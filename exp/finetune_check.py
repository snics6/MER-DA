"""E63．微調整の確認——WavLM-large を LoRA で微調整した版でも，出発点への罰則と候補全体への弱いラベルの得が残るか
（docs/exploratory-analyses.md E63．設計と予言 F1〜F4 は計算の前に書いた）．

AVT の 4 群（IEMOCAP 即興・台本，MELD，MOSEI），音声の生徒だけ．顔と言葉は凍結したまま（弱いラベルの先生の側）．

段（--stage）
  cache    元の音声を 16 kHz・平均 0・分散 1 の波形にして data/cache/wave_<群>.npy（float16）と wave_<群>.csv に保存する（60 秒まで．特徴抽出と同じ）
  teacher  凍結した特徴の第 1 エンコーダの AVT の世界で，E61 の oursx が音声の生徒に選んだ先生（2 段目の組み直し ref）を候補の上に作り，
           弱いラベルの行の重み（K=0 は釣り合い，K=5 はそのまま，総重み n）と K 本の重みを results/finetune_check/teacher_<T>.npz に保存する．
           凍結版の値（E39・私たちの形・融合の出発点）も同じ反復で測って teacher_<T>.csv に書く
  source   ターゲットのコーパスを除いたソース（3 通り）で出発点 θ₀（LoRA＋頭）を作る → results/finetune_check/start_<組>.pt
  adapt    ターゲットごとに，反復 × K ∈ {0, 5} × 手法（K 本だけ／出発点への罰則／罰則＋弱いラベル）× 罰則の重み λ を解き，テストの UAR を <T>.csv に追記する
  analyze  F1〜F4 を判定して results/explore_relation/e63_targets.csv・e63_tests.csv に書く

目的関数（微調整）：(1/M) Σ_i Σ_k s_ik·(−log p_k(x_i)) ＋ (λ/2)‖θ − θ₀‖²．s_ik は行の重み（候補は先生の見込みから作った弱いラベルの重み，
K 本は感情の釣り合いの重みを本当の感情に），M は重みの総和，θ は LoRA と頭の重み．ミニバッチでは (N/M)·平均 で同じ期待値にする．
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "exp"), str(ROOT / "measure"), str(ROOT / "features")]

import build_audio as BA  # noqa: E402  features/build_audio.py（decode・normalize）
import common as fcommon  # noqa: E402  features/common.py（発話一覧）
import compare_b1 as CB  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from baselines_distill import Prob, temp_rows  # noqa: E402
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from distill_belief import uar_of, weak_rows  # noqa: E402
from geom_e0 import settings  # noqa: E402
from multimodal_distill import WORLDS, beliefs1, blocks, make_students, sub_group  # noqa: E402

WORLD = "fuse-audio-video-text"
AUDIO = "audio_wavlm"
TARGETS = ["IEMOCAP-impro", "IEMOCAP-script", "MELD", "MOSEI"]
SETS = {"IEMOCAP": ["MELD", "MOSEI"], "MELD": ["IEMOCAP-impro", "IEMOCAP-script", "MOSEI"], "MOSEI": ["IEMOCAP-impro", "IEMOCAP-script", "MELD"]}
SET_OF = {"IEMOCAP-impro": "IEMOCAP", "IEMOCAP-script": "IEMOCAP", "MELD": "MELD", "MOSEI": "MOSEI"}
REPS = [0, 1, 2]
KS = ["0", "5"]
LAMS = (0.001, 0.01, 0.1)                 # 罰則の重み λ．固定版は 0.01（L2-SP の既定），他はオラクルとして併記
LAM_FIXED = 0.01
MODEL = "microsoft/wavlm-large"
SR = BA.SR
CROP_SEC, MAX_SEC = 10.0, BA.MAX_SEC      # 学習時は 10 秒まで（ランダムな位置），評価時は 60 秒まで（特徴抽出と同じ）
BATCH = 16
LR_LORA, LR_HEAD, WARMUP = 1e-4, 1e-3, 0.1
EPOCHS_SRC, EPOCHS_K, EPOCHS_OURS = 3, 20, 2
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
MOSEI_SRC_PER_CLASS, MOSEI_CAND, HOLDOUT = 1500, 4000, 0.1
EVAL_BUDGET_SEC, EVAL_MAX = 240, 64
OUT = ROOT / "results" / "finetune_check"
OUT_REL = ROOT / "results" / "explore_relation"
CACHE = ROOT / "data" / "cache"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- 波形
def group_table(g):
    """融合の世界の行（map.load と同じ順）．sample_id・label_4・speaker_group と，元の音声の場所。"""
    df, _ = load_any(g, WORLD)
    df = df.reset_index(drop=True)
    man = fcommon.load_manifest(GROUP_SPEC[g][0]).set_index("sample_id")
    m = man.loc[df.sample_id]
    df["path"] = [a if isinstance(a, str) and a else v for a, v in zip(m.path_audio, m.path_video)]
    return df


def build_cache(g, workers, limit=0):
    df = group_table(g)
    if limit:
        df = df.iloc[:limit]
    t0 = time.perf_counter()

    def one(path):
        w, sec, trunc = BA.decode(path)
        ok = len(w) >= 400                                             # CNN の 1 枠（400 標本）に届かないものだけ失敗扱い（MELD の 0.06〜0.09 秒の 12 本は特徴抽出と同じく使う）
        w = BA.normalize(w) if ok else np.zeros(SR, np.float32)
        return w.astype(np.float16), sec, trunc, ok
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, df.path.tolist()))
    lens = np.array([len(r[0]) for r in res], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    wave = np.concatenate([r[0] for r in res])
    np.save(CACHE / f"wave_{g}.npy", wave)
    pd.DataFrame({"sample_id": df.sample_id, "offset": off, "length": lens, "duration_sec": [r[1] for r in res],
                  "truncated": [r[2] for r in res], "ok": [r[3] for r in res]}).to_csv(CACHE / f"wave_{g}.csv", index=False)
    n_bad = int(sum(not r[3] for r in res))
    print(f"cache {g}：{len(df)} 本，{lens.sum() / SR / 3600:.2f} 時間，失敗 {n_bad}，{time.perf_counter() - t0:.0f} 秒", flush=True)


class Waves:
    """群ごとの波形（memmap）．行番号は融合の世界の Group の行番号と同じ。"""

    def __init__(self, g):
        self.g = g
        self.idx = pd.read_csv(CACHE / f"wave_{g}.csv")
        self.wave = np.load(CACHE / f"wave_{g}.npy", mmap_mode="r")
        df, _ = load_any(g, WORLD)
        assert (df.sample_id.to_numpy() == self.idx.sample_id.to_numpy()).all(), f"{g}：波形の並びが Group と違う"
        self.off, self.len, self.ok = self.idx.offset.to_numpy(), self.idx.length.to_numpy(), self.idx.ok.to_numpy()

    def get(self, i, max_len=None, rng=None):
        o, n = int(self.off[i]), int(self.len[i])
        if max_len is not None and n > max_len:
            s = int(rng.integers(0, n - max_len + 1)) if rng is not None else 0
            o, n = o + s, max_len
        return np.asarray(self.wave[o:o + n], dtype=np.float32)


def collate(ws):
    n = max(len(w) for w in ws)
    x = torch.zeros(len(ws), n)
    for r, w in enumerate(ws):
        x[r, :len(w)] = torch.from_numpy(w)
    lens = torch.tensor([len(w) for w in ws])
    return x, lens


# ---------------------------------------------------------------- 模型
def make_model(seed=0):
    from peft import LoraConfig, get_peft_model
    from transformers import WavLMModel
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    torch.manual_seed(seed)
    base = WavLMModel.from_pretrained(MODEL, layerdrop=0.0)        # layerdrop は切る（全部の層を通す）．SpecAugment の既定（mask_time_prob 0.075）は学習時だけ効く
    base.freeze_feature_encoder()
    pm = get_peft_model(base, LoraConfig(**LORA)).to(DEV)
    head = nn.Linear(base.config.hidden_size, 4).to(DEV)
    return pm, head


def trainable(pm, head):
    return [p for p in pm.parameters() if p.requires_grad], list(head.parameters())


def state_of(pm, head):
    return {"lora": {n: p.detach().cpu().clone() for n, p in pm.named_parameters() if p.requires_grad}, "head": {k: v.cpu().clone() for k, v in head.state_dict().items()}}


def load_state(pm, head, st):
    names = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    assert set(names) == set(st["lora"]), "LoRA の重みの名前が合わない"
    with torch.no_grad():
        for n, p in names.items():
            p.copy_(st["lora"][n].to(p.device))
    head.load_state_dict(st["head"])


def forward(pm, head, x, lens):
    x, lens = x.to(DEV), lens.to(DEV)
    mask = (torch.arange(x.shape[1], device=DEV)[None] < lens[:, None]).long()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
        h = pm(x, attention_mask=mask).last_hidden_state
    flen = pm.get_base_model()._get_feat_extract_output_lengths(lens)
    fmask = (torch.arange(h.shape[1], device=DEV)[None] < flen[:, None]).to(h.dtype)
    pooled = (h * fmask[..., None]).sum(1) / fmask.sum(1, keepdim=True)
    return head(pooled.float())


def train(pm, head, items, epochs, lam, theta0, seed, tag):
    """items = [(Waves, 行番号, 重みの列 (4,))]．(1/M) Σ s_ik ℓ_ik ＋ (λ/2)‖θ−θ₀‖² を AdamW で下げる。"""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    W = torch.tensor(np.stack([it[2] for it in items]), dtype=torch.float32, device=DEV)
    N, Mass = len(items), float(W.sum())
    scale = N / Mass
    lora, hp = trainable(pm, head)
    opt = torch.optim.AdamW([{"params": lora, "lr": LR_LORA}, {"params": hp, "lr": LR_HEAD}], weight_decay=0.0)
    spe = math.ceil(N / BATCH)
    total = epochs * spe
    warm = max(1, math.ceil(WARMUP * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    th0 = None if theta0 is None else [t.to(DEV) for t in theta0]
    pm.train()
    head.train()
    t0, step, losses = time.perf_counter(), 0, []
    max_len = int(CROP_SEC * SR)
    for ep in range(epochs):
        perm = rng.permutation(N)
        for b in range(spe):
            ix = perm[b * BATCH:(b + 1) * BATCH]
            x, lens = collate([items[i][0].get(items[i][1], max_len, rng) for i in ix])
            logits = forward(pm, head, x, lens)
            lp = torch.log_softmax(logits, 1)
            loss = -scale * (W[ix] * lp).sum() / len(ix)
            if th0 is not None and lam > 0:
                loss = loss + 0.5 * lam * sum(((p - p0) ** 2).sum() for p, p0 in zip(lora + hp, th0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
    sec = time.perf_counter() - t0
    print(f"    {tag}：{N} 本 × {epochs} エポック = {step} 歩，損失 {np.mean(losses[:spe]):.3f} → {np.mean(losses[-spe:]):.3f}，{sec:.0f} 秒", flush=True)
    return {"n_train": N, "steps": step, "train_sec": round(sec, 1), "loss_first": float(np.mean(losses[:spe])), "loss_last": float(np.mean(losses[-spe:]))}


@torch.inference_mode()
def predict(pm, head, waves, idx):
    pm.eval()
    head.eval()
    order = sorted(idx, key=lambda i: waves.len[i])
    out = np.zeros(len(order), dtype=np.int64)
    pos = {int(i): j for j, i in enumerate(order)}
    budget, i0 = EVAL_BUDGET_SEC * SR, 0
    while i0 < len(order):
        i1 = i0 + 1
        while i1 < len(order) and i1 - i0 < EVAL_MAX and (i1 + 1 - i0) * int(waves.len[order[i1]]) <= budget:
            i1 += 1
        x, lens = collate([waves.get(i) for i in order[i0:i1]])
        pred = forward(pm, head, x, lens).argmax(1).cpu().numpy()
        for i, p in zip(order[i0:i1], pred):
            out[pos[int(i)]] = p
        i0 = i1
    return np.array([out[pos[int(i)]] for i in idx])


def uar(y, pred):
    return CB.uar(np.asarray(y), np.asarray(pred))


# ---------------------------------------------------------------- 段 teacher（凍結した特徴）
def build_teacher(T):
    groups = groups_for(WORLD)
    data_f = {g: M.load(g, WORLD) for g in groups}
    data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(WORLD)}
    S = [s for s in groups if GROUP_SPEC[s][0] != GROUP_SPEC[T][0]]
    mods = WORLDS[WORLD]
    sets = {k: settings(k, T) for k in mods + [WORLD]}
    cache = {k: {} for k in mods + [WORLD]}
    info = {k: M5.pool_info(data_m[k] if k != WORLD else data_f, S, 0.01, cache[k]) for k in sets}
    ev = data_f[T]
    ctx = CB.Ctx(ev, REPS, ["5"], center="ind")
    saved, rows = {}, []
    for r in REPS:
        pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
        pos_of = {int(i): j for j, i in enumerate(pool)}
        k5 = np.array([pos_of[int(i)] for i in ctx.idx[(r, "5")]])
        label_sets = {"0": np.array([], int), "5": k5}
        stu = make_students(WORLD, T, data_f, data_m, S, pool, te, sets, info)
        yq, yte = ev.y[pool], ev.y[te]
        prs = {k: Prob(stu[k], yq, label_sets) for k in mods + [WORLD]}
        n = prs[WORLD].n
        none_f = prs[WORLD].solve([(K, None, K) for K in KS])
        P0_f = {K: prs[WORLD].probs(none_f[K], prs[WORLD].s["Z"]) for K in KS}
        lg = {K: 0 for K in KS}                                                # 2 段目の先生 ref（baselines_distill.run_world_unit と同じ）
        for m in mods:
            sol = prs[m].solve([(K, weak_rows(P0_f[K], 500), K) for K in KS])
            for K in KS:
                lg[K] = lg[K] + prs[m].probs(sol[K], prs[m].s["Z"]).clamp(min=1e-12).log()
        ref = {K: torch.softmax(lg[K], 1) for K in KS}
        w4 = {"0": weak_rows(ref["0"], float(n)), "5": temp_rows(ref["5"], 1, "n")}   # E61 の AVT の音声で oursx が選んだ形：K=0 釣り合い・n，K=5 そのまま・n
        a = prs[AUDIO]
        none_a = a.solve([(K, None, K) for K in KS])
        ours_a = a.solve([(K, w4[K], K) for K in KS])
        for K in KS:
            rows.append({"target": T, "rep": r, "K": K, "method": "frozen_e39", "uar": float(uar_of(*none_a[K], a.s["Zte"], yte))})
            rows.append({"target": T, "rep": r, "K": K, "method": "frozen_ours", "uar": float(uar_of(*ours_a[K], a.s["Zte"], yte))})
            rows.append({"target": T, "rep": r, "K": K, "method": "frozen_fused_start", "uar": float(uar_of(prs[WORLD].W0, prs[WORLD].b0, stu[WORLD]["Zte"], yte))})
            rows.append({"target": T, "rep": r, "K": K, "method": "frozen_fused_K", "uar": float(uar_of(*none_f[K], stu[WORLD]["Zte"], yte))})
        saved[f"pool_{r}"], saved[f"te_{r}"], saved[f"k5_{r}"] = pool, te, pool[k5]
        saved[f"S0_{r}"] = w4["0"].reshape(4, n).T.cpu().numpy()                # 候補 i の感情 k の重み
        saved[f"S5_{r}"] = w4["5"].reshape(4, n).T.cpu().numpy()
        saved[f"wt5_{r}"] = a.wt["5"].reshape(4, n).T.sum(1).cpu().numpy()      # K 本の重み（本当の感情の行だけ非 0）
        saved[f"ref0_{r}"], saved[f"ref5_{r}"] = ref["0"].cpu().numpy(), ref["5"].cpu().numpy()
        print(f"  teacher {T} 反復 {r}：候補 {n}，テスト {len(te)}，K 本 {len(k5)}，凍結 E39 {rows[-8]['uar']:.3f}，凍結 ours {rows[-7]['uar']:.3f}", flush=True)
    np.savez(OUT / f"teacher_{T}.npz", y=ev.y, **saved)
    pd.DataFrame(rows).to_csv(OUT / f"teacher_{T}.csv", index=False)


# ---------------------------------------------------------------- 段 source
def train_source(set_name, smoke=False):
    groups = SETS[set_name]
    waves = {g: Waves(g) for g in groups}
    rng = np.random.default_rng(0)
    items, hold = [], []
    for g in groups:
        df = group_table(g)
        y = df.label_4.to_numpy().astype(int)
        spk = df.speaker_group.astype(str).to_numpy()
        ok = waves[g].ok
        sel = np.flatnonzero(ok)
        if g == "MOSEI":                                                       # 感情ごとに 1,500 本に間引く
            sel = np.concatenate([rng.permutation(sel[y[sel] == k])[:MOSEI_SRC_PER_CLASS] for k in range(4)])
        us = np.unique(spk[sel])
        hs = set(rng.permutation(us)[:max(1, int(round(HOLDOUT * len(us))))])  # 話者の 10% を抜く（過学習を見るだけ．選ばない）
        tr = np.array([i for i in sel if spk[i] not in hs])
        ho = np.array([i for i in sel if spk[i] in hs])
        cnt = np.bincount(y[tr], minlength=4).astype(float)
        for i in tr:
            w = np.zeros(4, np.float32)
            w[y[i]] = 1.0 / (len(groups) * cnt[y[i]])                          # 群と感情で等しい重み（γ=0 に対応）
            items.append((waves[g], int(i), w))
        hold.append((g, ho, y))
        print(f"  {set_name} ← {g}：学習 {len(tr)} 本，抜いた話者 {len(hs)}／{len(us)}（{len(ho)} 本）", flush=True)
    tot = sum(it[2].sum() for it in items)
    items = [(w_, i_, v_ * len(items) / tot) for w_, i_, v_ in items]           # 平均の重みを 1 にする
    if smoke:
        items = items[:64]
    pm, head = make_model(seed=0)
    log = train(pm, head, items, 1 if smoke else EPOCHS_SRC, 0.0, None, seed=0, tag=f"source {set_name}")
    res = {}
    for g, ho, y in hold:
        ho = ho[:32] if smoke else ho
        res[g] = uar(y[ho], predict(pm, head, waves[g], ho))
    print(f"  {set_name}：抜いた話者の UAR {res}", flush=True)
    if not smoke:
        torch.save(state_of(pm, head), OUT / f"start_{set_name}.pt")
        pd.DataFrame([{"set": set_name, **log, **{f"holdout_{g}": v for g, v in res.items()}}]).to_csv(OUT / f"start_{set_name}.csv", index=False)


# ---------------------------------------------------------------- 段 adapt
def plan(K):
    if K == "0":
        return [("start", 0.0)] + [("ours", lam) for lam in LAMS]
    return [("k_only", 0.0)] + [("anchor", lam) for lam in LAMS] + [("ours", lam) for lam in LAMS]


def adapt(T, reps, smoke=False):
    tz = np.load(OUT / f"teacher_{T}.npz")
    y = tz["y"]
    waves = Waves(T)
    f = OUT / f"{T}{'_smoke' if smoke else ''}.csv"
    done = set()
    if f.exists():
        d = pd.read_csv(f, dtype={"K": str})
        done = {(int(a), str(b), c, float(e)) for a, b, c, e in zip(d.rep, d.K, d.method, d.lam)}
    sp = OUT / f"start_{SET_OF[T]}.pt"
    st = torch.load(sp, weights_only=True) if sp.exists() else None
    if st is None and not smoke:
        raise FileNotFoundError(sp)
    for r in reps:
        pool, te, k5 = tz[f"pool_{r}"], tz[f"te_{r}"], tz[f"k5_{r}"]
        S = {"0": tz[f"S0_{r}"], "5": tz[f"S5_{r}"]}
        wt5 = tz[f"wt5_{r}"]
        cand = pool
        if T == "MOSEI":                                                       # 候補を 4,000 本に間引く（K 本は常に入れる）
            cand = np.random.default_rng(100 + r).permutation(pool)[:MOSEI_CAND]
        if smoke:
            cand, te = cand[:48], te[:32]
        kpos = {int(i): j for j, i in enumerate(pool)}
        for K in KS:
            for method, lam in plan(K):
                if (r, K, method, lam) in done:
                    continue
                seed = 1000 * r + 10 * int(K) + {"start": 0, "k_only": 1, "anchor": 2, "ours": 3}[method] + int(round(1000 * lam))
                pm, head = make_model(seed=seed)
                if st is not None:
                    load_state(pm, head, st)
                th0 = [p.detach().clone() for p in sum(trainable(pm, head), [])]
                items = []
                if method in ("k_only", "anchor", "ours") and K != "0":
                    for i in k5:
                        w = np.zeros(4, np.float32)
                        w[y[i]] = wt5[kpos[int(i)]]
                        items.append((waves, int(i), w))
                if method == "ours":
                    ks = set(int(i) for i in k5)
                    for i in cand:
                        if int(i) in ks or not waves.ok[i]:
                            continue
                        items.append((waves, int(i), S[K][kpos[int(i)]].astype(np.float32)))
                torch.cuda.reset_peak_memory_stats() if DEV.type == "cuda" else None
                log = {}
                if items:
                    epochs = EPOCHS_OURS if method == "ours" else EPOCHS_K
                    log = train(pm, head, items, 1 if smoke else epochs, lam if method != "k_only" else 0.0, th0 if method != "k_only" else None, seed, f"{T} r{r} K{K} {method} λ{lam:g}")
                t0 = time.perf_counter()
                u = uar(y[te], predict(pm, head, waves, te))
                row = {"target": T, "rep": r, "K": K, "method": method, "lam": lam, "uar": u, "n_test": len(te), "eval_sec": round(time.perf_counter() - t0, 1),
                       "gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if DEV.type == "cuda" else np.nan, **log}
                pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
                print(f"  {T} r{r} K{K} {method} λ{lam:g}：UAR {u:.4f}（テスト {len(te)}，評価 {row['eval_sec']} 秒，GPU {row['gpu_gb']} GB）", flush=True)
                del pm, head, th0
                torch.cuda.empty_cache()


# ---------------------------------------------------------------- 段 analyze
def analyze():
    ft = pd.concat([pd.read_csv(OUT / f"{T}.csv", dtype={"K": str}) for T in TARGETS if (OUT / f"{T}.csv").exists()])
    fr = pd.concat([pd.read_csv(OUT / f"teacher_{T}.csv", dtype={"K": str}) for T in TARGETS if (OUT / f"teacher_{T}.csv").exists()])
    fr["lam"] = 0.0
    ft, fr = ft[ft.rep.isin(REPS)], fr[fr.rep.isin(REPS)]
    a = pd.concat([ft[["target", "rep", "K", "method", "lam", "uar"]], fr[["target", "rep", "K", "method", "lam", "uar"]]])
    a["uar"] = a.uar * 100
    D = a.groupby(["target", "K", "method", "lam"]).uar.agg(["mean", "count"]).reset_index().rename(columns={"mean": "uar", "count": "reps"})
    D.to_csv(OUT_REL / "e63_targets.csv", index=False)
    g = D.groupby(["K", "method", "lam"]).agg(uar=("uar", "mean"), targets=("target", "nunique")).reset_index()
    piv = D.pivot_table(index=["target", "K"], columns=["method", "lam"], values="uar")

    def col(method, lam=0.0):
        return piv[(method, lam)] if (method, lam) in piv.columns else pd.Series(np.nan, index=piv.index)
    tests = {}
    k5 = piv.index.get_level_values("K") == "5"
    k0 = ~k5
    tests["F1"] = (col("k_only")[k5] - col("frozen_e39")[k5]).mean() <= -1.0
    tests["F2"] = (col("anchor", LAM_FIXED)[k5] - col("k_only")[k5]).mean() >= 1.0
    d3 = {"0": (col("ours", LAM_FIXED)[k0] - col("start")[k0]).mean(), "5": (col("ours", LAM_FIXED)[k5] - col("anchor", LAM_FIXED)[k5]).mean()}
    tests["F3"] = all(v >= 1.0 for v in d3.values())
    d4 = {"0": (col("ours", LAM_FIXED)[k0] - col("frozen_ours")[k0]).mean(), "5": (col("ours", LAM_FIXED)[k5] - col("frozen_ours")[k5]).mean()}
    tests["F4"] = all(abs(v) <= 2.0 for v in d4.values())
    R = g.copy()
    for k, v in tests.items():
        R[k] = bool(v)
    R["F3_d0"], R["F3_d5"], R["F4_d0"], R["F4_d5"] = d3["0"], d3["5"], d4["0"], d4["5"]
    R.to_csv(OUT_REL / "e63_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(piv.round(2).to_string())
    print()
    print(g.pivot_table(index=["method", "lam"], columns="K", values="uar").round(2).to_string())
    print(f"\nF1 {tests['F1']}（K 本だけ − 凍結 E39，K=5：{(col('k_only')[k5] - col('frozen_e39')[k5]).mean():+.2f}）")
    print(f"F2 {tests['F2']}（罰則 − K 本だけ，K=5：{(col('anchor', LAM_FIXED)[k5] - col('k_only')[k5]).mean():+.2f}）")
    print(f"F3 {tests['F3']}（私たちの形 − 罰則だけ：K=0 {d3['0']:+.2f}，K=5 {d3['5']:+.2f}）")
    print(f"F4 {tests['F4']}（微調整版 − 凍結版の私たちの形：K=0 {d4['0']:+.2f}，K=5 {d4['5']:+.2f}）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["cache", "teacher", "source", "adapt", "analyze"])
    ap.add_argument("--groups", nargs="+", default=TARGETS, help="cache の群")
    ap.add_argument("--target", nargs="+", default=TARGETS, help="teacher・adapt のターゲット")
    ap.add_argument("--set", nargs="+", default=list(SETS), help="source のソースの組")
    ap.add_argument("--reps", nargs="+", type=int, default=REPS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="試走用．cache の件数")
    ap.add_argument("--smoke", action="store_true", help="試走用．source・adapt を 64 本・1 エポックで回し，adapt は <T>_smoke.csv に書く")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "analyze":
        return analyze()
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/finetune_check.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": MODEL, "lora": LORA, "batch": BATCH, "lr": [LR_LORA, LR_HEAD], "warmup": WARMUP,
            "epochs": [EPOCHS_SRC, EPOCHS_K, EPOCHS_OURS], "lams": LAMS, "lam_fixed": LAM_FIXED, "crop_sec": CROP_SEC, "max_sec": MAX_SEC,
            "mosei": [MOSEI_SRC_PER_CLASS, MOSEI_CAND], "holdout": HOLDOUT, "reps": args.reps, "ks": KS,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    with open(OUT / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "cache":
        for g in args.groups:
            if (CACHE / f"wave_{g}.npy").exists() and not args.limit:
                print(f"cache {g}：済み", flush=True)
                continue
            build_cache(g, args.workers, args.limit)
    elif args.stage == "teacher":
        for T in args.target:
            if (OUT / f"teacher_{T}.npz").exists():
                print(f"teacher {T}：済み", flush=True)
                continue
            t0 = time.perf_counter()
            build_teacher(T)
            print(f"teacher {T}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    elif args.stage == "source":
        for s in args.set:
            if (OUT / f"start_{s}.pt").exists() and not args.smoke:
                print(f"source {s}：済み", flush=True)
                continue
            t0 = time.perf_counter()
            train_source(s, args.smoke)
            print(f"source {s}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    elif args.stage == "adapt":
        for T in args.target:
            t0 = time.perf_counter()
            adapt(T, args.reps, args.smoke)
            print(f"adapt {T}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
