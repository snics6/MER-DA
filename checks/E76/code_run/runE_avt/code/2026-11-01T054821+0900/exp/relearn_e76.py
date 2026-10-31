"""E76（DAI 論文のステップ 2）：筋 1〜3 の学び直し。テストの発話ごとの点数と分類器の係数を保存する（GPU）。

設計は `DAI_paper/steps/step2_design.md`（関門 1 を通した後に `docs/exploratory-analyses.md` の E76 の節へ写す）。
TAC 版の模擬査読の指摘のうち，保存済みの結果からは出せないもの（話者を単位にした区間，性別ごとの値，
ターゲットのみの C の別の規則，q の当てはめ直しと言葉系の q，主張 1 の中心合わせの手の内訳）を，
1 回の計算で答えられるようにする。新しい手法は無い。すべて既存の台本の学習を同じ条件で解き直し，
**UAR だけでなくテストの発話ごとの点数（4 感情の対数オッズ）と係数を残す**。

学ぶもの（1 ターゲット × 反復 r．テストの袋・K 本・ソースの行・C は地図（`exp/map.py`）と C を固定した地図と同じ）

    tgt_only   ① ターゲットの K 本だけ．`map.train`（交差検証で C を選ぶ）．**候補の C 4 つすべての点数**を残す
    src_only   ② ソースだけ．`map.train`（交差検証で C を選ぶ）．**ターゲットの全発話**の点数を C 4 つで残す
    merged     ③ 合併．C ＝ 同じ反復のソースのみの C（C を固定した地図の規則）．ソースのみの解から温め直す（地図と同じ）
    h2〜h8     E25 の手（`exp/interventions.py`）．K は 1・5・10．h4・h5 は K 本で解いた切片のずらしを残す

反復は既定で 0〜9，K は 1・5・10・25・50・全件（手は 1・5・10）。

出力（`--out`．既定 `results/e76/`．既存の結果ファイルは上書きしない）

    provenance.json                     実行ごとの設定・コミット番号・GPU（実行のたびに runs に 1 つ足す）
    code/<開始時刻>/                     実行したコードの写しと git diff（`map.snapshot_code`）
    <kind>/<ターゲット>.npz              点数・係数・選んだ C（鍵の一覧は `KEYS_DOC`）
    <kind>/<ターゲット>.csv              条件ごとの UAR と，前の結果ファイルとの照合（再現の検算）
    <kind>/<ターゲット>.json             その npz を作った設定，コミット番号，所要時間，照合のまとめ

再現の検算（毎回）：地図（C を固定した地図）の同じ（ターゲット，ソース，K，反復）の UAR と C，
E25 の手の UAR，前の点数（`merged_scores_*`，`tgt_scores_*`）との差を条件ごとに `<ターゲット>.csv` に書き，
まとめを `<ターゲット>.json` と画面に出す。保存した係数から点数を作り直した差も測る。

途中から再開するとき：`<ターゲット>.npz` があれば，その json のコミット番号と設定が今と同じなら飛ばし，
違えば止まる（混ぜない）。

    # 小さく流す（ターゲット 1 つ，反復 1 つ）
    ./.venv/bin/python exp/relearn_e76.py --kinds audio_wavlm --targets IEMOCAP-impro --reps 0 \
        --out /tmp/e76_trial --allow-dirty
    # 本番（関門 1 の後．tmux の mer-da で）
    ./.venv/bin/python exp/relearn_e76.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import ablate  # noqa: E402
import interventions as IV  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import shift  # noqa: E402
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402

KINDS_MAIN = ("audio_wavlm", "video", "text_roberta", "fuse-audio-video", "fuse-audio-text",
              "fuse-video-text", "fuse-audio-video-text", "audio_hubert")   # A(HuBERT) は筋 2・3 の表に行があるので足した（2026-11-01）
K_ALL = M.K_GRID                      # ("1", "5", "10", "25", "50", "all")
K_HANDS = IV.KS                       # ("1", "5", "10")．E25 と同じ
HANDS_ALL = (2, 3, 4, 5, 6, 7, 8)
C_GRID = M.C_GRID
MEM_FRACTION = 0.6
E25_NAMES = {2: "2 中心合わせ＋ソースのみ", 3: "3 中心合わせ＋合併", 4: "4 ソースのみ＋切片直し",
             5: "5 中心合わせ＋ソースのみ＋切片直し", 6: "6 ターゲット専用の切片つき合併", 7: "7 仮説転移",
             8: "8 CORAL＋合併"}
TOL_EXACT = 1e-6                      # 同じ計算の道筋（tgt_only・src_only）で UAR が一致したとみなす幅
TOL_REPRO = 2e-3                      # 温め方や束ね方が違う学習で一致したとみなす幅の下限（0.2 点．summarize の tol）

KEYS_DOC = """npz の鍵（r は反復，K は K の文字列，s はソースの群の名前，j はソースの行の版．C の並びは C_GRID）
  sample_id, speaker_group, y                ターゲットの全発話（`data.load_any` の順．sample_id で並ぶ）
  reps                                       反復の一覧
  test_idx_r{r}                              テストの行番号（上の並びの中の位置）
  kidx_r{r}_K{K}                             K 本の行番号（K=all は無い：テスト以外の全部）
  t_logit_r{r}                               ① (len(K), 4 C, テストの数, 4) float32
  t_C_r{r}, t_rule_r{r}, t_cv_r{r}           ① 選んだ C，規則（cv/tie/fixed），交差検証の UAR (len(K), 4 C)
  t_A_r{r}, t_a0_r{r}                        ① 生の特徴の上の係数 (len(K), 4 C, 4, d) float32 と切片 (len(K), 4 C, 4)
  s_{s}__ver                                 ② 反復ごとの版 j（反復の一覧と同じ順．IEMOCAP 同士の組だけ反復で版が変わる）
  s_{s}__v{j}_logit                          ② ターゲットの全発話の点数 (4 C, 全発話, 4) float32
  s_{s}__v{j}_C, _rule, _cv, _A, _a0         ② 選んだ C，規則，交差検証の UAR (4 C)，係数 (4 C, 4, d)，切片 (4 C, 4)
  m_{s}_r{r}_logit                           ③ (len(K), テストの数, 4)．C ＝ ソースのみの C（--merged-all-c なら (len(K), 4 C, …)）
  m_{s}_r{r}_A, _a0                          ③ 係数 (len(K), 4, d)，切片 (len(K), 4)
  mean_t                                     ターゲットの全発話の平均（中心合わせ）
  h2_{s}__v{j}_logit, _A, _a0, mean_s        手 2：中心合わせしたソースのみ．全発話の点数 = (X − mean_t) A^T + a0
  h4_{s}_r{r}_b, h5_{s}_r{r}_b               手 4・5：K 本で解いた切片のずらし (len(K_HANDS), 4)．点数 = ソースのみ（手 5 は手 2）+ b
  h{3,6,7,8}_{s}_r{r}_logit, _A, _a0         手 3・6・7・8：テストの点数 (len(K_HANDS), テストの数, 4) と係数
                                             （手 3 は中心合わせした特徴，手 6 は特徴の末尾に「ターゲットか」の 1 列を足した特徴の上）
  coef_check                                 係数から作り直した点数と保存した点数の差の最大（ターゲットのテストで）
"""


# ---------------------------------------------------------------- 記録

def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def code_files() -> list[str]:
    """この台本が読み込んだ MER-DA の中のモジュール（コミット番号で固定したいもの）。"""
    out = []
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if (f and Path(f).is_absolute() and Path(f).exists() and Path(f).resolve().is_relative_to(ROOT)
                and ".venv" not in Path(f).parts):
            out.append(str(Path(f).resolve().relative_to(ROOT)))
    return sorted(set(out))


def code_state() -> dict:
    files = code_files()
    status = git("status", "--porcelain", "--", *files)
    return {"commit": git("rev-parse", "HEAD"), "code_files": files,
            "code_dirty": bool(status), "code_status": status.splitlines(),
            "tree_dirty": bool(git("status", "--porcelain"))}


# ---------------------------------------------------------------- データ

def load_with_ids(name: str, kind: str) -> tuple[M.Group, pd.DataFrame]:
    """`map.load` と同じ Group と，発話の sample_id・話者群を返す（同じ `load_any` の 1 回の呼び出しから作る）。"""
    df, X = load_any(name, kind)
    sp = df["speaker_group"].astype(str).reset_index(drop=True)
    person = (GROUP_SPEC[name][0] + ":" + sp).to_numpy()
    g = M.Group(name, logreg.to_device(X), df["label_4"].to_numpy(), sp, person)
    return g, df[["sample_id", "speaker_group"]].reset_index(drop=True)


def raw_coef(f: logreg.Fitted, c: float) -> tuple[np.ndarray, np.ndarray]:
    """生の特徴の上の係数：点数 = X A^T + a0（標準化を係数に畳み込む）。"""
    A = f.W[c] / f.sd
    a0 = f.b[c] - (f.mu / f.sd) @ f.W[c].T
    return A.cpu().numpy().astype(np.float32), a0.cpu().numpy()


def logits_np(f: logreg.Fitted, X: torch.Tensor, c: float) -> np.ndarray:
    L = f.logits(X, c).cpu().numpy()
    if not np.isfinite(L).all():
        raise FloatingPointError("点数に NaN か Inf がある")
    return L


def uar_of(L: np.ndarray, y: np.ndarray) -> float:
    return shift.uar(y, L.argmax(1))


def free_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- 照合の相手

def read_map(path: Path) -> dict:
    """C を固定した地図の (method, source, K, rep) → {C，C_rule，uar，test_uar_C*}。無ければ空。"""
    if not path.exists():
        return {}
    d = pd.read_csv(path, comment="#", dtype={"K": str, "source": str})
    d["K"], d["source"] = d.K.fillna(""), d.source.fillna("")
    cols = ["C", "C_rule", "uar"] + [f"test_uar_C{c:g}" for c in C_GRID]
    return {(x["method"], x["source"], x["K"], int(x["rep"])): {k: x[k] for k in cols}
            for x in d.to_dict("records")}


def read_e25(path: Path) -> dict:
    if not path.exists():
        return {}
    d = pd.read_csv(path, comment="#", dtype={"K": str, "source": str})
    d["K"] = d.K.fillna("")
    return {(r["method"], r["source"], r["K"], int(r["rep"])): float(r["uar"]) for r in d.to_dict("records")}


def read_old_scores(paths: list[Path]) -> dict:
    for p in paths:
        if p.exists():
            z = np.load(p)
            return {"path": str(p), "z": z}
    return {}


# ---------------------------------------------------------------- 1 ターゲット

class Recorder:
    def __init__(self, kind: str, target: str):
        self.kind, self.target = kind, target
        self.rows: list[dict] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.old_diff: list[dict] = []
        self.ustep: dict[int, float] = {}       # 反復ごとの，1 本の発話が UAR を動かす最大の量

    def row(self, **kw) -> None:
        self.rows.append({"kind": self.kind, "target": self.target, **kw})


def compare_old(rec: Recorder, old: dict, key: str, L_new: np.ndarray, what: str) -> None:
    """前に保存した同じ分類器の点数と比べる（`docs/methods.md` 9.3 節 I14）。"""
    if not old or key not in old["z"].files:
        return
    L_old = old["z"][key]
    if L_old.shape != L_new.shape:
        rec.old_diff.append({"what": what, "key": key, "max_abs": np.nan, "agree": np.nan,
                             "note": f"形が違う {L_old.shape} {L_new.shape}"})
        return
    rec.old_diff.append({"what": what, "key": key, "max_abs": float(np.abs(L_old - L_new).max()),
                         "agree": float((L_old.argmax(1) == L_new.argmax(1)).mean()), "note": ""})


def run_target(kind: str, tname: str, data: dict, ids: pd.DataFrame, sources: list[str], reps: list[int],
               Ks: tuple[str, ...], hands: tuple[int, ...], merged_all_c: bool, src_cache: dict,
               ref_map: dict, ref_e25: dict, old_m: dict, old_t: dict) -> Recorder:
    tgt = data[tname]
    rec = Recorder(kind, tname)
    A = rec.arrays
    nt = len(tgt.y)
    A["sample_id"] = ids["sample_id"].to_numpy().astype(str)
    A["speaker_group"] = ids["speaker_group"].to_numpy().astype(str)
    A["y"] = tgt.y.astype(np.int8)
    A["reps"] = np.array(reps)
    split = {}
    for r in reps:
        te = M.test_mask(tgt, r)
        order = M.kshot_order(tgt, np.flatnonzero(~te), r)
        idx = {K: M.take_k(order, K) for K in Ks}
        split[r] = (te, idx, np.unique(tgt.person[te]))
        rec.ustep[r] = 1.0 / (4 * int(np.bincount(tgt.y[te], minlength=4).min()))
        A[f"test_idx_r{r}"] = np.flatnonzero(te).astype(np.int32)
        for K in Ks:
            if K != "all":
                A[f"kidx_r{r}_K{K}"] = idx[K].astype(np.int32)

    # ---- ① ターゲットのみ（地図と同じ `map.train`．C 4 つすべての点数）
    t0 = time.perf_counter()
    for r in reps:
        te, idx, _ = split[r]
        Xte, yte = M.rows(tgt.X, te), tgt.y[te]
        Ls, As, a0s, Cs, rules, cvs = [], [], [], [], [], []
        for K in Ks:
            i = idx[K]
            t = M.train(M.rows(tgt.X, i), tgt.y[i], tgt.person[i], np.ones(len(i), bool), r)
            L = np.stack([logits_np(t.final, Xte, c) for c in C_GRID])
            co = [raw_coef(t.final, c) for c in C_GRID]
            Ls.append(L.astype(np.float32))
            As.append(np.stack([a for a, _ in co]))
            a0s.append(np.stack([b for _, b in co]))
            Cs.append(t.C)
            rules.append(t.rule)
            cvs.append([np.nan if t.cv is None else t.cv[c] for c in C_GRID])
            test = {c: uar_of(L[j], yte) for j, c in enumerate(C_GRID)}
            ref = ref_map.get(("tgt_only", "", K, r))
            rec.row(source="", method="tgt_only", K=K, rep=r, version=-1, C=t.C, C_rule=t.rule,
                    uar=test[t.C], ref_C=None if ref is None else ref["C"],
                    ref_rule=None if ref is None else ref["C_rule"],
                    ref_uar=None if ref is None else ref["uar"],
                    max_abs_diff_allC=None if ref is None else max(
                        abs(test[c] - ref[f"test_uar_C{c:g}"]) for c in C_GRID),
                    n_iter=int(t.final.n_iter[t.C]), grad=float(max(t.final.grad.values())),
                    **{f"test_uar_C{c:g}": test[c] for c in C_GRID})
            if K in ("1", "5", "10"):
                compare_old(rec, old_t, f"t_K{K}_r{r}", L[C_GRID.index(t.C)].astype(np.float32), "tgt_only")
        A[f"t_logit_r{r}"] = np.stack(Ls)
        A[f"t_A_r{r}"] = np.stack(As)
        A[f"t_a0_r{r}"] = np.stack(a0s)
        A[f"t_C_r{r}"] = np.array(Cs)
        A[f"t_rule_r{r}"] = np.array(rules)
        A[f"t_cv_r{r}"] = np.array(cvs)
        free_gpu()
    t_tgt = time.perf_counter() - t0

    # ---- ② ソースのみ，③ 合併，手
    yt = M.labels(tgt.y)
    Xt_c = None
    if hands:
        mean_t = tgt.X.mean(0)
        Xt_c = tgt.X - mean_t
        A["mean_t"] = mean_t.cpu().numpy()
    timing = {"tgt_only": t_tgt, "src_only": 0.0, "merged": 0.0, **{f"h{h}": 0.0 for h in hands}}
    for sname in sources:
        s = data[sname]
        versions: dict[tuple, list[int]] = {}
        for r in reps:
            key = (s.name, tuple(np.intersect1d(split[r][2], s.person)))
            versions.setdefault(key, []).append(r)
        ver_of = {}
        for j, (key, rs) in enumerate(versions.items()):
            for r in rs:
                ver_of[r] = j
            keep = ~np.isin(s.person, split[rs[0]][2])
            Xs, ys, ps = M.rows(s.X, keep), s.y[keep], s.person[keep]
            t0 = time.perf_counter()
            cached = key in src_cache
            if not cached:
                src_cache[key] = M.train(Xs, ys, ps, np.ones(len(ys), bool), M.SRC_CV_SEED)
            ts = src_cache[key]
            C = ts.C
            Ls_all = np.stack([logits_np(ts.final, tgt.X, c) for c in C_GRID])      # (4 C, 全発話, 4)
            co = [raw_coef(ts.final, c) for c in C_GRID]
            pre = f"s_{sname}__v{j}"
            A[f"{pre}_logit"] = Ls_all.astype(np.float32)
            A[f"{pre}_A"] = np.stack([a for a, _ in co])
            A[f"{pre}_a0"] = np.stack([b for _, b in co])
            A[f"{pre}_C"] = np.array(C)
            A[f"{pre}_rule"] = np.array(ts.rule)
            A[f"{pre}_cv"] = np.array([np.nan if ts.cv is None else ts.cv[c] for c in C_GRID])
            for r in rs:
                te = split[r][0]
                test = {c: uar_of(Ls_all[jc][te], tgt.y[te]) for jc, c in enumerate(C_GRID)}
                ref = ref_map.get(("src_only", sname, "", r))
                rec.row(source=sname, method="src_only", K="", rep=r, version=j, C=C, C_rule=ts.rule,
                        uar=test[C], ref_C=None if ref is None else ref["C"],
                        ref_rule=None if ref is None else ref["C_rule"],
                        ref_uar=None if ref is None else ref["uar"],
                        max_abs_diff_allC=None if ref is None else max(
                            abs(test[c] - ref[f"test_uar_C{c:g}"]) for c in C_GRID),
                        n_iter=int(ts.final.n_iter[C]), grad=float(max(ts.final.grad.values())),
                        cached=cached, **{f"test_uar_C{c:g}": test[c] for c in C_GRID})
                compare_old(rec, old_m, f"s_{sname}_r{r}", Ls_all[C_GRID.index(C)][te].astype(np.float32),
                            "src_only")
            timing["src_only"] += time.perf_counter() - t0

            # ③ 合併：C ＝ ソースのみの C．ソースのみの最終の解から温め直す（地図の合併の最終の学習と同じ出発点）
            t0 = time.perf_counter()
            ns = len(ys)
            X = torch.cat([tgt.X, Xs])
            y_all = M.labels(np.r_[tgt.y, ys])
            jobs = [(r, K) for r in rs for K in Ks]

            def mmask(r: int, K: str, idx_of=split) -> np.ndarray:
                m = np.r_[np.zeros(nt, bool), np.ones(ns, bool)]
                m[idx_of[r][1][K]] = True
                return m

            Cm = C_GRID if merged_all_c else (C,)
            fits = logreg.fit_batch(X, y_all, [mmask(r, K) for r, K in jobs], Cm,
                                    [ts.final] * len(jobs))
            fm = dict(zip(jobs, fits))
            for r in rs:
                te = split[r][0]
                Xte, yte = M.rows(tgt.X, te), tgt.y[te]
                Lm, Am, a0m = [], [], []
                for K in Ks:
                    f = fm[(r, K)]
                    Lc = {c: logits_np(f, Xte, c) for c in Cm}
                    Lm.append(np.stack([Lc[c] for c in Cm]) if merged_all_c else Lc[C])
                    a, b = raw_coef(f, C)
                    Am.append(a)
                    a0m.append(b)
                    test = {c: uar_of(Lc[c], yte) for c in Cm}
                    ref = ref_map.get(("merged", sname, K, r))
                    extra = {}
                    if merged_all_c and ref is not None:
                        ds = [abs(test[c] - ref[f"test_uar_C{c:g}"]) for c in C_GRID
                              if not pd.isna(ref[f"test_uar_C{c:g}"])]
                        extra["max_abs_diff_allC"] = max(ds) if ds else None
                    rec.row(source=sname, method="merged", K=K, rep=r, version=j, C=C, C_rule="src",
                            uar=test[C], ref_C=None if ref is None else ref["C"],
                            ref_rule=None if ref is None else ref["C_rule"],
                            ref_uar=None if ref is None else ref["uar"],
                            n_iter=int(f.n_iter[C]), grad=float(f.grad[C]),
                            **{f"test_uar_C{c:g}": test.get(c, np.nan) for c in C_GRID}, **extra)
                    if K in ("1", "5", "10"):
                        compare_old(rec, old_m, f"m_{sname}_K{K}_r{r}", Lc[C].astype(np.float32), "merged")
                A[f"m_{sname}_r{r}_logit"] = np.stack(Lm).astype(np.float32)
                A[f"m_{sname}_r{r}_A"] = np.stack(Am)
                A[f"m_{sname}_r{r}_a0"] = np.stack(a0m)
            del X, fits, fm
            free_gpu()
            timing["merged"] += time.perf_counter() - t0

            if hands:
                run_hands(rec, kind, tgt, s, sname, j, rs, key, src_cache, Xs, ys, ts, C, Xt_c, yt, split, hands,
                          Ls_all[C_GRID.index(C)], ref_e25, timing)
        A[f"s_{sname}__ver"] = np.array([ver_of[r] for r in reps])
        print(f"  {kind} {tname} ← {sname}：版 {len(versions)}（累計 ①{timing['tgt_only']:.0f} ②{timing['src_only']:.0f}"
              f" ③{timing['merged']:.0f} 手{sum(v for k, v in timing.items() if k.startswith('h')):.0f} 秒）", flush=True)
    rec.timing = timing
    return rec


def run_hands(rec: Recorder, kind: str, tgt: M.Group, s: M.Group, sname: str, j: int, rs: list[int],
              key: tuple, src_cache: dict, Xs: torch.Tensor, ys: np.ndarray, ts: M.Trained, C: float,
              Xt_c: torch.Tensor, yt: torch.Tensor, split: dict, hands: tuple[int, ...],
              Ls_all_C: np.ndarray, ref_e25: dict, timing: dict) -> None:
    """E25 の手（`exp/interventions.py` と同じ作り方）。ソースのみは E76 の ② の解（`map.train` の最終の解）を使う。"""
    A = rec.arrays
    nt, ns = len(tgt.y), len(ys)
    y_all = M.labels(np.r_[tgt.y, ys])
    jobs = [(r, K) for r in rs for K in K_HANDS]

    def mmask(r, K):
        m = np.r_[np.zeros(nt, bool), np.ones(ns, bool)]
        m[split[r][1][K]] = True
        return m

    def tmask(r, K):
        m = np.zeros(nt, bool)
        m[split[r][1][K]] = True
        return m

    def put(h: int, r: int, K: str, L: np.ndarray, extra: dict | None = None) -> None:
        te = split[r][0]
        u = uar_of(L, tgt.y[te])
        ref = ref_e25.get((E25_NAMES[h], sname, K, r))
        rec.row(source=sname, method=f"h{h}", K=K, rep=r, version=j, C=C, C_rule="src", uar=u, ref_uar=ref,
                **(extra or {}))

    need_c = any(h in hands for h in (2, 3, 5))
    Xs_c, L2_all = None, None
    if need_c:
        mean_s = Xs.mean(0)
        Xs_c = Xs - mean_s
    # 手 2（と手 5 の土台）：中心合わせしたソースのみ（E25 と同じく logreg.fit で 1 つの C）
    if 2 in hands or 5 in hands:
        t0 = time.perf_counter()
        ckey = ("centered", key, C)                  # 中心合わせしたソースのみはソースの行だけで決まる（ターゲットをまたいで使い回す）
        if ckey not in src_cache:
            src_cache[ckey] = logreg.fit(Xs_c, M.labels(ys), (C,))
        src_c = src_cache[ckey]
        L2_all = logits_np(src_c, Xt_c, C)
        a, b = raw_coef(src_c, C)
        pre = f"h2_{sname}__v{j}"
        A[f"{pre}_logit"], A[f"{pre}_A"], A[f"{pre}_a0"] = L2_all.astype(np.float32), a, b
        A[f"{pre}_mean_s"] = mean_s.cpu().numpy()
        for r in rs:
            te = split[r][0]
            u = uar_of(L2_all[te], tgt.y[te])
            rec.row(source=sname, method="h2", K="", rep=r, version=j, C=C, C_rule="src", uar=u,
                    ref_uar=ref_e25.get((E25_NAMES[2], sname, "", r)), n_iter=int(src_c.n_iter[C]),
                    grad=float(src_c.grad[C]))
        timing["h2"] += time.perf_counter() - t0
    # 手 4・5：K 本で切片のずらしを解く（CPU．`measure/ablate.fit_class_bias`．l2 は既定の 1e-2）
    for h, base in ((4, Ls_all_C), (5, L2_all)):
        if h not in hands:
            continue
        t0 = time.perf_counter()
        for r in rs:
            te, idx, _ = split[r]
            bs = []
            for K in K_HANDS:
                i = idx[K]
                b = ablate.fit_class_bias(base[i], tgt.y[i])
                bs.append(b)
                put(h, r, K, base[te] + b)
            A[f"h{h}_{sname}_r{r}_b"] = np.stack(bs)
        timing[f"h{h}"] += time.perf_counter() - t0

    def batch(h: int, X: torch.Tensor, y: torch.Tensor, masks: list[np.ndarray], Xte_of, **kw) -> None:
        t0 = time.perf_counter()
        fits = logreg.fit_batch(X, y, masks, (C,), **kw)
        fd = dict(zip(jobs, fits))
        for r in rs:
            Xte = Xte_of(r)
            Ls, As_, a0s = [], [], []
            for K in K_HANDS:
                f = fd[(r, K)]
                L = logits_np(f, Xte, C)
                Ls.append(L)
                a, b = raw_coef(f, C)
                As_.append(a)
                a0s.append(b)
                put(h, r, K, L, {"n_iter": int(f.n_iter[C]), "grad": float(f.grad[C])})
            A[f"h{h}_{sname}_r{r}_logit"] = np.stack(Ls).astype(np.float32)
            A[f"h{h}_{sname}_r{r}_A"] = np.stack(As_)
            A[f"h{h}_{sname}_r{r}_a0"] = np.stack(a0s)
        del fits, fd
        free_gpu()
        timing[f"h{h}"] += time.perf_counter() - t0

    te_rows = {r: split[r][0] for r in rs}
    if 3 in hands:                                    # 中心合わせ＋合併（冷えた出発点．E25 と同じ）
        X = torch.cat([Xt_c, Xs_c])
        batch(3, X, y_all, [mmask(r, K) for r, K in jobs], lambda r: M.rows(Xt_c, te_rows[r]))
        del X
    if 6 in hands:                                    # ターゲット専用の切片つき合併
        one_t = torch.ones(nt, 1, dtype=tgt.X.dtype, device=tgt.X.device)
        zero_s = torch.zeros(ns, 1, dtype=tgt.X.dtype, device=tgt.X.device)
        X = torch.cat([torch.cat([tgt.X, one_t], 1), torch.cat([Xs, zero_s], 1)])
        Xt_a = torch.cat([tgt.X, one_t], 1)
        batch(6, X, y_all, [mmask(r, K) for r, K in jobs], lambda r: M.rows(Xt_a, te_rows[r]))
        del X, Xt_a
    if 7 in hands:                                    # 仮説転移（罰則の中心 ＝ ソースのみの解．C′ ＝ C）
        batch(7, tgt.X, yt, [tmask(r, K) for r, K in jobs], lambda r: M.rows(tgt.X, te_rows[r]),
              anchor=ts.final, anchor_c=C)
    if 8 in hands:                                    # CORAL＋合併
        X = torch.cat([tgt.X, IV.coral(Xs, tgt.X)])
        batch(8, X, y_all, [mmask(r, K) for r, K in jobs], lambda r: M.rows(tgt.X, te_rows[r]))
        del X
    free_gpu()


# ---------------------------------------------------------------- 照合のまとめ

def coef_check(rec: Recorder, tgt: M.Group, reps: list[int], Ks, sources, hands) -> dict:
    """保存した係数（float32）で点数を作り直し，保存した点数との差を測る（CPU）。"""
    A = rec.arrays
    X = tgt.X.cpu().numpy()
    out = {"max_abs": 0.0, "agree_min": 1.0}

    def upd(L_saved, L_new):
        out["max_abs"] = max(out["max_abs"], float(np.abs(L_saved - L_new).max()))
        out["agree_min"] = min(out["agree_min"], float((L_saved.argmax(-1) == L_new.argmax(-1)).mean()))

    for r in reps[:1]:
        te = A[f"test_idx_r{r}"]
        Xte = X[te]
        for kk in range(len(Ks)):
            for jc in range(len(C_GRID)):
                upd(A[f"t_logit_r{r}"][kk, jc], Xte @ A[f"t_A_r{r}"][kk, jc].T.astype(np.float64)
                    + A[f"t_a0_r{r}"][kk, jc])
        for s in sources:
            L = A[f"m_{s}_r{r}_logit"]
            for kk in range(len(Ks)):
                Ls = L[kk] if L.ndim == 3 else L[kk, C_GRID.index(float(A[f"s_{s}__v0_C"]))]
                upd(Ls, Xte @ A[f"m_{s}_r{r}_A"][kk].T.astype(np.float64) + A[f"m_{s}_r{r}_a0"][kk])
            if 3 in hands:
                Xc = Xte - A["mean_t"]
                for kk in range(len(K_HANDS)):
                    upd(A[f"h3_{s}_r{r}_logit"][kk], Xc @ A[f"h3_{s}_r{r}_A"][kk].T.astype(np.float64)
                        + A[f"h3_{s}_r{r}_a0"][kk])
    return out


def summarize(rec: Recorder) -> tuple[pd.DataFrame, dict]:
    """条件ごとの差をまとめる。幅 tol は max(TOL_REPRO, 2 × その反復のテストで 1 本の発話が UAR を動かす最大の量)．
    1 本の発話の分 ustep = 1 / (4 × テストで最も少ない感情の本数)．小さいテスト（AFEW・RAVDESS）では
    argmax が 1 本変わるだけで UAR が 0.007 前後動くので，固定の幅では判定できない（試し 5）。"""
    d = pd.DataFrame(rec.rows)
    d["diff"] = (d["uar"] - pd.to_numeric(d["ref_uar"], errors="coerce")).abs()
    d["ustep"] = d["rep"].astype(int).map(rec.ustep)
    d["tol"] = np.maximum(TOL_REPRO, 2 * d["ustep"])
    summ = {}
    for m, g in d.groupby("method"):
        x = g["diff"].dropna()
        e = {"n": int(len(g)), "n_ref": int(len(x)),
             "max_abs_diff": float(x.max()) if len(x) else None,
             "n_over_exact": int((x > TOL_EXACT).sum()),
             "n_over_tol": int((g["diff"] > g["tol"]).sum()),
             "max_diff_over_tol": float((g["diff"] / g["tol"]).max()) if len(x) else None}
        if "ref_C" in g and g["ref_C"].notna().any():
            gc = g[g["ref_C"].notna()]
            e["C_mismatch"] = int((~np.isclose(gc["C"].astype(float), gc["ref_C"].astype(float))).sum())
        if "ref_rule" in g and g["ref_rule"].notna().any() and m != "merged":
            gr = g[g["ref_rule"].notna()]
            e["rule_mismatch"] = int((gr["C_rule"] != gr["ref_rule"]).sum())
        if "max_abs_diff_allC" in g and g["max_abs_diff_allC"].notna().any():
            a = g["max_abs_diff_allC"].astype(float)
            e["max_abs_diff_allC"] = float(a.max())
            e["n_allC_over_exact"] = int((a > TOL_EXACT).sum())
            e["max_allC_over_tol"] = float((a / g["tol"]).max())
        if "grad" in g and g["grad"].notna().any():
            e["n_not_converged"] = int((g["grad"].astype(float) > logreg.TOL).sum())
        summ[m] = e
    if rec.old_diff:
        o = pd.DataFrame(rec.old_diff)
        summ["old_scores"] = {w: {"n": int(len(g)), "max_abs": float(g.max_abs.max()),
                                  "agree_min": float(g.agree.min())} for w, g in o.groupby("what")}
    return d, summ


def verdict(summ: dict) -> list[str]:
    """再現の検算の判定（設計 2.5 節の基準）。空なら合格。

    ① ② は選ばれた C の UAR，C，規則が地図と 1e-6 で一致すること．選ばれなかった C は，地図の 2026-09-30 の
    実行（未コミットのコードを含む）の弱い罰則（C = 1・10）の解と止める基準の範囲でずれる（試しで最大 0.0069 =
    AFEW の発話 1 本ぶん）ので，幅 tol の 2.5 倍以内とし，ずれた行の数を記録する（C の別の規則の解析は E76 の値を使う）．
    ③ と手は，差が tol を超える条件が 1% 以下で，どの条件も tol の 2.5 倍以内。"""
    bad = []
    for m in ("tgt_only", "src_only"):
        e = summ.get(m, {})
        if e.get("n_ref", 0) and (e.get("n_over_exact", 0) or e.get("C_mismatch", 0) or e.get("rule_mismatch", 0)
                                  or (e.get("max_allC_over_tol") or 0) > 2.5):
            bad.append(f"{m}：地図と一致しない {e}")
    for m, e in summ.items():
        if m in ("tgt_only", "src_only", "old_scores") or not e.get("n_ref"):
            continue
        if e["max_diff_over_tol"] > 2.5 or e["n_over_tol"] > 0.01 * e["n_ref"]:
            bad.append(f"{m}：前の値との差が基準を超える {e}")
    return bad


# ---------------------------------------------------------------- 本体

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS_MAIN))
    ap.add_argument("--targets", nargs="+", help="既定：その kind で使える全群")
    ap.add_argument("--sources", nargs="+", help="既定：ターゲット以外の全群（試しで絞るときだけ使う）")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--K", nargs="+", default=list(K_ALL), choices=K_ALL)
    ap.add_argument("--hands", nargs="*", type=int, default=list(HANDS_ALL), help="E25 の手（空にすると手を回さない）")
    ap.add_argument("--merged-all-c", action="store_true", help="合併を C 4 つすべてで解く（交差検証の地図の合併も発話ごとに読める）")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e76")
    ap.add_argument("--ref-map", type=Path, default=ROOT / "results" / "fixc_map",
                    help="照合する C を固定した地図（<ref-map>/<kind>/<ターゲット>.csv）")
    ap.add_argument("--ref-e25", type=str, default=str(ROOT / "results" / "interventions_{kind}"),
                    help="照合する E25 の結果のフォルダ（{kind} を置き換える）")
    ap.add_argument("--ref-merged-scores", nargs="+",
                    default=[str(ROOT / "results" / "fixc_analysis" / "merged_scores_{kind}"),
                             str(ROOT / "results" / "fixc_text" / "merged_scores_{kind}")],
                    help="照合する前の点数（C を固定した地図の合併とソースのみ）．先に見つかった方を使う")
    ap.add_argument("--ref-tgt-scores", nargs="+", default=[str(ROOT / "results" / "tgt_scores_{kind}")])
    ap.add_argument("--allow-dirty", action="store_true",
                    help="読み込むコードにコミットしていない変更があっても流す（試しだけ．本番では使わない）")
    args = ap.parse_args()

    Ks = tuple(k for k in K_ALL if k in args.K)
    hands = tuple(sorted(set(args.hands)))
    if any(h not in HANDS_ALL for h in hands):
        raise SystemExit(f"手は {HANDS_ALL} から選ぶ")
    if hands and not set(K_HANDS) <= set(Ks):
        raise SystemExit("手を回すときは K に 1・5・10 を含める")
    state = code_state()
    if state["code_dirty"] and not args.allow_dirty:
        raise SystemExit("読み込むコードにコミットしていない変更がある（--allow-dirty は試しだけ）:\n"
                         + "\n".join(state["code_status"]))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = {"reps": args.reps, "K": list(Ks), "hands": list(hands), "merged_all_c": args.merged_all_c,
                "sources": args.sources,
                "C_grid": list(C_GRID), "K_hands": list(K_HANDS), "src_cv_seed": M.SRC_CV_SEED,
                "n_test_folds": M.N_TEST_FOLDS, "n_cv_folds": M.N_CV_FOLDS,
                "solver": {"tol": logreg.TOL, "max_iter": logreg.MAX_ITER, "history": logreg.HISTORY,
                           "dtype": str(logreg.DTYPE)},
                "ablate_l2": 1e-2, "mem_fraction": MEM_FRACTION}
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run = {"script": "exp/relearn_e76.py", "argv": sys.argv[1:], **state, "started": started,
           "kinds": args.kinds, "targets": args.targets, **settings,
           "ref": {"map": str(args.ref_map), "e25": args.ref_e25, "merged_scores": args.ref_merged_scores,
                   "tgt_scores": args.ref_tgt_scores},
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__}
    # 再開：書き始める前に，残っているファイルのコミット番号と設定を全部照合する（混ぜない）
    plan: dict[str, list[str]] = {}
    for kind in args.kinds:
        groups = groups_for(kind)
        targets = args.targets or groups
        bad = [t for t in targets if t not in groups]
        if bad:
            raise SystemExit(f"{kind} で使えない群: {bad}")
        kdir = out / kind
        todo = []
        for t in targets:
            meta_p = kdir / f"{t}.json"
            if (kdir / f"{t}.npz").exists():
                meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
                same = (meta.get("commit") == state["commit"] and meta.get("settings") == settings
                        and not meta.get("code_dirty", True))
                if same and meta.get("check_failed"):
                    raise SystemExit(f"{kdir / t}.npz は再現の検算に落ちている：{meta['check_failed']}．"
                                     f"原因を直してから，消して流し直す")
                if same:
                    print(f"{kind} {t}：済み（同じコミット {state['commit'][:7]}．飛ばす）", flush=True)
                    continue
                raise SystemExit(f"{kdir / t}.npz が別のコミットか設定で作られている（{meta.get('commit')}）．"
                                 f"混ぜないので止める．別の --out にするか，消してから流す")
            todo.append(t)
        plan[kind] = todo

    code_dir = M.snapshot_code(out, started)
    run["code"] = str(code_dir)
    run["plan"] = plan
    prov_path = out / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {"runs": []}
    prov["runs"].append(run)
    prov_path.write_text(json.dumps(prov, ensure_ascii=False, indent=1))

    t_all = time.perf_counter()
    for kind, todo in plan.items():
        if not todo:
            continue
        groups = groups_for(kind)
        kdir = out / kind
        kdir.mkdir(exist_ok=True)
        t0 = time.perf_counter()
        loaded = {g: load_with_ids(g, kind) for g in groups}
        data = {g: v[0] for g, v in loaded.items()}
        print(f"{kind}：読み込み {time.perf_counter() - t0:.0f} 秒", flush=True)
        src_cache: dict = {}
        for t in todo:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            p = args.ref_map / kind / f"{t}.csv"
            ref_map = read_map(p)
            ref_e25 = read_e25(Path(args.ref_e25.format(kind=kind)) / f"{t}.csv")
            old_m = read_old_scores([Path(x.format(kind=kind)) / f"{t}.npz" for x in args.ref_merged_scores])
            old_t = read_old_scores([Path(x.format(kind=kind)) / f"{t}.npz" for x in args.ref_tgt_scores])
            sources = [g for g in groups if g != t and (args.sources is None or g in args.sources)]
            rec = run_target(kind, t, data, loaded[t][1], sources, args.reps, Ks, hands, args.merged_all_c,
                             src_cache, ref_map, ref_e25, old_m, old_t)
            cc = coef_check(rec, data[t], args.reps, Ks, sources, hands)
            rec.arrays["coef_check"] = np.array([cc["max_abs"], cc["agree_min"]])
            d, summ = summarize(rec)
            bad = verdict(summ)
            elapsed = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
            meta = {**state, "settings": settings, "started": started, "kind": kind, "target": t,
                    "sources": sources, "elapsed_sec": elapsed, "timing_sec": rec.timing,
                    "gpu_peak_alloc_gib": peak, "check": summ, "coef_check": cc,
                    "check_failed": bad, "ref": {"map": str(p), "e25": args.ref_e25.format(kind=kind),
                                                 "old_merged": old_m.get("path"), "old_tgt": old_t.get("path")},
                    "keys_doc": KEYS_DOC}
            rec.arrays["_meta"] = np.array(json.dumps({k: meta[k] for k in ("commit", "code_dirty", "settings",
                                                                             "kind", "target", "started")}))
            tmp = kdir / f".{t}.tmp.npz"
            np.savez_compressed(tmp, **rec.arrays)
            os.replace(tmp, kdir / f"{t}.npz")
            with open(kdir / f".{t}.tmp.csv", "w") as f:
                f.write("# " + json.dumps({k: meta[k] for k in ("commit", "code_dirty", "settings", "started")},
                                          ensure_ascii=False) + "\n")
                d.to_csv(f, index=False)
            os.replace(kdir / f".{t}.tmp.csv", kdir / f"{t}.csv")
            if rec.old_diff:
                pd.DataFrame(rec.old_diff).to_csv(kdir / f"{t}_old_scores.csv", index=False)
            (kdir / f"{t}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
            size = (kdir / f"{t}.npz").stat().st_size / 2**20
            print(f"{kind} {t}：{elapsed:.0f} 秒，GPU の最大 {peak:.2f} GiB，npz {size:.1f} MiB", flush=True)
            for m, e in summ.items():
                print(f"    照合 {m}: {e}", flush=True)
            print(f"    係数から作り直した点数との差 最大 {cc['max_abs']:.2e}，argmax の一致 最小 {cc['agree_min']:.4f}",
                  flush=True)
            if bad:
                raise SystemExit("再現の検算が基準を満たさない．止める：\n" + "\n".join(bad))
            free_gpu()
        del data, loaded
        free_gpu()
    print(f"終わり：{out}（{time.perf_counter() - t_all:.0f} 秒）", flush=True)


if __name__ == "__main__":
    main()
