"""E84：ソース × 感情の対 × モダリティ．ソースだけで学んだ分類器の方向は，どのターゲットのどの感情の対で使えるか．

設計：docs/exploratory-analyses.md の E84（規則は値を見る前に書いた）．新しい学習はしない．
  (A) kind × ターゲット × 対 × ソースの a = 100·Φ(d/2)（切片を直した後の 2 感情の正解率）．全発話と，話者の半分ごと（seed 0〜4）
  (B) 対ごとに最良のソースを選ぶ余地 G：半分 A で選び，半分 B で測る（入れ替えて平均）
  (C) 世界 AV・AVT で，対ごとに最良のモダリティと，融合に対する単モダリティの余地 R
  (D) 記述：音声の s* をコーパスの種類で集計

入力は results/shift_<kind>/logits/<ソース>.npz の t_<ターゲット>（E13 と同じ）．ラベルと話者は measure/data.load_any で読む．

使い方：./.venv/bin/python exp/analyze_source_pairs_e84.py [--out results/e84] [--kinds audio_wavlm ...] [--seeds 5] [--smoke]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_pairs import ACTED, NATURAL, PAIR_NAMES, PAIRS, ba  # noqa: E402
from split import speaker_folds  # noqa: E402

KINDS_MAIN = ("audio_wavlm", "video", "text_roberta")
KINDS_ROBUST = ("audio_hubert",)
WORLDS = {"AV": (("audio_wavlm", "video"), "fuse-audio-video"),
          "AVT": (("audio_wavlm", "video", "text_roberta"), "fuse-audio-video-text")}
E13 = ROOT / "results" / "explore_relation" / "snapshots" / "20261002-1339_e13" / "e13_pairs.csv"
ANGER = [PAIR_NAMES[i] for i, (k, j) in enumerate(PAIRS) if 3 in (k, j)]                      # 怒りを含む 3 対（P84-1）
AUDIO_WEAK = [PAIR_NAMES[i] for i, (k, j) in enumerate(PAIRS) if (k, j) in ((1, 3), (0, 2))]  # 喜び–怒り，平静–悲しみ（P84-2）
TOL = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def corp(g):
    return GROUP_SPEC[g][0]


def ctype(g):
    return "acted" if g in ACTED else ("natural" if g in NATURAL else "other")


def sources_for(kind, t):
    return [s for s in groups_for(kind) if s != t and corp(s) != corp(t)]


def pair_rows(L, y, mask=None):
    """6 対の d，o，a，d_auc，a_auc，発話数．mask は使う発話（None なら全部）．"""
    out = []
    for p, (k, j) in enumerate(PAIRS):
        mk = (y == k) if mask is None else (y == k) & mask
        mj = (y == j) if mask is None else (y == j) & mask
        zk = L[mk, k] - L[mk, j]
        zj = L[mj, k] - L[mj, j]
        r = dict(pair=PAIR_NAMES[p], n_k=int(mk.sum()), n_l=int(mj.sum()))
        if len(zk) < 2 or len(zj) < 2:
            r.update(d=np.nan, o=np.nan, a=np.nan, d_auc=np.nan, a_auc=np.nan)
        else:
            sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
            d = (zk.mean() - zj.mean()) / sd
            o = -(zk.mean() + zj.mean()) / 2 / sd
            auc = roc_auc_score(np.r_[np.ones(len(zk)), np.zeros(len(zj))], np.r_[zk, zj])
            auc = float(np.clip(auc, 1e-6, 1 - 1e-6))
            d_auc = np.sqrt(2) * stats.norm.ppf(auc)
            r.update(d=d, o=o, a=100 * ba(d, 0), d_auc=d_auc, a_auc=100 * ba(d_auc, 0))
        out.append(r)
    return out


def load_logits(kind, s, t):
    z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
    return z[f"t_{t}"]


# ---------------------------------------------------------------- (A)(B)(D)：kind ごと

def cells_for_kind(kind, seeds):
    rows = []
    for t in groups_for(kind):
        df, _ = load_any(t, kind)
        y = df.label_4.to_numpy()
        halves = {sd: speaker_folds(df.speaker_group, k=2, seed=sd).to_numpy() for sd in seeds}
        for s in sources_for(kind, t):
            L = load_logits(kind, s, t)
            if len(L) != len(y):
                raise SystemExit(f"{kind} {s}→{t}：点数 {len(L)} 行と発話 {len(y)} 本が合わない")
            for r in pair_rows(L, y):
                rows.append(dict(kind=kind, target=t, source=s, split="full", **r))
            for sd in seeds:
                for h in (0, 1):
                    for r in pair_rows(L, y, halves[sd] == h):
                        rows.append(dict(kind=kind, target=t, source=s, split=f"s{sd}h{h}", **r))
    return pd.DataFrame(rows)


def check_e13(cells):
    """全発話の d が E13 の保存物と一致するか（kind が共通の行）．"""
    if not E13.exists():
        return {"checked": False, "reason": "e13_pairs.csv が無い"}
    e13 = pd.read_csv(E13, comment="#")
    full = cells[cells.split == "full"][["kind", "target", "source", "pair", "d"]]
    m = full.merge(e13[["kind", "target", "source", "pair", "d"]], on=["kind", "target", "source", "pair"], suffixes=("", "_e13"))
    if len(m) == 0:
        return {"checked": False, "reason": "共通の行が無い"}
    diff = float((m.d - m.d_e13).abs().max())
    if diff > TOL:
        raise SystemExit(f"E13 の d と合わない：最大の差 {diff:.3g} > {TOL}（{len(m)} 行）")
    return {"checked": True, "rows": int(len(m)), "max_abs_diff": diff}


def best_sources(a):
    """a: DataFrame(source, pair, a)．対ごとの最良 s* と 6 対の平均で最良 s°．6 対がそろわないソースは s° から外す．"""
    P = a.pivot(index="source", columns="pair", values="a")
    star = pd.Series({p: (P[p].idxmax() if P[p].notna().any() else None) for p in P.columns}, dtype=object)  # pair → source
    mean = P.mean(axis=1, skipna=False)
    circ = mean.idxmax() if mean.notna().any() else None
    return P, star, circ


def gain_rows(cells, kind):
    rows = []
    c = cells[cells.kind == kind]
    for t, ct in c.groupby("target"):
        full = ct[ct.split == "full"]
        P, star, circ = best_sources(full[["source", "pair", "a"]])
        g_in = np.nan if circ is None else float(np.mean([P.loc[star[p], p] - P.loc[circ, p] for p in P.columns]))
        n_distinct = int(star.dropna().nunique())
        seeds = sorted({int(sp[1]) for sp in ct.split.unique() if sp.startswith("s")})
        for sd in seeds:
            Ph = {}
            for h in (0, 1):
                Ph[h] = ct[ct.split == f"s{sd}h{h}"][["source", "pair", "a"]].pivot(index="source", columns="pair", values="a")
            gs = {}
            for hs, he in ((0, 1), (1, 0)):
                _, st, ci = best_sources(Ph[hs].stack().rename("a").reset_index())
                if ci is None or st.isna().any():
                    gs[f"G_{hs}{he}"] = np.nan
                    continue
                vals = [Ph[he].loc[st[p], p] - Ph[he].loc[ci, p] for p in Ph[he].columns]
                gs[f"G_{hs}{he}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            rows.append(dict(kind=kind, target=t, unit=corp(t), seed=sd, G_in=g_in, n_distinct_star=n_distinct,
                             G=np.nanmean([gs["G_01"], gs["G_10"]]), **gs))
    return pd.DataFrame(rows)


def delta_for(cells, kind, seed=0):
    c = cells[(cells.kind == kind) & cells.split.isin((f"s{seed}h0", f"s{seed}h1"))]
    P = c.pivot_table(index=["target", "source", "pair"], columns="split", values="a")
    if P.shape[1] < 2:
        return np.nan
    return float((P.iloc[:, 0] - P.iloc[:, 1]).abs().median())


def best_table(cells, kind):
    rows = []
    c = cells[(cells.kind == kind) & (cells.split == "full")]
    for t, ct in c.groupby("target"):
        P, star, circ = best_sources(ct[["source", "pair", "a"]])
        Pa, star_auc, _ = best_sources(ct[["source", "pair", "a_auc"]].rename(columns={"a_auc": "a"}))
        for p in P.columns:
            rows.append(dict(kind=kind, target=t, unit=corp(t), target_type=ctype(t), pair=p,
                             s_star=star[p], s_star_type=ctype(star[p]) if isinstance(star[p], str) else None,
                             a_star=P.loc[star[p], p] if isinstance(star[p], str) else np.nan,
                             s_circ=circ, a_circ=P.loc[circ, p] if circ else np.nan,
                             gap=(P.loc[star[p], p] - P.loc[circ, p]) if (circ and isinstance(star[p], str)) else np.nan,
                             star_ne_circ=(star[p] != circ) if (circ and isinstance(star[p], str)) else np.nan,
                             s_star_auc=star_auc[p], agree_auc=(star[p] == star_auc[p]),
                             n_sources=int(P[p].notna().sum()), anger_pair=p in ANGER))
    return pd.DataFrame(rows)


def summarize_kind(best, gain, delta):
    """単位（コーパス）ごとに G を平均し，判定 R1 の材料を出す．"""
    per_unit = gain.groupby(["kind", "unit", "seed"]).G.mean().groupby(["kind", "unit"]).agg(["mean", "min", "max"]).reset_index()
    per_unit = per_unit.rename(columns={"mean": "G_mean", "min": "G_min", "max": "G_max"})
    g_in = gain.groupby(["kind", "unit"]).G_in.mean().reset_index()
    per_unit = per_unit.merge(g_in)
    kind = per_unit.kind.iloc[0]
    n = len(per_unit)
    n_pos = int((per_unit.G_mean > 0).sum())
    n_ge = int((per_unit.G_mean >= delta).sum())
    G_med = float(per_unit.G_mean.median())
    r1 = bool(G_med >= delta and n_pos > n / 2)
    # P84-1：s* ≠ s° の割合（怒りの対 対 その他）．単位は (コーパス, 対)．IEMOCAP は 2 群の平均
    b = best.groupby(["unit", "pair", "anger_pair"]).star_ne_circ.mean().reset_index()
    p1 = b.groupby("anger_pair").star_ne_circ.mean()
    agree = float(best.agree_auc.mean())
    # 偶然の見込み：ソースが入れ替え可能なら s* ≠ s° は (n_sources − 1)/n_sources で起きる
    ns = best.n_sources[best.n_sources > 0]
    chance = float((1 - 1 / ns).mean()) if len(ns) else np.nan
    summ = dict(kind=kind, n_units=n, delta=delta, G_median=G_med, n_units_G_pos=n_pos, n_units_G_ge_delta=n_ge,
                G_in_median=float(per_unit.G_in.median()), R1_room=r1,
                p1_share_ne_anger=float(p1.get(True, np.nan)), p1_share_ne_other=float(p1.get(False, np.nan)),
                p1_chance_share_ne=chance, agree_auc_share=agree)
    return per_unit, summ


# ---------------------------------------------------------------- (C)：世界ごとのモダリティ

def world_rows(world, seeds):
    singles, fused = WORLDS[world]
    kinds = tuple(singles) + (fused,)
    rows, drows = [], []
    for t in groups_for(fused):
        dfs = {k: load_any(t, k)[0] for k in kinds}
        common = None
        for k in kinds:
            ids = set(dfs[k].sample_id)
            common = ids if common is None else common & ids
        common = sorted(common)
        idx = {k: pd.Index(dfs[k].sample_id).get_indexer(common) for k in kinds}
        if any((ix < 0).any() for ix in idx.values()):
            raise SystemExit(f"{world} {t}：sample_id の対応が取れない")
        base = dfs[fused].iloc[idx[fused]].reset_index(drop=True)
        y = base.label_4.to_numpy()
        for k in kinds:
            if not np.array_equal(dfs[k].iloc[idx[k]].label_4.to_numpy(), y):
                raise SystemExit(f"{world} {t} {k}：ラベルが融合と合わない")
        halves = {sd: speaker_folds(base.speaker_group, k=2, seed=sd).to_numpy() for sd in seeds}
        per_kind = {}
        for k in kinds:
            kc = []
            for s in sources_for(k, t):
                L = load_logits(k, s, t)
                if len(L) != len(dfs[k]):
                    raise SystemExit(f"{k} {s}→{t}：点数 {len(L)} 行と発話 {len(dfs[k])} 本が合わない")
                L = L[idx[k]]
                for r in pair_rows(L, y):
                    kc.append(dict(split="full", source=s, **r))
                for sd in seeds:
                    for h in (0, 1):
                        for r in pair_rows(L, y, halves[sd] == h):
                            kc.append(dict(split=f"s{sd}h{h}", source=s, **r))
            per_kind[k] = pd.DataFrame(kc)
            for r in kc:
                drows.append(dict(world=world, target=t, kind=k, **r))
        # 対ごとの a：全発話（s° を全発話で選ぶ）と，半分で選んで半分で測る（seed ごと）
        for k in kinds:
            c = per_kind[k]
            P, _, circ = best_sources(c[c.split == "full"][["source", "pair", "a"]])
            for p in P.columns:
                rows.append(dict(world=world, target=t, unit=corp(t), seed=-1, kind=k, pair=p, s_circ=circ,
                                 a=P.loc[circ, p] if circ else np.nan, n_common=len(common)))
            for sd in seeds:
                Ph = {h: c[c.split == f"s{sd}h{h}"][["source", "pair", "a"]].pivot(index="source", columns="pair", values="a") for h in (0, 1)}
                ci = {h: best_sources(Ph[h].stack().rename("a").reset_index())[2] for h in (0, 1)}
                for p in P.columns:
                    vals = [Ph[he].loc[ci[hs], p] for hs, he in ((0, 1), (1, 0)) if ci[hs] is not None]
                    rows.append(dict(world=world, target=t, unit=corp(t), seed=sd, kind=k, pair=p, s_circ=f"{ci[0]}|{ci[1]}",
                                     a=float(np.mean(vals)) if vals else np.nan, n_common=len(common)))
    return pd.DataFrame(rows), pd.DataFrame(drows)


def summarize_world(world, mrows, mcells):
    singles, fused = WORLDS[world]
    best_rows = []
    # δ：世界の全 kind の seed 0 の半分どうしの差の中央値（共通の発話の上）
    c0 = mcells[mcells.split.isin(("s0h0", "s0h1"))]
    P0 = c0.pivot_table(index=["target", "kind", "source", "pair"], columns="split", values="a")
    delta = float((P0.iloc[:, 0] - P0.iloc[:, 1]).abs().median()) if P0.shape[1] == 2 else np.nan
    for (t, sd), g in mrows.groupby(["target", "seed"]):
        P = g.pivot(index="kind", columns="pair", values="a")
        for p in P.columns:
            col = P[p]
            single_best = col.loc[list(singles)].idxmax() if col.loc[list(singles)].notna().any() else None
            m_star = col.idxmax() if col.notna().any() else None
            best_rows.append(dict(world=world, target=t, unit=corp(t), seed=sd, pair=p, m_star=m_star,
                                  single_best=single_best, a_single_best=col.loc[single_best] if single_best else np.nan,
                                  a_fused=col.get(fused, np.nan),
                                  room=max(0.0, col.loc[single_best] - col[fused]) if (single_best and np.isfinite(col.get(fused, np.nan))) else np.nan,
                                  audio_weak_pair=p in AUDIO_WEAK))
    B = pd.DataFrame(best_rows)
    Bo = B[B.seed >= 0]
    per_unit = Bo.groupby(["unit", "seed"]).room.mean().groupby("unit").agg(["mean", "min", "max"]).reset_index()
    per_unit = per_unit.rename(columns={"mean": "R_mean", "min": "R_min", "max": "R_max"}).assign(world=world)
    r_in = B[B.seed < 0].groupby("unit").room.mean().rename("R_in").reset_index()
    per_unit = per_unit.merge(r_in)
    n = len(per_unit)
    n_ge = int((per_unit.R_mean >= delta).sum())
    # m* の seed 平均（out-of-sample）：(単位, 対) ごとに，各 kind が最良になった割合
    share = Bo.assign(is_audio=Bo.m_star == "audio_wavlm").groupby(["unit", "pair", "audio_weak_pair"]).is_audio.mean().reset_index()
    p2 = share.groupby("audio_weak_pair").is_audio.mean()
    n_kinds = len(singles) + 1
    summ = dict(world=world, n_units=n, delta=delta, R_median=float(per_unit.R_mean.median()), n_units_R_ge_delta=n_ge,
                R2_room=bool(n_ge > n / 2), R_in_median=float(per_unit.R_in.median()),
                p2_share_audio_best_weak=float(p2.get(True, np.nan)), p2_share_audio_best_other=float(p2.get(False, np.nan)),
                p2_chance_share=1 / n_kinds)
    return B, per_unit, summ


# ---------------------------------------------------------------- main

def write(df, path, header):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {header}\n")
        df.to_csv(fh, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e84")
    ap.add_argument("--kinds", nargs="*", default=list(KINDS_MAIN + KINDS_ROBUST))
    ap.add_argument("--worlds", nargs="*", default=list(WORLDS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="試走（seed 1 つ，音声だけ，世界は AVT だけ．出力は --out か results/e84_smoke）")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.kinds, args.worlds = 1, ["audio_wavlm"], ["AVT"]
        if args.out == "results/e84":
            args.out = "results/e84_smoke"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "exp", "measure"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    header = f"E84 commit={commit} dirty_exp_measure={bool(dirty)} seeds={args.seeds}"
    prov = {"commit": commit, "dirty_exp_measure": bool(dirty), "script_sha256": sha256(Path(__file__)),
            "args": vars(args) | {"out": str(out)}, "inputs": {}, "checks": {}}

    # (A)(B)(D)
    cells = pd.concat([cells_for_kind(k, seeds) for k in args.kinds], ignore_index=True)
    prov["checks"]["e13"] = check_e13(cells)
    if E13.exists():
        prov["inputs"]["e13_pairs.csv"] = sha256(E13)
    for k in args.kinds:
        for s in groups_for(k):
            prov["inputs"][f"shift_{k}/logits/{s}.npz"] = sha256(ROOT / "results" / f"shift_{k}" / "logits" / f"{s}.npz")
    write(cells, out / "cells.csv", header)
    bests, gains, units, summs = [], [], [], []
    for k in args.kinds:
        b, g = best_table(cells, k), gain_rows(cells, k)
        dl = delta_for(cells, k)
        u, sm = summarize_kind(b, g, dl)
        bests.append(b)
        gains.append(g)
        units.append(u)
        summs.append(sm)
    write(pd.concat(bests, ignore_index=True), out / "best.csv", header)
    write(pd.concat(gains, ignore_index=True), out / "gain.csv", header)
    write(pd.concat(units, ignore_index=True), out / "gain_units.csv", header)
    write(pd.DataFrame(summs), out / "summary_kind.csv", header)
    # (D) 記述：音声の s* のコーパスの種類
    ba_ = pd.concat(bests, ignore_index=True)
    if "audio_wavlm" in args.kinds:
        d = ba_[ba_.kind == "audio_wavlm"].groupby(["target_type", "pair"]).apply(
            lambda g: pd.Series(dict(n=len(g), share_star_acted=(g.s_star_type == "acted").mean(),
                                     share_star_natural=(g.s_star_type == "natural").mean(),
                                     share_circ_acted=(g.s_circ.map(ctype) == "acted").mean()))).reset_index()
        write(d, out / "source_type_audio.csv", header)

    # (C)
    msum = []
    for w in args.worlds:
        mrows, mcells = world_rows(w, seeds)
        B, pu, sm = summarize_world(w, mrows, mcells)
        write(mcells, out / f"modality_cells_{w}.csv", header)
        write(mrows, out / f"modality_a_{w}.csv", header)
        write(B, out / f"modality_best_{w}.csv", header)
        write(pu, out / f"modality_units_{w}.csv", header)
        msum.append(sm)
    if msum:
        write(pd.DataFrame(msum), out / "summary_world.csv", header)

    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out, "| E13 check:", prov["checks"]["e13"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
