"""【探索的】E32 の 5 節 段 2 の解析：下書きを取り替えた仮説転移（`exp/combine_drafts.py`）と P4 の比べ（予言 D1）。新しい学習なし。

規則は `docs/exploratory-analyses.md` E32 の 5 節「段 2 の作り方と予言」。P4 は E31 の既定（`results/compare_<kind>/<ターゲット>.csv`
の HT_ALL，cen = 1，C′ = 0.01，C_b = 8）。単位はターゲット（反復 0〜9 の平均）。価値は 4 節と同じく，ターゲットのみの学習曲線
（地図の tgt_only，K = 1〜50）で読んだ本数から K を引いた値で，段 1 の予言（`e32d_targets.csv` の n_K）と比べる。

    ./.venv/bin/python exp/explore_combine.py [--in-root results]

出力：`results/explore_relation/e32e_targets.csv`（種類 × ターゲット × K × 下書き），`e32e_tests.csv`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import explore_compare as EC  # noqa: E402
import explore_source_value as SV  # noqa: E402

PRED_HALF = {"audio_wavlm": 1.1, "audio_hubert": 0.8}       # D1：段 1 の予言（K=5．W7 +2.2，Hu7 +1.6）の半分
DRAFTS = ["equal", "count", "value"]


def load_kind(kind: str, in_root: Path) -> pd.DataFrame:
    fs = sorted((in_root / f"combine_{kind}").glob("*.csv"))
    if not fs:
        return pd.DataFrame()
    a = pd.concat([pd.read_csv(f, dtype={"K": str}) for f in fs])
    a = a[a.rep.isin(EC.REPS)]
    T = a.pivot_table(index=["target", "K"], columns="source", values="uar", aggfunc="mean") * 100
    n_rep = a.groupby(["target", "K", "source"]).rep.nunique().unstack().min(axis=1).rename("n_rep")
    c = pd.concat([pd.read_csv(ROOT / "results" / f"compare_{kind}" / f"{t}.csv", dtype={"K": str, "cb": str})
                   for t in T.index.get_level_values("target").unique()])
    p4 = c[(c.method == "HT_ALL") & (c.cen == 1) & (c.cp == 0.01) & (c.cb == "8") & c.rep.isin(a.rep.unique())]
    T["P4"] = p4.groupby(["target", "K"]).uar.mean() * 100
    T = T.join(n_rep).reset_index()
    T.insert(0, "kind", kind)
    cur = SV.curves(kind, sorted(T.target.unique()))
    pred = pd.read_csv(SV.OUT / "e32d_targets.csv").query("kind == @kind")
    for d in DRAFTS + ["P4"]:
        if d not in T:
            continue
        nf = [SV.n_of(cur[t], u, int(K)) for t, K, u in zip(T.target, T.K, T[d])]
        T[f"n_{d}"] = [n for n, _ in nf]
        T[f"flag_{d}"] = [f for _, f in nf]
    for d, src in (("equal", "equal"), ("count", "count"), ("value", "value"), ("P4", "pooled")):
        m = pred[pred.draft == src].set_index("target")
        T[f"npred_{d}"] = [float(m.loc[t, f"n_K{K}"]) if t in m.index else np.nan for t, K in zip(T.target, T.K)]
    return T


def tests(T: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (kind, K), g in T.groupby(["kind", "K"]):
        row = {"kind": kind, "K": K, "targets": len(g), "n_rep_min": int(g.n_rep.min())}
        for d in DRAFTS:
            if d not in g or g[d].isna().all():
                continue
            diff = g[d] - g.P4
            row[f"{d}-P4"] = float(diff.mean())
            row[f"{d}_wins"] = int((diff > 0).sum())
            row[f"{d}_p_greater"] = float(stats.wilcoxon(diff, alternative="greater").pvalue) if len(g) >= 5 and diff.abs().sum() > 0 else np.nan
        out.append(row)
    R = pd.DataFrame(out)
    for kind, half in PRED_HALF.items():
        m = (R.kind == kind) & R.K.isin(["5", "10"])
        if m.sum() == 2 and "equal-P4" in R:
            R.loc[R.kind == kind, "D1_kind"] = bool((R[m]["equal-P4"] >= half).all())
    return R


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-root", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    T = pd.concat([load_kind(k, args.in_root) for k in SV.MAIN], ignore_index=True)
    R = tests(T)
    T.to_csv(SV.OUT / "e32e_targets.csv", index=False)
    R.to_csv(SV.OUT / "e32e_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    for kind, g in T.groupby("kind"):
        for d in DRAFTS + ["P4"]:
            if f"n_{d}" not in g:
                continue
            ok = (g[f"flag_{d}"] == 0) & g[f"npred_{d}"].notna() & g.K.isin(["5", "10"])
            if ok.any():
                print(f"{kind} {d}：測った価値 ÷ 段 1 の予言の中央値 {np.median(g[ok][f'n_{d}'] / g[ok][f'npred_{d}']):.2f}（{int(ok.sum())} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
