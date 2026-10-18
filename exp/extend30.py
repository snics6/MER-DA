"""【探索的】P4 を出発点にした実験を反復 30 回に延ばした結果の判定（CPU）。

規則は `docs/exploratory-analyses.md` の「P4 を出発点にした実験を反復 30 回に延ばす」（判定の基準は計算の前に書いた）。
反復 0〜9（もとの出力）と反復 10〜29（`_r30` の出力）を合わせ，ターゲットの平均で比べる。

- D1：`results/combine_<kind>/`（source=equal）と `results/combine_<kind>_r30/`
- P4：`results/compare_<kind>/<T>.csv`（反復 0〜9）と `<T>_final.csv`（反復 10〜29）の HT_ALL，cen=1，C′=0.01，C_b=8
- E39 の 5 つの数の版：`results/meta5_<kind>/` と `results/meta5_<kind>_r30/` の role=target のうち，
  `results/explore_relation/e39_targets.csv` でターゲットごとに選んだ組
- teach（`e37_targets.csv` の policy_star の歩幅），herding，herding_feat，random：`results/select_teach_<kind>/` と `_r30/`

    ./.venv/bin/python exp/extend30.py

出力：`results/explore_relation/ext30_targets.csv`（ターゲット × K × 反復の範囲），`ext30_tests.csv`（種類 × K の差と判定）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REL = ROOT / "results" / "explore_relation"
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
KS = ["1", "5", "10", "25"]
SPANS = {"all": range(30), "r0_9": range(10), "r10_29": range(10, 30)}
D1_MIN = {"audio_wavlm": 1.1, "audio_hubert": 0.8}                      # D1 の基準（E32 の 5 節：段 1 の予言の半分）
KEYS = ["gamma", "rho_s", "cb", "cp", "rho_M"]


def read(paths, **kw) -> pd.DataFrame:
    fs = [p for p in paths if p.exists()]
    return pd.concat([pd.read_csv(p, **kw) for p in fs]) if fs else pd.DataFrame()


def per_target(kind: str, T: str, sel39: pd.Series, star37: str) -> pd.DataFrame:
    """1 つのターゲットについて，手法ごと・反復・K の UAR（%）を縦に並べる。"""
    res = ROOT / "results"
    out = []
    d1 = read([res / f"combine_{kind}" / f"{T}.csv", res / f"combine_{kind}_r30" / f"{T}.csv"], dtype={"K": str})
    d1 = d1[d1.source == "equal"]
    out.append(d1.assign(method="D1")[["method", "rep", "K", "uar"]])
    p4 = read([res / f"compare_{kind}" / f"{T}.csv", res / f"compare_{kind}" / f"{T}_final.csv"], dtype={"K": str, "cb": str})
    p4 = p4[(p4.method == "HT_ALL") & (p4.cen == 1) & (p4.cp == 0.01) & (p4.cb == "8")]
    out.append(p4.assign(method="P4")[["method", "rep", "K", "uar"]])
    m5 = read([res / f"meta5_{kind}" / f"{T}.csv", res / f"meta5_{kind}_r30" / f"{T}.csv"], dtype={"K": str, "cb": str})
    m5 = m5[m5.role == "target"]
    hit = (m5.gamma == sel39.gamma) & (m5.rho_s == sel39.rho_s) & (m5.cb == str(sel39.cb)) & (m5.cp == sel39.cp) & (m5.rho_M == sel39.rho_M)
    out.append(m5[hit].assign(method="E39")[["method", "rep", "K", "uar"]])
    st = read([res / f"select_teach_{kind}" / f"{T}.csv", res / f"select_teach_{kind}_r30" / f"{T}.csv"], dtype={"K": str})
    st = st[st.role == "target"]
    for pol, name in ((star37, "teach"), ("herding", "herding"), ("herding_feat", "herding_feat"), ("random", "random")):
        out.append(st[st.policy == pol].assign(method=name)[["method", "rep", "K", "uar"]])
    a = pd.concat(out)
    dup = a.duplicated(["method", "rep", "K"])
    if dup.any():
        raise ValueError(f"{kind} {T}：同じ手法・反復・K の行が重なる（{int(dup.sum())} 行）")
    return a.assign(kind=kind, target=T, uar=a.uar * 100)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=MAIN)
    ap.add_argument("--suffix", default="", help="入力の e39・e37 の表と出力の名前の接尾辞（言葉系は _text）")
    ap.add_argument("--out-suffix", help="出力の名前の接尾辞だけを変える（既定は --suffix と同じ）")
    args = ap.parse_args()
    sx = args.suffix
    ox = sx if args.out_suffix is None else args.out_suffix
    e39 = pd.read_csv(REL / ("e39_targets.csv" if not sx else f"e39{sx}_targets.csv"), dtype={"K": str, "cb": str})
    e37 = pd.read_csv(REL / f"e37_targets{sx}.csv", dtype={"K": str})
    rows = []
    for kind in args.kinds:
        for T in sorted(e39[e39.kind == kind].target.unique()):
            sel39 = e39[(e39.kind == kind) & (e39.target == T)].iloc[0]
            star37 = e37[(e37.kind == kind) & (e37.target == T)].policy_star.iloc[0]
            rows.append(per_target(kind, T, sel39, star37))
    A = pd.concat(rows)
    tabs = []
    for span, reps in SPANS.items():
        a = A[A.rep.isin(list(reps))]
        g = a.groupby(["kind", "target", "K", "method"]).agg(uar=("uar", "mean"), n_rep=("rep", "nunique")).reset_index()
        tabs.append(g.assign(span=span))
    D = pd.concat(tabs)
    W = D.pivot_table(index=["span", "kind", "target", "K"], columns="method", values="uar").reset_index()
    N = D.pivot_table(index=["span", "kind", "target", "K"], columns="method", values="n_rep").reset_index()
    pairs = {"D1-P4": ("D1", "P4"), "herding-random": ("herding", "random"), "herding_feat-random": ("herding_feat", "random"),
             "herding-herding_feat": ("herding", "herding_feat"), "teach-herding": ("teach", "herding"),
             "teach-random": ("teach", "random"), "E39-D1": ("E39", "D1"), "E39-P4": ("E39", "P4")}
    tests = []
    for (span, kind, K), g in W.groupby(["span", "kind", "K"]):
        n = N[(N.span == span) & (N.kind == kind) & (N.K == K)]
        row = {"span": span, "kind": kind, "K": K, "targets": len(g), "min_reps": int(n.drop(columns=["span", "kind", "target", "K"]).min().min())}
        for m in ("P4", "D1", "E39", "random", "herding_feat", "herding", "teach"):
            row[m] = float(g[m].mean())
        for name, (x, y) in pairs.items():
            diff = g[x] - g[y]
            row[name] = float(diff.mean())
            row[f"{name}_wins"] = int((diff > 0).sum())
        tests.append(row)
    R = pd.DataFrame(tests)
    R["K"] = pd.Categorical(R.K, KS, ordered=True)
    R = R.sort_values(["span", "kind", "K"]).reset_index(drop=True)
    verdict = []
    for span in SPANS:
        for kind in [k for k in ("audio_wavlm", "audio_hubert") if k in args.kinds]:   # 言葉系は記述だけ（判定しない）
            r = R[(R.span == span) & (R.kind == kind)].set_index("K")
            v = {"span": span, "kind": kind,
                 "D1": bool(all(r.loc[K, "D1-P4"] >= D1_MIN[kind] for K in ("5", "10"))),
                 "G1": bool(all(r.loc[K, "herding-random"] > 0 for K in ("1", "5", "10"))),
                 "F1_E33": bool(all(r.loc[K, "herding_feat-random"] > 0 and r.loc[K, "herding-herding_feat"] > 0 for K in ("5", "10"))),
                 "T1": bool(all(r.loc[K, "teach-herding"] > 0 for K in ("5", "10"))),
                 "T2": bool(all(r.loc[K, "teach-random"] > 0 for K in ("1", "5", "10"))),
                 "F1_E39": bool(all(r.loc[K, "E39-D1"] > 0 for K in ("5", "10")))}
            verdict.append(v)
    V = pd.DataFrame(verdict)
    W.to_csv(REL / f"ext30_targets{ox}.csv", index=False)
    R.to_csv(REL / f"ext30_tests{ox}.csv", index=False)
    V.to_csv(REL / f"ext30_verdict{ox}.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["span", "kind", "K", "min_reps", "D1-P4", "herding-random", "herding_feat-random", "herding-herding_feat",
            "teach-herding", "teach-random", "E39-D1", "E39-P4"]
    print(R[cols].round(2).to_string(index=False))
    print(V.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
