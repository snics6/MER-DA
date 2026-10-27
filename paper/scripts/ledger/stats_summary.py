"""台帳のモジュール（stats_summary）：6 節の検定の要約（有意なセルの数，効果量の範囲）．

入力は exp/stats_compare.py（E64）の出力 results/stats_compare/tests.csv．比較 (a)（oursx − 最も強い相手の調整版）の
11 の生徒（AVT の 4 生徒，AV・VT・AT の融合，第 2 エンコーダの世界の 4 生徒）× K ∈ {0, 5} を数える．
"""
from __future__ import annotations

import pandas as pd

from .common import entry, RESULTS

F = RESULTS / "stats_compare" / "tests.csv"
SRC = "results/stats_compare/tests.csv"


def entries() -> dict[str, dict]:
    t = pd.read_csv(F, comment="#", dtype={"K": str}, keep_default_na=False)
    a = t[(t.experiment.isin(["E61", "E62"])) & (t.comparison == "a")].copy()
    for c in ("p_holm", "p_dedup_holm", "dz", "mean"):
        a[c] = pd.to_numeric(a[c])
    out = {}
    for K in ("0", "5"):
        g = a[a.K == K]
        if len(g) != 11:
            raise ValueError(f"stats_summary: 比較 (a) の K={K} のセルが {len(g)} 個（11 のはず）")
        base = f"{SRC} | 比較 (a) oursx − 最も強い相手の調整版，K={K}，11 の生徒（E61 の 7，E62 の 4），ターゲット × 反復の対"
        out[f"st-a-k{K}-cells-n"] = entry(len(g), base + " | セルの数", "%d")
        out[f"st-a-k{K}-pos-n"] = entry(int((g["mean"] > 0).sum()), base + " | 差の平均が正のセルの数", "%d")
        out[f"st-a-k{K}-sig-n"] = entry(int((g.p_holm < 0.05).sum()), base + " | Holm 後の p（族 F-a，55 検定）が 0.05 未満のセルの数", "%d")
        out[f"st-a-k{K}-sigdedup-n"] = entry(int((g.p_dedup_holm < 0.05).sum()),
                                             base + " | 同じテストの話者の集合を持つ反復を平均した単位での Holm 後の p（p_dedup_holm）が 0.05 未満のセルの数", "%d")
        out[f"st-a-k{K}-dz-lo"] = entry(g.dz.min(), base + " | 効果量 d_z（差の平均 ÷ 差の標準偏差）の最小", "%.2f")
        out[f"st-a-k{K}-dz-hi"] = entry(g.dz.max(), base + " | 効果量 d_z の最大", "%.2f")
    # ターゲットごとの oursx − E39（比較 (b)）：出発点を下回るターゲットのあるセルの数と，最も大きい損
    b = t[(t.experiment.isin(["E61", "E62"])) & (t.comparison == "b") & (t.K.isin(["0", "5"]))]
    if len(b) != 22:
        raise ValueError(f"stats_summary: 比較 (b) の K=0・5 のセルが {len(b)} 個（22 のはず）")
    vals = []
    ncell = 0
    for s in b.tgt_means:
        v = [float(x.rsplit(" ", 1)[1]) for x in s.split("；")]
        vals += v
        ncell += any(x < 0 for x in v)
    srcb = f"{SRC} | 比較 (b) oursx − E39，K ∈ {{0, 5}}，11 の生徒 × 2 の 22 セル，列 tgt_means（ターゲットごとに反復を平均した差）"
    out["st-b-tgtneg-cells-n"] = entry(ncell, srcb + " | 出発点を下回るターゲットが 1 つ以上あるセルの数", "%d")
    out["st-b-tgtneg-worst"] = entry(min(vals), srcb + " | ターゲットごとの差の最小（ポイント）", "%.1f")
    out["st-b-tgt-n"] = entry(len(vals), srcb + " | ターゲットとセルの組の数", "%d")
    out["st-b-tgtneg-n"] = entry(sum(x < 0 for x in vals), srcb + " | 差が負のターゲットとセルの組の数", "%d")
    return out
