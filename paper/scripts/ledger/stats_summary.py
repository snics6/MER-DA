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
    return out
