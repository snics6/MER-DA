"""Fig. lambda：合併が判断基準のずれを直す割合 λ の予言と実測（E15，C を固定した段階 B）．

横軸は E15 の式の λ = ρ/(1 + ρ)（反復 0〜9 の C で計算して平均．lam_pred_rep09），縦軸は合併の分類器で測った λ
（|o_s| ≥ 0.3 の感情の対で，|o_s| で重みをつけた 1 − o_m/o_s の平均．lam_obs）．1 点が 組 × K（K = 1，5，10）．
凡例の値は種類ごとのスピアマンの順位相関（台帳 p1-e15-lam-rho-<kind> と同じ計算）．

入力：results/fixc_analysis/explore_relation/snapshots/20261009-0105_fixc/stageB/e15b_pairs.csv

    ../.venv/bin/python scripts/figures/fig_lambda.py      # paper/ で
"""
from __future__ import annotations

import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import style  # noqa: E402
from ledger import part1 as P1  # noqa: E402

KINDS = ("a", "ah", "v", "av")


def main() -> None:
    style.setup()
    e = pd.read_csv(P1.STAGEB / "e15b_pairs.csv", dtype={"K": str})
    fig, ax = plt.subplots(figsize=(style.COL, 2.3))
    fig.subplots_adjust(left=0.135, right=0.67, bottom=0.16, top=0.98)
    lo, hi = -0.22, 1.06
    ax.plot([lo, hi], [lo, hi], color=style.INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for k in KINDS:
        c = style.CONFIG[k]
        g = e[e.kind == c["kind"]]
        rho = P1._spearman(g.lam_obs, g.lam_pred_rep09)
        ax.scatter(g.lam_pred_rep09, g.lam_obs, s=7, marker=c["marker"], color=c["color"], alpha=0.75,
                   lw=0.25, edgecolor="white", zorder=3, label=f"{c['label']}: {rho:.2f}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks, ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks(ticks, ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel(r"Predicted $\lambda = \rho/(1+\rho)$")
    ax.set_ylabel(r"Measured $\lambda$ of the pooled classifier")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.3, borderaxespad=0.0,
              labelspacing=0.5, markerscale=1.4, title="Spearman correlation", title_fontsize=6.5,
              alignment="left")
    style.save(fig, "lambda")


if __name__ == "__main__":
    main()
