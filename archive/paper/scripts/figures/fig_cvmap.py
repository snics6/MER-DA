"""Fig. S3 cvmap：交差検証で合併の C を選んだ地図の得（補足資料）．

本文の地図（fig_gainmap，合併の C ＝ ソースのみの C）と同じ形で，合併の C を K 本の交差検証で選んだ地図の
得（合併 − ターゲットのみ）の組の平均を K に対して描き，C を固定した地図の線を薄く重ねる．
入力：results/map_<kind>/summary/pairs.csv（交差検証の地図．音声 2 種類・V・AV は 30 回，言葉を含む 4 種類は 5 回）と
      results/fixc_map/<kind>/summary/pairs.csv

    ../.venv/bin/python scripts/figures/fig_cvmap.py      # paper/ で
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import style  # noqa: E402

KS = ["1", "5", "10", "25", "50", "all"]
KLAB = ["1", "5", "10", "25", "50", "All"]


def mean_gain(mapdir: str, k: str):
    """mapdir は "map" か "fixc_map"．組の平均の得（ポイント）を K ごとに返す．"""
    kind = style.CONFIG[k]["kind"]
    base = style.RESULTS / (f"map_{kind}" if mapdir == "map" else f"fixc_map/{kind}")
    p = pd.read_csv(base / "summary" / "pairs.csv", dtype={"K": str, "source": str})
    return [p[p.K == K].diff_mean.mean() * 100 for K in KS]


def main() -> None:
    style.setup()
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE, 2.1), sharey=True)
    x = range(len(KS))
    for ax, ks, title in ((axes[0], ["a", "ah", "v", "av"], "(a)"),
                          (axes[1], ["t", "vt", "at", "avt"], "(b)")):
        for k in ks:
            c = style.CONFIG[k]
            ax.plot(x, mean_gain("map", k), color=c["color"], marker=c["marker"], label=c['label'])
            ax.plot(x, mean_gain("fixc_map", k), color=c["color"], lw=0.8, ls=":", alpha=0.8)
        ax.axhline(0, color=style.INK, lw=0.9)
        ax.set_xticks(list(x)); ax.set_xticklabels(KLAB)
        ax.set_xlabel("Target labels per emotion, $K$")
        ax.text(0.01, 0.97, title, transform=ax.transAxes, va="top", fontsize=7, fontweight="bold")
        ax.legend(loc="upper right", fontsize=6.5, ncol=2)
    axes[0].set_ylabel("Gain of pooling (UAR points)")
    fig.tight_layout()
    style.save(fig, "cvmap")


if __name__ == "__main__":
    main()
