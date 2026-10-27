"""Fig. gainmap：C を固定した地図の得（合併 − ターゲットのみ）．

(a) 組の平均の得を K に対して 8 種類の線で（台帳 map-gain-<kind>-k<K> と同じ値）
(b) A，V，AV，T について，得の 95% 区間が 0 を上回る組・0 をまたぐ組・0 を下回る組の割合
    （台帳 map-gain-pos-*・map-gain-neg-* と同じ数え方．区間は summary/pairs.csv の diff_lo・diff_hi）

入力：results/fixc_map/<kind>/summary/pairs.csv（exp/summarize_map.py の出力．同じコーパスの組を含む全組，各組 30 回）

    ../.venv/bin/python scripts/figures/fig_gainmap.py      # paper/ で
"""
from __future__ import annotations

import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import style  # noqa: E402

KS = ["1", "5", "10", "25", "50", "all"]
KLAB = ["1", "5", "10", "25", "50", "All"]
BAR_KINDS = ["a", "v", "av", "t"]
# (b) の 3 色：緑系・灰・赤系を避け，style の色から選ぶ（青＝上回る，黄の薄い色＝またぐ，橙＝下回る）
C_POS, C_MID, C_NEG = style.SLOT[0], style.SLOT[3], style.SLOT[1]


def load(k: str) -> pd.DataFrame:
    kind = style.CONFIG[k]["kind"]
    p = pd.read_csv(style.RESULTS / "fixc_map" / kind / "summary" / "pairs.csv", dtype={"K": str, "source": str})
    p["gain"] = p["diff_mean"] * 100
    return p


def main() -> None:
    style.setup()
    data = {k: load(k) for k in style.CONFIG_ORDER}

    fig = plt.figure(figsize=(style.PAGE, 2.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.0], wspace=0.15,
                          left=0.068, right=0.998, bottom=0.2, top=0.9)
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[1])

    # ---- (a) 組の平均の得
    x = np.arange(len(KS))
    ax.axhline(0, color=style.INK, lw=0.9, zorder=2)
    for k in style.CONFIG_ORDER:
        c = style.CONFIG[k]
        g = data[k].groupby("K")["gain"].mean().reindex(KS)
        ax.plot(x, g.to_numpy(), color=c["color"], marker=c["marker"], ms=3.6, lw=1.2,
                label=c["label"], zorder=3, markeredgewidth=0.4, markeredgecolor="white")
    ax.set_xticks(x, KLAB)
    ax.set_xlim(-0.3, len(KS) - 0.7)
    ax.set_yticks([-5, 0, 5, 10])
    ax.set_xlabel("Target labels per emotion, $K$")
    ax.set_ylabel("Gain of pooling (UAR points)")
    ax.legend(ncol=2, loc="upper right", handlelength=1.6, columnspacing=0.9, labelspacing=0.3,
              borderaxespad=0.2)
    ax.text(-0.17, 1.0, "(a)", transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")

    # ---- (b) 区間の符号の割合
    gap = 0.9
    w = 0.78
    xt, xl, centers, names = [], [], [], []
    for gi, k in enumerate(BAR_KINDS):
        p = data[k]
        xs = gi * (len(KS) + gap) + np.arange(len(KS))
        pos, mid, neg = [], [], []
        for K in KS:
            q = p[p["K"] == K]
            n = len(q)
            a = (q["diff_lo"] > 0).sum() / n
            b = (q["diff_hi"] < 0).sum() / n
            pos.append(a)
            neg.append(b)
            mid.append(1 - a - b)
        pos, mid, neg = map(np.array, (pos, mid, neg))
        bx.bar(xs, pos, w, color=C_POS, lw=0)
        bx.bar(xs, mid, w, bottom=pos, color=C_MID, alpha=0.35, lw=0)
        bx.bar(xs, neg, w, bottom=pos + mid, color=C_NEG, lw=0)
        xt += list(xs)
        xl += KLAB
        centers.append(xs.mean())
        npairs = p.groupby("K").size().unique()
        assert len(npairs) == 1
        names.append(f"{style.CONFIG[k]['label']} ({npairs[0]} pairs)")
    bx.set_xticks(xt, xl, fontsize=6.5)
    bx.tick_params(axis="x", length=0, pad=1.5)
    for cx, nm in zip(centers, names):
        bx.text(cx, -0.13, nm, transform=bx.get_xaxis_transform(), ha="center", va="top")
    bx.set_ylim(0, 1)
    bx.set_xlim(-0.6, xt[-1] + 0.6)
    bx.set_ylabel("Fraction of ordered pairs")
    bx.grid(axis="x", visible=False)
    bx.set_axisbelow(True)
    bx.spines["bottom"].set_visible(False)
    handles = [Patch(color=C_POS, label="95% interval above 0"),
               Patch(color=C_MID, alpha=0.35, label="Interval contains 0"),
               Patch(color=C_NEG, label="Interval below 0")]
    bx.legend(handles=handles, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=1.2,
              columnspacing=1.2, borderaxespad=0.1)
    bx.text(-0.075, 1.0, "(b)", transform=bx.transAxes, fontweight="bold", va="bottom", ha="left")
    bx.set_xlabel(" ", labelpad=6)   # 種類の名前の場所を空ける

    style.save(fig, "gainmap")


if __name__ == "__main__":
    main()
