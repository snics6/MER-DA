"""Fig. thresholdshare：合併の上乗せのうち判断基準の分の割合（E29）と，K 本で減る 2 つの誤差の理論の曲線（E28-9）．

(a) share_pos = pos/(pos + dir) の組の中央値を K = 1，5，10 で，8 種類の線で（台帳 p1-e29-critshare-k<K>-<kind>）．
    pos は合併がソースのみから判断基準 o を動かした分，dir は向きを動かした分（2 つの山の模型の対の正解率の点）．
    同じコーパスの組を除く（A・A(HuBERT) 40 組，V・AV 28 組，言葉を含む 4 種類 10 組）
(b) K 本の推定の誤差：判断基準の分散 1/(2K) と，向きの雑音と信号の比 q/K（q は q_dir のターゲットの中央値．
    台帳 p1-qdir-<kind>）．どちらも同じ係数 (Δ/4)φ(Δ/2) を掛けると対の正解率の期待の損になる

入力：results/explore_relation/snapshots/20261011-1524_e29/e29_pairs.csv，
      results/explore_relation/snapshots/20261011-0139_e28t9/e28t9_targets.csv

    ../.venv/bin/python scripts/figures/fig_thresholdshare.py      # paper/ で
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
from matplotlib.lines import Line2D  # noqa: E402

import style  # noqa: E402
from ledger import part1 as P1  # noqa: E402

KS = ["1", "5", "10"]
SINGLE = ("a", "ah", "v")          # 主張の範囲（単独の音声と顔）
QKINDS = ("a", "ah", "av", "v")    # q_dir のある 4 種類


def repel(ys, min_gap):
    """縦の位置 ys（表示の単位）を，並び順を保って min_gap 以上離す．"""
    order = np.argsort(ys)
    y = np.asarray(ys, float)[order].copy()
    for _ in range(200):
        moved = False
        for i in range(1, len(y)):
            if y[i] - y[i - 1] < min_gap:
                push = (min_gap - (y[i] - y[i - 1])) / 2
                y[i - 1] -= push
                y[i] += push
                moved = True
        if not moved:
            break
    out = np.empty_like(y)
    out[order] = y
    return out


def label_right(ax, x_end, y_end, texts, colors, x_text, min_gap_pt=7.2, log=False):
    """線の右端に名前を付ける．重なるときは縦にずらし，細い引き出し線でつなぐ．"""
    fig = ax.figure
    fig.canvas.draw()
    to_disp = ax.transData.transform
    disp = np.array([to_disp((x_end, y))[1] for y in y_end])
    gap = min_gap_pt * fig.dpi / 72
    new = repel(disp, gap)
    inv = ax.transData.inverted()
    for y0, yd, t, c in zip(y_end, new, texts, colors):
        yt = inv.transform((0, yd))[1]
        ax.annotate(t, xy=(x_end, y0), xytext=(x_text, yt), textcoords="data", ha="left", va="center",
                    color=c, fontsize=6.5, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=c, shrinkA=0.5, shrinkB=2.5))


def main() -> None:
    style.setup()
    p = pd.read_csv(P1.E29 / "e29_pairs.csv", dtype={"K": str})
    T = pd.read_csv(P1.E28T9 / "e28t9_targets.csv")
    q = {k: float(T[T.kind == style.CONFIG[k]["kind"]].q_dir.median()) for k in QKINDS}

    fig = plt.figure(figsize=(style.COL, 3.0))
    gs = fig.add_gridspec(2, 1, hspace=0.5, left=0.16, right=0.78, bottom=0.1, top=0.95)
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[1])

    # ---- (a) 割合の中央値
    x = np.arange(len(KS))
    ax.axhline(0.5, color=style.INK, lw=0.8, zorder=2)
    ends, labs, cols = [], [], []
    for k in style.CONFIG_ORDER:
        c = style.CONFIG[k]
        g = p[p.kind == c["kind"]]
        med = [float(g[g.K == K].share_pos.median()) for K in KS]
        solid = k in SINGLE
        ax.plot(x, med, color=c["color"], marker=c["marker"], ms=3.6 if solid else 3.0,
                lw=1.5 if solid else 0.9, ls="-" if solid else (0, (3, 1.5)), zorder=4 if solid else 3,
                markeredgewidth=0.4, markeredgecolor="white")
        ends.append(med[-1])
        labs.append(c["label"])
        cols.append(c["color"])
    ax.set_xticks(x, KS)
    ax.set_xlim(-0.15, len(KS) - 0.85)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Target labels per emotion, $K$")
    ax.set_ylabel("Threshold share of the\ngain over source-only")
    label_right(ax, x[-1], ends, labs, cols, x[-1] + 0.2)
    handles = [Line2D([], [], color=style.INK2, lw=1.5, label="Single A or V"),
               Line2D([], [], color=style.INK2, lw=0.9, ls=(0, (3, 1.5)), label="Fused, or with T")]
    ax.legend(handles=handles, loc="upper right", handlelength=2.0, borderaxespad=0.2)
    ax.text(-0.3, 1.03, "(a)", transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")

    # ---- (b) 理論の曲線
    K = np.geomspace(1, 50, 200)
    bx.plot(K, 1 / (2 * K), color=style.INK, lw=1.4, zorder=4)
    ends, labs, cols = [], [], []
    for k in ("a", "ah", "av", "v"):
        c = style.CONFIG[k]
        bx.plot(K, q[k] / K, color=c["color"], lw=1.2, zorder=3)
        mk = np.array([1, 5, 10, 25, 50])
        bx.plot(mk, q[k] / mk, ls="none", marker=c["marker"], ms=3.0, color=c["color"],
                markeredgewidth=0.4, markeredgecolor="white", zorder=3)
        ends.append(q[k] / K[-1])
        labs.append(c["label"])
        cols.append(c["color"])
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(1, 50)
    bx.set_ylim(0.005, 40)
    bx.set_xticks([1, 5, 10, 25, 50], ["1", "5", "10", "25", "50"])
    bx.minorticks_off()
    bx.set_xlabel("Target labels per emotion, $K$")
    bx.set_ylabel("Error of the estimate\nfrom $K$ labels")
    label_right(bx, 50, ends, labs, cols, 50 * 1.15, log=True)
    bx.text(1.08, 0.055, "Threshold, $1/(2K)$", ha="left", va="top", color=style.INK)
    bx.text(48, 12, "Direction, $q/K$\n(equals $q$ at $K=1$)", ha="right", va="center", color=style.INK2,
            linespacing=1.1)
    bx.text(-0.3, 1.03, "(b)", transform=bx.transAxes, fontweight="bold", va="bottom", ha="left")

    style.save(fig, "thresholdshare")


if __name__ == "__main__":
    main()
