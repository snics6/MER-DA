"""Figures for the Japanese draft ver4 (2026-11-17 night; researcher: "図を作って", one figure = one conclusion,
following Mensh & Kording 2017 rules 7 and 9 and the MIT CommLab results formula).

fig_v4_overview_ja : Fig. 1, overview: pseudo-labels made from the other modalities of the same utterance, combined
                     with equal weights (product) or with the proposal's per-emotion weights, versus the 11 methods
                     that use no pseudo-labels including other modalities (conceptual, no data).
fig_v4_methods_ja  : Fig. 3, audio classifier, distillation - each of the 11 methods (4 corpora x K = 5, 10, 25 =
                     132 comparisons) with 95% intervals. Supports Sect. 4.1.
fig_v4_prop_ja     : Fig. 5, (a) proposal - product (fused predictions) and (b) AVT retrained on the proposal's
                     pseudo-labels - AVT retrained on the product's pseudo-labels, 12 conditions each. Supports
                     Sects. 4.4 and 4.6.

Values:
  Fig. 3: results/dai2/stats/prop_minus_method.csv (student = A, K in 5/10/25, the 11 methods of Table 2;
          columns effect, lo, hi). The script checks 132 rows, 132 positive, 122 with lo > 0, 10 not above
          zero all on IEMOCAP (Sect. 4.1).
  Fig. 5: results/dai20_eval/summary/effects.csv (a = prop, b = prod, seeds = all) and
          results/dai21_eval5/summary/effects.csv (part = 4corpora, a = stud_prop, b = stud_prod, seeds = all).
          The script checks 11 of 12 positive in each panel, and the counts of intervals above zero (7 and 5).
The values drawn are written to ../figures/data/fig_v4.csv.
Run:  /home/snics/MER-DA/.venv/bin/python make_figs_v4.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from make_figs_overview import C_K, C_PROP, C_SRC, C_U, FS, GREY, MM, MOD, W, arrow, box, canvas, save, setup, txt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "figures" / "data"
CORPORA = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
KS = [5, 10, 25]
KMARK = {5: "o", 10: "s", 25: "^"}
METHODS = [("self", "自己学習"), ("entmin", "エントロピー最小化"), ("im", "IM"), ("t3a", "T3A"),
           ("lp", "ラベル伝播"), ("own", "自身の確率"), ("st", "S+T"), ("coral", "CORAL"),
           ("dann", "DANN"), ("mme", "MME"), ("xmuda", "xMUDA 型")]
rows = []


def k_legend(ax, color, loc, ncol=5, extra=True):
    h = [Line2D([], [], ls="", marker=KMARK[k], ms=4.2, mfc=color, mec=color, label=f"K = {k}") for k in KS]
    if extra:
        h += [Line2D([], [], ls="", marker="o", ms=4.2, mfc=color, mec=color, label="区間が 0 より上"),
              Line2D([], [], ls="", marker="o", ms=4.2, mfc="white", mec=color, label="区間が 0 を含む")]
    ax.legend(handles=h, loc=loc, ncol=ncol, frameon=False, fontsize=FS - 0.4, handletextpad=0.2,
              columnspacing=1.0, borderaxespad=0.0)


# ------------------------------------------------------------------ Fig. 3
def fig_methods():
    d = pd.read_csv(ROOT / "results/dai2/stats/prop_minus_method.csv")
    keys = [m for m, _ in METHODS]
    s = d[(d.student == "A") & (d.K.isin(KS)) & (d.method.isin(keys))].copy()
    assert len(s) == 132 and (s.effect > 0).sum() == 132 and (s.lo > 0).sum() == 122
    assert set(s[~(s.lo > 0)]["T"]) == {"IEMOCAP"} and (~(s.lo > 0)).sum() == 10
    setup("ja")
    xmax, xmin = 16.0, -5.0
    fig, axes = plt.subplots(1, 4, figsize=(W * MM, 92 * MM), sharey=True)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.86, bottom=0.12, wspace=0.08)
    col = MOD["A"][0]
    for ax, c in zip(axes, CORPORA):
        ax.axvline(0, color="black", lw=0.6, zorder=1)
        for i, (m, _) in enumerate(METHODS):
            for j, K in enumerate(KS):
                r = s[(s["T"] == c) & (s.method == m) & (s.K == K)].iloc[0]
                y = i + (j - 1) * 0.26
                lo, hi, e = r.lo, r.hi, r.effect
                rows.append(dict(fig="fig3", corpus=c, method=m, K=K, effect=e, lo=lo, hi=hi))
                ax.plot([max(lo, xmin), min(hi, xmax)], [y, y], color=col, lw=0.7, zorder=2, solid_capstyle="butt")
                filled = lo > 0
                if e <= xmax:
                    ax.plot(e, y, ls="", marker=KMARK[K], ms=3.6, mfc=col if filled else "white", mec=col,
                            mew=0.8, zorder=3)
                else:
                    ax.plot(xmax, y, ls="", marker=">", ms=3.6, mfc=col if filled else "white", mec=col, mew=0.8,
                            zorder=3, clip_on=False)
            big = s[(s["T"] == c) & (s.method == m)].effect
            if big.max() > xmax:
                ax.text(xmax - 0.4, i - 0.55, f"+{big.min():.0f}〜+{big.max():.0f}", ha="right", va="center",
                        fontsize=FS - 1.4, color=GREY)
        ax.set_xlim(xmin, xmax)
        ax.set_title(c if c != "IEMOCAP" else "IEMOCAP（即興）", fontsize=FS + 0.4, pad=3)
        ax.set_xticks([0, 5, 10, 15])
        ax.tick_params(labelsize=FS - 0.6, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
    axes[0].set_yticks(range(len(METHODS)))
    axes[0].set_yticklabels([n for _, n in METHODS], fontsize=FS - 0.2)
    axes[0].invert_yaxis()
    fig.text(0.575, 0.02, "蒸留 − 比べる手法（声の分類器の UAR の差，点）", ha="center", fontsize=FS)
    k_legend(axes[0], col, loc="lower left")
    axes[0].get_legend().set_bbox_to_anchor((0.0, 1.10), transform=axes[0].transAxes)
    save(fig, "fig_v4_methods_ja")


# ------------------------------------------------------------------ Fig. 5
def fig_prop():
    a = pd.read_csv(ROOT / "results/dai20_eval/summary/effects.csv")
    a = a[(a.a == "prop") & (a.b == "prod") & (a.seeds == "all")]
    b = pd.read_csv(ROOT / "results/dai21_eval5/summary/effects.csv")
    b = b[(b.part == "4corpora") & (b.a == "stud_prop") & (b.b == "stud_prod") & (b.seeds == "all")]
    for x, n_above in ((a, 7), (b, 5)):
        assert len(x) == 12 and (x.est > 0).sum() == 11 and (x.lo > 0).sum() == n_above, (len(x), (x.lo > 0).sum())
    setup("ja")
    fig, axes = plt.subplots(1, 2, figsize=(W * MM, 62 * MM), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.12, wspace=0.06)
    titles = ["(a) まとめた予測：提案 − 積", "(b) 学び直した AVT：提案の疑似ラベル − 積の疑似ラベル"]
    for ax, x, t, tag in zip(axes, (a, b), titles, ("fig5a", "fig5b")):
        ax.axhline(0, color="black", lw=0.6, zorder=1)
        for ci, c in enumerate(CORPORA):
            for j, K in enumerate(KS):
                r = x[(x.c == c) & (x.K == K)].iloc[0]
                xx = ci + (j - 1) * 0.24
                rows.append(dict(fig=tag, corpus=c, K=K, effect=r.est, lo=r.lo, hi=r.hi))
                ax.plot([xx, xx], [r.lo, r.hi], color=C_PROP, lw=0.8, zorder=2)
                ax.plot(xx, r.est, ls="", marker=KMARK[K], ms=4.0, mfc=C_PROP if r.lo > 0 else "white", mec=C_PROP,
                        mew=0.8, zorder=3)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["AFEW", "IEMOCAP\n（即興）", "MOSEI", "MELD"], fontsize=FS - 0.2)
        ax.set_xlim(-0.6, 3.6)
        ax.set_title(t, fontsize=FS + 0.2, loc="left", pad=3)
        ax.tick_params(labelsize=FS - 0.6, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", lw=0.5, zorder=0)
    axes[0].set_ylabel("UAR の差（点）", fontsize=FS)
    k_legend(axes[0], C_PROP, loc="lower left")
    axes[0].get_legend().set_bbox_to_anchor((0.0, 1.13), transform=axes[0].transAxes)
    save(fig, "fig_v4_prop_ja")


# ------------------------------------------------------------------ Fig. 1
def fig_overview():
    setup("ja")
    H = 80
    fig, ax = canvas(H, W)
    fs = FS
    NAME = {"A": "声", "V": "顔", "T": "言葉"}
    # column 1: inputs
    txt(ax, 1, H - 1, "使えるデータ", size=fs, weight="bold", color=GREY)
    box(ax, 1, 56, 36, 13, fc="#EDE7F3", ec=C_SRC, lw=0.9)
    txt(ax, 3, 67, "ソース（ほかの\nコーパス）の\nラベル付きの発話", size=fs - 0.3, color=C_SRC)
    box(ax, 1, 25, 36, 24, fc="white", ec=C_U, lw=0.8, ls="--")
    txt(ax, 3, 47, "ターゲット（新しい\nコーパス）の\nラベル無しの発話", size=fs - 0.3, color="#444444")
    for k, m in enumerate("AVT"):
        ax.plot(6 + k * 10, 31.5, marker=MOD[m][1], ms=4, color=MOD[m][0], ls="")
        txt(ax, 8.2 + k * 10, 33.2, NAME[m], size=fs - 0.4, color=MOD[m][0])
    box(ax, 1, 6, 36, 12, fc="white", ec=C_K, lw=0.9)
    txt(ax, 3, 16, "ターゲットの\n感情ごとに K 本のラベル", size=fs - 0.3, color=C_K)

    # top lane: the 11 methods
    box(ax, 44, 63, 84, 15, fc="#F4F4F4", ec="#9A9A9A", lw=0.8)
    txt(ax, 46, 76, "比べる 11 の手法（自己学習，CORAL，DANN，MME など）", size=fs - 0.2, weight="bold", color=GREY)
    txt(ax, 46, 70.6, "ほかのモダリティを含む疑似ラベルを使わずに，ラベル無しの\n発話（声の分類器なら声）やソースで分類器をターゲットに合わせる",
        size=fs - 1.0, color=GREY, ls=1.15)
    arrow(ax, (37, 44), (44, 66), color=C_U, lw=0.7, head=6, ls="--")

    # column 2: classifiers trained on the sources
    box(ax, 44, 22, 34, 36, fc="white", ec="#9A9A9A", lw=0.7)
    txt(ax, 46, 56.5, "ソースで学んだ分類器", size=fs - 0.4, color=GREY)
    for k, m in enumerate("AVT"):
        y = 48 - k * 8
        box(ax, 47, y - 3, 28, 6.2, fc="white", ec=MOD[m][0], lw=0.9)
        txt(ax, 61, y + 0.1, NAME[m] + "の分類器", ha="center", va="center", size=fs - 0.4, color=MOD[m][0])
    txt(ax, 46, 27.6, "ラベル無しの発話ごとに\n4 感情の確率を付ける", size=fs - 1.3, color=GREY, ls=1.15)
    arrow(ax, (37, 61), (44, 52), color=C_SRC, lw=0.8, head=6)
    arrow(ax, (37, 37), (44, 37), color=C_U, lw=0.8, head=6, ls="--")

    # column 3: how the three probabilities are combined into pseudo-labels
    txt(ax, 84, 59.5, "疑似ラベルの作り方", size=fs - 0.2, weight="bold", color=GREY)
    box(ax, 84, 44, 44, 12, fc="white", ec=GREY, lw=0.8)
    txt(ax, 86, 54, "積（同じ重み）", size=fs - 0.2, weight="bold", color=GREY)
    txt(ax, 86, 49.4, "3 つの確率を同じ重みで掛ける", size=fs - 1.0, color=GREY)
    box(ax, 84, 4, 44, 36, fc="#FFF4EC", ec=C_PROP, lw=1.2)
    txt(ax, 86, 38, "提案（感情ごとの重み）", size=fs - 0.2, weight="bold", color=C_PROP)
    txt(ax, 86, 33.4, "分類器 × 感情の重みでまとめる", size=fs - 1.0, color=C_PROP)
    gx, gy, cw, chh = 92, 8.5, 6.0, 3.6
    vals = np.array([[0.9, 1.1, 1.0, 0.8], [1.3, 1.0, 0.9, 1.2], [0.8, 0.9, 1.1, 1.0]])
    for r_, m in enumerate("AVT"):
        txt(ax, gx - 1.0, gy + (2 - r_) * chh + chh / 2 - 0.2, NAME[m], ha="right", va="center", size=fs - 1.4,
            color=MOD[m][0])
        for c_ in range(4):
            v = vals[r_, c_]
            ax.add_patch(plt.Rectangle((gx + c_ * cw, gy + (2 - r_) * chh), cw - 0.4, chh - 0.4,
                                       fc=plt.cm.Oranges(0.15 + 0.55 * (v - 0.8) / 0.5), ec="none"))
    for c_, e in enumerate(["N", "H", "S", "A"]):
        txt(ax, gx + c_ * cw + cw / 2 - 0.2, gy - 0.4, e, ha="center", size=fs - 1.6, color=GREY)
    txt(ax, 86, 29.0, "強さ：ターゲットの K 本で測る", size=fs - 1.1, color=C_K)
    txt(ax, 86, 25.0, "感情ごとの形：ソースで測る", size=fs - 1.1, color=C_SRC)
    txt(ax, 126.5, 18.5, "重みは説明\nのための例", ha="right", va="bottom", size=fs - 1.8, color=GREY)
    arrow(ax, (78, 48), (84, 50), color=GREY, lw=0.8, head=6)
    arrow(ax, (78, 32), (84, 26), color=GREY, lw=0.8, head=6)
    arrow(ax, (37, 12), (84, 14), color=C_K, lw=0.8, head=6)

    # column 4: retrain and compare
    box(ax, 134, 22, 38, 36, fc="white", ec="black", lw=1.0)
    txt(ax, 136, 56, "学び直して比べる", size=fs - 0.2, weight="bold")
    txt(ax, 136, 50.5, "K 本と疑似ラベルで\n分類器を学び直す", size=fs - 0.6, ls=1.15)
    txt(ax, 136, 40.5, "テストの発話の UAR を\n比べる", size=fs - 0.6, color=GREY, ls=1.15)
    txt(ax, 136, 29.6, "まとめた予測そのものの\nUAR も比べる", size=fs - 1.2, color=GREY, ls=1.15)
    arrow(ax, (128, 50), (134, 48), color=GREY, lw=0.8, head=6)
    arrow(ax, (128, 22), (134, 30), color=C_PROP, lw=1.2, head=7)
    arrow(ax, (128, 70), (153, 58), color="#9A9A9A", lw=0.8, head=6)
    txt(ax, 172, 2.5, "全体図（模式図）", ha="right", va="bottom", size=fs - 1.4, color=GREY)
    save(fig, "fig_v4_overview_ja")


if __name__ == "__main__":
    fig_methods()
    fig_prop()
    fig_overview()
    pd.DataFrame(rows).to_csv(DATA / "fig_v4.csv", index=False)
    print("wrote", DATA / "fig_v4.csv", len(rows))
