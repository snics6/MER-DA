"""Figures for the co-researcher slides v8 (2026-11-16 night, researcher's requests):

fig_pl_flow_ja        : how the pseudo-labels are made and used (Sec. 4.4), in the style of the pipeline figure
fig_setting_single_ja : single-source setting (Sec. 5.1-5.3), shown just before its results
fig_setting_all_ja    : all-source setting (Sec. 5.4)
fig_setting_loco_ja   : LOCO setting (Sec. 6, 7)
fig_margin_def_ja     : what the "margin" (当たる幅) and the gain are, with two measured examples (Sec. 6.6)
fig_margin_counts_ja  : the 360 points per K split by the sign of the margin and by the gain interval (Sec. 6.6)
fig_D_def_ja          : how D is measured, and D of the voice vs. the face features for the text judgments (Sec. 6.5)
fig_reverse_ja        : D of the original voice, the reversed voice and the face features (Sec. 6.5, experiment 23)

Values:
  margin examples and counts : ../figures/data/fig10_teacher_margin.csv (the data of Fig. 8)
  D values                   : results/dai23/summary/main.csv (subset = all, K = 5): DAT (original voice),
                               DBT (reversed voice), L3_B (reversed voice - face); face = DBT - L3_B
The pseudo-label flow uses made-up probabilities (marked as an example in the figure).
Output: ../figures/<name>.pdf and .png; the values used are written to ../figures/data/fig_slides_v8.csv.
Run:  /home/snics/MER-DA/.venv/bin/python make_figs_slides.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_figs_overview import C_K, C_PROP, C_SRC, C_U, FS, GREY, MOD, W, arrow, box, canvas, mark, save, setup, txt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "figures" / "data"
C_UP = "#0072B2"
C_DOWN = "#D55E00"
C_NS = "#C8C8C8"
EMO = ["N", "H", "S", "A"]
rows = []   # values written to fig_slides_v8.csv


def minibars(ax, x, y, w, h, vals, color, labels=True, fs=FS - 1.6, show=False):
    """4 bars (N, H, S, A) in the box (x, y, w, h); y is the bottom."""
    n = len(vals)
    bw = w / n * 0.62
    for i, v in enumerate(vals):
        cx = x + (i + 0.5) * w / n
        ax.add_patch(__import__("matplotlib").patches.Rectangle((cx - bw / 2, y), bw, h * v, fc=color, ec="none", zorder=4))
        if labels:
            txt(ax, cx, y - 0.6, EMO[i], ha="center", va="top", size=fs, color=GREY)
        if show:
            txt(ax, cx, y + h * v + 0.5, f"{v:.2f}", ha="center", va="bottom", size=fs - 0.2)
    ax.plot([x, x + w], [y, y], color=GREY, lw=0.4, zorder=4)


# ------------------------------------------------------------------ pseudo-label flow (Sec. 4.4)
def fig_pl_flow():
    setup("ja")
    H = 80
    fig, ax = canvas(H)
    fs = FS + 0.6
    xs = [1, 33, 68, 106, 139]
    ws = [30, 33, 36, 31, 34]
    heads = ["① ターゲットの候補", "② 1 段目：AVT の分類器", "③ 2 段目：声・顔・言葉", "④ 積：確率を掛ける", "⑤ 学び直して比べる"]
    hcol = ["#EEEEEE", "#F2F2F2", "#E3EEF7", "#FDF0E8", "#ECE6F2"]
    ecol = ["#777777", "#444444", "#2F6FA6", C_PROP, C_SRC]
    top = H - 2
    for x, w, h, fc, ec in zip(xs, ws, heads, hcol, ecol):
        box(ax, x, 9, w, top - 9, fc=fc, ec=ec, lw=0.9, r=1.2)
        box(ax, x, top - 6.5, w, 6.5, fc=ec, ec=ec, lw=0.9, r=1.0, z=2)
        txt(ax, x + w / 2, top - 3.25, h, ha="center", va="center", size=fs - 0.6, color="white", weight="bold")
    # (1) candidates
    x, w = xs[0], ws[0]
    box(ax, x + 2, top - 22, w - 4, 12, fc="white", ec=C_K, lw=1.1, z=3)
    txt(ax, x + 3.5, top - 11.2, "K 本\n感情ごとに K 本．\nラベルを使う", size=fs - 0.8, color=C_K, ls=1.15)
    box(ax, x + 2, top - 42, w - 4, 17, fc="white", ec=C_U, lw=1.0, ls="--", z=3)
    txt(ax, x + 3.5, top - 26.2, "ほかの候補\nラベルは使わない\n（疑似ラベルを\n付ける相手）", size=fs - 0.8, color="#444444", ls=1.15)
    txt(ax, x + 2, top - 46, "候補 ＝ テスト以外の\nターゲットの発話．\nテストの発話は\nどの段でも使わない", size=fs - 1.6, color=GREY, ls=1.15)
    # (2) AVT stage 1
    x, w = xs[1], ws[1]
    box(ax, x + 2, top - 24, w - 4, 14, fc="white", ec="black", lw=1.0, z=3)
    txt(ax, x + 3.5, top - 11.2, "K 本の基準の\nAVT の分類器\n（アンカーから K 本\nで学び直したもの）", size=fs - 0.9, ls=1.15)
    txt(ax, x + 2.5, top - 27, "候補の発話ごとに\n4 感情の確率を付ける", size=fs - 0.9, ls=1.15)
    avt = [0.25, 0.45, 0.15, 0.15]
    minibars(ax, x + 6, top - 52, w - 12, 18, avt, "#555555", show=True)
    # (3) three classifiers
    x, w = xs[2], ws[2]
    txt(ax, x + 2.5, top - 8.5, "それぞれをアンカーから，\nK 本 ＋ ② の確率で学び直し，\n候補に確率を付ける", size=fs - 1.2, ls=1.15)
    probs = {"A": [0.40, 0.35, 0.15, 0.10], "V": [0.20, 0.60, 0.10, 0.10], "T": [0.30, 0.50, 0.10, 0.10]}
    names = {"A": "声", "V": "顔", "T": "言葉"}
    for r, m in enumerate("AVT"):
        y = top - 33 - r * 15
        col = MOD[m][0]
        box(ax, x + 2, y - 4, w - 4, 13.5, fc="white", ec=col, lw=1.0, z=3)
        mark(ax, x + 4.6, y + 6.3, m, size=3.6)
        txt(ax, x + 7, y + 6.3, names[m] + "の分類器", va="center", size=fs - 0.8, weight="bold", color=col)
        minibars(ax, x + 6, y - 1.2, w - 12, 5.5, probs[m], col, labels=(r == 2), fs=FS - 2.0)
        for i, v in enumerate(probs[m]):
            cx = x + 6 + (i + 0.5) * (w - 12) / 4
            txt(ax, cx, y - 1.2 + 5.5 * v + 0.3, f"{v:.2f}", ha="center", va="bottom", size=FS - 2.3)
        arrow(ax, (x + w - 2, y + 2.5), (xs[3] + 2, top - 34), lw=0.7, head=5, color=col)
    arrow(ax, (xs[1] + ws[1] - 2, top - 17), (xs[2] + 2, top - 30), lw=0.8, head=6, color=GREY)
    arrow(ax, (xs[0] + ws[0] - 2, top - 16), (xs[1] + 2, top - 17), lw=0.8, head=6, color=GREY)
    # (4) product
    x, w = xs[3], ws[3]
    txt(ax, x + 2.5, top - 9, "発話ごとに，3 つの確率を\n感情ごとに掛けて，\n合計が 1 になるように割る", size=fs - 1.2, ls=1.15)
    q = np.array(probs["A"]) * np.array(probs["V"]) * np.array(probs["T"])
    q = q / q.sum()
    box(ax, x + 2, top - 52, w - 4, 26, fc="white", ec=C_PROP, lw=1.1, z=3)
    txt(ax, x + w / 2, top - 27.5, "疑似ラベル q（積）", ha="center", size=fs - 0.6, weight="bold", color=C_PROP)
    minibars(ax, x + 6, top - 48, w - 12, 13, list(q), C_PROP, show=True)
    txt(ax, x + 2, top - 56, "3 つが揃って高い感情\n（例では Happy）が\n強く残る", size=fs - 1.3, color=GREY, ls=1.15)
    # (5) retrain and compare
    x, w = xs[4], ws[4]
    box(ax, x + 2, top - 33, w - 4, 24, fc="white", ec=C_SRC, lw=1.1, z=3)
    txt(ax, x + 3.5, top - 10.2, "学び直す分類器\n（声・顔・言葉・AVT\nのどれか 1 つ）を，\nアンカーに引き戻し\nながら K 本の正解 ＋\n候補の q で学ぶ", size=fs - 1.0, ls=1.15)
    txt(ax, x + 2.5, top - 35, "q の重みの合計は感情\nごとにそろえる", size=fs - 1.5, color=GREY, ls=1.1)
    box(ax, x + 2, top - 60, w - 4, 15, fc="white", ec="black", lw=1.0, z=3)
    txt(ax, x + 3.5, top - 46.2, "テストの UAR を比べる\ngain ＝ 学び直し\n　− K 本の基準の分類器", size=fs - 1.0, ls=1.15)
    arrow(ax, (x + w / 2, top - 39.5), (x + w / 2, top - 45), lw=0.8, head=6)
    arrow(ax, (xs[3] + ws[3] - 2, top - 39), (xs[4] + 2, top - 21), lw=0.9, head=6, color=C_PROP)
    txt(ax, W - 1, 1.2, "確率の数値は説明のための例．q は学び直す分類器自身のモダリティの予測も含む．7 節の提案は ④ のまとめ方を替える",
        ha="right", va="bottom", size=FS - 1.4, color=GREY)
    save(fig, "fig_pl_flow_ja")


# ------------------------------------------------------------------ settings, one per figure
def node(ax, x, y, w, h, s, ec, fc="white", fs=FS + 0.6, color="black", lw=1.0, ls="-", weight="normal"):
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, ls=ls, z=3)
    txt(ax, x + w / 2, y + h / 2, s, ha="center", va="center", size=fs, color=color, ls=1.15, weight=weight)


def fig_setting_single():
    setup("ja")
    H = 52
    fig, ax = canvas(H)
    fs = FS + 1.0
    node(ax, 2, 34, 46, 13, "ソース：1 つの群\n（全部の発話にラベル）", C_SRC, fc="#EDE7F3", color=C_SRC, fs=fs)
    node(ax, 2, 13, 46, 13, "ターゲット：別の 1 つの群\nの K 本（感情ごと）", C_K, color=C_K, fs=fs)
    node(ax, 74, 38, 48, 10, "ソースだけの分類器", "black", fs=fs)
    node(ax, 74, 24, 48, 10, "合併の分類器（ソース ＋ K 本）", "black", fs=fs, lw=1.4)
    node(ax, 74, 10, 48, 10, "ターゲットだけの分類器（K 本）", "black", fs=fs)
    arrow(ax, (48, 41), (74, 43), lw=0.9, head=7, color=C_SRC)
    arrow(ax, (48, 40), (74, 30), lw=0.9, head=7, color=C_SRC)
    arrow(ax, (48, 20), (74, 28), lw=0.9, head=7, color=C_K)
    arrow(ax, (48, 19), (74, 15), lw=0.9, head=7, color=C_K)
    node(ax, 140, 21, 32, 16, "ターゲットの\nテストの発話\nで UAR", "black", fs=fs)
    for y in (43, 29, 15):
        arrow(ax, (122, y), (140, 29), lw=0.6, head=5, color=GREY)
    txt(ax, 2, 8.5, "群は 7 つ（IEMOCAP の即興と台本を別に数える）．ソースとターゲットの組を替えて平均．K = 1，5，10，25，50，全件．10 反復",
        size=fs - 1.6, color=GREY)
    save(fig, "fig_setting_single_ja")


def fig_setting_all():
    setup("ja")
    H = 52
    fig, ax = canvas(H)
    fs = FS + 1.0
    node(ax, 2, 34, 44, 13, "ソース：ターゲット以外\nの全部の群", C_SRC, fc="#EDE7F3", color=C_SRC, fs=fs)
    node(ax, 2, 18, 44, 11, "ターゲットの K 本", C_K, color=C_K, fs=fs)
    node(ax, 60, 2, 38, 11, "ターゲットの\nラベル無しの発話", C_U, color="#444444", fs=fs - 0.4, ls="--")
    node(ax, 60, 34, 38, 13, "ソースの全群から\n作ったアンカー", C_SRC, color=C_SRC, fs=fs)
    node(ax, 60, 16, 38, 12, "K 本の基準の分類器\n（アンカー ＋ K 本）", "black", fs=fs)
    node(ax, 112, 30, 60, 15, "疑似ラベルによる学び直し\n（ほかのモダリティを含む分類器の疑似ラベル）", C_PROP, fs=fs - 0.6, color=C_PROP)
    node(ax, 112, 8, 60, 15, "同じモダリティのラベル無しの\n発話を使う 8 手法（自己学習など）", "#444444", fs=fs - 0.6)
    arrow(ax, (46, 40), (60, 40), lw=0.9, head=7, color=C_SRC)
    arrow(ax, (79, 34), (79, 28), lw=0.9, head=7, color=C_SRC)
    arrow(ax, (46, 23), (60, 22), lw=0.9, head=7, color=C_K)
    arrow(ax, (98, 22), (112, 36), lw=0.9, head=7)
    arrow(ax, (98, 21), (112, 15), lw=0.9, head=7)
    arrow(ax, (98, 7), (112, 11), lw=0.7, head=6, color=C_U, ls="--")
    arrow(ax, (98, 8), (112, 32), lw=0.7, head=6, color=C_U, ls="--")
    save(fig, "fig_setting_all_ja")


def fig_setting_loco():
    setup("ja")
    H = 52
    fig, ax = canvas(H)
    fs = FS + 1.0
    node(ax, 2, 34, 44, 13, "ソース：残る 3 コーパス\n（コーパスごとに分類器）", C_SRC, fc="#EDE7F3", color=C_SRC, fs=fs)
    node(ax, 2, 18, 44, 11, "ターゲットの K 本", C_K, color=C_K, fs=fs)
    node(ax, 60, 2, 38, 11, "ターゲットの候補\n（ラベル無し）", C_U, color="#444444", fs=fs - 0.4, ls="--")
    node(ax, 60, 34, 38, 13, "3 つの分類器の\n平均 ＝ アンカー", C_SRC, color=C_SRC, fs=fs)
    node(ax, 60, 16, 38, 12, "K 本の基準の分類器\n（アンカー ＋ K 本）", "black", fs=fs)
    node(ax, 112, 26, 60, 15, "疑似ラベルによる学び直し\n（K 本 ＋ 候補の疑似ラベル）", C_PROP, fs=fs - 0.4, color=C_PROP)
    arrow(ax, (46, 40), (60, 40), lw=0.9, head=7, color=C_SRC)
    arrow(ax, (79, 34), (79, 28), lw=0.9, head=7, color=C_SRC)
    arrow(ax, (46, 23), (60, 22), lw=0.9, head=7, color=C_K)
    arrow(ax, (98, 22), (112, 31), lw=0.9, head=7)
    arrow(ax, (98, 8), (112, 28), lw=0.7, head=6, color=C_U, ls="--")
    txt(ax, 112, 21.5, "gain ＝ 学び直し − K 本の基準の分類器\n（テストの発話の UAR の差）", size=fs - 0.8, ls=1.15)
    save(fig, "fig_setting_loco_ja")


# ------------------------------------------------------------------ margin (Sec. 6.6)
def hbar(ax, x0, y, v, scale, color, h=3.2, label=None, fs=FS):
    ax.add_patch(__import__("matplotlib").patches.Rectangle((x0 if v >= 0 else x0 + v * scale, y - h / 2), abs(v) * scale, h,
                                                            fc=color, ec="none", zorder=4))
    if label:
        if v >= 0:
            txt(ax, x0 + v * scale + 1, y, label, va="center", size=fs, weight="bold")
        else:
            txt(ax, x0 + v * scale - 1, y, label, ha="right", va="center", size=fs, weight="bold")


def fig_margin_def(d):
    setup("ja")
    H = 74
    fig, ax = canvas(H)
    fs = FS + 0.6
    # definition flow (top)
    node(ax, 2, 48, 36, 20, "ターゲットの候補の発話\n（K 本を除く）\n解析のときだけ\n正解ラベルを使う", C_U, fs=fs - 0.4, ls="--")
    node(ax, 48, 58, 64, 11, "疑似ラベルの出所の判定の UAR　…ⓐ", C_PROP, fs=fs - 0.2, color=C_PROP)
    node(ax, 48, 45, 64, 11, "学び直す前の分類器（K 本の基準）の UAR　…ⓑ", "black", fs=fs - 0.2)
    arrow(ax, (38, 60), (48, 63.5), lw=0.8, head=6)
    arrow(ax, (38, 56), (48, 50.5), lw=0.8, head=6)
    node(ax, 122, 51, 50, 15, "当たる幅 ＝ ⓐ − ⓑ\n疑似ラベルのほうが\n何点当たるか", C_PROP, fc="#FDF0E8", fs=fs - 0.2, weight="bold")
    arrow(ax, (112, 63.5), (122, 59), lw=0.8, head=6)
    arrow(ax, (112, 50.5), (122, 55), lw=0.8, head=6)
    txt(ax, 122, 49.6, "この疑似ラベルで学び直したときの\ngain（テストの UAR の差）と並べる", size=fs - 1.2, color=GREY, ls=1.1)
    # two examples (bottom)
    ex = [("MOSEI，K = 25", "声の分類器を，顔と言葉をつないだ\n分類器の疑似ラベルで学び直す", "MOSEI", "A", 25, "pair_VT", C_UP, "上がった"),
          ("AFEW，K = 5", "顔の分類器（AFEW で最も強い）を，声と\n言葉をつないだ分類器の疑似ラベルで学び直す", "AFEW", "V", 5, "pair_AT", C_DOWN, "下がった")]
    scale = 1.55
    for j, (head, desc, T, st, K, tea, col, word) in enumerate(ex):
        x = 2 + j * 86
        r = d[(d["T"] == T) & (d.student == st) & (d.K == K) & (d.teacher == tea)].iloc[0]
        rows.append(dict(item=f"margin example {T} {st} K={K} {tea}", margin=r.margin, gain=r.gain, gain_lo=r.gain_lo, gain_hi=r.gain_hi))
        box(ax, x, 3, 84, 37, fc="white", ec=col, lw=1.2, z=2)
        txt(ax, x + 2, 38.3, "例 " + str(j + 1) + "：" + head, size=fs, weight="bold", color=col)
        txt(ax, x + 2, 33.6, desc, size=fs - 1.0, ls=1.15)
        x0 = x + (30 if r.gain >= 0 else 76)
        ax.plot([x0, x0], [8.5, 24], color="black", lw=0.6, zorder=5)
        txt(ax, x + 2, 20, "当たる幅", va="center", size=fs - 0.4)
        txt(ax, x + 2, 13, "gain", va="center", size=fs - 0.4)
        hbar(ax, x0, 20, r.margin, scale, C_PROP, label=f"{r.margin:+.1f}".replace("-", "−"), fs=fs - 0.4)
        hbar(ax, x0, 13, r.gain, scale, col, label=f"{r.gain:+.2f}".replace("-", "−"), fs=fs - 0.4)
        lo, hi = r.gain_lo, r.gain_hi
        ax.plot([x0 + lo * scale, x0 + hi * scale], [10.2, 10.2], color=col, lw=0.8, zorder=5)
        for e in (lo, hi):
            ax.plot([x0 + e * scale] * 2, [9.5, 10.9], color=col, lw=0.8, zorder=5)
        txt(ax, x + 2, 5.6, f"gain の 95% 区間 [{lo:+.2f}, {hi:+.2f}]：0 を含まない → {word}".replace("-", "−"),
            va="center", size=fs - 1.0, color=col, weight="bold")
    txt(ax, W - 1, 0.3, "値は UAR の点．縦線が 0", ha="right", va="bottom", size=FS - 1.6, color=GREY)
    save(fig, "fig_margin_def_ja")


def fig_margin_counts(d):
    setup("ja")
    H = 64
    fig, ax = canvas(H)
    fs = FS + 0.8
    expect = {5: ((173, 0, 125), (187, 58, 16)), 25: ((184, 0, 144), (176, 42, 20))}
    scale = 0.52
    x0 = 58
    y = H - 17
    # legend
    for i, (c, s) in enumerate([(C_UP, "上がった（区間の下端 > 0）"), (C_NS, "区別できない（区間が 0 を含む）"), (C_DOWN, "下がった（区間の上端 < 0）")]):
        lx = [2, 44, 92][i]
        ax.add_patch(__import__("matplotlib").patches.Rectangle((lx, H - 10.2), 3, 2.6, fc=c, ec="none"))
        txt(ax, lx + 4, H - 8.9, s, va="center", size=fs - 1.2)
    txt(ax, 2, H - 3.2, "gain の 95% 区間で分けた点の数", va="center", size=fs - 0.4, weight="bold")
    for K in (5, 25):
        g = d[d.K == K]
        txt(ax, 2, y + 3.5, f"K = {K}（360 点）", size=fs, weight="bold")
        y -= 5
        for sgn, lab in ((-1, "当たる幅 < 0\n疑似ラベルのほうが当たらない"), (1, "当たる幅 > 0\n疑似ラベルのほうが当たる")):
            s = g[g.margin * sgn > 0]
            up = int((s.gain_lo > 0).sum())
            down = int((s.gain_hi < 0).sum())
            ns = len(s) - up - down
            e = expect[K][0 if sgn < 0 else 1]
            assert (len(s), up, down) == e, (K, sgn, len(s), up, down)
            rows.append(dict(item=f"counts K={K} margin{'<0' if sgn < 0 else '>0'}", n=len(s), up=up, ns=ns, down=down))
            txt(ax, 2, y + 0.2, lab, va="center", size=fs - 1.2, ls=1.1)
            txt(ax, x0 - 1.5, y, f"{len(s)} 点", ha="right", va="center", size=fs - 1.0)
            xx = x0
            for n, c in ((up, C_UP), (ns, C_NS), (down, C_DOWN)):
                if n > 0:
                    ax.add_patch(__import__("matplotlib").patches.Rectangle((xx, y - 2.6), n * scale, 5.2, fc=c, ec="white", lw=0.5, zorder=4))
                    txt(ax, xx + n * scale / 2, y, str(n), ha="center", va="center", size=fs - 0.4,
                        color="white" if c != C_NS else "black", weight="bold", z=7)
                xx += n * scale
            if up == 0:
                txt(ax, xx + 1.5, y, "上がった点は 0", va="center", size=fs - 0.6, color=C_UP, weight="bold")
            y -= 9
        y -= 3
    txt(ax, W - 1, 0.3, "1 点 ＝（コーパス，学び直す分類器，疑似ラベルの出所）の 1 つ．gain はテストの UAR の差", ha="right", va="bottom",
        size=FS - 1.6, color=GREY)
    save(fig, "fig_margin_counts_ja")


# ------------------------------------------------------------------ D (Sec. 6.5, experiment 23)
def d_values():
    m = pd.read_csv(ROOT / "results" / "dai23" / "summary" / "main.csv")
    m = m[(m.subset == "all") & (m.K == 5)]
    out = {}
    for T in ("IEMOCAP", "MOSEI", "MELD"):
        g = m[m["T"] == T].set_index("quantity")
        a, b, l3 = g.loc["DAT", "effect"], g.loc["DBT", "effect"], g.loc["L3_B", "effect"]
        out[T] = dict(voice=a, reversed=b, face=b - l3)
        rows.append(dict(item=f"D text judgment {T}", voice=a, reversed=b, face=b - l3))
    return out


def dbars(ax, x0, y0, hmax, vals, keys, cols, names, scale, bw=6.5, gap=1.2, gw=34, fs=FS):
    for i, T in enumerate(("IEMOCAP", "MOSEI", "MELD")):
        gx = x0 + i * gw
        for j, k in enumerate(keys):
            v = vals[T][k]
            bx = gx + j * (bw + gap)
            ax.add_patch(__import__("matplotlib").patches.Rectangle((bx, y0), bw, v * scale, fc=cols[j], ec="none", zorder=4))
            txt(ax, bx + bw / 2, y0 + v * scale + 0.6, f"{v:.1f}", ha="center", va="bottom", size=fs - 0.8)
        txt(ax, gx + len(keys) * (bw + gap) / 2 - gap / 2, y0 - 1.2, T, ha="center", va="top", size=fs - 0.2)
    ax.plot([x0 - 2, x0 + 3 * gw - 6], [y0, y0], color="black", lw=0.6, zorder=5)
    for j, (c, n) in enumerate(zip(cols, names)):
        ax.add_patch(__import__("matplotlib").patches.Rectangle((x0 + j * 30, y0 + hmax + 2), 3, 2.6, fc=c, ec="none"))
        txt(ax, x0 + j * 30 + 4, y0 + hmax + 3.3, n, va="center", size=fs - 0.8)


def fig_D_def(vals):
    setup("ja")
    H = 66
    fig, ax = canvas(H)
    fs = FS + 0.4
    steps = ["① 言葉の分類器（K 本の基準）が，発話ごとに\n　感情を 1 つ判定する（これを目標にする）",
             "② 声の特徴だけを入力にした線形分類器で，\n　その判定を当てる → 当たり方 R",
             "③ 判定を「同じ感情・同じ話者のクラスタ」の\n　発話の中で入れ替えた目標でも ② を行う → R′\n　（感情と話者だけで当たる分）",
             "④ D ＝ R − R′\n　感情と話者では説明できない一致"]
    y = H - 3
    for i, s in enumerate(steps):
        h = 11 if i != 2 else 14
        node_ec = C_PROP if i == 3 else "black"
        box(ax, 2, y - h, 82, h, fc="#FDF0E8" if i == 3 else "white", ec=node_ec, lw=1.0, z=3)
        txt(ax, 4, y - 1.4, s, size=fs - 0.6, ls=1.15, weight="bold" if i == 3 else "normal")
        y -= h + 2.5
        if i < 3:
            arrow(ax, (43, y + 2.5), (43, y), lw=0.7, head=5)
    txt(ax, 2, y + 0.5, "顔の特徴でも同じ D を測って比べる", size=fs - 0.6, color=GREY)
    dbars(ax, 98, 12, 40, vals, ["voice", "face"], [MOD["A"][0], MOD["V"][0]], ["声の特徴", "顔の特徴"], scale=1.75, bw=8, gw=25, fs=fs)
    txt(ax, 96, H - 2.5, "言葉の分類器の判定を再現する D（K = 5）", size=fs - 0.2, weight="bold")
    txt(ax, W - 1, 0.3, "AFEW は言葉が音声の書き起こしなので除いた．値は UAR の点", ha="right", va="bottom", size=FS - 1.6, color=GREY)
    save(fig, "fig_D_def_ja")


def fig_reverse(vals):
    setup("ja")
    H = 62
    fig, ax = canvas(H)
    fs = FS + 0.4
    node(ax, 2, 44, 36, 11, "元の音声", MOD["A"][0], fs=fs)
    node(ax, 2, 24, 36, 11, "逆再生した音声", "#8C6D1F", fs=fs, fc="#F7EBD3")
    node(ax, 48, 34, 30, 11, "WavLM-large\n最終層", "black", fs=fs - 0.4)
    arrow(ax, (38, 49.5), (48, 41), lw=0.8, head=6)
    arrow(ax, (38, 29.5), (48, 38), lw=0.8, head=6)
    txt(ax, 2, 18.5, "Whisper large-v3 の書き起こしが元の\n言葉と一致した度合い（偶然を超えた分）", size=fs - 1.0, ls=1.15)
    txt(ax, 4, 9.5, "元の音声：0.79〜0.80\n逆再生：0.005〜0.012（ほぼ聞き取れない）", size=fs - 0.6, ls=1.3, weight="bold")
    txt(ax, 48, 30, "どちらの特徴でも\n言葉の分類器の判定の\nD を測る", size=fs - 1.0, ls=1.15)
    cols = [MOD["A"][0], "#C9A66B", MOD["V"][0]]
    dbars(ax, 90, 12, 37, vals, ["voice", "reversed", "face"], cols, ["元の声", "逆再生の声", "顔"], scale=1.75, bw=7, gap=1.0, gw=28, fs=fs)
    txt(ax, 88, H - 2.5, "言葉の分類器の判定を再現する D（K = 5）", size=fs - 0.2, weight="bold")
    txt(ax, W - 1, 0.3, "値は UAR の点．顔の D は（逆再生の声の D）−（逆再生の声 − 顔の D）から求めた", ha="right", va="bottom",
        size=FS - 1.6, color=GREY)
    save(fig, "fig_reverse_ja")


if __name__ == "__main__":
    d = pd.read_csv(DATA / "fig10_teacher_margin.csv")
    fig_pl_flow()
    fig_setting_single()
    fig_setting_all()
    fig_setting_loco()
    fig_margin_def(d)
    fig_margin_counts(d)
    v = d_values()
    fig_D_def(v)
    fig_reverse(v)
    pd.DataFrame(rows).to_csv(DATA / "fig_slides_v8.csv", index=False)
    print(json.dumps(v, indent=1))
