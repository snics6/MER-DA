"""Slide figures for the proposed fusion, second try (2026-11-16 night, researcher: "提案は各コーパスごとに感情ごとに
強いモダリティが違うっていう考えからじゃん？それを説明したいしそれを学ぶためのメタ学習ちっくなことをやった" / "文ばっか過ぎ"):

fig_prop_idea_ja    : for each corpus and emotion, the modality with the highest reliability d (design-time values,
                      ../figures/data/fig7_strength_profile.csv: provisional-target runs, full candidate labels, K = 5, 30 reps).
fig_prop_decomp_ja  : the 12-cell weight table = strength (one number per classifier, from the K labels)
                      + per-emotion shape (from the source corpora). Made-up numbers.
fig_prop_episode_ja : practising on the source corpora (target AFEW): each source becomes a provisional target in turn,
                      the settings are chosen there and used once on AFEW; the shape's transfer (eta) is measured
                      source -> source.
Output: ../figures/<name>.pdf and .png
Run:  /home/snics/MER-DA/.venv/bin/python make_figs_prop_v2.py
"""
from pathlib import Path

import matplotlib.patches as mp
import numpy as np
import pandas as pd

from make_figs_overview import C_K, C_PROP, C_SRC, FS, GREY, MOD, W, arrow, box, canvas, save, setup, txt

DATA = Path(__file__).resolve().parents[1] / "figures" / "data"
EMO = ["Neutral", "Happy", "Sad", "Angry"]
MODN = {"A": "声", "V": "顔", "T": "言葉"}
HATCH = {"A": "", "V": "///", "T": "..."}
CORP = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
LIGHT = {"A": "#F6D78F", "V": "#8FC1E3", "T": "#8FD3B9"}


def fig_idea():
    setup("ja")
    d = pd.read_csv(DATA / "fig7_strength_profile.csv")
    H = 70
    fig, ax = canvas(H)
    fs = FS + 1.0
    x0, y0, cw, ch = 32, 8, 25, 12
    for j, e in enumerate(EMO):
        txt(ax, x0 + (j + 0.5) * cw, y0 + 4 * ch + 3, e, ha="center", size=fs, weight="bold")
    for i, c in enumerate(CORP):
        y = y0 + (3 - i) * ch
        txt(ax, x0 - 3, y + ch / 2, c if c != "IEMOCAP" else "IEMOCAP\n（即興）", ha="right", va="center", size=fs, ls=1.1)
        for j, e in enumerate(EMO):
            vals = {m: d[(d.modality == m) & (d.corpus == c)][f"d_{e}"].values[0] for m in "AVT"}
            best = max(vals, key=vals.get)
            ax.add_patch(mp.Rectangle((x0 + j * cw + 0.4, y + 0.4), cw - 0.8, ch - 0.8, fc=LIGHT[best], ec="white",
                                      hatch=HATCH[best], lw=0.0, zorder=2))
            ax.add_patch(mp.Rectangle((x0 + j * cw + 0.4, y + 0.4), cw - 0.8, ch - 0.8, fc="none", ec="#777777", lw=0.5, zorder=3))
            box(ax, x0 + (j + 0.5) * cw - 6, y + ch / 2 - 3, 12, 6, fc="white", ec="none", r=0.8, z=4)
            txt(ax, x0 + (j + 0.5) * cw, y + ch / 2, MODN[best], ha="center", va="center", size=fs + 1.5, weight="bold",
                color=MOD[best][0])
    # legend
    lx = x0 + 4 * cw + 5
    txt(ax, lx, y0 + 4 * ch + 3, "マスの字：その感情を\n最もよく見分けた\nモダリティ", size=fs - 1.0, va="top", ls=1.25)
    for k, m in enumerate("AVT"):
        yy = y0 + 2.6 * ch - k * 8
        ax.add_patch(mp.Rectangle((lx, yy), 8, 5.5, fc=LIGHT[m], hatch=HATCH[m], ec="#777777", lw=0.5))
        txt(ax, lx + 10, yy + 2.75, MODN[m], va="center", size=fs, color=MOD[m][0], weight="bold")
    txt(ax, W - 1, 0.5, "値は手法を決めるときに使った信頼度（仮のターゲットの回，候補の全ラベル，K = 5，30 反復の平均）．評価の回の値ではない",
        ha="right", va="bottom", size=FS - 1.6, color=GREY)
    save(fig, "fig_prop_idea_ja")


def heat(ax, x, y, vals, cw, ch, cmap, vmin, vmax, fmt, fs, labels_row=None, labels_col=None):
    import matplotlib.cm as cm
    cmap = __import__("matplotlib").colormaps[cmap]
    vals = np.asarray(vals)
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            v = vals[r, c]
            col = cmap((v - vmin) / (vmax - vmin))
            ax.add_patch(mp.Rectangle((x + c * cw, y - (r + 1) * ch), cw - 0.5, ch - 0.5, fc=col, ec="#999999", lw=0.4))
            lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
            txt(ax, x + c * cw + cw / 2, y - (r + 0.5) * ch, fmt.format(v).replace("-", "−"), ha="center", va="center",
                size=fs, color="white" if lum < 0.5 else "black")
    if labels_row:
        for r, l in enumerate(labels_row):
            txt(ax, x - 1.5, y - (r + 0.5) * ch, l, ha="right", va="center", size=fs, color=MOD["AVT"[r]][0], weight="bold")
    if labels_col:
        for c, l in enumerate(labels_col):
            txt(ax, x + c * cw + cw / 2, y + 2, l, ha="center", size=fs - 0.6, color=GREY)


def fig_decomp():
    setup("ja")
    H = 64
    fig, ax = canvas(H)
    fs = FS + 0.6
    level = np.array([[1.2], [1.9], [1.0]])
    prof = np.array([[-0.25, -0.10, 0.05, 0.30], [-0.45, 0.55, -0.15, 0.05], [-0.20, 0.10, 0.05, 0.05]])
    D = np.maximum(0, level + prof)
    E = ["N", "H", "S", "A"]
    rows = [MODN[m] for m in "AVT"]
    ytop = H - 14
    # weight table
    txt(ax, 2, H - 3, "分類器 × 感情の重みの表（12 マス）", size=fs + 0.4, weight="bold", va="top")
    heat(ax, 10, ytop, D, 9, 7, "Oranges", 0, 2.6, "{:.2f}", fs - 0.8, rows, E)
    txt(ax, 50, ytop - 10.5, "＝", size=fs + 8, va="center")
    # strength
    txt(ax, 60, H - 3, "① 強さ", size=fs + 0.2, weight="bold", va="top", color=C_K)
    heat(ax, 68, ytop, level, 11, 7, "Blues", 0, 2.6, "{:.1f}", fs - 0.8, rows, ["強さ"])
    txt(ax, 84, ytop - 10.5, "＋", size=fs + 8, va="center")
    # profile
    txt(ax, 92, H - 3, "② 感情ごとの形（行の平均は 0）", size=fs + 0.2, weight="bold", va="top", color=C_SRC)
    heat(ax, 98, ytop, prof, 9, 7, "PuOr_r", -0.8, 0.8, "{:+.2f}", fs - 0.8, rows, E)
    # where from
    box(ax, 52, 3, 40, 22, fc="#E8F0F8", ec=C_K, lw=1.1)
    txt(ax, 54, 22.5, "ターゲットの K 本で測る", size=fs - 0.2, weight="bold", color=C_K, va="top")
    txt(ax, 54, 16.5, "コーパスごとに入れ替わるので，\nターゲットで測る．K 本から\n決める数は 3 つだけ", size=fs - 1.3, va="top", ls=1.3)
    box(ax, 95, 3, 41, 22, fc="#F0EAF5", ec=C_SRC, lw=1.1)
    txt(ax, 97, 22.5, "ソースのコーパスで測る", size=fs - 0.2, weight="bold", color=C_SRC, va="top")
    txt(ax, 97, 16.5, "12 マスは K 本では測れない．\nラベルの多いソースで測る", size=fs - 1.3, va="top", ls=1.3)
    box(ax, 140, 3, 32, 47, fc="#FDF0E8", ec=C_PROP, lw=1.1)
    txt(ax, 142, 47.5, "この表の重みで", size=fs - 0.2, weight="bold", color=C_PROP, va="top")
    txt(ax, 142, 41.5, "声・顔・言葉の\n分類器の答えを\n感情ごとに\n重みづけて足す\n\n→ 疑似ラベル\n→ AVT を\n　学び直す", size=fs - 1.0, va="top", ls=1.35)
    arrow(ax, (137, 30), (139.5, 30), lw=0.9, head=7)
    txt(ax, 2, 3.5, "数値は説明のための例", size=FS - 1.6, color=GREY)
    save(fig, "fig_prop_decomp_ja")


def corp_box(ax, x, y, w, h, name, role, fs):
    col = C_K if role == "t" else C_SRC
    fc = "#E8F0F8" if role == "t" else "#F0EAF5"
    box(ax, x, y, w, h, fc=fc, ec=col, lw=1.0, z=3)
    txt(ax, x + w / 2, y + h / 2, name, ha="center", va="center", size=fs, color=col, weight="bold", z=4)


def fig_episode():
    setup("ja")
    H = 96
    fig, ax = canvas(H)
    fs = FS + 0.4
    txt(ax, 2, H - 2, "ハイパーパラメータの選択：ソースを 1 つずつ仮のターゲットにした 3 つのエピソード（ターゲットが AFEW のとき．AFEW は使わない）",
        size=fs - 0.2, weight="bold", va="top")
    eps = [("IEMOCAP", ["MOSEI", "MELD"]), ("MOSEI", ["IEMOCAP", "MELD"]), ("MELD", ["IEMOCAP", "MOSEI"])]
    for k, (pt, ss) in enumerate(eps):
        x = 2 + k * 37
        box(ax, x, 42, 35, 43, fc="white", ec="#888888", lw=0.8)
        txt(ax, x + 2, 83, f"エピソード {k + 1}", size=fs, weight="bold", va="top")
        corp_box(ax, x + 1.5, 63, 15.5, 9, pt, "t", fs - 1.4)
        txt(ax, x + 9.2, 61.5, "仮のターゲット", ha="center", va="top", size=fs - 2.2, color=C_K)
        for j, s in enumerate(ss):
            corp_box(ax, x + 19.5, 68 - j * 10, 14, 8, s, "s", fs - 1.8)
        txt(ax, x + 26.5, 55.5, "ソース", ha="center", va="top", size=fs - 2.2, color=C_SRC)
        txt(ax, x + 2, 51, "強さ：K 本で　形：ソースで\n→ ハイパーパラメータの候補\n　ごとにテストの UAR", size=fs - 1.9, va="top", ls=1.3)
    arrow(ax, (113, 63), (116.5, 63), lw=0.9, head=7, color="#555555")
    box(ax, 117, 42, 55, 43, fc="#FDF0E8", ec=C_PROP, lw=1.1)
    txt(ax, 119, 83, "選ぶハイパーパラメータ", size=fs, weight="bold", color=C_PROP, va="top")
    txt(ax, 119, 77.5, "3 つのエピソード（30 反復）の UAR の平均が\n最大の組を選ぶ（16 通りから 1 つ）", size=fs - 1.7, va="top", ls=1.3)
    items = [("n0：K 本の強さに既定の強さをどれだけ混ぜるか", "ラベル何本ぶん混ぜるか（0，20，80，∞）．\n0 なら K 本の値だけ，∞ なら既定の値だけ"),
             ("戻し先：混ぜる既定の強さは何か", "3 つの分類器の強さの平均（3 つを同じ強さ\nに近づける）か，ソースで測った強さか"),
             ("β：感情ごとの形を使うか", "1 なら使う，0 なら使わない（感情に共通の重み）")]
    y = 70.5
    for name, body in items:
        txt(ax, 119, y, "・" + name, size=fs - 1.8, weight="bold", va="top", color=C_PROP)
        txt(ax, 121.5, y - 3.6, body, size=fs - 2.1, va="top", ls=1.25)
        y -= 9.8
    arrow(ax, (147, 41.5), (147, 36.5), lw=1.0, head=8, color=C_PROP)
    txt(ax, 149, 39, "選んだ値で", size=fs - 1.6, color=C_PROP, va="center")
    # eta inset
    box(ax, 2, 3, 74, 31, fc="#F7F7F7", ec="#999999", lw=0.6)
    txt(ax, 4, 31.5, "形がほかのコーパスでどれだけ残るか（η）", size=fs - 0.2, weight="bold", va="top")
    corp_box(ax, 6, 8, 18, 9, "IEMOCAP", "s", fs - 1.0)
    corp_box(ax, 54, 8, 18, 9, "MELD", "s", fs - 1.0)
    txt(ax, 15, 22.5, "同じコーパスの中で\n測る信頼度 d_in", ha="center", va="center", size=fs - 1.8, color=C_SRC, ls=1.15)
    arrow(ax, (24.5, 12.5), (53.5, 12.5), lw=0.9, head=7, color=C_SRC)
    txt(ax, 39, 16.5, "別のコーパスに当てた\n信頼度 d_out", ha="center", va="center", size=fs - 1.8, color=C_SRC, ls=1.15)
    txt(ax, 4, 5.5, "η ＝ Σ d_out ÷ Σ d_in を形に掛ける（ソースどうしで測る）", size=fs - 1.8)
    # final
    box(ax, 82, 3, 90, 31, fc="#E8F0F8", ec=C_K, lw=1.1)
    txt(ax, 84, 31.5, "評価：AFEW", size=fs, weight="bold", color=C_K, va="top")
    corp_box(ax, 86, 10, 20, 10, "AFEW", "t", fs - 0.4)
    for j, s in enumerate(["IEMOCAP", "MOSEI", "MELD"]):
        corp_box(ax, 112, 21.5 - j * 6.6, 18, 5.6, s, "s", fs - 1.8)
    txt(ax, 134, 26, "強さ：AFEW の K 本で\n形：3 つのソースで\nハイパーパラメータ：\n　エピソードで選んだ値\n→ 疑似ラベルで AVT を学び\n　直し，テストで 1 回だけ評価",
        size=fs - 1.7, va="top", ls=1.3)
    save(fig, "fig_prop_episode_ja")


def bars(ax, x0, y0, vals, col, bw=5.0, gap=1.6, scale=8.0, fs=FS, signed=False, hatch=""):
    for i, v in enumerate(vals):
        x = x0 + i * (bw + gap)
        if signed:
            ax.add_patch(mp.Rectangle((x, y0 if v >= 0 else y0 + v * scale), bw, abs(v) * scale, fc=col, ec="black", lw=0.3, hatch=hatch))
            txt(ax, x + bw / 2, y0 + v * scale + (1.0 if v >= 0 else -1.0), f"{v:+.2f}".replace("-", "−"), ha="center",
                va="bottom" if v >= 0 else "top", size=fs - 1.6)
        else:
            ax.add_patch(mp.Rectangle((x, y0), bw, v * scale, fc=col, ec="black", lw=0.3, hatch=hatch))
            txt(ax, x + bw / 2, y0 + v * scale + 0.8, f"{v:.2f}", ha="center", va="bottom", size=fs - 1.6)
        txt(ax, x + bw / 2, y0 - (0.8 if not signed else 7.5), "NHSA"[i], ha="center", va="top", size=fs - 1.2, color=GREY)
    ax.plot([x0 - 1, x0 + 4 * (bw + gap) - gap + 1], [y0, y0], color="black", lw=0.5)


def fig_calc():
    setup("ja")
    H = 84
    fig, ax = canvas(H)
    fs = FS + 0.4
    colV = MOD["V"][0]
    # left: strength from K labels
    box(ax, 2, 2, 84, H - 4, fc="#F4F8FC", ec=C_K, lw=1.1)
    txt(ax, 4, H - 4.5, "① 強さ：ターゲットの K 本だけで計算（例：顔の分類器，K = 5）", size=fs - 0.2, weight="bold", color=C_K, va="top")
    # 4 x 5 grid of K-labelled utterances coloured by group
    gcol = ["#CFE3F3", "#F9E0C7", "#D8EFD6", "#EADCF0", "#F3D3D8"]
    gx, gy, cw, ch = 6, H - 34, 5.2, 4.6
    for e in range(4):
        txt(ax, gx - 1, gy + (3 - e) * ch + ch / 2, "NHSA"[e], ha="right", va="center", size=fs - 1.2, color=GREY)
        for g in range(5):
            ax.add_patch(mp.Rectangle((gx + g * cw, gy + (3 - e) * ch), cw - 0.6, ch - 0.6, fc=gcol[g], ec="#888888", lw=0.4))
    for g in range(5):
        txt(ax, gx + g * cw + cw / 2, gy - 1.2, f"組{g + 1}", ha="center", va="top", size=fs - 2.2, color=GREY)
    txt(ax, gx, gy + 4 * ch + 2, "K 本 ＝ 4 感情 × 5 本 ＝ 20 本", size=fs - 1.4)
    txt(ax, 40, H - 13, "1. 20 本を 5 組に分ける（各組に\n　 各感情 1 本）\n2. 4 組で分類器を学び直し，\n　 残る 1 組に答えを付ける．\n　 これを 5 回くり返す\n3. 20 本すべてに，その発話を\n　 学びに使っていない答えが付く",
        size=fs - 1.6, va="top", ls=1.3)
    dK = [1.40, 2.55, 1.60, 2.05]
    txt(ax, 6, 33, "4. その答えで感情ごとに d を求める（前のページの方法）", size=fs - 1.4)
    bars(ax, 10, 9, dK, colV, scale=7.0, fs=fs, hatch="///")
    m = float(np.mean(dK))
    ax.plot([8.5, 37.5], [9 + m * 7.0] * 2, color=C_K, lw=1.2, ls="--")
    txt(ax, 39, 9 + m * 7.0, f"平均 ＝ 強さ {m:.2f}", va="center", size=fs - 0.6, color=C_K, weight="bold")
    txt(ax, 40, 18.5, "意味：このターゲットで，\n顔の分類器が感情全体を\nどれだけよく見分けるか", size=fs - 1.6, va="top", ls=1.3)
    txt(ax, 40, 8.5, "K 本が少ないとぶれるので，\nn0 本ぶん既定の値に寄せる", size=fs - 1.8, va="top", ls=1.3, color=GREY)
    # right: shape from sources
    box(ax, 90, 2, 82, H - 4, fc="#F8F4FA", ec=C_SRC, lw=1.1)
    txt(ax, 92, H - 4.5, "② 感情ごとの形：ソースのコーパスで計算（K 本は使わない）", size=fs - 0.2, weight="bold", color=C_SRC, va="top")
    txt(ax, 93, H - 12, "1. 各ソースの中で，話者を分けた交差検証で\n　 顔の分類器の 4 感情の d を求め，ソースどうしで\n　 平均する（ほかのコーパスで残る割合 η も掛ける）",
        size=fs - 1.6, va="top", ls=1.3)
    dS = [0.95, 1.95, 1.25, 1.45]
    bars(ax, 98, 45, dS, colV, scale=7.0, fs=fs, hatch="///")
    ms = float(np.mean(dS))
    ax.plot([96.5, 125.5], [45 + ms * 7.0] * 2, color=C_SRC, lw=1.0, ls="--")
    txt(ax, 127, 45 + ms * 7.0, f"平均 {ms:.2f}", va="center", size=fs - 0.8, color=C_SRC)
    arrow(ax, (112, 42), (112, 35), lw=0.9, head=7, color=C_SRC)
    txt(ax, 114, 38.5, "2. 平均を引く", size=fs - 1.2, color=C_SRC, va="center")
    r = [v - ms for v in dS]
    bars(ax, 98, 22, r, "#C9B8DA", scale=7.0, fs=fs, signed=True)
    txt(ax, 128, 28, "＝ 感情ごとの形", size=fs - 0.4, color=C_SRC, weight="bold", va="center")
    txt(ax, 128, 21, "意味：顔は自分の平均\nより Happy が得意で，\nNeutral が苦手．\n強さの情報は持たない", size=fs - 1.6, va="top", ls=1.3)
    txt(ax, W - 4, 5.0, "数値は説明のための例", ha="right", size=FS - 1.6, color=GREY)
    save(fig, "fig_prop_calc_ja")


if __name__ == "__main__":
    fig_calc()
    fig_idea()
    fig_decomp()
    fig_episode()
