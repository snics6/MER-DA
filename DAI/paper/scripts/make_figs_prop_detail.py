"""Slide figures for the proposed fusion (2026-11-16 night, researcher: "p24 ... ここがキモなので．なぜ良くなると
思ったのか，あと学習中の他のコーパスの使い方とか．図を用いてめちゃ詳しく"):

fig_prop_why_ja       : why we expected the proposal to raise UAR. (a) RQ1(b): gain ranks with the margin of the
                        pseudo-label; (b) the strength (level) of each classifier swaps between corpora;
                        (c) the per-emotion shape (profile) is similar across corpora.
                        (b), (c) are the design-time values of ../figures/data/fig7_strength_profile.csv
                        (provisional-target runs, K = 5, full candidate labels, 30 repetitions; not the evaluation).
fig_prop_corpora_ja   : what each corpus is used for when the target is AFEW (anchors, profile, hyperparameters).
fig_prop_reliability_ja : how one reliability d_k(i) is measured (AUC of the centred score -> d), with made-up scores.
fig_prop_table_ja     : from level and profile to the table, the weights, the fused score of one utterance,
                        the label-free intercept shift and the pseudo-label (made-up numbers).
Output: ../figures/<name>.pdf and .png
Run:  /home/snics/MER-DA/.venv/bin/python make_figs_prop_detail.py
"""
from pathlib import Path

import matplotlib.patches as mp
import numpy as np
import pandas as pd
from scipy.stats import norm

from make_figs_overview import C_K, C_PROP, C_SRC, C_U, FS, GREY, MOD, W, arrow, box, canvas, save, setup, txt

DATA = Path(__file__).resolve().parents[1] / "figures" / "data"
EMO = ["N", "H", "S", "A"]
MODN = {"A": "声", "V": "顔", "T": "言葉"}
CORP = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]


def fig_why():
    setup("ja")
    d = pd.read_csv(DATA / "fig7_strength_profile.csv")
    H = 96
    fig, ax = canvas(H)
    fs = FS + 0.4
    heads = ["(a) 問い 1(b) から：疑似ラベルが当たるほど\n　 学び直しの gain が大きかった",
             "(b) 各分類器の強さ（水準）は\n　 コーパスによって入れ替わった",
             "(c) 感情ごとの得意・不得意の形（相性）は\n　 コーパスをまたいで似ていた"]
    xs = [2, 60, 118]
    for x, h in zip(xs, heads):
        txt(ax, x, H - 2, h, size=fs, weight="bold", va="top", ls=1.15)
    # (a)
    box(ax, 2, 25, 52, 55, fc="#F7F7F7", ec="#999999", lw=0.6)
    txt(ax, 4, 77, "疑似ラベルの出所を替えて学び直す\nと，当たる幅（疑似ラベル − 学び\n直す前の分類器の UAR）と gain\nの順位がそろった", size=fs - 0.9, va="top", ls=1.3)
    txt(ax, 4, 56, "・手法を決める前（仮のターゲットの回，\n　280 条件）：順位相関 0.68\n　（でたらめに並べたときの水準 0.32）\n・評価の回（計算の後に集計）：\n　K = 5 で 0.80，K = 25 で 0.83", size=fs - 0.9, va="top", ls=1.3)
    box(ax, 2, 3, 52, 18, fc="#FDF0E8", ec=C_PROP, lw=1.0)
    txt(ax, 4, 18.5, "→ 3 つの分類器の答えのまとめ方を\n　変えて，疑似ラベルが当たるように\n　すれば，gain が上がると考えた", size=fs - 0.6, va="top", ls=1.3, color=C_PROP, weight="bold")
    # (b) level bars
    sub = fig.add_axes([67 / W, 37 / H, 44 / W, 42 / H])
    bw = 0.26
    for j, m in enumerate(["A", "V", "T"]):
        vals = [d[(d.modality == m) & (d.corpus == c)].level_L.values[0] for c in CORP]
        col, mk = MOD[m]
        sub.bar(np.arange(4) + (j - 1) * bw, vals, bw, color=col, edgecolor="black", lw=0.3,
                hatch=["", "///", "..."][j], label=MODN[m])
    for i, c in enumerate(CORP):
        best = max(["A", "V", "T"], key=lambda m: d[(d.modality == m) & (d.corpus == c)].level_L.values[0])
        lab = "声と顔が\nほぼ同じ" if c == "MOSEI" else "最強：\n" + MODN[best]
        sub.text(i, 2.05, lab, ha="center", fontsize=FS - 1.8, linespacing=1.1)
    sub.set_xticks(range(4))
    sub.set_xticklabels(["AFEW", "IEMO-\nCAP", "MOSEI", "MELD"], fontsize=FS - 1.6)
    sub.set_ylim(0, 2.45)
    sub.set_ylabel("水準（信頼度の平均）", fontsize=FS - 1.4)
    sub.tick_params(labelsize=FS - 1.4)
    sub.legend(fontsize=FS - 1.6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, handlelength=1.4, columnspacing=0.8)
    for s in ["top", "right"]:
        sub.spines[s].set_visible(False)
    box(ax, 60, 3, 54, 18, fc="#E8F0F8", ec=C_K, lw=1.0)
    txt(ax, 62, 18.5, "→ 強さは，ターゲットの K 本で\n　測る（分類器ごとに 1 つ，\n　3 つの数だけ）", size=fs - 0.6, va="top", ls=1.3, color=C_K, weight="bold")
    # (c) profile lines per modality
    for j, m in enumerate(["A", "V", "T"]):
        sp = fig.add_axes([(125 + j * 16) / W, 37 / H, 14 / W, 42 / H])
        col, mk = MOD[m]
        for c, ls in zip(CORP, ["-", "--", ":", "-."]):
            r = d[(d.modality == m) & (d.corpus == c)][[f"affinity_{e}" for e in ["Neutral", "Happy", "Sad", "Angry"]]].values[0]
            sp.plot(range(4), r, ls=ls, marker=mk, ms=2.5, color=col, lw=0.9, label=c)
        sp.axhline(0, color=GREY, lw=0.4, ls="--")
        sp.set_ylim(-0.85, 0.8)
        sp.set_xticks(range(4))
        sp.set_xticklabels(EMO, fontsize=FS - 1.4)
        sp.set_title(MODN[m], fontsize=FS - 0.6, pad=2)
        sp.tick_params(labelsize=FS - 1.6)
        if j > 0:
            sp.set_yticklabels([])
        else:
            sp.set_ylabel("相性（信頼度 − 水準）", fontsize=FS - 1.6)
        for s in ["top", "right"]:
            sp.spines[s].set_visible(False)
        if j == 2:
            from matplotlib.lines import Line2D
            hs = [Line2D([0], [0], color="black", ls=ls, lw=0.9) for ls in ["-", "--", ":", "-."]]
            sp.legend(hs, CORP, fontsize=FS - 2.2, loc="upper center", bbox_to_anchor=(-0.65, -0.13), ncol=4, frameon=False,
                      handlelength=2.0, columnspacing=0.8)
    box(ax, 118, 3, 54, 18, fc="#F0EAF5", ec=C_SRC, lw=1.0)
    txt(ax, 120, 18.5, "→ 形は，ラベルの多いソースの\n　コーパスで測る（K 本からは\n　推さない）", size=fs - 0.6, va="top", ls=1.3, color=C_SRC, weight="bold")
    txt(ax, W - 1, 0.2, "(b)(c) は手法を決めるときに使った値（仮のターゲットの回，候補の全ラベル，K = 5，30 反復の平均）で，評価の回の値ではない",
        ha="right", va="bottom", size=FS - 1.8, color=GREY)
    save(fig, "fig_prop_why_ja")


def fig_corpora():
    setup("ja")
    H = 100
    fig, ax = canvas(H)
    fs = FS + 0.4
    txt(ax, 2, H - 2.5, "例：ターゲットが AFEW のとき（ソースは IEMOCAP，MOSEI，MELD）", size=fs + 0.4, weight="bold", va="top")
    srcs = ["IEMOCAP", "MOSEI", "MELD"]
    for i, s in enumerate(srcs):
        y = H - 22 - i * 15
        box(ax, 2, y, 33, 12, fc="#F0EAF5", ec=C_SRC, lw=1.0)
        txt(ax, 4, y + 9.5, f"ソース：{s}", size=fs - 0.2, weight="bold", color=C_SRC)
        txt(ax, 4, y + 5.6, "全発話にラベルがある．\n声・顔・言葉・AVT を学ぶ", size=fs - 1.8, va="top", ls=1.2)
    box(ax, 2, 3, 33, 37, fc="#E8F0F8", ec=C_K, lw=1.1)
    txt(ax, 4, 37.5, "ターゲット：AFEW", size=fs - 0.2, weight="bold", color=C_K)
    txt(ax, 4, 33, "・K 本：感情ごとに K 本\n　のラベル\n・候補：ほかの発話\n　（ラベルは使わない）\n・テスト：話者（クラスタ）\n　を分けた別の発話．\n　評価だけに使う",
        size=fs - 1.4, va="top", ls=1.3)
    uses = [
        ("① アンカー（学び直しの出発点）を作る", C_SRC,
         "3 つのソースの分類器の重みと切片を平均し，L2 正則化の中心\nにする．積を作る声・顔・言葉の分類器と AVT の分類器は，\nこのアンカーから AFEW の K 本（と 1 段目の確率）で学び直す", H - 30),
        ("② 感情ごとの形（相性）を測る", C_SRC,
         "各ソースの中で，話者を分けた 5 分割で信頼度 d_in を測る．\nあるソースで学んだ分類器を別のソースに当てて d_out を測る．\n残る割合 η ＝ Σ d_out ÷ Σ d_in を掛け，行の平均を引いて相性にする",
         H - 55),
        ("③ 設定（ハイパーパラメータ）を選ぶ", "#555555",
         "IEMOCAP，MOSEI，MELD を 1 つずつ仮のターゲットにし，残る 2 つ\nをソースにして同じ手順を流す（3 × 5 分割 × 種 2 ＝ 30 反復）．\nテストの UAR の平均が最大の設定（n0，戻し先，β）に固定する",
         H - 80),
    ]
    ys = []
    for head, col, body, y in uses:
        box(ax, 41, y - 4, 95, 21, fc="white", ec=col, lw=1.0)
        txt(ax, 43, y + 14.5, head, size=fs - 0.2, weight="bold", color=col, va="top")
        txt(ax, 43, y + 9, body, size=fs - 1.5, va="top", ls=1.35)
        ys.append(y + 6.5)
        arrow(ax, (38, y + 6.5), (40.5, y + 6.5), lw=0.8, head=6, color=C_SRC)
    ax.plot([35.5, 38], [H - 16, H - 16], color=C_SRC, lw=0.8)
    ax.plot([35.5, 38], [H - 31, H - 31], color=C_SRC, lw=0.8)
    ax.plot([35.5, 38], [H - 46, H - 46], color=C_SRC, lw=0.8)
    ax.plot([38, 38], [min(ys), H - 16], color=C_SRC, lw=0.8)
    box(ax, 141, 3, 31, H - 12, fc="#E8F0F8", ec=C_K, lw=1.0)
    txt(ax, 143, H - 11, "AFEW で行うこと", size=fs - 0.2, weight="bold", color=C_K, va="top")
    txt(ax, 143, H - 17, "④ 各分類器の強さ\n（水準）を K 本だけ\nで測る（分類器ごと\nに 1 つ）\n\n⑤ ① の分類器の答え\nを，② と ④ の表で\nまとめて候補に付け，\n疑似ラベルにして\nAVT を学び直す\n\n⑥ テストの UAR で\n評価する（③ で\n固定した設定で\n1 回だけ）",
        size=fs - 1.1, va="top", ls=1.3)
    for y in ys:
        arrow(ax, (136.5, y), (140.5, y), lw=0.8, head=6, color="#555555")
    arrow(ax, (35.5, 6), (140.5, 6), lw=0.8, head=6, color=C_K)
    txt(ax, 88, 8.6, "AFEW の K 本と候補", ha="center", size=fs - 1.4, color=C_K)
    txt(ax, W - 1, 0.2, "AFEW のテストの発話とラベルは，①〜⑤ のどこにも使わない", ha="right", va="bottom", size=FS - 1.6, color=GREY)
    save(fig, "fig_prop_corpora_ja")


def fig_reliability():
    setup("ja")
    H = 70
    fig, ax = canvas(H)
    fs = FS + 0.4
    rng = np.random.default_rng(1)
    pos = rng.normal(1.0, 1.0, 400)
    neg = rng.normal(-0.4, 1.0, 400)
    auc = float((pos[:, None] > neg[None, :]).mean())
    dval = np.sqrt(2) * norm.ppf(auc)
    sub = fig.add_axes([6 / W, 16 / H, 70 / W, 44 / H])
    bins = np.linspace(-4, 4.5, 30)
    sub.hist(neg, bins, color="#BBBBBB", edgecolor="#555555", lw=0.3, label="本当の感情が Sad の発話", hatch="///")
    sub.hist(pos, bins, color=MOD["V"][0], alpha=0.7, edgecolor="black", lw=0.3, label="本当の感情が Happy の発話")
    sub.set_xlabel("顔の分類器の Happy の点 z（中心化した対数確率）", fontsize=FS - 0.6)
    sub.set_yticks([])
    sub.legend(fontsize=FS - 1.2, frameon=False, loc="upper left")
    for s in ["top", "right", "left"]:
        sub.spines[s].set_visible(False)
    sub.tick_params(labelsize=FS - 1.2)
    txt(ax, 2, H - 2.5, "① 分類器 1 つ（例：顔）と感情 1 つ（例：Happy）について，本当の感情が Happy の発話と，ほかの感情の発話を，Happy の点 z で並べる",
        size=fs - 0.6, va="top")
    x = 86
    box(ax, x, 36, 86, 22, fc="#F7F7F7", ec="#999999", lw=0.6)
    txt(ax, x + 2, 55.5, f"② AUC：Happy の発話の点がほかの感情の発話の点より\n　大きい組の割合．この例では {auc:.2f}", size=fs - 0.4, va="top", ls=1.3)
    txt(ax, x + 2, 44.5, "　Sad，Angry，Neutral の 3 つで測って平均する", size=fs - 1.0, va="top", color=GREY)
    box(ax, x, 9, 86, 24, fc="#FDF0E8", ec=C_PROP, lw=1.0)
    txt(ax, x + 2, 30.5, f"③ 信頼度 d ＝ √2 × $\\Phi^{{-1}}$(AUC) ＝ {dval:.2f}", size=fs + 0.2, va="top", weight="bold", color=C_PROP)
    txt(ax, x + 2, 24, "$\\Phi^{-1}$ は標準正規分布の分位点．AUC が 0.5（でたらめ）なら 0，\n大きいほど Happy をよく見分ける．上限は 2.90（AUC を 0.98 で切る）．\n点の順位だけを使うので，切片やターゲットの感情の割合に左右されない",
        size=fs - 1.1, va="top", ls=1.35)
    txt(ax, 2, 6, "④ これを 3 つの分類器 × 4 感情で測ると，12 個の信頼度の表になる．水準 ＝ 4 感情の平均，相性 ＝ 信頼度 − 水準",
        size=fs - 0.4, weight="bold")
    txt(ax, W - 1, 0.2, "点の分布は説明のために作った例で，実験の値ではない", ha="right", va="bottom", size=FS - 1.6, color=GREY)
    save(fig, "fig_prop_reliability_ja")
    return auc, dval


def fig_table():
    setup("ja")
    H = 90
    fig, ax = canvas(H)
    fs = FS + 0.3
    mods = ["A", "V", "T"]
    prof = {"A": [-0.25, -0.10, 0.05, 0.30], "V": [-0.45, 0.55, -0.15, 0.05], "T": [-0.20, 0.10, 0.05, 0.05]}
    level = {"A": 1.2, "V": 1.9, "T": 1.0}
    Dt = {m: [max(0.0, level[m] + r) for r in prof[m]] for m in mods}
    mean = np.mean([v for m in mods for v in Dt[m]])
    Wt = {m: [v / mean for v in Dt[m]] for m in mods}
    z = {"A": [1.0, -0.3, -0.4, -0.3], "V": [-0.4, 1.0, -0.3, -0.3], "T": [0.9, -0.2, -0.4, -0.3]}
    s = [sum(Wt[m][i] * z[m][i] for m in mods) for i in range(4)]
    s_eq = [sum(z[m][i] for m in mods) for i in range(4)]

    def table(x, y, title, vals, col, fmt="{:+.2f}", cw=10.5, ch=5.2, note=None):
        txt(ax, x, y + 3.2, title, size=fs - 0.2, weight="bold", color=col)
        for i, e in enumerate(EMO):
            txt(ax, x + 9 + (i + 0.5) * cw, y - 1.0, e, ha="center", size=fs - 1.2, color=GREY)
        for r, m in enumerate(mods):
            yy = y - 3 - (r + 1) * ch
            txt(ax, x, yy + ch / 2, MODN[m], va="center", size=fs - 0.8)
            for i in range(4):
                box(ax, x + 9 + i * cw + 0.3, yy, cw - 0.6, ch - 0.6, fc="white", ec="#AAAAAA", lw=0.4, r=0.4)
                txt(ax, x + 9 + (i + 0.5) * cw, yy + ch / 2 - 0.3, fmt.format(vals[m][i]).replace("-", "−"), ha="center",
                    va="center", size=fs - 1.0)
        if note:
            txt(ax, x, y - 3 - 3 * ch - 3.5, note, size=fs - 1.6, color=GREY, va="top", ls=1.2)

    # row 1: profile + level -> table -> weights
    table(2, H - 6, "① 感情ごとの形 r（ソースで測る）", prof, C_SRC, note="行の平均は 0")
    txt(ax, 54.5, H - 19, "＋", size=fs + 4, va="center")
    # level column
    x = 58
    txt(ax, x, H - 2.8, "② 強さ L（K 本で）", size=fs - 0.2, weight="bold", color=C_K)
    for r, m in enumerate(mods):
        yy = H - 9 - (r + 1) * 5.2
        txt(ax, x, yy + 2.3, MODN[m], va="center", size=fs - 0.8)
        box(ax, x + 9, yy, 12, 4.6, fc="white", ec="#AAAAAA", lw=0.4, r=0.4)
        txt(ax, x + 15, yy + 2.3, f"{level[m]:.1f}", ha="center", va="center", size=fs - 1.0)
    arrow(ax, (82, H - 22), (88, H - 22), lw=0.9, head=7)
    table(89, H - 6, "③ 信頼度の表 D ＝ max(0, L ＋ r)", Dt, "black", fmt="{:.2f}")
    arrow(ax, (142, H - 22), (146, H - 22), lw=0.9, head=7)
    txt(ax, 147, H - 2.8, "④ 重み W", size=fs - 0.2, weight="bold", color=C_PROP)
    txt(ax, 147, H - 9, "D を 12 個の\n平均で割る\n\n顔の H " + f"{Wt['V'][1]:.2f}" + "\n声の H " + f"{Wt['A'][1]:.2f}" + "\n言葉の H " + f"{Wt['T'][1]:.2f}",
        size=fs - 1.0, va="top", ls=1.3)
    # row 2: one utterance
    y2 = H - 44
    ax.plot([2, W - 2], [y2 + 9, y2 + 9], color="#CCCCCC", lw=0.6)
    table(2, y2 + 2, "⑤ 候補の発話 1 つの点 z（3 つの分類器）", z, "black")
    txt(ax, 56, y2 + 4, "⑥ 感情ごとに，重みを掛けて足す\n　 s(i) ＝ Σ_k W(k, i) × z_k(i)", size=fs - 0.2, weight="bold", va="top", ls=1.3)
    for i, e in enumerate(EMO):
        xx = 75 + i * 10
        txt(ax, xx, y2 - 6, e, ha="center", size=fs - 1.0, color=GREY)
        txt(ax, xx, y2 - 11, f"{s[i]:+.2f}".replace("-", "−"), ha="center", size=fs, weight="bold" if i == int(np.argmax(s)) else "normal",
            color=C_PROP if i == int(np.argmax(s)) else "black")
        txt(ax, xx, y2 - 16.5, f"{s_eq[i]:+.2f}".replace("-", "−"), ha="center", size=fs - 1.0, color=GREY)
    txt(ax, 56, y2 - 11, "提案", size=fs - 1.0, color=C_PROP)
    txt(ax, 116, y2 - 11, "→ " + EMO[int(np.argmax(s))], size=fs, weight="bold", color=C_PROP)
    txt(ax, 116, y2 - 16.5, "→ " + EMO[int(np.argmax(s_eq))], size=fs, color=GREY)
    txt(ax, 56, y2 - 16.5, "同じ重み", size=fs - 1.0, color=GREY)
    txt(ax, 56, y2 - 21, "同じ重み（積と同じ答え）は声と言葉に引かれて N．\n提案は強い顔の H を重く数えて H",
        size=fs - 1.3, color="#333333", va="top", ls=1.3)
    box(ax, 126, y2 - 26, 46, 33, fc="#FDF0E8", ec=C_PROP, lw=1.0)
    txt(ax, 128, y2 + 4.5, "⑦ 切片をずらして疑似ラベルに", size=fs - 0.2, weight="bold", color=C_PROP, va="top")
    txt(ax, 128, y2 - 1.5, "答えが 1 つの感情に寄らない\nように，ラベルを使わずに\n切片 c をずらす\n\nq ＝ softmax(s ＋ c) を\n候補に付けて疑似ラベルにし，\nAVT を学び直す",
        size=fs - 1.2, va="top", ls=1.3)
    txt(ax, W - 1, 0.2, "表と点の値は説明のための例で，実験の値ではない．N：Neutral，H：Happy，S：Sad，A：Angry", ha="right", va="bottom",
        size=FS - 1.6, color=GREY)
    save(fig, "fig_prop_table_ja")


if __name__ == "__main__":
    fig_why()
    fig_corpora()
    print("auc, d =", fig_reliability())
    fig_table()
