"""Slide figures (2026-11-16 night, researcher: "p15について，もう少し詳しく図を書いて伝わるようにして"):

fig_margin_toy_ja   : how the margin (当たる幅) is computed, on 8 made-up candidate utterances
                      (true labels, judgments of the pseudo-label source, judgments of the classifier before retraining,
                      per-emotion recall, UAR, margin), and how it is paired with the gain on the test utterances.
fig_margin_real_ja  : the two measured examples with the two candidate UARs that make the margin
                      (teacher_uar_pool = (a), self_uar_pool = (b)) and the gain with its 95% interval.
Values of the real examples: results/dai1/stats/teacher_margin.csv
  (MOSEI, student A, K = 25, teacher pair_VT; AFEW, student V, K = 5, teacher pair_AT).
Output: ../figures/<name>.pdf and .png
Run:  /home/snics/MER-DA/.venv/bin/python make_fig_margin_detail.py
"""
from pathlib import Path

import matplotlib.patches as mp
import pandas as pd

from make_figs_overview import C_PROP, C_U, FS, GREY, W, arrow, box, canvas, save, setup, txt

ROOT = Path(__file__).resolve().parents[2]
C_OK, C_NG = "#009E73", "#D55E00"
C_UP, C_DOWN = "#0072B2", "#D55E00"
EMO = ["N", "H", "S", "A"]


def fig_toy():
    setup("ja")
    H = 84
    fig, ax = canvas(H)
    fs = FS + 0.5
    true = ["N", "N", "H", "H", "S", "S", "A", "A"]
    rows = [("正解（解析のときだけ使う）", true, "black"),
            ("ⓐ 疑似ラベルの出所の判定", ["N", "N", "H", "H", "S", "A", "A", "A"], C_PROP),
            ("ⓑ 学び直す前の分類器（K 本の基準）の判定", ["N", "H", "H", "N", "N", "S", "A", "N"], "black")]
    txt(ax, 2, H - 3, "① ターゲットの候補の発話（K 本を除く．ここでは 8 本の例）", size=fs, weight="bold")
    x0, cw, ch = 66, 8.2, 7.5
    for i in range(8):
        txt(ax, x0 + (i + 0.5) * cw, H - 9.5, f"発話 {i + 1}", ha="center", va="center", size=fs - 1.8, color=GREY)
    ys = [H - 20, H - 30, H - 40]
    for (name, pred, col), y in zip(rows, ys):
        txt(ax, 2, y + ch / 2, name, va="center", size=fs - 0.3, color=col, weight="bold" if col != "black" else "normal")
        for i, p in enumerate(pred):
            ok = p == true[i]
            fc = "#F2F2F2" if col == "black" and name.startswith("正解") else ("#E5F5EF" if ok else "#FBE3D6")
            box(ax, x0 + i * cw + 0.4, y, cw - 0.8, ch, fc=fc, ec="#999999", lw=0.5, r=0.6, z=2)
            mark = "" if name.startswith("正解") else (" ○" if ok else " ×")
            txt(ax, x0 + (i + 0.5) * cw, y + ch / 2, p + mark, ha="center", va="center", size=fs - 0.3,
                color="black" if name.startswith("正解") else (C_OK if ok else C_NG), weight="bold")
    # per-emotion recall and UAR
    xr = x0 + 8 * cw + 3
    txt(ax, xr, H - 9.5, "感情ごとの再現率 → UAR", size=fs - 1.0, color=GREY)
    for (name, pred, col), y in zip(rows[1:], ys[1:]):
        rec = [sum(1 for t, p in zip(true, pred) if t == e and p == e) / 2 for e in EMO]
        uar = 100 * sum(rec) / 4
        s = "，".join(f"{e} {r:g}" for e, r in zip(EMO, rec))
        txt(ax, xr, y + ch / 2 + 1.6, s, va="center", size=fs - 1.2)
        txt(ax, xr, y + ch / 2 - 2.0, f"UAR {uar:.1f}", va="center", size=fs, weight="bold", color=col)
    txt(ax, 2, ys[2] - 4, "○ は正解と同じ判定，× は違う判定．N：Neutral，H：Happy，S：Sad，A：Angry．"
        "UAR は 4 感情の再現率の平均", size=fs - 1.2, color=GREY)
    # margin
    box(ax, 2, 11, 80, 22, fc="#FDF0E8", ec=C_PROP, lw=1.1)
    txt(ax, 5, 29.5, "② 当たる幅 ＝ ⓐ の UAR − ⓑ の UAR", size=fs, weight="bold", color=C_PROP)
    txt(ax, 5, 24.5, "この例では 87.5 − 50.0 ＝ +37.5 点．\n疑似ラベルのほうが，学び直す前の分類器より\n37.5 点多く当たる．負なら，疑似ラベルのほうが当たらない",
        size=fs - 0.6, ls=1.3, va="top")
    arrow(ax, (82, 22), (92, 22), lw=1.0, head=8)
    box(ax, 92, 11, 80, 22, fc="#E8F0F8", ec=C_UP, lw=1.1)
    txt(ax, 95, 29.5, "③ gain：この疑似ラベルで学び直して比べる", size=fs, weight="bold", color=C_UP)
    txt(ax, 95, 24.5, "学び直す前の分類器を，ⓐ の判定の確率を\n疑似ラベルにして学び直す．テストの発話で\ngain ＝ 学び直した分類器 − 学び直す前の分類器",
        size=fs - 0.6, ls=1.3, va="top")
    txt(ax, 2, 8.0, "④ 疑似ラベルの出所を替えて，②（当たる幅）と ③（gain）の組を K = 5 と 25 で 360 組ずつ測り，2 つの関係を調べる", size=fs - 0.6)
    txt(ax, 2, 3.6, "8 本の発話と判定は説明のための例で，実験の値ではない", size=fs - 1.4, color=GREY)
    save(fig, "fig_margin_toy_ja")


def fig_real():
    setup("ja")
    d = pd.read_csv(ROOT / "results" / "dai1" / "stats" / "teacher_margin.csv", comment="#")
    ex = [("例 1：MOSEI，K = 25", "声の分類器を，顔と言葉をつないだ分類器の\n疑似ラベルで学び直す", "MOSEI", "A", 25, "pair_VT", C_UP, "上がった"),
          ("例 2：AFEW，K = 5", "顔の分類器（AFEW で最も強い）を，声と言葉を\nつないだ分類器の疑似ラベルで学び直す", "AFEW", "V", 5, "pair_AT", C_DOWN, "下がった")]
    H = 74
    fig, ax = canvas(H)
    fs = FS + 0.5
    for j, (head, desc, T, st, K, tea, col, word) in enumerate(ex):
        r = d[(d["T"] == T) & (d.student == st) & (d.K == K) & (d.teacher == tea)].iloc[0]
        x = 2 + j * 86
        box(ax, x, 2, 84, H - 4, fc="white", ec=col, lw=1.2)
        txt(ax, x + 2, H - 5.5, head, size=fs + 0.4, weight="bold", color=col)
        txt(ax, x + 2, H - 9, desc, size=fs - 0.8, va="top", ls=1.2)
        # candidate UARs
        txt(ax, x + 2, H - 20, "候補の上の UAR（K 本を除く）", size=fs - 0.6, weight="bold")
        sc = 0.62
        bx = x + 26
        for k, (lab, v, c) in enumerate([("ⓐ 疑似ラベル", r.teacher_uar_pool, C_PROP), ("ⓑ 学び直す前", r.self_uar_pool, "#555555")]):
            y = H - 26.5 - k * 6.5
            txt(ax, x + 3, y, lab, va="center", size=fs - 0.6, color=c)
            ax.add_patch(mp.Rectangle((bx, y - 2.2), v * sc, 4.4, fc=c, ec="none", alpha=0.85, zorder=3))
            txt(ax, bx + v * sc + 1, y, f"{v:.1f}", va="center", size=fs - 0.4, weight="bold")
        txt(ax, x + 3, H - 41, f"当たる幅 ＝ ⓐ − ⓑ ＝ {r.teacher_uar_pool:.1f} − {r.self_uar_pool:.1f} ＝ {r.margin:+.1f} 点".replace("-", "−").replace("＝ −", "＝ −"),
            size=fs - 0.2, weight="bold", color=C_PROP)
        # gain
        txt(ax, x + 2, H - 49, "テストの UAR の差（gain．学び直し − 学び直す前）", size=fs - 0.6, weight="bold")
        x0 = x + 42
        gs = 4.0
        ax.plot([x0, x0], [10, 20], color="black", lw=0.6)
        ax.add_patch(mp.Rectangle((min(x0, x0 + r.gain * gs), 14), abs(r.gain) * gs, 4, fc=col, ec="none", zorder=3))
        lab = f"{r.gain:+.2f}".replace("-", "−")
        txt(ax, x0 + r.gain * gs + (1 if r.gain > 0 else -1), 16, lab, ha="left" if r.gain > 0 else "right", va="center",
            size=fs, weight="bold")
        ax.plot([x0 + r.gain_lo * gs, x0 + r.gain_hi * gs], [11.8, 11.8], color=col, lw=0.9)
        for e in (r.gain_lo, r.gain_hi):
            ax.plot([x0 + e * gs] * 2, [11.0, 12.6], color=col, lw=0.9)
        txt(ax, x0, 9.6, "0", ha="center", va="top", size=fs - 1.2, color=GREY)
        txt(ax, x + 3, 6.2, f"95% 区間 [{r.gain_lo:+.2f}, {r.gain_hi:+.2f}] は 0 を含まない → UAR が{word}".replace("-", "−"),
            size=fs - 0.9, color=col, weight="bold")
    save(fig, "fig_margin_real_ja")


if __name__ == "__main__":
    fig_toy()
    fig_real()
