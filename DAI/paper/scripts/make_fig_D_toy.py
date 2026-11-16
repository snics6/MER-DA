"""Slide figure (2026-11-16 night, researcher: "これぐらいの説明がないとわからんね" for the D page):

fig_D_toy_ja : how R, R' and D are computed, on 8 made-up utterances in two (emotion, speaker) groups.
  J  = the text classifier's answer (the target), predicted from the voice features -> R (agreement with J)
  J' = J shuffled within each (true emotion, speaker cluster) group, predicted again -> R'
  D  = R - R' : agreement that the true emotion and the speaker cluster do not explain.
For readability the toy counts the agreement rate; the experiment uses the UAR with J (or J') as the truth.
Illustration only: not experimental values.
Output: ../figures/fig_D_toy_ja.pdf and .png
Run:  /home/snics/MER-DA/.venv/bin/python make_fig_D_toy.py
"""
from make_figs_overview import FS, GREY, W, arrow, box, canvas, save, setup, txt

C_OK, C_NG = "#009E73", "#D55E00"
C_A, C_T = "#E69F00", "#009E73"


def main():
    setup("ja")
    H = 92
    fig, ax = canvas(H)
    fs = FS + 0.5
    box(ax, 2, H - 14, W - 4, 12, fc="#F5F5F5", ec="#999999", lw=0.6)
    txt(ax, 4, H - 4.2, "ねらい：積の疑似ラベルには言葉の分類器の答えが入っている．声の特徴から言葉の分類器の答えを真似できるなら，",
        size=fs - 0.6)
    txt(ax, 4, H - 9.6, "声の分類器はその分を疑似ラベルから学べるはずである．真似できる度合いを，声と顔の特徴で測って比べる",
        size=fs - 0.6)
    group = ["感情 N・話者 a"] * 4 + ["感情 A・話者 b"] * 4
    J = ["N", "H", "N", "S", "A", "A", "S", "A"]
    PJ = ["N", "H", "N", "N", "A", "A", "S", "A"]
    Js = ["H", "N", "S", "N", "A", "S", "A", "A"]
    PJs = ["N", "N", "N", "N", "A", "A", "A", "A"]
    x0, cw, ch = 70, 9.2, 6.6
    yt = H - 21
    for i in range(8):
        txt(ax, x0 + (i + 0.5) * cw, yt, f"発話 {i + 1}", ha="center", va="center", size=fs - 1.8, color=GREY)
    for g, gx in ((0, 0), (1, 4)):
        box(ax, x0 + gx * cw + 0.3, yt - 9, 4 * cw - 0.6, 5.6, fc="#EEEEEE", ec="#999999", lw=0.5, r=0.6)
        txt(ax, x0 + (gx + 2) * cw, yt - 6.2, group[gx], ha="center", va="center", size=fs - 0.6)
    txt(ax, 2, yt - 6.2, "本当の感情・話者のまとまり", va="center", size=fs - 0.4)

    def row(y, name, vals, ref, col, bold=False):
        txt(ax, 2, y + ch / 2, name, va="center", size=fs - 0.5, color=col, weight="bold" if bold else "normal")
        for i, v in enumerate(vals):
            if ref is None:
                fc, s, c = "white", v, col
            else:
                ok = v == ref[i]
                fc, s, c = ("#E5F5EF", v + " ○", C_OK) if ok else ("#FBE3D6", v + " ×", C_NG)
            box(ax, x0 + i * cw + 0.4, y, cw - 0.8, ch, fc=fc, ec="#999999", lw=0.5, r=0.6, z=2)
            txt(ax, x0 + (i + 0.5) * cw, y + ch / 2, s, ha="center", va="center", size=fs - 0.4, color=c, weight="bold")

    y1 = yt - 18
    row(y1, "① 言葉の分類器の答え J（これを目標にする）", J, None, C_T, True)
    row(y1 - 8, "② 声の特徴だけで J を当てた答え", PJ, J, C_A, True)
    txt(ax, x0 + 8 * cw + 2, y1 - 8 + ch / 2, "R ＝ 7/8\n＝ 87.5", va="center", size=fs, weight="bold", ls=1.15)
    row(y1 - 19, "③ J を同じまとまりの中で入れ替えた J′", Js, None, "#555555")
    row(y1 - 27, "　 声の特徴だけで J′ を当てた答え", PJs, Js, C_A)
    txt(ax, x0 + 8 * cw + 2, y1 - 27 + ch / 2, "R′ ＝ 5/8\n＝ 62.5", va="center", size=fs, weight="bold", ls=1.15)
    txt(ax, 2, y1 - 31.5, "J′ は感情と話者のまとまりを保つので，R′ は「同じ感情・同じ話者だから一致する」分だけを表す．○ は目標と同じ，× は違う",
        size=fs - 1.3, color=GREY)
    box(ax, 2, 3, 112, 14, fc="#FDF0E8", ec="#D55E00", lw=1.1)
    txt(ax, 5, 13.6, "④ D ＝ R − R′ ＝ 87.5 − 62.5 ＝ 25 点", size=fs + 0.4, weight="bold", color="#D55E00")
    txt(ax, 5, 7.6, "感情と話者だけでは説明できない一致の大きさ．顔の特徴でも同じ D を測り，声と比べる",
        size=fs - 0.6)
    txt(ax, 117, 13.2, "例では数えやすいように一致の割合で数えた．\n実験では J を正解とみなした UAR で数え，\n当てる分類器は別の発話で学んで予測する．\n発話と答えは説明のための例で，実験の値ではない",
        size=fs - 1.5, color=GREY, va="top", ls=1.2)
    save(fig, "fig_D_toy_ja")


if __name__ == "__main__":
    main()
