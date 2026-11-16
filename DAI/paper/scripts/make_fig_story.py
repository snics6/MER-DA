"""Slide figure (2026-11-16 night, researcher: "前半の解析が後半にどう繋がるとか，そこら辺のつながりが伝わるように"):
fig_story_ja : the three questions in a row, each with its answer, and what each answer decided for the next step.
Text only (no values beyond the slide pages). Output: ../figures/fig_story_{ja,en}.pdf and .png
fig_story_en (2026-11-17): the English version for the manuscript (text 8-12 pt at the final width of 174 mm).
The Japanese version is drawn exactly as before.
Run:  /home/snics/MER-DA/.venv/bin/python make_fig_story.py [ja|en]
"""
import sys

from make_figs_overview import C_K, C_PROP, C_SRC, FS, GREY, W, W_EN, arrow, box, canvas, save, setup, txt

Q1A, Q1B, Q2 = "#8E2457", "#3B2F8F", "#8A5A00"


T = {
    "ja": dict(
        cols=[("問い 1(a)（解析）", "4 つの情報源は，答えの偏りと\n残りの誤りのどちらを直すか",
               "偏りは，ラベル無しで特徴の平均\nを引くだけでほぼ直った（声と顔）．\n残りの誤りは，ほかのモダリティ\nを含む疑似ラベルで減った"),
              ("問い 1(b)（解析）", "疑似ラベルで学び直すと，\nどの分類器の UAR が\nどんな場合に上がるか",
               "声の分類器は 12 条件すべてで\n上がった（点推定）．はっきり\n上がったのは，疑似ラベルが学び\n直す前の分類器より当たるときだけ"),
              ("問い 2（手法）", "疑似ラベルを作るために，\n3 つの分類器の答えを\nどうまとめるか",
               "強さを K 本で，感情ごとの形を\nソースで測った重みでまとめると，\n積を 12 条件のうち 11 で上回った\n（点推定．AFEW で +2.1〜+3.5 点）")],
        q="問い", a="答え", links_head="答えが次の段で決めたこと",
        links=[("問い 1(a) → 問い 1(b)", "残りの誤りを減らした疑似ラベルに絞り，\nどんな場合に UAR が上がるかを調べる"),
               ("問い 1(b) → 問い 2", "疑似ラベルを当てることが要る．\nそこで答えのまとめ方を変えて当てにいく"),
               ("問い 2 の中の決め事", "強いモダリティはコーパスと感情ごとに違う → 分類器 × 感情で重みを変える．\nK 本は少ない → K 本で決めるのは強さの 3 つだけ．感情ごとの形はソースで，\nハイパーパラメータはソースのエピソードで決める")],
    ),
    "en": dict(
        cols=[("RQ1(a) (analysis)", "Which part of the error do the\nfour sources correct: answer\nbias or remaining error?",
               "Subtracting the feature mean\nwithout labels almost removed\nthe bias (audio and visual;\nsingle-source). Pseudo-labels\n"
               "that include other modalities\nreduced the remaining error\n(all-source)."),
              ("RQ1(b) (analysis)", "Retrained on pseudo-labels,\nwhich classifiers gain UAR,\nand when?",
               "The audio classifier gained in\nall 12 conditions (point\nestimates). UAR rose (95%\ninterval above zero) only\n"
               "when the pseudo-labels had\nhigher UAR than the classifier\nbefore retraining."),
              ("RQ2 (method)", "How should the answers of the\nthree classifiers be combined\ninto pseudo-labels?",
               "Fusion with strengths from the\nK labels and per-emotion\nprofiles from the sources gave\nhigher UAR than the product in\n"
               "11 of 12 conditions (point\nestimates; +2.1 to +3.5\npoints on AFEW).")],
        q=None, a=None,   # the question is set in italics and the answer in colour, without the small labels
        links_head="What each answer decided for the next step",
        links=[("RQ1(a) → RQ1(b)", "Focus on the pseudo-labels that reduced the remaining error, and\nexamine when UAR rises."),
               ("RQ1(b) → RQ2", "UAR rose only with pseudo-labels of higher UAR, so change how the\nanswers are combined to raise the pseudo-labels’ UAR."),
               ("Within RQ2", "The strongest modality differed by corpus and emotion → weights vary\nby classifier × emotion. "
                "K labels are few → only the 3 strengths come\nfrom the K labels; the per-emotion profile comes from the sources,\n"
                "and the hyperparameters from episodes on the sources.")],
    ),
}


def main(lang="ja"):
    """The Japanese layout is the one drawn before the English version was added (first argument of L)."""
    setup(lang)
    t = T[lang]
    L = (lambda ja, en: ja) if lang == "ja" else (lambda ja, en: en)
    H = L(92, 78)
    fig, ax = canvas(H, L(W, W_EN))
    fs = FS + 0.6
    cols = [(c,) + x for c, x in zip((Q1A, Q1B, Q2), t["cols"])]
    bw = L(52, 41.1)
    xs = L([2, 60, 118], [0.6, 45.05, 89.5])
    yb, yt = L(44, 37.0), L(88, 77.4)     # bottom and top of the three boxes
    hh = L(8, 6.4)                     # height of the coloured header
    for (col, head, q, a), x in zip(cols, xs):
        box(ax, x, yb, bw, yt - yb, fc="white", ec=col, lw=1.3)
        box(ax, x, yt - hh, bw, hh, fc=col, ec=col, lw=1.3, r=1.0, z=2)
        txt(ax, x + bw / 2, yt - hh / 2, head, ha="center", va="center", size=L(fs + 0.4, 9), color="white", weight="bold", z=3)
        if lang == "ja":
            txt(ax, x + 3, yt - 11, t["q"], size=fs - 1.2, color=GREY, va="top")
            txt(ax, x + 3, yt - 15, q, size=fs - 0.8, va="top", ls=1.25)
            txt(ax, x + 3, yt - 27, t["a"], size=fs - 1.2, color=GREY, va="top")
            txt(ax, x + 3, yt - 31, a, size=fs - 1.4, va="top", ls=1.25, color=col, weight="bold")
        else:
            txt(ax, x + 1.6, yt - hh - 1.2, q, size=8, va="top", ls=1.1, style="italic")
            ax.plot([x + 1.6, x + bw - 1.6], [yt - hh - 11.6] * 2, color=col, lw=0.6)
            txt(ax, x + 1.6, yt - hh - 12.6, a, size=8, va="top", ls=1.1, color=col)
    for x in L((54.5, 112.5), (41.85, 86.3)):
        arrow(ax, (x, L(66, 58.5)), (x + L(5, 3.0), L(66, 58.5)), lw=L(1.6, 1.3), head=L(10, 7), color="#555555")
    # what each answer decided
    txt(ax, L(2, 0.6), L(39, 34.8), t["links_head"], size=L(fs + 0.2, 8.5), weight="bold")
    y = L(33, 30.0)
    for col, (h, body) in zip((Q1A, Q1B, Q2), t["links"]):
        nb = body.count("\n") + 1
        bh = L(8.5, 3.1 * nb + 0.2)
        box(ax, L(2, 0.6), y - L(9, bh + 0.2), L(3, 2.4), bh, fc=col, ec=col, lw=0.6, r=0.5)
        txt(ax, L(8, 4.4), y - 0.5, h, size=L(fs - 0.6, 8), weight="bold", color=col, va="top")
        txt(ax, L(46, 29.6), y - 0.5, body, size=L(fs - 1.4, 8), va="top", ls=L(1.25, 1.1))
        y -= L(10.5, bh + 1.6)
    save(fig, f"fig_story_{lang}")


if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["ja", "en"]):
        main(lang)
