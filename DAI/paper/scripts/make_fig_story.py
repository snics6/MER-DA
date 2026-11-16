"""Slide figure (2026-11-16 night, researcher: "前半の解析が後半にどう繋がるとか，そこら辺のつながりが伝わるように"):
fig_story_ja : the three questions in a row, each with its answer, and what each answer decided for the next step.
Text only (no values beyond the slide pages). Output: ../figures/fig_story_ja.pdf and .png
Run:  /home/snics/MER-DA/.venv/bin/python make_fig_story.py
"""
from make_figs_overview import C_K, C_PROP, C_SRC, FS, GREY, W, arrow, box, canvas, save, setup, txt

Q1A, Q1B, Q2 = "#8E2457", "#3B2F8F", "#8A5A00"


def main():
    setup("ja")
    H = 92
    fig, ax = canvas(H)
    fs = FS + 0.6
    cols = [
        (Q1A, "問い 1(a)（解析）", "4 つの情報源は，答えの偏りと\n残りの誤りのどちらを直すか",
         "偏りは，ラベル無しで特徴の平均\nを引くだけでほぼ直った（声と顔）．\n残りの誤りは，ほかのモダリティ\nを含む疑似ラベルで減った"),
        (Q1B, "問い 1(b)（解析）", "疑似ラベルで学び直すと，\nどの分類器の UAR が\nどんな場合に上がるか",
         "声の分類器は 12 条件すべてで\n上がった（点推定）．はっきり\n上がったのは，疑似ラベルが学び\n直す前の分類器より当たるときだけ"),
        (Q2, "問い 2（手法）", "疑似ラベルを作るために，\n3 つの分類器の答えを\nどうまとめるか",
         "強さを K 本で，感情ごとの形を\nソースで測った重みでまとめると，\n積を 12 条件のうち 11 で上回った\n（点推定．AFEW で +2.1〜+3.5 点）"),
    ]
    xs = [2, 60, 118]
    for (col, head, q, a), x in zip(cols, xs):
        box(ax, x, 44, 52, 44, fc="white", ec=col, lw=1.3)
        box(ax, x, 80, 52, 8, fc=col, ec=col, lw=1.3, r=1.0, z=2)
        txt(ax, x + 26, 84, head, ha="center", va="center", size=fs + 0.4, color="white", weight="bold", z=3)
        txt(ax, x + 3, 77, "問い", size=fs - 1.2, color=GREY, va="top")
        txt(ax, x + 3, 73, q, size=fs - 0.8, va="top", ls=1.25)
        txt(ax, x + 3, 61, "答え", size=fs - 1.2, color=GREY, va="top")
        txt(ax, x + 3, 57, a, size=fs - 1.4, va="top", ls=1.25, color=col, weight="bold")
    for x in (54.5, 112.5):
        arrow(ax, (x, 66), (x + 5, 66), lw=1.6, head=10, color="#555555")
    # what each answer decided
    txt(ax, 2, 39, "答えが次の段で決めたこと", size=fs + 0.2, weight="bold")
    links = [
        (Q1A, "問い 1(a) → 問い 1(b)", "残りの誤りを減らした疑似ラベルに絞り，\nどんな場合に UAR が上がるかを調べる"),
        (Q1B, "問い 1(b) → 問い 2", "疑似ラベルを当てることが要る．\nそこで答えのまとめ方を変えて当てにいく"),
        (Q2, "問い 2 の中の決め事", "強いモダリティはコーパスと感情ごとに違う → 分類器 × 感情で重みを変える．\nK 本は少ない → K 本で決めるのは強さの 3 つだけ．感情ごとの形はソースで，\nハイパーパラメータはソースのエピソードで決める"),
    ]
    y = 33
    for col, h, body in links:
        box(ax, 2, y - 9, 3, 8.5, fc=col, ec=col, lw=0.6, r=0.5)
        txt(ax, 8, y - 0.5, h, size=fs - 0.6, weight="bold", color=col, va="top")
        txt(ax, 46, y - 0.5, body, size=fs - 1.4, va="top", ls=1.25)
        y -= 10.5
    save(fig, "fig_story_ja")


if __name__ == "__main__":
    main()
