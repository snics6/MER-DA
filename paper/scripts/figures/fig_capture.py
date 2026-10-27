"""Fig. capture：出発点の向きの誤り E_0 を，K=5 の 20 本が張る空間が捕らえる割合（ランダムに対する倍率）．

出所は台帳 part2.py の E58 の部分と同じ：results/explore_relation/snapshots/20261021-2245_e58/e58_targets.csv
（1 行 = 種類 × ターゲット．反復 0〜9 の平均）．
- 点：ターゲットの平均の r(S) どうしの比 r(S)／r(random_true)（台帳の p2-e58-ratio-* と同じ計算）
- 横線：ターゲットごとの比 r(S)／r(random_true) の最小から最大
ラベル無しの選び方 7 通り（台帳の unlab と同じ）と，条件 1 の教える選び方 LLL（候補のラベルを使う上限）．
"""
from __future__ import annotations

import pandas as pd
from matplotlib.lines import Line2D

import style

SRC = style.RESULTS / "explore_relation" / "snapshots" / "20261021-2245_e58" / "e58_targets.csv"
KINDS = [("audio_wavlm", "a"), ("video", "v"), ("text_roberta", "t")]
# 上から下へ描く順．最後の 1 つは候補のラベルを使う
RULES = [
    ("random_pool", "Random (pool order)"),
    ("margin", "Smallest margin"),
    ("herding_pred", "Herding on predictions"),
    ("typiclust", "TypiClust"),
    ("coreset_raw", "Core-set (k-center)"),
    ("kcenter_white", "k-center, whitened"),
    ("norm_white", "Largest whitened norm"),
    ("LLL", "Teaching with pool labels"),
]


def main() -> None:
    style.setup()
    d = pd.read_csv(SRC, comment="#")
    R = d.groupby("kind")[[f"r_{s}" for s, _ in RULES] + ["r_random_true"]].mean()

    fig, ax = style.plt.subplots(figsize=(style.COL, 2.25), layout="constrained")
    n = len(RULES)
    ys = {s: n - 1 - i + (0 if s != "LLL" else -0.35) for i, (s, _) in enumerate(RULES)}
    off = {"a": 0.22, "v": 0.0, "t": -0.22}
    for kind, tok in KINDS:
        c = style.CONFIG[tok]
        g = d[d.kind == kind]
        for s, _ in RULES:
            y = ys[s] + off[tok]
            per = g[f"r_{s}"] / g["r_random_true"]
            ax.plot([per.min(), per.max()], [y, y], color=c["color"], lw=0.8, alpha=0.55, solid_capstyle="butt")
            ax.plot(R.loc[kind, f"r_{s}"] / R.loc[kind, "r_random_true"], y, marker=c["marker"], ls="none",
                    color=c["color"], mec="white", mew=0.4, ms=4.2, zorder=3)
    ax.axvline(1.0, color=style.INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(1.03, n - 0.6, "random draw", color=style.INK2, fontsize=6.5, ha="left", va="bottom")
    sep = (ys["norm_white"] + ys["LLL"]) / 2
    ax.axhline(sep, color=style.INK2, lw=0.5)
    ax.text(2.48, sep + 0.08, "label-free rules", fontsize=6.5, color=style.INK2, ha="right", va="bottom")
    ax.set_yticks([ys[s] for s, _ in RULES])
    ax.set_yticklabels([lab for _, lab in RULES])
    ax.set_ylim(ys["LLL"] - 0.5, n + 0.05)
    ax.set_xlim(0.3, 2.5)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel(r"Share of $E_0$ in the span of 20 labels (× random)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    hs = [Line2D([], [], color=style.CONFIG[t]["color"], marker=style.CONFIG[t]["marker"], ls="-", lw=0.8,
                 mec="white", mew=0.4, ms=4.2, label=style.CONFIG[t]["label"]) for _, t in KINDS]
    ax.legend(handles=hs, loc="center right", bbox_to_anchor=(1.0, 0.40), ncol=1, handlelength=1.4, borderaxespad=0.2)
    style.save(fig, "capture")


if __name__ == "__main__":
    main()
