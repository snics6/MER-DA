"""Fig. kcurves：AVT の世界（第 1 エンコーダ）の 4 生徒で，K に対する E39 との UAR の差．

線は ours（E60 の選んだ形），oursx（拡張版），最も強い相手の調整版（K ごとに選び直す．略称を点の脇に書く）．
出所と集計は台帳 comparison.py と同じ（fig_compare.world_table → comparison._world）：
  results/baselines_fuse-audio-video-text/<ターゲット>.csv，results/multimodal_distill_fuse-audio-video-text/<ターゲット>.csv
値は反復 0〜9 の平均をターゲットごとに取り，4 ターゲット（IEMOCAP 即興・台本，MELD，MOSEI）で平均した差．
"""
from __future__ import annotations

import fig_compare as F
import style

STUDENTS = [("audio_wavlm", "A student"), ("video", "V student"), ("text_roberta", "T student"),
            ("fuse-audio-video-text", "AVT student")]
KS = ["0", "1", "5", "10", "25"]


def main() -> None:
    style.setup()
    fig, axes = style.plt.subplots(2, 2, figsize=(style.COL, 2.75), sharex=True, sharey=True, layout="constrained")
    xs = list(range(len(KS)))
    for ax, (s, title), tag in zip(axes.flat, STUDENTS, "abcd"):
        best = [F.best_of("avt", s, K) for K in KS]
        lab = [F.best_label("avt", s, K) for K in KS]
        yb = [F.diffs("avt", s, K, f"{b}:tuned").mean() for K, b in zip(KS, best)]
        for key, col, mk in (("best", None, "s"), ("ours", "ours", "o"), ("oursx", "oursx:tuned", "D")):
            y = yb if key == "best" else [F.diffs("avt", s, K, col).mean() for K in KS]
            ax.plot(xs, y, color=style.METHOD[key]["color"], marker=mk, ms=3.2, lw=1.2,
                    mec="white", mew=0.4, label=style.METHOD[key]["label"], zorder=3)
        for x, t in zip(xs, lab):                    # 最も強い相手の略称（K ごと）を下の段に並べる
            ax.text(x, -1.95, t, fontsize=6.5, ha="center", va="bottom", color=style.METHOD["best"]["color"])
        ax.axhline(0, color=style.INK2, lw=0.6, zorder=1)
        ax.text(0.03, 0.97, f"({tag}) {title}", transform=ax.transAxes, fontsize=7, va="top", ha="left")
        ax.set_xticks(xs)
        ax.set_xticklabels(KS)
    for ax in axes[-1]:
        ax.set_xlabel("Labels per emotion, K")
    for ax in axes[:, 0]:
        ax.set_ylabel("UAR − start (points)")
    axes[0, 0].set_ylim(-2.05, 6.1)
    axes[0, 0].set_yticks([0, 2, 4])
    axes[0, 0].set_xlim(-0.4, len(KS) - 0.6)
    h, l = axes[0, 0].get_legend_handles_labels()
    order = [2, 1, 0]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="outside upper center",
               ncol=2, columnspacing=1.0, handlelength=1.8, borderaxespad=0.0)
    fig.get_layout_engine().set(hspace=0.02, wspace=0.02)
    style.save(fig, "kcurves")


if __name__ == "__main__":
    main()
