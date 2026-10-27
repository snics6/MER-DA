"""Fig. entmin：K=0 の融合の生徒で，エントロピー最小化の固定版と oursx の，E39 との UAR の差（世界 × ターゲット）．

出所と集計は台帳 comparison.py と同じ（fig_compare.world_table → comparison._world）：
  results/baselines_<世界>/<ターゲット>.csv（世界は AVT・VT・AT・AV．AV は 6 ターゲット，ほかは 4）
- 点：ターゲットごとの差（反復 0〜9 の平均）．台帳の cmp-entmin-fixed-vs-e39-<世界>-f-k0-<ターゲット> と
  cmp-oursx-vs-e39-avt-f-k0-<ターゲット> と同じ値
- 黒い横棒：ターゲットの平均（台帳の cmp-entmin-fixed-vs-e39-<世界>-f-k0 と cmp-oursx-vs-e39-<世界>-f-k0）
- 細い線：同じターゲットの 2 点を結ぶ
エントロピー最小化の固定版は候補 1 本あたり重み 1（entmin|-|n．文献の既定）．
"""
from __future__ import annotations

from matplotlib.lines import Line2D

import fig_compare as F
import style

WORLDS = [("avt", "fuse-audio-video-text", "AVT"), ("vt", "fuse-video-text", "VT"),
          ("at", "fuse-audio-text", "AT"), ("av", "fuse-audio-video", "AV")]
# ターゲットの印（種類の印とは別の図なので，ここだけの対応）
TMARK = {"IEMOCAP-impro": ("o", "IEMOCAP impro."), "IEMOCAP-script": ("s", "IEMOCAP script."),
         "MELD": ("^", "MELD"), "MOSEI": ("D", "CMU-MOSEI"), "AFEW": ("v", "AFEW"), "CREMA-D": ("P", "CREMA-D")}


def main() -> None:
    style.setup()
    fig, ax = style.plt.subplots(figsize=(style.COL, 2.35), layout="constrained")
    dx = 0.19
    for i, (wk, s, lab) in enumerate(WORLDS):
        d = F.world_table(wk)
        g = d[(d.student == s) & (d.K == "0")].set_index("target")
        ent = g["entmin:fixed"] - g["e39"]
        ox = g["oursx:tuned"] - g["e39"]
        for t in g.index:
            m = TMARK[t][0]
            ax.plot([i - dx, i + dx], [ent[t], ox[t]], color="#c9c8c4", lw=0.6, zorder=1)
            ax.plot(i - dx, ent[t], marker=m, ls="none", ms=3.6, color=style.METHOD["entmin"]["color"],
                    mec="white", mew=0.4, zorder=3)
            ax.plot(i + dx, ox[t], marker=m, ls="none", ms=3.6, color=style.METHOD["oursx"]["color"],
                    mec="white", mew=0.4, zorder=3)
        for x, v in ((i - dx, ent.mean()), (i + dx, ox.mean())):
            ax.plot([x - 0.11, x + 0.11], [v, v], color=style.INK, lw=1.1, zorder=4)
    ax.axhline(0, color=style.INK2, lw=0.6, zorder=0)
    ax.set_xticks(range(len(WORLDS)))
    ax.set_xticklabels([f"{w[2]} student" for w in WORLDS])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.5, len(WORLDS) - 0.5)
    ax.set_ylabel("UAR − start at K = 0 (points)")
    meth = [Line2D([], [], marker="o", ls="none", ms=3.6, color=style.METHOD[k]["color"], mec="white", mew=0.4,
                   label=style.METHOD[k]["label"]) for k in ("entmin", "oursx")]
    meth.append(Line2D([], [], color=style.INK, lw=1.1, label="Mean over targets"))
    tg = [Line2D([], [], marker=m, ls="none", ms=3.6, color=style.GRAY, mec="white", mew=0.4, label=n)
          for m, n in TMARK.values()]
    leg = ax.legend(handles=meth, loc="lower left", borderaxespad=0.2, handlelength=1.2)
    ax.add_artist(leg)
    ax.legend(handles=tg, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, borderaxespad=0.1,
              handlelength=1.0, columnspacing=1.0)
    ax.set_ylim(-42, 7)
    style.save(fig, "entmin")


if __name__ == "__main__":
    main()
