"""Fig. ladder：手法の積み上げ（音声 WavLM の生徒）．

(a) UAR の水準：ターゲットのみ，合併（ソース 1 つ），仮説転移 4（P4），5 つの値を選ぶ出発点（E39）．
    7 ターゲット，反復 0〜29．同じコーパスの組は除く．
    - ターゲットのみ（B1）と合併（B4．C はソースのみの C）：results/explore_relation/snapshots/20261017-1654_e31final/e31_finals30.csv
      （組ごとの値をソースで平均してターゲットの値にし，ターゲットで平均）
    - P4 と E39：results/explore_relation/snapshots/20261018-0213_ext30/ext30_targets.csv（span = all）
(b) 出発点（E39）の上に弱いラベルを足した得（UAR − E39．どれも同じ実験の中の E39 との対の差）．反復 0〜9．
    - 同じモダリティの見込みの蒸留（E59 の追加．音声だけの世界，7 ターゲット）：
      results/explore_relation/snapshots/20261022-0055_e59k/e59k_targets.csv の meta − e39
    - 別のモダリティの蒸留（E60 の選んだ形．AV の世界 6 ターゲットと AVT の世界 4 ターゲットの音声の生徒）：
      results/explore_relation/snapshots/20261022-0810_e60/e60_targets.csv の meta − e39
    - 拡張版 oursx（AVT の世界の音声の生徒，4 ターゲット）：台帳 comparison.py と同じ（fig_compare.world_table）
値はターゲットごとに反復を平均し，ターゲットで平均したもの．
"""
from __future__ import annotations

import pandas as pd

import fig_compare as F
import style

SNAP = style.RESULTS / "explore_relation" / "snapshots"
F_E31 = SNAP / "20261017-1654_e31final" / "e31_finals30.csv"
F_X30 = SNAP / "20261018-0213_ext30" / "ext30_targets.csv"
F_E59K = SNAP / "20261022-0055_e59k" / "e59k_targets.csv"
F_E60 = SNAP / "20261022-0810_e60" / "e60_targets.csv"
KS = ["0", "1", "5", "10", "25"]
KIND = "audio_wavlm"


def _read(p):
    return pd.read_csv(p, comment="#", dtype={"K": str})


def levels() -> dict[str, pd.Series]:
    e31 = _read(F_E31)
    e31 = e31[(e31.kind == KIND) & ~e31.same]
    per_t = e31.groupby(["target", "K"])[["B1", "B4"]].mean()          # ソースで平均（B1 はソースに依らない）
    x = _read(F_X30)
    x = x[(x.span == "all") & (x.kind == KIND)]
    if set(x.target) != set(e31.target):
        raise ValueError("fig_ladder: e31 と ext30 のターゲットがそろわない")
    out = {"target": per_t.B1.groupby("K").mean(), "pooled": per_t.B4.groupby("K").mean(),
           "p4": x.groupby("K").P4.mean(), "e39": x.groupby("K").E39.mean()}
    return {k: v.reindex(KS) for k, v in out.items()}


def gains() -> list[tuple[str, pd.Series, int]]:
    e59 = _read(F_E59K)
    e59 = e59[e59.kind == KIND]
    e60 = _read(F_E60)
    out = [("e59", (e59.meta - e59.e39).groupby(e59.K).mean(), e59.target.nunique())]
    for w, key in (("fuse-audio-video", "e60av"), ("fuse-audio-video-text", "e60avt")):
        g = e60[(e60.world == w) & (e60.student == KIND)]
        out.append((key, (g.meta - g.e39).groupby(g.K).mean(), g.target.nunique()))
    d = F.world_table("avt")
    g = d[d.student == KIND]
    out.append(("oursx", (g["oursx:tuned"] - g.e39).groupby(g.K).mean(), g.target.nunique()))
    return [(k, v.reindex(KS), n) for k, v, n in out]


def main() -> None:
    style.setup()
    fig, (a1, a2) = style.plt.subplots(2, 1, figsize=(style.COL, 3.0), sharex=True, layout="constrained")
    xs = list(range(len(KS)))
    L = levels()
    sty = {"target": dict(color=style.INK, ls=":", marker="v", label="Target only"),
           "pooled": dict(color=style.SLOT[3], ls="--", marker="s", label="Pooled"),
           "p4": dict(color=style.SLOT[5], ls="-.", marker="D", label="HT, all sources"),
           "e39": dict(color=style.METHOD["e39"]["color"], ls="-", marker="o", lw=1.8, label="Start")}
    for k, s in sty.items():
        a1.plot(xs, L[k].values, ms=3.2, mec="white", mew=0.4, **{"lw": 1.2, **s})
        a1.text(xs[1] - 0.12, L[k].iloc[1], s["label"], color=s["color"] if k != "pooled" else "#a87200",
                fontsize=6.5, ha="right", va="center")
    nT = _read(F_X30).query("span == 'all' and kind == @KIND").target.nunique()
    a1.set_ylabel("UAR (%)")
    a1.text(0.02, 0.97, f"(a) UAR of the classifiers, {nT} targets", transform=a1.transAxes, fontsize=7, va="top")
    a1.set_ylim(30, 62)

    G = gains()
    gsty = {"e59": dict(color=style.METHOD["ours"]["color"], ls="--", marker="o", mfc="white",
                        label="Same modality"),
            "e60av": dict(color=style.METHOD["ours"]["color"], ls="-.", marker="s", label="Fused, AV world"),
            "e60avt": dict(color=style.METHOD["ours"]["color"], ls="-", marker="X", label="Fused, AVT world"),
            "oursx": dict(color=style.METHOD["oursx"]["color"], ls="-", marker="D", label="Extended, AVT world")}
    for k, v, n in G:
        s = dict(gsty[k])
        s["label"] = f"{s['label']} ({n})"
        a2.plot(xs, v.values, ms=3.2, lw=1.2, mew=0.6 if k == "e59" else 0.4,
                mec=s["color"] if k == "e59" else "white", **s)
    a2.axhline(0, color=style.METHOD["e39"]["color"], lw=1.8, zorder=1)
    a2.set_ylabel("UAR − start (points)")
    a2.text(0.02, 0.97, "(b)", transform=a2.transAxes, fontsize=7, va="top")
    a2.set_ylim(-1.2, 9.0)
    a2.set_yticks([0, 2, 4, 6, 8])
    a2.legend(loc="upper right", ncol=2, handlelength=2.2, columnspacing=0.8, borderaxespad=0.2,
              title="Teacher of the weak labels (number of targets)", title_fontsize=6.5)
    a2.set_xticks(xs)
    a2.set_xticklabels(KS)
    a2.set_xlim(-0.4, len(KS) - 0.6)
    a2.set_xlabel("Labels per emotion, K")
    style.save(fig, "ladder")


if __name__ == "__main__":
    main()
