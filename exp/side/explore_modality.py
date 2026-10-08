"""【探索的】音声と映像で，得のばらつきの性質が違う理由を切り分ける（E6）。

E2 では，K=5 の得のばらつきのうち「組に固有の部分」が，音声で 46%，映像で 13% だった。
この違いを 3 段で調べる。得は反復ごとの「合併 − ターゲットのみ」の UAR の差（地図の CSV から作る）。

1. **比べる組をそろえる**：音声は 7 群，映像は RAVDESS の無い 6 群。音声から RAVDESS を除き，
   さらに反復も映像と同じ 0〜4 にそろえて，内訳（ターゲットの効果・ソースの効果・組に固有の部分）を出し直す
2. **揺れを調べる**：反復を偶数回目と奇数回目に分け，それぞれで組に固有の部分を出して相関をとる
   （半分ずつの一致の度合い）。Spearman–Brown の式 2r/(1+r) で，全反復の平均の一致の度合いに直す
3. **音声と映像を同じ組で比べる**：共通の 30 組で，組に固有の部分，ターゲットの効果，ソースの効果，
   距離の指標の組に固有の部分を，それぞれ音声と映像で相関をとる。組に固有の部分の相関は，
   段 2 の一致の度合いで割り戻した値（揺れで薄まる前の相関の目安）も出す

    ./.venv/bin/python exp/side/explore_modality.py

出力：`results/explore_relation/e6_share.csv`・`e6_reliability.csv`・`e6_crossmodal.csv`・`fig_e6.png`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # exp/side/ から見たリポジトリの根
sys.path.insert(0, str(ROOT / "exp"))
from pair_table import OUT, SIX, fit, gains  # noqa: E402
from summarize_map import C_MERGED, C_SRC, C_TGT, GRID, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

K_ORDER = ["1", "5", "10", "25", "50", "all"]
DIST = ["pad", "mmd2", "frechet", "coverage_t_by_s", "density_t_by_s"]


def pair_means(g: pd.DataFrame, groups: list[str], reps: list[int], K: str) -> pd.DataFrame:
    h = g[(g.K == K) & g.target.isin(groups) & g.source.isin(groups) & g.rep.isin(reps)]
    return h.groupby(["target", "source"], as_index=False)["gain"].mean()


def main() -> int:
    G = {k: gains(k) for k in ("audio_wavlm", "video", "fuse-audio-video")}
    common = lambda k: sorted(set.intersection(*[set(G[k][G[k].target == t].rep)  # noqa: E731
                                                  for t in G[k].target.unique()]))
    reps_all = {k: common(k) for k in G}
    seven = sorted(G["audio_wavlm"].target.unique())
    settings = [("audio 7 groups, all reps", "audio_wavlm", seven, reps_all["audio_wavlm"]),
                ("audio 6 groups, all reps", "audio_wavlm", SIX, reps_all["audio_wavlm"]),
                ("audio 6 groups, reps 0-4", "audio_wavlm", SIX, [0, 1, 2, 3, 4]),
                ("video 6 groups, reps 0-4", "video", SIX, [0, 1, 2, 3, 4]),
                ("audio+video 6 groups, reps 0-4", "fuse-audio-video", SIX, [0, 1, 2, 3, 4])]

    # ---- 段 1 と段 2
    share, rel = [], []
    for name, kind, groups, reps in settings:
        for K in K_ORDER:
            f = fit(pair_means(G[kind], groups, reps, K), "gain")
            share.append({"setting": name, "K": K, "n_reps": len(reps), "target": f["target"],
                          "source": f["source"], "pair": f["pair"]})
            ev, od = [r for r in reps if r % 2 == 0], [r for r in reps if r % 2 == 1]
            e1 = fit(pair_means(G[kind], groups, ev, K), "gain")["resid"]
            e2 = fit(pair_means(G[kind], groups, od, K), "gain")["resid"]
            j = pd.concat([e1.rename("e1"), e2.rename("e2")], axis=1).dropna()
            r = float(np.corrcoef(j.e1, j.e2)[0, 1])
            rel.append({"setting": name, "K": K, "n_reps": len(reps), "half_r": r,
                        "reliability": 2 * r / (1 + r) if r > -1 else np.nan,
                        "pair_share": f["pair"],
                        "true_pair_share": f["pair"] * max(2 * r / (1 + r), 0.0)})
    share, rel = pd.DataFrame(share), pd.DataFrame(rel)
    share.to_csv(OUT / "e6_share.csv", index=False)
    rel.to_csv(OUT / "e6_reliability.csv", index=False)

    # ---- 段 3：音声と映像（と音声＋映像）を同じ 30 組・同じ反復 0〜4 で比べる
    cross = []
    reps5 = [0, 1, 2, 3, 4]
    for other in ("video", "fuse-audio-video"):
        for K in K_ORDER:
            fa = fit(pair_means(G["audio_wavlm"], SIX, reps5, K), "gain")
            fo = fit(pair_means(G[other], SIX, reps5, K), "gain")
            ra = rel[(rel.setting == "audio 6 groups, reps 0-4") & (rel.K == K)].reliability.iloc[0]
            ro_name = "video 6 groups, reps 0-4" if other == "video" else "audio+video 6 groups, reps 0-4"
            ro = rel[(rel.setting == ro_name) & (rel.K == K)].reliability.iloc[0]
            j = pd.concat([fa["resid"].rename("a"), fo["resid"].rename("o")], axis=1).dropna()
            r_pair = stats.spearmanr(j.a, j.o)[0]
            pr = float(np.corrcoef(j.a, j.o)[0, 1])
            dis = pr / np.sqrt(ra * ro) if ra > 0 and ro > 0 else np.nan
            ga = pair_means(G["audio_wavlm"], SIX, reps5, K).set_index(["target", "source"]).gain
            go = pair_means(G[other], SIX, reps5, K).set_index(["target", "source"]).gain
            cross.append({"other": other, "K": K, "quantity": "gain (raw pair means)",
                          "rho": stats.spearmanr(ga, go.reindex(ga.index))[0], "n": len(ga)})
            cross.append({"other": other, "K": K, "quantity": "pair-specific gain",
                          "rho": r_pair, "pearson": pr, "rel_audio": ra, "rel_other": ro,
                          "disattenuated": dis, "n": len(j)})
            cross.append({"other": other, "K": K, "quantity": "target effect a_t",
                          "rho": stats.spearmanr(fa["a"], fo["a"].reindex(fa["a"].index))[0], "n": 6})
            cross.append({"other": other, "K": K, "quantity": "source effect b_s",
                          "rho": stats.spearmanr(fa["b"], fo["b"].reindex(fa["b"].index))[0], "n": 6})
        per = pd.read_csv(OUT / "per_pair.csv")
        pa = per[(per.kind == "audio_wavlm") & per.target.isin(SIX) & per.source.isin(SIX)]
        po = per[per.kind == other]
        for met in DIST:
            ea = fit(pa, met)["resid"]
            eo = fit(po, met)["resid"]
            j = pd.concat([ea.rename("a"), eo.rename("o")], axis=1).dropna()
            cross.append({"other": other, "K": "", "quantity": f"pair-specific {met}",
                          "rho": stats.spearmanr(j.a, j.o)[0], "n": len(j)})
    cross = pd.DataFrame(cross)
    cross.to_csv(OUT / "e6_crossmodal.csv", index=False)
    fig(share, rel, cross, OUT / "fig_e6.png")
    print(json.dumps({"reps_all": {k: [min(v), max(v), len(v)] for k, v in reps_all.items()}}))
    print(f"書いた: {OUT}")
    return 0


def fig(share: pd.DataFrame, rel: pd.DataFrame, cross: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    x = np.arange(len(K_ORDER))
    names = {"audio 7 groups, all reps": ("audio, 7 groups, all reps", MUTED, ":"),
             "audio 6 groups, reps 0-4": ("audio, 6 groups, reps 0-4", C_MERGED, "-"),
             "video 6 groups, reps 0-4": ("video, 6 groups, reps 0-4", C_TGT, "-"),
             "audio+video 6 groups, reps 0-4": ("audio+video, 6 groups, reps 0-4", C_SRC, "-")}
    for ax, (df, col, title) in zip(axes[:2], ((share, "pair", "pair-specific share of gain variance"),
                                              (rel, "true_pair_share", "pair-specific share × reliability"))):
        _style(ax)
        for s, (lab, c, ls) in names.items():
            h = df[df.setting == s].set_index("K").reindex(K_ORDER)
            ax.plot(x, h[col], color=c, linestyle=ls, linewidth=2, marker="o", markersize=5, label=lab)
        ax.set_xticks(x, K_ORDER)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("K", fontsize=8, color=INK2)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    ax = axes[2]
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for q, c, lab in (("pair-specific gain", C_MERGED, "pair-specific gain"),
                      ("target effect a_t", C_TGT, "target effect"),
                      ("source effect b_s", C_SRC, "source effect")):
        h = cross[(cross.other == "video") & (cross.quantity == q)].set_index("K").reindex(K_ORDER)
        ax.plot(x, h.rho, color=c, linewidth=2, marker="o", markersize=5, label=lab)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xticks(x, K_ORDER)
    ax.set_ylim(-1, 1)
    ax.set_title("audio vs video on the same 30 pairs (Spearman ρ)", fontsize=9, color=INK, loc="left")
    ax.set_xlabel("K", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("EXPLORATORY E6: why do audio and video differ?", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
