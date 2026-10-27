"""Fig. sourcevalue：ソースの価値（ターゲットのラベル何本ぶんか）の予言と実測（E32）．

横軸：予言の n（E13 の感情の対ごとの向きの効率 e と E28-9 の q_dir から 2 つの山の模型で計算．n_pred），
縦軸：測った n ＝ ターゲットのみの学習曲線で，ソースから出発した仮説転移の UAR に当たる本数 − K（感情ごとの本数）．
学習曲線は results/fixc_map/<kind>/*.csv の tgt_only（反復 0〜9 の平均，K=1・5・10・25・50 を K の対数で直線補間，
累積の最大）．台帳 part1._e32 と同じ関数（_curves，_k_equiv）で読み直す．
1 ソース（e32_pairs.csv の uar_P1．同じコーパスの組を除く）と全ソース（e32_allsrc.csv の P4）を印で分け，
K = 5 と 10 を描く（台帳の順位相関 p1-e32-rho-*・p1-e32-allsrc-rho-* と同じ範囲）．
学習曲線の範囲を超えた点（K=50 のターゲットのみより良い）は上の帯，測った n が 0 以下の点は下の帯に置く．

入力：results/explore_relation/snapshots/20261016-0954_e32/e32_pairs.csv，e32_allsrc.csv，results/fixc_map/<kind>/*.csv

    ../.venv/bin/python scripts/figures/fig_sourcevalue.py      # paper/ で
"""
from __future__ import annotations

import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import style  # noqa: E402
from ledger import part1 as P1  # noqa: E402

KINDS = ("a", "ah", "v", "av")
KUSE = ("5", "10")
XLO, XHI = 0.05, 1500
YLO, YHI = 0.1, 70


def measured(kind: str, tab: pd.DataFrame, ucol: str) -> pd.DataFrame:
    cur = P1._curves(kind, list(range(10)))
    res = [P1._k_equiv(cur[t], u) for t, u in zip(tab.target, tab[ucol])]
    tab = tab.copy()
    tab["n2"] = [kp - int(K) if fl == 0 else np.nan for (kp, fl), K in zip(res, tab.K)]
    tab["f2"] = [fl for _, fl in res]
    return tab


def main() -> None:
    style.setup()
    P = pd.read_csv(P1.E32 / "e32_pairs.csv", dtype={"K": str})
    A = pd.read_csv(P1.E32 / "e32_allsrc.csv", dtype={"K": str})
    rows = []
    for k in KINDS:
        kind = style.CONFIG[k]["kind"]
        g = measured(kind, P[(P.kind == kind) & P.K.isin(KUSE)], "uar_P1")
        assert ((g.f2 == g.flag_meas).all() and np.allclose(g.n2.fillna(-999), g.n_meas.fillna(-999)))
        g["src"] = "one"
        a = measured(kind, A[(A.kind == kind) & A.K.isin(KUSE)], "P4")
        assert ((a.f2 == a.flag).all() and np.allclose(a.n2.fillna(-999), a.n_meas.fillna(-999)))
        a["src"] = "all"
        for t in (g, a):
            t["k"] = k
            rows.append(t[["k", "src", "target", "K", "n_pred", "n2", "f2"]])
    D = pd.concat(rows, ignore_index=True)
    assert (D.n_pred > XLO).all() and (D.n_pred < XHI).all()

    fig = plt.figure(figsize=(style.COL, 2.45))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.13, 1.0, 0.13], hspace=0.08,
                          left=0.14, right=0.735, bottom=0.15, top=0.98)
    top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=top)
    bot = fig.add_subplot(gs[2], sharex=top)
    rng = np.random.default_rng(0)

    def draw(axx, d, y, k, src):
        c = style.CONFIG[k]
        if src == "one":
            axx.scatter(d.n_pred, y, s=8, marker=c["marker"], color=c["color"], alpha=0.7, lw=0.25,
                        edgecolor="white", zorder=3)
        else:
            axx.scatter(d.n_pred, y, s=22, marker=c["marker"], color=c["color"], lw=0.7,
                        edgecolor=style.INK, zorder=4)

    lo, hi = min(XLO, YLO), max(XHI, YHI)
    ax.plot([lo, hi], [lo, hi], color=style.INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for k in KINDS:
        for src in ("one", "all"):
            d = D[(D.k == k) & (D.src == src)]
            ok = (d.f2 == 0) & (d.n2 > 0)
            draw(ax, d[ok], d[ok].n2, k, src)
            hi_ = d[d.f2 == 1]
            draw(top, hi_, rng.uniform(0.25, 0.75, len(hi_)), k, src)
            le0 = d[(d.f2 == 0) & (d.n2 <= 0)]
            draw(bot, le0, rng.uniform(0.25, 0.75, len(le0)), k, src)
            assert (d.f2 != -1).all()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    ticks = [0.1, 1, 10, 100, 1000]
    tl = ["0.1", "1", "10", "100", "1000"]
    for axx in (top, bot):
        axx.set_ylim(0, 1)
        axx.set_yticks([])
        axx.grid(axis="y", visible=False)
        axx.tick_params(axis="x", which="both", length=0)
    top.text(-0.02, 0.5, r"$>50-K$", transform=top.transAxes, ha="right", va="center", fontsize=6.5,
             color=style.INK2)
    bot.text(-0.02, 0.5, r"$\leq 0$", transform=bot.transAxes, ha="right", va="center", fontsize=6.5,
             color=style.INK2)
    plt.setp(top.get_xticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(axis="x", which="both", length=0)
    bot.set_xticks(ticks, tl)
    bot.tick_params(axis="x", which="both", length=2.5)
    bot.minorticks_off()
    ax.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.minorticks_off()
    top.spines["bottom"].set_visible(False)
    bot.spines["top"].set_visible(True)
    bot.spines["top"].set_linestyle((0, (2, 2)))
    bot.spines["top"].set_linewidth(0.5)
    top.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for axx in (top, ax, bot):
        axx.grid(axis="x", visible=True)
    bot.set_xlabel(r"Predicted $n = q\,e^2/(1-e^2)$ (labels per emotion)")
    ax.set_ylabel("Measured $n$ (labels per emotion)")

    handles = [Line2D([], [], ls="none", marker=style.CONFIG[k]["marker"], color=style.CONFIG[k]["color"],
                      ms=4, label=style.CONFIG[k]["label"]) for k in KINDS]
    handles += [Line2D([], [], ls="none", marker="o", color=style.GRAY, ms=3.2, alpha=0.8, mec="white", mew=0.25,
                       label="One source"),
                Line2D([], [], ls="none", marker="o", color=style.GRAY, ms=5, mec=style.INK, mew=0.7,
                       label="All sources")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.12), borderaxespad=0.0,
              handletextpad=0.3, labelspacing=0.5)
    print(D.groupby(["k", "src"]).apply(lambda d: pd.Series({
        "n": len(d), "main": int(((d.f2 == 0) & (d.n2 > 0)).sum()), "beyond": int((d.f2 == 1).sum()),
        "le0": int(((d.f2 == 0) & (d.n2 <= 0)).sum())}), include_groups=False))
    style.save(fig, "sourcevalue")


if __name__ == "__main__":
    main()
