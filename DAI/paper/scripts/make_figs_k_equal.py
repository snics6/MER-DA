"""Figures redrawn so that K = 5, 10 and 25 are treated alike (2026-11-16, researcher's instruction).

Fig. 3 (Q1): pseudo-label retraining − K-label baseline for K = 0, 5, 10, 25 (K = 0, 5, 25 from families_flagged.csv family 1;
K = 10 from desc/outside_families.csv family 1, the pre-specified "outside the families" numbers computed at commit ba98974).
Fig. 6 (Q2-c): imitable part P (R0) vs pseudo-label advantage Δ, marker area = room, two panels K = 5 and K = 25
(dai14 conditions.csv; room from dai3 expl4_corpus_restored.csv).
Shared style and helpers are imported from make_figs_results.py (fonts, colours, markers, save, xerr, panel).
Output: figures/fig3_q1_retraining.{pdf,png}, figures/fig6_conditions.{pdf,png} (overwrite the K = 5-centred versions),
figures/data/<name>.csv, figures/data/figs_k_equal.provenance.json.
Usage: .venv/bin/python paper/scripts/make_figs_k_equal.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figs_results as M  # noqa: E402
from make_figs_results import plt, Line2D, STYLE, CORP4, ZERO, EB, MM, W2, read, save, xerr, panel, PROV, FIG  # noqa: E402

ABBR = {"AFEW": "AF", "IEMOCAP": "IE", "MOSEI": "MO", "MELD": "ME"}


def fig3() -> None:
    name = "fig3_q1_retraining"
    f = read(name, "dai1/stats/families_flagged.csv").rename(columns={"T": "corpus"})
    f = f[(f.family == 1) & (f.A == "prop") & (f.B == "none")]
    o = read(name, "dai1/stats/desc/outside_families.csv").rename(columns={"T": "corpus"})
    o = o[(o.family == 1) & (o.K == 10)]
    allk = pd.concat([f[["student", "corpus", "K", "effect", "lo", "hi", "n_units"]], o[["student", "corpus", "K", "effect", "lo", "hi", "n_units"]]])
    STUD, KS = ["A", "V", "T", "AVT"], [0, 5, 10, 25]

    def g(s, c, K):
        x = allk[(allk.student == s) & (allk.corpus == c) & (allk.K == K)]
        assert len(x) == 1, (s, c, K)
        return x.iloc[0]

    gap = 1.0
    nk = len(KS)
    ypos = {(c, K): i * (nk + gap) + j for i, c in enumerate(CORP4) for j, K in enumerate(KS)}
    fig, axes = plt.subplots(1, 4, figsize=(W2, 82 * MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.125, right=0.99, top=0.92, bottom=0.13, wspace=0.12)
    rows = []
    titles = ["(a) Audio", "(b) Visual", "(c) Text", "(d) AVT (concatenated)"]
    for ax, s, t in zip(axes, STUD, titles):
        col, mk, _ = STYLE[s]
        for i in range(1, 4):
            ax.axhline(i * (nk + gap) - (gap + 1) / 2, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, **ZERO)
        for c in CORP4:
            for K in KS:
                x = g(s, c, K)
                filled = K != 0
                ax.errorbar(x.effect, ypos[(c, K)], xerr=xerr(x.effect, x.lo, x.hi), fmt=mk, ms=3.8, color=col,
                            mfc=col if filled else "white", mec=col, mew=0.8, **EB)
                rows.append(dict(panel=t, classifier=s, corpus=c, K=K, effect=x.effect, lo=x.lo, hi=x.hi, n_repetitions=int(x.n_units),
                                 source=("desc/outside_families.csv" if K == 10 else "families_flagged.csv")))
        panel(ax, t)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[(c, K)] for c in CORP4 for K in KS])
    axes[0].set_yticklabels([str(K) for c in CORP4 for K in KS])
    axes[0].set_ylim(3 * (nk + gap) + nk - 0.4, -1.5)
    axes[0].set_xlim(-7.5, 11.5)
    axes[0].set_xticks([-5, 0, 5, 10])
    fig.text(0.5575, 0.02, "UAR difference: pseudo-label retraining − K-label baseline (points)", ha="center", va="bottom")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    ticks = axes[0].get_yticklabels()
    x0 = min(t.get_window_extent(r).x0 for t in ticks)
    x1 = max(t.get_window_extent(r).x1 for t in ticks)
    xf0, xf1 = inv.transform((x0 - 6, 0))[0], inv.transform((x1, 0))[0]
    for i, c in enumerate(CORP4):
        yf = inv.transform(axes[0].transData.transform((0, i * (nk + gap) + (nk - 1) / 2)))[1]
        fig.text(xf0, yf, "IEMOCAP\n(impro.)" if c == "IEMOCAP" else c, ha="right", va="center", linespacing=1.1)
    yf = inv.transform(axes[0].transData.transform((0, -1.05)))[1]
    fig.text(xf1, yf, "K", ha="right", va="center")
    save(fig, name, pd.DataFrame(rows))


def fig6() -> None:
    name = "fig6_conditions"
    cd0 = read(name, "dai14/summary/conditions.csv").rename(columns={"T": "corpus"})
    room0 = read(name, "dai3/eval/expl4_corpus_restored.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 92 * MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.80, bottom=0.17, wspace=0.08)
    SZ = 4.0
    XL, YL = (-4.5, 10.8), (-4.0, 34)
    rows = []
    for ax, K in zip(axes, (5, 25)):
        cd = cd0[cd0.K == K]
        room = room0[(room0.quantity == "room") & (room0.K == K) & (room0.subset == "all")]
        ax.fill_between([0, XL[1]], 0, YL[1], color="0.93", lw=0, zorder=0)
        ax.axvline(0, **ZERO)
        ax.axhline(0, **ZERO)
        for s in ["A", "V", "T"]:
            col, mk, _ = STYLE[s]
            for c in CORP4:
                q = lambda n: cd[(cd.student == s) & (cd.corpus == c) & (cd.quantity == n)].iloc[0]
                P, D = q("imitable_R0"), q("delta")
                rr = room[room.corpus == c]
                assert len(rr) == 1, (c, K)
                R = float(rr.iloc[0]["T_student" if s == "T" else s])
                ax.errorbar(P.effect, D.effect, xerr=xerr(P.effect, P.lo, P.hi), yerr=xerr(D.effect, D.lo, D.hi), fmt="none",
                            ecolor=col, elinewidth=0.7, capsize=0, zorder=2)
                ax.scatter(P.effect, D.effect, s=max(R, 0.5) * SZ, marker=mk, color=col, edgecolors="black", linewidths=0.4, zorder=3)
                ax.annotate(ABBR[c], (P.effect, D.effect), xytext=(5, 3), textcoords="offset points", fontsize=7, zorder=4)
                rows.append(dict(classifier=s, corpus=c, K=K, P=P.effect, P_lo=P.lo, P_hi=P.hi, delta=D.effect, delta_lo=D.lo,
                                 delta_hi=D.hi, room=R))
        ax.set_xlim(*XL)
        ax.set_ylim(*YL)
        ax.set_title(f"K = {K}")
        ax.set_xlabel("Imitable part P (R0), UAR difference:\nimitated pseudo-labels − K-label baseline (points)")
    axes[0].set_ylabel("Pseudo-label advantage Δ, UAR difference:\npseudo-labels − K-label baseline (points)")
    hm = [Line2D([], [], ls="none", marker=STYLE[s][1], ms=5, color=STYLE[s][0], mec="black", mew=0.4, label=lab)
          for s, lab in [("A", "Audio"), ("V", "Visual"), ("T", "Text")]]
    hz = [plt.scatter([], [], s=v * SZ, marker="o", color="white", edgecolors="black", linewidths=0.6, label=f"{v}") for v in [2, 6, 10]]
    l1 = fig.legend(handles=hm, title="Classifier", loc="upper left", bbox_to_anchor=(0.085, 0.995), ncol=3, title_fontsize=8)
    fig.legend(handles=hz, title="Room (marker area): all-label classifier − K-label baseline (points)", loc="upper right",
               bbox_to_anchor=(0.99, 0.995), ncol=3, title_fontsize=8, handletextpad=0.2, columnspacing=0.8)
    fig.add_artist(l1)
    fig.text(0.01, 0.008, "AF: AFEW, IE: IEMOCAP (impro.), MO: MOSEI, ME: MELD", ha="left", va="bottom")
    save(fig, name, pd.DataFrame(rows))


def main() -> None:
    for f in (fig3, fig6):
        print(f.__name__)
        f()
    json.dump(PROV, open(FIG / "data" / "figs_k_equal.provenance.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
