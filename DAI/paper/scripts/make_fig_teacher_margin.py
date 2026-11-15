"""Figure 10: teacher margin (teacher UAR - own K-label baseline UAR on candidates) vs retraining gain (test), K = 5 and K = 25.

Source: results/dai1/stats/teacher_margin.csv (tools/dai1_teacher_margin.py; points with an exclusion mark dropped).
Style: same fonts and Okabe-Ito colour/marker per student as scripts/make_figs_results.py.
Output: figures/fig10_teacher_margin.{pdf,png} and figures/data/fig10_teacher_margin.csv (the plotted points).
Usage: .venv/bin/python paper/scripts/make_fig_teacher_margin.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DAI = Path(__file__).resolve().parents[2]
SRC = DAI / "results" / "dai1" / "stats" / "teacher_margin.csv"
FIG = DAI / "paper" / "figures"
MM = 1 / 25.4
W2 = 174 * MM
WINF = Path("/mnt/c/Windows/Fonts")
if (WINF / "arial.ttf").exists():
    for f in ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]:
        if (WINF / f).exists():
            font_manager.fontManager.addfont(str(WINF / f))
    FONT = "Arial"
else:
    FONT = "Nimbus Sans"
plt.rcParams.update({
    "font.family": FONT, "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "axes.titlepad": 3,
})
STYLE = {"A": ("#E69F00", "o", "Audio"), "V": ("#0072B2", "s", "Visual"), "T": ("#009E73", "^", "Text"), "AVT": ("#000000", "D", "AVT")}


def main() -> None:
    D = pd.read_csv(SRC)
    D = D[D.excluded.isna() | (D.excluded == "")]
    FIG.mkdir(exist_ok=True)
    (FIG / "data").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 0.46 * W2), sharey=True)
    for ax, K in zip(axes, (5, 25)):
        x = D[D.K == K]
        lim = 1.05 * max(x.margin.abs().max(), 1)
        ax.axhline(0, color="0.5", ls="--", lw=0.7, zorder=0)
        ax.axvline(0, color="0.5", ls="--", lw=0.7, zorder=0)
        for s, (c, m, _) in STYLE.items():
            y = x[x.student == s]
            sig = (y.gain_lo > 0) | (y.gain_hi < 0)
            ax.scatter(y.margin[sig], y.gain[sig], s=12, marker=m, color=c, edgecolors=c, linewidths=0.5, alpha=0.9, zorder=3)
            ax.scatter(y.margin[~sig], y.gain[~sig], s=12, marker=m, facecolors="none", edgecolors=c, linewidths=0.5, alpha=0.8, zorder=2)
            q = y[y.teacher == "main_q"]
            ax.scatter(q.margin, q.gain, s=46, marker=m, facecolors="none", edgecolors="#D55E00", linewidths=0.9, zorder=4)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Pseudo-label source UAR − own K-label baseline UAR (candidates, points)")
        ax.set_title(f"K = {K}")
    axes[0].set_ylabel("Retraining − K-label baseline (test, points)")
    handles = [Line2D([], [], marker=m, color=c, ls="", label=lab, markersize=4) for c, m, lab in STYLE.values()]
    handles += [Line2D([], [], marker="o", color="0.3", ls="", markerfacecolor="none", label="95% CI includes 0", markersize=4),
                Line2D([], [], marker="o", color="#D55E00", ls="", markerfacecolor="none", label="Product (main pseudo-label)", markersize=6)]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig10_teacher_margin.{ext}", dpi=300)
    D[["T", "student", "K", "teacher", "margin", "gain", "gain_lo", "gain_hi", "no_own_modality"]].to_csv(FIG / "data" / "fig10_teacher_margin.csv", index=False)
    prov = {"source": str(SRC.relative_to(DAI)), "sha256_16": hashlib.sha256(SRC.read_bytes()).hexdigest()[:16], "n_points": int(len(D)), "font": FONT}
    json.dump(prov, open(FIG / "data" / "fig10_teacher_margin.provenance.json", "w"), indent=1)
    print(prov)


if __name__ == "__main__":
    main()
