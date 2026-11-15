#!/usr/bin/env python
"""Result figures 3, 4, 5, 6, 7 and 9 of the DAI paper (Japanese draft, second edition).

No training and no new computation of results: every plotted value is read from
DAI/results/.  Before a figure is drawn, its values are compared with the tables (and a few
sentences) of the frozen first-edition draft DAI/paper/ja/第1版/ (read only).  Any mismatch
stops the script with exit code 1 and nothing more is drawn.

Outputs (DAI/paper/figures/):
  <name>.pdf, <name>.png (300 dpi)       the figure, at its final width (174 mm; Fig. 6 84 mm)
  data/<name>.csv                         the plotted values
  make_figs_results_provenance.json       source files with sha256[:16], first-edition files
                                          checked, number of values checked, commit, font
Run:
  /home/snics/MER-DA/.venv/bin/python /home/snics/MER-DA/DAI/paper/scripts/make_figs_results.py

Figure rules (paper/notes/図_岡田石井と有名なMER論文_2026-11-15.md, section 4.0): Okabe-Ito
colour and marker shape per modality, order A, A(HuBERT), V, T, AV, VT, AT, AVT, one corpus
order (AFEW, IEMOCAP, MOSEI, MELD, CREMA-D), 95% intervals as error bars, grey dashed zero
line, no title inside the figure, panel labels "(a) ..." at the top left, Arial 8-9 pt.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

DAI = Path(__file__).resolve().parents[2]
RES = DAI / "results"
PAPER = DAI / "paper"
FIG = PAPER / "figures"
JA1 = PAPER / "ja" / "第1版"
MM = 1 / 25.4
W2, W1 = 174 * MM, 84 * MM  # Springer double / single column

# ---------------------------------------------------------------- fonts
WINF = Path("/mnt/c/Windows/Fonts")
if (WINF / "arial.ttf").exists() and (WINF / "arialbd.ttf").exists():
    for f in ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]:
        if (WINF / f).exists():
            font_manager.fontManager.addfont(str(WINF / f))
    FONT = "Arial"
else:  # fallback reported in the provenance file
    FONT = "Nimbus Sans"
plt.rcParams.update({
    "font.family": FONT, "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold", "mathtext.cal": FONT, "mathtext.sf": FONT,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType, keep text as text
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 0.9,
    "hatch.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handletextpad": 0.4, "legend.columnspacing": 1.0,
    "axes.titlepad": 3,
})

# ---------------------------------------------------------------- shared style
STYLE = {  # colour, marker, filled  (Okabe-Ito; one pair per feature set in every figure)
    "A": ("#E69F00", "o", True), "A(HuBERT)": ("#E69F00", "o", False),
    "V": ("#0072B2", "s", True), "T": ("#009E73", "^", True),
    "AV": ("#D55E00", "v", True), "VT": ("#56B4E9", "p", True),
    "AT": ("#CC79A7", "h", True), "AVT": ("#000000", "D", True),
}
KINDS = ["A", "A(HuBERT)", "V", "T", "AV", "VT", "AT", "AVT"]
KLABEL = {"A": "A (WavLM)", "A(HuBERT)": "A (HuBERT)"}
CORP4 = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
CORP5 = CORP4 + ["CREMA-D"]
ZERO = dict(color="0.5", ls="--", lw=0.7, zorder=0)
EB = dict(elinewidth=0.8, capsize=0)

PROV: dict = {"figures": {}}


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def rec(fig: str) -> dict:
    return PROV["figures"].setdefault(fig, {"sources": [], "first_edition": [], "n_checked": 0})


def read(fig: str, rel: str) -> pd.DataFrame:
    """Read a results csv (leading '#' provenance lines skipped) and record its sha256[:16]."""
    p = RES / rel
    h = sha16(p)
    rec(fig)["sources"].append({"path": f"results/{rel}", "sha256_16": h})
    print(f"  source results/{rel}  sha256[:16]={h}")
    lines = p.read_text(encoding="utf-8").splitlines()
    skip = 0
    while skip < len(lines) and lines[skip].startswith("#"):
        skip += 1
    return pd.read_csv(p, skiprows=skip)


def md(fig: str, name: str) -> str:
    p = JA1 / name
    h = sha16(p)
    r = rec(fig)["first_edition"]
    if not any(x["path"].endswith(name) for x in r):
        r.append({"path": f"paper/ja/第1版/{name}", "sha256_16": h})
        print(f"  first edition paper/ja/第1版/{name}  sha256[:16]={h}")
    return p.read_text(encoding="utf-8")


def md_table(text: str, marker: str):
    """Header and body rows of the first Markdown table after the line starting with marker."""
    lines = text.splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(marker))
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("|"))
    rows = []
    while j < len(lines) and lines[j].startswith("|"):
        rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
        j += 1
    return rows[0], rows[2:]


NUMRE = re.compile(r"[+\-−]?\d+(?:\.\d+)?")


def nums(cell: str) -> list[str]:
    return NUMRE.findall(cell.replace(",", " "))


def tofloat(s: str) -> float:
    return float(s.replace("−", "-"))


class Check:
    """Compare figure values with numbers printed in the first edition (to their decimals)."""

    def __init__(self, fig: str):
        self.fig, self.bad, self.n = fig, [], 0

    @staticmethod
    def agree(value: float, s: str) -> bool:
        t = s.replace("−", "-").lstrip("+")
        nd = len(t.split(".")[1]) if "." in t else 0
        return bool(np.isfinite(value)) and abs(float(value) - float(t)) <= 0.5 * 10 ** -nd + 1e-9

    def eq(self, value: float, s: str, what: str) -> None:
        self.n += 1
        if not self.agree(value, s):
            self.bad.append(f"{what}: figure {value:.6g} vs first edition {s}")

    def one_of(self, value: float, ss: list[str], what: str) -> None:
        self.n += 1
        if not any(self.agree(value, s) for s in ss):
            self.bad.append(f"{what}: figure {value:.6g} vs first edition one of {ss}")

    def true(self, cond: bool, what: str) -> None:
        self.n += 1
        if not cond:
            self.bad.append(what)

    def done(self) -> None:
        rec(self.fig)["n_checked"] = self.n
        if self.bad:
            print(f"STOP: {self.fig} does not match the first edition:")
            for b in self.bad:
                print("   ", b)
            sys.exit(1)
        print(f"  {self.n} values agree with the first edition")


def panel(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9)


def save(fig, name: str, data: pd.DataFrame) -> None:
    (FIG / "data").mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "DAI/paper/scripts/make_figs_results.py", "CreationDate": None}
    fig.savefig(FIG / f"{name}.pdf", metadata=meta)
    fig.savefig(FIG / f"{name}.png", dpi=300)
    data.to_csv(FIG / "data" / f"{name}.csv", index=False)
    w, h = fig.get_size_inches() / MM
    rec(name)["size_mm"] = [round(float(w), 1), round(float(h), 1)]
    print(f"  wrote figures/{name}.pdf, .png, data/{name}.csv  ({w:.0f} x {h:.0f} mm)")
    plt.close(fig)


def xerr(e, lo, hi):
    return [[e - lo], [hi - e]]


# =============================================================== Fig. 3
def fig3() -> None:
    name = "fig3_q1_retraining"
    print(name)
    f = read(name, "dai1/stats/families_flagged.csv").rename(columns={"T": "corpus"})
    f = f[(f.family == 1) & (f.A == "prop") & (f.B == "none")]
    STUD, KS = ["A", "V", "T", "AVT"], [0, 5, 25]

    def g(s, c, K):
        x = f[(f.student == s) & (f.corpus == c) & (f.K == K)]
        assert len(x) == 1, (s, c, K)
        return x.iloc[0]

    ck = Check(name)
    text = md(name, "07_本番の評価.md")
    hdr, body = md_table(text, "**表 7.1**")
    ci = hdr.index("形 (a) − K 本だけ")
    lab = {"声 A": "A", "顔 V": "V", "言葉 T": "T", "融合 AVT": "AVT"}
    ck.true(len(body) == 16, "table 7.1 has 16 rows")
    for r in body:
        x = g(lab[r[0]], r[1], 5)
        e, lo, hi = nums(r[ci])
        for v, s, w in [(x.effect, e, "effect"), (x.lo, lo, "lo"), (x.hi, hi, "hi")]:
            ck.eq(v, s, f"table 7.1 {r[0]} {r[1]} K=5 {w}")
    # sentences of section 7.3 (K = 0, 5, 25)

    def rng(s, c, a, b, col="effect"):
        y = f[(f.student == s) & (f.corpus == c)][col]
        ck.eq(y.min(), a, f"7.3 {s} {c} min {col}")
        ck.eq(y.max(), b, f"7.3 {s} {c} max {col}")

    A = f[f.student == "A"]
    ck.true(bool((A.effect > 0).all()), "7.3: audio differences all positive")
    ck.true(set(map(tuple, A[A.lo <= 0][["corpus", "K"]].values.tolist())) == {("IEMOCAP", 5), ("IEMOCAP", 25)},
            "7.3: audio lower ends <= 0 only IEMOCAP K=5, 25")
    for c, a, b in [("AFEW", "+3.3", "+4.9"), ("IEMOCAP", "+1.4", "+4.8"), ("MELD", "+2.9", "+3.9"), ("MOSEI", "+2.1", "+3.4")]:
        rng("A", c, a, b)
    rng("T", "IEMOCAP", "+5.0", "+7.5")
    rng("T", "IEMOCAP", "2.4", "4.7", "lo")
    for c, a, b in [("AFEW", "+0.7", "+1.6"), ("MELD", "−1.2", "+0.2"), ("MOSEI", "−0.7", "+0.0")]:
        rng("T", c, a, b)
    rng("V", "IEMOCAP", "−2.1", "−0.7")
    v25 = g("V", "MOSEI", 25)
    for v, s in [(v25.effect, "+2.57"), (v25.lo, "+1.17"), (v25.hi, "+3.98")]:
        ck.eq(v, s, "7.3 V MOSEI K=25")
    rng("AVT", "MELD", "+1.4", "+3.4")
    rng("AVT", "MELD", "0.6", "1.7", "lo")
    o = f[(f.student == "AVT") & (f.corpus != "MELD")]
    ck.eq(o.effect.min(), "−0.8", "7.3 AVT other corpora min")
    ck.eq(o.effect.max(), "+1.7", "7.3 AVT other corpora max")
    ck.true(int((o.lo > 0).sum()) == 1, "7.3 AVT other corpora: 8 of 9 include 0")
    for s, n in [("A", 10), ("V", 1), ("T", 3), ("AVT", 4)]:  # section 7 lead
        ck.true(int((f[f.student == s].lo > 0).sum()) == n, f"7 lead: {s} lower end > 0 in {n} of 12")
    ck.true(bool((f[f.K == 0].n_units == 5).all() and (f[f.K > 0].n_units == 50).all()), "7.2: 50 repetitions (K=0: 5)")
    ck.done()

    # K is written on the y-axis (one row per K), so the marker shape keeps the meaning it has in every other
    # figure: the classifier's modality (circle A, square V, triangle T, diamond AVT).
    gap = 1.0
    ypos = {(c, K): i * (3 + gap) + j for i, c in enumerate(CORP4) for j, K in enumerate(KS)}
    fig, axes = plt.subplots(1, 4, figsize=(W2, 70 * MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.125, right=0.99, top=0.91, bottom=0.15, wspace=0.12)
    rows = []
    titles = ["(a) Audio", "(b) Visual", "(c) Text", "(d) AVT (concatenated)"]
    for ax, s, t in zip(axes, STUD, titles):
        col, mk, _ = STYLE[s]
        for i in range(1, 4):
            ax.axhline(i * (3 + gap) - (gap + 1) / 2, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, **ZERO)
        for c in CORP4:
            for K in KS:
                x = g(s, c, K)
                ax.errorbar(x.effect, ypos[(c, K)], xerr=xerr(x.effect, x.lo, x.hi), fmt=mk, ms=3.8,
                            color=col, mfc=col, mec=col, mew=0.8, **EB)
                rows.append(dict(panel=t, classifier=s, corpus=c, K=K, effect=x.effect, lo=x.lo, hi=x.hi,
                                 n_repetitions=int(x.n_units)))
        panel(ax, t)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[(c, K)] for c in CORP4 for K in KS])
    axes[0].set_yticklabels([str(K) for c in CORP4 for K in KS])
    axes[0].set_ylim(3 * (3 + gap) + 2.6, -1.5)
    axes[0].set_xlim(-7.5, 11.5)
    axes[0].set_xticks([-5, 0, 5, 10])
    fig.text(0.5575, 0.02, "UAR difference: pseudo-label retraining − K-label baseline (points)", ha="center",
             va="bottom")
    # corpus names to the left of the K column, and the K header above it
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    ticks = axes[0].get_yticklabels()
    x0 = min(t.get_window_extent(r).x0 for t in ticks)
    x1 = max(t.get_window_extent(r).x1 for t in ticks)
    xf0, xf1 = inv.transform((x0 - 6, 0))[0], inv.transform((x1, 0))[0]
    for i, c in enumerate(CORP4):
        yf = inv.transform(axes[0].transData.transform((0, i * (3 + gap) + 1)))[1]
        fig.text(xf0, yf, "IEMOCAP\n(impro.)" if c == "IEMOCAP" else c, ha="right", va="center", linespacing=1.1)
    yf = inv.transform(axes[0].transData.transform((0, -1.05)))[1]
    fig.text(xf1, yf, "K", ha="right", va="center")
    save(fig, name, pd.DataFrame(rows))


# =============================================================== Fig. 4
def fig4() -> None:
    name = "fig4_mixing_gain"
    print(name)
    d = read(name, "e76_analysis/e76a_1_kind.csv")
    d["K"] = d["K"].astype(str)
    KL = ["1", "5", "10", "25", "50", "all"]

    def g(s):
        x = d[d.short == s].set_index("K").loc[KL]
        assert len(x) == 6, s
        return x

    ck = Check(name)
    hdr, body = md_table(md(name, "03_混ぜるgain.md"), "**表 1**")
    lab = {"A（WavLM）": "A", "A（HuBERT）": "A(HuBERT)"}
    ck.true([lab.get(r[0], r[0]) for r in body] == KINDS, "table 1 rows are the eight feature sets")
    ck.true(hdr[2:] == ["K=1", "5", "10", "25", "50", "全件"], f"table 1 columns {hdr}")
    for r in body:
        s = lab.get(r[0], r[0])
        x = g(s)
        nt, npair = r[1].split("/")
        ck.true(int(x.loc["1", "n_targets"]) == int(nt) and int(x.loc["1", "n_pairs"]) == int(npair), f"table 1 {s} counts")
        for K, cell in zip(KL, r[2:]):
            e, lo, hi = nums(cell)
            for v, sv, w in [(x.loc[K, "mean_gain"], e, "mean"), (x.loc[K, "lo"], lo, "lo"), (x.loc[K, "hi"], hi, "hi")]:
                ck.eq(v, sv, f"table 1 {s} K={K} {w}")
    ck.done()

    XP = np.array([0, 1, 2, 3, 4, 5.4])
    off = np.linspace(-0.15, 0.15, 4)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 66 * MM), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.17, wspace=0.08)
    rows = []
    groups = [["A", "A(HuBERT)", "V", "T"], ["AV", "VT", "AT", "AVT"]]
    for ax, grp, t in zip(axes, groups, ["(a) Single-modality features", "(b) Concatenated features"]):
        ax.axhline(0, **ZERO)
        hs = []
        for o, s in zip(off, grp):
            x = g(s)
            col, m, filled = STYLE[s]
            ax.plot(XP[:5] + o, x.mean_gain.values[:5], color=col, lw=0.8, zorder=1)
            ax.errorbar(XP + o, x.mean_gain, yerr=[x.mean_gain - x.lo, x.hi - x.mean_gain], fmt=m, ms=3.8,
                        color=col, mfc=col if filled else "white", mec=col, mew=0.8, zorder=2, **EB)
            hs.append(Line2D([], [], color=col, marker=m, ms=3.8, mfc=col if filled else "white", mew=0.8, lw=0.8,
                             label=KLABEL.get(s, s)))
            for K in KL:
                rows.append(dict(panel=t, feature=s, K=K, mean_gain=x.loc[K, "mean_gain"], lo=x.loc[K, "lo"],
                                 hi=x.loc[K, "hi"], n_targets=int(x.loc[K, "n_targets"]), n_pairs=int(x.loc[K, "n_pairs"])))
        ax.set_xticks(XP)
        ax.set_xticklabels(["1", "5", "10", "25", "50", "All"])
        ax.set_xlim(-0.5, 5.9)
        ax.set_xlabel("Labelled target utterances per emotion, K")
        ax.legend(handles=hs, loc="upper right", ncol=2)
        panel(ax, t)
    axes[0].set_ylim(-7, 16.5)
    axes[0].set_ylabel("UAR difference:\nmerged − target-only (points)")
    save(fig, name, pd.DataFrame(rows))


# =============================================================== Fig. 5
E75_GROUP = {"A": "audio_wavlm", "A(HuBERT)": "audio_hubert", "V": "video", "T": "text_roberta",
             "AV": "fuse-audio-video", "VT": "fuse-video-text", "AT": "fuse-audio-text", "AVT": "fuse-audio-video-text"}


def fig5() -> None:
    name = "fig5_bias_and_direction"
    print(name)
    k4 = read(name, "e76_analysis/e76a_4_kind.csv")
    k4 = k4[k4.K.astype(str) == "5"]
    e75 = read(name, "e75/e75_5_summary.csv")
    e75 = e75[e75.K.astype(str) == "5"]
    dT = read(name, "e76_analysis/e76a_3_dT.csv")
    qs = read(name, "e76_analysis/e76a_3_q_summary.csv")
    QN = ["G_merge", "G_center", "G_crit_c"]

    def g4(s, q):
        x = k4[(k4.short == s) & (k4.quantity == q)]
        assert len(x) == 1, (s, q)
        return x.iloc[0]

    def g75(s):
        x = e75[e75.group == E75_GROUP[s]]
        assert len(x) == 1, s
        return x.iloc[0]

    dT = dT[dT["rule"] == "main"].assign(r=lambda x: x.d_T / x.D)
    pts = dT.groupby(["short", "target", "K"]).r.mean().reset_index().groupby(["short", "K"]).r.median()
    cells = dT.drop_duplicates(["short", "target", "pair"])
    dpd = (cells.Dprime / cells.D).groupby(cells.short).median()  # D'/D, median over (target, pair)

    def qmain(s):
        x = qs[(qs.short == s) & (qs["rule"] == "main") & (qs.D_version == "D") & (qs.fit == "3pt")]
        assert len(x) == 1, s
        return float(x.iloc[0].median_q_incl_edge)

    ck = Check(name)
    hdr, body = md_table(md(name, "04_切片.md"), "**表 4（題）**")
    ck.true(sorted(r[0].replace("A(WavLM)", "A") for r in body) == sorted(KINDS), "table 4 rows")
    for r in body:
        s = r[0].replace("A(WavLM)", "A")
        ck.true(int(g4(s, "G_merge").n_pairs) == int(r[1]), f"table 4 {s} pairs")
        for q, col in [("G_merge", 2), ("G_center", 5), ("G_crit_c", 6)]:
            e, lo, hi = nums(r[col])
            x = g4(s, q)
            for v, sv, w in [(x["mean"], e, "mean"), (x.lo, lo, "lo"), (x.hi, hi, "hi")]:
                ck.eq(v, sv, f"table 4 {s} {q} {w}")
        ratio, med = nums(r[3])
        ck.eq(g75(s).uar_ratio_of_sums_Gcrit_Gmerge, ratio, f"table 4 {s} ratio")
        ck.eq(g75(s).uar_median_Gcrit_over_Gmerge, med, f"table 4 {s} median ratio")
    text5 = md(name, "05_向き.md")
    hdr, body = md_table(text5, "**表 5（題）**")
    b5 = {r[0]: r for r in body}
    for s, lab in [("A", "A(WavLM)"), ("V", "V"), ("T", "T")]:
        r = b5[lab]
        for K, col in [("1", 2), ("5", 3), ("10", 4)]:
            ck.eq(pts[(s, K)], nums(r[col])[0], f"table 5 {s} d_T/D K={K}")
        ck.one_of(qmain(s), nums(r[6]), f"table 5 {s} q(D) (main rule is one end of the range)")
    m = re.search(r"全件の分類器の d_T/D の中央値は A\(WavLM\) ([\d.]+)，T ([\d.]+)", text5)
    ck.eq(dpd["A"], m.group(1), "5.2 A all-label ratio")
    ck.eq(dpd["T"], m.group(2), "5.2 T all-label ratio")
    m = re.search(r"D′/D の中央値は ([\d.]+)", text5)
    ck.eq(dpd["V"], m.group(1), "5.2 V D'/D median")
    ck.done()

    fig = plt.figure(figsize=(W2, 88 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.68, 1], left=0.06, right=0.985, top=0.75, bottom=0.13, wspace=0.18)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    rows = []
    # (a) grouped bars
    w = 0.26
    BAR = {"G_merge": dict(facecolor="0.6", edgecolor="0.15"),
           "G_center": dict(facecolor="white", edgecolor="0.15", hatch="//////", hatchcolor="0.15"),
           "G_crit_c": dict(facecolor="white", edgecolor="0.15")}
    ax.axhline(0, **ZERO)
    for i, s in enumerate(KINDS):
        for j, q in enumerate(QN):
            x = g4(s, q)
            xp = i + (j - 1) * w
            ax.bar(xp, x["mean"], width=w, lw=0.6, zorder=2, **BAR[q])
            ax.errorbar(xp, x["mean"], yerr=xerr(x["mean"], x.lo, x.hi), fmt="none", ecolor="black", zorder=3, **EB)
            rows.append(dict(panel="a", feature=s, quantity=q, mean=x["mean"], lo=x.lo, hi=x.hi, n_pairs=int(x.n_pairs)))
    # The ratio (intercept-only fit − source-only) / (merged − source-only) is checked above against the first
    # edition but is not drawn: the intercept-only fit has no bar here, so the ratio could not be read off the bars
    # (review round3_ja/presentation_mer.md, m1).  It is given in the text and the appendix table.
    ax.set_xticks(range(8))
    ax.set_xticklabels(["A\nWavLM", "A\nHuBERT", "V", "T", "AV", "VT", "AT", "AVT"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(-1.5, 6.3)
    ax.set_ylabel("UAR difference (points)")
    hd = [Patch(lw=0.6, label="Merged − source-only", **BAR["G_merge"]),
          Patch(lw=0.6, label="Mean subtraction (no labels) − source-only", **BAR["G_center"]),
          Patch(lw=0.6, label="Intercept fit after mean subtraction − mean subtraction", **BAR["G_crit_c"])]
    ax.legend(handles=hd, loc="lower left", bbox_to_anchor=(-0.01, 1.03), ncol=1, handlelength=1.6,
              borderaxespad=0, labelspacing=0.3)
    # (b) d_T(K)/D
    Kg = np.geomspace(1, 50, 300)
    hs = []
    for s in ["A", "V", "T"]:
        col, mk, _ = STYLE[s]
        q = qmain(s)
        y = 1 / np.sqrt(1 + q / Kg)
        bx.plot(Kg[Kg <= 10], y[Kg <= 10], color=col, lw=1.0, zorder=2)
        bx.plot(Kg[Kg >= 10], y[Kg >= 10], color=col, lw=1.0, ls=(0, (4, 2)), zorder=2)
        bx.axhline(dpd[s], color=col, ls=":", lw=1.0, zorder=1)
        kf, kn = ["1", "5", "10"], ["25", "50"]
        bx.plot([int(k) for k in kf], [pts[(s, k)] for k in kf], ls="none", marker=mk, ms=4, color=col, mec=col, zorder=3)
        bx.plot([int(k) for k in kn], [pts[(s, k)] for k in kn], ls="none", marker=mk, ms=4, color=col, mfc="white",
                mec=col, mew=0.8, zorder=3)
        hs.append(Line2D([], [], color=col, marker=mk, ms=4, lw=1.0, label=KLABEL.get(s, s)))
        for k in kf + kn:
            rows.append(dict(panel="b", feature=s, quantity="d_T/D median over targets", K=int(k), mean=pts[(s, k)],
                             used_in_fit=k in kf))
        rows.append(dict(panel="b", feature=s, quantity="q (main C rule, K=1,5,10 fit, median over targets)", mean=q))
        rows.append(dict(panel="b", feature=s, quantity="D'/D median over (target, pair)", mean=float(dpd[s])))
    hs += [Line2D([], [], color="0.3", lw=1.0, label="Fit to K = 1, 5, 10"),
           Line2D([], [], color="0.3", lw=1.0, ls=(0, (4, 2)), label="Extension"),
           Line2D([], [], color="0.3", lw=1.0, ls=":", label="D′/D (all labels)")]
    bx.legend(handles=hs, loc="lower left", bbox_to_anchor=(-0.01, 1.03), ncol=2, handlelength=2.0, borderaxespad=0,
              labelspacing=0.3)
    bx.set_xscale("log")
    bx.set_xticks([1, 5, 10, 25, 50])
    bx.set_xticklabels(["1", "5", "10", "25", "50"])
    bx.minorticks_off()
    bx.set_xlim(0.85, 60)
    bx.set_ylim(0, 1.4)
    bx.set_xlabel("Labelled target utterances per emotion, K")
    bx.set_ylabel(r"$d_T(K)\,/\,D$")
    # panel labels sit above the legends, at the left edge of each panel's y-axis label
    fig.text(0.008, 0.99, "(a) Single-source setting, K = 5", ha="left", va="top", fontsize=9)
    fig.text(bx.get_position().x0 - 0.055, 0.99, "(b) Weight direction, target-only classifier", ha="left",
             va="top", fontsize=9)
    save(fig, name, pd.DataFrame(rows))


# =============================================================== Fig. 6
ABBR = {"AFEW": "AF", "IEMOCAP": "IE", "MOSEI": "MO", "MELD": "ME"}


def fig6() -> None:
    name = "fig6_conditions"
    print(name)
    cd = read(name, "dai14/summary/conditions.csv").rename(columns={"T": "corpus"})
    cd = cd[cd.K == 5]
    room = read(name, "dai3/eval/expl4_corpus_restored.csv")
    room = room[(room.quantity == "room") & (room.K == 5) & (room.subset == "all")]

    def gq(s, c, q):
        x = cd[(cd.student == s) & (cd.corpus == c) & (cd.quantity == q)]
        assert len(x) == 1, (s, c, q)
        return x.iloc[0]

    def groom(s, c):
        x = room[room.corpus == c]
        assert len(x) == 1, c
        return float(x.iloc[0]["T_student" if s == "T" else s])  # column T_student holds the text classifier

    ck = Check(name)
    hdr, body = md_table(md(name, "08_3条件.md"), "**表 8.1**")
    lab = {"声": "A", "顔": "V", "言葉": "T"}
    n = 0
    for r in body:
        if r[0] not in lab:
            continue
        s, c = lab[r[0]], r[1]
        n += 1
        ck.eq(groom(s, c), nums(r[2])[0], f"table 8.1 {s} {c} room")
        ck.eq(gq(s, c, "uar_judge").effect, nums(r[3])[0], f"table 8.1 {s} {c} K-only UAR")
        ck.eq(gq(s, c, "uar_pseudo").effect, nums(r[4])[0], f"table 8.1 {s} {c} q UAR")
        for q, col in [("delta", 5), ("imitable_R0", 6)]:
            e, lo, hi = nums(r[col])
            x = gq(s, c, q)
            for v, sv, w in [(x.effect, e, "effect"), (x.lo, lo, "lo"), (x.hi, hi, "hi")]:
                ck.eq(v, sv, f"table 8.1 {s} {c} {q} {w}")
    ck.true(n == 12, "table 8.1 has 12 rows for A, V, T")
    ck.done()

    # label offsets in points; True = draw a leader line (crowded points near P = 0, delta = 5-9)
    OFF = {("A", "AFEW"): (7, 3, False), ("A", "IEMOCAP"): (-30, -8, True), ("A", "MOSEI"): (8, -6, False),
           ("A", "MELD"): (7, 4, False), ("V", "AFEW"): (6, -6, False), ("V", "IEMOCAP"): (-8, 4, False),
           ("V", "MOSEI"): (-12, -9, True), ("V", "MELD"): (6, 2, False), ("T", "AFEW"): (-7, 3, False),
           ("T", "IEMOCAP"): (8, 3, False), ("T", "MOSEI"): (-20, 11, True), ("T", "MELD"): (-7, -6, False)}
    SZ = 5.0  # marker area (pt^2) per UAR point of room
    H6 = 104
    fig, ax = plt.subplots(figsize=(W1, H6 * MM))
    fig.subplots_adjust(left=0.165, right=0.98, top=1 - 23.5 / H6, bottom=18 / H6)
    XL, YL = (-4.5, 10.8), (-4.0, 34)
    ax.fill_between([0, XL[1]], 0, YL[1], color="0.93", lw=0, zorder=0)
    ax.axvline(0, **ZERO)
    ax.axhline(0, **ZERO)
    rows = []
    for s in ["A", "V", "T"]:
        col, mk, _ = STYLE[s]
        for c in CORP4:
            P, D, R = gq(s, c, "imitable_R0"), gq(s, c, "delta"), groom(s, c)
            ax.errorbar(P.effect, D.effect, xerr=xerr(P.effect, P.lo, P.hi), yerr=xerr(D.effect, D.lo, D.hi),
                        fmt="none", ecolor=col, elinewidth=0.7, capsize=0, zorder=2)
            ax.scatter(P.effect, D.effect, s=R * SZ, marker=mk, color=col, edgecolors="black", linewidths=0.4, zorder=3)
            dx, dy, lead = OFF[(s, c)]
            ax.annotate(ABBR[c], (P.effect, D.effect), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                        va="center", ha="left" if dx > 0 else "right", zorder=4,
                        arrowprops=dict(arrowstyle="-", color="0.3", lw=0.5, shrinkA=1, shrinkB=4) if lead else None)
            rows.append(dict(classifier=s, corpus=c, K=5, P=P.effect, P_lo=P.lo, P_hi=P.hi, delta=D.effect,
                             delta_lo=D.lo, delta_hi=D.hi, room=R))
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    # axis names give both sides of each difference (直しの決め事 2 節; review m2: "reproducibility" read as
    # replicability, so P is called the imitable part)
    ax.set_xlabel("Imitable part P, UAR difference:\nimitated pseudo-labels − K-label baseline (points)")
    ax.set_ylabel("Pseudo-label advantage Δ, UAR difference:\npseudo-labels − K-label baseline (points)")
    hm = [Line2D([], [], ls="none", marker=STYLE[s][1], ms=5, color=STYLE[s][0], mec="black", mew=0.4, label=lab)
          for s, lab in [("A", "Audio"), ("V", "Visual"), ("T", "Text")]]
    hz = [plt.scatter([], [], s=v * SZ, marker="o", color="white", edgecolors="black", linewidths=0.6, label=f"{v}")
          for v in [2, 6, 10]]
    x_left = -0.165 / (0.98 - 0.165)  # left edge of the figure in axes coordinates
    l1 = ax.legend(handles=hm, title="Classifier", loc="lower left", bbox_to_anchor=(x_left + 0.01, 1.245), ncol=3,
                   borderaxespad=0, title_fontsize=8, alignment="left")
    ax.add_artist(l1)
    ax.legend(handles=hz, title="Room (marker area), UAR difference:\nall-label classifier − K-label baseline (points)",
              loc="lower left", bbox_to_anchor=(x_left + 0.01, 1.015), ncol=3, borderaxespad=0, title_fontsize=8,
              alignment="left", handletextpad=0.2, columnspacing=0.8, frameon=False)
    fig.text(0.01, 0.008, "AF: AFEW, IE: IEMOCAP (impro.), MO: MOSEI, ME: MELD", ha="left", va="bottom")
    save(fig, name, pd.DataFrame(rows))


# =============================================================== Fig. 7
EMO = ["Neutral", "Happy", "Sad", "Angry"]


def fig7() -> None:
    name = "fig7_strength_profile"
    print(name)
    rel = read(name, "dai16_probe/summary/reliability.csv")
    r0 = read(name, "dai16_probe/summary/r0.csv")
    orc = rel[(rel.K == 5) & (rel.source == "M_orc")]
    rows_i = [(m, c) for m in ["A", "V", "T"] for c in CORP4]
    Dm = np.array([orc[(orc.c == c) & (orc.modality == m)][["d_0", "d_1", "d_2", "d_3"]].values[0] for m, c in rows_i])
    L = Dm.mean(1)
    R = Dm - L[:, None]

    ck = Check(name)
    text = md(name, "10_手法と評価.md")
    sec = text[text.index("### 10.1.1"):text.index("### 10.1.2")]
    r5 = r0[(r0.K == 5) & (r0.fam == "M")]
    ext, kk = r5[r5.source == "ext"], r5[r5.source == "K"]
    m = re.search(r"相関は ([−+\d.]+)〜([−+\d.]+)，最強のモダリティが一致した割合は ([\d.]+)〜([\d.]+) だった（K 本で測った表では ([\d.]+)〜([\d.]+)）", sec)
    ck.eq(ext.levelcorr.min(), m.group(1), "10.1.1 level correlation min")
    ck.eq(ext.levelcorr.max(), m.group(2), "10.1.1 level correlation max")
    ck.eq(ext.topok.min(), m.group(3), "10.1.1 strongest-modality agreement min (source)")
    ck.eq(ext.topok.max(), m.group(4), "10.1.1 strongest-modality agreement max (source)")
    ck.eq(kk.topok.min(), m.group(5), "10.1.1 strongest-modality agreement min (K)")
    ck.eq(kk.topok.max(), m.group(6), "10.1.1 strongest-modality agreement max (K)")
    m = re.search(r"相関が ([\d.]+)〜([\d.]+) で，感情の並びを入れ替えて元の相関以上になる割合は ([\d.]+)〜([\d.]+)", sec)
    ck.eq(ext.intcorr.min(), m.group(1), "10.1.1 component correlation min")
    ck.eq(ext.intcorr.max(), m.group(2), "10.1.1 component correlation max")
    ck.eq(ext.intperm.min(), m.group(3), "10.1.1 permutation share min")
    ck.eq(ext.intperm.max(), m.group(4), "10.1.1 permutation share max")
    ck.true(len(orc) == 12 and bool((orc.n_units == 30).all()), "full-label table: 12 rows, 30 repetitions each")
    ck.true(bool((Dm >= 0).all() and (Dm <= 2.9).all()), "reliabilities within [0, 2.90] (eq. 10.2)")
    ck.done()

    fig = plt.figure(figsize=(W2, 88 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 0.04], left=0.215, right=0.985,
                          top=0.85, bottom=0.085, wspace=0.07, hspace=0.12)
    ax, bx, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            v = R[i, j]
            ax.text(j, i, f"{v:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.62 else "black")
    for yb in [3.5, 7.5]:
        ax.axhline(yb, color="white", lw=2.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(EMO)
    ax.xaxis.tick_top()
    ax.set_yticks(range(12))
    ax.set_yticklabels(["IEMOCAP (impro.)" if c == "IEMOCAP" else c for _, c in rows_i])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    names = {"A": "Audio", "V": "Visual", "T": "Text"}
    xmod = 0.022  # modality labels at a fixed distance from the left edge of the figure
    tmod = blended_transform_factory(fig.transFigure, ax.transData)
    for k, mod in enumerate(["A", "V", "T"]):
        col, mk, _ = STYLE[mod]
        ax.text(xmod, 4 * k + 1.5, names[mod], transform=tmod, rotation=90, ha="center", va="center", fontsize=8)
        ax.plot([xmod], [4 * k - 0.1], marker=mk, color=col, ms=5, transform=tmod, clip_on=False)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(length=2)
    cb.outline.set_linewidth(0.5)
    # words as in Fig. 8 and the English abstract: strength (水準) and per-emotion profile (相性)
    cax.set_xlabel("Per-emotion profile r = d − L (reliability d minus the classifier's strength L)", labelpad=1)
    # (b) strength
    for yb in [3.5, 7.5]:
        bx.axhline(yb, color="0.8", lw=0.6)
    rows = []
    for i, (mod, c) in enumerate(rows_i):
        col, mk, _ = STYLE[mod]
        bx.plot([0, L[i]], [i, i], color="0.85", lw=0.6, zorder=1)
        bx.plot(L[i], i, marker=mk, ms=4.5, color=col, mec=col, zorder=2)
        rows.append(dict(modality=mod, corpus=c, level_L=L[i], **{f"affinity_{e}": R[i, j] for j, e in enumerate(EMO)},
                         **{f"d_{e}": Dm[i, j] for j, e in enumerate(EMO)}))
    bx.set_ylim(11.5, -0.5)
    bx.set_yticks(range(12))
    bx.set_yticklabels([])
    bx.tick_params(axis="y", length=0)
    bx.set_xlim(0, 2.1)
    bx.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    bx.set_xlabel("Strength L (mean d over the four emotions)", labelpad=2)
    top = ax.get_position().y1
    fig.text(0.008, 0.99, "(a) Per-emotion profile", ha="left", va="top", fontsize=9)
    fig.text(bx.get_position().x0 - 0.01, 0.99, "(b) Strength", ha="left", va="top", fontsize=9)
    assert top < 0.9
    save(fig, name, pd.DataFrame(rows))


# =============================================================== Fig. 9
ROWS9 = [("prod", "Product", "A", "積（7 節の手法の推定時の答え）"),
         ("tcal", "Temperature calibration", "A", "分類器ごとの温度の較正"),
         ("wf1000", "Weight-table search", "A", "K 本で重みの表を探す（乱数の表 1,000 個"),
         ("none", "K-label baseline", "B", "K 本だけの融合"),
         ("stud_prod", "Pseudo-label retraining", "B", "7 節の方法（積を疑似ラベルにした学び直し）")]


def fig9() -> None:
    name = "fig9_fusion_vs_others"
    print(name)
    d20 = read(name, "dai20_eval/summary/effects.csv")
    d21 = read(name, "dai21_eval5/summary/effects.csv")
    KS = [5, 10, 25]

    def g(c, b, grp, K):
        if grp == "A":
            src = d21[d21.part == "cremad"] if c == "CREMA-D" else d20
            a = "prop"
        else:
            src = d21[d21.part == ("cremad" if c == "CREMA-D" else "4corpora")]
            a = "stud_prop"
            if b == "none":
                b = "none_AV" if c == "CREMA-D" else "none_AVT"
        x = src[(src.a == a) & (src.b == b) & (src.c == c) & (src.K == K) & (src.seeds == "all")]
        assert len(x) == 1, (c, b, K)
        return x.iloc[0]

    ck = Check(name)
    text = md(name, "10_手法と評価.md")
    hdr, body = md_table(text, "表 10.2")
    ck.true([hdr[1 + 3 * i] for i in range(5)] == [f"{c} K=5" for c in CORP5], f"table 10.2 columns {hdr}")
    for b, _, grp, lab in ROWS9:
        r = [x for x in body if x[0].startswith(lab)]
        ck.true(len(r) == 1, f"table 10.2 row {lab}")
        for ci, c in enumerate(CORP5):
            for ki, K in enumerate(KS):
                cell = nums(r[0][1 + 3 * ci + ki])
                x = g(c, b, grp, K)
                for v, sv, w in [(x.est, cell[1], "effect"), (x.lo, cell[2], "lo"), (x.hi, cell[3], "hi")]:
                    ck.eq(v, sv, f"table 10.2 {b} {c} K={K} {w}")
                ck.true(int(x.n_fr) == 50, f"{b} {c} K={K}: 50 repetitions")
    # sentences of 10.2.3 that the caption repeats
    for b, grp, a, z in [("prod", "A", "−0.09", "+0.83"), ("stud_prod", "B", "−0.05", "+0.82")]:
        v = [g(c, b, grp, K).est for c in CORP5[1:] for K in KS]
        ck.eq(min(v), a, f"10.2.3 {b} other corpora min")
        ck.eq(max(v), z, f"10.2.3 {b} other corpora max")
    ck.done()

    # K is written on the y-axis (one row per K) instead of being coded by marker shape, because circles,
    # squares and triangles mean audio, visual and text in the other figures (review round3_ja m3).  The marker
    # is a cross, a shape no feature set uses.
    YB = {"prod": 0, "tcal": 3.9, "wf1000": 7.8, "none": 12.0, "stud_prod": 15.9}  # first row of each block
    SEP = 10.9  # line between the two groups of rows
    KJ = {5: 0, 10: 1, 25: 2}
    H9 = 78
    fig = plt.figure(figsize=(W2, H9 * MM))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.13, 1], left=0.185, right=0.99, top=1 - 10 / H9,
                          bottom=10 / H9, wspace=0.1)
    axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 2, 3, 5]]
    rows = []
    titles = {"AFEW": "AFEW", "IEMOCAP": "IEMOCAP\n(impro.)", "MOSEI": "MOSEI", "MELD": "MELD",
              "CREMA-D": "CREMA-D\n(audio, visual)"}
    for k, (ax, c) in enumerate(zip(axes, CORP5)):
        ax.axvline(0, **ZERO)
        ax.axhline(SEP, color="0.6", lw=0.6)
        for b, lab, grp, _ in ROWS9:
            for K in KS:
                x = g(c, b, grp, K)
                ax.errorbar(x.est, YB[b] + KJ[K], xerr=xerr(x.est, x.lo, x.hi), fmt="X", ms=3.8, color="black",
                            mfc="black", mec="black", mew=0.3, **EB)
                rows.append(dict(corpus=c, comparison=lab, group=grp, a=x.a, b=x.b, K=K, effect=x.est, lo=x.lo, hi=x.hi,
                                 n_repetitions=int(x.n_fr)))
        ax.set_xlim(-6.8, 6.8)
        ax.set_xticks([-5, 0, 5])
        ax.set_ylim(YB["stud_prod"] + 2.6, -1.5)
        ax.set_yticks([YB[b] + KJ[K] for b, _, _, _ in ROWS9 for K in KS])
        ax.set_yticklabels([str(K) for _ in ROWS9 for K in KS] if k == 0 else [])
        ax.tick_params(axis="y", length=0)
        # titles hang from a common top line so that the one- and two-line titles start at the same height
        ax.annotate(f"({'abcde'[k]}) {titles[c]}", xy=(0, 1), xycoords="axes fraction", xytext=(0, 25),
                    textcoords="offset points", ha="left", va="top", fontsize=9, linespacing=1.1)
    fig.text(0.5875, 0.012, "UAR difference: proposed fusion − method in the row (points)", ha="center", va="bottom")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax0 = axes[0]
    inv = fig.transFigure.inverted()
    ticks = ax0.get_yticklabels()
    x0 = min(t.get_window_extent(r).x0 for t in ticks)
    x1 = max(t.get_window_extent(r).x1 for t in ticks)
    xf0, xf1 = inv.transform((x0 - 5, 0))[0], inv.transform((x1, 0))[0]
    yf = inv.transform(ax0.transData.transform((0, -1.05)))[1]
    fig.text(xf1, yf, "K", ha="right", va="center")
    NAME9 = {"prod": "Product", "tcal": "Temperature\ncalibration", "wf1000": "Weight-table\nsearch",
             "none": "K-label\nbaseline", "stud_prod": "Pseudo-label\nretraining"}
    for b, _, _, _ in ROWS9:
        yf = inv.transform(ax0.transData.transform((0, YB[b] + 1)))[1]
        fig.text(xf0, yf, NAME9[b], ha="right", va="center", linespacing=1.1)
    fig.canvas.draw()
    xg = min(t.get_window_extent(r).x0 for t in fig.texts if "\n" in t.get_text() or t.get_text() == "Product")
    xg = inv.transform((xg - 4, 0))[0]
    for y0, y1, t in [(YB["prod"], YB["wf1000"] + 2, "Late-fusion\nanswers"),
                      (YB["none"], YB["stud_prod"] + 2, "Retrained AVT\n(concatenated)")]:
        yf = inv.transform(ax0.transData.transform((0, (y0 + y1) / 2)))[1]
        fig.text(xg, yf, t, rotation=90, ha="right", va="center", fontsize=8, linespacing=1.1)
    save(fig, name, pd.DataFrame(rows))


# =============================================================== main
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"font: {FONT}")
    for f in [fig3, fig4, fig5, fig6, fig7, fig9]:
        f()
    head = subprocess.run(["git", "-C", str(DAI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(DAI), "status", "--porcelain", "--", "results"],
                           capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(["git", "-C", str(DAI), "ls-files", "--error-unmatch", "paper/scripts/make_figs_results.py"],
                             capture_output=True, text=True).returncode == 0
    PROV.update({"script": "paper/scripts/make_figs_results.py", "script_sha256_16": sha16(Path(__file__)),
                 "commit": head, "results_dirty": bool(dirty), "script_tracked_in_git": tracked, "font": FONT,
                 "matplotlib": matplotlib.__version__, "pandas": pd.__version__, "numpy": np.__version__})
    (FIG / "make_figs_results_provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1) + "\n",
                                                           encoding="utf-8")
    print("wrote figures/make_figs_results_provenance.json")


if __name__ == "__main__":
    main()
