"""Concept figures of the DAI paper: Fig. 1 (research questions), Fig. 2 (error
decomposition) and Fig. 8 (proposed fusion for pseudo-labelling).

All numbers drawn in Fig. 2 and Fig. 8 are illustrative values, not results.
Fig. 2 computes A(d, o) = 0.5 [Phi(d/2 - o) + Phi(d/2 + o)] for D = 2.0, d = 1.5,
o = 0.5. Fig. 8 computes every derived table (profile, reliability table,
weights, fused scores, pseudo-label) from the illustrative inputs below.

Output: ../figures/<name>.pdf and ../figures/<name>.png (300 dpi), width 174 mm.
Run:  /home/snics/MER-DA/.venv/bin/python make_figs_concept.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
from matplotlib.textpath import TextToPath
from matplotlib.font_manager import FontProperties
from scipy.stats import norm

OUT = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------- fonts
ARIAL_DIR = Path("/mnt/c/Windows/Fonts")
if (ARIAL_DIR / "arial.ttf").exists():
    for f in ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]:
        fm.fontManager.addfont(str(ARIAL_DIR / f))
    FAMILY = "Arial"
else:  # Helvetica-metric fallback
    FAMILY = "Nimbus Sans"
plt.rcParams.update({
    "font.family": FAMILY,
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": FAMILY,
    "mathtext.it": f"{FAMILY}:italic",
    "mathtext.bf": f"{FAMILY}:bold",
    "mathtext.cal": FAMILY,
    "mathtext.sf": FAMILY,
    "mathtext.tt": FAMILY,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "hatch.linewidth": 0.5,
})
FS = 8          # body text in figures (pt)
MM = 1 / 25.4   # inch per mm
W_FIG = 174     # two-column width (mm)

# modality colours and markers (figure notes 4.0)
MOD = {
    "A": dict(color="#E69F00", marker="o", name="Audio"),
    "V": dict(color="#0072B2", marker="s", name="Visual"),
    "T": dict(color="#009E73", marker="^", name="Text"),
}
# research questions: colour + line style (Fig. 1)
QCOL = {"RQ1(a)": "#882255", "RQ1(b)": "#882255", "RQ2": "#8C510A"}
GREY = "#666666"
EMO = ["Neu", "Hap", "Sad", "Ang"]

_t2p = TextToPath()


def text_w_mm(s, size=FS, weight="normal", style="normal"):
    """Rendered width in mm (mathtext between $ signs is measured as math)."""
    prop = FontProperties(family=FAMILY, size=size, weight=weight, style=style)
    w, _, _ = _t2p.get_text_width_height_descent(s, prop, ismath=("$" in s))
    return w * 25.4 / 72


def _tokens(s):
    """Split on plain spaces that are outside $...$."""
    toks, cur, in_math = [], "", False
    for ch in s:
        if ch == "$":
            in_math = not in_math
        if ch == " " and not in_math:
            if cur:
                toks.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        toks.append(cur)
    return toks


def wrap(s, width_mm, size=FS, weight="normal", style="normal"):
    """Greedy word wrap so that each line is at most width_mm wide."""
    lines, cur = [], ""
    for word in _tokens(s):
        trial = word if not cur else cur + " " + word
        if text_w_mm(trial, size, weight, style) <= width_mm or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def canvas(h_mm):
    fig = plt.figure(figsize=(W_FIG * MM, h_mm * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_FIG)
    ax.set_ylim(0, h_mm)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc="white", ec="black", lw=0.7, ls="-", r=1.2, z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color="black", lw=0.9, ls="-", head=7, z=3, style="-|>",
          shrink=0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=head, color=color,
                        lw=lw, linestyle=ls, shrinkA=shrink, shrinkB=shrink,
                        zorder=z, joinstyle="miter", capstyle="butt")
    ax.add_patch(a)
    return a


def double_arrow_v(ax, x, y0, y1, color, head_len=2.4):
    """Vertical arrow drawn as a double line (two parallel strokes) with a head."""
    from matplotlib.patches import Polygon
    sgn = -1 if y1 < y0 else 1
    ye = y1 - sgn * head_len
    ax.plot([x, x], [y0, ye], color=color, lw=3.2, solid_capstyle="butt", zorder=3)
    ax.plot([x, x], [y0 - 0.3 * sgn, ye + 0.3 * sgn], color="white", lw=1.2,
            solid_capstyle="butt", zorder=4)
    ax.add_patch(Polygon([(x - 1.6, ye), (x + 1.6, ye), (x, y1)], closed=True,
                         fc=color, ec=color, lw=0.5, zorder=4))


def marker(ax, x, y, m, size=4.2):
    ax.plot([x], [y], marker=MOD[m]["marker"], ms=size, mfc=MOD[m]["color"],
            mec="black", mew=0.4, zorder=6, clip_on=False)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf", "and .png")




# ======================================================================= Fig. 1
def modality_row(ax, x, y, labels=("audio", "visual", "text")):
    """Row of modality chips: marker + word."""
    cx = x
    for m, lab in zip("AVT", labels):
        marker(ax, cx + 1.2, y, m, size=4.0)
        ax.text(cx + 2.6, y, lab, ha="left", va="center", fontsize=FS, zorder=6)
        cx += 2.6 + text_w_mm(lab) + 2.6


def chip(ax, x, y, label, w=None, h=3.6):
    w = w or text_w_mm(label) + 2.0
    box(ax, x, y - h / 2, w, h, fc="#F2F2F2", ec="#999999", lw=0.5, r=0.8, z=4)
    ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=FS, zorder=6)
    return w


def icon_intercept(ax, x, y, w=12, h=8):
    """Two bumps (emotion k solid, l dashed); boundary (thick) off the midpoint (dotted)."""
    z = np.linspace(-3.4, 2.6, 200)
    sx, sy = w / 6.0, h / 0.5
    X = lambda v: x + (v + 3.4) * sx
    for mu, ls in [(-1.25, "--"), (0.25, "-")]:
        ax.plot(X(z), y + norm.pdf(z, mu) * sy, color="black", lw=0.6, ls=ls, zorder=6)
    ax.plot([x, x + w], [y, y], color="black", lw=0.5, zorder=6)
    ax.plot([X(-0.5)] * 2, [y, y + h * 0.95], color=GREY, lw=0.7, ls=":", zorder=6)
    ax.plot([X(0.0)] * 2, [y, y + h * 0.95], color="black", lw=1.4, zorder=6)


def icon_direction(ax, x, y, w=12, h=8):
    """Two clouds of utterances; the current boundary (solid) is rotated away
    from the best boundary (dashed)."""
    c1, c2 = np.array([x + 3.2, y + 5.6]), np.array([x + 8.8, y + 2.4])
    for c, ls in [(c1, "--"), (c2, "-")]:
        ax.add_patch(Ellipse(c, 4.6, 2.4, angle=-30, fc="none", ec="black", lw=0.6,
                             ls=ls, zorder=6))
    mid = (c1 + c2) / 2
    v = c2 - c1
    perp = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    rot = lambda u, a: np.array([u[0] * np.cos(a) - u[1] * np.sin(a),
                                 u[0] * np.sin(a) + u[1] * np.cos(a)])
    for u, ls, col, lw in [(perp, "--", GREY, 0.8), (rot(perp, 0.55), "-", "black", 1.3)]:
        p0, p1 = mid - 4.2 * u, mid + 4.2 * u
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=lw, ls=ls, zorder=7)


def qtext(ax, x, y_top, q, sec, question, width, extra=None):
    """Question label: coloured bold header and coloured italic question.
    Returns the y of the bottom of the block."""
    col = QCOL[q]
    ax.text(x, y_top, f"{q} (Sec. {sec})", ha="left", va="top", fontsize=FS,
            weight="bold", color=col, zorder=6)
    body = wrap(f"“{question}”", width, style="italic")
    ax.text(x, y_top - 3.6, body, ha="left", va="top", fontsize=FS, style="italic",
            color=col, linespacing=1.15, zorder=6)
    n = body.count("\n") + 1
    y_end = y_top - 3.6 - n * 3.25
    if extra:
        ex = wrap(extra, width)
        ax.text(x, y_end - 0.5, ex, ha="left", va="top", fontsize=FS, color=GREY,
                linespacing=1.15, zorder=6)
        y_end -= 0.5 + (ex.count("\n") + 1) * 3.25
    return y_end


def leader(ax, x, y0, y1, color):
    ax.plot([x, x], [y0, y1], color=color, lw=0.5, zorder=2)
    ax.plot([x], [y1], marker="o", ms=1.8, color=color, zorder=5)


def fig1():
    H = 116
    fig, ax = canvas(H)

    # ---- data (left)
    box(ax, 1, 51, 39, 61, fc="#FAFAFA", ec="black", lw=0.8)
    ax.text(20.5, 109.2, "New target corpus", ha="center", va="center", fontsize=FS,
            weight="bold")
    box(ax, 3, 81, 35, 25, fc="white", ec="#777777", lw=0.6, r=0.8, z=2)
    ax.text(20.5, 103.2, "K labelled utterances\nper emotion", ha="center", va="top",
            fontsize=FS, linespacing=1.1, zorder=6)
    cw = 15.0
    chip(ax, 4.5, 93.6, "Neutral", w=cw); chip(ax, 4.5 + cw + 1.5, 93.6, "Happy", w=cw)
    chip(ax, 4.5, 89.4, "Sad", w=cw);     chip(ax, 4.5 + cw + 1.5, 89.4, "Angry", w=cw)
    modality_row(ax, 4.0, 84.4)
    box(ax, 3, 54, 35, 24, fc="white", ec="#777777", lw=0.6, r=0.8, z=2)
    ax.text(20.5, 75.6, "Unlabelled utterances", ha="center", va="top", fontsize=FS,
            zorder=6)
    for k in range(5):  # a stack of utterance cards
        box(ax, 7.5 + k * 1.3, 66.2 - k * 0.9, 11, 5, fc="white", ec="#999999",
            lw=0.5, r=0.5, z=3 + k * 0.01)
    ax.text(25.0, 63.0, "… many", ha="left", va="center", fontsize=FS, zorder=6)
    modality_row(ax, 4.0, 56.8)
    box(ax, 1, 26, 39, 21, fc="#FAFAFA", ec="black", lw=0.8)
    ax.text(20.5, 43.6, "Source corpora", ha="center", va="center", fontsize=FS,
            weight="bold")
    ax.text(20.5, 37.4, "other corpora with all\nutterances labelled", ha="center",
            va="center", fontsize=FS, linespacing=1.1)
    modality_row(ax, 4.0, 30.4)

    # ---- encoders
    box(ax, 44, 26, 7.5, 86, fc="#E8E8E8", ec="black", lw=0.8)
    ax.text(47.75, 69, "Frozen pretrained encoders", rotation=90, ha="center",
            va="center", fontsize=FS, weight="bold")
    for y0 in (93.5, 66, 36.5):
        arrow(ax, (40, y0), (44, y0), lw=0.8, head=6)

    # ---- three linear classifiers
    cx0, cwid = 55.5, 28.5
    clf_y = {"A": 82, "V": 69, "T": 56}
    for m, y0 in clf_y.items():
        box(ax, cx0, y0, cwid, 10, fc="white", ec=MOD[m]["color"], lw=1.4, r=1.0)
        marker(ax, cx0 + 2.8, y0 + 5, m, size=4.6)
        ax.text(cx0 + 5.1, y0 + 6.4, f"{MOD[m]['name']} classifier", ha="left",
                va="center", fontsize=FS, weight="bold", zorder=6)
        ax.text(cx0 + 5.1, y0 + 2.9, "linear", ha="left", va="center", fontsize=FS,
                zorder=6)
        arrow(ax, (51.5, y0 + 5), (cx0, y0 + 5), lw=0.8, head=6)

    # ---- pseudo-labels (fusion)
    px, py, pw, ph = 93.5, 61, 28.5, 22
    box(ax, px, py, pw, ph, fc="#F3ECE2", ec=QCOL["RQ2"], lw=1.2)
    ax.text(px + pw / 2, py + ph - 3.2, "Pseudo-labels", ha="center", va="center",
            fontsize=FS, weight="bold", zorder=6)
    ax.text(px + pw / 2, py + ph - 6.0,
            "late fusion of the\nclassifiers’ answers\non unlabelled target\nutterances",
            ha="center", va="top", fontsize=FS, linespacing=1.1, zorder=6)
    ends = {"A": py + 16, "V": py + 11, "T": py + 6}
    for m in "AVT":
        arrow(ax, (cx0 + cwid, clf_y[m] + 5), (px, ends[m]), color=QCOL["RQ2"], lw=1.3,
              ls=(0, (5, 1.6, 1, 1.6)), head=7)

    # ---- retrained classifiers
    rx, ry, rw, rh = 128, 61, 44, 22
    box(ax, rx, ry, rw, rh, fc="#F5E9EF", ec=QCOL["RQ1(b)"], lw=1.2)
    ax.text(rx + rw / 2, ry + rh - 3.2, "Retrained classifiers", ha="center",
            va="center", fontsize=FS, weight="bold", zorder=6)
    ax.text(rx + rw / 2, ry + rh - 6.0,
            "audio, visual, text and\nAVT (concatenated),\ntrained on the K labels\nand the pseudo-labels",
            ha="center", va="top", fontsize=FS, linespacing=1.1, zorder=6)
    arrow(ax, (px + pw, py + ph / 2), (rx, ry + rh / 2), color=QCOL["RQ1(b)"], lw=2.0,
          head=9)

    # ---- two parts of the error
    ex, ey, ew, eh = 128, 3, 44, 35
    box(ax, ex, ey, ew, eh, fc="#ECEBF4", ec=QCOL["RQ1(a)"], lw=1.2)
    ax.text(ex + ew / 2, ey + eh - 3.3, "Two parts of the error", ha="center",
            va="center", fontsize=FS, weight="bold", zorder=6)
    box(ax, ex + 1.5, ey + 16.2, ew - 3, 12.4, fc="white", ec="#999999", lw=0.5, r=0.6,
        z=2)
    icon_intercept(ax, ex + 2.8, ey + 18.0, w=11, h=8)
    ax.text(ex + 15.2, ey + 22.4, "Bias of the answers\ntowards some\nemotions (intercept)",
            ha="left", va="center", fontsize=FS, linespacing=1.1, zorder=6)
    box(ax, ex + 1.5, ey + 1.6, ew - 3, 12.4, fc="white", ec="#999999", lw=0.5, r=0.6,
        z=2)
    icon_direction(ax, ex + 2.6, ey + 3.8, w=12, h=8)
    ax.text(ex + 15.2, ey + 7.8, "Remaining error\n(weight direction)", ha="left",
            va="center", fontsize=FS, linespacing=1.1, zorder=6)
    double_arrow_v(ax, 150, ry, ey + eh, QCOL["RQ1(a)"])

    # ---- previous approaches (dashed)
    xr = 98.5
    prev = [(cx0 + 13, 56), (cx0 + 13, 35.5), (xr, 35.5)]
    ax.plot(*zip(*prev), color=GREY, lw=0.9, ls=(0, (4, 2)), zorder=2)
    arrow(ax, (xr, 35.5), (xr, py), color=GREY, lw=0.9, ls=(0, (4, 2)), head=7)
    ax.text(89.0, 33.2,
            wrap("Previous: product of the classifiers’ probabilities / "
                 "pseudo-labels from one modality", 70),
            ha="center", va="top", fontsize=FS, color=GREY, linespacing=1.15)

    # ---- question texts with leaders to their arrows
    yq = qtext(ax, 60.5, 111.5, "RQ2", 7,
               "How should the classifiers’ answers be combined into pseudo-labels "
               "(late fusion)?", 58)
    xa = 89.5  # point on the arrow from the audio classifier
    ya = (clf_y["A"] + 5) + (ends["A"] - clf_y["A"] - 5) * (xa - cx0 - cwid) / (px - cx0 - cwid)
    leader(ax, xa, yq - 0.6, ya, QCOL["RQ2"])

    yq = qtext(ax, 121.5, 111.5, "RQ1(b)", 6,
               "With only a few target labels, for which classifiers and when do pseudo-labels "
               "from classifiers that include other modalities raise UAR?", 50)
    leader(ax, 123.5, yq - 0.6, py + ph / 2, QCOL["RQ1(b)"])

    qtext(ax, 102.5, 59.0, "RQ1(a)", 5,
          "Which part of the error do they correct?", 44,
          extra="compared with the K labels, source corpora and unlabelled "
                "utterances of the same modality")
    save(fig, "fig1_questions")


# ======================================================================= Fig. 2
def A(d, o):
    return 0.5 * (norm.cdf(d / 2 - o) + norm.cdf(d / 2 + o))


XL = (-3.8, 3.1)
YL = (0, 0.70)


def bumps(ax, mu_k, mu_l, ghost=None):
    z = np.linspace(*XL, 400)
    if ghost is not None:
        for mu, ls in zip(ghost, ["-", "--"]):
            ax.plot(z, norm.pdf(z, mu), color="#BBBBBB", lw=0.6, ls=ls, zorder=1)
    pk, pl = norm.pdf(z, mu_k), norm.pdf(z, mu_l)
    ax.fill_between(z, 0, pl, facecolor="none", edgecolor="#777777", hatch="....",
                    lw=0, zorder=2)
    ax.fill_between(z, 0, pk, facecolor="none", edgecolor="#333333", hatch="////",
                    lw=0, zorder=2)
    ax.plot(z, pl, color="#777777", lw=1.0, ls="--", zorder=3)
    ax.plot(z, pk, color="black", lw=1.0, ls="-", zorder=3)


def axis_style(ax):
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels([str(v).replace("-", "−") for v in [-3, -2, -1, 0, 1, 2, 3]])
    ax.tick_params(axis="x", labelsize=FS, length=2.5, pad=1.5)
    ax.set_xlabel(r"Score difference $z_{kl}$ (pooled SD units)", fontsize=FS,
                  labelpad=2)


def hdim(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<|-|>", color="black", lw=0.7,
                                mutation_scale=5, shrinkA=0, shrinkB=0), zorder=6)


def answer_labels(ax, xb):
    kw = dict(va="center", fontsize=FS, zorder=5,
              bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(xb - 0.22, 0.075, "← answer $l$", ha="right", **kw)
    ax.text(xb + 0.22, 0.075, "answer $k$ →", ha="left", **kw)


def fig2():
    D, d, o = 2.0, 1.5, 0.5
    a_best, a_dir, a_now = A(D, 0), A(d, 0), A(d, o)
    loss_dir, loss_int = a_best - a_dir, a_dir - a_now
    print(f"Fig.2: A(D,0)={a_best:.4f} A(d,0)={a_dir:.4f} A(d,o)={a_now:.4f} "
          f"direction loss={loss_dir:.4f} intercept loss={loss_int:.4f}")
    assert f"{a_best:.3f}" == "0.841" and f"{a_dir:.3f}" == "0.773" \
        and f"{a_now:.3f}" == "0.747"

    H = 62
    fig = plt.figure(figsize=(W_FIG * MM, H * MM))
    fx = lambda mm: mm / W_FIG
    fy = lambda mm: mm / H
    axa = fig.add_axes([fx(3), fy(11), fx(49), fy(42)])
    axb = fig.add_axes([fx(60), fy(11), fx(49), fy(42)])
    axc = fig.add_axes([fx(126), fy(11), fx(47), fy(38)])
    top = 0.60  # height of the boundary / midpoint lines (data units)

    # (a) current classifier: midpoint at -o, boundary at 0
    mu_k, mu_l = d / 2 - o, -d / 2 - o
    bumps(axa, mu_k, mu_l)
    axis_style(axa)
    axa.plot([0, 0], [0, top], color="black", lw=1.8, zorder=7)
    axa.plot([-o, -o], [0, top], color="black", lw=0.7, ls=":", zorder=7)
    hdim(axa, mu_l, mu_k, 0.455)
    axa.text(mu_l - 0.12, 0.455, r"$d$ = 1.5", ha="right", va="center", fontsize=FS)
    hdim(axa, -o, 0, 0.535)
    axa.text(0.12, 0.535, r"$o$ = 0.5", ha="left", va="center", fontsize=FS)
    axa.text(0.08, 0.645, "boundary", ha="left", va="center", fontsize=FS)
    axa.text(-o - 0.08, 0.645, "midpoint", ha="right", va="center", fontsize=FS)
    axa.text(mu_l - 1.05, 0.29, "emotion $l$", ha="right", va="center", fontsize=FS,
             color="#555555")
    axa.text(mu_k + 1.05, 0.29, "emotion $k$", ha="left", va="center", fontsize=FS)
    answer_labels(axa, 0)

    # (b) intercept changed by +o: all scores shift by +o, the boundary stays at 0
    bumps(axb, mu_k + o, mu_l + o, ghost=(mu_k, mu_l))
    axis_style(axb)
    axb.plot([0, 0], [0, top], color="black", lw=1.8, zorder=7)
    hdim(axb, mu_l + o, mu_k + o, 0.455)
    axb.text(mu_k + o + 0.12, 0.47, "$d$ = 1.5\n(unchanged)", ha="left", va="center",
             fontsize=FS, linespacing=1.05)
    axb.annotate("", xy=(mu_l + o, 0.535), xytext=(mu_l, 0.535),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7,
                                 mutation_scale=5, shrinkA=0, shrinkB=0))
    axb.text(mu_l - 0.12, 0.535, r"all scores +$o$", ha="right", va="center",
             fontsize=FS)
    axb.text(0.08, 0.645, "boundary = midpoint", ha="left", va="center", fontsize=FS)
    axb.text(0.12, 0.535, r"$o$ = 0", ha="left", va="center", fontsize=FS)
    answer_labels(axb, 0)

    # (c) pairwise accuracy and its two losses
    vals = [a_best, a_dir, a_now]
    xs = [0, 1.3, 2.6]
    hatches = ["....", "////", "xxxx"]
    for x, v, h in zip(xs, vals, hatches):
        axc.bar(x, v - 0.5, bottom=0.5, width=0.72, fc="white", ec="black", lw=0.7,
                hatch=h, zorder=2)
    axc.set_xlim(-0.6, 5.1)
    axc.set_ylim(0.5, 0.9)
    axc.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    axc.set_xticks(xs)
    axc.set_xticklabels([f"{lab}\n{v:.3f}" for lab, v in
                         zip([r"$A(D, 0)$", r"$A(d, 0)$", r"$A(d, o)$"], vals)],
                        fontsize=FS, linespacing=1.3)
    axc.tick_params(axis="both", labelsize=FS, length=2.5, pad=1.5)
    axc.set_ylabel("Pairwise accuracy", fontsize=FS, labelpad=2)
    for s in ("right", "top"):
        axc.spines[s].set_visible(False)
    xb = 3.15
    for x, v in zip(xs, vals):
        axc.plot([x + 0.36, xb], [v, v], color="#999999", lw=0.5, ls=":", zorder=1)
    for t, b in [(a_best, a_dir), (a_dir, a_now)]:
        axc.plot([xb, xb + 0.14, xb + 0.14, xb], [t, t, b, b], color="black", lw=0.7)
    axc.text(xb + 0.32, (a_best + a_dir) / 2, f"direction\nloss {loss_dir:.3f}",
             ha="left", va="center", fontsize=FS, linespacing=1.05)
    axc.text(xb + 0.32, a_now - 0.004, f"intercept\nloss {loss_int:.3f}",
             ha="left", va="top", fontsize=FS, linespacing=1.05)

    fig.text(fx(1.5), fy(58.5), "(a) Current classifier", ha="left", va="center",
             fontsize=FS, weight="bold")
    fig.text(fx(58.5), fy(58.5), "(b) Only the intercept changed", ha="left",
             va="center", fontsize=FS, weight="bold")
    fig.text(fx(116), fy(58.5), "(c) Loss of pairwise accuracy", ha="left",
             va="center", fontsize=FS, weight="bold")
    fig.text(fx(116), fy(53.6), r"$D$ = 2.0, $d$ = 1.5, $o$ = 0.5 (illustrative values)",
             ha="left", va="center", fontsize=FS)
    save(fig, "fig2_decomposition")


# ======================================================================= Fig. 8
def table(ax, x0, y_top, row_keys, col_labels, values, col_w=7.0, row_h=4.0,
          lab_w=7.0, fmt="{:.1f}", rule_before_last=False, row_label_fn=None,
          shade=None, header=True):
    """Draw a small table; returns the y of its bottom rule."""
    ncol = len(col_labels)
    xr = x0 + lab_w + ncol * col_w
    y = y_top
    if header:
        for j, c in enumerate(col_labels):
            ax.text(x0 + lab_w + (j + 0.5) * col_w, y - row_h / 2, c, ha="center",
                    va="center", fontsize=FS, zorder=6)
        y -= row_h
    ax.plot([x0, xr], [y, y], color="black", lw=0.6, zorder=6)
    y_first = y
    for r, key in enumerate(row_keys):
        yc = y_first - row_h * (r + 0.5)
        if row_label_fn is None:
            marker(ax, x0 + 1.6, yc, key, size=3.8)
            ax.text(x0 + 3.2, yc, key, ha="left", va="center", fontsize=FS, zorder=6,
                    weight="bold")
        else:
            row_label_fn(ax, x0, yc, key)
        for j in range(ncol):
            v = values[r][j]
            s = v if isinstance(v, str) else fmt.format(v).replace("-", "−")
            if shade is not None and shade[r][j]:
                ax.add_patch(Rectangle((x0 + lab_w + j * col_w + 0.1, yc - row_h / 2 + 0.35),
                                       col_w - 0.2, row_h - 0.7, fc="#D9D9D9", ec="black",
                                       lw=0.6, zorder=5))
            ax.text(x0 + lab_w + (j + 0.5) * col_w, yc, s, ha="center", va="center",
                    fontsize=FS, zorder=6)
    y_bot = y_first - row_h * len(row_keys)
    ax.plot([x0, xr], [y_bot, y_bot], color="black", lw=0.6, zorder=6)
    if rule_before_last:
        xv = x0 + lab_w + (ncol - 1) * col_w
        ax.plot([xv, xv], [y_bot, y_top], color="black", lw=0.4, zorder=6)
    return y_bot


def panel(ax, x, y, w, h, title):
    box(ax, x, y, w, h, fc="white", ec="#4D4D4D", lw=0.8, r=1.0)
    hb = 5.4
    ax.add_patch(FancyBboxPatch((x, y + h - hb), w, hb,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                fc="#4D4D4D", ec="#4D4D4D", lw=0.8, zorder=2))
    ax.add_patch(Rectangle((x, y + h - hb), w, 1.2, fc="#4D4D4D", ec="none", zorder=2))
    ax.text(x + 2.0, y + h - hb / 2, title, ha="left", va="center", fontsize=FS,
            weight="bold", color="white", zorder=6)
    return y + h - hb


def ptext(ax, x, y_top, s, width, **kw):
    t = wrap(s, width)
    ax.text(x, y_top, t, ha="left", va="top", fontsize=FS, linespacing=1.15,
            zorder=6, **kw)
    return y_top - (t.count("\n") + 1) * 3.25


def fig8():
    # ---- illustrative inputs
    d_src = np.array([[0.9, 1.0, 1.4, 1.5],   # A: d' on the source corpora
                      [0.8, 1.8, 0.9, 0.9],   # V
                      [1.0, 1.4, 1.3, 1.1]])  # T
    L = np.array([0.6, 1.0, 0.5])            # strength from the K target labels
    z = np.array([[-0.2, -0.7, 0.0, 0.9],    # z_A(i|x)
                  [0.1, -0.3, 0.1, 0.1],     # z_V(i|x)
                  [0.8, 0.0, -0.3, -0.5]])   # z_T(i|x)
    c = np.array([0.1, 0.1, 0.0, -0.2])      # intercept shift
    # ---- derived; each step uses the rounded values shown in the figure
    lvl_src = np.round(d_src.mean(1), 1)
    r = np.round(d_src - lvl_src[:, None], 1)
    Dt = np.round(np.maximum(0, L[:, None] + r), 1)
    W = np.round(Dt / Dt.mean(), 2)
    s_prod = np.round(z.sum(0), 2)
    s = np.round((W * z).sum(0), 2)
    q = np.exp(s + c) / np.exp(s + c).sum()
    assert np.allclose(z.sum(1), 0)
    assert int(np.argmax(s_prod)) == 0 and int(np.argmax(s)) == 3
    print("Fig.8: source mean", lvl_src, "\nprofile r\n", r, "\nD\n", Dt,
          "\nmean D", round(Dt.mean(), 4), "\nW\n", W, "\nsum z", s_prod,
          "\ns (unrounded)", (W * z).sum(0), "\nq", np.round(q, 2))

    H = 97
    fig, ax = canvas(H)
    x1, w1 = 1, 61
    # (i) source corpora -> per-emotion profile
    yt = panel(ax, x1, 38.5, w1, 57.5, "(i) Source corpora: per-emotion profile")
    y = ptext(ax, x1 + 2, yt - 1.5,
              "Reliability $d′$ = √2 Φ$^{−1}$(AUC) of each classifier "
              "for each emotion, measured within and across the labelled source corpora",
              w1 - 4)
    tx = x1 + 6
    yb = table(ax, tx, y - 1.0, "AVT", EMO + ["Mean"],
               np.column_stack([d_src, lvl_src]), rule_before_last=True, col_w=8.0)
    ax.text(x1 + 2, yb - 2.7, "Subtract each row’s mean → profile $r_k(i)$",
            ha="left", va="center", fontsize=FS)
    table(ax, tx, yb - 4.6, "AVT", EMO, r, fmt="{:+.1f}", col_w=8.0)

    # (ii) K target labels -> strength
    yt2 = panel(ax, x1, 1, w1, 33.5, "(ii) K target labels: strength")
    y2 = ptext(ax, x1 + 2, yt2 - 1.5,
               "Split the K labelled target utterances per emotion into 5 folds, "
               "retrain on 4 folds and score the held-out fold", 37)
    ptext(ax, x1 + 2, y2 - 1.2,
          "Strength $L_k$ = mean of $d′$ over the 4 emotions (one number per "
          "classifier)", 37)
    table(ax, x1 + 42, yt2 - 3.5, "AVT", [r"$L_k$"], L[:, None], col_w=9.0)

    # (iii) reliability table
    x3, w3 = 66.5, 38
    arrow(ax, (x1 + w1, 67), (x3, 67), lw=1.0, head=7)
    arrow(ax, (x1 + w1, 17), (x3, 17), lw=1.0, head=7)
    yt3 = panel(ax, x3, 1, w3, 95, "(iii) Reliability table")
    ax.text(x3 + w3 / 2, yt3 - 3.4, "$D(k, i)$ = max(0, $L_k$ + $r_k(i)$)",
            ha="center", va="center", fontsize=FS)
    ax.text(x3 + w3 / 2, yt3 - 7.2, "strength (ii) + profile (i)", ha="center",
            va="center", fontsize=FS, color=GREY)
    yb3 = table(ax, x3 + 2.0, yt3 - 10.6, "AVT", EMO, Dt, col_w=7.0)
    ax.text(x3 + w3 / 2, yb3 - 6.0,
            "Weights\n$W(k, i)$ = $D(k, i)$ / mean of $D$", ha="center",
            va="center", fontsize=FS, linespacing=1.25)
    yb3 = table(ax, x3 + 2.0, yb3 - 10.8, "AVT", EMO, W, col_w=7.0, fmt="{:.2f}")
    ptext(ax, x3 + 2.0, yb3 - 3.0,
          "A: audio, V: visual, T: text classifier. Neu: neutral, Hap: happy, "
          "Ang: angry", w3 - 4)
    ax.text(x3 + 2.0, 5.0, "All numbers are illustrative", ha="left", va="center",
            fontsize=FS, style="italic", color=GREY)

    # (iv) fusion of one unlabelled utterance
    x4, w4 = 108.5, 64.5
    arrow(ax, (x3 + w3, 67), (x4, 67), lw=1.0, head=7)
    yt4 = panel(ax, x4, 38.5, w4, 57.5, "(iv) Late fusion for one unlabelled utterance")
    y4 = ptext(ax, x4 + 2, yt4 - 1.5,
               "Scores $z_k(i | x)$: centred log-probabilities of each "
               "classifier for utterance $x$; $s$ = Σ$_k$ $W z_k$", w4 - 4)

    def lab_fn(ax_, x0, yc, key):
        if key in "AVT":
            marker(ax_, x0 + 1.6, yc, key, size=3.8)
            ax_.text(x0 + 3.2, yc, f"$z_{key}(i | x)$", ha="left",
                     va="center", fontsize=FS, zorder=6)
        else:
            ax_.text(x0 + 0.4, yc, key, ha="left", va="center", fontsize=FS, zorder=6)

    lab4, cw4 = 29.5, 8.0
    yb4 = table(ax, x4 + 1.5, y4 - 0.6, "AVT", EMO, z, col_w=cw4, lab_w=lab4,
                fmt="{:+.1f}", row_label_fn=lab_fn)
    rows2 = ["Product: Σ$_k$ $z_k$", "Proposed fusion: $s$",
             "Intercept shift $c$"]
    fmt2 = ["{:+.2f}", "{:+.2f}", "{:+.1f}"]
    vstr = [[f.format(v).replace("-", "−") for v in row]
            for f, row in zip(fmt2, [s_prod, s, c])]
    vstr[2] = ["0.0" if v == "+0.0" else v for v in vstr[2]]
    shade = [[j == int(np.argmax(s_prod)) for j in range(4)],
             [j == int(np.argmax(s)) for j in range(4)], [False] * 4]
    yb4 = table(ax, x4 + 1.5, yb4 - 1.4, rows2, EMO, vstr, col_w=cw4, lab_w=lab4,
                row_label_fn=lab_fn, shade=shade, header=False)
    ptext(ax, x4 + 2, yb4 - 1.4,
          "Grey cell: the answer. $c$ is set without labels: the mean of "
          "softmax($s$ + $c$) over the unlabelled target utterances equals that "
          "of the product.", w4 - 5)

    # (v) pseudo-labels and retraining
    arrow(ax, (x4 + w4 / 2, 38.5), (x4 + w4 / 2, 34.5), lw=1.0, head=7)
    yt5 = panel(ax, x4, 1, w4, 33.5, "(v) Pseudo-labels and retraining")
    y5 = ptext(ax, x4 + 2, yt5 - 1.5,
               "Soft pseudo-label $q(i | x)$ = softmax($s$ + $c$)", w4 - 4)
    y5 = table(ax, x4 + 1.5, y5 - 0.4, ["q"], EMO, np.round(q, 2)[None, :], col_w=cw4,
               lab_w=lab4, fmt="{:.2f}",
               row_label_fn=lambda a_, x0, yc, k: a_.text(
                   x0 + 0.4, yc, "$q(i | x)$", ha="left", va="center",
                   fontsize=FS, zorder=6))
    ptext(ax, x4 + 2, y5 - 1.6,
          "Retrain the AVT (concatenated) classifier (linear) on the K labels and "
          "the pseudo-labels of all unlabelled target utterances", w4 - 4)
    save(fig, "fig8_method")


if __name__ == "__main__":
    fig1()
    fig2()
    fig8()
