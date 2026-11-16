"""Fig. C (2026-11-16): which corpora gain from the proposed fusion, and why.

(a) One panel per corpus: test UAR of the three classifiers that are fused (audio, visual, text; the
    second-stage classifiers of Sec. 5) as bars, and the fused answers (product, temperature scaling,
    proposed fusion) as markers, for K = 5, 10, 25.
(b) Fused − product (UAR points) against product − best single classifier (UAR points), 15 conditions
    (5 corpora x 3 K). Proposed fusion with its 95% interval (effects.csv), temperature scaling as
    the difference of two levels (no interval).
Inputs: results/dai20_eval/summary/levels.csv, effects.csv (seeds = all); results/dai21_eval5/summary/levels.csv,
effects.csv (CREMA-D, part = cremad). Values are the ones in docs/結果/計算の後の解析_提案が効くコーパスの型_2026-11-16.md.
Output: ../figures/fig_corpus_types{,_ja}.pdf/.png and ../figures/data/fig_corpus_types.csv.
Run:  /home/snics/MER-DA/.venv/bin/python make_fig_corpus_types.py
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "figures"
R20 = ROOT / "results" / "dai20_eval" / "summary"
R21 = ROOT / "results" / "dai21_eval5" / "summary"
MM = 1 / 25.4
FS = 7.2

ARIAL_DIR = Path("/mnt/c/Windows/Fonts")
if (ARIAL_DIR / "arial.ttf").exists():
    for f in ["arial.ttf", "arialbd.ttf", "ariali.ttf"]:
        fm.fontManager.addfont(str(ARIAL_DIR / f))
    EN_FONT = "Arial"
else:
    EN_FONT = "Nimbus Sans"
fm.fontManager.addfont("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")
JA_FONT = "IPAexGothic"

COL = {"A": "#E69F00", "V": "#0072B2", "T": "#009E73"}
C_PROP, C_TCAL = "#D55E00", "#444444"
CORPORA = ["AFEW", "MELD", "IEMOCAP", "MOSEI", "CREMA-D"]
CCOL = {"AFEW": "#000000", "MELD": "#CC79A7", "IEMOCAP": "#56B4E9", "MOSEI": "#7B3294", "CREMA-D": "#A6A600"}
ABBR = {"AFEW": "AF", "MELD": "ME", "IEMOCAP": "IE", "MOSEI": "MO", "CREMA-D": "CR"}
KS = [5, 10, 25]

L = {
    "en": dict(mods={"A": "Audio", "V": "Visual", "T": "Text"}, prod="Product", tcal="Temperature scaling",
               prop="Proposed fusion", ylab_a="Test UAR (%)", kname="K", iemocap="IEMOCAP (impro.)",
               xlab_b="Product − best single classifier (UAR points)", ylab_b="Fused − product (UAR points)",
               left="Product below\nthe best single\nclassifier", right="Product above the best single classifier",
               title_a="(a) UAR of the fused classifiers (bars) and of the fused answers (markers)",
               title_b="(b) Gain of the fused answer over the product", iemo_mosei="IE: IEMOCAP (impro.), MO: MOSEI; number = K", ci="", prop_ci="Proposed fusion (with 95% interval)", tcal_ci="Temperature scaling (with 95% interval)"),
    "ja": dict(mods={"A": "声", "V": "顔", "T": "言葉"}, prod="積", tcal="温度の較正", prop="提案",
               ylab_a="テストの UAR（%）", kname="K", iemocap="IEMOCAP（即興）",
               xlab_b="積 − 最も強い単独の分類器（UAR の点）", ylab_b="まとめた答え − 積（UAR の点）",
               left="積が最も強い\n単独の分類器に\n届かない", right="積が最も強い単独の分類器を上回る",
               title_a="(a) まとめる分類器の UAR（棒）と，まとめた答えの UAR（印）",
               title_b="(b) まとめた答えの積からの差", iemo_mosei="IE：IEMOCAP（即興），MO：MOSEI．数字は K", ci="", prop_ci="提案（95% 区間つき）", tcal_ci="温度の較正（95% 区間つき）"),
}


def load():
    l20 = pd.read_csv(R20 / "levels.csv", comment="#")
    l20 = l20[l20.seeds.astype(str) == "all"]
    l21 = pd.read_csv(R21 / "levels.csv", comment="#")
    l21 = l21[(l21.seeds.astype(str) == "all") & (l21.c == "CREMA-D")]
    lev = pd.concat([l20, l21])
    lev = lev[lev.version.isin(["ret_A", "ret_V", "ret_T", "prod", "prop", "tcal"])]
    lv = lev.pivot_table(index=["c", "K"], columns="version", values="uar").reset_index()
    e20 = pd.read_csv(R20 / "effects.csv", comment="#")
    e20 = e20[e20.seeds.astype(str) == "all"]
    e21 = pd.read_csv(R21 / "effects.csv", comment="#")
    e21 = e21[(e21.seeds.astype(str) == "all") & (e21.part == "cremad")]
    d = lv
    for a, name in (("prop", "prop_minus_prod"), ("tcal", "tcal_minus_prod_eff")):
        x = pd.concat([e20[(e20.a == a) & (e20.b == "prod")], e21[(e21.a == a) & (e21.b == "prod")]])[["c", "K", "est", "lo", "hi"]]
        d = d.merge(x.rename(columns={"est": name, "lo": name + "_lo", "hi": name + "_hi"}), on=["c", "K"])
    d["best"] = d[["ret_A", "ret_V", "ret_T"]].max(axis=1)
    d["prod_minus_best"] = d["prod"] - d["best"]
    d["tcal_minus_prod"] = d["tcal"] - d["prod"]
    if (d["tcal_minus_prod"] - d["tcal_minus_prod_eff"]).abs().max() > 1e-6:
        raise SystemExit("tcal - prod from levels.csv and effects.csv differ")
    if len(d) != 15:
        raise SystemExit(f"rows {len(d)} (expected 15)")
    return d


def draw(d, lang):
    t = L[lang]
    plt.rcParams.update({"font.family": EN_FONT if lang == "en" else JA_FONT, "font.size": FS,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig = plt.figure(figsize=(174 * MM, 132 * MM))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.05], hspace=0.62, wspace=0.42,
                          left=0.065, right=0.99, top=0.88, bottom=0.085)
    # (a)
    for ci, c in enumerate(CORPORA):
        ax = fig.add_subplot(gs[0, ci])
        x = d[d.c == c].sort_values("K")
        mods = [m for m in "AVT" if pd.notna(x[f"ret_{m}"]).all()]
        bw = 0.8 / 3
        for gi, (_, r) in enumerate(x.iterrows()):
            for mi, m in enumerate(mods):
                ax.bar(gi + (mi - (len(mods) - 1) / 2) * bw, r[f"ret_{m}"], width=bw * 0.95, color=COL[m], ec="black", lw=0.3)
            ax.plot([gi - 0.42, gi + 0.42], [r["prod"]] * 2, color="black", lw=1.1, zorder=4)
            ax.plot([gi + 0.30], [r["tcal"]], marker="D", ms=3.6, mfc="white", mec=C_TCAL, mew=0.8, zorder=5)
            ax.plot([gi + 0.30], [r["prop"]], marker="*", ms=6.5, mfc=C_PROP, mec="black", mew=0.3, zorder=6)
        lo = min(x[[f"ret_{m}" for m in mods]].min().min(), x["prod"].min()) - 6
        hi = max(x[["prod", "prop", "tcal"] + [f"ret_{m}" for m in mods]].max()) + 2
        ax.set_ylim(max(0, lo), hi)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"K={k}" for k in KS], fontsize=FS - 0.6)
        ax.set_title(t["iemocap"] if c == "IEMOCAP" else c, fontsize=FS + 0.2, pad=3)
        ax.tick_params(length=2, labelsize=FS - 0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ci == 0:
            ax.set_ylabel(t["ylab_a"])
    fig.text(0.065, 0.985, t["title_a"], ha="left", va="center", fontsize=FS + 0.6, weight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[m], ec="black", lw=0.3, label=t["mods"][m]) for m in "AVT"]
    handles += [Line2D([], [], color="black", lw=1.1, label=t["prod"]),
                Line2D([], [], ls="none", marker="D", ms=3.6, mfc="white", mec=C_TCAL, mew=0.8, label=t["tcal"]),
                Line2D([], [], ls="none", marker="*", ms=6.5, mfc=C_PROP, mec="black", mew=0.3, label=t["prop"])]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.955), ncol=6, fontsize=FS - 0.4,
               frameon=False, handlelength=1.4, columnspacing=1.0)
    # (b)
    ax = fig.add_subplot(gs[1, 0:3])
    ax.axhline(0, color="#999999", lw=0.6, zorder=1)
    ax.axvline(0, color="#999999", lw=0.6, ls="--", zorder=1)
    xmin, xmax = -2.2, 7.0
    ax.axvspan(xmin, 0, color="#F2F2F2", zorder=0)
    for _, r in d.iterrows():
        xx = r.prod_minus_best
        cc = CCOL[r.c]
        ax.errorbar(xx, r.prop_minus_prod, yerr=[[r.prop_minus_prod - r.prop_minus_prod_lo], [r.prop_minus_prod_hi - r.prop_minus_prod]],
                    fmt="none", ecolor=cc, elinewidth=0.7, capsize=1.2, zorder=3)
        ax.plot(xx, r.prop_minus_prod, marker="*", ms=7.0, mfc=cc, mec="black", mew=0.3, ls="none", zorder=5)
        ax.errorbar(xx + 0.07, r.tcal_minus_prod, yerr=[[r.tcal_minus_prod - r.tcal_minus_prod_eff_lo], [r.tcal_minus_prod_eff_hi - r.tcal_minus_prod]],
                    fmt="none", ecolor=cc, elinewidth=0.5, capsize=1.0, zorder=2, alpha=0.6)
        ax.plot(xx + 0.07, r.tcal_minus_prod, marker="D", ms=3.4, mfc="white", mec=cc, mew=0.9, ls="none", zorder=4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.8, 8.8)
    ax.set_xlabel(t["xlab_b"])
    ax.set_ylabel(t["ylab_b"])
    ax.text(xmin + 0.1, 8.7, t["left"], ha="left", va="top", fontsize=FS - 0.8, color="#666666")
    ax.text(0.25, 8.7, t["right"], ha="left", va="top", fontsize=FS - 0.8, color="#666666")
    ax.tick_params(length=2, labelsize=FS - 0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    hb = [Line2D([], [], ls="-", marker="*", ms=7.0, mfc="#BBBBBB", mec="black", mew=0.3, color="#888888", lw=0.6, label=t["prop_ci"]),
          Line2D([], [], ls="-", marker="D", ms=3.4, mfc="white", mec="#555555", mew=0.9, color="#AAAAAA", lw=0.5, label=t["tcal_ci"])]
    hb += [Line2D([], [], ls="none", marker="s", ms=5, mfc=CCOL[c], mec="none", label=(t["iemocap"] if c == "IEMOCAP" else c)) for c in CORPORA]
    ax.legend(handles=hb, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=FS - 0.4, frameon=False)
    fig.text(0.065, 0.455, t["title_b"], ha="left", va="center", fontsize=FS + 0.6, weight="bold")
    name = "fig_corpus_types" + ("_ja" if lang == "ja" else "")
    from matplotlib.transforms import Bbox                      # slide versions: panel (a) and panel (b) as separate files
    Hf = 132 * MM
    for part, (fa, fb) in {"a": (0.505, 1.0), "b": (0.0, 0.475)}.items():
        bb = Bbox([[0, fa * Hf], [174 * MM, fb * Hf]])
        fig.savefig(OUT / f"fig_corpus_types_{part}{'_ja' if lang == 'ja' else ''}.pdf", bbox_inches=bb)
        fig.savefig(OUT / f"fig_corpus_types_{part}{'_ja' if lang == 'ja' else ''}.png", dpi=300, bbox_inches=bb)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def main():
    d = load()
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    cols = ["c", "K", "ret_A", "ret_V", "ret_T", "prod", "tcal", "prop", "best", "prod_minus_best", "tcal_minus_prod",
            "tcal_minus_prod_eff_lo", "tcal_minus_prod_eff_hi", "prop_minus_prod", "prop_minus_prod_lo", "prop_minus_prod_hi"]
    d[cols].to_csv(OUT / "data" / "fig_corpus_types.csv", index=False)
    prov = {"inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [R20 / "levels.csv", R20 / "effects.csv", R21 / "levels.csv", R21 / "effects.csv"]},
            "script": str(Path(__file__).relative_to(ROOT)), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    json.dump(prov, open(OUT / "data" / "fig_corpus_types.provenance.json", "w"), indent=1)
    for lang in ("en", "ja"):
        draw(d, lang)


if __name__ == "__main__":
    main()
