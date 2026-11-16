"""Answer grids (2026-11-16, researcher's request: show the answers to Q1 and Q3 at a glance).

Fig. Q1 grid: gain (pseudo-label retraining with the product − K-label baseline, test UAR points) for the audio, visual, text
  and AVT classifiers x 4 corpora x K = 0, 5, 10, 25. K = 0, 5, 25: results/dai1/stats/families_flagged.csv (family 1, prop − none);
  K = 10: results/dai1/stats/desc/outside_families.csv (family 1). Same values as Table 5/6 and Fig. 3 of the paper.
Fig. Q3 grid: proposed fusion − each fusion method (F1–F10), inference-time answers, 4 corpora + CREMA-D x K = 5, 10, 25.
  results/dai20_eval/summary/effects.csv (seeds = all, a = prop) and results/dai21_eval5/summary/effects.csv (part = cremad).
  Same values as Table 11 of the paper and point 20 of docs/結果/計算の後の解析_提案が効くコーパスの型_2026-11-16.md.
Cell colour = point estimate; mark * = 95% interval above 0, † = below 0.
Output: ../figures/fig_q1_grid{,_ja}.pdf/.png, fig_q3_grid{,_ja}.pdf/.png, ../figures/data/fig_q1_grid.csv, fig_q3_grid.csv.
Run:  /home/snics/MER-DA/.venv/bin/python make_fig_grids.py
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "figures"
MM = 1 / 25.4
FS = 7.0
ARIAL_DIR = Path("/mnt/c/Windows/Fonts")
if (ARIAL_DIR / "arial.ttf").exists():
    for f in ["arial.ttf", "arialbd.ttf"]:
        fm.fontManager.addfont(str(ARIAL_DIR / f))
    EN_FONT = "Arial"
else:
    EN_FONT = "Nimbus Sans"
fm.fontManager.addfont("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")
JA_FONT = "IPAexGothic"
CORP4 = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
CMAP = plt.get_cmap("RdBu")

METHODS = [  # order and Japanese names of Table 10 of the paper (07_Q3.md); (key in effects.csv, number, English, Japanese)
    ("prod", "F1", "Product (equal weights)", "積（3 つの確率を同じ重みで掛ける）"),
    ("avg", "F2", "Mean of probabilities", "確率の平均"),
    ("psel", "F3", "Product with exponents chosen on other corpora", "別のコーパスで選んだ指数の積"),
    ("conf", "F4", "Confidence-weighted mean", "確信度で重みづける平均"),
    ("M_F2_K", "F5", "Level-only weights (K labels)", "水準だけの重み（K 本）"),
    ("M_F6_K", "F6", "All 12 reliabilities from K labels", "全部を K 本で測った表"),
    ("cf_D_l05", "F7", "Product after confusion-table re-reading", "表で読み替えて掛ける（K 本）"),
    ("tcal", "F8", "Per-classifier temperature scaling (K labels)", "分類器ごとの温度の較正（K 本）"),
    ("temp", "F9", "Modality weights fitted on K labels", "K 本で合わせたモダリティごとの重み"),
    ("wf1000", "F10", "Best of 1,001 weight tables (K labels)", "K 本で重みの表を探す（1,001 個）"),
]


def font(lang):
    plt.rcParams.update({"font.family": EN_FONT if lang == "en" else JA_FONT, "font.size": FS,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5})


def mark(lo, hi):
    return "*" if lo > 0 else ("†" if hi < 0 else "")


def q1_data():
    f = pd.read_csv(ROOT / "results/dai1/stats/families_flagged.csv", comment="#")
    f = f[(f.family == 1) & (f.A == "prop") & (f.B == "none")][["T", "student", "K", "effect", "lo", "hi"]]
    o = pd.read_csv(ROOT / "results/dai1/stats/desc/outside_families.csv", comment="#")
    o = o[(o.family == 1) & (o.K == 10) & (o.A == "prop") & (o.B == "none")][["T", "student", "K", "effect", "lo", "hi"]]
    d = pd.concat([f, o])
    if len(d) != 64:
        raise SystemExit(f"Q1 rows {len(d)} (expected 64)")
    return d


def q3_data():
    e = pd.read_csv(ROOT / "results/dai20_eval/summary/effects.csv", comment="#")
    e = e[(e.seeds.astype(str) == "all") & (e.a == "prop")][["c", "K", "b", "est", "lo", "hi"]]
    e21 = pd.read_csv(ROOT / "results/dai21_eval5/summary/effects.csv", comment="#")
    e21 = e21[(e21.seeds.astype(str) == "all") & (e21.part == "cremad") & (e21.a == "prop")][["c", "K", "b", "est", "lo", "hi"]]
    d = pd.concat([e, e21])
    d = d[d.b.isin([m[0] for m in METHODS])]
    return d


def grid(ax, vals, marks, row_labels, col_groups, ks, vmax, lang, cell_fmt="{:+.1f}"):
    nr, nc = vals.shape
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for i in range(nr):
        for j in range(nc):
            v = vals[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j, nr - 1 - i), 1, 1, fc="#F0F0F0", ec="white", lw=0.8))
                ax.text(j + 0.5, nr - 1 - i + 0.5, "—", ha="center", va="center", fontsize=FS - 1, color="#999999")
                continue
            c = CMAP(norm(np.clip(v, -vmax, vmax)))
            ax.add_patch(plt.Rectangle((j, nr - 1 - i), 1, 1, fc=c, ec="white", lw=0.8))
            dark = abs(norm(v) - 0.5) > 0.32
            ax.text(j + 0.5, nr - 1 - i + 0.5, (cell_fmt.format(v) if round(v, 1) != 0 else "0.0").replace("-", "\u2212") + marks[i][j], ha="center", va="center",
                    fontsize=FS - 1.2, color="white" if dark else "black")
    ax.set_xlim(0, nc)
    ax.set_ylim(0, nr)
    ax.set_yticks([nr - 1 - i + 0.5 for i in range(nr)])
    ax.set_yticklabels(row_labels, fontsize=FS - 0.2)
    ax.set_xticks([j + 0.5 for j in range(nc)])
    ax.set_xticklabels([f"{k}" for _ in col_groups for k in ks], fontsize=FS - 1.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for g, name in enumerate(col_groups):
        ax.text(g * len(ks) + len(ks) / 2, nr + 0.25, name, ha="center", va="bottom", fontsize=FS - 0.2)
        if g:
            ax.axvline(g * len(ks), color="black", lw=0.8)
    ax.text(-0.15, -0.55, "K =", ha="right", va="center", fontsize=FS - 1.2, transform=ax.transData)
    return norm


def fig_q1(lang):
    font(lang)
    d = q1_data()
    rows = ["A", "V", "T", "AVT"]
    names = {"en": {"A": "Audio", "V": "Visual", "T": "Text", "AVT": "AVT (concatenated)"},
             "ja": {"A": "声", "V": "顔", "T": "言葉", "AVT": "AVT（つないだ特徴）"}}[lang]
    ks = [0, 5, 10, 25]
    vals = np.full((4, 16), np.nan)
    marks = [[""] * 16 for _ in range(4)]
    for i, s in enumerate(rows):
        for g, c in enumerate(CORP4):
            for k_i, K in enumerate(ks):
                r = d[(d.student == s) & (d["T"] == c) & (d.K == K)]
                if len(r) != 1:
                    raise SystemExit(f"Q1 missing {s} {c} {K}")
                r = r.iloc[0]
                vals[i, g * 4 + k_i] = r.effect
                marks[i][g * 4 + k_i] = mark(r.lo, r.hi)
    fig = plt.figure(figsize=(174 * MM, 54 * MM))
    ax = fig.add_axes([0.15, 0.24, 0.72, 0.6])
    cg = [c if c != "IEMOCAP" else ("IEMOCAP (impro.)" if lang == "en" else "IEMOCAP（即興）") for c in CORP4]
    norm = grid(ax, vals, marks, [names[s] for s in rows], cg, ks, 6.0, lang)
    cax = fig.add_axes([0.89, 0.24, 0.012, 0.6])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.ax.tick_params(labelsize=FS - 1.2, length=2)
    cb.set_label("gain (UAR points)" if lang == "en" else "gain（UAR の点）", fontsize=FS - 1)
    note = ("gain = retraining on the product pseudo-labels − K-label baseline (test UAR). * 95% interval above 0, † below 0. "
            "\nK = 0: no target labels (baseline: average of the source classifiers, the anchor)." if lang == "en" else
            "gain ＝ 積の疑似ラベルで学び直した分類器 − K 本の基準の分類器（テストの UAR）．* は 95% 区間の下端 > 0，† は上端 < 0．\nK = 0 はターゲットのラベルを使わない（基準はソースの分類器の平均．アンカー）．")
    fig.text(0.02, 0.03, note, ha="left", va="bottom", fontsize=FS - 1.4, color="#444444")
    name = "fig_q1_grid" + ("_ja" if lang == "ja" else "")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    rec = [dict(student=s, corpus=c, K=K, effect=vals[i, g * 4 + k_i], mark=marks[i][g * 4 + k_i])
           for i, s in enumerate(rows) for g, c in enumerate(CORP4) for k_i, K in enumerate(ks)]
    return pd.DataFrame(rec)


def fig_q3(lang):
    font(lang)
    d = q3_data()
    corp = CORP4 + ["CREMA-D"]
    ks = [5, 10, 25]
    vals = np.full((len(METHODS), 15), np.nan)
    marks = [[""] * 15 for _ in METHODS]
    for i, (key, num, en, ja) in enumerate(METHODS):
        for g, c in enumerate(corp):
            for k_i, K in enumerate(ks):
                r = d[(d.b == key) & (d.c == c) & (d.K == K)]
                if len(r) > 1:
                    raise SystemExit(f"Q3 duplicate {key} {c} {K}")
                if len(r) == 1:
                    r = r.iloc[0]
                    vals[i, g * 3 + k_i] = r.est
                    marks[i][g * 3 + k_i] = mark(r.lo, r.hi)
                elif not (key == "psel" and c == "CREMA-D"):
                    raise SystemExit(f"Q3 missing {key} {c} {K}")
    fig = plt.figure(figsize=(174 * MM, 66 * MM))
    ax = fig.add_axes([0.31, 0.15, 0.58, 0.72])
    labels = [f"{num}  {en if lang == 'en' else ja}" for key, num, en, ja in METHODS]
    cg = [c if c != "IEMOCAP" else ("IEMOCAP (impro.)" if lang == "en" else "IEMOCAP（即興）") for c in corp]
    norm = grid(ax, vals, marks, labels, cg, ks, 4.0, lang)
    cax = fig.add_axes([0.905, 0.15, 0.012, 0.72])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.ax.tick_params(labelsize=FS - 1.2, length=2)
    cb.set_label("proposed − method (UAR points)" if lang == "en" else "提案 − 各方法（UAR の点）", fontsize=FS - 1)
    note = ("Proposed fusion − each method, fused answers on the test utterances. Blue: proposed higher. * 95% interval above 0, † below 0. "
            "F3 does not exist for CREMA-D (two classifiers)." if lang == "en" else
            "提案 − 各方法（まとめた答えのテストの UAR）．青は提案のほうが高い．* は 95% 区間の下端 > 0，† は上端 < 0．CREMA-D は分類器が 2 つで F3 は無い．")
    fig.text(0.02, 0.025, note, ha="left", va="bottom", fontsize=FS - 1.4, color="#444444")
    name = "fig_q3_grid" + ("_ja" if lang == "ja" else "")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    rec = [dict(method=METHODS[i][1], key=METHODS[i][0], corpus=c, K=K, effect=vals[i, g * 3 + k_i], mark=marks[i][g * 3 + k_i])
           for i in range(len(METHODS)) for g, c in enumerate(corp) for k_i, K in enumerate(ks)]
    return pd.DataFrame(rec)


# --- slide versions (2026-11-16 evening): proposed − product per corpus, and the Q3 grid with rows grouped by type
TYPES = {
    "ja": [("ラベルを\n使わない", ["F1", "F2", "F4"]), ("別のコーパス\nで決める", ["F3"]),
           ("K 本で 3 つの\n数を決める", ["F5", "F8", "F9"]), ("K 本で表を\n測る・探す", ["F6", "F7", "F10"])],
    "en": [("No labels\nused", ["F1", "F2", "F4"]), ("Set on other\ncorpora", ["F3"]),
           ("3 numbers set\non K labels", ["F5", "F8", "F9"]), ("Table set\non K labels", ["F6", "F7", "F10"])],
}
SHORT = {
    "ja": {"F1": "F1 積（同じ重み）", "F2": "F2 確率の平均", "F4": "F4 確信度で重みづけ", "F3": "F3 別のコーパスで選んだ指数",
           "F5": "F5 水準だけの重み", "F8": "F8 温度の較正", "F9": "F9 K 本で合わせた重み",
           "F6": "F6 全部を K 本で測った表", "F7": "F7 取り違えの表で読み替え", "F10": "F10 重みの表を探す"},
    "en": {"F1": "F1 Product (equal)", "F2": "F2 Mean", "F4": "F4 Confidence-weighted", "F3": "F3 Exponents from other corpora",
           "F5": "F5 Level-only weights", "F8": "F8 Temperature scaling", "F9": "F9 Weights fitted on K",
           "F6": "F6 All-K table", "F7": "F7 Confusion re-reading", "F10": "F10 Table search"},
}


def fig_prop_vs_prod(lang):
    font(lang)
    d = q3_data()
    d = d[d.b == "prod"]
    corp = CORP4 + ["CREMA-D"]
    ks = [5, 10, 25]
    kcol = {5: "#FDB863", 10: "#E66101", 25: "#7F2704"}
    fig = plt.figure(figsize=(174 * MM, 70 * MM))
    ax = fig.add_axes([0.08, 0.16, 0.9, 0.72])
    rec = []
    for g, c in enumerate(corp):
        for k_i, K in enumerate(ks):
            r = d[(d.c == c) & (d.K == K)]
            if len(r) != 1:
                raise SystemExit(f"prop-prod missing {c} {K}")
            r = r.iloc[0]
            x = g * 4 + k_i
            ax.bar(x, r.est, width=0.8, color=kcol[K], label=f"K = {K}" if g == 0 else None, zorder=2)
            ax.errorbar(x, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5, zorder=3)
            ytxt = max(r.hi, 0) + 0.12
            ax.text(x, ytxt, f"{r.est:+.2f}".replace("-", "\u2212") + mark(r.lo, r.hi), ha="center", va="bottom", fontsize=FS - 1.6)
            rec.append(dict(corpus=c, K=K, est=r.est, lo=r.lo, hi=r.hi, mark=mark(r.lo, r.hi)))
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks([g * 4 + 1 for g in range(len(corp))])
    ax.set_xticklabels([c if c != "IEMOCAP" else ("IEMOCAP (impro.)" if lang == "en" else "IEMOCAP（即興）") for c in corp], fontsize=FS)
    ax.set_ylabel("proposed − product (UAR points)" if lang == "en" else "提案 − 積（UAR の点）", fontsize=FS - 0.5)
    ax.tick_params(labelsize=FS - 1, length=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_ylim(-1.2, 6.2)
    ax.axvspan(-0.6, 2.6, color="#FDF0E8", zorder=0)
    ax.legend(fontsize=FS - 1, frameon=False, loc="upper right", ncol=3)
    note = ("Fused answers on the test utterances (before retraining). Bars: mean of 50 repetitions; lines: 95% intervals; * interval above 0. CREMA-D: audio and visual only." if lang == "en" else
            "まとめた答え（学び直す前）のテストの UAR の差．棒は 50 反復の平均，線は 95% 区間，* は区間の下端 > 0．CREMA-D は声と顔の 2 つの分類器．")
    fig.text(0.02, 0.02, note, ha="left", va="bottom", fontsize=FS - 1.4, color="#444444")
    name = "fig_q3_prop_vs_prod" + ("_ja" if lang == "ja" else "")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    return pd.DataFrame(rec)


def fig_q3_grid_typed(lang):
    font(lang)
    d = q3_data()
    corp = CORP4 + ["CREMA-D"]
    ks = [5, 10, 25]
    meta = {num: (key, en, ja) for key, num, en, ja in METHODS}
    order = [f for _, fs in TYPES[lang] for f in fs]
    vals = np.full((len(order), 15), np.nan)
    marks = [[""] * 15 for _ in order]
    for i, f in enumerate(order):
        key = meta[f][0]
        for g, c in enumerate(corp):
            for k_i, K in enumerate(ks):
                r = d[(d.b == key) & (d.c == c) & (d.K == K)]
                if len(r) == 1:
                    vals[i, g * 3 + k_i] = r.iloc[0].est
                    marks[i][g * 3 + k_i] = mark(r.iloc[0].lo, r.iloc[0].hi)
    fig = plt.figure(figsize=(174 * MM, 72 * MM))
    ax0, ay0, aw, ah = 0.33, 0.13, 0.58, 0.74
    ax = fig.add_axes([ax0, ay0, aw, ah])
    labels = [SHORT[lang][f] for f in order]
    norm = grid(ax, vals, marks, labels, [c if c != "IEMOCAP" else ("IEMOCAP (impro.)" if lang == "en" else "IEMOCAP（即興）") for c in corp], ks, 4.0, lang)
    nr = len(order)
    row = 0
    for gi, (gname, fs) in enumerate(TYPES[lang]):
        top = nr - row
        bot = nr - row - len(fs)
        if gi:
            yf = ay0 + ah * top / nr
            fig.add_artist(plt.Line2D([0.01, ax0 + aw], [yf, yf], color="black", lw=0.9))
        fig.text(0.015, ay0 + ah * (top + bot) / 2 / nr, gname, ha="left", va="center", fontsize=FS - 0.8,
                 color="#333333", linespacing=1.1)
        row += len(fs)
    cax = fig.add_axes([0.925, 0.13, 0.012, 0.74])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.ax.tick_params(labelsize=FS - 1.2, length=2)
    cb.set_label("proposed − method" if lang == "en" else "提案 − 各方法（点）", fontsize=FS - 1)
    note = ("Blue: proposed fusion higher; red: the method higher. * 95% interval above 0, † below 0. F3 does not exist for CREMA-D." if lang == "en" else
            "青：提案のほうが高い．赤：その方法のほうが高い．* は 95% 区間の下端 > 0，† は上端 < 0．CREMA-D には F3 が無い．")
    fig.text(0.02, 0.025, note, ha="left", va="bottom", fontsize=FS - 1.4, color="#444444")
    name = "fig_q3_grid_typed" + ("_ja" if lang == "ja" else "")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def main():
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    for lang in ("en", "ja"):
        q1 = fig_q1(lang)
        q3 = fig_q3(lang)
        pp = fig_prop_vs_prod(lang)
        fig_q3_grid_typed(lang)
    q1.to_csv(OUT / "data" / "fig_q1_grid.csv", index=False)
    q3.to_csv(OUT / "data" / "fig_q3_grid.csv", index=False)
    pp.to_csv(OUT / "data" / "fig_q3_prop_vs_prod.csv", index=False)
    ins = ["results/dai1/stats/families_flagged.csv", "results/dai1/stats/desc/outside_families.csv",
           "results/dai20_eval/summary/effects.csv", "results/dai21_eval5/summary/effects.csv"]
    json.dump({"inputs": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ins},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              open(OUT / "data" / "fig_grids.provenance.json", "w"), indent=1)
    print("wrote grids")


if __name__ == "__main__":
    main()
