"""集計の各ファイルの行の数・欠け・重複・NaN（値は出さない）。"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA"); OUT = ROOT / "results/dai9_probe"
sys.path.insert(0, str(ROOT / "DAI/exp"))
TAUS = ("raw", "gc", "nap10", "nap50", "pca64", "pca256"); KS = (1, 5, 10, 25)
prov = json.loads((OUT / "provenance.json").read_text())
UD = ROOT / "results/dai1/pseudo/units"
bad = [k for k, v in prov["inputs_sha256"].items() if hashlib.sha256((UD / f"{k}_p1.npz").read_bytes()).hexdigest() != v]
print("入力の sha256 が今のファイルと違う", len(bad), "／", len(prov["inputs_sha256"]))
print("runs", len(prov["runs"]), "名前の重複", len(prov["runs"]) - len({r["name"] for r in prov["runs"]}))
C = pd.read_csv(OUT / "cells.csv")
print("cells", C.shape, "c×key の重複", C.duplicated(["c", "key"]).sum(), "c ごとの key の数", C.groupby("c").key.nunique().to_dict(),
      "count が 10 でない", int((C["count"] != 10).sum()), "NaN", int(C.isna().sum().sum()),
      "UAR（百分率）の 0〜100 の外", int(((C["mean"] < 0) | (C["mean"] > 100)).sum()), "1 以下の行", int((C["mean"] <= 1).sum()), "P_ や pool の鍵", int(C.key.str.startswith(("P_", "pool")).sum()))
D = pd.read_csv(OUT / "diffs.csv")
print("diffs", D.shape, "列", list(D.columns))
print("  c×compare×K の重複", int(D.duplicated(["c", "compare", "K"]).sum()), "c ごとの行", D.groupby("c").size().to_dict())
print("  NaN の列ごとの数", {k: int(v) for k, v in D.isna().sum().items() if v})
print("  Inf", int(np.isinf(D.select_dtypes("number")).sum().sum()), "lo>hi", int((D.lo > D.hi).sum()),
      "点推定が区間の外", int(((D.effect < D.lo) | (D.effect > D.hi)).sum()))
exp = []
for t in TAUS:
    exp += [(f"{t}_ldaO_vs_raw", 0), (f"{t}_emtruth_ldaO", 0), (f"{t}_emtruth_vs_raw", 0)]
    for K in KS:
        exp += [(f"{t}_emK_ldaK", K), (f"{t}_emKconv_ldaK", K), (f"{t}_emK_ldaKraw", K), (f"{t}_ldaK_ldaKraw", K)]
for c in sorted(D.c.unique()):
    have = set(zip(D[D.c == c]["compare"], D[D.c == c].K))
    print("  ", c, "予定の比べの欠け", len(set(exp) - have), "余り", len(have - set(exp)))
DG = pd.read_csv(OUT / "diag_units.csv")
print("diag_units", DG.shape, "c×fold×seed×key の重複", int(DG.duplicated(["c", "fold", "seed", "key"]).sum()),
      "単位の数", DG.groupby(["c", "fold", "seed"]).ngroups, "単位ごとの key の数", sorted(DG.groupby(["c", "fold", "seed"]).key.nunique().unique()))
# 鍵の型ごとに，要る列の NaN
def kind(k):
    if k.endswith("_info"): return "info_nap" if k.startswith("nap") else ("info_gc" if k.startswith("gc") else ("info_pca" if k.startswith("pca") else "info_raw"))
    if k.endswith("_ldaO"): return "ldaO"
    if k.endswith(("_emtruth", "_emrand")): return "emdiag"
    return "emK"
need = {"ldaO": ["pool_uar"], "emdiag": ["pool_uar", "keep_init", "n_iter_done", "hist"],
        "emK": ["pool_uar", "n_iter_done", "pool_uar_init", "moved_at10", "hist_at10", "hist"],
        "info_nap": ["nap_r_used", "dim", "n_over_d", "eig_top", "mod_share"], "info_pca": ["dim", "n_over_d", "mod_share"],
        "info_gc": ["dim", "n_over_d", "small_group_share_pool", "small_group_share_test"], "info_raw": ["dim", "n_over_d"]}
DG["kind"] = DG.key.map(kind)
for kd, cols in need.items():
    s = DG[DG.kind == kd]
    print("  ", kd, len(s), "行，要る列の NaN", {c_: int(s[c_].isna().sum()) for c_ in cols if c_ in s} , "無い列", [c_ for c_ in cols if c_ not in s])
print("  pca の dim が p と一致", int(sum(DG[(DG.kind == "info_pca")].apply(lambda r: r.dim == int(r.key[3:-5]), axis=1))), "/", int((DG.kind == "info_pca").sum()))
print("  hist が空の行", int((DG["hist"].astype(str) == "[]").sum()))
Q = pd.read_csv(OUT / "q1_units.csv")
print("q1_units", Q.shape, "c×fold×tau の重複", int(Q.duplicated(["c", "fold", "tau"]).sum()), "NaN", int(Q.isna().sum().sum()),
      "Inf", int(np.isinf(Q.select_dtypes("number")).sum().sum()), "c ごと", Q.groupby("c").size().to_dict(), "commit", Q.commit.unique().tolist())
Cr = pd.read_csv(OUT / "criteria.csv", keep_default_na=False, na_values=["nan", "NaN", ""])
print("criteria", Cr.shape, "c", Cr.c.tolist())
pref = {}
for col in Cr.columns:
    p = col.split("_")[0]
    pref.setdefault(p, []).append(col)
print("  頭ごとの列の数", {k: len(v) for k, v in pref.items()})
for t in TAUS:
    need_c = [f"S1_{t}_n_K", f"S1_{t}_n_hi_neg", f"S1_{t}", f"S1_{t}_harm", f"S2_{t}_truth_minus_ldaO", f"S2_{t}_keep_truth", f"S2_{t}_keep_rand", f"S2_{t}",
              f"S4_{t}_top3ex_emo", f"S4_{t}_top3ex_grp_xe", f"S4_{t}_top3ex_dlg_xe", f"S4_{t}_nonemo_separates", f"S5_{t}_top3ex_emo_ratio",
              f"S0_{t}_moved_at10", f"S0_{t}_not_moved", f"S0_{t}_n_not_conv_at10", f"S0_{t}_n_over_d", f"S1_{t}_n_corpora"]
    if t in ("gc", "nap10", "nap50"):
        need_c += [f"HYP_{t}_n_fold_pos", f"HYP_{t}_emtruth_vs_raw", f"HYP_{t}_emtruth_vs_raw_lo", f"S5_{t}_ldaO_vs_raw", f"S5_{t}_removed_emotion", f"HYP_{t}_blocked_by_emotion_loss"]
    if t.startswith("nap"): need_c.append(f"{t}_r_used_min")
    miss = [x for x in need_c if x not in Cr.columns]
    nan = {x: int(Cr[x].isna().sum()) for x in need_c if x in Cr.columns and Cr[x].isna().any()}
    print("  ", t, "要る列", len(need_c), "無い", miss, "NaN のある列", list(nan))
for x in ("HYP_taus", "HYP_n_taus", "HYP", "HYP_n_corpora", "HYP_nap_same"):
    print("  ", x, "列あり" if x in Cr.columns else "列なし", (int(Cr[x].isna().sum()) if x in Cr.columns else "-"))
# S5 比の NaN が n2 の規則どおりか（q1 から作り直して NaN の位置だけを比べる）
qr = Q[Q.tau == "raw"]
okraw = {c: int((qr[qr.c == c].top3_emo > qr[qr.c == c].top3nullmax_emo).sum()) >= 3 for c in Cr.c}
mism = 0
for t in TAUS:
    for _, r in Cr.iterrows():
        mism += int(pd.isna(r[f"S5_{t}_top3ex_emo_ratio"]) == okraw[r.c])
print("  S5 比の NaN の位置が n2 の規則と食い違う数", mism)
# 予想の下端の NaN が無いか（IEMOCAP の jackknife）
print("  Inf", int(np.isinf(Cr.select_dtypes("number")).sum().sum()))
