"""DAI-8 の集計のファイルの点検（gate-code）．値は出さず，行の数・重複・NaN の数だけを出す．"""
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path("/home/snics/MER-DA/results/dai8_probe")
print("出力先のファイル", sorted(p.name for p in OUT.iterdir()))
C = pd.read_csv(OUT / "cells.csv")
print("cells.csv", len(C), "行，(c,key) の重複", C.duplicated(["c", "key"]).sum(), "count の分布", dict(Counter(C["count"])),
      "c ごと", dict(Counter(C.c)), "mean の NaN", int(C["mean"].isna().sum()), "0〜100 の外", int(((C["mean"] < 0) | (C["mean"] > 100)).sum()))
D = pd.read_csv(OUT / "diffs.csv")
print("diffs.csv", len(D), "行，(c,K,space,compare,subset) の重複", D.duplicated(["c", "K", "space", "compare", "subset"]).sum())
print("  subset ごと", dict(Counter(zip(D.c, D.subset))))
print("  c×subset=all の行", dict(Counter(D[D.subset == "all"].c)))
for col in ("effect", "lo", "hi", "se", "n_folds", "n_units"):
    print(f"  {col} の NaN", int(D[col].isna().sum()))
print("  n_folds の分布", dict(Counter(D.n_folds)), "n_units の分布", dict(Counter(D.n_units)), "method", dict(Counter(zip(D.c, D.method))))
print("  folds_dropped の分布", dict(Counter(D.folds_dropped.astype(str))), "identical が真", int(D.identical.astype(bool).sum()))
print("  lo ≤ effect ≤ hi に反する", int(((D.lo > D.effect + 1e-9) | (D.hi < D.effect - 1e-9)).sum()))
# 規則の比べの行
for nm in ("em_none", "em_joint", "emfromK_ldaK", "em_ldaK", "em_emmu", "em_ttfused", "emT_noneT"):
    n_ = D[(D["compare"] == nm) & (D.space == "AVT") & (D.subset == "all") & D.K.isin([1, 5, 10, 25])].groupby("c").size().to_dict()
    print("  規則の比べ", nm, n_)
for nm in ("ldaO_oracle", "emtruth_ldaO", "emrand_ldaO", "ldaOblk_ldaO"):
    print("  K に依らない比べ", nm, D[(D["compare"] == nm) & (D.space == "AVT")].groupby("c").size().to_dict())
# 予定の比べの数
exp_rows = 0
for s in ("A", "V", "T", "AVT"):
    exp_rows += (4 if s == "AVT" else 3) + 4 * (19 if s == "AVT" else 18) + (15 if s == "AVT" else 14)
print("  予定の all の行（c あたり）", exp_rows, "実際", dict(Counter(D[D.subset == "all"].c)))
print("  比べの名前ごとの行の数", dict(Counter(D[D.subset == "all"]["compare"])))
G = pd.read_csv(OUT / "diag_units.csv")
print("diag_units.csv", len(G), "行，(c,T,fold,seed,key) の重複", G.duplicated(["c", "T", "fold", "seed", "key"]).sum(),
      "単位の数", G.groupby(["c", "T", "fold", "seed"]).ngroups, "単位あたりの鍵", dict(Counter(G.groupby(["c", "T", "fold", "seed"]).size())))
def fam(k):
    p = k.split("_")
    if k.startswith("K"):
        return "_".join(p[2:])
    return "_".join(p[1:])
G["fam"] = G.key.map(fam)
num = [c for c in G.columns if c not in ("c", "T", "key", "fam")]
part = []
for f_, g in G.groupby("fam"):
    for col in num:
        n = int(g[col].isna().sum())
        if 0 < n < len(g):
            part.append((f_, col, n, len(g)))
print("  一部の行だけ NaN の（族，列）", part if part else "なし")
allnan = sorted({(f_.split("_")[0], col) for f_, g in G.groupby("fam") for col in num if g[col].isna().all()})
print("  全部の行で NaN の（族の頭，列）", allnan)
inf = int(np.isinf(G[num].select_dtypes("number").to_numpy()).sum())
print("  Inf の数", inf)
for col in ("pool_uar", "pool_uar_none", "pool_uar_at10"):
    v = G[col].dropna(); print(f"  {col} の 0〜100 の外", int(((v < 0) | (v > 100)).sum()))
for col in ("moved", "moved_at10", "keep_init", "keep_init_at10", "agree_none_test", "beta_flip_last") + tuple(f"pool_rec_e{e}" for e in range(4)):
    v = G[col].dropna(); print(f"  {col} の 0〜1 の外", int(((v < 0) | (v > 1)).sum()))
v = G.entropy.dropna(); print("  entropy の 0〜log4 の外", int(((v < -1e-9) | (v > np.log(4) + 1e-9)).sum()))
print("  n_iter_done の分布（収束の版の族）", {f_: (int(g.n_iter_done.min()), int(g.n_iter_done.max()), int((g.n_iter_done == 100).sum())) for f_, g in G.groupby("fam")
       if f_ in ("em_a0.5_b1.0", "emblk_a0.5_b1.0", "emtruth_a0.5_b0.0", "emrand_a0.5_b0.0")})
DG = pd.read_csv(OUT / "diag.csv")
print("diag.csv", len(DG), "行，(c,key) の重複", DG.duplicated(["c", "key"]).sum())
K = pd.read_csv(OUT / "knn_units.csv")
print("knn_units.csv", len(K), "行，(c,fold,space) の重複", K.duplicated(["c", "fold", "space"]).sum(), "c ごと", dict(Counter(K.c)))
kn = {c: int(K[c].isna().sum()) for c in K.columns if K[c].isna().any()}
print("  NaN のある列", kn, "IEMOCAP の行の scen の NaN", int(K.loc[K.c == "IEMOCAP", ["scen", "emo_xscen", "emo_gain_xscen"]].isna().sum().sum()))
print("  Inf", int(np.isinf(K.select_dtypes("number").to_numpy()).sum()))
A = pd.read_csv(OUT / "ami_units.csv")
print("ami_units.csv", len(A), "行，重複", A.duplicated(["c", "T", "fold", "seed", "key"]).sum(), "単位あたり", dict(Counter(A.groupby(["c", "T", "fold", "seed"]).size())),
      "NaN", int(A[["ami_emo", "ami_grp", "ami_dlg"]].isna().sum().sum()), "鍵の種類", A.key.map(lambda k: "_".join(k.split("_")[2:]) if k.startswith("K") else k).nunique())
M = pd.read_csv(OUT / "cmi_units.csv")
print("cmi_units.csv", len(M), "行，重複", M.duplicated(["c", "T", "fold", "seed", "key"]).sum(), "単位あたり", dict(Counter(M.groupby(["c", "T", "fold", "seed"]).size())))
xc = [c for c in M.columns if c.startswith(("x_", "cmi"))]
nanby = {}
for c in xc:
    m = M[c].isna()
    if m.any():
        nanby[c] = dict(Counter(M.key[m].map(lambda k: k if k.startswith("ref") else "assign")))
print("  NaN の（列：鍵の種類→数）", nanby)
print("  Inf", int(np.isinf(M[xc].to_numpy()).sum()))
R = pd.read_csv(OUT / "criteria.csv")
print("criteria.csv", len(R), "行", len(R.columns), "列，c", list(R.c))
print("  NaN のある列とその c", {c: list(R.c[R[c].isna()]) for c in R.columns if R[c].isna().any()})
print("  空の文字列の列", [c for c in R.columns if R[c].dtype == object and (R[c].astype(str).str.strip() == "").any()])
need = ["R1", "R2", "R3", "R3x", "R3_harm", "R4", "R4_effect", "R6", "R7", "R8", "not_moved", "M1_reading", "M1_drifted", "R10_top", "R10_em_top_ami",
        "R10_ami_sentence_ok", "R5_top3_top", "R5_nonemo_separates", "R5_n_fold", "n_not_conv_at_rule"]
print("  規則の列の欠け", [c for c in need if c not in R.columns])
print("  R*_n_corpora の列", sorted(c for c in R.columns if c.endswith("_n_corpora")))
print("  R5_n_fold の値（分割の数）", list(R.R5_n_fold))
