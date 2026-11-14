"""DAI-20 計算の後の点検（単位の保存）．results/dai20_eval は読むだけ．値（UAR）は画面に出さない．
数・一致の割合・ずれの大きさだけを出す．"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA")
R = ROOT / "results" / "dai20_eval"
U = R / "units"
CORP = ("AFEW", "IEMOCAP", "MOSEI", "MELD")

csvs = sorted(U.glob("*.csv"))
npzs = sorted(U.glob("*.npz"))
print("csv", len(csvs), "npz", len(npzs))
stem_c = {p.stem for p in csvs}
stem_n = {p.stem for p in npzs}
print("csv だけ", sorted(stem_c - stem_n), "npz だけ", sorted(stem_n - stem_c))
exp = {f"{T}_ALL_f{f}_r{r}" for T in CORP for f in range(5) for r in range(10)}
print("期待の 200 との差：欠け", sorted(exp - stem_c), "余り", sorted(stem_c - exp))

dfs = []
for p in csvs:
    d = pd.read_csv(p, low_memory=False)
    d["_file"] = p.stem
    dfs.append(d)
rowsn = pd.Series([len(d) for d in dfs])
print("行の数の分布", rowsn.value_counts().to_dict())
A = pd.concat(dfs, ignore_index=True)
print("全部の行", len(A))
# 名前と列の中身
bad = A[(A["T"] + "_ALL_f" + A.fold.astype(str) + "_r" + A.seed.astype(str)) != A._file]
print("ファイル名と (T, fold, seed) の食い違い", len(bad), "c != T の行", int((A.c != A["T"]).sum()), "tag", A.tag.unique().tolist())
print("重複 (T,c,fold,seed,K,version)", int(A.duplicated(["T", "c", "fold", "seed", "K", "version"]).sum()))
print("K の値", sorted(A.K.unique().tolist()))
# 版の並びが全部の単位で同じか
sets = A.groupby("_file").apply(lambda d: tuple(sorted(zip(d.K, d.version.astype(str)))), include_groups=False)
print("(K, 版) の組の種類の数", sets.nunique(), "1 単位あたり", len(sets.iloc[0]))
vers = sorted({v for _, v in sets.iloc[0]})
kv = pd.Series([k for k, _ in sets.iloc[0]]).value_counts().to_dict()
print("K ごとの版の数", kv, "版の種類", len(vers))
need = (["prod", "psel", "avg", "conf", "M_F2_K", "M_F6_K", "cf_D_l05", "tcal", "temp", "tempb", "wf10", "wf100", "wf1000", "M_F6_ext", "M_F8", "M_F6_orc",
         "none_AVT", "joint_AVT", "ret_A", "ret_V", "ret_T", "b_kd", "b_agree", "b_mix", "b_cotrain", "b_coreg", "_check", "_check20", "prod_sel"]
        + [f"M_F6_mix{i}" for i in range(16)] + [f"M_F6n_mix{i}" for i in range(16)] + [f"M_F6h_mix{i}" for i in range(16) if i in ()]
        + [f"est_M_F6_mix{i}" for i in range(16)])
print("必要な版で無いもの", [v for v in need if v not in vers])
print("M_F6h_mix の番号", sorted(int(v.replace("M_F6h_mix", "")) for v in vers if v.startswith("M_F6h_mix")))
# commit / dirty / host / gpu
for col in ("commit", "dirty", "host", "gpu", "dai1_commits"):
    print(col, A[col].astype(str).unique().tolist()[:5], A[col].nunique())
# NaN / inf
num = A.select_dtypes(include=[np.number])
inf = np.isinf(num.to_numpy()).sum(0)
print("無限大のある列", {c: int(n) for c, n in zip(num.columns, inf) if n})
for col in ("uar_raw", "uar_adj", "uar_pool"):
    print(col, "NaN の行の版", A[A[col].isna()].version.astype(str).str.replace(r"\d+$", "#", regex=True).value_counts().to_dict())
    v = A[col].dropna()
    print(col, "0〜100 の外", int(((v < 0) | (v > 100)).sum()))
allnan = [c for c in A.columns if A[c].isna().all()]
print("全部 NaN の列", allnan)
# 検算の量
ck = A[A.version == "_check"]
c2 = A[A.version == "_check20"]
print("_check の行", len(ck), "_check20 の行", len(c2))
print("teacher_dev 最大", ck.teacher_dev.max(), "A/V/T", ck.teacher_dev_A.max(), ck.teacher_dev_V.max(), ck.teacher_dev_T.max())
print("grad 最大", ck.grad.max(), "stalled 合計", ck.stalled.sum())
print("adj_resid 最大", A.adj_resid.max(), "版", A.loc[A.adj_resid.idxmax(), "version"])
print("recheck_agree_min 最小", c2.recheck_agree_min.min(), "eq_agree 最小", c2.eq_agree.min(), "psel_agree 最小", c2.psel_agree.min())
wf = A[A.version.astype(str).str.fullmatch(r"wf\d+")]
print("wf_n_capped 合計", wf.wf_n_capped.fillna(0).sum())
print("tcal/temp/tempb の w_bound の平均", A[A.version.isin(["tcal", "temp", "tempb"])].groupby("version").w_bound.mean().round(4).to_dict())
print("cf_D_l05 n_unused の合計", A[A.version == "cf_D_l05"].n_unused.sum() if "n_unused" in A else None)
print("所要（秒）：単位の合計", A.groupby("_file").sec.first().sum(), "最大", A.groupby("_file").sec.first().max())
print("T ごとの所要の平均", A.groupby("T").sec.first().round(0).to_dict() if False else A.groupby(["T", "_file"]).sec.first().groupby("T").mean().round(0).to_dict())

# npz の中身：キーの集合，NaN/inf
keysets, nanrep, n = {}, {}, 0
for p in npzs:
    z = np.load(p, allow_pickle=False)
    ks = tuple(sorted(z.files))
    keysets[ks] = keysets.get(ks, 0) + 1
    for k in z.files:
        a = z[k]
        if a.dtype.kind == "f":
            nn, ni = int(np.isnan(a).sum()), int(np.isinf(a).sum())
            if nn or ni:
                key = k
                for KK in ("_K5", "_K10", "_K25"):
                    key = key.replace(KK, "_K*")
                s = nanrep.setdefault(key, [0, 0, 0])
                s[0] += 1; s[1] += nn; s[2] += ni
    n += 1
print("npz のキーの集合の種類", len(keysets), "キーの数", [len(k) for k in keysets])
print("NaN/inf を含む npz の配列（ファイル数, NaN 数, inf 数）", nanrep)
ks0 = list(keysets)[0]
print("pred_ の配列の数", sum(k.startswith("pred_") for k in ks0))
# K ごとの pred_adj 版の名前がそろうか
for K in (5, 10, 25):
    print(K, "pred_adj", sum(k.startswith(f"pred_adj_K{K}_") for k in ks0), "pred_raw", sum(k.startswith(f"pred_raw_K{K}_") for k in ks0))
