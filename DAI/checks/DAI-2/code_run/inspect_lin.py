"""gate-code の検算：lin の 1 単位の csv と npz の形・有限・収束・格子の数を数える（UAR の値は出さない）。
引数：csv のパス（npz は同じ名前）"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

p = Path(sys.argv[1])
d = pd.read_csv(p, low_memory=False)
z = np.load(str(p).replace(".csv", ".npz"))
print("行", len(d), "列", len(d.columns))
print("列:", list(d.columns))
print("\n手法ごとの行の数（生徒 × K）")
print(d.groupby(["method", "student"]).size().unstack(fill_value=0))
print("\nK ごとの行の数（主の手法）")
main = d[~d.method.str.startswith("_")]
print(main.groupby(["method", "K"]).size().unstack(fill_value=0))
print("\nuar の範囲が 0〜100 の中か:", bool(((main.uar >= 0) & (main.uar <= 100)).all()), "uar の NaN:", int(main.uar.isna().sum()))
for c in ("grad", "n_iter"):
    if c in d:
        g = d[c]
        print(c, "最大", float(g.max()), "NaN", int(g.isna().sum()))
print("grad > 1e-5 の行（主の手法）:", main[main.grad > 1e-5].groupby("method").size().to_dict())
print("stalled の行:", main[main.stalled.astype(str) == "True"].groupby("method").size().to_dict())
print("finite が偽:", int((d.finite.astype(str) == "False").sum()))
nn = d[d.method == "none"]
print("none_agree_dai1:", nn[["student", "K", "none_agree_dai1", "dai1_same_units"]].to_string(index=False))
lp = d[d.method == "_lp_diag"]
print("LP 共役勾配：回数の最大", int(lp.cg_iters.max()), "残差の最大", float(lp.cg_resid.max()), "疑似ラベルの本数の最小", int(lp.n_pseudo.min()))
sr = d[d.method == "_self_round"]
print("自己学習：残った本数 0 の巡の数", int((sr.n_pseudo == 0).sum()), "/", len(sr))
print("_conv_self grad_max の最大", float(d[d.method == "_conv_self"].grad_max.max()))
nc = d[d.method.isin(["entmin", "im"])]
print("entmin・IM：grad > 1e-5 の問の数", int((nc.grad > 1e-5).sum()), "/", len(nc), " n_iter の最大", int(nc.n_iter.max()))
st = d[d.method == "st"]
print("S+T：行", len(st), "C", sorted(st.C.unique()), "C_edge", st.groupby("student").C_edge.first().to_dict(), "beta_dummy の数", int(st.beta_dummy.sum()))
print("\nnpz の鍵の数", len(z.files))
bad = [k for k in z.files if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all()]
print("非有限を含む npz の配列:", bad)
const = [k for k in z.files if k.endswith("_pred") and z[k].ndim >= 1 and len(np.unique(z[k])) == 1]
print("判定が 1 種類だけの _pred:", const)
print("sample_test", z["sample_test"].shape, "sample_pool", z["sample_pool"].shape, "commit", str(z["commit"]), "dirty", str(z["dirty"]))
for k in ("self_A_K5_items", "lin_A_K5_items", "nc_A_K5_items"):
    print(k, len(z[k]), json.loads(str(z[k][0])))
print("self_A_K5_pl", z["self_A_K5_pl"].shape, z["self_A_K5_pl"].dtype, "lin_A_K5_W", z["lin_A_K5_W"].shape, "map 型の W·A は無し")
print("st_A_items", [json.loads(str(x)) for x in z["st_A_items"]])
print("sec", float(d.sec.iloc[0]))
