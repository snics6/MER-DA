"""gate-code 再検査（DAI-22 第 3 版）：試走 t1 の出力の形と，独立に作り直した値との差だけを出す．量の値は出さない．"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai14 as D14  # noqa: E402
import dai_solver as S  # noqa: E402

C = Path("/home/snics/MER-DA/DAI/checks/DAI-22/code_run_v3/t1")
R1 = Path("/home/snics/MER-DA/DAI/results/dai1/eval/units")
csvs = pd.concat([pd.read_csv(p) for p in sorted((C / "units").glob("*.csv"))])
print("単位の csv：行", len(csvs), "列", csvs.shape[1], "識別子の重複", int(csvs.duplicated(["T", "fold", "seed", "K", "readout"]).sum()))
r0 = csvs[csvs.readout == "R0"]
newc = [c for c in csvs.columns if c.startswith(("n_iter_", "stalled_tol", "n_pred_diff", "grad_tol")) or c in ("n_tie", "n_linear_diff_nontie_unclamped", "tol_resolved")]
print("足された列：", newc)
res = r0[r0.tol_resolved.astype(str) == "True"]
print("解き直した行：", len(res), res[["T", "fold", "seed", "K"] + [c for c in newc if c.startswith(("n_iter_", "stalled_tol", "n_pred_diff", "grad_tol"))]].to_dict("records"))
print("n_tie の最大", int(r0.n_tie.max()), "同点でも切られてもいない発話の違いの最大", int(r0.n_linear_diff_nontie_unclamped.max()),
      "agree_linear の最小が 1 か", bool(r0.agree_linear.min() == 1.0), "cat の差の最大", float(r0.cat_max_abs_diff.max()))

# J_none・J_prop の UAR と実験 1 の p2.csv の差（台本は差を残さないので作り直す）
dmax = 0.0
for _, row in r0.iterrows():
    a = np.load(C / "units" / f"{row['T']}_f{row.fold}_r{row.seed}.npz")
    K = int(row.K)
    nk = a[f"nk__K{K}"].astype(bool)
    y = a["y"].astype(int)
    p2 = pd.read_csv(R1 / f"{row['T']}_ALL_f{row.fold}_r{row.seed}_p2.csv")
    rn = p2[(p2.method == "teacher") & (p2.teacher == "none_AVT") & (p2.K == K)]
    rp = p2[(p2.method == "product") & (p2.teacher == "(1.0, 1.0, 1.0)") & (p2.K == K)]
    assert len(rn) == 1 and len(rp) == 1
    dmax = max(dmax, abs(100 * S.uar(y[nk], a[f"none__K{K}"].astype(int)[nk]) - float(rn.uar_pool.iloc[0])),
               abs(100 * S.uar(y[nk], a[f"tgt__K{K}"].astype(int)[nk]) - float(rp.uar_pool.iloc[0])))
print("J_none・J_prop の UAR と p2.csv の差の最大", dmax, "（", len(r0), "行 × 2 ）")
# p2.csv の none_AVT と (1.0,1.0,1.0) の行が，ほかの列（生徒など）で複数に分かれていないか
p2 = pd.read_csv(R1 / "AFEW_ALL_f0_r0_p2.csv")
print("p2.csv の列：", list(p2.columns))
print("p2.csv の method の種類：", sorted(p2.method.unique()))

# 要約：一致の行を npz から作り直した値との差，start_minus_none と R0_minus_start の恒等式
sdir = C / "summary"
asm = pd.read_csv(sdir / "assumptions.csv")
cond = pd.read_csv(sdir / "conditions.csv")
print("assumptions の kind ごとの行：", asm.kind.value_counts().to_dict())
print("conditions：行", len(cond), "重複", int(cond.duplicated(["T", "K", "subset", "quantity"]).sum()), "Inf", int(np.isinf(cond.select_dtypes(float)).sum().sum()),
      "quantity の種類", cond.quantity.nunique())
ag_d = 0.0
for T in ("AFEW", "IEMOCAP", "MELD", "MOSEI"):
    U = [np.load(p) for p in sorted((C / "units").glob(f"{T}_f*_r*.npz"))]
    for K in (5, 25):
        for ro in ("R0", "R1"):
            for other in ("tgt", "start"):
                v = []
                for a in U:
                    nk = a[f"nk__K{K}"].astype(bool)
                    o = a[f"tgt__K{K}"] if other == "tgt" else a["start"]
                    v.append(float((a[f"K{K}__{ro}"][nk] == o[nk]).mean()))
                x = asm[(asm.kind == f"agree_{ro}_{other}") & (asm["T"] == T) & (asm.K == K)]
                assert len(x) == 1
                ag_d = max(ag_d, abs(x.agree_mean.iloc[0] - np.mean(v)), abs(x.agree_min.iloc[0] - min(v)), abs(x.agree_max.iloc[0] - max(v)))
print("一致の行と作り直しの差の最大", ag_d)
idd = 0.0
for (T, K, sub), x in cond.groupby(["T", "K", "subset"]):
    e = x.set_index("quantity").effect
    idd = max(idd, abs(e["start_minus_none"] - (e["uar_start"] - e["uar_none"])), abs(e["R0_minus_start"] - (e["uar_imitation_R0"] - e["uar_start"])))
print("start_minus_none・R0_minus_start の恒等式の差の最大", idd)
ivl = cond[cond.quantity.isin(["start_minus_none", "R0_minus_start"])]
print("区間の順序が崩れた行", int(((ivl.lo > ivl.effect) | (ivl.effect > ivl.hi)).sum()), "NaN の lo/hi", int(ivl[["lo", "hi"]].isna().sum().sum()))
cv = json.load(open(sdir / "convergence.json"))
print("convergence：", {k: cv[k] for k in ("n_rows", "n_units", "grad_max", "n_stalled", "inner_grad_max", "inner_stalled_max", "r0_saved_n_over_tol",
                                           "r0_n_resolved", "r0_resolved_abs_diff_max", "cat_ok", "grad_ok", "inner_grad_ok")})
a = np.load(C / "units" / "MOSEI_f0_r1.npz")
print("npz の鍵の数", len(a.files))
