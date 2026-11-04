"""896136c：共正則化の ρ = 0 の列が比較の基準（K≥1）と出発点（K=0）に一致するか，統計の台本の coreg の取り出し，選ぶ段の格子と端の数え上げ（UAR は出さない）。"""
import sys, json
import numpy as np, pandas as pd
import dai1, dai1_stats as ST, dai1_select as SEL
X = sys.argv[1]; base = f"{X}/pseudo/units/MOSEI_AFEW_f0_r0"
z1, z3 = np.load(base + "_p1.npz"), np.load(base + "_p3.npz")
tw = json.load(open(f"{X}/pseudo/MOSEI_tether_w.json"))
print("COREG_RHO", dai1.COREG_RHO, "選ぶ段の格子", SEL.GRIDS["coreg"])
for K in dai1.KS_ALL:
    for m in "AVT":
        pr0 = z3[f"p3g_K{K}_{m}_pred"][0]
        if K == 0:
            ref = z1[f"p1_{m}_start_pred"]
        else:
            cw, cb = tw["tether"][f"{m}|{K}"]
            items = [str(x) for x in z1[f"p1_{m}_items"]]
            ref = z1[f"p1_{m}_pred"][items.index(f"{K}|{cw}|{cb}")]
        print(f"K={K} {m}: ρ=0 の判定と比較の基準の食い違い {int((pr0 != ref).sum())}/{len(ref)}")
d = pd.read_csv(base + "_p3.csv", low_memory=False)
cv = d[d.method == "_conv_coreg"]; cr = d[d.method == "coreg"]
print("coreg の行", len(cr), "ρ の値", sorted(cr.rho.unique()), "grad の最大", float(cr.grad.max()), "stalled", int((cr.stalled.astype(str) == "True").sum()),
      "finite 偽", int((cr.finite.astype(str) == "False").sum()), "S の最小", float(cr.S.min()))
print(cv[["K", "grad_max", "n_iter", "n_stalled", "finite"]].to_string(index=False))
up = ST.UnitPreds(__import__("pathlib").Path(base))
ok = all((up.coreg(s, K, rho) == z3[f"p3g_K{K}_{s}_pred"][j]).all() for s in ("A", "V", "T", "AVT") for K in dai1.KS_ALL for j, rho in enumerate(dai1.COREG_RHO))
print("UnitPreds.coreg が COREG_RHO の並びで取り出す", ok)
# 格子の端の数え上げ（作り物の選択）
sel = {"A|5|joint_a1|one": {"rho": 0.25, "a": 1}, "A|5|joint|one": {"rho": 0.25, "a": 16}, "A|5|kbias|-": {"Cb": 10.0},
       "A|5|coreg|cr": {"rho": 0.0}, "V|5|coreg|cr": {"rho": 0.25}}
edge = {}
for key, ch in sel.items():
    meth = key.split("|")[2]; gm = "joint" if meth == "joint_a1" else meth
    keys = [(k, v) for k, v in SEL.GRIDS[gm] if len(v) > 2 and k in ch and not (meth == "joint_a1" and k == "a")]
    if not keys: continue
    hit = any(float(ch[k]) in (float(min(v)), float(max(v))) for k, v in keys)
    edge.setdefault(meth, [0, 0]); edge[meth][0] += int(hit); edge[meth][1] += 1
print("端の数え上げ（作り物．本番の格子）", edge)
