"""評価の回の実験 1：保存した int8 の判定から UAR を作り直し，段 1・3 の csv の uar と照合する（c9_predcheck.py の評価の回版．
c9 が照合の外に置いた行に加え，段 3 の main の格子（p3_*），引き戻しの曲線（p3t_*）も照合する）。"""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
ROOT = Path("/home/snics/MER-DA"); U = ROOT / "results/dai1/eval/units"
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]; ST = ["A", "V", "T", "AVT"]; KS = [0, 1, 5, 10, 25]
RHO = [0.0625, 0.25, 1.0, 4.0]; A = [1, 4, 16]; CR = [0.0, 0.0625, 0.25, 1.0, 4.0]
sp = {c: pd.read_csv(ROOT / f"DAI/splits/{c}.csv", keep_default_na=False).set_index("sample_id") for c in C}
def U_(y, p):
    ok = [k for k in range(4) if (y == k).any()]; return 100 * np.mean([(p[y == k] == k).mean() for k in ok])
def one(args):
    T, f, r = args; u = f"{T}_ALL_f{f}_r{r}"; n = 0; bad = []; cov = {}
    z1 = np.load(U / f"{u}_p1.npz"); z3 = np.load(U / f"{u}_p3.npz")
    d1 = pd.read_csv(U / f"{u}_p1.csv", low_memory=False); d3 = pd.read_csv(U / f"{u}_p3.csv", low_memory=False)
    y = sp[T].loc[z1["sample_test"].astype(str), "label_4"].to_numpy()
    assert (z3["tether"] is not None)
    def chk(tag, yhat_pred, q):
        nonlocal n
        n += 1; cov[tag] = cov.get(tag, 0) + 1
        if len(q) != 1 or not np.isclose(U_(y, yhat_pred.astype(int)), q.uar.iloc[0], atol=1e-6): bad.append((u, tag, len(q)))
    for kind in ["A", "V", "T", "AV", "AT", "VT", "AVT"]:
        rr = d1[d1.kind == kind]
        for nm in ("start", "start_srcstd"):
            chk("p1_" + nm, z1[f"p1_{kind}_{nm}_pred"], rr[rr.method == nm])
        items = [str(x) for x in z1[f"p1_{kind}_items"]]; pr = z1[f"p1_{kind}_pred"]
        for j, it in enumerate(items):
            K, cw, cb = it.split("|"); meth = "oracle" if K == "all" else "none"; Kv = -1 if K == "all" else int(K)
            q = rr[(rr.method == meth) & (rr.K == Kv) & np.isclose(rr.Cw, float(cw)) & np.isclose(rr.Cb, float(cb))]
            chk("p1_" + meth, pr[j], q)
        if kind in ST:
            for key, meth, col in (("kbias", "kbias", "Cb"), ("tgt", "tgtonly", "C")):
                its = [str(x) for x in z1[f"p1_{kind}_{key}_items"]]; pr = z1[f"p1_{kind}_{key}_pred"]
                for j, it in enumerate(its):
                    K, v = it.split("|"); q = rr[(rr.method == meth) & (rr.K == int(K)) & np.isclose(rr[col].astype(float), float(v))]
                    chk("p1_" + meth, pr[j], q)
    for s in ST:
        for K in KS:
            b = d3[(d3.student == s) & (d3.K == K)]
            # main grid and tether
            for pref, ptag in (("p3", "p3main"), ("p3t", "p3tether"), ("p3z", "p3scratch")):
                its = [json.loads(str(x)) for x in z3[f"{pref}_{s}_K{K}_items"]]; pr = z3[f"{pref}_{s}_K{K}_pred"]
                for j, it in enumerate(its):
                    q = b[b.method == it["method"]]
                    for k, v in it.items():
                        if k == "method": continue
                        col = q[k]
                        if isinstance(v, str): q = q[col.astype(str) == v]
                        else: q = q[np.isclose(pd.to_numeric(col, errors="coerce").astype(float), float(v))]
                    chk(ptag + ":" + it["method"], pr[j], q)
            rs = b[b.method == "rolesep"]
            key = "pred_a" if K == 0 else "pred_b"; pr = z3[f"p3r_{s}_K{K}_{key}"]
            for _, row in rs.iterrows():
                ja = [(tv, x) for tv in ("sel", "one") for x in RHO].index((row.teacher, float(row.rho)))
                j = ja if K == 0 else ja * 3 + A.index(int(float(row.a)))
                n += 1; cov["rolesep"] = cov.get("rolesep", 0) + 1
                if not np.isclose(U_(y, pr[j].astype(int)), row.uar, atol=1e-6): bad.append((u, s, K, "rolesep"))
            pr = z3[f"p3g_K{K}_{s}_pred"]; cg = b[b.method == "coreg"]
            for j, rho in enumerate(CR):
                chk("coreg", pr[j], cg[np.isclose(cg.rho.astype(float), rho)])
            if s != "AVT":
                pr = z3[f"p3c_K{K}_{s}_pred"]; ct = b[b.method == "cotrain"]
                for j, rho in enumerate(RHO):
                    chk("cotrain", pr[j], ct[np.isclose(ct.rho.astype(float), rho)])
    return u, n, bad, cov
if __name__ == "__main__":
    jobs = [(T, f, r) for T in C for f in range(5) for r in range(10)]
    with Pool(8) as pool: res = pool.map(one, jobs, chunksize=2)
    tot = sum(r[1] for r in res); bad = [b for r in res for b in r[2]]
    cov = {}
    for r in res:
        for k, v in r[3].items(): cov[k] = cov.get(k, 0) + v
    print("checked", tot, "mismatch", len(bad), bad[:10]); print("coverage by tag:", cov)
