"""段 1 の全部の問と，段 3 の照合の外の行（rolesep，単一の生徒の cotrain，coreg，joint_scratch）の UAR を int8 の判定から作り直して rows.csv と照合する。"""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/home/snics/MER-DA"); P = ROOT / "DAI/checks/DAI-1/code_run_post"; U = ROOT / "results/dai1/pseudo/units"
p1 = pd.read_pickle(P / "rows_p1.pkl"); p3 = pd.read_pickle(P / "rows_p3.pkl")
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]; ST = ["A", "V", "T", "AVT"]; RHO = [0.0625, 0.25, 1.0, 4.0]; A = [1, 4, 16]
CR = [0.0, 0.0625, 0.25, 1.0, 4.0]; CG = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]
sp = {c: pd.read_csv(ROOT / f"DAI/splits/{c}.csv", keep_default_na=False).set_index("sample_id") for c in C}
def U_(y, p):
    ok = [k for k in range(4) if (y == k).any()]; return 100 * np.mean([(p[y == k] == k).mean() for k in ok])
n = 0; bad = []; npred = {}
for T in C:
    for c in C:
        if c == T: continue
        for f in range(5):
            for r in (0, 1):
                u = f"{T}_{c}_f{f}_r{r}"
                z1 = np.load(U / f"{u}_p1.npz"); z3 = np.load(U / f"{u}_p3.npz")
                y = sp[c].loc[z1["sample_test"].astype(str), "label_4"].to_numpy()
                a1 = p1[p1.unit == u]; a3 = p3[(p3.unit == u) & (p3.stage == 3)]
                for kind in ["A", "V", "T", "AV", "AT", "VT", "AVT"]:
                    rr = a1[a1.kind == kind]
                    for nm, meth in (("start", "start"), ("start_srcstd", "start_srcstd")):
                        n += 1; v = rr[rr.method == meth].uar.iloc[0]
                        if not np.isclose(U_(y, z1[f"p1_{kind}_{nm}_pred"].astype(int)), v, atol=1e-6): bad.append((u, kind, meth))
                    items = [str(x) for x in z1[f"p1_{kind}_items"]]; pr = z1[f"p1_{kind}_pred"]
                    for j, it in enumerate(items):
                        K, cw, cb = it.split("|"); meth = "oracle" if K == "all" else "none"; Kv = -1 if K == "all" else int(K)
                        q = rr[(rr.method == meth) & (rr.K == Kv) & np.isclose(rr.Cw, float(cw)) & np.isclose(rr.Cb, float(cb))]
                        n += 1
                        if len(q) != 1 or not np.isclose(U_(y, pr[j].astype(int)), q.uar.iloc[0], atol=1e-6): bad.append((u, kind, it))
                    if kind in ST:
                        for key, meth, col in (("kbias", "kbias", "Cb"), ("tgt", "tgtonly", "C")):
                            its = [str(x) for x in z1[f"p1_{kind}_{key}_items"]]; pr = z1[f"p1_{kind}_{key}_pred"]
                            for j, it in enumerate(its):
                                K, v = it.split("|"); q = rr[(rr.method == meth) & (rr.K == int(K)) & np.isclose(rr[col].astype(float), float(v))]
                                n += 1
                                if len(q) != 1 or not np.isclose(U_(y, pr[j].astype(int)), q.uar.iloc[0], atol=1e-6): bad.append((u, kind, meth, it))
                for K in (0, 1, 5, 10, 25):
                    for s in ST:
                        b = a3[(a3.student == s) & (a3.K == K)]
                        # rolesep
                        rs = b[b.method == "rolesep"]
                        key = "pred_a" if K == 0 else "pred_b"; pr = z3[f"p3r_{s}_K{K}_{key}"]
                        for _, row in rs.iterrows():
                            ja = [(tv, x) for tv in ("sel", "one") for x in RHO].index((row.teacher, float(row.rho)))
                            j = ja if K == 0 else ja * 3 + A.index(int(float(row.a)))
                            n += 1
                            if not np.isclose(U_(y, pr[j].astype(int)), row.uar, atol=1e-6): bad.append((u, s, K, "rolesep"))
                        # joint_scratch
                        its = [json.loads(x) for x in z3[f"p3z_{s}_K{K}_items"]]; pr = z3[f"p3z_{s}_K{K}_pred"]
                        js = b[b.method == "joint_scratch"]
                        for j, it in enumerate(its):
                            q = js[(js.teacher == it["teacher"]) & np.isclose(js.C.astype(float), it["C"])]
                            n += 1
                            if len(q) != 1 or not np.isclose(U_(y, pr[j].astype(int)), q.uar.iloc[0], atol=1e-6): bad.append((u, s, K, "scratch"))
                        # coreg
                        pr = z3[f"p3g_K{K}_{s}_pred"]; cg = b[b.method == "coreg"]
                        for j, rho in enumerate(CR):
                            q = cg[np.isclose(cg.rho.astype(float), rho)]; n += 1
                            if len(q) != 1 or not np.isclose(U_(y, pr[j].astype(int)), q.uar.iloc[0], atol=1e-6): bad.append((u, s, K, "coreg"))
                        if s != "AVT":
                            pr = z3[f"p3c_K{K}_{s}_pred"]; ct = b[b.method == "cotrain"]
                            for j, rho in enumerate(RHO):
                                q = ct[np.isclose(ct.rho.astype(float), rho)]; n += 1
                                if len(q) != 1 or not np.isclose(U_(y, pr[j].astype(int)), q.uar.iloc[0], atol=1e-6): bad.append((u, s, K, "cotrain"))
print("checked", n, "mismatch", len(bad), bad[:10])
