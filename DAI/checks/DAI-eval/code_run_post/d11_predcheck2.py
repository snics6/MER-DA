"""評価の回の実験 2：保存した int8 の判定から UAR を作り直して csv の uar と照合する（線形・CORAL・DANN/MME/mapctl・xMUDA 型）。"""
import json, numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
ROOT = Path("/home/snics/MER-DA"); U = ROOT / "results/dai2/eval/units"
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]; ST = ["A", "V", "T", "AVT"]; KS = [0, 1, 5, 10, 25]
sp = {c: pd.read_csv(ROOT / f"DAI/splits/{c}.csv", keep_default_na=False).set_index("sample_id") for c in C}
def U_(y, p):
    ok = [k for k in range(4) if (y == k).any()]; return 100 * np.mean([(p[y == k] == k).mean() for k in ok])
def match(q, it, skip=("method", "chunk", "chunk_pos", "chunk_size", "seed_init", "roles", "rep")):
    for k, v in it.items():
        if k in skip: continue
        if k not in q.columns: raise KeyError(k)
        col = q[k]
        q = q[col.astype(str) == v] if isinstance(v, str) else q[np.isclose(pd.to_numeric(col, errors="coerce").astype(float), float(v))]
    return q
def one(args):
    T, f, r = args; u = f"{T}_ALL_f{f}_r{r}"; n = 0; bad = []; cov = {}; amb = {}
    def chk(tag, pred, q, y):
        nonlocal n
        n += 1; cov[tag] = cov.get(tag, 0) + 1
        if len(q) == 0: bad.append((u, tag, "norow")); return
        v = U_(y, pred.astype(int))
        if not np.isclose(v, q.uar.to_numpy(float), atol=1e-6).all(): bad.append((u, tag, len(q)))
        if len(q) > 1: amb[tag] = amb.get(tag, 0) + 1
    # lin
    z = np.load(U / f"{u}_lin.npz"); d = pd.read_csv(U / f"{u}_lin.csv", low_memory=False)
    y = sp[T].loc[z["sample_test"].astype(str), "label_4"].to_numpy()
    for s in ST:
        for K in KS:
            b = d[(d.student == s) & (d.K == K)]
            q = b[b.method == "none"]; pn = z[f"none_{s}_K{K}_pred"]; chk("lin:none", pn[0] if pn.ndim == 2 else pn, q, y)
            for tag, kk in (("t3a", "t3a"), ("t3a_b", "t3a_pred_b")):
                pr = z[f"t3a_{s}_K{K}_pred" if tag == "t3a" else f"t3a_{s}_K{K}_pred_b"]
                q0 = b[b.method == tag].sort_values("M", key=lambda c: c.replace(-1, 1e9))
                for j in range(len(q0)): chk("lin:" + tag, pr[j], q0.iloc[[j]], y)
            chk("lin:saerens", z[f"sae_{s}_K{K}_pred"], b[b.method == "saerens"], y)
            for pref in ("lin", "nc", "self"):
                kk = f"{pref}_{s}_K{K}_items"
                if kk not in z.files: continue
                pr = z[f"{pref}_{s}_K{K}_pred"]
                for j, it in enumerate(json.loads(str(x)) for x in z[kk]):
                    chk("lin:" + it["method"], pr[j], match(b[b.method == it["method"]], it), y)
    for s in ST:
        kk = f"st_{s}_items"; pr = z[f"st_{s}_pred"]; b = d[(d.student == s) & (d.method == "st")]
        for j, it in enumerate(json.loads(str(x)) for x in z[kk]): chk("lin:st", pr[j], match(b, it), y)
    # coral
    z = np.load(U / f"{u}_coral.npz"); d = pd.read_csv(U / f"{u}_coral.csv", low_memory=False)
    for s in ST:
        its = [json.loads(str(x)) for x in z[f"coral_{s}_items"]]; pr = z[f"coral_{s}_pred"]; b = d[(d.student == s)]
        for j, it in enumerate(its):
            m = it["method"]; q = b[b.method == m]
            q = match(q, {k: v for k, v in it.items() if k != "r"} | ({"r": it["r"]} if "r" in q.columns else {}))
            chk("coral:" + m, pr[j], q, y)
    # map / xmuda
    for kind, pre in ((("map", "map"), ("map", "maprep"), ("xmuda", "xm"), ("xmuda", "xmrep")) if r < 3 else ()):
        z = np.load(U / f"{u}_{kind}.npz"); d = pd.read_csv(U / f"{u}_{kind}.csv", low_memory=False)
        for s in (["A", "V", "T"] if kind == "xmuda" else ST):
            items = [json.loads(str(x)) for x in (z[f"{pre}_items"] if pre.startswith("xm") else z[f"{pre}_{s}_items"])]
            pr = z[f"{pre}_{s}_pred"]; b = d[d.student == s]
            for j, it in enumerate(items):
                q = b[b.method == it["method"]] if "method" in it else b[b.method.isin(["xmuda", "xmuda_rep"])]
                if pre.endswith("rep"):
                    q = b[b.method == (it.get("method", "xmuda") + "_rep")]
                    q = q[q.rep.astype(str) == it["rep"]]
                    q = q[q.roles.astype(str) == "|".join(it["roles"])]
                    q = match(q, it, skip=("method", "chunk", "chunk_pos", "chunk_size", "seed_init", "roles", "rep"))
                else:
                    q = match(q, it)
                chk(f"{kind}:{pre}:" + it.get("method", "xmuda"), pr[j], q, y)
    return u, n, bad, cov, amb
if __name__ == "__main__":
    jobs = [(T, f, r) for T in C for f in range(5) for r in range(10)]
    with Pool(8) as pool: res = pool.map(one, jobs, chunksize=2)
    tot = sum(r[1] for r in res); bad = [b for r in res for b in r[2]]
    cov = {}; amb = {}
    for r in res:
        for k, v in r[3].items(): cov[k] = cov.get(k, 0) + v
        for k, v in r[4].items(): amb[k] = amb.get(k, 0) + v
    print("checked", tot, "mismatch", len(bad), bad[:10]); print("coverage", cov); print("multi-row matches", amb)
