"""段 3 の行の数・NaN・収束・保存の並び（items と行と判定）を確かめる（UAR の値は出さない）。"""
import sys, glob, json
import numpy as np, pandas as pd
sys.path.insert(0, "DAI/exp")
import dai_data as DD, dai1
O = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "pseudo"
cache = {}
def uar(y, p):
    return 100 * np.mean([(p[y == k] == k).mean() for k in range(4) if (y == k).any()])
def match(df, it):
    m = np.ones(len(df), bool)
    for k, v in it.items():
        if k not in df: return None
        col = df[k]
        if isinstance(v, str): m &= (col.astype(str) == v).to_numpy()
        else: m &= np.isclose(pd.to_numeric(col, errors="coerce").to_numpy(dtype=float), float(v), equal_nan=False)
    return np.flatnonzero(m)
for f in sorted(glob.glob(f"{O}/{tag}/units/*_p3.csv")):
    df = pd.read_csv(f, low_memory=False); z = np.load(f.replace(".csv", ".npz"))
    z1 = np.load(f.replace("_p3.csv", "_p1.npz"))
    c = f.split("/")[-1].split("_")[1]
    if c not in cache:
        d, _ = DD.load_corpus(c); cache[c] = dict(zip(d.sample_id, d.label_4))
    y = np.array([cache[c][s] for s in z1["sample_test"]])
    d3 = df[(df.stage == 3) & df.student.notna() & ~df.method.astype(str).str.startswith("_")]
    cnt = d3.groupby(["student", "K", "method"]).size().unstack("method").fillna(0).astype(int)
    tot = mism = nomatch = multi = 0
    for s in DD.STUDENTS:
        for K in dai1.KS_ALL:
            ds = d3[(d3.student == s) & (d3.K == K)]
            for key in (f"p3_{s}_K{K}", f"p3t_{s}_K{K}", f"p3z_{s}_K{K}"):
                items = [json.loads(x) for x in z[f"{key}_items"]]; P = z[f"{key}_pred"]
                assert len(items) == len(P) == z[f"{key}_W"].shape[0], key
                for it, p in zip(items, P):
                    ix = match(ds, it)
                    if ix is None or len(ix) == 0: nomatch += 1; continue
                    if len(ix) > 1: multi += 1
                    tot += 1; mism += abs(uar(y, p) - ds.iloc[ix[0]].uar) > 1e-9
            # 役割を分けた版
            rs = ds[ds.method == "rolesep"]
            PA = z[f"p3r_{s}_K{K}_pred_a"]
            order = [(tv, r) for tv in ("sel", "one") for r in dai1.RHO_GRID]
            for ja, (tv, r) in enumerate(order):
                for ia, a in enumerate(dai1.A_GRID):
                    row = rs[(rs.teacher == tv) & np.isclose(rs.rho, r) & np.isclose(rs.a, a)]
                    assert len(row) == 1, (s, K, tv, r, a, len(row))
                    p = PA[ja] if K == 0 else z[f"p3r_{s}_K{K}_pred_b"][ja * 3 + ia]
                    tot += 1; mism += abs(uar(y, p) - row.uar.iloc[0]) > 1e-9
            if s != "AVT":
                ct = ds[ds.method == "cotrain"]
                for jr, r in enumerate(dai1.RHO_GRID):
                    row = ct[np.isclose(ct.rho, r)]
                    tot += 1; mism += abs(uar(y, z[f"p3c_K{K}_{s}_pred"][jr]) - row.uar.iloc[0]) > 1e-9
    g = d3[d3.grad.notna()]
    conv = df[df.method.astype(str).str.startswith("_conv")]
    print(f.split("/")[-1], "行", len(df), "uar の NaN", int(d3.uar.isna().sum()), "grad 最大", f"{g.grad.max():.2e}", "1e-5 超", int((g.grad > 1e-5).sum()),
          "_conv の grad_max 最大", f"{conv.grad_max.max():.2e}", "stalled", int(g.stalled.fillna(False).astype(bool).sum()),
          "照合", f"{mism}/{tot}", "items に行が無い", nomatch, "行が 2 つ以上", multi, "所要", round(df.sec.iloc[0]))
    print(cnt.to_string())
