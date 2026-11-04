"""選ぶ段（設計 8 節）を dai1_select.py を使わずに作り直し，frozen.json と照合する。"""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/home/snics/MER-DA"); P = ROOT / "DAI/checks/DAI-1/code_run_post"
p1 = pd.read_pickle(P / "rows_p1.pkl"); p2 = pd.read_pickle(P / "rows_p2.pkl"); p3 = pd.read_pickle(P / "rows_p3.pkl")
fz = json.load(open(ROOT / "results/dai1/select/frozen.json"))
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]; ST = ["A", "V", "T", "AVT"]; KS = [0, 1, 5, 10, 25]
RHO = [0.0625, 0.25, 1.0, 4.0]; A = [1, 4, 16]; CG = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]
GR = {"joint": [("rho", RHO), ("a", A)], "rolesep": [("rho", RHO), ("a", A)], "kd": [("temp", [1, 2, 4]), ("rho", RHO)],
      "agree": [("tau", [0.5, 0.7, 0.9]), ("f", [0.25, 1, 4])], "mix": [("alpha", [0.2, 0.5, 0.8]), ("f", [0.25, 1, 4])],
      "cotrain": [("rho", RHO)], "coreg": [("rho", [0.0, 0.0625, 0.25, 1.0, 4.0])], "joint_scratch": [("C", CG)],
      "tgtonly": [("C", CG)], "kbias": [("Cb", [1.0, 10.0])]}
DF = {"joint": dict(rho=1.0, a=1), "rolesep": dict(rho=1.0, a=1), "kd": dict(temp=2, rho=1.0), "agree": dict(tau=0.9, f=1),
      "mix": dict(alpha=0.2, f=1), "cotrain": dict(rho=1.0), "coreg": dict(rho=1.0), "joint_scratch": dict(C=1.0),
      "tgtonly": dict(C=1.0), "kbias": dict(Cb=10.0)}
FAM = {"joint", "rolesep", "kd", "agree", "mix", "cotrain", "coreg"}
diffs = []; counts_bad = []
def gi(grid, v):
    return [i for i, x in enumerate(grid) if np.isclose(x, v)][0]
def choose(g, meth, delta, units_expected=30):
    keys = [k for k, _ in GR[meth]]; grids = dict(GR[meth])
    g = g.copy()
    for k in keys: g[k] = g[k].astype(float)
    cnt = g.groupby(keys).size()
    full = int(np.prod([len(v) for _, v in GR[meth]]))
    if not ((cnt == units_expected).all() and len(cnt) == full):
        counts_bad.append((meth, len(cnt), cnt.min(), cnt.max()))
    m = g.groupby(keys).uar.mean()
    best = m.max()
    cands = [ix if isinstance(ix, tuple) else (ix,) for ix in m.index[m >= best - delta]]
    d0 = [gi(grids[k], float(DF[meth][k])) for k in keys]
    def key(ix):
        ii = [gi(grids[k], v) for k, v in zip(keys, ix)]
        return (sum(abs(a - b) for a, b in zip(ii, d0)), ii)
    ch = min(cands, key=key)
    am = m.idxmax(); am = am if isinstance(am, tuple) else (am,)
    edge = any(gi(grids[k], v) in (0, len(grids[k]) - 1) for k, v in zip(keys, ch) if len(grids[k]) > 2)
    return dict(zip(keys, ch)), float(m.loc[ch if len(ch) > 1 else ch[0]]), tuple(ch) != tuple(am), edge, len(cands)
