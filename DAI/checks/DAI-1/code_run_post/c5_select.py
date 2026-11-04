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
summary = {}
for T in C:
    F = fz[T]
    q1 = p1[p1["T"] == T]; q3 = p3[p3["T"] == T]
    dsel = {}
    for s in ST:
        for K in [1, 5, 10, 25]:
            cw, cb = F["tether"][f"{s}|{K}"]
            g = q1[(q1.method == "none") & (q1.kind == s) & (q1.K == K) & np.isclose(q1.Cw, cw) & np.isclose(q1.Cb, cb)]
            assert len(g) == 30
            v = g.groupby(["c", "fold"]).uar.var(ddof=1)
            assert len(v) == 15
            dsel[(s, K)] = 2 * np.sqrt(v.mean() / 30)
        dsel[(s, 0)] = dsel[(s, 1)]
    for k, v in dsel.items():
        if not np.isclose(v, F["delta_sel"][f"{k[0]}|{k[1]}"], rtol=1e-12, atol=1e-12): diffs.append((T, "dsel", k))
    sel = {}; tie_f = tie_x = 0; edge = {}; ncand = []
    for meth in ["joint", "rolesep", "kd", "agree", "mix", "cotrain", "coreg", "joint_scratch"]:
        for s in ST:
            for K in KS:
                g0 = q3[(q3.stage == 3) & (q3.method == meth) & (q3.student == s) & (q3.K == K)]
                for ver, g in g0.groupby("teacher"):
                    ch, pu, tie, ed, nc = choose(g, meth, dsel[(s, K)])
                    sel[f"{s}|{K}|{meth}|{ver}"] = (ch, pu)
                    ncand.append(dict(T=T, s=s, K=K, meth=meth, ver=ver, ncand=nc))
                    tie_f += tie if meth in FAM else 0; tie_x += 0 if meth in FAM else tie
                    edge.setdefault(meth, [0, 0]); edge[meth][0] += ed; edge[meth][1] += 1
                    if meth == "joint":
                        ch1, pu1, tie1, ed1, _ = choose(g[g.a.astype(float) == 1.0], "joint", dsel[(s, K)])
                        sel[f"{s}|{K}|joint_a1|{ver}"] = (ch1, pu1); tie_f += tie1
                        # joint_a1 の a は数えない：rho（4 点）だけで端
                        e1 = gi(RHO, ch1["rho"]) in (0, 3)
                        edge.setdefault("joint_a1", [0, 0]); edge["joint_a1"][0] += e1; edge["joint_a1"][1] += 1
    for meth in ["tgtonly", "kbias"]:
        for s in ST:
            for K in [1, 5, 10, 25]:
                g = q1[(q1.method == meth) & (q1.kind == s) & (q1.K == K)]
                ch, pu, tie, ed, _ = choose(g, meth, dsel[(s, K)])
                sel[f"{s}|{K}|{meth}|-"] = (ch, pu); tie_x += tie
                if meth == "tgtonly":
                    edge.setdefault(meth, [0, 0]); edge[meth][0] += ed; edge[meth][1] += 1
    # 照合
    if set(sel) != set(F["select"]): diffs.append((T, "select keys", len(set(sel) ^ set(F["select"]))))
    for k, (ch, pu) in sel.items():
        got = F["select"].get(k)
        if got is None: continue
        if not np.isclose(pu, got["pseudo_uar"], atol=1e-9) or any(not np.isclose(float(got[kk]), float(vv)) for kk, vv in ch.items()):
            diffs.append((T, k, ch, pu, got))
    if (tie_f, tie_x) != (F["tie_rule_used_family"], F["tie_rule_used_extra"]): diffs.append((T, "ties", tie_f, tie_x, F["tie_rule_used_family"], F["tie_rule_used_extra"]))
    if {k: list(v) for k, v in edge.items()} != F["grid_edge_cells"]: diffs.append((T, "edge", edge, F["grid_edge_cells"]))
    # 主の版
    for K in KS:
        ms = np.mean([sel[f"{s}|{K}|joint|sel"][1] for s in ST]); mo = np.mean([sel[f"{s}|{K}|joint|one"][1] for s in ST])
        dm = np.mean([dsel[(s, K)] for s in ST])
        w = tuple(F["w"][str(K)])
        mv = "sel" if (w != (1.0, 1.0, 1.0) and ms - mo > dm) else "one"
        if mv != F["main"][str(K)]: diffs.append((T, "main", K, mv, F["main"][str(K)]))
        summary.setdefault(T, {})[K] = (mv, round(ms - mo - dm, 3))
        for s in ST:
            cands = []
            for pm in ["kd", "agree", "mix", "cotrain", "coreg"]:
                vers = {"cotrain": ["ct"], "coreg": ["cr"]}.get(pm, ["sel", "one"])
                if len(vers) == 2:
                    a, b = sel[f"{s}|{K}|{pm}|sel"][1], sel[f"{s}|{K}|{pm}|one"][1]
                    v = "sel" if a > b else "one"
                else:
                    v = vers[0]
                cands.append((sel[f"{s}|{K}|{pm}|{v}"][1], pm, v))
            top = max(x[0] for x in cands)
            near = [x for x in cands if x[0] >= top - dsel[(s, K)]]
            ch = near[0]   # 並び順が kd → agree → mix → cotrain → coreg
            got = F["partner"][f"{s}|{K}"]
            if (ch[1], ch[2]) != (got["method"], got["version"]) or not np.isclose(ch[0], got["pseudo_uar"]): diffs.append((T, "partner", s, K, ch, got))
    # 指数の安定さ
    q2 = p2[(p2["T"] == T) & (p2.method == "product")]
    for K in KS:
        hits = 0
        for f in range(5):
            g = q2[(q2.K == K) & (q2.fold != f)]
            m = g.groupby("teacher").uar_pool.mean(); best = m.max()
            c = [tuple(float(x) for x in w.strip("()").split(",")) for w in m.index[m >= best - 1e-9]]
            wf = min(c, key=lambda w: (sum(abs(x - 1) for x in w), abs(max(w) - 1), w))
            hits += wf == tuple(F["w"][str(K)])
        if hits != F["w_lofo_agree_of5"][str(K)]: diffs.append((T, "lofo", K, hits))
print("grid count problems:", counts_bad[:5], len(counts_bad))
print("differences vs frozen:", len(diffs)); print(diffs[:20])
print("main version and margin (ms-mo-δmain):", summary)
pd.DataFrame(ncand).to_csv(P / "select_ncand.csv", index=False)
