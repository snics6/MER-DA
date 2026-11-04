"""効果の見込み（設計 9.2 節）を rows.csv の UAR と自前の選び直しで作り直し，frozen の expected_effect と照合する。"""
import numpy as np, pandas as pd, json
from c8_lib import *   # p1, p2, p3, fz, choose, GR, DF, C, ST, KS, RHO
def reselect(T, keepc):
    F = fz[T]; q1 = p1[(p1["T"] == T) & p1.c.isin(keepc)]; q3 = p3[(p3["T"] == T) & p3.c.isin(keepc)]
    n = 10 * len(keepc); dsel = {}
    for s in ST:
        for K in [1, 5, 10, 25]:
            cw, cb = F["tether"][f"{s}|{K}"]
            g = q1[(q1.method == "none") & (q1.kind == s) & (q1.K == K) & np.isclose(q1.Cw, cw) & np.isclose(q1.Cb, cb)]
            dsel[(s, K)] = 2 * np.sqrt(g.groupby(["c", "fold"]).uar.var(ddof=1).mean() / n)
        dsel[(s, 0)] = dsel[(s, 1)]
    sel = {}
    for meth in ["joint", "rolesep", "kd", "agree", "mix", "cotrain", "coreg"]:
        for s in ST:
            for K in (0, 5, 25):
                g0 = q3[(q3.method == meth) & (q3.student == s) & (q3.K == K)]
                for ver, g in g0.groupby("teacher"):
                    ch, pu, *_ = choose(g, meth, dsel[(s, K)], units_expected=n)
                    sel[(s, K, meth, ver)] = (ch, pu)
    main = {}; part = {}
    for K in (0, 5, 25):
        ms = np.mean([sel[(s, K, "joint", "sel")][1] for s in ST]); mo = np.mean([sel[(s, K, "joint", "one")][1] for s in ST])
        dm = np.mean([dsel[(s, K)] for s in ST])
        main[K] = "sel" if (tuple(F["w"][str(K)]) != (1.0, 1.0, 1.0) and ms - mo > dm) else "one"
        for s in ST:
            cands = []
            for pm in ["kd", "agree", "mix", "cotrain", "coreg"]:
                vers = {"cotrain": ["ct"], "coreg": ["cr"]}.get(pm, ["sel", "one"])
                v = vers[0] if len(vers) == 1 else ("sel" if sel[(s, K, pm, "sel")][1] > sel[(s, K, pm, "one")][1] else "one")
                cands.append((sel[(s, K, pm, v)][1], pm, v))
            top = max(x[0] for x in cands); part[(s, K)] = [x for x in cands if x[0] >= top - dsel[(s, K)]][0]
    return sel, main, part
def u_of(rows, **w):
    m = np.ones(len(rows), bool)
    for k, v in w.items():
        m &= (rows[k].astype(float).to_numpy() == float(v)) if not isinstance(v, str) else (rows[k].astype(str) == v).to_numpy()
    return rows[m]
bad = []; rep = {}
for X in C:
    acc = {}
    for c in [x for x in C if x != X]:
        sel, main, part = reselect(X, [x for x in C if x not in (X, c)])
        F = fz[X]; r1 = p1[(p1["T"] == X) & (p1.c == c)]; r3 = p3[(p3["T"] == X) & (p3.c == c) & (p3.stage == 3)]
        for s in ST:
            for K in (0, 5, 25):
                v = main[K]; j = sel[(s, K, "joint", v)][0]
                def ser(g):
                    g = g.sort_values(["fold", "seed"]); assert len(g) == 10, len(g); return g.uar.to_numpy(float)
                prop = ser(u_of(r3, method="joint", student=s, K=K, teacher=v, rho=j["rho"], a=j["a"]))
                dflt = ser(u_of(r3, method="joint", student=s, K=K, teacher="one", rho=1.0, a=1))
                if K == 0: none = ser(u_of(r1, method="start", kind=s, K=0))
                else:
                    cw, cb = F["tether"][f"{s}|{K}"]; none = ser(u_of(r1, method="none", kind=s, K=K, Cw=cw, Cb=cb))
                _, pm, pv = part[(s, K)]; pp = sel[(s, K, pm, pv)][0]
                partner = ser(u_of(r3, method=pm, student=s, K=K, teacher=pv, **pp))
                vals = {"prop": prop, "default": dflt, "none": none, "partner": partner}
                if K > 0:
                    rs = sel[(s, K, "rolesep", v)][0]; vals["rolesep"] = ser(u_of(r3, method="rolesep", student=s, K=K, teacher=v, **rs))
                for fam, (Aa, Bb, ks) in {1: ("prop", "none", (0, 5, 25)), 2: ("rolesep", "prop", (5, 25)), 3: ("prop", "partner", (0, 5, 25)), 4: ("prop", "default", (0, 5, 25))}.items():
                    if K in ks: acc.setdefault(f"{fam}|{s}|{K}", []).append(float(np.mean(vals[Aa] - vals[Bb])))
    for k, v in acc.items():
        got = fz[X]["detect"]["cells"][k]["expected_effect"]
        if not np.isclose(np.mean(v), got, atol=1e-6): bad.append((X, k, np.mean(v), got))
    rep[X] = len(acc)
print("cells checked", rep, "mismatches", len(bad), bad[:8])
