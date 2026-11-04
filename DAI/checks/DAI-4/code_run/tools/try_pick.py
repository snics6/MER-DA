import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/checks/DAI-4/code_run/bin")
import numpy as np, pandas as pd
import dai4 as D4, dai1_select as SEL
SEL.GRIDS["cons"] = [("rho", D4.RHO_GRID), ("gamma", D4.GAMMA_GRID)]
rng = np.random.default_rng(1); bad = 0; ties = 0
for t in range(500):
    rows = [dict(rho=r, gamma=g, uar=50 + rng.normal(0, 0.4)) for r in D4.RHO_GRID for g in D4.GAMMA_GRID for _ in range(3)]
    g = pd.DataFrame(rows); delta = float(rng.choice([0.0, 0.2, 0.5, 1.0]))
    a, ua = D4.pick4(g, delta); b, ub = SEL.pick(g, "cons", delta, D4.DEFAULT)
    ties += int(ua)
    if (a["rho"], a["gamma"], ua) != (b["rho"], b["gamma"], ub): bad += 1
print("pick4 vs dai1_select.pick: 不一致", bad, "/ 500，同点の規則が働いた", ties)
