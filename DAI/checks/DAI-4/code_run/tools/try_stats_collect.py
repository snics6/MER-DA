import sys, json
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/checks/DAI-4/code_run/bin")
import numpy as np, pandas as pd
import dai4_stats as D, dai4 as D4, dai1, dai_data as DD
import dai1_stats as ST
out4, T, c, f, r = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
fz1 = json.load(open(D4.FROZEN1))
y_of, cl_of = ST.label_maps(c)
sc = {(s, K): (1.0, 0.25) for s in DD.STUDENTS for K in dai1.KS_ALL}
cells, mism, total = D.collect(out4, D4.IN_DIR, "pseudo", T, c, [(f, r)], y_of, cl_of, sc, fz1[T], extras=True, check_rows=True)
print("collect OK. mismatch/total:", mism, total, "cells:", len(cells))
names = sorted({k[2] for k in cells}); print([n for n in names if not n.startswith("g|")], "n g|:", sum(n.startswith("g|") for n in names))
# cons vs joint on this single unit
for s in DD.STUDENTS:
    for K in (0, 5, 25):
        a, b = cells[(s, K, "cons")][(f, r)], cells[(s, K, "joint")][(f, r)]
        assert (a[0] == b[0]).all() and (a[2] == b[2]).all()   # 塊の名前と感情ごとの件数
print("cluster names and test counts identical between cons and joint: OK")
# g1 point in cells (selftest) vs joint(rho=1,a=1) predictions
import collections
up = ST.UnitPreds(Path(D4.IN_DIR) / "pseudo/units" / f"{T}_{c}_f{f}_r{r}")
p4 = D4.P4(out4 / "pseudo/units" / f"{T}_{c}_f{f}_r{r}")
agree = []
for s in DD.STUDENTS:
    for K in dai1.KS_ALL:
        a = p4.pred(s, K, 1.0, 1.0); b = up.p3item(s, K, method="joint", teacher="one", rho=1.0, a=1)
        agree.append(float((a == b).mean()))
print("gamma=1 (rho=1) vs joint(one, rho=1, a=1): min agree over 20 cells", min(agree))
# effect_stats on one unit: does it run?
rng = np.random.default_rng(0)
g_eff, _ = ST.g_eff_of(T)
try:
    e = ST.effect_stats(cells[("A", 5, "cons")], cells[("A", 5, "joint")], g_eff, rng)
    print("effect_stats ran: n_units", e["n_units"], "method", e["method"])
except Exception as ex:
    print("effect_stats on 1 unit failed:", type(ex).__name__, ex)
