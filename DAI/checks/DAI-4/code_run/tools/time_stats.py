import sys, json, time
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np
import dai4 as D4, dai4_stats as D, dai1, dai_data as DD, dai1_stats as ST
CR = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run"); fz1 = json.load(open(D4.FROZEN1))
y_of, cl_of = ST.label_maps("MOSEI")
sel = {(s, K): (1.0, 0.25) for s in DD.STUDENTS for K in dai1.KS_ALL}
cells, *_ = D.collect(CR / "out2/a", D4.IN_DIR, "pseudo", "IEMOCAP", "MOSEI", [(4, 1)], y_of, cl_of, sel, fz1["IEMOCAP"])
A, B = cells[("A", 5, "cons")][(4, 1)], cells[("A", 5, "joint")][(4, 1)]
rng = np.random.default_rng(0)
cA = {(f, r): A for f in range(5) for r in range(10)}; cB = {(f, r): B for f in range(5) for r in range(10)}
for g_eff in (50.0, 6.0):
    t = time.time(); e = ST.effect_stats(cA, cB, g_eff, rng); print("g_eff", g_eff, "n clusters", len(A[0]), "method", e["method"], "秒", round(time.time() - t, 2))
