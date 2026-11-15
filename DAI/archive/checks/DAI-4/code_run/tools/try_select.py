import sys, json, shutil
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/checks/DAI-4/code_run/bin")
import pandas as pd, numpy as np
import dai4 as D4, dai1
fz1 = json.load(open(D4.FROZEN1))
src = pd.read_csv("/home/snics/MER-DA/DAI/checks/DAI-4/code_run/out/g1/pseudo/units/MELD_AFEW_f2_r1_p4.csv", low_memory=False)
dest = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run/out/fake_sel"); shutil.rmtree(dest, ignore_errors=True)
(ud := dest / "pseudo" / "units").mkdir(parents=True)
T = "MELD"
def write(drop_g1: bool, jitter: bool):
    shutil.rmtree(ud, ignore_errors=True); ud.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for n in fz1[T]["units"]:
        c, f, r = n.split("_"); d = src.copy()
        if drop_g1: d = d[~(np.isclose(d.gamma.astype(float), 1.0) & d.method.isin(["cons", "cons_self", "cons_own", "cons_conf", "cons_rand", "_weights"]))]
        d["c"], d["fold"], d["seed"] = c, int(f[1:]), int(r[1:])
        d["commit"] = "abc"; d["dirty"] = False
        if jitter: d["uar"] = d["uar"] + rng.normal(0, 0.3, len(d))
        d.to_csv(ud / f"{T}_{n}_p4.csv", index=False)
dai1.git_commit = lambda: ("abc", False)
for drop, label in ((False, "γ=1 の行が混じる"), (True, "γ=1 を除く")):
    write(drop, jitter=True)
    try:
        D4.select_stage(dest, [T], fz1)
        print(label, "-> 通った")
    except SystemExit as e:
        print(label, "-> SystemExit:", e)
fz = json.load(open(dest / "select" / "frozen.json"))
print("keys:", list(fz[T].keys()), "n_cells", fz[T]["n_cells"], "edge", fz[T]["grid_edge_cells"], "tie", fz[T]["tie_rule_used"])
print("sample cell A|5:", fz[T]["select"]["A|5"], " K=0:", fz[T]["select"]["A|0"])
# eval 出力があれば止まるか
(dest / "eval" / "units").mkdir(parents=True); (dest / "eval" / "units" / "x").write_text("")
try: D4.select_stage(dest, [T], fz1); print("eval 出力あり -> 通った")
except SystemExit as e: print("eval 出力あり -> SystemExit:", e)
# 30 でない単位
shutil.rmtree(dest / "eval")
import os; os.remove(ud / f"{T}_AFEW_f0_r0_p4.csv")
try: D4.select_stage(dest, [T], fz1); print("29 単位 -> 通った")
except SystemExit as e: print("29 単位 -> SystemExit:", e)
