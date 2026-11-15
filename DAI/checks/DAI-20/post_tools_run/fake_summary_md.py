"""md_tables_dai20.py の検算用に，dai20_eval.stage_summary と同じ列と行の作り物の集計を作る（値は作り物）．
版 v の UAR を val(v, c, K) とし，差は val(a) − val(b) にする．表に出た数がこの式と合うかを確かめる．"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai20_eval as E  # noqa: E402

OUT = Path("/home/snics/MER-DA/DAI/checks/DAI-20/post_tools_run/fake_summary")
OUT.mkdir(parents=True, exist_ok=True)
CORP, KS = E.CORP, E.KS
pick_names = ("prop", "prop_noprof", "prop_noprof_sel", "prop_n0zero", "prop_loto", "oracle_grid", "prop_levelfree")
vers = (list(E.A_VERS) + list(E.REF_VERS) + list(E.ORC_VERS) + [f"{v}@raw" for v in E.A_VERS] + list(pick_names) + ["prop@raw", "prop_noint", "prop_home"]
        + list(E.PLAIN))
names = sorted(set(vers))


def val(v, c, K):
    h = sum(ord(ch) * (i + 1) for i, ch in enumerate(v)) % 997
    return 30 + h / 50 + CORP.index(c) * 3 + K / 10


SAME = {"prod@raw": "prod", "psel": "prod"}        # AFEW では psel は積と同じ数（作り物）
lv = []
for c in CORP:
    for K in KS:
        for v in names:
            vv = val(SAME.get(v, v), c, K) if (v in SAME and c == "AFEW") or v == "prod@raw" else val(v, c, K)
            sa = ""
            if v == "prod" or v == "prod@raw" or (v == "psel" and c == "AFEW"):
                sa = "prod"
            for sn in ("all", "0-2"):
                lv.append(dict(c=c, K=K, version=v, role=E.role_of(v), seeds=sn, uar=vv + (0 if sn == "all" else 0.5), lo=vv - 1, hi=vv + 1, se=0.5, df=13,
                               method="jk+seed", n_fr=50 if sn == "all" else 15, grid=-1, same_as=sa))
pd.DataFrame(lv).to_csv(OUT / "levels.csv", index=False)
lvd = {(r["c"], r["K"], r["version"]): r["uar"] for r in lv if r["seeds"] == "all"}
con = ([("prop", b) for b in E.A_VERS + E.ABL + ("none_AVT", "joint_AVT") + E.B_VERS + E.REF_VERS]
       + [(a, "prod") for a in E.A_VERS[1:] + ("prop_noprof", "prop_noprof_sel", "prop_loto", "oracle_grid") + E.ORC_VERS + E.REF_VERS]
       + [("prop_loto", "prop"), ("oracle_grid", "prop"), ("prop", "prop@raw"), ("prop_levelfree", "prod"), ("prop_noint", "prop_noprof")]
       + [("prop@raw", f"{b}@raw") for b in E.A_VERS] + [("prop", f"{b}@raw") for b in E.A_VERS[1:]])
ef = []
for a, b in con:
    for c in CORP:
        for K in KS:
            same = (c == "AFEW" and {a, b} <= {"prod", "psel", "prod@raw"})
            for sn in ("all", "0-1", "2-9"):
                est = lvd[(c, K, a)] - lvd[(c, K, b)] + {"all": 0.0, "0-1": 0.11, "2-9": -0.07}[sn]
                ef.append(dict(a=a, b=b, c=c, K=K, seeds=sn, role_a=E.role_of(a), role_b=E.role_of(b), same_counts=same, a_same_as="", b_same_as="",
                               est=0.0 if same else est, lo=est - 0.4, hi=est + 0.4, se=0.2, df=13, method="jk+seed", n_fr=50, n_folds=5, folds_dropped="",
                               v_cluster=0.03, v_seed=0.01, v_seed_all=0.01, mark=("↑" if est - 0.4 > 0 and est >= 1 else "↓" if est + 0.4 < 0 and est <= -1 else "")))
pd.DataFrame(ef).to_csv(OUT / "effects.csv", index=False)
oth = []
for c in CORP:
    for K in KS:
        for row, v in E.B_ROWS.items():
            oth.append(dict(group="B", c=c, K=K, method=row, version=v, uar=lvd[(c, K, v)], lo=0, hi=0, se=0, n_units=50, seeds="0-9", uar_here=lvd[(c, K, v)], source="x"))
        for m, n in (("own", 50), ("dann", 15), ("coral", 50)):
            oth.append(dict(group="C", c=c, K=K, method=m, version="", uar=40 + len(m) + K / 100, lo=np.nan, hi=np.nan, se=np.nan, n_units=n, seeds="0-2" if n == 15 else "0-9",
                            gradient=m == "dann", appendix=False, point="p", descriptive_version=False, exp2_group="g", exp1prop_minus_method=1.5 + len(m) / 10,
                            exp1prop_minus_method_lo=0.5, exp1prop_minus_method_hi=2.5, uar_exp1prop_same_units=45.0, source="y"))
pd.DataFrame(oth).to_csv(OUT / "others.csv", index=False)
gr = []
for c in CORP:
    for K in KS:
        for i in range(E.NG):
            gr.append(dict(c=c, K=K, grid=i, n0=E.GRID[i][0], beta=E.GRID[i][1], prior=E.GRID[i][2], uar_adj=50 + i / 10, uar_adj_others=49, frozen=i == 6, loto=i == 2, oracle=i == 14))
pd.DataFrame(gr).to_csv(OUT / "grid_levels.csv", index=False)
json.dump(dict(n_units=200, teacher_dev_max=1e-7, grad_max=1e-6, stalled=0, adj_resid_max=1e-8, recheck_agree_min=1.0, eq_agree_min=1.0, psel_agree_min=1.0,
               others_B_dev_max=0.0), open(OUT / "convergence.json", "w"))
print("作り物の集計", OUT, len(lv), len(ef), len(oth), len(gr))
