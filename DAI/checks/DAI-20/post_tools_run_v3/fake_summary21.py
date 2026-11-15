"""md_tables_dai21.py の検算用に，dai21_eval5.stage_summary・table_block と同じ列・版の名前・差の対の作り物の集計を作る．
版の名前と差の対の一覧は，stage_summary の names と con の式を，台本の定数（EXTRA_T・EX1・A2・B_VERS・P20.ABL）から同じ形で組み立てる．
値は作り物：版 v の UAR は val(v, c, K)（版ごとに違う乱数），差は val(a) − val(b) ＋ 種の集合ごとのずれ．
CREMA-D の K = 10 の (stud_tcal, stud_prod) の判定を同じにした（same_pred の印の試し）．"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai21_eval5 as D  # noqa: E402

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
C4, CD, KS = list(D.C4), D.CD, D.KS
ST_EX = tuple(f"stud_{t}" for t in D.EXTRA_T)
EX1 = D.EX1
fk = "AV"
names4 = (["stud_prop", "stud_prod", "stud_prop_sel", "stud_prod_sel"] + list(ST_EX) + [f"{m}_{t}" for m in EX1 for t in ("prod", "prop")]
          + ["prop", "prod", "none_AVT", "joint_AVT"] + list(D.B_VERS))
con4 = ([("stud_prop", b) for b in ("stud_prod", "joint_AVT", "none_AVT", "prop", "prod") + ST_EX + D.B_VERS]
        + [("stud_prop_sel", "stud_prod_sel"), ("stud_prod", "joint_AVT"), ("joint_AVT", "none_AVT"), ("prop", "joint_AVT")]
        + [(s, "stud_prod") for s in ST_EX] + [(b, "joint_AVT") for b in D.B_VERS]
        + [("stud_prop", f"{m}_prop") for m in EX1] + [(f"{m}_prop", f"b_{m}") for m in EX1] + [(f"{m}_prod", f"b_{m}") for m in EX1]
        + [(f"{m}_prop", f"{m}_prod") for m in EX1])
namesC = (list(D.A2) + ["M_F6_orc", "M_F6_ext"] + [f"{v}@raw" for v in D.A2] + ["prop", "prop@raw", "prop_noprof", "prop_noprof_sel", "prop_noint", "prop_n0zero", "prop_home",
          "prop_levelfree", "stud_prod", "stud_prop", "stud_prop_sel"] + list(ST_EX) + [f"none_{k}" for k in ("A", "V", fk)] + ["ret_A", "ret_V"]
          + [f"full_{k}" for k in ("A", "V", fk)] + [f"start_{k}" for k in ("A", "V", fk)])
nf = f"none_{fk}"
conC = ([("prop", b) for b in D.A2 + D.P20.ABL + (nf, "stud_prod", "M_F6_ext")] + [(a, "prod") for a in D.A2[1:] + ("prop_noprof", "prop_levelfree", "M_F6_orc", "M_F6_ext")]
        + [("prop_noint", "prop_noprof"), ("prop", "prop@raw")] + [("prop@raw", f"{b}@raw") for b in D.A2] + [("prop", f"{b}@raw") for b in D.A2[1:]]
        + [("stud_prop", b) for b in ("stud_prod", nf, "prop", "prod") + ST_EX] + [("stud_prop_sel", "stud_prod"), ("stud_prod", nf), ("stud_prod", "prod")]
        + [(s, "stud_prod") for s in ST_EX])


def val(v, c, K):
    h = int(hashlib.sha256(f"{v}|{c}|{K}".encode()).hexdigest()[:8], 16)
    return 30 + 40 * h / 2 ** 32


SAMEP = {(CD, 10, "stud_tcal"): "stud_prod"}
SHIFT = {"all": 0.0, "0-1": 0.123, "2-9": -0.071}
lv, ef = [], []
for part, cs, names, con in (("4corpora", C4, names4, con4), ("cremad", [CD], namesC, conC)):
    for c in cs:
        for K in KS:
            for v in names:
                sp = "stud_prod" if (c, K, v) in SAMEP else ("stud_prod" if (v == "stud_prod" and (c, K, "stud_tcal") in SAMEP) else "")
                vv = val(SAMEP.get((c, K, v), v), c, K)
                for sn in D.SEED_SETS:
                    lv.append(dict(part=part, c=c, K=K, version=v, role=D.role_of(v), seeds=sn, uar=vv + SHIFT[sn], lo=vv - 2, hi=vv + 2, se=1.0, df=13, method="jk+seed", n_fr=50,
                                   same_as="", same_pred=sp))
            for a, b in con:
                for sn in D.SEED_SETS:
                    same = SAMEP.get((c, K, a)) == b or SAMEP.get((c, K, b)) == a
                    est = 0.0 if same else val(SAMEP.get((c, K, a), a), c, K) - val(SAMEP.get((c, K, b), b), c, K) + SHIFT[sn]
                    lo, hi = est - 0.37, est + 0.41
                    ef.append(dict(part=part, a=a, b=b, c=c, K=K, seeds=sn, role_a=D.role_of(a), role_b=D.role_of(b), a_same_as="", b_same_as="", same_pred=bool(same),
                                   est=est, lo=(0.0 if same else lo), hi=(0.0 if same else hi), se=0.2, df=13, method="jk+seed", n_fr=50, n_folds=5, folds_dropped="",
                                   v_cluster=0.03, v_seed=0.01, v_seed_all=0.01,
                                   mark=("↑" if lo > 0 and est >= 1 else "↓" if hi < 0 and est <= -1 else "") if not same else ""))
pd.DataFrame(lv).to_csv(OUT / "levels.csv", index=False)
pd.DataFrame(ef).to_csv(OUT / "effects.csv", index=False)
# 設定の格子（flatten した列の名前は stage_summary 734 行と同じ）
rows = []
for c in C4 + [CD]:
    for K in KS:
        for v in ("stud_prop", "stud_prod"):
            for g in range(D.NS):
                rows.append({"c": c, "K": K, "version": v, "grid": g, "rho": 1.0, "a": 1.0, "uar_mean": 40 + g + (0.5 if v == "stud_prop" else 0), "uar_count": 50,
                             "mimic_mean": 0.8, "mimic_count": 50, "teacher_uar_mean": 50.0, "teacher_uar_count": 50, "part": "cremad" if c == CD else "4corpora"})
pd.DataFrame(rows).to_csv(OUT / "student_grid.csv", index=False)
json.dump(dict(students4=dict(n_units=200, grad_max=1e-6, stalled=0, joint_repro_min=0.999, joint_repro_mean=0.9995, joint_repro_ok=True, n_joint_repro_below=0, agree_min=1.0,
                              kd_repro_min=0.998, agree_repro_min=0.997, mix_repro_min=0.996, n_kd_repro_below=0, n_agree_repro_below=0, n_mix_repro_below=0, commits=["x"]),
               cremad=dict(n_units=50, grad_max=2e-6, stalled=0, adj_resid_max=1e-8, wf_n_capped=0, commits=["x"]), summary_commit="x"), open(OUT / "convergence.json", "w"), indent=1)
print("作り物の集計", OUT, len(lv), len(ef))
