import sys, json
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np
import dai4 as D4, dai4_stats as D, dai1, dai_data as DD, dai1_stats as ST
CR = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run"); fz1 = json.load(open(D4.FROZEN1))
T, c, f, r, out = "IEMOCAP", "MOSEI", 4, 1, CR / "out2/a"
y_of, cl_of = ST.label_maps(c)
sel = {(s, K): (1.0, 0.25) for s in DD.STUDENTS for K in dai1.KS_ALL}
sc = {m: sel for m in D4.CTRL_PREFIX}
cells, mism, total = D.collect(out, D4.IN_DIR, "pseudo", T, c, [(f, r)], y_of, cl_of, sel, fz1[T], extras=True, check_rows=True, sel_ctrl=sc, sel_rho25=sel)
names = sorted({k[2] for k in cells}); print("mismatch", mism, "/", total, " cell 名（g|・jg| 以外）:", [n for n in names if not n.startswith(("g|", "jg|"))])
print("g| の数", sum(n.startswith("g|") for n in names), " jg| の数", sum(n.startswith("jg|") for n in names))
# 設計 2 節の比べを全部作れるか
need = ["none", "kd_one", "partner", "agree", "cotrain", "coreg", "cons_g1", "cons0", "joint", "joint_default", "joint_a1", "cons_rho25"] + [x for m in D4.CTRL_PREFIX for x in (m, m + "@cons")]
miss = [(s, K, n) for s in DD.STUDENTS for K in dai1.KS_ALL for n in need if (s, K, n) not in cells and not (n.startswith("cons_self") and s == "AVT")]
print("欠けるセル:", miss[:5], len(miss))
# 同じ (s,K) の cons と各比べ相手で塊の名前・件数が同じか（effect_stats の前提）
bad = 0
for (s, K, n), v in cells.items():
    a = cells[(s, K, "cons")][(f, r)]; b = v[(f, r)]
    bad += int(not ((a[0] == b[0]).all() and (a[2] == b[2]).all()))
print("塊と件数の不一致", bad)
# 部分集合
import pandas as pd
sub = pd.read_csv("/home/snics/MER-DA/DAI/frozen/subsets.csv", keep_default_na=False)
print("IEMOCAP sample_id の例:", sub[sub.corpus == "IEMOCAP"].sample_id.iloc[:2].tolist())
