import sys, json
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd
import dai1, dai1_stats as ST
base = Path("/home/snics/MER-DA/results/dai1/pseudo/units/MELD_AFEW_f2_r1")
up = ST.UnitPreds(base)
print("p3 npz に sample_test / sample_K があるか:", [k for k in up.p3.files if k.startswith("sample")])
rows = pd.read_csv(f"{base}_p3.csv", low_memory=False)
y_of, _ = ST.label_maps("AFEW"); y = y_of(up.sid)
items = [json.loads(x) for x in up.p3["p3_A_K5_items"]]
j = [i for i, it in enumerate(items) if it["method"] == "joint" and it["teacher"] == "one" and it["rho"] == 1.0 and float(it["a"]) == 1.0][0]
pred = up.p3["p3_A_K5_pred"][j]
u_mine = 100*np.mean([(pred[y == k] == k).mean() for k in range(4) if (y == k).any()])
r = rows[(rows.stage == 3) & (rows.student == "A") & (rows.K == 5) & (rows.method == "joint") & (rows.teacher == "one") & np.isclose(rows.rho, 1.0) & (rows.a.astype(float) == 1.0)]
print("joint の p3 判定（p1 の test 順で解釈）から作った UAR と p3 の行の UAR の差:", abs(u_mine - float(r.uar.iloc[0])))
perm = np.random.default_rng(0).permutation(len(y))
u_bad = 100*np.mean([(pred[y[perm] == k] == k).mean() for k in range(4) if (y == k).any()])
print("test の順を入れ替えて解釈すると差:", abs(u_bad - float(r.uar.iloc[0])))
