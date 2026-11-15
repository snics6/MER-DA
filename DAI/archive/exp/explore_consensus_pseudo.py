# 診断の仮のターゲットの回での再現：教師（3 モダリティの積）の正しさが，生徒以外の 2 モダリティの一致で分かれるか．
# 単位は (評価のターゲット T, 仮のターゲット c, 分割, 種)．候補とラベルは c のもの．評価のテストのラベルは使わない．
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1_stats as ST
from pathlib import Path
ROOT = Path("/home/snics/MER-DA"); U = ROOT/"results/dai1/pseudo/units"
C4 = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
rows = []
for T in C4:
    for c in C4:
        if c == T: continue
        y_of, _ = ST.label_maps(c)
        for f in range(5):
            for r in range(2):
                b = U/f"{T}_{c}_f{f}_r{r}"
                z1, z2 = np.load(f"{b}_p1.npz"), np.load(f"{b}_p2.npz")
                ids = z1["sample_pool"].astype(str); y = y_of(ids)
                for K in (5, 25):
                    nk = ~np.isin(ids, z1[f"sample_K{K}"].astype(str))
                    P = {m: z2[f"p2_K{K}_ret_{m}_pool"].astype(np.float64) for m in "AVT"}
                    L = sum(np.log(np.clip(P[m], 1e-12, None)) for m in "AVT"); tq = L.argmax(1)
                    am = {m: P[m].argmax(1) for m in "AVT"}
                    for s in ("A", "V", "T", "AVT"):
                        sp = z2[f"p2_K{K}_none_{s}_pool"].astype(np.float32).argmax(1)
                        others = [m for m in "AVT" if m != s] if s != "AVT" else list("AVT")
                        cons = np.all([am[others[0]] == am[o] for o in others[1:]], axis=0)
                        tr, sr = tq == y, sp == y
                        for name, g in (("一致", cons & nk), ("不一致", ~cons & nk)):
                            n = g.sum()
                            rows.append(dict(T=T, c=c, K=K, student=s, group=name, share=n/nk.sum(), teacher_acc=tr[g].mean(),
                                             student_acc=sr[g].mean(), fix=(tr & ~sr)[g].mean(), harm=(~tr & sr)[g].mean(), n=int(n)))
D = pd.DataFrame(rows)
D.to_csv("results/explore_consensus/consensus_pseudo_units.csv", index=False)
C = D.groupby(["c", "K", "student", "group"]).mean(numeric_only=True).reset_index()   # 仮のターゲット c ごと（T と分割・種で平均）
C.to_csv("results/explore_consensus/consensus_pseudo_cells.csv", index=False)
pd.set_option("display.width", 250)
x = C[C.K == 5].copy()
for col in ("share", "teacher_acc", "student_acc", "fix", "harm"): x[col] = (x[col]*100).round(1)
print("単位", D.groupby(["T","c","K","student"]).size().max(), "× (T,c) の組", D.groupby(["T","c"]).ngroups)
print(x.pivot_table(index=["c", "student"], columns="group", values=["share", "teacher_acc", "fix", "harm"]).to_string())
