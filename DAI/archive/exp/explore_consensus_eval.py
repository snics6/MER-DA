# 探索：教師（3 モダリティの積）の正しさが，生徒以外の 2 モダリティの判定の一致で分かれるか（KD の欠点の診断．学習はしない）
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1_stats as ST
from pathlib import Path
ROOT = Path("/home/snics/MER-DA"); U = ROOT/"results/dai1/eval/units"
fz = json.load(open(ROOT/"DAI/frozen/dai1_frozen.json"))
rows = []
for T in ("AFEW", "IEMOCAP", "MELD", "MOSEI"):
    y_of, _ = ST.label_maps(T)
    for f in range(5):
        for r in range(10):
            b = U/f"{T}_ALL_f{f}_r{r}"
            if not Path(f"{b}_p3.csv").exists(): continue
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
                        rows.append(dict(T=T, K=K, student=s, group=name, share=n/nk.sum(), teacher_acc=tr[g].mean(), student_acc=sr[g].mean(),
                                         fix=(tr & ~sr)[g].mean(), harm=(~tr & sr)[g].mean(), n=int(n)))
D = pd.DataFrame(rows)
C = D.groupby(["T", "K", "student", "group"]).mean(numeric_only=True).reset_index()
C.to_csv("results/explore_consensus/consensus_cells.csv", index=False)
pd.set_option("display.width", 250)
x = C[C.K == 5].copy()
for c in ("share", "teacher_acc", "student_acc", "fix", "harm"): x[c] = (x[c]*100).round(1)
print(x.pivot_table(index=["T", "student"], columns="group", values=["share", "teacher_acc", "student_acc", "fix", "harm"]).to_string())
