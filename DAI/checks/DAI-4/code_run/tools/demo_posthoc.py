import sys, json
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd
import dai1, dai_data as DD, dai1_stats as ST
out = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run/out/g1/pseudo/units/MELD_AFEW_f2_r1")
z4 = np.load(f"{out}_p4.npz"); T, c = "MELD", "AFEW"
up = ST.UnitPreds(Path("/home/snics/MER-DA/results/dai1/pseudo/units/MELD_AFEW_f2_r1"))
cp = dai1.Corpus.get(c); df = cp.df.set_index(cp.df["sample_id"].astype(str))
ids = z4["sample_test"].astype(str); y = df.loc[ids, "label_4"].to_numpy().astype(int)
man = pd.read_csv("/home/snics/MER-DA/data/metadata/afew_utterance_manifest.csv").set_index("sample_id")
gender = man.loc[ids, "gender"].to_numpy()
items = [json.loads(x) for x in z4["p4_A_K5_items"]]
j = [i for i, it in enumerate(items) if it["rho"] == 1.0 and it["gamma"] == 0.25][0]
pc = z4["p4_A_K5_pred"][j]; pj = up.p3item("A", 5, method="joint", teacher="one", rho=1.0, a=1)
def uar(yy, pp): return 100*np.mean([(pp[yy == k] == k).mean() for k in range(4) if (yy == k).any()])
print("(a) の実演（値は仮のターゲットの単位 1 つ．要約には使わない）")
for g in ("Male", "Female"):
    m = gender == g; print(" 性別", g, "n", int(m.sum()), "cons", round(uar(y[m], pc[m]), 2), "joint", round(uar(y[m], pj[m]), 2))
print(" 感情ごとの再現率 cons", [round(100*(pc[y == k] == k).mean(), 1) for k in range(4)], "joint", [round(100*(pj[y == k] == k).mean(), 1) for k in range(4)])
from collections import Counter
print(" 混同行列の行（cons）:", [np.bincount(pc[y == k], minlength=4).tolist() for k in range(4)])
