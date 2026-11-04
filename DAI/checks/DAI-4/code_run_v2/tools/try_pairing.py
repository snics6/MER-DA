import sys, json, os
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd
import dai1, dai4 as D4, dai4_stats as D, dai_data as DD
CR = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run_v2")
fz1 = json.load(open(D4.FROZEN1)); fz1["AFEW"]["n_seeds_eval"] = 1
DD.CORPORA = ["AFEW"]; DD.N_FOLDS = 1
out4 = CR / "out_e"
def run(label, ind):
    try:
        D.pairing(out4, ind, fz1); print(f"[{label}] 通った", pd.read_csv(out4 / "stats/pairing.csv").to_dict("records"))
    except SystemExit as e:
        print(f"[{label}] SystemExit: {str(e)[:100]}", pd.read_csv(out4 / "stats/pairing.csv").to_dict("records"))
run("本物の入力", D4.IN_DIR)
ind = CR / "in_perm"; u = ind / "eval/units"; u.mkdir(parents=True, exist_ok=True)
src = D4.IN_DIR / "eval/units"
for w in (1, 2):
    for ext in ("npz", "csv"):
        p = u / f"AFEW_ALL_f0_r0_p{w}.{ext}"
        if not p.exists(): os.symlink(src / f"AFEW_ALL_f0_r0_p{w}.{ext}", p)
p = u / "AFEW_ALL_f0_r0_p3.csv"
if not p.exists(): os.symlink(src / "AFEW_ALL_f0_r0_p3.csv", p)
z = dict(np.load(src / "AFEW_ALL_f0_r0_p3.npz")); n = z["p3_A_K5_pred"].shape[1]
perm = np.random.default_rng(0).permutation(n)
for k in list(z):
    if k.startswith("p3_AVT_K5_pred"): z[k] = z[k][:, perm]          # 1 つの (生徒, K) だけ並びを入れ替える
np.savez_compressed(u / "AFEW_ALL_f0_r0_p3.npz", **z)
run("p3 の AVT・K=5 の判定の並びを入れ替え", ind)
