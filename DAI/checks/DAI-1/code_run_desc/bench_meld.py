"""gate-code（DAI-1 code_v5）：MELD をテストにした区間 1 つ（話者の jackknife）の所要を，単位 10 と 50 で測る。
値は使わない（所要だけ）。仮のターゲットの AFEW_MOSEI の単位の判定を種の数だけ繰り返して 50 単位の形を作る。"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import dai1_stats as ST
from pathlib import Path

U = Path("/home/snics/MER-DA/results/dai1/pseudo/units")
lab = pd.read_csv("/home/snics/MER-DA/DAI/splits/MELD.csv", keep_default_na=False, dtype={"cluster": str}).set_index("sample_id")
names = np.array(sorted(set(lab.cluster)))
ge, _ = ST.g_eff_of("MELD")
cells = {}
for f in range(5):
    for r in (0, 1):
        z = np.load(U / f"AFEW_MELD_f{f}_r{r}_p1.npz")
        sid = z["sample_test"].astype(str)
        y = lab.loc[sid].label_4.to_numpy().astype(int); cl = np.searchsorted(names, lab.loc[sid].cluster.to_numpy())
        p = z["p1_A_start_pred"].astype(int)
        cells[(f, r)] = ST.counts(y, p, cl)
zero = {k: (n, np.zeros_like(c), t) for k, (n, c, t) in cells.items()}
for R in (2, 10):
    cA = {(f, r): cells[(f, r % 2)] for f in range(5) for r in range(R)}
    cB = {(f, r): zero[(f, r % 2)] for f in range(5) for r in range(R)}
    rng = np.random.default_rng(0)
    t0 = time.time()
    for _ in range(3):
        ST.effect_stats(cA, cB, ge, rng)
    print(f"単位 {5 * R}：区間 1 つ {((time.time() - t0) / 3):.3f} 秒", flush=True)
