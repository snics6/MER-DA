"""gate-code（DAI-5）：最大の候補（c = MOSEI）の 1 単位で solve_unit の GPU メモリの最大と所要を測る。保存しない。成績は出さない。"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1, dai5, dai_data as DD
T, c, f, r = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
fz1 = json.load(open(dai5.FROZEN1))[T]
tether = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in fz1["tether"].items()}
z1 = np.load(ROOT / "results/dai1/pseudo/units" / f"{T}_{c}_f{f}_r{r}_p1.npz")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
u = dai1.Unit(dai5.IN_DIR, T, c, [s for s in DD.others(T) if s != c], f, r)
store, rows = {}, []
dai5.solve_unit(u, tether, fz1, store, rows, z1, None, False)
print("pool", u.n, "sec", round(time.time() - t0, 1), "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2),
      "reserved GB", round(torch.cuda.max_memory_reserved() / 1e9, 2), "npz MB est", round(sum(np.asarray(v).nbytes for v in store.values()) / 1e6, 1))
