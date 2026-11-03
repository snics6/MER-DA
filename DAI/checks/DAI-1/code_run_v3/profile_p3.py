"""段 3 の部品ごとの所要（1 単位．UAR は出さない）。引数：out T c f r [部品…]（cotrain coreg rest）"""
import sys, json, time
import numpy as np, torch
import dai1, dai_data as DD
from pathlib import Path
S = dai1.S
out, T, c, f, r = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
parts = sys.argv[6:] or ["coreg"]
Ks = [int(x) for x in __import__("os").environ.get("KS", ",".join(map(str, dai1.KS_ALL))).split(",")]
srcs = [s for s in DD.others(T) if s != c]
t0 = time.time()
u = dai1.Unit(Path(out), T, c, srcs, f, r)
tw = json.load(open(f"{out}/pseudo/{T}_tether_w.json"))
teth = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in tw["tether"].items()}
base = f"{out}/pseudo/units/{T}_{c}_f{f}_r{r}"
p1, p2 = np.load(base + "_p1.npz"), np.load(base + "_p2.npz")
TK = dai1.build_tk(u, p1, p2, teth)
print(f"準備 {time.time()-t0:.1f} 秒 n={u.n}", flush=True)
for K in Ks:
    if "cotrain" in parts:
        t1 = time.time(); st, rows = {}, []
        dai1.cotrain(u, TK[K], K, teth, st, rows)
        print(f"K={K} cotrain {time.time()-t1:.1f} 秒", flush=True)
    if "coreg" in parts:
        for rho in dai1.RHO_GRID:
            keep = dai1.RHO_GRID
            dai1.RHO_GRID = (rho,)
            t1 = time.time(); st, rows = {}, []
            dai1.coreg(u, K, teth, st, rows)
            dai1.RHO_GRID = keep
            cv = [x for x in rows if x["method"] == "_conv_coreg"][0]
            print(f"K={K} ρ={rho} coreg {time.time()-t1:.1f} 秒 反復 {cv['n_iter']} 勾配の最大 {cv['grad_max']:.1e}", flush=True)
