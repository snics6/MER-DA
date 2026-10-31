"""E77 関門 1：共訓練（cotrain_rec）の 1 巡ごとの所要と，学習器の反復の回数・勾配の最大（AV，AFEW，反復 0）。"""
import sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/snics/MER-DA"); sys.path.insert(0, str(ROOT / "exp")); sys.path.insert(0, str(ROOT / "measure"))
torch.cuda.set_per_process_memory_fraction(0.3)
import compare_dai as CD, baselines_distill as BD, compare_b1 as CB, committee as CM, map as M, meta5 as M5, logreg
from data import groups_for
from geom_e0 import settings
from multimodal_distill import WORLDS, blocks, make_students, sub_group
_fb = logreg.fit_batch
t0 = time.perf_counter()
def fb(*a, **k):
    t = time.perf_counter(); out = _fb(*a, **k)
    it = [list(f.n_iter.values())[0] for f in out]; gr = [list(f.grad.values())[0] for f in out]
    print(f"  fit_batch {len(out)} 問：{time.perf_counter() - t:.1f} 秒，反復の最大 {max(it)}，中央値 {int(np.median(it))}，勾配の最大 {max(gr):.1e}（累計 {time.perf_counter() - t0:.0f} 秒）", flush=True)
    return out
logreg.fit_batch = fb
world, T, r = sys.argv[1], sys.argv[2], 0
groups = groups_for(world)
data_f = {g: M.load(g, world) for g in groups}
data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(world)}
S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
sets = {k: settings(k, T) for k in WORLDS[world] + [world]}
print({k: sets[k][3:5] for k in sets})
info = {k: M5.pool_info(data_m[k] if k != world else data_f, S, CD.C_FIT, {}) for k in sets}
ev = data_f[T]; ctx = CB.Ctx(ev, [r], list(CD.KS_LAB), center="ind")
pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
stu = make_students(world, T, data_f, data_m, S, pool, te, sets, info)
mods = WORLDS[world]
prs = {k: CD.Prob(stu[k], ev.y[pool], BD.label_sets_of(ctx, r, pool)) for k in mods + [world]}
rec = CD.Rec({"target": T, "rep": r}, ev.y[te], keep=False)
print(f"準備 {time.perf_counter() - t0:.0f} 秒", flush=True)
CD.cotrain_rec({m: prs[m] for m in mods}, prs[world], rec, world)
print(f"共訓練 終わり {time.perf_counter() - t0:.0f} 秒", flush=True)
