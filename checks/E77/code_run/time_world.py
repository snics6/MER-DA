"""E77 関門 1：世界の 1 反復の所要を部品ごとに測る（AV，AFEW，反復 0．出力は保存しない）。"""
import sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/snics/MER-DA"); sys.path.insert(0, str(ROOT / "exp")); sys.path.insert(0, str(ROOT / "measure"))
torch.cuda.set_per_process_memory_fraction(float(sys.argv[3]) if len(sys.argv) > 3 else 0.3)
import compare_dai as CD, baselines_distill as BD, compare_b1 as CB, committee as CM, map as M, meta5 as M5
from data import groups_for
from geom_e0 import settings
from multimodal_distill import WORLDS, blocks, make_students, sub_group
world, T, r = sys.argv[1], sys.argv[2], 0
t0 = time.perf_counter()
def wrap(name, fn):
    def g(*a, **k):
        t = time.perf_counter(); out = fn(*a, **k); torch.cuda.synchronize()
        print(f"{name} {a[2] if name in ('run_ab', 'run_c') else ''}：{time.perf_counter() - t:.0f} 秒（累計 {time.perf_counter() - t0:.0f}）　GPU 最大 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        return out
    return g
for nm in ("run_ab", "run_c", "cotrain_rec", "coreg_rec"):
    setattr(CD, nm, wrap(nm, getattr(CD, nm)))
_ab_inner = {}
groups = groups_for(world)
data_f = {g: M.load(g, world) for g in groups}
data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(world)}
S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
sets = {k: settings(k, T) for k in WORLDS[world] + [world]}
info = {k: M5.pool_info(data_m[k] if k != world else data_f, S, CD.C_FIT, {}) for k in sets}
ev = data_f[T]; ctx = CB.Ctx(ev, [r], list(CD.KS_LAB), center="ind")
pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
print("候補", len(pool), "テスト", len(te), f"準備 {time.perf_counter() - t0:.0f} 秒", flush=True)
stu = make_students(world, T, data_f, data_m, S, pool, te, sets, info)
print(f"生徒 {time.perf_counter() - t0:.0f} 秒", flush=True)
rec = CD.Rec({"target": T, "rep": r}, ev.y[te], keep=True)
CD.world_unit(world, stu, ev.y[pool], ev.y[te], BD.label_sets_of(ctx, r, pool), rec)
print("行", len(rec.rows), f"計 {time.perf_counter() - t0:.0f} 秒")
