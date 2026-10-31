"""E77 関門 1：世界の 1 反復を共正則化を除いて流し（10 分の制限のため），E61 の保存済みの値（反復 0）と突き合わせる。
共正則化の再現は check_coreg.py で別に測る。出力は checks/E77/code_run/repro_<world>_<T>.csv。"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path("/home/snics/MER-DA"); sys.path.insert(0, str(ROOT / "exp")); sys.path.insert(0, str(ROOT / "measure"))
torch.cuda.set_per_process_memory_fraction(0.3)
import compare_dai as CD, baselines_distill as BD, compare_b1 as CB, committee as CM, map as M, meta5 as M5
from data import groups_for
from geom_e0 import settings
from multimodal_distill import WORLDS, blocks, make_students, sub_group
world, T, r = sys.argv[1], sys.argv[2], 0
skip = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else {"coreg_rec"}
for nm in skip:
    setattr(CD, nm, lambda *a, **k: None)
t0 = time.perf_counter()
groups = groups_for(world)
data_f = {g: M.load(g, world) for g in groups}
data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(world)}
S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
sets = {k: settings(k, T) for k in WORLDS[world] + [world]}
info = {k: M5.pool_info(data_m[k] if k != world else data_f, S, CD.C_FIT, {}) for k in sets}
ev = data_f[T]; ctx = CB.Ctx(ev, [r], list(CD.KS_LAB), center="ind")
pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
stu = make_students(world, T, data_f, data_m, S, pool, te, sets, info)
rec = CD.Rec({"target": T, "rep": r}, ev.y[te], keep=True)
CD.world_unit(world, stu, ev.y[pool], ev.y[te], BD.label_sets_of(ctx, r, pool), rec)
a = pd.DataFrame(rec.rows)
out = ROOT / "checks/E77/code_run" / f"repro_{world}_{T}.csv"
a.to_csv(out, index=False)
np.save(str(out).replace(".csv", "_probs.npy"), np.stack(rec.P)); np.save(str(out).replace(".csv", "_te.npy"), te)
print("行", len(a), f"{time.perf_counter() - t0:.0f} 秒")
a = pd.read_csv(out, dtype={"K": str, "meff": str, "param": str})
e = pd.read_csv(ROOT / f"results/baselines_{world}/{T}.csv", dtype={"K": str, "meff": str, "param": str}).query("rep == 0")
e = e.assign(method=e.method.replace({"oursx": "dist"}))
m = a.merge(e, on=["student", "method", "param", "meff", "K"], suffixes=("", "_e61"))
print("E61 の反復 0 の行", len(e), "突き合わせた行", len(m), "差の最大", float((m.uar - m.uar_e61).abs().max()))
print("突き合わせた (生徒, 手法) の行数", m.groupby(["student", "method"]).size().to_dict())
print("突き合わない E61 の行の手法", e.merge(a, on=["student", "method", "param", "meff", "K"], how="left", indicator=True).query("_merge == 'left_only'").groupby(["student", "method"]).size().to_dict())
