"""E77 関門 1 の検算（I13）：共正則化の罰則の重みを 0 に近づけた版（m_eff = 1e-8）が，各モダリティの none（E39 の学習器）と同じ解になるか。
AV の世界，AFEW，反復 0。参照の積 ref の作り直しが E61 の作り方と同じかも見る。"""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/snics/MER-DA"); sys.path.insert(0, str(ROOT / "exp")); sys.path.insert(0, str(ROOT / "measure"))
torch.cuda.set_per_process_memory_fraction(0.25)
import compare_dai as CD, baselines_distill as BD, compare_b1 as CB, committee as CM, map as M, meta5 as M5
from data import groups_for
from geom_e0 import settings
from multimodal_distill import WORLDS, blocks, make_students, sub_group
world, T, r = "fuse-audio-video", "AFEW", 0
groups = groups_for(world)
data_f = {g: M.load(g, world) for g in groups}
data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(world)}
S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
sets = {k: settings(k, T) for k in WORLDS[world] + [world]}
info = {k: M5.pool_info(data_m[k] if k != world else data_f, S, CD.C_FIT, {}) for k in sets}
ev = data_f[T]; ctx = CB.Ctx(ev, [r], list(CD.KS_LAB), center="ind")
pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
stu = make_students(world, T, data_f, data_m, S, pool, te, sets, info)
ls = BD.label_sets_of(ctx, r, pool)
mods = WORLDS[world]
prs = {k: CD.Prob(stu[k], ev.y[pool], ls) for k in mods + [world]}
none = {m: prs[m].solve([(K, None, K) for K in CD.KS]) for m in mods}
import time
CD.MEFFS = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("1e-8", "1e-4")          # 罰則の総重みを 0 に近づける
t0 = time.perf_counter()
rec = CD.Rec({"target": T, "rep": r}, ev.y[te], keep=True)
W64 = {}
_add = CD.Rec.add
def add64(self, student, method, param, meff, K, probs=None, Wb=None, Zte=None):
    if Wb is not None:
        W64[len(self.rows)] = (Wb[0].detach().double().cpu().numpy(), Wb[1].detach().double().cpu().numpy())
    return _add(self, student, method, param, meff, K, probs=probs, Wb=Wb, Zte=Zte)
CD.Rec.add = add64
CD.coreg_rec({m: prs[m] for m in mods}, rec, world)
Wrows = {st: lst for st, lst in rec.W.items()}
for m in mods:
    for (rid, W16, b16) in Wrows[m]:
        row = rec.rows[rid]
        W, b = W64[rid]
        e16 = float(np.abs(W16.astype(np.float64) - W).max() / np.abs(W).max())
        Wn, bn = none[m][row["K"]]
        print(m, "m_eff", row["meff"], "K", row["K"], "W の差の最大（float64．W の最大で割る）", float(np.abs(W.astype(np.float64) - Wn.cpu().numpy()).max() / np.abs(Wn.cpu().numpy()).max()),
              "b の差", float(np.abs(b - bn.cpu().numpy()).max()), "float16 で保存した W の丸めの誤差（W の最大で割る）", e16)
print(f"共正則化 {time.perf_counter() - t0:.0f} 秒")
for c in rec.conv: print(c)
