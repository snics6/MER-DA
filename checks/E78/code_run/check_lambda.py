"""E78 関門 1：DA の項の重み λ が小さい端（0.01）で，DANN・MME が対照（λ = 0）に近づくか．Adam は座標ごとに勾配を正規化するので，
K = 0（写像 A の勾配が DA の項だけ）では λ の大きさに依らず A が動きうる。あわせて種 0 の全候補（400 問）の所要を測る。
種類 audio_wavlm，評価のターゲット RAVDESS の仮のターゲット AFEW（反復 0．台本の仮のターゲットと同じ作り方）．評価のターゲットの値は見ない．結果は標準出力だけ。"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
torch.cuda.set_per_process_memory_fraction(0.12)
torch.backends.cuda.matmul.allow_tf32 = True
import compare_dai2_9044116 as E  # noqa: E402
E.ROOT = ROOT
import baselines_distill as BD  # noqa: E402
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from baselines_distill import Prob  # noqa: E402
from geom_e0 import settings  # noqa: E402
from teach_student39 import Student  # noqa: E402
from data import groups_for  # noqa: E402

kind, T, r = "audio_wavlm", "RAVDESS", 0
DEV = E.DEV
t0 = time.perf_counter()
data = {g: M.load(g, kind) for g in groups_for(kind)}
S = [s for s in data if CM.corp(s) != CM.corp(T)]
gamma, rho_s, rho_m, cb, cp, _ = settings(kind, T)
C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
c = "AFEW"
S = [s_ for s_ in S if CM.corp(s_) != CM.corp(c)]
C_T = E.C_FIT
src_pc, lin, n_of = M5.pool_info(data, S, C_T, {})
ev = data[c]
ctx = CB.Ctx(ev, [r], list(E.KS_LAB), center="ind")
pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
ct = ctx.ct[r]
Xu, te_X = M.rows(ev.X, pool) - ct, M.rows(ev.X, te) - ct
srcs = {s: (data[s].X, data[s].y) for s in S}
yq, yte = ev.y[pool], ev.y[te]
label_sets = BD.label_sets_of(ctx, r, pool)
st = Student(ev, Xu, src_pc, lin, n_of, S, C_T, gamma, rho_s, [rho_m])
Am = st.mats[rho_m][0]
s = {"st": st, "rho_m": rho_m, "cb": cb, "cp": cp, "Z": st.Xstd @ Am, "Zte": ((te_X - st.src.mu) / st.src.sd) @ Am}
Zs, ys = E.src_feats(st, rho_m, srcs)
pr = Prob(s, yq, label_sets)
none = pr.solve([(K, None, K) for K in E.KS])
lab = E.lab_of(label_sets, yq)
pos, ylab, wK = E.lab_tensors(lab)
I = torch.eye(pr.d, dtype=E.F32, device=DEV)
args = (s["Z"].to(E.F32), s["Z"].to(E.F32)[pos], ylab, wK, Zs.to(E.F32), torch.as_tensor(ys, device=DEV), pr.W0.to(E.F32), pr.b0.to(E.F32), float(cp), cb)
print(f"準備 {time.perf_counter() - t0:.0f} 秒．n={pr.n}")


def uar_of(Wb):
    return CB.uar(yte, (s["Zte"].double() @ Wb[0].T.double() + Wb[1].double()).argmax(1).cpu().numpy())


print("none の UAR:", {K: round(uar_of(none[K]), 4) for K in E.KS})
for lr in E.MAP_LRS:
    for iters in E.MAP_ITERS:
        chunk = [(m_, lam, mu, sw, K) for m_, lam, sw in (("mapctl", 0.0, 0.0), ("dann", 0.01, 0.0), ("mme", 0.01, 0.0)) for mu in (10.0,) for K in ("0", "5")]
        out = E._train_map(*args, chunk, lr, iters, 0, I)
        res = []
        for c, (Wf, bf) in zip(chunk, out):
            res.append(f"{c[0]}(λ={c[1]},K={c[4]}) {uar_of((Wf, bf)):.4f}")
        print(f"lr={lr} it={iters} μ=10：" + "，".join(res))
# 種 0 の全候補の所要
for meths in (("dann", "mme", "mapctl"),):
    rec = E.Rec({"target": T, "rep": r}, yte, keep=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    E.run_map(pr, Zs, ys, rec, kind, lab, E.map_problems(meths, (0,)))
    torch.cuda.synchronize()
    d = pd.DataFrame(rec.rows)
    print(f"種 0 の全候補 {len(d)} 問：{time.perf_counter() - t1:.1f} 秒（E77 と GPU を分け合った値）")
    print(d.groupby(["method", "K"]).uar.agg(["min", "median", "max"]).round(3).to_string())
    print("でたらめ（0.25）を下回る行:", d[d.uar < 0.25].groupby("method").size().to_dict(), "/", d.groupby("method").size().to_dict())
print(f"GPU 最大 {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB，全体 {time.perf_counter() - t0:.0f} 秒")

# CORAL の所要の内訳（r = 32，ソース 1 つ分の特異値分解を float64 と float32 で）
torch.cuda.synchronize(); t1 = time.perf_counter()
_ = E.coral_lin(srcs, Xu, rho_s, 32)
torch.cuda.synchronize()
print(f"coral_lin（r=32，ソース {len(srcs)}）：{time.perf_counter() - t1:.1f} 秒")
s0 = max(srcs, key=lambda k_: len(srcs[k_][0]))
both = torch.cat([srcs[s0][0] - srcs[s0][0].mean(0), Xu])
for dt in (torch.float64, torch.float32):
    x = (both - both.mean(0)).to(dt)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    torch.linalg.svd(x, full_matrices=False)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    torch.linalg.eigh(x.T @ x)
    torch.cuda.synchronize()
    print(f"{s0}（{tuple(x.shape)}，{dt}）：svd {t2 - t1:.1f} 秒，共分散の eigh {time.perf_counter() - t2:.2f} 秒")
