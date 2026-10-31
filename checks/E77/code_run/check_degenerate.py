"""E77 関門 1 の検算（GPU，RAVDESS 反復 0，audio_wavlm）：IM の重み → 0 で none に戻るか，学習器の収束（勾配の最大と反復の回数），
ラベル伝播の共役勾配法の残差，float16 で保存した確率の argmax の食い違い。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path("/home/snics/MER-DA"); sys.path.insert(0, str(ROOT / "exp")); sys.path.insert(0, str(ROOT / "measure"))
torch.cuda.set_per_process_memory_fraction(0.25)
import compare_dai as CD, baselines_distill as BD, compare_b1 as CB, committee as CM, map as M, meta5 as M5, logreg
from geom_e0 import settings
from teach_student39 import Student
from data import groups_for
LOG = []
_fb = logreg.fit_batch
def fb(*a, **k):
    out = _fb(*a, **k)
    LOG.append([(list(f.n_iter.values())[0], list(f.grad.values())[0]) for f in out])
    return out
CD.logreg.fit_batch = fb
kind, T, r = "audio_wavlm", sys.argv[1] if len(sys.argv) > 1 else "RAVDESS", 0
data = {g: M.load(g, kind) for g in groups_for(kind)}
S = [s for s in data if CM.corp(s) != CM.corp(T)]
gamma, rho_s, rho_m, cb, cp, _ = settings(kind, T)
C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
src, lin, n_of = M5.pool_info(data, S, C_T, {})
ev = data[T]; ctx = CB.Ctx(ev, [r], list(CD.KS_LAB), center="ind")
pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
st = Student(ev, M.rows(ev.X, pool) - ctx.ct[r], src, lin, n_of, S, C_T, gamma, rho_s, [rho_m])
Am = st.mats[rho_m][0]
s = {"st": st, "rho_m": rho_m, "cb": cb, "cp": cp, "Z": st.Xstd @ Am, "Zte": ((M.rows(ev.X, te) - ctx.ct[r] - st.src.mu) / st.src.sd) @ Am}
pr = CD.Prob(s, ev.y[pool], BD.label_sets_of(ctx, r, pool))
none = pr.solve([(K, None, K) for K in CD.KS])
print("T", T, "n 候補", pr.n, "d", pr.d, "設定 (γ, ρ_s, ρ_M, C_b, C′)", (gamma, rho_s, rho_m, cb, cp), "C_T", C_T)
print("none の学習器の (反復, 勾配の最大)", LOG[-1])
f = logreg.Fitted
print("Fitted の属性", [a for a in vars(logreg.fit_batch.__wrapped__ if hasattr(logreg.fit_batch, '__wrapped__') else _fb).keys()] if False else "")
# 1. IM の重みを 0 に近づける
for K in ("1", "5", "25"):
    sol = CD.im_solve(pr, none, [(0.3, "1e-8", K)])
    W, b = sol[(0.3, "1e-8", K)]
    print(f"IM m=1e-8 K={K}：none との W の差の最大 {float((W - none[K][0]).abs().max()):.2e}，b {float((b - none[K][1]).abs().max()):.2e}")
# 2. IM の全格子の収束
LOG.clear()
sol = CD.im_solve(pr, none, [(a, m, K) for a in CD.IM_ALPHAS for m in CD.MEFFS for K in CD.KS])
it = np.array([x[0] for x in LOG[-1]]); gm = np.array([x[1] for x in LOG[-1]], dtype=float)
print(f"IM 60 問：反復の回数 最大 {it.max()}，勾配の最大の最大 {np.nanmax(gm):.2e}（TOL={logreg.TOL}），TOL を超えた問 {(gm > logreg.TOL).sum()}")
# 3. 自己学習・ラベル伝播・dist の収束（Prob.solve の中の fit_batch）
LOG.clear()
lab = CD.lab_of(BD.label_sets_of(ctx, r, pool), ev.y[pool])
rec = CD.Rec({"target": T, "rep": r}, ev.y[te], keep=False)
CD.run_ab(pr, rec, kind, lab)
allf = [x for L in LOG for x in L]
it = np.array([x[0] for x in allf]); gm = np.array([x[1] for x in allf], dtype=float)
print(f"run_ab の全 fit_batch {len(allf)} 問：反復の回数 最大 {it.max()}，TOL を超えた問 {(gm > logreg.TOL).sum()}，勾配の最大の最大 {np.nanmax(gm):.2e}")
# 4. ラベル伝播：共役勾配法 50 回の残差（相対）と，密な解との差
K = "5"
for k in CD.LP_KS:
    for al in CD.LP_ALPHAS:
        Fs = CD.lp_scores(s["Z"], *lab[K], k, al)
        # 密な行列で解き直す
        n = pr.n
        zn = torch.nn.functional.normalize(s["Z"].float(), dim=1)
        Sm = zn @ zn.T; Sm.fill_diagonal_(-2.0)
        v, i = Sm.topk(min(k, n - 1), 1)
        Wd = torch.zeros(n, n, device=Sm.device); Wd.scatter_(1, i, v.clamp(min=0) ** 3)
        Wd = (Wd + Wd.T).double(); dd = Wd.sum(1); dm = torch.where(dd > 0, dd.rsqrt(), torch.zeros_like(dd))
        Wn = dm[:, None] * Wd * dm[None]
        Y = torch.zeros(n, 4, dtype=torch.float64, device=Sm.device); Y[torch.as_tensor(lab[K][0]), torch.as_tensor(lab[K][1])] = 1
        Fd = torch.linalg.solve(torch.eye(n, dtype=torch.float64, device=Sm.device) - al * Wn, Y).clamp(min=0)
        hd = (Fd.argmax(1) == Fs.argmax(1)).float().mean()
        print(f"LP k={k} α={al}：CG と密な解の差の最大（相対） {float((Fs - Fd).abs().max() / Fd.abs().max()):.2e}，硬いラベルの一致 {float(hd):.4f}，全部 0 の行 {(Fs.sum(1) == 0).sum().item()}")
