"""共正則化の検算（UAR は出さない）：(1) 保存した解での目的関数の勾配の最大，(2) ρ = 0 にしたとき K 本だけの比較の基準（段 1 の同じ引き戻し）と一致するか，
(3) 別の解き方（scipy の L-BFGS-B，float64，強い許容）の解と判定が一致するか，(4) AVT の判定が 3 つの点数の和の argmax か，(5) 最終の .grad と解き直した勾配の差。"""
import sys, json, time
import numpy as np, torch
import dai1, dai_data as DD
S = dai1.S
out, T, c, f, r = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
from pathlib import Path
srcs = [s for s in DD.others(T) if s != c]
u = dai1.Unit(Path(out), T, c, srcs, f, r)
tw = json.load(open(f"{out}/pseudo/{T}_tether_w.json"))
teth = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in tw["tether"].items()}
base = f"{out}/pseudo/units/{T}_{c}_f{f}_r{r}"
z1, z3 = np.load(base + "_p1.npz"), np.load(base + "_p3.npz")

def obj(Ws, bs, K, rho):
    gk = u.gK(K) if K else torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
    tot, cen = 0.0, []
    for m in "AVT":
        W0, b0 = u.start[m]; cw, cb = teth[(m, max(K, 1))]
        Z = u.Xp[m] @ Ws[m].T + bs[m]
        tot = tot - (gk * torch.log_softmax(Z, 1)).sum() + ((Ws[m] - W0) ** 2).sum() / (2 * cw) + ((bs[m] - b0) ** 2).sum() / (2 * cb)
        cen.append(Z - Z.mean(1, keepdim=True))
    dis = sum(((cen[a] - cen[b]) ** 2).sum() for a in range(3) for b in range(a + 1, 3))
    return (tot + 0.5 * rho * dis) / (3 * float(gk.sum()) + rho * u.n)

for K in dai1.KS_ALL:
    for j, rho in enumerate(dai1.RHO_GRID):
        Ws = {m: S.t(z3[f"p3g_K{K}_{m}_W"][j]).requires_grad_(True) for m in "AVT"}
        bs = {m: S.t(z3[f"p3g_K{K}_{m}_b"][j]).requires_grad_(True) for m in "AVT"}
        fval = obj(Ws, bs, K, rho); fval.backward()
        g = max(float(x.grad.abs().max()) for x in list(Ws.values()) + list(bs.values()))
        Zs = sum(u.Xt[m] @ Ws[m].detach().T + bs[m].detach() for m in "AVT")
        avt_ok = bool((Zs.argmax(1).cpu().numpy() == z3[f"p3g_K{K}_AVT_pred"][j]).all())
        print(f"K={K} ρ={rho}: 保存した解（float32）での勾配の最大 {g:.2e}，AVT の判定が和の argmax {avt_ok}")
# (2) ρ = 0（K ≥ 1）
dai1_RHO = dai1.RHO_GRID
dai1.RHO_GRID = (0.0,)
store, rows = {}, []
for K in [k for k in dai1.KS_ALL if k > 0]:
    dai1.coreg(u, K, teth, store, rows)
    for m in "AVT":
        cw, cb = teth[(m, K)]
        items = [str(x) for x in z1[f"p1_{m}_items"]]
        jj = items.index(f"{K}|{cw}|{cb}")
        Wn, bn = z1[f"p1_{m}_W"][jj], z1[f"p1_{m}_b"][jj]
        Wc, bc = store[f"p3g_K{K}_{m}_W"][0], store[f"p3g_K{K}_{m}_b"][0]
        pn = z1[f"p1_{m}_pred"][jj]
        print(f"ρ=0 K={K} {m}: W の差の最大 {np.abs(Wn - Wc).max():.2e}，b {np.abs(bn - bc).max():.2e}，判定の食い違い {int((pn != store[f'p3g_K{K}_{m}_pred'][0]).sum())}/{len(pn)}")
dai1.RHO_GRID = dai1_RHO
# (3) 別の解き方（K=5，ρ=1）
from scipy.optimize import minimize
K, rho = 5, 1.0
j = list(dai1.RHO_GRID).index(rho)
shapes = {m: u.start[m][0].shape for m in "AVT"}
def unpack(th):
    th = torch.as_tensor(th, dtype=S.DT, device=S.DEV); o = 0; Ws, bs = {}, {}
    for m in "AVT":
        n_ = shapes[m][0] * shapes[m][1]; Ws[m] = th[o:o + n_].view(shapes[m]); o += n_
        bs[m] = th[o:o + 4]; o += 4
    return Ws, bs
def fg(th):
    tt = torch.as_tensor(th, dtype=S.DT, device=S.DEV).requires_grad_(True)
    Ws, bs = unpack(tt); fv = obj(Ws, bs, K, rho); fv.backward()
    return float(fv), tt.grad.cpu().numpy()
x0 = np.concatenate([np.concatenate([u.start[m][0].cpu().numpy().ravel(), u.start[m][1].cpu().numpy()]) for m in "AVT"])
t0 = time.time()
res = minimize(fg, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=50000, gtol=1e-10, ftol=0, maxcor=20))
Ws, bs = unpack(res.x)
gmax = float(np.abs(res.jac).max())
print(f"scipy：{time.time()-t0:.0f} 秒，反復 {res.nit}，勾配の最大 {gmax:.1e}，終わり方 {res.message}")
for m in "AVT":
    Wsv = z3[f"p3g_K{K}_{m}_W"][j]
    pr = (u.Xt[m] @ Ws[m].T + bs[m]).argmax(1).cpu().numpy()
    print(f"  {m}: W の差の最大 {np.abs(Ws[m].cpu().numpy() - Wsv).max():.2e}，判定の食い違い {int((pr != z3[f'p3g_K{K}_{m}_pred'][j]).sum())}")
# (5) 台本の n_iter と grad の記録
import pandas as pd
d = pd.read_csv(base + "_p3.csv", low_memory=False)
print(d[d.method == "_conv_coreg"][["K", "rho", "grad_max", "n_iter", "finite"]].to_string(index=False))
