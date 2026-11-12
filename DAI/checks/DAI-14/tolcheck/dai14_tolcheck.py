import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd, torch
import dai14 as M, dai3, dai1, dai_data as DD, dai_solver as S
dai1.ENC["enc"] = DD.ENC1
fz1, _ = dai3.load_fz1(M.OUT1)
T, f, r, K, s, nm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
sv = dai3.Saved(M.OUT1, "eval", T, T, f, r, fz1[T]); u = sv.u
TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
cl = sv.cl_pool.astype(str); h = dai3.pool_halves(cl); halves = [np.flatnonzero(h == 0), np.flatnonzero(h == 1)]
Q = dai3.pool_teachers(sv, TK[K], s, K, [nm]); qy = Q[nm].argmax(1).cpu().numpy().astype(int)
a = np.load(f"/home/snics/MER-DA/results/dai14/units/{T}_f{f}_r{r}.npz")
assert (a[f"tgt__K{K}__{nm}__true"] == qy).all()
unit_pred = a[f"K{K}__main__R0__{s}__{nm}__true"]
saved = pd.read_csv(M.REPRO_SAVED, low_memory=False)
sv_ = saved[(saved["T"] == T) & (saved.fold == f) & (saved.seed == r) & (saved.K == K) & (saved.student == s) & (saved.teacher == nm) & (saved.subset == "all")].agree.iloc[0]
unit_ag = pd.read_csv(f"/home/snics/MER-DA/results/dai14/units/{T}_f{f}_r{r}.csv").query("K==@K and readout=='R0' and student==@s and target==@nm and version=='true' and split=='main'").agree_halfmean.iloc[0]
n = u.n; W0, b0 = u.start[s]; cw, cb = sv.teth[(s, K)]
G = torch.stack([S.balanced_onehot(qy, halves[hh], n) for hh in (0, 1)])
for tol in (1e-6, 1e-8, 1e-10):
    res = S.fit(u.Xp[s], G, W0, b0, Cw=cw, Cb=cb, tol=tol, max_iter=200000)
    pr = S.logits(u.Xp[s], res["W"], res["b"]).argmax(2).cpu().numpy()
    full = np.empty(n, int); ag = []
    for hh in (0, 1):
        o = halves[1 - hh]; full[o] = pr[hh][o]; ag.append(100 * S.uar(qy[o], pr[hh][o]))
    ag = float(np.mean(ag))
    print(f"tol={tol:g}: |alone-saved|={abs(ag-sv_):.4f} |alone-unit|={abs(ag-unit_ag):.4f} n_pred_diff_vs_unit={(full!=unit_pred).sum()} grad={res['grad_p'].max():.1e}")
print(f"|unit-saved|={abs(unit_ag-sv_):.4f}  n_candidates={n}")
