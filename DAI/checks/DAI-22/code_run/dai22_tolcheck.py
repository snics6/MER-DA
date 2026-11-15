"""gate-code の検算：dai22 の R0（AVT，prop_one）が実験 3 の保存と 0.001 を超えて違った単位を，
止める勾配の基準を変えて 1 問だけで解き直し，実験 3 と同じ束（repro_types + prop_sel, prop_one）でも解き直す．
差と予測の違いの数だけを出す（一致の値そのものは出さない）．"""
import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd, torch
import dai3, dai1, dai_data as DD, dai_solver as S
dai1.ENC["enc"] = DD.ENC1
OUT1 = DD.ROOT / "results" / "dai1"
fz1, _ = dai3.load_fz1(OUT1)
T, f, r, K, out_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
s, nm = "AVT", "prop_one"
sv = dai3.Saved(OUT1, "eval", T, T, f, r, fz1[T]); u = sv.u
TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
cl = sv.cl_pool.astype(str); h = dai3.pool_halves(cl); halves = [np.flatnonzero(h == 0), np.flatnonzero(h == 1)]
a = np.load(f"{out_dir}/units/{T}_f{f}_r{r}.npz")
unit_pred = a[f"K{K}__R0"].astype(int)
unit_ag = pd.read_csv(f"{out_dir}/units/{T}_f{f}_r{r}.csv").query("K==@K and readout=='R0'").agree_halfmean.iloc[0]
saved = pd.read_csv(DD.ROOT / "results" / "dai3" / "eval" / "repro_units.csv", low_memory=False)
sv_ = saved[(saved["T"] == T) & (saved.fold == f) & (saved.seed == r) & (saved.K == K) & (saved.student == s) & (saved.teacher == nm) & (saved.subset == "all")].agree.iloc[0]
n = u.n; W0, b0 = u.start[s]; cw, cb = sv.teth[(s, K)]


def cross(pr, qy, j0):
    full = np.empty(n, int); ag = []
    for hh in (0, 1):
        o = halves[1 - hh]; full[o] = pr[j0 + hh][o]; ag.append(100 * S.uar(qy[o], pr[j0 + hh][o]))
    return full, float(np.mean(ag))


Q = dai3.pool_teachers(sv, TK[K], s, K, [nm]); qy = Q[nm].argmax(1).cpu().numpy().astype(int)
G = torch.stack([S.balanced_onehot(qy, halves[hh], n) for hh in (0, 1)])
for tol in (1e-6, 1e-8, 1e-10):
    res = S.fit(u.Xp[s], G, W0, b0, Cw=cw, Cb=cb, tol=tol, max_iter=200000)
    pr = S.logits(u.Xp[s], res["W"], res["b"]).argmax(2).cpu().numpy()
    full, ag = cross(pr, qy, 0)
    print(f"alone tol={tol:g}: |alone-saved|={abs(ag - sv_):.4f} |alone-unit|={abs(ag - unit_ag):.4f} n_pred_diff_vs_unit={(full != unit_pred).sum()} grad={res['grad_p'].max():.1e}")
# 実験 3 と同じ束
names = dai3.repro_types(s, sv.srcs) + ["prop_sel", "prop_one"]
Qa = dai3.pool_teachers(sv, TK[K], s, K, names)
qya = {k: Qa[k].argmax(1).cpu().numpy() for k in names}
Gb = torch.stack([S.balanced_onehot(qya[k], halves[hh], n) for k in names for hh in (0, 1)])
res = S.fit(u.Xp[s], Gb, W0, b0, Cw=cw, Cb=cb)
pr = S.logits(u.Xp[s], res["W"], res["b"]).argmax(2).cpu().numpy()
j = names.index(nm)
full, ag = cross(pr, qy, 2 * j)
print(f"batch(dai3 names {len(names)}) tol=1e-6: |batch-saved|={abs(ag - sv_):.4f} |batch-unit|={abs(ag - unit_ag):.4f} n_pred_diff_vs_unit={(full != unit_pred).sum()}")
print(f"|unit-saved|={abs(unit_ag - sv_):.4f}  n_candidates={n}")
