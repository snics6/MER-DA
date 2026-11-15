"""E58 の枝：ランダム 20 本の直しの向きを，ラベルを増やさずに E_0 へそろえる 3 つの案の下調べ（A の 7 ターゲット，反復 0・1）。
  rank   ：E_0 の特異値の分布（ランク 1 に近いか）と，20 本の直し D̂ をランク 1・2 に切ったときの cos
  aug    ：各ラベル付き発話の近い k 本に同じラベルを弱い重みで付けて学ぶ（ターゲットの近所の構造でラベルを広げる）
  loo    ：他の 19 本で学んだ直しがその発話の正解側へ点数を動かすか（整合）で重みを付け直す
"""
import sys, zlib, numpy as np, pandas as pd, torch
sys.path.insert(0, "exp"); sys.path.insert(0, "measure")
import committee as CM, compare_b1 as CB, logreg, map as M, meta5 as M5
from belief_check import knn
from data import groups_for
from select_labelfree import present_weights
from teach_student39 import Student
import geom_e0 as G

kind = "audio_wavlm"
groups = groups_for(kind); data = {g: M.load(g, kind) for g in groups}
cache = {}; rows = []
def fit_D(Z, yq, mask, w, st, rho_m, C_T, cp, cb, W0):
    zero = torch.zeros(Z.shape[1], dtype=Z.dtype, device=Z.device)
    fit = logreg.fit_batch(Z, M.labels(yq), [mask], (cp,), anchor=st.anchor(rho_m), anchor_c=C_T, stats=(zero, torch.ones_like(zero)), Cb=[cb], weights=[w])[0]
    return fit.W[cp] - W0
def cosE(D, E0): return float((D * E0).sum() / (D.norm() * E0.norm()).clamp(min=1e-12))
for T in groups:
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    gamma, rho_s, rho_m, cb, cp, c_t = G.settings(kind, T)
    C_T = float(pd.read_csv(f"results/multisource_scores_{kind}/{T}.csv", dtype={"K": str}).C.iloc[0])
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    ev = data[T]; ctx = CB.Ctx(ev, [0, 1], ["1", "5"], center="ind")
    for r in [0, 1]:
        pool = np.flatnonzero(~ctx.te[r]); yq = ev.y[pool]; pos_of = {int(i): j for j, i in enumerate(pool)}
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, gamma, rho_s, [rho_m]); Am, Mh = st.mats[rho_m]
        Z = st.Xstd @ Am; W0 = st.W_std @ Mh
        Wst, bst, c_star, _ = G.fit_star(Z, yq, ev.speaker.iloc[pool], st.anchor(rho_m), C_T, 1000 + r); E0 = Wst - W0
        sv = torch.linalg.svdvals(E0); svf = (sv ** 2 / (sv ** 2).sum()).cpu().numpy()
        idx = np.array([pos_of[int(i)] for i in ctx.idx[(r, "5")]]); n = len(pool)
        yt = M.labels(yq); dev = Z.device
        # plain
        mask = np.zeros(n, bool); mask[idx] = True
        w = torch.zeros(n, dtype=logreg.DTYPE, device=dev); w[torch.as_tensor(mask, device=dev)] = present_weights(yt[torch.as_tensor(mask, device=dev)])
        D = fit_D(Z, yq, mask, w, st, rho_m, C_T, cp, cb, W0)
        row = {"target": T, "rep": r, "c_star": c_star, "sv1": svf[0], "sv12": svf[:2].sum(), "cos_plain": cosE(D, E0), "mov_plain": float(D.norm() / E0.norm())}
        U, s_, Vh = torch.linalg.svd(D, full_matrices=False)
        for k in (1, 2):
            Dk = (U[:, :k] * s_[:k]) @ Vh[:k]; row[f"cos_rank{k}"] = cosE(Dk, E0)
        Ek = (torch.linalg.svd(E0, full_matrices=False)); 
        # aug: neighbours get the label
        nb = knn(Z.float(), 10).cpu().numpy()
        for wn in (0.3, 1.0):
            m2 = mask.copy(); y2 = yq.copy(); w2 = w.clone()
            hit = []
            for i in idx:
                for j in nb[i]:
                    if j in idx: continue
                    hit.append(yq[j] == yq[i])
                    m2[j] = True; y2[j] = yq[i]
            yt2 = M.labels(y2); w2 = torch.zeros(n, dtype=logreg.DTYPE, device=dev)
            mt = torch.as_tensor(m2, device=dev); bal = present_weights(yt2[mt]); w2[mt] = bal
            lab_t = torch.as_tensor(mask, device=dev); nb_only = mt & ~lab_t
            w2[nb_only] = w2[nb_only] * wn
            Da = fit_D(Z, y2, m2, w2, st, rho_m, C_T, cp, cb, W0)
            row[f"cos_aug{wn}"] = cosE(Da, E0); row[f"mov_aug{wn}"] = float(Da.norm() / E0.norm()); row["nb_label_acc"] = float(np.mean(hit))
        # loo consistency weights
        masks = []; ws = []
        for i in idx:
            mi = mask.copy(); mi[i] = False
            wi = torch.zeros(n, dtype=logreg.DTYPE, device=dev); mt = torch.as_tensor(mi, device=dev); wi[mt] = present_weights(yt[mt])
            masks.append(mi); ws.append(wi)
        zero = torch.zeros(Z.shape[1], dtype=Z.dtype, device=dev)
        fits = logreg.fit_batch(Z, yt, masks, (cp,), anchor=st.anchor(rho_m), anchor_c=C_T, stats=(zero, torch.ones_like(zero)), Cb=[cb] * len(masks), weights=ws)
        p0 = torch.softmax(st.L0, 1)
        cons = []
        for i, f in zip(idx, fits):
            Di = f.W[cp] - W0; g = torch.nn.functional.one_hot(yt[i], 4).to(Z.dtype) - p0[i]
            cons.append(float((Di @ Z[i]) @ g))
        cons = np.array(cons); row["cons_pos_frac"] = float((cons > 0).mean())
        for name, wsel in (("loo_pos", cons > 0), ("loo_top10", cons >= np.sort(cons)[-10])):
            m3 = np.zeros(n, bool); m3[idx[wsel]] = True
            w3 = torch.zeros(n, dtype=logreg.DTYPE, device=dev); mt = torch.as_tensor(m3, device=dev); w3[mt] = present_weights(yt[mt])
            D3 = fit_D(Z, yq, m3, w3, st, rho_m, C_T, cp, cb, W0); row[f"cos_{name}"] = cosE(D3, E0); row[f"n_{name}"] = int(wsel.sum())
        # larger step for reference
        D5 = fit_D(Z, yq, mask, w, st, rho_m, C_T, cp * 10, cb, W0); row["cos_plain_c10"] = cosE(D5, E0); row["mov_plain_c10"] = float(D5.norm() / E0.norm())
        rows.append(row)
    print(T, "済み", flush=True)
d = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(d.drop(columns=["rep"]).groupby("target").mean(numeric_only=True).round(3).T.to_string())
print("\nmean"); print(d.drop(columns=["target", "rep"]).mean().round(3).to_string())
d.to_csv("results/explore_relation/e58_invert1.csv", index=False)
