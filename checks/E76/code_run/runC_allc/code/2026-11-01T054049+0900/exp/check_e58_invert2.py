"""E58 の枝 2：ターゲット固有の構造をどう使うかの下調べ（A の 7 ターゲット，反復 0・1．20 本＝K=5）。
  pca_k   ：直し D を候補の上位 k 主成分の中に限る（ターゲットの分散構造を事前分布にする）
  alpha   ：主成分 j の事前分散を λ_j^α に比例させる（α=0 が今の等方）
  maxchg  ：見本 4 本から始め，今の直しが点数を最も動かす発話に次のラベルを付ける（直しを確かめる選び方．4 本ごとに学び直す）
  margin_r：同じ手順で，学び直した生徒の迷いが大きい発話を選ぶ（E57 の margin．対照）
測るのは，E_0 との cos，動く長さ，テストの UAR（W_0 + D，b）．
"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "exp"); sys.path.insert(0, "measure")
import committee as CM, compare_b1 as CB, logreg, map as M, meta5 as M5
from data import groups_for
from select_labelfree import present_weights
from teach_student39 import Student
import geom_e0 as G

kind = "audio_wavlm"
groups = groups_for(kind); data = {g: M.load(g, kind) for g in groups}
cache = {}; rows = []
def fit_on(F, y, mask, anchorW, anchorb, C_T, cp, cb):
    yt = M.labels(y); dev = F.device
    w = torch.zeros(len(y), dtype=logreg.DTYPE, device=dev); mt = torch.as_tensor(mask, device=dev); w[mt] = present_weights(yt[mt])
    zero = torch.zeros(F.shape[1], dtype=F.dtype, device=dev)
    anc = logreg.Fitted(zero, torch.ones_like(zero), {C_T: anchorW}, {C_T: anchorb}, {C_T: 0}, {C_T: 0.0})
    fit = logreg.fit_batch(F, yt, [mask], (cp,), anchor=anc, anchor_c=C_T, stats=(zero, torch.ones_like(zero)), Cb=[cb], weights=[w])[0]
    return fit.W[cp], fit.b[cp]
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
        Z = st.Xstd @ Am; W0 = st.W_std @ Mh; b0 = st.b_std
        te = np.flatnonzero(ctx.te[r]); yte = ev.y[te]
        Zte = ((M.rows(ev.X, te) - ctx.ct[r] - st.src.mu) / st.src.sd) @ Am
        uar = lambda W, b: CB.uar(yte, (Zte @ W.T + b).argmax(1).cpu().numpy())
        Wst, bst, c_star, _ = G.fit_star(Z, yq, ev.speaker.iloc[pool], st.anchor(rho_m), C_T, 1000 + r); E0 = Wst - W0
        n = len(pool); idx5 = np.array([pos_of[int(i)] for i in ctx.idx[(r, "5")]]); idx1 = np.array([pos_of[int(i)] for i in ctx.idx[(r, "1")]])
        mask5 = np.zeros(n, bool); mask5[idx5] = True
        row = {"target": T, "rep": r, "uar_w0": uar(W0, b0), "uar_star": uar(Wst, bst)}
        W, b = fit_on(Z, yq, mask5, W0, b0, C_T, cp, cb); row["uar_random"] = uar(W, b); row["cos_random"] = cosE(W - W0, E0)
        Zc = Z - Z.mean(0); lam, V = torch.linalg.eigh(Zc.T @ Zc / n); lam, V = lam.flip(0).clamp(min=1e-12), V.flip(1)
        for k in (10, 20, 50, 100):
            Vk = V[:, :k]; Fk = Z @ Vk
            for mult in (1, 10):
                Wk, bk = fit_on(Fk, yq, mask5, W0 @ Vk, b0, C_T, cp * mult, cb)
                D = (Wk - W0 @ Vk) @ Vk.T; row[f"uar_pca{k}_c{mult}"] = uar(W0 + D, bk); row[f"cos_pca{k}_c{mult}"] = cosE(D, E0)
        for alpha in (0.5, 1.0):
            s_ = (lam / lam.mean()) ** (alpha / 2); Fa = (Z @ V) * s_                      # 事前分散 ∝ λ^α（座標を伸ばす）
            Wa, ba = fit_on(Fa, yq, mask5, (W0 @ V) / s_, b0, C_T, cp, cb)
            D = ((Wa - (W0 @ V) / s_) * s_) @ V.T; row[f"uar_alpha{alpha}"] = uar(W0 + D, ba); row[f"cos_alpha{alpha}"] = cosE(D, E0)
        for name in ("maxchg", "margin_r"):
            lab = list(idx1)
            while len(lab) < 20:
                m = np.zeros(n, bool); m[lab] = True
                W, b = fit_on(Z, yq, m, W0, b0, C_T, cp, cb)
                if name == "maxchg":
                    score = ((Z @ (W - W0).T) ** 2).sum(1)
                else:
                    P = torch.softmax(Z @ W.T + b, 1); top = P.topk(2, 1).values; score = -(top[:, 0] - top[:, 1])
                score[torch.as_tensor(lab, device=Z.device)] = -float("inf")
                lab += torch.topk(score, 4).indices.cpu().numpy().tolist()
            m = np.zeros(n, bool); m[lab] = True
            W, b = fit_on(Z, yq, m, W0, b0, C_T, cp, cb); row[f"uar_{name}"] = uar(W, b); row[f"cos_{name}"] = cosE(W - W0, E0)
            cnt = np.bincount(yq[lab], minlength=4); row[f"min_{name}"] = int(cnt.min())
        rows.append(row)
    print(T, "済み", flush=True)
d = pd.DataFrame(rows); pd.set_option("display.width", 250)
g = d.drop(columns=["rep"]).groupby("target").mean(numeric_only=True)
for c in [c for c in g.columns if c.startswith("uar_") and c not in ("uar_random", "uar_w0", "uar_star")]:
    g[c + "-rnd"] = g[c] - g.uar_random
print((g[[c for c in g.columns if c.startswith("uar_")]] * 100).round(2).T.to_string())
print("\n== 平均（UAR は ×100）")
m = d.drop(columns=["target", "rep"]).mean(); m[[i for i in m.index if i.startswith("uar_")]] *= 100
print(m.round(3).to_string())
d.to_csv("results/explore_relation/e58_invert2.csv", index=False)
