"""E58 の枝 3：見込みを「選ぶため」ではなく「候補全体の弱いラベル」として生徒に流し込む（蒸留）下調べ（A の 7 ターゲット，反復 0・1）。
  弱いラベル：候補の各発話に見込み q の確率をそのまま重みにした 4 つのラベル（soft label）．総重み m_eff 本ぶん（20・100・500）．
  見込み：p（出発点自身．対照．動かないはず），q（E44 の C），pq（{p, q} の積み上げ．E52）．
  それに K=5 の 20 本の本当のラベルを足した版も測る．量：E_0 との cos，動く長さ，テストの UAR．
"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "exp"); sys.path.insert(0, "measure")
import committee as CM, compare_b1 as CB, logreg, map as M, meta5 as M5
from data import groups_for
from select_labelfree import present_weights
from teach_student39 import Student
from teach_stack import c_pool_on, draft_logits, fit_stack, logp, softmax_t
import geom_e0 as G

kind = "audio_wavlm"
groups = groups_for(kind); data = {g: M.load(g, kind) for g in groups}
f, Cs, P, Y = CM.prepare(kind, data)
cache = {}; rows = []
def cosE(D, E0): return float((D * E0).sum() / (D.norm() * E0.norm()).clamp(min=1e-12))
def fit_weighted(Z, y_rows, w_rows, anchorW, anchorb, C_T, cp, cb):
    """行ごとのラベルと重みで学ぶ（soft label は 4 行に展開済み）。"""
    dev = Z.device; zero = torch.zeros(Z.shape[1], dtype=Z.dtype, device=dev)
    anc = logreg.Fitted(zero, torch.ones_like(zero), {C_T: anchorW}, {C_T: anchorb}, {C_T: 0}, {C_T: 0.0})
    fit = logreg.fit_batch(Z, y_rows, [np.ones(len(y_rows), bool)], (cp,), anchor=anc, anchor_c=C_T, stats=(zero, torch.ones_like(zero)), Cb=[cb], weights=[w_rows])[0]
    return fit.W[cp], fit.b[cp]
for T in groups:
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    gamma, rho_s, rho_m, cb, cp, c_t = G.settings(kind, T)
    C_T = float(pd.read_csv(f"results/multisource_scores_{kind}/{T}.csv", dtype={"K": str}).C.iloc[0])
    lam44 = float(pd.read_csv(f"results/committee_{kind}/{T}.csv").query("variant == 'C'").lam.iloc[0])
    pool_q = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), lam44)
    mem = {g: {"p": softmax_t(draft_logits(data, S, g, gamma, rho_s, cache)), "q": c_pool_on(g, S, P, Y, lam44)} for g in S}
    pool_pq, _ = fit_stack(T, "pq", S, mem, Y, [])
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    ev = data[T]; ctx = CB.Ctx(ev, [0, 1], ["1", "5"], center="ind")
    for r in [0, 1]:
        pool = np.flatnonzero(~ctx.te[r]); yq = ev.y[pool]; pos_of = {int(i): j for j, i in enumerate(pool)}
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, gamma, rho_s, [rho_m]); Am, Mh = st.mats[rho_m]
        Z = st.Xstd @ Am; W0 = st.W_std @ Mh; b0 = st.b_std; n = len(pool); dev = Z.device
        te = np.flatnonzero(ctx.te[r]); yte = ev.y[te]
        Zte = ((M.rows(ev.X, te) - ctx.ct[r] - st.src.mu) / st.src.sd) @ Am
        uar = lambda W, b: CB.uar(yte, (Zte @ W.T + b).argmax(1).cpu().numpy())
        Wst, bst, c_star, _ = G.fit_star(Z, yq, ev.speaker.iloc[pool], st.anchor(rho_m), C_T, 1000 + r); E0 = Wst - W0
        Ps = [softmax_t(f[s].logits(Xc, Cs[s])) for s in S]
        p = softmax_t(st.L0)
        tens = lambda cc: (torch.as_tensor(np.stack(cc, 1), dtype=logreg.DTYPE, device=dev), torch.ones(n, len(cc), dtype=logreg.DTYPE, device=dev))
        q = pool_q.predict(*tens([logp(x) for x in Ps])); pq = pool_pq.predict(*tens([logp(p), logp(q)]))
        beliefs = {"p": p, "q": q, "pq": pq}
        idx5 = np.array([pos_of[int(i)] for i in ctx.idx[(r, "5")]])
        yt = M.labels(yq)
        row = {"target": T, "rep": r, "uar_w0": uar(W0, b0), "uar_star": uar(Wst, bst), "acc_q": float((q.argmax(1) == yq).mean()), "acc_pq": float((pq.argmax(1) == yq).mean()), "acc_p": float((p.argmax(1) == yq).mean())}
        # 本当の 20 本だけ（参照）
        w20 = torch.zeros(n, dtype=logreg.DTYPE, device=dev); m5 = torch.zeros(n, dtype=torch.bool, device=dev); m5[torch.as_tensor(idx5, device=dev)] = True
        w20[m5] = present_weights(yt[m5])
        W, b = fit_weighted(Z, yt, w20, W0, b0, C_T, cp, cb); row["uar_true20"] = uar(W, b); row["cos_true20"] = cosE(W - W0, E0)
        Z4 = Z.repeat(4, 1); y4 = torch.arange(4, device=dev).repeat_interleave(n)          # soft label を 4 行に展開
        for name, B in beliefs.items():
            Bt = torch.as_tensor(B, dtype=logreg.DTYPE, device=dev)
            cls_w = 1.0 / Bt.sum(0).clamp(min=1e-6)                                            # 感情の釣り合い（見込みの総量で）
            base = (Bt * cls_w).T.reshape(-1)                                                   # 行 = 感情 k の n 本
            for meff in (20, 100, 500):
                w4 = base / base.sum() * meff
                W, b = fit_weighted(Z4, y4, w4, W0, b0, C_T, cp, cb)
                row[f"uar_{name}{meff}"] = uar(W, b); row[f"cos_{name}{meff}"] = cosE(W - W0, E0); row[f"mov_{name}{meff}"] = float((W - W0).norm() / E0.norm())
                if name != "p":
                    Zm = torch.cat([Z4, Z]); ym = torch.cat([y4, yt]); wm = torch.cat([w4, w20])  # 弱いラベル＋本当の 20 本
                    W, b = fit_weighted(Zm, ym, wm, W0, b0, C_T, cp, cb)
                    row[f"uar_{name}{meff}+20"] = uar(W, b); row[f"cos_{name}{meff}+20"] = cosE(W - W0, E0)
        rows.append(row)
    print(T, "済み", flush=True)
d = pd.DataFrame(rows); pd.set_option("display.width", 250)
g = d.drop(columns=["rep"]).groupby("target").mean(numeric_only=True)
uc = [c for c in g.columns if c.startswith("uar_")]
print((g[uc] * 100).round(2).T.to_string())
print("\n== 平均（UAR・acc は ×100）")
m = d.drop(columns=["target", "rep"]).mean(); m[[i for i in m.index if i.startswith(("uar_", "acc_"))]] *= 100
print(m.round(3).to_string())
d.to_csv("results/explore_relation/e58_invert3.csv", index=False)
