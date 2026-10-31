"""E60 の下調べ 3：運用時も複数モダリティ．融合の E39 を超える 2 つの筋（AV の 6 群，AVT の 4 群，反復 0・1，K ∈ {0, 5}）．
  組み直し：モダリティごとの生徒を（a）融合の出発点，（b）互いの票（AVT では他の 2 つの積）で鍛え（E59 の蒸留），確率の積（poe）と点数の和（sum）で融合し直す
  DS      ：ターゲットの候補の上で，A・V・T の出発点の票から Dawid–Skene の EM で各モダリティの混同の表を推定した事後確率（ラベル無し）
  蒸留    ：組み直しの poe と DS の事後確率を，融合の生徒に E59 の形で蒸留（m_eff ∈ {100, 500}）
  上限    ：候補全体のラベルで学んだ融合の分類器（W*）
"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "exp"); sys.path.insert(0, "measure")
import committee as CM, compare_b1 as CB, logreg, map as M, meta5 as M5
from data import groups_for
from geom_e0 import fit_star, settings
from teach_student39 import Student
from distill_belief import solve_all, uar_of
dev = logreg.DEVICE
BLOCKS = {"fuse-audio-video": [("audio_wavlm", 0, 1024), ("video", 1024, 1536)],
          "fuse-audio-video-text": [("audio_wavlm", 0, 1024), ("video", 1024, 1536), ("text_roberta", 1536, 2304)]}
MEFFS = (100, 500)
def sub_group(g, a, b): return M.Group(g.name, g.X[:, a:b], g.y, g.speaker, g.person)
def lg(p): return np.log(np.clip(p, 1e-12, None))
def dawid_skene(votes, n_iter=30, smooth=1.0):
    """votes: (n, M) の言い切りの票．ラベル無しで事前確率と各票の混同の表を推定し，(事後確率, π, C) を返す。"""
    n, Mn = votes.shape
    post = np.zeros((n, 4))
    for j in range(Mn):
        post[np.arange(n), votes[:, j]] += 1
    post = post / post.sum(1, keepdims=True)
    for _ in range(n_iter):
        pi = post.mean(0)
        Cs = []
        for j in range(Mn):
            Cj = np.full((4, 4), smooth)
            for l in range(4):
                Cj[:, l] += post[votes[:, j] == l].sum(0)
            Cs.append(Cj / Cj.sum(1, keepdims=True))
        logp = np.log(pi)[None, :].repeat(n, 0)
        for j in range(Mn):
            logp += np.log(Cs[j][:, votes[:, j]].T)
        post = np.exp(logp - logp.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
    return post, pi, Cs
def ds_apply(votes, pi, Cs):
    logp = np.log(pi)[None, :].repeat(len(votes), 0)
    for j in range(len(Cs)):
        logp += np.log(Cs[j][:, votes[:, j]].T)
    post = np.exp(logp - logp.max(1, keepdims=True)); return post / post.sum(1, keepdims=True)
rows = []
for fkind, blocks in BLOCKS.items():
    groups = groups_for(fkind); data_f = {g: M.load(g, fkind) for g in groups}
    mods = [k for k, _, _ in blocks]
    data_m = {k: {g: sub_group(data_f[g], a, b) for g in groups} for k, a, b in blocks}
    cache = {k: {} for k in mods + [fkind]}
    for T in groups:
        S = [s for s in groups if CM.corp(s) != CM.corp(T)]
        st_set = {k: settings(k, T) for k in mods + [fkind]}
        info = {k: M5.pool_info(data_m[k] if k != fkind else data_f, S, 0.01, cache[k]) for k in st_set}
        ev = data_f[T]; ctx = CB.Ctx(ev, [0, 1], ["1", "5"], center="ind")
        for r in [0, 1]:
            pool = np.flatnonzero(~ctx.te[r]); yq = ev.y[pool]; te = np.flatnonzero(ctx.te[r]); yte = ev.y[te]
            pos_of = {int(i): j for j, i in enumerate(pool)}
            label_sets = {"0": np.array([], int), "5": np.array([pos_of[int(i)] for i in ctx.idx[(r, "5")]])}
            stu, Z, Zte, P = {}, {}, {}, {}
            for k in st_set:
                dat = data_m[k][T] if k != fkind else ev
                mu_pool = M.rows(dat.X, pool).mean(0)
                Xc = M.rows(dat.X, pool) - mu_pool
                gamma, rho_s, rho_m, cb, cp, _ = st_set[k]
                src, lin, n_of = info[k]
                st = Student(dat, Xc, src, lin, n_of, S, 0.01, gamma, rho_s, [rho_m]); Am = st.mats[rho_m][0]
                stu[k] = (st, rho_m, cb, cp); Z[k] = st.Xstd @ Am
                Zte[k] = ((M.rows(dat.X, te) - mu_pool - st.src.mu) / st.src.sd) @ Am
                P[k] = torch.softmax(st.L0, 1).cpu().numpy()
            pF = P[fkind]
            stF, rhoF, cbF, cpF = stu[fkind]
            row = {"world": fkind, "target": T, "rep": r, "fused_K0": uar_of(stF.W_std @ stF.mats[rhoF][1], stF.b_std, Zte[fkind], yte)}
            Wst, bst, c_star, _ = fit_star(Z[fkind], yq, ev.speaker.iloc[pool], stF.anchor(rhoF), 0.01, 1000 + r)
            row["ceiling"] = uar_of(Wst, bst, Zte[fkind], yte)
            solF = solve_all(Z[fkind], yq, {"p": pF}, label_sets, [("p", 0), ("p", 100)], stF, rhoF, 0.01, cpF, cbF)
            row["fused_K5"] = uar_of(*solF[("p", 0, "5")], Zte[fkind], yte)
            for K in ("0", "5"): row[f"selfdistill_K{K}"] = uar_of(*solF[("p", 100, K)], Zte[fkind], yte)
            # --- 組み直し：モダリティごとの生徒を鍛える
            teachers = {"byF": {k: pF for k in mods}}
            other = {k: CM.softmax_np(sum(lg(P[o]) for o in mods if o != k)) for k in mods}
            teachers["byO"] = other
            refuse_beliefs = {}
            for tname, tdict in teachers.items():
                for meff in MEFFS:
                    probs_te, scores_te, probs_pool = {K: [] for K in ("0", "5")}, {K: [] for K in ("0", "5")}, {K: [] for K in ("0", "5")}
                    for k in mods:
                        st, rho_m, cb, cp = stu[k]
                        sol = solve_all(Z[k], yq, {"t": tdict[k]}, label_sets, [("t", meff)], st, rho_m, 0.01, cp, cb)
                        for K in ("0", "5"):
                            W, b = sol[("t", meff, K)]
                            s_te = (Zte[k] @ W.T + b); scores_te[K].append(s_te); probs_te[K].append(torch.softmax(s_te, 1).cpu().numpy())
                            probs_pool[K].append(torch.softmax(Z[k] @ W.T + b, 1).cpu().numpy())
                    for K in ("0", "5"):
                        poe_te = CM.softmax_np(sum(lg(p) for p in probs_te[K])); sum_te = torch.softmax(sum(scores_te[K]), 1).cpu().numpy()
                        row[f"refuse_{tname}{meff}_poe_K{K}"] = CB.uar(yte, poe_te.argmax(1)); row[f"refuse_{tname}{meff}_sum_K{K}"] = CB.uar(yte, sum_te.argmax(1))
                        refuse_beliefs[(tname, meff, K)] = CM.softmax_np(sum(lg(p) for p in probs_pool[K]))
            # --- Dawid–Skene（ターゲットの候補の票から）
            votes = np.stack([P[k].argmax(1) for k in mods], 1)
            post, pi, Cs = dawid_skene(votes)
            votes_te = np.stack([torch.softmax(Zte[k] @ (stu[k][0].W_std @ stu[k][0].mats[stu[k][1]][1]).T + stu[k][0].b_std, 1).argmax(1).cpu().numpy() for k in mods], 1)
            post_te = ds_apply(votes_te, pi, Cs)
            row["ds_pool_uar"] = CB.uar(yq, post.argmax(1)); row["ds_test_uar"] = CB.uar(yte, post_te.argmax(1))
            rq, rb = post.argmax(1) == yq, pF.argmax(1) == yq
            row["ds_net_vs_fused"] = float((rq & ~rb).mean() - (~rq & rb).mean())
            row["fused_pool_uar"] = CB.uar(yq, pF.argmax(1))
            # --- 蒸留：DS と組み直しの見込みを融合の生徒へ
            bel = {"ds": post, "refF500": refuse_beliefs[("byF", 500, "0")], "refO100": refuse_beliefs[("byO", 100, "0")]}
            sol = solve_all(Z[fkind], yq, bel, label_sets, [(b, m) for b in bel for m in MEFFS], stF, rhoF, 0.01, cpF, cbF)
            for (b, m, K), (W, bb) in sol.items():
                row[f"distill_{b}{m}_K{K}"] = uar_of(W, bb, Zte[fkind], yte)
            rows.append(row)
        print(fkind, T, "済み", flush=True)
d = pd.DataFrame(rows); pd.set_option("display.width", 250)
d.to_csv("results/explore_relation/e60_refuse_check.csv", index=False)
for w, g in d.groupby("world"):
    m = g.drop(columns=["world", "target", "rep"]).mean() * 100
    print("==", w, " 融合 K=0", round(m.fused_K0, 2), " K=5", round(m.fused_K5, 2), " 上限", round(m.ceiling, 2))
    base = {"0": m.fused_K0, "5": m.fused_K5}
    out = {}
    for c in m.index:
        if "_K" in c and not c.startswith("fused"):
            b, K = c.rsplit("_K", 1); out.setdefault(b, {})[f"K={K}"] = round(m[c] - base[K], 2)
    print(pd.DataFrame(out).T.to_string())
    print("DS: pool UAR", round(m.ds_pool_uar, 2), " fused pool UAR", round(m.fused_pool_uar, 2), " net vs fused", round(m.ds_net_vs_fused, 2), " DS test UAR", round(m.ds_test_uar, 2))
