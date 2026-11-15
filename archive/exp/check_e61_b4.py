"""E61 の下調べ（予言 B4 の軽い確かめ）：E60-1 の音声の生徒で，私たちの形と一般化蒸留を比べる（AV，反復 0・1，K ∈ {0, 5}．候補のラベルは測定にだけ使う）。
  ours  ：融合の出発点 pF を，感情の釣り合いを取って弱いラベルにする（E60-1 の pF．m_eff ∈ {100, 500, 2000}）
  oursK ：K 本で学び直した融合の生徒 pF_K を先生にして，同じく釣り合いを取る（改良の候補．K=0 では ours と同じ）
  gd    ：一般化蒸留．pF_K を温度 T ∈ {1, 2, 4} でなだらかにし，釣り合いを取らずに弱いラベルにする．総重み ∈ {100, 500, 2000, n}（n は 1 本あたり重み 1）．固定版は T=2・n
  kd    ：温度付きの蒸留．先生を K=0 の pF にした gd（K 本を先生に使わない版）
  ours2 ：E60-2 の組み直し（pF_K で鍛えた音声と映像の生徒（m_in = 500）の確率の積 ref_K）を先生にして，釣り合いを取る（2 段目）
  gd2   ：同じ先生 ref_K を温度 1・釣り合いなしで（ours2 の釣り合いの効果を見る）．refdirect は ref_K そのもの（テストでの積）
どの候補も K 本の本当のラベル（present_weights）と合わせて E39 の出発点から学び直す（罰則・座標は E60 と同じ）。

    CUDA_VISIBLE_DEVICES= MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/check_e61_b4.py AFEW IEMOCAP-script
"""
import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, "exp"); sys.path.insert(0, "measure")
import committee as CM, compare_b1 as CB, logreg, map as M, meta5 as M5
from data import groups_for
from geom_e0 import settings
from teach_student39 import Student
from distill_belief import weak_rows, uar_of
from select_labelfree import present_weights
fkind = "fuse-audio-video"; blocks = [("audio_wavlm", 0, 1024), ("video", 1024, 1536)]
groups = groups_for(fkind); data_f = {g: M.load(g, fkind) for g in groups}
def sub_group(g, a, b): return M.Group(g.name, g.X[:, a:b], g.y, g.speaker, g.person)
data_m = {k: {g: sub_group(data_f[g], a, b) for g in groups} for k, a, b in blocks}
cache = {k: {} for k in ["audio_wavlm", "video", fkind]}
MEFFS, TEMPS, WEIGHTS = (100, 500, 2000), (1, 2, 4), (100, 500, 2000, "n")
targets = sys.argv[1:] or ["AFEW", "IEMOCAP-script"]


def temp_rows(P, T, w):
    """先生の確率 P（n×4）を温度 T でなだらかにし，釣り合いを取らない弱いラベルの行の重み（長さ 4n）。総重みは w（"n" なら 1 本あたり 1）。"""
    Pt = torch.as_tensor(P, dtype=logreg.DTYPE, device=logreg.DEVICE)
    PT = torch.softmax(torch.log(Pt.clamp(min=1e-12)) / T, 1)
    r = PT.T.reshape(-1)
    return r if w == "n" else r / r.sum() * w


def solve_rows(Z, yq, items, label_sets, st, rho_m, C, cp, cb):
    """items = [(key, 弱いラベルの行の重み (4n,) か None, K)] を束ねて解く（distill_belief.solve_all と同じ問）。"""
    n, d = Z.shape; dev = Z.device
    X = torch.cat([Z.repeat(4, 1), Z]); yt = M.labels(yq)
    y = torch.cat([torch.arange(4, device=dev).repeat_interleave(n), yt])
    weights, keys = [], []
    for key, w4, K in items:
        wt = torch.zeros(n, dtype=logreg.DTYPE, device=dev)
        idx = label_sets[K]
        if len(idx):
            it = torch.as_tensor(np.asarray(idx), device=dev); wt[it] = present_weights(yt[it])
        w = torch.cat([w4 if w4 is not None else torch.zeros(4 * n, dtype=logreg.DTYPE, device=dev), wt])
        if float(w.sum()) == 0.0:
            continue
        weights.append(w); keys.append(key)
    zero = torch.zeros(d, dtype=Z.dtype, device=dev)
    fits = logreg.fit_batch(X, y, [np.ones(len(y), bool)] * len(weights), (cp,), anchor=st.anchor(rho_m), anchor_c=C,
                            stats=(zero, torch.ones_like(zero)), Cb=[cb] * len(weights), weights=weights)
    return {k: (f.W[cp], f.b[cp]) for k, f in zip(keys, fits)}


rows = []
for T in targets:
    t0 = time.perf_counter()
    S = [s for s in groups if CM.corp(s) != CM.corp(T)]
    st_set = {k: settings(k, T) for k in ["audio_wavlm", "video", fkind]}
    ev = data_f[T]; ctx = CB.Ctx(ev, [0, 1], ["1", "5"], center="ind")
    info = {k: M5.pool_info(data_m[k] if k != fkind else data_f, S, 0.01, cache[k]) for k in st_set}
    for r in [0, 1]:
        pool = np.flatnonzero(~ctx.te[r]); yq = ev.y[pool]; te = np.flatnonzero(ctx.te[r]); yte = ev.y[te]
        pos_of = {int(i): j for j, i in enumerate(pool)}
        label_sets = {"0": np.array([], int), "5": np.array([pos_of[int(i)] for i in ctx.idx[(r, "5")]])}
        stu, Z, Zte, P = {}, {}, {}, {}
        for k in st_set:
            dat = data_m[k][T] if k != fkind else ev
            mu = M.rows(dat.X, pool).mean(0); Xc = M.rows(dat.X, pool) - mu
            gamma, rho_s, rho_m, cb, cp, _ = st_set[k]
            src, lin, n_of = info[k]
            st = Student(dat, Xc, src, lin, n_of, S, 0.01, gamma, rho_s, [rho_m]); Am = st.mats[rho_m][0]
            stu[k] = (st, rho_m, cb, cp); Z[k] = st.Xstd @ Am
            Zte[k] = ((M.rows(dat.X, te) - mu - st.src.mu) / st.src.sd) @ Am
            P[k] = torch.softmax(st.L0, 1).cpu().numpy()
        pF = P[fkind]
        stf, rho_f, cb_f, cp_f = stu[fkind]                                     # 融合の生徒を K 本で学び直す → pF_K
        solf = solve_rows(Z[fkind], yq, [("5", None, "5")], label_sets, stf, rho_f, 0.01, cp_f, cb_f)
        W5, b5 = solf["5"]
        pFK = {"0": pF, "5": torch.softmax(Z[fkind] @ W5.T + b5, 1).cpu().numpy()}
        ref, ref_te = {}, {}                                                    # 2 段目の先生：pF_K で鍛えた音声と映像の生徒の積
        for K in ("0", "5"):
            lg_pool, lg_te = 0, 0
            for v in ("audio_wavlm", "video"):
                stv, rho_v, cb_v, cp_v = stu[v]
                Wv, bv = solve_rows(Z[v], yq, [((v, K), weak_rows(pFK[K], 500), K)], label_sets, stv, rho_v, 0.01, cp_v, cb_v)[(v, K)]
                lg_pool = lg_pool + torch.log_softmax(Z[v] @ Wv.T + bv, 1)
                lg_te = lg_te + torch.log_softmax(Zte[v] @ Wv.T + bv, 1)
            ref[K] = torch.softmax(lg_pool, 1).cpu().numpy()
            ref_te[K] = torch.softmax(lg_te, 1).cpu().numpy()
        st, rho_m, cb, cp = stu["audio_wavlm"]
        n = len(pool)
        items = []
        for K in ("0", "5"):
            items.append((("none", "", 0, K), None, K))
            for m in MEFFS:
                items.append((("ours", "", m, K), weak_rows(pF, m), K))
                items.append((("oursK", "", m, K), weak_rows(pFK[K], m), K))
                items.append((("ours2", "", m, K), weak_rows(ref[K], m), K))
            for Tt in TEMPS:
                for w in WEIGHTS:
                    items.append((("gd", f"T{Tt}", w, K), temp_rows(pFK[K], Tt, w), K))
                    items.append((("kd", f"T{Tt}", w, K), temp_rows(pF, Tt, w), K))
            for w in WEIGHTS:
                items.append((("gd2", "T1", w, K), temp_rows(ref[K], 1, w), K))
            rows.append({"target": T, "rep": r, "method": "refdirect", "param": "", "w": 0, "K": K, "uar": CB.uar(yte, ref_te[K].argmax(1)), "n": n})
        sol = solve_rows(Z["audio_wavlm"], yq, items, label_sets, st, rho_m, 0.01, cp, cb)
        rows.append({"target": T, "rep": r, "method": "none", "param": "", "w": 0, "K": "0",
                     "uar": uar_of(st.W_std @ st.mats[rho_m][1], st.b_std, Zte["audio_wavlm"], yte), "n": n})
        for (meth, par, w, K), (W, bb) in sol.items():
            rows.append({"target": T, "rep": r, "method": meth, "param": par, "w": w, "K": K, "uar": uar_of(W, bb, Zte["audio_wavlm"], yte), "n": n})
        rows.append({"target": T, "rep": r, "method": "pFK_fused", "param": "", "w": 0, "K": "5", "uar": uar_of(W5, b5, Zte[fkind], yte), "n": n})
        rows.append({"target": T, "rep": r, "method": "pF_fused", "param": "", "w": 0, "K": "0",
                     "uar": uar_of(stf.W_std @ stf.mats[rho_f][1], stf.b_std, Zte[fkind], yte), "n": n})
    print(T, f"済み（{time.perf_counter() - t0:.0f} 秒）", flush=True)
d = pd.DataFrame(rows); pd.set_option("display.width", 250)
d.to_csv("results/explore_relation/e61_b4_check.csv", index=False)
m = d.groupby(["K", "method", "param", "w"]).uar.mean() * 100
for K in ("0", "5"):
    base = m.loc[(K, "none", "", 0)]
    print(f"\n== K={K}  E39（音声）= {base:.2f}  （ターゲット {sorted(d.target.unique())}，反復 0・1 の平均）")
    sub = m.loc[K].reset_index(); sub["diff"] = (sub.uar - base).round(2)
    print(sub.pivot_table(index=["method", "param"], columns="w", values="diff", sort=False).to_string())
    for meth in ("ours", "oursK", "ours2", "gd", "kd", "gd2"):
        s = sub[sub.method == meth]
        print(f"  {meth:6s} 最良（テストで選ぶ上限）: {s['diff'].max():+.2f}", end="")
        if meth in ("gd", "kd"):
            f = s[(s.param == "T2") & (s.w == "n")]
            print(f"   固定版 T=2・重み 1: {float(f['diff'].iloc[0]):+.2f}", end="")
        print()
