"""E61 の下調べ 2（研究者の承認 2026-10-22）：融合の生徒で，エントロピー最小化に「感情ごとの予測の総量をそろえる項」を付けると潰れが消えるか，
私たちの蒸留（2 段目の先生 ref）の後に仕上げとして足すと両者の良い方を超えるか（AVT，IEMOCAP 即興・MELD・MOSEI，反復 0・1，K ∈ {0, 5}．候補のラベルは測定にだけ使う）。
  none      ：E39（K 本だけ）
  ent       ：エントロピー最小化（候補のエントロピーの和 × u．u = m_eff/n）．出発点・罰則の中心は E39
  im        ：同じく，ただし候補の上の平均の予測のエントロピーを最大にする項を引く（情報量最大化．釣り合いの項が潰れを防ぐ）
  ours      ：2 段目の先生 ref（pFK を釣り合いを取って m_in=500 で蒸留したモダリティごとの生徒の確率の積）を，釣り合い 500／そのまま n で蒸留
  ours+ent  ：ours の解を新しい出発点（罰則の中心）にして ent を足す
  ours+im   ：同じく im を足す
総重み m_eff ∈ {2000, n}．出力に，候補の上の予測の感情ごとの割合の最小値（潰れの目印）も残す。

    MERDA_MAP_ROOT=results/fixc_map MERDA_GPU_FRACTION=0.4 ./.venv/bin/python exp/check_e61_im.py IEMOCAP-impro MELD MOSEI
"""
import os, sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, "exp"); sys.path.insert(0, "measure")
import committee as CM, compare_b1 as CB, logreg, map as M, meta5 as M5
from data import groups_for
from geom_e0 import settings
from distill_belief import uar_of, weak_rows
from baselines_distill import Prob, temp_rows, mass, label_sets_of, DT, DEV
from multimodal_distill import WORLDS, blocks, make_students, sub_group
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MERDA_GPU_FRACTION", "0.4")))
world = "fuse-audio-video-text"; mods = WORLDS[world]
groups = groups_for(world); data_f = {g: M.load(g, world) for g in groups}
data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(world)}
cache = {k: {} for k in mods + [world]}
targets = sys.argv[1:] or ["IEMOCAP-impro", "MELD", "MOSEI"]
KS2, MEFFS, REPS = ("0", "5"), ("2000", "n"), [0, 1]


def solve_ent(pr, K, u, balanced, anchor_Wb, init_Wb):
    """ラベルの重み wt[K] と，エントロピー（balanced なら情報量最大化）の項 u·n を，罰則の中心 anchor_Wb・出発点 init_Wb で解く。"""
    s, cp, n = pr.s, pr.s["cp"], pr.n
    C = s["st"].C
    anchor = s["st"].anchor(s["rho_m"]) if anchor_Wb is None else logreg.Fitted(pr.zero, pr.one, {C: anchor_Wb[0]}, {C: anchor_Wb[1]}, {C: 0}, {C: 0.0})
    init = logreg.Fitted(pr.zero, pr.one, {cp: init_Wb[0]}, {cp: init_Wb[1]}, {cp: 0}, {cp: 0.0})

    def fn(L, ix):
        lp = torch.log_softmax(L[:n], 2)
        p = lp.exp()
        Hc = -(p * lp).sum(2).sum(0)
        if not balanced:
            return u * Hc
        pbar = p.mean(0)
        Hm = -(pbar * pbar.clamp(min=1e-12).log()).sum(1)
        return u * (Hc - n * Hm)
    f = logreg.fit_batch(pr.X, pr.y, [np.ones(len(pr.y), bool)], (cp,), anchor=anchor, anchor_c=C, stats=(pr.zero, pr.one), Cb=[s["cb"]],
                         weights=[pr.wt[K]], extra=(fn, [u * n]), inits=[init])[0]
    return f.W[cp], f.b[cp]


rows = []
for T in targets:
    t0 = time.perf_counter()
    S = [s for s in groups if CM.corp(s) != CM.corp(T)]
    sets = {k: settings(k, T) for k in mods + [world]}
    info = {k: M5.pool_info(data_m[k] if k != world else data_f, S, 0.01, cache[k]) for k in sets}
    ev = data_f[T]; ctx = CB.Ctx(ev, REPS, ["1", "5", "10", "25"], center="ind")
    for r in REPS:
        pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r]); yq, yte = ev.y[pool], ev.y[te]
        label_sets = label_sets_of(ctx, r, pool)
        stu = make_students(world, T, data_f, data_m, S, pool, te, sets, info)
        prs = {k: Prob(stu[k], yq, label_sets) for k in mods + [world]}
        f = prs[world]; n = f.n
        none = f.solve([(K, None, K) for K in KS2])
        P0 = {K: f.probs(none[K], f.s["Z"]) for K in KS2}
        lg = {K: 0 for K in KS2}
        for m in mods:                                                            # 2 段目の先生 ref
            sol = prs[m].solve([(K, weak_rows(P0[K], 500), K) for K in KS2])
            for K in KS2:
                lg[K] = lg[K] + prs[m].probs(sol[K], prs[m].s["Z"]).clamp(min=1e-12).log()
        ref = {K: torch.softmax(lg[K], 1) for K in KS2}
        ours = f.solve([((K, "bal500"), weak_rows(ref[K], 500.0), K) for K in KS2] + [((K, "rawn"), temp_rows(ref[K], 1, "n"), K) for K in KS2])

        def rec(method, meff, K, Wb):
            Pp = f.probs(Wb, f.s["Z"]); share = Pp.mean(0)
            rows.append({"target": T, "rep": r, "method": method, "meff": meff, "K": K, "uar": uar_of(Wb[0], Wb[1], f.s["Zte"], yte),
                         "min_share": float(share.min()), "max_share": float(share.max()), "n": n})
        for K in KS2:
            rec("none", "0", K, none[K])
            rec("ours:bal500", "500", K, ours[(K, "bal500")])
            rec("ours:rawn", "n", K, ours[(K, "rawn")])
            for meff in MEFFS:
                u = mass(meff, n) / n
                rec("ent", meff, K, solve_ent(f, K, u, False, None, none[K]))
                rec("im", meff, K, solve_ent(f, K, u, True, None, none[K]))
                for oname, okey in (("ours:bal500", (K, "bal500")), ("ours:rawn", (K, "rawn"))):
                    o = ours[okey]
                    rec(f"{oname}+ent", meff, K, solve_ent(f, K, u, False, o, o))
                    rec(f"{oname}+im", meff, K, solve_ent(f, K, u, True, o, o))
        print(T, r, f"済み（{time.perf_counter() - t0:.0f} 秒）", flush=True)
d = pd.DataFrame(rows); pd.set_option("display.width", 250)
d.to_csv("results/explore_relation/e61_im_check.csv", index=False)
for T in targets:
    g = d[d.target == T]
    m = g.groupby(["K", "method", "meff"]).agg(uar=("uar", "mean"), min_share=("min_share", "mean")).reset_index()
    for K in KS2:
        base = float(m[(m.K == K) & (m.method == "none")].uar.iloc[0]) * 100
        sub = m[(m.K == K) & (m.method != "none")].copy(); sub["diff"] = (sub.uar * 100 - base).round(2); sub["min_share"] = sub.min_share.round(3)
        print(f"\n== {T} K={K}  E39 = {base:.2f}（反復 0・1）  値は差，min_share は候補の上の予測の感情の割合の最小（0.25 が釣り合い）")
        print(sub.pivot_table(index="method", columns="meff", values=["diff", "min_share"], sort=False).to_string())
