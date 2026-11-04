"""dai1_stats.detect を，台本の関数を使わずに作り直す（CPU）。
(1) 族 1〜4 の 44 セルの判定を npz から取り，分割の表の塊で jackknife（AFEW・IEMOCAP・MELD）か塊の引き直し（MOSEI．乱数は別）で分散を作る．
(2) 種の数 R，同じ水準の δ，検出できる差，半幅，記述に回すセル，同じ水準を出しうるセルを作る．
(3) 効果の見込みを，rows.csv の UAR（判定とは別の経路）と自前の選び直しで作る。"""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as st
ROOT = Path("/home/snics/MER-DA"); P = ROOT / "DAI/checks/DAI-1/code_run_post"; U = ROOT / "results/dai1/pseudo/units"
fz = json.load(open(ROOT / "results/dai1/select/frozen.json"))
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]; ST = ["A", "V", "T", "AVT"]
RHO = [0.0625, 0.25, 1.0, 4.0]; A = [1, 4, 16]; CR = [0.0, 0.0625, 0.25, 1.0, 4.0]
FAMS = {1: ("prop", "none", (0, 5, 25)), 2: ("rolesep", "prop", (5, 25)), 3: ("prop", "partner", (0, 5, 25)), 4: ("prop", "default", (0, 5, 25))}
sp = {c: pd.read_csv(ROOT / f"DAI/splits/{c}.csv", keep_default_na=False).set_index("sample_id") for c in C}
geff = {c: float((lambda n: n.sum() ** 2 / (n ** 2).sum())(sp[c].cluster.value_counts().to_numpy())) for c in C}
def fl(x): return float(x)
def preds(T, c, f, r):
    z1 = np.load(U / f"{T}_{c}_f{f}_r{r}_p1.npz"); z3 = np.load(U / f"{T}_{c}_f{f}_r{r}_p3.npz")
    F = fz[T]; out = {}
    for s in ST:
        for K in (0, 5, 25):
            v = F["main"][str(K)]; j = F["select"][f"{s}|{K}|joint|{v}"]; items = [json.loads(x) for x in z3[f"p3_{s}_K{K}_items"]]
            def item(**w):
                h = [i for i, it in enumerate(items) if all(k in it and np.isclose(fl(it[k]), fl(vv)) if not isinstance(vv, str) else it.get(k) == vv for k, vv in w.items())]
                assert len(h) == 1, (w, len(h)); return z3[f"p3_{s}_K{K}_pred"][h[0]]
            out[(s, K, "prop")] = item(method="joint", teacher=v, rho=j["rho"], a=j["a"])
            out[(s, K, "default")] = item(method="joint", teacher="one", rho=1.0, a=1)
            if K == 0: out[(s, K, "none")] = z1[f"p1_{s}_start_pred"]
            else:
                cw, cb = F["tether"][f"{s}|{K}"]; it1 = [str(x) for x in z1[f"p1_{s}_items"]]
                out[(s, K, "none")] = z1[f"p1_{s}_pred"][it1.index(f"{K}|{cw}|{cb}")]
            if K > 0:
                rs = F["select"][f"{s}|{K}|rolesep|{v}"]
                ja = [(tv, x) for tv in ("sel", "one") for x in RHO].index((v, fl(rs["rho"])))
                out[(s, K, "rolesep")] = z3[f"p3r_{s}_K{K}_pred_b"][ja * 3 + A.index(int(fl(rs["a"])))]
            pt = F["partner"][f"{s}|{K}"]; pm, pv = pt["method"], pt["version"]
            pp = {k: x for k, x in F["select"][f"{s}|{K}|{pm}|{pv}"].items() if k != "pseudo_uar"}
            if pm == "cotrain" and s != "AVT": out[(s, K, "partner")] = z3[f"p3c_K{K}_{s}_pred"][RHO.index(fl(pp["rho"]))]
            elif pm == "coreg": out[(s, K, "partner")] = z3[f"p3g_K{K}_{s}_pred"][CR.index(fl(pp["rho"]))]
            else: out[(s, K, "partner")] = item(method=pm, teacher=pv, **pp)
    return z1["sample_test"].astype(str), out
def cnt(y, p, cl, names):
    ix = pd.Index(names).get_indexer(cl); cor = np.zeros((len(names), 4)); tot = np.zeros((len(names), 4))
    np.add.at(tot, (ix, y), 1); h = p == y; np.add.at(cor, (ix[h], y[h]), 1); return cor, tot
def uar(cor, tot, strict):
    c, t = cor, tot
    if strict and (t.sum() < 30 or (t == 0).any()): return np.nan
    ok = t > 0; return 100 * (c[ok] / t[ok]).mean()
rng = np.random.default_rng(7)
res = []
for T in C:
    for c in C:
        if c == T: continue
        s_ = sp[c]; D = {}
        for f in range(5):
            for r in (0, 1):
                sid, pr = preds(T, c, f, r); y = s_.loc[sid, "label_4"].to_numpy(); cl = s_.loc[sid, "cluster"].to_numpy()
                names = np.unique(cl)
                for k, p in pr.items(): D[(k, f, r)] = (names, *cnt(y, p.astype(int), cl, names))
        for fam, (Aa, Bb, ks) in FAMS.items():
            for s in ST:
                for K in ks:
                    d = {f: [uar(D[((s, K, Aa), f, r)][1].sum(0), D[((s, K, Aa), f, r)][2].sum(0), False) - uar(D[((s, K, Bb), f, r)][1].sum(0), D[((s, K, Bb), f, r)][2].sum(0), False) for r in (0, 1)] for f in range(5)}
                    est = np.mean([np.mean(v) for v in d.values()]); s2 = {f: np.var(v, ddof=1) for f, v in d.items()}
                    vsa = sum(s2[f] / 2 for f in range(5)) / 25
                    if geff[c] >= 20:
                        bs = []
                        Wbs = {f: rng.multinomial(len(D[((s, K, Aa), f, 0)][0]), np.full(len(D[((s, K, Aa), f, 0)][0]), 1 / len(D[((s, K, Aa), f, 0)][0])), size=2000) for f in range(5)}
                        acc = np.zeros(2000); nv = np.zeros(2000)
                        for f in range(5):
                            W = Wbs[f].astype(float); dd = []
                            for r in (0, 1):
                                _, ca, ta = D[((s, K, Aa), f, r)]; _, cb, tb = D[((s, K, Bb), f, r)]
                                CA, TA, CB, TB = W @ ca, W @ ta, W @ cb, W @ tb
                                with np.errstate(all="ignore"):
                                    ua = 100 * (CA / TA).mean(1); ub = 100 * (CB / TB).mean(1)
                                bad = (TA.sum(1) < 30) | (TA == 0).any(1)
                                dd.append(np.where(bad, np.nan, ua - ub))
                            m = np.nanmean(np.stack(dd), 0); ok = np.isfinite(m); acc[ok] += m[ok]; nv[ok] += 1
                        b = acc[nv > 0] / nv[nv > 0]; vjk = np.var(b, ddof=1); vs = vsa; dropped = set()
                    else:
                        allc = sorted(set().union(*[set(D[((s, K, Aa), f, 0)][0]) for f in range(5)])); jk = []; dropped = set()
                        for g in allc:
                            vals = []
                            for f in range(5):
                                names = D[((s, K, Aa), f, 0)][0]; vv = []
                                for r in (0, 1):
                                    _, ca, ta = D[((s, K, Aa), f, r)]; _, cb, tb = D[((s, K, Bb), f, r)]
                                    if g in names:
                                        i = int(np.flatnonzero(names == g)[0])
                                        a_ = uar(ca.sum(0) - ca[i], ta.sum(0) - ta[i], True); b_ = uar(cb.sum(0) - cb[i], tb.sum(0) - tb[i], True)
                                        if np.isfinite(a_) and np.isfinite(b_): vv.append(a_ - b_)
                                    else: vv.append(uar(ca.sum(0), ta.sum(0), False) - uar(cb.sum(0), tb.sum(0), False))
                                if vv: vals.append(np.mean(vv))
                                else: dropped.add(f)
                            jk.append(np.mean(vals))
                        jk = np.array(jk); G = len(allc); vjk = (G - 1) / G * ((jk - jk.mean()) ** 2).sum()
                        vs = sum(s2[f] / 2 for f in range(5) if f not in dropped) / 25
                    res.append(dict(T=T, c=c, fam=fam, s=s, K=K, est=est, v_jk=vjk, v_seed=vs, v_seed_all=vsa, ndrop=len(dropped)))
        print("done", T, c, flush=True)
R = pd.DataFrame(res); R.to_csv(P / "detect_rebuild_cells.csv", index=False)
R["v_cl"] = (R.v_jk - (R.v_seed_all - R.v_seed)).clip(lower=0)
# (2) R，δ，セル
p1 = pd.read_pickle(P / "rows_p1.pkl"); p3 = pd.read_pickle(P / "rows_p3.pkl")
out = {}
for X in C:
    d = R[R.c == X]
    nse = 20; alt = 20
    for Rr in (10, 20):
        k = d[d.K > 0]; vs = k.v_seed_all * 2 / Rr; sh = np.where(k.v_cl + vs > 0, vs / (k.v_cl + vs), 0)
        if np.median(sh) <= 0.25: nse = Rr; break
    for Rr in (10, 20):                          # 別の読み：3 つの T で平均したセルの中央値
        g = d[d.K > 0].groupby(["fam", "s", "K"])[["v_cl", "v_seed_all"]].mean(); vs = g.v_seed_all * 2 / Rr
        if np.median(vs / (g.v_cl + vs)) <= 0.25: alt = Rr; break
    V = []
    for T in C:
        if T == X: continue
        for s in ST:
            for K in (5, 25):
                cw, cb = fz[T]["tether"][f"{s}|{K}"]
                q = p1[(p1["T"] == T) & (p1.c == X) & (p1.method == "none") & (p1.kind == s) & (p1.K == K) & np.isclose(p1.Cw, cw) & np.isclose(p1.Cb, cb)]
                V.append(q.groupby("fold").uar.var(ddof=1).mean())
    delta = 2 * np.sqrt(np.mean(V) / (5 * nse))
    df_ = max(1, int(np.floor(geff[X])) - 1)
    q95 = 1.959964 if geff[X] >= 20 else st.t.ppf(0.975, df_)
    det = fz[X]["detect"]; bad = []
    for (fam, s, K), g in d.groupby(["fam", "s", "K"]):
        vc = g.v_cl.mean(); vs = g.v_seed_all.mean() * 2 / nse if K > 0 else 0.0; se = np.sqrt(vc + vs)
        m = 12 if fam in (1, 3, 4) else 8
        qf = st.norm.ppf(1 - 0.05 / m / 2) if geff[X] >= 20 else st.t.ppf(1 - 0.05 / m / 2, df_)
        key = f"{fam}|{s}|{K}"; cell = det["cells"][key]
        tol = 0.10 if geff[X] >= 20 else 1e-6           # MOSEI は引き直しの乱数が別
        if abs(se - cell["se"]) > tol * max(cell["se"], 1e-9): bad.append((key, "se", se, cell["se"]))
        if not np.isclose(cell["detectable_first"], qf * cell["se"], rtol=1e-9): bad.append((key, "detfirst"))
        if not np.isclose(cell["halfwidth"], q95 * cell["se"], rtol=1e-6): bad.append((key, "halfwidth"))
        if cell["same_level_possible"] != bool(K > 0 and cell["halfwidth"] < det["delta_same"]): bad.append((key, "same"))
        if cell["descriptive"] != bool(cell["detectable_first"] > abs(cell["expected_effect"])): bad.append((key, "desc"))
        # 自前の se で記述の判定が変わるか
        if (qf * se > abs(cell["expected_effect"])) != cell["descriptive"]: bad.append((key, "desc-flip-with-rebuilt-se", qf * se, abs(cell["expected_effect"])))
    out[X] = dict(n_seeds=nse, n_seeds_alt_reading=alt, frozen_n_seeds=det["n_seeds_eval"], delta=delta, frozen_delta=det["delta_same"], bad=bad)
for X, v in out.items(): print(X, {k: v[k] for k in v if k != "bad"}, "problems", len(v["bad"]), v["bad"][:6])
