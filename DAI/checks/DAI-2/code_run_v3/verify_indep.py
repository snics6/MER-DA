"""gate-code（DAI-2 code_v3）：dai2_stats.py の pseudo の出力を，保存の npz・csv と DAI/splits だけから独立に作り直して照らす。
dai1_stats・dai2_stats の関数は使わない（格子の定義だけ dai2・dai2_grad から読む）．成績の値は画面に出さない（差の最大と件数だけ）。
使い方：python verify_indep.py <T> <c> <dai2_stats の出力の置き場>"""
import itertools, json, math, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
from scipy import stats as st
ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai2, dai2_grad as G  # noqa: E402  （格子と既定値の定義だけ）

T, C, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
D1 = ROOT / "results/dai1/pseudo/units"; D2 = ROOT / "results/dai2/pseudo/units"
fz1 = json.load(open(ROOT / "DAI/frozen/dai1_frozen.json"))[T]
fz2 = json.load(open(ROOT / "DAI/frozen/dai2_frozen.json"))[T]
sp = pd.read_csv(ROOT / f"DAI/splits/{C}.csv", keep_default_na=False)
Y = dict(zip(sp.sample_id.astype(str), sp.label_4.astype(int))); CL = dict(zip(sp.sample_id.astype(str), sp.cluster.astype(str)))
n = sp.cluster.value_counts().to_numpy(); GEFF = n.sum() ** 2 / (n ** 2).sum()
assert GEFF < 20, "この検算は jackknife のコーパスだけ"
DF = max(1, int(math.floor(GEFF)) - 1)
UNITS = [(f, r) for f in range(5) for r in (0, 1)]
GR = dict(dai2.GRIDS) | dict(G.GRIDS)


def close(a, b):
    return abs(float(a) - float(b)) < 1e-9


def match(it, want):
    return all(k in it and close(it[k], v) for k, v in want.items())


class U:
    def __init__(s, f, r):
        b1 = D1 / f"{T}_{C}_f{f}_r{r}"; b2 = D2 / f"{T}_{C}_f{f}_r{r}"
        s.p1, s.p3 = np.load(f"{b1}_p1.npz"), np.load(f"{b1}_p3.npz")
        s.z = {p: np.load(f"{b2}_{p}.npz") for p in ("lin", "coral", "map", "xmuda")}
        s.rows = {p: pd.read_csv(f"{b2}_{p}.csv", low_memory=False) for p in ("map", "xmuda")}
        sid = s.p1["sample_test"].astype(str)
        s.y = np.array([Y[x] for x in sid]); s.cl = np.array([CL[x] for x in sid]); s.f, s.r = f, r

    def items(s, z, ik, pk):
        return [(json.loads(str(i)), p) for i, p in zip(z[ik], z[pk])]

    def pred(s, m, st_, K, pt):
        if m == "prop":
            v = fz1["main"][str(K)]; j = fz1["select"][f"{st_}|{K}|joint|{v}"]
            h = [p for i, p in s.items(s.p3, f"p3_{st_}_K{K}_items", f"p3_{st_}_K{K}_pred")
                 if i["method"] == "joint" and i["teacher"] == v and close(i["rho"], j["rho"]) and close(i["a"], j["a"])]
            assert len(h) == 1; return h[0]
        if m == "none":
            if K == 0:
                return s.p1[f"p1_{st_}_start_pred"]
            cw, cb = fz1["tether"][f"{st_}|{K}"]
            its = [str(x) for x in s.p1[f"p1_{st_}_items"]]
            return s.p1[f"p1_{st_}_pred"][its.index(f"{K}|{cw}|{cb}")]
        z = s.z
        if m in ("self", "own", "entmin", "im", "lp"):
            pre = {"self": "self", "own": "lin", "lp": "lin", "entmin": "nc", "im": "nc"}[m]
            h = [p for i, p in s.items(z["lin"], f"{pre}_{st_}_K{K}_items", f"{pre}_{st_}_K{K}_pred") if i["method"] == m and match(i, pt)]
        elif m in ("t3a", "t3a_b"):
            j = list(dai2.T3A_M).index(int(pt["M"]))
            return z["lin"][f"t3a_{st_}_K{K}_pred" + ("_b" if m == "t3a_b" else "")][j]
        elif m == "st":
            h = [p for i, p in s.items(z["lin"], f"st_{st_}_items", f"st_{st_}_pred") if int(i["K"]) == K and match(i, pt)]
        elif m == "coral":
            if K == 0:
                return z["coral"][f"coral_{st_}_start_pred"][list(G.CORAL_R).index(int(pt["r"]))]
            h = [p for i, p in s.items(z["coral"], f"coral_{st_}_items", f"coral_{st_}_pred")
                 if i["method"] == "coral" and int(i["K"]) == K and match(i, pt)]
        elif m in ("dann", "mme", "mapctl"):
            h = [p for i, p in s.items(z["map"], f"map_{st_}_items", f"map_{st_}_pred") if i["method"] == m and int(i["K"]) == K and match(i, pt)]
        elif m == "xmuda":
            h = [p for i, p in s.items(z["xmuda"], "xm_items", f"xm_{st_}_pred") if int(i["K"]) == K and match(i, pt)]
        assert len(h) == 1, (m, st_, K, pt, len(h))
        return h[0]


def uar(y, p, msk=None):
    if msk is not None:
        y, p = y[msk], p[msk]
    if len(y) < 30 or len(set(y)) < 4:
        return float("nan")
    return 100 * np.mean([(p[y == k] == k).mean() for k in range(4)])


def uar_full(y, p):
    return 100 * np.mean([(p[y == k] == k).mean() for k in range(4) if (y == k).any()])


def interval(us, pa, pb):
    """設計 9.3 節の文どおりの jackknife と種の項（G_eff < 20）。"""
    folds = sorted({u.f for u in us})
    d = {f: [uar_full(u.y, pa[u]) - uar_full(u.y, pb[u]) for u in us if u.f == f] for f in folds}
    th = np.mean([np.mean(d[f]) for f in folds])
    gs = sorted({g for u in us for g in u.cl})
    jk, D = [], set()
    for g in gs:
        per = {}
        for f in folds:
            vals = []
            for u in [u for u in us if u.f == f]:
                if g in set(u.cl):
                    m = u.cl != g
                    a, b = uar(u.y, pa[u], m), uar(u.y, pb[u], m)
                    if np.isfinite(a) and np.isfinite(b):
                        vals.append(a - b)
                else:
                    vals.append(uar_full(u.y, pa[u]) - uar_full(u.y, pb[u]))
            if vals:
                per[f] = np.mean(vals)
            else:
                D.add(f)
        jk.append(np.mean(list(per.values())))
    jk = np.array(jk); Gn = len(gs)
    vjk = (Gn - 1) / Gn * ((jk - jk.mean()) ** 2).sum()
    vs = sum(np.var(d[f], ddof=1) / len(d[f]) for f in folds if f not in D and len(d[f]) > 1) / len(folds) ** 2
    se = math.sqrt(vjk + vs); q = st.t.ppf(0.975, DF)
    return th, th - q * se, th + q * se, se


def selpt(m, s, K):
    if m == "st" and K == 0:
        return {"beta": 1.0}
    return {k: v for k, v in fz2["select"][f"{s}|{K}|{m}"].items() if k != "pseudo_uar"}


us = [U(f, r) for f, r in UNITS]
res = {}
# 1. 提案 − 手法（表 1）
T1 = pd.read_csv(OUT / "prop_minus_method.csv"); T1 = T1[T1.c == C]
mx, nck = 0.0, 0
cases = [(m, s, K) for m in ("self", "entmin", "im", "t3a", "t3a_b", "lp", "own", "st", "coral", "dann", "mme", "mapctl", "xmuda")
         for s in (("A", "V", "T") if m == "xmuda" else ("A", "V", "T", "AVT")) for K in ((1, 5, 10, 25) if m == "lp" else (0, 1, 5, 10, 25))]
for m, s, K in cases:
    pt = selpt(m, s, K)
    pa = {u: u.pred("prop", s, K, None) for u in us}; pb = {u: u.pred(m, s, K, pt) for u in us}
    th, lo, hi, se = interval(us, pa, pb)
    row = T1[(T1.student == s) & (T1.K == K) & (T1.method == m)].iloc[0]
    mx = max(mx, abs(th - row.effect), abs(lo - row.lo), abs(hi - row.hi), abs(se - row.se)); nck += 1
res["表1 提案−手法（θ・lo・hi・se）"] = (nck, mx)
# 4. DA の項（表 5．役 default・ref）
T5 = pd.read_csv(OUT / "da_term.csv"); T5 = T5[T5.c == C]
bad = 0; n5 = 0
for m in ("dann", "mme", "xmuda"):
    for s in (("A", "V", "T") if m == "xmuda" else ("A", "V", "T", "AVT")):
        for K in (0, 1, 5, 10, 25):
            pt = dict(G.XM_REF) if m == "xmuda" else dict(G.DEFAULTS[m]); tok = "ref" if m == "xmuda" else "default"
            cpt = {k: pt[k] for k in ("lr", "iters", "mu")}
            cnt = [0, 0, 0]
            for u in us:
                part = "xmuda" if m == "xmuda" else "map"
                R = u.rows[part]; RM = u.rows["map"]
                sel = lambda d, meth, p: d[(d.method == meth) & (d.student == s) & (d.K == K)
                                           & np.logical_and.reduce([np.isclose(d[k].astype(float), v) for k, v in p.items()])]
                pr = sel(R, m, pt); rm = sel(R, m + "_rep", pt); rm = rm[rm.roles.astype(str).str.contains(tok)]
                rc = sel(RM, "mapctl_rep", cpt); rc = rc[rc.roles.astype(str).str.contains(f"ctl_of_{m}_{tok}")]
                assert len(pr) == 1 and len(rm) == 1 and len(rc) == 1, (m, s, K, len(pr), len(rm), len(rc))
                dc, em, ec = float(pr.dis_ctl.iloc[0]), float(rm.dis_rep.iloc[0]), float(rc.dis_rep.iloc[0])
                cnt[0 if dc <= ec + 0.01 else (1 if dc <= em else 2)] += 1
            row = T5[(T5.method == m) & (T5.student == s) & (T5.K == K) & (T5.role == tok)].iloc[0]
            bad += int([row.n_i, row.n_ii, row.n_iii] != cnt) + int(bool(row.not_worked) != (cnt[0] > 5)); n5 += 1
res["表5 DA の項の (i)〜(iii) と not_worked（役 default・ref）"] = (n5, bad)
# 6. 揺れ（表 6．dann の default：入れ替わりの平均と UAR の差の平均）
T6 = pd.read_csv(OUT / "rep_shift.csv"); T6 = T6[T6.c == C]
mx6 = 0.0
for s in ("A", "V", "T", "AVT"):
    for K in (0, 1, 5, 10, 25):
        pt = dict(G.DEFAULTS["dann"]); fl, du = [], []
        for u in us:
            prod = u.pred("dann", s, K, pt)
            h = [p for i, p in u.items(u.z["map"], f"maprep_{s}_items", f"maprep_{s}_pred")
                 if i["method"] == "dann" and int(i["K"]) == K and match(i, pt) and "default" in i["roles"]]
            assert len(h) == 1
            fl.append((h[0] != prod).mean()); du.append(uar_full(u.y, h[0]) - uar_full(u.y, prod))
        row = T6[(T6.method == "dann") & (T6.student == s) & (T6.K == K) & (T6.role == "default")].iloc[0]
        mx6 = max(mx6, abs(np.mean(fl) - row.flip_mean), abs(np.mean(du) - row.uar_diff_mean), abs(np.max(fl) - row.flip_max))
res["表6 dann default の入れ替わりと UAR の差"] = (20, mx6)
# 7. 印（表 7．dann と mme の全部の問．学習率ごと）
MK = pd.read_csv(OUT / "marks.csv"); MK = MK[MK.c == C]
bad7 = 0; n7 = 0
allmap = pd.concat([u.rows["map"] for u in us]); allxm = pd.concat([u.rows["xmuda"].assign(f_=u.f, r_=u.r) for u in us])
for (m, lr, s, K), g in allmap[allmap.method.isin(["dann", "mme", "mapctl"])].groupby(["method", "lr", "student", "K"]):
    ne = int((g.cdiff_after.astype(float) > 100 * g.cdiff_before.astype(float)).sum())
    nd = int((g.dloss_after.astype(float) > 10 * math.log(2)).sum())
    row = MK[(MK.method == m) & np.isclose(MK.lr, lr) & (MK.student == s) & (MK.K == K)].iloc[0]
    bad7 += int(row.n_problems != len(g)) + int(row.n_expand != ne) + int(m == "dann" and row.n_disc != nd); n7 += 1
xm = allxm[allxm.method == "xmuda"]
amax = xm.groupby(["f_", "r_", "lr", "iters", "kap", "mu", "K"]).anorm.max()
for (lr, s, K), g in xm.groupby(["lr", "student", "K"]):
    ne = sum(int(amax.loc[(a.f_, a.r_, a.lr, a.iters, a.kap, a.mu, a.K)] > 100) for a in g.itertuples())
    row = MK[(MK.method == "xmuda") & np.isclose(MK.lr, lr) & (MK.student == s) & (MK.K == K)].iloc[0]
    bad7 += int(row.n_expand != ne) + int(row.n_problems != len(g)); n7 += 1
res["表7 印の数（dann・mme・mapctl・xmuda）"] = (n7, bad7)
# 8. 取りこぼし（表 8．self・t3a・dann の θ が最大の点と取りこぼし．同点の数も数える）
RG = pd.read_csv(OUT / "regret.csv"); RG = RG[RG.c == C]
mx8, n8 = 0.0, 0
for m in ("self", "t3a", "dann", "coral"):
    keys = [k for k, _ in GR[m]]
    for s in ("A", "V", "T", "AVT"):
        for K in (0, 5, 25):
            pts = [dict(zip(keys, c)) for c in itertools.product(*[v for _, v in GR[m]])]
            if m == "mme":
                pts = [p for p in pts if p["sw"] == 0]
            th = [np.mean([np.mean([uar_full(u.y, u.pred(m, s, K, p)) for u in us if u.f == f]) for f in range(5)]) for p in pts]
            sp_ = selpt(m, s, K); isel = [i for i, p in enumerate(pts) if all(close(p[k], sp_[k]) for k in keys)][0]
            row = RG[(RG.method == m) & (RG.student == s) & (RG.K == K)].iloc[0]
            mx8 = max(mx8, abs(max(th) - th[isel] - row.regret)); n8 += 1
res["表8 取りこぼし（self・t3a・dann・coral，K 0・5・25）"] = (n8, mx8)
# 9. 符号の反転（表 9．群 method:dann の「上回る」の 32 通り）
NU = pd.read_csv(OUT / "null_counts.csv"); NU = NU[(NU.c == C) & (NU.group == "method:dann") & (NU.what == "above")]
cnts = np.zeros(32, int)
for i, pat in enumerate(itertools.product((1, -1), repeat=5)):
    for s in ("A", "V", "T", "AVT"):
        for K in (0, 1, 5, 10, 25):
            pt = selpt("dann", s, K)
            pa = {u: (u.pred("prop", s, K, None) if pat[u.f] == 1 else u.pred("dann", s, K, pt)) for u in us}
            pb = {u: (u.pred("dann", s, K, pt) if pat[u.f] == 1 else u.pred("prop", s, K, None)) for u in us}
            cnts[i] += int(interval(us, pa, pb)[1] > 0)
row = NU.iloc[0]
res["表9 dann の 32 通り（平均・上側 5% 点・最大・観測以上の割合の食い違い）"] = (
    4, int(not np.isclose(cnts.mean(), row.null_mean)) + int(np.sort(cnts)[-2] != row.null_q95) + int(cnts.max() != row.null_max)
    + int(not np.isclose((cnts >= cnts[0]).mean(), row.frac_ge_observed)))
# 10. K = 0 の mapctl と none
K0 = pd.read_csv(OUT / "k0_ctl_none.csv"); K0 = K0[K0.c == C]
b10 = 0
for s in ("A", "V", "T", "AVT"):
    for c_ in itertools.product(*[v for _, v in GR["mapctl"]]):
        pt = dict(zip([k for k, _ in GR["mapctl"]], c_))
        ag = [float((u.pred("mapctl", s, 0, pt) == u.pred("none", s, 0, None)).mean()) for u in us]
        row = K0[(K0.student == s) & (K0.point == json.dumps({k: float(v) for k, v in pt.items()}))].iloc[0]
        b10 += int(row.n_units_identical != sum(a == 1.0 for a in ag)) + int(not close(row.agree_min, min(ag)))
res["表4 付き K = 0 の mapctl と none の一致（32 点）"] = (32, b10)
for k, v in res.items():
    print(f"{k}: 照らした数 {v[0]}，食い違い（差の最大か数） {v[1]}")
