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
T1 = pd.read_csv(OUT / "prop_minus_method.csv"); T1 = T1[T1.c == C]
mx = 0; n = 0; mxb = 0
for m in ("self","t3a","own","st","coral","dann","mme","mapctl","xmuda"):
    for s in (("A","V","T") if m=="xmuda" else ("A","V","T","AVT")):
        for K in (0,5,25):
            pt = selpt(m, s, K)
            d = {f: [uar_full(u.y, u.pred("prop", s, K, None)) - uar_full(u.y, u.pred(m, s, K, pt)) for u in us if u.f == f] for f in range(5)}
            th = np.mean([np.mean(v) for v in d.values()])
            row = T1[(T1.student==s)&(T1.K==K)&(T1.method==m)].iloc[0]
            vs = sum(np.var(v, ddof=1)/len(v) for v in d.values() if len(v)>1)/25
            mx = max(mx, abs(th-row.effect)); mxb = max(mxb, abs(vs-row.v_seed)); n += 1
print("MOSEI(c) θ と種の項: 照らした数", n, "θ の最大差", mx, "v_seed の最大差", mxb, "ci_rule", T1.ci_rule.unique())
