"""dai20_posthoc_intercept.py の検算（下調べの単位で）．
1. 点推定と区間を，下調べの集計 effects.csv の (a, prod, where) の行と照らす（全体と向きの分）．
2. 切片の分（差の差）の区間を，effect_stats を使わずに単位ごとの差の差から直接作り直して照らす（jackknife のコーパス）．
   MOSEI（塊の重みの引き直し）は，同じ乱数の鍵で 4 つの UAR を別々に作って差の差を取り，同じ値になるかを見る．
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1_stats as ST  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai_data as DD  # noqa: E402

ROOT = Path("/home/snics/MER-DA")
OUT = ROOT / "DAI/checks/DAI-20/post_tools_run/intercept_probe"
ef = pd.read_csv(OUT / "intercept_direction.csv")
pe = pd.read_csv(ROOT / "results/dai16_probe/summary/effects.csv", low_memory=False)

# ---- 1. 集計の段との照合（全体：v@L − prod@L，向き：v@ob − prod@ob）
name = {"konly": "M_F6_K", "level_only": "M_F2_K", "confusion": "cf_D_l05", "source_only": "M_F6_ext", "all_labels": "M_F6_orc",
        "prop": "M_F6_mix2", "noprof": "M_F6_mix0", "noint": "M_F6n_mix2", "levelfree": "M_F6_mix14"}
rows = []
for r in ef[ef.quantity.isin(["total", "direction"])].itertuples():
    m = pe[(pe.c == r.c) & (pe.K == r.K) & (pe.a == name[r.version]) & (pe.b == "prod") & (pe["where"] == r.layer)]
    if len(m) == 1:
        rows.append(dict(c=r.c, K=r.K, v=r.version, q=r.quantity, w=r.layer, d_est=abs(r.est - m.est.iloc[0]), d_lo=abs(r.lo - m.lo.iloc[0]),
                         d_se=abs(r.se - m.se.iloc[0]), method=r.method))
cmp_ = pd.DataFrame(rows)
print("集計の段と照らせた行", len(cmp_), "／全体と向きの行", int(ef.quantity.isin(["total", "direction"]).sum()))
print(cmp_.groupby(["method"])[["d_est", "d_lo", "d_se"]].max())
print("照らせた (版, 層)：", sorted(set(zip(cmp_.v, cmp_.w))))

# ---- 2. 差の差の区間を直接作る
udir = ROOT / "results/dai16_probe/units"
want = [("AFEW", 5, "prop", "adj"), ("MELD", 25, "noint", "raw"), ("IEMOCAP", 25, "prop", "kb"), ("MOSEI", 5, "prop", "adj"), ("MOSEI", 25, "konly", "raw")]
need = {}
for c, K, v, L in want:
    b = name[v]
    for nm, kk in ((f"{v}@{L}", f"pred_{L}_K{K}_{b}"), (f"{v}@ob", f"pred_ob_K{K}_{b}"), (f"prod@{L}", f"pred_{L}_K{K}_prod"), ("prod@ob", f"pred_ob_K{K}_prod")):
        need.setdefault((c, K, nm), set()).add(kk)
cells = {}
for p in sorted(udir.glob("*.npz")):
    T, c, f, r = p.stem.rsplit("_", 3)
    f, r = int(f[1:]), int(r[1:])
    z = np.load(p, allow_pickle=False)
    yt, cl = z["y_test"].astype(int), z["cl_test"]
    for (cc, K, nm), kks in need.items():
        if cc != c:
            continue
        (kk,) = kks
        P12._add(cells.setdefault((c, K, nm), {}), (f, r), ST.counts(yt, z[kk].astype(int), cl), p.stem)
P12._finish(cells)


def uar(cor, tot):
    c_, t_ = cor.sum(0), tot.sum(0)
    ok = t_ > 0
    return 100.0 * (c_[ok] / t_[ok]).mean()


def uar_drop(cor, tot, i):
    c_, t_ = cor.sum(0) - cor[i], tot.sum(0) - tot[i]
    if t_.sum() < 30 or (t_ == 0).any():
        return np.nan
    return 100.0 * (c_ / t_).mean()


res = []
for c, K, v, L in want:
    four = [cells[(c, K, f"{v}@{L}")], cells[(c, K, "prod@ob")], cells[(c, K, f"{v}@ob")], cells[(c, K, f"prod@{L}")]]
    sg = (1, 1, -1, -1)
    keys = sorted(four[0])
    assert all(set(x) == set(keys) for x in four)
    folds = sorted({f for f, _ in keys})
    dd = {f: [sum(s * uar(x[(f, r)][1], x[(f, r)][2]) for s, x in zip(sg, four)) for ff, r in keys if ff == f] for f in folds}
    est = np.mean([np.mean(dd[f]) for f in folds])
    s2 = {f: (np.var(dd[f], ddof=1) / len(dd[f]) if len(dd[f]) > 1 else 0.0) for f in folds}
    geff = ST.g_eff_of(c)[0]
    row = ef[(ef.c == c) & (ef.K == K) & (ef.version == v) & (ef.quantity == "intercept") & (ef.layer == L)].iloc[0]
    if geff < 20:
        allc = sorted({g for k in keys for g in four[0][k][0]})
        jk, dropped = [], set()
        for g in allc:
            th = []
            for f in folds:
                vals = []
                for ff, r in keys:
                    if ff != f:
                        continue
                    nm = four[0][(f, r)][0]
                    if g in nm:
                        i = int(np.flatnonzero(nm == g)[0])
                        u = [uar_drop(x[(f, r)][1], x[(f, r)][2], i) for x in four]
                        if all(np.isfinite(u)):
                            vals.append(sum(s * q for s, q in zip(sg, u)))
                    else:
                        vals.append(sum(s * uar(x[(f, r)][1], x[(f, r)][2]) for s, x in zip(sg, four)))
                if vals:
                    th.append(np.mean(vals))
                else:
                    dropped.add(f)
            jk.append(np.mean(th))
        jk = np.array(jk); G = len(jk)
        vj = (G - 1) / G * ((jk - jk.mean()) ** 2).sum()
        se = np.sqrt(vj + sum(s2[f] for f in folds if f not in dropped) / len(folds) ** 2)
        q = st.t.ppf(0.975, max(1, int(np.floor(geff)) - 1))
        lo, hi = est - q * se, est + q * se
        how = "直接の jackknife"
    else:
        rng = P16.row_rng("dai20post", "I", c, K, v, L)
        boots = np.zeros(ST.N_BOOT); nval = np.zeros(ST.N_BOOT)
        for f in folds:
            rs = [r for ff, r in keys if ff == f]
            ncl = len(four[0][(f, rs[0])][0])
            Wb = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl), size=ST.N_BOOT).astype(float)
            u4 = [np.stack([ST.uar_rows(Wb @ x[(f, r)][1], Wb @ x[(f, r)][2]) for r in rs]) for x in four]
            dm = np.nanmean(u4[0] + u4[1] - u4[2] - u4[3], axis=0)
            ok = np.isfinite(dm); boots[ok] += dm[ok]; nval[ok] += 1
        boots = np.where(nval > 0, boots / np.maximum(nval, 1), np.nan); boots = boots[np.isfinite(boots)]
        se = np.sqrt(np.var(boots, ddof=1) + sum(s2[f] for f in folds) / len(folds) ** 2)
        lo, hi = est - 1.96 * se, est + 1.96 * se
        how = "4 つの UAR を別々に引き直した差の差（同じ鍵）"
    res.append(dict(c=c, K=K, v=v, L=L, how=how, est_direct=est, est_tool=row.est, lo_direct=lo, lo_tool=row.lo, hi_direct=hi, hi_tool=row.hi,
                    d_max=max(abs(est - row.est), abs(lo - row.lo), abs(hi - row.hi))))
pd.set_option("display.width", 250)
print(pd.DataFrame(res).to_string())
