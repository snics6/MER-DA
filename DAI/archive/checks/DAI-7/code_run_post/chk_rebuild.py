"""DAI-7 計算の後の点検 (4)：保存した rows/ と fits/ から集計の出力を作り直して突き合わせる．
台本の関数は使わず，発話の単位の重みで書き直した．値は出さず，一致・不一致の数と最大の差の桁だけを出す．"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as SS

OUT = Path("/home/snics/MER-DA/results/dai7_probe_v2")
CORP = ("AFEW", "IEMOCAP", "MELD", "MOSEI")
MODS = "AVT"
PAIRS = ("AV", "AT", "VT")
BLK = {"A": slice(0, 3), "V": slice(3, 6), "T": slice(6, 9)}
N_BOOT, MIN_N = 1000, 15
TOL = 1e-9


def mag(x):
    x = float(x)
    if not np.isfinite(x):
        return "非有限"
    if x == 0:
        return "0"
    return f"< 1e{int(np.floor(np.log10(x))) + 1}"


def cmp(name, a, b, tol=TOL):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(a - b)
    d[both_nan] = 0
    bad = int(((d > tol) | (np.isnan(a) != np.isnan(b))).sum())
    print(f"{name}：比べた {a.size}，不一致 {bad}，最大の差 {mag(np.nanmax(d) if d.size else 0)}")
    return bad


units = sorted(p.name[: -len("_units.csv")] for p in (OUT / "rows").glob("*_units.csv"))
U = pd.concat([pd.read_csv(OUT / "rows" / f"{u}_units.csv") for u in units], ignore_index=True).copy()
A = pd.concat([pd.read_csv(OUT / "rows" / f"{u}_uar.csv") for u in units], ignore_index=True)
cells = pd.read_csv(OUT / "cells.csv")
acells = pd.read_csv(OUT / "uar_cells.csv")
Df = pd.read_csv(OUT / "diffs.csv")
nbad = 0

print("== (4a) cells.csv を rows/*_units.csv から作り直す")
D_e = pd.DataFrame({f"D_e{e}": U[f"d_e{e}"] - U[f"d0_e{e}"] for e in range(4)})
U2 = pd.concat([U, D_e], axis=1)
qcols = [x for x in cells.columns if x not in ("c", "K", "set", "pair", "n_units", "commit", "summarize_commit", "summarize_dirty", "boot_seed", "n_boot")]
print("cells の量の列の数", len(qcols))
per_pair = U2.groupby(["c", "K", "set", "pair"])[qcols].mean()
unit_all = U2.groupby(["T", "c", "fold", "seed", "K", "set"])[qcols].mean()
allp = unit_all.groupby(["c", "K", "set"]).mean()
allp["pair"] = "all"
allp = allp.reset_index().set_index(["c", "K", "set", "pair"])
reb = pd.concat([per_pair, allp]).sort_index()
cs = cells.set_index(["c", "K", "set", "pair"]).sort_index()
assert reb.index.equals(cs.index)
nbad += cmp("cells の全部の量の列", reb[qcols].to_numpy(), cs[qcols].to_numpy())

print("== (4b) uar_cells.csv を rows/*_uar.csv から作り直す")
A2 = A.assign(h_excess=A.heterogeneity - A.heterogeneity_null)
ucols = [x for x in acells.columns if x not in ("c", "K", "set", "commit", "summarize_commit", "summarize_dirty", "boot_seed", "n_boot")]
print("uar_cells の量の列の数", len(ucols))
rA = A2.groupby(["c", "K", "set"])[ucols].mean().sort_index()
ac = acells.set_index(["c", "K", "set"]).sort_index()
nbad += cmp("uar_cells の全部の量の列", rA.to_numpy(), ac[ucols].to_numpy())
# 単位の行の L_dep・L_cal が UAR の差と同じか
nbad += cmp("uar の L_dep = uar_full − uar_blk（7,800 行）", A.L_dep, A.uar_full - A.uar_blk, 1e-9)
nbad += cmp("uar の L_cal = uar_blk − uar_prod（7,800 行）", A.L_cal, A.uar_blk - A.uar_prod, 1e-9)

print("== (4c) diffs.csv の点推定を cells と uar_cells から作り直す")
DDM = {"dD_joint_a1_none": ("joint_a1", "none"), "dD_joint_none": ("joint", "none"), "dD_joint_a1_oracle": ("joint_a1", "oracle"),
       "dD_rho_hi_lo": ("joint_a1_rho4.0", "joint_a1_rho0.0625"), "dD_rho1_ownret": ("joint_a1_rho1.0", "tt_ownret"),
       "dD_ownret_none": ("tt_ownret", "none")}
DPHI = {"dphi_joint_a1_none": ("joint_a1", "none"), "dphi_joint_none": ("joint", "none"),
        "dphi_rho1_ownret": ("joint_a1_rho1.0", "tt_ownret"), "dphi_ownret_none": ("tt_ownret", "none")}
est_r, est_g, dn_r, dn_g = [], [], [], []
for _, row in Df.iterrows():
    c, K, v, p, q = row.c, row.K, row.variant, row.pair, row.quantity
    x, dn = None, None
    if q in DDM:
        s1, s2 = DDM[q]
        if pd.notna(row.emotion):
            col, ncol = f"D_e{int(row.emotion)}", None
        elif v in ("main", "jk", "spk"):
            col, ncol = "D", "D_null"
        else:
            col, ncol = f"D_{v}", f"Dnull_{v}"
        x = cs.loc[(c, K, s1, p), col] - cs.loc[(c, K, s2, p), col]
        if ncol:
            dn = cs.loc[(c, K, s1, p), ncol] - cs.loc[(c, K, s2, p), ncol]
    elif q.startswith("D_"):
        x = cs.loc[(c, K, q[2:], p), "D"]
    elif q in DPHI:
        s1, s2 = DPHI[q]
        x = cs.loc[(c, K, s1, "all"), "phi"] - cs.loc[(c, K, s2, "all"), "phi"]
    elif q.startswith("dLdep_") and q.endswith("_none"):
        s = q[len("dLdep_"):-len("_none")]
        x = ac.loc[(c, K, s), "L_dep"] - ac.loc[(c, K, "none"), "L_dep"]
    elif q.startswith("Ldep_"):
        x = ac.loc[(c, K, q[5:]), "L_dep"]
    elif q == "dDt_joint_a1_none":
        x = cs.loc[(c, K, "joint_a1", "all"), "D_partial_t"] - cs.loc[(c, K, "none", "all"), "D_partial_t"]
    else:
        for pre in ("G_", "S1b_", "S1_", "Scal_", "S_", "dLdep_", "dLcal_", "share_dep_"):
            if q.startswith(pre):
                kind, s = pre[:-1], q[len(pre):]
                break
        us = {m: ac.loc[(c, K, s), f"uar_{m}"] for m in MODS}
        un = {m: ac.loc[(c, K, "none"), f"uar_{m}"] for m in MODS}
        gbar = np.mean([us[m] - un[m] for m in MODS])
        gcal = np.mean([ac.loc[(c, K, s), f"uar_lda_{m}"] - ac.loc[(c, K, "none"), f"uar_lda_{m}"] for m in MODS])
        gp = ac.loc[(c, K, s), "uar_prod"] - ac.loc[(c, K, "none"), "uar_prod"]
        gf = ac.loc[(c, K, s), "uar_full"] - ac.loc[(c, K, "none"), "uar_full"]
        ldep = ac.loc[(c, K, s), "L_dep"] - ac.loc[(c, K, "none"), "L_dep"]
        lcal = ac.loc[(c, K, s), "L_cal"] - ac.loc[(c, K, "none"), "L_cal"]
        S = gbar - gp
        x = {"G": -S, "S": S, "S1": gbar - gf, "Scal": gbar - gcal, "S1b": gcal - gf, "dLdep": ldep, "dLcal": lcal,
             "share_dep": ldep / S if S != 0 else np.nan}[kind]
        if kind == "G":
            for col, val in (("G_A", us["A"] - un["A"]), ("G_V", us["V"] - un["V"]), ("G_T", us["T"] - un["T"]), ("G_prod", gp), ("G_full", gf),
                             ("G_cal", gcal), ("G_max", max(us[m] - un[m] for m in MODS))):
                est_r.append(row[col]); est_g.append(val)
    est_r.append(row.est); est_g.append(x)
    if dn is not None:
        dn_r.append(row.d_null_diff); dn_g.append(dn)
nbad += cmp("diffs の est（3,105 行）と G の行の G_A〜G_max", est_r, est_g)
nbad += cmp("diffs の d_null_diff（ΔD の行）", dn_r, dn_g)
# S の恒等式（3.5 節）
for s in ("joint_a1", "joint_a1_rho1.0"):
    g = {q: Df[(Df.variant == "main") & (Df.quantity == f"{q}_{s}")].set_index(["c", "K"]).est.sort_index() for q in ("S", "S1", "Scal", "S1b", "dLdep", "dLcal")}
    nbad += cmp(f"恒等式 S = Scal + S1b + dLdep + dLcal（{s}，20 セル）", g["S"], g["Scal"] + g["S1b"] + g["dLdep"] + g["dLcal"], 1e-9)
    nbad += cmp(f"恒等式 S1 = Scal + S1b（{s}）", g["S1"], g["Scal"] + g["S1b"], 1e-9)
# n_test_5folds
n5_bad = 0
for c in CORP:
    n5 = np.zeros(4, int)
    for f in range(5):
        u = next(x for x in units if x.split("_")[1] == c and x.split("_")[2] == f"f{f}")
        n5 += np.bincount(np.load(OUT / "fits" / f"{u}.npz")["test_y"].astype(int), minlength=4)
    r = Df[(Df.c == c) & Df.emotion.notna()]
    n5_bad += int((r.n_test_5folds.to_numpy() != n5[r.emotion.astype(int).to_numpy()]).sum())
print("diffs の n_test_5folds が fits の本数と違う行", n5_bad)
nbad += n5_bad


# ------------------------------------------------------------------ (4d) 区間を発話の単位の重みで作り直す
def boot_w(cl_by_fold, jk=False):
    names, fold_of, code_of = [], [], {}
    for f in sorted(cl_by_fold):
        uq = sorted(set(cl_by_fold[f]))
        code_of[f] = {x: len(names) + i for i, x in enumerate(uq)}
        names += uq; fold_of += [f] * len(uq)
    k = len(names)
    w = np.random.default_rng(0).multinomial(k, np.full(k, 1.0 / k), size=N_BOOT).astype(float)
    if jk:
        w = 1.0 - np.eye(k)
    return w, np.array(fold_of), code_of


def combine(per_fold, w, fold_of):
    folds = sorted(per_fold)
    num = np.zeros(w.shape[0]); den = np.zeros(w.shape[0])
    for f in folds:
        cnt = w[:, fold_of == f].sum(1)
        wf = cnt / (fold_of == f).sum()
        v = per_fold[f]
        ok = (cnt > 0) & np.isfinite(v)
        num += np.where(ok, v * wf, 0); den += np.where(ok, wf, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def wcov_all(Z, y, W):
    """発話の重み W (B, n) で感情ごとの重みつき共分散 (B, 9, 9) と，重みつきの本数が MIN_N 以上の回の印を返す．"""
    out = []
    for e in range(4):
        ix = y == e
        We, Ze = W[:, ix], Z[ix]
        n = We.sum(1)
        ok = n >= MIN_N
        with np.errstate(all="ignore"):
            mu = We @ Ze / n[:, None]
            M2 = (We @ (Ze[:, :, None] * Ze[:, None, :]).reshape(len(Ze), -1)).reshape(-1, Z.shape[1], Z.shape[1]) / n[:, None, None]
        out.append((ok, M2 - mu[:, :, None] * mu[:, None, :]))
    return out


def d_from_cov(covs, d0, sa, sb):
    B = covs[0][1].shape[0]
    res = np.full((B, 4), np.nan)
    I3 = np.eye(3)
    for e in range(4):
        ok, Cv = covs[e]
        if not np.isfinite(d0[e]) or not ok.any():
            continue
        Cv = Cv[ok]
        Saa, Sbb, Sab = Cv[:, sa, sa], Cv[:, sb, sb], Cv[:, sa, sb]
        ra = 1e-6 * np.trace(Saa, axis1=1, axis2=2)[:, None, None] / 3 * I3
        rb = 1e-6 * np.trace(Sbb, axis1=1, axis2=2)[:, None, None] / 3 * I3
        Am = np.linalg.solve(Saa + ra, Sab)
        Bm = np.linalg.solve(Sbb + rb, np.swapaxes(Sab, 1, 2))
        res[ok, e] = np.trace(Am @ Bm, axis1=1, axis2=2) / 3 - d0[e]
    return _nanmean_rows(res)


def nm(stack):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(np.stack(stack), 0)


def wuar(pred, y, W):
    r = []
    bad = np.zeros(W.shape[0], bool)
    for e in range(4):
        ix = y == e
        n = W[:, ix].sum(1)
        bad |= n < MIN_N
        with np.errstate(all="ignore"):
            r.append(W[:, ix] @ (pred[ix] == e).astype(float) / n)
    out = 100 * np.mean(r, 0)
    out[bad] = np.nan
    return out


def wphi(pa, pb, y, W):
    vals = []
    for e in range(4):
        ix = y == e
        Wi = W[:, ix]
        n = Wi.sum(1)
        ca, cb = (pa[ix] == e).astype(float), (pb[ix] == e).astype(float)
        with np.errstate(all="ignore"):
            a_ = Wi @ ca / n; b_ = Wi @ cb / n; ab = Wi @ (ca * cb) / n
            ph = (ab - a_ * b_) / np.sqrt(a_ * (1 - a_) * b_ * (1 - b_))
        ph[(n < MIN_N) | ~np.isfinite(ph)] = np.nan
        vals.append(ph)
    return _nanmean_rows(np.stack(vals, 1))


def _nanmean_rows(M):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(M, 1)


def rebuild(c, K, var="main", jk=False):
    us = sorted({(u.split("_")[0], int(u.split("_")[2][1:]), int(u.split("_")[3][1:])) for u in units if u.split("_")[1] == c})
    fits = {k: np.load(OUT / "fits" / f"{k[0]}_{c}_f{k[1]}_r{k[2]}.npz") for k in us}
    clkey = "test_group" if var == "spk" else "test_cluster"
    masked = var not in ("main", "spk")
    pre = f"v_{var}_" if masked else ""
    cl_by_fold = {}
    for k, z in sorted(fits.items()):
        mk = z[f"mask_{var}"].astype(bool) if masked else np.ones(len(z["test_y"]), bool)
        if k[1] not in cl_by_fold and mk.any():
            cl_by_fold[k[1]] = list(z[clkey].astype(str)[mk])
    w, fold_of, code_of = boot_w(cl_by_fold, jk)
    Uc = U[(U.c == c) & (U.K == K)].set_index(["T", "fold", "seed", "set", "pair"])
    sets_D = ("none", "joint_a1", "oracle", "joint_a1_rho1.0", "tt_ownret")
    accD = {s: {} for s in sets_D}
    accphi = {s: {} for s in ("none", "joint_a1")}
    accL = {s: {} for s in ("none", "joint_a1")}
    accS = {}
    for f in sorted(cl_by_fold):
        lD = {s: [] for s in sets_D}; lphi = {s: [] for s in accphi}; lL = {s: [] for s in accL}; lS = []
        for k, z in sorted(fits.items()):
            if k[1] != f:
                continue
            mk = z[f"mask_{var}"].astype(bool) if masked else np.ones(len(z["test_y"]), bool)
            yfull = z["test_y"].astype(int)
            y = yfull[mk]
            Wu = w[:, [code_of[f][x] for x in z[clkey].astype(str)[mk]]]
            for s in sets_D:
                Z = z[f"K{K}_{s}_z"].astype(np.float64)
                if var in ("cent", "scen"):
                    g = z["test_group" if var == "cent" else "test_scenario"].astype(str)
                    key = pd.Series([f"{a}|{b}" for a, b in zip(g, yfull)])
                    Zc = Z.copy()
                    for _, ix in key.groupby(key).groups.items():
                        ix = np.asarray(ix)
                        if len(ix) >= 3:
                            Zc[ix] = Z[ix] - Z[ix].mean(0)
                    Z = Zc
                Z = Z[mk]
                vals = []
                covs = wcov_all(Z, y, Wu)
                for a, b in PAIRS:
                    rw = Uc.loc[(k[0], k[1], k[2], s, a + b)]
                    d0 = np.array([rw[f"{pre}d0_e{e}"] if np.isfinite(rw[f"{pre}d_e{e}"]) else np.nan for e in range(4)])
                    vals.append(d_from_cov(covs, d0, BLK[a], BLK[b]))
                lD[s].append(nm(vals))
            if var == "main" and not jk:
                yy = yfull
                for s in accphi:
                    lphi[s].append(nm([wphi(z[f"K{K}_{s}_pred_{a}"].astype(int), z[f"K{K}_{s}_pred_{b}"].astype(int), yy, Wu) for a, b in PAIRS]))
                bu = {s: {q: wuar(z[f"K{K}_{s}_pred_{q}"].astype(int), yy, Wu) for q in ("prod", "blk", "full", *MODS, "lda_A", "lda_V", "lda_T")}
                      for s in ("none", "joint_a1")}
                for s in accL:
                    lL[s].append(bu[s]["full"] - bu[s]["blk"])
                gbar = np.mean([bu["joint_a1"][m] - bu["none"][m] for m in MODS], 0)
                lS.append(gbar - (bu["joint_a1"]["prod"] - bu["none"]["prod"]))
        for s in sets_D:
            accD[s][f] = nm(lD[s])
        if var == "main" and not jk:
            for s in accphi:
                accphi[s][f] = nm(lphi[s]); accL[s][f] = nm(lL[s])
            accS[f] = nm(lS)
    BD = {s: combine(accD[s], w, fold_of) for s in sets_D}
    res = {"dD_joint_a1_none": BD["joint_a1"] - BD["none"], "dD_joint_a1_oracle": BD["joint_a1"] - BD["oracle"],
           "dD_rho1_ownret": BD["joint_a1_rho1.0"] - BD["tt_ownret"]}
    if var == "main" and not jk:
        res["dphi_joint_a1_none"] = combine(accphi["joint_a1"], w, fold_of) - combine(accphi["none"], w, fold_of)
        res["dLdep_joint_a1_none"] = combine(accL["joint_a1"], w, fold_of) - combine(accL["none"], w, fold_of)
        res["G_joint_a1"] = -combine(accS, w, fold_of)
        res["D_none"] = BD["none"]
    return res


def q_ci(x):
    x = x[np.isfinite(x)]
    return np.quantile(x, 0.025), np.quantile(x, 0.975), len(x)


print("== (4d) 区間の作り直し（発話の単位の重み．台本の関数は使わない）")
plan = [("IEMOCAP", 5, "main"), ("AFEW", 10, "main"), ("MELD", 1, "main"), ("MOSEI", 25, "main"),
        ("IEMOCAP", 25, "cent"), ("IEMOCAP", 10, "scen"), ("IEMOCAP", 1, "spk"), ("AFEW", 5, "keep"), ("MOSEI", 10, "chan"), ("MELD", 25, "sub_face_single")]
lo_r, lo_g, nv_bad, nci = [], [], 0, 0
for c, K, var in plan:
    res = rebuild(c, K, var)
    for q, x in res.items():
        r = Df[(Df.c == c) & (Df.K == K) & (Df.variant == var) & (Df.quantity == q) & (Df.pair == "all") & Df.emotion.isna()]
        assert len(r) == 1, (c, K, var, q, len(r))
        lo, hi, nv = q_ci(x)
        lo_r += [r.lo.iloc[0], r.hi.iloc[0]]; lo_g += [lo, hi]
        nv_bad += int(nv != r.n_boot_valid.iloc[0]); nci += 1
    print(f"{c} K={K} {var}：作り直した区間の数 {len(res)}", flush=True)
nbad += cmp(f"区間の端（{nci} 個の区間の lo と hi）", lo_r, lo_g, 1e-8)
print("有効な回の数が違う区間", nv_bad); nbad += nv_bad

print("== (4e) jackknife と t の区間の作り直し")
jr, jg, jbad = [], [], 0
for c, K in (("AFEW", 1), ("IEMOCAP", 5), ("MELD", 25)):
    res = rebuild(c, K, "main", jk=True)
    for q, th in res.items():
        r = Df[(Df.c == c) & (Df.K == K) & (Df.variant == "jk") & (Df.quantity == q)]
        th = th[np.isfinite(th)]
        n = len(th)
        se = np.sqrt((n - 1) / n * np.sum((th - th.mean()) ** 2))
        df = max(1, int(np.floor(r.n_eff.iloc[0])) - 1)
        tq = SS.t.ppf(0.975, df)
        est = r.est.iloc[0]
        jr += [r.lo.iloc[0], r.hi.iloc[0]]; jg += [est - tq * se, est + tq * se]
        jbad += int(n != r.n_boot_valid.iloc[0]) + int(df != r.df.iloc[0])
nbad += cmp("jackknife の区間の端（9 個の区間）", jr, jg, 1e-8)
print("jackknife の回の数か自由度が違う行", jbad); nbad += jbad
print("== 不一致の合計", nbad)
