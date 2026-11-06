"""gate-code（DAI-5）：dai5.py の pseudo の 1 単位の保存を numpy で作り直して照合する。成績の値は出さない（差と一致の数だけ）。
使い方：./.venv/bin/python DAI/checks/DAI-5/code_run/verify.py <out_dir> T c f r
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai_data as DD  # noqa: E402

out, T, c, f, r = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
name = f"{T}_{c}_f{f}_r{r}"
z = np.load(out / "pseudo" / "units" / f"{name}_p5.npz")
rows = pd.read_csv(out / "pseudo" / "units" / f"{name}_p5.csv", low_memory=False)
z1 = np.load(ROOT / "results" / "dai1" / "pseudo" / "units" / f"{name}_p1.npz")
p3 = pd.read_csv(ROOT / "results" / "dai1" / "pseudo" / "units" / f"{name}_p3.csv", low_memory=False)
fz1 = json.load(open(ROOT / "DAI" / "frozen" / "dai1_frozen.json"))[T]
srcs = [s for s in DD.others(T) if s != c]

# --- データ：DD.load_corpus から独立に（標準化は torch の std と同じ ddof=1）
df, feats = DD.load_corpus(c, DD.ENC1)
y = df["label_4"].to_numpy().astype(int)
sid = df["sample_id"].to_numpy().astype(str)
pool = np.flatnonzero(DD.pool_mask(df, f)); test = np.flatnonzero(df["fold"].to_numpy() == f)
assert (sid[pool] == z1["sample_pool"]).all() and (sid[test] == z1["sample_test"]).all()
assert (sid[test] == z["sample_test"]).all()
pos = {s: i for i, s in enumerate(sid[pool])}
kpos = {0: np.array([], int), **{K: np.array([pos[s] for s in z1[f"sample_K{K}"]]) for K in DD.KS}}
for K in DD.KS:
    assert (z[f"sample_K{K}"] == z1[f"sample_K{K}"]).all()
nest = all(set(kpos[a]) <= set(kpos[b]) for a, b in zip(DD.KS[:-1], DD.KS[1:]))
yp, yt = y[pool], y[test]
n = len(pool)


def std(kind):
    X = DD.kind_matrix(feats, kind).astype(np.float64)
    mu, sd = X[pool].mean(0), np.maximum(X[pool].std(0, ddof=1), 1e-6)
    return (X[pool] - mu) / sd, (X[test] - mu) / sd


def lsm(Z):
    Z = Z - Z.max(-1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(-1, keepdims=True))


def uar(yy, pp):
    return float(np.mean([(pp[yy == k] == k).mean() for k in range(4) if (yy == k).any()]))


Xp, Xt = {}, {}
for k in DD.STUDENTS:
    Xp[k], Xt[k] = std(k)
names, LP, TPRED = [], [], []
for m in "AVT":
    for s in srcs:
        q = np.load(ROOT / "results" / "dai1" / "source" / f"{s}_{m}.npz")
        names.append(f"{s}_{m}")
        LP.append(lsm(Xp[m] @ q["W"].T.astype(np.float64) + q["b"]))
        TPRED.append((Xt[m] @ q["W"].T.astype(np.float64) + q["b"]).argmax(1))
res = {}
res["names_equal"] = list(z["p5_src_names"].astype(str)) == names
res["src_test_pred_mismatch"] = int((np.stack(TPRED) != z["p5_src_test_pred"]).sum())
res["K_nested"] = nest

# --- 採点・重み・Q・G
BETAS = (0.0, 2.5, 5.0, 10.0, 20.0)
tether = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in fz1["tether"].items()}
maxdiff = dict(score=0.0, w=0.0, maxp=0.0, ent=0.0, neffw=0.0, neff=0.0, colsum=0.0)
heaviest_bad = 0
grad_ok, grad_alt, pred_mm, uar_mm, n_items, wb_eq0 = [], [], 0, 0, 0, 0
order_bad, rho_a_bad = 0, 0
tstat = rows[rows.method == "_teacher_stats"]
srow = rows[rows.method == "srcw"]
for K in (0,) + DD.KS:
    kp = kpos[K]
    if K:
        sc = np.array([100 * uar(yp[kp], LP[j][kp].argmax(1)) for j in range(len(names))])
        maxdiff["score"] = max(maxdiff["score"], float(np.abs(sc - z[f"p5_K{K}_scores"]).max()))
    else:
        assert np.isnan(z[f"p5_K{K}_scores"]).all()
    betas = (0.0,) if K == 0 else BETAS
    Qs = {}
    for be in betas:
        if K:
            e = np.exp(be * sc / 100 - (be * sc / 100).max()); w = 3 * e / e.sum()
        else:
            w = np.full(len(names), 3 / len(names))
        if be == 0.0:
            wb_eq0 += int(np.allclose(z[f"p5_K{K}_w_b{be}"], 3 / len(names), atol=1e-15, rtol=0))
        maxdiff["w"] = max(maxdiff["w"], float(np.abs(w - z[f"p5_K{K}_w_b{be}"]).max()))
        L = sum(wj * LP[j] for j, wj in enumerate(w))
        Q = np.exp(lsm(L)); Qs[be] = Q
        ts = tstat[(tstat.K == K) & np.isclose(tstat.beta, be)]
        assert len(ts) == 1
        ts = ts.iloc[0]
        Pn = Q / Q.sum(0, keepdims=True); Wt = Pn / Pn.sum() * n
        neff = Wt.sum(0) ** 2 / (Wt ** 2).sum(0)
        maxdiff["maxp"] = max(maxdiff["maxp"], abs(Q.max(1).mean() - ts.maxp))
        maxdiff["ent"] = max(maxdiff["ent"], abs(-(Q * np.log(np.maximum(Q, 1e-12))).sum(1).mean() - ts.entropy))
        maxdiff["neffw"] = max(maxdiff["neffw"], abs(w.sum() ** 2 / (w ** 2).sum() - ts.neff_w))
        maxdiff["neff"] = max(maxdiff["neff"], max(abs(neff[k] - ts[f"neff_{k}"]) / neff[k] for k in range(4)))
        maxdiff["colsum"] = max(maxdiff["colsum"], max(abs(Q.sum(0)[k] - ts[f"colsum_{k}"]) / Q.sum(0)[k] for k in range(4)))
        heaviest_bad += int(ts.heaviest != names[int(np.argmax(w))])
    # gK：感情の釣り合いの one-hot
    gk = np.zeros((n, 4))
    if K:
        cnt = np.bincount(yp[kp], minlength=4).astype(float)
        gk[kp, yp[kp]] = len(kp) / ((cnt > 0).sum() * cnt[yp[kp]])
    for s in DD.STUDENTS:
        v = fz1["main"][str(K)]
        js = fz1["select"][f"{s}|{K}|joint|{v}"]
        rho, a = float(js["rho"]), float(js["a"])
        cw, cb = tether[(s, max(K, 1))]
        q0 = [np.load(ROOT / "results" / "dai1" / "source" / f"{c_}_{s}.npz") for c_ in srcs]
        W0 = np.mean([x["W"] for x in q0], 0).astype(np.float64); b0 = np.mean([x["b"] for x in q0], 0).astype(np.float64)
        items = [json.loads(x) for x in z[f"p5_{s}_K{K}_items"]]
        order_bad += int([it["beta"] for it in items] != list(betas))
        rho_a_bad += int(any(it["rho"] != rho or it["a"] != a or it["method"] != "srcw" for it in items))
        Wsv, bsv, prd = z[f"p5_{s}_K{K}_W"].astype(np.float64), z[f"p5_{s}_K{K}_b"].astype(np.float64), z[f"p5_{s}_K{K}_pred"]
        for j, be in enumerate(betas):
            def grad(G, W, b):
                Z = Xp[s] @ W.T + b
                P = np.exp(lsm(Z))
                dZ = G.sum(1, keepdims=True) * P - G
                gW = dZ.T @ Xp[s] + (W - W0) / cw; gb = dZ.sum(0) + (b - b0) / cb
                return max(np.abs(gW).max(), np.abs(gb).max()) / G.sum()
            Qn = Qs[be] / Qs[be].sum(0, keepdims=True)
            G = a * gk + Qn / Qn.sum() * rho * n
            grad_ok.append(grad(G, Wsv[j], bsv[j]))
            # 判別力の確かめ：別の β（K=0 は ρ を 2 倍）の G で同じ解の勾配
            if K:
                be2 = betas[(j + 1) % len(betas)]
                Qn2 = Qs[be2] / Qs[be2].sum(0, keepdims=True)
                G2 = a * gk + Qn2 / Qn2.sum() * rho * n
            else:
                G2 = a * gk + Qn / Qn.sum() * 2 * rho * n
            grad_alt.append(grad(G2, Wsv[j], bsv[j]))
            pr = (Xt[s] @ Wsv[j].T + bsv[j]).argmax(1)
            pred_mm += int((pr != prd[j]).sum())
            rr = srow[(srow.student == s) & (srow.K == K) & np.isclose(srow.beta, be)]
            assert len(rr) == 1
            uar_mm += int(abs(100 * uar(yt, prd[j].astype(int)) - rr.uar.iloc[0]) > 1e-9)
            n_items += 1
res.update({f"maxdiff_{k}": v for k, v in maxdiff.items()})
res.update(heaviest_bad=heaviest_bad, w_beta0_equal=wb_eq0, item_order_bad=order_bad, rho_a_method_bad=rho_a_bad,
           n_items=n_items, grad_at_saved_max=float(max(grad_ok)), grad_alt_G_min=float(min(grad_alt)),
           pred_recompute_mismatch=pred_mm, n_test=len(test), uar_row_mismatch=uar_mm)

# --- 教師の検算と ρ = 0 の自己検査
tc = rows[rows.method == "_teacher_check"]
mine_bad = 0
for K in (0,) + DD.KS:
    nk = np.setdiff1d(np.arange(n), kpos[K])
    for j, nm in enumerate(names):
        mine = 100 * uar(yp[nk], LP[j][nk].argmax(1))
        ref = p3[(p3.method == "_teacher_stats") & (p3.teacher.astype(str) == f"tt:src_{nm}") & (p3.K == K)]
        t5 = tc[(tc.K == K) & (tc.src == nm)]
        mine_bad += int(len(ref) == 0 or abs(mine - ref.uar_pool.iloc[0]) > 1e-9 or len(t5) != 1 or abs(mine - t5.uar_pool_mine.iloc[0]) > 1e-9)
res["teacher_check_rows"] = int(len(tc)); res["teacher_check_match_rows"] = int(tc.match.astype(bool).sum())
res["teacher_check_independent_bad"] = mine_bad
r0 = rows[rows.method == "_rho0check"]
res["rho0_rows"] = int(len(r0)); res["rho0_min_agree"] = float(r0.pred_agree.min()) if len(r0) else None
cv = rows[rows.method == "srcw"]
res["srcw_rows"] = int(len(cv)); res["dup_ids"] = int(cv.duplicated(["student", "K", "beta"]).sum())
res["grad_max_row"] = float(cv.grad.max()); res["stalled"] = int(cv.stalled.astype(bool).sum())
res["finite_false"] = int((cv.finite.astype(str) == "False").sum())
num = rows.select_dtypes("number")
res["nonfinite_numeric_cells_srcw"] = int((~np.isfinite(cv.select_dtypes("number").drop(columns=[x for x in cv.columns if cv[x].isna().all()], errors="ignore").to_numpy(float))).sum())
res["npz_keys"] = len(z.files)
res["npz_has_sharp_agree"] = ("p5_src_sharp" in z.files, "p5_src_agree" in z.files)
if "p5_src_sharp" in z.files:
    P = [np.exp(lp) for lp in LP]
    sh = np.array([[p.max(1).mean(), -(p * lp).sum(1).mean()] for p, lp in zip(P, LP)])
    am = [p.argmax(1) for p in P]
    ag = np.array([[(x == y_).mean() for y_ in am] for x in am])
    res["sharp_maxdiff"] = float(np.abs(sh - z["p5_src_sharp"]).max()); res["agree_maxdiff"] = float(np.abs(ag - z["p5_src_agree"]).max())
res["commit"] = str(z["commit"]); res["dirty"] = bool(z["dirty"]); res["gpu"] = str(z["gpu"])
for k, v in res.items():
    print(k, v)
