"""gate-code 第 2 版（DAI-5）：コミット 818d07f で足した部分を，pseudo の試走の 1 単位の保存から numpy で独立に作り直して照合する。
対象：srcw_mod・srcw_src・srcw_top1 の重みと教師，影響の割合（大きさ・判定を変える），p5_src_sharp の 3 列目，
引き戻しの曲線（--test-tether の p5t），_teacher_stats の variant ごとの行，_poecheck の行，問の数と重複，保存した W・b での勾配。
成績の値は出さない（差・一致の数・勾配だけ）。
使い方：./.venv/bin/python DAI/checks/DAI-5/code_run_v2/verify_v2.py <out_dir> T c f r
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
z3 = np.load(ROOT / "results" / "dai1" / "pseudo" / "units" / f"{name}_p3.npz")
fz1 = json.load(open(ROOT / "DAI" / "frozen" / "dai1_frozen.json"))[T]
TETHER = [(cw, cb) for cw in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0) for cb in (1.0, 10.0)]   # dai1.TETHER を写した（独立の照合のため import しない）
srcs = [s for s in DD.others(T) if s != c]
nS = len(srcs)

df, feats = DD.load_corpus(c, DD.ENC1)
y = df["label_4"].to_numpy().astype(int)
sid = df["sample_id"].to_numpy().astype(str)
pool = np.flatnonzero(DD.pool_mask(df, f)); test = np.flatnonzero(df["fold"].to_numpy() == f)
assert (sid[pool] == z1["sample_pool"]).all() and (sid[test] == z["sample_test"]).all()
pos = {s: i for i, s in enumerate(sid[pool])}
kpos = {0: np.array([], int), **{K: np.array([pos[s] for s in z1[f"sample_K{K}"]]) for K in DD.KS}}
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


def sm3(v, beta):
    e = np.exp(beta * v / 100 - (beta * v / 100).max())
    return e / e.sum()


Xp, Xt = {}, {}
for k in DD.STUDENTS:
    Xp[k], Xt[k] = std(k)
names, LP, Zs = [], [], []
for m in "AVT":
    for s in srcs:
        q = np.load(ROOT / "results" / "dai1" / "source" / f"{s}_{m}.npz")
        Z = Xp[m] @ q["W"].T.astype(np.float64) + q["b"]
        names.append(f"{s}_{m}"); LP.append(lsm(Z)); Zs.append(Z)
res = {"names_equal": list(z["p5_src_names"].astype(str)) == names}

# 出発点のモダリティごとの識別器（ソースの W・b の平均）を直接作る．D1：その argmax は対数確率の平均の argmax と同じはず
startZ = {}
for m in "AVT":
    qs = [np.load(ROOT / "results" / "dai1" / "source" / f"{s}_{m}.npz") for s in srcs]
    W0 = np.mean([x["W"] for x in qs], 0).astype(np.float64); b0 = np.mean([x["b"] for x in qs], 0).astype(np.float64)
    startZ[m] = Xp[m] @ W0.T + b0

# p5_src_sharp の 3 列目：感情の平均を引いた点数の標準偏差（torch の std は ddof=1，全要素）
sd3 = np.array([np.std(lp - lp.mean(1, keepdims=True), ddof=1) for lp in LP])
sdZ = np.array([np.std(Zc - Zc.mean(1, keepdims=True), ddof=1) for Zc in Zs])
res["sharp_col3_maxdiff"] = float(np.abs(sd3 - z["p5_src_sharp"][:, 2]).max())
res["sharp_col3_eq_logit_sd"] = float(np.abs(sd3 - sdZ).max())

tether = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in fz1["tether"].items()}
md = dict(scores_mod=0.0, w_mod=0.0, w_src=0.0, w_top1=0.0, infl_mag=0.0, infl_flip=0.0, ts_maxp=0.0, ts_ent=0.0)
cnt = dict(mod_sum_ok=0, src_permod_sum_ok=0, top1_ties=0, n_infl=0, ts_rows=0, ts_missing=0, heaviest_bad=0)
grads, grads_alt, pred_mm, uar_mm, items_bad, n_items = [], [], 0, 0, 0, 0
ts = rows[rows.method == "_teacher_stats"]
fixed_pt_agree = []
for K in (0,) + DD.KS:
    kp = kpos[K]
    Q = {}
    if K:
        sc = np.array([100 * uar(yp[kp], LP[j][kp].argmax(1)) for j in range(len(names))])
        scm_lp = np.array([100 * uar(yp[kp], (sum(LP[j] for j in range(len(names)) if names[j].endswith("_" + m)) / nS)[kp].argmax(1)) for m in "AVT"])
        scm_start = np.array([100 * uar(yp[kp], startZ[m][kp].argmax(1)) for m in "AVT"])
        md["scores_mod"] = max(md["scores_mod"], float(np.abs(scm_lp - z[f"p5_K{K}_scores_mod"]).max()), float(np.abs(scm_start - z[f"p5_K{K}_scores_mod"]).max()))
        for be in (0.0, 2.5, 5.0, 10.0, 20.0):
            w = 3 * sm3(sc, be)
            Q[("srcw", be)] = (w, np.exp(lsm(sum(wj * LP[j] for j, wj in enumerate(w)))))
            # 影響の割合
            Qa = Q[("srcw", be)][1].argmax(1)
            mag = np.array([w[j] * np.abs(LP[j] - LP[j].mean(1, keepdims=True)).mean() for j in range(len(names))]); mag /= mag.sum()
            flip = []
            for j in range(len(names)):
                w2 = w.copy(); w2[j] = 0
                flip.append(float((lsm(sum(wj * LP[i] for i, wj in enumerate(w2))).argmax(1) != Qa).mean()))
            I = z[f"p5_K{K}_infl_b{be}"]
            md["infl_mag"] = max(md["infl_mag"], float(np.abs(I[:, 0] - mag).max())); md["infl_flip"] = max(md["infl_flip"], float(np.abs(I[:, 1] - flip).max()))
            cnt["n_infl"] += 1
        if K in (5, 25):
            for be in (2.5, 5.0, 10.0, 20.0):
                wm = 3 * sm3(scm_start, be)
                wmod = np.array([wm["AVT".index(nm.split("_")[-1])] / nS for nm in names])
                wsrc = np.zeros(len(names))
                for mi, m in enumerate("AVT"):
                    ix = [j for j, nm in enumerate(names) if nm.endswith("_" + m)]
                    wsrc[ix] = sm3(sc[ix], be)
                md["w_mod"] = max(md["w_mod"], float(np.abs(wmod - z[f"p5m_K{K}_w_b{be}"]).max()))
                md["w_src"] = max(md["w_src"], float(np.abs(wsrc - z[f"p5s_K{K}_w_b{be}"]).max()))
                cnt["mod_sum_ok"] += int(np.isclose(z[f"p5m_K{K}_w_b{be}"].sum(), 3))
                cnt["src_permod_sum_ok"] += int(all(np.isclose(z[f"p5s_K{K}_w_b{be}"][[j for j, nm in enumerate(names) if nm.endswith("_" + m)]].sum(), 1) for m in "AVT"))
                Q[("srcw_mod", be)] = (wmod, np.exp(lsm(sum(wj * LP[j] for j, wj in enumerate(wmod)))))
                Q[("srcw_src", be)] = (wsrc, np.exp(lsm(sum(wj * LP[j] for j, wj in enumerate(wsrc)))))
        top = np.flatnonzero(sc == sc.max())
        wt = np.zeros(len(names)); wt[top] = 3 / len(top)
        cnt["top1_ties"] += int(len(top) > 1)
        md["w_top1"] = max(md["w_top1"], float(np.abs(wt - z[f"p5o_K{K}_w_binf"]).max()))
        Q[("srcw_top1", np.inf)] = (wt, np.exp(lsm(sum(wj * LP[j] for j, wj in enumerate(wt)))))
    else:
        w = np.full(len(names), 3 / len(names))
        Q[("srcw", 0.0)] = (w, np.exp(lsm(sum(wj * LP[j] for j, wj in enumerate(w)))))
    # _teacher_stats の行（variant ごと）
    for (meth, be), (w, Qv) in Q.items():
        t = ts[(ts.variant == meth) & (ts.K == K) & (np.isclose(ts.beta.astype(float), be) | (np.isinf(be) & np.isinf(ts.beta.astype(float))))]
        if len(t) != 1:
            cnt["ts_missing"] += 1; continue
        t = t.iloc[0]; cnt["ts_rows"] += 1
        md["ts_maxp"] = max(md["ts_maxp"], abs(Qv.max(1).mean() - t.maxp))
        md["ts_ent"] = max(md["ts_ent"], abs(-(Qv * np.log(np.maximum(Qv, 1e-12))).sum(1).mean() - t.entropy))
        topw = np.flatnonzero(w >= w.max() - 1e-12)
        exp_h = "" if (be == 0.0 or len(topw) == len(w)) else "|".join(names[i] for i in topw)
        cnt["heaviest_bad"] += int((str(t.heaviest) if pd.notna(t.heaviest) else "") != exp_h)
    # 生徒：保存した W・b で勾配（独立に作った G）．別の教師の G でも測って見分けられるか
    gk = np.zeros((n, 4))
    if K:
        ct = np.bincount(yp[kp], minlength=4).astype(float)
        gk[kp, yp[kp]] = len(kp) / ((ct > 0).sum() * ct[yp[kp]])
    for s in DD.STUDENTS:
        v = fz1["main"][str(K)]
        js = fz1["select"][f"{s}|{K}|joint|{v}"]
        rho, a = float(js["rho"]), float(js["a"])
        cw0, cb0 = tether[(s, max(K, 1))]
        q0 = [np.load(ROOT / "results" / "dai1" / "source" / f"{c_}_{s}.npz") for c_ in srcs]
        W0 = np.mean([x["W"] for x in q0], 0).astype(np.float64); b0 = np.mean([x["b"] for x in q0], 0).astype(np.float64)

        def grad(G, W, b, cw, cb):
            Zs_ = Xp[s] @ W.T + b
            P = np.exp(lsm(Zs_))
            dZ = G.sum(1, keepdims=True) * P - G
            gW = dZ.T @ Xp[s] + (W - W0) / cw; gb = dZ.sum(0) + (b - b0) / cb
            return max(np.abs(gW).max(), np.abs(gb).max()) / G.sum()

        def Gof(Qv):
            Qn = Qv / Qv.sum(0, keepdims=True)
            return a * gk + Qn / Qn.sum() * rho * n
        for px, meth in (("p5m", "srcw_mod"), ("p5s", "srcw_src"), ("p5o", "srcw_top1"), ("p5t", "srcw_tether")):
            key = f"{px}_{s}_K{K}"
            if f"{key}_items" not in z.files:
                continue
            its = [json.loads(x) for x in z[f"{key}_items"]]
            Wsv, bsv, prd = z[f"{key}_W"].astype(np.float64), z[f"{key}_b"].astype(np.float64), z[f"{key}_pred"]
            for j, it in enumerate(its):
                n_items += 1
                items_bad += int(it["method"] != meth or it["rho"] != rho or it["a"] != a)
                if meth == "srcw_tether":
                    be = 5.0; cw, cb = it["Cw"], it["Cb"]
                    items_bad += int((cw, cb) != TETHER[j] or it["beta"] != 5.0)
                else:
                    be = float(it["beta"]); cw, cb = cw0, cb0
                Qv = Q[("srcw" if meth == "srcw_tether" else meth, be)][1]
                grads.append(grad(Gof(Qv), Wsv[j], bsv[j], cw, cb))
                alt = Q[("srcw", 0.0)][1] if K else Qv
                grads_alt.append(grad(Gof(alt), Wsv[j], bsv[j], cw, cb) if meth != "srcw_tether" else grad(Gof(Qv), Wsv[j], bsv[j], cw * 3, cb))
                pr = (Xt[s] @ Wsv[j].T + bsv[j]).argmax(1)
                pred_mm += int((pr != prd[j]).sum())
                rr = rows[(rows.method == meth) & (rows.student == s) & (rows.K == K)]
                rr = rr[np.isclose(rr.Cw.astype(float), cw) & np.isclose(rr.Cb.astype(float), cb)] if meth == "srcw_tether" else \
                    rr[np.isclose(rr.beta.astype(float), be) | (np.isinf(be) & np.isinf(rr.beta.astype(float)))]
                uar_mm += int(len(rr) != 1 or abs(100 * uar(yt, prd[j].astype(int)) - float(rr.uar.iloc[0])) > 1e-9)
                if meth == "srcw_tether" and (cw, cb) == (cw0, cb0):          # 固定の点は本番の srcw（β = 5）と同じ問
                    it5 = [json.loads(x)["beta"] for x in z[f"p5_{s}_K{K}_items"]]
                    fixed_pt_agree.append(float((prd[j] == z[f"p5_{s}_K{K}_pred"][it5.index(5.0)]).mean()))
res.update({f"maxdiff_{k}": v for k, v in md.items()}, **cnt)
res.update(n_items_new=n_items, items_bad=items_bad, grad_at_saved_max=float(max(grads)), grad_alt_min=float(min(grads_alt)),
           pred_recompute_mismatch=pred_mm, uar_row_mismatch=uar_mm, tether_fixed_point_agree_min=min(fixed_pt_agree) if fixed_pt_agree else None,
           n_tether_fixed_points=len(fixed_pt_agree))
# 問の数・重複・収束
for meth, want in (("srcw", 84), ("srcw_mod", 32), ("srcw_src", 32), ("srcw_top1", 16), ("srcw_tether", 112)):
    g = rows[rows.method == meth]
    keycols = ["student", "K", "beta"] + (["Cw", "Cb"] if meth == "srcw_tether" else [])
    res[f"rows_{meth}"] = (len(g), want, int(g.duplicated(keycols).sum()), int(((g.grad > 1e-5) | g.stalled.astype(bool)).sum()))
# _poecheck：K = 0 の srcw と実験 1 の tt:poe_none（独立に照合）
pc = rows[rows.method == "_poecheck"]
agree = []
for s in DD.STUDENTS:
    it3 = [json.loads(x) for x in z3[f"p3_{s}_K0_items"]]
    j3 = [j for j, x in enumerate(it3) if x.get("method") == "tt" and x.get("teacher") == "poe_none"]
    agree.append((s, len(j3), float((z[f"p5_{s}_K0_pred"][0] == z3[f"p3_{s}_K0_pred"][j3[0]]).mean())))
res["poecheck_rows"] = [(x.student, bool(x.same_problem), round(float(x.pred_agree), 6)) for x in pc.itertuples()]
res["poecheck_independent"] = agree
# ρ = 0 の自己検査
r0 = rows[rows.method == "_rho0check"]
res["rho0"] = (len(r0), float(r0.pred_agree.min()))
for k, v in res.items():
    print(k, v)
