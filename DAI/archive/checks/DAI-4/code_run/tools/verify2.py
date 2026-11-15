"""最新版（b018ae0）の p4 の保存を独立に作り直して照合する。使い方：verify2.py <out_dir> T c f r"""
import sys, json, zlib
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd
import dai1, dai_data as DD
out, T, c, f, r = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
name = f"{T}_{c}_f{f}_r{r}"; ib = Path("/home/snics/MER-DA/results/dai1/pseudo/units") / name
z4 = np.load(out / "pseudo/units" / f"{name}_p4.npz"); rows = pd.read_csv(out / "pseudo/units" / f"{name}_p4.csv", low_memory=False)
z1, z2, z3 = np.load(f"{ib}_p1.npz"), np.load(f"{ib}_p2.npz"), np.load(f"{ib}_p3.npz")
fz1 = json.load(open("/home/snics/MER-DA/DAI/frozen/dai1_frozen.json"))
u = dai1.Unit(Path("/home/snics/MER-DA/results/dai1"), T, c, [s for s in DD.others(T) if s != c], f, r)
Xp = {k: u.Xp[k].cpu().numpy() for k in DD.KINDS}; y, yt, n = u.ypool, u.ytest, u.n
print("unit", name, "n", n, "ntest", len(yt), "npz MB", round((out / "pseudo/units" / f"{name}_p4.npz").stat().st_size / 1e6, 1))
print("methods:", rows.method.value_counts().to_dict()); print("gpu", rows.gpu.iloc[0], "commit", rows.commit.iloc[0][:10], "dirty", rows.dirty.iloc[0])
for col in ["uar", "grad", "S"]:
    v = rows[col].dropna(); print(col, "NaN/Inf", int((~np.isfinite(v)).sum()), "range", round(float(v.min()), 3), round(float(v.max()), 3))
print("finite False", int((rows.finite.astype(str) == "False").sum()), "stalled", int(rows.stalled.fillna(False).astype(bool).sum()), "max grad", float(rows.grad.max()))
M = ["cons", "cons_self", "cons_own", "cons_conf", "cons_rand"]; PFX = dict(zip(M, ["p4", "p4s", "p4o", "p4c", "p4r"]))
cr = rows[rows.method.isin(M) & rows.uar.notna()]; print("rows/method", cr.groupby("method").size().to_dict(), "dup", int((cr.groupby(["method", "student", "K", "rho", "gamma"]).size() > 1).sum()))
def uar_np(p): return 100 * np.mean([(p[yt == k] == k).mean() for k in range(4) if (yt == k).any()])
mx = 0; nchk = 0
for meth, px in PFX.items():
    for s in DD.STUDENTS:
        for K in dai1.KS_ALL:
            key = f"{px}_{s}_K{K}"
            if f"{key}_items" not in z4.files: assert meth == "cons_self" and s == "AVT"; continue
            its = [json.loads(x) for x in z4[f"{key}_items"]]
            for j, it in enumerate(its):
                rr = rows[(rows.method == meth) & (rows.student == s) & (rows.K == K) & np.isclose(rows.rho, it["rho"]) & np.isclose(rows.gamma, it["gamma"])]
                assert len(rr) == 1; mx = max(mx, abs(uar_np(z4[f"{key}_pred"][j]) - float(rr.uar.iloc[0]))); nchk += 1
print(f"UAR 作り直し {nchk} 問 最大差 {mx:.1e}")
def probs(X, W, b):
    L = X @ W.T + b; L -= L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)
bad = {k: 0 for k in ("teacher", "am", "Q", "mark", "own", "conf", "rand", "self")}; wd = dict(neff=0.0, S=0.0, unm=0.0); per_emotion_ok = True; nrow = 0
for K in dai1.KS_ALL:
    P = {m: probs(Xp[m], z2[f"p2_K{K}_ret_{m}_W"].astype(np.float64), z2[f"p2_K{K}_ret_{m}_b"].astype(np.float64)) for m in "AVT"}
    L = sum(np.log(np.clip(P[m], 1e-12, None)) for m in "AVT"); L -= L.max(1, keepdims=True); Q = np.exp(L); Q /= Q.sum(1, keepdims=True)
    am = {m: P[m].argmax(1) for m in "AVT"}; tp = Q.argmax(1); qmax = Q.max(1)
    bad["teacher"] += int((z4[f"p4_K{K}_teacher"] != tp).sum()); bad["Q"] += int(not np.allclose(z4[f"p4_K{K}_Q"], Q, atol=1e-6))
    bad["am"] += sum(int((z4[f"p4_K{K}_ret_{m}_am"] != am[m]).sum()) for m in "AVT")
    nk = np.setdiff1d(np.arange(n), u.kpos[K]); oth = {"A": "VT", "V": "AT", "T": "AV", "AVT": "AVT"}
    for s in DD.STUDENTS:
        o = oth[s]; mk = np.ones(n, bool)
        for a in o[1:]: mk &= am[o[0]] == am[a]
        bad["mark"] += int((z4[f"p4_K{K}_{s}_mark"].astype(bool) != mk).sum())
        if s != "AVT": bad["self"] += int((z4[f"p4_K{K}_{s}_markself"].astype(bool) != ((am["A"] == am["V"]) & (am["A"] == am["T"]))).sum())
        if K == 0: W, b = u.start[s][0].cpu().numpy(), u.start[s][1].cpu().numpy()
        else:
            cw, cb = fz1[T]["tether"][f"{s}|{K}"]; its = [str(x) for x in z1[f"p1_{s}_items"]]; jj = its.index(f"{K}|{cw}|{cb}")
            W, b = z1[f"p1_{s}_W"][jj].astype(np.float64), z1[f"p1_{s}_b"][jj].astype(np.float64)
        bad["own"] += int((z4[f"p4_K{K}_{s}_markown"].astype(bool) != (probs(Xp[s], W, b).argmax(1) == tp)).sum())
        rng = np.random.default_rng(zlib.crc32(f"{T}|{c}|{f}|{r}|{s}|{K}".encode())); mc = np.zeros(n, bool); mr = np.zeros(n, bool)
        for k in range(4):
            ik = np.flatnonzero(tp == k); nm = int(mk[ik].sum())
            mc[ik[np.argsort(-qmax[ik], kind="stable")[:nm]]] = True; mr[ik[rng.permutation(len(ik))[:nm]]] = True
        bad["conf"] += int((z4[f"p4_K{K}_{s}_markconf"].astype(bool) != mc).sum()); bad["rand"] += int((z4[f"p4_K{K}_{s}_markrand"].astype(bool) != mr).sum())
        for tg in ("conf", "rand"):
            mm = z4[f"p4_K{K}_{s}_mark{tg}"].astype(bool)
            per_emotion_ok &= all(mm[tp == k].sum() == mk[tp == k].sum() for k in range(4))
        for rho in dai1.RHO_GRID:
            for g in (0.0, 0.25, 0.5, 1.0):
                m_ = np.where(mk, 1.0, g); Sk = (Q * m_[:, None]).sum(0); G = Q * m_[:, None] * rho * n / (4 * np.maximum(Sk, 1e-12))
                neff = G.sum(0) ** 2 / np.maximum((G ** 2).sum(0), 1e-30); unm = G[~mk].sum(0) / np.maximum(G.sum(0), 1e-30)
                wr = rows[(rows.method == "_weights") & (rows.variant == "cons") & (rows.student == s) & (rows.K == K) & np.isclose(rows.rho, rho) & np.isclose(rows.gamma, g)]
                if len(wr) != 1: continue
                nrow += 1
                wd["neff"] = max(wd["neff"], max(abs(neff[k] - float(wr[f"neff_{k}"].iloc[0])) for k in range(4)))
                wd["S"] = max(wd["S"], max(abs(Sk[k] - float(wr[f"S_{k}"].iloc[0])) for k in range(4)))
                wd["unm"] = max(wd["unm"], max(abs(unm[k] - float(wr[f"unmarked_wfrac_{k}"].iloc[0])) for k in range(4)))
print("独立の作り直しとの食い違い", bad, "感情ごとの印の数が本番と同じ（conf・rand）", per_emotion_ok)
print("重みの行", nrow, "最大差", wd)
tc = rows[rows.method == "_teacher_check"]; print("_teacher_check", tc.match.tolist(), "max|diff|", float(tc.uar_pool_diff.abs().max()))
g1 = rows[rows.method == "_g1check"]; print("_g1check", len(g1), "pred_agree min", float(g1.pred_agree.min()), "W max", float(g1.W_maxdiff.max()), "b max", float(g1.b_maxdiff.max()))
w = rows[rows.method == "_weights"]; print("n_zero_weight_classes>0 行", int((w.n_zero_weight_classes > 0).sum()), "neff min cons γ=0", round(float(w[(w.variant == "cons") & (w.gamma == 0)][[f"neff_{k}" for k in range(4)]].min().min()), 1))
mkr = rows[rows.method == "_marks"]; print("_marks zero_marked_classes", int(mkr.zero_marked_classes.sum()), "n_cls sum==n_cand", bool(all(mkr[[f"n_cls{k}" for k in range(4)]].sum(1) == mkr.n_cand)))
print("npz: cluster_test", "cluster_test" in z4.files, "Q", f"p4_K5_Q" in z4.files, "grid", len(z4["grid"]), "gpu", str(z4["gpu"]), "commit", str(z4["commit"])[:10], "frozen_sha", repr(str(z4["frozen_sha256"])))
