"""gate-code DAI-4：1 単位の p4 の保存を，独立に（numpy で）作り直して照合する。使い方：verify_unit.py <out_dir> T c f r"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, zlib
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1, dai_data as DD, dai_solver as S

out, T, c, f, r = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
name = f"{T}_{c}_f{f}_r{r}"
ib = Path("/home/snics/MER-DA/results/dai1/pseudo/units") / name
z4 = np.load(out / "pseudo/units" / f"{name}_p4.npz")
rows = pd.read_csv(out / "pseudo/units" / f"{name}_p4.csv", low_memory=False)
z1, z2, z3 = np.load(f"{ib}_p1.npz"), np.load(f"{ib}_p2.npz"), np.load(f"{ib}_p3.npz")
p3rows = pd.read_csv(f"{ib}_p3.csv", low_memory=False)
fz1 = json.load(open("/home/snics/MER-DA/DAI/frozen/dai1_frozen.json"))
srcs = [s for s in DD.others(T) if s != c]
u = dai1.Unit(Path("/home/snics/MER-DA/results/dai1"), T, c, srcs, f, r)
Xp = {k: u.Xp[k].cpu().numpy() for k in DD.KINDS}
y, yt, n = u.ypool, u.ytest, u.n
print("unit", name, "n", n, "ntest", len(yt))
print("npz keys:", len(z4.files), "size MB", (out / "pseudo/units" / f"{name}_p4.npz").stat().st_size / 1e6)
print("methods in rows:", rows.method.value_counts(dropna=False).to_dict())
print("gpu", rows.gpu.iloc[0], "commit", rows.commit.iloc[0][:10], "dirty", rows.dirty.iloc[0])
for col in ["uar", "grad", "S"]:
    v = rows[col].dropna()
    print(col, "NaN/Inf:", int((~np.isfinite(v)).sum()), "range", float(v.min()), float(v.max()))
fin = rows[rows.finite.notna()]
print("finite False:", int((fin.finite.astype(str) == "False").sum()), "stalled:", int(rows.stalled.fillna(False).astype(bool).sum()),
      "max grad:", float(rows.grad.max()))
# 1. 各 (method, student, K, rho, gamma) の重複
cons_rows = rows[rows.method.isin(["cons", "cons_self", "cons_own", "cons_conf", "cons_rand"]) & rows.uar.notna()]
dup = cons_rows.groupby(["method", "student", "K", "rho", "gamma"]).size()
print("rows per method:", cons_rows.groupby("method").size().to_dict(), "dup:", int((dup > 1).sum()))
# 2. 判定 → UAR の作り直し
def uar_np(pred):
    return 100 * np.mean([(pred[yt == k] == k).mean() for k in range(4) if (yt == k).any()])
PFX = {"cons": "p4", "cons_self": "p4s", "cons_own": "p4o", "cons_conf": "p4c", "cons_rand": "p4r"}
mx, n_chk = 0.0, 0
for meth, px in PFX.items():
    for s in DD.STUDENTS:
        for K in dai1.KS_ALL:
            key = f"{px}_{s}_K{K}"
            if f"{key}_items" not in z4.files:
                assert meth == "cons_self" and s == "AVT", (meth, s, K); continue
            items = [json.loads(x) for x in z4[f"{key}_items"]]
            pred = z4[f"{key}_pred"]
            assert pred.shape == (len(items), len(yt)), (key, pred.shape)
            for j, it in enumerate(items):
                assert it["method"] == meth, (it, meth)
                rr = rows[(rows.method == meth) & (rows.student == s) & (rows.K == K) & np.isclose(rows.rho, it["rho"]) & np.isclose(rows.gamma, it["gamma"])]
                assert len(rr) == 1, (meth, s, K, it, len(rr))
                d = abs(uar_np(pred[j]) - float(rr.uar.iloc[0])); mx = max(mx, d); n_chk += 1
print(f"UAR 作り直し：{n_chk} 問，行との最大差 {mx:.2e}")
# 3. 教師と印を numpy float64 で独立に作り直す
def probs(X, W, b):
    L = X @ W.T + b; L -= L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)
bad = dict(teacher=0, mark=0, own=0, conf=0, rand=0, self=0)
nm = {}
for K in dai1.KS_ALL:
    P = {m: probs(Xp[m], z2[f"p2_K{K}_ret_{m}_W"].astype(np.float64), z2[f"p2_K{K}_ret_{m}_b"].astype(np.float64)) for m in "AVT"}
    L = sum(np.log(np.clip(P[m], 1e-12, None)) for m in "AVT"); L -= L.max(1, keepdims=True); Q = np.exp(L); Q /= Q.sum(1, keepdims=True)
    am = {m: P[m].argmax(1) for m in "AVT"}
    tp = Q.argmax(1)
    bad["teacher"] += int((z4[f"p4_K{K}_teacher"] != tp).sum())
    kp = u.kpos[K]; nk = np.setdiff1d(np.arange(n), kp)
    mine_uar = 100 * np.mean([(tp[nk][y[nk] == k] == k).mean() for k in range(4)])
    ref = p3rows[(p3rows.method == "_teacher_stats") & (p3rows.teacher == "one") & (p3rows.K == K)].uar_pool.iloc[0]
    nm[K] = (round(mine_uar, 6), round(float(ref), 6))
    oth = {"A": "VT", "V": "AT", "T": "AV", "AVT": "AVT"}
    for s in DD.STUDENTS:
        o = oth[s]; mk = np.ones(n, bool)
        for a in o[1:]: mk &= am[o[0]] == am[a]
        bad["mark"] += int((z4[f"p4_K{K}_{s}_mark"].astype(bool) != mk).sum())
        cnt = z4[f"p4_K{K}_{s}_counts"]
        assert list(cnt) == [n, mk.sum(), len(nk), mk[nk].sum()], (K, s, cnt, n, mk.sum(), len(nk), mk[nk].sum())
        if s != "AVT":
            allm = (am["A"] == am["V"]) & (am["A"] == am["T"])
            bad["self"] += int((z4[f"p4_K{K}_{s}_markself"].astype(bool) != allm).sum())
        # own
        if K == 0: W, b = u.start[s][0].cpu().numpy(), u.start[s][1].cpu().numpy()
        else:
            cw, cb = fz1[T]["tether"][f"{s}|{K}"]
            its = [str(x) for x in z1[f"p1_{s}_items"]]; j = its.index(f"{K}|{cw}|{cb}")
            W, b = z1[f"p1_{s}_W"][j].astype(np.float64), z1[f"p1_{s}_b"][j].astype(np.float64)
        own = probs(Xp[s], W, b).argmax(1) == tp
        bad["own"] += int((z4[f"p4_K{K}_{s}_markown"].astype(bool) != own).sum())
        # conf
        order = np.argsort(-Q.max(1), kind="stable"); mc = np.zeros(n, bool); mc[order[:int(mk.sum())]] = True
        bad["conf"] += int((z4[f"p4_K{K}_{s}_markconf"].astype(bool) != mc).sum())
        seed = zlib.crc32(f"{T}|{c}|{f}|{r}|{s}|{K}".encode())
        mr = np.zeros(n, bool); mr[np.random.default_rng(seed).permutation(n)[:int(mk.sum())]] = True
        bad["rand"] += int((z4[f"p4_K{K}_{s}_markrand"].astype(bool) != mr).sum())
        assert z4[f"p4_K{K}_{s}_markrand"].sum() == mk.sum()
    # 重み：G を式 m_i Q_ik ρ n /(4 S_k) で作り neff を行と照合
    for s in DD.STUDENTS:
        mk = z4[f"p4_K{K}_{s}_mark"].astype(bool)
        for rho in dai1.RHO_GRID:
            for g in (0.0, 0.25, 0.5, 1.0):
                m = np.where(mk, 1.0, g); Sk = (Q * m[:, None]).sum(0)
                G = Q * m[:, None] * rho * n / (4 * np.maximum(Sk, 1e-12))
                neff = G.sum(0) ** 2 / np.maximum((G ** 2).sum(0), 1e-30)
                wr = rows[(rows.method == "_weights") & (rows.variant == "cons") & (rows.student == s) & (rows.K == K) &
                          np.isclose(rows.rho, rho) & np.isclose(rows.gamma, g)]
                if len(wr) != 1:
                    continue
                d = max(abs(neff[k] - float(wr[f"neff_{k}"].iloc[0])) for k in range(4))
                bad.setdefault("neff_maxdiff", 0.0); bad["neff_maxdiff"] = max(bad["neff_maxdiff"], d)
                bad.setdefault("colsum_ok", 0); bad["colsum_ok"] += int(np.allclose(G.sum(0), rho * n / 4) if g > 0 or (Sk > 1e-12).all() else 1)
print("独立の作り直しとの食い違い：", bad)
print("教師の UAR（作り直し，実験 1 の行）：", nm)
tc = rows[rows.method == "_teacher_check"]
print("_teacher_check match:", tc.match.tolist(), "max |diff|", float(tc.uar_pool_mine.sub(tc.uar_pool_dai1).abs().max()))
g1 = rows[rows.method == "_g1check"]
print("_g1check 行", len(g1), "pred_agree min", float(g1.pred_agree.min()), "W_maxdiff max", float(g1.W_maxdiff.max()), "b_maxdiff max", float(g1.b_maxdiff.max()))
w = rows[rows.method == "_weights"]
print("_weights 行", len(w), "列", [c_ for c_ in w.columns if c_.startswith(("S_", "unmarked", "n_zero", "neff"))][:8])
print("n_zero_weight_classes>0:", int((w.n_zero_weight_classes > 0).sum()), "gamma0 of cons:", int(((w.n_zero_weight_classes > 0) & (w.gamma == 0)).sum()))
print("neff min (cons γ=0):", float(w[(w.variant == "cons") & (w.gamma == 0)][[f"neff_{k}" for k in range(4)]].min().min()))
mk_rows = rows[rows.method == "_marks"]
print("_marks 行", len(mk_rows), "zero_marked_classes sum", int(mk_rows.zero_marked_classes.sum()))
print("share marked (cand) by student K=5:", mk_rows[mk_rows.K == 5][["student", "n_cand", "n_cand_marked"]].assign(share=lambda d: d.n_cand_marked / d.n_cand).round(3).values.tolist())
mc = rows[rows.method == "_marks_ctrl"]
print("_marks_ctrl: n_cand_marked equals cons?", bool(all(mc.n_pool_marked.values[i] == mk_rows.set_index(['student','K']).loc[(mc.student.values[i], mc.K.values[i])].n_pool_marked for i in range(len(mc)))))
print("sample_test equal p1:", bool((z4["sample_test"] == z1["sample_test"]).all()), "sample_pool equal:", bool((z4["sample_pool"] == z1["sample_pool"]).all()))
print("npz has cluster_test:", "cluster_test" in z4.files, "has Q:", any(k.endswith("_Q") for k in z4.files))
print("gpu", str(z4["gpu"]), "commit", str(z4["commit"])[:10], "frozen_sha", repr(str(z4["frozen_sha256"])), "dai1_frozen_sha", str(z4["dai1_frozen_sha256"])[:10])
print("grid in npz:", len(z4["grid"]))
