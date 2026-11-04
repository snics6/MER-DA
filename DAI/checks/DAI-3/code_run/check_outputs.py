"""gate-code の検算（DAI-3）．試走の出力（oute/）と実験 1 の試走の保存（d1e/）を読み，差と件数だけを出す（成績の値は出さない）．
1. 保存済みの値の再現：作り直した全体のテストの UAR と，実験 1 の rows（p1・p3 の csv）の UAR の差
2. 恒等式：和 = 位置 + 方向，和 = 位置' + 方向'（区間の表）
3. b* を scipy で独立に解き直し，交差の当て直しの UAR と fullb を作り直して pickle と比べる
4. テストの sample_id の並び（ラベルの中身ごとの再現率が使う並び）と Saved の並び
5. 候補の上の予測 Q の作り直しと，実験 1 の段 2 の保存（float16）の一致，提案の予測の maxp と段 3 の記録の一致
6. 族 5 の 1 セルの区間を pickle から独立に作り直す．Holm の並び
"""
import sys, json, pickle, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
exec(open(HERE / "run_trial.py").read().split("sys.argv = [")[0])
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize
from scipy import stats as st
import dai3 as D3, dai1, dai_solver as S, dai1_stats as ST, dai_data as DD

D3.RUN["commit"], D3.RUN["dirty"] = dai1.git_commit()
d1e, oute = HERE / "d1e", HERE / "oute"
fz1 = json.load(open(d1e / "select" / "frozen.json"))
fz3 = json.load(open(oute / "frozen.json"))

# 1. 保存済みの値の再現
mx = {}
for T in ("AFEW", "IEMOCAP"):
    for f in (0, 1):
        for r in (0, 1):
            R = pickle.load(open(oute / "eval" / "units" / f"{T}_f{f}_r{r}.pkl", "rb"))
            base = d1e / "eval" / "units" / f"{T}_ALL_f{f}_r{r}"
            p1 = pd.read_csv(f"{base}_p1.csv", low_memory=False); p3 = pd.read_csv(f"{base}_p3.csv", low_memory=False)
            fzT = fz1[T]
            for s in DD.STUDENTS:
                cw, cb = fzT["tether"][f"{s}|5"]
                h = p1[(p1.method == "none") & (p1.kind == s) & (p1.K == 5) & np.isclose(p1.Cw, cw) & np.isclose(p1.Cb, cb)]
                mx.setdefault("none5 全体の UAR − p1 の行", []).append(abs(R["refit"][f"{s}|none|5"]["full"][0] - h.uar.iloc[0]))
                mx.setdefault("none5 判定の数から − p1 の行", []).append(abs(ST.uar_full(R["counts"][(s, 5, "none")]) - h.uar.iloc[0]))
                h0 = p1[(p1.method == "start") & (p1.kind == s)]
                mx.setdefault("start 全体の UAR − p1 の行", []).append(abs(R["refit"][f"{s}|start"]["full"][0] - h0.uar.iloc[0]))
                for K in (0, 5):
                    v = fzT["main"][str(K)]; j = fzT["select"][f"{s}|{K}|joint|{v}"]
                    hh = p3[(p3.stage == 3) & (p3.student == s) & (p3.K == K) & (p3.method == "joint") & (p3.teacher == v)
                            & np.isclose(p3.rho, float(j["rho"])) & np.isclose(p3.a, float(j["a"]))]
                    mx.setdefault("prop 全体の UAR − p3 の行", []).append(abs(R["refit"][f"{s}|prop|{K}"]["full"][0] - hh.uar.iloc[0]))
                ocw, ocb = fz3["oracle_tether"][T][s]
                ho = p1[(p1.method == "oracle") & (p1.kind == s) & np.isclose(p1.Cw, ocw) & np.isclose(p1.Cb, ocb)]
                mx.setdefault("oracle 全体の UAR − p1 の行", []).append(abs(R["refit"][f"{s}|oracle"]["full"][0] - ho.uar.iloc[0]))
                kb = float(fzT["select"][f"{s}|5|kbias|-"]["Cb"])
                hk = p1[(p1.method == "kbias") & (p1.kind == s) & (p1.K == 5) & np.isclose(p1.Cb, kb)]
                mx.setdefault("kbias 判定の数から − p1 の行", []).append(abs(ST.uar_full(R["counts"][(s, 5, "kbias")]) - hk.uar.iloc[0]))
for k, v in mx.items():
    print(f"1. {k}：最大の差 {max(v):.2e}（{len(v)} 個）")

# 2. 恒等式
d = pd.read_csv(oute / "eval" / "decomp.csv")
p = d.pivot_table(index=["T", "item", "version"], columns="quantity", values="effect")
print("2. 和 − (位置 + 方向) の最大", f"{(p['sum'] - p['position'] - p['direction']).abs().max():.2e}",
      "／ 和 − (位置' + 方向') の最大", f"{(p['sum'] - p['position_swap'] - p['direction_swap']).abs().max():.2e}",
      "／ (位置 − 水準) − (位置 − position_null) の最大", f"{(p['position_minus_null'] - (p['position'] - p['position_null'])).abs().max():.2e}",
      "／ 行", len(p))

# 3. b* の独立の解き直し
def bstar_scipy(Z, yl, m, k):
    sel = np.flatnonzero(m & (yl >= 0)); ys = yl[sel]
    cnt = np.bincount(ys, minlength=k); w = len(sel) / (k * cnt[ys])
    def obj(b):
        zz = Z[sel] + b
        mm = zz.max(1, keepdims=True)
        lse = np.log(np.exp(zz - mm).sum(1)) + mm[:, 0]
        p_ = np.exp(zz - lse[:, None])
        g = (w[:, None] * (p_ - np.eye(k)[ys])).sum(0) / w.sum()
        return float((w * (lse - zz[np.arange(len(sel)), ys])).sum() / w.sum()), g
    return minimize(obj, np.zeros(k), jac=True, method="BFGS", options={"gtol": 1e-12, "maxiter": 10000}).x

T, f, r = "AFEW", 1, 0
R = pickle.load(open(oute / "eval" / "units" / f"{T}_f{f}_r{r}.pkl", "rb"))
sv = D3.Saved(d1e, "eval", T, T, f, r, fz1[T])
grp = D3.test_halves(sv.y, sv.cl, sv.sid)
keys, keep = D3.rep_keep(sv.cl, sv.mode, sv.gmap)
dif = []
for name in ("A|none|5", "AVT|prop|5", "V|oracle", "T|start"):
    kind = name.split("|")[0]
    W, b = {"A|none|5": lambda: sv.none("A", 5), "AVT|prop|5": lambda: sv.prop("AVT", 5),
            "V|oracle": lambda: sv.oracle("V", *fz3["oracle_tether"][T]["V"]), "T|start": lambda: sv.u.start["T"]}[name]()
    Z = D3.scores(sv.u.Xt[kind], W).cpu().numpy(); bn = b.cpu().numpy()
    for vi, cls in enumerate(D3.VERSIONS):
        k = len(cls); yl = D3.local_labels(sv.y, cls); Zc = Z[:, list(cls)]
        for j in (0, 1, len(keys) - 1):
            if not R["refit"][name]["valid"][vi, j]:
                continue
            A = [keep[j] & (grp == g) & (yl >= 0) for g in (0, 1)]
            def uar(bb, m):
                pr = np.argmax(Zc + bb, 1)
                return 100 * np.mean([np.mean(pr[m & (yl == q)] == q) for q in range(k)])
            ba, bb_ = bstar_scipy(Zc, yl, A[0], k), bstar_scipy(Zc, yl, A[1], k)
            bf = bstar_scipy(Zc, yl, A[0] | A[1], k)
            refit = 0.5 * (uar(ba, A[1]) + uar(bb_, A[0])); fullb = 0.5 * (uar(bf, A[0]) + uar(bf, A[1]))
            orig = 0.5 * (uar(bn[list(cls)], A[0]) + uar(bn[list(cls)], A[1]))
            dif.append((abs(refit - R["refit"][name]["refit"][vi, j]), abs(fullb - R["refit"][name]["fullb"][vi, j]),
                        abs(orig - R["refit"][name]["orig"][vi, j])))
dif = np.array(dif)
print(f"3. scipy で解いた b* との差（UAR のポイント）：refit {dif[:,0].max():.2e}，fullb {dif[:,1].max():.2e}，orig {dif[:,2].max():.2e}，"
      f"比べた値 {len(dif)}（うち refit の差が 1e-6 を超えた数 {(dif[:,0] > 1e-6).sum()}）")

# 4. sample_id の並び
bad = 0
for T in ("AFEW", "IEMOCAP"):
    df = D3.corpus_df(T)
    for f in (0, 1):
        sv2 = D3.Saved(d1e, "eval", T, T, f, 0, fz1[T])
        sid_all = df["sample_id"].to_numpy().astype(str)[df["fold"].to_numpy() == f]
        bad += int(not (sid_all == sv2.sid).all())
print("4. recall_content の並び（df の分割の順）と Saved の sample_id の並びが違う単位の数", bad)

# 5. 候補の上の Q の作り直しと段 2 の保存
p2 = sv.p2
TK = dai1.build_tk(sv.u, sv.p1, sv.p2, sv.teth)
res = []
for K in (0, 5):
    for name in ["none_A", "none_V", "none_T", "none_AVT", "ret_A", "ret_V", "ret_T", "poe_none"]:
        Q = (TK[K][name][0] if name != "poe_none" else dai1.poe(TK[K], "AVT", 0)).cpu().numpy()
        Qs = p2[f"p2_K{K}_{name}_pool"].astype(np.float64)
        res.append((K, name, float(np.abs(Q - Qs).max()), float((Q.argmax(1) == Qs.argmax(1)).mean())))
print("5. 作り直した Q と段 2 の保存（float16）：確率の差の最大", f"{max(x[2] for x in res):.2e}", "判定の一致の最小", f"{min(x[3] for x in res):.4f}")
p3c = pd.read_csv(f"{sv.base}_p3.csv", low_memory=False)
for K in (0, 5):
    for nm, tv in (("prop_sel", "sel"), ("prop_one", "one")):
        Q = D3.pool_teachers(sv, TK[K], "A", K, [nm])[nm]
        h = p3c[(p3c.method == "_teacher_stats") & (p3c.teacher == tv) & (p3c.K == K) & p3c.student.isna()]
        print(f"5b. {nm} K={K}：maxp の差 {abs(float(Q.max(1).values.mean()) - h.maxp.iloc[0]):.2e}")
    for nm in ("single_V", "src_IEMOCAP_T", "start_AVT", "noret_sel"):
        Q = D3.pool_teachers(sv, TK[K], "A", K, [nm])[nm]
        h = p3c[(p3c.method == "_teacher_stats") & (p3c.teacher == f"tt:{nm}") & (p3c.K == K) & (p3c.student == "A")]
        print(f"5c. tt:{nm} K={K}：maxp の差 {abs(float(Q.max(1).values.mean()) - h.maxp.iloc[0]):.2e}")

# 6. 族 5 の 1 セルを独立に作り直す（AFEW，塊の jackknife）
T = "AFEW"
U = {(f, r): pickle.load(open(oute / "eval" / "units" / f"{T}_f{f}_r{r}.pkl", "rb")) for f in (0, 1) for r in (0, 1)}
fam = pd.read_csv(oute / "eval" / "family5.csv")
out = []
for s in DD.STUDENTS:
    for vi in range(1, 7):
        dfr = {}; th = {}
        for (f, r), R in U.items():
            a, b = R["refit"][f"{s}|prop|5"], R["refit"][f"{s}|none|5"]
            v = a["refit"][vi] - b["refit"][vi]
            v = np.where(a["valid"][vi], v, np.nan)
            dfr[(f, r)] = dict(zip(R["keys"], v))
        folds = sorted({f for f, _ in dfr})
        est = np.mean([np.mean([dfr[(f, r)]["full"] for r in (0, 1)]) for f in folds])
        cls_all = sorted({g for (f, r), d in dfr.items() for g in d if g != "full"})
        jk, D = [], set()
        for g in cls_all:
            fv = []
            for f in folds:
                xs = [dfr[(f, r)].get(g, dfr[(f, r)]["full"]) for r in (0, 1)]
                xs = [x for x in xs if np.isfinite(x)]
                if xs: fv.append(np.mean(xs))
                else: D.add(f)
            jk.append(np.mean(fv))
        jk = np.array(jk); G = len(jk)
        vjk = (G - 1) / G * ((jk - jk.mean()) ** 2).sum()
        vs = sum(np.var([dfr[(f, r)]["full"] for r in (0, 1)], ddof=1) / 2 for f in folds if f not in D) / len(folds) ** 2
        n = pd.Series([c for R in [U[(f, 0)] for f in folds] for c, m in R["sizes"].items() for _ in range(m)]).value_counts().to_numpy()
        geff = n.sum() ** 2 / (n ** 2).sum(); dfree = max(1, int(np.floor(geff)) - 1)
        se = math.sqrt(vjk + vs); p = 2 * st.t.sf(abs(est) / se, dfree)
        row = fam[(fam["T"] == T) & (fam.student == s) & (fam.pair == D3.VNAME[vi])].iloc[0]
        out.append((abs(est - row.effect), abs(se - row.se), abs(p - row.p), dfree == row.df))
out = np.array(out, dtype=float)
print(f"6. 族 5（AFEW 24 セル）の作り直しとの差：効果 {out[:,0].max():.2e}，標準誤差 {out[:,1].max():.2e}，p {out[:,2].max():.2e}，自由度の一致 {int(out[:,3].sum())}/24")
fa = fam[fam["T"] == T]
print("6b. Holm の作り直しとの差", f"{np.abs(ST.holm(fa.p.fillna(1).to_numpy()) - fa.p_holm.to_numpy()).max():.2e}",
      "／ p ≥ p_holm でない行", int((fa.p_holm + 1e-12 < fa.p).sum()))
