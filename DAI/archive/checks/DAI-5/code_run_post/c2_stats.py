"""計算の後の点検 2：集計の表の行の数・重複・選んだ β の読み方と，int8 の判定とラベルからの作り直し（台本の関数は使わない）。"""
import json, itertools, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA"); S = ROOT / "results/dai5/stats"; U5 = ROOT / "results/dai5/eval/units"; U1 = ROOT / "results/dai1/eval/units"
Ts, STU, KS, BET = ["AFEW", "IEMOCAP", "MOSEI", "MELD"], ["A", "V", "T", "AVT"], [0, 1, 5, 10, 25], [0.0, 2.5, 5.0, 10.0, 20.0]
fz5 = json.load(open(ROOT / "results/dai5/select/frozen.json")); fz1 = json.load(open(ROOT / "DAI/frozen/dai1_frozen.json"))
sel = {(T, int(K)): float(v["beta"]) for T in Ts for K, v in fz5[T]["select"].items()}
selv = {(T, k.split("|")[0], int(k.split("|")[1])): float(v["beta"]) for T in Ts for k, v in fz5[T]["select_variants"].items()}
splits = {T: pd.read_csv(ROOT / f"DAI/splits/{T}.csv", keep_default_na=False) for T in Ts}
ylab = {T: dict(zip(splits[T].sample_id.astype(str), splits[T].label_4.astype(int))) for T in Ts}
rd = {n: pd.read_csv(S / f"{n}.csv", low_memory=False) for n in ["families7", "desc", "regret", "weights", "scores", "teacher_check", "teacher_check_units", "teacher_sharp",
      "src_agree", "poe_check", "rule_b_chance", "gpu_counts", "sensitivity", "recall", "confusion", "src_recall", "premise_pseudo", "premise_pseudo_by_c", "premise_pseudo_units", "pairing"]}
print("== 1. 行の数と重複")
exp_rows = dict(families7=32, desc=16 * (10 + 19 + 19 + 47 + 47), regret=4 * 4 * 5 * 3, weights=20, scores=16, teacher_check=20, teacher_check_units=200 * 5 * 9,
                teacher_sharp=200 * 5, src_agree=4 * 36, poe_check=16, rule_b_chance=1, gpu_counts=4, sensitivity=8 + 8 + 32, recall=96, confusion=96, src_recall=16,
                premise_pseudo=160, premise_pseudo_by_c=480, premise_pseudo_units=4800, pairing=200)
keys = dict(families7=["T", "student", "K"], desc=["T", "student", "K", "A", "B", "read", "Cw", "Cb"], regret=["T", "student", "K", "which"], weights=["T", "K"], scores=["T", "K"],
            teacher_check=["T", "K"], teacher_check_units=["T", "fold", "seed", "K", "src"], teacher_sharp=["T", "fold", "seed", "K"], src_agree=["T", "a", "b"],
            poe_check=["T", "student"], gpu_counts=["T"], sensitivity=["T", "subset", "student", "K", "B"], recall=["T", "student", "K", "row"], confusion=["T", "student", "K", "row"],
            src_recall=["T", "K"], premise_pseudo=["T", "student", "K", "which"], premise_pseudo_by_c=["T", "c", "student", "K", "which"],
            premise_pseudo_units=["T", "c", "fold", "seed", "student", "K", "which"], pairing=["T", "fold", "seed"], rule_b_chance=["n_cells"])
for n, d in rd.items():
    k = keys[n]; dup = int(d.assign(**{c: d[c].fillna(-1) for c in k if c in d}).duplicated(k).sum())
    num = d.select_dtypes("number"); ninf = int(np.isinf(num.to_numpy(float)).sum())
    cm = set(d.commit.astype(str)) if "commit" in d else {"-"}; uc = set(d.unit_commit.astype(str)) if "unit_commit" in d else {"-"}
    print(f"{n}: 行 {len(d)}（期待 {exp_rows[n]}），重複 {dup}，Inf {ninf}，commit {[c[:7] for c in cm]}，unit_commit {[c[:7] for c in uc]}，dirty {set(d.dirty.astype(str)) if 'dirty' in d else '-'}")
f7 = rd["families7"]
print("族 7 の NaN の数（effect・lo・hi・p・p_holm）", int(f7[["effect", "lo", "hi", "p", "p_holm"]].isna().sum().sum()))
print("desc の effect の NaN", int(rd["desc"].effect.isna().sum()), "．うち read=fixed_point_check 以外の lo の NaN", int(rd["desc"][rd["desc"].read.fillna("") != "fixed_point_check"].lo.isna().sum()))
print("== 2. 選んだ β の読み方")
print("族 7 の beta と固定のファイルの食い違い", int(sum(abs(r.beta - sel[(r.T, r.K)]) > 0 for r in f7.itertuples())))
dd = rd["desc"]; base = dd[dd.read.fillna("") == ""]
print("desc（read 空）の beta と固定の食い違い", int(sum(abs(r.beta - sel[(r.T, r.K)]) > 0 for r in base.itertuples())))
vv = dd[dd.read.fillna("") == "variant_vs_b0"]
print("desc の軸を分けた版の beta と固定の食い違い", int(sum(abs(r.beta_variant - selv[(r.T, r.A, r.K)]) > 0 for r in vv.itertuples())), "/", len(vv))
w = rd["weights"]; print("weights の beta と固定の食い違い", int(sum(abs(r.beta - sel[(r.T, r.K)]) > 0 for r in w.itertuples())))
# 3. 作り直し
def uar(y, p): return 100 * np.mean([(p[y == c] == c).mean() for c in range(4) if (y == c).any()])
def p3item(z1, s, K, **want):
    it = [json.loads(x) for x in z1[f"p3_{s}_K{K}_items"]]
    h = [j for j, x in enumerate(it) if all((str(x.get(k)) == str(v)) if isinstance(v, str) else (x.get(k) is not None and np.isclose(float(x.get(k)), float(v))) for k, v in want.items())]
    assert len(h) == 1, (s, K, want, len(h)); return z1[f"p3_{s}_K{K}_pred"][h[0]]
def p5pred(z5, s, K, beta, px="p5"):
    it = [json.loads(x) for x in z5[f"{px}_{s}_K{K}_items"]]
    h = [j for j, x in enumerate(it) if (np.isinf(beta) and np.isinf(x["beta"])) or np.isclose(x["beta"], beta)]
    assert len(h) == 1; return z5[f"{px}_{s}_K{K}_pred"][h[0]]
print("== 3. 作り直し（全 200 単位）")
D7 = {}; IDENT = {}; POE = []; SC = []; WS = {}
for T in Ts:
    for f in range(5):
        for r in range(10):
            nm = f"{T}_ALL_f{f}_r{r}"
            z5 = np.load(U5 / f"{nm}_p5.npz"); z1 = np.load(U1 / f"{nm}_p3.npz")
            y = np.array([ylab[T][x] for x in z5["sample_test"].astype(str)])
            for s in STU:
                for K in KS:
                    j = fz1[T]["select"][f"{s}|{K}|joint|one"]
                    pj = p3item(z1, s, K, method="joint", teacher="one", rho=j["rho"], a=j["a"])
                    ps = p5pred(z5, s, K, sel[(T, K)] if K else 0.0)
                    pb = p5pred(z5, s, K, 0.0)
                    D7.setdefault((T, s, K), {}).setdefault(f, []).append(uar(y, ps) - uar(y, pj))
                    IDENT.setdefault((T, s, K), []).append(bool((ps == pb).all()))
                if True:
                    j0 = fz1[T]["select"][f"{s}|0|joint|one"]
                    POE.append(dict(T=T, s=s, f=f, r=r, same=bool(float(j0["rho"]) == 1.0 and float(j0["a"]) == 1.0),
                                    agree=float((p5pred(z5, s, 0, 0.0) == p3item(z1, s, 0, method="tt", teacher="poe_none")).mean())))
            tp = z5["p5_src_test_pred"].astype(int); tu = np.array([uar(y, tp[j]) for j in range(len(tp))]); names = [str(x) for x in z5["p5_src_names"]]
            for K in (1, 5, 10, 25):
                sc = z5[f"p5_K{K}_scores"]; top = np.flatnonzero(sc >= sc.max() - 1e-12)
                SC.append(dict(T=T, f=f, r=r, K=K, reg=tu[top].mean() - tu.max(), null=(tu - tu.max()).mean()))
                for b in BET:
                    e = np.exp(b * sc / 100 - (b * sc / 100).max()); wm = 3 * e / e.sum()
                    WS.setdefault((T, K, b), []).append(np.abs(wm - z5[f"p5_K{K}_w_b{b}"]).max())
                b = sel[(T, K)]
                e = np.exp(b * sc / 100 - (b * sc / 100).max()); WS.setdefault(("share", T, K), []).append(e / e.sum())
                WS.setdefault(("names", T), names)
theta = lambda d: float(np.mean([np.mean(v) for v in d.values()]))
diffs = [abs(theta(D7[(r.T, r.student, r.K)]) - r.effect) for r in f7.itertuples()]
print(f"(a) 族 7 の点推定：32 セルすべてを作り直し，差の最大 {max(diffs):.2e}")
for r in f7.head(0).itertuples(): pass
print("    例：", [(r.T, r.student, r.K, round(r.effect, 4), round(theta(D7[(r.T, r.student, r.K)]), 4)) for r in f7.iloc[[0, 9, 18, 27]].itertuples()])
# Holm
hd = []
for T, g in f7.groupby("T"):
    p = g.p.to_numpy(); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, j in enumerate(o): run = max(run, min(1, (len(p) - i) * p[j])); adj[j] = run
    hd.append(np.abs(adj - g.p_holm.to_numpy()).max())
print(f"    Holm（コーパスごと 8 セル）を p から作り直した差の最大 {max(hd):.2e}")
# K = 0 の srcw − joint（記述）
k0 = dd[(dd.read.fillna("") == "") & (dd.B == "joint") & (dd.K == 0)]
print(f"    K = 0 の desc の srcw − joint 16 行と作り直しの差の最大 {max(abs(theta(D7[(r.T, r.student, 0)]) - r.effect) for r in k0.itertuples()):.2e}")
# (b) identical
b0 = dd[(dd.read.fillna("") == "") & (dd.B == "srcw_b0")]
n_sel0 = sum(1 for (T, s, K) in IDENT if (K == 0 or sel[(T, K)] == 0.0)); all_same = sum(1 for (T, s, K), v in IDENT.items() if (K == 0 or sel[(T, K)] == 0.0) and all(v))
flag_ok = sum(1 for r in b0.itertuples() if bool(r.identical_by_construction) == (r.K == 0 or sel[(r.T, r.K)] == 0.0))
eff0 = sum(1 for r in b0.itertuples() if r.identical_by_construction and r.effect == 0.0 and bool(r.identical))
nonsel_same = [(T, s, K) for (T, s, K), v in IDENT.items() if K > 0 and sel[(T, K)] != 0.0 and all(v)]
nonsel_any = sum(1 for (T, s, K), v in IDENT.items() if K > 0 and sel[(T, K)] != 0.0 and any(v))
print(f"(b) β = 0 が選ばれた (T, 生徒, K) {n_sel0} 個（K = 0 を含む）のうち，全 50 単位で srcw と srcw_b0 の判定が同じもの {all_same}．desc の identical_by_construction の印が規則と合う行 {flag_ok}/{len(b0)}，"
      f"印のある行で effect が厳密に 0 かつ identical が真 {eff0}/{int(b0.identical_by_construction.sum())}．β ≠ 0 なのに全単位で判定が同じセル {nonsel_same}，1 単位でも同じセル {nonsel_any}")
# (c) poe
P = pd.DataFrame(POE); g = P.groupby(["T", "s", "same"]).agree.agg(["mean", "min"]).reset_index()
pc = rd["poe_check"].merge(g, left_on=["T", "student", "same_problem"], right_on=["T", "s", "same"])
print(f"(c) K = 0 の srcw と tt:poe_none：同じ問のセル {int(g.same.sum())}/16，そのうち全単位で一致率 1.0 のセル {int(((g['min'] == 1.0) & g.same).sum())}．"
      f"違う問のセルの一致率の平均 {g[~g.same]['mean'].round(4).tolist()}．poe_check.csv との差の最大 {float((pc.pred_agree_mean - pc['mean']).abs().max()):.1e}（突き合わせた行 {len(pc)}）")
# (d) scores
SCd = pd.DataFrame(SC); sr = rd["scores"]
for T, K in (("AFEW", 25), ("MOSEI", 25), ("IEMOCAP", 5)):
    g = SCd[(SCd["T"] == T) & (SCd.K == K)].groupby("f")[["reg", "null"]].mean()
    v = (g.reg - g.null).to_numpy(); G = len(v); jk = np.array([np.delete(v, i).mean() for i in range(G)]); se = np.sqrt((G - 1) / G * ((jk - jk.mean()) ** 2).sum())
    from scipy import stats as st
    lo = v.mean() - st.t.ppf(0.975, G - 1) * se
    row = sr[(sr["T"] == T) & (sr.K == K)].iloc[0]
    usable = bool(K == 25 and g.reg.mean() >= 0.5 * g.null.mean() and lo > 0)
    print(f"(d) {T} K={K}：採点で選んだ損 作り直し {g.reg.mean():.4f} / 表 {row.regret_sel_mean:.4f}，でたらめの損 {g.null.mean():.4f} / {row.null_mean_mean:.4f}，"
          f"diff_vs_null の下端 {lo:.4f} / {row.diff_vs_null_lo:.4f}，usable_rule 作り直し {usable} / 表 {row.usable_rule}")
print("    全 16 行の usable_rule を表の列から作り直した食い違い",
      int(sum(bool(r.K == 25 and r.regret_sel_mean >= 0.5 * r.null_mean_mean and r.diff_vs_null_lo > 0) != bool(r.usable_rule) for r in sr.itertuples())))
# (e) weights
print(f"(e) 保存した重みと，保存した採点から作り直した 3·softmax(βU/100) の差の最大 {max(max(v) for k, v in WS.items() if len(k) == 3 and isinstance(v[0], float) ):.2e}（全単位・K ≥ 1・β 5 点）")
for T, K in (("IEMOCAP", 25), ("AFEW", 25), ("MOSEI", 10)):
    sh = np.mean(WS[("share", T, K)], 0); names = WS[("names", T)]; row = w[(w["T"] == T) & (w.K == K)].iloc[0]
    tab = np.array([row[f"share_{n}"] for n in names])
    mods = {m: sum(sh[j] for j, n in enumerate(names) if n.endswith("_" + m)) for m in "AVT"}
    print(f"    {T} K={K} β={sel[(T, K)]}：指数の割合の差の最大 {np.abs(sh - tab).max():.2e}．モダリティの和 作り直し {{{', '.join(f'{m}: {100*v:.1f}%' for m, v in mods.items())}}} / 表 "
          f"{{{', '.join(f'{m}: {100*row[f'share_mod_{m}']:.1f}%' for m in 'AVT')}}}")
# 5. 規則の列
print("== 5. 規則の列")
avt = b0[(b0.student == "AVT") & b0.K.isin([5, 25])]
def rule(r, d):
    if r.identical_by_construction: return ""
    return "差が小さい" if (-d < r.lo and r.hi < d) else ("見込みと違う" if ((r.lo > 0 or r.hi < 0) and abs(r.effect) > d) else "判定できない")
print("rule_d：AVT × K{5,25} の行", len(avt), "，作り直しとの食い違い", int(sum(rule(r, r.delta_same) != (r.rule_d if isinstance(r.rule_d, str) else "") for r in avt.itertuples())),
      "，rule_d が空でない行", int(avt.rule_d.notna().sum()), "，AVT 以外で rule_d がある行", int(dd[(dd.student != "AVT")].rule_d.notna().sum()))
iw = dd[dd.K > 0]; print("interval_within_delta の作り直しの食い違い（K > 0，lo のある行）",
      int(sum(bool(-r.delta_same < r.lo and r.hi < r.delta_same) != bool(r.interval_within_delta) for r in iw.itertuples() if pd.notna(r.lo))))
rg = rd["regret"]; print("split_half_regret_ge_delta の作り直しの食い違い", int(sum(bool(pd.notna(r.split_half_regret) and r.split_half_regret >= r.delta_same) != bool(r.split_half_regret_ge_delta) for r in rg.itertuples())),
      "．split_half_regret の NaN", int(rg.split_half_regret.isna().sum()), "．使った分割の数", sorted(rg.split_half_folds_used.unique()))
print("regret の同じテストの値が負の行（生徒ごとの選び方の行で）", int(((rg.which != "srcw_shared_best") & (rg.effect < -1e-12)).sum()), "．shared_best の負の行", int(((rg.which == "srcw_shared_best") & (rg.effect < -1e-12)).sum()))
print("regret の n_grid_points", rg.groupby("which").n_grid_points.unique().to_dict())
# rule_b_chance：MELD の identical と AFEW の 3 セルで作り直す
vecs = {}
for T in ("AFEW", "MELD"):
    for s in "AVT":
        if sel[(T, 25)] != 0.0:
            # 分割ごとの srcw − srcw_b0（種の平均）
            dv = []
            for f in range(5):
                xs = []
                for r in range(10):
                    z5 = np.load(U5 / f"{T}_ALL_f{f}_r{r}_p5.npz"); y = np.array([ylab[T][x] for x in z5["sample_test"].astype(str)])
                    xs.append(uar(y, p5pred(z5, s, 25, sel[(T, 25)])) - uar(y, p5pred(z5, s, 25, 0.0)))
                dv.append(np.mean(xs))
            vecs[(T, s)] = np.array(dv)
Tsb = sorted({t for t, _ in vecs}); pats = list(itertools.product([1, -1], repeat=5)); ge4 = le2 = 0
for combo in itertools.product(pats, repeat=len(Tsb)):
    sg = dict(zip(Tsb, map(np.array, combo))); npos = sum(int((v * sg[t]).mean() > 0) for (t, _), v in vecs.items()); ge4 += npos >= 4; le2 += npos <= 2
rb = rd["rule_b_chance"].iloc[0]
print(f"rule_b_chance：分母 作り直し {len(vecs)} / 表 {rb.n_cells}，4 以上の割合 {ge4/len(pats)**len(Tsb):.4f} / {rb.chance_ge4}，2 以下 {le2/len(pats)**len(Tsb):.4f} / {rb.chance_le2}，"
      f"点推定が正 {sum(int(v.mean() > 0) for v in vecs.values())} / {rb.observed_positive}")
tc = rd["teacher_check_units"]; print("teacher_check_units：一致しない行", int((~tc.match.astype(bool)).sum()), "，実験 1 の行が無い（NaN）行", int(tc.uar_pool_dai1.isna().sum()), "，差の最大", float(tc["diff"].abs().max()))
pr = rd["pairing"]; print("pairing：一致しない単位", int((~(pr.test_ok & pr.K_ok & pr.cluster_ok) | (pr.joint_uar_mismatch > 0)).sum()))
ts = rd["teacher_sharp"]; print("teacher_sharp の NaN", int(ts.isna().sum().sum()), "，fixed_point の行", end=" ")
fp = dd[dd.read.fillna("") == "fixed_point_check"]; print(len(fp), "，counts の一致率の最小", float(fp.fixed_point_counts_agree.min()), "，判定の一致率の最小", float(fp.fixed_point_pred_agree.min()),
      "，srcw の判定の一致率の最小", float(fp[fp.A == "srcw"].fixed_point_pred_agree.min()), "，joint", float(fp[fp.A == "joint"].fixed_point_pred_agree.min()))
