"""1 つの置き場の段 3 の csv の行の数・NaN・行き詰まり・勾配，保存の鍵，判定からの UAR の作り直し（stats.collect の照合と，項目ごとの照合）を数える（UAR は出さない）。"""
import sys, glob, json
import numpy as np, pandas as pd
import dai1, dai1_stats as ST, dai_data as DD
out = sys.argv[1]
for f in sorted(glob.glob(f"{out}/pseudo/units/*_p3.csv")):
    d = pd.read_csv(f, low_memory=False)
    base = f[:-7]
    z1, z3 = np.load(base + "_p1.npz"), np.load(base + "_p3.npz")
    c = d.c.iloc[0]
    cp = dai1.Corpus.get(c)
    ymap = dict(zip(cp.df.sample_id.astype(str), cp.y))
    y = np.array([ymap[s] for s in z1["sample_test"].astype(str)])
    real = d[~d.method.astype(str).str.startswith("_")]
    cnt = real.groupby(["student", "method"]).size().unstack(fill_value=0)
    conv = d[d.method.astype(str).str.startswith("_conv")]
    gcol = pd.concat([real.grad.dropna(), conv.grad_max.dropna()])
    print(base.split("/")[-1], "行", len(d), "手法の行", len(real), "uar の NaN", int(real.uar.isna().sum()),
          "grad>1e-5", int((gcol > 1e-5).sum()), "grad の最大", float(gcol.max()),
          "stalled 真", int((real.stalled.astype(str) == "True").sum()), "n_stalled 計", float(conv.n_stalled.fillna(0).sum()),
          "finite 偽", int((d.finite.astype(str) == "False").sum()))
    print(cnt.to_string())
    print("coreg の収束", d[d.method == "_conv_coreg"][["K", "rho", "grad_max", "n_iter", "finite"]].to_string(index=False))
    # 項目ごとの照合：p3_<s>_K<K> の items の順と，csv の同じ問の行の順（solve_items が書いた順）
    bad, tot = 0, 0
    for s in DD.STUDENTS:
        for K in dai1.KS_ALL:
            for key in (f"p3_{s}_K{K}", f"p3t_{s}_K{K}", f"p3z_{s}_K{K}"):
                if f"{key}_items" not in z3.files:
                    continue
                items = [json.loads(x) for x in z3[f"{key}_items"]]
                mine = np.array([100 * dai1.S.uar(y, p) for p in z3[f"{key}_pred"]])
                for it, v in zip(items, mine):
                    m = (d.stage == 3) & (d.student == s) & (d.K == K) & (d.method == it["method"]) & (d.teacher.astype(str) == str(it["teacher"]))
                    for k, val in it.items():
                        if k in ("method", "teacher", "tag", "n_kept"):
                            continue
                        m &= np.isclose(pd.to_numeric(d[k], errors="coerce"), float(val))
                    r = d[m]
                    tot += 1
                    bad += int(len(r) != 1 or abs(float(r.uar.iloc[0]) - v) > 1e-6)
    # coreg・共訓練（単一）・役割を分けた版
    for K in dai1.KS_ALL:
        for s in DD.STUDENTS:
            CR = getattr(dai1, "COREG_RHO", dai1.RHO_GRID)
            assert z3[f"p3g_K{K}_{s}_pred"].shape[0] == len(CR)
            for j, rho in enumerate(CR):
                v = 100 * dai1.S.uar(y, z3[f"p3g_K{K}_{s}_pred"][j])
                r = d[(d.method == "coreg") & (d.student == s) & (d.K == K) & np.isclose(d.rho, rho)]
                tot += 1; bad += int(len(r) != 1 or abs(float(r.uar.iloc[0]) - v) > 1e-6)
            for j, rho in enumerate(dai1.RHO_GRID):
                if s != "AVT":
                    v = 100 * dai1.S.uar(y, z3[f"p3c_K{K}_{s}_pred"][j])
                    r = d[(d.method == "cotrain") & (d.student == s) & (d.K == K) & np.isclose(d.rho, rho)]
                    tot += 1; bad += int(len(r) != 1 or abs(float(r.uar.iloc[0]) - v) > 1e-6)
    print("項目ごとの照合 食い違い", bad, "/", tot)
    keys = sorted({k.split("_K")[0] if "_K" in k else k for k in z3.files})
    print("p3 の鍵の種類", keys, "p3 の tether・wsel", "tether" in z3.files, "wsel" in z3.files)
    print("mix の残った本数の列", "n_kept" in d.columns, int(d[d.method == "mix"].n_kept.isna().sum()))
