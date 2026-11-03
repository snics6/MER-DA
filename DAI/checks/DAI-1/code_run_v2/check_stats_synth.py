"""dai1_stats.effect_stats を人工の判定で確かめる（実データの UAR は使わない）。"""
import numpy as np
import dai1_stats as ST
rng = np.random.default_rng(1)
def make(nf, ncl_per_fold, n_per_cl, R, acc_a, acc_b):
    A, B = {}, {}
    for f in range(nf):
        cl = np.repeat([f"f{f}c{j}" for j in range(ncl_per_fold)], n_per_cl)
        y = rng.integers(0, 4, len(cl))
        for r in range(R):
            pa = np.where(rng.random(len(y)) < acc_a[f], y, rng.integers(0, 4, len(y)))
            pb = np.where(rng.random(len(y)) < acc_b, y, rng.integers(0, 4, len(y)))
            A[(f, r)] = ST.counts(y, pa, cl); B[(f, r)] = ST.counts(y, pb, cl)
    return A, B
# 1) 同じ判定どうし：差 0，区間の幅 0
A, _ = make(5, 3, 60, 2, [0.5] * 5, 0.5)
e = ST.effect_stats(A, A, 5.0, rng); print("同じ判定 jk:", e["effect"], e["se"], e["lo"], e["hi"])
e = ST.effect_stats(A, A, 25.0, rng); print("同じ判定 boot:", e["effect"], e["se"], e["lo"], e["hi"], e["p"])
# 2) IEMOCAP 型（分割ごとに塊 1 つ）：V_jk が分割の平均の差の分散 / 5 に一致するか
A, B = make(5, 1, 200, 3, [0.4, 0.5, 0.6, 0.7, 0.55], 0.4)
e = ST.effect_stats(A, B, 4.9, rng)
dbar = np.array([np.mean([ST.uar_full(A[(f, r)]) - ST.uar_full(B[(f, r)]) for r in range(3)]) for f in range(5)])
print("IEMOCAP 型 V_jk", round(e["v_jk"], 6), "var(d̄_f)/5", round(np.var(dbar, ddof=1) / 5, 6), "df", e["df"], "θ", round(e["effect"], 6), "mean d̄", round(dbar.mean(), 6))
# 3) 塊の多い型：jackknife と 2 段の引き直しの標準誤差が同じ桁か
A, B = make(5, 40, 25, 2, [0.55] * 5, 0.45)
ej = ST.effect_stats(A, B, 15.0, rng); eb = ST.effect_stats(A, B, 25.0, rng)
print("多い型 jk se", round(ej["se"], 3), "boot se", round(eb["se"], 3), "v_seed", round(ej["v_seed"], 4), "boot p", eb["p"], "jk p", ej["p"])
# 4) Holm
print("holm", ST.holm(np.array([0.01, 0.04, 0.03, 0.2])))
