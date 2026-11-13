"""RAVDESS の形（24 塊，分割ごとのテストの塊 5,5,5,5,4，塊ごとに 28 本，種 2）で，dai1_stats.effect_stats の 95% 区間が真の差 0 を含む割合．
塊の重みの引き直し（g_eff=24 の道）と，塊を除く jackknife（g_eff=19 を渡した道）を比べる．人工の数だけ．"""
import sys, numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1_stats as ST
rng0 = np.random.default_rng(7)
sizes = [5, 5, 5, 5, 4]
per = np.array([4, 8, 8, 8])
def make(rng, tau):
    A, B = {}, {}
    g = 0
    for f, n in enumerate(sizes):
        names = np.array([f"s{g+i:02d}" for i in range(n)]); g += n
        dg = rng.normal(0, tau, (n, 4))            # 塊 × 感情ごとの差（真の平均は 0）
        base = rng.uniform(0.4, 0.8, (n, 4))
        for r in (0, 1):
            pa = np.clip(base + dg / 2 + rng.normal(0, 0.01, (n, 4)), 0.02, 0.98)
            pb = np.clip(base - dg / 2 + rng.normal(0, 0.01, (n, 4)), 0.02, 0.98)
            tot = np.tile(per, (n, 1)).astype(float)
            A[(f, r)] = (names, rng.binomial(per, pa).astype(float), tot)
            B[(f, r)] = (names, rng.binomial(per, pb).astype(float), tot)
    return A, B
for tau in (0.05, 0.15):
    res = {24: [], 19: []}
    se = {24: [], 19: []}
    for rep in range(300):
        A, B = make(rng0, tau)
        for ge in (24, 19):
            e = ST.effect_stats(A, B, ge, np.random.default_rng(rep))
            res[ge].append(e["lo"] <= 0 <= e["hi"]); se[ge].append(e["se"])
    print(f"tau={tau}：含む割合 引き直し {np.mean(res[24]):.3f}，jackknife {np.mean(res[19]):.3f}．se の中央値の比（引き直し÷jackknife）{np.median(np.array(se[24]) / np.array(se[19])):.3f}")
