import sys, numpy as np
sys.argv = [sys.argv[0], "0.22", "0.42"]
sys.path.insert(0, ".")
import sim_pop as S
import sim_finite as F
imb = [0.45, 0.25, 0.2, 0.1]
w = S.make_world(imb, seed=0)
w["src_noise"] = np.random.default_rng(5).normal(size=w["mu"].shape) * 0.12
for n_unl in (5000, 20000):
    for K in (5, 25):
        rows = [F.one(w, "ci", K, n_unl, np.random.default_rng(2000 + i)) for i in range(12)]
        m = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}
        d = {k: np.array([r[k] - r["KD"] for r in rows]) for k in ("KD+A4", "BWD(Mhat)", "BWD(Mtrue)", "sLDA-R", "sLDA", "hard0.7", "bayes")}
        print(f"[ci K={K} n={n_unl}] teacher {m['teacher']*100:.1f} bayes {m['bayes']*100:.1f} KD {m['KD']*100:.1f}")
        print("   minus KD:", {k: (round(np.nanmean(v)*100, 2), round(np.nanstd(v)*100, 2)) for k, v in d.items()})
