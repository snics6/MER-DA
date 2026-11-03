"""effect_stats の確かめ（作り物の判定．本物の塊とラベルの上で）。"""
import numpy as np, pandas as pd
import dai1_stats as ST
from pathlib import Path
rng = np.random.default_rng(0)
for c in ("AFEW", "IEMOCAP", "MELD", "MOSEI"):
    g_eff, sp = ST.g_eff_of(c)
    y_all = sp.label_4.to_numpy(int)
    def cell(acc, R=2, shift=0.0):
        out = {}
        for f in range(5):
            idx = np.flatnonzero(sp.fold.to_numpy() == f)
            y, cl = y_all[idx], sp.cluster.to_numpy(str)[idx]
            for r in range(R):
                rr = np.random.default_rng(1000 * f + r)
                hit = rr.random(len(y)) < acc + shift * (r - 0.5)
                pred = np.where(hit, y, (y + 1) % 4)
                out[(f, r)] = ST.counts(y, pred, cl)
        return out
    A = cell(0.6, shift=0.1); B = cell(0.5)
    e = ST.effect_stats(A, A, g_eff, rng)
    e2 = ST.effect_stats(A, B, g_eff, rng)
    print(c, f"G_eff {g_eff:.2f}", "同じ 2 行：", dict(p=e["p"], identical=e["identical"], se=e["se"]),
          "違う 2 行：", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e2.items() if k in ("method", "se", "v_jk", "v_seed", "v_seed_all", "folds_dropped", "df", "p", "identical")})
