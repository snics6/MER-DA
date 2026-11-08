import numpy as np
from ncm_sim import run
for sep in (0.7, 1.0, 1.5):
    for nuis, nsd in ((5, 3.0), (20, 3.0), (100, 2.0), (100, 3.0)):
        for K in (5, 25):
            R = [run(300, 1500, 1500, sep, nuis, nsd, K, s) for s in range(6)]
            m = {k: np.mean([r[k] for r in R]) for k in R[0]}
            flag = "★" if (m['LDA_T'] > m['LDA_K'] and m['LDA_T'] < m['NCM']) else ""
            print(f"sep={sep} nuis={nuis}×{nsd} K={K}: LDA_K {m['LDA_K']:.1f} LDA_T {m['LDA_T']:.1f} NCM {m['NCM']:.1f} LDA_SO {m['LDA_SO']:.1f} {flag}")
