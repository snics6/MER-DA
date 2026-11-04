import sys, json
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd, torch
import dai1, dai4 as D4, dai4_stats as D, dai_data as DD, dai_solver as S, dai1_stats as ST
CR = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run_v2")
p4 = D4.P4(CR / "out_e/eval/units/AFEW_ALL_f0_r0")
y_of, _ = ST.label_maps("AFEW")
ids = p4.z["sample_pool"].astype(str); y = y_of(ids); n = len(ids)
for K in (0, 5, 25):
    nk = ~np.isin(ids, p4.z[f"sample_K{K}"].astype(str)) if K else np.ones(n, bool)
    Q = torch.as_tensor(p4.z[f"p4_K{K}_Q"].astype(np.float64), device=S.DEV); yy = torch.as_tensor(y, device=S.DEV)
    mk = p4.z[f"p4_K{K}_AVT_mark"].astype(bool); m = torch.as_tensor(np.where(mk, 1.0, 0.25), device=S.DEV)
    P = S.soft_rows(Q * m[:, None], dai1.mval(1.0, n), balance=True)
    nkt = torch.as_tensor(nk, device=S.DEV)
    new = float(np.mean([(P[nkt & (yy == k), k].sum() / P[nkt, k].sum().clamp(min=1e-30)).item() for k in range(4)]))
    Pn = P.cpu().numpy()
    ref = float(np.mean([Pn[nk & (y == k), k].sum() / Pn[nk, k].sum() for k in range(4)]))
    old = float(np.mean([(P[yy == k, k].sum() / P[:, k].sum().clamp(min=1e-30)).item() for k in range(4)]))
    print(K, "n", n, "除く本数", int((~nk).sum()), "新(K除外)", round(new, 6), "numpyで作り直し", round(ref, 6), "旧(全pool)", round(old, 6))
