"""収束：勾配の最大が 1e-5 を超えた問，行き詰まりの数を段と手法ごとに数える。"""
import numpy as np, pandas as pd
from pathlib import Path
P = Path("/home/snics/MER-DA/DAI/checks/DAI-1/code_run_post")
for p in (1, 2, 3):
    D = pd.read_pickle(P / f"rows_p{p}.pkl")
    g = pd.to_numeric(D.get("grad"), errors="coerce") if "grad" in D else None
    if g is not None:
        D["_g"] = g
        t = D[D._g.notna()].groupby("method")._g.agg(n="size", over1e5=lambda x: int((x > 1e-5).sum()), over1e6=lambda x: int((x > 1e-6).sum()), mx="max")
        print(f"stage {p} per-problem grad:\n{t}")
    cv = D[D.method.astype(str).str.startswith("_conv")]
    if len(cv):
        cv = cv.assign(gm=pd.to_numeric(cv.grad_max, errors="coerce"), ns=pd.to_numeric(cv.n_stalled, errors="coerce"),
                       ni=pd.to_numeric(cv.get("n_iter"), errors="coerce") if "n_iter" in cv else np.nan)
        t2 = cv.groupby(["method"] + (["kind"] if p < 3 else [])).agg(n=("gm", "size"), gm_over1e5=("gm", lambda x: int((x > 1e-5).sum())),
                                                                          gm_max=("gm", "max"), stalled_sum=("ns", "sum"), iter_max=("ni", "max"))
        print(f"stage {p} _conv rows:\n{t2}")
    if "stalled" in D:
        s = D[D.stalled.notna()]
        print(f"stage {p} stalled True by method:", s[s.stalled.astype(str) == "True"].groupby("method").size().to_dict(),
              "rows with stalled col by method", s.groupby("method").size().to_dict())
