"""measure の段の複製（boot・bootcl・jk・jkcl）の形と NaN・Inf の数（読むだけ）。値は出さない。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
d = Path(sys.argv[1])
keys = {f.stem[5:]: pd.read_csv(f, dtype=str) for f in d.glob("keys_*.csv")}
rows = []
for f in sorted(d.glob("*.npz")):
    kind, rest = f.stem.split("_", 1)
    run = next(r for r in sorted(keys, key=len, reverse=True) if rest.startswith(r + "_"))
    T = rest[len(run) + 1:]
    nk = int(((keys[run].target == T)).sum())
    z = np.load(f)
    for q in z.files:
        a = z[q]
        if a.dtype.kind not in 'fc':
            continue
        allnan_cols = int(np.isnan(a).all(0).sum()) if a.ndim == 2 else -1
        rows.append({"file": f.name, "kind": kind, "run": run, "target": T, "q": q, "shape": a.shape, "n_keys_csv": nk,
                     "ncol_match": a.shape[-1] == nk, "nan": int(np.isnan(a).sum()), "inf": int(np.isinf(a).sum()),
                     "allnan_cols": allnan_cols, "nan_rows": int(np.isnan(a).all(1).sum()) if a.ndim == 2 else -1})
R = pd.DataFrame(rows)
print(R.groupby("kind").agg(n=("file", "nunique"), ncol_mismatch=("ncol_match", lambda s: int((~s).sum())), nan=("nan", "sum"), inf=("inf", "sum"),
                            allnan_cols=("allnan_cols", "sum"), nan_rows_max=("nan_rows", "max")).to_string())
x = R[(R.allnan_cols > 0) | (R.inf > 0)]
print(x[["file", "q", "shape", "allnan_cols", "nan_rows", "inf"]].to_string() if len(x) else "全部 NaN の鍵は無い")
y = R[~R.ncol_match]
print(y[["file", "q", "shape", "n_keys_csv"]].drop_duplicates("file").to_string() if len(y) else "列の数は keys と一致")
R.to_csv(Path(__file__).parent / f"boot_scan_{d.parent.name}.csv", index=False)
