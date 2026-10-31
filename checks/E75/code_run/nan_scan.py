import sys, glob, numpy as np, pandas as pd
for f in sorted(glob.glob(sys.argv[1] + "/*.csv")):
    d = pd.read_csv(f, comment="#")
    num = d.select_dtypes("number")
    nan = {c: int(num[c].isna().sum()) for c in num if num[c].isna().any()}
    inf = {c: int(np.isinf(num[c]).sum()) for c in num if np.isinf(num[c]).any()}
    const = [c for c in d.columns if d[c].nunique(dropna=False) == 1 and len(d) > 1]
    print(f.split("/")[-1], len(d), "NaN:", nan, "Inf:", inf, "const:", const)
