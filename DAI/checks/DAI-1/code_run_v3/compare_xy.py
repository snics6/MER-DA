"""2 つの置き場の同じ単位の npz（全部の配列）と csv（uar 以外の列も）が一致するかを数える（UAR の値は出さない）。"""
import sys, glob, os
import numpy as np, pandas as pd
A, B = sys.argv[1], sys.argv[2]
skip = {"commit", "dirty", "frozen_sha256", "gate_sha256"}
fa = sorted(os.path.basename(f) for f in glob.glob(f"{A}/pseudo/units/*.npz"))
fb = sorted(os.path.basename(f) for f in glob.glob(f"{B}/pseudo/units/*.npz"))
print("npz の数", len(fa), len(fb), "片方だけ", sorted(set(fa) ^ set(fb)))
for f in sorted(set(fa) & set(fb)):
    za, zb = np.load(f"{A}/pseudo/units/{f}"), np.load(f"{B}/pseudo/units/{f}")
    ka, kb = set(za.files), set(zb.files)
    diff, nd = [], 0
    for k in sorted(ka & kb):
        x, y = za[k], zb[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            diff.append((k, "shape")); continue
        if x.dtype.kind in "fc":
            d = float(np.nanmax(np.abs(x.astype(float) - y.astype(float)))) if x.size else 0.0
            if d > 0: diff.append((k, d))
        elif not np.array_equal(x, y):
            diff.append((k, "neq"))
    ca, cb = pd.read_csv(f"{A}/pseudo/units/{f[:-4]}.csv", low_memory=False), pd.read_csv(f"{B}/pseudo/units/{f[:-4]}.csv", low_memory=False)
    cols = [c for c in ca.columns if c not in ("sec",)]
    same_rows = len(ca) == len(cb) and list(ca.columns) == list(cb.columns)
    neq = 0
    if same_rows:
        for c in cols:
            x, y = ca[c], cb[c]
            if x.dtype.kind in "fc" and y.dtype.kind in "fc":
                neq += int((~(np.isclose(x, y, rtol=0, atol=0) | (x.isna() & y.isna()))).sum())
            else:
                neq += int((x.astype(str) != y.astype(str)).sum())
    print(f, "鍵", len(ka), len(kb), "鍵の差", sorted(ka ^ kb)[:5], "配列の違い", len(diff), diff[:5], "csv 行", len(ca), len(cb), "セルの違い", neq if same_rows else "列か行が違う")
