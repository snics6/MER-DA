"""measure の rows_*.csv.gz：反復ごとの行の数がそろうか，重複，NaN・Inf，UAR が 0〜1 の外の数（読むだけ）。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
d = Path(sys.argv[1])
for f in sorted(d.glob("rows_*.csv.gz")):
    x = pd.read_csv(f, low_memory=False)
    num = x.select_dtypes("number")
    key = [c for c in ("wave", "target", "rep", "student", "method", "param", "meff", "K") if c in x.columns]
    per = x.groupby(["target", "rep"]).size().groupby("target").nunique()
    u = x["uar"] if "uar" in x else None
    print(f.name, x.shape, "dup", int(x.duplicated(key).sum()), "reps/target", x.groupby("target").rep.nunique().unique().tolist(),
          "targets whose row count varies by rep", int((per > 1).sum()),
          "inf", int(np.isinf(num.to_numpy(dtype=float)).sum()),
          "uar nan", int(u.isna().sum()) if u is not None else None, "uar out of [0,1]", int(((u < 0) | (u > 1)).sum()) if u is not None else None)
    nan = num.isna().sum()
    print("   NaN cols", {c: int(v) for c, v in nan.items() if v})
    if f.name.startswith("rows_audio") or len(sys.argv) > 2:
        print("   cols", list(x.columns))
