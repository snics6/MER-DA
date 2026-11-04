"""全 npz の浮動小数の配列に NaN・Inf が無いか，int8 の判定が 0〜3 の中かを走査する（読むだけ）。"""
import numpy as np, sys
from pathlib import Path
from multiprocessing import Pool
U = Path("/home/snics/MER-DA/results/dai1/pseudo/units")
def scan(p):
    z = np.load(p); nf = 0; bad8 = 0; nfloat = 0
    for k in z.files:
        a = z[k]
        if a.dtype.kind == "f":
            nfloat += a.size; nf += int((~np.isfinite(a)).sum())
        elif a.dtype == np.int8:
            bad8 += int(((a < 0) | (a > 3)).sum())
    return p.name, nf, bad8, nfloat
if __name__ == "__main__":
    fs = sorted(U.glob("*.npz"))
    with Pool(8) as pool:
        res = pool.map(scan, fs, chunksize=4)
    print("files", len(res), "nonfinite total", sum(r[1] for r in res), "int8 out of 0..3", sum(r[2] for r in res), "float values", sum(r[3] for r in res))
    print([r for r in res if r[1] or r[2]][:5])
    src = sorted(Path("/home/snics/MER-DA/results/dai1/source").glob("*.npz"))
    print("source nonfinite", sum(scan(p)[1] for p in src))
