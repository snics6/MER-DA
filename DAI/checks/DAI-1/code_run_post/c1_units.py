"""gate-code 計算の後の点検：DAI-1 の仮のターゲットの単位の欠け・重複・行の数・コミット・有限・収束を数える（読むだけ）。"""
import glob, json, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA")
U = ROOT / "results/dai1/pseudo/units"
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
exp = {(T, c, f, r, p) for T in C for c in C if c != T for f in range(5) for r in (0, 1) for p in (1, 2, 3)}
have = set()
for fp in U.glob("*"):
    n = fp.name
    stem, ext = n.rsplit(".", 1)
    T, c, f, r, p = stem.split("_")
    have.add((T, c, int(f[1:]), int(r[1:]), int(p[1:]), ext))
for ext in ("npz", "csv"):
    h = {k[:5] for k in have if k[5] == ext}
    print(ext, "missing", len(exp - h), "extra", len(h - exp))
rows = {1: [], 2: [], 3: []}
npzmeta = []
for (T, c, f, r, p) in sorted(exp):
    base = U / f"{T}_{c}_f{f}_r{r}_p{p}"
    d = pd.read_csv(f"{base}.csv", low_memory=False)
    d["unit"] = f"{T}_{c}_f{f}_r{r}"
    rows[p].append(d)
    z = np.load(f"{base}.npz")
    npzmeta.append(dict(T=T, c=c, f=f, r=r, p=p, commit=str(z["commit"]), dirty=bool(z["dirty"]),
                        tether=("tether" in z), wsel=("wsel" in z), nkeys=len(z.files)))
M = pd.DataFrame(npzmeta)
print("npz commits", M.commit.value_counts().to_dict(), "dirty", M.dirty.value_counts().to_dict())
print("npz tether present by stage", M.groupby("p").tether.sum().to_dict(), "wsel", M.groupby("p").wsel.sum().to_dict())
print("npz nkeys by stage/T", M.groupby(["p"]).nkeys.agg(["min", "max"]).to_dict())
for p in (1, 2, 3):
    D = pd.concat(rows[p], ignore_index=True)
    D.to_pickle(ROOT / f"DAI/checks/DAI-1/code_run_post/rows_p{p}.pkl")
    n = D.groupby("unit").size()
    print(f"stage {p}: units {n.size}, rows per unit min {n.min()} max {n.max()}, total {len(D)}")
    print("  commit", D.commit.astype(str).value_counts().to_dict(), "dirty", D.dirty.astype(str).value_counts().to_dict(),
          "gpu", D.gpu.astype(str).value_counts().to_dict())
    if "finite" in D:
        print("  finite False:", int((D.finite.astype(str) == "False").sum()), " finite NaN rows:", int(D.finite.isna().sum()))
    if "stalled" in D:
        print("  stalled True:", int((D.stalled.astype(str) == "True").sum()))
    for col in ("uar", "uar_pool", "grad", "grad_max", "S"):
        if col in D:
            v = pd.to_numeric(D[col], errors="coerce")
            print(f"  {col}: nonfinite(non-null) {int((~np.isfinite(v[v.notna()])).sum())}", end="")
            if col in ("uar", "uar_pool"):
                print(f" out-of-range {int(((v < 0) | (v > 100)).sum())}", end="")
            print()
