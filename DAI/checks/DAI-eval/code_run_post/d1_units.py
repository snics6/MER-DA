"""評価の回の単位の欠け・重複・行の数・コミット・dirty・sha・計算機・有限・収束を数える（読むだけ）。"""
import json, sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA"); P = ROOT / "DAI/checks/DAI-eval/code_run_post"
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]; R = 10
# ---- dai1
U1 = ROOT / "results/dai1/eval/units"
files = sorted(U1.glob("*"))
exp = {(T, f, r, p, e) for T in C for f in range(5) for r in range(R) for p in (1, 2, 3) for e in ("npz", "csv")}
have = set()
for fp in files:
    stem, e = fp.name.rsplit(".", 1); T, a, f, r, p = stem.split("_"); assert a == "ALL"
    have.add((T, int(f[1:]), int(r[1:]), int(p[1:]), e))
print("dai1 files", len(files), "missing", len(exp - have), "extra", len(have - exp))
allrows = {}
meta = []
for T in C:
    for f in range(5):
        for r in range(R):
            for p in (1, 2, 3):
                b = U1 / f"{T}_ALL_f{f}_r{r}_p{p}"
                d = pd.read_csv(f"{b}.csv", low_memory=False); d["unit"] = f"{T}_f{f}_r{r}"; d["stage_file"] = p
                allrows.setdefault(p, []).append(d)
                z = np.load(f"{b}.npz")
                meta.append(dict(T=T, f=f, r=r, p=p, commit=str(z["commit"]), dirty=bool(z["dirty"]) if "dirty" in z else None,
                                 frozen=str(z["frozen_sha256"]) if "frozen_sha256" in z else None,
                                 gate=str(z["gate_sha256"]) if "gate_sha256" in z else None, nkeys=len(z.files)))
M = pd.DataFrame(meta)
print("npz commit", M.commit.value_counts().to_dict(), "dirty", M.dirty.value_counts().to_dict())
print("npz frozen", M.frozen.value_counts().to_dict()); print("npz gate", M.gate.value_counts().to_dict())
print("nkeys by T,p", M.groupby(["T", "p"]).nkeys.agg(["min", "max"]).to_dict())
for p in (1, 2, 3):
    D = pd.concat(allrows[p], ignore_index=True)
    D.to_pickle(P / f"rows_p{p}.pkl")
    n = D.groupby("unit").size()
    print(f"stage {p}: units {n.size}, rows/unit min {n.min()} max {n.max()}, total {len(D)}; unique rows/unit by T:",
          D.groupby("T").size().div(50).to_dict())
    print("  commit", D.commit.astype(str).value_counts().to_dict(), "dirty", D.dirty.astype(str).value_counts().to_dict())
    print("  gpu", D.gpu.astype(str).value_counts().to_dict(), " frozen", D.frozen_sha256.nunique(), " gate", D.gate_sha256.nunique())
    for col in ("finite",):
        print("  finite:", D[col].astype(str).value_counts(dropna=False).to_dict())
    if "stalled" in D: print("  stalled:", D.stalled.astype(str).value_counts(dropna=False).to_dict())
    for col in ("uar", "uar_pool", "grad", "grad_max", "S"):
        if col in D:
            v = pd.to_numeric(D[col], errors="coerce")
            print(f"  {col}: notnull {int(v.notna().sum())} nonfinite {int((~np.isfinite(v[v.notna()])).sum())} max {v.max():.3g} min {v.min():.3g}", end="")
            if col in ("uar", "uar_pool"): print(f" out-of-range {int(((v < 0) | (v > 100)).sum())}", end="")
            print()
# per-unit rows identical across units of a T (grid)
for p in (1, 2, 3):
    D = pd.read_pickle(P / f"rows_p{p}.pkl")
    cnt = D.groupby(["T", "unit"]).size().groupby("T").agg(["min", "max", "nunique"])
    print("stage", p, "rows/unit by T\n", cnt)
