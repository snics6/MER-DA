import pandas as pd, numpy as np, json
from pathlib import Path
ROOT=Path("/home/snics/MER-DA"); U=ROOT/"results/dai2/eval/units"; P=ROOT/"DAI/checks/DAI-eval/code_run_post"
C=["AFEW","IEMOCAP","MOSEI","MELD"]; R=10
res={}
for kind,nr in (("lin",10),("coral",10),("map",3),("xmuda",3)):
    L=[]; meta=[]
    for T in C:
        for f in range(5):
            for r in range(nr):
                b=U/f"{T}_ALL_f{f}_r{r}_{kind}"
                d=pd.read_csv(f"{b}.csv",low_memory=False); d["unit"]=f"{T}_f{f}_r{r}"; L.append(d)
                z=np.load(f"{b}.npz")
                meta.append(dict(commit=str(z["commit"]) if "commit" in z else None,dirty=bool(z["dirty"]) if "dirty" in z else None,
                  frozen=str(z["frozen_sha256"]) if "frozen_sha256" in z else None,gate=str(z["gate_sha256"]) if "gate_sha256" in z else None))
    D=pd.concat(L,ignore_index=True); D.to_pickle(P/f"rows2_{kind}.pkl"); M=pd.DataFrame(meta)
    n=D.groupby("unit").size()
    print(f"== {kind}: units {n.size} (expected {4*5*nr}) rows/unit min {n.min()} max {n.max()} total {len(D)}")
    print("  npz commit",M.commit.value_counts().to_dict(),"dirty",M.dirty.value_counts().to_dict(),"frozen",M.frozen.nunique(),"gate",M.gate.nunique())
    print("  csv commit",D.commit.astype(str).value_counts().to_dict(),"dirty",D.dirty.astype(str).value_counts().to_dict(),"gpu",D.gpu.astype(str).value_counts().to_dict(),"frozen",D.frozen_sha256.nunique(),"gate",D.gate_sha256.nunique())
    print("  finite",D.finite.astype(str).value_counts(dropna=False).to_dict() if "finite" in D else None)
    if "stalled" in D: print("  stalled",D.stalled.astype(str).value_counts(dropna=False).to_dict())
    for col in ("uar","grad","grad_max","S","anorm","cdiff_after","dloss_after","dloss_before","cdiff_before","dis_ctl","dis_rep","kl_last","ent_last","W_dist","cg_resid"):
        if col in D:
            v=pd.to_numeric(D[col],errors="coerce"); nn=v.notna().sum()
            print(f"  {col}: notnull {nn} nonfinite {int((~np.isfinite(v[v.notna()])).sum())} min {v.min():.3g} max {v.max():.3g}"+(f" out-of-range {int(((v<0)|(v>100)).sum())}" if col=="uar" else ""))
    print("  methods:",D.groupby("method").size().to_dict())
    res[kind]=D
