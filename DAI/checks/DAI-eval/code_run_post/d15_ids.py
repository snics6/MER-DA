import numpy as np, pandas as pd
from pathlib import Path
ROOT=Path("/home/snics/MER-DA"); C=["AFEW","IEMOCAP","MOSEI","MELD"]
sp={c:pd.read_csv(ROOT/f"DAI/splits/{c}.csv",keep_default_na=False).set_index("sample_id") for c in C}
bad=[]; n=0; kdraw={}
for T in C:
    s=sp[T]
    for f in range(5):
        te_exp=set(s.index[s.fold==f]); po_exp=set(s.index[s.fold!=f])
        if T=="MELD":
            dl=set(s.loc[list(te_exp),"dialogue"]); po_exp={x for x in po_exp if s.at[x,"dialogue"] not in dl}
        for r in range(10):
            u=f"{T}_ALL_f{f}_r{r}"; z=np.load(ROOT/f"results/dai1/eval/units/{u}_p1.npz")
            te,po=set(z["sample_test"]),set(z["sample_pool"]); n+=1
            if te!=te_exp: bad.append((u,"test"))
            if po!=po_exp: bad.append((u,"pool"))
            if len(te)!=len(z["sample_test"]) or len(po)!=len(z["sample_pool"]): bad.append((u,"dup"))
            if set(s.loc[list(te),"cluster"])&set(s.loc[list(po),"cluster"]): bad.append((u,"cluster overlap"))
            prev=set(); ks={}
            for K in (1,5,10,25):
                k=z[f"sample_K{K}"]; lab=s.loc[list(k),"label_4"].value_counts(); ks[K]=tuple(k)
                if not set(k)<=po or len(set(k))!=4*K or len(lab)!=4 or not (lab==K).all(): bad.append((u,f"K{K}"))
                if not prev<=set(k): bad.append((u,f"nest{K}"))
                prev=set(k)
            kdraw[(T,f,r)]=ks
            # other kinds: same ids
            for kind,rmax in (("lin",10),("coral",10),("map",3),("xmuda",3)):
                if r>=rmax: continue
                z2=np.load(ROOT/f"results/dai2/eval/units/{u}_{kind}.npz")
                for key in ("sample_test","sample_pool","sample_K1","sample_K5","sample_K10","sample_K25"):
                    if not (z2[key]==z[key]).all(): bad.append((u,kind,key))
print("units",n,"problems",len(bad),bad[:8])
dist={}
for T in C:
    for f in range(5):
        v=[kdraw[(T,f,r)][5] for r in range(10)]; dist[(T,f)]=len(set(v))
print("distinct K=5 draws among 10 seeds per (T,fold): min",min(dist.values()),"max",max(dist.values()))
# same K draw across tests T? (not expected). test of set sizes
print({T:int((sp[T].fold>=0).sum()) for T in C})
