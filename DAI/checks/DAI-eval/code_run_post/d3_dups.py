import pandas as pd, numpy as np, json
from pathlib import Path
P=Path("/home/snics/MER-DA/DAI/checks/DAI-eval/code_run_post")
keys={1:['unit','stage','method','kind','K','Cw','Cb','C'],2:['unit','stage','method','kind','K','key','tag','round','Cw','Cb','teacher','student','rho','a','temp','tau','f','alpha','C'],
      3:['unit','stage','method','teacher','K','kind','round','student','rho','a','temp','tau','f','alpha','key','tag','Cw','Cb','C']}
for p in (1,2,3):
    D=pd.read_pickle(P/f"rows_p{p}.pkl"); k=[c for c in keys[p] if c in D.columns]
    X=D[k].copy()
    for c in k: X[c]=X[c].astype(str)
    d=int(X.duplicated().sum()); print("stage",p,"cols",k,"dup rows",d)
    if d and p!=2:
        print(D[X.duplicated(keep=False)].groupby("method").size())
# tether grid per cell
D=pd.read_pickle(P/"rows_p3.pkl"); fz=json.load(open("/home/snics/MER-DA/results/dai1/select/frozen.json"))
t=D[D.method.astype(str).str.endswith("_tether")]
print(t.groupby(["T","method"]).size().unstack(fill_value=0))
g=t.groupby(["T","unit","method","student","K"]).size()
print("tether rows per (unit,method,student,K):",g.value_counts().to_dict())
for T in ["AFEW","IEMOCAP","MOSEI","MELD"]:
    pt=fz[T]["partner"]; print(T, {k:(v if not isinstance(v,dict) else v) for k,v in list(pt.items())[:3]})
    break
