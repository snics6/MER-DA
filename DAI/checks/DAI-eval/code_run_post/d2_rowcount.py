import pandas as pd, numpy as np
from pathlib import Path
P=Path("/home/snics/MER-DA/DAI/checks/DAI-eval/code_run_post"); U=Path("/home/snics/MER-DA/results/dai1/pseudo/units")
for p in (1,2,3):
    E=pd.read_pickle(P/f"rows_p{p}.pkl"); E=E[E.unit=="MOSEI_f0_r0"]
    Pp=pd.read_csv(U/f"MOSEI_AFEW_f0_r0_p{p}.csv",low_memory=False)
    k=["method"]
    ge=E.groupby(k,dropna=False).size(); gp=Pp.groupby(k,dropna=False).size()
    t=pd.concat([ge.rename("eval"),gp.rename("pseudo")],axis=1)
    print("stage",p,"eval rows",len(E),"pseudo rows",len(Pp)); print(t.to_string())
