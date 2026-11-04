import json, numpy as np, pandas as pd
P="/home/snics/MER-DA/DAI/checks/DAI-eval/code_run_post/"
p1=pd.read_pickle(P+"rows_p1.pkl"); p3=pd.read_pickle(P+"rows_p3.pkl"); p3=p3[p3.stage==3]
fz=json.load(open("/home/snics/MER-DA/DAI/frozen/dai1_frozen.json")); fam=pd.read_csv("/home/snics/MER-DA/results/dai1/stats/families.csv")
def pick(df, **kw):
    m=np.ones(len(df),bool)
    for k,v in kw.items():
        col=df[k]
        m&= (col.astype(str)==v).to_numpy() if isinstance(v,str) else np.isclose(pd.to_numeric(col,errors="coerce").astype(float),float(v))
    return df[m]
def theta(T,a,b):
    # a,b : frames with columns fold, seed, uar (per unit)
    m=a.merge(b,on=["fold","seed"],suffixes=("_a","_b")); assert len(m)==len(a)==len(b), (len(m),len(a),len(b))
    m["d"]=m.uar_a-m.uar_b
    return m.groupby("fold").d.mean().mean()
bad=[]; n=0; mx=0
for T in ["AFEW","IEMOCAP","MOSEI","MELD"]:
    F=fz[T]; r1=p1[p1["T"]==T]; r3=p3[p3["T"]==T]
    for _,r in fam[fam["T"]==T].iterrows():
        s=r.student; K=int(r.K); main=F["main"][str(K)]
        def J(meth,ver,params,**extra): 
            return pick(pick(r3[(r3.method==meth)&(r3.student==s)&(r3.K==K)&(r3.teacher.astype(str)==ver)],**params),**extra)
        sj=F["select"][f"{s}|{K}|joint|{main}"]; pj={k:v for k,v in sj.items() if k!="pseudo_uar"}
        prop=J("joint",main,pj)
        if r.family==1:
            if K==0: base=r1[(r1.method=="start")&(r1.kind==s)]
            else:
                cw,cb=F["tether"][f"{s}|{K}"]; base=pick(r1[(r1.method=="none")&(r1.kind==s)&(r1.K==K)],Cw=cw,Cb=cb)
            th=theta(T,prop,base)
        elif r.family==2:
            sr=F["select"][f"{s}|{K}|rolesep|{main}"]; pr={k:v for k,v in sr.items() if k!="pseudo_uar"}
            th=theta(T,J("rolesep",main,pr),prop)
        elif r.family==3:
            pt=F["partner"][f"{s}|{K}"]; m=pt["method"]; v=pt["version"]
            if m in("cotrain","coreg"): v={"cotrain":"ct","coreg":"cr"}[m]
            sp=F["select"][f"{s}|{K}|{m}|{v}"]; pp={k:x for k,x in sp.items() if k!="pseudo_uar"}
            th=theta(T,prop,J(m,v,pp))
        else:
            th=theta(T,prop,J("joint","one",dict(rho=1.0,a=1)))
        n+=1; d=abs(th-r.effect); mx=max(mx,d)
        if d>1e-6: bad.append((T,r.family,s,K,th,r.effect))
print("cells recomputed",n,"max abs diff in theta",mx,"bad",len(bad),bad[:5])
