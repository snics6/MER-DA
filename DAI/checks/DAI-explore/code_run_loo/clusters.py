import sys, numpy as np, pandas as pd
sys.path.insert(0,"/home/snics/MER-DA/DAI/exp")
import dai1, dai_data as DD
for T in sys.argv[1:]:
    srcs=[c for c in DD.CORPORA if c!=T]; out=[]
    for f,r in [(0,0),(2,3),(4,7)]:
        u=dai1.Unit(dai1.DD.ROOT/"results/dai1",T,T,srcs,f,r)
        cl=u.cp.df["cluster"].to_numpy()[u.pool]
        s=[]
        for K in (5,25):
            c=cl[u.kpos[K]]; vc=pd.Series(c).value_counts()
            s.append((K,len(vc),int(vc.max()), int((vc.reindex(c).values>1).sum())))
        out.append((f,r,u.n,len(set(cl)),s))
    print(T,out)
