import sys, json, numpy as np, pandas as pd, torch, time
sys.path.insert(0,"/home/snics/MER-DA/DAI/exp")
import dai1, dai_data as DD, dai_solver as S
ROOT=DD.ROOT
def uar(y,p): return 100.0*np.mean([np.mean(p[y==k]==k) for k in range(4) if (y==k).any()])
T=sys.argv[1]; fz=json.load(open(ROOT/"DAI/frozen/dai1_frozen.json"))[T]; srcs=[c for c in DD.CORPORA if c!=T]
ties=0; tot=0
for f,r in [(0,0),(0,1),(1,2),(2,3)]:
    u=dai1.Unit(ROOT/"results/dai1",T,T,srcs,f,r)
    cl=u.cp.df["cluster"].to_numpy()[u.pool]
    for K in (5,25):
        kp=u.kpos[K]; nK=len(kp); yk=u.ypool[kp]; ck=cl[kp]; uc=pd.unique(ck)
        res_u={}; res_c={}
        for m in "AVT":
            cw,cb=fz["tether"][f"{m}|{K}"]; W0,b0=u.start[m]
            # utterance LOO
            Gs=torch.stack([S.balanced_onehot(u.ypool,np.delete(kp,i),u.n) for i in range(nK)])
            t0=time.time(); rr=S.fit(u.Xp[m],Gs,W0,b0,Cw=cw,Cb=cb); Z=S.logits(u.Xp[m][kp],rr["W"],rr["b"])
            res_u[m]=uar(yk,Z[torch.arange(nK),torch.arange(nK)].argmax(1).cpu().numpy())
            # cluster LOO
            Gc=torch.stack([S.balanced_onehot(u.ypool,kp[ck!=c],u.n) for c in uc])
            rc=S.fit(u.Xp[m],Gc,W0,b0,Cw=cw,Cb=cb); Zc=S.logits(u.Xp[m][kp],rc["W"],rc["b"])
            pred=np.zeros(nK,int)
            for p,c in enumerate(uc):
                h=np.flatnonzero(ck==c); pred[h]=Zc[p][torch.as_tensor(h,device=Zc.device)].argmax(1).cpu().numpy()
            res_c[m]=uar(yk,pred); assert rc["ok"] and rr["ok"]
        vals=list(res_u.values()); tot+=1; ties+= int(len(set(np.round(vals,6)))<3)
        print(T,f,r,K,"nclusters",len(uc),"rank-order utt-LOO:",''.join(sorted("AVT",key=lambda m:-res_u[m])),"cluster-LOO:",''.join(sorted("AVT",key=lambda m:-res_c[m])),"tie(utt)",len(set(np.round(vals,6)))<3,"gap utt-clu A/V/T", [round(res_u[m]-res_c[m],1) for m in "AVT"])
print("ties",ties,"/",tot)
