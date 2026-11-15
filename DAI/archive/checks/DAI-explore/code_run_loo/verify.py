import sys, json, numpy as np, pandas as pd, torch, time
sys.path.insert(0,"/home/snics/MER-DA/DAI/exp")
import dai1, dai_data as DD, dai_solver as S
ROOT=dai1.DD.ROOT
T=sys.argv[1]; f=0; r=0
fz=json.load(open(ROOT/"DAI/frozen/dai1_frozen.json"))[T]
srcs=[c for c in DD.CORPORA if c!=T]
u=dai1.Unit(ROOT/"results/dai1",T,T,srcs,f,r)
base=ROOT/f"results/dai1/eval/units/{T}_ALL_f{f}_r{r}"
p1=np.load(f"{base}_p1.npz"); p2=np.load(f"{base}_p2.npz")
print("pool n",u.n,"d",{m:u.Xp[m].shape[1] for m in "AVT"})
for K in (5,25):
    kp=u.kpos[K]; nK=len(kp); yk=u.ypool[kp]
    cl=u.cp.df["cluster"].to_numpy()[u.pool][kp]
    print(f"K={K} nK={nK} distinct clusters among K picks={len(set(cl))}; max per cluster={pd.Series(cl).value_counts().max()}")
    for m in "AVT":
        cw,cb=fz["tether"][f"{m}|{K}"]; W0,b0=u.start[m]
        # (1) reproduce stored none W,b
        res=S.fit(u.Xp[m],u.gK(K)[None],W0,b0,Cw=cw,Cb=cb)
        items=[str(x) for x in p1[f"p1_{m}_items"]]; j=items.index(f"{K}|{cw}|{cb}")
        dW=float(np.abs(res["W"][0].cpu().numpy()-p1[f"p1_{m}_W"][j]).max()); db=float(np.abs(res["b"][0].cpu().numpy()-p1[f"p1_{m}_b"][j]).max())
        # exact none test uar vs half-argmax stored
        Z=S.logits(u.Xt[m],res["W"][0],res["b"][0]); pe=Z.argmax(1).cpu().numpy()
        ph=p2[f"p2_K{K}_none_{m}_test"].astype(np.float32).argmax(1)
        print(f" {m}: stored-none reproduction dW={dW:.1e} db={db:.1e}; exact-vs-half argmax disagree n={int((pe!=ph).sum())}/{len(pe)}")
        # (2) batch vs single LOO
        torch.cuda.reset_peak_memory_stats(); t0=time.time()
        Gs=torch.stack([S.balanced_onehot(u.ypool,np.delete(kp,i),u.n) for i in range(nK)])
        rb=S.fit(u.Xp[m],Gs,W0,b0,Cw=cw,Cb=cb)
        torch.cuda.synchronize(); print(f"   batch fit {time.time()-t0:.1f}s, peak GPU {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, n_iter {rb['n_iter']} grad {rb['grad_max']:.1e} stalled {rb['stalled']} ok {rb['ok']}")
        Zb=S.logits(u.Xp[m][kp],rb["W"],rb["b"]); held=Zb[torch.arange(nK),torch.arange(nK)].argmax(1).cpu().numpy()
        for i in (0,nK//2,nK-1):
            G1=S.balanced_onehot(u.ypool,np.delete(kp,i),u.n)
            assert float(G1[kp[i]].sum())==0 and abs(float(G1.sum())-(nK-1))<1e-9
            r1=S.fit(u.Xp[m],G1[None],W0,b0,Cw=cw,Cb=cb)
            z1=S.logits(u.Xp[m][kp[i]][None],r1["W"][0],r1["b"][0])
            dz=float((z1[0]-Zb[i,i]).abs().max()); 
            assert int(z1.argmax(1)[0])==held[i], "held mismatch"
            print(f"   i={i}: single vs batch held-logit diff {dz:.1e}")
        # (3) ties
        print(f"   loo acc {np.mean(held==yk):.3f}")
