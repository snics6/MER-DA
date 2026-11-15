import pandas as pd, re
D='/home/snics/MER-DA/checks/E79/results_draft_p4.md'
c=pd.read_csv('/home/snics/MER-DA/results/e79_p4/judge/combos.csv',low_memory=False)
st={'A':'audio_wavlm','V':'video','T':'text_roberta','AVT 融合':'fuse-audio-video-text','AV 融合':'fuse-audio-video'}
lines=open(D).read().splitlines()[99:171]
bad=0
for L in lines:
    f=[x.strip() for x in L.strip('|').split('|')]
    fam="P4'" if f[0]=='P4′' else 'P4'
    cc=c[(c.tag=='main')&(c.family==fam)&(c.metric=='dir')&(c.comparison=='none-'+f[1])&(c.student==st[f[2]])&(c.K==int(f[3]))]
    r=cc.iloc[0]
    got=[int(r.n_pos),int(r.n_same),int(r.n_neg),int(r.n_undet),int(r.n_inactive),int(r.n_missing),int(r.n_model_dependent)]
    exp=[int(x) for x in f[5:12]]
    nomd=f[12].split('／'); gotn=[int(r.n_pos_nomd),int(r.n_same_nomd),int(r.n_neg_nomd)]
    hw=f[13]; goth='—' if pd.isna(r.ci_halfwidth_median) else f'{r.ci_halfwidth_median:.2f}'
    seen='見た後' if str(r.seen_before_prediction)=='True' else '見る前'
    if got!=exp or [int(x) for x in nomd]!=gotn or hw!=goth or seen!=f[15]:
        bad+=1; print('MISMATCH',f[:4],got,exp,gotn,nomd,goth,hw,seen,f[15])
    # ctl
    if f[16] not in ('—',):
        cf="P4'_ctl" if fam=="P4'" else 'P4_ctl'
        meth={'coral':'coralctl-coral','dann':'mapctl-dann','mme':'mapctl-mme'}[f[1]]
        k=c[(c.tag=='main')&(c.family==cf)&(c.metric=='dir')&(c.comparison==meth)&(c.student==st[f[2]])&(c.K==int(f[3]))].iloc[0]
        g=f'{int(k.n_pos)}/{int(k.n_same)}/{int(k.n_neg)}/{int(k.n_undet)}'
        if g!=f[16]: bad+=1; print('CTL MISMATCH',f[:4],g,f[16])
print('rows',len(lines),'bad',bad)
