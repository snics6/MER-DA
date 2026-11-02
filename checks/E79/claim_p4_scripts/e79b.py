import pandas as pd, numpy as np
R='/home/snics/MER-DA/results/e79_p4/'
J=R+'judge/'
u=pd.read_csv(J+'units.csv',low_memory=False)
c=pd.read_csv(J+'combos.csv',low_memory=False)
ch=pd.read_csv(R+'select/chosen.csv',low_memory=False)
kind={'audio_wavlm','video','text_roberta'}
def m(fam,sens='main',met='dir',var='excl'):
    return u[(u.family==fam)&(u.sensitivity==sens)&(u.metric==met)&(u.variant==var)]
# per (2) combos: corpus details
for fam,combos in [('P4',[('none-im','audio_wavlm',0),('none-t3a','video',0),('none-t3a','video',5)]),
                   ("P4'",[('none-entmin','fuse-audio-video-text',0),('none-entmin','fuse-audio-video-text',5),('none-im','fuse-audio-video-text',5),('none-self','fuse-audio-video',0),('none-t3a','fuse-audio-video',0),('none-t3a','fuse-audio-video',5),('none-t3a','fuse-audio-video-text',0),('none-t3a','fuse-audio-video-text',5)])]:
    mm=m(fam)
    for cmp,st,K in combos:
        x=mm[(mm.comparison==cmp)&(mm.student==st)&(mm.K==K)]
        meth=cmp.split('-')[1]
        sel=ch[(ch.student==st)&(ch.K==K)&(ch.selector==meth)&(ch.variant=='incl')][['target','chose_none']]
        print(fam,cmp,st,K, '|', '; '.join(f"{r.corpus}:{r.judgment}:{r.point:+.3f}{'md' if r.model_dependent else ''}" for r in x.itertuples()), '| incl none:', dict(zip(sel.target,sel.chose_none)))
# (2') combos
print()
for fam,combos in [('P4',[('none-im','text_roberta',0),('none-im','video',0),('none-im','video',5),('none-self','text_roberta',0)]),("P4'",[('none-im','fuse-audio-video',0),('none-im','fuse-audio-video',5),('none-self','fuse-audio-video',5),('none-self','fuse-audio-video-text',0),('none-self','fuse-audio-video-text',5)])]:
    mm=m(fam)
    for cmp,st,K in combos:
        x=mm[(mm.comparison==cmp)&(mm.student==st)&(mm.K==K)]
        meth=cmp.split('-')[1]
        sel=ch[(ch.student==st)&(ch.K==K)&(ch.selector==meth)&(ch.variant=='incl')][['target','chose_none']]
        print(fam,cmp,st,K,'|','; '.join(f"{r.corpus}:{r.judgment}:{r.point:+.3f}{'md' if r.model_dependent else ''}" for r in x.itertuples()),'| incl none:',dict(zip(sel.target,sel.chose_none)))
# P4' unit counts by corpus
mm=m("P4'"); pos=mm[mm.judgment=='pos']
print(pos.corpus.value_counts().to_dict(), 'md',pos.model_dependent.sum(), 'IEMOCAP nonmd', pos[(pos.corpus=='IEMOCAP')&(~pos.model_dependent.astype(bool))][['comparison','K']].values.tolist())
# entmin IEMOCAP AVT paragraph
x=mm[(mm.comparison=='none-entmin')&(mm.student=='fuse-audio-video-text')]
print(x[['K','corpus','point','point_auc','judgment']].to_string())
d=m("P4_desc_uar" if False else 'P4_desc_uar',met='uar')
print(d.comparison.unique()[:20])
d=u[(u.family=='P4_desc_uar')&(u.sensitivity=='main')&(u.student=='fuse-audio-video-text')&(u.comparison.str.contains('entmin'))&(u.corpus=='IEMOCAP')]
print(d[['comparison','K','point','lo','hi','judgment','variant']].to_string())
# P4 DANN inactive / da_effect
da=pd.read_csv(J+'da_effect.csv')
dk=da[da.student.isin(kind)]
print(dk.groupby(['method','K']).da_changed_predictions.agg(['sum','size']))
# ctl
for fam in ['P4_ctl',"P4'_ctl"]:
    x=m(fam); print(fam,len(x),x.judgment.value_counts().to_dict())
    print(x[x.judgment=='neg'].comparison.value_counts().to_dict())
cd=pd.read_csv(J+'ctl_disagree.csv')
cdk=cd[cd.K.isin([0,5])]
for grp,sub in [('P4',cdk[cdk.student.isin(kind)]),("P4'",cdk[cdk.student.str.startswith('fuse')])]:
    print(grp,len(sub),'disagree',sub.disagree.sum(), 'inactive involved', ((sub.judgment_none=='inactive')&sub.disagree).sum())
x=cdk[(cdk.method=='dann')&(cdk.K==5)&(cdk.student.isin(kind))&(cdk.judgment_none=='neg')]
print(x[['student','corpus','judgment_ctl']].to_string())
