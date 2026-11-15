import pandas as pd, numpy as np
J='/home/snics/MER-DA/results/e79_p4/judge/'
u=pd.read_csv(J+'units.csv',low_memory=False)
c=pd.read_csv(J+'combos.csv',low_memory=False)
s=pd.read_csv(J+'statements.csv')

def rejudge(r):
    p,lo,hi,d=r.point,r.lo,r.hi,r.delta
    if pd.isna(p) or pd.isna(lo): return 'missing'
    if lo>0 and p>=d: return 'pos'
    if hi<0 and p<=-d: return 'neg'
    if lo>=-d and hi<=d: return 'same'
    return 'undet'

for fam in ['P4',"P4'"]:
    m=u[(u.family==fam)&(u.sensitivity=='main')&(u.metric=='dir')&(u.variant=='excl')].copy()
    m['rj']=m.apply(rejudge,axis=1)
    # inactive replace
    jr=m.judgment_raw.fillna(m.judgment)
    mism=(m.rj!=jr)&~m.judgment.isin(['missing'])
    print(fam,len(m),'raw-judgment mismatches',mism.sum())
    print(m.judgment.value_counts().to_dict())
    print('inactive raw',m[m.judgment=='inactive'].judgment_raw.value_counts().to_dict())
    pos=m[m.judgment=='pos']
    print(pos[['comparison','student','K','corpus','point','lo','hi','delta','model_dependent','point_auc']].to_string())
    und=m[m.judgment=='undet']
    hw=(und.hi-und.lo)/2
    print('undet halfwidth',round(hw.min(),3),round(hw.max(),3),round(hw.median(),3))
    print('model_dependent count all units', m.model_dependent.sum(), 'pos md',pos.model_dependent.sum())

# combos main P4 dir
for fam in ['P4',"P4'"]:
    cc=c[(c.tag=='main')&(c.family==fam)&(c.metric=='dir')]
    print(fam,'ncombos',len(cc),'all_missing',cc.all_missing.sum())
    cc=cc[~cc.all_missing.astype(bool)]
    print(' r1',cc.r1.sum(),'r2',cc.r2.sum(),'r2p',cc.r2p.sum(),'r3ok',cc.r3ok.sum(),'r4',cc.r4.sum())
    print(' nomd r1',cc.r1_nomd.sum(),'r2',cc.r2_nomd.sum(),'r2p',cc.r2p_nomd.sum(),'r3ok',cc.r3ok_nomd.sum(),'r4',cc.r4_nomd.sum())
    print(' sums pos',cc.n_pos.sum(),'neg',cc.n_neg.sum(),'same',cc.n_same.sum(),'undet',cc.n_undet.sum(),'inact',cc.n_inactive.sum(),'miss',cc.n_missing.sum(), 'md',cc.n_model_dependent.sum())
    print(cc[cc.r2.astype(bool)][['comparison','student','K','n_pos','n_neg','r2_selection_agrees','seen_before_prediction']].to_string())
    print(cc[cc.r2p.astype(bool)][['comparison','student','K','n_pos','n_neg','n_same','n_undet','r2p_rest_same_or_inc','r2p_selection_chose_method','ci_halfwidth_median']].to_string())
    print(' sign_p_holm min',cc.sign_p_holm.min())
    print(cc.groupby('student')[['delta_d','none_dirloss','delta_d_ratio_to_none_dirloss']].first() if False else cc[['student','K','delta_d','none_dirloss','delta_d_ratio_to_none_dirloss']].drop_duplicates(['student','K']).to_string())
print(s[(s.tag=='main')][['family','subset','rule','n_combos','combos','statement_key','notes','delta_d']].to_string())
