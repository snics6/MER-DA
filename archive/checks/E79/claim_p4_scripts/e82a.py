import pandas as pd
C='/home/snics/MER-DA/results/e82_analysis/compare/'
mu=pd.read_csv(C+'multipliers.csv')
print(mu.level.unique(), mu.variant.unique())
x=mu[(mu.variant=='excl')]
print(x[x.level=='all3'][['n_chosen','n_mult1','n_mult3','n_mult10','n_tie_multi_mult','n_same_as_fixed']].to_string())
print(x[x.level=='method'][['selector','n_chosen','n_mult1','n_mult3','n_mult10','n_tie_multi_mult','n_same_as_fixed']].to_string())
print(x[x.level=='family×K'][['family','selector','K','n_chosen','n_mult1','n_mult3','n_mult10']].to_string())
print('tie any',mu.n_tie_multi_mult.sum())
ca=pd.read_csv(C+'candidates.csv'); print(ca.to_string())
j=pd.read_csv(C+'judgments.csv')
print(j.sensitivity.unique(), j.metric.unique(), j.grid.unique())
jm=j[(j.sensitivity=='main')&(j.metric=='dir')]
print(jm[['family','method','K','grid','n_units','n_pos','n_same','n_neg','n_undet','n_inactive','n_missing']].to_string())
for g in jm.grid.unique():
    s=jm[jm.grid==g]; print(g, s[['n_pos','n_same','n_neg','n_undet','n_inactive','n_units']].sum().to_dict())
ju=pd.read_csv(C+'judgment_units.csv')
print(ju.sensitivity.unique(), ju.metric.unique(), ju.variant.unique())
m=ju[(ju.sensitivity=='main')&(ju.metric=='dir')]
print(len(m), m.changed_a.sum(), m.changed_b.sum())
print(m.groupby(['family','method']).changed_a.sum())
e=m[m.method=='entmin']
print(pd.crosstab(e.judgment_fixed,e.judgment_a))
p=m[(m.method=='entmin')&(m.family=='P4')]
print('entmin P4 neg fixed',(p.judgment_fixed=='neg').sum(),'a',(p.judgment_a=='neg').sum(),'b',(p.judgment_b=='neg').sum(),'same fixed',(p.judgment_fixed=='same').sum(),'a',(p.judgment_a=='same').sum(),'b',(p.judgment_b=='same').sum())
pos=m[(m.judgment_fixed=='pos')|(m.judgment_a=='pos')|(m.judgment_b=='pos')]
print(pos[['family','method','student','K','corpus','judgment_fixed','judgment_a','judgment_b']].to_string())
