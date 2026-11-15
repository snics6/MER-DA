import pandas as pd
J='/home/snics/MER-DA/results/e79_p4/judge/'
c=pd.read_csv(J+'combos.csv',low_memory=False)
s=pd.read_csv(J+'statements.csv')
for tag in s.tag.unique():
    row=[tag]
    for fam in ['P4',"P4'"]:
        k=[s[(s.tag==tag)&(s.family==fam)&(s.subset==sub)&(s.rule!='chance')].rule.tolist() for sub in ['all','unseen_only']]
        met='dir_rev' if tag.startswith('rev_order') else 'dir'
        base=tag.split('+')[0]
        cc=c[(c.tag==tag)&(c.family==fam)&(c.metric==met)]
        cc=cc[~cc.all_missing.astype(bool)]
        r2=cc[cc.r2.astype(bool)]
        row.append(f"{fam}: {k} r1={int(cc.r1.sum())} r2={len(r2)}[{int(r2.r2_selection_agrees.astype(bool).sum())}] r2p={int(cc.r2p.sum())} pos={int(cc.n_pos.sum())} neg={int(cc.n_neg.sum())} N={int(cc[['n_pos','n_neg','n_same','n_undet','n_inactive']].sum().sum())}")
        n=s[(s.tag==tag)&(s.family==fam)&(s.subset=='all')&(s.rule!='chance')].notes.tolist()
        if n and isinstance(n[0],str): row.append('NOTE:'+n[0][:120])
    print(' | '.join(row))
# delta_half (1) combo
cc=c[(c.tag=='delta_half')&(c.family=='P4')&(c.metric=='dir')&(c.r1.astype(bool))]
print(cc[['comparison','student','K','n_pos','seen_before_prediction']])
cc=c[(c.tag=='rev_order')&(c.family=='P4')&(c.metric=='dir_rev')&(c.r2.astype(bool))]
print(cc[['comparison','student','K','n_pos','n_neg']])
print(c.tag.unique())
