"""関門 2 の修正案の数を確かめる（E79 p4 と E82）"""
import pandas as pd
T = lambda v: v.astype(str).eq('True')
C = pd.read_csv('results/e79_p4/judge/combos.csv', low_memory=False)
S = pd.read_csv('results/e79_p4/judge/statements.csv')
# (1) 固定版を P4 の手法・定義から同じ版・対照に分ける
m = C[(C.tag == 'main') & (C.family == 'fixed') & (C.K.isin([0, 5])) & (C.metric == 'dir') & (C.run == C.student)]
m = m.assign(scope=m.run.str.startswith('fuse').map({True: "P4'", False: 'P4'}))
grp = lambda c: 'ident' if c in ('none-massbal', 'none-saerens') else ('ctl' if c == 'none-mapctl_fix' else 'meth')
m = m.assign(g=m.comparison.map(grp))
print(m.groupby(['scope', 'g']).agg(n=('comparison', 'nunique'), units=('n_corpora', 'sum'), miss=('n_missing', 'sum'), pos=('n_pos', 'sum'), same=('n_same', 'sum'), neg=('n_neg', 'sum'), undet=('n_undet', 'sum')))
# (2) 表 J4 の注を *_nomd から付け直す：決め手の印の組み合わせ x のどれも r1/r2/r2p_nomd に当たらなければ注
order = ['main', 'delta_half', 'delta_double', 'cluster_boot', 'cluster_boot,delta_half', 'cluster_boot,delta_double',
         'main+excl_mosei_afew', 'delta_half+excl_mosei_afew', 'iemocap_impro_only', 'iemocap_impro_only,delta_half', 'iemocap_impro_only,delta_double',
         'noscript_reselect', 'noscript_reselect,delta_half', 'noscript_reselect,delta_double', 'nc_seed0', 'nc_seed0,delta_half', 'nc_seed0,delta_double',
         'nc_seed0+excl_mosei_afew', 'nc_seed0,delta_half+excl_mosei_afew', 'rev_order', 'rev_order,delta_half', 'rev_order,delta_double']
n_new = 0
for tag in order:
    met = 'dir_rev' if 'rev_order' in tag else 'dir'
    for fam in ['P4', "P4'"]:
        g = C[(C.tag == tag) & (C.family == fam) & (C.metric == met)]
        g = g[~T(g.all_missing)]
        x = None
        for f in ['r1', 'r2', 'r2p']:
            if T(g[f]).any():
                x = g[T(g[f])]; break
        if x is None:
            continue
        keep = (T(x.r1_nomd) | T(x.r2_nomd) | T(x.r2p_nomd))
        note = S[(S.tag == tag) & (S.family == fam) & (S.subset == 'all') & (S.rule != 'chance')].notes.iloc[0]
        has = isinstance(note, str) and '模型に依る単位だけ' in note
        md_only = not keep.any()
        if md_only and not has:
            n_new += 1
        if md_only or has:
            print(f'{tag:38s} {fam:4s} 決め手の組み合わせ {len(x)}，模型を除いて残る {int(keep.sum())}，台本の注 {has}')
print('台本の注が無く付け直す行', n_new)
# (3) 塊のブートストラップで entmin・AVT・K=0 の IEMOCAP
U = pd.read_csv('results/e79_p4/judge/units.csv', low_memory=False, usecols=['family', 'comparison', 'student', 'K', 'corpus', 'sensitivity', 'metric', 'point', 'lo', 'hi', 'delta', 'judgment'])
u = U[(U.family == "P4'") & (U.comparison == 'none-entmin') & (U.student == 'fuse-audio-video-text') & (U.corpus == 'IEMOCAP') & (U.metric == 'dir') & U.sensitivity.isin(['main', 'cluster_boot'])]
print(u.round(3).to_string())
a = U[(U.family.isin(['P4', "P4'"])) & (U.metric == 'dir') & (U.sensitivity.isin(['main', 'cluster_boot']))]
p = a.pivot_table(index=['family', 'comparison', 'student', 'K', 'corpus'], columns='sensitivity', values='judgment', aggfunc='first')
print('main と cluster_boot で判定が違う単位'); print(p[p['main'] != p['cluster_boot']])
