"""固定版（選ばない版）の判定の数（results/e79_p4/judge/combos.csv，tag=main，family=fixed，metric=dir，K=0・5，種類の生徒と融合の生徒）"""
import pandas as pd
C = pd.read_csv('results/e79_p4/judge/combos.csv', low_memory=False)
m = C[(C.tag == 'main') & (C.family == 'fixed') & (C.K.isin([0, 5])) & (C.metric == 'dir') & (C.run == C.student)]
m = m.assign(scope=m.run.str.startswith('fuse').map({True: "P4'", False: 'P4'}))
for sc, g in m.groupby('scope'):
    units = int((g.n_corpora - g.n_missing).sum())
    print(sc, '固定版の数', g.comparison.nunique(), '単位（欠けを除く）', units, '減らした', int(g.n_pos.sum()), '増やした', int(g.n_neg.sum()), '同じ水準', int(g.n_same.sum()), '判定できない', int(g.n_undet.sum()), '欠け', int(g.n_missing.sum()))
    print('  massbal・saerens を除く単位', int((g[~g.comparison.isin(['none-massbal', 'none-saerens'])].n_corpora - g[~g.comparison.isin(['none-massbal', 'none-saerens'])].n_missing).sum()))
