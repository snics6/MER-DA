"""E82：主の行の単位ごとの判定の移り（results/e82_analysis/compare/judgment_units.csv．sensitivity=main，metric=dir）"""
import pandas as pd
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 300)
J = pd.read_csv('results/e82_analysis/compare/judgment_units.csv')
m = J[(J.sensitivity == 'main') & (J.metric == 'dir')]
print('単位', len(m), m.groupby('family').size().to_dict())
print('変わった (a)', int(m.changed_a.sum()), '(b)', int(m.changed_b.sum()), '(a)≠(b)', int((~m.same_a_b).sum()))
print(m.groupby(['family', 'method']).agg(n=('changed_a', 'size'), ch_a=('changed_a', 'sum'), ch_b=('changed_b', 'sum')))
print(pd.crosstab([m.method, m.judgment_fixed], m.judgment_a))
print(pd.crosstab([m.method, m.judgment_fixed], m.judgment_b))
# 「減らした」に入った・出た単位
x = m[(m.judgment_fixed == 'pos') | (m.judgment_a == 'pos') | (m.judgment_b == 'pos')]
print(x[['family', 'method', 'student', 'K', 'corpus', 'judgment_fixed', 'judgment_a', 'judgment_b']].to_string())
