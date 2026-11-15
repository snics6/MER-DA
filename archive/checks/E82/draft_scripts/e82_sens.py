"""E82：感度の行（tag ≠ main）で，比べ (a)・(b) の鍵が固定の版と違う行を数える（results/e82_analysis/compare/statements.csv，subset=all）"""
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
S = pd.read_csv('results/e82_analysis/compare/statements.csv')
x = S[(S.subset == 'all') & (S.tag != 'main')]
x = x.assign(ch_a=x.rule_fixed != x.rule_a, ch_b=x.rule_fixed != x.rule_b)
print(x.groupby(['scope']).agg(rows=('tag', 'size'), ch_a=('ch_a', 'sum'), ch_b=('ch_b', 'sum')))
print(x[x.ch_a | x.ch_b].groupby(['scope', 'dir_a', 'dir_b', 'category']).size())
y = S[(S.subset == 'all') & (S.tag == 'main')]
print(y[['scope', 'family', 'rule_fixed', 'rule_a', 'rule_b', 'same_combos_a', 'same_combos_b']].to_string())
