"""E82：格子の最良（向きの損）の none − 手法が δ_d 以上の単位を，E82 の範囲（種類の 3 生徒と 2 つの融合の生徒，K=0・5，entmin・IM・自己学習）で数える
（results/e82_analysis/compare/oracle_dir_both.csv）"""
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
B = pd.read_csv('results/e82_analysis/compare/oracle_dir_both.csv')
B = B[(B.run == B.student) & B.K.isin([0, 5]) & B.method.isin(['entmin', 'im', 'self'])]
B['fam'] = B.run.str.startswith('fuse').map({True: "P4'", False: 'P4'})
t = B.groupby(['grid', 'fam', 'method']).agg(n=('at_least_delta_d', 'size'), hit=('at_least_delta_d', 'sum'), grid_min=('grid_size', 'min'), grid_max=('grid_size', 'max'))
print(t)
print(B.groupby(['grid']).at_least_delta_d.agg(['sum', 'size']))
w = B.pivot_table(index=['fam', 'method', 'student', 'K', 'corpus'], columns='grid', values='at_least_delta_d', aggfunc='first')
print('変わった単位'); print(w[w['fixed'] != w['reselect']])
