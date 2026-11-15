"""results/e79_p4/judge/oracle_dir.csv：P4（種類の生徒）と P4'（融合の生徒）の K=0・5 で，格子の最良の none − 手法が δ_d 以上の単位を数える"""
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
O = pd.read_csv('results/e79_p4/judge/oracle_dir.csv')
O = O[O.K.isin([0, 5])]
kind = O.run == O.student
fus = O.run.str.startswith('fuse') & (O.run == O.student)
for name, sel in [('P4', kind & ~O.run.str.startswith('fuse')), ("P4'", fus)]:
    x = O[sel]
    print(name, '単位', len(x), 'NaN', int(x.none_minus_best_dirloss.isna().sum()), 'δ_d 以上', int(x.at_least_delta_d.sum()))
    print(x[x.at_least_delta_d].round(3).to_string())
    print(x.groupby('method').at_least_delta_d.agg(['sum', 'count']))
