"""results/e79_p4/judge/none_chosen.csv：P4（種類）と P4'（融合の生徒）の K=0・5 で，none を含めて選んだ版（incl）で none が選ばれた群の数（sum／size．単位は評価のターゲットの群）"""
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
D = pd.read_csv('results/e79_p4/judge/none_chosen.csv')
D = D[D.K.isin([0, 5]) & D.selector.isin(['entmin', 'self', 'im', 't3a', 'lp', 'coral', 'dann', 'mme'])]
print(D.role.unique())
for nm, s in [('P4', D.role == 'kind'), ("P4'", D.role == 'fused')]:
    x = D[s]
    t = x.groupby('selector')[['sum', 'size']].sum()
    print(nm); print(t); print('計', int(t['sum'].sum()), '/', int(t['size'].sum()))
