"""E79 p4 の補助の数え上げ
(1) results/e79_p4/judge/ctl_disagree.csv：DANN・MME・CORAL の none との判定と対照との判定（P4 の範囲，K=0・5）
(2) results/e79_p4/judge/units.csv：主の行の P4・P4' の「判定できない」単位の区間の半分の幅"""
import pandas as pd
pd.set_option('display.width', 250)
D = pd.read_csv('results/e79_p4/judge/ctl_disagree.csv')
D = D[D.K.isin([0, 5]) & (D.run == D.student) & ~D.run.str.startswith('fuse')]
print(pd.crosstab([D.method, D.K, D.judgment_none], D.judgment_ctl))
m = pd.read_pickle('/tmp/claude-1000/-home-snics-MER-DA/4cd03bf4-8db6-4e9a-a1e4-1871eadc639a/scratchpad/p4_main_units.pkl')
u = m[m.judgment == 'undet']
hw = (u.hi - u.lo) / 2
print(hw.groupby(u.family).agg(['size', 'min', 'median', 'max']).round(3))
