"""E82 6 節の表の下書き：3 手法 × 族 × K ごとの判定の数（results/e82_analysis/compare/judgments.csv，sensitivity=main，metric=dir），
選ばれた倍率（multipliers.csv，level=family×K，variant=excl），候補の数（candidates.csv）"""
import pandas as pd
J = pd.read_csv('results/e82_analysis/compare/judgments.csv')
M = pd.read_csv('results/e82_analysis/compare/multipliers.csv')
Cn = pd.read_csv('results/e82_analysis/compare/candidates.csv')
J = J[(J.sensitivity == 'main') & (J.metric == 'dir')]
G = {'fixed': '固定', 'reselect (a) none(E39)': '(a)', 'reselect (b) none(同じ C′)': '(b)'}
fmt = lambda r: f'{int(r.n_pos)}／{int(r.n_same)}／{int(r.n_neg)}／{int(r.n_undet)}'
print('| 族 | 手法 | K | 単位 | 固定 減／同／増／判 | (a) 減／同／増／判 | (b) 減／同／増／判 | 選ばれた倍率 1／3／10（単位） | 候補の数 固定 → 選び直し |')
print('|' + '---|' * 9)
tot = {g: [0, 0, 0, 0, 0] for g in G}
for fam in ['P4', "P4'"]:
    for meth in ['entmin', 'im', 'self']:
        for K in [0, 5]:
            row = []
            for g in G:
                r = J[(J.family == fam) & (J.method == meth) & (J.K == K) & (J.grid == g)].iloc[0]
                row.append(fmt(r)); n = int(r.n_units)
                for i, c in enumerate(['n_pos', 'n_same', 'n_neg', 'n_undet', 'n_inactive']):
                    tot[g][i] += int(r[c])
            m = M[(M.level == 'family×K') & (M.family == fam) & (M.selector == meth) & (M.K == K) & (M.variant == 'excl')].iloc[0]
            c0 = Cn[(Cn.family == fam) & (Cn.selector == meth) & (Cn.K == K) & (Cn.grid == 'fixed')].n_cand_median.iloc[0]
            c1 = Cn[(Cn.family == fam) & (Cn.selector == meth) & (Cn.K == K) & (Cn.grid != 'fixed')].n_cand_median.iloc[0]
            print(f'| {fam} | {meth} | {K} | {n} | {row[0]} | {row[1]} | {row[2]} | {int(m.n_mult1)}／{int(m.n_mult3)}／{int(m.n_mult10)}（{int(m.n_chosen)}） | {int(c0)} → {int(c1)} |')
print(tot)
print(Cn.grid.unique())
