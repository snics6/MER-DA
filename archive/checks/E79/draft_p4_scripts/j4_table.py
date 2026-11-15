"""表 J4（感度の行）の下書き：results/e79_p4/judge/statements.csv の rule（subset all と unseen_only）と，
combos.csv の P4・P4' の組み合わせの印と単位の数（rev_order の行は dir_rev，ほかは dir）"""
import pandas as pd
C = pd.read_csv('results/e79_p4/judge/combos.csv', low_memory=False)
S = pd.read_csv('results/e79_p4/judge/statements.csv')
T = lambda v: v.astype(str).eq('True')
order = ['main', 'delta_half', 'delta_double', 'cluster_boot', 'cluster_boot,delta_half', 'cluster_boot,delta_double',
         'main+excl_mosei_afew', 'delta_half+excl_mosei_afew', 'iemocap_impro_only', 'iemocap_impro_only,delta_half', 'iemocap_impro_only,delta_double',
         'noscript_reselect', 'noscript_reselect,delta_half', 'noscript_reselect,delta_double', 'nc_seed0', 'nc_seed0,delta_half', 'nc_seed0,delta_double',
         'nc_seed0+excl_mosei_afew', 'nc_seed0,delta_half+excl_mosei_afew', 'rev_order', 'rev_order,delta_half', 'rev_order,delta_double']
print('| 行（tag） | P4 の鍵（all／unseen_only） | P4 の組み合わせ (1)／(2)［一致］／(2′) | P4 の単位 減らした／増やした／全体 | P4′ の鍵（all／unseen_only） | P4′ の組み合わせ (1)／(2)［一致］／(2′) | P4′ の単位 減らした／増やした／全体 | 文の注（notes） |')
print('|' + '---|' * 8)
for tag in order:
    met = 'dir_rev' if 'rev_order' in tag else 'dir'
    cells = [f'`{tag}`']
    notes = []
    for fam in ['P4', "P4'"]:
        g = C[(C.tag == tag) & (C.family == fam) & (C.metric == met)]
        g = g[~T(g.all_missing)]
        r2 = g[T(g.r2)]
        ra = S[(S.tag == tag) & (S.family == fam) & (S.subset == 'all') & (S.rule != 'chance')].iloc[0]
        ru = S[(S.tag == tag) & (S.family == fam) & (S.subset == 'unseen_only') & (S.rule != 'chance')].iloc[0]
        if isinstance(ra.notes, str) and ra.notes:
            notes.append(f'{fam}：{ra.notes}')
        cells += [f'{ra.rule}／{ru.rule}', f'{int(T(g.r1).sum())}／{len(r2)}［{int(T(r2.r2_selection_agrees).sum())}］／{int(T(g.r2p).sum())}',
                  f'{int(g.n_pos.sum())}／{int(g.n_neg.sum())}／{int((g.n_corpora - g.n_missing).sum())}']
    cells.append('；'.join(notes) if notes else '—')
    print('| ' + ' | '.join(cells) + ' |')
