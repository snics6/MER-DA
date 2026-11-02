"""表 J4（感度の行）の第 2 版：注の列を combos.csv の *_nomd から付け直す（関門 2 checks/E79/claim_p4.md の [A1-1]）．
注を付ける条件：行（tag × 族 × subset）の決め手の印（(1)，なければ (2)，なければ (2′)）の組み合わせのどれも，
模型に依る単位を除くと (1)・(2)・(2′) に当たらない．台本の注のうち対照の族の注はそのまま残し，〈手法〉〈コーパス〉を埋める"""
import pandas as pd
C = pd.read_csv('results/e79_p4/judge/combos.csv', low_memory=False)
S = pd.read_csv('results/e79_p4/judge/statements.csv')
T = lambda v: v.astype(str).eq('True')
order = ['main', 'delta_half', 'delta_double', 'cluster_boot', 'cluster_boot,delta_half', 'cluster_boot,delta_double',
         'main+excl_mosei_afew', 'delta_half+excl_mosei_afew', 'iemocap_impro_only', 'iemocap_impro_only,delta_half', 'iemocap_impro_only,delta_double',
         'noscript_reselect', 'noscript_reselect,delta_half', 'noscript_reselect,delta_double', 'nc_seed0', 'nc_seed0,delta_half', 'nc_seed0,delta_double',
         'nc_seed0+excl_mosei_afew', 'nc_seed0,delta_half+excl_mosei_afew', 'rev_order', 'rev_order,delta_half', 'rev_order,delta_double']
FAMN = {'P4': 'P4', "P4'": 'P4′'}
def md_only(g):
    x = None
    for f in ['r1', 'r2', 'r2p']:
        if T(g[f]).any():
            x = g[T(g[f])]; break
    if x is None:
        return False
    return not (T(x.r1_nomd) | T(x.r2_nomd) | T(x.r2p_nomd)).any()
print('| 行（tag） | P4 の鍵（all／unseen_only） | P4 の組み合わせ (1)／(2)［一致］／(2′) | P4 の単位 減らした／増やした／全体 | P4′ の鍵（all／unseen_only） | P4′ の組み合わせ (1)／(2)［一致］／(2′) | P4′ の単位 減らした／増やした／全体 | 注 |')
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
        subs = [s for s, gg in (('all', g), ('unseen_only', g[~T(g.seen_before_prediction)])) if md_only(gg)]
        if subs:
            notes.append(f'{FAMN[fam]}（{"・".join(subs)}）：この印は模型に依る単位だけで決まった')
        if isinstance(ra.notes, str) and 'DA の項' in ra.notes:
            notes.append(f'{FAMN[fam]}：MME の DA の項は同じ形の対照より向きの損を減らした（V の K=0 の MOSEI．模型に依る単位）が，none より減らしたとは言えない')
        cells += [f'{ra.rule}／{ru.rule}'.replace("'", '′'), f'{int(T(g.r1).sum())}／{len(r2)}［{int(T(r2.r2_selection_agrees).sum())}］／{int(T(g.r2p).sum())}',
                  f'{int(g.n_pos.sum())}／{int(g.n_neg.sum())}／{int((g.n_corpora - g.n_missing).sum())}']
    cells.append('；'.join(notes) if notes else '—')
    print('| ' + ' | '.join(cells) + ' |')
