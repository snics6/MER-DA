"""表 J4 の「模型に依る単位だけで決まった」の注を combos.csv の *_nomd から付け直す．
定義 A：行（tag × 族 × subset）の決め手の印の組み合わせ（(1) があれば (1)，なければ (2)，なければ (2′)）のどれも，
模型に依る単位を除くと (1)・(2)・(2′) に当たらない．定義 B：決め手の印の組み合わせのうち，模型に依る単位を除くと同じ印に当たらない組み合わせの数"""
import pandas as pd
T = lambda v: v.astype(str).eq('True')
C = pd.read_csv('results/e79_p4/judge/combos.csv', low_memory=False)
S = pd.read_csv('results/e79_p4/judge/statements.csv')
tot = {'A_all': 0, 'A_unseen': 0, 'B_rows': 0}
for tag in C.tag.unique():
    met = 'dir_rev' if 'rev_order' in tag else 'dir'
    for fam in ['P4', "P4'"]:
        for sub in ['all', 'unseen_only']:
            g = C[(C.tag == tag) & (C.family == fam) & (C.metric == met)]
            g = g[~T(g.all_missing)]
            if sub == 'unseen_only':
                g = g[~T(g.seen_before_prediction)]
            x, f = None, None
            for f in ['r1', 'r2', 'r2p']:
                if T(g[f]).any():
                    x = g[T(g[f])]; break
            if x is None:
                continue
            keepA = (T(x.r1_nomd) | T(x.r2_nomd) | T(x.r2p_nomd)).any()
            nB = int((~T(x[f + '_nomd'])).sum())
            note = S[(S.tag == tag) & (S.family == fam) & (S.subset == sub) & (S.rule != 'chance')].notes.iloc[0]
            has = isinstance(note, str) and '模型に依る単位だけ' in note
            if not keepA and not has:
                tot['A_' + ('all' if sub == 'all' else 'unseen')] += 1
                print('A', tag, fam, sub, len(x))
            if nB and not has:
                tot['B_rows'] += 1
print(tot)
