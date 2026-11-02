"""(2) と (2') の組み合わせで，none を含む選択（incl）が手法を選んだか（results/e79_p4/select/chosen.csv の chose_none）と，判定（units.csv の main）を並べる"""
import pandas as pd
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 400)
S = pd.read_csv('results/e79_p4/select/chosen.csv', low_memory=False)
m = pd.read_pickle('/tmp/claude-1000/-home-snics-MER-DA/4cd03bf4-8db6-4e9a-a1e4-1871eadc639a/scratchpad/p4_main_units.pkl')
C = pd.read_csv('results/e79_p4/judge/combos.csv', low_memory=False)
cm = C[(C.tag == 'main') & (C.family.isin(['P4', "P4'"])) & (C.metric == 'dir')]
corp = lambda t: 'IEMOCAP' if t.startswith('IEMOCAP') else t
S['corpus'] = S.target.map(corp)
for flag in ['r2', 'r2p']:
    hits = cm[cm[flag].astype(str) == 'True']
    print('=====', flag, len(hits))
    for r in hits.itertuples():
        meth = r.comparison.split('-', 1)[1]
        u = m[(m.comparison == r.comparison) & (m.student == r.student) & (m.K == r.K)]
        sel = S[(S.run == r.run) & (S.student == r.student) & (S.K == r.K) & (S.selector == meth) & (S.variant == 'incl')]
        s = sel.groupby('corpus').chose_none.apply(lambda v: ','.join(map(str, v)))
        out = []
        for x in u.itertuples():
            out.append(f'{x.corpus}:{x.judgment}({x.point:+.2f};md={x.model_dependent};incl_chose_none={s.get(x.corpus)})')
        print(r.family, r.comparison, r.student, 'K', r.K, 'agree=', r.r2_selection_agrees, 'chose_method=', r.r2p_selection_chose_method, 'rest_same_or_inc=', r.r2p_rest_same_or_inc)
        print('   ', ' | '.join(out))
