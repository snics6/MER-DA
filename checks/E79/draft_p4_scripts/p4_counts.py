"""E79 p4 の結果の数え上げ（下書き用）．results/e79_p4/judge/combos.csv から数える．"""
import pandas as pd, sys
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 300)
C = pd.read_csv('results/e79_p4/judge/combos.csv', low_memory=False)
tags = sys.argv[1:] or ['main']
for tag in tags:
    for fam in ['P4', "P4'", 'P4_ctl', "P4'_ctl"]:
        g = C[(C.tag == tag) & (C.family == fam) & (C.metric == 'dir')]
        if g.empty: continue
        nm = g[~g.all_missing.astype(str).eq('True')]
        def cnt(col): return int(nm[col].astype(str).eq('True').sum())
        print(f'== {tag} {fam}: combos(全部欠けを除く)={len(nm)}  units={int((nm.n_corpora-nm.n_missing).sum())}')
        print('  単位: pos', int(g.n_pos.sum()), 'neg', int(g.n_neg.sum()), 'same', int(g.n_same.sum()), 'undet', int(g.n_undet.sum()),
              'inactive', int(g.n_inactive.sum()), 'missing', int(g.n_missing.sum()), 'model_dep', int(g.n_model_dependent.sum()))
        if 'r1' in g and g.r1.notna().any():
            print('  組: r1', cnt('r1'), 'r2', cnt('r2'), "r2p", cnt('r2p'), 'r3ok', cnt('r3ok'), 'r4', cnt('r4'))
            print('  組(nomd): r1', cnt('r1_nomd'), 'r2', cnt('r2_nomd'), "r2p", cnt('r2p_nomd'), 'r3ok', cnt('r3ok_nomd'), 'r4', cnt('r4_nomd'))
            print('  hw中央値の範囲', round(nm.ci_halfwidth_median.min(),3), round(nm.ci_halfwidth_median.max(),3))
            for meth, h in nm.groupby('method'):
                print(f'   {meth:7s} pos {int(h.n_pos.sum())} neg {int(h.n_neg.sum())} same {int(h.n_same.sum())} undet {int(h.n_undet.sum())} inact {int(h.n_inactive.sum())} miss {int(h.n_missing.sum())}')
