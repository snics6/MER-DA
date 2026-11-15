"""none の向きの損の水準（コーパスごと．反復の平均，IEMOCAP は 2 群の平均）．台本の none_levels と同じ取り方（results/e79_p4/measure/rows_<実行>.csv.gz の method=none，wave=e77 の dirloss）"""
import pandas as pd
pd.set_option('display.width', 250)
corp = lambda t: 'IEMOCAP' if t.startswith('IEMOCAP') else t
for run in ['audio_wavlm', 'video', 'text_roberta', 'fuse-audio-video-text', 'fuse-audio-video']:
    P = pd.read_csv(f'results/e79_p4/measure/rows_{run}.csv.gz', usecols=lambda c: c in ('target', 'student', 'K', 'method', 'wave', 'dirloss', 'rep'), low_memory=False)
    x = P[(P.method == 'none') & (P.wave == 'e77') & (P.K.isin([0, 5]))].copy()
    x['c'] = x.target.map(corp)
    g = x.groupby(['student', 'K', 'target', 'c']).dirloss.mean().groupby(['student', 'K', 'c']).mean()
    for (st, K), h in g.groupby(level=[0, 1]):
        if run.startswith('fuse') and st != run: continue
        print(run, st, K, ' '.join(f'{c}:{v:+.2f}' for (_, _, c), v in h.items()), f'| mean {h.mean():+.3f}')
