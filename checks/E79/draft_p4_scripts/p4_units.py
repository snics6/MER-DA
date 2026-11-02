"""E79 p4：主の行の P4・P4' の単位を書き出す（results/e79_p4/judge/units.csv．sensitivity=main，metric=dir，variant=excl）"""
import pandas as pd
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 400)
U = pd.read_csv('results/e79_p4/judge/units.csv', low_memory=False)
m = U[(U.family.isin(['P4', "P4'"])) & (U.sensitivity == 'main') & (U.metric == 'dir')]
print(m.variant.unique(), len(m))
m.to_pickle('/tmp/claude-1000/-home-snics-MER-DA/4cd03bf4-8db6-4e9a-a1e4-1871eadc639a/scratchpad/p4_main_units.pkl')
cols = ['family','comparison','student','K','corpus','point','lo','hi','delta','judgment','judgment_raw','inactive','inactive_a','model_dependent','point_auc','model_err_diff','interval_rule']
r2 = [('none-im','audio_wavlm',0),('none-t3a','video',0),('none-t3a','video',5),
      ('none-entmin','fuse-audio-video-text',0),('none-entmin','fuse-audio-video-text',5),('none-im','fuse-audio-video-text',5),
      ('none-self','fuse-audio-video',0),('none-t3a','fuse-audio-video',0),('none-t3a','fuse-audio-video',5),('none-t3a','fuse-audio-video-text',0),('none-t3a','fuse-audio-video-text',5)]
for c,s,k in r2:
    print(m[(m.comparison==c)&(m.student==s)&(m.K==k)][cols].round(3).to_string(header=False))
    print()
