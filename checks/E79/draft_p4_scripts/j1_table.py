"""表 J1 の下書き：P4・P4' の組み合わせごとの判定の数（results/e79_p4/judge/combos.csv，tag=main，metric=dir）．
対照の族（P4_ctl・P4'_ctl）の数を並べる．"""
import pandas as pd
C = pd.read_csv('results/e79_p4/judge/combos.csv', low_memory=False)
T = lambda v: str(v) == 'True'
SN = {'audio_wavlm': 'A', 'video': 'V', 'text_roberta': 'T', 'fuse-audio-video-text': 'AVT 融合', 'fuse-audio-video': 'AV 融合'}
m = C[(C.tag == 'main') & (C.family.isin(['P4', "P4'"])) & (C.metric == 'dir')]
ctl = C[(C.tag == 'main') & (C.family.isin(['P4_ctl', "P4'_ctl"])) & (C.metric == 'dir')]
CTL = {}
for r in ctl.itertuples():
    meth = r.comparison.split('-')[1]
    CTL[(r.family.replace('_ctl', ''), meth, r.student, r.K)] = f'{r.n_pos}/{r.n_same}/{r.n_neg}/{r.n_undet}'
order = {'entmin': 0, 'self': 1, 'im': 2, 't3a': 3, 'lp': 4, 'coral': 5, 'dann': 6, 'mme': 7}
m = m.assign(o=m.method.map(order), s=m.student.map(list(SN).index)).sort_values(['family', 'o', 's', 'K'])
print('| 族 | 手法 | 生徒 | K | コーパス（過半） | 減らした | 同じ水準 | 増やした | 判定できない | 働かなかった | 欠け | 模型に依る単位 | 除いた後の 減らした／同じ水準／増やした | 区間の半分の幅の中央値 | 印 | 予言 | 対照の族 減／同／増／判 |')
print('|' + '---|' * 17)
for r in m.itertuples():
    if T(r.all_missing):
        mark = '全部欠け'
    else:
        mark = '(1)' if T(r.r1) else '(2)' if T(r.r2) else "(2')" if T(r.r2p) else '(3) に合う' if T(r.r3ok) else '(4)' if T(r.r4) else '?'
    if T(r.r2):
        mark += '・一致' if T(r.r2_selection_agrees) else '・不一致'
    if T(r.r2p):
        mark += '・残りの過半が同じ水準か増やした' if T(r.r2p_rest_same_or_inc) else '・残りの過半に届かない'
        mark += '・選択が手法を選んだ' if T(r.r2p_selection_chose_method) else '・選択は none'
    pred = '見た後' if T(r.seen_before_prediction) else '見る前'
    hw = '—' if pd.isna(r.ci_halfwidth_median) else f'{r.ci_halfwidth_median:.2f}'
    print(f'| {r.family} | {r.method} | {SN[r.student]} | {r.K} | {r.n_corpora}（{r.majority}） | {r.n_pos} | {r.n_same} | {r.n_neg} | {r.n_undet} | {r.n_inactive} | {r.n_missing} | {r.n_model_dependent} | {r.n_pos_nomd}／{r.n_same_nomd}／{r.n_neg_nomd} | {hw} | {mark} | {pred} | {CTL.get((r.family, r.method, r.student, r.K), "—")} |')
