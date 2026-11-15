"""round5（G2，2026-11-16）：05_Q1.md の表 5・表 6・表 新4 と 06_Q2.md の表 9 を結果ファイルから Markdown で書き出す．
結果ファイルは読むだけ．使い方（DAI の直下で）：./.venv/bin/python reviews/round5_ja/G2_表の書き出し.py <出力の置き場>
出力：table5.md，table6.md（K = 5，10，25），tablenew4.md（K = 0），table9.md（K = 5，25）．
丸め：水準は小数 1 桁．表 6・新4 は 2 桁，表 9 は 1 桁．区間の端がその桁で 0 に丸まるときは 1 桁増やして符号を残す．
"""
import sys
import pandas as pd
OUT = sys.argv[1].rstrip('/') + '/'
L = pd.read_csv('results/dai1/stats/desc/levels.csv')
F = pd.read_csv('results/dai1/stats/families_flagged.csv')
O = pd.read_csv('results/dai1/stats/desc/outside_families.csv')      # K = 10（族の外の数．正本）
c14 = pd.read_csv('results/dai14/summary/conditions.csv')
c22 = pd.read_csv('results/dai22/summary/conditions.csv'); c22 = c22[c22.subset == 'all']
D = pd.read_csv('results/dai3/eval/decomp.csv')
C = ['AFEW', 'IEMOCAP', 'MOSEI', 'MELD']; S = ['A', 'V', 'T', 'AVT']
nm5 = {'A': '声 A', 'V': '顔 V', 'T': '言葉 T', 'AVT': 'AVT'}; nm9 = {'A': '声', 'V': '顔', 'T': '言葉', 'AVT': 'AVT'}
M = '−'
def sg(v, d):
    r = round(v, d)
    if r == 0 and v != 0 and d < 4:
        return sg(v, d + 1)
    return f'{r:+.{d}f}'.replace('-', M)
def sgp(v, d): return f'{v:+.{d}f}'.replace('-', M)
def ns(v, d=1): return f'{v:.{d}f}'.replace('-', M)
def one(df, **kw):
    x = df
    for k, v in kw.items(): x = x[x[k] == v]
    assert len(x) == 1, (kw, len(x)); return x.iloc[0]
def fam(f, T, s, K): return one(O if K == 10 else F, family=f, T=T, student=s, K=K)
def cell2(r): return f'{sgp(r.effect, 2)} [{sg(r.lo, 2)}, {sg(r.hi, 2)}]'
# 表 5
t = ['| 分類器 | コーパス | K = 0：ソースだけ（アンカー） | K = 0：学び直し | K = 5：基準 | K = 5：学び直し | K = 10：基準 | K = 10：学び直し | K = 25：基準 | K = 25：学び直し |',
     '|-----|-------|--------|-------|------|-------|------|-------|------|-------|']
mx = 0
for s in S:
    for T in C:
        v = []
        for K in (0, 5, 10, 25):
            b = one(L, T=T, student=s, K=K, row='none'); r = one(L, T=T, student=s, K=K, row='joint|one|main')
            assert b.n_units == r.n_units == (5 if K == 0 else 50)
            mx = max(mx, abs(r.effect - b.effect - fam(1, T, s, K).effect))
            v += [ns(b.effect), ns(r.effect)]
        t.append(f'| {nm5[s]} | {T} | ' + ' | '.join(v) + ' |')
assert mx < 1e-9, mx
open(OUT + 'table5.md', 'w').write('\n'.join(t) + '\n')
# 表 6（K = 5，10，25．族 1 と族 3）
t = ['| 分類器 | コーパス | K = 5 | K = 10 | K = 25 |', '|----|-----|--------------|--------------|--------------|']
for f, lab in ((1, '**gain**'), (3, '**形 (a) − 選ばれた形**')):
    t.append(f'| {lab} | | | | |')
    for s in S:
        for T in C:
            t.append(f'| {nm5[s]} | {T} | ' + ' | '.join(cell2(fam(f, T, s, K)) for K in (5, 10, 25)) + ' |')
for s in S:
    for T in C:
        for K in (5, 25): assert bool(fam(4, T, s, K).identical)
        assert bool(fam(4, T, s, 10).se_zero)
open(OUT + 'table6.md', 'w').write('\n'.join(t) + '\n')
# 表 新4（K = 0．族 1・3・4）
t = ['| 分類器 | コーパス | gain | 形 (a) − 選ばれた形 | 選んだ値 − 調整しない値 |', '|----|-----|--------------|--------------|--------------|']
for s in S:
    for T in C:
        r4 = fam(4, T, s, 0)
        c4 = '0（同じ分類器）' if bool(r4.identical) else cell2(r4)
        t.append(f'| {nm5[s]} | {T} | {cell2(fam(1, T, s, 0))} | {cell2(fam(3, T, s, 0))} | {c4} |')
open(OUT + 'tablenew4.md', 'w').write('\n'.join(t) + '\n')
# 表 9（K = 5，25）
def q(src, T, s, K, qq): return one(src, T=T, student=s, K=K, quantity=qq)
def iv(r): return f'[{sg(r.lo, 1)}, {sg(r.hi, 1)}]'
t = ['| K | 分類器 | コーパス | 余地（テスト） | K 本の基準（候補） | q（候補） | 当たる幅 Δ（候補） | 再現できる分 P（候補．R0 / R1） | gain（テスト） |',
     '|--|---|-----|---------|----|----|---------|----------------|---------|']
for K in (5, 25):
    for s in S:
        for T in C:
            src = c22 if s == 'AVT' else c14
            b = q(src, T, s, K, 'uar_none'); qq = q(src, T, s, K, 'uar_pseudo'); de = q(src, T, s, K, 'delta')
            p0 = q(src, T, s, K, 'imitable_R0'); p1 = q(src, T, s, K, 'imitable_R1')
            if s != 'AVT': assert abs(q(src, T, s, K, 'uar_judge').effect - b.effect) < 1e-9
            rm = one(D, T=T, item=f'{s}|none|{K}', version='four', quantity='sum')
            g = one(F, family=1, T=T, student=s, K=K)
            t.append(f'| {K} | {nm9[s]} | {T} | {ns(rm.effect)} {iv(rm)} | {ns(b.effect)} | {ns(qq.effect)} | {sgp(de.effect, 1)} {iv(de)} | '
                     f'{ns(p0.effect)} {iv(p0)} / {ns(p1.effect)} {iv(p1)} | {sgp(g.effect, 1)} {iv(g)} |')
open(OUT + 'table9.md', 'w').write('\n'.join(t) + '\n')
print('wrote', OUT)
