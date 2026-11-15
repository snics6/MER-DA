"""E82 の離れの表の要約（results/e82_analysis/dist/dist_selected_units.csv と dist_grid.csv．記述．判定には使わない）
dist_selected_units：単位（実行, T, 生徒, K, 手法）ごとに反復の中央値．ここでは手法ごと・版ごとに単位の中央値を取る．"""
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
D = pd.read_csv('results/e82_analysis/dist/dist_selected_units.csv')
print(D.groupby(['version', 'method']).size())
f = lambda g: pd.Series({'n': len(g), 'dW_anchor': g.dW_anchor.median(), 'dW_samecp': g.dW_samecp.median(), 'norm_ratio': g.norm_ratio.median(), 'cos': g.cos.median()})
print('手法 × 版'); print(D.groupby(['method', 'version']).apply(f).round(3))
print('手法 × K × 版'); print(D.groupby(['method', 'K', 'version']).apply(f).round(3))
# 同じ単位で固定 → 選び直しの比（単位ごと）
W = D.pivot_table(index=['run', 'target', 'student', 'K', 'method'], columns='version', values=['dW_anchor', 'dW_samecp', 'cos'])
W.columns = [f'{a}_{b}' for a, b in W.columns]
W = W.reset_index()
W['changed'] = (W.dW_anchor_reselect != W.dW_anchor_fixed)
W['up'] = W.dW_anchor_reselect > W.dW_anchor_fixed
print('選び直しで離れが増えた単位（手法ごと）'); print(W.groupby('method').agg(n=('up', 'size'), changed=('changed', 'sum'), up=('up', 'sum')))
# 選び直した版：倍率ごと
print('選び直した版の倍率ごと'); print(D[D.version == 'reselect'].groupby(['method', 'mult']).apply(f).round(3))
G = pd.read_csv('results/e82_analysis/dist/dist_grid.csv')
G = G[G.method != 'none']
print('格子の全設定（dist_grid）：倍率ごとの中央値の範囲（実行 × K で）'); print(G.groupby(['method', 'mult']).agg(med_min=('dW_anchor_med', 'min'), med_max=('dW_anchor_med', 'max'), cos_min=('cos_med', 'min'), cos_max=('cos_med', 'max')).round(3))
N = pd.read_csv('results/e82_analysis/dist/dist_grid.csv'); N = N[N.method == 'none']
print('none の倍率ごと（K=5 だけ動く）'); print(N[['run', 'K', 'mult', 'n', 'dW_anchor_med']].round(3).to_string())
