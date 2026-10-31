"""(a) に分けた解析を，台本の出力だけから計算できることを示す（値の中身は出さず，検算の差と行の数だけ出す）。"""
import numpy as np, pandas as pd
from scipy import stats
A, B = "runA", "runB"
P = pd.read_csv(f"{A}/e75_1_pairs.csv", dtype={"K": str})
J = P.n_rep; tq = stats.t.ppf(.975, J - 1)
s = (P.naive_hi - P["mean"]) / tq * np.sqrt(J)                 # 素朴な区間の幅から標準偏差を戻す
lo = P["mean"] - tq * s * np.sqrt(1 / J + P.rho)
print("1) 補正の区間を出力だけで作り直した差の最大", float(np.abs(lo - P.corr_lo).max()), "→ ρ を 0.25 など別の値にして数え直せる")
# 2) 損に占める o の割合を，同時に取り戻せる量で作り直す（項目 6 × 項目 8）
p6 = pd.read_csv(f"{B}/e75_6_pairs.csv"); p8 = pd.read_csv(f"{A}/e75_8_pairs.csv")
m = p6.merge(p8, on=["kind", "target", "source"])
print("2) 項目 6 と 8 を突き合わせた行", len(m), "／", len(p6), "；位置の損の差の最大", float(np.abs(m.loc_loss_pairwise - m.L_crit).max()))
# 3) 項目 10 を同じコーパスの組を除いて作り直す
p10 = pd.read_csv(f"{A}/e75_10_pairs.csv", dtype={"K": str})
corp = lambda g: g.split("-")[0] if g.startswith("IEMOCAP") else g
ex = p10[[corp(t) != corp(s_) for t, s_ in zip(p10.target, p10.source)]]
print("3) 項目 10 の組の数（全 / 同じコーパスを除く）", p10.groupby(["kind", "K"]).size().unique(), ex.groupby(["kind", "K"]).size().unique())
# 4) 項目 5 の感情の対ごと（6 対）の内訳は出力に無い（平静を含むか否かの 2 群だけ）
n5 = pd.read_csv(f"{A}/e75_5_neutral_pairs.csv")
print("4) e75_5_neutral_pairs の列", list(n5.columns))
# 5) 項目 11 の図 11-1 の誤差の棒（6 対の平均の d の反復の標準偏差）は出力から出るか
r11 = pd.read_csv(f"{A}/e75_11_residuals.csv")
print("5) e75_11_residuals の列", list(r11.columns), "反復ごとの d の列は無い")
