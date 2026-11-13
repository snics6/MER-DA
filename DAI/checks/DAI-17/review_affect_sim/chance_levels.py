"""感情認識の査読役の検算（DAI-17 第 1 版）：発話一覧（DAI/splits/*.csv）の感情の割合だけを使う．特徴も結果も読まない．
設計 3.7 節・4 節の D3 の「でたらめに引いたときの当たる割合」（テストの感情の割合と表のラベルの割合の内積）が，
引いた発話のラベルの割合 q の動きだけでどれだけ変わるかを出す．
"""
import itertools
import numpy as np
import pandas as pd

ROOT = "/home/snics/MER-DA"
C = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
sp = {c: pd.read_csv(f"{ROOT}/DAI/splits/{c}.csv") for c in C}
cnt = {c: np.bincount(sp[c].label_4.astype(int), minlength=4).astype(float) for c in C}
p = {c: cnt[c] / cnt[c].sum() for c in C}
rows = []
for c in C:
    for srcs in itertools.combinations([x for x in C if x != c], 2):
        pdb = (cnt[srcs[0]] + cnt[srcs[1]]) / (cnt[srcs[0]].sum() + cnt[srcs[1]].sum())
        for f in range(5):
            y = sp[c].label_4[sp[c].fold == f].astype(int).to_numpy()
            pt = np.bincount(y, minlength=4) / len(y)
            ch = float(pt @ pdb)
            # 引いた発話が全部ソース s から来たとき（発話ごとの感情の情報は 0）
            only = [float(pt @ p[s]) for s in srcs]
            # 引いた発話が全部 Neutral のとき／全部 Happy のとき（同上）
            rows.append(dict(c=c, srcs="+".join(srcs), fold=f, chance=100 * ch, d3_all_src0=100 * (only[0] - ch), d3_all_src1=100 * (only[1] - ch),
                             d3_all_neutral=100 * (pt[0] - ch), d3_q_neutral_x2=100 * (float(pt @ (lambda q: q / q.sum())(pdb * np.array([2, 1, 1, 1.0]))) - ch),
                             macro_null=25.0))
d = pd.DataFrame(rows)
g = d.groupby(["c", "srcs"]).mean(numeric_only=True).drop(columns="fold").round(2)
print(g.to_string())
print()
print("c ごとの平均（3 つのソースの組 × 5 分割）")
print(d.groupby("c").mean(numeric_only=True).drop(columns="fold").round(2).to_string())
g.to_csv(f"{ROOT}/DAI/checks/DAI-17/review_affect_sim/chance_levels.csv")
