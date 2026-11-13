"""感情認識の査読役の検算（DAI-17 第 3 版）：テストの分割ごとの感情の件数（4 感情がどの分割にもあるか）．
rank_bal（感情ごとの当たる割合の平均）は，テストに無い感情を除いて平均するので，欠ける感情があると「無関係なら 25%」が崩れる．"""
import pandas as pd
ROOT = "/home/snics/MER-DA"
for c in ("AFEW", "IEMOCAP", "MOSEI", "MELD"):
    d = pd.read_csv(f"{ROOT}/DAI/splits/{c}.csv", dtype={"label_4": int}, keep_default_na=False)
    t = pd.crosstab(d.fold, d.label_4)
    print(c, "発話", len(d)); print(t.to_string()); print("最小", int(t.values.min()))
