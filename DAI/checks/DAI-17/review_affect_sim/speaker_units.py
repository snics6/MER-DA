"""感情認識の査読役の検算（DAI-17 第 1 版）：発話一覧（DAI/splits/*.csv）の塊の列だけを使う．
台本の distinct_speakers_top20（上位 20 本の中の別々の「話者」の数）は，dai_data の cluster（塊）で数える．
ソースの組ごとに，表からでたらめに 20 本を引いたときの別々の塊の数の期待値を出す（偏りを読むための基準）．
"""
import itertools
import numpy as np
import pandas as pd

ROOT = "/home/snics/MER-DA"
C = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
sp = {c: pd.read_csv(f"{ROOT}/DAI/splits/{c}.csv", dtype=str, keep_default_na=False) for c in C}
for c in C:
    vc = sp[c].cluster.value_counts()
    print(c, "発話", len(sp[c]), "塊", len(vc), "最大の塊の割合", round(vc.iloc[0] / len(sp[c]), 3), "上位 6 塊の割合", round(vc.iloc[:6].sum() / len(sp[c]), 3))
rows = []
for srcs in itertools.combinations(C, 2):
    n = np.concatenate([sp[s].cluster.value_counts().to_numpy() for s in srcs]).astype(float)
    p = n / n.sum()
    rows.append(dict(srcs="+".join(srcs), n_utt=int(n.sum()), n_clusters=len(n), expect_distinct_in_20=float((1 - (1 - p) ** 20).sum())))
    for s in srcs:
        q = sp[s].cluster.value_counts(normalize=True).to_numpy()
        rows.append(dict(srcs=f"  {s} だけ", n_utt=len(sp[s]), n_clusters=len(q), expect_distinct_in_20=float((1 - (1 - q) ** 20).sum())))
d = pd.DataFrame(rows).drop_duplicates("srcs").round(2)
print(d.to_string(index=False))
d.to_csv(f"{ROOT}/DAI/checks/DAI-17/review_affect_sim/speaker_units.csv", index=False)
