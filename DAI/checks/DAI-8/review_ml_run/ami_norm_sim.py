"""dai-review-ml の DAI-8 の第 3 回の検査（人工のラベルだけ）：R10 の AMI の比べで，組の数の多いラベル（グループ・対話）と
4 つの感情を同じ AMI（scikit-learn の既定．正規化は 2 つのエントロピーの算術平均）で比べると，割り当てがグループだけで
決まる場合でもグループの AMI が小さく出るかを見る。正規化を min（割り当ての側のエントロピー）にした場合と並べる。
使い方：.venv/bin/python DAI/checks/DAI-8/review_ml_run/ami_norm_sim.py
"""
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as AMI

rng = np.random.default_rng(0)
for n, G in ((632, 51), (2350, 8), (15000, 2000)):
    sizes = rng.zipf(1.6, G).astype(float)
    grp = rng.choice(G, n, p=sizes / sizes.sum())
    emo = rng.choice(4, n, p=(0.37, 0.32, 0.21, 0.10))
    g2c = rng.integers(0, 4, G)
    part_grp = g2c[grp]                                                      # 割り当てがグループだけで決まる
    noise = rng.random(n) < 0.5
    part_emo = np.where(noise, rng.integers(0, 4, n), emo)                   # 割り当ての半分が感情に従い，残りはでたらめ
    for nm, part in (("グループだけで決まる割り当て", part_grp), ("半分が感情に従う割り当て", part_emo)):
        out = {f"{lab}_{m}": round(AMI(L, part, average_method=m), 3)
               for lab, L in (("emo", emo), ("grp", grp)) for m in ("arithmetic", "min")}
        print(dict(n=n, groups=len(np.unique(grp)), case=nm, **out), flush=True)

# 混ざった場合：発話ごとに確率 w でグループの写像，1 − w で感情に従う（どちらが割り当てを多く決めるかを w で変える）
print("--- 混ざった場合 ---")
for n, G in ((632, 51), (15000, 2000)):
    rng = np.random.default_rng(1)
    sizes = rng.zipf(1.6, G).astype(float)
    grp = rng.choice(G, n, p=sizes / sizes.sum())
    emo = rng.choice(4, n, p=(0.37, 0.32, 0.21, 0.10))
    g2c = rng.integers(0, 4, G)
    for w in (0.5, 0.6, 0.7):
        part = np.where(rng.random(n) < w, g2c[grp], emo)
        out = {f"{lab}_{m}": round(AMI(L, part, average_method=m), 3)
               for lab, L in (("emo", emo), ("grp", grp)) for m in ("arithmetic", "min")}
        print(dict(n=n, groups=len(np.unique(grp)), w_group=w, **out), flush=True)
