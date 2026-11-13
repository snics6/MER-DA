import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/snics/MER-DA/DAI/checks/DAI-18/review_ml_sim")
import v1_copy as V
# 単位ごとのソースの数
cnt = {}
for T, c, f, r in V.all_units(folds=[0], seeds=[0]):
    for k in V.kinds_of(c):
        s = [x for x in V.CORP6 if x not in (T, c) and set(k) <= set(V.MODS_OF[x])]
        cnt.setdefault(k, set()).add(len(s))
print("ソースの数", {k: sorted(v) for k, v in cnt.items()})
for c in ("CREMA-D", "RAVDESS"):
    cp = V.C6.get(c)
    df = cp.df
    print(c, len(df), "塊", df.cluster.nunique(), "g_eff", round(V.g_eff_of(c), 1), "ラベル", np.bincount(cp.y).tolist(), "分割", df.fold.value_counts().sort_index().tolist())
    for f in range(5):
        pool = np.flatnonzero(V.DD.pool_mask(df, f))
        print("  分割", f, "候補の感情ごとの最小", np.bincount(cp.y[pool], minlength=4).min(), "塊の数（テスト）", df[df.fold == f].cluster.nunique())
    print("  speaker_id と cluster が同じか", (df.speaker_id.astype(str) == df.cluster).mean(), df.cluster.unique()[:5])
