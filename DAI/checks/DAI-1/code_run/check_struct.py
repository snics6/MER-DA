"""gate-code: 分割・K 本の構造の検査（CPU．ラベルの成績は使わない）。"""
import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd
import dai_data as DD
for c in DD.CORPORA:
    df, feats = DD.load_corpus(c)
    y = df["label_4"].to_numpy().astype(int)
    print(f"== {c}: 発話 {len(df)}，塊 {df.cluster.nunique()}，話者 {df.speaker_id.nunique()}，次元",
          {m: feats[m].shape[1] for m in feats}, "NaN", int(sum(np.isnan(v).sum() for v in feats.values())))
    sp = pd.read_csv(f"DAI/splits/{c}.csv")
    print("  splits 表との一致：sample_id", bool((sp.sample_id.astype(str).values == df.sample_id.values).all()),
          "fold", bool((sp.fold.values == df.fold.values).all()))
    for f in range(DD.N_FOLDS):
        pm = DD.pool_mask(df, f); te = df.fold.values == f
        pool = np.flatnonzero(pm)
        ov_cl = len(set(df.cluster[pm]) & set(df.cluster[te]))
        ov_sp = len(set(df.speaker_id[pm].astype(str)) & set(df.speaker_id[te].astype(str)))
        ov_dl = len((set(df.dialogue[pm]) & set(df.dialogue[te])) - {""})
        excl = int((~pm & ~te).sum())
        minc = np.bincount(y[pool], minlength=4).min()
        Kd = {}
        for r in range(10):
            ko = DD.k_orders(y, pool, c, f, r)
            assert all(np.isin(ko[K], pool).all() for K in DD.KS)
            assert all((np.bincount(y[ko[K]], minlength=4) == K).all() for K in DD.KS)
            assert all(np.isin(ko[a], ko[b]).all() for a, b in zip(DD.KS[:-1], DD.KS[1:]))
            Kd[r] = tuple(ko[25])
        print(f"  f{f}: テスト {te.sum()}，候補 {len(pool)}，除いた {excl}，塊の重なり {ov_cl}，話者の重なり {ov_sp}，対話の重なり {ov_dl}，"
              f"候補の感情ごとの最少 {minc}，種 0〜9 の K=25 の組の異なり {len(set(Kd.values()))}")
    if c == "MELD":
        top = df.speaker_id.value_counts().head(8)
        print("  MELD の話者の上位と，その話者の塊の数：", {s: df.loc[df.speaker_id == s, "cluster"].nunique() for s in top.index})
