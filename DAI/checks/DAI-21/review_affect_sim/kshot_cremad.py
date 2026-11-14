"""感情認識の査読者の検算（DAI-21）：CREMA-D の K 本の中身（話者，文）を，分割 5 × 種 10 で数える．CPU だけ．UAR は計算しない．
実験 18 の分割の表（results/dai18_probe/splits_new.csv）と，台本と同じ乱数の作り方（features/common.rng_for("dai-K", 4, 分割, 種)）を使う．"""
import sys
import numpy as np
import pandas as pd

ROOT = "/home/snics/MER-DA"
sys.path.insert(0, ROOT + "/features")
import common  # noqa: E402

sp = pd.read_csv(ROOT + "/results/dai18_probe/splits_new.csv", dtype=str, keep_default_na=False)
sp = sp[sp.corpus == "CREMA-D"].reset_index(drop=True)
assert list(sp.sample_id) == sorted(sp.sample_id)
y = sp.label_4.astype(int).to_numpy()
fold = sp.fold.astype(int).to_numpy()
spk = sp.speaker_id.to_numpy()
sent = sp.sample_id.str.split("_").str[1].to_numpy()
KS = (1, 5, 10, 25)

def korder(f, r):
    pool = np.flatnonzero(fold != f)
    rng = common.rng_for("dai-K", 4, f, r)
    perm = {k: rng.permutation(pool[y[pool] == k]) for k in range(4)}
    return pool, {K: np.sort(np.concatenate([perm[k][:K] for k in range(4)])) for K in KS}

# 実験 18 の保存（RAVDESS を除いた単位．分割 0・種 0）の K 本と同じになるか
z = np.load(ROOT + "/results/dai18_probe/units/RAVDESS_CREMA-D_f0_r0.npz", allow_pickle=False)
pool, ko = korder(0, 0)
for K in (5, 10, 25):
    assert (np.searchsorted(pool, ko[K]) == z[f"kpos_K{K}"]).all(), K
assert (sp.sample_id.to_numpy()[fold == 0] == z["sid_test"].astype(str)).all()
print("実験 18 の保存の K 本・テストの並びと一致（分割 0・種 0）")

rows = []
for f in range(5):
    test_spk = set(spk[fold == f])
    for r in range(10):
        pool, ko = korder(f, r)
        for K in (5, 10, 25):
            ix = ko[K]
            assert not (set(spk[ix]) & test_spk)
            ieo = sent[ix] == "IEO"
            rows.append(dict(f=f, r=r, K=K, n_spk=len(set(spk[ix])), n_sent=len(set(sent[ix])),
                             ieo_neu=int(ieo[y[ix] == 0].sum()), ieo_other=int(ieo[y[ix] != 0].sum()),
                             max_per_spk=int(pd.Series(spk[ix]).value_counts().max()),
                             min_sent_per_emo=min(len(set(sent[ix][y[ix] == k])) for k in range(4))))
d = pd.DataFrame(rows)
print("単位の数", len(d) // 3, "（K ごと）．テストの話者が K 本に入る単位：0")
g = d.groupby("K").agg(n_spk_min=("n_spk", "min"), n_spk_mean=("n_spk", "mean"), n_sent_min=("n_sent", "min"), n_sent_mean=("n_sent", "mean"),
                       ieo_neu_mean=("ieo_neu", "mean"), ieo_other_mean=("ieo_other", "mean"), max_per_spk=("max_per_spk", "max"),
                       min_sent_per_emo=("min_sent_per_emo", "min"))
print(g.round(2).to_string())
for K in (5, 10, 25):
    q = d[d.K == K]
    print(f"K={K}：K 本のうち IEO の割合  Neutral {q.ieo_neu.sum() / (K * len(q)):.3f}，ほかの 3 感情 {q.ieo_other.sum() / (3 * K * len(q)):.3f}．"
          f"IEO が Neutral に 1 本も無い単位 {int((q.ieo_neu == 0).sum())}/{len(q)}，ほかの 3 感情に 1 本も無い単位 {int((q.ieo_other == 0).sum())}/{len(q)}")
# テスト：分割ごとの話者，感情の件数，IEO を除いた件数
for f in range(5):
    te = fold == f
    print(f"分割 {f}：テストの話者 {len(set(spk[te]))}，発話 {te.sum()}，感情ごと {np.bincount(y[te], minlength=4).tolist()}，"
          f"IEO を除く {np.bincount(y[te & (sent != 'IEO')], minlength=4).tolist()}")
