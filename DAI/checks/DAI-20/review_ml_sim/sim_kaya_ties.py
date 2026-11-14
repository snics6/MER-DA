"""ML の査読者の検算（DAI-20．人工の数だけ）．K 本の上の UAR が同点の表からの選び方で，テストの UAR がどれだけ変わるか．
(a) 番号の小さい表（設計の規則），(b) 等しい重みに最も近い表，(c) 同点の表の平均．表は全単位で共通の 1,000 個（列の和は 3 にそろえた）．"""
import numpy as np
from sim_kaya_scale import softmax, centered, uar, make

STRENGTH = np.array([[1.6, 1.0, 1.8, 1.4], [0.8, 1.6, 0.6, 1.0], [1.4, 1.2, 1.0, 1.8]])
PRIOR = np.array([0.55, 0.20, 0.10, 0.15])
TAB = np.random.default_rng(7).dirichlet(np.ones(3), size=(1000, 4)).transpose(0, 2, 1) * 3.0
TAB[0] = 1.0


def unit(seed, K):
    rng = np.random.default_rng(5000 + seed)
    bias = rng.normal(scale=0.5, size=(3, 4))
    yk = np.repeat(np.arange(4), K)
    Lk = []
    for k in range(3):
        Z = rng.normal(size=(4 * K, 4)) + bias[k]
        Z[np.arange(4 * K), yk] += STRENGTH[k, yk]
        Lk.append(centered(softmax(Z)))
    Lk = np.stack(Lk)
    yt, Pt = make(rng, 3000, PRIOR, STRENGTH, bias)
    Lt = np.stack([centered(p) for p in Pt])
    uk = np.array([uar(yk, np.einsum("ky,kny->ny", w, Lk).argmax(1)) for w in TAB])
    tied = np.flatnonzero(uk >= uk.max() - 1e-9)
    ut = lambda w: uar(yt, np.einsum("ky,kny->ny", w, Lt).argmax(1))
    a = 0 if 0 in tied else int(tied[0])
    b = int(tied[np.argmin([((TAB[i] - 1.0) ** 2).sum() for i in tied])])
    return ut(TAB[0]), ut(TAB[a]), ut(TAB[b]), ut(TAB[tied].mean(0)), a


for K in (5, 10, 25):
    r = np.array([unit(s, K) for s in range(300)])
    top = np.bincount(r[:, 4].astype(int)).argsort()[::-1][:3]
    print(f"K={K}：テストの UAR  等しい重み {r[:, 0].mean():.2f}，(a) 番号の小さい表 {r[:, 1].mean():.2f}，(b) 等しい重みに最も近い表 {r[:, 2].mean():.2f}，"
          f"(c) 同点の平均 {r[:, 3].mean():.2f}．(a) で選ばれた表の番号が 20 未満の単位の割合 {np.mean(r[:, 4] < 20):.2f}，よく選ばれた番号 {top.tolist()}")
