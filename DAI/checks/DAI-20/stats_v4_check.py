"""stats_v4 の確かめ：n0 = 0，β = 0 の提案の重み（F6 の道）と水準だけの重み（F2 の道）が浮動小数点で同じ判定になるか．
dai16_probe.py の mix_rel・to_w・combine・match_shift を同じ式で写した（読み込みはしない）．"""
import math
import numpy as np
from scipy.special import logsumexp

def norm_log(L): return L - logsumexp(L, axis=1, keepdims=True)
def to_w(D): m = float(D.mean()); return np.ones_like(D) if m <= 0 else D / m
def mix_rel(DK, De, n, n0, beta, prior):
    LK = DK.mean(1); L0 = np.full_like(LK, LK.mean()) if prior == "eq" else De.mean(1)
    w = 0.0 if math.isinf(n0) else n / (n + n0); L = w * LK + (1 - w) * L0
    return np.maximum(0.0, L[:, None] + beta * (De - De.mean(1, keepdims=True)))
def match_shift(Lp, t):
    d = np.zeros(4); lt = np.log(np.clip(t, 1e-12, None))
    for _ in range(3000):
        m = np.exp(norm_log(Lp + d)).mean(0)
        if np.abs(m - t).max() < 1e-8: break
        d = d + lt - np.log(np.clip(m, 1e-12, None)); d = d - d.mean()
    return d
rng = np.random.default_rng(0)
nflip_raw = nflip_adj = ntot = 0; wdiff = 0.0
for rep in range(200):
    DK, De = rng.uniform(0.0, 2.0, (3, 4)), rng.uniform(0.0, 2.0, (3, 4))
    def lm(n):
        P = rng.dirichlet(np.ones(4) * 0.8, size=(3, n)).astype(np.float32).astype(float)
        L = np.log(np.clip(P, 1e-12, None)); return L - L.mean(-1, keepdims=True)
    Lt, Lp = lm(3000), lm(2000)
    W6 = to_w(mix_rel(DK, De, 20, 0.0, 0.0, "eq")); w2 = to_w(DK).mean(1)
    wdiff = max(wdiff, float(np.abs(W6[:, 0] - w2).max()))
    s6t, s6p = norm_log(np.einsum("ky,kny->ny", W6, Lt)), norm_log(np.einsum("ky,kny->ny", W6, Lp))
    s2t, s2p = norm_log(np.tensordot(w2, Lt, axes=1)), norm_log(np.tensordot(w2, Lp, axes=1))
    tgt = np.exp(norm_log(Lp.sum(0))).mean(0)
    nflip_raw += int((s6t.argmax(1) != s2t.argmax(1)).sum())
    nflip_adj += int(((s6t + match_shift(s6p, tgt)).argmax(1) != (s2t + match_shift(s2p, tgt)).argmax(1)).sum())
    ntot += len(Lt[0])
print(f"重みの差の最大 {wdiff:.2e}，判定が違った発話 raw {nflip_raw}/{ntot}，adj {nflip_adj}/{ntot}")
