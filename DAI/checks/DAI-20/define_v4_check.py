"""関門 1（定義）第 4 版の再検査の確かめ（読むだけ．人工の数．CPU で 1 秒）．
1. 格子の点 14（n0 = ∞，β = 1，戻し先 eq）の重みが K 本の水準の平均 L̄_K に依るか．
2. prop_noint − prop_noprof の重みの差に，列の差（感情ごとの倍率）以外の成分（λ の項，行の差）が残るか．
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp"))
import dai16_probe as P  # noqa: E402

rng = np.random.default_rng(0)
i14 = P.GRID.index((float("inf"), 1.0, "eq"))
print("点 14 =", P.GRID[i14])

# 1. 同じ相性（Dext）で，K 本の水準 DK の全体の大きさだけを変えたときに点 14 の重みが変わるか
Dext = rng.uniform(0.2, 2.0, size=(3, 4))
for scale in (0.3, 1.0, 3.0):
    DK = scale * rng.uniform(0.5, 1.5, size=(3, 4))
    W = P.to_w(P.mix_rel(DK, Dext, 20, float("inf"), 1.0, "eq"))
    Lbar = DK.mean()
    W_pred = np.maximum(0.0, Lbar + (Dext - Dext.mean(1, keepdims=True)))
    W_pred = W_pred / W_pred.mean()
    print(f"L̄_K={Lbar:.3f}  重みの範囲 {W.min():.3f}〜{W.max():.3f}  1+相性/L̄_K（切り捨てと割り直し込み）との差の最大 {np.abs(W - W_pred).max():.1e}")

# 2. prop_noint − prop_noprof の分け
n_lam, n_row, n_clip = 0, 0, 0
lam_share = []
for t in range(500):
    DK = rng.uniform(0.0, 2.0, size=(3, 4))
    Dext = rng.uniform(0.0, 2.0, size=(3, 4))
    for n0 in (0.0, 20.0):
        Dp = P.mix_rel(DK, Dext, 20, n0, 1.0, "eq")
        Wp = P.to_w(Dp)
        Wn = P.noint(Wp)
        Wq = P.to_w(P.mix_rel(DK, Dext, 20, n0, 0.0, "eq"))
        g = Wp.mean(); a = Wp.mean(1) - g; b = Wp.mean(0) - g
        den = (a ** 2).sum() * (b ** 2).sum()
        lam = (a[:, None] * b[None, :] * Wp).sum() / den if den > 1e-12 else 0.0
        diff = Wn - Wq
        col = b[None, :] * np.ones((3, 1))
        lam_part = lam * a[:, None] * b[None, :]
        rest = diff - col - lam_part
        n_lam += np.abs(lam_part).max() > 1e-6
        n_row += np.abs(rest).max() > 1e-6
        n_clip += bool((Dp <= 0).any() or (Wn <= 0).any())
        lam_share.append(np.abs(lam_part).sum() / max(np.abs(diff).sum(), 1e-12))
print(f"2. 1,000 組のうち λ の項が 0 でない組 {n_lam}，列の差と λ の項の外の残り（行の差．切り捨てで出る）が 0 でない組 {n_row}，切り捨てが起きた組 {n_clip}")
print(f"   差の絶対値の和に占める λ の項の割合：中央 {np.median(lam_share):.2f}，90% 点 {np.quantile(lam_share, 0.9):.2f}")
