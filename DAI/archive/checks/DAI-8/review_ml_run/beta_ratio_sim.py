"""dai-review-ml の DAI-8 の検査（人工データだけ）：第 2 版の 3.3 節の「項の大きさの比」（1 回目の E 段の
max−min δ の中央値 ÷ max−min log q の中央値）と，β log q を外すと E 段の argmax が変わる割合（直接の量）を並べる。
データの作りは em_selfterm_sim.py と同じ。
使い方：CUDA_VISIBLE_DEVICES="" .venv/bin/python DAI/checks/DAI-8/review_ml_run/beta_ratio_sim.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import em_selfterm_sim as E  # noqa: E402
P = E.P
DT = torch.float64


def one(name, seed, sep=0.55, alpha=0.5, K=5, d=2304):
    n_pool, props = E.CONF[name]
    rng = np.random.default_rng(seed)
    gen = E.Gen(d, rng, sep)
    Xp, yp = gen.sample(n_pool, rng, props, 16)
    mu, sd = Xp.mean(0), Xp.std(0)
    Xp_t = torch.tensor((Xp - mu) / sd, dtype=DT)
    yp_t = torch.tensor(yp)
    kpos = np.sort(np.concatenate([rng.permutation(np.flatnonzero(yp == k))[:K] for k in range(4)]))
    kpos_t = torch.tensor(kpos)
    U = np.ones(n_pool, bool); U[kpos] = False
    Ut = torch.tensor(U)
    Xs, ys = gen.sample(6000, rng, (0.25, 0.25, 0.25, 0.25), 16, src=E.SRC)
    Ws, _ = E.fit_head(torch.tensor((Xs - mu) / sd, dtype=DT), torch.tensor(ys), 1e-3, torch.zeros(d, 4, dtype=DT))
    W, b = E.fit_head(Xp_t[kpos_t], yp_t[kpos_t], E.LAM_K, Ws)
    q = torch.softmax(Xp_t @ W + b, -1)
    R = q.clone(); R[kpos_t] = 0.0; R[kpos_t, yp_t[kpos_t]] = 1.0
    A, c, pi = P.lda_from_R(Xp_t, R, Xp_t.T @ Xp_t, alpha)               # 1 回目の M 段
    dl = P.delta(Xp_t[Ut], A, c)
    lq = torch.log(q[Ut].clamp(min=1e-12))
    ratio = float((dl.max(1).values - dl.min(1).values).median() / (lq.max(1).values - lq.min(1).values).median())
    base = torch.log(pi) + dl
    flip = 100 * float(((base + lq).argmax(1) != base.argmax(1)).double().mean())
    return dict(conf=name, seed=seed, n_pool=n_pool, ratio_v2=round(ratio, 2), beta_flip_first_E_pct=round(flip, 2))


if __name__ == "__main__":
    for name in E.CONF:
        for seed in (0, 1, 2):
            print(one(name, seed), flush=True)
