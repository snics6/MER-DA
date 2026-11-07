"""dai-review-ml の DAI-8 の検査（人工データだけ）：正しい形のデータ（感情ごとに共通の共分散のガウス）でも，
台本の em_fit が初期値から離れて判定を悪くするか．α と次元と初期値を変える。
初期値は (i) 正解の one-hot（真の割り当て），(ii) 正解率の高い頭の確率，の 2 つ。
使い方：CUDA_VISIBLE_DEVICES="" .venv/bin/python DAI/checks/DAI-8/review_ml_run/em_drift_sim.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
import dai8_probe as P  # noqa: E402

DT = torch.float64
PROPS = np.array([0.37, 0.32, 0.21, 0.10])


def rec(y, p):
    return np.array([100 * (p[y == k] == k).mean() for k in range(4)])


def data(d, n, rng, sep, rc=20, rank=60, s_low=2.0):
    B = np.linalg.qr(rng.standard_normal((d, rc)))[0]
    M = rng.standard_normal((4, rc)) * sep
    L = rng.standard_normal((d, rank)) * s_low / np.sqrt(rank)

    def draw(m):
        y = rng.choice(4, m, p=PROPS)
        return M[y] @ B.T + rng.standard_normal((m, rank)) @ L.T + rng.standard_normal((m, d)), y
    Xp, yp = draw(n)
    Xt, yt = draw(3000)
    mu, sd = Xp.mean(0), Xp.std(0)
    return (torch.tensor((Xp - mu) / sd, dtype=DT), yp, torch.tensor((Xt - mu) / sd, dtype=DT), yt)


def run(d, n, seed, sep, alpha):
    rng = np.random.default_rng(seed)
    Xp, yp, Xt, yt = data(d, n, rng, sep)
    yp_t = torch.tensor(yp)
    oh = torch.nn.functional.one_hot(yp_t, 4).to(DT)
    XtX = Xp.T @ Xp
    A, c, _ = P.lda_from_R(Xp, oh, XtX, alpha)
    out = dict(d=d, n=n, alpha=alpha, ldaO=round(rec(yt, P.delta(Xt, A, c).argmax(1).numpy()).mean(), 1))
    # 初期値 (ii)：全ラベルの LDA の候補の上の確率を温度 3 で鈍らせたもの（正解率は高い）
    q_good = torch.softmax(P.delta(Xp, A, c) / 3.0, -1)
    out["q_good_pool"] = round(rec(yp, q_good.argmax(1).numpy()).mean(), 1)
    kpos = torch.tensor(np.sort(np.concatenate([rng.permutation(np.flatnonzero(yp == k))[:5] for k in range(4)])))
    for nm, q in (("truth", oh.clone().clamp(1e-6, 1)), ("good", q_good)):
        q = q / q.sum(1, keepdim=True)
        for beta in (0.0, 1.0):
            A2, c2, pi, R, hist = P.em_fit(Xp, XtX, kpos, yp_t[kpos], q, alpha, beta)
            out[f"{nm}_b{int(beta)}"] = round(rec(yt, P.delta(Xt, A2, c2).argmax(1).numpy()).mean(), 1)
            out[f"{nm}_b{int(beta)}_pool"] = round(rec(yp, R.argmax(1).numpy()).mean(), 1)
            out[f"{nm}_b{int(beta)}_pi_min"] = round(float(pi[3]), 3)
    return out


if __name__ == "__main__":
    for d, sep in ((128, 0.5), (2304, 0.5)):
        for alpha in (0.1, 0.5, 0.9):
            print(run(d, 2350, 0, sep, alpha), flush=True)
