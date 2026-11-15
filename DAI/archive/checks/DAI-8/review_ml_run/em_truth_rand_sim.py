"""dai-review-ml の DAI-8 の検査（人工データだけ）：候補の正解から始める EM と，同じ割合ででたらめに並べ替えた
ラベルから始める EM で，最後の割り当てが初期値と同じ割合（残った割合）と，テストの UAR を比べる。
データの作りは em_drift_sim.py と同じ（感情ごとに共通の共分散のガウス．形は正しい）。
使い方：CUDA_VISIBLE_DEVICES="" .venv/bin/python DAI/checks/DAI-8/review_ml_run/em_truth_rand_sim.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import em_drift_sim as D  # noqa: E402
P = D.P
DT = torch.float64


def run(d, n, seed, sep, alpha):
    rng = np.random.default_rng(seed)
    Xp, yp, Xt, yt = D.data(d, n, rng, sep)
    XtX = Xp.T @ Xp
    oh = torch.nn.functional.one_hot(torch.tensor(yp), 4).to(DT)
    A, c, _ = P.lda_from_R(Xp, oh, XtX, alpha)
    out = dict(d=d, n=n, alpha=alpha, ldaO=round(D.rec(yt, P.delta(Xt, A, c).argmax(1).numpy()).mean(), 1))
    yr = rng.permutation(yp)
    for nm, y0 in (("truth", yp), ("rand", yr)):
        q = torch.nn.functional.one_hot(torch.tensor(y0), 4).to(DT).clamp(1e-6, 1)
        q = q / q.sum(1, keepdim=True)
        A2, c2, pi, R, hist = P.em_fit(Xp, XtX, torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), q, alpha, 0.0)
        out[f"{nm}_test_uar"] = round(D.rec(yt, P.delta(Xt, A2, c2).argmax(1).numpy()).mean(), 1)
        out[f"{nm}_kept_pct"] = round(100 * float((R.argmax(1).numpy() == y0).mean()), 1)
    return out


if __name__ == "__main__":
    for d, n in ((2304, 632), (2304, 2350), (2304, 8500), (128, 2350)):
        for alpha in (0.1, 0.5):
            print(run(d, n, 0, 0.5, alpha), flush=True)
