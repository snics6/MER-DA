"""dai-review-ml の DAI-8 の再検査（人工データだけ）：第 3 版の台本の em_fit・lda_from_R をそのまま呼び，
R9（流れた理由の読み分け．台本 summarize の M1_reading と同じ式）がどの読みを出すかを，テストの UAR と並べる。
データは em_drift_sim.py と同じ作り（感情ごとに共通の共分散のガウス．形は正しい）。
使い方：CUDA_VISIBLE_DEVICES="" .venv/bin/python DAI/checks/DAI-8/review_ml_run/r9_reading_sim.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "exp"))
import dai8_probe as P  # noqa: E402   第 3 版（コミット 1bde2d5）
import em_drift_sim as D  # noqa: E402

DT = torch.float64


def uar(y, p):
    return 100 * np.mean([(p[y == k] == k).mean() for k in range(4)])


def run(d, n, seed, sep, props):
    D.PROPS = np.array(props)
    rng = np.random.default_rng(seed)
    Xp, yp, Xt, yt = D.data(d, n, rng, sep)
    XtX = Xp.T @ Xp
    oh = torch.nn.functional.one_hot(torch.tensor(yp), 4).to(DT)
    a = P.FIX[0]
    fO = P.lda_from_R(Xp, oh, XtX, a)
    out = dict(d=d, n=n, seed=seed, ldaO_test=round(uar(yt, P.delta(Xt, fO).argmax(1).numpy()), 1),
               ldaO_pool_insample=round(uar(yp, P.delta(Xp, fO).argmax(1).numpy()), 1))
    empty = torch.zeros(0, dtype=torch.long)
    yr = np.random.default_rng(P.SEED).permutation(yp)
    for tag, lab in (("truth", yp), ("rand", yr)):
        q = torch.nn.functional.one_hot(torch.tensor(lab), 4).to(DT)
        res = P.em_fit(Xp, XtX, empty, empty, q, a, 0.0, n_iter=P.MAX_EM, tol=P.TOL_EM, snaps=(P.N_EM,), yU=yp)
        prc = res["R"].argmax(1).numpy()
        out[f"{tag}_keep"] = round(float((prc == lab).mean()), 3)
        out[f"{tag}_pool_uar"] = round(uar(yp, prc), 1)
        out[f"{tag}_test"] = round(uar(yt, P.delta(Xt, res["fit"]).argmax(1).numpy()), 1)
        out[f"{tag}_iters"] = res["info"]["n_iter_done"]
    if out["rand_keep"] >= P.KEEP_RAND:
        out["R9"] = "c_keeps_any_init"
    elif (out["truth_pool_uar"] - 25) < 0.5 * (out["ldaO_pool_insample"] - 25):
        out["R9"] = "b_optimum_away_from_emotion"
    else:
        out["R9"] = "a_init_or_other"
    return out


if __name__ == "__main__":
    for d, n, props in ((2304, 632, (0.26, 0.27, 0.22, 0.25)), (2304, 1200, (0.26, 0.27, 0.22, 0.25)),
                        (2304, 2350, (0.37, 0.32, 0.21, 0.10)), (2304, 8500, (0.57, 0.20, 0.09, 0.14))):
        for seed in (0, 1):
            print(run(d, n, seed, 0.5, props), flush=True)
