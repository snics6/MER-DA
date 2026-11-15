"""dai-review-ml の DAI-9 の検査（人工データだけ．実データは読まない）：
感情でないまとまりがまったく無いデータ（グループの平均の成分 0）でも，グループの中の感情の割合が偏っていると，
gc と nap が感情の成分を消して全ラベルの LDA を下げ，「予想の書き方」の 2 つの条件（raw が (b)，τ が (b) でない，
EM_truth^τ − LDA_O^τ が raw より 2 ポイント以上大きい）が満たされるかを見る。
台本 DAI/exp/dai9_probe.py の transform と，DAI/exp/dai8_probe.py の lda_from_R・em_fit・delta をそのまま呼ぶ。
使い方：CUDA_VISIBLE_DEVICES="" .venv/bin/python DAI/checks/DAI-9/review_ml_run/hyp_mech_sim.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
import dai9_probe as P9  # noqa: E402
P8 = P9.P8
DT = torch.float64
PROPS = np.array([0.37, 0.32, 0.21, 0.10])


def uar(y, p):
    return 100 * np.mean([(p[y == k] == k).mean() for k in range(4)])


def draw(n, n_grp, rng, M, B, L, Bg, s_grp, p_dom, g0):
    gsize = rng.poisson(n / n_grp, n_grp) + 1
    g = np.repeat(np.arange(n_grp), gsize)[:n]
    dom = rng.choice(4, n_grp, p=PROPS)
    y = np.where(rng.random(len(g)) < p_dom, dom[g], rng.choice(4, len(g), p=PROPS))
    G = rng.standard_normal((n_grp, Bg.shape[1])) * s_grp
    d = B.shape[0]
    x = (M[y] @ B.T + G[g] @ Bg.T + rng.standard_normal((len(g), L.shape[1])) @ L.T + rng.standard_normal((len(g), d)))
    return x, y, (g + g0).astype(str)


def run(seed, s_grp, p_dom, d=256, n=2350, sep=0.5, n_grp=300):
    rng = np.random.default_rng(seed)
    B = np.linalg.qr(rng.standard_normal((d, 20)))[0]
    M = rng.standard_normal((4, 20)) * sep
    L = rng.standard_normal((d, 60)) * 2.0 / np.sqrt(60)
    Bg = np.linalg.qr(rng.standard_normal((d, 30)))[0]
    Xp, yp, gp = draw(n, n_grp, rng, M, B, L, Bg, s_grp, p_dom, 0)
    Xt, yt, gt = draw(1500, 190, rng, M, B, L, Bg, s_grp, p_dom, 10000)
    mu, sd = Xp.mean(0), Xp.std(0)
    X0p, X0t = torch.tensor((Xp - mu) / sd, dtype=DT), torch.tensor((Xt - mu) / sd, dtype=DT)
    oh = torch.nn.functional.one_hot(torch.tensor(yp), 4).to(DT)
    yr = np.random.default_rng(0).permutation(yp)
    empty = torch.zeros(0, dtype=torch.long)
    out = {}
    for tau in ("raw", "gc", "nap10", "nap50"):
        Xa, Xb = P9.transform(tau, X0p, X0t, gp, gt)
        XtX = Xa.T @ Xa
        fO = P8.lda_from_R(Xa, oh, XtX, 0.5)
        r = dict(ldaO=uar(yt, P8.delta(Xb, fO).argmax(1).numpy()))
        keep = {}
        for tag, lab in (("truth", yp), ("rand", yr)):
            q = torch.nn.functional.one_hot(torch.tensor(lab), 4).to(DT)
            res = P8.em_fit(Xa, XtX, empty, empty, q, 0.5, 0.0, n_iter=100, tol=1e-4)
            keep[tag] = float((res["R"].argmax(1).numpy() == lab).mean())
            if tag == "truth":
                r["emtruth"] = uar(yt, P8.delta(Xb, res["fit"]).argmax(1).numpy())
        r["diff"] = r["emtruth"] - r["ldaO"]
        r["S2"] = "b" if r["diff"] <= -2 else ("c" if keep["rand"] >= 0.9 or keep["truth"] - keep["rand"] < 0.2 else "a")
        out[tau] = r
    hyp = [t for t in ("gc", "nap10", "nap50")
           if out["raw"]["S2"] == "b" and out[t]["S2"] != "b" and out[t]["diff"] >= out["raw"]["diff"] + 2]
    line = {t: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out[t].items()} for t in out}
    print(dict(seed=seed, sep=sep, s_grp=s_grp, p_dom=p_dom, HYP=hyp, **line), flush=True)


def sweep():
    """raw の EM_truth の落ち方が中くらいになる感情の離れ方で，同じ確かめをする（グループの成分は 0）。"""
    for sep in (0.7, 0.8, 0.9):
        for seed in (0, 1):
            run(seed, 0.0, 0.6, sep=sep)


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep()
    else:
        for s_grp, p_dom in ((0.0, 0.6), (0.0, 0.0), (1.0, 0.6)):
            for seed in (0, 1):
                run(seed, s_grp, p_dom)
