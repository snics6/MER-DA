"""ML の査読者の検査（DAI-7）：教師を使わない場合（各モダリティが自分の特徴の共有の向きを拾うだけ）に，
誤りの相関 φ（設計 3.4 節）と D がどう動くかを人工データで確かめる．実データは読まない．
partial_t_sim.py の場合 C と同じ作り（教師の因子は平均だけ，共有の大きさ 0.6）．φ は台本の phi_err をそのまま使う．"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dai7_probe_v2 as P  # noqa: E402
import partial_t_sim as Q  # noqa: E402


def run(seed, gamma, sep=0.6, ks=2, p=10, n_pool=1500, n_test=400, K=5, gscale=0.6):
    rng = np.random.default_rng(seed)
    mu = {m: sep * rng.normal(size=(4, p)) for m in "AVT"}
    G = {m: gscale * rng.normal(size=(ks, p)) for m in "AVT"}
    Xp, yp = Q.make(rng, n_pool, mu, G, ks, p, True)
    Xt, yt = Q.make(rng, n_test, mu, G, ks, p, True)
    kix = np.concatenate([np.flatnonzero(yp == e)[:K] for e in range(4)])
    none4, drift4 = {}, {}
    for m in "AVT":
        muk = np.stack([Xp[m][kix][yp[kix] == e].mean(0) for e in range(4)])
        Z = Xt[m] @ muk.T - 0.5 * np.sum(muk ** 2, 1)
        Gm = G[m] / np.linalg.norm(G[m], axis=1, keepdims=True)
        proj = Xt[m] @ Gm.T
        M = rng.normal(size=(ks, 4))
        add = proj @ M
        none4[m] = Z
        drift4[m] = Z + gamma * add * np.std(Z) / max(np.std(add), 1e-9)
    res = {}
    for nm, Z4 in (("none", none4), ("drift", drift4)):
        z = {m: Z4[m] @ P.H for m in "AVT"}
        D = Q.Dpair(z, yt, np.random.default_rng(7))
        ph = np.mean([P.phi_err(Z4[a].argmax(1), Z4[b].argmax(1), yt)[0] for a, b in Q.PAIRS])
        u = np.mean([P.uar(yt, Z4[m].argmax(1)) for m in "AVT"])
        res[nm] = (D, ph, u)
    return res


def main():
    n_both = n_d = 0
    for seed in range(10):
        for gamma in (0.5, 1.0):
            r = run(seed, gamma)
            dD = r["drift"][0] - r["none"][0]
            dphi = r["drift"][1] - r["none"][1]
            if dD >= 0.02:
                n_d += 1
                n_both += int(dphi >= 0.02)
            if seed < 4:
                print(f"seed {seed} γ={gamma}: ΔD={dD:+.3f} Δφ={dphi:+.3f}  単独の UAR の平均 none={r['none'][2]:.1f} drift={r['drift'][2]:.1f}")
    print(f"教師なしの場合 C：ΔD ≥ 0.02 の {n_d} 回のうち Δφ ≥ 0.02 も満たしたのは {n_both} 回")


if __name__ == "__main__":
    main()
