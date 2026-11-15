"""ML の査読者の検査（DAI-7）：教師の点数を除いた依存 D_t（設計 3.3 節の副）の振る舞いを人工データで確かめる．

実データは読まない．台本 `DAI/exp/dai7_probe_v2.py` の d_with_null・residualize・H をそのまま使う．

作り：感情 e ごとに，3 つのモダリティの特徴 x_m = mu_m[e] + G_m s + u_m（s は 3 つで共有する潜在の量，u_m は各モダリティだけの雑音）．
- 疑似ラベルのモデルの因子 r_m：候補の全ラベルで学んだ判別分析の点数（各モダリティ別）．教師の点数 t = Σ_m r_m（H で 3 個）．
- none_m：K 本（感情ごとに 5 本）の平均だけから作った点数．
- 場合 0：s が無い（感情の中でモダリティは独立）．none の D と D_t を比べる．
- 場合 B（教師を写す）：生徒 m の点数 = 候補の上で t を x_m に回帰した予測（教師の写し）．
- 場合 C（教師を写さない）：生徒 m の点数 = none_m + γ·(x_m の共有の向きへの射影)．各モダリティが教師と関係なく共有の向きを拾う．
各場合で D（3 組の平均）と D_t を出し，設計 4 節の書き方 2 の比 ΔD_t / ΔD を出す．
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
import dai7_probe_v2 as P  # noqa: E402

PAIRS = (("A", "V"), ("A", "T"), ("V", "T"))


def make(rng, n_e, mu, G, ks, p, shared):
    y = np.repeat(np.arange(4), n_e)
    s = rng.normal(size=(len(y), ks)) if shared else np.zeros((len(y), ks))
    X = {m: mu[m][y] + s @ G[m] + rng.normal(size=(len(y), p)) for m in "AVT"}
    return X, y


def lda_scores(X, y, Xe):
    mu = np.stack([X[y == e].mean(0) for e in range(4)])
    R = X - mu[y]
    S = R.T @ R / len(X) + 1e-3 * np.eye(X.shape[1])
    W = np.linalg.solve(S, mu.T).T
    return Xe @ W.T - 0.5 * np.sum(W * mu, 1)


def Dpair(z, y, rng, resid_t=None):
    vals = []
    for a, b in PAIRS:
        za, zb = z[a], z[b]
        if resid_t is not None:
            za, zb = P.residualize(za, resid_t, y), P.residualize(zb, resid_t, y)
        d, d0, _ = P.d_with_null(za, zb, y, rng, n_perm=100)
        vals.append(np.nanmean(d - d0))
    return float(np.mean(vals))


def nm_scores(X, y, Xe):
    mu = np.stack([X[y == e].mean(0) for e in range(4)])
    return Xe @ mu.T


def one(seed, shared, gamma, sep=0.6, ks=2, p=10, n_pool=1500, n_test=400, K=5, gscale=1.5, teacher="lda"):
    rng = np.random.default_rng(seed)
    mu = {m: sep * rng.normal(size=(4, p)) for m in "AVT"}
    G = {m: gscale * rng.normal(size=(ks, p)) for m in "AVT"}
    Xp, yp = make(rng, n_pool, mu, G, ks, p, shared)
    Xt, yt = make(rng, n_test, mu, G, ks, p, shared)
    # 疑似ラベルのモデルの因子（候補の全ラベルの判別分析）と教師の点数
    fsc = lda_scores if teacher == "lda" else nm_scores
    r_t = {m: fsc(Xp[m], yp, Xt[m]) @ P.H for m in "AVT"}
    r_p = {m: fsc(Xp[m], yp, Xp[m]) @ P.H for m in "AVT"}
    t_t, t_p = sum(r_t.values()), sum(r_p.values())
    # none：K 本の平均だけ（共分散は単位行列）
    kix = np.concatenate([np.flatnonzero(yp == e)[:K] for e in range(4)])
    none = {}
    for m in "AVT":
        muk = np.stack([Xp[m][kix][yp[kix] == e].mean(0) for e in range(4)])
        none[m] = (Xt[m] @ muk.T) @ P.H
    # 場合 B：教師を写す（候補の上で t を x_m に回帰）
    copy = {}
    for m in "AVT":
        A1 = np.c_[np.ones(len(yp)), Xp[m]]
        beta, *_ = np.linalg.lstsq(A1, t_p, rcond=None)
        copy[m] = np.c_[np.ones(len(yt)), Xt[m]] @ beta
    # 場合 C：教師を写さず，各モダリティが自分の特徴の共有の向きを拾う
    drift = {}
    for m in "AVT":
        Gm = G[m] / np.linalg.norm(G[m], axis=1, keepdims=True)       # 共有の向き（特徴の空間）
        proj = Xt[m] @ Gm.T                                              # (n, ks)
        M = rng.normal(size=(ks, 3))
        drift[m] = none[m] + gamma * proj @ M * np.std(none[m]) / max(np.std(proj @ M), 1e-9)
    rng2 = np.random.default_rng(seed + 1000)
    out = {}
    for nm, z in (("none", none), ("copy_B", copy), ("drift_C", drift), ("ret", r_t)):
        out[nm] = (Dpair(z, yt, np.random.default_rng(7)), Dpair(z, yt, np.random.default_rng(7), resid_t=t_t))
    return out


def main():
    print("場合 0：感情の中でモダリティは独立（s なし）．教師の因子は判別分析")
    for seed in range(3):
        o = one(seed, shared=False, gamma=0.0)
        print(f"  seed {seed}: D(none)={o['none'][0]:+.3f}  D_t(none)={o['none'][1]:+.3f}   D(ret)={o['ret'][0]:+.3f}  D_t(ret)={o['ret'][1]:+.3f}")
    for teacher, gs in (("lda", 1.5), ("nm", 0.6)):
        print(f"共有の潜在の量あり（教師の因子 {teacher}，共有の大きさ {gs}）")
        cnt = {"copy_B": [0, 0], "drift_C": [0, 0]}
        for seed in range(10):
            for gamma in (0.5, 1.0):
                o = one(seed, shared=True, gamma=gamma, gscale=gs, teacher=teacher)
                Dn, Dtn = o["none"]
                for nm in ("copy_B", "drift_C"):
                    if nm == "copy_B" and gamma == 1.0:
                        continue
                    Ds, Dts = o[nm]
                    dD, dDt = Ds - Dn, Dts - Dtn
                    if dD < 0.02:
                        rule = "ΔD < 0.02（数えない）"
                    else:
                        cnt[nm][1] += 1
                        w = dDt <= 0.5 * dD
                        cnt[nm][0] += int(w)
                        rule = "教師の写しと書く" if w else "書かない"
                    if seed < 3:
                        print(f"  seed {seed} {nm:8s} γ={gamma if nm=='drift_C' else '-'}: D(none)={Dn:+.3f} D_t(none)={Dtn:+.3f}  "
                              f"D={Ds:+.3f} D_t={Dts:+.3f}  ΔD={dD:+.3f} ΔD_t={dDt:+.3f} → {rule}")
        for nm, (a, b) in cnt.items():
            print(f"  10 種 × γ：{nm} で ΔD ≥ 0.02 の {b} 回のうち「教師の写し」の条件を満たしたのは {a} 回")


if __name__ == "__main__":
    main()
