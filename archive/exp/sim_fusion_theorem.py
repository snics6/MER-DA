"""E73 の検算（CPU．人工データ）：見方を K に合わせた物差しの食い違いで重みづけて平均し，K 本で学び直すときの定理 1・補題 2 の確かめ。

2 次の模型（P = I の座標）：ターゲットの損失 R(θ) = R* + ½ (θ − θ*)ᵀ F (θ − θ*)，K 本の損失の 2 階微分 H（階数 < 次元）．
中心 c から学び直した解は θ(c) = θ* + (I + H)⁻¹ (c − θ* + ξ)，ラベルの揺れ ξ について平均した超過損失は ½ G_K(c) + 雑音の項，
G_K(c) = (c − θ*)ᵀ M_K (c − θ*)，M_K = (I + H)⁻¹ F (I + H)⁻¹．
  1. 学び直しの式の確かめ：ξ を引いて実際に解いた超過損失の平均と，½ G_K(c) + ½ tr(M_K Cov ξ) を比べる
  2. 定理 1：M_K の食い違い D_st から最小 2 乗で Ê を出し，w ∝ 1/Ê で平均した中心の G_K が，上界
     ((1 + ε)/(1 − ε))² (1 + (V − 1) ρ_K) (Σ 1/E_t)⁻¹ を超えないこと（ε = max |Ê_t/E_t − 1|，ρ_K は M_K の内積での実現した相関の最大）
  3. K に合わせた物差しの効果：見方 a は H の大きい方向で良く固有の方向（H の零空間）で悪い，見方 b はその逆，のとき，
     M_K の重みと F（K = 0 の物差し）の重みで，学び直した後の G_K を比べる
  4. 補題 2：2 つの見方の差が H の零空間にあるとき，どんな重みでも，片方の見方が θ* である 2 つの世界のどちらかで G_K ≥ ¼ ‖差‖²_M

    ./.venv/bin/python exp/sim_fusion_theorem.py
出力は標準出力だけ（数値は `docs/exploratory-analyses.md` E73 に写す）
"""

from __future__ import annotations

import itertools

import numpy as np

RNG = np.random.default_rng(20261028)


def psd(q, rank, rng, scale=1.0):
    B = rng.normal(size=(rank, q)) * scale / np.sqrt(q)
    return B.T @ B


def metric(F, H):
    Ai = np.linalg.inv(np.eye(len(F)) + H)
    return Ai @ F @ Ai


def hat(X, Mk):
    V = len(X)
    pairs = list(itertools.combinations(range(V), 2))
    A = np.zeros((len(pairs), V))
    D = np.zeros(len(pairs))
    for i, (s, t) in enumerate(pairs):
        A[i, s] = A[i, t] = 1.0
        d = X[s] - X[t]
        D[i] = d @ Mk @ d
    return np.linalg.lstsq(A, D, rcond=None)[0]


def g(c, th, Mk):
    e = c - th
    return float(e @ Mk @ e)


def check_refit(n_mc=4000):
    q, r = 30, 12
    F, H = psd(q, q, RNG, 2.0), psd(q, r, RNG, 3.0)
    th = RNG.normal(size=q)
    c = th + RNG.normal(size=q)
    Mk = metric(F, H)
    L = np.linalg.cholesky(H + 1e-12 * np.eye(q))
    Ai = np.linalg.inv(np.eye(q) + H)
    sims = []
    for _ in range(n_mc):
        xi = L @ RNG.normal(size=q)
        t = th + Ai @ (c - th + xi)
        sims.append(0.5 * (t - th) @ F @ (t - th))
    pred = 0.5 * g(c, th, Mk) + 0.5 * np.trace(Mk @ H)
    print(f"1. 学び直しの式：実際に解いた平均 {np.mean(sims):.4f} ± {np.std(sims) / np.sqrt(n_mc):.4f}，式 {pred:.4f}")


def check_theorem(n=20000):
    viol, better_best, better_mean, better_pf, valid = 0, 0, 0, 0, 0
    ratios = []
    for _ in range(n):
        q = int(RNG.integers(20, 200))
        r = int(RNG.integers(0, q))
        V = int(RNG.integers(3, 6))
        F, H = psd(q, q, RNG, RNG.uniform(0.5, 3)), psd(q, r, RNG, RNG.uniform(0.5, 5)) if r else np.zeros((q, q))
        Mk = metric(F, H)
        th = RNG.normal(size=q)
        shared = RNG.normal(size=q) * RNG.uniform(0, 1.0)
        X = []
        for t in range(V):
            e = RNG.normal(size=q) * RNG.uniform(0.2, 2.0) * RNG.uniform(0.2, 1.0, size=q)
            if RNG.uniform() < 0.3:
                e = e + shared
            X.append(th + e)
        X = np.array(X)
        Ee = np.array([g(x, th, Mk) for x in X])
        Eh = hat(X, Mk)
        eps = float(np.max(np.abs(Eh / Ee - 1)))
        if not (eps < 1 and np.all(Eh > 0)):
            continue
        valid += 1
        C = np.array([[float((X[s] - th) @ Mk @ (X[t] - th)) for t in range(V)] for s in range(V)])
        rho = float(np.max(np.abs(C / np.sqrt(np.outer(Ee, Ee)))[~np.eye(V, dtype=bool)]))
        w = (1 / Eh) / np.sum(1 / Eh)
        Gw = g(w @ X, th, Mk)
        Hh = 1 / np.sum(1 / Ee)
        bound = ((1 + eps) / (1 - eps)) ** 2 * (1 + (V - 1) * rho) * Hh
        viol += Gw > bound * (1 + 1e-10)
        better_best += Gw <= Ee.min()
        better_mean += Gw <= g(X.mean(0), th, Mk)
        better_pf += Gw <= Ee[1]
        ratios.append(Gw / Ee.min())
    print(f"2. 定理 1：上界が成り立つ前提（ε < 1，Ê > 0）を満たした {valid} / {n} 例で，上界を超えた例 {viol}．"
          f"一番良い見方以下 {better_best / valid:.3f}，同じ重みの平均以下 {better_mean / valid:.3f}，"
          f"見方 1 つ（pF の役）以下 {better_pf / valid:.3f}．G/一番良い見方の中央値 {np.median(ratios):.3f}，90% 点 {np.quantile(ratios, 0.9):.3f}")


def check_k_aware(n=2000):
    """見方 a は H の値域で誤差小・零空間で誤差大，b は逆，c は両方で中くらい。"""
    wins, gains = 0, []
    for _ in range(n):
        q, r = 120, 40
        F = psd(q, q, RNG, 2.0)
        B = RNG.normal(size=(r, q))
        H = B.T @ B * 5.0 / q
        _, U = np.linalg.eigh(H)
        null, rng_ = U[:, : q - r], U[:, q - r:]
        Mk = metric(F, H)
        th = RNG.normal(size=q)

        def err(s_rng, s_null):
            return rng_ @ RNG.normal(size=r) * s_rng + null @ RNG.normal(size=q - r) * s_null

        X = np.array([th + err(0.2, 1.5), th + err(1.5, 0.4), th + err(0.9, 0.9)])
        w_k = 1 / hat(X, Mk)
        w_0 = 1 / hat(X, F)
        if np.any(w_k <= 0) or np.any(w_0 <= 0):
            continue
        Gk = g((w_k / w_k.sum()) @ X, th, Mk)
        G0 = g((w_0 / w_0.sum()) @ X, th, Mk)
        wins += Gk < G0
        gains.append(1 - Gk / G0)
    print(f"3. K に合わせた物差し：M_K の重みが F の重みより学び直した後の G_K を下げた割合 {wins / len(gains):.3f}（{len(gains)} 例），"
          f"下げた割合の中央値 {np.median(gains):.3f}")


def check_lemma2(n=2000):
    worst = []
    for _ in range(n):
        q, r = 50, 20
        F = psd(q, q, RNG, 2.0)
        B = RNG.normal(size=(r, q))
        H = B.T @ B / q
        _, U = np.linalg.eigh(H)
        v = U[:, 0]                                      # 零空間の方向
        Mk = metric(F, H)
        xs = RNG.normal(size=q)
        delta = RNG.uniform(0.5, 3.0)
        xt = xs - delta * v
        w = RNG.uniform()                                # どんな重みでも
        c = w * xs + (1 - w) * xt
        worst.append(max(g(c, xs, Mk), g(c, xt, Mk)) / (0.25 * delta ** 2 * float(v @ Mk @ v)))
    print(f"4. 補題 2：max(G_A, G_B) / (¼ ‖差‖²_M) の最小値 {min(worst):.4f}（1 以上なら補題どおり．一番良い見方はそれぞれの世界で 0）")


if __name__ == "__main__":
    check_refit()
    check_theorem()
    check_k_aware()
    check_lemma2()
