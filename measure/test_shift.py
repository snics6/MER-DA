"""`shift.py` と `ablate.py` の検算（答えが分かっている人工データで確かめる）。

    ./.venv/bin/python measure/test_shift.py      # 1 分以内に終わる

指標や操作を変えたら必ず流す。1 つでも落ちたら，その指標を実験に使わない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate  # noqa: E402
import shift  # noqa: E402

D, N = 64, 300


def gauss(rng, n, mean=0.0, scale=1.0, d=D):
    return rng.normal(mean, scale, size=(n, d))


def spk(n, n_spk, prefix):
    return np.array([f"{prefix}{i % n_spk}" for i in range(n)])


def check(name, cond, detail):
    print(("OK   " if cond else "FAIL ") + f"{name}: {detail}")
    return bool(cond)


def main() -> int:
    rng = np.random.default_rng(0)
    ok = True

    # ---- 軸 1：同一分布ではほぼ 0，平均をずらすと大きい
    a, b = gauss(rng, N), gauss(rng, N)
    c = gauss(rng, N, mean=0.3)
    pad_same = shift.pad(a, spk(N, 20, "a"), b, spk(N, 20, "b"))
    pad_diff = shift.pad(a, spk(N, 20, "a"), c, spk(N, 20, "c"))
    # 平均の差のノルム 0.3·√64 = 2.4 → ベイズ誤り率 Φ(−1.2) ≈ 0.115 → 理論上の PAD ≈ 1.54
    ok &= check("PAD", pad_same < 0.3 and 1.2 < pad_diff <= 1.6,
                f"同一 {pad_same:.3f} / ずれ {pad_diff:.3f}（理論上限 約 1.54）")
    m_same, m_diff = shift.mmd2_unbiased(a, b), shift.mmd2_unbiased(a, c)
    ok &= check("MMD²", abs(m_same) < 0.01 and m_diff > 10 * abs(m_same),
                f"同一 {m_same:.5f}（負もありうる） / ずれ {m_diff:.5f}")
    f_same, f_diff = shift.frechet(a, b, r=16), shift.frechet(a, c, r=16)
    ok &= check("Fréchet", f_diff > 2 * f_same, f"同一 {f_same:.3f} / ずれ {f_diff:.3f}")

    # ---- 軸 2：coverage は同一分布で理論値に近く，向きがある
    covs = [shift.density_coverage(gauss(rng, N), gauss(rng, N))[1] for _ in range(10)]
    exp_cov = shift.expected_coverage(N, N)
    ok &= check("coverage 同一分布", abs(np.mean(covs) - exp_cov) < 0.02,
                f"実測 {np.mean(covs):.3f} / 理論 {exp_cov:.3f}")
    # ソースは 2 つの塊，ターゲットはその片方だけ。ターゲットは覆われ，ソースの半分は覆われない
    src = np.vstack([gauss(rng, N // 2, d=8), gauss(rng, N // 2, mean=6.0, d=8)])
    tgt = gauss(rng, N, d=8)
    _, cov_t_by_s = shift.density_coverage(tgt, src)
    _, cov_s_by_t = shift.density_coverage(src, tgt)
    ok &= check("coverage の向き", cov_t_by_s > 0.7 and cov_s_by_t < 0.6,
                f"ターゲットが覆われる {cov_t_by_s:.3f} / ソースが覆われる {cov_s_by_t:.3f}")
    # 密度の比にも反応する：広く薄いソースは，範囲を含んでいても狭いターゲットを覆わない
    wide, narrow = gauss(rng, N, scale=2.0, d=8), gauss(rng, N, scale=1.0, d=8)
    _, cov_n_by_w = shift.density_coverage(narrow, wide)
    ok &= check("coverage は密度の比に反応", cov_n_by_w < 0.5,
                f"狭いターゲットが広いソースに覆われる {cov_n_by_w:.3f}（低いのが正しい）")
    cd_wide = shift.cov_dominance(gauss(rng, 2000, scale=2.0, d=16), gauss(rng, 2000, d=16), r=8)
    cd_narrow = shift.cov_dominance(gauss(rng, 2000, scale=0.5, d=16), gauss(rng, 2000, d=16), r=8)
    ok &= check("共分散の支配", cd_wide > 1.0 > cd_narrow,
                f"広いソース {cd_wide:.2f} / 狭いソース {cd_narrow:.2f}")

    # ---- ゼロ点：同じ群を話者で割った値は小さい
    X = gauss(rng, 900)
    z = shift.zero_point(X, spk(900, 30, "p"), n=N, reps=2)
    ok &= check("ゼロ点", z["pad"] < 0.4, f"PAD {z['pad']:.3f}，coverage {z['coverage_t_by_s']:.3f}")

    # ---- 測る空間（md 0.1）：PCA 64 次元でも同一分布は小さく，ずれは大きい
    a128, b128 = gauss(rng, N, d=128), gauss(rng, N, d=128)
    c128 = gauss(rng, N, mean=0.3, d=128)
    g_a, g_b = spk(N, 20, "a"), spk(N, 20, "b")
    m_same = shift.input_metrics(a128, g_a, b128, g_b, n=200, reps=2, space="pca64")
    m_diff = shift.input_metrics(a128, g_a, c128, g_b, n=200, reps=2, space="pca64")
    ok &= check("PCA 64 次元の空間", m_same["pad"] < 0.4 < 1.0 < m_diff["pad"]
                and m_same["dim"] == 128 and m_same["space"] == "pca64",
                f"PAD 同一 {m_same['pad']:.3f} / ずれ {m_diff['pad']:.3f}")
    # 次元が小さい空間（対数オッズの 4 次元）でも全部の指標が値を返す
    m4 = shift.input_metrics(gauss(rng, N, d=4), g_a, gauss(rng, N, mean=1.0, d=4), g_b,
                             n=200, reps=2)
    ok &= check("4 次元でも動く", all(np.isfinite(m4[k]) for k in
                                   ("pad", "mmd2", "frechet", "coverage_t_by_s", "cov_dominance")),
                f"PAD {m4['pad']:.3f}，フレシェ {m4['frechet']:.3f}，c {m4['cov_dominance']:.3f}")
    # 広がりが 0 の方向がある空間（和が一定の 4 つの点数）でも共分散の支配が落ちない
    L4 = gauss(rng, N, d=4)
    L4 -= L4.mean(1, keepdims=True)                   # 4 つの和が 0 → 実質 3 次元
    L4s = gauss(rng, N, scale=2.0, d=4)
    L4s -= L4s.mean(1, keepdims=True)
    cd4 = shift.cov_dominance(L4s, L4)
    ok &= check("広がり 0 の方向を除く", np.isfinite(cd4) and cd4 > 1.0,
                f"広いソースの c {cd4:.2f}（1 より大きいのが正しい）")
    # 割合をそろえた抜き出し（md C9）
    yb = np.repeat([0, 1, 2, 3], [400, 100, 60, 40])
    ib = shift.subsample_balanced(yb, 160, rng)
    ok &= check("感情ごとに同数を抜く", np.all(np.bincount(yb[ib], minlength=4) == 40),
                f"{np.bincount(yb[ib], minlength=4)}")
    try:
        shift.subsample_balanced(yb, 200, rng)
        short_ok = False
    except ValueError:
        short_ok = True
    ok &= check("足りなければ例外", short_ok, "感情 3 が 40 本しかない所で 50 本は抜かない")

    # ---- 軸 3：割合
    ys, yt = np.repeat([0, 1, 2, 3], [25, 25, 25, 25]), np.repeat([0, 1, 2, 3], [70, 10, 10, 10])
    ok &= check("label_l1", abs(shift.label_l1(ys, yt) - 0.9) < 1e-9, f"{shift.label_l1(ys, yt):.3f}（理論 0.900）")

    # MLLS＋BCTS：ソースは均等，ターゲットは偏った割合。p(x|y) は同じ（ラベルシフト）
    means = rng.normal(0, 1.0, size=(4, 8))

    def draw(counts):
        y = np.repeat(np.arange(4), counts)
        return means[y] + rng.normal(0, 1.0, size=(len(y), 8)), y

    Xtr, ytr = draw([500] * 4)
    Xho, yho = draw([500] * 4)
    Xt, yt = draw([1400, 200, 200, 200])
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    temp, bias = shift.fit_bcts(clf.decision_function(Xho), yho)
    p_ho = shift.apply_bcts(clf.decision_function(Xho), temp, bias)
    q = shift.mlls_prior(shift.apply_bcts(clf.decision_function(Xt), temp, bias), p_ho.mean(0))
    ok &= check("MLLS の割合推定", np.abs(q - [0.7, 0.1, 0.1, 0.1]).max() < 0.05, f"推定 {np.round(q, 3)}")

    # ---- 軸 4：混同行列・UAR・BER・Δ_CE
    yt4 = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    yp4 = np.array([0, 1, 1, 1, 2, 2, 3, 0])
    C = shift.confusion(yt4, yp4)
    ok &= check("UAR", abs(shift.uar(yt4, yp4) - 0.75) < 1e-9, f"{shift.uar(yt4, yp4):.3f}（理論 0.750）")
    ok &= check("BER は最大値", abs(shift.ber(C) - 0.5) < 1e-9, f"{shift.ber(C):.3f}（理論 0.500）")
    dce, _ = shift.delta_ce(np.eye(4), C)
    ok &= check("Δ_CE", abs(dce - 0.5) < 1e-9, f"{dce:.3f}（理論 0.500）")

    # ---- ablate
    y = np.repeat([0, 1, 2, 3], [300, 100, 50, 80])
    i = ablate.balanced_subset(y, rng)
    ok &= check("P1 均等化", np.all(np.bincount(y[i]) == 50), f"{np.bincount(y[i])}")
    i = ablate.prior_matched_subset(y, np.array([0.7, 0.1, 0.1, 0.1]), rng)
    ok &= check("P2 割合合わせ", np.allclose(np.bincount(y[i]) / len(i), [0.7, 0.1, 0.1, 0.1], atol=0.01),
                f"{np.round(np.bincount(y[i]) / len(i), 3)}（{len(i)} 本）")
    yf = ablate.flip_labels(y, 0.2, rng)
    ok &= check("D3s 一様な付け替え", abs(np.mean(yf != y) - 0.2) < 1e-3, f"付け替え率 {np.mean(yf != y):.3f}")
    yp = ablate.flip_labels_pair(y, 0, 2, 0.3, rng)
    moved = np.mean(yp[y == 0] == 2)
    ok &= check("D3a 対の付け替え", abs(moved - 0.3) < 1e-2 and np.all(yp[y != 0] == y[y != 0]),
                f"Neutral→Sad の割合 {moved:.3f}，他は不変")
    true_b = np.array([1.0, -0.5, 0.0, -0.5])
    Xk, yk = draw([200] * 4)
    logit = clf.decision_function(Xk)
    y_shift = np.array([rng.choice(4, p=p) for p in shift.apply_bcts(logit, 1.0, true_b)])
    b = ablate.fit_class_bias(logit, y_shift, l2=0.0)
    ok &= check("D1 閾値の補正", np.abs(b - (true_b - true_b.mean())).max() < 0.3,
                f"推定 {np.round(b, 2)} / 真 {np.round(true_b - true_b.mean(), 2)}")
    Xs_c = gauss(rng, 2000, d=8) @ np.diag([3, 1, 1, 1, 1, 1, 1, 0.3])
    Xt_c = gauss(rng, 2000, d=8)
    before = np.abs(np.cov(Xs_c, rowvar=False) - np.cov(Xt_c, rowvar=False)).max()
    after = np.abs(np.cov(ablate.coral(Xs_c, Xt_c, reg=1e-6), rowvar=False) - np.cov(Xt_c, rowvar=False)).max()
    ok &= check("I3 CORAL", after < 0.1 * before, f"共分散の差 {before:.3f} → {after:.3f}")
    Xs_n = np.vstack([gauss(rng, 200, d=8), gauss(rng, 200, mean=5.0, d=8)])
    sel = ablate.near_target_subset(Xs_n, gauss(rng, 300, d=8), 200)
    ok &= check("C1 近傍選択", np.mean(sel < 200) > 0.95, f"ターゲット側から選んだ割合 {np.mean(sel < 200):.3f}")
    Xz, n_per = ablate.standardize_by_speaker(gauss(rng, 100, mean=3.0, d=4), spk(100, 5, "q"))
    ok &= check("I2 話者標準化", np.abs(Xz.mean(0)).max() < 1e-9, f"平均の最大 {np.abs(Xz.mean(0)).max():.1e}")

    # ---- 理論の検算（protocol 7.3・7.4 の導出）
    eta = rng.dirichlet(np.ones(4), size=20000)
    for rho in (0.1, 0.2, 0.4):
        k = 4
        eta_s = eta * (1 - rho * k / (k - 1)) + rho / (k - 1)          # D3s 後の事後確率
        same = np.all(eta_s.argmax(1) == eta.argmax(1))
        srt, srt_s = np.sort(eta, 1), np.sort(eta_s, 1)
        ratio = (srt_s[:, -1] - srt_s[:, -2]) / (srt[:, -1] - srt[:, -2])
        ok &= check(f"7.4 一様な付け替え ρ={rho}", same and np.allclose(ratio, 1 - rho * k / (k - 1)),
                    f"最大の感情は不変，余裕は {ratio.mean():.3f} 倍（理論 {1 - rho * k / (k - 1):.3f}）")
    rho = 0.2
    e2 = eta.copy()
    e2[:, 0], e2[:, 2] = (1 - rho) * eta[:, 0], eta[:, 2] + rho * eta[:, 0]   # D3a：0→2
    pick0 = e2[:, 0] > e2[:, 2]
    pred = eta[:, 0] / eta[:, 2] > 1 / (1 - 2 * rho)
    ok &= check("7.4 対の付け替えの境界", np.all(pick0 == pred),
                f"0 と 2 の境界が比 1/(1−2ρ)={1 / (1 - 2 * rho):.3f} に動く")
    worst = 0.0
    for _ in range(2000):
        Cs, Ct = rng.dirichlet(np.ones(4), 4), rng.dirichlet(np.ones(4), 4)
        gap = abs(np.mean(np.diag(Cs)) - np.mean(np.diag(Ct)))
        worst = max(worst, gap / (3 * shift.delta_ce(Cs, Ct)[0]))
    ok &= check("7.3 |UAR差| ≤ 3Δ_CE", worst <= 1 + 1e-12, f"乱数の混同行列 2,000 組で 比の最大 {worst:.3f}（≤1 なら成立）")
    w = np.r_[np.full(300, 1 / 300), np.full(50, 1 / 50)]
    ok &= check("Kish n_eff", abs(shift.kish_neff(w) - 171.43) < 0.01, f"{shift.kish_neff(w):.2f}（理論 171.43）")

    print("すべて通過" if ok else "失敗あり")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
