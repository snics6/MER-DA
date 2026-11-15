"""dai-review-ml の DAI-8 の検査：人工データだけで lda_from_R と em_fit の性質を確かめる（実データは読まない）。

確かめること
  (a) 次元 d = 2,304 で感情ごとの候補の本数 n_k が数百のとき，推定した平均の雑音が −½ μ̂_kᵀΣ⁻¹μ̂_k を
      ½ tr(Σ⁻¹S)/n_k だけ大きくし，本数の少ない感情を判定で不利にするか（等しい事前で判定しても残る）。
  (b) E 段の重み r が 0 か 1 に張り付き，初期値の判定がそのまま残るか．β log q の項が判定を変える割合。
  (c) EM − LDA_K に，none[K]（ソースの向きのずれを含む頭）の初期値の効果が混ざるか．LDA_K の確率から始める β = 0 の EM と比べる。

台本 DAI/exp/dai8_probe.py の lda_from_R，em_fit，delta をそのまま呼ぶ（補正した判定は検査のための計算で，台本には無い）。
使い方：CUDA_VISIBLE_DEVICES="" .venv/bin/python DAI/checks/DAI-8/review_ml_run/em_selfterm_sim.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
import dai8_probe as P  # noqa: E402

DT = torch.float64
SRC = 0.7        # ソースの感情の平均のずれ（感情の部分空間の中の別の向き．大きさは sep の 0.7 倍）
LAM_K = 0.01     # none[K] のソースの頭への引き戻しの強さ
CONF = {   # 名前 → (候補の本数, 感情の割合)．本数と割合は DAI/splits/*.csv の 4 割（テストの分割を除く）に合わせた
    "AFEW型": (632, (0.26, 0.27, 0.22, 0.25)),
    "IEMOCAP型": (2350, (0.37, 0.32, 0.21, 0.10)),
    "MELD型": (8500, (0.57, 0.20, 0.09, 0.14)),
}


class Gen:
    """x = 感情の平均（rc 次元の部分空間）+ グループの平均（30 次元）+ 低い階数の相関した雑音 + 各次元の雑音。"""

    def __init__(self, d, rng, sep, rc=20, s_grp=1.0, rank=60, s_low=2.0):
        self.d = d
        self.B = np.linalg.qr(rng.standard_normal((d, rc)))[0]
        self.M = rng.standard_normal((4, rc)) * sep
        self.Bg = np.linalg.qr(rng.standard_normal((d, 30)))[0]
        self.L = rng.standard_normal((d, rank)) * s_low / np.sqrt(rank)
        self.s_grp = s_grp
        self.shift = rng.standard_normal((4, rc)) * sep         # ソースの頭の向きのずれ

    def sample(self, n, rng, props, n_grp, src=0.0):
        y = rng.choice(4, n, p=np.array(props) / sum(props))
        g = rng.integers(0, n_grp, n)
        G = rng.standard_normal((n_grp, 30)) * self.s_grp
        M = self.M + src * self.shift                           # src > 0：ソースのコーパス（感情の平均がずれる）
        x = (M[y] @ self.B.T + G[g] @ self.Bg.T
             + rng.standard_normal((n, self.L.shape[1])) @ self.L.T + rng.standard_normal((n, self.d)))
        return x, y


def rec(y, p):
    return np.array([100 * (p[y == k] == k).mean() for k in range(4)])


def fit_head(XL, yL, lam, ref):
    d = XL.shape[1]
    W = ref.clone().contiguous().requires_grad_(True)
    b = torch.zeros(4, dtype=DT, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

    def clos():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(XL @ W + b, yL) + lam * ((W - ref) ** 2).sum()
        loss.backward()
        return loss
    opt.step(clos)
    return W.detach(), b.detach()


def penalty(X, R, alpha):
    """½ tr(Σ⁻¹S)/n_k（推定した平均の雑音が −½ μ̂ᵀΣ⁻¹μ̂ に足す分の期待値）。"""
    n_k = R.sum(0)
    mu = (R.T @ X) / n_k[:, None]
    N = n_k.sum()
    Sm = (X.T @ X - (mu.T * n_k) @ mu) / N
    Sig = (1 - alpha) * Sm + alpha * torch.eye(X.shape[1], dtype=DT)
    tr = float(torch.trace(torch.linalg.solve(Sig, Sm)))
    return 0.5 * tr / n_k, tr


def one(name, d, seed, sep, alpha=0.5, K=5):
    n_pool, props = CONF[name]
    rng = np.random.default_rng(seed)
    gen = Gen(d, rng, sep)
    Xp, yp = gen.sample(n_pool, rng, props, 16)
    Xt, yt = gen.sample(3000, rng, props, 4)
    mu, sd = Xp.mean(0), Xp.std(0)
    Xp_t, Xt_t = torch.tensor((Xp - mu) / sd, dtype=DT), torch.tensor((Xt - mu) / sd, dtype=DT)
    yp_t = torch.tensor(yp)
    kpos = np.sort(np.concatenate([rng.permutation(np.flatnonzero(yp == k))[:K] for k in range(4)]))
    kpos_t = torch.tensor(kpos)
    ylab = yp_t[kpos_t]
    U = np.ones(n_pool, bool); U[kpos] = False
    Ut = torch.tensor(U)
    out = dict(conf=name, d=d, seed=seed)
    # 全ラベルの LDA（α = 0.5）と，平均の雑音の分を戻した判定（検査のための計算）
    onehot = torch.nn.functional.one_hot(yp_t, 4).to(DT)
    A, c, _ = P.lda_from_R(Xp_t, onehot, Xp_t.T @ Xp_t, alpha)
    pen, tr = penalty(Xp_t, onehot, alpha)
    p_lda = P.delta(Xt_t, A, c).argmax(1).numpy()
    p_cor = (P.delta(Xt_t, A, c) + pen).argmax(1).numpy()
    out["trSinvS"] = round(tr, 1)
    out["pen_max_minus_min"] = round(float(pen.max() - pen.min()), 2)
    out["ldaO_rec"] = rec(yt, p_lda).round(1).tolist()
    out["ldaO_uar"] = round(rec(yt, p_lda).mean(), 2)
    out["ldaO_corr_uar"] = round(rec(yt, p_cor).mean(), 2)
    # 全ラベルのロジスティック（線形の上限の代わり）
    W, b = fit_head(Xp_t, yp_t, 1e-3 / n_pool * 100, torch.zeros(d, 4, dtype=DT))
    out["logreg_all_uar"] = round(rec(yt, (Xt_t @ W + b).argmax(1).numpy()).mean(), 2)
    # none[K]：K 本で，ずれたソースの頭に引き戻したロジスティック
    Xs, ys = gen.sample(6000, rng, (0.25, 0.25, 0.25, 0.25), 16, src=SRC)
    Ws, _ = fit_head(torch.tensor((Xs - mu) / sd, dtype=DT), torch.tensor(ys), 1e-3, torch.zeros(d, 4, dtype=DT))
    W, b = fit_head(Xp_t[kpos_t], ylab, LAM_K, Ws)                      # K 本で，ソースの頭に引き戻す
    q = torch.softmax(Xp_t @ W + b, -1)
    out["none_uar"] = round(rec(yt, (Xt_t @ W + b).argmax(1).numpy()).mean(), 2)
    XL = Xp_t[kpos_t]
    A, c, _ = P.lda_from_R(XL, torch.nn.functional.one_hot(ylab, 4).to(DT), XL.T @ XL, alpha)
    out["ldaK_uar"] = round(rec(yt, P.delta(Xt_t, A, c).argmax(1).numpy()).mean(), 2)
    qK = torch.softmax(P.delta(Xp_t, A, c), -1)
    XtX = Xp_t.T @ Xp_t
    true_prop = np.bincount(yp, minlength=4) / n_pool
    for beta in (0.0, 1.0):
        A, c, pi, R, hist = P.em_fit(Xp_t, XtX, kpos_t, ylab, q, alpha, beta)
        pr = R.argmax(1).numpy()
        pt = P.delta(Xt_t, A, c).argmax(1).numpy()
        tag = f"em_b{int(beta)}"
        out[f"{tag}_uar"] = round(rec(yt, pt).mean(), 2)
        out[f"{tag}_rec"] = rec(yt, pt).round(1).tolist()
        out[f"{tag}_pi_minus_true"] = (pi.numpy() - true_prop).round(3).tolist()
        out[f"{tag}_keep_init_pct"] = round(100 * float((pr[U] == q.argmax(1).numpy()[U]).mean()), 1)
        out[f"{tag}_mean_maxpost"] = round(float(R[Ut].max(1).values.mean()), 3)
        dl = P.delta(Xp_t[Ut], A, c)
        t2 = torch.topk(dl, 2, 1).values
        out[f"{tag}_pool_gap_median_nat"] = round(float((t2[:, 0] - t2[:, 1]).median()), 2)
        dt = P.delta(Xt_t, A, c)
        t2 = torch.topk(dt, 2, 1).values
        out[f"{tag}_test_gap_median_nat"] = round(float((t2[:, 0] - t2[:, 1]).median()), 2)
        if beta == 1.0:
            lq = torch.log(q[Ut].clamp(min=1e-12))
            t2 = torch.topk(lq, 2, 1).values
            out["logq_gap_median_nat"] = round(float((t2[:, 0] - t2[:, 1]).median()), 2)
            base = torch.log(pi) + dl
            out["beta_changes_argmax_pct"] = round(100 * float(((base + lq).argmax(1) != base.argmax(1)).double().mean()), 2)
    A, c, pi, R, hist = P.em_fit(Xp_t, XtX, kpos_t, ylab, qK, alpha, 0.0)
    out["em_from_ldaK_b0_uar"] = round(rec(yt, P.delta(Xt_t, A, c).argmax(1).numpy()).mean(), 2)
    return out


def main():
    torch.manual_seed(0)
    for name in CONF:
        for seed in (0, 1, 2):
            r = one(name, 2304, seed, sep=0.55)
            print(r, flush=True)


if __name__ == "__main__":
    main()
