"""DAI 実験の学習器：重みつきの多クラスのロジスティック回帰を，多くの問を一度に GPU で解く。

問 p の目的関数（S_p は重みの和）：

    L_p(W, b) = [ Σ_i Σ_k G[p,i,k] · (−log softmax(x_i W_pᵀ + b_p)_k)
                  + ‖W_p − W0_p‖² / (2 Cw_p) + ‖b_p − b0_p‖² / (2 Cb_p) ] / S_p

- G[p,i,k] ≥ 0 は発話 i の感情 k の重み．K 本の正解（one-hot × 感情の釣り合いの重み × 倍率 a），
  疑似ラベル（確率 × 重み），の和で作る．発話ごとに複数の感情に重みを置いてよい（柔らかいラベル）．
- Cw = inf は W の罰則なし，Cb = inf は b の罰則なし．W0 = 0 なら普通の L2 正則化のロジスティック回帰である．
- 凸なので解は 1 つに決まる．L-BFGS（強い Wolfe の直線探索）で，問ごとの勾配の最大が tol 以下になるまで解く．
- fix_W=True なら W を W0 に固定して b だけを解く（役割を分けた版の切片の段）．

X は呼ぶ側で標準化してから渡す（計画 8.4 節の M9：同じ分割の中では全手法が同じ統計）。
`--selftest` で scikit-learn と同じ解になるかを確かめる。
"""

from __future__ import annotations

import math

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64 if DEV == "cpu" else torch.float32


def t(x, dtype=None):
    return x.to(DEV, dtype or DT) if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), device=DEV, dtype=dtype or DT)


def balanced_onehot(y: np.ndarray, idx: np.ndarray, n: int, scale: float = 1.0) -> torch.Tensor:
    """idx の発話に，感情の釣り合いの重み（len/(感情の数 × その感情の件数)）× scale の one-hot を置いた (n, 4)。"""
    G = torch.zeros(n, 4, device=DEV, dtype=DT)
    if len(idx) == 0:
        return G
    yy = y[idx]
    cnt = np.bincount(yy, minlength=4).astype(float)
    present = (cnt > 0).sum()
    w = len(idx) / (present * cnt[yy])
    G[torch.as_tensor(idx, device=DEV), torch.as_tensor(yy, device=DEV)] = t(w * scale)
    return G


def soft_rows(P: torch.Tensor, total: float, balance: bool = True) -> torch.Tensor:
    """疑似ラベルの重み (n, 4)。balance=True なら感情ごとの総量をそろえてから，全体を total にする。"""
    P = t(P)
    if balance:
        P = P / P.sum(0, keepdim=True).clamp(min=1e-12)
    return P / P.sum().clamp(min=1e-12) * total


def fit(X: torch.Tensor, G: torch.Tensor, W0: torch.Tensor, b0: torch.Tensor,
        Cw, Cb, fix_W: bool = False, max_iter: int = 3000, tol: float = 1e-6) -> dict:
    """G: (P, n, 4)．W0: (P, 4, d) か (4, d)．b0: (P, 4) か (4,)．Cw, Cb: 長さ P か数．
    戻り：W (P,4,d)，b (P,4)，n_iter，grad_max（問ごとの勾配の最大の絶対値の最大），ok。"""
    X = t(X)
    G = t(G)
    P, n, k = G.shape
    d = X.shape[1]
    W0 = t(W0).expand(P, k, d).contiguous() if t(W0).dim() == 2 else t(W0)
    b0 = t(b0).expand(P, k).contiguous() if t(b0).dim() == 1 else t(b0)
    Cw = t(np.broadcast_to(np.asarray(Cw, float), (P,)).copy())
    Cb = t(np.broadcast_to(np.asarray(Cb, float), (P,)).copy())
    S = G.sum((1, 2)).clamp(min=1e-12)
    lw = torch.where(torch.isinf(Cw), torch.zeros_like(Cw), 1.0 / (2 * Cw))
    lb = torch.where(torch.isinf(Cb), torch.zeros_like(Cb), 1.0 / (2 * Cb))

    W = W0.clone().requires_grad_(not fix_W)
    b = b0.clone().requires_grad_(True)
    params = [b] if fix_W else [W, b]

    def objective():
        Z = torch.einsum("nd,pkd->pnk", X, W) + b[:, None, :]
        nll = -(G * torch.log_softmax(Z, 2)).sum((1, 2))
        reg = lw * ((W - W0) ** 2).sum((1, 2)) + lb * ((b - b0) ** 2).sum(1)
        return ((nll + reg) / S).sum()

    opt = torch.optim.LBFGS(params, lr=1, max_iter=max_iter, tolerance_grad=tol, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    opt.step(closure)
    opt.zero_grad()
    objective().backward()
    gmax = max(float(p.grad.abs().max()) for p in params)
    n_iter = opt.state[opt._params[0]].get("n_iter", -1)
    return {"W": W.detach(), "b": b.detach(), "n_iter": int(n_iter), "grad_max": gmax, "ok": gmax <= 10 * tol}


def logits(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(P,4,d),(P,4) → (P, n, 4)。W が (4,d) なら (n,4)。"""
    X = t(X)
    if W.dim() == 2:
        return X @ W.T + b
    return torch.einsum("nd,pkd->pnk", X, W) + b[:, None, :]


def uar(y: np.ndarray, pred: np.ndarray) -> float:
    rs = [float((pred[y == k] == k).mean()) for k in range(4) if (y == k).any()]
    return float(np.mean(rs))


def uar_batch(y: np.ndarray, Z: torch.Tensor) -> np.ndarray:
    """Z: (P, n, 4) → 問ごとの UAR（%）。"""
    pred = Z.argmax(2).cpu().numpy()
    return np.array([100.0 * uar(y, pr) for pr in pred])


def selftest() -> None:
    """scikit-learn の多項ロジスティック回帰（class_weight='balanced'）と同じ解になるか。"""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    n, d = 600, 20
    X = rng.normal(size=(n, d))
    Wt = rng.normal(size=(4, d))
    y = (X @ Wt.T + rng.gumbel(size=(n, 4))).argmax(1)
    y[:20] = 3                                   # 釣り合いを崩す
    Xs = (X - X.mean(0)) / X.std(0)
    for C in (0.01, 0.3, 10.0):
        sk = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, tol=1e-10).fit(Xs, y)
        G = balanced_onehot(y, np.arange(n), n)[None]
        # sklearn：½‖W‖² + C Σ w_i ℓ_i ↔ ここ：(Σ w_i ℓ_i + ‖W‖²/(2C')) / S は C' = C で同じ最小点
        r = fit(t(Xs), G, torch.zeros(4, d), torch.zeros(4), Cw=C, Cb=math.inf, tol=1e-9)
        dW = float((r["W"][0].cpu().numpy() - sk.coef_).__abs__().max())
        # 多項の切片は定数をずらしても同じなので，平均を引いて比べる
        bb = r["b"][0].cpu().numpy(); skb = sk.intercept_
        db = float(np.abs((bb - bb.mean()) - (skb - skb.mean())).max())
        print(f"C={C}: W の差の最大 {dW:.2e}，b の差の最大 {db:.2e}，反復 {r['n_iter']}，勾配 {r['grad_max']:.1e}")
        assert dW < 1e-3 and db < 1e-3, "scikit-learn と解が違う"
    # 引き戻し：Cw→0 で W0 に一致，fix_W で W が動かない
    W0 = torch.randn(4, d); b0 = torch.randn(4)
    G = balanced_onehot(y, np.arange(50), n)[None]
    r = fit(t(Xs), G, W0, b0, Cw=1e-8, Cb=1e-8)
    assert float((r["W"][0].cpu() - W0.to(r["W"].dtype)).abs().max()) < 1e-4
    r = fit(t(Xs), G, W0, b0, Cw=1.0, Cb=1.0, fix_W=True)
    assert float((r["W"][0].cpu() - W0.to(r["W"].dtype)).abs().max()) == 0.0
    print("selftest: 通った")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
