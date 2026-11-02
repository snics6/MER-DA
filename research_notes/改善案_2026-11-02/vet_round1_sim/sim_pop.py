"""Population-level checks of the 'confusion-corrected distillation' theory (toy, Gaussian).

C1  soft means = R mu under CI
C2  R^-1 corrected means + true Sigma_W -> Bayes UAR
C3  R^-1 corrected means + total covariance + intercept-only fit -> loss?
C4  KD-logistic weights lie in span{Sigma^-1 (mu_k - mu_j)} under CI (Gaussian)
C5  KD + 4x4 affine map (fit with labels) -> Bayes UAR
C6  symmetric uniform noise M with imbalanced priors -> direction rotates?
C7  CI violated (teacher also sees x_S through a wrong direction) -> what survives?
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

rng = np.random.default_rng(0)
C, p, pT = 4, 20, 10
import sys
SEPS, SEPT = float(sys.argv[1]), float(sys.argv[2])


def uar(y, pred):
    return np.mean([np.mean(pred[y == c] == c) for c in range(C)])


def make_world(pi, sepS=SEPS, sepT=SEPT, seed=0):
    r = np.random.default_rng(seed)
    A = r.normal(size=(p, p)) / np.sqrt(p)
    Sig = A @ A.T + 0.5 * np.eye(p)
    mu = r.normal(size=(C, p)) * sepS
    nu = r.normal(size=(C, pT)) * sepT
    return dict(pi=np.array(pi), Sig=Sig, mu=mu, nu=nu)


def sample(w, n, r):
    y = r.choice(C, size=n, p=w["pi"])
    L = np.linalg.cholesky(w["Sig"])
    xs = w["mu"][y] + r.normal(size=(n, p)) @ L.T
    xt = w["nu"][y] + r.normal(size=(n, pT))
    return y, xs, xt


def teacher_post(w, xt, prior=None):
    prior = w["pi"] if prior is None else prior
    lg = -0.5 * ((xt[:, None, :] - w["nu"][None]) ** 2).sum(-1) + np.log(prior)
    return softmax(lg, axis=1)


def lda_logits(x, mu, Sig, pi):
    Si = np.linalg.inv(Sig)
    W = mu @ Si
    b = -0.5 * np.einsum("kp,kp->k", W, mu) + np.log(pi)
    return x @ W.T + b


def fit_bias_balanced(z, y):
    """Only an intercept vector is fit (class-balanced CE)."""
    wts = np.array([1.0 / np.mean(y == c) for c in range(C)])[y]

    def f(b):
        s = z + b
        lse = logsumexp(s, axis=1)
        loss = np.sum(wts * (lse - s[np.arange(len(y)), y])) / len(y)
        P = np.exp(s - lse[:, None])
        Y = np.eye(C)[y]
        g = ((P - Y) * wts[:, None]).sum(0) / len(y)
        return loss, g

    return minimize(f, np.zeros(C), jac=True, method="L-BFGS-B").x


def fit_affine_balanced(z, y, lam=1e-4):
    """4x4 affine map on logits fit with class-balanced CE (ridge toward identity)."""
    wts = np.array([1.0 / np.mean(y == c) for c in range(C)])[y]
    Y = np.eye(C)[y]

    def f(th):
        Amat = th[: C * C].reshape(C, C)
        b = th[C * C:]
        s = z @ Amat.T + b
        lse = logsumexp(s, axis=1)
        loss = np.sum(wts * (lse - s[np.arange(len(y)), y])) / len(y) + lam * np.sum((Amat - np.eye(C)) ** 2)
        P = np.exp(s - lse[:, None])
        G = (P - Y) * wts[:, None] / len(y)
        gA = G.T @ z + 2 * lam * (Amat - np.eye(C))
        return loss, np.concatenate([gA.ravel(), G.sum(0)])

    th0 = np.concatenate([np.eye(C).ravel(), np.zeros(C)])
    th = minimize(f, th0, jac=True, method="L-BFGS-B").x
    return th[: C * C].reshape(C, C), th[C * C:]


def fit_softlogreg(x, T, l2=1e-6, W0=None):
    """Multinomial LR with soft (possibly signed, rows sum to 1) targets T."""
    n, d = x.shape
    X1 = np.hstack([x, np.ones((n, 1))])

    def f(th):
        Wb = th.reshape(C, d + 1)
        s = X1 @ Wb.T
        lse = logsumexp(s, axis=1)
        loss = np.mean(lse - (T * s).sum(1)) + l2 * np.sum(Wb[:, :d] ** 2)
        P = np.exp(s - lse[:, None])
        G = (P - T) / n
        g = G.T @ X1
        g[:, :d] += 2 * l2 * Wb[:, :d]
        return loss, g.ravel()

    th0 = np.zeros(C * (d + 1)) if W0 is None else W0.ravel()
    th = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=2000)).x
    Wb = th.reshape(C, d + 1)
    return Wb[:, :d], Wb[:, d]


def subspace_residual(W, w):
    """Fraction of ||W_centered||^2 outside span{Sig^-1 (mu_k - mu_1)}."""
    Si = np.linalg.inv(w["Sig"])
    B = (w["mu"][1:] - w["mu"][0]) @ Si  # (C-1, p)
    Q, _ = np.linalg.qr(B.T)
    Wc = W - W.mean(0)
    proj = Wc @ Q @ Q.T
    return np.sum((Wc - proj) ** 2) / np.sum(Wc ** 2)


def run(label, pi, M_override=None, ci=True, wrong_dir_scale=0.0, seed=0, n=200_000):
    w = make_world(pi, seed=seed)
    r = np.random.default_rng(seed + 100)
    y, xs, xt = sample(w, n, r)
    ylab, xslab, xtlab = sample(w, 40_000, r)  # population-level labeled set (for fits that use labels)
    yte, xste, xtte = sample(w, 100_000, r)

    if M_override is None:
        q = teacher_post(w, xt)
        qlab = teacher_post(w, xtlab)
    else:  # sample-level synthetic noise: q is one-hot of a label drawn from M[y]
        cum = np.cumsum(M_override[y], 1)
        lab = (r.random(n)[:, None] > cum).sum(1)
        q = np.eye(C)[lab]
        cuml = np.cumsum(M_override[ylab], 1)
        labl = (r.random(len(ylab))[:, None] > cuml).sum(1)
        qlab = np.eye(C)[labl]
    if not ci:
        # teacher also uses x_S through a WRONG direction (e.g., source-trained block)
        r2 = np.random.default_rng(seed + 7)
        mu_src = w["mu"] + r2.normal(size=w["mu"].shape) * 0.8
        add = lda_logits(xs, mu_src, w["Sig"], np.ones(C) / C) * wrong_dir_scale
        addl = lda_logits(xslab, mu_src, w["Sig"], np.ones(C) / C) * wrong_dir_scale
        q = softmax(np.log(q + 1e-12) + add, axis=1)
        qlab = softmax(np.log(qlab + 1e-12) + addl, axis=1)

    out = {}
    # Bayes for the student
    zb = lda_logits(xste, w["mu"], w["Sig"], w["pi"])
    bb = fit_bias_balanced(lda_logits(xslab, w["mu"], w["Sig"], w["pi"]), ylab)
    out["bayes"] = uar(yte, (zb + bb).argmax(1))
    out["teacher_uar"] = uar(y, q.argmax(1))

    # M (row y: E[q | y]) estimated at population level from labels (GLC style), R from M and pi~
    M = np.stack([qlab[ylab == c].mean(0) for c in range(C)])
    pit = q.mean(0)
    pi_hat = np.linalg.solve(M.T, pit)
    R = (M * pi_hat[:, None]).T / pit[:, None]  # R[j, y] = P(y | teacher j)
    out["M_offdiag_asym"] = np.abs(M - M.T).sum() / 2

    # soft means
    mut = (q.T @ xs) / q.sum(0)[:, None]
    out["C1_softmean_vs_Rmu"] = np.abs(mut - R @ w["mu"]).max() / np.abs(w["mu"]).max()

    # soft LDA with TRUE Sigma_W + intercept-only fit
    z = lda_logits(xslab, mut, w["Sig"], pit)
    b = fit_bias_balanced(z, ylab)
    out["softLDA_trueSig_bias"] = uar(yte, (lda_logits(xste, mut, w["Sig"], pit) + b).argmax(1))
    # R^-1 corrected means + true Sigma_W
    muc = np.linalg.solve(R, mut)
    z = lda_logits(xslab, muc, w["Sig"], pi_hat)
    b = fit_bias_balanced(z, ylab)
    out["C2_Rinv_trueSig_bias"] = uar(yte, (lda_logits(xste, muc, w["Sig"], pi_hat) + b).argmax(1))
    # R^-1 corrected means + TOTAL covariance + intercept-only
    SigT = np.cov(xs.T)
    z = lda_logits(xslab, muc, SigT, pi_hat)
    b = fit_bias_balanced(z, ylab)
    out["C3_Rinv_totalSig_bias"] = uar(yte, (lda_logits(xste, muc, SigT, pi_hat) + b).argmax(1))
    # R^-1 corrected means + Sigma_W reconstructed = SigT - between(corrected)
    mbar = pi_hat @ muc
    SigW = SigT - ((muc - mbar).T * pi_hat) @ (muc - mbar)
    z = lda_logits(xslab, muc, SigW, pi_hat)
    b = fit_bias_balanced(z, ylab)
    out["C3b_Rinv_reconSigW_bias"] = uar(yte, (lda_logits(xste, muc, SigW, pi_hat) + b).argmax(1))

    # KD logistic (soft labels)
    sub = slice(0, 60_000)
    Wk, bk = fit_softlogreg(xs[sub], q[sub])
    out["C4_KD_subspace_resid"] = subspace_residual(Wk, w)
    zk_l = xslab @ Wk.T + bk
    zk_t = xste @ Wk.T + bk
    b = fit_bias_balanced(zk_l, ylab)
    out["KD_bias"] = uar(yte, (zk_t + b).argmax(1))
    Am, bm = fit_affine_balanced(zk_l, ylab)
    out["C5_KD_affine4x4"] = uar(yte, (zk_t @ Am.T + bm).argmax(1))
    # backward-corrected KD (targets q M^-1)
    Tb = q[sub] @ np.linalg.inv(M)
    Wc, bc = fit_softlogreg(xs[sub], Tb)
    b = fit_bias_balanced(xslab @ Wc.T + bc, ylab)
    out["KD_backward_bias"] = uar(yte, (xste @ Wc.T + bc + b).argmax(1))
    out["KD_backward_subspace_resid"] = subspace_residual(Wc, w)
    # hard-threshold transform (keep max q > 0.7, one-hot)
    keep = q[sub].max(1) > 0.7
    Th = np.eye(C)[q[sub][keep].argmax(1)]
    Wh, bh = fit_softlogreg(xs[sub][keep], Th)
    b = fit_bias_balanced(xslab @ Wh.T + bh, ylab)
    out["hard0.7_bias"] = uar(yte, (xste @ Wh.T + bh + b).argmax(1))
    out["hard0.7_subspace_resid"] = subspace_residual(Wh, w)
    print(label, {k: (round(v * 100, 2) if "uar" in k.lower() or k in ("bayes",) or "bias" in k or "affine" in k else round(v, 4)) for k, v in out.items()})
    return out


if __name__ == "__main__":
    imb = [0.45, 0.25, 0.2, 0.1]
    bal = [0.25] * 4
    run("CI, posterior teacher, imbalanced", imb)
    run("CI, posterior teacher, balanced  ", bal)
    a = 0.4
    Msym = a * np.eye(C) + (1 - a) / C * np.ones((C, C))
    run("CI, symmetric-uniform M, imbalanced", imb, M_override=Msym)
    run("CI, symmetric-uniform M, balanced  ", bal, M_override=Msym)
    run("CI violated (wrong x_S dir x1.0)   ", imb, ci=False, wrong_dir_scale=1.0)
    run("CI violated (wrong x_S dir x0.5)   ", imb, ci=False, wrong_dir_scale=0.5)
