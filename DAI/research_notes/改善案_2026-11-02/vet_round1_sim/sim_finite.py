"""Finite-sample toy: K labels per class + n unlabeled with a fixed teacher.

Compares (UAR, class-balanced intercept from the K labels everywhere):
  KD        : ridge soft-label logistic on unlabeled + intercept from K
  KD+A4     : KD logits -> 4x4 affine map fit on K labels (ridge to identity)
  BWD(Mhat) : backward-corrected KD with M estimated from K labels (GLC: class means of q)
  BWD(Mtrue): same with the true M (population)
  sLDA-R    : soft means corrected by Rhat^-1, shrunk total covariance, LDA
  hard0.7   : one-hot labels where max q > 0.7
Teacher modes: 'ci' (posterior on x_T only), 'mild' (x_T posterior + x_S logits of a partly
wrong source direction), 'benign' (x_T posterior + x_S logits of the true direction).
"""
import sys
import numpy as np
from scipy.special import softmax
sys.path.insert(0, ".")
import sim_pop as S  # reuses helpers; SEPS/SEPT come from argv

C = 4
p = S.p


def ledoit_shrink(X):
    Xc = X - X.mean(0)
    n = len(X)
    Sg = Xc.T @ Xc / n
    mu = np.trace(Sg) / Sg.shape[0]
    d2 = np.sum((Sg - mu * np.eye(len(Sg))) ** 2)
    b2 = sum(np.sum((np.outer(x, x) - Sg) ** 2) for x in Xc[:400]) / (min(n, 400) * n)
    a = min(b2, d2) / d2
    return a * mu * np.eye(len(Sg)) + (1 - a) * Sg


def one(w, mode, K, n_unl, r, lam4=0.01, l2=1e-3):
    y, xs, xt = S.sample(w, n_unl, r)
    yk, xsk, xtk = S.sample(w, 400, r)
    # K per class labeled
    idx = np.concatenate([np.where(yk == c)[0][:K] for c in range(C)])
    yk, xsk, xtk = yk[idx], xsk[idx], xtk[idx]
    yte, xste, xtte = S.sample(w, 20000, r)

    def teach(xt_, xs_):
        q = S.teacher_post(w, xt_)
        if mode == "ci":
            return q
        musrc = w["mu"] + w["src_noise"] if mode == "mild" else w["mu"]
        return softmax(np.log(q + 1e-12) + S.lda_logits(xs_, musrc, w["Sig"], np.ones(C) / C), axis=1)

    q = teach(xt, xs)
    qk = teach(xtk, xsk)
    res = {}
    # population-level M for oracle
    yP, xsP, xtP = S.sample(w, 40000, r)
    qP = teach(xtP, xsP)
    Mtrue = np.stack([qP[yP == c].mean(0) for c in range(C)])
    res["teacher"] = S.uar(yte, teach(xtte, xste).argmax(1))
    zb = S.lda_logits(xste, w["mu"], w["Sig"], w["pi"])
    res["bayes"] = S.uar(yte, (zb + S.fit_bias_balanced(S.lda_logits(xsk, w["mu"], w["Sig"], w["pi"]), yk)).argmax(1))
    Wk, bk = S.fit_softlogreg(xs, q, l2=l2)
    zt, zk = xste @ Wk.T + bk, xsk @ Wk.T + bk
    res["KD"] = S.uar(yte, (zt + S.fit_bias_balanced(zk, yk)).argmax(1))
    A, b = S.fit_affine_balanced(zk, yk, lam=lam4)
    res["KD+A4"] = S.uar(yte, (zt @ A.T + b).argmax(1))
    Mhat = np.stack([qk[yk == c].mean(0) for c in range(C)])
    for nm, M in (("BWD(Mhat)", Mhat), ("BWD(Mtrue)", Mtrue)):
        try:
            Tb = q @ np.linalg.inv(M)
            Wc, bc = S.fit_softlogreg(xs, Tb, l2=l2)
            res[nm] = S.uar(yte, (xste @ Wc.T + bc + S.fit_bias_balanced(xsk @ Wc.T + bc, yk)).argmax(1))
        except np.linalg.LinAlgError:
            res[nm] = np.nan
    # soft LDA with Rhat
    pit = q.mean(0)
    pih = np.clip(np.linalg.solve(Mhat.T, pit), 1e-3, None); pih /= pih.sum()
    R = (Mhat * pih[:, None]).T / pit[:, None]
    mut = (q.T @ xs) / q.sum(0)[:, None]
    Sg = ledoit_shrink(xs)
    try:
        muc = np.linalg.solve(R, mut)
        z = S.lda_logits(xsk, muc, Sg, pih)
        res["sLDA-R"] = S.uar(yte, (S.lda_logits(xste, muc, Sg, pih) + S.fit_bias_balanced(z, yk)).argmax(1))
    except np.linalg.LinAlgError:
        res["sLDA-R"] = np.nan
    z = S.lda_logits(xsk, mut, Sg, pit)
    res["sLDA"] = S.uar(yte, (S.lda_logits(xste, mut, Sg, pit) + S.fit_bias_balanced(z, yk)).argmax(1))
    keep = q.max(1) > 0.7
    if keep.sum() > 20 and len(np.unique(q[keep].argmax(1))) == C:
        Wh, bh = S.fit_softlogreg(xs[keep], np.eye(C)[q[keep].argmax(1)], l2=l2)
        res["hard0.7"] = S.uar(yte, (xste @ Wh.T + bh + S.fit_bias_balanced(xsk @ Wh.T + bh, yk)).argmax(1))
    else:
        res["hard0.7"] = np.nan
    res["cond_M"] = np.linalg.cond(Mtrue)
    res["Mhat_err"] = np.abs(Mhat - Mtrue).mean()
    return res


if __name__ == "__main__":
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    imb = [0.45, 0.25, 0.2, 0.1]
    w = S.make_world(imb, seed=0)
    w["src_noise"] = np.random.default_rng(5).normal(size=w["mu"].shape) * 0.12
    for mode in ("ci", "benign", "mild"):
        for K in (5, 25):
            for n_unl in (1500,):
                rows = [one(w, mode, K, n_unl, np.random.default_rng(1000 + i)) for i in range(reps)]
                keys = rows[0].keys()
                m = {k: np.nanmean([r[k] for r in rows]) for k in keys}
                d = {k: np.array([r[k] - r["KD"] for r in rows]) for k in ("KD+A4", "BWD(Mhat)", "BWD(Mtrue)", "sLDA-R", "sLDA", "hard0.7", "bayes")}
                print(f"[{mode:6s} K={K:2d} n={n_unl}] teacher {m['teacher']*100:.1f} bayes {m['bayes']*100:.1f} KD {m['KD']*100:.1f} cond(M) {m['cond_M']:.1f} |Mhat-M| {m['Mhat_err']:.3f}")
                print("   minus KD (mean, sd, frac>0):", {k: (round(np.nanmean(v) * 100, 2), round(np.nanstd(v) * 100, 2), round(np.nanmean(v > 0), 2)) for k, v in d.items()})
