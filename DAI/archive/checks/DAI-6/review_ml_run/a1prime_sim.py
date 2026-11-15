"""人工データで，第 4 版の A1′（S3 と LDA の 3 次元の主角の余弦の 2 乗の平均 ≥ 0.8）が，母集団の「上位 3 個が感情か」と合うかを数える．
summarize の A1′ と同じ手順（主成分の白色化した座標で正準相関，_lda_basis，sigma_orth，principal_cos2）．実データは読まない。"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai6_probe as P
torch.set_num_threads(8)
V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)   # (1/4)Σ v vᵀ = I

def gen(n, p, bts, wt, rng):
    y = rng.integers(0, 4, n)
    b = np.sqrt(np.array(bts) / (1 - np.array(bts)))       # 次元 j の感情の間の正準相関が bts[j]
    s = np.sqrt(wt / (1 - wt)) if wt > 0 else 0.0           # 感情の中の共有の正準相関 wt
    z = rng.normal(size=n)
    X = {}
    for m in "ab":
        x = rng.normal(size=(n, p))
        x[:, :3] += V[y] * b
        x[:, 3] += s * z
        X[m] = x
    return y, X

def a1p(y, X, p):
    Xa = P.tt(X["a"] - X["a"].mean(0)); Xb = P.tt(X["b"] - X["b"].mean(0))
    Ea, la, Za, _ = P.pca(Xa, p); Zb = P.pca(Xb, p)[2]
    _, Da, _ = P.cca(Za, Zb)
    Sg = Xa.T @ Xa / (len(y) - 1)
    B3 = P.sigma_orth((Ea * la.rsqrt()) @ Da[:, :3], Sg)
    L = P.sigma_orth((Ea * la.rsqrt()) @ P._lda_basis(Za, y), Sg)
    P3 = P.sigma_orth(Ea[:, :3], Sg)
    return P.principal_cos2(B3, L, Sg), P.principal_cos2(P3, L, Sg)

settings = [("AFEW-like", 632, 13, 20), ("IEMOCAP-like", 2212, 48, 10), ("MELD-like", 5995, 131, 5), ("MOSEI-like", 14907, 200, 3)]
cases = [((0.25, 0.25, 0.25), 0.0, "感情"), ((0.25, 0.25, 0.25), 0.20, "感情"), ((0.15, 0.15, 0.15), 0.0, "感情（弱い）"),
         ((0.40, 0.25, 0.12), 0.0, "感情（不ぞろい）"), ((0.40, 0.25, 0.12), 0.20, "共有が 3 番目より強い"), ((0.25, 0.25, 0.25), 0.40, "共有が強い")]
rng = np.random.default_rng(0)
for name, n, p, reps in settings:
    for bts, wt, truth in cases:
        cs, cp = [], []
        for r in range(reps):
            y, X = gen(n, p, bts, wt, rng)
            a, b = a1p(y, X, p); cs.append(a); cp.append(b)
        cs = np.array(cs)
        print(f"{name:13s} 間={bts} 中={wt:.2f}（母集団：{truth}）| S3–LDA 平均 {cs.mean():.3f} [最小 {cs.min():.3f}，最大 {cs.max():.3f}]，"
              f"0.8 以上 {int((cs >= 0.8).sum())}/{reps} | 主成分3–LDA 平均 {np.mean(cp):.3f}", flush=True)
