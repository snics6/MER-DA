"""A1′ の読み方を，単位ごとの参照（前提が成り立つとした人工データ）で較正する案が働くかを，人工データで確かめる．実データは読まない．
参照：観測の主成分の白色化した座標で感情ごとの平均を推定し，見方ごとに独立な雑音（共分散 I − 感情の間の共分散）を足したデータを，
同じ n・p・感情の並びで作り，S3–LDA の余弦の 2 乗の平均の分布を作る．観測が参照の 5% 点を下回れば「上位 3 個に感情以外の方向が入る」。"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai6_probe as P
torch.set_num_threads(8)
V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)

def gen(n, p, bts, wt, rng):
    y = rng.integers(0, 4, n)
    b = np.sqrt(np.array(bts) / (1 - np.array(bts)))
    s = np.sqrt(wt / (1 - wt)) if wt > 0 else 0.0
    z = rng.normal(size=n)
    X = {}
    for m in "ab":
        x = rng.normal(size=(n, p)); x[:, :3] += V[y] * b; x[:, 3] += s * z; X[m] = x
    return y, X

def cos2_from_Z(Za, Zb, y):
    """白色化した座標（計量は I）での S3–LDA。"""
    _, Da, _ = P.cca(Za, Zb)
    I = torch.eye(Za.shape[1], dtype=P.DT, device=P.DEV)
    Sg = Za.T @ Za / (len(y) - 1)
    return P.principal_cos2(P.sigma_orth(Da[:, :3], Sg), P.sigma_orth(P._lda_basis(Za, y), Sg), Sg)

def whiten(X, p):
    return P.pca(P.tt(X - X.mean(0)), p)[2]

def reference(Za, Zb, y, rng, reps=40):
    out = []
    n, p = Za.shape
    parts = {}
    for nm, Z in (("a", Za), ("b", Zb)):
        Zn = Z.cpu().numpy()
        mu = np.stack([Zn[y == k].mean(0) for k in range(4)])
        pi = np.bincount(y, minlength=4) / n
        mu = mu - pi @ mu
        SB = (mu * pi[:, None]).T @ mu
        W = np.eye(p) - SB
        ev, U = np.linalg.eigh((W + W.T) / 2); ev = np.clip(ev, 1e-3, None)
        parts[nm] = (mu, U * np.sqrt(ev))
    for _ in range(reps):
        Zs = {nm: P.tt(mu[y] + rng.normal(size=(n, p)) @ L.T) for nm, (mu, L) in parts.items()}
        out.append(cos2_from_Z(whiten(Zs["a"].cpu().numpy(), p), whiten(Zs["b"].cpu().numpy(), p), y))
    return np.array(out)

settings = [("AFEW-like", 632, 13, 12), ("IEMOCAP-like", 2212, 48, 6)]
cases = [((0.25, 0.25, 0.25), 0.0, "前提が成り立つ"), ((0.40, 0.25, 0.12), 0.0, "前提が成り立つ（不ぞろい）"),
         ((0.25, 0.25, 0.25), 0.20, "感情が上位 3 個（共有 0.20）"), ((0.40, 0.25, 0.12), 0.20, "共有が 3 番目より強い"),
         ((0.25, 0.25, 0.25), 0.40, "共有が強い")]
rng = np.random.default_rng(1)
for name, n, p, reps in settings:
    for bts, wt, truth in cases:
        below, obs, q05s = 0, [], []
        for r in range(reps):
            y, X = gen(n, p, bts, wt, rng)
            Za, Zb = whiten(X["a"], p), whiten(X["b"], p)
            o = cos2_from_Z(Za, Zb, y)
            q05 = np.quantile(reference(Za, Zb, y, rng), 0.05)
            below += o < q05; obs.append(o); q05s.append(q05)
        print(f"{name:13s} 間={bts} 中={wt:.2f}（{truth}）| 観測の平均 {np.mean(obs):.3f}，参照の 5% 点の平均 {np.mean(q05s):.3f}，"
              f"観測 < 5% 点（「感情以外が入る」）{below}/{reps}", flush=True)
