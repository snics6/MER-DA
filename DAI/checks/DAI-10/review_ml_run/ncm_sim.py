"""ML の査読想定（DAI-10）M1：LDA_T − LDA_K の符号だけでは，ラベル無しの計量が等方の計量（Σ = I，最近傍の平均）より良いかは分からない例（人工データ）。
感情の平均は 3 次元の部分空間，感情でない分散は nuis 次元の強い成分（話者などの代わり）と等方の雑音。候補の統計で標準化する。
"""
import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, torch
import dai10_probe as P10
import dai8_probe as P8
import dai1_stats as ST
DT = torch.float64

def uar(y, p):
    return 100 * np.mean([np.mean(p[y == k] == k) for k in range(4)])

def run(d, Np, Nt, sep, nuis, nuis_sd, K, seed):
    g = np.random.default_rng(seed)
    Q = np.linalg.qr(g.normal(size=(d, d)))[0]
    mu = np.zeros((4, d)); mu[:, :3] = g.normal(size=(4, 3)) * sep
    mu = mu @ Q.T
    sd = np.ones(d); sd[3:3 + nuis] = nuis_sd
    def draw(n):
        y = g.choice(4, size=n, p=(0.37, 0.32, 0.21, 0.10))
        Z = g.normal(size=(n, d)) * sd
        return mu[y] + Z @ Q.T, y
    Xp, yp = draw(Np); Xt, yt = draw(Nt)
    m, s = Xp.mean(0), Xp.std(0, ddof=1)
    Xp, Xt = (Xp - m) / s, (Xt - m) / s
    Xp, Xt = torch.tensor(Xp, dtype=DT), torch.tensor(Xt, dtype=DT)
    kpos = np.concatenate([g.permutation(np.flatnonzero(yp == k))[:K] for k in range(4)])
    XL, yl = Xp[kpos], torch.tensor(yp[kpos])
    muK = P10.class_means(XL, yl)
    T = Xp.T @ Xp / Np
    out = {}
    out["LDA_K"] = P8.delta(Xt, P8.lda_from_R(XL, torch.nn.functional.one_hot(yl, 4).to(DT), XL.T @ XL, 0.5)).argmax(1).numpy()
    out["LDA_T"] = P8.delta(Xt, P10.lda_metric(muK, P10.metric_chol(T, 0.5))).argmax(1).numpy()
    out["NCM"] = P8.delta(Xt, P10.lda_metric(muK, torch.eye(d, dtype=DT))).argmax(1).numpy()
    fO = P8.lda_from_R(Xp, torch.nn.functional.one_hot(torch.tensor(yp), 4).to(DT), Xp.T @ Xp, 0.5)
    out["LDA_SO"] = P8.delta(Xt, P10.lda_metric(muK, fO["chol"])).argmax(1).numpy()
    return {k: uar(yt, v) for k, v in out.items()}

if __name__ == "__main__":
      for (sep, nuis, nuis_sd, lab) in [(2.0, 0, 1.0, "感情でない強い分散なし"), (1.0, 20, 6.0, "感情でない強い分散 20 次元")]:
        for K in (5, 25):
            R = [run(300, 1500, 1500, sep, nuis, nuis_sd, K, s) for s in range(10)]
            mean = {k: np.mean([r[k] for r in R]) for k in R[0]}
            print(f"{lab}，K={K}（10 回の平均）：" + "，".join(f"{k} {v:.1f}" for k, v in mean.items()) +
                  f"｜LDA_T−LDA_K {mean['LDA_T']-mean['LDA_K']:+.1f}，LDA_T−NCM {mean['LDA_T']-mean['NCM']:+.1f}，LDA_K−NCM {mean['LDA_K']-mean['NCM']:+.1f}，LDA_SO−LDA_K {mean['LDA_SO']-mean['LDA_K']:+.1f}，LDA_SO−NCM {mean['LDA_SO']-mean['NCM']:+.1f}")
