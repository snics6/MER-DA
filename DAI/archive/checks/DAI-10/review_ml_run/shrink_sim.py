"""ML の査読想定（DAI-10）：縮みの係数の式の検算（人工データ）。
A = (1-α)T + αI，Σ_O = (1-α)S_O + αI，T = S_O + B（候補の全ラベル）のとき
 (1) A⁻¹B の固有値 λ と Σ_O⁻¹B の固有値 λ_O の関係 λ = λ_O/(1 + (1-α)λ_O)
 (2) A⁻¹M = Σ_O⁻¹M·C で，B の固有の向きでの縮みが 1 − (1−α)λ（設計の 1 − λ ではない）
 (3) α = 0.5 で λ > 1 になる例（1 − λ が負になる）
"""
import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, torch
import dai10_probe as P10
import dai8_probe as P8

DT = torch.float64
def make(d, n, sep, seed, pi=(0.37, 0.32, 0.21, 0.10)):
    g = np.random.default_rng(seed)
    y = g.choice(4, size=n, p=pi)
    mu = g.normal(size=(4, d)) * sep / np.sqrt(d)
    L = g.normal(size=(d, d)) / np.sqrt(d)
    X = mu[y] + g.normal(size=(n, d)) @ (np.eye(d) + 0.7 * L)
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    return torch.tensor(X, dtype=DT), torch.tensor(y)

for (d, n, sep) in [(50, 2000, 1.0), (50, 2000, 4.0), (50, 2000, 8.0), (200, 600, 6.0)]:
    X, y = make(d, n, sep, 0)
    N = X.shape[0]
    T = X.T @ X / N
    R = torch.nn.functional.one_hot(y, 4).to(DT)
    for a in (0.01, 0.5, 0.9):
        chA = P10.metric_chol(T, a)
        lam = np.array(P10.emo_eigs(X, y, chA))                    # 台本の量
        fO = P8.lda_from_R(X, R, X.T @ X, a)                         # Σ_O = (1-α)S_O + αI
        lamO = np.array(P10.emo_eigs(X, y, fO["chol"]))              # Σ_O⁻¹B の固有値
        pred = lamO / (1 + (1 - a) * lamO)
        # 縮み：A⁻¹M と Σ_O⁻¹M を B の固有の向きで比べる
        M = fO["mu"]; pi = fO["pi"]
        Mc = M - (pi[:, None] * M).sum(0, keepdim=True)
        AiM = torch.cholesky_solve(Mc.T.contiguous(), chA)          # d×4
        SiM = torch.cholesky_solve(Mc.T.contiguous(), fO["chol"])
        # 4×4 で C を出す：AiM = SiM C
        C = torch.linalg.lstsq(SiM, AiM).solution
        W = torch.diag(pi.sqrt())
        G = W @ Mc @ SiM @ W                                         # Π^{1/2} M Σ_O⁻¹ Mᵀ Π^{1/2}
        ev, U = torch.linalg.eigh((G + G.T) / 2)
        # C を Π^{1/2} の基底で対角化：W C W⁻¹ の固有値
        cval = torch.linalg.eigvals(torch.linalg.pinv(W) @ C @ W).real.sort(descending=False).values.numpy()
        print(f"d={d} n={n} sep={sep} α={a}: λ(A⁻¹B)={np.round(lam,3)}  λ_O/(1+(1−α)λ_O)={np.round(pred,3)}  "
              f"縮み 1−(1−α)λ={np.round(1-(1-a)*lam,3)}  設計の 1−λ={np.round(1-lam,3)}  C の固有値={np.round(cval[:3],3)}")
