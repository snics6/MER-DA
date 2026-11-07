"""gate-code の検算（DAI-6）：AFEW の分割 0 で，台本の関数を直接呼んで退化した場合を確かめる。results/ には書かない。"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai6_probe as P  # noqa: E402
import dai1  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

U = P.ProbeUnit("AFEW", 0)
print("n", U.n, "p", U.p, "speaker/cluster", U.cluster[:3])
m = "A"
E, lam, Z, kept = P.pca(U.Xp[m], U.p)
Bp = E * lam.rsqrt()                                   # 主成分 p 次元（Σ 正規直交）
# (1) 基底を回しても情報の量の行は同じ（罰則が Σ 正規直交の回転で不変）
r1 = P.fit_info(U.Xp[m] @ Bp, U.Xt[m] @ Bp, U.y, U.yt, U.cluster)
g = torch.Generator().manual_seed(0)
Q, _ = torch.linalg.qr(torch.randn(U.p, U.p, generator=g, dtype=torch.float64))
Br = Bp @ Q.to(P.DEV)
r2 = P.fit_info(U.Xp[m] @ Br, U.Xt[m] @ Br, U.y, U.yt, U.cluster)
print(f"(1) 回す前 C={r1['C']} UAR={r1['uar']:.4f} cv={r1['cv_uar']:.4f}；回した後 C={r2['C']} UAR={r2['uar']:.4f} cv={r2['cv_uar']:.4f}")
# (1b) Σ 正規直交にしない基底（主成分の固有ベクトルのまま）では値が変わる（罰則が座標に依ることの確認）
r3 = P.fit_info(U.Xp[m] @ E, U.Xt[m] @ E, U.y, U.yt, U.cluster)
print(f"(1b) 白色化しない主成分の座標 C={r3['C']} UAR={r3['uar']:.4f}")
# (2) 部分空間が主成分の全部なら，射影は恒等（Σ のノルムで外の割合 0）
W = torch.randn(4, U.Xp[m].shape[1], dtype=torch.float64, device=P.DEV)
Bfull = P.sigma_orth(torch.linalg.eigh(U.Sig[m])[1][:, -(U.n - 1):], U.Sig[m])
Pw = P.proj_sigma(W, Bfull, U.Sig[m])
print("(2) 全部の列空間への射影の外の割合", P.sig_norm2(W - Pw, U.Sig[m]) / P.sig_norm2(W, U.Sig[m]), "基底の次元", Bfull.shape[1])
# (3) 手法の形の行の座標の写し方：Bo = 単位行列なら全部の次元の行と同じ解
T, s = "IEMOCAP", "A"
srcs = [x for x in DD.CORPORA if x not in (T, "AFEW")]
Ws, bs = zip(*[dai1.load_source(P.SRC_OUT, x, s) for x in srcs])
W0, b0 = torch.stack(Ws).mean(0), torch.stack(bs).mean(0)
import json
cw, cb = json.loads((ROOT / "DAI" / "frozen" / "dai3_frozen.json").read_text())["oracle_tether"][T][s]
G = S.balanced_onehot(U.y, np.arange(U.n), U.n)[None]
Bo = torch.block_diag(torch.linalg.qr(torch.eye(U.Xp[m].shape[1], dtype=torch.float64, device=P.DEV))[0])
ra = S.fit(U.Xp[m] @ Bo, G, W0 @ Bo, b0, Cw=[cw], Cb=[cb])
rb = S.fit(U.Xp[m], G, W0, b0, Cw=[cw], Cb=[cb])
ua = P.uar_np(U.yt, (U.Xt[m] @ Bo) @ ra["W"][0].T + ra["b"][0]); ub = P.uar_np(U.yt, U.Xt[m] @ rb["W"][0].T + rb["b"][0])
print(f"(3) Bo=I の UAR {ua:.6f}，全部の次元の UAR {ub:.6f}，W の差の最大 {float((ra['W'][0] @ Bo.T - rb['W'][0]).abs().max()):.2e}")
# (3b) QR で直した Bo の行の解が部分空間の中にある（ΔW ∈ S）
B = P.sigma_orth(Bp[:, :5], U.Sig[m])
Bq = torch.linalg.qr(B)[0]
PW0 = P.proj_sigma(W0, B, U.Sig[m])
rc = S.fit(U.Xp[m] @ Bq, G, PW0 @ Bq, b0, Cw=[cw], Cb=[cb])
Wd = rc["W"][0] @ Bq.T
print("(3b) 解を元の座標に戻して部分空間へ射影した差", float((P.proj_sigma(Wd, B, U.Sig[m]) - Wd).abs().max()),
      "；PW0 @ Bq @ Bqᵀ と PW0 の差", float((PW0 @ Bq @ Bq.T - PW0).abs().max()))
# (4) 切片の直し：候補の上で予測の確率の平均が 1/4
bb = P.balance_intercept(U.Xp[m] @ W0.T, b0)
print("(4) 直した後の確率の平均", torch.softmax(U.Xp[m] @ W0.T + bb, 1).mean(0).cpu().numpy())
# (5) ブートストラップの重みがすべて 1 なら合わせた UAR
y = U.yt; pr = np.random.default_rng(0).integers(0, 4, len(y))
codes = pd.factorize(U.cp.df["cluster"].to_numpy()[U.test])[0]
k = codes.max() + 1
n_kc = np.zeros((k, 4)); np.add.at(n_kc, (codes, y), 1)
cor = np.zeros((k, 4)); np.add.at(cor, (codes, y), (pr == y).astype(float))
w = np.ones(k)
print("(5) 重み 1 のブートストラップの UAR", 100 * np.nanmean((w @ cor) / (w @ n_kc)), "直接の UAR", 100 * S.uar(y, pr),
      "テストの塊の数", k, "最大の塊の割合", np.bincount(codes).max() / len(codes))
# (6) 版 (iv) の中心化：塊ごとの平均が 0
Xc = U.centered("A").cpu().numpy()
gm = pd.DataFrame(Xc).groupby(U.center_group).mean().abs().to_numpy().max()
print("(6) 中心化の後の塊ごとの平均の絶対値の最大", gm)
