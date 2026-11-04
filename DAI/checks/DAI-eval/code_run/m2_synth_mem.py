"""gate-code（評価の回の起動の台本）：評価の回の最も大きい問と同じ形の乱数のデータで，GPU のメモリの最大を測る．
本物の特徴もラベルも使わない（評価のターゲットのデータに触れない）．反復は数十回で打ち切る（1 反復のメモリは反復の数に依らない）．"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
import numpy as np
import torch
import dai_solver as S
import dai2, dai2_grad as G

GB = 1024 ** 3
g = torch.Generator(device=S.DEV).manual_seed(0)


def base():
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def report(name, b0, t0):
    print(f"{name}: 最大の確保 {(torch.cuda.max_memory_allocated()-b0)/GB:.2f} GB（入力を含む），{time.time()-t0:.0f} 秒", flush=True)


d = 2304
# 1. 実験 1 の段 3：AVT の生徒，MOSEI の候補 14,907 本，問 113 + 温度の項（solve_items）
n, P = 14907, 113
b0 = base(); t0 = time.time()
X = torch.randn(n, d, device=S.DEV, dtype=S.DT, generator=g)
Q = torch.softmax(torch.randn(P, n, 4, device=S.DEV, dtype=S.DT, generator=g), -1)
r = S.fit(X, Q, torch.zeros(4, d), torch.zeros(4), Cw=0.1, Cb=1.0, G2=Q * 0.5, T2=np.full(P, 2.0), max_iter=20)
report("実験 1 段 3 の束（AVT，n=14,907，問 113，温度の項つき）", b0, t0)
del X, Q, r
# 2. 実験 2 の S+T：ソース 3 つ（最大は T=AFEW：IEMOCAP 2,943 ＋ MOSEI 18,634 ＋ MELD 11,267）＋ K 本 100，問 13
n = 2943 + 18634 + 11267 + 100
b0 = base(); t0 = time.time()
X = torch.randn(n, d, device=S.DEV, dtype=S.DT, generator=g)
Gm = torch.softmax(torch.randn(13, n, 4, device=S.DEV, dtype=S.DT, generator=g), -1)
r = S.fit(X, Gm, torch.zeros(4, d), torch.zeros(4), Cw=0.01, Cb=float("inf"), max_iter=20)
report(f"実験 2 の S+T（AVT，n={n}，問 13）", b0, t0)
del X, Gm, r
# 3. 実験 2 のラベル伝播のグラフ（MOSEI の候補 14,907 本，k = 50）
n = 14907
b0 = base(); t0 = time.time()
X = torch.randn(n, d, device=S.DEV, dtype=S.DT, generator=g)
Wm, dm = dai2.knn_graph(X, 50)
report("実験 2 の k 近傍のグラフ（n=14,907，k=50）", b0, t0)
del X, Wm, dm
# 4. 実験 2 の DANN・MME・対照の束（AVT，束 12 問，候補 14,907，ソース 32,844．反復 10）
n, ns = 14907, 2943 + 18634 + 11267
b0 = base(); t0 = time.time()
Z = torch.randn(n, d, device=S.DEV, dtype=torch.float32, generator=g)
Zs = torch.randn(ns, d, device=S.DEV, dtype=torch.float32, generator=g)
ys = torch.randint(0, 4, (ns,), device=S.DEV, generator=g)
ylab = torch.randint(0, 4, (100,), device=S.DEV, generator=g)
wl = torch.ones(12, 100, device=S.DEV)
meth = ["dann"] * 4 + ["mme"] * 4 + ["mapctl"] * 4
W0, b0_ = torch.zeros(4, d, device=S.DEV), torch.zeros(4, device=S.DEV)
out = G.train_map(Z, Z[:100], ylab, wl, Zs, ys, W0, b0_, [0.1] * 12, [1.0] * 12, meth, [0.1] * 12, [1.0] * 12, [0.1] * 12, 1e-3, 10, 0,
                  Z[:2000], Zs[:2000])
report("実験 2 の map の束（AVT，12 問，反復 10）", b0, t0)
