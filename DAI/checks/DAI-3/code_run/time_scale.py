"""gate-code の所要の見積もり：本番の大きさの人工の問で refit_item（MOSEI：テスト 3,727 本，50 群の複製 51．MELD：2,105 本，複製 67）と
再現できる度合いの学習（候補 14,907 本，次元 1,024・2,304，問 44・8）の時間を測る．値は人工．"""
import sys, time, math
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, torch
import dai3 as D3, dai_solver as S
rng = np.random.default_rng(0)
def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
for name, n, J in (("MOSEI", 3727, 51), ("MELD", 2105, 67), ("AFEW", 158, 15), ("IEMOCAP", 731, 2)):
    y = rng.integers(0, 4, n); cl = np.array([f"c{i % (J-1):03d}" for i in range(n)]); sid = np.array([f"s{i:05d}" for i in range(n)])
    Z = S.t(rng.normal(size=(n, 4)) * 2 + np.eye(4)[y] * 1.5); b = S.t(np.zeros(4))
    grp = D3.test_halves(y, cl, sid); keys, keep = D3.rep_keep(cl, "cluster", None)
    D3.refit_item(Z, b, y, grp, keep); sync()
    t0 = time.time()
    for _ in range(3):
        R, cv = D3.refit_item(Z, b, y, grp, keep)
    sync(); print(name, "refit_item 1 回", f"{(time.time()-t0)/3:.2f} 秒", "問", cv["n_problems"], "GPU 最大", f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
for n, d, P in ((14907, 1024, 44), (14907, 2304, 8), (9162, 1024, 44), (2212, 1024, 44)):
    X = S.t(rng.normal(size=(n, d)).astype(np.float32)); W0 = S.t(rng.normal(size=(4, d)) * 0.01); b0 = S.t(np.zeros(4))
    qy = rng.integers(0, 4, (P, n)); half = np.arange(n) < n // 2
    G = torch.stack([S.balanced_onehot(qy[p], np.flatnonzero(half if p % 2 == 0 else ~half), n) for p in range(P)])
    sync(); t0 = time.time()
    r = S.fit(X, G, W0, b0, Cw=0.01, Cb=1.0); sync()
    print("再現の学習", n, d, P, f"{time.time()-t0:.1f} 秒", "反復", r["n_iter"], "GPU 最大", f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    del X, G, r; torch.cuda.empty_cache()
