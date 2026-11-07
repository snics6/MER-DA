"""人工データで，設計 3.4.1 の A1 の読み方の規則（台本 summarize の 682-683 行）がどう振る舞うかを数える．
実データは読まない．母集団の値：感情の間の正準相関（3 個とも bt），感情の中の共有の正準相関（1 個，wt）．"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai6_probe as P
torch.set_num_threads(8)
P.N_PERM_DECOMP = int(sys.argv[1]) if len(sys.argv) > 1 else 200

def gen(n, p, bt, wt, rng):
    y = rng.integers(0, 4, n)
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)  # 正四面体，(1/4)Σ v vᵀ = I
    b2 = bt / (1 - bt)            # 感情の間の分散 b² で正準相関 b²/(1+b²) = bt
    s2 = wt / (1 - wt)            # 共有の要因の分散 s² で s²/(1+s²) = wt
    z = rng.normal(size=n)
    X = {}
    for m in "ab":
        x = rng.normal(size=(n, p))
        x[:, :3] += np.sqrt(b2) * V[y]
        x[:, 3] += np.sqrt(s2) * z
        X[m] = x
    return y, X

def label(row):
    if row["sv_within_1"] <= row["wnull_q95_sv_within_1"]:
        return "holds"
    return "broken_top3_emotion" if row["sv_within_1"] < row["sv_between_3"] else "broken_within_stronger"

settings = [("AFEW-like", 632, 13, 10), ("IEMOCAP-like", 2212, 48, 6), ("MELD-like", 5995, 131, 3), ("MOSEI-like", 14907, 200, 2)]
cases = [(0.25, 0.0), (0.25, 0.10), (0.25, 0.20), (0.35, 0.20), (0.35, 0.30)]
rng = np.random.default_rng(0)
for name, n, p, reps in settings:
    for bt, wt in cases:
        labs, s3, w1, wq = [], [], [], []
        for r in range(reps):
            y, X = gen(n, p, bt, wt, rng)
            Za, Zb = (P.tt(X[m] - X[m].mean(0)) for m in "ab")
            row = P.decomp_row(Za, Zb, y, {}, seed=r)
            labs.append(label(row)); s3.append(row["sv_between_3"]); w1.append(row["sv_within_1"]); wq.append(row["wnull_q95_sv_within_1"])
        u, c = np.unique(labs, return_counts=True)
        print(f"{name:13s} n={n:5d} p={p:3d} 母集団：間の3番目={bt:.2f} 中の1番目={wt:.2f} | 標本：間3={np.mean(s3):.3f} 中1={np.mean(w1):.3f} 中1の帰無95%={np.mean(wq):.3f} | 判定 {dict(zip(u, c))}", flush=True)
