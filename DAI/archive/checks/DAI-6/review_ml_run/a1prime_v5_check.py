"""第 5 版の台本の a1prime（単位ごとの参照の分布）を，そのまま人工データに当てて，前提が成り立つときの「離れている」の割合と，
共有が強いときの検出の割合を数える．実データは読まない。"""
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
    Z = []
    for m in "ab":
        x = rng.normal(size=(n, p)); x[:, :3] += V[y] * b; x[:, 3] += s * z
        Z.append(P.pca(P.tt(x - x.mean(0)), p)[2])           # summarize と同じく白色化した主成分の座標
    return y, Z

settings = [("AFEW-like", 632, 13, 20), ("IEMOCAP-like", 2212, 48, 12), ("MELD-like", 5995, 131, 6), ("MOSEI-like", 14907, 200, 4)]
cases = [((0.25, 0.25, 0.25), 0.0, "前提が成り立つ"), ((0.40, 0.25, 0.12), 0.0, "前提が成り立つ（不ぞろい）"),
         ((0.15, 0.15, 0.15), 0.0, "前提が成り立つ（弱い）"), ((0.25, 0.25, 0.25), 0.40, "共有が強い")]
rng = np.random.default_rng(3)
for name, n, p, reps in settings:
    for bts, wt, truth in cases:
        ba = bb = 0
        for r in range(reps):
            y, (Za, Zb) = gen(n, p, bts, wt, rng)
            res = P.a1prime(Za, Zb, y, seed=r, n_ref=100)
            ba += res["a"]["below_ref_q05"]; bb += res["b"]["below_ref_q05"]
        print(f"{name:13s} 間={bts} 中={wt:.2f}（{truth}）| 「離れている」 a の側 {ba}/{reps}，b の側 {bb}/{reps}", flush=True)
