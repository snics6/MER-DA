"""A1′ の参照を「言葉の側（b）の行を感情の中だけで並べ替える」形（A1 の帰無と同じ並べ替え）で作ったときの較正を，
第 5 版の a1prime（平均を差し込む正規の参照）と同じ人工データで比べる．実データは読まない。"""
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
        Z.append(P.pca(P.tt(x - x.mean(0)), p)[2])
    return y, Z

def a1prime_perm(Za, Zb, y, seed, n_ref=100):
    obs_a, obs_b = P._s3_lda_cos2(Za, Zb, y)
    perms = P.make_perms(len(y), n_ref, y, seed)          # 感情の中だけの並べ替え（A1 の帰無と同じ）
    ra, rb = [], []
    for pm in perms:
        a_, b_ = P._s3_lda_cos2(Za, Zb[torch.as_tensor(pm, device=P.DEV)], y)
        ra.append(a_); rb.append(b_)
    return obs_a < np.quantile(ra, 0.05), obs_b < np.quantile(rb, 0.05)

settings = [("AFEW-like", 632, 13, 20), ("IEMOCAP-like", 2212, 48, 12), ("MELD-like", 5995, 131, 6)]
cases = [((0.25, 0.25, 0.25), 0.0, "前提が成り立つ"), ((0.40, 0.25, 0.12), 0.0, "前提が成り立つ（不ぞろい）"),
         ((0.15, 0.15, 0.15), 0.0, "前提が成り立つ（弱い）"), ((0.25, 0.25, 0.25), 0.40, "共有が強い"),
         ((0.40, 0.25, 0.12), 0.20, "共有が 3 番目より強い")]
rng = np.random.default_rng(3)
for name, n, p, reps in settings:
    for bts, wt, truth in cases:
        pa = pb = 0
        for r in range(reps):
            y, (Za, Zb) = gen(n, p, bts, wt, rng)
            x, z = a1prime_perm(Za, Zb, y, seed=r)
            pa += x; pb += z
        print(f"{name:13s} 間={bts} 中={wt:.2f}（{truth}）| 並べ替えの参照で「離れている」 a の側 {pa}/{reps}，b の側 {pb}/{reps}", flush=True)
