"""第 6 版（cc07f13）の台本の a1prime が，本検査の a1prime_perm_check.py の並べ替えの参照と同じ値を返すかを，人工データで確かめる．実データは読まない。"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp"); sys.path.insert(0, "/home/snics/MER-DA/DAI/checks/DAI-6/review_ml_run")
import dai6_probe as P
torch.set_num_threads(8)
V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
def gen(n, p, bts, wt, rng):
    y = rng.integers(0, 4, n); b = np.sqrt(np.array(bts) / (1 - np.array(bts))); s = np.sqrt(wt / (1 - wt)) if wt > 0 else 0.0
    z = rng.normal(size=n); Z = []
    for m in "ab":
        x = rng.normal(size=(n, p)); x[:, :3] += V[y] * b; x[:, 3] += s * z; Z.append(P.pca(P.tt(x - x.mean(0)), p)[2])
    return y, Z
def mine(Za, Zb, y, seed, n_ref=100):
    obs_a, obs_b = P._s3_lda_cos2(Za, Zb, y); perms = P.make_perms(len(y), n_ref, y, seed); ra, rb = [], []
    for pm in perms:
        a_, b_ = P._s3_lda_cos2(Za, Zb[torch.as_tensor(pm, device=P.DEV)], y); ra.append(a_); rb.append(b_)
    return np.quantile(ra, 0.05), np.quantile(rb, 0.05)
rng = np.random.default_rng(11)
mx = 0.0
for (n, p) in ((632, 13), (2212, 48)):
    for (bts, wt) in (((0.25,) * 3, 0.0), ((0.25,) * 3, 0.4)):
        y, (Za, Zb) = gen(n, p, bts, wt, rng)
        r = P.a1prime(Za, Zb, y, seed=7)
        qa, qb = mine(Za, Zb, y, seed=7)
        mx = max(mx, abs(r["a"]["ref_q05"] - qa), abs(r["b"]["ref_q05"] - qb))
        print(n, p, bts, wt, "台本の 5% 点", round(r["a"]["ref_q05"], 6), round(r["b"]["ref_q05"], 6), "本検査", round(qa, 6), round(qb, 6), "離れている", r["a"]["below_ref_q05"], r["b"]["below_ref_q05"])
print("差の最大", mx)
