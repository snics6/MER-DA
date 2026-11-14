"""関門 1（定義）第 3 版の確かめ．読むだけの計算．
1. 3 ソースの残り方 ret_m が 1 を超えるか（`prop_home` の「残り方で相性を縮めた分」の向き）．ソースのラベルだけの表から．
2. 格子の点 12（n0 = inf，β = 0，戻し先 eq）が積と同じ判定になるか（人工の数．raw と adj）．
3. 相性を使わない点 0（n0 = 0，β = 0）が水準だけの重み M_F2_K と同じ判定になるか（人工の数）．
4. 重みの表を探す版で，同点の表を平均した表の K 本の UAR が最大の値を下回る割合（人工の数）．
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai16_probe as P16  # noqa: E402
import dai15  # noqa: E402
import dai_solver as S  # noqa: E402

# ---- 1
d = pd.read_csv("/home/snics/MER-DA/results/dai16_probe/summary/reliability_src.csv")
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
H = {(r.source, r.modality): np.array([r.d_0, r.d_1, r.d_2, r.d_3]) for r in d.itertuples() if r.where == "home"}
A = {(r.source, r.where.split("→")[1], r.modality): np.array([r.d_0, r.d_1, r.d_2, r.d_3]) for r in d.itertuples() if r.where != "home"}
for T in C:
    o = [x for x in C if x != T]
    pairs = [(s, t) for s in o for t in o if s != t]
    ret = {m: float(np.clip(sum(A[(s, t, m)].sum() for s, t in pairs) / sum(H[(s, m)].sum() for s, t in pairs), 0, 1.5)) for m in "AVT"}
    print("1.", T, {k: round(v, 3) for k, v in ret.items()})

# ---- 2, 3
rng = np.random.default_rng(1)
GRID = P16.GRID
assert GRID[12] == (float("inf"), 0.0, "eq") and GRID[0] == (0.0, 0.0, "eq")
bad12, bad0 = 0, 0
for it in range(500):
    n = 200
    P = rng.dirichlet(np.ones(4) * 0.6, size=(3, n))
    LM = np.stack([P16.centered(p) for p in P])
    Pt = rng.dirichlet(np.ones(4) * 0.6, size=(3, 150))
    LMt = np.stack([P16.centered(p) for p in Pt])
    DK, De = rng.uniform(0.0, 2.5, size=(3, 4)), rng.uniform(0.0, 2.5, size=(3, 4))
    K = int(rng.choice([5, 10, 25]))
    prod_p = P16.norm_log(LM.sum(0))
    target = np.exp(prod_p).mean(0)
    prod_t = P16.norm_log(LMt.sum(0)).argmax(1)
    # 点 12
    W = P16.to_w(P16.mix_rel(DK, De, 4 * K, *GRID[12]))
    Sp, St = P16.combine("F6", LM, W), P16.combine("F6", LMt, W)
    dl, _ = P16.match_shift(Sp, target)
    if not ((St.argmax(1) == prod_t).all() and ((St + dl).argmax(1) == prod_t).all()):
        bad12 += 1
    # 点 0 と M_F2_K
    W0 = P16.to_w(P16.mix_rel(DK, De, 4 * K, *GRID[0]))
    WK = P16.to_w(DK)
    S0p, S0t = P16.combine("F6", LM, W0), P16.combine("F6", LMt, W0)
    S2p, S2t = P16.combine("F2", LM, WK), P16.combine("F2", LMt, WK)
    d0, _ = P16.match_shift(S0p, target)
    d2, _ = P16.match_shift(S2p, target)
    if not (((S0t + d0).argmax(1) == (S2t + d2).argmax(1)).all() and (S0t.argmax(1) == S2t.argmax(1)).all()):
        bad0 += 1
print("2. 点 12 と積の判定が 1 本でも違った組：", bad12, "/ 500")
print("3. 点 0 と M_F2_K の判定が 1 本でも違った組：", bad0, "/ 500")

# ---- 4（台本の wf_search と同じ式を numpy で．表は台本の WS と同じ作り方）
SEED = 20261114 + 20
N_WF = 1000
WS = np.concatenate([np.ones((1, 3, 4)), 3.0 * np.random.default_rng([SEED, 1]).dirichlet(np.ones(3), size=(N_WF, 4)).transpose(0, 2, 1)])


def kshot_uar(Wt, LMh, yk, fold):
    pred = np.empty(len(yk), int)
    for j in range(5):
        te = np.flatnonzero(fold == j)
        pred[te] = P16.norm_log(np.einsum("ky,kny->ny", Wt, LMh[:, te])).argmax(1)   # 切片は省いた近似（同点の起こり方を見るため）
    return 100 * S.uar(yk, pred)


below, ntie = [], []
for it in range(40):
    K = 5
    yk = np.repeat(np.arange(4), K)
    fold = dai15.kfold_of(yk)
    Ph = rng.dirichlet(np.ones(4), size=(3, 4 * K))
    Ph[:, np.arange(4 * K), yk] += rng.uniform(0.1, 0.6, size=(3, 1))
    Ph /= Ph.sum(-1, keepdims=True)
    LMh = np.stack([P16.centered(p) for p in Ph])
    ua = np.array([kshot_uar(WS[r], LMh, yk, fold) for r in range(101)])
    tied = np.flatnonzero(ua >= ua.max() - 1e-9)
    ntie.append(len(tied))
    below.append(kshot_uar(WS[tied].mean(0), LMh, yk, fold) < ua.max() - 1e-9)
print("4. K=5，候補 101：同点の数の中央値", int(np.median(ntie)), "，平均した表の K 本の UAR が最大を下回った割合", float(np.mean(below)))
