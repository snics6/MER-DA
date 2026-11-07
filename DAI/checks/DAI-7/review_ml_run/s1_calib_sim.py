"""dai-review-ml の再検査（v2）の M4 の確かめ：情報が同じで，none の単独の分類器だけに打ち消し合う偏りがあるとき，
S = (Ḡ − G_full) + ΔL_dep + ΔL_cal の第 1 項 S1 がどう出るか．実データは読まない．台本の関数をそのまま呼ぶ．"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np
import dai7_probe_v2 as P

def run(seed, beta, n=3000, s=1.2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    mu = s * np.eye(4)
    info = {m: mu[y] + rng.normal(size=(n, 4)) for m in P.MODS}            # 3 つのモダリティの情報（joint と none で同じ）
    bias = {"A": beta * np.array([1., -1, 0, 0]), "V": beta * np.array([-1., 1, 0, 0]), "T": np.zeros(4)}   # 積では打ち消し合う
    sets = {"joint": {m: info[m] for m in P.MODS}, "none": {m: info[m] + bias[m] for m in P.MODS}}
    half = P.split_halves(y, np.array(["one"] * n))[0]
    u = {}
    for k, Z in sets.items():
        z9 = np.concatenate([Z[m] @ P.H for m in P.MODS], 1)
        d = {m: P.uar(y, Z[m].argmax(1)) for m in P.MODS}
        d["prod"] = P.uar(y, sum(Z.values()).argmax(1))
        d["blk"] = P.uar(y, P.crossfit(z9, y, half, True))
        d["full"] = P.uar(y, P.crossfit(z9, y, half, False))
        for i, m in enumerate(P.MODS):                                       # 直し方の案：各モダリティの 3 次元の判別分析（感情の事前を等しく）
            d[f"cal_{m}"] = P.uar(y, P.crossfit(z9[:, 3 * i:3 * i + 3], y, half, False))
        u[k] = d
    dec = P.decompose(u["joint"], u["none"])
    Gcal = np.mean([u["joint"][f"cal_{m}"] - u["none"][f"cal_{m}"] for m in P.MODS])
    return dec, Gcal

for beta in (0.0, 1.5, 3.0):
    for seed in range(3):
        dec, Gcal = run(seed, beta)
        print(f"beta={beta} seed={seed}  Gbar={dec['Gbar']:+.2f}  G_prod={dec['G_prod']:+.2f}  G_full={dec['G_full']:+.2f}  "
              f"S={dec['S']:+.2f}  S1={dec['S1']:+.2f}  dLdep={dec['dLdep']:+.2f}  dLcal={dec['dLcal']:+.2f}  "
              f"| Gbar-Gcal={dec['Gbar'] - Gcal:+.2f}  Gcal-G_full={Gcal - dec['G_full']:+.2f}")
