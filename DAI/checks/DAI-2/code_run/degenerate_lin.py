"""gate-code の検算（線形の手法，I7・I13）：CPU で 1 単位（T=MOSEI，c=AFEW，分割 0，種 0），生徒 A，K=5。UAR は出さない。
1. fit_nc で追加の項の重み 0（mass = 0）の解が dai_solver.fit の none[K] と同じ
2. entmin・IM の勾配が有限差分と合う（目的関数の実装）
3. 保存した lin の csv の S の列が設計の式（self：4K + ρ × 残った本数，own：a·4K + ρn，lp：4K + n，entmin・IM：4K + ρn）と合う
引数：trial3 の置き場，lin の csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1  # noqa: E402
import dai2  # noqa: E402
import dai_solver as S  # noqa: E402

d1 = Path(sys.argv[1])
teth, _ = dai2.tether_pseudo(d1, "MOSEI", True)
u = dai1.Unit(d1, "MOSEI", "AFEW", ["IEMOCAP", "MELD"], 0, 0)
s, K = "A", 5
X, (W0, b0) = u.Xp[s], u.start[s]
cw, cb = teth[(s, K)]
gk = u.gK(K)
rn = S.fit(X, gk[None], W0, b0, Cw=cw, Cb=cb)
r0 = dai2.fit_nc(X, gk[None].expand(2, -1, -1), W0, b0, cw, cb, W0, b0, [0.0, 0.0], [1.0, 0.1], [0.0, 1.0])
print("1. mass=0 の fit_nc と none[K]：W の最大差", float((r0["W"] - rn["W"][0]).abs().max()), "（W の最大", float(rn["W"].abs().max()), "）勾配の最大", r0["grad_max"])
# 2. 有限差分（IM，α=0.3，ρ=1）
n = u.n
mass = n * 1.0
Wn, bn = rn["W"][0], rn["b"][0]


def obj(W, b, alpha, beta):
    Z = X @ W.T + b
    lp = torch.log_softmax(Z, 1)
    p = lp.exp()
    nll = -(gk * lp).sum()
    cond = -(p * lp).sum(1).mean()
    pb = p.mean(0)
    marg = -(pb * pb.log()).sum()
    reg = ((W - W0) ** 2).sum() / (2 * cw) + ((b - b0) ** 2).sum() / (2 * cb)
    return (nll + mass * (alpha * cond - beta * marg) + reg) / (gk.sum() + mass)


r1 = dai2.fit_nc(X, gk[None], W0, b0, cw, cb, Wn, bn, [mass], [0.3], [1.0])
Wf, bf = r1["W"][0], r1["b"][0]
print("2. IM α=0.3 ρ=1：目的関数（台本）", float(r1["f_p"][0]), "（独立に書いた式）", float(obj(Wf, bf, 0.3, 1.0)), "勾配の最大", r1["grad_max"], "反復", r1["n_iter"])
Wv = Wf.clone().requires_grad_(True)
bv = bf.clone().requires_grad_(True)
f = obj(Wv, bv, 0.3, 1.0)
g, = torch.autograd.grad(f, Wv)
print("   独立の式での停留点の勾配の最大", float(g.abs().max()))
# 3. S の列
d = pd.read_csv(sys.argv[2], low_memory=False)
nn_ = int(d.n_pseudo.dropna().iloc[0]) if False else None
sr = d[(d.method == "_self_round") & (d["round"] == 3)]
se = d[d.method == "self"].drop(columns=["n_pseudo"]).merge(sr[["student", "K", "tau", "rho", "n_pseudo"]], on=["student", "K", "tau", "rho"])
print("3. self の S − (4K + ρ × 残った本数) の最大絶対", float((se.S - (4 * se.K + se.rho * se.n_pseudo)).abs().max()), "行", len(se))
npool = len(u.pool)
ow = d[d.method == "own"]
print("   own の S − (a·4K + ρn) の最大絶対", float((ow.S - (ow.a * 4 * ow.K + ow.rho * npool)).abs().max()))
lp = d[d.method == "lp"]
print("   lp の S − (4K + n) の最大絶対", float((lp.S - (4 * lp.K + npool)).abs().max()))
nc = d[d.method.isin(["entmin", "im"])]
print("   entmin・IM の S − (4K + ρn) の最大絶対", float((nc.S - (4 * nc.K + nc.rho * npool)).abs().max()))
sa = d[d.method == "saerens"]
print("   saerens の π の和 − 1 の最大絶対", float((sa[["pi_0", "pi_1", "pi_2", "pi_3"]].sum(1) - 1).abs().max()))
