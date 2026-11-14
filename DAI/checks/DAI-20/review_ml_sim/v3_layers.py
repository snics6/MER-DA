"""確率の平均（avg）と確信度で重みづける平均（conf）の raw と adj の層の UAR を人工の数で並べる（CPU．MER-DA のデータは読まない）．"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np
from scipy.special import logsumexp
import dai20_eval as E
P16, S = E.P16, E.S
rng = np.random.default_rng(7)
def probs(Z): return np.exp(Z - logsumexp(Z, axis=-1, keepdims=True))
def mk(y, st, bs):
    Z = rng.normal(0, 1, size=(len(y), 4)) + bs[None]; Z[np.arange(len(y)), y] += st[y]; return probs(Z)
out = []
for rep in range(40):
    share = np.array([0.55, 0.20, 0.10, 0.15]) if rep % 2 else np.full(4, 0.25)
    yp, yt = rng.choice(4, 3000, p=share), rng.choice(4, 3000, p=share)
    st, bs = rng.uniform(0.6, 1.8, size=(3, 4)), rng.normal(0, 0.5, size=(3, 4))
    Pp, Pt = [mk(yp, st[k], bs[k]) for k in range(3)], [mk(yt, st[k], bs[k]) for k in range(3)]
    LMp, LMt = np.stack([P16.centered(p) for p in Pp]), np.stack([P16.centered(p) for p in Pt])
    target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)
    r = dict(skew=rep % 2, prod=100 * S.uar(yt, LMt.sum(0).argmax(1)))
    ct, cp = np.stack([p.max(1) for p in Pt]), np.stack([p.max(1) for p in Pp])
    for v, St, Sp in (("avg", P16.lg(np.mean(Pt, 0)), P16.lg(np.mean(Pp, 0))),
                      ("conf", P16.lg((ct[:, :, None] * np.stack(Pt)).sum(0) / ct.sum(0)[:, None]), P16.lg((cp[:, :, None] * np.stack(Pp)).sum(0) / cp.sum(0)[:, None]))):
        dl, _ = P16.match_shift(Sp, target)
        r[f"{v}_raw"], r[f"{v}_adj"] = 100 * S.uar(yt, St.argmax(1)), 100 * S.uar(yt, (St + dl).argmax(1))
    out.append(r)
import pandas as pd
d = pd.DataFrame(out)
print(d.groupby("skew").mean().round(2).to_string())
print("adj − raw の範囲：avg", (d.avg_adj - d.avg_raw).min().round(2), (d.avg_adj - d.avg_raw).max().round(2), "conf", (d.conf_adj - d.conf_raw).min().round(2), (d.conf_adj - d.conf_raw).max().round(2))
