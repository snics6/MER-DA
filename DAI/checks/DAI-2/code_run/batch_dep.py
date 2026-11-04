"""gate-code の検算 2：同じ問を束の大きさ（B）だけ変えて解いたときの差（数値の丸めへの依存）。UAR は出さない。
引数：trial3 の置き場，学習率，反復，生徒"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1  # noqa: E402
import dai2  # noqa: E402
import dai2_grad as G  # noqa: E402

d1 = Path(sys.argv[1])
lr, iters, s = float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
dai2.RUN["dai1_dir"] = str(d1)
teth, _ = dai2.tether_pseudo(d1, "MOSEI", True)
u = dai1.Unit(d1, "MOSEI", "AFEW", ["IEMOCAP", "MELD"], 0, 0)
pt, ylab, wK = G.kshot(u)
Z, Zt = u.Xp[s].to(G.F32), u.Xt[s]
Zl = Z[pt]
Zs, ys = G.src_feats(u, s)
W0, b0 = (x.to(G.F32) for x in u.start[s])


def run(probs):
    tw = [teth[(s, max(c[4], 1))] for c in probs]
    return G.train_map(Z, Zl, ylab, torch.stack([wK[c[4]] for c in probs]), Zs, ys, W0, b0, [x[0] for x in tw], [x[1] for x in tw],
                       [c[0] for c in probs], [c[1] for c in probs], [c[2] for c in probs], [c[3] for c in probs], lr, iters, 0)


def pred(W, b):
    return G.S.logits(Zt, W.to(G.S.DT), b.to(G.S.DT)).argmax(-1).cpu().numpy()


def cmp(name, Wa, ba, Wb, bb):
    print(f"{name}：W·A の最大差 {float((Wa - Wb).abs().max()):.2e}（W0 の最大 {float(W0.abs().max()):.2e}），判定の一致 "
          f"{float((pred(Wa[None], ba[None])[0] == pred(Wb[None], bb[None])[0]).mean()):.4f}", flush=True)


ctl = ("mapctl", 0.0, 1.0, 0.0, 5)
dann = ("dann", 0.1, 1.0, 0.0, 5)
mme = ("mme", 0.1, 1.0, 0.0, 5)
filler = [("mapctl", 0.0, 10.0, 0.0, K) for K in (1, 10, 25)]
t0 = time.time()
W7, b7, _ = run([ctl, dann, mme] + filler)
W1c, b1c, _ = run([ctl])
W2c, b2c, _ = run([ctl, filler[0]])
W1d, b1d, _ = run([dann])
W2d, b2d, _ = run([dann, filler[0]])
W1m, b1m, _ = run([mme])
print(f"lr={lr} iters={iters} 生徒 {s}：{time.time()-t0:.0f} 秒")
cmp("mapctl：B=1 と B=6", W1c[0], b1c[0], W7[0], b7[0])
cmp("mapctl：B=2 と B=6", W2c[0], b2c[0], W7[0], b7[0])
cmp("DANN λ=0.1：B=1 と B=6", W1d[0], b1d[0], W7[1], b7[1])
cmp("DANN λ=0.1：B=2 と B=6", W2d[0], b2d[0], W7[1], b7[1])
cmp("MME λ=0.1：B=1 と B=6", W1m[0], b1m[0], W7[2], b7[2])
cmp("参考 DANN λ=0.1（B=6）と mapctl（B=6）", W7[1], b7[1], W7[0], b7[0])
if s in "AVT" and len(s) == 1:
    mods = "AVT"
    Zm = {m: u.Xp[m].to(G.F32) for m in mods}
    Zlm = {m: Zm[m][pt] for m in mods}
    cw = {m: [teth[(m, 5)][0]] * 2 for m in mods}
    cb = {m: [teth[(m, 5)][1]] * 2 for m in mods}
    o = G.train_xmuda(Zm, Zlm, ylab, torch.stack([wK[5], wK[5]]), u.start, cw, cb, [0.0, 1.0], [1.0, 1.0], lr, iters, 0, u.n)
    cmp(f"xMUDA 型 κ=0（B=2）の {s} と mapctl（B=6）", o[s][0][0], o[s][1][0], W7[0], b7[0])
    cmp(f"xMUDA 型 κ=0（B=2）の {s} と mapctl（B=1）", o[s][0][0], o[s][1][0], W1c[0], b1c[0])
    cmp(f"xMUDA 型 κ=1 と κ=0（同じ束）の {s}", o[s][0][1], o[s][1][1], o[s][0][0], o[s][1][0])
print("GPU の最大メモリ（GB）", torch.cuda.max_memory_allocated() / 1e9)
