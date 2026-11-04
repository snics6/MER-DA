"""gate-code の検算（勾配法の手法の退化した場合と束の組み方への依存）。UAR の値は出さず，W·A・b の差とテストの判定の一致だけを出す。
単位：T = MOSEI，c = AFEW，分割 0，種 0（実験 1 の試走 trial3 の段 source と tether_w.json を使う）。生徒 A（と xMUDA 型の A・V・T）。
引数：trial3 の置き場，学習率，反復"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1  # noqa: E402
import dai2  # noqa: E402
import dai2_grad as G  # noqa: E402

d1 = Path(sys.argv[1])
lr, iters = float(sys.argv[2]), int(sys.argv[3])
dai2.RUN["dai1_dir"] = str(d1)
teth, _ = dai2.tether_pseudo(d1, "MOSEI", True)
u = dai1.Unit(d1, "MOSEI", "AFEW", ["IEMOCAP", "MELD"], 0, 0)
pt, ylab, wK = G.kshot(u)
s = "A"
Z, Zt = u.Xp[s].to(G.F32), u.Xt[s]
Zl = Z[pt]
Zs, ys = G.src_feats(u, s)
W0, b0 = (x.to(G.F32) for x in u.start[s])


def run(probs):
    """probs：[(方法, λ, μ, sw, K)]．戻り：W·A（P,4,d），b，記録。"""
    tw = [teth[(s, max(c[4], 1))] for c in probs]
    t0 = time.time()
    out = G.train_map(Z, Zl, ylab, torch.stack([wK[c[4]] for c in probs]), Zs, ys, W0, b0, [x[0] for x in tw], [x[1] for x in tw],
                      [c[0] for c in probs], [c[1] for c in probs], [c[2] for c in probs], [c[3] for c in probs], lr, iters, 0)
    return out, time.time() - t0


def pred(W, b):
    return G.S.logits(Zt, W.to(G.S.DT), b.to(G.S.DT)).argmax(-1).cpu().numpy()


chunk = [("mapctl", 0.0, 1.0, 0.0, 0), ("mapctl", 0.0, 1.0, 0.0, 5), ("dann", 0.0, 1.0, 0.0, 5), ("mme", 0.0, 1.0, 0.0, 5),
         ("dann", 0.1, 1.0, 0.0, 5), ("mme", 0.1, 1.0, 0.0, 5), ("mme", 0.1, 1.0, 0.1, 5)]
(Wa, ba, dg), sec = run(chunk)
print(f"束 7 問：{sec:.1f} 秒，lr={lr}，iters={iters}")
Ws0, bs0 = W0, b0
print("1. mapctl K=0：‖W·A − W0‖_max =", float((Wa[0] - Ws0).abs().max()), " ‖b − b0‖_max =", float((ba[0] - bs0).abs().max()),
      " 出発点との判定の一致 =", float((pred(Wa[0][None], ba[0][None])[0] == pred(Ws0[None], bs0[None])[0]).mean()))
for j, name in ((2, "DANN λ=0"), (3, "MME λ=0,sw=0")):
    print(f"2. {name} と mapctl K=5 の差：W·A の最大差 {float((Wa[j] - Wa[1]).abs().max()):.2e}，b の最大差 {float((ba[j] - ba[1]).abs().max()):.2e}，"
          f"判定の一致 {float((pred(Wa[j][None], ba[j][None])[0] == pred(Wa[1][None], ba[1][None])[0]).mean()):.4f}")
for j, name in ((4, "DANN λ=0.1"), (5, "MME λ=0.1,sw=0"), (6, "MME λ=0.1,sw=0.1")):
    print(f"3. {name}：mapctl との判定の一致 {float((pred(Wa[j][None], ba[j][None])[0] == pred(Wa[1][None], ba[1][None])[0]).mean()):.4f}，"
          f"‖A−I‖_F {dg[j]['anorm']:.3g}，dom {dg[j]['dom_first']:.3g}→{dg[j]['dom_last']:.3g}，ent {dg[j]['ent_first']:.3g}→{dg[j]['ent_last']:.3g}，S {dg[j]['S']:.1f}")
# 束の組み方への依存：DANN λ=0.1 を 1 問だけで解く／別の並びの束で解く
(W1, b1, _), _ = run([chunk[4]])
(W2, b2, _), _ = run([chunk[5], chunk[4], chunk[0]])
for nm, Wx, bx in (("1 問だけ", W1[0], b1[0]), ("3 問の束", W2[1], b2[1])):
    print(f"4. DANN λ=0.1 の {nm} と 7 問の束の差：W·A の最大差 {float((Wx - Wa[4]).abs().max()):.2e}，判定の一致 "
          f"{float((pred(Wx[None], bx[None])[0] == pred(Wa[4][None], ba[4][None])[0]).mean()):.4f}")
(W3, b3, _), _ = run([chunk[4]])
print(f"5. DANN λ=0.1 を同じ 1 問で 2 回：W·A の最大差 {float((W3[0] - W1[0]).abs().max()):.2e}，判定の一致 "
      f"{float((pred(W3, b3)[0] == pred(W1, b1)[0]).mean()):.4f}")
# xMUDA 型 κ=0 と mapctl（設計 2.3 節の 2-C：模倣の重み 0 の問はモダリティごとの mapctl と同じ）
mods = "AVT"
Zm = {m: u.Xp[m].to(G.F32) for m in mods}
Zlm = {m: Zm[m][pt] for m in mods}
cw = {m: [teth[(m, 5)][0]] for m in mods}
cb = {m: [teth[(m, 5)][1]] for m in mods}
t0 = time.time()
o = G.train_xmuda(Zm, Zlm, ylab, wK[5][None], u.start, cw, cb, [0.0], [1.0], lr, iters, 0, u.n)
print(f"xMUDA 型 1 問：{time.time()-t0:.1f} 秒")
print(f"6. xMUDA 型 κ=0 の A と mapctl K=5 の差：W·A の最大差 {float((o['A'][0][0] - Wa[1]).abs().max()):.2e}，b の最大差 "
      f"{float((o['A'][1][0] - ba[1]).abs().max()):.2e}，判定の一致 {float((pred(o['A'][0], o['A'][1])[0] == pred(Wa[1][None], ba[1][None])[0]).mean()):.4f}")
print("GPU の最大メモリ（GB）", torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else "cpu")
