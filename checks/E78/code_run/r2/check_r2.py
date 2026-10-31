"""E78 関門 1 の再検査（957fa35）：DANN の 1 設定を単独と束で解いたときの一致，所要の測り直し（DANN・MME・対照の種 0 の全候補，CORAL）。
種類 audio_wavlm，評価のターゲット RAVDESS の仮のターゲット AFEW（反復 0）．評価のターゲットの値は使わない。引数：parts（例 batch,map,coral）"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "exp")); sys.path.insert(0, str(ROOT / "measure")); sys.path.insert(0, str(Path(__file__).resolve().parent))
torch.cuda.set_per_process_memory_fraction(0.1)
torch.backends.cuda.matmul.allow_tf32 = True
import compare_dai2_957fa35 as E  # noqa: E402
E.ROOT = ROOT
import baselines_distill as BD  # noqa: E402
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from baselines_distill import Prob  # noqa: E402
from geom_e0 import settings  # noqa: E402
from teach_student39 import Student  # noqa: E402
from data import groups_for  # noqa: E402

parts = set((sys.argv[1] if len(sys.argv) > 1 else "batch,map,coral").split(","))
kind, T, r, c = "audio_wavlm", "RAVDESS", 0, "AFEW"
DEV = E.DEV
t0 = time.perf_counter()
data = {g: M.load(g, kind) for g in groups_for(kind)}
S = [s for s in data if CM.corp(s) != CM.corp(T) and CM.corp(s) != CM.corp(c)]
gamma, rho_s, rho_m, cb, cp, _ = settings(kind, T)
C = E.C_FIT
src_pc, lin, n_of = M5.pool_info(data, S, C, {})
ev = data[c]
ctx = CB.Ctx(ev, [r], list(E.KS_LAB), center="ind")
pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
ct = ctx.ct[r]
Xu, te_X = M.rows(ev.X, pool) - ct, M.rows(ev.X, te) - ct
srcs = {s: (data[s].X, data[s].y) for s in S}
yq, yte = ev.y[pool], ev.y[te]
label_sets = BD.label_sets_of(ctx, r, pool)
st = Student(ev, Xu, src_pc, lin, n_of, S, C, gamma, rho_s, [rho_m])
Am = st.mats[rho_m][0]
s = {"st": st, "rho_m": rho_m, "cb": cb, "cp": cp, "Z": st.Xstd @ Am, "Zte": ((te_X - st.src.mu) / st.src.sd) @ Am}
Zs, ys = E.src_feats(st, rho_m, srcs)
pr = Prob(s, yq, label_sets)
lab = E.lab_of(label_sets, yq)
pos, ylab, wK = E.lab_tensors(lab)
I = torch.eye(pr.d, dtype=E.F32, device=DEV)
args = (s["Z"].to(E.F32), s["Z"].to(E.F32)[pos], ylab, wK, Zs.to(E.F32), torch.as_tensor(ys, device=DEV), pr.W0.to(E.F32), pr.b0.to(E.F32), float(cp), cb)
print(f"準備 {time.perf_counter() - t0:.0f} 秒．n={pr.n}，ソース {S}")

if "batch" in parts:
    probe = ("dann", 1.0, 1.0, 0.0, "5")
    fill = [x for x in [("dann", lam, m_, 0.0, K) for lam in E.MAP_LAMS for m_ in E.MAP_MUS for K in E.KS] if x != probe]
    mme = [("mme", lam, m_, sw, K) for lam in E.MAP_LAMS for m_ in E.MAP_MUS for sw in E.MME_SRCW for K in E.KS][:29]
    for lr, iters in ((1e-2, 200), (1e-3, 1000)):
        alone = E._train_map(*args, [probe], lr, iters, 0, I)[0]
        alone2 = E._train_map(*args, [probe], lr, iters, 0, I)[0]
        inb = E._train_map(*args, [probe] + fill[:29], lr, iters, 0, I)[0]
        inb2 = E._train_map(*args, fill[:10] + [probe] + mme[:9], lr, iters, 0, I)        # 位置と同じ束のほかの問を替える
        W = lambda o: torch.einsum("ke,ed->kd", o[0], torch.eye(o[0].shape[1], device=DEV)) if False else o[0]  # noqa: E731
        print(f"DANN λ=1 μ=1 K=5 lr={lr} it={iters}：単独 2 回の差 {float((alone[0] - alone2[0]).abs().max()):.2e}，"
              f"単独と束 30（先頭）の差 {float((alone[0] - inb[0]).abs().max()):.2e}，単独と束 20（11 番目，MME と同じ束）の差 {float((alone[0] - inb2[10][0]).abs().max()):.2e}，"
              f"|W − W0| {float((alone[0] - pr.W0.to(E.F32)).abs().max()):.2e}")
        p_a = (s["Zte"].float() @ alone[0].T + alone[1]).argmax(1)
        p_b = (s["Zte"].float() @ inb2[10][0].T + inb2[10][1]).argmax(1)
        print(f"   予測の食い違い（単独と束 20）{int((p_a != p_b).sum())}/{len(yte)}")
    for seed in (1, 2):
        o1 = E._train_map(*args, [probe], 1e-2, 200, seed, I)[0]
        o0 = E._train_map(*args, [probe], 1e-2, 200, 0, I)[0]
        print(f"   種 {seed} と種 0 の W の差 {float((o1[0] - o0[0]).abs().max()):.2e}（種で値が変わることの確かめ）")

if "tf32" in parts:
    probe = ("dann", 1.0, 1.0, 0.0, "5")
    fill = [x for x in [("dann", lam, m_, 0.0, K) for lam in E.MAP_LAMS for m_ in E.MAP_MUS for K in E.KS] if x != probe]
    ctl = ("mapctl", 0.0, 1.0, 0.0, "5")
    for tf in (True, False):
        torch.backends.cuda.matmul.allow_tf32 = tf
        for pb in (probe, ctl):
            a1 = E._train_map(*args, [pb], 1e-2, 200, 0, I)[0]
            a2 = E._train_map(*args, [pb, pb], 1e-2, 200, 0, I)[0]
            a30 = E._train_map(*args, [pb] + fill[:29], 1e-2, 200, 0, I)[0]
            print(f"TF32={tf} {pb[0]}：B=1 と B=2 の差 {float((a1[0] - a2[0]).abs().max()):.2e}，B=1 と B=30 の差 {float((a1[0] - a30[0]).abs().max()):.2e}")
    # 種 0 の DANN λ=1 に 1e-6 の乱れを入れると解がどれだけ動くか（初期値への敏感さ）
    torch.backends.cuda.matmul.allow_tf32 = True
    W0p = args[6] + 1e-6 * torch.randn_like(args[6])
    args_p = args[:6] + (W0p,) + args[7:]
    o = E._train_map(*args, [probe] + fill[:29], 1e-2, 200, 0, I)[0]
    op = E._train_map(*args_p, [probe] + fill[:29], 1e-2, 200, 0, I)[0]
    print(f"DANN λ=1：W0 に 1e-6 の乱れを足したときの W の差 {float((o[0] - op[0]).abs().max()):.2e}")

if "map" in parts:
    for meths in (("dann",), ("mme",), ("mapctl",)):
        rec = E.Rec({"pseudo": c, "rep": r}, yte, keep=False)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        E.run_map(pr, Zs, ys, rec, kind, lab, E.map_problems(meths, (0,)))
        torch.cuda.synchronize()
        print(f"run_map {meths[0]}（種 0 の全候補 {len(rec.rows)} 問）：{time.perf_counter() - t1:.1f} 秒")

if "coral" in parts:
    rec = E.Rec({"pseudo": c, "rep": r}, yte, keep=False)
    rec.conv = []
    torch.cuda.synchronize(); t1 = time.perf_counter()
    E.run_coral(ev, Xu, te_X, src_pc, srcs, n_of, S, C, gamma, rho_s, label_sets, yq, rec, kind, lin, ct, cp, cb)
    torch.cuda.synchronize()
    d = pd.DataFrame(rec.rows)
    print(f"run_coral：{time.perf_counter() - t1:.1f} 秒，{len(d)} 行，conv：{[(x['what'], round(x.get('sec', x.get('rel_diff', 0)), 3)) for x in rec.conv]}")
print(f"GPU 最大 {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB，全体 {time.perf_counter() - t0:.0f} 秒")
