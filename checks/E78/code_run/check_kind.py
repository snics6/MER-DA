"""E78 関門 1：種類の生徒（audio_wavlm，評価のターゲット RAVDESS，反復 0）での検算。結果は標準出力だけ（results/ に書かない）。

1. none の UAR を E77 の試走（dc600b8）の保存値と突き合わせる
2. lab_tensors の重みと Prob.wt の一致，罰則の中心 W0 と anchor の一致
3. 退化：mapctl を μ = 1e6（A を I に固定）で解くと none の解に近づくか．K=0 は W0 のまま
4. 退化：dann の λ=0，mme の λ=0・sw=0 が mapctl と同じ値になるか（同じ束の大きさ）
5. 同じ問を束の大きさを変えて解くと値が変わるか（判別器の乱数の消費が束の大きさに依る）
6. CORAL の整合：そろえを恒等にした coral_lin が meta5.pool_info の lin と一致するか，K=0 の行が none の K=0 と一致するか
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
torch.cuda.set_per_process_memory_fraction(0.12)
torch.backends.cuda.matmul.allow_tf32 = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_dai2_9044116 as E  # noqa: E402  （検査するコミットの版を固定）
E.ROOT = ROOT  # 写した版の ROOT は checks/E78 を指すので直す
import compare_dai as E77  # noqa: E402
import baselines_distill as BD  # noqa: E402
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from baselines_distill import Prob  # noqa: E402
from geom_e0 import settings  # noqa: E402
from teach_student39 import Student  # noqa: E402
from data import groups_for  # noqa: E402

kind, T, r = "audio_wavlm", "RAVDESS", 0
DEV = E.DEV
t0 = time.perf_counter()
data = {g: M.load(g, kind) for g in groups_for(kind)}
S = [s for s in data if CM.corp(s) != CM.corp(T)]
gamma, rho_s, rho_m, cb, cp, _ = settings(kind, T)
C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
src_pc, lin, n_of = M5.pool_info(data, S, C_T, {})
ev = data[T]
ctx = CB.Ctx(ev, [r], list(E.KS_LAB), center="ind")
pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
ct = ctx.ct[r]
Xu = M.rows(ev.X, pool) - ct
te_X = M.rows(ev.X, te) - ct
srcs = {s: (data[s].X, data[s].y) for s in S}
yq = ev.y[pool]
label_sets = BD.label_sets_of(ctx, r, pool)
st = Student(ev, Xu, src_pc, lin, n_of, S, C_T, gamma, rho_s, [rho_m])
Am = st.mats[rho_m][0]
s = {"st": st, "rho_m": rho_m, "cb": cb, "cp": cp, "Z": st.Xstd @ Am, "Zte": ((te_X - st.src.mu) / st.src.sd) @ Am}
Zs, ys = E.src_feats(st, rho_m, srcs)
pr = Prob(s, yq, label_sets)
none = pr.solve([(K, None, K) for K in E.KS])
yte = ev.y[te]
print(f"準備 {time.perf_counter() - t0:.0f} 秒．n={pr.n} d={pr.d} テスト {len(te)}．E39 cp={cp} cb={cb} rho_m={rho_m} C_T={C_T}")


def uar_of(Wb, Zte=s["Zte"]):
    return CB.uar(yte, (Zte.to(torch.float64) @ Wb[0].T.to(torch.float64) + Wb[1].to(torch.float64)).argmax(1).cpu().numpy())


# 1. none の再現
ref = pd.read_csv(ROOT / "checks/E77/code_run/kind_audio_dc6/RAVDESS.csv", dtype={"K": str})
ref = ref[(ref.method == "none") & (ref.rep == 0)].set_index("K").uar
print("1. none の UAR（E78 の解 / E77 試走）:", {K: (round(uar_of(none[K]), 4), round(float(ref[K]), 4)) for K in E.KS})

# 2. 重みと罰則の中心
pos, ylab, wK = E.lab_tensors(E.lab_of(label_sets, yq))
for K in E.KS:
    w_prob = pr.wt[K].view(4, pr.n).sum(0)
    w_e78 = torch.zeros(pr.n, dtype=torch.float64, device=DEV)
    w_e78[pos] = wK[K].to(torch.float64)
    lab_ok = bool((pr.wt[K].view(4, pr.n).argmax(0)[pos][wK[K] > 0] == ylab[wK[K] > 0]).all()) if K != "0" else True
    print(f"2. K={K}: 重みの差の最大 {float((w_prob - w_e78).abs().max()):.2e}，ラベル一致 {lab_ok}")
anc = st.anchor(rho_m)
Wa, ba = anc.moved_to(pr.zero, pr.one, st.C)
print(f"2. 罰則の中心：|W0 − anchor| 最大 {float((pr.W0 - Wa).abs().max()):.2e}，|b0 − anchor| 最大 {float((pr.b0 - ba).abs().max()):.2e}")

# 3. mapctl を μ = 1e6 で
I = torch.eye(pr.d, dtype=E.F32, device=DEV)
args = (s["Z"].to(E.F32), s["Z"].to(E.F32)[pos], ylab, wK, Zs.to(E.F32), torch.as_tensor(ys, device=DEV), pr.W0.to(E.F32), pr.b0.to(E.F32), float(cp), cb)
t1 = time.perf_counter()
out = E._train_map(*args[:2], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9],
                   [("mapctl", 0.0, 1e6, 0.0, K) for K in E.KS], 1e-2, 1000, 0, I)
print(f"3. mapctl μ=1e6 lr=1e-2 1000 回：{time.perf_counter() - t1:.1f} 秒")
for K, (Wf, bf) in zip(E.KS, out):
    Wn, bn = none[K]
    rel = float((Wf.double() - Wn).norm() / (Wn - pr.W0).norm().clamp(min=1e-12))
    print(f"   K={K}: |W − W_none|/|W_none − W0| = {rel:.3e}，|W − W0| = {float((Wf.double() - pr.W0).norm()):.3e}，UAR {uar_of((Wf, bf)):.4f} / none {uar_of(none[K]):.4f}")

# 4. dann λ=0・mme λ=0 sw=0 と mapctl（同じ束の大きさ 5）
mu = 1.0
for lr, iters in ((1e-2, 200),):
    ctl = E._train_map(*args, [("mapctl", 0.0, mu, 0.0, K) for K in E.KS], lr, iters, 0, I)
    dn = E._train_map(*args, [("dann", 0.0, mu, 0.0, K) for K in E.KS], lr, iters, 0, I)
    mm = E._train_map(*args, [("mme", 0.0, mu, 0.0, K) for K in E.KS], lr, iters, 0, I)
    dmax = max(float((a[0] - b[0]).abs().max()) for a, b in zip(ctl, dn))
    mmax = max(float((a[0] - b[0]).abs().max()) for a, b in zip(ctl, mm))
    print(f"4. lr={lr} it={iters}：|dann(λ=0) − mapctl| 最大 {dmax:.2e}，|mme(λ=0,sw=0) − mapctl| 最大 {mmax:.2e}")
    print("   mapctl の UAR:", [round(uar_of(w), 4) for w in ctl], " none:", [round(uar_of(none[K]), 4) for K in E.KS])

# 5. 束の大きさを変えると同じ問の値が変わるか（dann λ=1 μ=1 K=5，lr 1e-2，200 回）
probe = ("dann", 1.0, 1.0, 0.0, "5")
fill = [("dann", lam, m_, 0.0, K) for lam in E.MAP_LAMS for m_ in E.MAP_MUS for K in E.KS]
fill = [c for c in fill if c != probe]
alone = E._train_map(*args, [probe], 1e-2, 200, 0, I)[0]
alone2 = E._train_map(*args, [probe], 1e-2, 200, 0, I)[0]
inbatch = E._train_map(*args, [probe] + fill[:29], 1e-2, 200, 0, I)[0]
print(f"5. dann λ=1 μ=1 K=5：単独 2 回の差 {float((alone[0] - alone2[0]).abs().max()):.2e}，単独と束 30 の差 {float((alone[0] - inbatch[0]).abs().max()):.2e}，"
      f"UAR 単独 {uar_of(alone):.4f} 束 {uar_of(inbatch):.4f}")
for seed in (0, 1, 2):
    o = E._train_map(*args, [probe], 1e-2, 200, seed, I)[0]
    print(f"   種 {seed} 単独の UAR {uar_of(o):.4f}")

# 6. CORAL の整合
orig_ms = E.msqrt


def coral_lin_identity(srcs_, Xu_, rho_s_, r_):
    """そろえを恒等にした coral_lin（Pn = Ps）。"""
    E.msqrt = lambda S_, inv=False: torch.eye(S_.shape[0], dtype=S_.dtype, device=S_.device)
    try:
        return E.coral_lin(srcs_, Xu_, rho_s_, r_)
    finally:
        E.msqrt = orig_ms


lin_id = coral_lin_identity(srcs, Xu, rho_s, 64)
for s_ in S[:3]:
    A1, a1 = lin[(s_, rho_s)]
    A2, a2 = lin_id[(s_, rho_s)]
    cos = float((A1 * A2).sum() / (A1.norm() * A2.norm()))
    print(f"6. {s_}：恒等のそろえの lin と pool_info の lin の相対差 {float((A1 - A2).norm() / A1.norm()):.3e}，余弦 {cos:.6f}")
st_id = Student(ev, Xu, src_pc, lin_id, n_of, S, C_T, gamma, rho_s, [rho_m])
W0id = st_id.W_std @ st_id.mats[rho_m][1]
print(f"6. 恒等のそろえの E39 の下書き：K=0 の UAR {uar_of((W0id, st_id.b_std)):.4f}（none の K=0 {uar_of(none['0']):.4f}），"
      f"|W0id − W0|/|W0| {float((W0id - pr.W0).norm() / pr.W0.norm()):.3e}")
# 恒等そろえの lin を pool_info と同じ sd_p で作り直すと一致するか（原因の切り分け）
Xp = torch.cat([data[s_].X for s_ in S])
sdp_raw = Xp.std(0, unbiased=False)
Xc = torch.cat([data[s_].X - data[s_].X.mean(0) for s_ in S])
sdp_c = Xc.std(0, unbiased=False)
print(f"6. pool_info の sd_p（中心化しない）と coral_lin の sd_p（ソースごとに中心化）の比：中央値 {float((sdp_raw / sdp_c).median()):.3f}，"
      f"最小 {float((sdp_raw / sdp_c).min()):.3f}，最大 {float((sdp_raw / sdp_c).max()):.3f}")
del Xp, Xc
# CORAL の行（本物の run_coral）の K=0 と none の K=0
rec = E.Rec({"target": T, "rep": r}, yte, keep=True)
t1 = time.perf_counter()
E.run_coral(ev, Xu, te_X, src_pc, srcs, n_of, S, C_T, gamma, rho_s, label_sets, yq, rec, kind)
d = pd.DataFrame(rec.rows)
print(f"6. run_coral：{time.perf_counter() - t1:.1f} 秒，{len(d)} 行．K=0 の UAR の範囲 {d[d.K == '0'].uar.min():.4f}〜{d[d.K == '0'].uar.max():.4f}")
print("   r ごとの行の数:", d.param.str.extract(r"^(r\d+)")[0].value_counts().to_dict())
print("   W の保存のキー:", list(rec.W), " 型:", rec.W[kind + "@std"][0][1].dtype if rec.W else None, " 生の特徴の W のキー（xform）:", list(rec.xform))
print(f"GPU 最大 {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB，全体 {time.perf_counter() - t0:.0f} 秒")
