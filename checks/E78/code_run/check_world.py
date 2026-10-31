"""E78 関門 1：AV の世界（評価のターゲット AFEW，反復 0）での検算。結果は標準出力だけ。

1. 第 2 波の参照 ref（ref だけを束ねて解く）が，第 1 波の作り方（ref と ref0 を同じ束で解く）と同じか
2. 退化：Co-teaching の候補の重みを 0（m_eff = 0）にすると，f は none の解に近づくか
3. 退化：xMUDA 型の KL の重みを 0 にすると，各モダリティの主の頭が mapctl と同じになるか
4. Co-teaching・xMUDA 型・dann/mme/mapctl を本番の格子で解く所要時間（1 つの生徒，1 反復）
"""
import sys
import time
from pathlib import Path

import numpy as np
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
from baselines_distill import Prob, tens  # noqa: E402
from distill_belief import weak_rows  # noqa: E402
from geom_e0 import settings  # noqa: E402
from multimodal_distill import WORLDS, blocks, make_students, sub_group  # noqa: E402
from data import groups_for  # noqa: E402

world, T, r = "fuse-audio-video", "AFEW", 0
modes = set((sys.argv[1] if len(sys.argv) > 1 else "ref,cot0,xm0,time").split(","))
DEV = E.DEV
t0 = time.perf_counter()
groups = groups_for(world)
data_f = {g: M.load(g, world) for g in groups}
data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(world)}
S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
mods = WORLDS[world]
sets = {k: settings(k, T) for k in mods + [world]}
info = {k: M5.pool_info(data_m[k] if k != world else data_f, S, E.C_FIT, {}) for k in sets}
ev = data_f[T]
ctx = CB.Ctx(ev, [r], list(E.KS_LAB), center="ind")
pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
stu = make_students(world, T, data_f, data_m, S, pool, te, sets, info)
for k in stu:
    stu[k]["st_ref"] = stu[k]["st"]
yq, yte = ev.y[pool], ev.y[te]
label_sets = BD.label_sets_of(ctx, r, pool)
lab = E.lab_of(label_sets, yq)
prs = {k: Prob(stu[k], yq, label_sets) for k in mods + [world]}
print(f"準備 {time.perf_counter() - t0:.0f} 秒．n={prs[world].n}")


def uar(k, Wb):
    return CB.uar(yte, (prs[k].s["Zte"].double() @ Wb[0].T.double() + Wb[1].double()).argmax(1).cpu().numpy())


none = {k: prs[k].solve([(K, None, K) for K in E.KS]) for k in prs}

if "ref" in modes:
    # 1. 参照 ref：第 2 波（ref だけ）と第 1 波（ref と ref0 を同じ束）
    pFK = {K: prs[world].probs(none[world][K], prs[world].s["Z"]) for K in E.KS}
    pF = tens(stu[world]["P"])
    lg2, lg1 = {K: 0 for K in E.KS}, {K: 0 for K in E.KS}
    for m in mods:
        sol2 = prs[m].solve([(K, weak_rows(pFK[K], E77.REF_MIN), K) for K in E.KS])
        sol1 = prs[m].solve([(("ref", K), weak_rows(pFK[K], E77.REF_MIN), K) for K in E.KS] + [(("ref0", K), weak_rows(pF, E77.REF_MIN), K) for K in E.KS])
        for K in E.KS:
            lg2[K] = lg2[K] + prs[m].probs(sol2[K], prs[m].s["Z"]).clamp(min=1e-12).log()
            lg1[K] = lg1[K] + prs[m].probs(sol1[("ref", K)], prs[m].s["Z"]).clamp(min=1e-12).log()
    for K in E.KS:
        p2, p1 = torch.softmax(lg2[K], 1), torch.softmax(lg1[K], 1)
        print(f"1. K={K}: ref の確率の差の最大 {float((p2 - p1).abs().max()):.2e}，硬いラベルの食い違い {int((p2.argmax(1) != p1.argmax(1)).sum())}/{len(p1)}")

def lp_of(k, Wb):
    return torch.log_softmax(prs[k].s["Zte"].double() @ Wb[0].T.double() + Wb[1].double(), 1)


if "cot0" in modes:
    # 2. Co-teaching の候補の重み 0（評価のターゲットの UAR は出さず，none との出力の差だけを出す）
    k = "audio_wavlm"
    E.CT_MEFFS = ("0",)
    E.CT_TAUS = (0.0, 0.4)
    E.CT_LRS = (1e-2,)
    rec = E.Rec({"target": T, "rep": r}, yte, keep=True)
    t1 = time.perf_counter()
    teachers = {"pFK": {K: prs[world].probs(none[world][K], prs[world].s["Z"]) for K in E.KS}}
    E.run_cot(prs[k], rec, k, lab, teachers)
    print(f"2. cot（m_eff=0，τ 0 と 0.4，lr 1e-2，50 エポック）：{time.perf_counter() - t1:.1f} 秒")
    for K in E.KS:
        ref_lp = lp_of(k, none[k][K])
        ds, dis = [], []
        for row, P_ in zip(rec.rows, rec.P):
            if row["K"] == K and row["method"] in ("cot", "cotctl"):
                P_ = torch.as_tensor(P_, device=ref_lp.device).double()
                ds.append(float((P_ - ref_lp).abs().max()))
                dis.append(int((P_.argmax(1) != ref_lp.argmax(1)).sum()))
        print(f"   K={K}: f と none の対数の確率の差の最大 {max(ds):.3e}，予測の食い違い 最大 {max(dis)}/{len(yte)}")

if "xm0" in modes:
    # 3. xMUDA 型の KL の重み 0 と mapctl（lr 1e-2，200 回，μ 1）
    E.XM_KAPPAS = (0.0,)
    rec = E.Rec({"target": T, "rep": r}, yte, keep=True)
    t1 = time.perf_counter()
    E.run_xmuda(prs, mods, rec, lab, [(0.0, 1.0, K) for K in E.KS] and [(1e-2, 200, 0.0, 1.0, 0, K) for K in E.KS])
    print(f"3. xmuda κ=0：{time.perf_counter() - t1:.1f} 秒")
    for m in mods:
        Zs, ys = E.src_feats_from(stu[m], {s: (data_m[m][s].X, data_m[m][s].y) for s in S})
        rc = E.Rec({"target": T, "rep": r}, yte, keep=True)
        E.run_map(prs[m], Zs, ys, rc, m, lab, [("mapctl", 1e-2, 200, 0.0, 1.0, 0.0, 0, K) for K in E.KS])
        px = [P_ for row, P_ in zip(rec.rows, rec.P) if row["student"] == m]
        pc = rc.P
        dmax = max(float(np.abs(a_ - c_).max()) for a_, c_ in zip(px, pc))
        dis = max(int((a_.argmax(1) != c_.argmax(1)).sum()) for a_, c_ in zip(px, pc))
        print(f"   {m}: xmuda(κ=0) と mapctl の対数の確率の差の最大 {dmax:.3e}，予測の食い違い 最大 {dis}/{len(yte)}")

if "time" in modes:
    # 4. 本番の格子での所要（音声の生徒）
    import pandas as pd
    E.CT_MEFFS, E.CT_TAUS, E.CT_LRS = ("500", "n"), (0.0, 0.2, 0.4, 0.6), (1e-3, 1e-2)
    k = "audio_wavlm"
    Zs, ys = E.src_feats_from(stu[k], {s: (data_m[k][s].X, data_m[k][s].y) for s in S})
    for meths in ():
        rec = E.Rec({"target": T, "rep": r}, yte, keep=False)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        E.run_map(prs[k], Zs, ys, rec, k, lab, E.map_problems(meths, (0,)))
        torch.cuda.synchronize()
        print(f"4. {meths[0]}（種 0 の全候補 {len(rec.rows)} 問，音声の生徒）：{time.perf_counter() - t1:.1f} 秒")
    rec = E.Rec({"target": T, "rep": r}, yte, keep=False)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    E.run_xmuda(prs, mods, rec, lab, E.xm_problems((0,)))
    torch.cuda.synchronize()
    print(f"4. xmuda（種 0 の全候補，{len(rec.rows)} 行）：{time.perf_counter() - t1:.1f} 秒")
    pFK = {K: prs[world].probs(none[world][K], prs[world].s["Z"]) for K in E.KS}
    rec = E.Rec({"target": T, "rep": r}, yte, keep=False)
    diag = []
    torch.cuda.synchronize(); t1 = time.perf_counter()
    E.run_cot(prs[k], rec, k, lab, {"pF": tens(stu[world]["P"]), "pFK": pFK, "ref": pFK}, tau_fix={"pF": 0.3, "pFK": 0.3, "ref": 0.3}, diag=diag)
    torch.cuda.synchronize()
    print(f"4. cot（参照 3 つ，全格子と固定版，{len(rec.rows)} 行）：{time.perf_counter() - t1:.1f} 秒")
    dd = pd.DataFrame(diag)
    print("   ṽ（τ>0）の範囲:", round(dd.v_norm.min(), 3), round(dd.v_norm.max(), 3), " v（τ=0）:", sorted(set(dd[dd.method == 'cotctl'].v.round(3)))[:5])
print(f"GPU 最大 {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB，全体 {time.perf_counter() - t0:.0f} 秒")
