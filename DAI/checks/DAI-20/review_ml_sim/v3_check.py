"""ML の査読者の再検査（DAI-20 第 3 版）：台本 DAI/exp/dai20_eval.py の関数を人工の数に当てる検算．CPU だけ．MER-DA のデータと結果は読まない．
使い方：CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DAI/checks/DAI-20/review_ml_sim/v3_check.py [A B C D E]
  A  重みの表 WS の大きさと，等しい重みの表を足した版の道（put と同じ式）に通したときの判定（M1 の直し方 1・2）
  B  add_extra を人工の 1 単位に通す（K = 5・10・25）．保存した量から同点の表と選んだ重みを作り直せるか（M1 の直し方 3〜5）
  C  K 本で合わせるモダリティごとの重み（temp・tempb）の当てはめが途中で止まらないか
  D  同点の規則（最大に並んだ表の平均）と候補の数 m ごとの，テストの UAR（人工の数）
  E  水準と再現率の区間の作り方（同じ件数で正解 0 を相手にする．感情の列だけ残して 4 倍する）の恒等式
"""
import itertools
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

import dai20_eval as E  # noqa: E402

P16, S, ST, dai15 = E.P16, E.S, E.ST, E.dai15
NF, WS, GRID = E.NF, E.WS, E.GRID
PARTS = set(sys.argv[1:]) or set("ABCDE")


def make_logits(rng, y, strength, bias, noise=1.0):
    n = len(y)
    Z = rng.normal(0, noise, size=(n, 4)) + bias[None]
    Z[np.arange(n), y] += strength[y]
    return Z


def probs(Z):
    return np.exp(Z - logsumexp(Z, axis=-1, keepdims=True))


def f6(W, LM):
    return P16.norm_log(np.einsum("ky,kny->ny", W, LM))


def draw_y(rng, n, share):
    return rng.choice(4, size=n, p=share)


# ------------------------------------------------------------------ A
if "A" in PARTS:
    print("=== A：重みの表の大きさと，等しい重みの表")
    print("WS の形", WS.shape, "感情ごとの和の範囲", float(WS.sum(1).min()), float(WS.sum(1).max()), "12 個の平均の範囲",
          float(WS.mean((1, 2)).min()), float(WS.mean((1, 2)).max()), "表 0 は全部 1：", bool((WS[0] == 1).all()), "最小", float(WS.min()))
    rng = np.random.default_rng(1)
    worst_raw = worst_adj = 0.0
    worst_d = 0.0
    for rep in range(20):
        share = np.array([0.55, 0.20, 0.10, 0.15]) if rep % 2 else np.full(4, 0.25)
        yp, yt = draw_y(rng, 3000, share), draw_y(rng, 3000, share)
        st = rng.uniform(0.6, 1.8, size=(3, 4))
        bs = rng.normal(0, 0.5, size=(3, 4))
        Pp = [probs(make_logits(rng, yp, st[k], bs[k])).astype(np.float32).astype(np.float64) for k in range(3)]
        Pt = [probs(make_logits(rng, yt, st[k], bs[k])).astype(np.float32).astype(np.float64) for k in range(3)]
        LMp, LMt = np.stack([P16.centered(p) for p in Pp]), np.stack([P16.centered(p) for p in Pt])
        target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)
        prod = LMt.sum(0).argmax(1)
        St, Sp = f6(np.ones((3, 4)), LMt), f6(np.ones((3, 4)), LMp)
        dl, _ = P16.match_shift(Sp, target)
        worst_d = max(worst_d, float(np.abs(dl).max()))
        worst_raw = max(worst_raw, float((St.argmax(1) != prod).mean()))
        worst_adj = max(worst_adj, float(((St + dl).argmax(1) != prod).mean()))
        if rep < 2:      # 参考：全部 1/3 の表（第 2 版の作り）だと判定が変わる
            St3, Sp3 = f6(np.full((3, 4), 1 / 3), LMt), f6(np.full((3, 4), 1 / 3), LMp)
            d3, _ = P16.match_shift(Sp3, target)
            print(f"  参考 rep {rep}：全部 1/3 の表は，adj の判定の {100 * ((St3 + d3).argmax(1) != prod).mean():.1f}% が積と違う（切片のずれの最大 {np.abs(d3).max():.3f}）")
    print(f"等しい重みの表（全部 1）：切片のずれの最大 {worst_d:.2e}．積と違う判定の割合の最大 raw {worst_raw:.4f}，adj {worst_adj:.4f}（20 通り）")

# ------------------------------------------------------------------ B
if "B" in PARTS:
    print("=== B：add_extra を人工の 1 単位に通す")
    rng = np.random.default_rng(2)
    n_pool, n_test = 500, 800
    share = np.array([0.45, 0.25, 0.12, 0.18])
    yp, yt = draw_y(rng, n_pool, share), draw_y(rng, n_test, share)
    st = rng.uniform(0.6, 1.8, size=(3, 4))
    bs = rng.normal(0, 0.5, size=(3, 4))
    Zp = np.stack([make_logits(rng, yp, st[k], bs[k]) for k in range(3)])
    Zt = np.stack([make_logits(rng, yt, st[k], bs[k]) for k in range(3)])
    pairs = list(itertools.combinations(range(NF), 2))
    arr = dict(y_test=yt.astype(np.int8), y_pool=yp.astype(np.int8))
    home = rng.uniform(0.5, 2.0, size=(3, 4))
    arr["SR_M_home"], arr["SR_M_ext"] = home, home * np.array([0.8, 0.5, 0.9])[:, None]
    bp = {}
    for K in E.KS:
        kpos = np.sort(np.concatenate([rng.choice(np.flatnonzero(yp == c), K, replace=False) for c in range(4)]))
        yk = yp[kpos]
        fold = dai15.kfold_of(yk)
        nk = np.setdiff1d(np.arange(n_pool), kpos)
        var = [Zp] + [Zp + rng.normal(0, 0.15, size=(3, 1, 4)) + rng.normal(0, 0.1, size=Zp.shape) for _ in range(NF + len(pairs))]   # 作り直した分類器の代わり
        Pv = [probs(v) for v in var]
        held = np.stack([np.stack([Pv[1 + fold[a]][k][kpos[a]] for a in range(len(kpos))]) for k in range(3)])
        Pp64, Pt64 = Pv[0], probs(Zt)
        LMp, LMt = np.stack([P16.centered(p) for p in Pp64]), np.stack([P16.centered(p) for p in Pt64])
        target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)
        DK = np.stack([P16.dprime(held[k], yk)[0] for k in range(3)])
        arr[f"kpos_K{K}"], arr[f"kfold_K{K}"], arr[f"D_M_K_K{K}"] = kpos.astype(np.int32), fold.astype(np.int8), DK
        for k, m in enumerate("AVT"):
            arr[f"p_test_ret_{m}_K{K}"], arr[f"p_pool_ret_{m}_K{K}"] = Pt64[k].astype(np.float32), Pp64[k].astype(np.float32)
            arr[f"pcf_{m}_K{K}"] = held[k].astype(np.float32)
            arr[f"pcf2_{m}_K{K}"] = np.stack([Pv[1 + NF + n][k][kpos] for n in range(len(pairs))]).astype(np.float32)
            arr[f"ppj_{m}_K{K}"] = np.stack([Pv[1 + j][k] for j in range(NF)]).astype(np.float32)
        arr[f"pred_raw_K{K}_prod"] = LMt.sum(0).argmax(1).astype(np.int8)
        arr[f"pred_pool_K{K}_prod"] = LMp.sum(0)[nk].argmax(1).astype(np.int8)
        for i, (n0, beta, prior) in enumerate(GRID):                       # 実験 16 の単位の計算と同じ式（float64 の確率から）
            W = P16.to_w(P16.mix_rel(DK, arr["SR_M_ext"], 4 * K, n0, beta, prior))
            dl, _ = P16.match_shift(f6(W, LMp), target)
            arr[f"pred_adj_K{K}_M_F6_mix{i}"] = (f6(W, LMt) + dl).argmax(1).astype(np.int8)
        bp[K] = {f"b_{m}": rng.integers(0, 4, n_test) for m in E.B_METHODS}
    rows = []
    t0 = time.time()
    E.add_extra(arr, rows, bp)
    print(f"add_extra：{time.time() - t0:.1f} 秒．行 {len(rows)}")
    import pandas as pd
    df = pd.DataFrame(rows)
    for K in E.KS:
        d = df[df.K == K].set_index("version")
        need = ["avg", "conf", "temp", "tempb"] + [f"wf{m}" for m in E.WF_MS] + [f"M_F6h_mix{i}" for i, g in enumerate(GRID) if g[1] == 1.0] + list(E.B_VERS) + ["_check20"]
        miss = [v for v in need if v not in d.index]
        keys = [f"pred_{w}_K{K}_{v}" for v in ("avg", "conf", "temp", "tempb") + tuple(f"wf{m}" for m in E.WF_MS) for w in ("raw", "adj", "pool")]
        miss_k = [k for k in keys if k not in arr]
        ua = arr[f"wf_uar_K{K}"]
        ok_w = []
        for m in E.WF_MS:
            u = ua[:m + 1].astype(np.float64)
            tied = np.flatnonzero(u >= u.max() - 1e-9)                          # 保存した float32 の UAR から同点の表を作り直す
            ok_w.append(bool(np.allclose(WS[tied].mean(0), arr[f"W_wf{m}_K{K}"]) and len(tied) == int(d.loc[f"wf{m}", "wf_n_tied"]) and tied[0] == int(d.loc[f"wf{m}", "wf_index"])))
        # K 本の上の UAR を，台本の関数を使わずに 30 個の表で作り直す
        kpos, fold = arr[f"kpos_K{K}"], arr[f"kfold_K{K}"].astype(int)
        yk = yp[kpos]
        LMh = np.stack([P16.centered(arr[f"pcf_{m}_K{K}"].astype(np.float64)) for m in "AVT"])
        LMpj = [np.stack([P16.centered(arr[f"ppj_{m}_K{K}"][j].astype(np.float64)) for m in "AVT"]) for j in range(NF)]
        dev = 0.0
        for r_ in list(range(12)) + list(np.random.default_rng(5).integers(12, 1001, 18)):
            pred = np.empty(len(yk), int)
            for j in range(NF):
                te = np.flatnonzero(fold == j)
                dl, _ = P16.match_shift(f6(WS[r_], LMpj[j]), np.exp(P16.norm_log(LMpj[j].sum(0))).mean(0))
                pred[te] = (f6(WS[r_], LMh[:, te]) + dl).argmax(1)
            dev = max(dev, abs(100 * S.uar(yk, pred) - float(ua[r_])))
        print(f"K={K}：欠けた行 {miss}．欠けた判定 {miss_k}．wf_uar の形 {ua.shape}．K 本の UAR の作り直しとの差の最大 {dev:.1e}．"
              f"保存から同点の表と重みを作り直せる（m=10,100,1000）：{ok_w}．同点の数 {[int(d.loc[f'wf{m}', 'wf_n_tied']) for m in E.WF_MS]}．"
              f"作り直した提案の判定の一致 {d.loc['_check20', 'recheck_agree_min']:.4f}．等しい重みの一致 {d.loc['_check20', 'eq_agree']:.4f}．"
              f"temp の重み {np.round(arr[f'W_temp_K{K}'], 2)}，tempb {np.round(arr[f'W_tempb_K{K}'], 2)}．"
              f"UAR(adj) 積 {100 * S.uar(yt, arr[f'pred_raw_K{K}_prod'].astype(int)):.1f}，" + "，".join(f"{v} {d.loc[v, 'uar_adj']:.1f}" for v in ("avg", "conf", "temp", "tempb", "wf10", "wf100", "wf1000")))
    # 等しい重みの表だけが最大のときの道：K 本の UAR を作って wf_pick に渡す
    W0, t0_ = E.wf_pick(np.r_[80.0, np.full(1000, 70.0)], 1000)
    print("最大が表 0 だけ → 選んだ重みは全部 1：", bool((W0 == 1).all()), list(t0_))

# ------------------------------------------------------------------ C
if "C" in PARTS:
    print("=== C：temp・tempb の当てはめが止まらないか（fit_wb は勾配の最大 > 1e-3 で SystemExit）")
    rng = np.random.default_rng(3)
    cases = {"弱い（強さ 0.3）": (0.3, 0.3, 0.3), "中（1.5）": (1.5, 1.5, 1.5), "強い（4）": (4.0, 4.0, 4.0), "1 つだけ完全（12，ほかは 0）": (12.0, 0.0, 0.0),
             "全部ほぼ完全（12）": (12.0, 12.0, 12.0), "1 つが逆（−3），ほかは 2": (-3.0, 2.0, 2.0), "鋭い確率（雑音 6，強さ 2）": (2.0, 2.0, 2.0)}
    for name, sts in cases.items():
        for K in (5, 10, 25):
            fail, at_hi, at_lo, n_try = 0, 0, 0, 150
            for _ in range(n_try):
                yk = np.repeat(np.arange(4), K)
                noise = 6.0 if name.startswith("鋭い") else 1.0
                P = [probs(make_logits(rng, yk, np.full(4, s), rng.normal(0, 0.5, 4), noise)).astype(np.float32).astype(np.float64) for s in sts]
                LM = np.stack([P16.centered(p) for p in P])
                for fb in (False, True):
                    try:
                        w = E.temp_w(LM, yk, fb)
                        at_hi += int((w >= P16.W_MAX - 1e-6).any()); at_lo += int((w <= 1e-6).any())
                    except SystemExit as e:
                        fail += 1
                        if fail == 1:
                            print("   止まった例：", name, K, fb, str(e)[:80])
            print(f"  {name}，K={K}：止まった回数 {fail}/{2 * n_try}．重みが上限 5 に当たった {at_hi}，0 に当たった {at_lo}")

# ------------------------------------------------------------------ D
if "D" in PARTS:
    print("=== D：同点の規則と候補の数 m ごとのテストの UAR（人工の数．adj の層）")
    rng = np.random.default_rng(4)
    n_pool, n_test, n_units = 400, 3000, 30
    Wt = torch.tensor(WS)
    res = {}
    first = True
    for K in (5, 10, 25):
        acc = {k: [] for k in ("prod",) + tuple(f"{r}{m}" for m in E.WF_MS for r in ("avg", "first", "near"))}
        ties, eqonly = {m: [] for m in E.WF_MS}, {m: [] for m in E.WF_MS}
        for u_ in range(n_units):
            share = np.array([0.55, 0.20, 0.10, 0.15]) if u_ % 2 else np.full(4, 0.25)
            yp, yt = draw_y(rng, n_pool, share), draw_y(rng, n_test, share)
            st = rng.uniform(0.6, 1.8, size=(3, 4))
            bs = rng.normal(0, 0.5, size=(3, 4))
            LMp = np.stack([P16.centered(probs(make_logits(rng, yp, st[k], bs[k]))) for k in range(3)])
            LMt = np.stack([P16.centered(probs(make_logits(rng, yt, st[k], bs[k]))) for k in range(3)])
            kpos = np.sort(np.concatenate([rng.choice(np.flatnonzero(yp == c), K, replace=False) for c in range(4)]))
            yk = yp[kpos]
            fold = dai15.kfold_of(yk)
            target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)
            # 抜いた分類器は元の分類器と同じと置く（候補は全部の組で共通）ので，切片は表ごとに 1 回だけ決める
            Sp = torch.einsum("rky,kny->rny", Wt, torch.tensor(LMp))
            Sp = Sp - torch.logsumexp(Sp, -1, keepdim=True)
            d = E.match_shift_batch(Sp, torch.tensor(target))
            Sh = torch.einsum("rky,kny->rny", Wt, torch.tensor(LMp[:, kpos]))
            Sh = Sh - torch.logsumexp(Sh, -1, keepdim=True)
            pred = (Sh + d[:, None, :]).argmax(-1).numpy()
            ua = np.array([100 * S.uar(yk, p) for p in pred])
            if first:                                                           # 台本の wf_search と同じ値になる
                ua2 = E.wf_search(LMp[:, kpos], yk, fold, [LMp] * NF)
                print(f"  台本の wf_search との差の最大 {np.abs(ua - ua2).max():.1e}")
                first = False
            prod = 100 * S.uar(yt, LMt.sum(0).argmax(1))
            acc["prod"].append(prod)

            def test_uar(W):
                dl, _ = P16.match_shift(f6(W, LMp), target)
                return 100 * S.uar(yt, (f6(W, LMt) + dl).argmax(1))
            for m in E.WF_MS:
                W, tied = E.wf_pick(ua, m)
                ties[m].append(len(tied)); eqonly[m].append(float(len(tied) == 1 and tied[0] == 0))
                acc[f"avg{m}"].append(prod if (len(tied) == 1 and tied[0] == 0) else test_uar(W))
                acc[f"first{m}"].append(test_uar(WS[tied[0]]))
                acc[f"near{m}"].append(test_uar(WS[tied[np.argmin(((WS[tied] - 1) ** 2).sum((1, 2)))]]))
        print(f"  K={K}（{n_units} 単位）：積 {np.mean(acc['prod']):.1f}｜" + "｜".join(
            f"m={m}：平均（設計）{np.mean(acc[f'avg{m}']):.1f}，番号の小さい表 {np.mean(acc[f'first{m}']):.1f}，等しい重みに近い表 {np.mean(acc[f'near{m}']):.1f}，"
            f"同点の数 {np.mean(ties[m]):.0f}，表 0 だけ {np.mean(eqonly[m]):.2f}" for m in E.WF_MS))

# ------------------------------------------------------------------ E
if "E" in PARTS:
    print("=== E：水準と再現率の区間の作り方の恒等式（dai1_stats.effect_stats）")
    rng = np.random.default_rng(6)
    for g_eff, label in ((8.0, "jackknife の道"), (40.0, "引き直しの道")):
        cell = {}
        names = np.array([f"g{i}" for i in range(12)])
        for f in range(5):
            for r in range(4):
                tot = rng.integers(8, 30, size=(12, 4)).astype(float)
                cor = np.floor(tot * rng.uniform(0.2, 0.9, size=(12, 4)))
                cell[(f, r)] = (names, cor, tot)
        zero = {k: (x[0], np.zeros_like(x[1]), x[2]) for k, x in cell.items()}
        es = ST.effect_stats(cell, zero, g_eff, np.random.default_rng(0))
        direct = np.mean([np.mean([100 * (cell[(f, r)][1].sum(0) / cell[(f, r)][2].sum(0)).mean() for r in range(4)]) for f in range(5)])
        rec = []
        for i in range(4):
            m = np.zeros(4); m[i] = 1.0
            only = {k: (x[0], x[1] * m, x[2]) for k, x in cell.items()}
            ei = ST.effect_stats(only, zero, g_eff, np.random.default_rng(0))
            di = np.mean([np.mean([100 * cell[(f, r)][1].sum(0)[i] / cell[(f, r)][2].sum(0)[i] for r in range(4)]) for f in range(5)])
            rec.append((4 * ei["effect"], di, 4 * ei["lo"], 4 * ei["hi"]))
        print(f"  {label}（{es['method']}）：水準 {es['effect']:.6f}，単位の UAR の平均 {direct:.6f}，区間 [{es['lo']:.2f}, {es['hi']:.2f}]．"
              f"再現率（4 倍）と直接の値の差の最大 {max(abs(a - b) for a, b, _, _ in rec):.1e}．再現率の平均 − 水準 {np.mean([a for a, *_ in rec]) - es['effect']:.1e}")
