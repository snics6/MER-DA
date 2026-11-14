"""ML の査読者の再検査（DAI-20 第 3 版）：下調べ（実験 16．仮のターゲットの回）の単位の保存に，足した相手の計算の道を当てる．CPU だけ．
テストのラベル（y_test）は読まない．UAR は出さない．見るのは次の 3 つだけである．
  1. 切片のずらし（adj）で，テストの判定がどれだけ変わるか（raw の判定と違う割合．ラベルを使わない量）．相手ごと．
  2. K 本で合わせるモダリティごとの重み（temp・tempb）の当てはめが，全 120 単位 × K = 5・25 で途中で止まらないか．重みの大きさ（K 本のラベルだけを使う）．
  3. 重みの表を探す版の切片の反復の回数と所要（1 単位．CPU）．
評価の回の保存（results/dai1/eval），実験 1・2 の集計，results/dai18_probe は開かない．
使い方：CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DAI/checks/DAI-20/review_ml_sim/v3_probe_labelfree.py [1 2 3]
"""
import itertools
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import dai20_eval as E  # noqa: E402

P16 = E.P16
NF, WS = E.NF, E.WS
UD = E.ROOT / "results" / "dai16_probe" / "units"
PARTS = set(sys.argv[1:]) or set("123")
CORP = ("AFEW", "IEMOCAP", "MOSEI", "MELD")


def f6(W, LM):
    return P16.norm_log(np.einsum("ky,kny->ny", W, LM))


def load(p, K):
    z = np.load(p, allow_pickle=False)
    Pt = [z[f"p_test_ret_{m}_K{K}"].astype(np.float64) for m in "AVT"]
    Pp = [z[f"p_pool_ret_{m}_K{K}"].astype(np.float64) for m in "AVT"]
    LMt, LMp = np.stack([P16.centered(x) for x in Pt]), np.stack([P16.centered(x) for x in Pp])
    return z, Pt, Pp, LMt, LMp


if "1" in PARTS:
    print("=== 1：切片のずらしで変わるテストの判定の割合（%．ラベルを使わない量）．単位は分割 0・種 0 の 12 個")
    rows = []
    for T in CORP:
        for c in CORP:
            p = UD / f"{T}_{c}_f0_r0.npz"
            if T == c or not p.exists():
                continue
            for K in (5, 25):
                z, Pt, Pp, LMt, LMp = load(p, K)
                target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)
                kpos = z[f"kpos_K{K}"]
                yk = z["y_pool"].astype(int)[kpos]                      # K 本のラベルだけ
                LMh = np.stack([P16.centered(z[f"pcf_{m}_K{K}"].astype(np.float64)) for m in "AVT"])
                ct, cp = np.stack([x.max(1) for x in Pt]), np.stack([x.max(1) for x in Pp])
                wt, wtb = E.temp_w(LMh, yk, False), E.temp_w(LMh, yk, True)
                vers = {"prod": (f6(np.ones((3, 4)), LMt), f6(np.ones((3, 4)), LMp)),
                        "avg": (P16.lg(np.mean(Pt, 0)), P16.lg(np.mean(Pp, 0))),
                        "conf": (P16.lg((ct[:, :, None] * np.stack(Pt)).sum(0) / ct.sum(0)[:, None]), P16.lg((cp[:, :, None] * np.stack(Pp)).sum(0) / cp.sum(0)[:, None])),
                        "temp": (P16.norm_log(np.tensordot(wt, LMt, axes=1)), P16.norm_log(np.tensordot(wt, LMp, axes=1))),
                        "tempb": (P16.norm_log(np.tensordot(wtb, LMt, axes=1)), P16.norm_log(np.tensordot(wtb, LMp, axes=1))),
                        "temp_n（平均 1 に割った重み）": (P16.norm_log(np.tensordot(wt / max(wt.mean(), 1e-12), LMt, axes=1)), P16.norm_log(np.tensordot(wt / max(wt.mean(), 1e-12), LMp, axes=1)))}
                r = dict(c=c, K=K)
                for v, (St, Sp) in vers.items():
                    dl, _ = P16.match_shift(Sp, target)
                    r[v] = 100 * float(((St + dl).argmax(1) != St.argmax(1)).mean())
                    if v in ("avg", "temp"):
                        r[f"{v}_最大の感情の割合_raw"] = 100 * float(np.bincount(St.argmax(1), minlength=4).max() / len(St))
                        r[f"{v}_最大の感情の割合_adj"] = 100 * float(np.bincount((St + dl).argmax(1), minlength=4).max() / len(St))
                r["積_最大の感情の割合"] = 100 * float(np.bincount(LMt.sum(0).argmax(1), minlength=4).max() / LMt.shape[1])
                for i in (2, 6, 7):                                             # 提案の格子の点（下調べの保存の判定）
                    r[f"mix{i}"] = 100 * float((z[f"pred_adj_K{K}_M_F6_mix{i}"] != z[f"pred_raw_K{K}_M_F6_mix{i}"]).mean())
                r["M_F2_K"] = 100 * float((z[f"pred_adj_K{K}_M_F2_K"] != z[f"pred_raw_K{K}_M_F2_K"]).mean())
                r["cf_D_l05"] = 100 * float((z[f"pred_adj_K{K}_cf_D_l05"] != z[f"pred_raw_K{K}_cf_D_l05"]).mean())
                r["temp の重みの平均"] = float(wt.mean())
                rows.append(r)
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(d.groupby(["c", "K"]).mean(numeric_only=True).round(1).to_string())
    print("全体の平均：")
    print(d.drop(columns=["c"]).groupby("K").mean(numeric_only=True).round(1).T.to_string())

if "2" in PARTS:
    print("=== 2：temp・tempb の当てはめ（全 120 単位 × K = 5・25．入れ子の 5 回を含む）")
    pairs = list(itertools.combinations(range(NF), 2))
    pidx = {p: n for n, p in enumerate(pairs)}
    fails, ws, n_fit = [], [], 0
    t0 = time.time()
    for p in sorted(UD.glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        c = p.stem.split("_")[1]
        for K in (5, 25):
            kpos, fold = z[f"kpos_K{K}"], z[f"kfold_K{K}"].astype(int)
            yk = z["y_pool"].astype(int)[kpos]
            LMh = np.stack([P16.centered(z[f"pcf_{m}_K{K}"].astype(np.float64)) for m in "AVT"])
            P2 = [z[f"pcf2_{m}_K{K}"].astype(np.float64) for m in "AVT"]
            for fb in (False, True):
                try:
                    n_fit += 1
                    w = E.temp_w(LMh, yk, fb)
                    ws.append(dict(c=c, K=K, fb=fb, w_A=w[0], w_V=w[1], w_T=w[2], mean=w.mean(), at0=float((w <= 1e-6).any()), at5=float((w >= 5 - 1e-6).any())))
                except SystemExit as e:
                    fails.append((p.stem, K, fb, "全部", str(e)))
                for j in range(NF):
                    tr = np.flatnonzero(fold != j)
                    Ltr = np.stack([P16.centered(np.stack([P2[k][pidx[tuple(sorted((int(fold[a]), j)))]][a] for a in tr])) for k in range(3)])
                    try:
                        n_fit += 1
                        E.temp_w(Ltr, yk[tr], fb)
                    except SystemExit as e:
                        fails.append((p.stem, K, fb, f"組 {j}", str(e)))
    print(f"当てはめ {n_fit} 回，止まった回数 {len(fails)}（{time.time() - t0:.0f} 秒）")
    for f_ in fails[:10]:
        print("  ", f_)
    w = pd.DataFrame(ws)
    print(w.groupby(["c", "K", "fb"])[["w_A", "w_V", "w_T", "mean", "at0", "at5"]].mean().round(2).to_string())

if "3" in PARTS:
    print("=== 3：重みの表を探す版の切片の反復（CPU．1 単位，組 0 だけ）")
    for name in ("AFEW_IEMOCAP_f0_r0", "AFEW_MOSEI_f0_r0"):
        p = UD / f"{name}.npz"
        z = np.load(p, allow_pickle=False)
        K = 5
        Lj = torch.tensor(np.stack([P16.centered(z[f"ppj_{m}_K{K}"][0].astype(np.float64)) for m in "AVT"]))
        tgt = torch.softmax(Lj.sum(0), -1).mean(0)
        W = torch.tensor(WS[:201])
        Sp = torch.einsum("rky,kny->rny", W, Lj)
        Sp = Sp - torch.logsumexp(Sp, -1, keepdim=True)
        # 台本の match_shift_batch と同じ式で，表ごとの反復の回数を数える
        d = torch.zeros(Sp.shape[0], 4, dtype=Sp.dtype)
        lt = torch.log(tgt.clamp(min=1e-12))
        live = torch.arange(Sp.shape[0])
        iters = np.zeros(Sp.shape[0], int)
        t0 = time.time()
        for it in range(P16.ADJ_ITERS):
            if not len(live):
                break
            m = torch.softmax(Sp[live] + d[live, None, :], -1).mean(1)
            done = (m - tgt).abs().amax(1) < P16.ADJ_TOL * 1e-2
            upd = d[live] + lt - torch.log(m.clamp(min=1e-12))
            upd = upd - upd.mean(1, keepdim=True)
            d[live[~done]] = upd[~done]
            iters[live[~done].numpy()] += 1
            live = live[~done]
        sec = time.time() - t0
        d2 = E.match_shift_batch(Sp, tgt)
        resid = float((torch.softmax(Sp + d2[:, None, :], -1).mean(1) - tgt).abs().max())
        print(f"{name}：候補 {Lj.shape[1]} 本，表 201 個．反復の回数 中央 {int(np.median(iters))}，90% 点 {int(np.percentile(iters, 90))}，最大 {iters.max()}（上限 {P16.ADJ_ITERS}）．"
              f"上限に当たった表 {int((iters >= P16.ADJ_ITERS).sum())}．残りの最大 {resid:.1e}．台本の関数との差 {float((d - d2).abs().max()):.1e}．CPU で {sec:.1f} 秒．切片のずれの最大 {float(d.abs().max()):.2f}")
