"""実験 7 の下調べ（DAI-7）：ほかのモダリティの疑似ラベルで蒸留すると，単独の分類器どうしの点数の依存が増えるか．

設計は `DAI/docs/実験7_下調べ_設計.md`（第 4 版）．第 1 版の台本 `DAI/exp/dai7_probe.py` は関門の前の本計算（`results/dai7_probe/`）の記録として残す．実験 1 の仮のターゲットの回の保存（`results/dai1/pseudo/units/`）だけを読み，
保存した W・b を候補とテストの特徴に当てて点数を作り直す．新しく学ぶのは 3.5 節の 9 次元の判別分析と，3.3 節の教師の点数への回帰だけである．
評価の回の保存は読まない．

使い方：
  python DAI/exp/dai7_probe_v2.py --out results/dai7_probe_v2                    # 120 単位（出力が既にある単位があれば止める）
  python DAI/exp/dai7_probe_v2.py --out <dir> --units AFEW_IEMOCAP_f0_r0          # 試走（glob）
  python DAI/exp/dai7_probe_v2.py --out results/dai7_probe_v2 --summarize [--allow-partial]
  python DAI/exp/dai7_probe_v2.py --selftest
"""
from __future__ import annotations

import argparse
import ast
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = HERE.parents[1]
UD = ROOT / "results" / "dai1" / "pseudo" / "units"
FROZEN = ROOT / "DAI" / "frozen" / "dai1_frozen.json"
KS = (0, 1, 5, 10, 25)
KS_RULE = (1, 5, 10, 25)
RHOS = (0.0625, 0.25, 1.0, 4.0)
TT = ("poe_none", "start_AVT", "fused_AVT")          # fused_AVT は ret と同じ問（数値の確かめだけ．設計 3.2 節）
PAIRS = (("A", "V"), ("A", "T"), ("V", "T"))
MODS = "AVT"
N_NULL = 200          # 3.3 節：テストの帰無
N_NULL_POOL = 50      # 3.3 節：候補の上の D（副）の帰無
N_NULL_H = 50         # 3.6 節：h の偶然の水準
N_BOOT = 1000
MIN_N = 15            # 3.3 節：感情の発話が 15 本未満の (分割, 感情) は使わない．4 節：抜き直しの回でも同じ
MIN_CELL = 3          # 3.3 節：話者で中心化した版の (グループ, 感情) の最小
MIN_HALF = 5          # 3.5 節 (a)：両方の半分の各感情が 5 本以上
MIN_SHARE = 0.30      # 3.5 節 (b)：小さい半分がテストの 30% 以上
FLOOR_D = 0.02        # 4 節：ΔD・Δφ の下限
FLOOR_L = 0.5         # 4 節：ΔL の下限（UAR のポイント）
NULL_RATIO = 0.1      # 3.6 節：|Δd⁰| < 0.1·|ΔD|
H_MAX = 0.5           # 3.6 節：h − h⁰ ≥ 0.5 のセルは L_dep を「この結合器での差」とだけ書く
HALF_T = 0.5          # 4 節の書き方 2：教師の点数を除いた ΔD_t が ΔD の半分以下
EPS_D = 1e-6
EPS_LDA = 1e-3
H = np.linalg.svd(np.eye(4) - 0.25)[0][:, :3]           # 4×3．列は 1 のベクトルに直交
RHO_SETS = tuple(f"joint_a1_rho{r}" for r in RHOS)
BOOT_SETS = ("none", "joint", "joint_a1", "oracle", RHO_SETS[0], RHO_SETS[-1])   # 4 節の区間を作る組
L_SETS = ("none", "joint", "joint_a1")
BLK = {m: slice(3 * i, 3 * i + 3) for i, m in enumerate(MODS)}


# ------------------------------------------------------------------ 量（3.3〜3.5 節）

def dstat(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
    """(…, n, 3) の 2 つの点数 → 正準相関の 2 乗の平均 (1/3)·tr(Saa⁻¹ Sab Sbb⁻¹ Sba)（3.3 節）。"""
    za = za - za.mean(-2, keepdims=True)
    zb = zb - zb.mean(-2, keepdims=True)
    n = za.shape[-2]
    Saa = np.swapaxes(za, -1, -2) @ za / n
    Sbb = np.swapaxes(zb, -1, -2) @ zb / n
    Sab = np.swapaxes(za, -1, -2) @ zb / n
    return dstat_cov(Saa, Sbb, Sab)


def dstat_cov(Saa, Sbb, Sab):
    I3 = np.eye(3)
    ra = EPS_D * np.trace(Saa, axis1=-2, axis2=-1)[..., None, None] / 3 * I3
    rb = EPS_D * np.trace(Sbb, axis1=-2, axis2=-1)[..., None, None] / 3 * I3
    A = np.linalg.solve(Saa + ra, Sab)                                  # Saa⁻¹ Sab
    B = np.linalg.solve(Sbb + rb, np.swapaxes(Sab, -1, -2))             # Sbb⁻¹ Sba
    return np.trace(A @ B, axis1=-2, axis2=-1) / 3


def perm_within(groups: np.ndarray, rng, n_perm: int) -> np.ndarray:
    """groups（発話ごとの組の番号）の中で並べ替えた添字 (n_perm, n)。"""
    n = len(groups)
    out = np.tile(np.arange(n), (n_perm, 1))
    for g in np.unique(groups):
        ix = np.flatnonzero(groups == g)
        if len(ix) > 1:
            out[:, ix] = ix[np.argsort(rng.random((n_perm, len(ix))), 1)]
    return out


def d_with_null(za, zb, y, rng, n_perm=N_NULL, cells=None):
    """感情ごとの d_e と，m′ を同じ感情の中で（cells を渡せばその組の中で）並べ替えた n_perm 回の平均 d_e⁰。
    発話が MIN_N 未満の感情は NaN。"""
    d, d0, n = np.full(4, np.nan), np.full(4, np.nan), np.zeros(4, int)
    for e in range(4):
        ix = np.flatnonzero(y == e)
        n[e] = len(ix)
        if len(ix) < MIN_N:
            continue
        a, b = za[ix], zb[ix]
        d[e] = float(dstat(a, b))
        g = np.zeros(len(ix), int) if cells is None else pd.factorize(cells[ix])[0]
        perm = perm_within(g, rng, n_perm)
        d0[e] = float(dstat(np.broadcast_to(a, (n_perm,) + a.shape), b[perm]).mean())
    return d, d0, n


def center_cells(z, y, grp):
    """(グループ, 感情) ごとの平均を引く．MIN_CELL 本未満の発話は外す（3.3 節の副）。"""
    keep = np.zeros(len(y), bool)
    out = z.copy()
    key = pd.Series([f"{g}|{e}" for g, e in zip(grp, y)])
    for _, ix in key.groupby(key).groups.items():
        ix = np.asarray(ix)
        if len(ix) >= MIN_CELL:
            out[ix] = z[ix] - z[ix].mean(0)
            keep[ix] = True
    return out, keep, key.to_numpy()


def residualize(z, t, y):
    """感情ごとに z を [1, t] に最小二乗で回帰した残り（3.3 節の教師の点数を除いた依存）。"""
    out = np.zeros_like(z)
    for e in range(4):
        ix = y == e
        if ix.sum() == 0:
            continue
        Xd = np.concatenate([np.ones((ix.sum(), 1)), t[ix]], 1)
        beta, *_ = np.linalg.lstsq(Xd, z[ix], rcond=None)
        out[ix] = z[ix] - Xd @ beta
    return out


def phi_err(pa, pb, y):
    """感情の中の正誤の φ の 4 感情の平均（3.4 節）。使えない感情は除く。"""
    vals = []
    for e in range(4):
        ix = y == e
        if ix.sum() < MIN_N:
            continue
        ea, eb = (pa[ix] == e).astype(float), (pb[ix] == e).astype(float)
        if ea.std() == 0 or eb.std() == 0:
            continue
        vals.append(float(np.corrcoef(ea, eb)[0, 1]))
    return float(np.mean(vals)) if vals else np.nan, len(vals)


def phi_counts(pa, pb, y, codes, k):
    """(塊, 感情) ごとの 2×2 の表（正誤 a × 正誤 b）の数（4 節のブートストラップ用）。"""
    c = np.zeros((k, 4, 2, 2))
    np.add.at(c, (codes, y, (pa == y).astype(int), (pb == y).astype(int)), 1)
    return c


def lda_fit(z, y, blk):
    mu = np.stack([z[y == e].mean(0) for e in range(4)])
    R = z - mu[y]
    Sig = R.T @ R / len(z)
    if blk:
        B = np.zeros_like(Sig)
        for s in BLK.values():
            B[s, s] = Sig[s, s]
        Sig = B
    P = np.linalg.inv(Sig + EPS_LDA * np.trace(Sig) / Sig.shape[0] * np.eye(Sig.shape[0]))
    A = mu @ P
    return A, 0.5 * np.sum(A * mu, 1)


def lda_pred(z, A, c):
    return (z @ A.T - c).argmax(1)


def split_halves(y, cl):
    """3.5 節の 2 つの半分．塊を大きい順（同じ大きさは名前の順）に発話の少ない半分へ入れ，(a) 両方の半分の各感情が MIN_HALF 本以上で
    (b) 小さい半分がテストの MIN_SHARE 以上なら塊の単位．そうでなければ感情で層にした発話の単位（種 0）。"""
    codes, names = pd.factorize(pd.Series(cl).astype(str), sort=True)
    if len(names) >= 2:
        sizes = np.bincount(codes)
        order = sorted(range(len(names)), key=lambda j: (-sizes[j], names[j]))
        half_of, tot = np.zeros(len(names), int), [0, 0]
        for j in order:
            h = int(tot[1] < tot[0])
            half_of[j] = h; tot[h] += sizes[j]
        h = half_of[codes]
        ok_e = all((y[h == s] == e).sum() >= MIN_HALF for s in (0, 1) for e in range(4))
        if ok_e and min(tot) >= MIN_SHARE * len(y):
            return h, "cluster"
    rng = np.random.default_rng(0)
    h = np.zeros(len(y), int)
    for e in range(4):
        ix = np.flatnonzero(y == e)
        ix = ix[rng.permutation(len(ix))]
        h[ix[1::2]] = 1
    return h, "utterance"


def crossfit(z, y, half, blk):
    pred = np.zeros(len(y), int)
    for s in (0, 1):
        tr, te = half != s, half == s
        A, c = lda_fit(z[tr], y[tr], blk)
        pred[te] = lda_pred(z[te], A, c)
    return pred


def heterogeneity(z, y):
    """感情ごとの共分散 Σ_e と共通の Σ のフロベニウスの比の平均 h（3.6 節）。"""
    mu = np.stack([z[y == e].mean(0) for e in range(4)])
    R = z - mu[y]
    Sig = R.T @ R / len(z)
    vals = [np.linalg.norm(np.cov(z[y == e].T, bias=True) - Sig) / np.linalg.norm(Sig) for e in range(4) if (y == e).sum() >= MIN_N]
    return float(np.mean(vals)) if vals else np.nan


def heterogeneity_null(z, y, rng, n_perm=N_NULL_H):
    """感情のラベルをテストの中で並べ替えた n_perm 回の h の平均 h⁰（3.6 節）。"""
    return float(np.mean([heterogeneity(z, y[rng.permutation(len(y))]) for _ in range(n_perm)]))


def uar(y, p):
    return 100.0 * np.mean([np.mean(p[y == e] == e) for e in range(4) if (y == e).any()])


# ------------------------------------------------------------------ 分類器の組（3.2 節）

def pick(z3, key, want):
    items = [json.loads(x) for x in z3[f"{key}_items"]]
    w = {k: ST.norm(v) for k, v in want.items()}
    hits = [j for j, it in enumerate(items) if all(ST.norm(it.get(k)) == v for k, v in w.items())]
    if len(hits) != 1:
        raise SystemExit(f"{key}：項目 {want} が {len(hits)} 個ある（1 個のはず）")
    return z3[f"{key}_W"][hits[0]], z3[f"{key}_b"][hits[0]]


def parse_tether(arr) -> dict:
    out = {}
    for s in arr:
        kind, K, cw, cb = str(s).split("|")
        out[(kind, int(K))] = (float(cw), float(cb))
    return out


def parse_wsel(arr) -> dict:
    return {int(s.split("|")[0]): tuple(float(x) for x in ast.literal_eval(str(s).split("|", 1)[1])) for s in arr}


def p1_wb(z1, kind, tag):
    items = [str(x) for x in z1[f"p1_{kind}_items"]]
    if items.count(tag) != 1:
        raise SystemExit(f"p1_{kind}_items に {tag} が {items.count(tag)} 個ある")
    j = items.index(tag)
    return z1[f"p1_{kind}_W"][j], z1[f"p1_{kind}_b"][j]


def classifier_sets(u, z1, z2, z3, F, K, tether):
    """組の名前 → {モダリティ: (W, b)}（numpy）。tt_single_X は X 以外の 2 つのモダリティだけを持つ。"""
    ver = F["main"][str(K)]
    names = ("none", "ret", "joint", "joint_a1", "oracle", *RHO_SETS, *[f"tt_{q}" for q in TT], *[f"tt_single_{x}" for x in MODS])
    sets = {n: {} for n in names}
    for m in MODS:
        if K == 0:
            W0, b0 = u.start[m]
            sets["none"][m] = (W0.cpu().numpy(), b0.cpu().numpy())
        else:
            cw, cb = tether[(m, K)]
            sets["none"][m] = p1_wb(z1, m, f"{K}|{cw}|{cb}")
        cw25, cb25 = tether[(m, 25)]
        sets["oracle"][m] = p1_wb(z1, m, f"all|{cw25}|{cb25}")
        sets["ret"][m] = (z2[f"p2_K{K}_ret_{m}_W"], z2[f"p2_K{K}_ret_{m}_b"])
        key = f"p3_{m}_K{K}"
        sel = {k: v for k, v in F["select"][f"{m}|{K}|joint|{ver}"].items() if k != "pseudo_uar"}
        sets["joint"][m] = pick(z3, key, dict(method="joint", teacher=ver, **sel))
        sel1 = {k: v for k, v in F["select"][f"{m}|{K}|joint_a1|{ver}"].items() if k != "pseudo_uar"}
        if ST.norm(sel1.get("a", 1)) != "1":
            raise SystemExit(f"{m}|{K}|joint_a1|{ver} の a が 1 でない：{sel1}")
        sets["joint_a1"][m] = pick(z3, key, dict(method="joint", teacher=ver, rho=sel1["rho"], a=1))
        for r, nm in zip(RHOS, RHO_SETS):
            sets[nm][m] = pick(z3, key, dict(method="joint", teacher=ver, rho=r, a=1))
        for q in TT:
            sets[f"tt_{q}"][m] = pick(z3, key, dict(method="tt", teacher=q, rho=1.0, a=1))
        for x in MODS:
            if x != m:
                sets[f"tt_single_{x}"][m] = pick(z3, key, dict(method="tt", teacher=f"single_{x}", rho=1.0, a=1))
    return sets, ver


def to_t(a, like):
    return torch.as_tensor(np.asarray(a), dtype=like.dtype, device=like.device)


# ------------------------------------------------------------------ 単位

def unit_name(p3: Path) -> str:
    return p3.name[: -len("_p3.npz")]


def run_unit(p3: Path, out: Path, fz: dict, commit: str, dirty: bool) -> dict:
    name = unit_name(p3)
    T, c, fs, rs = name.split("_")
    f, r = int(fs[1:]), int(rs[1:])
    rows_dir, fits_dir = out / "rows", out / "fits"
    for p in (rows_dir / f"{name}_units.csv", rows_dir / f"{name}_uar.csv", fits_dir / f"{name}.npz"):
        if p.exists():
            raise SystemExit(f"{p} は既にある（重複を作らない．消すか別の --out にする）")
    t0 = time.time()
    F = fz[T]
    srcs = [x for x in DD.CORPORA if x not in (T, c)]
    z1, z2, z3 = (np.load(UD / f"{name}_p{i}.npz") for i in (1, 2, 3))
    for i, z in enumerate((z1, z2, z3), 1):
        if str(z["commit"]) != F["pseudo_commit"]:
            raise SystemExit(f"{name}_p{i} のコミット {z['commit']} が固定のファイルの pseudo_commit と違う")
    tether = parse_tether(z3["tether"])
    wsel = parse_wsel(z3["wsel"])
    u = dai1.Unit(ROOT / "results" / "dai1", T, c, srcs, f, r)
    sid_t = u.cp.df["sample_id"].to_numpy()[u.test].astype(str)
    sid_p = u.cp.df["sample_id"].to_numpy()[u.pool].astype(str)
    if not ((sid_t == z1["sample_test"].astype(str)).all() and (sid_p == z1["sample_pool"].astype(str)).all()):
        raise SystemExit(f"{name}：発話の並びが保存と違う")
    y_of, _ = ST.label_maps(c)
    yt, yp = u.ytest, u.ypool
    if not ((y_of(sid_t) == yt).all() and (y_of(sid_p) == yp).all()):
        raise SystemExit(f"{name}：ラベルが発話一覧と違う")
    cl_t = u.cp.df["cluster"].to_numpy()[u.test].astype(str)
    grp_t = (u.cp.df["speaker_id"] if c == "IEMOCAP" else u.cp.df["cluster"]).to_numpy()[u.test].astype(str)
    half, split_kind = split_halves(yt, cl_t)
    n_half = (int((half == 0).sum()), int((half == 1).sum()))
    seed = int.from_bytes(name.encode(), "little") % (2 ** 32)
    urows, arows = [], []
    store = {"test_sample_id": sid_t, "test_cluster": cl_t, "test_group": grp_t, "test_y": yt.astype(np.int8), "half": half.astype(np.int8)}
    base = dict(T=T, c=c, fold=f, seed=r)
    for K in KS:
        sets, ver = classifier_sets(u, z1, z2, z3, F, K, tether)
        rng = np.random.default_rng([seed, K])
        # 教師の点数 t（主の版の積．one は指数 (1,1,1)，sel は固定のファイルの指数）．テストの上で作る（3.3 節の副）
        if ver == "sel" and tuple(float(x) for x in F["w"][str(K)]) != tuple(wsel[K]):
            raise SystemExit(f"{name}：K = {K} の wsel {wsel[K]} が固定のファイルの w {F['w'][str(K)]} と違う")
        w_t = (1.0, 1.0, 1.0) if ver == "one" else wsel[K]
        Lt = 0.0
        for wm, m in zip(w_t, MODS):
            W, b = sets["ret"][m]
            Zr = S.logits(u.Xt[m], to_t(W, u.Xt[m]), to_t(b, u.Xt[m]))
            Lt = Lt + float(wm) * torch.log_softmax(Zr, -1).double().cpu().numpy()
        t_te = Lt @ H
        store[f"K{K}_teacher_t"] = t_te.astype(np.float32)
        for sname, wb in sets.items():
            Zt, Zp = {}, {}
            for m, (W, b) in wb.items():
                Wt, bt = to_t(W, u.Xt[m]), to_t(b, u.Xt[m])
                Zt[m] = S.logits(u.Xt[m], Wt, bt).double().cpu().numpy()
                Zp[m] = S.logits(u.Xp[m], Wt, bt).double().cpu().numpy()
            zt = {m: Zt[m] @ H for m in Zt}
            zp = {m: Zp[m] @ H for m in Zp}
            pr = {m: Zt[m].argmax(1) for m in Zt}
            for a, b in PAIRS:
                if a not in zt or b not in zt:
                    continue
                d, d0, n = d_with_null(zt[a], zt[b], yt, rng)
                dp, dp0, _ = d_with_null(zp[a], zp[b], yp, rng, n_perm=N_NULL_POOL)
                zc, keep, cellkey = center_cells(np.concatenate([zt[a], zt[b]], 1), yt, grp_t)
                dc, dc0, _ = d_with_null(zc[keep, :3], zc[keep, 3:], yt[keep], rng, cells=cellkey[keep])
                ra, rb = residualize(zt[a], t_te, yt), residualize(zt[b], t_te, yt)
                dr, dr0, _ = d_with_null(ra, rb, yt, rng)
                ph, nph = phi_err(pr[a], pr[b], yt)
                row = dict(**base, K=K, set=sname, pair=a + b, teacher_version=ver,
                           D=float(np.nanmean(d - d0)), D_raw=float(np.nanmean(d)), D_null=float(np.nanmean(d0)),
                           n_emotions=int(np.isfinite(d).sum()),
                           D_pool=float(np.nanmean(dp - dp0)), D_cent=float(np.nanmean(dc - dc0)),
                           n_dropped_cent=int((~keep).sum()), D_partial_t=float(np.nanmean(dr - dr0)),
                           phi=ph, n_emotions_phi=nph)
                for e in range(4):
                    row.update({f"d_e{e}": d[e], f"d0_e{e}": d0[e], f"n_e{e}": int(n[e])})
                urows.append(row)
            store[f"K{K}_{sname}_z"] = np.concatenate([zt[m] for m in MODS if m in zt], 1).astype(np.float32)
            store[f"K{K}_{sname}_mods"] = np.array("".join(m for m in MODS if m in zt))
            for m in pr:
                store[f"K{K}_{sname}_pred_{m}"] = pr[m].astype(np.int8)
            if len(zt) == 3:
                z9 = np.concatenate([zt[m] for m in MODS], 1)
                p_prod = sum(Zt.values()).argmax(1)
                p_blk, p_full = crossfit(z9, yt, half, True), crossfit(z9, yt, half, False)
                store.update({f"K{K}_{sname}_pred_prod": p_prod.astype(np.int8), f"K{K}_{sname}_pred_blk": p_blk.astype(np.int8),
                              f"K{K}_{sname}_pred_full": p_full.astype(np.int8)})
                arows.append(dict(**base, K=K, set=sname, teacher_version=ver, split=split_kind,
                                  n_half0=n_half[0], n_half1=n_half[1], n_per_param=min(n_half) / 9,
                                  **{f"uar_{m}": uar(yt, pr[m]) for m in MODS},
                                  uar_prod=uar(yt, p_prod), uar_blk=uar(yt, p_blk), uar_full=uar(yt, p_full),
                                  L_dep=uar(yt, p_full) - uar(yt, p_blk), L_cal=uar(yt, p_blk) - uar(yt, p_prod),
                                  heterogeneity=heterogeneity(z9, yt), heterogeneity_null=heterogeneity_null(z9, yt, rng)))
    rows_dir.mkdir(parents=True, exist_ok=True); fits_dir.mkdir(parents=True, exist_ok=True)
    tag = dict(commit=commit, dirty=dirty)
    np.savez_compressed(fits_dir / f"{name}.npz", **store, commit=np.array(commit))
    pd.DataFrame(urows).assign(**tag).to_csv(rows_dir / f"{name}_units.csv", index=False)
    pd.DataFrame(arows).assign(**tag).to_csv(rows_dir / f"{name}_uar.csv", index=False)
    sec = time.time() - t0
    print(f"{name}：{sec:.0f} 秒（分け方 {split_kind}，半分 {n_half[0]}/{n_half[1]}）", flush=True)
    if S.DEV == "cuda":
        torch.cuda.empty_cache()
    return dict(unit=name, sec=sec, split=split_kind, n_half=n_half)


# ------------------------------------------------------------------ 集計（4 節）

def boot_weights(cl_by_fold: dict, seed: int = 0):
    """c の 5 分割のテストの塊をまとめた集まりから塊を抜き直す重み（実験 6 の `_boot_fold_mean` と同じ作り）。
    戻り：w (N_BOOT, 塊の総数)，fold_of (塊の総数,)，code_of {分割: 塊の名前 → 通し番号}。"""
    names, fold_of, code_of = [], [], {}
    for f in sorted(cl_by_fold):
        uniq = sorted(set(cl_by_fold[f]))
        code_of[f] = {x: len(names) + i for i, x in enumerate(uniq)}
        names += uniq; fold_of += [f] * len(uniq)
    Kc = len(names)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(Kc, np.full(Kc, 1.0 / Kc), size=N_BOOT).astype(float)
    return w, np.array(fold_of), code_of


def fold_combine(per_fold: dict, w, fold_of):
    """分割ごとの (N_BOOT,) の値を，抜かれた分割について「抜かれた塊の数 ÷ 分割の塊の数」の重みで平均する（4 節の 4）。"""
    folds = sorted(per_fold)
    V = np.stack([per_fold[f] for f in folds], 1)
    cnt = np.stack([w[:, fold_of == f].sum(1) for f in folds], 1)
    size = np.array([(fold_of == f).sum() for f in folds])
    wf = cnt / size[None, :]
    ok = (cnt > 0) & np.isfinite(V)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, V * wf, 0).sum(1) / np.where(ok, wf, 0).sum(1)


def boot_D_unit(z, y, codes, d0, wcl):
    """1 単位・1 組の D の抜き直し．z (n, 9)，d0 {組: ((slice_a, slice_b), (4,) の d⁰)}，wcl (N_BOOT, 塊)。
    その回で重みつきの本数が MIN_N 未満の感情は外す（4 節の 2）。戻り {組: (N_BOOT,)}。"""
    k = wcl.shape[1]
    dim = z.shape[1]
    n = np.zeros((k, 4)); np.add.at(n, (codes, y), 1)
    s = np.zeros((k, 4, dim)); np.add.at(s, (codes, y), z)
    ss = np.zeros((k, 4, dim, dim)); np.add.at(ss, (codes, y), z[:, :, None] * z[:, None, :])
    N = wcl @ n
    Sx = (wcl @ s.reshape(k, -1)).reshape(-1, 4, dim)
    SS = (wcl @ ss.reshape(k, -1)).reshape(-1, 4, dim, dim)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = Sx / N[..., None]
        C = SS / N[..., None, None] - mu[..., :, None] * mu[..., None, :]
    out = {}
    for pair, ((sa, sb), d0e) in d0.items():
        dd = np.full(N.shape, np.nan)
        for e in range(4):
            if not np.isfinite(d0e[e]):
                continue
            ok = N[:, e] >= MIN_N
            if ok.any():
                Ce = C[ok, e]
                dd[ok, e] = dstat_cov(Ce[:, sa, sa], Ce[:, sb, sb], Ce[:, sa, sb]) - d0e[e]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[pair] = np.nanmean(dd, 1)
    return out


def boot_uar_unit(pred, y, codes, wcl):
    k = wcl.shape[1]
    n = np.zeros((k, 4)); np.add.at(n, (codes, y), 1)
    cor = np.zeros((k, 4)); np.add.at(cor, (codes, y), (pred == y).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (wcl @ cor) / (wcl @ n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return 100.0 * np.nanmean(r, 1)


def boot_phi_unit(z, y, codes, wcl, K, s):
    ph = []
    for a_, b_ in PAIRS:
        cnt = phi_counts(z[f"K{K}_{s}_pred_{a_}"].astype(int), z[f"K{K}_{s}_pred_{b_}"].astype(int), y, codes, wcl.shape[1])
        Wc = np.einsum("bk,keij->beij", wcl, cnt)
        n_ = Wc.sum((2, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            pa = Wc[:, :, 1, :].sum(2) / n_; pb = Wc[:, :, :, 1].sum(2) / n_; pab = Wc[:, :, 1, 1] / n_
            phe = (pab - pa * pb) / np.sqrt(pa * (1 - pa) * pb * (1 - pb))
        phe[(n_ < MIN_N) | ~np.isfinite(phe)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ph.append(np.nanmean(phe, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(ph), 0)


def ci(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)), len(x)


def nanmean_stack(lst):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(lst), 0)


def summarize(out: Path, allow_partial: bool) -> None:
    U = pd.concat([pd.read_csv(p) for p in sorted((out / "rows").glob("*_units.csv"))], ignore_index=True)
    A = pd.concat([pd.read_csv(p) for p in sorted((out / "rows").glob("*_uar.csv"))], ignore_index=True)
    want = {unit_name(p) for p in UD.glob("*_p3.npz")}
    have = set(U["T"] + "_" + U["c"] + "_f" + U["fold"].astype(str) + "_r" + U["seed"].astype(str))
    dup = U.duplicated(["T", "c", "fold", "seed", "K", "set", "pair"]).sum() + A.duplicated(["T", "c", "fold", "seed", "K", "set"]).sum()
    commits = set(U["commit"]) | set(A["commit"])
    if dup:
        raise SystemExit(f"重複した行が {dup} 個ある")
    if (have != want or len(commits) != 1) and not allow_partial:
        raise SystemExit(f"単位がそろっていない（{len(have)}/{len(want)}）か，コミットが複数ある（{commits}）")
    qcols = ["D", "D_raw", "D_null", "D_pool", "D_cent", "D_partial_t", "phi"]
    Ua = U.groupby(["T", "c", "fold", "seed", "K", "set"])[qcols].mean().reset_index().assign(pair="all")
    Uall = pd.concat([U, Ua], ignore_index=True)
    cells = Uall.groupby(["c", "K", "set", "pair"])[qcols].mean().reset_index()
    cells["n_units"] = Uall.groupby(["c", "K", "set", "pair"]).size().values
    A["h_excess"] = A["heterogeneity"] - A["heterogeneity_null"]
    ucols = [x for x in A.columns if x.startswith("uar_") or x in ("L_dep", "L_cal", "heterogeneity", "heterogeneity_null", "h_excess", "n_per_param")]
    acells = A.groupby(["c", "K", "set"])[ucols].mean().reset_index()
    diffs = []
    for c, Uc in Uall.groupby("c"):
        units = sorted(set(zip(Uc["T"], Uc["fold"], Uc["seed"])))
        fits = {(T, f, r): np.load(out / "fits" / f"{T}_{c}_f{f}_r{r}.npz") for T, f, r in units}
        cl_by_fold = {}
        for (T, f, r), z in fits.items():
            cl_by_fold.setdefault(f, list(z["test_cluster"].astype(str)))
        w, fold_of, code_of = boot_weights(cl_by_fold)
        for K in KS:
            bD = {s: {p: {} for p in ("AV", "AT", "VT", "all")} for s in BOOT_SETS}
            bphi = {s: {} for s in BOOT_SETS}
            bL = {s: {} for s in L_SETS}
            bG, bG1 = {}, {}
            for f in sorted(cl_by_fold):
                cols = np.flatnonzero(fold_of == f)
                wcl = w[:, cols]
                acc = {s: {p: [] for p in ("AV", "AT", "VT", "all")} for s in BOOT_SETS}
                accphi = {s: [] for s in BOOT_SETS}
                accL = {s: [] for s in L_SETS}
                accG, accG1 = [], []
                for (T, ff, r), z in fits.items():
                    if ff != f:
                        continue
                    y = z["test_y"].astype(int)
                    codes = np.array([code_of[f][x] for x in z["test_cluster"].astype(str)]) - cols[0]
                    sel_u = Uc[(Uc["T"] == T) & (Uc["fold"] == f) & (Uc["seed"] == r) & (Uc["K"] == K)]
                    for s in BOOT_SETS:
                        rs_ = sel_u[sel_u["set"] == s]
                        d0 = {}
                        for a_, b_ in PAIRS:
                            rw = rs_[rs_["pair"] == a_ + b_].iloc[0]
                            d0[a_ + b_] = ((BLK[a_], BLK[b_]), np.array([rw[f"d0_e{e}"] if np.isfinite(rw[f"d_e{e}"]) else np.nan for e in range(4)]))
                        res = boot_D_unit(z[f"K{K}_{s}_z"].astype(np.float64), y, codes, d0, wcl)
                        for p, v in res.items():
                            acc[s][p].append(v)
                        acc[s]["all"].append(nanmean_stack([res[p] for p in ("AV", "AT", "VT")]))
                        accphi[s].append(boot_phi_unit(z, y, codes, wcl, K, s))
                    for s in L_SETS:
                        accL[s].append(boot_uar_unit(z[f"K{K}_{s}_pred_full"].astype(int), y, codes, wcl)
                                       - boot_uar_unit(z[f"K{K}_{s}_pred_blk"].astype(int), y, codes, wcl))
                    bu = {(s, q): boot_uar_unit(z[f"K{K}_{s}_pred_{q}"].astype(int), y, codes, wcl)
                          for s in ("joint_a1", "none") for q in ("prod", *MODS)}
                    bu.update({("rho1", q): boot_uar_unit(z[f"K{K}_joint_a1_rho1.0_pred_{q}"].astype(int), y, codes, wcl) for q in ("prod", *MODS)})
                    accG.append((bu[("joint_a1", "prod")] - bu[("none", "prod")])
                                - np.mean([bu[("joint_a1", m)] - bu[("none", m)] for m in MODS], 0))
                    accG1.append((bu[("rho1", "prod")] - bu[("none", "prod")])
                                 - np.mean([bu[("rho1", m)] - bu[("none", m)] for m in MODS], 0))
                for s in BOOT_SETS:                                         # 4 節の 3：分割の値は 6 単位の平均
                    for p in acc[s]:
                        bD[s][p][f] = nanmean_stack(acc[s][p])
                    bphi[s][f] = nanmean_stack(accphi[s])
                for s in L_SETS:
                    bL[s][f] = nanmean_stack(accL[s])
                bG[f] = nanmean_stack(accG)
                bG1[f] = nanmean_stack(accG1)
            Bd = {s: {p: fold_combine(bD[s][p], w, fold_of) for p in bD[s]} for s in BOOT_SETS}
            Bphi = {s: fold_combine(bphi[s], w, fold_of) for s in BOOT_SETS}
            BL = {s: fold_combine(bL[s], w, fold_of) for s in L_SETS}
            BG = fold_combine(bG, w, fold_of)
            BG1 = fold_combine(bG1, w, fold_of)
            pt = cells[(cells.c == c) & (cells.K == K)].set_index(["set", "pair"])
            ptA = acells[(acells.c == c) & (acells.K == K)].set_index("set")
            dd_pairs = {"dD_joint_a1_none": ("joint_a1", "none"), "dD_joint_none": ("joint", "none"), "dD_joint_a1_oracle": ("joint_a1", "oracle"),
                        "dD_rho_hi_lo": (RHO_SETS[-1], RHO_SETS[0])}
            for p in ("AV", "AT", "VT", "all"):
                for nm, (s1, s2) in dd_pairs.items():
                    lo, hi, nv = ci(Bd[s1][p] - Bd[s2][p])
                    diffs.append(dict(c=c, K=K, pair=p, quantity=nm, est=pt.loc[(s1, p), "D"] - pt.loc[(s2, p), "D"], lo=lo, hi=hi, n_boot_valid=nv,
                                      d_null_diff=pt.loc[(s1, p), "D_null"] - pt.loc[(s2, p), "D_null"]))
                for s in BOOT_SETS:
                    lo, hi, nv = ci(Bd[s][p])
                    diffs.append(dict(c=c, K=K, pair=p, quantity=f"D_{s}", est=pt.loc[(s, p), "D"], lo=lo, hi=hi, n_boot_valid=nv))
            for nm, (s1, s2) in {"dphi_joint_a1_none": ("joint_a1", "none"), "dphi_joint_none": ("joint", "none")}.items():
                lo, hi, nv = ci(Bphi[s1] - Bphi[s2])
                diffs.append(dict(c=c, K=K, pair="all", quantity=nm, est=pt.loc[(s1, "all"), "phi"] - pt.loc[(s2, "all"), "phi"], lo=lo, hi=hi, n_boot_valid=nv))
            for nm, (s1, s2) in {"dLdep_joint_a1_none": ("joint_a1", "none"), "dLdep_joint_none": ("joint", "none")}.items():
                lo, hi, nv = ci(BL[s1] - BL[s2])
                diffs.append(dict(c=c, K=K, pair="all", quantity=nm, est=ptA.loc[s1, "L_dep"] - ptA.loc[s2, "L_dep"], lo=lo, hi=hi, n_boot_valid=nv))
            for s in L_SETS:
                lo, hi, nv = ci(BL[s])
                diffs.append(dict(c=c, K=K, pair="all", quantity=f"Ldep_{s}", est=ptA.loc[s, "L_dep"], lo=lo, hi=hi, n_boot_valid=nv))
            for s in ("joint", "joint_a1", "joint_a1_rho1.0"):              # 得の足し合わせ（3.5 節，3.6 節の (iv)．rho1.0 は選んでいない対照）
                G = {m: ptA.loc[s, f"uar_{m}"] - ptA.loc["none", f"uar_{m}"] for m in MODS}
                gp = ptA.loc[s, "uar_prod"] - ptA.loc["none", "uar_prod"]
                gf = ptA.loc[s, "uar_full"] - ptA.loc["none", "uar_full"]
                lo, hi, nv = ci(BG) if s == "joint_a1" else ci(BG1) if s == "joint_a1_rho1.0" else (np.nan, np.nan, 0)
                diffs.append(dict(c=c, K=K, pair="all", quantity=f"G_{s}", est=gp - np.mean(list(G.values())), lo=lo, hi=hi, n_boot_valid=nv,
                                  **{f"G_{m}": G[m] for m in MODS}, G_prod=gp, G_full=gf, G_max=max(G.values())))
            for s1, s2 in (("joint_a1", "none"),):                          # 教師の点数を除いた依存の差（4 節の書き方 2．点推定）
                diffs.append(dict(c=c, K=K, pair="all", quantity="dDt_joint_a1_none",
                                  est=pt.loc[(s1, "all"), "D_partial_t"] - pt.loc[(s2, "all"), "D_partial_t"]))
        print(f"{c}：区間を作った", flush=True)
    Df = pd.DataFrame(diffs)
    crit = []

    def count(c, q, floor, null_check=False):
        d = Df[(Df.c == c) & (Df.quantity == q) & (Df.pair == "all") & Df.K.isin(KS_RULE)]
        lo_pos = d.lo > 0
        ok = lo_pos & (d.est >= floor)
        if null_check:
            ok &= d.d_null_diff.abs() < NULL_RATIO * d.est.abs()
        return int(lo_pos.sum()), int(ok.sum()), int((d.hi < 0).sum())

    for c in sorted(Df.c.unique()):
        r = dict(c=c)
        for nm, q, fl, nc in (("H1", "dD_joint_a1_none", FLOOR_D, True), ("H1_joint", "dD_joint_none", FLOOR_D, True),
                              ("H1p", "dD_joint_a1_oracle", FLOOR_D, True), ("phi", "dphi_joint_a1_none", FLOOR_D, False),
                              ("H3", "dLdep_joint_a1_none", FLOOR_L, False), ("H3_joint", "dLdep_joint_none", FLOOR_L, False),
                              ("H2_size", "dD_rho_hi_lo", FLOOR_D, False)):
            a_, b_, n_ = count(c, q, fl, nc)
            r.update({f"{nm}_n_lo_pos": a_, f"{nm}_n_lo_pos_floor": b_, f"{nm}_n_hi_neg": n_, nm: bool(b_ >= 3)})
        sp, size_ok = [], []
        for K in KS_RULE:
            v = [cells[(cells.c == c) & (cells.K == K) & (cells.set == nm) & (cells.pair == "all")]["D"].iloc[0] for nm in RHO_SETS]
            sp.append(float(pd.Series(v).corr(pd.Series(range(4)), method="spearman")))
            d = Df[(Df.c == c) & (Df.quantity == "dD_rho_hi_lo") & (Df.pair == "all") & (Df.K == K)].iloc[0]
            size_ok.append(bool(d.lo > 0 and d.est >= FLOOR_D))
        r.update({f"H2_spearman_K{K}": v for K, v in zip(KS_RULE, sp)})
        r["H2"] = bool(sum((v >= 0.8 - 1e-9) and s_ for v, s_ in zip(sp, size_ok)) >= 3)
        dD = Df[(Df.c == c) & (Df.quantity == "dD_joint_a1_none") & (Df.pair == "all") & Df.K.isin(KS_RULE)].set_index("K")["est"]
        dDt = Df[(Df.c == c) & (Df.quantity == "dDt_joint_a1_none") & Df.K.isin(KS_RULE)].set_index("K")["est"]
        dDr = Df[(Df.c == c) & (Df.quantity == "dD_joint_a1_none") & (Df.pair == "all") & Df.K.isin(KS_RULE)].set_index("K")
        h1_ok = [K for K in KS_RULE if dDr.loc[K, "lo"] > 0 and dDr.loc[K, "est"] >= FLOOR_D
                 and abs(dDr.loc[K, "d_null_diff"]) < NULL_RATIO * abs(dDr.loc[K, "est"])]
        r["n_K_teacher_explains"] = int(sum(dDt[K] <= HALF_T * dD[K] for K in h1_ok))          # H1 の条件を満たした K だけで数える
        phi_small = r["phi_n_lo_pos"] - r["phi_n_lo_pos_floor"] >= 3
        if not r["H1"]:
            r["H1_wording"] = "満たさなかった"
        elif not r["H1p"]:
            r["H1_wording"] = "正解で別々に学んだ分類器の依存の水準の中に収まる"
        elif r["phi"]:
            r["H1_wording"] = "誤りの写しを含む依存の増加"
        elif r["n_K_teacher_explains"] >= 3:
            r["H1_wording"] = "教師の点数の揺れの写しによる依存の増加" + ("（誤りの相関の増加は下限 0.02 に届かない）" if phi_small else "")
        else:
            r["H1_wording"] = "依存は増えたが，誤りの写しとも教師の点数の揺れの写しとも分けられない" + ("（誤りの相関の増加は下限 0.02 に届かない）" if phi_small else "")
        g = Df[(Df.c == c) & (Df.quantity == "G_joint_a1") & Df.K.isin(KS_RULE)]
        r["additivity_n_hi_neg"] = int((g.hi < 0).sum())
        r["additivity_fails"] = bool(r["additivity_n_hi_neg"] >= 3)
        h = acells[(acells.c == c) & acells.K.isin(KS_RULE) & acells.set.isin(L_SETS)]
        r["n_cells_h_excess_ge_0.5"] = int((h.h_excess >= H_MAX).sum())
        crit.append(r)
    C = pd.DataFrame(crit)
    C = C.assign(**{f"{nm}_overall": bool(C[nm].sum() >= 3) for nm in ("H1", "H1_joint", "H1p", "H2", "H3")})
    commit = sorted(commits)[0] if len(commits) == 1 else "multiple"
    for df_, fn in ((cells, "cells.csv"), (acells, "uar_cells.csv"), (Df, "diffs.csv"), (C, "criteria.csv")):
        df_.assign(commit=commit).to_csv(out / fn, index=False)
    print(f"書いた：cells.csv，uar_cells.csv，diffs.csv，criteria.csv（{len(have)} 単位）", flush=True)


# ------------------------------------------------------------------ 検算

def selftest() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    # (1) 独立なら d − d⁰ ≈ 0
    a, b = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    y = rng.integers(0, 4, n)
    d, d0, _ = d_with_null(a, b, y, rng)
    assert np.all(np.abs(d - d0) < 0.01), (d, d0)
    # (2) 共有の因子：a = s + e，b = s + e′（各次元）で正準相関の 2 乗は 1/4
    sh = rng.normal(size=(n, 3)); a, b = sh + rng.normal(size=(n, 3)), sh + rng.normal(size=(n, 3))
    d, d0, _ = d_with_null(a, b, y, rng)
    assert np.all(np.abs(d - 0.25) < 0.04), d
    # (3) モダリティごとの可逆な 1 次変換でほぼ変わらない
    M1, M2 = rng.normal(size=(3, 3)) + 3 * np.eye(3), rng.normal(size=(3, 3)) + 3 * np.eye(3)
    gap = abs(float(dstat(a, b)) - float(dstat(a @ M1 + 5, b @ M2 - 2)))
    assert gap < 1e-5, gap
    # (4) 重み 1 の抜き直しの共分散は直接の値と同じ
    codes = rng.integers(0, 20, n)
    z = np.concatenate([a, b, rng.normal(size=(n, 3))], 1)
    res = boot_D_unit(z, y, codes, {"AV": ((BLK["A"], BLK["V"]), np.zeros(4))}, np.ones((2, 20)))
    direct = np.mean([float(dstat(a[y == e], b[y == e])) for e in range(4)])
    assert abs(res["AV"][0] - direct) < 1e-8, (res["AV"][0], direct)
    # (5) 依存が無いとき L_dep ≈ 0，強い依存で full > blk
    mu = np.eye(4)[:, :3] * 1.0

    def gen(rho):
        yy = rng.integers(0, 4, n)
        e1 = rng.normal(size=(n, 3)); e2 = rho * e1 + np.sqrt(1 - rho ** 2) * rng.normal(size=(n, 3)); e3 = rng.normal(size=(n, 3))
        return np.concatenate([mu[yy] + e1, mu[yy] + e2, mu[yy] + e3], 1), yy
    half = np.arange(n) % 2
    z0, y0 = gen(0.0); zr, yr = gen(0.9)
    l0 = uar(y0, crossfit(z0, y0, half, False)) - uar(y0, crossfit(z0, y0, half, True))
    lr = uar(yr, crossfit(zr, yr, half, False)) - uar(yr, crossfit(zr, yr, half, True))
    assert abs(l0) < 1.5 and lr > 1.0, (l0, lr)
    # (6) 分け方：塊がそろっていれば塊の単位，最大の塊が 97% なら発話の単位，塊 1 つなら発話の単位
    cl = np.array([f"s{i % 7}" for i in range(n)])
    h, kind = split_halves(y, cl)
    assert kind == "cluster" and all((y[h == s] == e).sum() >= MIN_HALF for s in (0, 1) for e in range(4))
    big = np.where(np.arange(n) < int(0.97 * n), "big", np.array([f"t{i % 2}" for i in range(n)]))
    assert split_halves(y, big)[1] == "utterance"
    assert split_halves(y, np.array(["one"] * n))[1] == "utterance"
    # (7) fold_combine の重み
    w = np.ones((N_BOOT, 3)); fo = np.array([0, 0, 1])
    v = fold_combine({0: np.full(N_BOOT, 1.0), 1: np.full(N_BOOT, 3.0)}, w, fo)
    assert np.allclose(v, 2.0), v[:3]
    # (8) 塊の中の並べ替えは塊の外に出ない
    g = np.repeat(np.arange(5), 7)
    P = perm_within(g, rng, 30)
    assert all((g[P[i]] == g).all() for i in range(30))
    # (9) 教師の点数を除くと，教師だけを共有する 2 つの点数の依存は偶然の水準まで下がる
    t = rng.normal(size=(n, 3))
    a2, b2 = t @ rng.normal(size=(3, 3)) + rng.normal(size=(n, 3)), t @ rng.normal(size=(3, 3)) + rng.normal(size=(n, 3))
    dr, dr0, _ = d_with_null(residualize(a2, t, y), residualize(b2, t, y), y, rng)
    dd, dd0, _ = d_with_null(a2, b2, y, rng)
    assert np.all(np.abs(dr - dr0) < 0.01) and np.all(dd - dd0 > 0.1), (dr - dr0, dd - dd0)
    print("検算：9 項目すべて通った")


# ------------------------------------------------------------------ 入口

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--units", default="*", help="単位の名前の glob（例：AFEW_IEMOCAP_f0_r0）")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="試走用：単位がそろわなくても集計する")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.summarize:
        summarize(a.out, a.allow_partial); return
    commit, dirty = dai1.git_commit()
    fz = json.load(open(FROZEN))
    a.out.mkdir(parents=True, exist_ok=True)
    units = sorted(UD.glob(f"{a.units}_p3.npz"))
    prov_path = a.out / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {"runs": []}
    run = dict(commit=commit, dirty=dirty, argv=sys.argv, host=platform.node(), python=platform.python_version(),
               torch=torch.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               frozen_sha256=dai1.sha(FROZEN), n_null=N_NULL, n_null_pool=N_NULL_POOL, n_boot=N_BOOT, min_n=MIN_N, min_cell=MIN_CELL,
               min_half=MIN_HALF, min_share=MIN_SHARE, floor_d=FLOOR_D, floor_l=FLOOR_L, null_ratio=NULL_RATIO, h_max=H_MAX,
               n_null_h=N_NULL_H, half_t=HALF_T,
               eps_d=EPS_D, eps_lda=EPS_LDA, units=[])
    prov["runs"].append(run)
    for p3 in units:
        res = run_unit(p3, a.out, fz, commit, dirty)
        res["inputs_sha256"] = {f"p{i}": dai1.sha(UD / f"{unit_name(p3)}_p{i}.npz") for i in (1, 2, 3)}
        run["units"].append(res)
        prov_path.write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    print(f"{len(units)} 単位を終えた", flush=True)


if __name__ == "__main__":
    main()
