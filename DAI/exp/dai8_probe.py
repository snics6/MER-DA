"""DAI-8 の下調べ：ターゲットのラベル無しの発話の分布で，各感情の平均と共通の共分散を推定し直す
（設計 DAI/docs/実験8_下調べ_設計.md）。

単位の段：  python DAI/exp/dai8_probe.py --out results/dai8_probe [--units 名前,...]
集計の段：  python DAI/exp/dai8_probe.py --out results/dai8_probe --summarize
自己検査：  python DAI/exp/dai8_probe.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
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
import dai7_probe_v2 as P7  # noqa: E402   parse_tether，p1_wb，pick，UD，FROZEN（同じ保存の読み方）

ROOT = HERE.parents[1]
UD, FROZEN = P7.UD, P7.FROZEN
SPACES = ("A", "V", "T", "AVT")
BLOCKS = {"AVT": (("A", 1024), ("V", 512), ("T", 768))}
ALPHAS = (0.1, 0.5, 0.9)
BETAS = (0.0, 1.0)
FIX = (0.5, 1.0)              # 規則で数える点（4 節．計算の前に固定）
N_EM = 10
KNN = 10
KS_ALL = (0, 1, 5, 10, 25)
KS_RULE = (1, 5, 10, 25)
MIN_D = 0.5                   # UAR の差の下限（ポイント）
R4_TOL = -2.0                 # R4：LDA_oracle − oracle の許す差（ポイント）
SEED = 0
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64


# ------------------------------------------------------------------ ガウスの判別

def lda_from_R(X: torch.Tensor, R: torch.Tensor, XtX: torch.Tensor, alpha: float, block=None):
    """重み R (n, 4)（各行の和は 1）で μ と共通の共分散を推定し，判別の係数 (A (d,4), c (4,)) と π を返す。
    S = (XᵀX − Σ_k n_k μ_k μ_kᵀ)/N（行の和が 1 なので Σ_i (Σ_k R_ik) x_i x_iᵀ = XᵀX）．Σ = (1 − α)S + αI（特徴は候補で標準化）。"""
    n_k = R.sum(0).clamp(min=1e-9)
    mu = (R.T @ X) / n_k[:, None]                                            # (4, d)
    N = n_k.sum()
    Sm = (XtX - (mu.T * n_k) @ mu) / N
    if block is not None:
        mask = torch.zeros_like(Sm, dtype=torch.bool)
        o = 0
        for _, w in block:
            mask[o:o + w, o:o + w] = True
            o += w
        Sm = torch.where(mask, Sm, torch.zeros_like(Sm))
    d = X.shape[1]
    Sig = (1.0 - alpha) * Sm + alpha * torch.eye(d, dtype=X.dtype, device=X.device)
    Lc = torch.linalg.cholesky(Sig)
    A = torch.cholesky_solve(mu.T.contiguous(), Lc)                          # Σ⁻¹ μᵀ (d, 4)
    c = -0.5 * (mu * A.T).sum(1)
    return A, c, (n_k / N)


def delta(X, A, c):
    return X @ A + c


def em_fit(X, XtX, kpos, ylab, q, alpha, beta, block=None, n_iter=N_EM):
    """K 本（kpos，正解 ylab）を one-hot に固定し，ラベル無しの候補の重みを EM で直す。q は none[K] の候補の確率。"""
    n = X.shape[0]
    R = q.clone()
    if len(kpos):
        R[kpos] = 0.0
        R[kpos, ylab] = 1.0
    U = torch.ones(n, dtype=torch.bool, device=X.device)
    U[kpos] = False
    logq = torch.log(q[U].clamp(min=1e-12))
    hist = []
    for _ in range(n_iter):
        A, c, pi = lda_from_R(X, R, XtX, alpha, block)
        lg = beta * logq + torch.log(pi.clamp(min=1e-12)) + delta(X[U], A, c)
        newU = torch.softmax(lg, -1)
        hist.append(float((newU - R[U]).abs().sum(1).mean()))
        R[U] = newU
    A, c, pi = lda_from_R(X, R, XtX, alpha, block)
    return A, c, pi, R, hist


# ------------------------------------------------------------------ 近傍（Q1）

def knn_stats(X: torch.Tensor, y: np.ndarray, g: np.ndarray, k=KNN) -> dict:
    Xn = torch.nn.functional.normalize(X.float(), dim=1)
    n = Xn.shape[0]
    nb = torch.empty((n, k), dtype=torch.long, device=X.device)
    for s in range(0, n, 2048):
        sim = Xn[s:s + 2048] @ Xn.T
        idx = torch.arange(s, min(s + 2048, n), device=X.device)
        sim[torch.arange(len(idx), device=X.device), idx] = -2.0             # 自分を除く
        nb[s:s + 2048] = sim.topk(k, dim=1).indices
    nb = nb.cpu().numpy()
    same_y = (y[nb] == y[:, None]).mean(1)
    emo = float(np.mean([same_y[y == e].mean() for e in range(4) if (y == e).any()]))
    _, gi = np.unique(g, return_inverse=True)
    grp = float((gi[nb] == gi[:, None]).mean())
    p = np.bincount(gi) / len(gi)
    g0 = float((p ** 2).sum())
    return dict(emo=emo, emo0=0.25, emo_gain=(emo - 0.25) / 0.75, grp=grp, grp0=g0, grp_gain=((grp - g0) / (1 - g0) if g0 < 1 else np.nan))


# ------------------------------------------------------------------ 単位

def to_dev(a):
    return torch.as_tensor(np.asarray(a), dtype=DT, device=DEV)


def logits_np(X: torch.Tensor, W, b) -> torch.Tensor:
    return X @ to_dev(W).T + to_dev(b)


def run_unit(p3: Path, out: Path, fz: dict) -> dict:
    name = P7.unit_name(p3)
    T, c, fs, rs = name.split("_")
    f, r = int(fs[1:]), int(rs[1:])
    dst = out / "rows" / f"{name}.npz"
    if dst.exists() or (out / "rows" / f"{name}_knn.csv").exists():
        raise SystemExit(f"{dst} は既にある（重複を作らない）")
    t0 = time.time()
    F = fz[T]
    srcs = [x for x in DD.CORPORA if x not in (T, c)]
    z1, z3 = np.load(UD / f"{name}_p1.npz"), np.load(UD / f"{name}_p3.npz")
    for i, z in ((1, z1), (3, z3)):
        if str(z["commit"]) != F["pseudo_commit"]:
            raise SystemExit(f"{name}_p{i} のコミットが固定のファイルの pseudo_commit と違う")
    tether = P7.parse_tether(z3["tether"])
    u = dai1.Unit(ROOT / "results" / "dai1", T, c, srcs, f, r)
    sid_t = u.cp.df["sample_id"].to_numpy()[u.test].astype(str)
    sid_p = u.cp.df["sample_id"].to_numpy()[u.pool].astype(str)
    if not ((sid_t == z1["sample_test"].astype(str)).all() and (sid_p == z1["sample_pool"].astype(str)).all()):
        raise SystemExit(f"{name}：発話の並びが保存と違う")
    y_of, cl_of = ST.label_maps(c)
    yt, yp = u.ytest.astype(int), u.ypool.astype(int)
    if not ((y_of(sid_t) == yt).all() and (y_of(sid_p) == yp).all()):
        raise SystemExit(f"{name}：発話の並びとラベルの対応が読み込みと違う")
    gcol = "speaker_id" if c == "IEMOCAP" else "cluster"
    grp_p = u.cp.df[gcol].to_numpy()[u.pool].astype(str)
    ver = {K: F["main"][str(K)] for K in KS_ALL}
    pred, diag, knn_rows, checks = {}, {}, [], dict(none_mismatch=0, none_compared=0)
    yp_t = torch.as_tensor(yp, device=DEV)
    onehot_all = torch.nn.functional.one_hot(yp_t, 4).to(DT)
    for s in SPACES:
        Xp, Xt = u.Xp[s].to(DEV, DT), u.Xt[s].to(DEV, DT)
        XtX = Xp.T @ Xp
        block = BLOCKS.get(s)
        knn_rows.append(dict(space=s, **knn_stats(Xp, yp, grp_p)))
        cw25, cb25 = tether[(s, 25)]
        Wo, bo = P7.p1_wb(z1, s, f"all|{cw25}|{cb25}")
        pred[f"{s}_oracle"] = logits_np(Xt, Wo, bo).argmax(1)
        for a in ALPHAS:
            A, cc, _ = lda_from_R(Xp, onehot_all, XtX, a)
            pred[f"{s}_ldaO_a{a}"] = delta(Xt, A, cc).argmax(1)
        if block is not None:
            A, cc, _ = lda_from_R(Xp, onehot_all, XtX, FIX[0], block)
            pred[f"{s}_ldaOblk_a{FIX[0]}"] = delta(Xt, A, cc).argmax(1)
        for K in KS_ALL:
            if K == 0:
                W0, b0 = u.start[s]
                Wn, bn = W0.cpu().numpy(), b0.cpu().numpy()
                saved = z1[f"p1_{s}_start_pred"]
            else:
                cw, cb = tether[(s, K)]
                Wn, bn = P7.p1_wb(z1, s, f"{K}|{cw}|{cb}")
                items = [str(x) for x in z1[f"p1_{s}_items"]]
                saved = z1[f"p1_{s}_pred"][items.index(f"{K}|{cw}|{cb}")]
            pn = logits_np(Xt, Wn, bn).argmax(1)
            checks["none_mismatch"] += int((pn.cpu().numpy() != np.asarray(saved)).sum())
            checks["none_compared"] += len(pn)
            pred[f"K{K}_{s}_none"] = pn
            sel = {k: v for k, v in F["select"][f"{s}|{K}|joint|{ver[K]}"].items() if k != "pseudo_uar"}
            Wj, bj = P7.pick(z3, f"p3_{s}_K{K}", dict(method="joint", teacher=ver[K], **sel))
            pred[f"K{K}_{s}_joint"] = logits_np(Xt, Wj, bj).argmax(1)
            q = torch.softmax(logits_np(Xp, Wn, bn), -1)
            kpos = u.kpos[K]
            kpos_t = torch.as_tensor(kpos, dtype=torch.long, device=DEV)
            ylab = yp_t[kpos_t]
            Umask = np.ones(len(yp), bool)
            Umask[kpos] = False
            d0 = dict(pool_uar_none=float(ST.uar_full(ST.counts(yp[Umask], q.argmax(1).cpu().numpy()[Umask], np.zeros(Umask.sum(), int)))))
            if K >= 1:
                XL = Xp[kpos_t]
                RL = torch.nn.functional.one_hot(ylab, 4).to(DT)
                for a in ALPHAS:
                    A, cc, _ = lda_from_R(XL, RL, XL.T @ XL, a)
                    pred[f"K{K}_{s}_ldaK_a{a}"] = delta(Xt, A, cc).argmax(1)
            for a in ALPHAS:
                for b in BETAS:
                    A, cc, pi, R, hist = em_fit(Xp, XtX, kpos_t, ylab, q, a, b)
                    key = f"K{K}_{s}_em_a{a}_b{b}"
                    pred[key] = delta(Xt, A, cc).argmax(1)
                    pr = R.argmax(1).cpu().numpy()
                    diag[key] = dict(hist=hist, pi=pi.cpu().tolist(),
                                     pool_uar=float(ST.uar_full(ST.counts(yp[Umask], pr[Umask], np.zeros(Umask.sum(), int)))),
                                     **d0)
            if block is not None:
                A, cc, pi, R, hist = em_fit(Xp, XtX, kpos_t, ylab, q, FIX[0], FIX[1], block)
                key = f"K{K}_{s}_emblk_a{FIX[0]}_b{FIX[1]}"
                pred[key] = delta(Xt, A, cc).argmax(1)
                pr = R.argmax(1).cpu().numpy()
                diag[key] = dict(hist=hist, pi=pi.cpu().tolist(),
                                 pool_uar=float(ST.uar_full(ST.counts(yp[Umask], pr[Umask], np.zeros(Umask.sum(), int)))), **d0)
        del Xp, Xt, XtX
        torch.cuda.empty_cache()
    pool_prop = (np.bincount(yp, minlength=4) / len(yp)).tolist()
    (out / "rows").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, test_sample_id=sid_t, test_y=yt.astype(np.int8),
                        test_cluster=cl_of(sid_t).astype(str), pool_prop=np.array(pool_prop),
                        diag=json.dumps(diag), checks=json.dumps(checks),
                        **{k: v.cpu().numpy().astype(np.int8) for k, v in pred.items()})
    pd.DataFrame(knn_rows).assign(T=T, c=c, fold=f, seed=r).to_csv(out / "rows" / f"{name}_knn.csv", index=False)
    del u
    torch.cuda.empty_cache()
    sec = time.time() - t0
    print(f"{name}：{sec:.0f} 秒（none の判定の照合の不一致 {checks['none_mismatch']} / {checks['none_compared']}）", flush=True)
    return dict(name=name, seconds=sec, **checks)


# ------------------------------------------------------------------ 集計

COMPARE = {   # 名前 → (A の方法, B の方法)．{s} は空間，{K} は K
    "em_none": ("K{K}_{s}_em_a0.5_b1.0", "K{K}_{s}_none"),
    "em_joint": ("K{K}_{s}_em_a0.5_b1.0", "K{K}_{s}_joint"),
    "em_ldaK": ("K{K}_{s}_em_a0.5_b1.0", "K{K}_{s}_ldaK_a0.5"),
    "ldaK_none": ("K{K}_{s}_ldaK_a0.5", "K{K}_{s}_none"),
    "joint_none": ("K{K}_{s}_joint", "K{K}_{s}_none"),
    "emblk_em": ("K{K}_{s}_emblk_a0.5_b1.0", "K{K}_{s}_em_a0.5_b1.0"),
    **{f"em_a{a}_b{b}_none": (f"K{{K}}_{{s}}_em_a{a}_b{b}", "K{K}_{s}_none") for a in ALPHAS for b in BETAS if (a, b) != FIX},
}
COMPARE_K0 = {   # K に依らない比べ（K = 0 の行に書く）
    "ldaO_oracle": ("{s}_ldaO_a0.5", "{s}_oracle"),
    "ldaOblk_ldaO": ("{s}_ldaOblk_a0.5", "{s}_ldaO_a0.5"),
}


def summarize(out: Path) -> None:
    files = sorted((out / "rows").glob("*.npz"))
    if len(files) != 120:
        print(f"注意：単位の出力が {len(files)} 個（120 のはず）", flush=True)
    rng = np.random.default_rng(SEED)
    cells, cellrows, diagrows = {}, [], []
    for p in files:
        T, c, fs, rs = p.name[:-4].split("_")
        f, r = int(fs[1:]), int(rs[1:])
        z = np.load(p, allow_pickle=False)
        y, cl = z["test_y"].astype(int), z["test_cluster"].astype(str)
        for k in z.files:
            if k.startswith(("K", "A_", "V_", "T_", "AVT_")) and z[k].dtype == np.int8 and z[k].shape == y.shape:
                cnt = ST.counts(y, z[k].astype(int), cl)
                cells.setdefault((c, k), {})[(f, (T, r))] = cnt
                cellrows.append(dict(c=c, T=T, fold=f, seed=r, key=k, uar=ST.uar_full(cnt)))
        for k, d in json.loads(str(z["diag"])).items():
            diagrows.append(dict(c=c, T=T, fold=f, seed=r, key=k, pool_uar=d["pool_uar"], pool_uar_none=d["pool_uar_none"],
                                 hist_last=d["hist"][-1], hist_first=d["hist"][0],
                                 pi_err=float(np.abs(np.array(d["pi"]) - z["pool_prop"]).sum())))
    U = pd.DataFrame(cellrows)
    Cm = U.groupby(["c", "key"]).uar.agg(["mean", "count"]).reset_index()
    Cm.to_csv(out / "cells.csv", index=False)
    Dg = pd.DataFrame(diagrows).groupby(["c", "key"]).mean(numeric_only=True).reset_index()
    Dg.to_csv(out / "diag.csv", index=False)
    rows = []
    for c in sorted({k[0] for k in cells}):
        g_eff = ST.g_eff_of(c)[0]
        for s in SPACES:
            for nm, (a, b) in COMPARE_K0.items():
                ka, kb = a.format(s=s), b.format(s=s)
                if (c, ka) in cells and (c, kb) in cells:
                    rows.append(dict(c=c, K=0, space=s, compare=nm, **ST.effect_stats(cells[(c, ka)], cells[(c, kb)], g_eff, rng)))
            for K in KS_ALL:
                for nm, (a, b) in COMPARE.items():
                    ka, kb = a.format(K=K, s=s), b.format(K=K, s=s)
                    if (c, ka) in cells and (c, kb) in cells:
                        rows.append(dict(c=c, K=K, space=s, compare=nm, **ST.effect_stats(cells[(c, ka)], cells[(c, kb)], g_eff, rng)))
        print(f"{c}：区間を作った", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(out / "diffs.csv", index=False)
    knn = pd.concat([pd.read_csv(p) for p in sorted((out / "rows").glob("*_knn.csv"))])
    knn.to_csv(out / "knn_units.csv", index=False)
    # 規則（4 節）
    crit = []
    for c in sorted(D.c.unique()):
        r = dict(c=c)
        for tag, nm in (("R1", "em_none"), ("R2", "em_joint"), ("R3", "em_ldaK")):
            d = D[(D.c == c) & (D.space == "AVT") & (D.compare == nm) & D.K.isin(KS_RULE)]
            ok = (d.lo > 0) & (d.effect >= MIN_D)
            r[f"{tag}_n_K"] = int(ok.sum())
            r[f"{tag}_n_lo_pos"] = int((d.lo > 0).sum())
            r[f"{tag}_n_hi_neg"] = int((d.hi < 0).sum())
            r[tag] = bool(ok.sum() >= 3)
        r["R3_harm"] = bool(r["R3_n_hi_neg"] >= 3)
        d4 = D[(D.c == c) & (D.space == "AVT") & (D.compare == "ldaO_oracle")]
        r["R4_effect"] = float(d4.effect.iloc[0]) if len(d4) else np.nan
        r["R4"] = bool(len(d4) and d4.effect.iloc[0] >= R4_TOL)
        kf = knn[(knn.c == c) & (knn.space == "AVT")].groupby("fold")[["emo_gain", "grp_gain"]].mean()
        r["R5_n_fold_emo_lt_grp"] = int((kf.emo_gain < kf.grp_gain).sum())
        r["R5_n_fold"] = int(len(kf))
        crit.append(r)
    Cr = pd.DataFrame(crit)
    for tag in ("R1", "R2", "R3", "R4"):
        Cr[f"{tag}_n_corpora"] = int(Cr[tag].sum())
        Cr[f"{tag}_n_corpora_excl_IEMOCAP"] = int(Cr.loc[Cr.c != "IEMOCAP", tag].sum())
    Cr.to_csv(out / "criteria.csv", index=False)
    commit, dirty = dai1.git_commit()
    prov = json.loads((out / "provenance.json").read_text()) if (out / "provenance.json").exists() else {}
    prov.update(summarize_commit=commit, summarize_dirty=dirty, boot_seed=SEED, n_boot=ST.N_BOOT)
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"書いた：cells.csv，diag.csv，diffs.csv（{len(D)} 行），knn_units.csv，criteria.csv", flush=True)


# ------------------------------------------------------------------ 自己検査

def selftest() -> None:
    g = torch.Generator().manual_seed(0)
    d, n = 6, 4000
    mu = torch.randn(4, d, generator=g, dtype=DT) * 2
    y = torch.randint(0, 4, (n,), generator=g)
    X = (mu[y] + torch.randn(n, d, generator=g, dtype=DT)).to(DEV)
    y = y.to(DEV)
    R = torch.nn.functional.one_hot(y, 4).to(DT)
    A, c, pi = lda_from_R(X, R, X.T @ X, 0.0)
    acc = float((delta(X, A, c).argmax(1) == y).double().mean())
    # 直接の推定と比べる
    mu_hat = torch.stack([X[y == k].mean(0) for k in range(4)])
    Sw = sum(((X[y == k] - mu_hat[k]).T @ (X[y == k] - mu_hat[k])) for k in range(4)) / n
    A2 = torch.linalg.solve(Sw, mu_hat.T)
    print(f"(1) 係数の一致：最大の差 {float((A - A2).abs().max()):.2e}，正解率 {acc:.3f}")
    assert float((A - A2).abs().max()) < 1e-8
    # (2) EM：正しい初期値から始めると崩れない／K 本の行は固定
    q = torch.full((n, 4), 0.25, dtype=DT, device=DEV)
    kpos = torch.arange(0, 40, device=DEV)
    A3, c3, pi3, R3, hist = em_fit(X, X.T @ X, kpos, y[kpos], q, 0.1, 0.0)
    acc3 = float((delta(X, A3, c3).argmax(1) == y).double().mean())
    fixed = bool((R3[kpos].argmax(1) == y[kpos]).all() and torch.allclose(R3[kpos].sum(1), torch.ones(40, dtype=DT, device=DEV)))
    print(f"(2) 一様の初期値と K 本 40 からの EM の正解率 {acc3:.3f}（直接の LDA {acc:.3f}），K 本の行の固定 {fixed}，r の変化 {hist[0]:.3f} → {hist[-1]:.4f}")
    assert fixed and acc3 > acc - 0.05
    # (3) ブロック対角：ブロックの外が 0
    Xb = torch.cat([X, X[:, :2] + 0.1 * torch.randn(n, 2, generator=g, dtype=DT).to(DEV)], 1)
    blk = (("a", d), ("b", 2))
    A4, c4, _ = lda_from_R(Xb, R, Xb.T @ Xb, 0.5, blk)
    print(f"(3) ブロック対角の解は有限：{bool(torch.isfinite(A4).all())}")
    # (4) 近傍：同じ感情しか近傍に無い作りで 1 に近い
    ks = knn_stats(X, y.cpu().numpy(), np.zeros(n, int))
    print(f"(4) 近傍の感情の一致 {ks['emo']:.3f}（よく分かれた人工データ），グループ 1 つの一致 {ks['grp']:.3f}")
    assert ks["grp"] == 1.0
    print("自己検査：通った")


# ------------------------------------------------------------------ 入口

def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "dai8_probe")
    ap.add_argument("--units", default=None, help="単位の名前をコンマで（試走）")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    out = a.out if a.out.is_absolute() else ROOT / a.out
    if a.summarize:
        summarize(out)
        return
    commit, dirty = dai1.git_commit()
    fz = json.loads(FROZEN.read_text())
    p3s = sorted(UD.glob("*_p3.npz"))
    if a.units:
        want = set(a.units.split(","))
        p3s = [p for p in p3s if P7.unit_name(p) in want]
    else:
        if out.exists():
            raise SystemExit(f"{out} は既にある（本計算は空の出力先に書く）")
        if dirty:
            raise SystemExit("作業木に未コミットの変更がある（本計算はコミットしてから）")
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(commit=commit, dirty=dirty, started=time.strftime("%F %T"), python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                cuda=torch.version.cuda, device=str(DEV), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                frozen_sha256=sha(FROZEN), alphas=ALPHAS, betas=BETAS, fix=FIX, n_em=N_EM, knn=KNN, min_d=MIN_D, r4_tol=R4_TOL,
                units=[P7.unit_name(p) for p in p3s],
                inputs_sha256={P7.unit_name(p): {i: sha(UD / f"{P7.unit_name(p)}_p{i}.npz") for i in (1, 3)} for p in p3s})
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"開始 {prov['started']} コミット {commit[:7]} 単位 {len(p3s)} 個 機器 {prov['gpu']}", flush=True)
    runs = [run_unit(p, out, fz) for p in p3s]
    prov["runs"] = runs
    prov["finished"] = time.strftime("%F %T")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"{len(runs)} 単位を終えた", flush=True)


if __name__ == "__main__":
    main()
