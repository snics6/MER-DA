"""DAI-8 の下調べ：ターゲットのラベル無しの発話の分布で，各感情の平均と共通の共分散を推定し直す
（設計 DAI/docs/実験8_下調べ_設計.md 第 3 版）。

単位の段：  python DAI/exp/dai8_probe.py --out results/dai8_probe [--units 名前,...]
集計の段：  python DAI/exp/dai8_probe.py --out results/dai8_probe --summarize [--allow-partial]
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
import dai7_probe_v2 as P7  # noqa: E402   parse_tether，p1_wb，pick，unit_name，UD，FROZEN（同じ保存の読み方）
import dai3  # noqa: E402   部分集合の印と IEMOCAP の筋書き

ROOT = P7.ROOT
UD, FROZEN = P7.UD, P7.FROZEN
SPACES = ("A", "V", "T", "AVT")
BLOCKS = {"AVT": (("A", 1024), ("V", 512), ("T", 768))}
DIM = {"A": 1024, "V": 512, "T": 768, "AVT": 2304}
ALPHAS = (0.1, 0.5, 0.9)
BETAS = (0.0, 1.0)
FIX = (0.5, 1.0)              # 規則で数える点（4 節．計算の前に固定）
N_EM = 10                     # 規則で数える回数
MAX_EM, TOL_EM = 100, 1e-4    # 収束まで回す版（固定の点とブロック対角．gate-code の 1）
SNAPS = (0, 10, 30)           # 保存する回（0 = 初期の重みで 1 回だけ解いた版．gate-code の 4）
FIX_TETHER = (0.01, 1.0)      # 選ばない引き戻しの点（gate-code の 7．計算の前に格子の中から固定）
KNN = 10
KS_ALL = (0, 1, 5, 10, 25)
KS_RULE = (1, 5, 10, 25)
MIN_D = 0.5                   # UAR の差の下限（ポイント）
R4_TOL = -2.0                 # R4：LDA_oracle − oracle の点推定の許す差（ポイント）
MOVED_MIN = 0.05              # 3.3 節：動いた割合がこれ未満なら「ほとんど動かなかった」
RATIO_MAX = 10.0              # 3.3 節：項の大きさの比がこれを超えれば β の区別がほとんど無い
CONV = 0.01                   # 3.3 節：規則の回（10 回目）の r の変化がこれ未満で「収束した」
N_PERM = 5                    # 上位 3 つの固有値の和の並べ替えの帰無の回数
KEEP_RAND = 0.9               # review_ml の M1：でたらめのラベルから始めた EM がこの割合以上を保てば「どの初期値も保つ」
KEEP_GAP = 0.2                # review_ml の第 2 回の N1：正解とでたらめの保った割合の差がこれ未満でも「どの初期値も保つ」
R9_TEST_DROP = -2.0           # R9 の段 1：EM_truth − LDA_oracle のテストの UAR の点推定がこれ以下なら「頂上が感情から離れる」
DRIFT = 0.5                   # 「流れた」：none から始めた EM の候補の上の UAR が初期値より 0.5 ポイント以上下がった単位が過半（gate-probe の第 2 回の 4）
N_UNITS = 120
SEED = 0
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.6)                       # WSL2 の割り当ての落とし穴（記憶 wsl2-cuda-allocator）


# ------------------------------------------------------------------ ガウスの判別

def lda_from_R(X: torch.Tensor, R: torch.Tensor, XtX: torch.Tensor, alpha: float, block=None, chol=None):
    """重み R (n, 4)（各行の和は 1）で μ と共通の共分散を推定し，判別の係数を返す。
    S = (XᵀX − Σ_k n_k μ_k μ_kᵀ)/N（行の和が 1 なので Σ_i (Σ_k R_ik) x_i x_iᵀ = XᵀX）．Σ = (1 − α)S + αI（特徴は候補で標準化）。
    戻り：dict(A (d,4) = Σ⁻¹μᵀ，c (4,)，pi (4,)，mu (4,d)，chol)。chol を与えれば Σ を固定する（EM_μ）。"""
    n_k = R.sum(0).clamp(min=1e-9)
    mu = (R.T @ X) / n_k[:, None]
    N = n_k.sum()
    if chol is None:
        Sm = (XtX - (mu.T * n_k) @ mu) / N
        if block is not None:
            mask = torch.zeros_like(Sm, dtype=torch.bool)
            o = 0
            for _, w in block:
                mask[o:o + w, o:o + w] = True
                o += w
            Sm = torch.where(mask, Sm, torch.zeros_like(Sm))
        Sig = (1.0 - alpha) * Sm + alpha * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
        chol = torch.linalg.cholesky(Sig)
    A = torch.cholesky_solve(mu.T.contiguous(), chol)
    c = -0.5 * (mu * A.T).sum(1)
    if not (torch.isfinite(A).all() and torch.isfinite(c).all()):
        raise FloatingPointError("判別の係数に有限でない値がある")
    return dict(A=A, c=c, pi=n_k / N, mu=mu, chol=chol, n_k=n_k)


def tr_sinv_s(f, alpha, d):
    """tr(Σ⁻¹S) = (d − α·tr(Σ⁻¹))/(1 − α)（review_ml の m3：少ない本数の感情の平均の誤差の項 ½·tr(Σ⁻¹S)/n_k）。"""
    if alpha >= 1:
        return float("nan")
    tri = float(torch.cholesky_inverse(f["chol"]).diagonal().sum())
    return (d - alpha * tri) / (1 - alpha)


def delta(X, f):
    return X @ f["A"] + f["c"]


def em_fit(X, XtX, kpos, ylab, q, alpha, beta, block=None, n_iter=N_EM, fix_sigma=False, tol=None, snaps=(), yU=None):
    """K 本（kpos，正解 ylab）を one-hot に固定し，ラベル無しの候補の重みを EM で直す。q は初期値と事前の候補の確率。
    回 t の分類器 = t 回の E 段の後の M 段の解（t = 0 は初期の重みで 1 回だけ解いた版）。tol を与えれば変化が tol 未満で止める。
    戻り：dict(fit = n_iter 回目（tol のときは最後）の解，snap = {t: 解}，R，hist，info)。"""
    n = X.shape[0]
    R = q.clone()
    if len(kpos):
        R[kpos] = 0.0
        R[kpos, ylab] = 1.0
    U = torch.ones(n, dtype=torch.bool, device=X.device)
    U[kpos] = False
    logq = torch.log(q[U].clamp(min=1e-12))
    hist, pool_hist, snap, chol, ratio, beta_flip = [], [], {}, None, float("nan"), 0.0
    t = 0
    while True:
        f = lda_from_R(X, R, XtX, alpha, block, chol=chol)
        if fix_sigma and chol is None:
            chol = f["chol"]                                                 # 最初の M 段の Σ を固定する（gate-probe の 5）
        if t in snaps:
            snap[t] = f
        if yU is not None:
            pool_hist.append(float(ST.uar_full(ST.counts(yU, R[U].argmax(1).cpu().numpy(), np.zeros(len(yU), int)))))
        stop = (t == n_iter) or (tol is not None and t > 0 and hist[-1] < tol)
        if stop:
            break
        dl = delta(X[U], f)
        if t == 0:                                                           # 3.3 節の項の大きさの比
            rd = (dl.max(1).values - dl.min(1).values).median()
            rq = (logq.max(1).values - logq.min(1).values).median()
            ratio = float(rd / rq) if float(rq) > 0 else float("inf")
        base = torch.log(f["pi"].clamp(min=1e-12)) + dl
        newU = torch.softmax(beta * logq + base, -1)
        beta_flip = float(((beta * logq + base).argmax(1) != base.argmax(1)).double().mean()) if beta != 0 else 0.0
        hist.append(float((newU - R[U]).abs().sum(1).mean()))
        R[U] = newU
        t += 1
    rU = R[U]
    info = dict(ratio=ratio, moved=float((rU.argmax(1) != q[U].argmax(1)).double().mean()),
                entropy=float(-(rU.clamp(min=1e-12).log() * rU).sum(1).mean()), n_iter_done=t,
                beta_flip_last=beta_flip)                                       # 最後の E 段で β の項が判定を変えた割合（review_ml の m1）
    return dict(fit=f, snap=snap, R=R, hist=hist, pool_hist=pool_hist, info=info, U=U)


# ------------------------------------------------------------------ 近傍と散らばり（Q1）

def _topk(Xn, k, block_same=None):
    """コサインの k 近傍（自分を除く）．block_same（整数の列）が与えられれば，同じ値の発話を近傍から外す。"""
    n = Xn.shape[0]
    nb = torch.empty((n, k), dtype=torch.long, device=Xn.device)
    for s in range(0, n, 2048):
        sim = Xn[s:s + 2048] @ Xn.T
        idx = torch.arange(s, min(s + 2048, n), device=Xn.device)
        sim[torch.arange(len(idx), device=Xn.device), idx] = -2.0
        if block_same is not None:
            sim[block_same[s:s + 2048][:, None] == block_same[None, :]] = -2.0
        nb[s:s + 2048] = sim.topk(k, dim=1).indices
    return nb.cpu().numpy()


def _bal(v, y):
    return float(np.mean([v[y == e].mean() for e in range(4) if (y == e).any()]))


def _means(X, lab):
    lab_t = torch.as_tensor(lab, device=X.device)
    nk = torch.bincount(lab_t).to(X.dtype)
    M = torch.zeros((len(nk), X.shape[1]), dtype=X.dtype, device=X.device).index_add_(0, lab_t, X) / nk.clamp(min=1)[:, None]
    return M, nk


def _between(X, lab):
    """tr(S_b)/tr(S_t)：ラベル lab の平均の違いが全体の散らばりに占める割合。"""
    Xc = X - X.mean(0)
    M, nk = _means(Xc, lab)
    return float((nk[:, None] * M ** 2).sum() / (Xc ** 2).sum())


def _perm(lab, seed=SEED):
    return np.random.default_rng(seed).permutation(lab)


def _fisher(X, lab, Ach):
    """白色化した計量での分かれやすさ tr(A⁻¹ S_b)，A = (1 − α)T + αI（ラベルに依らない計量．Woodbury の導出．設計 3.4 節）。"""
    Xc = X - X.mean(0)
    M, nk = _means(Xc, lab)
    W = torch.cholesky_solve(M.T.contiguous(), Ach)                          # A⁻¹ Mᵀ
    return float((nk * (M.T * W).sum(0)).sum() / X.shape[0])


def _top3(X, lab, Ach):
    """白色化した計量での上位 3 つの固有値の和：S_b v = λ A v の上位 3 つ（4 成分の EM が使える向きは 3 つ．gate-probe の第 2 回の 1）。"""
    Xc = X - X.mean(0)
    M, nk = _means(Xc, lab)
    w = (nk / X.shape[0]).sqrt()
    if M.shape[0] <= X.shape[1]:
        W = torch.cholesky_solve(M.T.contiguous(), Ach)
        Gm = (M @ W) * w[:, None] * w[None, :]
        ev = torch.linalg.eigvalsh((Gm + Gm.T) / 2)
    else:
        Z = torch.linalg.solve_triangular(Ach, (M.T * w).contiguous(), upper=False)
        ev = torch.linalg.eigvalsh(Z @ Z.T)
    return float(ev.sort(descending=True).values[:3].sum())


def _top3_null(X, lab, Ach, n_perm=N_PERM):
    return float(np.mean([_top3(X, np.random.default_rng(SEED + 1 + i).permutation(lab), Ach) for i in range(n_perm)]))


def knn_stats(X: torch.Tensor, y: np.ndarray, g: np.ndarray, dlg: np.ndarray, scen=None, k=KNN) -> dict:
    """Q1：近傍の感情・グループ・対話の一致（review_affect の A2），同じ対話を外した近傍，グループで中心化した版，
    散らばりの割合と白色化した計量での分かれやすさ。"""
    Xd = X.double()
    Xn = torch.nn.functional.normalize(Xd.float(), dim=1)
    _, gi = np.unique(g, return_inverse=True)
    _, di = np.unique(dlg, return_inverse=True)
    nb = _topk(Xn, k)
    out = dict(emo=_bal((y[nb] == y[:, None]).mean(1), y), emo0=0.25,
               grp=float((gi[nb] == gi[:, None]).mean()), dlg=float((di[nb] == di[:, None]).mean()))
    pg, pdl = np.bincount(gi) / len(gi), np.bincount(di) / len(di)
    out["grp0"], out["dlg0"] = float((pg ** 2).sum()), float((pdl ** 2).sum())
    out["emo_gain"] = (out["emo"] - 0.25) / 0.75
    out["grp_gain"] = (out["grp"] - out["grp0"]) / (1 - out["grp0"]) if out["grp0"] < 1 else np.nan
    out["dlg_gain"] = (out["dlg"] - out["dlg0"]) / (1 - out["dlg0"]) if out["dlg0"] < 1 else np.nan
    if scen is not None:
        _, si = np.unique(scen, return_inverse=True)
        out["scen"] = float((si[nb] == si[:, None]).mean())
        out["scen0"] = float(((np.bincount(si) / len(si)) ** 2).sum())
    if len(pdl) < 2:
        raise SystemExit("対話の印が 1 種類しか無い")
    nbx = _topk(Xn, k, torch.as_tensor(di, device=X.device))                # 同じ対話を外した近傍
    out["emo_xdlg"] = _bal((y[nbx] == y[:, None]).mean(1), y)
    out["emo0_xdlg"] = _bal(_excl_chance(y, di), y)
    out["emo_gain_xdlg"] = (out["emo_xdlg"] - out["emo0_xdlg"]) / (1 - out["emo0_xdlg"])
    if scen is not None:                                                     # 同じ筋書きを外した近傍（IEMOCAP．review_affect の第 2 回の B2）
        nbs = _topk(Xn, k, torch.as_tensor(si, device=X.device))
        out["emo_xscen"] = _bal((y[nbs] == y[:, None]).mean(1), y)
        out["emo0_xscen"] = _bal(_excl_chance(y, si), y)
        out["emo_gain_xscen"] = (out["emo_xscen"] - out["emo0_xscen"]) / (1 - out["emo0_xscen"])
    for nm, lab in (("emo", y), ("grp", gi), ("dlg", di)):                 # でたらめの水準：同じ大きさの組に並べ替えたラベル
        out[f"{nm}_between"], out[f"{nm}_between0"] = _between(Xd, lab), _between(Xd, _perm(lab))
    out["n_groups"], out["n_dialogues"] = int(len(pg)), int(len(pdl))
    Xc = Xd - Xd.mean(0)
    T = Xc.T @ Xc / Xd.shape[0]
    Ach = torch.linalg.cholesky((1 - FIX[0]) * T + FIX[0] * torch.eye(Xd.shape[1], dtype=Xd.dtype, device=X.device))
    for nm, lab in (("emo", y), ("grp", gi), ("dlg", di)):
        out[f"fisher_{nm}"], out[f"fisher0_{nm}"] = _fisher(Xd, lab, Ach), _fisher(Xd, _perm(lab), Ach)
        out[f"top3_{nm}"], out[f"top3null_{nm}"] = _top3(Xd, lab, Ach), _top3_null(Xd, lab, Ach)
        out[f"top3ex_{nm}"] = out[f"top3_{nm}"] - out[f"top3null_{nm}"]
    Gm, _ = _means(Xd, gi)                                                   # グループの平均を引いた版（次の試しの手がかり）
    Xg = Xd - Gm[torch.as_tensor(gi, device=X.device)]
    nbg = _topk(torch.nn.functional.normalize(Xg.float(), dim=1), k)
    out["emo_gc"] = _bal((y[nbg] == y[:, None]).mean(1), y)
    out["emo_gain_gc"] = (out["emo_gc"] - 0.25) / 0.75
    out["emo_between_gc"] = _between(Xg, y)
    return out


def dialogue_ids(c: str, sid: np.ndarray, df_dialogue: np.ndarray) -> np.ndarray:
    """対話の印（review_affect の第 2 回の B1：dai_data の dialogue 列は MELD 以外で空）。
    IEMOCAP は sample_id の最後の _ より前（対話），MOSEI は同じ（動画），AFEW は発話一覧の movie_title（作品），MELD は dialogue 列。"""
    if c in ("IEMOCAP", "MOSEI"):
        out = np.array([x.rsplit("_", 1)[0] for x in sid])
    elif c == "AFEW":
        man = pd.read_csv(ROOT / "data" / "metadata" / "afew_utterance_manifest.csv", usecols=["sample_id", "movie_title"])
        mp = dict(zip(man.sample_id.astype(str), man.movie_title.astype(str)))
        out = np.array([mp.get(x, "") for x in sid])
    else:
        out = np.asarray(df_dialogue).astype(str)
    if (out == "").any() or len(np.unique(out)) < 2:
        raise SystemExit(f"{c}：対話の印に空があるか，1 種類しか無い")
    return out


def _excl_chance(y, blk):
    """同じ blk の発話を外した後の候補での，自分の感情の割合（発話ごと）。"""
    cy = np.zeros((blk.max() + 1, 4))
    np.add.at(cy, (blk, y), 1)
    tot_y = np.bincount(y, minlength=4)
    rest = len(y) - cy.sum(1)[blk]
    if rest.min() < KNN:
        raise SystemExit("同じ印の発話を外すと，近傍の候補が k 本に足りない発話がある")
    return (tot_y[y] - cy[blk, y]) / rest


# ------------------------------------------------------------------ 単位

def to_dev(a):
    return torch.as_tensor(np.asarray(a), dtype=DT, device=DEV)


def logits_np(X: torch.Tensor, W, b) -> torch.Tensor:
    return X @ to_dev(W).T + to_dev(b)


def _uar_pool(yU, pr):
    return float(ST.uar_full(ST.counts(yU, pr, np.zeros(len(yU), int))))


def run_unit(p3: Path, out: Path, fz: dict, commit: str) -> dict:
    name = P7.unit_name(p3)
    T, c, fs, rs = name.split("_")
    f, r = int(fs[1:]), int(rs[1:])
    dst = out / "rows" / f"{name}.npz"
    if dst.exists():
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
    grp_t = u.cp.df[gcol].to_numpy()[u.test].astype(str)
    dlg_p = dialogue_ids(c, sid_p, u.cp.df["dialogue"].to_numpy()[u.pool])
    scen_p = dai3.scenario(sid_p) if c == "IEMOCAP" else None
    ver = {K: F["main"][str(K)] for K in KS_ALL}
    q1_here = (T == sorted(x for x in DD.CORPORA if x != c)[0]) and r == 0   # Q1 は (c, f) ごとに 1 回だけ（T が辞書順で最初，r = 0）
    pred, dlt, coef, poolpred, diag, knn_rows = {}, {}, {}, {}, {}, []
    checks = dict(none_mismatch=0, none_compared=0)
    yp_t = torch.as_tensor(yp, device=DEV)
    onehot_all = torch.nn.functional.one_hot(yp_t, 4).to(DT)

    def keep(key, ft, Xt, save_delta=False, save_coef=False):
        dt = delta(Xt, ft)
        pred[key] = dt.argmax(1)
        if save_delta:
            dlt[key] = dt
        if save_coef:
            coef[key] = ft

    for s in SPACES:
        Xp, Xt = u.Xp[s].to(DEV, DT), u.Xt[s].to(DEV, DT)
        XtX = Xp.T @ Xp
        block = BLOCKS.get(s)
        if q1_here:
            knn_rows.append(dict(space=s, **knn_stats(Xp, yp, grp_p, dlg_p, scen_p)))
        cw25, cb25 = tether[(s, 25)]
        Wo, bo = P7.p1_wb(z1, s, f"all|{cw25}|{cb25}")
        pred[f"{s}_oracle"] = logits_np(Xt, Wo, bo).argmax(1)
        for a in ALPHAS:
            keep(f"{s}_ldaO_a{a}", lda_from_R(Xp, onehot_all, XtX, a), Xt, save_delta=(a == FIX[0]), save_coef=(a == FIX[0]))
        if block is not None:
            keep(f"{s}_ldaOblk_a{FIX[0]}", lda_from_R(Xp, onehot_all, XtX, FIX[0], block), Xt, save_delta=True)
        fO = lda_from_R(Xp, onehot_all, XtX, FIX[0])
        diag[f"{s}_ldaO_a{FIX[0]}"] = dict(n_k=fO["n_k"].cpu().tolist(), tr_sinv_s=tr_sinv_s(fO, FIX[0], DIM[s]),
                                            pool_uar=_uar_pool(yp, delta(Xp, fO).argmax(1).cpu().numpy()))
        empty = torch.zeros(0, dtype=torch.long, device=DEV)
        yrand = np.random.default_rng(SEED).permutation(yp)                   # 同じ割合ででたらめに並べ替えたラベル（review_ml の M1）
        for tag, lab in (("truth", yp), ("rand", yrand)):
            q_ = torch.nn.functional.one_hot(torch.as_tensor(lab, device=DEV), 4).to(DT)
            res = em_fit(Xp, XtX, empty, empty, q_, FIX[0], 0.0, n_iter=MAX_EM, tol=TOL_EM, snaps=(N_EM,), yU=yp)
            f10 = res["snap"].get(N_EM, res["fit"])
            keep(f"{s}_em{tag}_a{FIX[0]}_b0.0", f10, Xt)
            keep(f"{s}_em{tag}conv_a{FIX[0]}_b0.0", res["fit"], Xt)
            prc = res["R"].argmax(1).cpu().numpy()
            poolpred[f"{s}_em{tag}conv_a{FIX[0]}_b0.0"] = prc
            p10 = delta(Xp, f10).argmax(1).cpu().numpy()
            diag[f"{s}_em{tag}_a{FIX[0]}_b0.0"] = dict(
                hist=res["hist"], pool_hist=res["pool_hist"], pi=res["fit"]["pi"].cpu().tolist(), **res["info"],
                n_k=res["fit"]["n_k"].cpu().tolist(), tr_sinv_s=tr_sinv_s(res["fit"], FIX[0], DIM[s]),
                pool_uar=_uar_pool(yp, prc), pool_uar_at10=_uar_pool(yp, p10),
                keep_init=float((prc == lab).mean()), keep_init_at10=float((p10 == lab).mean()),
                pool_rec=[float((prc[yp == e] == e).mean()) for e in range(4)], pool_uar_none=float("nan"),
                pool_rec_none=[float("nan")] * 4, agree_none_test=None)
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
            Wf, bf = P7.pick(z3, f"p3_{s}_K{K}", dict(method="tt", teacher="fused_AVT", rho=1.0, a=1))   # 同じ教師の識別型（gate-code の 8）
            pred[f"K{K}_{s}_ttfused"] = logits_np(Xt, Wf, bf).argmax(1)
            q = torch.softmax(logits_np(Xp, Wn, bn), -1)
            kpos = u.kpos[K]
            kpos_t = torch.as_tensor(kpos, dtype=torch.long, device=DEV)
            ylab = yp_t[kpos_t]
            Umask = np.ones(len(yp), bool)
            Umask[kpos] = False
            yU = yp[Umask]
            qa = q.argmax(1).cpu().numpy()[Umask]
            d0 = dict(pool_uar_none=_uar_pool(yU, qa), pool_rec_none=[float((qa[yU == e] == e).mean()) for e in range(4)])

            def record(key, res, q_=q):
                pr = res["R"][res["U"]].argmax(1).cpu().numpy()
                diag[key] = dict(hist=res["hist"], pool_hist=res["pool_hist"], pi=res["fit"]["pi"].cpu().tolist(), **res["info"],
                                 n_k=res["fit"]["n_k"].cpu().tolist(), tr_sinv_s=tr_sinv_s(res["fit"], FIX[0], DIM[s]) if "_a0.5_" in key else None,
                                 pool_uar=_uar_pool(yU, pr), pool_rec=[float((pr[yU == e] == e).mean()) for e in range(4)],
                                 agree_none_test=None, **d0)
                return pr

            if K >= 1:
                XL = Xp[kpos_t]
                RL = torch.nn.functional.one_hot(ylab, 4).to(DT)
                for a in ALPHAS:
                    fk = lda_from_R(XL, RL, XL.T @ XL, a)
                    keep(f"K{K}_{s}_ldaK_a{a}", fk, Xt, save_delta=(a == FIX[0]), save_coef=(a == FIX[0]))
                    if a == FIX[0]:
                        q0 = torch.softmax(delta(Xp, fk), -1)                 # EM_fromK の初期値（gate-probe の 2）
                res = em_fit(Xp, XtX, kpos_t, ylab, q0, FIX[0], 0.0, yU=yU)
                key = f"K{K}_{s}_emfromK_a{FIX[0]}_b0.0"
                keep(key, res["fit"], Xt, save_delta=True, save_coef=True)
                record(key, res, q0)
                cwT, cbT = FIX_TETHER                                        # 選ばない引き戻しの点から始める EM（gate-code の 7）
                WT, bT = P7.p1_wb(z1, s, f"{K}|{cwT}|{cbT}")
                pred[f"K{K}_{s}_noneT"] = logits_np(Xt, WT, bT).argmax(1)
                qT = torch.softmax(logits_np(Xp, WT, bT), -1)
                res = em_fit(Xp, XtX, kpos_t, ylab, qT, FIX[0], FIX[1], yU=yU)
                key = f"K{K}_{s}_emT_a{FIX[0]}_b{FIX[1]}"
                keep(key, res["fit"], Xt, save_delta=True)
                record(key, res, qT)
            for a in ALPHAS:
                for b in BETAS:
                    key = f"K{K}_{s}_em_a{a}_b{b}"
                    if (a, b) == FIX:                                        # 固定の点：収束まで回し，0・10・30・最後の回を残す
                        res = em_fit(Xp, XtX, kpos_t, ylab, q, a, b, n_iter=MAX_EM, tol=TOL_EM, snaps=SNAPS, yU=yU)
                        for t_, ft in res["snap"].items():
                            keep(f"K{K}_{s}_emit{t_}_a{a}_b{b}", ft, Xt)
                        keep(key, res["snap"][N_EM] if N_EM in res["snap"] else res["fit"], Xt, save_delta=True, save_coef=True)
                        keep(f"K{K}_{s}_emconv_a{a}_b{b}", res["fit"], Xt, save_delta=True, save_coef=True)
                        record(key, res)
                        ph = res["pool_hist"]
                        diag[key]["pool_uar_at10"] = ph[N_EM] if len(ph) > N_EM else ph[-1]   # 「流れた」は t = 10 で数える（review_ml の第 3 回の n4）
                        poolpred[f"K{K}_{s}_emconv_a{a}_b{b}"] = res["R"].argmax(1).cpu().numpy()   # 候補の全部（gate-probe の第 2 回の 2）
                        h = res["hist"]
                        diag[key]["hist_at_rule"] = h[N_EM - 1] if len(h) >= N_EM else (h[-1] if h else 0.0)
                    else:
                        res = em_fit(Xp, XtX, kpos_t, ylab, q, a, b, yU=yU)
                        keep(key, res["fit"], Xt)
                        record(key, res)
            res = em_fit(Xp, XtX, kpos_t, ylab, q, FIX[0], FIX[1], fix_sigma=True, yU=yU)   # EM_μ（gate-probe の 5）
            key = f"K{K}_{s}_emmu_a{FIX[0]}_b{FIX[1]}"
            keep(key, res["fit"], Xt, save_delta=True, save_coef=True)
            record(key, res)
            if block is not None:
                res = em_fit(Xp, XtX, kpos_t, ylab, q, FIX[0], FIX[1], block, n_iter=MAX_EM, tol=TOL_EM, snaps=(N_EM,), yU=yU)
                key = f"K{K}_{s}_emblk_a{FIX[0]}_b{FIX[1]}"
                keep(key, res["snap"].get(N_EM, res["fit"]), Xt, save_delta=True, save_coef=True)
                keep(f"K{K}_{s}_emblkconv_a{FIX[0]}_b{FIX[1]}", res["fit"], Xt)
                record(key, res)
            for key in [k for k in diag if k.startswith(f"K{K}_{s}_")]:      # 3.3 節：テストで none と判定が違う割合
                diag[key]["agree_none_test"] = float((pred[key] == pn).double().mean()) if key in pred else None
            poolpred[f"K{K}_{s}_none"] = q.argmax(1).cpu().numpy()
        del Xp, Xt, XtX
        torch.cuda.empty_cache()
    pool_prop = (np.bincount(yp, minlength=4) / len(yp)).tolist()
    (out / "rows").mkdir(parents=True, exist_ok=True)
    cf = {}
    for k, ft in coef.items():
        cf[f"C_A_{k}"] = ft["A"].cpu().numpy().astype(np.float16)
        cf[f"C_mu_{k}"] = ft["mu"].cpu().numpy().astype(np.float16)
        cf[f"C_c_{k}"] = ft["c"].cpu().numpy().astype(np.float32)
        cf[f"C_pi_{k}"] = ft["pi"].cpu().numpy().astype(np.float32)
    np.savez_compressed(dst, commit=commit, test_sample_id=sid_t, test_y=yt.astype(np.int8),
                        test_cluster=cl_of(sid_t).astype(str), test_group=grp_t, pool_prop=np.array(pool_prop),
                        pool_y=yp.astype(np.int8), pool_group=grp_p, pool_dialogue=dlg_p,
                        diag=json.dumps(diag), checks=json.dumps(checks), n_pool=len(yp),
                        **{k: v.cpu().numpy().astype(np.int8) for k, v in pred.items()},
                        **{f"D_{k}": v.cpu().numpy().astype(np.float32) for k, v in dlt.items()},
                        **{f"P_{k}": np.asarray(v).astype(np.int8) for k, v in poolpred.items()},
                        **cf)
    if knn_rows:
        kp = out / "rows" / f"{name}_knn.csv"
        if kp.exists():
            raise SystemExit(f"{kp} は既にある")
        pd.DataFrame(knn_rows).assign(T=T, c=c, fold=f, seed=r, commit=commit).to_csv(kp, index=False)
    del u
    torch.cuda.empty_cache()
    sec = time.time() - t0
    print(f"{name}：{sec:.0f} 秒（none の判定の照合の不一致 {checks['none_mismatch']} / {checks['none_compared']}）", flush=True)
    return dict(name=name, seconds=sec, **checks)


# ------------------------------------------------------------------ 集計

FX = f"a{FIX[0]}_b{FIX[1]}"
COMPARE = {   # 名前 → (A の方法, B の方法)．{s} は空間，{K} は K
    "em_none": (f"K{{K}}_{{s}}_em_{FX}", "K{K}_{s}_none"),
    "em_joint": (f"K{{K}}_{{s}}_em_{FX}", "K{K}_{s}_joint"),
    "em_ttfused": (f"K{{K}}_{{s}}_em_{FX}", "K{K}_{s}_ttfused"),
    "emfromK_ldaK": (f"K{{K}}_{{s}}_emfromK_a{FIX[0]}_b0.0", f"K{{K}}_{{s}}_ldaK_a{FIX[0]}"),
    "em_ldaK": (f"K{{K}}_{{s}}_em_{FX}", f"K{{K}}_{{s}}_ldaK_a{FIX[0]}"),
    "ldaK_none": (f"K{{K}}_{{s}}_ldaK_a{FIX[0]}", "K{K}_{s}_none"),
    "joint_none": ("K{K}_{s}_joint", "K{K}_{s}_none"),
    "ttfused_none": ("K{K}_{s}_ttfused", "K{K}_{s}_none"),
    "emT_noneT": (f"K{{K}}_{{s}}_emT_{FX}", "K{K}_{s}_noneT"),
    "em0_none": (f"K{{K}}_{{s}}_emit0_{FX}", "K{K}_{s}_none"),
    "em_em0": (f"K{{K}}_{{s}}_em_{FX}", f"K{{K}}_{{s}}_emit0_{FX}"),
    "emconv_none": (f"K{{K}}_{{s}}_emconv_{FX}", "K{K}_{s}_none"),
    "em_emmu": (f"K{{K}}_{{s}}_em_{FX}", f"K{{K}}_{{s}}_emmu_{FX}"),
    "emblk_em": (f"K{{K}}_{{s}}_emblk_{FX}", f"K{{K}}_{{s}}_em_{FX}"),
    **{f"em_a{a}_b{b}_none": (f"K{{K}}_{{s}}_em_a{a}_b{b}", "K{K}_{s}_none") for a in ALPHAS for b in BETAS if (a, b) != FIX},
}
COMPARE_K0 = {   # K に依らない比べ（K = 0 の行に書く）
    "ldaO_oracle": (f"{{s}}_ldaO_a{FIX[0]}", "{s}_oracle"),
    "ldaOblk_ldaO": (f"{{s}}_ldaOblk_a{FIX[0]}", f"{{s}}_ldaO_a{FIX[0]}"),
    "emtruth_ldaO": (f"{{s}}_emtruthconv_a{FIX[0]}_b0.0", f"{{s}}_ldaO_a{FIX[0]}"),   # R9 の段 1（review_ml の第 2 回の N1）
    "emrand_ldaO": (f"{{s}}_emrandconv_a{FIX[0]}_b0.0", f"{{s}}_ldaO_a{FIX[0]}"),
}
SUB_COMPARE = ("em_none", "em_joint", "emfromK_ldaK", "emT_noneT")                                       # 部分集合で並べる比べ（AVT．review_affect の A3・a7）


def _add(d, key, cnt, label):
    if key in d:
        n0, c0, t0 = d[key]
        if not (n0 == cnt[0]).all():
            raise SystemExit(f"{label}：塊の並びが同じ (f, r) のほかの T と違う")
        d[key] = (n0, c0 + cnt[1], t0 + cnt[2])
    else:
        d[key] = cnt


def summarize(out: Path, allow_partial: bool) -> None:
    files = sorted((out / "rows").glob("*.npz"))
    if len(files) != N_UNITS and not allow_partial:
        raise SystemExit(f"単位の出力が {len(files)} 個（{N_UNITS} のはず）．試走なら --allow-partial")
    rng = np.random.default_rng(SEED)
    cells, cellrows, diagrows, commits = {}, [], [], set()
    same_cf, cf_mism = {}, []                                                # 同じ (c, f) の単位の一致（gate-code の 5）
    for p in files:
        T, c, fs, rs = p.name[:-4].split("_")
        f, r = int(fs[1:]), int(rs[1:])
        z = np.load(p, allow_pickle=False)
        commits.add(str(z["commit"]))
        y, cl, sid = z["test_y"].astype(int), z["test_cluster"].astype(str), z["test_sample_id"].astype(str)
        subs = {nm: np.asarray(mk, bool) for nm, mk in dai3.subset_masks(c, sid, cl).items()}
        for k in z.files:
            if k.startswith(("K", "A_", "V_", "T_", "AVT_")) and z[k].dtype == np.int8 and z[k].shape == y.shape:
                prd = z[k].astype(int)
                cnt = ST.counts(y, prd, cl)
                _add(cells.setdefault((c, k), {}), (f, r), cnt, f"{p.name} {k}")
                cellrows.append(dict(c=c, T=T, fold=f, seed=r, key=k, uar=ST.uar_full(cnt)))
                if "_ldaO" in k and not k.startswith("K"):
                    ref = same_cf.setdefault((c, f, k), prd)
                    mism = int((ref != prd).sum())
                    cf_mism.append(mism)
                    if mism > max(1, int(0.001 * len(prd))):                  # 数値の丸めでの同点の入れ替わりまでは許す
                        raise SystemExit(f"{p.name} {k}：同じ (c, f) の単位で全ラベルの LDA の判定が {mism} 本違う")
                if k.startswith("K") and "_AVT_" in k:
                    for nm, mk in subs.items():
                        if mk.any():
                            _add(cells.setdefault((c, f"{k}|sub_{nm}"), {}), (f, r), ST.counts(y[mk], prd[mk], cl[mk]), f"{p.name} {k} {nm}")
        for k, d in json.loads(str(z["diag"])).items():
            h = d.get("hist") or []
            row = dict(c=c, T=T, fold=f, seed=r, key=k, n_pool=int(z["n_pool"]),
                       hist_first=h[0] if h else np.nan, hist_last=h[-1] if h else np.nan,
                       hist_at_rule=d.get("hist_at_rule", h[-1] if h else np.nan),
                       pi_err=float(np.abs(np.array(d["pi"]) - z["pool_prop"]).sum()) if "pi" in d else np.nan,
                       tr_sinv_s=d.get("tr_sinv_s"), n_k_min=min(d["n_k"]) if "n_k" in d else np.nan)
            for nm in ("pool_uar", "pool_uar_none", "pool_uar_at10", "n_iter_done", "ratio", "moved", "entropy", "beta_flip_last",
                       "agree_none_test", "keep_init", "keep_init_at10"):
                row[nm] = d.get(nm, np.nan)
            for e in range(4):
                row[f"pool_rec_e{e}"] = d["pool_rec"][e] if "pool_rec" in d else np.nan
                row[f"pool_rec_none_e{e}"] = d["pool_rec_none"][e] if "pool_rec_none" in d else np.nan
            diagrows.append(row)
    knn = pd.concat([pd.read_csv(p) for p in sorted((out / "rows").glob("*_knn.csv"))])
    if knn.groupby(["c", "fold", "space"]).size().max() != 1 or (not allow_partial and len(knn) != 20 * len(SPACES)):
        raise SystemExit(f"Q1 の行が (c, f, 空間) ごとに 1 行でない（{len(knn)} 行）")
    knn.to_csv(out / "knn_units.csv", index=False)
    # 流れた先：最後の割り当てと感情・グループ・対話の調整済み相互情報量（gate-probe の第 2 回の 2．候補の全部）
    from sklearn.metrics import adjusted_mutual_info_score as AMI
    amirows = []
    for p in files:
        T, c, fs, rs = p.name[:-4].split("_")
        z = np.load(p, allow_pickle=False)
        labs = {"emo": z["pool_y"].astype(int), "grp": z["pool_group"].astype(str), "dlg": z["pool_dialogue"].astype(str)}
        amirows.append(dict(c=c, T=T, fold=int(fs[1:]), seed=int(rs[1:]), key="labels_emo",   # ラベルどうしの AMI（review_ml の第 3 回の n6）
                            **{f"ami_{nm}": float(AMI(lab, labs["emo"], average_method="min")) for nm, lab in labs.items()}))
        for k in [x for x in z.files if x.startswith("P_") and "_AVT_" in x or x.startswith("P_AVT_")]:
            a_ = z[k].astype(int)
            amirows.append(dict(c=c, T=T, fold=int(fs[1:]), seed=int(rs[1:]), key=k[2:],
                                **{f"ami_{nm}": float(AMI(lab, a_, average_method="min")) for nm, lab in labs.items()}))   # min で正規化（gate-probe の第 3 回の 1）
    AM = pd.DataFrame(amirows)
    AM.to_csv(out / "ami_units.csv", index=False)
    U = pd.DataFrame(cellrows)
    U.groupby(["c", "key"]).uar.agg(["mean", "count"]).reset_index().to_csv(out / "cells.csv", index=False)
    DGu = pd.DataFrame(diagrows)
    DGu.to_csv(out / "diag_units.csv", index=False)
    DGu.groupby(["c", "key"]).mean(numeric_only=True).reset_index().to_csv(out / "diag.csv", index=False)
    rows = []
    for c in sorted({k[0] for k in cells}):
        g_eff = ST.g_eff_of(c)[0]
        for s in SPACES:
            for nm, (a, b) in COMPARE_K0.items():
                ka, kb = a.format(s=s), b.format(s=s)
                if (c, ka) in cells and (c, kb) in cells:
                    rows.append(dict(c=c, K=0, space=s, compare=nm, subset="all", **ST.effect_stats(cells[(c, ka)], cells[(c, kb)], g_eff, rng)))
            for K in KS_ALL:
                for nm, (a, b) in COMPARE.items():
                    ka, kb = a.format(K=K, s=s), b.format(K=K, s=s)
                    if (c, ka) in cells and (c, kb) in cells:
                        rows.append(dict(c=c, K=K, space=s, compare=nm, subset="all", **ST.effect_stats(cells[(c, ka)], cells[(c, kb)], g_eff, rng)))
                    if s == "AVT" and nm in SUB_COMPARE:
                        for sk in [k for k in cells if k[0] == c and k[1].startswith(f"{ka}|sub_")]:
                            sub = sk[1].split("|sub_")[1]
                            kb2 = (c, f"{kb}|sub_{sub}")
                            if kb2 in cells:
                                rows.append(dict(c=c, K=K, space=s, compare=nm, subset=sub, **ST.effect_stats(cells[sk], cells[kb2], g_eff, rng)))
        print(f"{c}：区間を作った", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(out / "diffs.csv", index=False)
    crit = []
    for c in sorted(D.c.unique()):
        r = dict(c=c)
        for tag, nm in (("R1", "em_none"), ("R2", "em_joint"), ("R3", "emfromK_ldaK"), ("R3x", "em_ldaK"), ("R6", "em_emmu"),
                        ("R7", "em_ttfused"), ("R8", "emT_noneT")):
            d = D[(D.c == c) & (D.space == "AVT") & (D.compare == nm) & (D.subset == "all") & D.K.isin(KS_RULE)]
            ok = (d.lo > 0) & (d.effect >= MIN_D)
            r[f"{tag}_n_K"], r[f"{tag}_n_lo_pos"], r[f"{tag}_n_hi_neg"] = int(ok.sum()), int((d.lo > 0).sum()), int((d.hi < 0).sum())
            r[tag] = bool(ok.sum() >= 3)
        r["R3_harm"] = bool(r["R3_n_hi_neg"] >= 3)
        for tag, nm in (("R1", "em_none"), ("R3", "emfromK_ldaK"), ("R8", "emT_noneT")):   # MOSEI の channel の確かめ（review_affect の b4）
            full = D[(D.c == c) & (D.space == "AVT") & (D.compare == nm) & (D.subset == "all") & D.K.isin(KS_RULE)].set_index("K")
            ch = D[(D.c == c) & (D.space == "AVT") & (D.compare == nm) & (D.subset == "channel") & D.K.isin(KS_RULE)].set_index("K")
            okK = [K for K in KS_RULE if K in full.index and full.loc[K, "lo"] > 0 and full.loc[K, "effect"] >= MIN_D]
            r[f"{tag}_chan_n_K_robust"] = (int(sum(1 for K in okK if K in ch.index and ch.loc[K, "lo"] > 0
                                                   and ch.loc[K, "effect"] >= 0.5 * full.loc[K, "effect"])) if c == "MOSEI" else np.nan)
        d4 = D[(D.c == c) & (D.space == "AVT") & (D.compare == "ldaO_oracle")]
        r["R4_effect"] = float(d4.effect.iloc[0]) if len(d4) else np.nan
        r["R4"] = bool(len(d4) and d4.effect.iloc[0] >= R4_TOL)
        kf = knn[(knn.c == c) & (knn.space == "AVT")].groupby("fold")[["emo_gain_xdlg", "grp_gain"]].mean()
        r["R5_n_fold_emo_lt_grp"], r["R5_n_fold"] = int((kf.emo_gain_xdlg < kf.grp_gain).sum()), int(len(kf))
        fx = DGu[(DGu.c == c) & DGu.key.isin([f"K{K}_AVT_em_{FX}" for K in KS_RULE])]
        r["moved_AVT_fix"] = float(fx.moved.mean())
        r["not_moved"] = bool(r["moved_AVT_fix"] < MOVED_MIN)
        r["n_not_conv_at_rule"] = int((DGu[(DGu.c == c) & DGu.key.str.endswith(f"_em_{FX}")].hist_at_rule >= CONV).sum())
        # review_ml の M1：流れた理由の読み分け（AVT，候補の上．テストのラベルは使わない）
        o = DGu[(DGu.c == c) & (DGu.key == f"AVT_ldaO_a{FIX[0]}")].pool_uar.mean()
        tru = DGu[(DGu.c == c) & (DGu.key == f"AVT_emtruth_a{FIX[0]}_b0.0")]
        rnd = DGu[(DGu.c == c) & (DGu.key == f"AVT_emrand_a{FIX[0]}_b0.0")]
        r["M1_pool_uar_ldaO"], r["M1_pool_uar_truth"], r["M1_pool_uar_rand"] = float(o), float(tru.pool_uar.mean()), float(rnd.pool_uar.mean())
        r["M1_keep_truth"], r["M1_keep_rand"] = float(tru.keep_init.mean()), float(rnd.keep_init.mean())
        dt9 = D[(D.c == c) & (D.space == "AVT") & (D.compare == "emtruth_ldaO")]
        r["M1_test_truth_minus_ldaO"] = float(dt9.effect.iloc[0]) if len(dt9) else np.nan
        if r["M1_test_truth_minus_ldaO"] <= R9_TEST_DROP:                      # review_ml の第 2 回の N1：段の順と基準
            r["M1_reading"] = "b_optimum_away_from_emotion"
        elif r["M1_keep_rand"] >= KEEP_RAND or (r["M1_keep_truth"] - r["M1_keep_rand"]) < KEEP_GAP:
            r["M1_reading"] = "c_keeps_any_init"
        else:
            r["M1_reading"] = "a_init_or_other"
        em_pool = DGu[(DGu.c == c) & DGu.key.isin([f"K{K}_AVT_em_{FX}" for K in KS_RULE])]
        r["M1_drifted"] = bool(((em_pool.pool_uar_at10 - em_pool.pool_uar_none) <= -DRIFT).mean() > 0.5) if len(em_pool) else False
        for s_ in SPACES:
            fs_ = DGu[(DGu.c == c) & (DGu.key.str.endswith(f"_{s_}_em_{FX}"))]
            r[f"ratio_{s_}"] = float(fs_.ratio.median())
            r[f"beta_moot_{s_}"] = bool(r[f"ratio_{s_}"] > RATIO_MAX)
            r[f"n_over_d_{s_}"] = float(fs_.n_pool.mean() / DIM[s_])
        am = AM[AM.c == c] if len(AM) else AM
        for nm_, keys in (("none", [f"K{K}_AVT_none" for K in KS_RULE]), ("em", [f"K{K}_AVT_emconv_{FX}" for K in KS_RULE]),
                          ("truth", [f"AVT_emtruthconv_a{FIX[0]}_b0.0"]), ("rand", [f"AVT_emrandconv_a{FIX[0]}_b0.0"])):
            sub = am[am.key.isin(keys)] if len(am) else am
            for lab in ("emo", "grp", "dlg"):
                r[f"R10_ami_{lab}_{nm_}"] = float(sub[f"ami_{lab}"].mean()) if len(sub) else np.nan
        r["R10_em_top"] = max(("emo", "grp", "dlg"), key=lambda lab: r[f"R10_ami_{lab}_em"]) if len(am) else ""
        kq = knn[(knn.c == c) & (knn.space == "AVT")]
        r["R5_top3ex_emo"], r["R5_top3ex_grp"], r["R5_top3ex_dlg"] = (float(kq[f"top3ex_{x}"].mean()) for x in ("emo", "grp", "dlg"))
        r["R5_top3_top"] = max(("emo", "grp", "dlg"), key=lambda x: r[f"R5_top3ex_{x}"])
        crit.append(r)
    Cr = pd.DataFrame(crit)
    for tag in ("R1", "R2", "R3", "R4", "R7", "R8", "not_moved"):
        Cr[f"{tag}_n_corpora"] = int(Cr[tag].sum())
        Cr[f"{tag}_n_corpora_excl_IEMOCAP"] = int(Cr.loc[Cr.c != "IEMOCAP", tag].sum())
    Cr.to_csv(out / "criteria.csv", index=False)
    commit, dirty = dai1.git_commit()
    pp = out / "provenance.json"
    prov = json.loads(pp.read_text()) if pp.exists() else {}
    shards = sorted(out.glob("provenance_shard*.json"))
    if shards:
        prov["shards"] = [json.loads(x.read_text()) for x in shards]
        cs = {x["commit"] for x in prov["shards"]}
        if len(cs) != 1 or any(x["dirty"] for x in prov["shards"]):
            raise SystemExit(f"組ごとの単位の段のコミットが違うか dirty：{cs}")
    prov.update(summarize_commit=commit, summarize_dirty=dirty, rows_commits=sorted(commits), boot_seed=SEED, n_boot=ST.N_BOOT,
                n_units_found=len(files), same_cf_lda_mismatch_max=int(max(cf_mism) if cf_mism else 0))
    pp.write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"書いた：cells.csv，diag.csv，diag_units.csv，diffs.csv（{len(D)} 行），knn_units.csv，criteria.csv", flush=True)


# ------------------------------------------------------------------ 自己検査

def selftest() -> None:
    g = torch.Generator().manual_seed(0)
    d, n = 6, 4000
    mu = torch.randn(4, d, generator=g, dtype=DT) * 2
    y = torch.randint(0, 4, (n,), generator=g)
    X = (mu[y] + torch.randn(n, d, generator=g, dtype=DT)).to(DEV)
    X = X - X.mean(0)
    y = y.to(DEV)
    R = torch.nn.functional.one_hot(y, 4).to(DT)
    f1 = lda_from_R(X, R, X.T @ X, 0.0)
    acc = float((delta(X, f1).argmax(1) == y).double().mean())
    mu_hat = torch.stack([X[y == k].mean(0) for k in range(4)])
    Sw = sum(((X[y == k] - mu_hat[k]).T @ (X[y == k] - mu_hat[k])) for k in range(4)) / n
    A2 = torch.linalg.solve(Sw, mu_hat.T)
    print(f"(1) 係数の一致：最大の差 {float((f1['A'] - A2).abs().max()):.2e}，正解率 {acc:.3f}")
    assert float((f1["A"] - A2).abs().max()) < 1e-8
    q = torch.full((n, 4), 0.25, dtype=DT, device=DEV)
    kpos = torch.arange(0, 40, device=DEV)
    res = em_fit(X, X.T @ X, kpos, y[kpos], q, 0.1, 0.0)
    acc3 = float((delta(X, res["fit"]).argmax(1) == y).double().mean())
    fixed = bool((res["R"][kpos].argmax(1) == y[kpos]).all())
    print(f"(2) 一様の初期値と K 本 40 からの EM の正解率 {acc3:.3f}（直接の LDA {acc:.3f}），K 本の行の固定 {fixed}")
    assert fixed and acc3 > acc - 0.05
    # (2b) 回の数え方：tol なしの n_iter = 10 の解と，収束まで回した版の回 10 の保存が一致する
    q2 = torch.softmax(torch.randn(n, 4, generator=g, dtype=DT).to(DEV), -1)
    r10 = em_fit(X, X.T @ X, kpos, y[kpos], q2, 0.5, 1.0)
    rc = em_fit(X, X.T @ X, kpos, y[kpos], q2, 0.5, 1.0, n_iter=MAX_EM, tol=TOL_EM, snaps=SNAPS)
    dif = float((r10["fit"]["A"] - rc["snap"][10]["A"]).abs().max())
    print(f"(2b) 回 10 の保存の一致：最大の差 {dif:.2e}，収束までの回 {rc['info']['n_iter_done']}")
    assert dif < 1e-10
    # (2c) 共分散を固定する版：与えた Cholesky で解いた係数が，同じ Σ で直接解いた係数と一致する
    fa = lda_from_R(X, R, X.T @ X, 0.3)
    fb = lda_from_R(X, R, X.T @ X, 0.3, chol=fa["chol"])
    print(f"(2c) 固定した Σ の解の一致：最大の差 {float((fa['A'] - fb['A']).abs().max()):.2e}")
    assert float((fa["A"] - fb["A"]).abs().max()) < 1e-12
    # (2d) Woodbury：Σ⁻¹M = A⁻¹ M C（A = (1 − α)T + αI，候補の平均 0）
    Rr = torch.softmax(torch.randn(n, 4, generator=g, dtype=DT).to(DEV), -1)
    ff = lda_from_R(X, Rr, X.T @ X, 0.5)
    Tm = X.T @ X / n
    Am = 0.5 * Tm + 0.5 * torch.eye(d, dtype=DT, device=DEV)
    AiM = torch.linalg.solve(Am, ff["mu"].T)
    Cm = torch.linalg.inv(torch.eye(4, dtype=DT, device=DEV) - 0.5 * torch.diag(ff["pi"]) @ ff["mu"] @ AiM)
    print(f"(2d) Woodbury の一致：最大の差 {float((ff['A'] - AiM @ Cm).abs().max()):.2e}")
    assert float((ff["A"] - AiM @ Cm).abs().max()) < 1e-10
    Xb = torch.cat([X, X[:, :2] + 0.1 * torch.randn(n, 2, generator=g, dtype=DT).to(DEV)], 1)
    f4 = lda_from_R(Xb, R, Xb.T @ Xb, 0.5, (("a", d), ("b", 2)))
    print(f"(3) ブロック対角の解は有限：{bool(torch.isfinite(f4['A']).all())}")
    ks = knn_stats(X, y.cpu().numpy(), np.zeros(n, int), np.arange(n) // 10)
    try:
        knn_stats(X, y.cpu().numpy(), np.zeros(n, int), np.zeros(n, int))
        raise AssertionError("対話の印が 1 種類でも止まらなかった")
    except SystemExit:
        print("(4b) 対話の印が 1 種類なら止まる：確かめた")
    print(f"(4) 近傍の感情の一致 {ks['emo']:.3f}，対話を外した版 {ks['emo_xdlg']:.3f}，グループ 1 つの一致 {ks['grp']:.3f}，"
          f"感情の散らばりの割合 {ks['emo_between']:.3f}，白色化した分かれやすさ {ks['fisher_emo']:.3f}")
    assert ks["grp"] == 1.0
    # (5) EM_truth と EM_rand の検算（review_ml の第 2 回の n3）：よく分かれた人工データでは，正解から始めた EM は正解をほぼ保ち，
    #     でたらめのラベルから始めた EM はそれを保たない
    empty = torch.zeros(0, dtype=torch.long, device=DEV)
    yy = y.cpu().numpy()
    keeps = {}
    for tag, lab in (("truth", yy), ("rand", np.random.default_rng(0).permutation(yy))):
        q_ = torch.nn.functional.one_hot(torch.as_tensor(lab, device=DEV), 4).to(DT)
        rr = em_fit(X, X.T @ X, empty, empty, q_, 0.5, 0.0, n_iter=MAX_EM, tol=TOL_EM)
        keeps[tag] = float((rr["R"].argmax(1).cpu().numpy() == lab).mean())
    print(f"(5) 正解から始めた EM が保った割合 {keeps['truth']:.3f}，でたらめから {keeps['rand']:.3f}")
    assert keeps["truth"] > 0.9 and keeps["rand"] < 0.6
    print("自己検査：通った")


# ------------------------------------------------------------------ 入口

def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "dai8_probe")
    ap.add_argument("--units", default=None, help="単位の名前をコンマで（試走）")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="集計の段で 120 単位に満たなくても進める（試走）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n：単位を n 組に分けた i 番目だけを流す（本計算を並べて流す）")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    out = a.out if a.out.is_absolute() else ROOT / a.out
    if a.summarize:
        summarize(out, a.allow_partial)
        return
    commit, dirty = dai1.git_commit()
    fz = json.loads(FROZEN.read_text())
    p3s = sorted(UD.glob("*_p3.npz"))
    partial = bool(a.units)
    shard = None
    if partial:
        want = set(a.units.split(","))
        p3s = [p for p in p3s if P7.unit_name(p) in want]
    else:
        if dirty:
            raise SystemExit("作業木に未コミットの変更がある（本計算はコミットしてから）")
        if len(p3s) != N_UNITS:
            raise SystemExit(f"入力の単位が {len(p3s)} 個（{N_UNITS} のはず）")
        if a.shard:
            i, n = (int(x) for x in a.shard.split("/"))
            shard = (i, n)
            if (out / f"provenance_shard{i}.json").exists():
                raise SystemExit(f"組 {i} の provenance が既にある")
            p3s = p3s[i::n]
        elif out.exists():
            raise SystemExit(f"{out} は既にある（本計算は空の出力先に書く）")
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(commit=commit, dirty=dirty, partial=partial, started=time.strftime("%F %T"), python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                cuda=torch.version.cuda, device=str(DEV), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                script_sha256=sha(Path(__file__)), frozen_sha256=sha(FROZEN),
                alphas=ALPHAS, betas=BETAS, fix=FIX, n_em=N_EM, max_em=MAX_EM, tol_em=TOL_EM, snaps=SNAPS, fix_tether=FIX_TETHER,
                knn=KNN, min_d=MIN_D, r4_tol=R4_TOL, moved_min=MOVED_MIN, ratio_max=RATIO_MAX, conv=CONV, keep_rand=KEEP_RAND,
                keep_gap=KEEP_GAP, r9_test_drop=R9_TEST_DROP, drift=DRIFT, n_perm=N_PERM,
                units=[P7.unit_name(p) for p in p3s],
                inputs_sha256={P7.unit_name(p): {i: sha(UD / f"{P7.unit_name(p)}_p{i}.npz") for i in (1, 3)} for p in p3s})
    pname = (f"provenance_partial_{time.strftime('%Y%m%d-%H%M%S')}.json" if partial
             else f"provenance_shard{shard[0]}.json" if shard else "provenance.json")
    prov["shard"] = list(shard) if shard else None
    (out / pname).write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"開始 {prov['started']} コミット {commit[:7]} 単位 {len(p3s)} 個 機器 {prov['gpu']}", flush=True)
    runs = [run_unit(p, out, fz, commit) for p in p3s]
    prov["runs"] = runs
    prov["finished"] = time.strftime("%F %T")
    (out / pname).write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"{len(runs)} 単位を終えた", flush=True)


if __name__ == "__main__":
    main()
