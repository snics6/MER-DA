"""【探索的】E74 の下調べ（GPU）：定理の手順 1 段．標準の理論が置く仮定を，仮のターゲットのデータで測る．仮のターゲットだけ。

問い（2026-10-28．研究者の承認．`docs/exploratory-analyses.md` E74．仮定の表と基準は計算の前に書いた．手順は `docs/methods.md` 10 節）
  追う現象は 0 段の一覧の U9・U10 と，先行研究の未解決の問い 1（Ha & Chen）・6（Medvedev ほか）である．
  A：E70 の命題 5（2 次の模型）の仮定．B：Ha & Chen（arXiv 2507.14661v2）の仮定．C：Medvedev ほか（arXiv 2503.02877v3）の定理 4.3 の前提と結論
  生徒・教師モデル・分割・仮のターゲット・解き方は E69（`exp/explore_curvature_split.py`）と同じ．評価に使うターゲットのテストは使わない

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/explore_theory_assumptions.py --world fuse-audio-video [--target …] [--pseudo-reps …] [--out-dir …]
    ./.venv/bin/python exp/explore_theory_assumptions.py --analyze [--results-dir …]
出力：`results/theory_assump_<世界>/<T>.csv`（仮のターゲットの反復と生徒ごとに 1 行），`provenance.json`，`provenance_log.jsonl`．
      判定は `results/explore_relation/e74_check.csv`・`e74_units.csv`・`e74_groups.csv`・`e74_tests.csv`・`e74_within.csv`・`e74_caps.csv`

記号（感情 k = 4，生徒の座標 z（d 次），θ = (W, b) を感情ごとに (W_a, b_a) と並べた 4(d+1) 次．感情の平均は除く）
  θ*：候補の全部のラベルで 0 への正則化で解いた頭（E71 の W*．`fit_star`）．θ* の揺れは話者の半分ずつの解の差の半分
  R(θ)：候補の上の釣り合いの重みつきの交差エントロピーの平均．F = ∇²R(θ*)，g = ∇R(θ*)
  e0 = a0 − θ*，e_Q = θ_Q − θ*（θ_Q は pF・m=2000 で動かしたアンカー．E66 の K=0）
  H，λ_j，u_j：θ_s（a0 から K 本で学び直した解）での K 本の 2 階微分と，H v = λ P v の固有値と固有方向（E69 の `Curv`）
  命題 5：学び直しの後の誤差 r(e) = (I + H̃)⁻¹ e，得 G = ½ (r(e0)ᵀ F r(e0) − r(e_Q)ᵀ F r(e_Q))．K=0 では H = 0
  方向の部分空間 S（特徴の方向 V_r の r 本と切片）で直せる割合 cap(e, S) = eᵀF B (BᵀFB)⁺ BᵀF e / eᵀFe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_curvature_split as CS  # noqa: E402
import explore_move_then_learn as MTL  # noqa: E402
import explore_span_split as SS  # noqa: E402
import explore_view_fusion as VF  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
import multimodal_distill as MD  # noqa: E402
from data import groups_for  # noqa: E402
from distill_belief import solve_all, uar_of  # noqa: E402
from geom_e0 import settings  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import present_weights  # noqa: E402

WORLDS = MTL.WORLDS
BELIEF, MEFF = SS.BELIEF, 2000
KS_L = MTL.KS_L                                     # 分割は E69 と同じ（K = 1, 5, 10, 25 の組を作る）
KS_A = ("5", "25")                                  # 学び直しを測る K
PSEUDO_REPS = list(MTL.PSEUDO_REPS)
OUT_REL = ROOT / "results" / "explore_relation"
KEY = SS.KEY
R_GRID = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)     # 0 は切片だけ
N_RAND = 3
BASES = ("shift", "pca") + tuple(f"rand{i}" for i in range(N_RAND))
ERRS = ("e0", "eQ", "noise")


# ---------------------------------------------------------------- 量

def theta(W, b):
    """(W, b) を感情ごとに (W_a, b_a) と並べ，感情の平均を除く（4 × (d+1)）。"""
    t = torch.cat([W, b[:, None]], 1)
    return t - t.mean(0, keepdim=True)


def wb(t):
    d = t.shape[1] - 1
    return t[:, :d], t[:, d]


class Fisher:
    """候補の上の釣り合いの重みつき交差エントロピーの平均 R の，θ* での 2 階微分 F と勾配 g。"""

    def __init__(self, Z, yq, Wst, bst):
        self.Z, self.y = Z, M.labels(yq)
        n, d = Z.shape
        self.w = logreg.balanced_weights(self.y) / n
        Xt = torch.cat([Z, torch.ones(n, 1, dtype=Z.dtype, device=Z.device)], 1)
        pi = torch.softmax(Z @ Wst.T + bst, 1)
        F = torch.zeros(4, d + 1, 4, d + 1, dtype=Z.dtype, device=Z.device)
        for a in range(4):
            for c in range(a, 4):
                coef = self.w * ((pi[:, a] if a == c else 0.0) - pi[:, a] * pi[:, c])
                blk = Xt.T @ (coef[:, None] * Xt)
                F[a, :, c, :] = blk
                if c != a:
                    F[c, :, a, :] = blk.T
        self.D = 4 * (d + 1)
        self.F = F.reshape(self.D, self.D)
        Y = torch.nn.functional.one_hot(self.y, 4).to(Z.dtype)
        self.g = ((self.w[:, None] * (pi - Y)).T @ Xt).reshape(-1)
        self.R_star = self.loss(Wst, bst)

    def loss(self, W, b):
        lp = torch.log_softmax(self.Z @ W.T + b, 1)
        return float(-(self.w * lp.gather(1, self.y[:, None])[:, 0]).sum())

    def quad(self, t):
        v = t.reshape(-1)
        return float(v @ (self.F @ v))

    def lin(self, t):
        return float(self.g @ t.reshape(-1))


def unit_dirs(cv):
    """E69 の Curv の固有方向 u_j を元の座標へ戻したもの（J × 4 × (d+1)。P で正規直交）。"""
    J = cv.lam.shape[0]
    v = (cv.V / cv.lam.sqrt()[None, :]).view(cv.n, cv.k, J)
    q = torch.einsum("iab,ibj->iaj", cv.Mi, v)
    dW = cv.cp * torch.einsum("iaj,ip->jap", q, cv.Zk)
    db = cv.cb * q.sum(0).T
    return torch.cat([dW, db[:, :, None]], 2)


def refit_err(cv, t):
    """命題 5 の (I + H̃)⁻¹ e（元の座標）と，固有方向の係数 c，λ = 0 の部分 n。"""
    dW, db = wb(t)
    c = cv.coef(dW, db)
    rW, rb = cv.back(c * cv.lam / (1 + cv.lam))
    pW, pb = cv.back(c)
    return theta(dW - rW, db - rb), c, theta(dW - pW, db - pb)


def gains(fis, cv, f_dir, e0, eQ):
    """命題 5 の得の 2 次の項（F の全体と，F を u_j の座標で対角とみなしたもの），尺度，1 次の項 gᵀr(e0)・gᵀr(e_Q)。"""
    r0, c0, n0 = refit_err(cv, e0)
    rQ, cQ, nQ = refit_err(cv, eQ)
    q0, qQ = fis.quad(r0), fis.quad(rQ)
    full = 0.5 * (q0 - qQ)
    diag = 0.5 * (float((f_dir * (c0 ** 2 - cQ ** 2) / (1 + cv.lam) ** 2).sum()) + fis.quad(n0) - fis.quad(nQ))
    return full, diag, 0.5 * (q0 + qQ), fis.lin(r0), fis.lin(rQ), 0.5 * q0, 0.5 * qQ


def shift_basis(Zt, Zs_list):
    """候補と各ソース（それぞれの平均を除く）の共分散の差の方向（|γ| の大きい順）。(Σ_T − Σ_S) v = γ ((Σ_T + Σ_S)/2) v。"""
    St = torch.cov(Zt.T, correction=0)
    n_s = sum(z.shape[0] for z in Zs_list)
    Ss = sum((z - z.mean(0, keepdim=True)).T @ (z - z.mean(0, keepdim=True)) for z in Zs_list) / n_s
    Mm = (St + Ss) / 2
    Mm = Mm + 1e-6 * torch.trace(Mm) / Mm.shape[0] * torch.eye(Mm.shape[0], dtype=Mm.dtype, device=Mm.device)
    L = torch.linalg.cholesky(Mm)
    Li = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], dtype=L.dtype, device=L.device), upper=False)
    g, Y = torch.linalg.eigh(Li @ (St - Ss) @ Li.T)
    V = Li.T @ Y
    order = torch.argsort(g.abs(), descending=True)
    return V[:, order], g[order]


def pca_basis(Zt):
    s, V = torch.linalg.eigh(torch.cov(Zt.T, correction=0))
    return V[:, torch.argsort(s, descending=True)]


def caps(fis, V, errs):
    """方向の部分空間 S_r（V の先頭 r 本と切片）で直せる割合 cap(e, S_r)．R_GRID ごと。"""
    d = V.shape[0]
    rmax = min(R_GRID[-1], d)
    B = torch.zeros(4, d + 1, 4, rmax + 1, dtype=V.dtype, device=V.device)
    for a in range(4):
        B[a, :d, a, :rmax] = V[:, :rmax]
        B[a, d, a, rmax] = 1.0
    B = B.reshape(fis.D, 4 * (rmax + 1))
    FB = fis.F @ B
    Mx = B.T @ FB
    vs = {k: FB.T @ t.reshape(-1) for k, t in errs.items()}
    tot = {k: fis.quad(t) for k, t in errs.items()}
    j = torch.arange(rmax + 1, device=V.device).repeat(4)
    out = {}
    for r in R_GRID:
        if r > rmax:
            continue
        idx = torch.nonzero((j < r) | (j == rmax))[:, 0]
        Pinv = torch.linalg.pinv(Mx[idx][:, idx], hermitian=True, rtol=1e-10)
        for k, v in vs.items():
            vr = v[idx]
            out[(k, r)] = float(vr @ (Pinv @ vr)) / tot[k] if tot[k] > 0 else np.nan
    return out


def r90(cap, k):
    for r in R_GRID:
        x = cap.get((k, r), np.nan)
        if np.isfinite(x) and x >= 0.9:
            return r
    return np.nan


def l2_err(fis, P, Y):
    """釣り合いの重みつきの ‖p − y‖² の平均（‖y‖² = 1）。"""
    return float((fis.w * ((P - Y) ** 2).sum(1)).sum())


def alpha(fis, P, T):
    """縮める倍率の最適値 argmin_α E‖α p − t‖²（1 以上なら p は t について縮めても良くならない）。"""
    return float((fis.w * (P * T).sum(1)).sum() / (fis.w * (P * P).sum(1)).sum())


def w2s_bound(L):
    """Medvedev ほかの定理 4.3 の下限（L ≤ 1 のときだけ）。"""
    return (np.sqrt(1 + 3 * L) - np.sqrt(1 - L)) ** 2 / 4 if 0 <= L <= 1 else np.nan


class Origin:
    """疑似ラベルの学習で，罰則の中心を原点にする（C1 の対照）。"""

    def __init__(self, d, dt, dev):
        self.W, self.b = torch.zeros(4, d, dtype=dt, device=dev), torch.zeros(4, dtype=dt, device=dev)

    def anchor(self, rho_m):
        return SS.fitted(self.W, self.b)


# ---------------------------------------------------------------- 計算（GPU）

def source_z(st, rho_m, data_m, m, pool_c):
    Am = st.mats[rho_m][0]
    out = []
    for s in pool_c:
        X = data_m[m][s].X
        out.append(((X - X.mean(0, keepdim=True) - st.src.mu) / st.src.sd) @ Am)
    return out


def solve_assump(world, stu, yq, yte, label_sets, speaker, seed, data_m, pool_c, rows, tag):
    b1 = MD.beliefs1(stu, world)
    for m in MD.WORLDS[world]:
        s = stu[m]
        st, rho_m, cp, cb, Z, Zte = s["st"], s["rho_m"], s["cp"], s["cb"], s["Z"], s["Zte"]
        d = Z.shape[1]
        a0 = st.anchor(rho_m)
        W0, b0 = a0.W[MD.C_FIT], a0.b[MD.C_FIT]
        row = {**tag, "student": m, "d": d, "n_cand": Z.shape[0]}
        Wq, bq = solve_all(Z, yq, {BELIEF: b1[BELIEF]}, {"0": label_sets["0"]}, [(BELIEF, MEFF)], st, rho_m,
                           MD.C_FIT, cp, cb)[(BELIEF, MEFF, "0")]
        Wo, bo = solve_all(Z, yq, {BELIEF: b1[BELIEF]}, {"0": label_sets["0"]}, [(BELIEF, MEFF)],
                           Origin(d, Z.dtype, Z.device), rho_m, MD.C_FIT, cp, cb)[(BELIEF, MEFF, "0")]
        (Wst, bst), halves = VF.star(Z, yq, speaker, seed)
        fis = Fisher(Z, yq, Wst, bst)
        ts = theta(Wst, bst)
        e0, eQ = theta(W0, b0) - ts, theta(Wq, bq) - ts
        noise = (theta(*halves[0]) - theta(*halves[1])) / 2
        row.update(uar_a0=uar_of(W0, b0, Zte, yte), uar_Q=uar_of(Wq, bq, Zte, yte), uar_O=uar_of(Wo, bo, Zte, yte),
                   uar_star=uar_of(Wst, bst, Zte, yte), R_star=fis.R_star)
        # A1：2 次の近似と実際の損失の増え方
        for name, (W, b) in (("a0", (W0, b0)), ("Q", (Wq, bq))):
            t = theta(W, b) - ts
            row.update({f"act_{name}": fis.loss(W, b) - fis.R_star, f"quad_{name}": 0.5 * fis.quad(t), f"lin_{name}": fis.lin(t)})
        row["G_0"] = row["G_diag_0"] = 0.5 * (fis.quad(e0) - fis.quad(eQ))
        row["G_scale_0"] = 0.5 * (fis.quad(e0) + fis.quad(eQ))
        row.update(lin_r0_0=row["lin_a0"], lin_rQ_0=row["lin_Q"], quad_r0_0=row["quad_a0"], quad_rQ_0=row["quad_Q"])
        for K in KS_A:
            idx = label_sets[K]
            Ws, bs_ = SS.refit_one(a0, Z, yq, idx, cp, cb)
            WB, bB = SS.refit_one(SS.fitted(Wq, bq), Z, yq, idx, cp, cb)
            row.update({f"uar_s_{K}": uar_of(Ws, bs_, Zte, yte), f"uar_B_{K}": uar_of(WB, bB, Zte, yte)})
            for name, (W, b) in (("s", (Ws, bs_)), ("B", (WB, bB))):
                t = theta(W, b) - ts
                row.update({f"act_{name}_{K}": fis.loss(W, b) - fis.R_star, f"quad_{name}_{K}": 0.5 * fis.quad(t),
                            f"lin_{name}_{K}": fis.lin(t)})
            yk = M.labels(yq[idx])
            wk = present_weights(yk).to(Z.device, Z.dtype)
            cv = CS.Curv(Ws, bs_, Z[torch.as_tensor(idx, device=Z.device)], wk, cp, cb)
            U = unit_dirs(cv).reshape(cv.lam.shape[0], -1)
            f_dir = (U * (fis.F @ U.T).T).sum(1)
            full, diag, scale, l0, lQ, h0, hQ = gains(fis, cv, f_dir, e0, eQ)
            row.update({f"G_{K}": full, f"G_diag_{K}": diag, f"G_scale_{K}": scale, f"lin_r0_{K}": l0, f"lin_rQ_{K}": lQ,
                        f"quad_r0_{K}": h0, f"quad_rQ_{K}": hQ})
            # A4：λ_j と n_w f_j（n_w は K 本の重みの和）
            lam, f_np = cv.lam.cpu().numpy(), f_dir.cpu().numpy()
            nw = float(wk.sum())
            hi = lam > 1
            row.update({f"n_dir_{K}": len(lam), f"n_hi_{K}": int(hi.sum()), f"nw_{K}": nw,
                        f"rho_hi_{K}": float(np.median(lam[hi] / (nw * f_np[hi]))) if hi.any() else np.nan,
                        f"rho_all_{K}": float(np.median(lam / (nw * f_np))),
                        f"sp_lam_f_{K}": float(spearmanr(lam, f_np)[0]) if len(lam) > 2 else np.nan})
        # B：ラベル無しで決まる方向の部分空間で直せる割合
        errs = {"e0": e0, "eQ": eQ, "noise": noise}
        for k, t in errs.items():
            row[f"F_{k}"] = fis.quad(t)
        Zs = source_z(st, rho_m, data_m, m, pool_c)
        Vs, gam = shift_basis(Z, Zs)
        bases = {"shift": Vs, "pca": pca_basis(Z)}
        gen = torch.Generator(device="cpu").manual_seed(seed * 7 + d)
        for i in range(N_RAND):
            Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=Z.dtype).to(Z.device))
            bases[f"rand{i}"] = Q
        row["gamma_top4"] = float(gam[:4].abs().mean())
        for bn, V in bases.items():
            cp_ = caps(fis, V, errs)
            for (k, r), x in cp_.items():
                row[f"cap_{bn}_{k}_{r}"] = x
            for k in errs:
                row[f"r90_{bn}_{k}"] = r90(cp_, k)
        # C1：教師モデル，アンカーに寄せた生徒，原点に寄せた生徒の ‖p − f‖²．目標 f は θ* の予測確率 p*（主）と one-hot（記述）
        Y = torch.nn.functional.one_hot(fis.y, 4).to(Z.dtype)
        Ps = torch.softmax(Z @ Wst.T + bst, 1)
        pT = torch.as_tensor(b1[BELIEF], dtype=Z.dtype, device=Z.device)
        pQ, pO = torch.softmax(Z @ Wq.T + bq, 1), torch.softmax(Z @ Wo.T + bo, 1)
        for tg, Ft in (("s", Ps), ("y", Y)):
            LT = l2_err(fis, pT, Ft)
            row.update({f"L_T_{tg}": LT, f"L_Q_{tg}": l2_err(fis, pQ, Ft), f"L_O_{tg}": l2_err(fis, pO, Ft),
                        f"L_bound_{tg}": w2s_bound(LT), f"alpha_T_{tg}": alpha(fis, pT, Ft), f"norm_{tg}": l2_err(fis, Ft, 0 * Ft)})
        row.update(alpha_Q=alpha(fis, pQ, pT), alpha_O=alpha(fis, pO, pT), L_QT=l2_err(fis, pQ, pT), L_OT=l2_err(fis, pO, pT))
        rows.append(row)
        del fis, bases
        torch.cuda.empty_cache()


def run_target(world, T, data_f, data_m, out, cache):
    """E69 の run_target と同じ生徒・分割で解く。"""
    S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
    sets = {k: settings(k, T) for k in MD.WORLDS[world] + [world]}
    rows = []
    for c in S:
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c:
            continue
        info_c = {k: M5.pool_info(data_m[k] if k != world else data_f, pool_c, MD.C_FIT, cache[k]) for k in sets}
        evc = data_f[c]
        ctxc = CB.Ctx(evc, PSEUDO_REPS, list(KS_L), center="ind")
        for r in PSEUDO_REPS:
            pool = np.flatnonzero(~ctxc.te[r])
            te = np.flatnonzero(ctxc.te[r])
            pos_of = {int(i): j for j, i in enumerate(pool)}
            label_sets = {"0": np.array([], int), **{K: np.array([pos_of[int(i)] for i in ctxc.idx[(r, K)]]) for K in KS_L}}
            stu = MD.make_students(world, c, data_f, data_m, pool_c, pool, te, sets, info_c)
            solve_assump(world, stu, evc.y[pool], evc.y[te], label_sets, evc.speaker.iloc[pool], 1000 + r, data_m, pool_c,
                         rows, {"pseudo": c, "rep": r})
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def _meta(results):
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"script": "exp/explore_theory_assumptions.py", "commit": git("rev-parse", "--short", "HEAD"), "results": str(results)}


def write(df, name, meta):
    with open(OUT_REL / name, "w") as f:
        f.write("# " + json.dumps(meta, ensure_ascii=False) + "\n")
        df.to_csv(f, index=False)


GROUP = ["student", "pseudo"]                       # 独立な単位：仮のターゲットのコーパス（S7．同じ c は候補・分割・K 本・θ*・F を共有する）
NOISE_MAX = 0.5                                     # θ* の揺れ F_noise/F_eQ がこれを超える反復は B1・B3 から外す


def capx(X, b, k, r):
    """切片で直せる分を引いた割合 (cap_r − cap_0)/(1 − cap_0)（D10）。"""
    c0 = X[f"cap_{b}_{k}_0"]
    return (X[f"cap_{b}_{k}_{r}"] - c0) / (1 - c0)


def r90x(X, b, k):
    """切片を引いた割合が 0.9 に届く最小の r（1 以上．届かなければ d）。"""
    out = pd.Series(np.nan, index=X.index)
    for r in reversed([r for r in R_GRID if r > 0]):
        c = f"cap_{b}_{k}_{r}"
        if c in X:
            out = out.where(~(capx(X, b, k, r) >= 0.9), r)
    return out.fillna(X.d)


def per_rep(X):
    """反復ごとの量（判定に使うもの）。"""
    Y = X[KEY + ["d"]].copy()
    for n in ("a0", "Q"):
        Y[f"A0_{n}"] = X[f"lin_{n}"].abs() / X[f"quad_{n}"]
        Y[f"A1_{n}"] = (X[f"lin_{n}"] + X[f"quad_{n}"]) / X[f"act_{n}"]
        Y[f"A1q_{n}"] = X[f"quad_{n}"] / X[f"act_{n}"]
    for K in KS_A:
        for n in ("r0", "rQ"):
            Y[f"A0_{n}_{K}"] = X[f"lin_{n}_{K}"].abs() / X[f"quad_{n}_{K}"]
        for n in ("s", "B"):
            Y[f"A1_{n}_{K}"] = (X[f"lin_{n}_{K}"] + X[f"quad_{n}_{K}"]) / X[f"act_{n}_{K}"]
            Y[f"A1q_{n}_{K}"] = X[f"quad_{n}_{K}"] / X[f"act_{n}_{K}"]
    for K in ("0",) + KS_A:
        lin = X[f"lin_r0_{K}"] - X[f"lin_rQ_{K}"]
        Y[f"predG_{K}"] = X[f"G_{K}"] + lin                      # 2 次までのテイラー展開の得
        Y[f"predGd_{K}"] = X[f"G_diag_{K}"] + lin
        Y[f"A3_{K}"] = (X[f"G_diag_{K}"] - X[f"G_{K}"]).abs() / X[f"G_scale_{K}"]
    Y["uarG_0"] = X.uar_Q - X.uar_a0
    for K in KS_A:
        Y[f"uarG_{K}"] = X[f"uar_B_{K}"] - X[f"uar_s_{K}"]
        Y[f"A4_rho_{K}"] = X[f"rho_hi_{K}"]
        Y[f"A4_sp_{K}"] = X[f"sp_lam_f_{K}"]
    Y["noise_rel"] = X.F_noise / X.F_eQ
    ok = Y.noise_rel <= NOISE_MAX
    for k in ERRS:
        Y[f"r90_{k}"] = r90x(X, "shift", k).where(ok | (k == "noise"))
    for k in ERRS:
        Y[f"cap0_{k}"] = X[f"cap_shift_{k}_0"]
        for r in (4, 16, 64):
            Y[f"capx_shift_{k}_{r}"] = capx(X, "shift", k, r)
            Y[f"capx_pca_{k}_{r}"] = capx(X, "pca", k, r)
            Y[f"capx_rand_{k}_{r}"] = pd.concat([capx(X, f"rand{i}", k, r) for i in range(N_RAND)], axis=1).mean(1)
    for tg in ("s", "y"):
        for c in ("L_T", "L_Q", "L_O", "L_bound", "alpha_T"):
            Y[f"{c}_{tg}"] = X[f"{c}_{tg}"]
        Y[f"Q_below_{tg}"] = (X[f"L_Q_{tg}"] < X[f"L_bound_{tg}"]).astype(float).where(X[f"L_bound_{tg}"].notna())
        Y[f"O_below_{tg}"] = (X[f"L_O_{tg}"] < X[f"L_bound_{tg}"]).astype(float).where(X[f"L_bound_{tg}"].notna())
    for c in ("alpha_Q", "alpha_O", "uar_a0", "uar_Q", "uar_O"):
        Y[c] = X[c]
    return Y


def within(g, x, y):
    """群の中の順位相関（反復の行を点にする）。"""
    h = g[[x, y]].dropna()
    return spearmanr(h[x], h[y])[0] if len(h) > 2 else np.nan


def analyze(results=None) -> int:
    results = results or ROOT / "results"
    meta = _meta(results)
    X = SS.load(results, "theory_assump")
    # 確かめ：θ_Q と，start・full（m=2000）の UAR が E69 の同じ問題と一致するか
    E = SS.load(results, "curv_split")
    chk = []
    e = E[(E.variant == "full") & (E.K.astype(str) == KS_A[0]) & (E.meff == MEFF)].set_index(KEY).uar_anchor
    j = X.set_index(KEY).uar_Q.to_frame("x").join(e.rename("y"), how="inner")
    chk.append(dict(what="θ_Q K=0", n=len(j), n_x=len(X), max_abs_diff=float((j.x - j.y).abs().max()) if len(j) else np.nan))
    for K in KS_A:
        for name, col in (("start", f"uar_s_{K}"), ("full", f"uar_B_{K}")):
            e = E[(E.variant == name) & (E.K.astype(str) == K) & ((E.meff == MEFF) | (name == "start"))].set_index(KEY).uar
            j = X.set_index(KEY)[col].to_frame("x").join(e.rename("y"), how="inner")
            chk.append(dict(what=f"{name} K={K}", n=len(j), n_x=len(X), max_abs_diff=float((j.x - j.y).abs().max()) if len(j) else np.nan))
    chk = pd.DataFrame(chk)
    write(chk, "e74_check.csv", meta)
    print("== 確かめ（E69 との差）\n" + chk.to_string(index=False))
    Y = per_rep(X)
    cols = [c for c in Y.columns if c not in KEY + ["d"]]
    U = Y.groupby(["world", "target", "pseudo", "student"]).agg(
        {c: ("median" if c.startswith("r90") else "mean") for c in cols} | {"d": "first"}).reset_index()
    Gm = U.groupby(GROUP).agg({c: ("median" if c.startswith("r90") else "mean") for c in cols} | {"d": "first"}).reset_index()
    Gm["n_units"] = U.groupby(GROUP).size().values
    write(U, "e74_units.csv", meta)
    write(Gm, "e74_groups.csv", meta)
    tests, per_group = [], []
    for stu_, g in Gm.groupby("student"):
        n = len(g)
        Ys = Y[Y.student == stu_]
        a0 = {c: g[c].median() for c in ("A0_a0", "A0_Q", "A0_r0_5", "A0_rQ_5", "A0_r0_25", "A0_rQ_25")}
        tests.append(dict(rule="A0", student=stu_, n=n, detail=" ".join(f"{c[3:]}:{v:.3f}" for c, v in a0.items()),
                          ok=bool(all(v <= 0.25 for v in a0.values()))))
        a1 = {c: g[c].median() for c in ("A1_a0", "A1_Q", "A1_B_5", "A1_B_25")}
        tests.append(dict(rule="A1", student=stu_, n=n, detail=" ".join(f"{c[3:]}:{v:.2f}" for c, v in a1.items()),
                          ok=bool(all(0.75 <= v <= 1.25 for v in a1.values()))))
        a2 = {}
        for K in ("0",) + KS_A:
            rs = {c: within(h, f"predG_{K}", f"uarG_{K}") for c, h in Ys.groupby("pseudo")}
            a2[K] = np.nanmedian(list(rs.values()))
            per_group += [dict(rule="A2", student=stu_, K=K, pseudo=c, rho=v) for c, v in rs.items()]
        tests.append(dict(rule="A2", student=stu_, n=n, detail=" ".join(f"K={K}:{v:.2f}" for K, v in a2.items()),
                          ok=bool(all(v >= 0.5 for v in a2.values()))))
        a3 = {}
        for K in KS_A:
            rs = {c: within(h, f"predGd_{K}", f"predG_{K}") for c, h in Ys.groupby("pseudo")}
            per_group += [dict(rule="A3", student=stu_, K=K, pseudo=c, rho=v) for c, v in rs.items()]
            sign = float((np.sign(Ys[f"predGd_{K}"]) == np.sign(Ys[f"predG_{K}"])).mean())
            a3[K] = (g[f"A3_{K}"].median(), sign, np.nanmedian(list(rs.values())))
        tests.append(dict(rule="A3", student=stu_, n=n,
                          detail=" ".join(f"K={K}:差{a:.3f},符号{b:.2f},順位相関{c:.2f}" for K, (a, b, c) in a3.items()),
                          ok=bool(all(a <= 0.25 and b >= 0.9 and c >= 0.8 for a, b, c in a3.values()))))
        a4 = {K: (g[f"A4_rho_{K}"].median(), g[f"A4_sp_{K}"].median()) for K in KS_A}
        tests.append(dict(rule="A4", student=stu_, n=n, detail=" ".join(f"K={K}:比{a:.2f},順位相関{b:.2f}" for K, (a, b) in a4.items()),
                          ok=bool(all(0.5 <= a <= 2 and b >= 0.5 for a, b in a4.values()))))
        b1 = (g.r90_eQ / g.d).median()
        tests.append(dict(rule="B1", student=stu_, n=n, detail=f"r90(e_Q)/d:{b1:.3f}（r90 の中央値 {g.r90_eQ.median():.0f}．"
                          f"揺れで外した反復 {int((Ys.noise_rel > NOISE_MAX).sum())}/{len(Ys)}）", ok=bool(b1 <= 0.05)))
        cs, cr = g.capx_shift_eQ_16.median(), g.capx_rand_eQ_16.median()
        tests.append(dict(rule="B2", student=stu_, n=n, detail=f"r=16 shift:{cs:.3f} rand:{cr:.3f} 比:{cs / cr:.2f}（切片だけ {g.cap0_eQ.median():.3f}）",
                          ok=bool(cs >= 0.5 and cs >= 3 * cr)))
        b3 = float(((g.r90_eQ / g.r90_e0) <= 0.5).mean())
        tests.append(dict(rule="B3", student=stu_, n=n, detail=f"r90(e_Q)/r90(e0) ≤ 0.5 の群の割合:{b3:.2f}", ok=bool(b3 >= 2 / 3)))
        tests.append(dict(rule="C1", student=stu_, n=n, ok=None,
                          detail=f"目標 p*：L_T {g.L_T_s.median():.3f} L_Q {g.L_Q_s.median():.3f} L_O {g.L_O_s.median():.3f} "
                                 f"下限 {g.L_bound_s.median():.3f}．下限より下の割合：Q {g.Q_below_s.median():.2f} O {g.O_below_s.median():.2f}．"
                                 f"α_T≥1 の群：{(g.alpha_T_s >= 1).mean():.2f}"))
    tests = pd.DataFrame(tests)
    write(tests, "e74_tests.csv", meta)
    write(pd.DataFrame(per_group), "e74_within.csv", meta)
    print("\n== 判定（生徒の種類ごと．独立な単位は仮のターゲットのコーパス．中央値は群の平均の中央値）")
    print(tests.to_string(index=False))
    q = lambda s: f"{s.median():.3f} [{s.min():.3f}, {s.max():.3f}]"  # noqa: E731
    rows = []
    for stu_, g in Gm.groupby("student"):
        for c in [c for c in Gm.columns if c.startswith(("cap", "r90_", "A0_", "A1", "A3_", "A4_", "L_", "alpha_", "noise"))]:
            rows.append(dict(student=stu_, what=c, summary=q(g[c])))
    S = pd.DataFrame(rows)
    write(S, "e74_caps.csv", meta)
    print("\n== 記述（群の平均の中央値 [最小，最大]）")
    print(S.pivot(index="what", columns="student", values="summary").to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=WORLDS)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--pseudo-reps", nargs="+", type=int, help="試走用．既定は 0〜4")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--results-dir", type=Path, help="--analyze が theory_assump_* と curv_split_* を読む場所（既定は results/）")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.results_dir)
    global PSEUDO_REPS
    if args.pseudo_reps is not None:
        PSEUDO_REPS = args.pseudo_reps
    world = args.world
    out = args.out_dir or ROOT / "results" / f"theory_assump_{world}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(world)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{world}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data_f = {g: M.load(g, world) for g in groups}
    data_m = {m: {g: MD.sub_group(data_f[g], a, b) for g in groups} for m, a, b in MD.blocks(world)}
    prov = {"script": "exp/explore_theory_assumptions.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "exp", "measure")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "targets": todo, "map_dir": str(map_dir(world)), "world": world, "mods": MD.WORLDS[world], "pseudo_reps": PSEUDO_REPS,
            "c_fit": MD.C_FIT, "belief": BELIEF, "meff": MEFF, "ks": KS_A, "r_grid": R_GRID, "n_rand": N_RAND,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache = {k: {} for k in MD.WORLDS[world] + [world]}
    for t in todo:
        t0 = time.perf_counter()
        run_target(world, t, data_f, data_m, out, cache)
        print(f"{world} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
