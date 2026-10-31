"""【理論の検算】少数のターゲットの例が，合併の分類器の位置のずれをどれだけ直すか（E15）。

導出（`docs/exploratory-analyses.md` E15）：ソースのみの分類器のターゲットでの位置のずれを o_s とすると，
K 本の合併の位置のずれは o_m ≈ (1 − λ_K) o_s で，

    λ_K = ρ_K / (1 + ρ_K)
    ρ_K = N_t κ_t · [ 1/(N_s κ_s)  +  δᵀ (A_s + I/C)⁻¹ δ ]
          └ 切片を動かす道 ┘  └ 重みで「ターゲットらしさ」を使う道 ┘

- N_s, N_t：学習データでのソース・ターゲットの重みの和（クラスを均等にする重み s_i の和）
- κ：損失の曲がり（2 値なら p(1−p)，多クラスなら (1 − Σ_k p_k²)/(k−1)）の平均．ソースのみの分類器で測る
- A_s = Σ_{i∈ソース} s_i κ_i (z_i − z̄)(z_i − z̄)ᵀ（κ で重みをつけたソースの散らばり），z̄ は κ で重みをつけた平均
- δ = ターゲットの平均 − z̄（標準化した特徴の上．ターゲットはラベルを使わない全発話）
- C：学習器の罰則（目的関数は Σ s_i CE + ‖W‖²/(2C)．`exp/logreg.py` と同じ）

人工データ（白色化した空間で作り，散らばりの形 Σ = L Lᵀ をかける）：
- ソース：クラスの中心が離れ方 Δ で並ぶ．ターゲット：同じ散らばりで，クラスの中心を角度 θ だけ回し
  （向きの効率 e ≈ cos θ），全体を δ = L(γ_o u_o + γ_x u_x) だけずらす
- γ_o はソースのみの分類器の位置のずれを作る成分（白色化した空間でソースの境目の向きに沿う）．
  γ_x は境目と無関係な向き u_x のずれで，位置のずれは作らずにターゲットの見分けやすさだけを上げる．
  u_x は散らばりが最大の向き（high）か最小の向き（low）に置く（罰則 I/C の効き方が変わる）

学習器は sklearn の LogisticRegression（class_weight="balanced"，L2，切片あり．学習データで標準化）。
2 値（導出がそのまま当てはまる）と 4 クラス（実データと同じ．感情の対ごとに E13 と同じ d・o を測る）。

    ./.venv/bin/python exp/sim_position.py            # 2 値と 4 クラスの両方（CPU 8 並列で数分）

出力：`results/sim_position/{binary,multi}.csv`・`summary_*.csv`・`fig_lambda.png`
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "sim_position"

P = 128                      # 次元
DELTA = 2.5                  # クラスの中心どうしの離れ方（白色化した空間．2 値で正解率 0.89）
N_SRC_PER_CLASS = 1500       # 2 値のソース（4 クラスは 750 ずつ．どちらも 3,000 本）
N_TEST_PER_CLASS = 2000
K_MAX = 50
O_MIN = 0.3                  # λ を測るのは |o_s| がこれ以上の組だけ（割り算が不安定になるため）

GRID = {
    "binary": dict(classes=2, thetas=[0, 45, 70], gamma_o=[0.0, 0.75, 1.5], gamma_x=[0, 3, 10],
                   Ks=[1, 2, 5, 10, 25, 50], Cs=[0.01, 1.0], reps=10),
    "multi": dict(classes=4, thetas=[0, 45], gamma_o=[1.0], gamma_x=[0, 3, 10],
                  Ks=[1, 5, 10, 25], Cs=[0.01, 1.0], reps=5),
}


# ---------------------------------------------------------------- 人工データ

def world(seed: int, classes: int) -> dict:
    """散らばりの形（固有値が 1/j で減る．平均 1）と，クラス・ずれの向きを作る。"""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((P, P)))
    sig = 1.0 / np.sqrt(np.arange(1, P + 1))
    sig = sig / np.sqrt((sig ** 2).mean())
    L = Q * sig                                   # x = L ξ，ξ ~ N(0, I)
    # 白色化した空間での向き：クラスの部分空間 U，回した先 U2，見分けやすさの向き（high = 固有値最大）
    m = classes - 1
    R, _ = np.linalg.qr(rng.standard_normal((P, 2 * m)))
    U, U2 = R[:, :m], R[:, m:]
    def orth(v):
        v = v - R @ (R.T @ v)
        return v / np.linalg.norm(v)
    ux = {"high": orth(np.eye(P)[0]), "low": orth(np.eye(P)[P - 1])}
    # クラスの中心（白色化した空間）：2 値は ±Δ/2，4 クラスは正四面体（どの 2 つも Δ）
    if classes == 2:
        M0 = np.array([[-0.5], [0.5]]) * DELTA                      # (2, 1)
    else:
        E = np.eye(classes) - 1.0 / classes                         # 中心をそろえた頂点（4 次元，3 次元の部分空間）
        B = np.linalg.svd(E)[2][:m]                                 # その部分空間の基底 (3, 4)
        M0 = (E @ B.T) * DELTA / np.sqrt(2)                         # (4, 3)
    # 位置のずれを作る向き：2 値はソースの境目の向き，4 クラスはクラス 0 の向き（答えがクラス 0 に偏る）
    uo = U @ (M0[-1] - M0[0]) if classes == 2 else U @ M0[0]
    uo = uo / np.linalg.norm(uo)
    return dict(L=L, U=U, U2=U2, M0=M0, uo=uo, ux=ux)


def means(w: dict, theta: float, rotate: bool) -> np.ndarray:
    """クラスの中心（元の空間）。rotate なら U を角度 θ だけ U2 の側へ回す。"""
    th = np.deg2rad(theta) if rotate else 0.0
    Ub = np.cos(th) * w["U"] + np.sin(th) * w["U2"]
    return (Ub @ w["M0"].T).T @ w["L"].T                            # (classes, P)


def draw(rng, w: dict, n_per_class: int, mu: np.ndarray):
    k = len(mu)
    xi = rng.standard_normal((k * n_per_class, P))
    y = np.repeat(np.arange(k), n_per_class)
    return xi @ w["L"].T + mu[y], y


# ---------------------------------------------------------------- 学習と測定

def fit(X, y, C):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, tol=1e-6)
    clf.fit((X - mu) / sd, y)
    return clf, mu, sd


def scores(model, X):
    clf, mu, sd = model
    return clf.decision_function((X - mu) / sd)


def uar(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def d_o(z, y, k, l):
    """E13 と同じ：z = 点数の差（k の向きが正）．d は離れ方，o は位置のずれ。"""
    a, b = z[y == k], z[y == l]
    s = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / s, -(a.mean() + b.mean()) / (2 * s)


def pair_d_o(F, y, classes):
    """感情の対ごとの (k, l, d, o)。2 値は F が 1 列（クラス 1 の向きが正）。"""
    if classes == 2:
        return [(1, 0, *d_o(F, y, 1, 0))]
    return [(k, l, *d_o(F[:, k] - F[:, l], y, k, l)) for k, l in itertools.combinations(range(classes), 2)]


def predict_lambda(src_model, Xs, ys, Xt_pool, K, C, classes):
    """式の λ_K。ソースのみの分類器と，ソース・ラベルの無いターゲットだけから計算する。"""
    clf, mu, sd = src_model
    Zs, Zt = (Xs - mu) / sd, (Xt_pool - mu) / sd
    def curv(Z):
        if classes == 2:
            p = clf.predict_proba(Z)[:, 1]
            return p * (1 - p)
        p = clf.predict_proba(Z)
        return (1 - (p ** 2).sum(1)) / (classes - 1)
    ks, kt = curv(Zs), curv(Zt)
    # クラスを均等にする重み（合併の学習データ：ソースのクラス c に n_c 本，ターゲットに K 本ずつ）
    n_c = np.bincount(ys, minlength=classes) + K
    n = n_c.sum()
    s_src = n / (classes * n_c[ys])
    s_tgt = n / (classes * n_c)                                     # ターゲットのクラス c の 1 本の重み
    N_t = float((s_tgt * K).sum())
    wk = s_src * ks
    a = float(wk.sum())                                             # N_s κ_s
    zbar = (wk[:, None] * Zs).sum(0) / a
    D = Zs - zbar
    A = (D * wk[:, None]).T @ D
    delta = Zt.mean(0) - zbar
    w_route = float(delta @ np.linalg.solve(A + np.eye(P) / C, delta))
    rho = N_t * float(kt.mean()) * (1.0 / a + w_route)
    D2 = float(delta @ np.linalg.solve(A / a, delta))               # 罰則なしの見分けやすさ（マハラノビス距離の二乗）
    return dict(lam_pred=rho / (1 + rho), rho=rho, route_b=1.0 / a, route_w=w_route,
                N_t=N_t, N_s_kappa=a, kappa_s=float(ks.mean()), kappa_t=float(kt.mean()), D2=D2)


def acc_model(d, o):
    return 0.5 * (stats.norm.cdf(d / 2 + o) + stats.norm.cdf(d / 2 - o))


def run_job(mode: str, rep: int, C: float) -> list[dict]:
    g = GRID[mode]
    k = g["classes"]
    w = world(1000 + rep, k)
    rng = np.random.default_rng(10_000 * rep + int(C * 1000))
    mu_s = means(w, 0, rotate=False)
    Xs, ys = draw(rng, w, N_SRC_PER_CLASS * 2 // k, mu_s)
    src = fit(Xs, ys, C)
    # ターゲットの雑音は設定をまたいで共通にする（設定どうしを対にして比べられる）
    xi_tr = rng.standard_normal((k * K_MAX, P))
    y_tr = np.tile(np.arange(k), K_MAX)                             # 先頭から K·k 本で各クラス K 本
    xi_te = rng.standard_normal((k * N_TEST_PER_CLASS, P))
    y_te = np.repeat(np.arange(k), N_TEST_PER_CLASS)
    rows = []
    tgt_cache = {}
    configs = [(th, go, 0, "-") for th in g["thetas"] for go in g["gamma_o"]]
    configs += [(th, go, gx, wh) for th in g["thetas"] for go in g["gamma_o"]
                for gx in g["gamma_x"] if gx > 0 for wh in ("high", "low")]
    for th, go, gx, wh in configs:
        mu_t = means(w, th, rotate=True)
        shift = w["L"] @ (go * w["uo"] + (gx * w["ux"][wh] if gx > 0 else 0))
        Xtr = xi_tr @ w["L"].T + mu_t[y_tr] + shift
        Xte = xi_te @ w["L"].T + mu_t[y_te] + shift
        Fs = scores(src, Xte)
        s_pairs = pair_d_o(Fs, y_te, k)
        acc_s = uar(y_te, (Fs > 0).astype(int) if k == 2 else Fs.argmax(1))
        for K in g["Ks"]:
            sel = np.arange(k * K)
            # ターゲットのみ（標準化で全体のずれは消えるので，θ と K だけで決まる）
            key = (th, K)
            if key not in tgt_cache:
                tm = fit(Xtr[sel] - shift, y_tr[sel], C)
                Ft = scores(tm, Xte - shift)
                tgt_cache[key] = uar(y_te, (Ft > 0).astype(int) if k == 2 else Ft.argmax(1))
            acc_t = tgt_cache[key]
            mm = fit(np.vstack([Xs, Xtr[sel]]), np.r_[ys, y_tr[sel]], C)
            Fm = scores(mm, Xte)
            acc_m = uar(y_te, (Fm > 0).astype(int) if k == 2 else Fm.argmax(1))
            m_pairs = pair_d_o(Fm, y_te, k)
            pl = predict_lambda(src, Xs, ys, Xte, K, C, k)
            base = dict(mode=mode, rep=rep, C=C, theta=th, gamma_o=go, gamma_x=gx, where=wh, K=K,
                        acc_s=acc_s, acc_t=acc_t, acc_m=acc_m, **pl)
            for (kk, ll, ds, os_), (_, _, dm, om) in zip(s_pairs, m_pairs):
                lam = 1 - om / os_ if abs(os_) >= O_MIN else np.nan
                rows.append({**base, "k": kk, "l": ll, "d_s": ds, "o_s": os_, "d_m": dm, "o_m": om,
                             "lam_obs": lam,
                             "acc_pair_s": acc_model(ds, os_), "acc_pair_m": acc_model(dm, om),
                             "acc_pair_m_pred": acc_model(ds, (1 - pl["lam_pred"]) * os_)})
    return rows


# ---------------------------------------------------------------- まとめ

def summarize(d: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = []
    v = d.dropna(subset=["lam_obs"])
    for C, g in [("all", v)] + list(v.groupby("C")):
        r_p = stats.pearsonr(g.lam_pred, g.lam_obs)
        r_s = stats.spearmanr(g.lam_pred, g.lam_obs)
        out.append({"mode": mode, "C": C, "n": len(g), "pearson": r_p[0], "spearman": r_s[0],
                    "mean_lam_obs": g.lam_obs.mean(), "mean_lam_pred": g.lam_pred.mean(),
                    "mae": (g.lam_obs - g.lam_pred).abs().mean(),
                    "r_pair_acc": stats.pearsonr(g.acc_pair_m_pred, g.acc_pair_m)[0]})
    return pd.DataFrame(out)


def table_lambda(d: pd.DataFrame) -> pd.DataFrame:
    v = d.dropna(subset=["lam_obs"])
    v = v[v.theta == 0]
    return (v.groupby(["mode", "C", "gamma_x", "where", "K"])
             [["lam_obs", "lam_pred", "D2", "route_b", "route_w"]].mean().reset_index())


def table_synergy(d: pd.DataFrame) -> pd.DataFrame:
    """上乗せ = 合併 − max(ソースのみ, ターゲットのみ)（UAR）。組ごとに 1 行にまとめてから平均する。"""
    e = d.drop_duplicates(["mode", "rep", "C", "theta", "gamma_o", "gamma_x", "where", "K"]).copy()
    e["synergy"] = e.acc_m - e[["acc_s", "acc_t"]].max(axis=1)
    e["m_minus_s"] = e.acc_m - e.acc_s
    e["m_minus_t"] = e.acc_m - e.acc_t
    return (e.groupby(["mode", "C", "theta", "gamma_o", "gamma_x", "where", "K"])
             [["acc_s", "acc_t", "acc_m", "m_minus_s", "m_minus_t", "synergy", "lam_pred"]]
             .mean().reset_index())


def fig(d: pd.DataFrame, path: Path) -> None:
    fig_, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (mode, g) in zip(axes, d.dropna(subset=["lam_obs"]).groupby("mode")):
        for (C, wh), h in g.groupby(["C", "where"]):
            ax.scatter(h.lam_pred, h.lam_obs, s=6, alpha=0.4, label=f"C={C}, u_x={wh}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("predicted lambda"); ax.set_ylabel("observed lambda = 1 - o_m/o_s")
        ax.set_title(mode); ax.set_ylim(-0.5, 1.5); ax.legend(fontsize=7)
    fig_.tight_layout(); fig_.savefig(path, dpi=120); plt.close(fig_)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=["binary", "multi"], choices=list(GRID))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip())
    frames = []
    for mode in args.modes:
        g = GRID[mode]
        t0 = time.time()
        res = Parallel(n_jobs=args.jobs)(delayed(run_job)(mode, r, C)
                                         for r in range(g["reps"]) for C in g["Cs"])
        d = pd.DataFrame([row for rows in res for row in rows])
        header = json.dumps({"script": "exp/sim_position.py", "mode": mode, "grid": g, "P": P,
                             "DELTA": DELTA, "commit": commit + ("+dirty" if dirty else ""),
                             "seconds": round(time.time() - t0, 1)}, ensure_ascii=False)
        with open(OUT / f"{mode}.csv", "w") as f:
            f.write(f"# {header}\n")
            d.to_csv(f, index=False)
        print(f"{mode}: {len(d)} 行，{time.time() - t0:.0f} 秒", flush=True)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    s = pd.concat([summarize(g, m) for m, g in d.groupby("mode")], ignore_index=True)
    s.to_csv(OUT / "summary_fit.csv", index=False)
    table_lambda(d).to_csv(OUT / "summary_lambda.csv", index=False)
    table_synergy(d).to_csv(OUT / "summary_synergy.csv", index=False)
    fig(d, OUT / "fig_lambda.png")
    print(s.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
