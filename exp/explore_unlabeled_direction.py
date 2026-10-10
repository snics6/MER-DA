"""【探索的】E28 の 10 節：ラベル無しのターゲットの構造に向きの情報はどれだけあるか（V1〜V4）。新しい学習なし（CPU）。

E13 の own_d と同じ半分割（話者を 2 つに分け，片方で向きを決め，もう片方で測り，入れ替えて平均）で，
ラベルを使わずに決めた向き（1 歩，EM，錨つき EM）の離れ方を，ソースの向き d_s とラベルつきの天井 D と比べる。
特徴はソースの統計で標準化。初期値はソースのみの分類器のターゲットでの事後確率。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_unlabeled_direction.py [--kinds …]

出力：`OUT/e28t10_pairs.csv`（組 × 感情の対），`OUT/e28t10_tests.csv`，`OUT/e28t10_summary.csv`
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_synergy import PAIRS  # noqa: E402
from pair_table import OUT  # noqa: E402
from split import speaker_folds  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
N_ITER, W_FLOOR, N_SEMI, K_SEMI = 20, 0.02, 3, 10   # 計算量のため，登録時の 30 回・5 回から減らした（流す前に決めた）
N_FIT_MAX = 3000                                     # 向きを決める側の半分は最大 3,000 本をでたらめに選ぶ（計算量のため．測る側は全発話）


def m_step(X: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = R.sum(0) / len(X)
    w = np.maximum(w, W_FLOOR); w /= w.sum()
    mu = (R.T @ X) / R.sum(0)[:, None]
    resid = X - R @ mu                                        # 責任で重みづけた平均からの残差
    S = LedoitWolf(assume_centered=True).fit(resid).covariance_
    return w, mu, S


def e_step(X: np.ndarray, w: np.ndarray, mu: np.ndarray, S: np.ndarray, prior: np.ndarray | None) -> np.ndarray:
    L = np.linalg.cholesky(S)
    logp = np.empty((len(X), 4))
    for k in range(4):
        z = np.linalg.solve(L, (X - mu[k]).T)
        logp[:, k] = -0.5 * (z ** 2).sum(0) + np.log(w[k])
    if prior is not None:
        logp += np.log(np.clip(prior, 1e-12, None))
    return softmax(logp, axis=1)


def directions(mu: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.linalg.solve(S, np.stack([mu[k] - mu[j] for k, j in PAIRS]).T).T


def sep(U: np.ndarray, Xb: np.ndarray, yb: np.ndarray) -> np.ndarray:
    out = []
    for u, (k, j) in zip(U, PAIRS):
        pk, pj = Xb[yb == k] @ u, Xb[yb == j] @ u
        out.append((pk.mean() - pj.mean()) / np.sqrt((pk.var(ddof=1) + pj.var(ddof=1)) / 2))
    return np.array(out)


def sep_logits(Lb: np.ndarray, yb: np.ndarray) -> np.ndarray:
    out = []
    for k, j in PAIRS:
        zk, zj = Lb[yb == k, k] - Lb[yb == k, j], Lb[yb == j, k] - Lb[yb == j, j]
        out.append((zk.mean() - zj.mean()) / np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2))
    return np.array(out)


def fit_half(Xa: np.ndarray, Pa: np.ndarray, ya: np.ndarray, rng: np.random.Generator) -> dict:
    """半分 A（ラベル無し）で向きを決める。返り値は向きの行列（6, d）の辞書。"""
    out = {}
    w, mu, S = m_step(Xa, Pa)
    out["m1"] = directions(mu, S)
    for name, prior in (("plain", None), ("anch", Pa)):
        R = Pa.copy()
        for _ in range(N_ITER):
            w, mu, S = m_step(Xa, R)
            R = e_step(Xa, w, mu, S, prior)
        w, mu, S = m_step(Xa, R)
        out[name] = directions(mu, S)
    m = np.stack([Xa[ya == k].mean(0) for k in range(4)])
    out["sup"] = directions(m, LedoitWolf(assume_centered=True).fit(Xa - m[ya]).covariance_)
    semi = []
    for r in range(N_SEMI):
        idx = np.concatenate([rng.choice(np.flatnonzero(ya == k), K_SEMI, replace=False) for k in range(4)])
        onehot = np.eye(4)[ya[idx]]
        R = Pa.copy(); R[idx] = onehot
        for _ in range(N_ITER):
            w, mu, S = m_step(Xa, R)
            R = e_step(Xa, w, mu, S, Pa); R[idx] = onehot
        w, mu, S = m_step(Xa, R)
        semi.append(directions(mu, S))
    out["semi10"] = semi
    return out


def run_kind(kind: str) -> pd.DataFrame:
    groups = groups_for(kind)
    data = {g: load_any(g, kind) for g in groups}
    rows = []
    for s in groups:
        dfs, Xs = data[s]
        mu_s, sd_s = Xs.mean(0), Xs.std(0)
        sd_s = np.where(sd_s > 1e-12, sd_s, 1.0)
        z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
        for t in groups:
            if t == s or GROUP_SPEC[t][0] == GROUP_SPEC[s][0]:
                continue
            t0 = time.perf_counter()
            dft, Xt = data[t]
            y = dft.label_4.to_numpy()
            X = ((Xt - mu_s) / sd_s).astype(np.float64)
            Lt = z[f"t_{t}"].astype(np.float64)
            P = softmax(Lt, axis=1)
            half = speaker_folds(dft.speaker_group.astype(str).reset_index(drop=True), k=2, seed=0).to_numpy()
            rng = np.random.default_rng(0)
            acc = {k: [] for k in ("sup", "s", "m1", "plain", "anch", "semi10")}
            for a, b in ((0, 1), (1, 0)):
                A, B = half == a, half == b
                if min(np.bincount(y[A], minlength=4).min(), np.bincount(y[B], minlength=4).min()) < K_SEMI + 1:
                    continue
                ia = np.flatnonzero(A)
                if len(ia) > N_FIT_MAX:
                    ia = np.sort(rng.choice(ia, N_FIT_MAX, replace=False))
                U = fit_half(X[ia], P[ia], y[ia], rng)
                acc["sup"].append(sep(U["sup"], X[B], y[B]))
                acc["s"].append(sep_logits(Lt[B], y[B]))
                for name in ("m1", "plain", "anch"):
                    acc[name].append(sep(U[name], X[B], y[B]))
                acc["semi10"].append(np.mean([sep(Us, X[B], y[B]) for Us in U["semi10"]], axis=0))
            if not acc["sup"]:
                continue
            for i, (k, j) in enumerate(PAIRS):
                rows.append({"kind": kind, "target": t, "source": s, "pair": f"{k}-{j}",
                             **{f"d_{name}": float(np.mean([v[i] for v in vals])) for name, vals in acc.items()}})
            print(f"  {kind} {t} ← {s}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    args = ap.parse_args()
    R = pd.concat([run_kind(k) for k in args.kinds], ignore_index=True)
    R.to_csv(OUT / "e28t10_pairs.csv", index=False)
    q = pd.read_csv(OUT / "e28t9_targets.csv")[["kind", "target", "q_dir"]]
    ref = pd.read_csv(OUT / "e28t9_pairs.csv", dtype={"K": str})
    ref = ref[ref.K == "10"][["kind", "target", "source", "d_star_dir", "d_h_default", "d_h_best"]]
    A = R.groupby(["kind", "target", "source"])[[c for c in R.columns if c.startswith("d_")]].mean().reset_index()
    A = A.merge(q, on=["kind", "target"], how="left").merge(ref, on=["kind", "target", "source"], how="left")
    for name in ("m1", "plain", "anch", "semi10"):
        A[f"f_{name}"] = (A[f"d_{name}"] - A.d_s) / (A.d_sup - A.d_s)
    A.to_csv(OUT / "e28t10_summary.csv", index=False)
    tests = []
    for kind in args.kinds:
        g = A[A.kind == kind]
        frac_up = float((g.d_anch > g.d_s).mean()); med = float(g.f_anch.median())
        tests.append({"kind": kind, "test": "V1", "frac_anch_gt_s": frac_up, "median_f_anch": med, "holds": frac_up >= 0.6 and med >= 0.1, "n": len(g)})
        medp = float(g.f_plain.median()); frac_dn = float((g.d_plain < g.d_s).mean())
        tests.append({"kind": kind, "test": "V2", "median_f_plain": medp, "median_f_anch": med, "frac_plain_lt_s": frac_dn,
                      "holds": medp < med and frac_dn >= 0.3, "n": len(g)})
        rho, p = stats.spearmanr(g.f_anch, np.log(g.q_dir))
        tests.append({"kind": kind, "test": "V3", "rho_f_anch_logq": float(rho), "p": float(p), "holds": rho <= -0.3, "n": len(g)})
        tests.append({"kind": kind, "test": "V4", "frac_anch_ge_dstar10": float((g.d_anch >= g.d_star_dir).mean()),
                      "frac_semi10_ge_ht10": float((g.d_semi10 >= g.d_h_default).mean()),
                      "mean_d_s": g.d_s.mean(), "mean_d_m1": g.d_m1.mean(), "mean_d_plain": g.d_plain.mean(), "mean_d_anch": g.d_anch.mean(),
                      "mean_d_semi10": g.d_semi10.mean(), "mean_d_sup": g.d_sup.mean(), "mean_d_star10": g.d_star_dir.mean(),
                      "mean_d_ht10": g.d_h_default.mean(), "median_f_m1": float(g.f_m1.median()), "median_f_semi10": float(g.f_semi10.median())})
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "e28t10_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
