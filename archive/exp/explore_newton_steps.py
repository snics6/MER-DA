"""【探索的】E28 の 3 節：合併の向きの変化を，数歩の減衰つきニュートン法で予言する（T3-1〜T3-3）。新しい学習はしない。

目的関数は「ソースの部分は 2 次（H_S で固定），K 本の部分は厳密」の Q(Δ) = ½ Σ_行 Δ_行 H_S Δ_行ᵀ + Σ_j s_j CE(L_s(x_j) + Δ x_j, y_j)。
1 歩目は E16 の 1 歩（`explore_direction.newton_step`）と同じ。各歩で K 本の確率と曲がりを計算し直し，Armijo の後退で歩幅を決める。
入力は E16・E17 と同じ（C を固定した段階 B の点数，`results/shift_<kind>/logits/`）。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_newton_steps.py [--kinds …]

出力：`OUT/e28t3_rows.csv`（組 × K × 反復 × 感情の対の実測と 1 歩・数歩の予言），`OUT/e28t3_pairs.csv`，`OUT/e28t3_tests.csv`
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import log_softmax, softmax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import groups_for, load_any  # noqa: E402
from explore_direction import balanced_sw, d_o, hs_inv, map_draw, newton_step  # noqa: E402
from explore_synergy import KS, N_CLASSES, O_MIN, PAIRS, kappa, load_map, src_parts  # noqa: E402
from pair_table import OUT, OUT_ROOT  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
MAX_STEPS, MAX_HALVING, TOL = 10, 20, 1e-6


def hs_mul(sp: dict, C: float):
    """H_S を右から掛ける関数：M (k, d+1) → M H_S。"""
    V, lam, a = sp["V"], sp["ev"] + 1.0 / C, sp["a"]
    def f(M: np.ndarray) -> np.ndarray:
        out = np.empty_like(M)
        out[:, :-1] = ((M[:, :-1] @ V) * lam[None, :]) @ V.T
        out[:, -1] = M[:, -1] * a
        return out
    return f


def solve_h(Hinv, X: np.ndarray, D: np.ndarray, M: np.ndarray) -> np.ndarray:
    """M (H_S + Xᵀ diag(D) X)⁻¹ をウッドベリーで（M は (k, d+1)）。"""
    HiX = Hinv(X.T)                                               # (d+1, m)
    MHi = Hinv(M.T).T                                             # (k, d+1)
    inner = np.diag(1.0 / D) + X @ HiX
    return MHi - (MHi @ X.T) @ np.linalg.solve(inner, HiX.T)


def newton_steps(sp: dict, C: float, X: np.ndarray, L0: np.ndarray, y: np.ndarray, sw: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """数歩の減衰つきニュートン法。返り値は (1 歩の Δ，最終の Δ，歩数，収束したか)。"""
    Hinv, Hmul = hs_inv(sp, C), hs_mul(sp, C)
    k = N_CLASSES
    Y = np.eye(k)[y]

    def q(D: np.ndarray) -> float:
        L = L0 + X @ D.T
        return 0.5 * float((Hmul(D) * D).sum()) - float((sw * (log_softmax(L, axis=1) * Y).sum(1)).sum())

    Delta = np.zeros((k, X.shape[1]))
    first, f = None, q(Delta)
    conv = False
    for t in range(MAX_STEPS):
        P = softmax(L0 + X @ Delta.T, axis=1)
        G = ((Y - P) * sw[:, None]).T @ X                         # (k, d+1)
        grad = Hmul(Delta) - G
        if np.abs(grad).max() < TOL:
            conv = True
            break
        Dk = sw * np.maximum((1 - (P ** 2).sum(1)) / (k - 1), 1e-12)
        direction = -solve_h(Hinv, X, Dk, grad)
        gtd = float((grad * direction).sum())
        step = 1.0
        for _ in range(MAX_HALVING):
            cand = Delta + step * direction
            fc = q(cand)
            if fc <= f + 1e-4 * step * gtd:
                break
            step *= 0.5
        else:
            break                                                 # 行き詰まり
        Delta, f = cand, fc
        if first is None:
            first = Delta.copy()
    if first is None:
        first = Delta.copy()
    return first, Delta, t + 1, conv


def run_kind(kind: str) -> pd.DataFrame:
    mp = load_map(kind)
    groups = groups_for(kind)
    data = {g: load_any(g, kind) for g in groups}
    rows = []
    for s in groups:
        dfs, Xs = data[s]
        ys = dfs.label_4.to_numpy()
        mu, sd = Xs.mean(0), Xs.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Zs = ((Xs - mu) / sd).astype(np.float64)
        z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
        ks_all = kappa(z["oof"])
        n_src_c = np.bincount(ys, minlength=N_CLASSES)
        sps = {K: src_parts(Zs, ys, ks_all, int(K)) for K in KS}
        for t in groups:
            if t == s:
                continue
            dft, Xt = data[t]
            yt = dft.label_4.to_numpy()
            Zt = ((Xt - mu) / sd).astype(np.float64)
            Lt = z[f"t_{t}"].astype(np.float64)
            B = np.load(OUT_ROOT / f"merged_scores_{kind}" / f"{t}.npz")
            g = mp[(mp.target == t) & (mp.source == s)]
            reps_b = sorted(int(k.split("_r")[1]) for k in B.files if k.startswith("test_idx_r"))
            for K in KS:
                sp = sps[K]
                Xa = np.c_[Zt - sp["zbar"], np.ones(len(Zt))]
                swc = balanced_sw(n_src_c, int(K))
                for r in reps_b:
                    te, idx = map_draw(dft, t, r, int(K))
                    assert np.array_equal(np.flatnonzero(te), B[f"test_idx_r{r}"]), (kind, t, r)
                    C = float(g[(g.K == K) & (g.rep == r)].C_m.iloc[0])
                    sw = swc[yt[idx]]
                    d1, dT, n_steps, conv = newton_steps(sp, C, Xa[idx], Lt[idx], yt[idx], sw)
                    ref1 = newton_step(hs_inv(sp, C), Xa[idx], Lt[idx] * 0 + softmax(Lt[idx], axis=1), yt[idx], sw)
                    Ls_te, Lm_te = B[f"s_{s}_r{r}"].astype(np.float64), B[f"m_{s}_K{K}_r{r}"].astype(np.float64)
                    y_te = yt[te]
                    ds, dm = d_o(Ls_te, y_te), d_o(Lm_te, y_te)
                    dp1, dpT = d_o(Ls_te + Xa[te] @ d1.T, y_te), d_o(Ls_te + Xa[te] @ dT.T, y_te)
                    for (a, b), x_s, x_m, x_1, x_T in zip(PAIRS, ds, dm, dp1, dpT):
                        rows.append({"kind": kind, "target": t, "source": s, "K": K, "rep": r, "pair": f"{a}-{b}", "C": C,
                                     "d_s": x_s[0], "o_s": x_s[1], "d_m": x_m[0], "o_m": x_m[1],
                                     "d_p1": x_1[0], "o_p1": x_1[1], "d_pT": x_T[0], "o_pT": x_T[1],
                                     "n_steps": n_steps, "converged": conv,
                                     "step1_diff": float(np.abs(d1 - ref1).max())})
        print(f"  {kind} ソース {s} 済み", flush=True)
    return pd.DataFrame(rows)


def lam_obs(o_s: pd.Series, o_x: pd.Series) -> float:
    v = o_s.abs() >= O_MIN
    return 1 - float((np.sign(o_s[v]) * o_x[v]).sum() / o_s[v].abs().sum()) if v.any() else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    args = ap.parse_args()
    t0 = time.time()
    R = pd.concat([run_kind(k) for k in args.kinds], ignore_index=True)
    R.to_csv(OUT / "e28t3_rows.csv", index=False)
    for x in ("m", "p1", "pT"):
        R[f"dc_{x}"] = R[f"d_{x}"] - R.d_s
    A = R.groupby(["kind", "target", "source", "K"]).apply(lambda g: pd.Series({
        "dc_m": g.dc_m.mean(), "dc_p1": g.dc_p1.mean(), "dc_pT": g.dc_pT.mean(),
        "dc_p1_median": g.dc_p1.median(), "dc_pT_median": g.dc_pT.median(),
        "lam_m": lam_obs(g.o_s, g.o_m), "lam_p1": lam_obs(g.o_s, g.o_p1), "lam_pT": lam_obs(g.o_s, g.o_pT),
        "n_steps": g.n_steps.mean(), "conv": g.converged.mean()})).reset_index()
    A.to_csv(OUT / "e28t3_pairs.csv", index=False)
    tests = []
    for kind in args.kinds:
        a, r = A[A.kind == kind], R[R.kind == kind]
        rho1, rhoT = stats.spearmanr(a.dc_p1, a.dc_m)[0], stats.spearmanr(a.dc_pT, a.dc_m)[0]
        is_audio = kind.startswith("audio")
        tests.append({"kind": kind, "test": "T3-1", "rho_1step": rho1, "rho_multi": rhoT,
                      "holds": (rhoT > rho1 and rhoT >= 0.6) if is_audio else rhoT >= 0.8, "n": len(a)})
        out1, outT = float((r.dc_p1.abs() > 1).mean()), float((r.dc_pT.abs() > 1).mean())
        tests.append({"kind": kind, "test": "T3-2", "outlier_1step": out1, "outlier_multi": outT,
                      "holds": (outT < 0.02) if is_audio else np.nan, "n": len(r)})
        g = a.dropna(subset=["lam_m", "lam_pT"])
        tests.append({"kind": kind, "test": "T3-3", "rho_lam_1step": stats.spearmanr(g.lam_p1, g.lam_m)[0],
                      "rho_lam_multi": stats.spearmanr(g.lam_pT, g.lam_m)[0],
                      **{f"med_K{K}_{x}": float(a[a.K == K][f"dc_{x}"].median()) for K in KS for x in ("m", "p1", "pT")},
                      "mean_steps": a.n_steps.mean(), "conv_rate": a.conv.mean(), "step1_max_diff": r.step1_diff.max()})
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "e28t3_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"{len(R)} 行，{time.time() - t0:.0f} 秒")
    print(T.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
