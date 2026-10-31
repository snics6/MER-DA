"""【探索的】E28 の 1 節：仮説転移の式の検証（予言 T1-1〜T1-8）。新しい学習はしない。

入力は `exp/transfer_grid.py` の点数（ソースの統計で標準化した仮説転移．設定 × K × 反復），
C を固定した段階 B の合併（`OUT/e15b_rows.csv`），E15 の式の λ（`OUT/e15_lambda.csv`）。
向きの予言は，K 本と保存したソースのみの点数（`results/shift_<kind>/logits/`）から，
等方の罰則のニュートン法の 1 歩 [ΔW, Δb] = G_t (H_t + P)⁻¹ を計算し，ソースのみのテストの点数に足して作る。
H_t は階数が 4K 以下なので，ウッドベリーの公式で小さい行列だけを解く。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_transfer_theory.py [--kinds …]

出力：`OUT/e28t1_rows.csv`（種類 × 組 × K × 反復 × 設定 × 感情の対の d・o と予言），`OUT/e28t1_pairs.csv`（組 × K × 設定），
`OUT/e28t1_tests.csv`（T1-1〜T1-8 の判定），`OUT/e28t1_summary.csv`（種類 × 設定 × K の平均）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import groups_for, load_any  # noqa: E402
from explore_direction import map_draw  # noqa: E402
from explore_synergy import KS, O_MIN, PAIRS, d_o, kappa  # noqa: E402
from pair_table import OUT  # noqa: E402
from transfer_grid import CB_C, CB_LIST, FIXED_CS, FREE_CS, cfg_name  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
NAME = {"audio_wavlm": "W7", "audio_hubert": "Hu7", "video": "V", "fuse-audio-video": "AV"}
FREE = [cfg_name("free", c) for c in FREE_CS]
FIXED = [cfg_name("fixed", c) for c in FIXED_CS]
CBS = [cfg_name("cb", CB_C, cb) for cb in CB_LIST]
CONFIGS = {**{cfg_name("free", c): ("free", c, None) for c in FREE_CS},
           **{cfg_name("fixed", c): ("fixed", c, None) for c in FIXED_CS},
           **{cfg_name("cb", CB_C, cb): ("cb", CB_C, cb) for cb in CB_LIST}}
EPS_B = 1e-8                     # 自由な切片の数値の安定のための小さな罰則
GRID_ROOT = ROOT / "results"     # `--grid-root` で変えられる（試走の出力で配管を確かめるため）


def newton_ht(Zk: np.ndarray, P: np.ndarray, y: np.ndarray, mode: str, Cp: float, Cb: float | None) -> np.ndarray:
    """等方の罰則のニュートン法の 1 歩。返り値は (4, d+1)（最後の列が切片の変化）。"""
    n, d = Zk.shape
    kap = (1 - (P ** 2).sum(1)) / 3
    R = np.eye(4)[y] - P                                   # 残差 (n, 4)。重み s_j は K 本が各感情 K 本なので 1
    X = np.c_[Zk, np.ones(n)]
    G = R.T @ X                                            # (4, d+1)
    if mode == "fixed":
        U, Mdiag, c = Zk, kap, 1 / Cp
        Gu = G[:, :d]
    else:
        pen_b = EPS_B if mode == "free" else 1 / Cb
        U = np.vstack([X, np.r_[np.zeros(d), 1.0][None]])  # (n+1, d+1)
        Mdiag = np.r_[kap, pen_b - 1 / Cp]
        c, Gu = 1 / Cp, G
    # (cI + Uᵀ M U)⁻¹ = (1/c) I − (1/c²) Uᵀ M (I + U Uᵀ M / c)⁻¹ U
    S = U @ U.T
    core = np.linalg.solve(np.eye(len(Mdiag)) + S * Mdiag[None, :] / c, (U @ Gu.T))   # (n+1, 4)
    dW = Gu / c - ((Mdiag[:, None] * core).T @ U) / c ** 2
    if mode == "fixed":
        dW = np.c_[dW, np.zeros(4)]
    return dW


def rows_for_kind(kind: str) -> pd.DataFrame:
    groups = groups_for(kind)
    data = {g: load_any(g, kind) for g in groups}
    out = []
    for t in groups:
        f = GRID_ROOT / f"transfer_grid_{kind}" / f"{t}.npz"
        if not f.exists():
            print(f"  {kind} {t}：点数が無い（飛ばす）", flush=True)
            continue
        z = np.load(f)
        dft, Xt = data[t]
        reps = sorted(int(k.split("_r")[1]) for k in z.files if k.startswith("test_idx_r"))
        srcs = sorted({k[2:].rsplit("_r", 1)[0] for k in z.files if k.startswith("s_")})
        for s in srcs:
            dfs, Xs = data[s]
            mu, sd = Xs.mean(0), Xs.std(0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            Zt_all = ((Xt - mu) / sd).astype(np.float64)
            Lt = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")[f"t_{t}"]
            Pt = np.exp(Lt - Lt.max(1, keepdims=True)); Pt /= Pt.sum(1, keepdims=True)
            y_all = dft.label_4.to_numpy()
            for r in reps:
                te_idx, y_te = z[f"test_idx_r{r}"], z[f"y_test_r{r}"]
                Ls = z[f"s_{s}_r{r}"].astype(np.float64)
                ds = d_o(Ls, y_te)
                Xte_aug = np.c_[Zt_all[te_idx], np.ones(len(te_idx))]
                te_chk, _ = map_draw(dft, t, r, 1)
                assert np.array_equal(np.flatnonzero(te_chk), te_idx), (kind, t, r)
                for K in KS:
                    _, idx = map_draw(dft, t, r, int(K))
                    Zk, Pk, yk = Zt_all[idx], Pt[idx], y_all[idx]
                    for name, (mode, Cp, Cb) in CONFIGS.items():
                        key = f"{name}_{s}_K{K}_r{r}"
                        if key not in z.files:
                            continue
                        dh = d_o(z[key].astype(np.float64), y_te)
                        dW = newton_ht(Zk, Pk, yk, mode, Cp, Cb)
                        dp = d_o(Ls + Xte_aug @ dW.T, y_te)
                        for (a, b), (d_s, o_s), (d_h, o_h), (d_p, o_p) in zip(PAIRS, ds, dh, dp):
                            out.append({"kind": kind, "target": t, "source": s, "K": K, "rep": r, "config": name,
                                        "pair": f"{a}-{b}", "d_s": d_s, "o_s": o_s, "d_h": d_h, "o_h": o_h,
                                        "d_pred": d_p, "o_pred": o_p})
        print(f"  {kind} {t}：済み", flush=True)
    return pd.DataFrame(out)


def lam_obs(o_s: pd.Series, o_x: pd.Series) -> float:
    v = o_s.abs() >= O_MIN
    return 1 - float((np.sign(o_s[v]) * o_x[v]).sum() / o_s[v].abs().sum()) if v.any() else np.nan


def analyse(R: pd.DataFrame, kinds: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uar = pd.concat([pd.read_csv(GRID_ROOT / f"transfer_grid_{k}" / f"{t}.csv", dtype={"K": str}).assign(kind=k)
                     for k in kinds for t in groups_for(k)
                     if (GRID_ROOT / f"transfer_grid_{k}" / f"{t}.csv").exists()])
    uar = uar[uar.K != ""]
    U = uar.groupby(["kind", "target", "source", "K", "config"]).agg(uar=("uar", "mean"), uar_ref=("uar_ref", "mean"),
                                                                      db_norm=("db_norm", "mean"), dW_norm=("dW_norm", "mean")).reset_index()
    merged = pd.read_csv(OUT / "e15b_rows.csv", dtype={"K": str})
    lam = pd.read_csv(OUT / "e15_lambda.csv", comment="#", dtype={"K": str})[["kind", "target", "source", "K", "lam_pred_rep09", "G"]]
    # 組 × K × 設定
    agg = []
    for (kind, t, s, K, cfg), g in R.groupby(["kind", "target", "source", "K", "config"]):
        agg.append({"kind": kind, "target": t, "source": s, "K": K, "config": cfg,
                    "lam_h": lam_obs(g.o_s, g.o_h), "lam_pred_newton": lam_obs(g.o_s, g.o_pred),
                    "abs_o_s": g.o_s.abs().mean(), "abs_o_h": g.o_h.abs().mean(),
                    "d_s": g.d_s.mean(), "dchg_h": (g.d_h - g.d_s).mean(), "dchg_pred": (g.d_pred - g.d_s).mean()})
    A = pd.DataFrame(agg).merge(lam, on=["kind", "target", "source", "K"], how="left") \
                         .merge(U, on=["kind", "target", "source", "K", "config"], how="left")
    mA = merged.groupby(["kind", "target", "source", "K"]).apply(
        lambda g: pd.Series({"lam_m": lam_obs(g.o_s, g.o_m), "dchg_m": (g.d_m - g.d_s).mean()})).reset_index()
    A = A.merge(mA, on=["kind", "target", "source", "K"], how="left")
    S = A.groupby(["kind", "config", "K"])[["lam_h", "lam_pred_newton", "lam_m", "abs_o_s", "abs_o_h", "dchg_h", "dchg_pred",
                                          "dchg_m", "uar", "db_norm", "dW_norm"]].mean().reset_index()

    tests = []
    for kind in kinds:
        r = R[R.kind == kind]
        a = A[A.kind == kind]
        # T1-1 位置の揺れ：反復間の o_h の分散（|o_s| の反復平均 ≥ 0.3 の (組, 感情の対)）
        for cfg in ["free_C0.001", "free_C0.01"]:
            g = r[r.config == cfg].groupby(["target", "source", "K", "pair"]).agg(
                v=("o_h", "var"), n=("o_h", "size"), ao=("o_s", lambda x: x.abs().mean())).reset_index()
            g = g[(g.ao >= O_MIN) & (g.n >= 5)]
            for K in KS:
                med = float(g[g.K == K].v.median())
                tests.append({"kind": kind, "test": "T1-1", "config": cfg, "K": K, "value": med, "ref": 1 / (2 * int(K)),
                              "ratio": med * 2 * int(K), "holds": 0.5 <= med * 2 * int(K) <= 2.0, "n": int((g.K == K).sum())})
        # T1-2 期待値で全部直す：組 × 感情の対の反復平均 o_h を o_s に回帰した傾き
        for cfg in FREE:
            for K in KS:
                g = r[(r.config == cfg) & (r.K == K)].groupby(["target", "source", "pair"])[["o_s", "o_h"]].mean()
                slope = float(np.polyfit(g.o_s, g.o_h, 1)[0])
                tests.append({"kind": kind, "test": "T1-2", "config": cfg, "K": K, "value": slope, "holds": slope < 0.2, "n": len(g)})
        for K in KS:
            g = merged[(merged.kind == kind) & (merged.K == K)].groupby(["target", "source", "pair"])[["o_s", "o_m"]].mean()
            tests.append({"kind": kind, "test": "T1-2 参考（合併）", "config": "merged", "K": K,
                          "value": float(np.polyfit(g.o_s, g.o_m, 1)[0]), "n": len(g)})
        # T1-3 切片を固定した λ と式の λ
        for cfg in FIXED:
            g = a[(a.config == cfg) & (a.K == "5")].dropna(subset=["lam_h", "lam_pred_rep09"])
            fr = a[(a.config == "free_C0.01") & (a.K == "5")]
            rho, p = stats.spearmanr(g.lam_pred_rep09, g.lam_h)
            tests.append({"kind": kind, "test": "T1-3", "config": cfg, "K": "5", "value": float(rho), "p": float(p),
                          "mean_lam_fixed": g.lam_h.mean(), "mean_lam_free": fr.lam_h.mean(),
                          "holds": rho > 0.5 and g.lam_h.mean() < fr.lam_h.mean(), "n": len(g)})
        g = a[(a.config == "free_C0.01") & (a.K == "5")].dropna(subset=["lam_h", "lam_pred_rep09"])
        tests.append({"kind": kind, "test": "T1-3 参考（自由な切片の λ と式の λ）", "config": "free_C0.01", "K": "5",
                      "value": float(stats.spearmanr(g.lam_pred_rep09, g.lam_h)[0]), "n": len(g)})
        # T1-4 向きの予言（組 × K × C′ ≤ 0.1）
        g = a[a.config.isin(["free_C0.001", "free_C0.01", "free_C0.1"])].dropna(subset=["dchg_h", "dchg_pred"])
        rho, p = stats.spearmanr(g.dchg_pred, g.dchg_h)
        ref = {"audio_wavlm": 0.48, "audio_hubert": 0.35}.get(kind, np.nan)
        tests.append({"kind": kind, "test": "T1-4", "config": "free C′≤0.1", "K": "all", "value": float(rho), "p": float(p),
                      "ref": ref, "holds": rho > 0 and (np.isnan(ref) or rho > ref), "n": len(g)})
        for cfg in FREE:
            h = g[g.config == cfg] if cfg in g.config.unique() else a[a.config == cfg]
            tests.append({"kind": kind, "test": "T1-4 設定ごと", "config": cfg, "K": "all",
                          "value": float(stats.spearmanr(h.dchg_pred, h.dchg_h)[0]), "n": len(h)})
        # T1-5 K に比例，C′ で飽和
        rat = {}
        for cfg in FREE:
            w = a[a.config == cfg].pivot_table(index=["target", "source"], columns="K", values="dchg_h")
            ok = w["5"] > 0.005
            rat[cfg] = float((w.loc[ok, "10"] / w.loc[ok, "5"]).median())
            tests.append({"kind": kind, "test": "T1-5 比", "config": cfg, "K": "10/5", "value": rat[cfg], "n": int(ok.sum())})
        small = np.mean([rat["free_C0.001"], rat["free_C0.01"]])
        tests.append({"kind": kind, "test": "T1-5", "config": "C′≤0.01 vs C′=1", "K": "10/5", "value": small, "ref": rat["free_C1"],
                      "holds": all(1.5 <= rat[c] <= 2.5 for c in ["free_C0.001", "free_C0.01"]) and rat["free_C1"] < small})
        # T1-6 切片の縛り（K=1，C_b = 2）
        base = a[(a.config == "free_C0.01") & (a.K == "1")].set_index(["target", "source"])
        for cfg in CBS:
            cb = a[(a.config == cfg) & (a.K == "1")].set_index(["target", "source"])
            diff = (cb.uar - base.uar).dropna()
            ao = base.abs_o_s.loc[diff.index]
            lo, hi = ao.quantile(1 / 3), ao.quantile(2 / 3)
            m_lo, m_hi = float(diff[ao <= lo].mean()), float(diff[ao >= hi].mean())
            tests.append({"kind": kind, "test": "T1-6" if cfg == "cb2_C0.01" else "T1-6 記述", "config": cfg, "K": "1",
                          "value": m_lo, "ref": m_hi, "mean_all": float(diff.mean()),
                          "holds": (m_lo > 0 and m_hi < 0) if cfg == "cb2_C0.01" else np.nan, "n": len(diff)})
        # T1-7 d を最大にする C′ の K=5 と K=10 の一致
        best = {}
        for K in ("5", "10"):
            w = a[(a.K == K) & a.config.isin(FREE)].pivot_table(index=["target", "source"], columns="config", values="d_s")
            v = a[(a.K == K) & a.config.isin(FREE)].pivot_table(index=["target", "source"], columns="config", values="dchg_h")
            best[K] = v.idxmax(axis=1)
        agree = float((best["5"] == best["10"].loc[best["5"].index]).mean())
        tests.append({"kind": kind, "test": "T1-7", "config": "free", "K": "5 vs 10", "value": agree, "ref": 0.25, "n": len(best["5"])})
        # T1-8 標準化（free_Csrc と手 7）
        for K in ("1", "10"):
            g = U[(U.kind == kind) & (U.config == "free_Csrc") & (U.K == K)].dropna(subset=["uar_ref"])
            dm = float((g.uar - g.uar_ref).mean())
            tests.append({"kind": kind, "test": "T1-8", "config": "free_Csrc − 手 7", "K": K, "value": dm,
                          "n_pos": int((g.uar > g.uar_ref).sum()), "holds": (dm > 0) if K == "1" else (abs(dm) < 0.005), "n": len(g)})
    return A, S, pd.DataFrame(tests)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    ap.add_argument("--grid-root", type=Path)
    ap.add_argument("--rows-only", action="store_true", help="配管の確かめ：d・o の行だけ作り，検定はしない")
    args = ap.parse_args()
    global GRID_ROOT
    if args.grid_root:
        GRID_ROOT = args.grid_root
    t0 = time.time()
    R = pd.concat([rows_for_kind(k) for k in args.kinds], ignore_index=True)
    if args.rows_only:
        print(f"{len(R)} 行，欠損 {int(R.isna().sum().sum())}，設定 {sorted(R.config.unique())}，{time.time() - t0:.0f} 秒")
        return 0
    R.to_csv(OUT / "e28t1_rows.csv", index=False)
    A, S, T = analyse(R, args.kinds)
    A.to_csv(OUT / "e28t1_pairs.csv", index=False)
    S.to_csv(OUT / "e28t1_summary.csv", index=False)
    T.to_csv(OUT / "e28t1_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"{len(R)} 行，{time.time() - t0:.0f} 秒")
    print(T.round(3).to_string(index=False))
    print(S.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
