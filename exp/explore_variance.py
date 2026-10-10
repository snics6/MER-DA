"""【探索的】E28 の 2 節：引いた K 本による位置と向きの揺れ（T2-1〜T2-3）。新しい学習はしない。

入力：`OUT/e28t1_rows.csv`（仮説転移の設定 × 反復ごとの d・o．`exp/explore_transfer_theory.py` の出力），
`OUT/e15b_rows.csv`（C を固定した合併の反復ごとの d・o），`results/transfer_grid_<kind>/<ターゲット>.csv`（設定 × 反復の UAR），
C を固定した地図（反復ごとのターゲットのみ・ソースのみ・合併の UAR）。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_variance.py

出力：`OUT/e28t2_tests.csv`，`OUT/e28t2_var.csv`（組 × K × 感情の対の分散），`OUT/e28t2_safety.csv`（反復ごとの安全率）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
from data import groups_for  # noqa: E402
from explore_synergy import KS, O_MIN, load_map  # noqa: E402
from pair_table import OUT  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
REPS = range(10)
KEY = ["kind", "target", "source", "K"]


def lam_obs(o_s: pd.Series, o_x: pd.Series) -> float:
    v = o_s.abs() >= O_MIN
    return 1 - float((np.sign(o_s[v]) * o_x[v]).sum() / o_s[v].abs().sum()) if v.any() else np.nan


def main() -> int:
    H = pd.read_csv(OUT / "e28t1_rows.csv", dtype={"K": str})
    H = H[H.rep.isin(REPS)]
    Mg = pd.read_csv(OUT / "e15b_rows.csv", dtype={"K": str})
    Mg = Mg[Mg.rep.isin(REPS) & Mg.kind.isin(KINDS)]
    tests, var_rows = [], []
    # ---- T2-1：位置の揺れ．Var_rep(o_m) / Var_rep(o_h) と λ_m²
    h = H[H.config == "free_C0.01"]
    vh = h.groupby(KEY + ["pair"]).agg(var_h=("o_h", "var"), n_h=("o_h", "size"), ao=("o_s", lambda x: x.abs().mean())).reset_index()
    vm = Mg.groupby(KEY + ["pair"]).agg(var_m=("o_m", "var"), n_m=("o_m", "size")).reset_index()
    lam_m = Mg.groupby(KEY).apply(lambda g: lam_obs(g.o_s, g.o_m)).rename("lam_m").reset_index()
    V = vh.merge(vm, on=KEY + ["pair"]).merge(lam_m, on=KEY)
    V = V[(V.ao >= O_MIN) & (V.n_h >= 5) & (V.n_m >= 5)]
    V["ratio"] = V.var_m / V.var_h
    V["lam_m2"] = V.lam_m ** 2
    V.to_csv(OUT / "e28t2_var.csv", index=False)
    for kind in KINDS:
        for K in KS:
            g = V[(V.kind == kind) & (V.K == K)].dropna(subset=["ratio", "lam_m2"])
            rho, p = stats.spearmanr(g.ratio, g.lam_m2)
            med_r, med_l = float(g.ratio.median()), float(g.lam_m2.median())
            tests.append({"kind": kind, "test": "T2-1" if K == "5" else "T2-1 記述", "K": K, "rho": float(rho), "p": float(p),
                          "median_ratio": med_r, "median_lam2": med_l, "median_var_h": float(g.var_h.median()),
                          "median_var_m": float(g.var_m.median()), "ref_1_over_2K": 1 / (2 * int(K)),
                          "holds": (rho > 0.5 and 0.5 <= med_r / med_l <= 2.0) if K == "5" else np.nan, "n": len(g)})
    # ---- T2-2：向きの揺れ．組ごとの Var_rep(Δd)（反復ごとに 6 対の平均をとってから分散）
    D = H[H.config.isin(["free_C0.001", "free_C0.01"])].assign(dc=lambda x: x.d_h - x.d_s)
    per_rep = D.groupby(["kind", "config", "target", "source", "K", "rep"]).dc.mean().reset_index()
    vd = per_rep.groupby(["kind", "config", "target", "source", "K"]).dc.var().rename("var_dc").reset_index()
    for kind in KINDS:
        w = vd[vd.kind == kind].pivot_table(index=["config", "target", "source"], columns="K", values="var_dc")
        fK = float((w["10"] > w["1"]).mean())
        w5 = vd[(vd.kind == kind) & (vd.K == "5")].pivot_table(index=["target", "source"], columns="config", values="var_dc")
        fC = float((w5["free_C0.01"] > w5["free_C0.001"]).mean())
        tests.append({"kind": kind, "test": "T2-2", "frac_var_K10_gt_K1": fK, "frac_var_C0.01_gt_C0.001_K5": fC,
                      "median_var_K1": float(w["1"].median()), "median_var_K5": float(w["5"].median()),
                      "median_var_K10": float(w["10"].median()), "holds": fK >= 0.7 and fC >= 0.7, "n": len(w5)})
    # ---- T2-3：反復ごとの安全率
    saf = []
    for kind in KINDS:
        mp = load_map(kind)
        mp = mp[mp.rep.isin(REPS)]
        mp["best_simple"] = mp[["t", "s"]].max(axis=1)
        base = mp[["target", "source", "K", "rep", "t", "s", "m", "best_simple"]]
        g = pd.concat([pd.read_csv(ROOT / "results" / f"transfer_grid_{kind}" / f"{t}.csv", dtype={"K": str})
                       for t in groups_for(kind) if (ROOT / "results" / f"transfer_grid_{kind}" / f"{t}.csv").exists()])
        g = g[(g.K != "") & g.rep.isin(REPS)][["target", "source", "K", "rep", "config", "uar"]]
        g = g.pivot_table(index=["target", "source", "K", "rep"], columns="config", values="uar").reset_index()
        X = base.merge(g, on=["target", "source", "K", "rep"], how="left")
        cfgs = [c for c in X.columns if c.startswith(("free_", "fixed_", "cb"))]
        for K in KS:
            x = X[X.K == K]
            row = {"kind": kind, "K": K, "n": len(x)}
            for c in ["s", "m"] + cfgs:
                row[f"unsafe_{c}"] = float((x[c] < x.best_simple - 0.01).mean())
                row[f"mean_{c}"] = float(x[c].mean())
            saf.append(row)
    S = pd.DataFrame(saf)
    S.to_csv(OUT / "e28t2_safety.csv", index=False)
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "e28t2_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string(index=False))
    print(S[[c for c in S.columns if c.startswith(("kind", "K", "n", "unsafe_"))]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
