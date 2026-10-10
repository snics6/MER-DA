"""【探索的】E28 の 7 節：全ソースを混ぜた合併の位置の直り λ を，E15 の式で予言できるか（T7-1〜T7-3）。新しい学習はしない。

入力：`exp/multisource_scores.py` の点数と確率（`results/multisource_scores_<kind>/`），1 つずつ混ぜたときの λ
（C を固定した段階 B：`OUT/e15b_rows.csv`・`OUT/e15b_pairs.csv`，式の λ：`OUT/e15_lambda.csv`）。
式の λ は，混ぜたソースの統計で標準化した特徴と，混ぜたソースのみの分類器の確率（κ_s は学習データでの確率．近似）で計算する。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_multisource_lambda.py

出力：`OUT/e28t7_targets.csv`（ターゲット × K），`OUT/e28t7_tests.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_pairs import ba  # noqa: E402
from explore_synergy import KS, O_MIN, PAIRS, d_o, lam_at, parts, src_parts  # noqa: E402
from pair_table import OUT  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]


def lam_obs(o_s: pd.Series, o_x: pd.Series) -> float:
    v = o_s.abs() >= O_MIN
    return 1 - float((np.sign(o_s[v]) * o_x[v]).sum() / o_s[v].abs().sum()) if v.any() else np.nan


def main() -> int:
    single_rows = pd.read_csv(OUT / "e15b_rows.csv", dtype={"K": str})
    single_rows = single_rows[single_rows.rep < 10]
    lam_single_pred = pd.read_csv(OUT / "e15_lambda.csv", comment="#", dtype={"K": str})
    rows = []
    for kind in KINDS:
        groups = groups_for(kind)
        data = {g: load_any(g, kind) for g in groups}
        for t in groups:
            f = ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz"
            if not f.exists():
                print(f"  {kind} {t}：点数が無い（飛ばす）", flush=True)
                continue
            z = np.load(f)
            csv = pd.read_csv(f.with_suffix(".csv"), dtype={"K": str})
            C = float(csv.C.iloc[0])
            srcs = list(z["src_names"])
            Xs = np.concatenate([data[s][1] for s in srcs])
            ys = z["src_y"]
            mu, sd = Xs.mean(0), Xs.std(0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            Zs = ((Xs - mu) / sd).astype(np.float64)
            P_s, P_t = z["p_src_train"].astype(np.float64), z["p_tgt_all"].astype(np.float64)
            ks = (1 - (P_s ** 2).sum(1)) / 3
            kt = float(((1 - (P_t ** 2).sum(1)) / 3).mean())
            Zt_mean = ((data[t][1] - mu) / sd).mean(0)
            reps = sorted(int(k.split("_r")[1]) for k in z.files if k.startswith("test_idx_r"))
            # 1 つずつ混ぜたときの値（同じコーパスの群は除く）
            sr = single_rows[(single_rows.kind == kind) & (single_rows.target == t) & single_rows.source.isin(srcs)]
            lp = lam_single_pred[(lam_single_pred.kind == kind) & (lam_single_pred.target == t) & lam_single_pred.source.isin(srcs)]
            for K in KS:
                sp = src_parts(Zs, ys, ks, int(K))
                pred = lam_at(parts(sp, Zt_mean, kt), C)
                rec = []
                for r in reps:
                    y = z[f"y_test_r{r}"]
                    ds, dm = d_o(z[f"s_all_r{r}"].astype(np.float64), y), d_o(z[f"m_all_K{K}_r{r}"].astype(np.float64), y)
                    for (a, b), (d_s, o_s), (d_m, o_m) in zip(PAIRS, ds, dm):
                        rec.append({"rep": r, "pair": f"{a}-{b}", "d_s": d_s, "o_s": o_s, "d_m": d_m, "o_m": o_m})
                R = pd.DataFrame(rec)
                pos = (ba(R.d_s, R.o_m) - ba(R.d_s, R.o_s)).mean() * 100
                dire = (ba(R.d_m, R.o_m) - ba(R.d_s, R.o_m)).mean() * 100
                srK = sr[sr.K == K]
                lam_single = srK.groupby("source").apply(lambda g: lam_obs(g.o_s, g.o_m))
                pos_single = (ba(srK.d_s, srK.o_m) - ba(srK.d_s, srK.o_s)).mean() * 100
                dir_single = (ba(srK.d_m, srK.o_m) - ba(srK.d_s, srK.o_m)).mean() * 100
                u = csv[(csv.K == K)]
                us = csv[csv.method == "src_all"]
                rows.append({"kind": kind, "target": t, "K": K, "C": C, "n_src": int(len(ys)), "n_sources": len(srcs),
                             "lam_obs_all": lam_obs(R.o_s, R.o_m), "lam_pred_all": pred["lam"], "G_all": pred["G"], "N_t": sp["N_t"],
                             "lam_obs_single_mean": float(lam_single.mean()), "lam_obs_single_max": float(lam_single.max()),
                             "lam_pred_single_mean": float(lp[lp.K == K].lam_pred_rep09.mean()),
                             "pos_all": pos, "dir_all": dire, "pos_single_mean": pos_single, "dir_single_mean": dir_single,
                             "abs_o_s_all": R.o_s.abs().mean(), "abs_o_s_single": srK.o_s.abs().mean(),
                             "d_s_all": R.d_s.mean(), "d_s_single": srK.d_s.mean(), "dchg_all": (R.d_m - R.d_s).mean(),
                             "dchg_single": (srK.d_m - srK.d_s).mean(),
                             "uar_src_all": us.uar.mean(), "uar_merged_all": u.uar.mean()})
        print(f"  {kind} 済み", flush=True)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "e28t7_targets.csv", index=False)
    tests = []
    for kind in KINDS:
        a = A[A.kind == kind]
        a5 = a[a.K == "5"]
        n_less = int((a5.lam_obs_all < a5.lam_obs_single_mean).sum())
        tests.append({"kind": kind, "test": "T7-1", "n_targets": len(a5), "n_lam_all_lt_single_mean": n_less,
                      "mean_lam_all": a5.lam_obs_all.mean(), "mean_lam_single": a5.lam_obs_single_mean.mean(),
                      "holds": n_less >= 5 if len(a5) == 7 else n_less >= len(a5) - 2})
        g = a.dropna(subset=["lam_obs_all", "lam_pred_all"])
        rho, p = stats.spearmanr(g.lam_pred_all, g.lam_obs_all)
        tests.append({"kind": kind, "test": "T7-2", "rho": float(rho), "p": float(p), "holds": rho >= 0.5, "n": len(g)})
        for K in KS:
            h = a[a.K == K]
            tests.append({"kind": kind, "test": "T7-3", "K": K, "ratio_obs": float((h.lam_obs_all / h.lam_obs_single_mean).median()),
                          "ratio_pred": float((h.lam_pred_all / h.lam_pred_single_mean).median()),
                          "pos_all": h.pos_all.mean(), "pos_single": h.pos_single_mean.mean(),
                          "dir_all": h.dir_all.mean(), "dir_single": h.dir_single_mean.mean(),
                          "abs_o_s_all": h.abs_o_s_all.mean(), "abs_o_s_single": h.abs_o_s_single.mean(),
                          "d_s_all": h.d_s_all.mean(), "d_s_single": h.d_s_single.mean(),
                          "merged_minus_src_all": (h.uar_merged_all - h.uar_src_all).mean()})
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "e28t7_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
