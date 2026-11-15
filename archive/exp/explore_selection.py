"""【探索的】E28 の 8 節：K 本の交差検証で候補の C を選ぶことは，どの K から当てになるか（T8-1〜T8-4）。新しい学習はしない。

交差検証の地図（`results/map_<kind>/`）の合併の行の `cv_uar_C*`（K 本での成績），`test_uar_C*`（候補ごとのテストの UAR），
`C`・`C_rule`（同点の規則つきで選んだ C）と，ソースのみの行の `C`（固定の規則）を使う。

    ./.venv/bin/python exp/explore_selection.py

出力：`results/explore_relation/e28t8_rows.csv`，`e28t8_summary.csv`（種類 × K），`e28t8_tests.csv`，`e28t8_calib.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
from map import C_GRID  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT = ROOT / "results" / "explore_relation"
KS = ["1", "5", "10", "25", "50", "all"]
TC = [f"test_uar_C{c:g}" for c in C_GRID]


def rows(kind: str) -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str})
                   for f in sorted((ROOT / "results" / f"map_{kind}").glob("*.csv"))], ignore_index=True)
    s = d[d.method == "src_only"].set_index(["target", "source", "rep"]).C
    m = d[d.method == "merged"].copy()
    m["C_src"] = [s.get((t, so, r), np.nan) for t, so, r in zip(m.target, m.source, m.rep)]
    T = m[TC].to_numpy()
    m["best_test"] = T.max(1)
    m["second_test"] = np.sort(T, axis=1)[:, -2]
    m["best_C"] = [C_GRID[i] for i in T.argmax(1)]
    m["uar_fix"] = [row[f"test_uar_C{c:g}"] if not np.isnan(c) else np.nan for c, (_, row) in zip(m.C_src, m.iterrows())]
    m["regret_cv"] = m.best_test - m.uar
    m["regret_fix"] = m.best_test - m.uar_fix
    m["tie"] = m.C_rule.eq("tie")
    m["correct_cv"] = np.isclose(m.uar, m.best_test)
    m["n_val"] = 4 * m.K.map(lambda k: {"all": np.nan}.get(k, k)).astype(float)
    m["delta12"] = m.best_test - m.second_test
    p = m.best_test.clip(0.05, 0.95)
    m["p_pred"] = stats.norm.cdf(m.delta12 * np.sqrt(m.n_val) / np.sqrt(2 * p * (1 - p)))
    m["kind"] = kind
    return m


def main() -> int:
    R = pd.concat([rows(k) for k in KINDS], ignore_index=True)
    R.to_csv(OUT / "e28t8_rows.csv", index=False)
    summ, tests, calib = [], [], []
    for kind in KINDS:
        r = R[R.kind == kind]
        for K in KS:
            g = r[r.K == K]
            summ.append({"kind": kind, "K": K, "n": len(g), "tie_rate": g.tie.mean(), "within1_cv": (g.regret_cv <= 0.01).mean(),
                         "within1_fix": (g.regret_fix <= 0.01).mean(), "correct_cv": g.correct_cv.mean(),
                         "median_regret_cv": g.regret_cv.median(), "mean_regret_cv": g.regret_cv.mean(),
                         "median_regret_fix": g.regret_fix.median(), "mean_regret_fix": g.regret_fix.mean(),
                         "mean_best": g.best_test.mean(), "mean_uar_cv": g.uar.mean(), "mean_uar_fix": g.uar_fix.mean()})
        S = pd.DataFrame([x for x in summ if x["kind"] == kind]).set_index("K")
        tests.append({"kind": kind, "test": "T8-1", "within1_K1": S.loc["1", "within1_cv"], "tie_K1": S.loc["1", "tie_rate"],
                      "within1_K25": S.loc["25", "within1_cv"], "within1_K50": S.loc["50", "within1_cv"], "within1_all": S.loc["all", "within1_cv"],
                      "holds": S.loc["1", "within1_cv"] < 0.6 and S.loc["1", "tie_rate"] >= 0.5
                      and all(S.loc[k, "within1_cv"] >= 0.85 for k in ("25", "50", "all"))})
        med = [S.loc[k, "median_regret_cv"] for k in ("1", "5", "10", "25", "50")]
        tests.append({"kind": kind, "test": "T8-2", **{f"median_regret_K{k}": v for k, v in zip(("1", "5", "10", "25", "50"), med)},
                      "holds": all(a > b for a, b in zip(med[:-1], med[1:]))})
        tests.append({"kind": kind, "test": "T8-3", "K5_cv": S.loc["5", "median_regret_cv"], "K5_fix": S.loc["5", "median_regret_fix"],
                      "all_cv": S.loc["all", "median_regret_cv"], "all_fix": S.loc["all", "median_regret_fix"],
                      "K5_mean_cv": S.loc["5", "mean_regret_cv"], "K5_mean_fix": S.loc["5", "mean_regret_fix"],
                      "all_mean_cv": S.loc["all", "mean_regret_cv"], "all_mean_fix": S.loc["all", "mean_regret_fix"],
                      "holds": S.loc["5", "median_regret_cv"] > S.loc["5", "median_regret_fix"]
                      and S.loc["all", "median_regret_cv"] < S.loc["all", "median_regret_fix"]})
        g = r[r.K != "all"].dropna(subset=["p_pred"])
        g = g.assign(bin=pd.cut(g.p_pred, [0, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0], include_lowest=True))
        for b, h in g.groupby("bin", observed=True):
            calib.append({"kind": kind, "bin": str(b), "n": len(h), "pred_mean": h.p_pred.mean(), "observed_correct": h.correct_cv.mean(),
                          "observed_within1": (h.regret_cv <= 0.01).mean()})
    pd.DataFrame(summ).to_csv(OUT / "e28t8_summary.csv", index=False)
    T, Cb = pd.DataFrame(tests), pd.DataFrame(calib)
    T.to_csv(OUT / "e28t8_tests.csv", index=False)
    Cb.to_csv(OUT / "e28t8_calib.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print(T.round(4).to_string(index=False))
    print(Cb.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
