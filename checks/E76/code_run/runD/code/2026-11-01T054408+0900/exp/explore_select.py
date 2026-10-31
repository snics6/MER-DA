"""【探索的】E33 の解析：K 本の選び方ごとの UAR，得の中身（位置と向き），選び方の価値，予言 S1〜S3。新しい学習なし。

規則は `docs/exploratory-analyses.md` E33。入力は `results/select_<kind>/`（`exp/select_kshot.py`），random の学習曲線は E31 の P4
（`results/compare_<kind>/<ターゲット>[_k50all].csv` の HT_ALL，cen = 1，C′ = 0.01，C_b = 8），得の中身の基準はソースのみの下書きの
点数（`results/multisource_scores_<kind>/` の `s_all`．E31 の得の中身と同じ）。単位はターゲット（反復 0〜9 の平均）。

    ./.venv/bin/python exp/explore_select.py [--in-root results]

出力：`results/explore_relation/e33_targets.csv`（種類 × ターゲット × K × 選び方），`e33_tests.csv`，`e33_summary.csv`，`e33_diag.csv`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import explore_compare as EC  # noqa: E402
from explore_compare_mech import split_one  # noqa: E402

MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
STRATS = ["random", "pad_low", "pad_mid", "pad_high", "easy", "hard", "committee"]
KS = ["1", "5", "10", "25"]
CURVE_K = [1, 5, 10, 25, 50]
OUT = ROOT / "results" / "explore_relation"


def p4_curve(kind: str, t: str, reps) -> dict:
    fs = [ROOT / "results" / f"compare_{kind}" / f"{t}.csv", ROOT / "results" / f"compare_{kind}" / f"{t}_k50all.csv"]
    c = pd.concat([pd.read_csv(f, dtype={"K": str, "cb": str}) for f in fs if f.exists()])
    c = c[(c.method == "HT_ALL") & (c.cen == 1) & (c.cp == 0.01) & (c.cb == "8") & c.rep.isin(reps)]
    return (c.groupby("K").uar.mean() * 100).to_dict()


def k_equiv(curve: dict, u: float) -> float:
    ks = [k for k in CURVE_K if str(k) in curve]
    us = np.maximum.accumulate([curve[str(k)] for k in ks])
    if u > us[-1]:
        return np.inf
    if u < us[0]:
        return np.nan
    return float(np.exp(np.interp(u, us, np.log(ks))))


def load_kind(kind: str, in_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = in_root / f"select_{kind}"
    fs = sorted(f for f in d.glob("*.csv") if not f.stem.endswith("_sel")) if d.exists() else []
    if not fs:
        return pd.DataFrame(), pd.DataFrame()
    rows, diag = [], []
    for f in fs:
        t = f.stem
        a = pd.read_csv(f, dtype={"K": str})
        a = a[a.rep.isin(EC.REPS)]
        z = np.load(f.with_suffix(".npz"))
        zm = np.load(ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz")
        curve = p4_curve(kind, t, sorted(a.rep.unique()))     # 選び方の結果と同じ反復で比べる
        mech = []
        for r in sorted(a.rep.unique()):
            if not np.array_equal(z[f"test_idx_r{r}"], zm[f"test_idx_r{r}"]):
                raise ValueError(f"テストの行が合わない: {kind} {t} r{r}")
            y = z[f"y_test_r{r}"]
            Ls = zm[f"s_all_r{r}"].astype(np.float64)
            for st in STRATS:
                for K in KS:
                    key = f"{st}_K{K}_r{r}"
                    if key in z.files:
                        pos, dirc = split_one(Ls, z[key].astype(np.float64), y)
                        mech.append({"strategy": st, "K": K, "rep": r, "pos": pos, "dir": dirc})
        Mg = pd.DataFrame(mech).groupby(["strategy", "K"])[["pos", "dir"]].mean()
        U = a.groupby(["source", "K"]).uar.mean() * 100
        for (st, K), u in U.items():
            kp = k_equiv(curve, u)
            rows.append({"kind": kind, "target": t, "strategy": st, "K": K, "uar": u, "P4_E31": curve.get(K, np.nan),
                         "pos": Mg.loc[(st, K), "pos"], "dir": Mg.loc[(st, K), "dir"],
                         "k_equiv": kp, "value_ratio": kp / int(K)})
        s = pd.read_csv(d / f"{t}_sel.csv", dtype={"K": str})
        s = s[s.rep.isin(EC.REPS)]
        g1 = s[s.emotion == -1].groupby(["strategy", "K"])[["draft_right", "p_true", "disagree"]].mean()
        g2 = s[s.emotion >= 0].groupby(["strategy", "K"])[["pad", "cluster_size", "n_clusters"]].mean()
        diag.append(g1.join(g2, how="left").reset_index().assign(kind=kind, target=t))
    return pd.DataFrame(rows), pd.concat(diag, ignore_index=True)


def tests(T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    W = T.pivot_table(index=["kind", "target", "K"], columns="strategy", values=["uar", "pos", "dir"])
    summ, out = [], []
    for (kind, K), g in W.groupby(level=["kind", "K"]):
        row = {"kind": kind, "K": K, "targets": len(g),
               "random_vs_E31": float((T[(T.kind == kind) & (T.K == K) & (T.strategy == "random")].uar
                                       - T[(T.kind == kind) & (T.K == K) & (T.strategy == "random")].P4_E31).abs().max())}
        for st in STRATS[1:]:
            if ("uar", st) not in g:
                continue
            row[f"{st}-random"] = float((g["uar"][st] - g["uar"]["random"]).mean())
            row[f"{st}_wins"] = int((g["uar"][st] > g["uar"]["random"]).sum())
            row[f"{st}_pos-random"] = float((g["pos"][st] - g["pos"]["random"]).mean())
            row[f"{st}_dir-random"] = float((g["dir"][st] - g["dir"]["random"]).mean())
        summ.append(row)
        if K in ("5", "10"):
            bound = 100 * 0.121 / (2 * int(K))
            pos_gap = max(row.get(f"{st}_pos-random", -np.inf) for st in STRATS[1:])
            mu = {st: float(g["uar"][st].mean()) for st in STRATS if ("uar", st) in g}
            out.append({"kind": kind, "K": K,
                        "S1_easy-random": row.get("easy-random"), "S1": row.get("easy-random", np.nan) < 0,
                        "S2_low_mid_high": f"{mu.get('pad_low', np.nan):.2f} < {mu.get('pad_mid', np.nan):.2f} < {mu.get('pad_high', np.nan):.2f}",
                        "S2": mu.get("pad_low", np.nan) < mu.get("pad_mid", np.nan) < mu.get("pad_high", np.nan),
                        "S3_max_pos-random": pos_gap, "S3_bound": bound, "S3": pos_gap <= bound})
    return pd.DataFrame(summ), pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-root", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    parts = [load_kind(k, args.in_root) for k in MAIN]
    T = pd.concat([p[0] for p in parts], ignore_index=True)
    D = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    S, R = tests(T)
    T.to_csv(OUT / "e33_targets.csv", index=False)
    S.to_csv(OUT / "e33_summary.csv", index=False)
    R.to_csv(OUT / "e33_tests.csv", index=False)
    D.to_csv(OUT / "e33_diag.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(2).T.to_string())
    print(R.round(2).to_string(index=False))
    print(D.groupby(["kind", "strategy", "K"])[["draft_right", "p_true", "disagree", "pad"]].mean().round(3).to_string())
    print(T.groupby(["kind", "strategy", "K"]).value_ratio.median().unstack().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
