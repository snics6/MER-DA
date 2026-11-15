"""【探索的】E33 の追加の解析：勾配を合わせる選び方（herding）と random の比べ（G1・G2），全選び方のずれの量と損（G3）。新しい学習なし。

規則は `docs/exploratory-analyses.md` E33「勾配を合わせる選び方」。入力は `results/select_herd_<kind>/`（`exp/select_herding.py`），
E33 の 7 通りの UAR は `results/select_<kind>/`，得の中身の基準はソースのみの下書きの点数（`multisource_scores` の `s_all`）。
単位はターゲット（反復 0〜9 の平均）。

    ./.venv/bin/python exp/explore_herding.py

出力：`results/explore_relation/e33h_targets.csv`（種類 × ターゲット × K），`e33h_g3.csv`（選び方 × ターゲット × K），`e33h_tests.csv`
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
import explore_compare as EC  # noqa: E402
from explore_compare_mech import split_one  # noqa: E402
from explore_select import k_equiv, p4_curve  # noqa: E402

MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
KS = ["1", "5", "10", "25"]
OTHERS = ["pad_low", "pad_mid", "pad_high", "easy", "hard", "committee"]
OUT = ROOT / "results" / "explore_relation"


def load_kind(kind: str):
    d = ROOT / "results" / f"select_herd_{kind}"
    fs = sorted(f for f in d.glob("*.csv") if not f.stem.endswith("_mis")) if d.exists() else []
    rows, g3 = [], []
    for f in fs:
        t = f.stem
        h = pd.read_csv(f, dtype={"K": str})
        h = h[h.rep.isin(EC.REPS)]
        reps = sorted(h.rep.unique())
        e = pd.read_csv(ROOT / "results" / f"select_{kind}" / f"{t}.csv", dtype={"K": str})
        e = e[e.rep.isin(reps)]
        z = np.load(f.with_suffix(".npz"))
        zm = np.load(ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz")
        mech = []
        for r in reps:
            if not np.array_equal(z[f"test_idx_r{r}"], zm[f"test_idx_r{r}"]):
                raise ValueError(f"テストの行が合わない: {kind} {t} r{r}")
            y, Ls = z[f"y_test_r{r}"], zm[f"s_all_r{r}"].astype(np.float64)
            for st in ("herding", "random"):
                for K in KS:
                    pos, dirc = split_one(Ls, z[f"{st}_K{K}_r{r}"].astype(np.float64), y)
                    mech.append({"strategy": st, "K": K, "pos": pos, "dir": dirc})
        Mg = pd.DataFrame(mech).groupby(["strategy", "K"])[["pos", "dir"]].mean()
        U = h.groupby(["source", "K"]).uar.mean() * 100
        UE = e.groupby(["source", "K"]).uar.mean() * 100
        mis = pd.read_csv(d / f"{t}_mis.csv", dtype={"K": str})
        mis = mis[mis.rep.isin(reps)].groupby(["strategy", "K"])[["mismatch", "residual"]].mean()
        curve = p4_curve(kind, t, reps)
        for K in KS:
            kp = k_equiv(curve, U[("herding", K)])
            rows.append({"kind": kind, "target": t, "K": K, "herding": U[("herding", K)], "random": U[("random", K)],
                         "random_E33": UE[("random", K)], "pos_herding": Mg.loc[("herding", K), "pos"],
                         "pos_random": Mg.loc[("random", K), "pos"], "dir_herding": Mg.loc[("herding", K), "dir"],
                         "dir_random": Mg.loc[("random", K), "dir"], "mis_herding": mis.loc[("herding", K), "mismatch"],
                         "mis_random": mis.loc[("random", K), "mismatch"], "value_ratio": kp / int(K)})
            for st in OTHERS + ["herding"]:
                u = U[("herding", K)] if st == "herding" else UE.get((st, K), np.nan)
                g3.append({"kind": kind, "target": t, "K": K, "strategy": st, "mismatch": mis.loc[(st, K), "mismatch"],
                           "residual": mis.loc[(st, K), "residual"], "diff": u - UE[("random", K)]})
    return pd.DataFrame(rows), pd.DataFrame(g3)


def tests(T: pd.DataFrame, G: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kind, g in T.groupby("kind"):
        row = {"kind": kind, "targets": g.target.nunique(), "random_check": float((g.random - g.random_E33).abs().max())}
        for K in KS:
            h = g[g.K == K]
            row[f"herd-rand_K{K}"] = float((h.herding - h.random).mean())
            row[f"wins_K{K}"] = int((h.herding > h.random).sum())
            row[f"dpos_K{K}"] = float((h.pos_herding - h.pos_random).mean())
            row[f"ddir_K{K}"] = float((h.dir_herding - h.dir_random).mean())
            row[f"value_ratio_K{K}"] = float(h.value_ratio.median())
        row["G1"] = all(row[f"herd-rand_K{K}"] > 0 for K in ("1", "5", "10"))
        row["G2"] = row["dpos_K1"] > row["ddir_K1"] and all(row[f"ddir_K{K}"] > row[f"dpos_K{K}"] for K in ("10", "25"))
        x = G[(G.kind == kind) & G.strategy.isin(OTHERS)]
        rho = stats.spearmanr(x.mismatch, x["diff"])
        row["G3_rho"], row["G3_p"], row["G3_n"] = float(rho.statistic), float(rho.pvalue), len(x)
        row["G3"] = rho.statistic < 0 and rho.pvalue < 0.05
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    parts = [load_kind(k) for k in MAIN]
    T = pd.concat([p[0] for p in parts], ignore_index=True)
    G = pd.concat([p[1] for p in parts], ignore_index=True)
    R = tests(T, G)
    T.to_csv(OUT / "e33h_targets.csv", index=False)
    G.to_csv(OUT / "e33h_g3.csv", index=False)
    R.to_csv(OUT / "e33h_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).T.to_string())
    print(G.groupby(["kind", "strategy"])[["mismatch", "diff"]].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
