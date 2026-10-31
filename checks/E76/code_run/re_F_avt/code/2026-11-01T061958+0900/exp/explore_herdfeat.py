"""【探索的】E33 の追加 2 の解析：特徴だけを合わせる herding（herding_feat）と random・herding の比べ（予言 F1）。新しい学習なし。

規則は `docs/exploratory-analyses.md` E33「E33 の追加 2」。入力は `results/select_herdfeat_<kind>/`（herding_feat と random），
herding の UAR と点数は `results/select_herd_<kind>/`，得の中身の基準は `multisource_scores` の `s_all`。単位はターゲット（反復 0〜9 の平均）。

    ./.venv/bin/python exp/explore_herdfeat.py

出力：`results/explore_relation/e33f_targets.csv`（種類 × ターゲット × K），`e33f_tests.csv`
"""

from __future__ import annotations

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
KS = ["1", "5", "10", "25"]
OUT = ROOT / "results" / "explore_relation"


def load_kind(kind: str) -> pd.DataFrame:
    d = ROOT / "results" / f"select_herdfeat_{kind}"
    fs = sorted(f for f in d.glob("*.csv") if not f.stem.endswith("_mis")) if d.exists() else []
    rows = []
    for f in fs:
        t = f.stem
        a = pd.read_csv(f, dtype={"K": str})
        a = a[a.rep.isin(EC.REPS)]
        reps = sorted(a.rep.unique())
        h = pd.read_csv(ROOT / "results" / f"select_herd_{kind}" / f"{t}.csv", dtype={"K": str})
        h = h[h.rep.isin(reps)]
        z = np.load(f.with_suffix(".npz"))
        zh = np.load(ROOT / "results" / f"select_herd_{kind}" / f"{t}.npz")
        zm = np.load(ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz")
        mech = []
        for r in reps:
            if not (np.array_equal(z[f"test_idx_r{r}"], zm[f"test_idx_r{r}"]) and np.array_equal(zh[f"test_idx_r{r}"], zm[f"test_idx_r{r}"])):
                raise ValueError(f"テストの行が合わない: {kind} {t} r{r}")
            y, Ls = z[f"y_test_r{r}"], zm[f"s_all_r{r}"].astype(np.float64)
            for st, zz in (("herding_feat", z), ("random", z), ("herding", zh)):
                for K in KS:
                    pos, dirc = split_one(Ls, zz[f"{st}_K{K}_r{r}"].astype(np.float64), y)
                    mech.append({"strategy": st, "K": K, "pos": pos, "dir": dirc})
        Mg = pd.DataFrame(mech).groupby(["strategy", "K"])[["pos", "dir"]].mean()
        U = pd.concat([a, h[h.source == "herding"]]).groupby(["source", "K"]).uar.mean() * 100
        mis = pd.read_csv(d / f"{t}_mis.csv", dtype={"K": str})
        mis = mis[mis.rep.isin(reps)].groupby(["strategy", "K"])[["mismatch", "mismatch_feat"]].mean()
        for K in KS:
            row = {"kind": kind, "target": t, "K": K,
                   "random_check": abs(U[("random", K)] - h[(h.source == "random") & (h.K == K)].uar.mean() * 100)}
            for st in ("random", "herding_feat", "herding"):
                row[st] = U[(st, K)]
                row[f"pos_{st}"], row[f"dir_{st}"] = Mg.loc[(st, K), "pos"], Mg.loc[(st, K), "dir"]
                row[f"mis_{st}"], row[f"misfeat_{st}"] = mis.loc[(st, K), "mismatch"], mis.loc[(st, K), "mismatch_feat"]
            rows.append(row)
    return pd.DataFrame(rows)


def tests(T: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kind, g in T.groupby("kind"):
        row = {"kind": kind, "targets": g.target.nunique(), "random_check": float(g.random_check.max())}
        for K in KS:
            h = g[g.K == K]
            for a, b in (("herding_feat", "random"), ("herding", "herding_feat")):
                row[f"{a}-{b}_K{K}"] = float((h[a] - h[b]).mean())
                row[f"wins_{a}>{b}_K{K}"] = int((h[a] > h[b]).sum())
            for st in ("random", "herding_feat", "herding"):
                row[f"mis_{st}_K{K}"] = float(h[f"mis_{st}"].mean())
            row[f"dpos_feat-rand_K{K}"] = float((h.pos_herding_feat - h.pos_random).mean())
            row[f"ddir_feat-rand_K{K}"] = float((h.dir_herding_feat - h.dir_random).mean())
        row["F1"] = all(row[f"herding_feat-random_K{K}"] > 0 and row[f"herding-herding_feat_K{K}"] > 0 for K in ("5", "10"))
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    T = pd.concat([load_kind(k) for k in MAIN], ignore_index=True)
    R = tests(T)
    T.to_csv(OUT / "e33f_targets.csv", index=False)
    R.to_csv(OUT / "e33f_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).T.to_string())
    print(T.groupby(["kind", "K"])[["random", "herding_feat", "herding"]].mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
