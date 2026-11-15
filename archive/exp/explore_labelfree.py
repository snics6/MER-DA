"""【探索的】E34 の解析：ラベル無しで選ぶ herding（herding_pred）と random_pred・random_true（E31 の P4）の比べ（予言 L1・L2）。

規則は `docs/exploratory-analyses.md` E34。入力は `results/select_lf_<kind>/`（`exp/select_labelfree.py`），ラベルを知っている
herding_feat は `results/select_herdfeat_<kind>/`，得の中身の基準は `multisource_scores` の `s_all`。単位はターゲット（反復 0〜9 の平均）。

    ./.venv/bin/python exp/explore_labelfree.py

出力：`results/explore_relation/e34_targets.csv`（種類 × ターゲット × K），`e34_tests.csv`，`e34_diag.csv`
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
from explore_select import k_equiv, p4_curve  # noqa: E402

MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
KS = ["1", "5", "10", "25"]
STRATS = ["herding_pred", "random_pred", "herding_all", "random_true"]
OUT = ROOT / "results" / "explore_relation"


def load_kind(kind: str):
    d = ROOT / "results" / f"select_lf_{kind}"
    fs = sorted(f for f in d.glob("*.csv") if not f.stem.endswith("_sel")) if d.exists() else []
    rows, diag = [], []
    for f in fs:
        t = f.stem
        a = pd.read_csv(f, dtype={"K": str})
        a = a[a.rep.isin(EC.REPS)]
        reps = sorted(a.rep.unique())
        z = np.load(f.with_suffix(".npz"))
        zm = np.load(ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz")
        mech = []
        for r in reps:
            if not np.array_equal(z[f"test_idx_r{r}"], zm[f"test_idx_r{r}"]):
                raise ValueError(f"テストの行が合わない: {kind} {t} r{r}")
            y, Ls = z[f"y_test_r{r}"], zm[f"s_all_r{r}"].astype(np.float64)
            for st in STRATS:
                for K in KS:
                    pos, dirc = split_one(Ls, z[f"{st}_K{K}_r{r}"].astype(np.float64), y)
                    mech.append({"strategy": st, "K": K, "pos": pos, "dir": dirc})
        Mg = pd.DataFrame(mech).groupby(["strategy", "K"])[["pos", "dir"]].mean()
        U = a.groupby(["source", "K"]).uar.mean() * 100
        hf = ROOT / "results" / f"select_herdfeat_{kind}" / f"{t}.csv"
        H = pd.read_csv(hf, dtype={"K": str}) if hf.exists() else None
        H = H[H.rep.isin(reps) & (H.source == "herding_feat")].groupby("K").uar.mean() * 100 if H is not None else {}
        curve = p4_curve(kind, t, reps)
        e31 = curve
        for K in KS:
            row = {"kind": kind, "target": t, "K": K, "random_check": abs(U[("random_true", K)] - e31.get(K, np.nan)),
                   "herding_feat": H.get(K, np.nan) if len(H) else np.nan,
                   "value_ratio": k_equiv(curve, U[("herding_pred", K)]) / int(K)}
            for st in STRATS:
                row[st] = U[(st, K)]
                row[f"pos_{st}"], row[f"dir_{st}"] = Mg.loc[(st, K), "pos"], Mg.loc[(st, K), "dir"]
            rows.append(row)
        s = pd.read_csv(d / f"{t}_sel.csv", dtype={"K": str})
        s = s[s.rep.isin(reps)]
        diag.append(s.groupby(["strategy", "K"])[["missing", "min_count", "max_count", "purity"]].mean().reset_index()
                    .assign(kind=kind, target=t))
    return pd.DataFrame(rows), pd.concat(diag, ignore_index=True) if diag else pd.DataFrame()


def tests(T: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kind, g in T.groupby("kind"):
        row = {"kind": kind, "targets": g.target.nunique(), "random_check": float(g.random_check.max())}
        for K in KS:
            h = g[g.K == K]
            for a, b in (("herding_pred", "random_pred"), ("herding_pred", "random_true"), ("herding_all", "random_pred"),
                         ("herding_feat", "herding_pred")):
                row[f"{a}-{b}_K{K}"] = float((h[a] - h[b]).mean())
                row[f"wins_{a}>{b}_K{K}"] = int((h[a] > h[b]).sum())
            row[f"dpos_hp-rp_K{K}"] = float((h.pos_herding_pred - h.pos_random_pred).mean())
            row[f"ddir_hp-rp_K{K}"] = float((h.dir_herding_pred - h.dir_random_pred).mean())
            row[f"value_ratio_K{K}"] = float(h.value_ratio.median())
        row["L1"] = all(row[f"herding_pred-random_pred_K{K}"] > 0 for K in ("1", "5", "10"))
        row["L2"] = all(row[f"herding_pred-random_true_K{K}"] > 0 for K in ("5", "10"))
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    parts = [load_kind(k) for k in MAIN]
    T = pd.concat([p[0] for p in parts], ignore_index=True)
    D = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    R = tests(T)
    T.to_csv(OUT / "e34_targets.csv", index=False)
    R.to_csv(OUT / "e34_tests.csv", index=False)
    D.to_csv(OUT / "e34_diag.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).T.to_string())
    print(T.groupby(["kind", "K"])[STRATS + ["herding_feat"]].mean().round(2).to_string())
    print(D.groupby(["kind", "strategy", "K"])[["missing", "min_count", "purity"]].mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
