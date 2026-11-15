"""【探索的】E37 の下見（CPU．新しい学習なし）：下書きの 4 つの感情の点数を混ぜ直すだけで，向きの損をどれだけ取り戻せるか。

下書き（全ソースの合併．`results/multisource_scores_<kind>/` の `s_all_r<r>`）のテストの点数 L（4 列）について，感情の対 j = (k, l) ごとに
- d_s：下書きの点数の差 L_k − L_l の離れ方（E32 の 5 節と同じ `d_o`）
- d_span：L の 4 列の線形結合で最良の向き（テストの発話を話者で 2 つに分け，片方の LDA（4 × 4 の共通の共分散）で向きを決めて，もう片方で測り，入れ替えて平均）．混ぜ直しで届く上限の目安
- d_spanK：同じ混ぜ直しを，片方から感情ごとに K 本だけ引いて決めたもの（共分散は K 本の 4 × 4 を下書きの向きに縮める．5 回の平均）
- d_h：P4（E31 の既定）の K 本の仮説転移の点数（`results/compare_<kind>/` の `p4_ALL_K<K>_r<r>`）の離れ方
- D：E13 の own_d
を反復 0〜9 で平均し，「取り戻した割合」(d − d_s)/(D − d_s) を出す。

    ./.venv/bin/python exp/explore_span_room.py

出力：`results/explore_relation/e37pre_targets.csv`，`e37pre_summary.csv`
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
import explore_source_value as SV  # noqa: E402
from data import load_any  # noqa: E402
from explore_synergy import PAIRS, d_o  # noqa: E402
from split import speaker_folds  # noqa: E402

KS = (5, 10)
GAMMA = 1.0          # K 本の共分散を単位行列（下書きの点数の尺度）へ縮める強さ


def sep(z: np.ndarray, y: np.ndarray, k: int, l: int) -> float:
    a, b = z[y == k], z[y == l]
    return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))


def lda_dirs(L: np.ndarray, y: np.ndarray, shrink: float = 0.0) -> list[np.ndarray]:
    m = np.stack([L[y == k].mean(0) for k in range(4)])
    R = L - m[y]
    S = R.T @ R / max(len(y) - 4, 1)
    S = (1 - shrink) * S + shrink * np.trace(S) / 4 * np.eye(4) if shrink else S + 1e-9 * np.eye(4)
    return [np.linalg.solve(S, m[k] - m[l]) for k, l in PAIRS]


def run(kind: str) -> list[dict]:
    e13 = SV.load_e13(kind)
    rows = []
    for f in sorted((ROOT / "results" / f"multisource_scores_{kind}").glob("*.npz")):
        t = f.stem
        zm, zc = np.load(f), np.load(ROOT / "results" / f"compare_{kind}" / f"{t}.npz")
        own = e13[e13.target == t].groupby("pair").own_d.mean().reindex(SV.PAIR_NAMES).to_numpy()
        df, _ = load_any(t, kind)
        sp = df["speaker_group"].astype(str).to_numpy()
        acc = {k: [] for k in ["d_s", "d_span"] + [f"d_spanK{K}" for K in KS] + [f"d_h{K}" for K in KS]}
        for r in EC.REPS:
            if f"s_all_r{r}" not in zm.files:
                continue
            idx, y = zm[f"test_idx_r{r}"], zm[f"y_test_r{r}"]
            if not np.array_equal(idx, zc[f"test_idx_r{r}"]):
                raise ValueError(f"テストの行が合わない: {kind} {t} r{r}")
            L = zm[f"s_all_r{r}"].astype(np.float64)
            L = L - L.mean(0)
            acc["d_s"].append([d for d, _ in d_o(L, y)])
            half = speaker_folds(pd.Series(sp[idx]), k=2, seed=r).to_numpy()
            A, B = half == 0, half == 1
            if min(np.bincount(y[A], minlength=4).min(), np.bincount(y[B], minlength=4).min()) < 2:
                continue
            ds = []
            for P, Q in ((A, B), (B, A)):
                U = lda_dirs(L[P], y[P])
                ds.append([sep(L[Q] @ u, y[Q], k, l) for u, (k, l) in zip(U, PAIRS)])
            acc["d_span"].append(np.mean(ds, axis=0))
            rng = np.random.default_rng(r)
            for K in KS:
                dk = []
                for _ in range(5):
                    for P, Q in ((A, B), (B, A)):
                        pick = np.concatenate([rng.choice(np.flatnonzero(P & (y == c)), min(K, int((P & (y == c)).sum())), replace=False)
                                               for c in range(4)])
                        U = lda_dirs(L[pick], y[pick], shrink=GAMMA / (1 + GAMMA))
                        dk.append([sep(L[Q] @ u, y[Q], k, l) for u, (k, l) in zip(U, PAIRS)])
                acc[f"d_spanK{K}"].append(np.mean(dk, axis=0))
                acc[f"d_h{K}"].append([d for d, _ in d_o(zc[f"p4_ALL_K{K}_r{r}"].astype(np.float64), y)])
        row = {"kind": kind, "target": t, "D": float(own.mean())}
        for k, v in acc.items():
            row[k] = float(np.mean(v, axis=0).mean()) if v else np.nan
        for k in [c for c in acc if c != "d_s"]:
            row[f"f_{k}"] = (row[k] - row["d_s"]) / (row["D"] - row["d_s"])
        rows.append(row)
        print(f"  {kind} {t}", flush=True)
    return rows


def main() -> int:
    D = pd.DataFrame([r for k in SV.MAIN for r in run(k)])
    S = D.drop(columns="target").groupby("kind").median()
    D.to_csv(SV.OUT / "e37pre_targets.csv", index=False)
    S.to_csv(SV.OUT / "e37pre_summary.csv")
    pd.set_option("display.width", 250)
    print(S.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
