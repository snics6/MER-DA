"""【探索的】E39 の下見 2（CPU．新しい学習なし）：ソースごとの分類器の向きを，感情の対ごとに組み合わせ直すとどこまで届くか。

ソースごとのソースのみの分類器のテストの点数（`results/merged_scores_<kind>/` の `s_<ソース>_r<r>`．E32 の 5 節と同じ）から，
感情の対 j = (k, l) ごとに各ソースの点数の差 z_sj = L_s,k − L_s,l を並べた M 次元の量（M はソースの数）を作る。
- d_equal：z をラベル無しの尺度でそろえて等しい重みで足した向き（E32 の 5 節の equal と同じ）
- d_span：M 次元の z の最良の線形結合（テストの発話を話者で 2 つに分け，片方の LDA で重みを決めてもう片方で測り，入れ替えて平均）
- d_spanK：同じ重みを片方から感情ごとに K 本だけで決める（M × M の共分散を単位行列に縮め，平均の差を等しい重みの向きへ縮める．5 回の平均）
- D：E13 の own_d
を反復 0〜9 で平均し，取り戻した割合 (d − d_equal)/(D − d_equal) を出す。

    ./.venv/bin/python exp/explore_source_span.py

出力：`results/explore_relation/e39pre2_targets.csv`，`e39pre2_summary.csv`
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
import explore_source_combine as SC  # noqa: E402
import explore_source_value as SV  # noqa: E402
from data import load_any  # noqa: E402
from explore_synergy import PAIRS  # noqa: E402
from split import speaker_folds  # noqa: E402

KS = (5, 10)


def sep(z, y, k, l):
    a, b = z[y == k], z[y == l]
    return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))


def lda_w(Zj, y, k, l, shrink=0.0, prior=None):
    """2 つの感情 k, l の発話だけで M 次元の LDA の重み。shrink で共分散を単位行列へ，prior（向き）へ平均の差を寄せる。"""
    a, b = Zj[y == k], Zj[y == l]
    S = (np.cov(a, rowvar=False) * (len(a) - 1) + np.cov(b, rowvar=False) * (len(b) - 1)) / max(len(a) + len(b) - 2, 1)
    S = np.atleast_2d(S)
    S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S)) + 1e-9 * np.eye(len(S))
    dm = a.mean(0) - b.mean(0)
    if prior is not None:
        dm = dm + prior * np.linalg.norm(dm) / np.linalg.norm(prior)
    return np.linalg.solve(S, dm)


def run(kind: str) -> list[dict]:
    e13 = SV.load_e13(kind)
    rows = []
    for f in sorted((ROOT / "results" / f"multisource_scores_{kind}").glob("*.npz")):
        t = f.stem
        zm, zs = np.load(f), np.load(ROOT / "results" / f"merged_scores_{kind}" / f"{t}.npz")
        srcs = [str(s) for s in zm["src_names"]]
        own = e13[e13.target == t].groupby("pair").own_d.mean().reindex(SV.PAIR_NAMES).to_numpy()
        df, _ = load_any(t, kind)
        sp = df["speaker_group"].astype(str).to_numpy()
        acc = {k: [] for k in ["d_equal", "d_span"] + [f"d_spanK{K}" for K in KS]}
        for r in EC.REPS:
            if f"s_all_r{r}" not in zm.files:
                continue
            idx, y = zm[f"test_idx_r{r}"], zm[f"y_test_r{r}"]
            Ls = [zs[f"s_{s}_r{r}"].astype(np.float64) for s in srcs]
            Ls = [L / SC.scale(L) for L in Ls]
            acc["d_equal"].append(SC.d_pairs(sum(Ls) / len(Ls), y))
            Z = [np.stack([L[:, k] - L[:, l] for L in Ls], 1) for k, l in PAIRS]      # 対ごとに (n, M)
            half = speaker_folds(pd.Series(sp[idx]), k=2, seed=r).to_numpy()
            A, B = half == 0, half == 1
            if min(np.bincount(y[A], minlength=4).min(), np.bincount(y[B], minlength=4).min()) < 3:
                continue
            ds = []
            for P, Q in ((A, B), (B, A)):
                ds.append([sep(Z[j][Q] @ lda_w(Z[j][P], y[P], k, l), y[Q], k, l) for j, (k, l) in enumerate(PAIRS)])
            acc["d_span"].append(np.mean(ds, axis=0))
            rng = np.random.default_rng(r)
            ones = np.ones(len(srcs))
            for K in KS:
                dk = []
                for _ in range(5):
                    for P, Q in ((A, B), (B, A)):
                        pick = np.concatenate([rng.choice(np.flatnonzero(P & (y == c)), min(K, int((P & (y == c)).sum())), replace=False)
                                               for c in range(4)])
                        dk.append([sep(Z[j][Q] @ lda_w(Z[j][pick], y[pick], k, l, shrink=0.5, prior=ones), y[Q], k, l)
                                   for j, (k, l) in enumerate(PAIRS)])
                acc[f"d_spanK{K}"].append(np.mean(dk, axis=0))
        row = {"kind": kind, "target": t, "D": float(own.mean()), "M": len(srcs)}
        for k, v in acc.items():
            row[k] = float(np.mean(v, axis=0).mean()) if v else np.nan
        for k in [c for c in acc if c != "d_equal"]:
            row[f"f_{k}"] = (row[k] - row["d_equal"]) / (row["D"] - row["d_equal"])
        rows.append(row)
        print(f"  {kind} {t}", flush=True)
    return rows


def main() -> int:
    D = pd.DataFrame([r for k in SV.MAIN for r in run(k)])
    S = D.drop(columns="target").groupby("kind").median()
    D.to_csv(SV.OUT / "e39pre2_targets.csv", index=False)
    S.to_csv(SV.OUT / "e39pre2_summary.csv")
    pd.set_option("display.width", 250)
    print(S.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
