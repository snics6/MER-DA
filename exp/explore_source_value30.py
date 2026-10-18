"""【探索的】E32 の種類別の表（`e32_kinds.csv`）を反復 0〜29 で作り直す（8 種類．新しい学習なし．CPU）。

`docs/exploratory-analyses.md`「言葉系の反復の回数を主な種類にそろえる」の E32 の分。`e32_kinds.csv` は手法の値が反復 0〜9，
学習曲線が交差検証の地図のターゲットのみ（言葉系は反復 0〜4）だった。ここでは次の 2 つを取り替え，読み方
（`explore_source_value.kind_rows`：種類で平均した UAR が，ターゲットで平均した学習曲線で何本に当たるか）は変えない。

- 手法の値（P1_def，P3_def，P4_def，B4，H3，O1_meta）とターゲットのみの UAR（uar_B1）：E31 の段 3 の表
  （主な種類は `e31_finals30.csv`，言葉系は `explore_compare_text30.py` の `e31_finals30_text.csv`．どちらも反復 0〜29）
- 学習曲線：C を固定した地図（既定 `results/fixc_map/<kind>/`）のターゲットのみの行の反復 0〜29，K ∈ {1, 5, 10, 25, 50}
  （ターゲットのみの行は 2 つの地図で同じ）

    ./.venv/bin/python exp/explore_source_value30.py [--finals …] [--map-root …] [--out …] [--require-full]

`--require-full`：手法の値か学習曲線の反復が 30 に足りなければ，何も書かずに終了コード 1 で止まる（本番用）。
出力：`results/explore_relation/e32_kinds30.csv`（`e32_kinds.csv` と同じ列に，使った反復の数の最小
`reps_methods`・`reps_curve` を足したもの）。`e32_kinds.csv` は変えない。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
import explore_compare as EC  # noqa: E402
import explore_source_value as SV  # noqa: E402

REL = ROOT / "results" / "explore_relation"
REPS30 = list(range(30))


def curves30(kind: str, targets, map_root: Path, reps=REPS30) -> tuple[dict, int]:
    """C を固定した地図のターゲットのみの学習曲線（`explore_source_value.curves` と同じ形）と，反復の数の最小。"""
    rows = []
    for t in targets:
        d = pd.read_csv(map_root / kind / f"{t}.csv", comment="#", dtype={"K": str, "source": str},
                        usecols=["target", "method", "K", "rep", "uar"])
        rows.append(d[(d.method == "tgt_only") & d.rep.isin(reps)])
    m = pd.concat(rows, ignore_index=True)
    m["uar"] = 100 * m.uar
    g = m.groupby(["target", "K"]).agg(u=("uar", "mean"), n=("rep", "nunique"))
    cur = {t: np.maximum.accumulate(np.array([g.u[(t, str(k))] for k in SV.KS_CURVE])) for t in targets}
    n = int(min(g.n[(t, str(k))] for t in targets for k in SV.KS_CURVE))
    return cur, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finals", nargs="+", type=Path,
                    default=[REL / "e31_finals30.csv", REL / "e31_finals30_text.csv"])
    ap.add_argument("--map-root", type=Path, default=ROOT / "results" / "fixc_map")
    ap.add_argument("--out", type=Path, default=REL / "e32_kinds30.csv")
    ap.add_argument("--require-full", action="store_true")
    args = ap.parse_args()
    F = pd.concat([pd.read_csv(p, dtype={"K": str}) for p in args.finals], ignore_index=True)
    F = F[F.K.isin(SV.KS)]
    rows, short = [], []
    for kind in [k for k in EC.KINDS if k in set(F.kind)]:
        P = F[F.kind == kind]
        ncols = [c for c in P.columns if c.startswith("n_")]
        n_m = int(P[~P.same][ncols].min().min())
        cur, n_c = curves30(kind, sorted(P.target.unique()), args.map_root)
        for r in SV.kind_rows(kind, P, cur):
            rows.append({**r, "reps_methods": n_m, "reps_curve": n_c})
        print(f"  {kind}：手法の反復 {n_m}，学習曲線の反復 {n_c}", flush=True)
        if min(n_m, n_c) < len(REPS30):
            short.append(kind)
    if short and args.require_full:
        print(f"反復が 30 にそろっていない種類：{short}．何も書かずに止める", flush=True)
        return 1
    if short:
        print(f"注意：反復が 30 にそろっていない種類がある（{short}）．途中の確かめとしてだけ使う", flush=True)
    Kt = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Kt.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(Kt.pivot_table(index=["kind", "method"], columns="K", values="n", sort=False).reindex(columns=SV.KS).round(1).to_string())
    print(f"書いた：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
