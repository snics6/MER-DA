"""【探索的】E31 の段 3（最終候補の反復 0〜29）を言葉を含む 4 種類で作る。新しい学習なし（CPU）。

`docs/exploratory-analyses.md`「言葉系の反復の回数を主な種類にそろえる」の E31 の分。手法の値と設定の選び方は
`explore_compare.pair_table`・`explore_compare.finals` をそのまま使い（台本は変えない），地図の基準だけを
C を固定した地図（既定 `results/fixc_map/<kind>/`）の反復 0〜29 から取り直す。

- B1（ターゲットのみ）・B2（ソースのみ）：C を固定した地図の行．2 つの地図で違うのは合併の行だけで，この 2 つは
  交差検証の地図と同じ手順の同じ行である（言葉系の交差検証の地図は反復 0〜4 しかないので，30 回はこちらにしかない）
- B4（C を固定した合併）：同じ地図（`explore_compare.finals` と同じ値）
- B3（交差検証で C を選ぶ合併）：交差検証の地図にしか無い．30 回そろわない行では `B3` を空にし（30 回の比べから外れる），
  そろった反復だけの平均を `B3_cvmap`，反復の数を `n_B3_cvmap` に残す（言葉系は 5 回）．主な種類では B3 は 30 回そろう
- 30 回の比べ（`e31_main30<suffix>.csv`）は `explore_compare.main_comparisons` と同じ．B3 が空の比べ（n = 0）の行は書かない
  （Holm は空の p を数えないので，残りの行の値は B3 の行を書いても書かなくても同じ）

    ./.venv/bin/python exp/explore_compare_text30.py [--kinds …] [--out-dir …] [--suffix _text] [--require-full]

`--require-full`：使う値（P3_def，P1_def，H3，P4_def，P3_meta，P4_meta，O1_meta，B1，B2，B4）の反復が 1 つでも 30 に
足りなければ，何も書かずに終了コード 1 で止まる（本番用）。付けなければ，足りない数を表示して書く（途中の確かめ用）。

出力：`<out-dir>/e31_finals30<suffix>.csv`（`e31_finals30.csv` と同じ列に，反復の数 n_P4_meta・n_O1_meta・n_B1・n_B2・n_B4
と B3_cvmap・n_B3_cvmap を足したもの），`e31_main30<suffix>.csv`
"""

from __future__ import annotations

import argparse
import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
import explore_compare as EC  # noqa: E402

TEXT = EC.TEXT
REPS30 = list(range(30))
N_COLS = ["n_P3_def", "n_P1_def", "n_H3", "n_P4_def", "n_P3_meta", "n_P4_meta", "n_O1_meta", "n_B1", "n_B2", "n_B4"]


def map_baselines(kind: str, targets, map_root: Path, reps) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """C を固定した地図の B1・B2・B4（UAR の点の反復平均と反復の数）。"""
    rows = []
    for t in targets:
        d = pd.read_csv(map_root / kind / f"{t}.csv", comment="#", dtype={"K": str, "source": str},
                        usecols=["target", "source", "method", "K", "rep", "uar"])
        d["K"], d["source"] = d.K.fillna(""), d.source.fillna("")
        rows.append(d[d.rep.isin(reps)])
    m = pd.concat(rows, ignore_index=True)
    m["uar"] = 100 * m.uar

    def agg(x, by, name):
        return x.groupby(by).agg(**{name: ("uar", "mean"), f"n_{name}": ("rep", "nunique")}).reset_index()
    return (agg(m[m.method == "tgt_only"], ["target", "K"], "B1"),
            agg(m[m.method == "src_only"], ["target", "source"], "B2"),
            agg(m[m.method == "merged"], ["target", "source", "K"], "B4"))


def cv_b3(kind: str, targets, reps) -> pd.DataFrame:
    """交差検証の地図の合併（B3）の反復平均と反復の数。"""
    m = EC.load_map(kind, targets, reps)
    x = m[(m.src == "cv") & (m.method == "merged")]
    return x.groupby(["target", "source", "K"]).agg(B3_cvmap=("uar", "mean"), n_B3_cvmap=("rep", "nunique")).reset_index()


def meta_counts(kind: str, F: pd.DataFrame) -> tuple[list, list]:
    """P4_meta・O1_meta（`finals` は反復の数を残さない）の反復の数を，`finals` と同じ引き方で数える。"""
    d = EC.load_compare(kind, REPS30)
    ha = d[(d.method == "HT_ALL") & (d.cen == 1)].groupby(["target", "K", "cp", "cb"]).rep.nunique()
    o1 = d[d.method == "O1"].groupby(["target", "source", "K", "alpha"]).rep.nunique()
    n4, n1 = [], []
    for r in F.itertuples():
        cp4, cb4 = ast.literal_eval(r.P4_meta_set)
        n4.append(int(ha.loc[(r.target, r.K, cp4, cb4)]))
        n1.append(int(o1.loc[(r.target, r.source, r.K, float(r.O1_meta_set))]))
    return n4, n1


def finals30(kind: str, map_root: Path) -> pd.DataFrame:
    P = EC.pair_table(kind)                      # 設定の選び方（反復 0〜9）．段 1・2 と同じ
    F = EC.finals(P, kind)                       # 手法の値（反復 0〜29）．B1・B2・B3 は交差検証の地図，B4 は C を固定した地図
    targets = sorted(F.target.unique())
    b1, b2, b4 = map_baselines(kind, targets, map_root, REPS30)
    order = list(F.columns)
    F = F.drop(columns=["B1", "B2", "B4"])
    F = F.merge(b1, on=["target", "K"], how="left").merge(b2, on=["target", "source"], how="left")
    F = F.merge(b4, on=["target", "source", "K"], how="left")
    F = F.drop(columns=["B3"]).merge(cv_b3(kind, targets, REPS30), on=["target", "source", "K"], how="left")
    F["B3"] = np.where(F.n_B3_cvmap == len(REPS30), F.B3_cvmap, np.nan)
    F["n_P4_meta"], F["n_O1_meta"] = meta_counts(kind, F)
    extra = ["n_P4_meta", "n_O1_meta", "n_B1", "n_B2", "n_B4", "B3_cvmap", "n_B3_cvmap"]
    return F[order + extra]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=TEXT)
    ap.add_argument("--map-root", type=Path, default=ROOT / "results" / "fixc_map", help="C を固定した地図の根")
    ap.add_argument("--out-dir", type=Path, default=EC.OUT)
    ap.add_argument("--suffix", default="_text")
    ap.add_argument("--require-full", action="store_true", help="反復が 30 にそろわなければ書かずに止まる")
    args = ap.parse_args()
    F = pd.concat([finals30(k, args.map_root) for k in args.kinds], ignore_index=True)
    short = {c: (int(F[c].min()), int((F[c] < len(REPS30)).sum())) for c in N_COLS}
    bad = {c: v for c, v in short.items() if v[1]}
    print("反復の数（最小，30 未満の行）：", short, flush=True)
    if bad and args.require_full:
        print(f"反復が 30 にそろっていない：{bad}．何も書かずに止める", flush=True)
        return 1
    if bad:
        print(f"注意：反復が 30 にそろっていない行がある（{bad}）．途中の確かめとしてだけ使う", flush=True)
    with warnings.catch_warnings():              # B3 が空の比べ（n = 0）の平均の警告
        warnings.simplefilter("ignore", RuntimeWarning)
        M30 = EC.main_comparisons(F)
    M30 = M30[~((M30.b == "B3") & (M30.n == 0))]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    F.to_csv(args.out_dir / f"e31_finals30{args.suffix}.csv", index=False)
    M30.to_csv(args.out_dir / f"e31_main30{args.suffix}.csv", index=False)
    cols = ["B1", "B2", "B3", "B4", "H3", "P1_def", "P3_def", "P3_meta", "P4_def", "P4_meta", "O1_meta"]
    pd.set_option("display.width", 250)
    print(F[~F.same].groupby(["kind", "K"])[cols].mean().round(2).to_string())
    print(M30.round(3).to_string(index=False))
    print(f"書いた：{args.out_dir / f'e31_finals30{args.suffix}.csv'}，{args.out_dir / f'e31_main30{args.suffix}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
