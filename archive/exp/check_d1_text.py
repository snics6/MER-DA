"""【確かめ】D1 の台本（`combine_drafts.py --check`）の pooled の行が，E31 の P4 の既定と一致するか（CPU．新しい学習なし）。

`docs/exploratory-analyses.md`「言葉系の反復の回数を主な種類にそろえる」の確かめ G4。`<combine-root>/combine_<kind>/<T>.csv` の
source=pooled（method=HT_AVG．全ソースを 1 つにまとめた P4 の下書きそのもので解いた仮説転移）の行と，
`<compare-root>/compare_<kind>/<T>.csv` の P4 の既定（method=HT_ALL，cen=1，C′=0.01，C_b=8）の行を，(反復, K) ごとに比べる。
同じ下書き・同じ K 本・同じ学習器なので UAR は同じになるはず（発話 1 本の当たり外れで UAR は 10⁻⁴ 以上動くので，
許す差 `--tol`（既定 10⁻⁹．UAR は 0〜1 の値）は計算の丸めだけを見逃す大きさ）。

    ./.venv/bin/python exp/check_d1_text.py [--kinds …] [--reps 0 … 9] [--tol 1e-9] [--combine-root results] [--out …]

比べる反復は `--reps`（既定 0〜9．E31 の段 1 の `<T>.csv` にある反復）。P4 の行のある (反復, K) に pooled の行が無い，
同じ (反復, K) の行が 2 つ以上ある，差が tol を超える，のどれかがあれば終了コード 1。
出力：画面にターゲットごとの行数・最大の差・一致しない行の数。`--out` を与えるとその表を CSV に書く。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEXT = ["text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text"]
KEY = ["rep", "K"]


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"K": str, "cb": str, "source": str})


def check_target(kind: str, t: str, combine: Path, compare: Path, reps, tol: float) -> dict:
    p4 = read(compare / f"{t}.csv")
    p4 = p4[(p4.method == "HT_ALL") & (p4.cen == 1) & (p4.cp == 0.01) & (p4.cb == "8") & p4.rep.isin(reps)]
    row = {"kind": kind, "target": t, "n_p4": len(p4), "n_pooled": 0, "missing": len(p4), "dup": 0,
           "n_mismatch": 0, "max_abs_diff": float("nan")}
    f = combine / f"{t}.csv"
    if not f.exists():
        return row
    c = read(f)
    c = c[(c.source == "pooled") & c.rep.isin(reps)]
    if len(c) and not ((c.method == "HT_AVG") & (c.cen == 1) & (c.cp == 0.01) & (c.cb == "8")).all():
        raise SystemExit(f"{kind} {t}：pooled の行の設定が P4 の既定と違う")
    row["n_pooled"] = len(c)
    row["dup"] = int(p4.duplicated(KEY).sum() + c.duplicated(KEY).sum())
    m = p4[KEY + ["uar"]].merge(c[KEY + ["uar"]], on=KEY, how="left", suffixes=("_p4", "_pooled"))
    row["missing"] = int(m.uar_pooled.isna().sum())
    d = (m.uar_p4 - m.uar_pooled).abs()
    row["n_mismatch"] = int((d > tol).sum())
    row["max_abs_diff"] = float(d.max()) if d.notna().any() else float("nan")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=TEXT)
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--combine-root", type=Path, default=ROOT / "results", help="combine_<kind>/ を置いた場所")
    ap.add_argument("--compare-root", type=Path, default=ROOT / "results", help="compare_<kind>/ を置いた場所")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = []
    for kind in args.kinds:
        compare = args.compare_root / f"compare_{kind}"
        targets = sorted(p.stem for p in compare.glob("*.csv") if "_" not in p.stem)    # 段 1 の <T>.csv だけ
        if not targets:
            raise SystemExit(f"{kind}：{compare} に段 1 の表が無い")
        rows += [check_target(kind, t, args.combine_root / f"combine_{kind}", compare, args.reps, args.tol) for t in targets]
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        R.to_csv(args.out, index=False)
    bad = R[(R.n_p4 == 0) | (R.missing > 0) | (R.dup > 0) | (R.n_mismatch > 0)]
    if len(bad):
        print(f"一致しない：{len(bad)} ターゲット（{', '.join(bad.kind + ' ' + bad.target)}）", flush=True)
        return 1
    print(f"一致した：{len(R)} ターゲット，{int(R.n_p4.sum())} 行，最大の差 {R.max_abs_diff.max():.3g}（許す差 {args.tol:g}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
