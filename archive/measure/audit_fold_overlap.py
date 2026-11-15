"""テストの話者が反復の間でどれだけ重なるかを数える（`docs/seen-data.md` 3 節）。

地図と手法の比較は，反復 r のテストを `speaker_folds(…, seed=r)` の r mod 5 番の袋にする
（`docs/methods.md` 2.2 節）。袋は反復ごとに引き直すが，話者の多くは早い反復のうちに
一度はテストに出る。**まだ使っていない反復で確かめても，見ていない話者での確認には
ならない**ことを，群ごとの数で示す。

使い方：
    ./.venv/bin/python measure/audit_fold_overlap.py [--kind audio_wavlm] [--seen 10] [--reps 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_group  # noqa: E402
from split import speaker_folds  # noqa: E402

GROUPS = ["IEMOCAP-impro", "IEMOCAP-script", "MOSEI", "MELD", "AFEW", "RAVDESS", "CREMA-D"]
N_TEST_FOLDS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="audio_wavlm")
    ap.add_argument("--seen", type=int, default=10, help="見たとみなす反復の数（0 から）")
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()

    print("群,話者群の数,見た反復でテストに出た話者群の割合,残りの反復のテストの発話のうち未出の話者のものの割合,異なるテストの集合の数")
    for grp in GROUPS:
        df, _ = load_group(grp, a.kind)
        g = df["speaker_group"].reset_index(drop=True)
        tests = {r: set(g[speaker_folds(g, k=N_TEST_FOLDS, seed=r) == r % N_TEST_FOLDS])
                 for r in range(a.reps)}
        seen = set().union(*(tests[r] for r in range(a.seen)))
        new = [(~g[g.isin(tests[r])].isin(seen)).mean() for r in range(a.seen, a.reps)]
        distinct = len({frozenset(tests[r]) for r in range(a.reps)})
        print(f"{grp},{g.nunique()},{len(seen) / g.nunique():.3f},{sum(new) / len(new):.3f},{distinct}")


if __name__ == "__main__":
    main()
