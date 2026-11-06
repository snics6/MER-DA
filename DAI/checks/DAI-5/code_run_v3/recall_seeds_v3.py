"""gate-code 第 3 版（DAI-5）：12bfa1d の recall_rows（混同行列を種の数で割る）を，種が 2 つの偽の評価の回で確かめる。
try_stats_v3.py が作った偽の置き場の f{f}_r0 を f{f}_r1 に写し（判定は同じ），種 1 つと種 2 つで混同行列・再現率・n_test_total が同じになるかを見る。
使い方：CUDA_VISIBLE_DEVICES= ./.venv/bin/python DAI/checks/DAI-5/code_run_v3/recall_seeds_v3.py <偽の根>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1_stats as ST  # noqa: E402
import dai5 as D5  # noqa: E402
import dai5_stats as S5  # noqa: E402
import dai_data as DD  # noqa: E402

fake = Path(sys.argv[1])
fin, fout = fake / "in", fake / "out5"
for f in range(DD.N_FOLDS):
    for base in (fin / "eval" / "units", fout / "eval" / "units"):
        for p in base.glob(f"AFEW_ALL_f{f}_r0_*"):
            shutil.copy(p, base / p.name.replace("_r0_", "_r1_"))
y_of, _ = ST.label_maps("IEMOCAP")
with open(D5.FROZEN1, encoding="utf-8") as fh:
    fz1T = json.load(fh)["AFEW"]
DD.others = lambda T: ["MOSEI", "MELD"]
sb = {0: 0.0, 1: 2.5, 5: 5.0, 10: 0.0, 25: 5.0}
u1 = [(f, 0) for f in range(DD.N_FOLDS)]
u2 = [(f, r) for f in range(DD.N_FOLDS) for r in (0, 1)]
a = [pd.DataFrame(x) for x in S5.recall_rows(fout, fin, "AFEW", u1, y_of, sb, fz1T)]
b = [pd.DataFrame(x) for x in S5.recall_rows(fout, fin, "AFEW", u2, y_of, sb, fz1T)]
ntest = sum(len(np.load(fout / "eval" / "units" / f"AFEW_ALL_f{f}_r0_p5.npz")["sample_test"]) for f in range(DD.N_FOLDS))
for nm, x, y in zip(("recall", "confusion", "src_recall"), a, b):
    num = [c for c in x.columns if c.startswith(("cm_", "recall", "n_test_total", "n_", "top_", "median_")) and c not in ("n_seeds", "n_units")]
    print(nm, "行", len(x), len(y), "種 1 と種 2 の差の最大", float(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float)).max()),
          "n_seeds", sorted(y.n_seeds.unique().tolist()) if "n_seeds" in y else "-",
          "n_test_total", sorted(y.n_test_total.unique().tolist()) if "n_test_total" in y else "-", "テストの総数", ntest)
