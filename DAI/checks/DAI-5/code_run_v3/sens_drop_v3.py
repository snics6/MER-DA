"""gate-code 第 3 版（DAI-5）：sensitivity の「1 回のテストが 30 本未満か感情が欠ける分割を外す」規則を，try_stats_v3.py が作った偽の置き場で確かめる。
部分集合（keep）を，分割 0・1 のテストから 60 本ずつ，分割 2 から 10 本だけ選んで作る（分割 3・4 は 0 本）。外すべき分割は 2・3・4（と，60 本で感情が欠ければその分割）。
使い方：CUDA_VISIBLE_DEVICES= ./.venv/bin/python DAI/checks/DAI-5/code_run_v3/sens_drop_v3.py <偽の根>
"""
import json
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
fin, fout, froot = fake / "in", fake / "out5", fake / "root"
y_of, cl_of = ST.label_maps("IEMOCAP")
rng = np.random.default_rng(3)
keep, expect = [], set()
for f, n in ((0, 60), (1, 60), (2, 10), (3, 0), (4, 0)):
    sid = np.load(fout / "eval" / "units" / f"AFEW_ALL_f{f}_r0_p5.npz")["sample_test"].astype(str)
    pick = rng.choice(sid, n, replace=False) if n else np.array([], str)
    keep += list(pick)
    y = y_of(pick) if n else np.array([], int)
    if n < ST.MIN_TEST or (np.bincount(y, minlength=4) == 0).any():
        expect.add(f)
sub = pd.DataFrame(dict(sample_id=keep, corpus="AFEW", subset="face_compete", value="single"))
sub.to_csv(froot / "DAI" / "frozen" / "subsets.csv", index=False)
S5.ROOT = froot
S5.SUBSETS = {"AFEW": [("face_single", "face_compete", "single", ("V",))]}
with open(D5.FROZEN1, encoding="utf-8") as fh:
    fz1 = json.load(fh)
fz1["AFEW"]["n_seeds_eval"] = 1
sb = {0: 0.0, 1: 2.5, 5: 5.0, 10: 0.0, 25: 5.0}
units = [(f, 0) for f in range(DD.N_FOLDS)]
rows = S5.sensitivity(fout, fin, "AFEW", units, y_of, cl_of, sb, fz1["AFEW"], 15.0, np.random.default_rng(0), 1.0)
d = pd.DataFrame(rows)
print("外すべき分割", sorted(expect), "台本の folds_dropped", d.folds_dropped.unique().tolist(), "n_units", d.n_units.unique().tolist(),
      "effect の NaN", int(d.effect.isna().sum()) if "effect" in d else "列なし")
