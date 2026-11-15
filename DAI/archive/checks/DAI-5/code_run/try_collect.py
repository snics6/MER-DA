"""gate-code（DAI-5）：dai5_stats.collect（extras=True，check_rows=True）を仮のターゲットの回の試走の 1 単位で流し，
評価の集計で使う行の名前がすべて作れること，判定から作った UAR と行の uar の食い違い，evaluate の重み・採点の表の鍵の形を確かめる。
成績の値は出さない。使い方：./.venv/bin/python DAI/checks/DAI-5/code_run/try_collect.py <試走の置き場> T c f r
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai5 as D5  # noqa: E402
import dai5_stats as S5  # noqa: E402
import dai_data as DD  # noqa: E402

out, T, c, f, r = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
fz1 = json.load(open(D5.FROZEN1))
y_of, cl_of = ST.label_maps(c)
for sel in (0.0, 2.5, 20.0):
    sb = {K: (0.0 if K == 0 else sel) for K in dai1.KS_ALL}
    ss = {(s, K): (0.0 if K == 0 else 5.0) for s in DD.STUDENTS for K in dai1.KS_ALL}
    cells, mism, total = S5.collect(out, D5.IN_DIR, "pseudo", T, c, [(f, r)], y_of, cl_of, sb, fz1[T], extras=True, check_rows=True,
                                    sel_student=ss)
    names = sorted({k[2] for k in cells})
    print(f"beta={sel}: cells {len(cells)}, row names {len(names)}: {names[:8]} ... {names[-4:]}, mismatch {mism}/{total}")
# 対の条件：joint の判定から作った UAR と実験 1 の p3 の行の uar（pairing と同じ式．仮のターゲットの回の単位で）
import pandas as pd  # noqa: E402
name = f"{T}_{c}_f{f}_r{r}"
up = ST.UnitPreds(D5.IN_DIR / "pseudo" / "units" / name)
r3 = pd.read_csv(D5.IN_DIR / "pseudo" / "units" / f"{name}_p3.csv", low_memory=False)
y = y_of(up.sid)
nb = 0
for s in DD.STUDENTS:
    for K in dai1.KS_ALL:
        jj = D5.joint_sel(fz1[T], s, K)
        mine = dai1.S.uar(y, up.p3item(s, K, method="joint", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"]).astype(int)) * 100
        q = r3[(r3.stage == 3) & (r3.student == s) & (r3.K == K) & (r3.method == "joint") & (r3.teacher == jj["teacher"])]
        q = q[np.isclose(q.rho.astype(float), float(jj["rho"])) & np.isclose(q.a.astype(float), float(jj["a"]))]
        nb += int(len(q) != 1 or abs(float(q.uar.iloc[0]) - mine) > 1e-6)
print("joint pairing mismatch", nb, "/ 20")
# evaluate の重み・採点の鍵（json から読んだ β の float の文字列）
p5 = D5.P5(out / "pseudo" / "units" / name)
for b in json.loads(json.dumps(list(D5.BETA_GRID))):
    for K in DD.KS:
        assert f"p5_K{K}_w_b{b}" in p5.z.files, (K, b)
w0 = p5.z["p5_K5_w_b0.0"]
print("beta=0 weights equal:", bool(np.allclose(w0, w0[0])), "argmax at beta=0 gives", p5.z["p5_src_names"][int(np.argmax(w0))])
print("scores keys", [k for k in p5.z.files if k.endswith("_scores")])
