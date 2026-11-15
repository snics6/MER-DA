"""gate-code 第 3 版（DAI-5）：select_stage の教師の検算の行の数の守り（6674496）を偽の置き場で確かめる（uar は乱数．本物の成績は使わない）。
1 単位の試走の p5.csv を 30 単位ぶんに写し，(0) 何も替えない，(1) 1 単位の _teacher_check を全部消す，(2) 1 行だけ消す，(3) 1 行を重ねる，
(4) 一致しない行（前からの守り），(5) srcw_tether の行の勾配を 1e-4 にする（select は pseudo の 4 版だけを見る），の 6 通りを流す。
使い方：CUDA_VISIBLE_DEVICES= ./.venv/bin/python DAI/checks/DAI-5/code_run_v3/try_select_v3.py <試走の置き場> <偽の置き場>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1  # noqa: E402
import dai5  # noqa: E402

src, fake = Path(sys.argv[1]), Path(sys.argv[2])
rng = np.random.default_rng(11)
with open(dai5.FROZEN1, encoding="utf-8") as fh:
    fz1 = json.load(fh)
one = sorted((src / "pseudo" / "units").glob("AFEW_*_p5.csv"))[0]
base = pd.read_csv(one, low_memory=False)
npz = Path(str(one).replace(".csv", ".npz"))
tc0 = base[base.method == "_teacher_check"]
print("試走の 1 単位の _teacher_check の行", len(tc0), "（K, 識別器）の組", tc0.groupby(["K"]).size().to_dict() if "K" in tc0 else None,
      "method ごとの行", base.method.value_counts().to_dict())
if fake.exists():
    shutil.rmtree(fake)
(fake / "pseudo" / "units").mkdir(parents=True)
for nm in fz1["AFEW"]["units"]:
    c, f, r = nm.split("_")[0], int(nm.split("_")[1][1:]), int(nm.split("_")[2][1:])
    d = base.copy()
    d["c"], d["fold"], d["seed"] = c, f, r
    s = d.method.isin(["srcw", "srcw_mod", "srcw_src", "srcw_top1"])
    d.loc[s, "uar"] = rng.normal(50, 2, s.sum()).round(1)
    d.to_csv(fake / "pseudo" / "units" / f"AFEW_{nm}_p5.csv", index=False)
    shutil.copy(npz, fake / "pseudo" / "units" / f"AFEW_{nm}_p5.npz")
dai1.git_commit = lambda: (str(base.commit.iloc[0]), False)


def mutate(case):
    f2 = Path(str(fake) + "_" + case)
    if f2.exists():
        shutil.rmtree(f2)
    shutil.copytree(fake, f2)
    fs = sorted((f2 / "pseudo" / "units").glob("*.csv"))
    d0 = pd.read_csv(fs[3], low_memory=False)
    if case == "teacher_check_unit_missing":
        d0 = d0[d0.method != "_teacher_check"]
    elif case == "teacher_check_one_row_missing":
        d0 = d0.drop(d0.index[d0.method == "_teacher_check"][7])
    elif case == "teacher_check_one_row_duplicated":
        d0 = pd.concat([d0, d0[d0.method == "_teacher_check"].iloc[[4]]])
    elif case == "teacher_check_false":
        ix = d0.index[d0.method == "_teacher_check"][7]; d0.loc[ix, "match"] = False
    elif case == "tether_grad":
        ix = d0.index[d0.method == "srcw_tether"]
        if len(ix):
            d0.loc[ix[0], "grad"] = 1e-4
    d0.to_csv(fs[3], index=False)
    try:
        dai5.select_stage(f2, ["AFEW"], fz1)
        print(f"{case:34s} -> 通った")
    except SystemExit as e:
        print(f"{case:34s} -> 止まる：{str(e)[:110]}")
    shutil.rmtree(f2)


for case in ("none", "teacher_check_unit_missing", "teacher_check_one_row_missing", "teacher_check_one_row_duplicated", "teacher_check_false", "tether_grad"):
    mutate(case)
