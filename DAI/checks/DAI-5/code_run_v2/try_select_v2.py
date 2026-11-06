"""gate-code 第 2 版（DAI-5）：select_stage を偽の置き場で流す（本物の成績は使わない．uar は乱数に置き換え）。
確かめること：(1) 軸を分けた版の β の選択（β = 0 の候補を srcw の行から取る）を独立に作り直して一致するか，
(2) 収束・行き詰まり・非有限・教師の検算の守りで止まるか，(3) pseudo_unit_sha256 に csv が入り，select の後に csv を替えると評価の段の守りが止めるか。
使い方：CUDA_VISIBLE_DEVICES= ./.venv/bin/python DAI/checks/DAI-5/code_run_v2/try_select_v2.py <試走の置き場> <偽の置き場>
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1  # noqa: E402
import dai5  # noqa: E402
import dai_data as DD  # noqa: E402

src, fake = Path(sys.argv[1]), Path(sys.argv[2])
rng = np.random.default_rng(11)
fz1 = json.load(open(dai5.FROZEN1))
one = sorted((src / "pseudo" / "units").glob("AFEW_*_p5.csv"))[0]
base = pd.read_csv(one, low_memory=False)
npz = Path(str(one).replace(".csv", ".npz"))
if fake.exists():
    shutil.rmtree(fake)
(fake / "pseudo" / "units").mkdir(parents=True)
for nm in fz1["AFEW"]["units"]:
    c, f, r = nm.split("_")[0], int(nm.split("_")[1][1:]), int(nm.split("_")[2][1:])
    d = base.copy()
    d["c"], d["fold"], d["seed"] = c, f, r
    s = d.method.isin(["srcw", "srcw_mod", "srcw_src", "srcw_top1"])
    d.loc[s, "uar"] = rng.normal(50, 2, s.sum()).round(int(rng.integers(0, 2)))
    d.to_csv(fake / "pseudo" / "units" / f"AFEW_{nm}_p5.csv", index=False)
    shutil.copy(npz, fake / "pseudo" / "units" / f"AFEW_{nm}_p5.npz")
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
orig_git = dai1.git_commit
dai1.git_commit = lambda: (str(base.commit.iloc[0]), False)
dai5.select_stage(fake, ["AFEW"], fz1)
fz = json.load(open(fake / "select" / "frozen.json"))
A = fz["AFEW"]
print("select_variants keys", sorted(A["select_variants"]))
shk = A["pseudo_unit_sha256"]
print("pseudo_unit_sha256: entries", len(shk), "npz", sum(k.endswith(".npz") for k in shk), "csv", sum(k.endswith(".csv") for k in shk))

# (1) 軸を分けた版の選択を独立に作り直す
raw = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted((fake / "pseudo" / "units").glob("AFEW_*_p5.csv"))])
G = list(dai5.BETA_GRID)
mm, n = 0, 0
for meth in ("srcw_mod", "srcw_src"):
    for K in (5, 25):
        dl = np.mean([fz1["AFEW"]["delta_sel"][f"{s}|{K}"] for s in DD.STUDENTS])
        v = raw[(raw.method == meth) & (raw.K == K)]
        b0 = raw[(raw.method == "srcw") & (raw.K == K) & (raw.beta.astype(float) == 0.0)]
        allr = pd.concat([v, b0])
        assert len(allr) == 30 * 4 * 5, len(allr)
        g = allr.groupby("beta").uar.mean()
        ref = [b for b in G if g[b] >= g.max() - dl][0]
        mm += int(ref != A["select_variants"][f"{meth}|{K}"]["beta"]); n += 1
        # β = 0 の値が srcw の β = 0 の平均と同じか
        mm += int(not np.isclose(g[0.0], raw[(raw.method == "srcw") & (raw.K == K) & (raw.beta == 0.0)].uar.mean())); n += 1
print("variant beta independent mismatch", mm, "/", n)


# (2) 守り
def mutate(case):
    f2 = Path(str(fake) + "_" + case)
    if f2.exists():
        shutil.rmtree(f2)
    shutil.copytree(fake, f2)
    shutil.rmtree(f2 / "select")
    fs = sorted((f2 / "pseudo" / "units").glob("*.csv"))
    d0 = pd.read_csv(fs[3], low_memory=False)
    if case == "grad_mod":
        ix = d0.index[d0.method == "srcw_mod"][5]; d0.loc[ix, "grad"] = 1e-4
    elif case == "stalled_top1":
        ix = d0.index[d0.method == "srcw_top1"][2]; d0.loc[ix, "stalled"] = True
    elif case == "finite_src":
        ix = d0.index[d0.method == "srcw_src"][0]; d0.loc[ix, "finite"] = False
    elif case == "grad_srcw":
        ix = d0.index[d0.method == "srcw"][10]; d0.loc[ix, "grad"] = 2e-5
    elif case == "teacher_check_false":
        ix = d0.index[d0.method == "_teacher_check"][7]; d0.loc[ix, "match"] = False
    elif case == "teacher_check_missing":
        d0 = d0[d0.method != "_teacher_check"]
    elif case == "variant_row_missing":
        d0 = d0.drop(d0.index[d0.method == "srcw_src"][3])
    d0.to_csv(fs[3], index=False)
    try:
        dai5.select_stage(f2, ["AFEW"], fz1)
        print(f"{case:24s} -> 通った（止まらない）")
    except SystemExit as e:
        print(f"{case:24s} -> 止まる：{str(e)[:90]}")


for case in ("grad_mod", "stalled_top1", "finite_src", "grad_srcw", "teacher_check_false", "teacher_check_missing", "variant_row_missing"):
    mutate(case)
dai1.git_commit = orig_git

# (3) select の後に csv を替えたとき，評価の段の守りの pseudo の sha256 の照合が止めるか（check_eval_gate5 の該当の部分だけを同じ式で回す）
def pseudo_sha_loop(out_dir, fzj):
    for nm, h in fzj["AFEW"]["pseudo_unit_sha256"].items():
        fp = out_dir / "pseudo" / "units" / nm
        if not fp.exists() or dai1.sha(fp) != h:
            return f"止まる：{nm}"
    return "通った"


print("csv 替えない        ->", pseudo_sha_loop(fake, fz))
p = sorted((fake / "pseudo" / "units").glob("*.csv"))[4]
p.write_text(p.read_text().replace("srcw_top1", "srcw_top1", 1) + "\n")
print("csv に 1 行を足す   ->", pseudo_sha_loop(fake, fz))
src_line = [ln for ln in Path(dai5.__file__).read_text().splitlines() if "pseudo_unit_sha256=" in ln]
print("select の sha の行:", src_line[0].strip()[:160])
