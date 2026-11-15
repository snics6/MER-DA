"""gate-code（DAI-5）：pick5 と独立の実装の比べ，select_stage を偽の置き場で流す（uar は乱数に置き換え．本物の成績は使わない）。
使い方：./.venv/bin/python DAI/checks/DAI-5/code_run/try_select.py <試走の置き場> <偽の置き場>
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
import dai_data as DD  # noqa: E402

src, fake = Path(sys.argv[1]), Path(sys.argv[2])
rng = np.random.default_rng(3)

# 1. pick5 と独立の実装（最大の平均から δ 以内の候補のうち格子の番号が最も小さい点）
G = list(dai5.BETA_GRID)
bad, used = 0, 0
for _ in range(3000):
    nb = 5
    m = rng.normal(50, 1, nb).round(int(rng.integers(0, 3)))
    delta = float(rng.uniform(0, 1.5))
    g = pd.DataFrame({"beta": np.repeat(G, 3), "uar": np.repeat(m, 3)})
    b, u, mu = dai5.pick5(g, delta)
    cands = [G[i] for i in range(nb) if m[i] >= m.max() - delta]
    ref = cands[0]
    bad += int(b != ref or not np.isclose(mu, m[G.index(ref)]))
    used += int(u)
print("pick5 vs independent: mismatch", bad, "/ 3000, tie rule used", used)

# 2. select_stage を偽の置き場で：本物の 1 単位の csv を 30 単位ぶん写し，uar を乱数にする
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
    s = d.method == "srcw"
    d.loc[s, "uar"] = rng.normal(50, 3, s.sum())
    d.to_csv(fake / "pseudo" / "units" / f"AFEW_{nm}_p5.csv", index=False)
    shutil.copy(npz, fake / "pseudo" / "units" / f"AFEW_{nm}_p5.npz")
orig_git = dai1.git_commit
dai1.git_commit = lambda: (str(base.commit.iloc[0]), False)
dai5.select_stage(fake, ["AFEW"], fz1)
fz = json.load(open(fake / "select" / "frozen.json"))
A = fz["AFEW"]
print("select keys", sorted(A.keys()))
print("n_units", A["n_units"], "pseudo_unit_sha256", len(A["pseudo_unit_sha256"]), "partial", fz["_meta"].get("partial_targets"),
      "pseudo_gpu", fz["_meta"]["pseudo_gpu"], "select_student", len(A["select_student"]))
# 選んだ β を独立に作り直す
raw = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted((fake / "pseudo" / "units").glob("AFEW_*_p5.csv"))])
d = raw[raw.method == "srcw"]
mm = 0
for K in dai1.KS_ALL:
    g = d[d.K == K].groupby("beta").uar.mean()
    dl = np.mean([fz1["AFEW"]["delta_sel"][f"{s}|{K}"] for s in DD.STUDENTS])
    ref = [b for b in G if b in g.index and g[b] >= g.max() - dl][0]
    mm += int(ref != A["select"][str(K)]["beta"])
    for s in DD.STUDENTS:
        gs = d[(d.K == K) & (d.student == s)].groupby("beta").uar.mean()
        ds = fz1["AFEW"]["delta_sel"][f"{s}|{K}"]
        rs = [b for b in G if b in gs.index and gs[b] >= gs.max() - ds][0]
        mm += int(rs != A["select_student"][f"{s}|{K}"]["beta"])
print("select beta independent mismatch", mm, "/ 25")
# 3. 止まる場合：1 単位を消す，dirty の行，eval の出力がある
for case in ("drop_unit", "eval_exists", "dup_beta_row"):
    f2 = Path(str(fake) + "_" + case)
    if f2.exists():
        shutil.rmtree(f2)
    shutil.copytree(fake, f2)
    shutil.rmtree(f2 / "select")
    fs = sorted((f2 / "pseudo" / "units").glob("*.csv"))
    if case == "drop_unit":
        fs[0].unlink()
    elif case == "eval_exists":
        (f2 / "eval" / "units").mkdir(parents=True); (f2 / "eval" / "units" / "x").write_text("x")
    else:
        d0 = pd.read_csv(fs[0], low_memory=False)
        d0 = pd.concat([d0, d0[(d0.method == "srcw") & (d0.K == 5)].head(1)])
        d0.to_csv(fs[0], index=False)
    try:
        dai5.select_stage(f2, ["AFEW"], fz1)
        print(case, "-> 通った（止まらない）")
    except SystemExit as e:
        print(case, "-> 止まる：", str(e)[:80])
dai1.git_commit = orig_git
