"""gate-code 第 2 版（DAI-5）：dai5_stats.evaluate を偽の評価の回で最後まで流し，新しい関数（collect の extras・sel_var・halves，
regret_rows，rule_b_chance，scores_rows，jk_folds，重みの表，tc|・tj| の行，固定の点の照合）が保存した出力から動く形かを確かめる。
偽の評価の回：仮のターゲットの回の試走 AFEW_IEMOCAP_f{0..4}_r0（--test-tether，β = 5 の p5t つき）を AFEW_ALL_f{f}_r0 と名づけ，
実験 1 の仮のターゲットの回の同じ単位の p1・p2・p3 に，joint の選んだ点の判定を 14 点に写した p3t を足して入力にする。
評価のターゲットの本物のデータは読まない。成績の値は出さない（行の数・欠け・NaN の数・形だけ）。
使い方：CUDA_VISIBLE_DEVICES= ./.venv/bin/python DAI/checks/DAI-5/code_run_v2/try_stats_v2.py <試走の置き場> <偽の根>
"""
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai5 as D5  # noqa: E402
import dai5_stats as S5  # noqa: E402
import dai_data as DD  # noqa: E402

src, fake = Path(sys.argv[1]), Path(sys.argv[2])
if fake.exists():
    shutil.rmtree(fake)
fin, fout = fake / "in", fake / "out5"
(fin / "eval" / "units").mkdir(parents=True); (fout / "eval" / "units").mkdir(parents=True); (fout / "pseudo" / "units").mkdir(parents=True)
(fout / "select").mkdir(parents=True)
fz1_real = json.load(open(D5.FROZEN1))
fz1 = json.loads(json.dumps(fz1_real)); fz1["AFEW"]["n_seeds_eval"] = 1
for f in range(DD.N_FOLDS):
    a = ROOT / "results" / "dai1" / "pseudo" / "units" / f"AFEW_IEMOCAP_f{f}_r0"
    b = fin / "eval" / "units" / f"AFEW_ALL_f{f}_r0"
    for w in (1, 2):
        shutil.copy(f"{a}_p{w}.npz", f"{b}_p{w}.npz")
    shutil.copy(f"{a}_p3.csv", f"{b}_p3.csv"); shutil.copy(f"{a}_p1.csv", f"{b}_p1.csv")
    z3 = dict(np.load(f"{a}_p3.npz"))
    for s in DD.STUDENTS:
        for K in (5, 25):
            jj = D5.joint_sel(fz1["AFEW"], s, K)
            it = [json.loads(x) for x in z3[f"p3_{s}_K{K}_items"]]
            jm = [j for j, x in enumerate(it) if x.get("method") == "joint" and x.get("teacher") == jj["teacher"] and np.isclose(x["rho"], jj["rho"]) and np.isclose(x["a"], jj["a"])][0]
            items = [json.dumps(dict(method="joint_tether", tag="prop_sel", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"], Cw=cw, Cb=cb), sort_keys=True) for cw, cb in dai1.TETHER]
            z3[f"p3t_{s}_K{K}_items"] = np.array(items)
            z3[f"p3t_{s}_K{K}_pred"] = np.repeat(z3[f"p3_{s}_K{K}_pred"][jm][None], len(items), 0)
    np.savez(f"{b}_p3.npz", **z3)
    sp = src / "pseudo" / "units" / f"AFEW_IEMOCAP_f{f}_r0"
    shutil.copy(f"{sp}_p5.npz", fout / "eval" / "units" / f"AFEW_ALL_f{f}_r0_p5.npz")
    shutil.copy(f"{sp}_p5.csv", fout / "eval" / "units" / f"AFEW_ALL_f{f}_r0_p5.csv")
    shutil.copy(f"{sp}_p5.npz", fout / "pseudo" / "units" / f"AFEW_IEMOCAP_f{f}_r0_p5.npz")
gpu = str(np.load(fout / "eval" / "units" / "AFEW_ALL_f0_r0_p5.npz")["gpu"])
sel = {"0": 0.0, "1": 2.5, "5": 5.0, "10": 0.0, "25": 5.0}       # 偽の選択（p5t は β = 5 で解いたので K = 5・25 は 5．K = 10 は β = 0 で identical を作る）
frozen = {"_meta": {"pseudo_gpu": [gpu]},
          "AFEW": {"select": {K: {"beta": b} for K, b in sel.items()},
                   "select_student": {f"{s}|{K}": {"beta": b} for s in DD.STUDENTS for K, b in sel.items()},
                   "select_variants": {"srcw_mod|5": {"beta": 2.5}, "srcw_mod|25": {"beta": 0.0}, "srcw_src|5": {"beta": 10.0}, "srcw_src|25": {"beta": 20.0}},
                   "detect": {"cells": {}}}}
json.dump(frozen, open(fout / "select" / "frozen.json", "w"))

# 差し替え（本物の DAI/frozen には書かない）
DD.CORPORA = ("AFEW",)
orig_jload, orig_lm = S5.jload, ST.label_maps
shas = {}


def jload(p):
    p = Path(p)
    if p == D5.FROZEN1:
        return fz1
    if p == D5.GATE5:
        return {"input_sha256": {f"AFEW_ALL_f{f}_r0_p{w}.npz": dai1.sha(fin / "eval" / "units" / f"AFEW_ALL_f{f}_r0_p{w}.npz") for f in range(DD.N_FOLDS) for w in (1, 2, 3)}}
    return orig_jload(p)


S5.jload = jload
S5.guard_eval = lambda out5, fz1_: "fakecommit"
S5.pairing = lambda *a: None
S5.recheck_sources = lambda *a: pd.DataFrame()
S5.shutil = types.SimpleNamespace(copy=lambda *a, **k: None)
ST.label_maps = lambda T: orig_lm("IEMOCAP")
S5.evaluate(fout, fin)

st = fout / "stats"
for nm in ("families7", "desc", "regret", "weights", "scores", "teacher_sharp", "src_agree", "poe_check", "rule_b_chance", "gpu_counts"):
    p = st / f"{nm}.csv"
    if not p.exists() or p.stat().st_size < 5:
        print(f"{nm:14s} 無いか空"); continue
    d = pd.read_csv(p, low_memory=False)
    num = d.select_dtypes("number")
    print(f"{nm:14s} 行 {len(d):5d} 列 {d.shape[1]:4d} 数の列の NaN {int(num.isna().sum().sum()):6d} Inf {int(np.isinf(num.to_numpy(float)).sum())}")
d = pd.read_csv(st / "desc.csv", low_memory=False)
print("desc の read ごとの行:", d.read.fillna("").value_counts().to_dict())
print("desc の B（read が空）:", sorted(d[d.read.fillna("") == ""].B.unique()))
tc = d[d.read == "tether_curve"]
print("tether_curve：(生徒, K) ごとの点の数", tc.groupby(["student", "K"]).size().unique().tolist(), "K", sorted(tc.K.unique()))
fp = d[d.read == "fixed_point_check"] if "fixed_point_check" in set(d.read.dropna()) else None
print("fixed_point_check：行", 0 if fp is None else len(fp), "一致の最小", None if fp is None else float(fp.fixed_point_counts_agree.min()))
print("identical_by_construction の行（B ごと）:", d[d.identical_by_construction.astype(bool)].groupby("B").size().to_dict())
r = pd.read_csv(st / "regret.csv")
print("regret：which ごとの行", r.which.value_counts().to_dict(), "effect < 0 の行", int((r.effect < -1e-9).sum()),
      "null_level が NaN の行（which ごと）", r[r.null_level.isna()].which.value_counts().to_dict(), "n_grid", sorted(r.n_grid_points.unique()))
rb = pd.read_csv(st / "rule_b_chance.csv")
print("rule_b_chance：", rb[["T", "K", "n_cells", "chance_ge4", "observed_positive"]].to_dict("records"))
w = pd.read_csv(st / "weights.csv")
print("weights：K", w.K.tolist(), "uniform", w.uniform_by_construction.tolist(), "share の列", sum(c.startswith("share_") for c in w.columns),
      "rank の列", sum(c.startswith("rank_") for c in w.columns), "mag の列", sum(c.startswith("mag_") for c in w.columns))
print("weights：share の和（K ごと）", [round(float(x), 6) for x in w[[c for c in w.columns if c.startswith("share_") and not c.startswith(("share_mod_", "share_src_"))]].sum(1)])
s = pd.read_csv(st / "scores.csv")
print("scores：K", s.K.tolist(), "usable_rule", s.usable_rule.tolist(), "n_units", s.n_units.tolist())
print("jk_folds の検算：", S5.jk_folds({0: [1.0, 3.0], 1: [2.0], 2: [4.0], 3: [np.nan]}))
