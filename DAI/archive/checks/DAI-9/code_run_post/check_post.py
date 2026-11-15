"""DAI-9 計算の後の点検（値は出さず，数・欠け・重複・一致だけを出す）。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA")
OUT = ROOT / "results/dai9_probe"
UD = ROOT / "results/dai1/pseudo/units"
TAUS = ("raw", "gc", "nap10", "nap50", "pca64", "pca256")
KS = (1, 5, 10, 25)
CORP = ["AFEW", "IEMOCAP", "MELD", "MOSEI"]
exp_pred = set()
exp_P = set()
exp_C = set()
for t in TAUS:
    exp_pred |= {f"{t}_ldaO", f"{t}_emtruth", f"{t}_emrand"}
    exp_P |= {f"P_{t}_emtruth", f"P_{t}_emrand"}
    exp_C |= {f"C_{a}_{t}_ldaO" for a in "A c mu pi".split()}
    for K in KS:
        exp_pred |= {f"K{K}_{t}_ldaK", f"K{K}_{t}_emK", f"K{K}_{t}_emKconv"}
        exp_P |= {f"P_K{K}_{t}_emK", f"P_K{K}_{t}_emKconv"}
        exp_C |= {f"C_{a}_K{K}_{t}_emK" for a in "A c mu pi".split()}
exp_B = {f"B_{t}" for t in ("nap10", "nap50", "pca64", "pca256")}
exp_meta = {"commit", "T", "test_sample_id", "test_y", "test_cluster", "pool_y", "pool_group", "pool_dialogue", "diag"}
exp_all = exp_pred | exp_P | exp_C | exp_B | exp_meta
print("予定の鍵の数", len(exp_pred), len(exp_P), len(exp_C), len(exp_B), len(exp_meta), len(exp_all))
exp_diag = set()
for t in TAUS:
    exp_diag |= {f"{t}_info", f"{t}_ldaO", f"{t}_emtruth", f"{t}_emrand"} | {f"K{K}_{t}_emK" for K in KS}
names = sorted(p.name for p in (OUT / "rows").glob("*.npz"))
want = [f"{c}_f{f}_r{r}" for c in CORP for f in range(5) for r in (0, 1)]
print("npz", len(names), "予定", len(want), "欠け", sorted(set(w + ".npz" for w in want) - set(names)), "余り", sorted(set(names) - set(w + ".npz" for w in want)))
q1n = sorted(p.name for p in (OUT / "rows").glob("*_q1.csv"))
wq = [f"{c}_f{f}_r0_q1.csv" for c in CORP for f in range(5)]
print("q1", len(q1n), "欠け", sorted(set(wq) - set(q1n)), "余り", sorted(set(q1n) - set(wq)))
prob = []
commits = set(); Ts = {}
testsets = {}
for w in want:
    z = np.load(OUT / "rows" / f"{w}.npz", allow_pickle=False)
    ks = set(z.files)
    if ks != exp_all:
        prob.append((w, "鍵", sorted(exp_all - ks)[:5], sorted(ks - exp_all)[:5]))
    commits.add(str(z["commit"]))
    c = w.split("_")[0]; Ts.setdefault(c, set()).add(str(z["T"]))
    nt, npool = len(z["test_y"]), len(z["pool_y"])
    if nt == npool: prob.append((w, "テストと候補の本数が同じ（集計の鍵の選び方に注意）"))
    sid = z["test_sample_id"].astype(str)
    if len(set(sid)) != nt: prob.append((w, "テストの sample_id が重複"))
    for k in exp_pred:
        a = z[k]
        if a.dtype != np.int8 or a.shape != (nt,) or a.min() < 0 or a.max() > 3: prob.append((w, k, "判定の形か値の範囲"))
    for k in exp_P:
        a = z[k]
        if a.shape != (npool,) or a.min() < 0 or a.max() > 3: prob.append((w, k, "割り当ての形か値の範囲"))
    for k in ("pool_group", "pool_dialogue"):
        if z[k].shape != (npool,): prob.append((w, k, "長さ"))
    if z["test_cluster"].shape != (nt,): prob.append((w, "test_cluster 長さ"))
    if set(np.unique(z["test_y"])) != {0, 1, 2, 3} or set(np.unique(z["pool_y"])) != {0, 1, 2, 3}: prob.append((w, "ラベルが 4 種でない"))
    for k in exp_C | exp_B:
        a = z[k].astype(np.float64)
        if not np.isfinite(a).all(): prob.append((w, k, "有限でない"))
    for k in exp_B:
        p = int(k.split("_")[1][3:]) if "pca" in k else None
        if z[k].shape[0] != 2304: prob.append((w, k, "基底の行"))
        if p and z[k].shape[1] != p: prob.append((w, k, "基底の列"))
    d = json.loads(str(z["diag"]))
    if set(d) != exp_diag: prob.append((w, "diag の鍵", sorted(exp_diag - set(d))[:5]))
    for t in ("nap10", "nap50"):
        if z[f"B_{t}"].shape[1] != d[f"{t}_info"]["nap_r_used"]: prob.append((w, t, "基底の列と nap_r_used が違う"))
    # 入力の保存との照合（候補の sample_id と K 本は入力の p1 から引ける）
    T = str(z["T"])
    z1 = np.load(UD / f"{T}_{w}_p1.npz", allow_pickle=False)
    if not (z1["sample_test"].astype(str) == sid).all(): prob.append((w, "入力のテストの並びと違う"))
    if len(z1["sample_pool"]) != npool: prob.append((w, "入力の候補の本数と違う"))
    for K in KS:
        if f"sample_K{K}" not in z1.files: prob.append((w, f"入力に sample_K{K} が無い"))
        else:
            n = len(z1[f"sample_K{K}"])
            if n != 4 * K: prob.append((w, f"K{K} の本数 {n}"))
    f_ = w.split("_")[1]
    testsets.setdefault((c, f_), []).append(tuple(sid))
print("単位のコミット", commits)
print("T", {c: sorted(v) for c, v in Ts.items()})
print("同じ (c,f) で r0 と r1 のテストの並びが同じ", sum(len(set(v)) == 1 for v in testsets.values()), "/", len(testsets))
print("問題", len(prob))
for x in prob[:30]: print(" ", x)
