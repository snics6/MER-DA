"""gate-code（DAI-5）：評価の回と同じ経路（p3rows=None）で，K = 25 の K 本の外の候補のラベルとテストのラベルを並べ替えて
dai5.solve_unit を 2 回解き，採点・重み・Q の統計・W・b・判定が変わらないことを確かめる。成績の値は出さない。保存しない。
使い方：MERDA_GPU_FRACTION=0.3 ./.venv/bin/python DAI/checks/DAI-5/code_run/permute.py T c f r
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1  # noqa: E402
import dai5  # noqa: E402
import dai_data as DD  # noqa: E402

T, c, f, r = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
fz1 = json.load(open(dai5.FROZEN1))[T]
tether = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in fz1["tether"].items()}
srcs = [s for s in DD.others(T) if s != c]
z1 = np.load(ROOT / "results" / "dai1" / "pseudo" / "units" / f"{T}_{c}_f{f}_r{r}_p1.npz")
out = {}
for tag in ("orig", "perm"):
    u = dai1.Unit(dai5.IN_DIR, T, c, srcs, f, r)
    dai5.check_inputs(u, z1)
    if tag == "perm":
        rng = np.random.default_rng(7)
        outside = np.setdiff1d(np.arange(u.n), u.kpos[25])
        u.ypool = u.ypool.copy(); u.ypool[outside] = rng.permutation(u.ypool[outside])
        u.ytest = rng.permutation(u.ytest)
        assert (u.ypool[outside] != dai1.Corpus.get(c).y[u.pool][outside]).mean() > 0.3
    store, rows = {}, []
    t0 = time.time()
    dai5.solve_unit(u, tether, fz1, store, rows, z1, None, False)
    out[tag] = (store, pd.DataFrame(rows), time.time() - t0)
a, b = out["orig"][0], out["perm"][0]
assert set(a) == set(b)
d = {}
for k in a:
    x, y = np.asarray(a[k]), np.asarray(b[k])
    if x.dtype.kind in "fc":
        d[k] = float(np.nanmax(np.abs(x - y))) if x.size else 0.0
    else:
        d[k] = float((x != y).mean()) if x.size else 0.0
pred = [k for k in d if k.endswith("_pred")]
print("keys", len(a), "pred keys", len(pred), "pred mismatch frac max", max(d[k] for k in pred))
print("W max diff", max(d[k] for k in d if k.endswith("_W")), "b max diff", max(d[k] for k in d if k.endswith("_b")))
print("scores max diff", np.nanmax([d[k] for k in d if k.endswith("_scores")]), "w max diff", max(d[k] for k in d if "_w_b" in k))
ra, rb = out["orig"][1], out["perm"][1]
print("methods in eval path rows", sorted(ra.method.unique()))
ts_a = ra[ra.method == "_teacher_stats"].drop(columns=["stage"]).reset_index(drop=True)
ts_b = rb[rb.method == "_teacher_stats"].drop(columns=["stage"]).reset_index(drop=True)
num = ts_a.select_dtypes("number").columns
print("teacher_stats max diff", float(np.nanmax(np.abs(ts_a[num].to_numpy(float) - ts_b[num].to_numpy(float)))),
      "heaviest same", bool((ts_a.heaviest == ts_b.heaviest).all()))
sa, sb = ra[ra.method == "srcw"], rb[rb.method == "srcw"]
print("srcw uar rows differ (test labels permuted, expected True)", bool((sa.uar.to_numpy() != sb.uar.to_numpy()).any()))
print("sec", [round(out[t][2], 1) for t in out])
