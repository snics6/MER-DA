"""gate-code: 段 source を一時の置き場で流す（UAR は画面に出さない）。"""
import sys, time, io, contextlib
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import torch, numpy as np, pandas as pd
import dai1
out = Path(sys.argv[1]); corpora = sys.argv[2:]
t0 = time.time()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    dai1.stage_source(out, corpora)
print("所要", round(time.time() - t0), "秒; GPU 最大確保", round(torch.cuda.max_memory_allocated() / 2**30, 2), "GiB; 予約", round(torch.cuda.max_memory_reserved()/2**30,2))
sm = pd.read_csv(out / "source" / "summary.csv")
print("summary 行", len(sm), "列", list(sm.columns)[:4], "...", "NaN", int(sm.isna().sum().sum()))
print(sm[["corpus", "kind", "grad_max", "sec"]].to_string())
for c in corpora:
    for k in dai1.DD.KINDS:
        z = np.load(out / "source" / f"{c}_{k}.npz")
        print(c, k, "W", z["W"].shape, z["W"].dtype, "n", len(z["sample_id"]), "finite", bool(np.isfinite(z["W"]).all()), "cv shape", z["cv"].shape)
