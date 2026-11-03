"""gate-code: 1 単位（仮のターゲットの回．T=MOSEI，c=MELD，ソース AFEW・IEMOCAP，f0 r0）を段 1〜3 で流す。UAR は画面に出さない。"""
import sys, time
sys.path[:0] = ["/home/snics/MER-DA/DAI/checks/DAI-1/code_run/snap", "/home/snics/MER-DA/measure", "/home/snics/MER-DA/features"]
from pathlib import Path
import torch
torch.cuda.set_per_process_memory_fraction(0.5)
import numpy as np, pandas as pd
import dai1, dai_data as DD
out = Path(sys.argv[1])
teth = {(k, K): (0.01, 10.0) for k in DD.KINDS for K in DD.KS}
w = {K: (2.0, 0.5, 1.0) for K in dai1.KS_ALL}
t0 = time.time()
for which in (1, 2, 3):
    t1 = time.time()
    d = dai1.run_pass(out, "pseudo", "MELD", "MOSEI", ["AFEW", "IEMOCAP"], 0, 0, which, tether=teth, wsel=w)
    print("段", which, "行", len(d), "列", len(d.columns), "所要", round(time.time() - t1), "秒",
          "GPU 最大確保", round(torch.cuda.max_memory_allocated() / 2**30, 2), "GiB 予約", round(torch.cuda.max_memory_reserved() / 2**30, 2), flush=True)
    num = d.select_dtypes("number")
    print("  NaN（uar 列）", int(d["uar"].isna().sum()) if "uar" in d else None, " Inf", int(np.isinf(num.to_numpy()).sum()))
    conv = d[d.method.astype(str).str.startswith("_conv")]
    print("  収束の行", len(conv), "勾配の最大", conv.grad_max.max(), "反復の最大", conv.n_iter.max() if "n_iter" in conv else None)
print("合計", round(time.time() - t0), "秒")
