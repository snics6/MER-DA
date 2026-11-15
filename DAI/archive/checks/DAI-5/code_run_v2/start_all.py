"""gate-code 第 2 版（DAI-5）：check_inputs の出発点の照合（しきい値 0.999）が，本番の全単位（pseudo 120，eval 200）で
A5000 の再計算と実験 1 の保存（H200）の判定の一致として成り立つかを数える。ラベルは読まない（判定どうしの一致だけ）。
使い方：./.venv/bin/python DAI/checks/DAI-5/code_run_v2/start_all.py <pseudo|eval> [T ...]"""
import sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import json, dai1, dai5, dai_solver as S, dai_data as DD
mode = sys.argv[1]; Ts = sys.argv[2:] or list(DD.CORPORA)
fz1 = json.load(open(dai5.FROZEN1))
t0 = time.time(); worst = []
for T in Ts:
    seeds = [0, 1] if mode == "pseudo" else list(range(int(fz1[T]["n_seeds_eval"])))
    for c, srcs, f, r in dai1.units_for(T, mode, range(DD.N_FOLDS), seeds):
        cn = "ALL" if mode == "eval" else c
        z1 = np.load(dai5.IN_DIR / mode / "units" / f"{T}_{cn}_f{f}_r{r}_p1.npz")
        u = dai1.Unit(dai5.IN_DIR, T, c, srcs, f, r)
        for s in DD.STUDENTS:
            pr = S.logits(u.Xt[s], *u.start[s]).argmax(1).cpu().numpy()
            ref = z1[f"p1_{s}_start_pred"]
            worst.append((float((pr == ref).mean()), int((pr != ref).sum()), len(pr), f"{T}_{cn}_f{f}_r{r}", s))
        del u
    dai1.Corpus.cache.clear(); torch.cuda.empty_cache()
worst.sort()
print(mode, "units x students", len(worst), "below 0.999:", sum(w[0] < 0.999 for w in worst), "with any mismatch:", sum(w[1] > 0 for w in worst),
      "worst 3:", worst[:3], "sec", round(time.time() - t0))
