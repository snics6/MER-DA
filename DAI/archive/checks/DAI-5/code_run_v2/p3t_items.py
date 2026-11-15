"""gate-code 第 2 版（DAI-5）：実験 1 の評価の回の p3t（joint の選んだ点の引き戻しの 14 点）の並びが dai5_stats.collect の照合の形に合うか，
固定の (C′, C_b) の点の判定が joint の選んだ点の判定と同じかを数える。ラベルは読まない（判定どうしの一致だけ）。
使い方：./.venv/bin/python DAI/checks/DAI-5/code_run_v2/p3t_items.py"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1, dai5 as D5, dai_data as DD
fz1 = json.load(open(D5.FROZEN1))
for T in DD.CORPORA:
    bad, agree = 0, []
    for f in range(DD.N_FOLDS):
        for r in (0, 5):
            z = np.load(ROOT / f"results/dai1/eval/units/{T}_ALL_f{f}_r{r}_p3.npz")
            for s in DD.STUDENTS:
                for K in (5, 25):
                    it = [json.loads(x) for x in z[f"p3t_{s}_K{K}_items"]]
                    ps = [j for j, x in enumerate(it) if x.get("method") == "joint_tether" and x.get("tag") == "prop_sel"]
                    pts = sorted((it[j]["Cw"], it[j]["Cb"]) for j in ps)
                    bad += int(pts != sorted(dai1.TETHER))
                    jj = D5.joint_sel(fz1[T], s, K)
                    bad += int(any(it[j].get("teacher") != jj["teacher"] or it[j].get("rho") != jj["rho"] or it[j].get("a") != jj["a"] for j in ps))
                    cw, cb = fz1[T]["tether"][f"{s}|{K}"]
                    jf = [j for j in ps if np.isclose(it[j]["Cw"], cw) and np.isclose(it[j]["Cb"], cb)][0]
                    i3 = [json.loads(x) for x in z[f"p3_{s}_K{K}_items"]]
                    jm = [j for j, x in enumerate(i3) if x.get("method") == "joint" and x.get("teacher") == jj["teacher"] and np.isclose(x["rho"], jj["rho"]) and np.isclose(x["a"], jj["a"])]
                    agree.append(float((z[f"p3t_{s}_K{K}_pred"][jf] == z[f"p3_{s}_K{K}_pred"][jm[0]]).mean()))
    print(T, "items form bad", bad, "/ 80 checks x2; fixed-point pred agree min", min(agree), "n", len(agree), "n<1", sum(a < 1 for a in agree))
