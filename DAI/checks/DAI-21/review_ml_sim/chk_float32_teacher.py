"""ML の査読者の検算（DAI-21）：保存の確率（float32）から疑似ラベルを作り直す道の確かめ．ラベルを使わない．

実験 20 の単位の出力は開けないので，同じ関数（dai16_probe.run_unit）が作った実験 16 の下調べの単位（仮のターゲットの回）を使う．
 (1) 保存の確率 p_pool_ret（float32）に，0 に丸まった値（1.4e-45 未満）や桁落ちした値（1.2e-38 未満）が無いか．
 (2) 台本の teacher_probs（積と提案）で作り直した候補の上の判定が，保存の判定 pred_pool と一致する割合（台本の止める条件は 0.995 未満）．
 (3) 積と提案の疑似ラベルが違う判定を付ける候補の割合（疑似ラベルの違いの大きさ．ラベルを使わない）．
使い方：CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <この台本>
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
spec = importlib.util.spec_from_file_location("dai21", "/home/snics/MER-DA/DAI/docs/tools_pending/dai21_eval5.py")
D21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D21)
P16 = D21.P16
GRID = D21.GRID

UD = Path("/home/snics/MER-DA/results/dai16_probe/units")
files = sorted(UD.glob("*.npz"))
pts = [i for i in range(len(GRID)) if GRID[i][1] == 1.0 and GRID[i][0] in (0.0, 20.0)]      # 提案に選ばれうる点（n0 = 0・20，相性を使う）
print(f"単位 {len(files)} 個．調べる格子の点 {pts}")
tiny32 = float(np.finfo(np.float32).tiny)
res = []
for p in files:
    z = np.load(p, allow_pickle=False)
    for K in (5, 25):
        if f"kpos_K{K}" not in z.files:
            continue
        kpos = z[f"kpos_K{K}"]
        P = [z[f"p_pool_ret_{m}_K{K}"] for m in "AVT"]
        n = P[0].shape[0]
        nk = np.setdiff1d(np.arange(n), kpos)
        n_zero = int(sum((q == 0).sum() for q in P))
        n_sub = int(sum(((q > 0) & (q < tiny32)).sum() for q in P))
        pmin = float(min(q[q > 0].min() for q in P))
        LMp = np.stack([P16.centered(q.astype(np.float64)) for q in P])
        Q0 = D21.teacher_probs(LMp, None)
        row = dict(unit=p.stem, K=K, n=n, n_zero=n_zero, n_subnormal=n_sub, p_min=pmin,
                   agree_prod=float((Q0[nk].argmax(1) == z[f"pred_pool_K{K}_prod"].astype(int)).mean()))
        ag, dif = [], []
        for i in pts:
            W = P16.to_w(P16.mix_rel(z[f"D_M_K_K{K}"], z["SR_M_ext"], 4 * K, *GRID[i]))
            Q1 = D21.teacher_probs(LMp, W)
            ag.append(float((Q1[nk].argmax(1) == z[f"pred_pool_K{K}_M_F6_mix{i}"].astype(int)).mean()))
            dif.append(float((Q1[nk].argmax(1) != Q0[nk].argmax(1)).mean()))
        row.update(agree_prop_min=min(ag), teacher_differs_min=min(dif), teacher_differs_max=max(dif),
                   l1_mean=float(np.abs(Q1 - Q0).sum(1).mean()))
        res.append(row)
import pandas as pd  # noqa: E402

d = pd.DataFrame(res)
d["c"] = d.unit.str.split("_").str[1]
print(f"行 {len(d)}（単位 × K）")
print("0 に丸まった確率の数（合計）:", int(d.n_zero.sum()), "／桁落ちした確率の数（合計）:", int(d.n_subnormal.sum()), "／正の確率の最小:", f"{d.p_min.min():.3e}")
print("作り直した判定の一致の最小：積", f"{d.agree_prod.min():.6f}", "提案の点", f"{d.agree_prop_min.min():.6f}")
print("一致が 1 未満の行の数：積", int((d.agree_prod < 1).sum()), "提案の点", int((d.agree_prop_min < 1).sum()), "／ 0.995 未満の行の数", int(((d.agree_prod < 0.995) | (d.agree_prop_min < 0.995)).sum()))
print("\n仮のターゲット × K ごと：積と提案の疑似ラベルで判定が違う候補の割合（%．格子の点の最小〜最大の平均），確率の差の L1 の平均（最後の点）")
g = d.groupby(["c", "K"]).agg(units=("unit", "count"), n=("n", "mean"), differs_min=("teacher_differs_min", "mean"), differs_max=("teacher_differs_max", "mean"),
                              l1=("l1_mean", "mean"), agree_prop_min=("agree_prop_min", "min"), p_min=("p_min", "min"))
g[["differs_min", "differs_max"]] *= 100
print(g.round(4).to_string())
