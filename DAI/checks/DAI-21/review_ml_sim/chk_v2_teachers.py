"""ML の査読者の検算（DAI-21 第 2 版）：足した疑似ラベル（相性を外した表，分類器 × 感情の成分を消した表，水準の違いを外した表，K 本だけの表）を
保存の確率（float32）から作り直した判定が，保存の判定と一致するか．ラベルを使わない．

台本の unit_students4 は，7 通りの疑似ラベルの判定が実験 20 の保存と一致する割合が 0.995 未満だと止まる．
実験 20 の出力は開けないので，同じ関数（dai16_probe.run_unit）が作った実験 16 の下調べの単位で，台本の teachers_of をそのまま呼んで確かめる．
温度の較正（tcal）の保存は実験 16 の単位に無いので，ここでは重みを台本と同じ関数（P20.temp_w）で作り，疑似ラベルが作れることだけを見る
（判定の照合は chk_v2_students.py で実験 20 の add_extra と比べる）．
使い方：CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <この台本>
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
spec = importlib.util.spec_from_file_location("dai21", "/home/snics/MER-DA/DAI/docs/tools_pending/dai21_eval5.py")
D21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D21)
P16, P20, GRID = D21.P16, D21.P20, D21.GRID

UD = Path("/home/snics/MER-DA/results/dai16_probe/units")
files = sorted(UD.glob("*.npz"))
PTS = (2, 6, 7)                                                               # 実験 20 の固定のファイルに出る提案の点（n0 = 0，n0 = 20 で戻し先 eq・ext）
res = []
for p in files:
    z = np.load(p, allow_pickle=False)
    for K in (5, 25):
        kpos = z[f"kpos_K{K}"]
        n = z[f"p_pool_ret_A_K{K}"].shape[0]
        nk = np.setdiff1d(np.arange(n), kpos)
        yk = z["y_pool"].astype(int)[kpos]                                     # K 本のラベル（温度の重みを合わせるのに使う．手法が使うラベル）
        LMp = np.stack([P16.centered(z[f"p_pool_ret_{m}_K{K}"].astype(np.float64)) for m in "AVT"])
        LMh = np.stack([P16.centered(z[f"pcf_{m}_K{K}"].astype(np.float64)) for m in "AVT"])
        w_tcal = P20.temp_w(LMh, yk, "sep")
        for i in PTS:
            t = D21.teachers_of(LMp, z[f"D_M_K_K{K}"], z["SR_M_ext"], K, i, w_tcal)
            chk = (("prod", "prod"), ("prop", f"M_F6_mix{i}"), ("noprof", f"M_F6_mix{P20.same_but(i, beta=0.0)}"), ("noint", f"M_F6n_mix{i}"),
                   ("levelfree", f"M_F6_mix{GRID.index((float('inf'), 1.0, 'eq'))}"), ("konly", "M_F6_K"))
            ag = {a: float((t[a][nk].argmax(1) == z[f"pred_pool_K{K}_{v}"].astype(int)).mean()) for a, v in chk}
            fin = all(np.isfinite(q).all() and np.allclose(q.sum(1), 1.0) for q in t.values())
            dif = {a: float((t[a][nk].argmax(1) != t["prod"][nk].argmax(1)).mean()) for a in ("prop", "noprof", "noint", "levelfree", "konly", "tcal")}
            res.append(dict(unit=p.stem, c=p.stem.split("_")[1], K=K, i=i, finite=fin, **{f"ag_{a}": v for a, v in ag.items()}, **{f"dif_{a}": v for a, v in dif.items()},
                            w_tcal_min=float(w_tcal.min()), w_tcal_max=float(w_tcal.max())))
d = pd.DataFrame(res)
agc = [c for c in d.columns if c.startswith("ag_")]
print(f"行 {len(d)}（単位 {len(files)} × K 2 つ × 提案の点 {len(PTS)}）．疑似ラベルが有限で和が 1：{bool(d.finite.all())}")
print("作り直した判定と保存の判定の一致の最小：", {c[3:]: round(float(d[c].min()), 6) for c in agc})
print("一致が 1 未満の行の数：", {c[3:]: int((d[c] < 1).sum()) for c in agc}, "／ 0.995 未満の行の数：", int((d[agc].min(1) < 0.995).sum()))
print("温度の較正の重みの範囲：", round(float(d.w_tcal_min.min()), 3), "〜", round(float(d.w_tcal_max.max()), 3))
print("\n積の疑似ラベルと判定が違う候補の割合（%．仮のターゲット × K の平均．提案の点 2）")
g = d[d.i == 2].groupby(["c", "K"])[[c for c in d.columns if c.startswith("dif_")]].mean() * 100
print(g.round(2).to_string())
