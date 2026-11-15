"""gate-code 第 2 版（DAI-5）：check_inputs の出発点の照合（第 1 版の指摘 4）が，本物の保存で通り，出発点の判定を 1% 替えた偽の保存で止まるか。成績は出さない。
使い方：CUDA_VISIBLE_DEVICES= ./.venv/bin/python DAI/checks/DAI-5/code_run_v2/start_guard.py"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1, dai5, dai_data as DD
T, c, f, r = "AFEW", "IEMOCAP", 0, 0
z1 = dict(np.load(ROOT / f"results/dai1/pseudo/units/{T}_{c}_f{f}_r{r}_p1.npz"))
u = dai1.Unit(dai5.IN_DIR, T, c, [s for s in DD.others(T) if s != c], f, r)
dai5.check_inputs(u, z1); print("本物の保存 -> 通った")
for frac in (0.0005, 0.01):
    z2 = dict(z1); p = z2["p1_V_start_pred"].copy(); k = max(1, int(len(p) * frac)); p[:k] = (p[:k] + 1) % 4; z2["p1_V_start_pred"] = p
    try:
        dai5.check_inputs(u, z2); print(f"出発点 V の判定を {k}/{len(p)} 本替えた -> 通った")
    except SystemExit as e:
        print(f"出発点 V の判定を {k}/{len(p)} 本替えた -> 止まる：{str(e)[:80]}")
