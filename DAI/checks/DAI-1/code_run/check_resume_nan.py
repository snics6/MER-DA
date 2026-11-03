import sys
sys.path[:0] = ["/home/snics/MER-DA/DAI/checks/DAI-1/code_run/snap", "/home/snics/MER-DA/measure", "/home/snics/MER-DA/features"]
from pathlib import Path
import numpy as np, torch, math
import dai1, dai_data as DD, dai_solver as S
out = Path(sys.argv[1])
# I12：別の引き戻しの強さで段 2 を呼ぶと，保存済みのファイルを照合なしに返すか
teth2 = {(k, K): (0.3, 1.0) for k in DD.KINDS for K in DD.KS}
d = dai1.run_pass(out, "pseudo", "MOSEI", "MELD", ["AFEW", "IEMOCAP"], 0, 0, 2, tether=teth2)
z = np.load(out / "pseudo" / "units" / "MOSEI_MELD_f0_r0_p2.npz")
print("段 2 を引き戻し (0.3, 1.0) で呼んだ：返った行", len(d), "，保存の tether の先頭", z["tether"][0], "→ 照合なしに古い結果が返った")
# I11：勾配に NaN が出たとき，fit が収束したと返すか
X = torch.randn(50, 3, dtype=torch.float64); G = torch.zeros(1, 50, 4, dtype=torch.float64); G[0, :, 0] = 1.0
G[0, 0, 1] = float("nan")
r = S.fit(X, G, torch.zeros(4, 3), torch.zeros(4), Cw=1.0, Cb=1.0)
print("G に NaN を入れた問：ok =", r["ok"], " grad_max =", r["grad_max"], " n_iter =", r["n_iter"], " W に NaN", bool(torch.isnan(r["W"]).any()))
