"""gate-code の再検査（DAI-6）：_boot_fold_mean が，テストの塊が分割ごとに 1 つ（IEMOCAP）と数個（AFEW の分割 0，MELD の分割 4）のときに区間を作るか。CPU だけ。"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai6_probe as P
rng = np.random.default_rng(0)
def fold(n, ncl):
    y = rng.integers(0, 4, n); cl = np.array([f"c{j}" for j in rng.integers(0, ncl, n)])
    return (y, cl, {"S": np.where(rng.random(n) < .5, y, rng.integers(0, 4, n)), "full": np.where(rng.random(n) < .6, y, rng.integers(0, 4, n))})
for name, spec in [("IEMOCAP 型（分割ごとに塊 1）", [(600, 1)] * 5), ("MELD 型（分割 4 だけ塊 2）", [(2400, 66)] * 4 + [(2400, 2)]), ("塊 50 × 5", [(600, 50)] * 5)]:
    bt, nv = P._boot_fold_mean([fold(n, k) for n, k in spec], seed=1)
    d = bt["S"] - bt["full"]
    print(f"{name}：有効な回 {nv}，差の 95% 区間の幅 {np.percentile(d, 97.5) - np.percentile(d, 2.5):.3f}，S の標準偏差 {bt['S'].std():.3f}")
