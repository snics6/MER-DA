"""gate-code の再検査 2（DAI-6）：コミット 99e1bd5 の _boot_fold_mean を，IEMOCAP 型・MELD 型・AFEW 型・MOSEI の大きさの人工の判定で流す。CPU だけ。"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai6_probe as P
rng = np.random.default_rng(0)
def fold(n, ncl, big=None):
    y = rng.integers(0, 4, n)
    if big is None:
        cl = rng.integers(0, ncl, n)
    else:                                   # 1 つの塊が big の割合を占める
        cl = np.where(rng.random(n) < big, 0, rng.integers(1, ncl, n))
    cl = np.array([f"c{j}" for j in cl])
    return (y, cl, {"S": np.where(rng.random(n) < .5, y, rng.integers(0, 4, n)), "full": np.where(rng.random(n) < .6, y, rng.integers(0, 4, n))})
def uar(y, p): return 100 * np.mean([np.mean(p[y == k] == k) for k in range(4)])
specs = {"IEMOCAP 型（分割ごとに塊 1）": [(600, 1)] * 5,
         "MELD 型（分割 4 だけ塊 2）": [(2400, 66)] * 4 + [(2400, 2)],
         "AFEW 型（分割 0 は塊 3 で 1 つが 97%）": [(158, 3, .97)] + [(158, 11)] * 4,
         "MOSEI の大きさ（分割ごとに塊 523）": [(3727, 523)] * 5,
         "塊 50 × 5": [(600, 50)] * 5}
for name, spec in specs.items():
    pf = [fold(*s) for s in spec]
    t0 = time.time(); bt, nv = P._boot_fold_mean(pf, seed=1); sec = time.time() - t0
    d = bt["S"] - bt["full"]
    point = np.mean([uar(y, pr["S"]) - uar(y, pr["full"]) for y, _, pr in pf])
    print(f"{name}：有効な回 {nv}/2000，差の点推定 {point:.3f}，抜き直しの平均 {d.mean():.3f}，95% 区間 [{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]，幅 {np.percentile(d, 97.5) - np.percentile(d, 2.5):.3f}，{sec:.2f} 秒")
# 重みがすべて 1 の回は点推定（分割の平均）と一致するか：関数の中の式を重み 1 で再現する
pf = [fold(*s) for s in specs["MELD 型（分割 4 だけ塊 2）"]]
point = np.mean([uar(y, pr["S"]) for y, _, pr in pf])
import types
src_rng = np.random.default_rng
class One:
    def multinomial(self, K, p, size): return np.ones((size, K), dtype=np.int64)
np.random.default_rng = lambda seed=None: One()
bt, nv = P._boot_fold_mean(pf, n_boot=3, seed=0)
np.random.default_rng = src_rng
print(f"重み 1 の回の S の UAR {bt['S'][0]:.10f}，分割の平均の点推定 {point:.10f}，差 {abs(bt['S'][0] - point):.2e}")
