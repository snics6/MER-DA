"""uar_bias_fixed の 1 行あたりの所要（CPU．テストの発話数ごと）。値はでたらめ。"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = ["x"]
sys.path.insert(0, "/home/snics/MER-DA/exp")
import numpy as np
import analyze_e79 as E
rng = np.random.default_rng(0)
for n in (300, 1000, 3800):
    y = rng.integers(0, 4, n)
    ts = []
    for i in range(20):
        L = rng.normal(0, 1, (n, 4)) + 1.0 * np.eye(4)[y] - np.array([0, 1.0, 0.5, 0.2])
        L = L - np.log(np.exp(L).sum(1, keepdims=True))
        t0 = time.perf_counter(); E.uar_bias_fixed(L, y); ts.append(time.perf_counter() - t0)
    print(n, "1 行あたり", round(float(np.median(ts)) * 1000, 1), "ms")
