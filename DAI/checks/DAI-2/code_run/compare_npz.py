"""gate-code の検算：同じ単位の 2 つの npz（map・xmuda・coral・lin）の判定の一致と W の差を鍵ごとに出す（UAR は出さない）。
引数：npz 1，npz 2"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
keys = sorted(set(a.files) & set(b.files))
print("共通の鍵", len(keys), "片方だけ", sorted(set(a.files) ^ set(b.files))[:20])
for k in keys:
    x, y = a[k], b[k]
    if k.endswith("_pred") or k.endswith("_pred_b"):
        if x.shape != y.shape:
            print(k, "形が違う", x.shape, y.shape)
            continue
        agree = (x == y).mean(axis=-1)
        print(f"{k}: 問 {agree.size}，判定の一致の最小 {agree.min():.4f}，平均 {agree.mean():.5f}，一致が 99% 未満の問 {(agree < 0.99).sum()}")
    elif k.endswith("_W") and x.dtype.kind == "f":
        if x.shape != y.shape:
            print(k, "形が違う", x.shape, y.shape)
            continue
        rel = np.abs(x - y).max(axis=tuple(range(1, x.ndim))) / np.maximum(np.abs(y).max(axis=tuple(range(1, x.ndim))), 1e-30) if x.ndim > 2 else np.array([np.abs(x - y).max() / max(np.abs(y).max(), 1e-30)])
        print(f"{k}: W の相対の最大差の最大 {rel.max():.2e}")
    elif k.endswith("_items") or k.endswith("_chunks"):
        print(k, "同じ" if np.array_equal(x, y) else "違う")
