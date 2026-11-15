"""IEMOCAP 型（分割ごとにテストの塊 1 つ）で，_boot_fold_mean の抜き直しが重複の回数を使うかを確かめる．実データは読まない．"""
import sys, numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai6_probe as P
rng = np.random.default_rng(5)
acc = [0.45, 0.55, 0.60, 0.70, 0.80]          # セッションごとに正答率を変える
pf = []
for j, a in enumerate(acc):
    y = np.repeat(np.arange(4), 150)
    pr = np.where(rng.random(600) < a, y, (y + 1) % 4)
    pf.append((y, np.array([f"s{j}"] * 600), {"x": pr}))
bt, nv = P._boot_fold_mean(pf, n_boot=4000, seed=1)
u = np.array([100 * np.mean([np.mean(pr["x"][y == c] == c) for c in range(4)]) for y, _, pr in pf])
# 重複の回数で重みをつけた普通のブートストラップ（5 つの分割の UAR の平均）
w = rng.multinomial(5, np.full(5, 0.2), size=4000)
std_boot = (w * u).sum(1) / 5
print("分割ごとの UAR", np.round(u, 2), "点推定", round(u.mean(), 3))
print("台本の抜き直し：標準偏差", round(bt["x"].std(), 3), "95% 区間", np.round(np.percentile(bt["x"], [2.5, 97.5]), 2), "有効", nv)
print("重複の回数を使う抜き直し：標準偏差", round(std_boot.std(), 3), "95% 区間", np.round(np.percentile(std_boot, [2.5, 97.5]), 2))
# 台本の値が「抜かれた分割の集合」だけで決まるか：同じ集合で重複が違う回の値が等しいか
print("比 =", round(bt["x"].std() / std_boot.std(), 3))
