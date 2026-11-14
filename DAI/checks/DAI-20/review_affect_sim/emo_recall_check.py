"""感情認識の査読想定の役の検算（DAI-20 第 3 版）．人工の数だけを使う（結果のファイルは開かない．GPU は使わない）．
dai20_eval.stage_summary の「感情ごとの再現率と区間」の作り方（感情 i の列だけ残した数と，正解 0 の数の差を effect_stats に渡して 4 倍する）が，
感情 i の再現率（分割ごとに種の平均を取り，分割で平均する）と一致するか，per_emotion.csv の列 n が何を数えるかを確かめる．"""
import sys

import numpy as np

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1_stats as ST  # noqa: E402


def zero(A):
    return {k: (x[0], np.zeros_like(x[1]), x[2]) for k, x in A.items()}


def only(A, i):
    m = np.zeros(4); m[i] = 1.0
    return {k: (x[0], x[1] * m, x[2]) for k, x in A.items()}


def make(rng, n_folds, n_seeds, n_cl, n_per, acc):
    cells, ys = {}, {}
    for f in range(n_folds):
        y = rng.choice(4, size=n_per, p=[0.4, 0.3, 0.2, 0.1])
        cl = np.array([f"g{f}_{j}" for j in rng.integers(0, n_cl, n_per)])
        ys[f] = y
        for r in range(n_seeds):
            pred = np.where(rng.random(n_per) < acc[y], y, rng.integers(0, 4, n_per))
            cells[(f, r)] = ST.counts(y, pred, cl)
    return cells, ys


rng = np.random.default_rng(0)
acc = np.array([0.7, 0.6, 0.4, 0.3])
for name, g_eff, n_cl in (("jackknife（塊が少ない）", 8.0, 6), ("引き直し（塊が多い）", 200.0, 40)):
    dd, ys = make(rng, 5, 10, n_cl, 160, acc)
    lev = ST.effect_stats(dd, zero(dd), g_eff, np.random.default_rng(1))
    tot4 = 0.0
    for i in range(4):
        e = ST.effect_stats(only(dd, i), zero(dd), g_eff, np.random.default_rng(1))
        direct = np.mean([np.mean([100 * dd[(f, r)][1][:, i].sum() / dd[(f, r)][2][:, i].sum() for r in range(10)]) for f in range(5)])
        n_sum = sum(x[2].sum(0)[i] for x in dd.values())
        n_distinct = sum(int((ys[f] == i).sum()) for f in range(5))
        tot4 += e["effect"]
        print(f"{name} 感情 {i}：4×effect {4 * e['effect']:.4f}，直接の再現率 {direct:.4f}，差 {abs(4 * e['effect'] - direct):.1e}，"
              f"区間 [{4 * e['lo']:.2f}, {4 * e['hi']:.2f}]（{e['method']}），列 n に入る数 {n_sum:.0f}，別々の発話の数 {n_distinct}")
        assert abs(4 * e["effect"] - direct) < 1e-9
    print(f"{name}：4 感情の effect の和 {tot4:.4f}，UAR の水準 {lev['effect']:.4f}")
    assert abs(tot4 - lev["effect"]) < 1e-9
print("検算：通った")
