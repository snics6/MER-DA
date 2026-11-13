"""感情認識の査読役の検算（DAI-17 第 1 版）：人工の数だけを使う（特徴も結果も読まない）．
設計 3.7 節の「ラベルをそろえた信頼度の 3 分位」は，2 つのソースを合わせた表の中のラベル e の発話で 3 分位を切る（台本 run_unit）．
コーパスの中では信頼度と当たりやすさが無関係でも，2 つのソースで (a) 信頼度（コーパス × ラベルの中で平均 1 に割った値）の散らばり方と (b) 当たる割合が違うと，
D2（高い − 低い）が 0 からずれるかを確かめる．あわせて，3 分位をコーパス × ラベルの中で切る数え方も出す．
"""
import numpy as np

rng = np.random.default_rng(0)


def run(n1, n2, ab1, ab2, hit1, hit2, share2, n_pairs=400000):
    """n1・n2：ラベル e の発話の数（ソース 1・2）．ab：信頼度の元の値の Beta 分布の形．hit：そのソースの発話を引いたときに当たる割合（信頼度と無関係）．
    share2：引いたラベル e の発話のうちソース 2 の割合．"""
    r1 = rng.beta(*ab1, n1); r1 /= r1.mean()
    r2 = rng.beta(*ab2, n2); r2 /= r2.mean()
    allr = np.concatenate([r1, r2])
    q1, q2 = np.quantile(allr, [1 / 3, 2 / 3])
    from2 = rng.random(n_pairs) < share2
    R = np.where(from2, rng.choice(r2, n_pairs), rng.choice(r1, n_pairs))
    H = rng.random(n_pairs) < np.where(from2, hit2, hit1)
    pooled = 100 * (H[R > q2].mean() - H[R <= q1].mean())
    # コーパス × ラベルの中で 3 分位を切る
    t = np.empty(n_pairs, int)
    for mk, r in ((~from2, r1), (from2, r2)):
        a, b = np.quantile(r, [1 / 3, 2 / 3])
        t[mk] = (R[mk] > a).astype(int) + (R[mk] > b).astype(int)
    within = 100 * (H[t == 2].mean() - H[t == 0].mean())
    low2 = from2[R <= q1].mean(); high2 = from2[R > q2].mean()
    return pooled, within, low2, high2


print("ソース 1 の信頼度の形, ソース 2 の形, 当たる割合 (1, 2), 引いた発話のソース 2 の割合 → D2（合わせた表の 3 分位）, D2（コーパスの中の 3 分位）, 低い群・高い群のソース 2 の割合")
for ab1, ab2 in (((1, 3), (2, 2)), ((0.5, 2), (3, 2)), ((2, 2), (1, 3))):
    for hit1, hit2 in ((0.30, 0.40), (0.40, 0.30), (0.35, 0.35)):
        for share2 in (0.1, 0.3):
            p, w, l2, h2 = run(8000, 1000, ab1, ab2, hit1, hit2, share2)
            print(f"Beta{ab1}, Beta{ab2}, ({hit1:.2f}, {hit2:.2f}), {share2:.1f} → {p:+.2f}, {w:+.2f}, {l2:.2f}・{h2:.2f}")
