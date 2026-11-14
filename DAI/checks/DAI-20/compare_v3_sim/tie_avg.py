"""gate-compare 第 3 回の確かめ：重みの表を探す版の同点の規則（最大に並んだ表の平均）が，選んだ表を等しい重みへ寄せる大きさ．
人工の数だけを使う（データは読まない）．台本の下書きの wf_search・wf_pick をそのまま呼ぶ．CPU で流す．"""
import sys
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/dai20")
import dai20_eval as E  # noqa: E402
P16, dai15 = E.P16, E.dai15
rng = np.random.default_rng(7)
for K in (5, 25):
    res = {m: [] for m in E.WF_MS}
    for rep in range(4):
        yk = np.repeat(np.arange(4), K)
        fold = dai15.kfold_of(yk)
        # 3 つの分類器：強さを変えた人工の確率（分類器と感情で当てやすさが違う）
        Ph = np.empty((3, 4 * K, 4))
        for k, s in enumerate((0.6, 0.35, 0.2)):
            P = rng.dirichlet(np.ones(4), size=4 * K)
            P[np.arange(4 * K), yk] += s * rng.uniform(0.3, 1.7, size=4)[yk]
            Ph[k] = P / P.sum(1, keepdims=True)
        LMh = np.stack([P16.centered(p) for p in Ph])
        LMpj = [np.stack([P16.centered(p) for p in rng.dirichlet(np.ones(4) * 0.7, size=(3, 120))]) for _ in range(E.NF)]
        ua = E.wf_search(LMh, yk, fold, LMpj)
        for m in E.WF_MS:
            W, tied = E.wf_pick(ua, m)
            res[m].append((len(tied), float(np.abs(W - 1).mean()), float(np.abs(E.WS[tied[0]] - 1).mean()), float(ua[:m + 1].max() - ua[0])))
    for m in E.WF_MS:
        a = np.array(res[m])
        print(f"K={K:2d} m={m:4d}  並んだ表の数 中央値 {np.median(a[:,0]):6.1f} 最大 {a[:,0].max():5.0f}  "
              f"|平均した表−1| の平均 {a[:,1].mean():.3f}  |並んだ最初の表−1| の平均 {a[:,2].mean():.3f}  K 本の UAR の上積み（最大−表0）の平均 {a[:,3].mean():.1f}")
