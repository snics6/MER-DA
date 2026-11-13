"""ML の査読役の検算 3：adj の層（候補の上の確率の平均を積にそろえる）で，判定の割合はそろうか．
3 つの分類器は同じ作り（モダリティごとの違いは無い）．重みは感情ごとに違うが分類器で共通．"""
import sys
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import importlib.util
_sp = importlib.util.spec_from_file_location("p16c", "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v3/dai16_probe_556a06f.py")
P = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(P)

def uar(y, pred):
    return 100 * float(np.mean([(pred[y == i] == i).mean() for i in range(4)]))

def draw(rng, n, sep, prior):
    y = rng.choice(4, size=n, p=prior)
    Ps = []
    for k in range(sep.shape[0]):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]
        Ps.append(P.softmax(Z))
    return y, Ps

for name, e, prior in (("感情の当てやすさの差が大きい・割合は等しい", [0.3, 0.8, 1.5, 2.5], [0.25] * 4),
                       ("感情の当てやすさの差が小さい・割合は等しい", [0.8, 1.0, 1.2, 1.5], [0.25] * 4),
                       ("感情の当てやすさの差が小さい・割合が偏る", [0.8, 1.0, 1.2, 1.5], [0.47, 0.36, 0.10, 0.07]),
                       ("感情の当てやすさの差が大きい・割合が偏る", [0.3, 0.8, 1.5, 2.5], [0.47, 0.36, 0.10, 0.07])):
    sep = np.ones((3, 1)) * np.array(e)
    acc = {}
    for rep in range(20):
        rng = np.random.default_rng(300 + rep)
        yp, Pp = draw(rng, 20000, sep, prior); yt, Pt = draw(rng, 20000, sep, prior)
        Lp, Lt = np.stack([P.centered(p) for p in Pp]), np.stack([P.centered(p) for p in Pt])
        q1p, q1t = P.softmax(Lp.sum(0)), P.softmax(Lt.sum(0))
        target = q1p.mean(0)
        W = P.to_w(np.stack([P.dprime(p, yp)[0] for p in Pp]))
        We = np.repeat(W.mean(0, keepdims=True), 3, 0)
        base = Lt.sum(0).argmax(1)
        r = {"prod": uar(yt, base)}
        for f in ("F3", "F6"):
            Sp, St = P.combine(f, Lp, We, q1p), P.combine(f, Lt, We, q1t)
            dl, _ = P.match_shift(Sp, target)
            pa = (St + dl).argmax(1)
            r[f + "_adj"] = uar(yt, pa)
            r[f + "_share_gap"] = float(np.abs(np.bincount(pa, minlength=4) / len(pa) - np.bincount(base, minlength=4) / len(base)).max())
            r[f + "_changed"] = float((pa != base).mean())
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    g = lambda a: np.mean(np.array(acc[a]) - np.array(acc["prod"]))
    s = lambda a: np.std(np.array(acc[a]) - np.array(acc["prod"]), ddof=1) / np.sqrt(20)
    print(f"{name}：積 {np.mean(acc['prod']):.1f}  共通の重み・F3 の形（adj）−積 {g('F3_adj'):+.2f}±{s('F3_adj'):.2f}（判定の割合の差の最大 {np.mean(acc['F3_share_gap']):.3f}，判定が変わった割合 {np.mean(acc['F3_changed']):.3f}）"
          f"  F6 の形 {g('F6_adj'):+.2f}±{s('F6_adj'):.2f}（{np.mean(acc['F6_share_gap']):.3f}，{np.mean(acc['F6_changed']):.3f}）")
