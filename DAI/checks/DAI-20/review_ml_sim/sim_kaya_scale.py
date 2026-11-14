"""ML の査読者の検算（DAI-20．人工の数だけ．MER-DA のデータと結果は読まない）．
(1) 「列ごとに和が 1」の等しい重みの表（全部 1/3）を，提案と同じ「候補の上の確率の平均を積に合わせる切片のずらし」に通すと，積と同じ判定になるか．
(2) K 本（感情ごとに K 本．1 組を抜いた確率の代わりに独立な発話）の上の UAR で 1,000 個の表から選ぶと，等しい重みの表が残る割合と，同点の数．
match_shift は DAI/exp/dai16_probe.py の同名の関数と同じ式．"""
import numpy as np
from scipy.special import logsumexp

ADJ_ITERS, ADJ_TOL = 3000, 1e-6


def norm_log(L):
    return L - logsumexp(L, axis=1, keepdims=True)


def softmax(L):
    return np.exp(norm_log(L))


def centered(P):
    L = np.log(np.clip(P, 1e-300, None))
    return L - L.mean(1, keepdims=True)


def match_shift(Lp, target):
    d = np.zeros(4)
    lt = np.log(np.clip(target, 1e-12, None))
    for _ in range(ADJ_ITERS):
        m = softmax(Lp + d).mean(0)
        if np.abs(m - target).max() < ADJ_TOL * 1e-2:
            break
        d = d + lt - np.log(np.clip(m, 1e-12, None))
        d = d - d.mean()
    return d


def uar(y, p):
    return 100 * np.mean([(p[y == c] == c).mean() for c in range(4) if (y == c).any()])


def make(rng, n, prior, strength, bias):
    """3 つの分類器の確率．strength：(3, 4) 分類器 × 感情の強さ．bias：(3, 4) 切片の癖．"""
    y = rng.choice(4, size=n, p=prior)
    P = []
    for k in range(3):
        Z = rng.normal(size=(n, 4)) + bias[k]
        Z[np.arange(n), y] += strength[k, y]
        P.append(softmax(Z))
    return y, P


def run(seed, prior, label):
    rng = np.random.default_rng(seed)
    strength = np.array([[1.6, 1.0, 1.8, 1.4], [0.8, 1.6, 0.6, 1.0], [1.4, 1.2, 1.0, 1.8]])
    bias = rng.normal(scale=0.5, size=(3, 4))
    yp, Pp = make(rng, 4000, prior, strength, bias)       # 候補
    yt, Pt = make(rng, 4000, prior, strength, bias)       # テスト
    Lp, Lt = np.stack([centered(p) for p in Pp]), np.stack([centered(p) for p in Pt])
    prod_p, prod_t = norm_log(Lp.sum(0)), norm_log(Lt.sum(0))
    target = np.exp(prod_p).mean(0)
    out = {}
    for name, w in (("全部 1（積と同じ尺度）", 1.0), ("全部 1/3（列の和が 1）", 1 / 3)):
        Sp, St = norm_log(w * Lp.sum(0)), norm_log(w * Lt.sum(0))
        d = match_shift(Sp, target)
        pa = (St + d).argmax(1)
        out[name] = (float((pa != prod_t.argmax(1)).mean()), uar(yt, pa) - uar(yt, prod_t.argmax(1)), d)
    print(f"[{label}] 種 {seed}  積の UAR {uar(yt, prod_t.argmax(1)):.2f}")
    for name, (dis, du, d) in out.items():
        print(f"    {name}：積と判定が違う割合 {100 * dis:.2f}%，UAR の差 {du:+.2f} 点，切片のずれ {np.round(d, 3)}")
    return out, (rng, strength, bias, prior, Lt, yt)


def ties(seed, K, prior, label, n_tab=1000):
    rng = np.random.default_rng(1000 + seed)
    strength = np.array([[1.6, 1.0, 1.8, 1.4], [0.8, 1.6, 0.6, 1.0], [1.4, 1.2, 1.0, 1.8]])
    bias = rng.normal(scale=0.5, size=(3, 4))
    # K 本：感情ごとに K 本
    yk = np.repeat(np.arange(4), K)
    Pk = []
    for k in range(3):
        Z = rng.normal(size=(4 * K, 4)) + bias[k]
        Z[np.arange(4 * K), yk] += strength[k, yk]
        Pk.append(softmax(Z))
    Lk = np.stack([centered(p) for p in Pk])
    yt, Pt = make(rng, 4000, prior, strength, bias)
    Lt = np.stack([centered(p) for p in Pt])
    W = rng.dirichlet(np.ones(3), size=(n_tab, 4)).transpose(0, 2, 1) * 3.0     # (表, 分類器, 感情)．列の和が 3
    W[0] = 1.0                                                                  # 等しい重みの表
    uk = np.array([uar(yk, np.einsum("ky,kny->ny", w, Lk).argmax(1)) for w in W])
    best = uk.max()
    tied = np.flatnonzero(uk >= best - 1e-9)
    pick = 0 if 0 in tied else int(tied[0])
    ut = lambda w: uar(yt, np.einsum("ky,kny->ny", w, Lt).argmax(1))
    return dict(eq_kept=(pick == 0), n_tied=len(tied), k_uar_eq=uk[0], k_uar_best=best, test_eq=ut(W[0]), test_pick=ut(W[pick]),
                test_tied_sd=float(np.std([ut(W[i]) for i in tied[:50]])))


if __name__ == "__main__":
    for label, prior in (("割合がそろう", np.full(4, 0.25)), ("割合が偏る", np.array([0.55, 0.20, 0.10, 0.15]))):
        res = [run(s, prior, label)[0] for s in range(5)]
        a = np.array([[r[n][0] for r in res] for n in res[0]])
        b = np.array([[r[n][1] for r in res] for n in res[0]])
        for i, n in enumerate(res[0]):
            print(f"  ==> [{label}] {n}：判定が違う割合 平均 {100 * a[i].mean():.2f}%（{100 * a[i].min():.2f}〜{100 * a[i].max():.2f}），UAR の差 平均 {b[i].mean():+.2f}（{b[i].min():+.2f}〜{b[i].max():+.2f}）")
    for K in (5, 10, 25):
        rs = [ties(s, K, np.array([0.55, 0.20, 0.10, 0.15]), "偏る") for s in range(200)]
        print(f"K={K}：等しい重みが残る割合 {np.mean([r['eq_kept'] for r in rs]):.2f}，同点の表の数の中央値 {np.median([r['n_tied'] for r in rs]):.0f}，"
              f"K 本の UAR（等しい重み → 最大） {np.mean([r['k_uar_eq'] for r in rs]):.1f} → {np.mean([r['k_uar_best'] for r in rs]):.1f}，"
              f"テストの UAR（等しい重み → 選んだ表） {np.mean([r['test_eq'] for r in rs]):.2f} → {np.mean([r['test_pick'] for r in rs]):.2f}，"
              f"同点の表どうしのテストの UAR の標準偏差の平均 {np.mean([r['test_tied_sd'] for r in rs]):.2f}")
