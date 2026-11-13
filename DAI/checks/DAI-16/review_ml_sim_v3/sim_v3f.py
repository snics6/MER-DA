"""ML の査読役の検算 6：分類器 × 感情の成分から，「水準の差 × 感情の当てやすさの差」の形の成分（行の効果 × 列の効果．1 自由度）も除くと，
水準と感情の当てやすさだけでできた表（足し算の形・掛け算の形・その間の形）で相関と UAR の差が 0 に戻るか．台本の関数は 556a06f の版．"""
import sys, importlib.util
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
SP = "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v3"
sp = importlib.util.spec_from_file_location("p16c", SP + "/dai16_probe_556a06f.py"); P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

def uar(y, pred):
    return 100 * float(np.mean([(pred[y == i] == i).mean() for i in range(4)]))

def draw(rng, n, sep):
    y = rng.integers(0, 4, n)
    Ps = []
    for k in range(sep.shape[0]):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]
        Ps.append(P.softmax(Z))
    return y, Ps

def fit_add(X):                      # 行の平均 ＋ 列の平均 − 全体の平均（第 4 版の noint と同じ式．切る前）
    return X.mean(1, keepdims=True) + X.mean(0, keepdims=True) - X.mean()

def fit_tukey(X):                    # 上に，行の効果 × 列の効果の成分を最小 2 乗で足す
    r, c = X.mean(1, keepdims=True) - X.mean(), X.mean(0, keepdims=True) - X.mean()
    Pm = r * c
    lam = float(((X - fit_add(X)) * Pm).sum() / max(float((Pm ** 2).sum()), 1e-12))
    return fit_add(X) + lam * Pm

def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else float("nan")

def run(name, sepS, sepT, n=20000, reps=20):
    acc = {}
    for rep in range(reps):
        rng = np.random.default_rng(9000 + rep)
        ys, Ps = draw(rng, n, sepS); yp, Pp = draw(rng, n, sepT); yt, Pt = draw(rng, n, sepT)
        Lp, Lt = np.stack([P.centered(p) for p in Pp]), np.stack([P.centered(p) for p in Pt])
        q1p, q1t = P.softmax(Lp.sum(0)), P.softmax(Lt.sum(0)); target = q1p.mean(0)
        Ds = np.stack([P.dprime(p, ys)[0] for p in Ps]); Dt = np.stack([P.dprime(p, yp)[0] for p in Pp])
        W = P.to_w(Dt)
        r = {"c_add": corr(Ds - fit_add(Ds), Dt - fit_add(Dt)), "c_tuk": corr(Ds - fit_tukey(Ds), Dt - fit_tukey(Dt)),
             "resid_add": float(np.abs(Dt - fit_add(Dt)).mean()), "resid_tuk": float(np.abs(Dt - fit_tukey(Dt)).mean())}
        for tag, f, Wx in (("F6", "F6", W), ("F6a", "F6", np.maximum(0, fit_add(W))), ("F6t", "F6", np.maximum(0, fit_tukey(W))),
                           ("F3", "F3", W), ("F3a", "F3", np.maximum(0, fit_add(W))), ("F3t", "F3", np.maximum(0, fit_tukey(W)))):
            Sp, St = P.combine(f, Lp, Wx, q1p), P.combine(f, Lt, Wx, q1t)
            dl, _ = P.match_shift(Sp, target)
            r[tag] = uar(yt, (St + dl).argmax(1))
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    m = lambda k: float(np.nanmean(acc[k]))
    d = lambda a, b: (float(np.mean(np.array(acc[a]) - np.array(acc[b]))), float(np.std(np.array(acc[a]) - np.array(acc[b]), ddof=1) / np.sqrt(reps)))
    print(f"{name}：相関 足し算の形 {m('c_add'):+.2f}（残りの大きさ {m('resid_add'):.3f}）  積の形も除く {m('c_tuk'):+.2f}（{m('resid_tuk'):.3f}）"
          f"  | adj：F6−対照(足し算) {d('F6','F6a')[0]:+.2f}±{d('F6','F6a')[1]:.2f}  F6−対照(積の形も) {d('F6','F6t')[0]:+.2f}±{d('F6','F6t')[1]:.2f}"
          f"  F3−対照(足し算) {d('F3','F3a')[0]:+.2f}±{d('F3','F3a')[1]:.2f}  F3−対照(積の形も) {d('F3','F3t')[0]:+.2f}±{d('F3','F3t')[1]:.2f}")

e = np.array([0.5, 0.9, 1.4, 2.0]); col = lambda a: np.array(a)[:, None]
run("A 掛け算の形・水準の順は同じ", col([1.4, 1.0, 0.6]) * e, col([1.3, 1.0, 0.7]) * e)
run("B 掛け算の形・水準の順が逆", col([1.4, 1.0, 0.6]) * e, col([0.6, 1.0, 1.4]) * e)
run("C 足し算の形", col([0.4, 0.0, -0.4]) + e, col([0.3, 0.0, -0.3]) + e)
sq = lambda r: (col(r) + np.sqrt(e)) ** 2
run("E 間の形（(水準＋当てやすさ)の 2 乗）", sq([0.25, 0.0, -0.25]), sq([0.2, 0.0, -0.2]))
inter = np.array([[2.5, 0.3, 0.3, 0.8], [0.3, 2.5, 0.3, 0.8], [0.3, 0.3, 2.5, 0.8]])
run("D 得意な感情が違う（同じ）", inter, inter)
inter2 = col([1.3, 1.0, 0.7]) * e + np.array([[0.5, -0.5, 0.0, 0.0], [-0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
run("F 掛け算の形 ＋ 得意な感情の違い（同じ）", inter2, inter2)
inter3 = col([1.3, 1.0, 0.7]) * e + np.array([[-0.5, 0.5, 0.0, 0.0], [0.5, -0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
run("G 掛け算の形は同じ，得意な感情の違いはソースとターゲットで逆", inter2, inter3)
