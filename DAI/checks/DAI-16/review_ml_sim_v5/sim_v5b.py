"""ML の査読役の検算 2（DAI-16 第 5 版．コミット 5600b47）．人工の数だけ．
第 5 版の対照（noint：水準・共通の感情の当てやすさ・その掛け算だけを残した重み）で，「版 − 対照」が，
(i) 水準と感情の当てやすさだけでできた分類器では 0 に近く，(ii) 得意な感情が本当に違う分類器では正になるかを，adj・raw・ob の 3 つの層で確かめる．
ob の層は台本と同じに，候補の全ラベルで fit_wb（罰則なし）の切片を決めて raw の点に足す．"""
import sys, importlib.util
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
SP = "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v5"
sp = importlib.util.spec_from_file_location("p16v5", SP + "/dai16_probe_5600b47.py"); P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

def uar(y, pred):
    return 100 * float(np.mean([(pred[y == i] == i).mean() for i in range(4)]))

def draw(rng, n, sep):
    y = rng.integers(0, 4, n)
    Ps = []
    for k in range(sep.shape[0]):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]
        Ps.append(P.softmax(Z))
    return y, Ps

def run(name, sep, n=20000, reps=12):
    acc = {}
    for rep in range(reps):
        rng = np.random.default_rng(600 + rep)
        yp, Pp = draw(rng, n, sep); yt, Pt = draw(rng, n, sep)
        Lp, Lt = np.stack([P.centered(p) for p in Pp]), np.stack([P.centered(p) for p in Pt])
        q1p, q1t = P.softmax(Lp.sum(0)), P.softmax(Lt.sum(0)); target = q1p.mean(0)
        W = P.to_w(np.stack([P.dprime(p, yp)[0] for p in Pp]))
        r = {}
        for tag, f, Wx in (("prod", "F2", np.ones((3, 4))), ("F6", "F6", W), ("F6n", "F6", P.noint(W)), ("F3", "F3", W), ("F3n", "F3", P.noint(W)), ("F2", "F2", W)):
            Sp, St = P.combine(f, Lp, Wx, q1p), P.combine(f, Lt, Wx, q1t)
            dl, _ = P.match_shift(Sp, target)
            bo = P.fit_wb(Sp[None], yp, np.ones(1), 0.0, fit_w=False, fit_b=True)[1]
            r[tag + "_raw"], r[tag + "_adj"], r[tag + "_ob"] = uar(yt, St.argmax(1)), uar(yt, (St + dl).argmax(1)), uar(yt, (St + bo).argmax(1))
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    d = lambda a, b: (float(np.mean(np.array(acc[a]) - np.array(acc[b]))), float(np.std(np.array(acc[a]) - np.array(acc[b]), ddof=1) / np.sqrt(reps)))
    print(f"== {name}（積 {np.mean(acc['prod_raw']):.1f}）")
    for lay in ("adj", "raw", "ob"):
        print(f"  {lay}：F6−積 {d('F6_'+lay,'prod_'+lay)[0]:+.2f}  F6−対照 {d('F6_'+lay,'F6n_'+lay)[0]:+.2f}±{d('F6_'+lay,'F6n_'+lay)[1]:.2f}  対照−積 {d('F6n_'+lay,'prod_'+lay)[0]:+.2f}"
              f" | F3−積 {d('F3_'+lay,'prod_'+lay)[0]:+.2f}  F3−対照 {d('F3_'+lay,'F3n_'+lay)[0]:+.2f}±{d('F3_'+lay,'F3n_'+lay)[1]:.2f}  対照−積 {d('F3n_'+lay,'prod_'+lay)[0]:+.2f}"
              f" | F6−F3 {d('F6_'+lay,'F3_'+lay)[0]:+.2f}  (F6−対照)−(F3−対照) {d('F6_'+lay,'F6n_'+lay)[0]-d('F3_'+lay,'F3n_'+lay)[0]:+.2f}")

e = np.array([0.5, 0.9, 1.4, 2.0]); col = lambda a: np.array(a)[:, None]
run("1 3 つとも同じ作り（感情の当てやすさだけ．差が大きい）", np.ones((3, 1)) * np.array([0.3, 0.8, 1.5, 2.5]))
run("A 水準 × 当てやすさ", col([1.3, 1.0, 0.7]) * e)
run("C 水準 ＋ 当てやすさ", col([0.3, 0.0, -0.3]) + e)
run("E (水準 ＋ 当てやすさ) の 2 乗", (col([0.2, 0.0, -0.2]) + np.sqrt(e)) ** 2)
run("D 得意な感情が違う", np.array([[2.5, 0.3, 0.3, 0.8], [0.3, 2.5, 0.3, 0.8], [0.3, 0.3, 2.5, 0.8]]))
run("F 水準 × 当てやすさ ＋ 小さな得意の違い", col([1.3, 1.0, 0.7]) * e + np.array([[0.5, -0.5, 0.0, 0.0], [-0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
