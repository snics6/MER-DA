"""ML の査読役の検算 4（DAI-16 第 3 版）．人工の数だけ．台本はコミット 556a06f の版を取り出して読み込む．
問い：「分類器 × 感情の成分を消した対照」と「その成分だけの相関」を，足し算の形（行の平均＋列の平均−全体の平均）で作る場合と，
掛け算の形（行の平均×列の平均÷全体の平均）で作る場合で，読みが変わるか．"""
import sys, importlib.util
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
SP = "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v3"
spec = importlib.util.spec_from_file_location("p16c", SP + "/dai16_probe_556a06f.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

def uar(y, pred):
    return 100 * float(np.mean([(pred[y == i] == i).mean() for i in range(4)]))

def draw(rng, n, sep):
    y = rng.integers(0, 4, n)
    Ps = []
    for k in range(sep.shape[0]):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]
        Ps.append(P.softmax(Z))
    return y, Ps

def n_add(W):
    return np.maximum(0.0, W.mean(1, keepdims=True) + W.mean(0, keepdims=True) - W.mean())

def n_mul(W):
    return W.mean(1, keepdims=True) * W.mean(0, keepdims=True) / W.mean()

def dc_add(D):
    return D - D.mean(1, keepdims=True) - D.mean(0, keepdims=True) + D.mean()

def dc_mul(D):
    return D / (D.mean(1, keepdims=True) * D.mean(0, keepdims=True) / D.mean()) - 1.0

def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

def run(name, sepS, sepT, reps=20, n=20000):
    acc = {}
    for rep in range(reps):
        rng = np.random.default_rng(4000 + rep)
        ys, Ps = draw(rng, n, sepS); yp, Pp = draw(rng, n, sepT); yt, Pt = draw(rng, n, sepT)
        Lp, Lt = np.stack([P.centered(p) for p in Pp]), np.stack([P.centered(p) for p in Pt])
        q1p, q1t = P.softmax(Lp.sum(0)), P.softmax(Lt.sum(0))
        target = q1p.mean(0)
        Ds = np.stack([P.dprime(p, ys)[0] for p in Ps]); Dt = np.stack([P.dprime(p, yp)[0] for p in Pp])
        W = P.to_w(Dt)
        r = {"prof": corr(Ds - Ds.mean(1, keepdims=True), Dt - Dt.mean(1, keepdims=True)), "int_add": corr(dc_add(Ds), dc_add(Dt)), "int_mul": corr(dc_mul(Ds), dc_mul(Dt)),
             "lvl": corr(Ds.mean(1), Dt.mean(1)), "emo": corr(Ds.mean(0), Dt.mean(0)), "prod": uar(yt, Lt.sum(0).argmax(1))}
        preds = {}
        for tag, f, Wx in (("F6", "F6", W), ("F6a", "F6", n_add(W)), ("F6m", "F6", n_mul(W)), ("F3", "F3", W), ("F3a", "F3", n_add(W)), ("F3m", "F3", n_mul(W)), ("F2", "F2", W)):
            Sp, St = P.combine(f, Lp, Wx, q1p), P.combine(f, Lt, Wx, q1t)
            dl, _ = P.match_shift(Sp, target)
            r[tag + "_adj"] = uar(yt, (St + dl).argmax(1)); r[tag + "_raw"] = uar(yt, St.argmax(1)); preds[tag] = St.argmax(1)
        r["F3m_raw_eq_F2_raw"] = float((preds["F3m"] == preds["F2"]).mean())
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    m = lambda k: float(np.mean(acc[k]))
    d = lambda a, b: (float(np.mean(np.array(acc[a]) - np.array(acc[b]))), float(np.std(np.array(acc[a]) - np.array(acc[b]), ddof=1) / np.sqrt(reps)))
    print(f"\n== {name}")
    print(f"  相関：行の平均だけ引く（設計）{m('prof'):+.2f}  足し算の形で両方引く {m('int_add'):+.2f}  掛け算の形 {m('int_mul'):+.2f}  （水準の相関 {m('lvl'):+.2f}，感情の平均の相関 {m('emo'):+.2f}）")
    for lay in ("adj", "raw"):
        print(f"  層 {lay}：F6−積 {d('F6_'+lay,'prod')[0]:+.2f}  F6−F2 {d('F6_'+lay,'F2_'+lay)[0]:+.2f}  F6−対照(足し算) {d('F6_'+lay,'F6a_'+lay)[0]:+.2f}±{d('F6_'+lay,'F6a_'+lay)[1]:.2f}  F6−対照(掛け算) {d('F6_'+lay,'F6m_'+lay)[0]:+.2f}±{d('F6_'+lay,'F6m_'+lay)[1]:.2f}"
              f"  | F3−積 {d('F3_'+lay,'prod')[0]:+.2f}  F3−F2 {d('F3_'+lay,'F2_'+lay)[0]:+.2f}  F3−対照(足し算) {d('F3_'+lay,'F3a_'+lay)[0]:+.2f}±{d('F3_'+lay,'F3a_'+lay)[1]:.2f}  F3−対照(掛け算) {d('F3_'+lay,'F3m_'+lay)[0]:+.2f}±{d('F3_'+lay,'F3m_'+lay)[1]:.2f}")
    print(f"  F3 の掛け算の対照の raw の判定が F2 の raw の判定と同じ割合 {m('F3m_raw_eq_F2_raw'):.6f}")

e = np.array([0.5, 0.9, 1.4, 2.0])
col = lambda a: np.array(a)[:, None]
run("場合 A：水準 × 感情の当てやすさ（掛け算の形．どの感情でもモダリティの比は同じ）．水準の順はソースとターゲットで同じ", col([1.4, 1.0, 0.6]) * e, col([1.3, 1.0, 0.7]) * e)
run("場合 B：場合 A で，水準の順がソースとターゲットで逆", col([1.4, 1.0, 0.6]) * e, col([0.6, 1.0, 1.4]) * e)
run("場合 C：水準 ＋ 感情の当てやすさ（足し算の形）．順は同じ", col([0.4, 0.0, -0.4]) + e, col([0.3, 0.0, -0.3]) + e)
inter = np.array([[2.5, 0.3, 0.3, 0.8], [0.3, 2.5, 0.3, 0.8], [0.3, 0.3, 2.5, 0.8]])
run("場合 D：モダリティごとに得意な感情が違う（ソースとターゲットで同じ）", inter, inter)
