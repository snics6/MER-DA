"""ML の査読役の検算 5（DAI-16 第 5 版）．検算 4 で R0 の規則を満たした「水準と当てやすさだけの曲がった表」（2 乗の形・平方根の形・頭打ち）で，
UAR の「版 − 対照」（M_F6 − M_F6n，M_F3 − M_F3n に当たる差）が 0 に近いままかを確かめる．重みは候補の全ラベルで測った信頼度から作る．"""
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
        rng = np.random.default_rng(800 + rep)
        yp, Pp = draw(rng, n, sep); yt, Pt = draw(rng, n, sep)
        Lp, Lt = np.stack([P.centered(p) for p in Pp]), np.stack([P.centered(p) for p in Pt])
        q1p, q1t = P.softmax(Lp.sum(0)), P.softmax(Lt.sum(0)); target = q1p.mean(0)
        D = np.stack([P.dprime(p, yp)[0] for p in Pp]); W = P.to_w(D)
        r = {"mag": float(np.abs(P.dcenter(D)).mean()), "wmag": float(np.abs(W - P.noint(W)).mean())}
        for tag, f, Wx in (("prod", "F2", np.ones((3, 4))), ("F6", "F6", W), ("F6n", "F6", P.noint(W)), ("F3", "F3", W), ("F3n", "F3", P.noint(W))):
            Sp, St = P.combine(f, Lp, Wx, q1p), P.combine(f, Lt, Wx, q1t)
            dl, _ = P.match_shift(Sp, target)
            r[tag + "_raw"], r[tag + "_adj"] = uar(yt, St.argmax(1)), uar(yt, (St + dl).argmax(1))
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    d = lambda a, b: (float(np.mean(np.array(acc[a]) - np.array(acc[b]))), float(np.std(np.array(acc[a]) - np.array(acc[b]), ddof=1) / np.sqrt(reps)))
    print(f"{name}：成分の大きさ {np.mean(acc['mag']):.3f}（重みでは {np.mean(acc['wmag']):.3f}） | adj：F6−積 {d('F6_adj','prod_adj')[0]:+.2f}  F6−対照 {d('F6_adj','F6n_adj')[0]:+.2f}±{d('F6_adj','F6n_adj')[1]:.2f}"
          f"  F3−積 {d('F3_adj','prod_adj')[0]:+.2f}  F3−対照 {d('F3_adj','F3n_adj')[0]:+.2f}±{d('F3_adj','F3n_adj')[1]:.2f} | raw：F6−対照 {d('F6_raw','F6n_raw')[0]:+.2f}±{d('F6_raw','F6n_raw')[1]:.2f}  F3−対照 {d('F3_raw','F3n_raw')[0]:+.2f}±{d('F3_raw','F3n_raw')[1]:.2f}")

col = lambda a: np.array(a)[:, None]
e = np.array([0.5, 0.9, 1.4, 2.0]); es = np.array([0.3, 0.5, 0.8, 1.2])
run("E2 2 乗の形・頭打ちあり", (col([0.45, 0.0, -0.45]) + np.sqrt(e)) ** 2)
run("E3 2 乗の形・頭打ちなし", (col([0.35, 0.0, -0.35]) + np.sqrt(es)) ** 2)
run("E5 平方根の形", np.sqrt(np.maximum(0, col([0.45, 0.0, -0.45]) + 2 * es)))
run("F 掛け算の形 ＋ 小さな得意の違い", col([1.3, 1.0, 0.7]) * e + np.array([[0.25, -0.25, 0.0, 0.0], [-0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
