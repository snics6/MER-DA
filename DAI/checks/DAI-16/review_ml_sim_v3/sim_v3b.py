"""ML の査読役の検算 2（DAI-16 第 3 版）．人工の数だけ．
C：各分類器の確率が正しい事後確率（校正されている）のとき，全ラベルで測った d′ の重みは積を上回るか．
D：仮定 (b) の旗（候補とテストの d′ の差の絶対値の平均 > 0.05）は，ずれが無いときに立つか．
E：K 本（感情ごとに 5 本・25 本）で測った d′ の偏り．"""
import sys
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import importlib.util
_sp = importlib.util.spec_from_file_location("p16c", "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v3/dai16_probe_556a06f.py")
P = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(P)

def uar(y, pred):
    return 100 * float(np.mean([(pred[y == i] == i).mean() for i in range(4)]))

def posterior(rng, n, mu, temp=1.0):
    """特徴 x ~ N(mu[y], I)．確率は正しい事後確率（感情の割合は等しい）を温度 temp で尖らせるか平らにしたもの．"""
    y = rng.integers(0, 4, n)
    x = mu[y] + rng.normal(size=(n, mu.shape[1]))
    ll = -0.5 * ((x[:, None, :] - mu[None]) ** 2).sum(-1)
    return y, P.softmax(ll / temp)

def case_C(name, mus, temps, reps=20, n=20000):
    res = {}
    for rep in range(reps):
        rng = np.random.default_rng(500 + rep)
        st = rng.bit_generator.state
        rp = np.random.default_rng(900 + rep); rt = np.random.default_rng(1900 + rep)
        yp = rp.integers(0, 4, n); yt = rt.integers(0, 4, n)
        Pp, Pt = [], []
        for mu, tp in zip(mus, temps):
            for (y, store, r) in ((yp, Pp, rp), (yt, Pt, rt)):
                x = mu[y] + r.normal(size=(n, mu.shape[1]))
                ll = -0.5 * ((x[:, None, :] - mu[None]) ** 2).sum(-1)
                store.append(P.softmax(ll / tp))
        Lp, Lt = np.stack([P.centered(p) for p in Pp]), np.stack([P.centered(p) for p in Pt])
        q1p, q1t = P.softmax(Lp.sum(0)), P.softmax(Lt.sum(0))
        target = q1p.mean(0)
        D = np.stack([P.dprime(p, yp)[0] for p in Pp])
        W = P.to_w(D)
        r = {"prod": uar(yt, Lt.sum(0).argmax(1)), "lvl_spread": float(D.mean(1).max() - D.mean(1).min())}
        for f in ("F6", "F3", "F2"):
            Sp, St = P.combine(f, Lp, W, q1p), P.combine(f, Lt, W, q1t)
            dl, _ = P.match_shift(Sp, target)
            r[f + "_adj"] = uar(yt, (St + dl).argmax(1)); r[f + "_raw"] = uar(yt, St.argmax(1))
        for k, v in r.items():
            res.setdefault(k, []).append(v)
    d = lambda a: (float(np.mean(np.array(res[a]) - np.array(res["prod"]))), float(np.std(np.array(res[a]) - np.array(res["prod"]), ddof=1) / np.sqrt(reps)))
    print(f"== {name}：積 {np.mean(res['prod']):.1f}  水準の幅 {np.mean(res['lvl_spread']):.2f}  "
          + "  ".join(f"{f}−積 adj {d(f+'_adj')[0]:+.2f}±{d(f+'_adj')[1]:.2f} raw {d(f+'_raw')[0]:+.2f}" for f in ("F6", "F3", "F2")))

rng0 = np.random.default_rng(7)
def mk(strengths, dim=3):
    """感情 i の平均を，大きさ strengths[i] の乱数の向きに置く．"""
    v = rng0.normal(size=(4, dim)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.array(strengths)[:, None]

print("---- C：校正された確率（モダリティは本当の感情を与えると独立）")
mus_lvl = [mk([2.0] * 4), mk([1.2] * 4), mk([0.6] * 4)]
case_C("C1 水準だけが違う・校正", mus_lvl, (1, 1, 1))
mus_int = [mk([2.5, 0.6, 0.6, 1.0]), mk([0.6, 2.5, 0.6, 1.0]), mk([0.6, 0.6, 2.5, 1.0])]
case_C("C2 得意な感情が違う・校正", mus_int, (1, 1, 1))
case_C("C3 得意な感情が違う・3 つとも尖りすぎ（温度 0.3）", mus_int, (0.3, 0.3, 0.3))
case_C("C4 水準だけが違う・弱い 2 つが尖りすぎ（温度 1, 0.3, 0.3）", mus_lvl, (1, 0.3, 0.3))
case_C("C5 得意な感情が違う・3 つとも平らすぎ（温度 3）", mus_int, (3, 3, 3))

print("---- D：候補とテストが同じ分布のときの，d′ の差の絶対値の平均（仮定 (b) の旗の基準 0.05）")
sep = np.array([[1.5, 1.0, 0.8, 1.2], [0.8, 1.4, 1.0, 0.6], [1.0, 0.7, 1.3, 1.1]])
for ntest, prior in ((200, None), (800, None), (3000, None), (3000, np.array([0.47, 0.36, 0.10, 0.07]))):
    vals, sgn = [], []
    for rep in range(200):
        rng = np.random.default_rng(rep)
        def dr(n):
            y = rng.choice(4, size=n, p=prior) if prior is not None else rng.integers(0, 4, n)
            Ps = []
            for k in range(3):
                Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]
                Ps.append(P.softmax(Z))
            return y, Ps
        yp, Pp = dr(8000); yt, Pt = dr(ntest)
        Dp = np.stack([P.dprime(p, yp)[0] for p in Pp]); Dt = np.stack([P.dprime(p, yt)[0] for p in Pt])
        vals.append(np.abs(Dt - Dp).mean()); sgn.append((Dt - Dp).mean())
    print(f"  テスト {ntest} 本（割合 {'等しい' if prior is None else '偏り'}）：差の絶対値の平均 {np.mean(vals):.3f}（0.05 を超えた回 {np.mean(np.array(vals) > 0.05):.2f}）  符号つきの平均 {np.mean(sgn):+.3f}")

print("---- E：K 本で測った d′ の偏り（本当の d′ は大きな標本の値）")
for K in (5, 25):
    for s in (0.0, 0.5, 1.0, 2.0):
        ds = []
        for rep in range(400):
            rng = np.random.default_rng(rep)
            y = np.repeat(np.arange(4), K)
            Z = rng.normal(size=(4 * K, 4)); Z[np.arange(4 * K), y] += s
            ds.append(P.dprime(P.softmax(Z), y)[0].mean())
        rng = np.random.default_rng(1)
        y = rng.integers(0, 4, 40000); Z = rng.normal(size=(40000, 4)); Z[np.arange(40000), y] += s
        print(f"  K={K} sep={s}：K 本の水準の平均 {np.mean(ds):.2f}（標準偏差 {np.std(ds):.2f}），大きな標本の水準 {P.dprime(P.softmax(Z), y)[0].mean():.2f}")
