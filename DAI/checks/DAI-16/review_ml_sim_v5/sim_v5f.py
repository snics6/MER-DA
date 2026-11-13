"""ML の査読役の検算 6（DAI-16 第 5 版）．直し方の例に書いた順位の量の確かめ．
感情ごとの一致：感情ごとに最も強い分類器がソースの表とターゲットの表で同じ割合．水準だけの一致：ソースで水準が最も高い分類器を 4 感情すべてに当てたときに合う割合．
1 回 = ソースの表 3 つとターゲットの表 1 つ（3 つで平均）．差（感情ごと − 水準だけ）の平均と，0.1 以上になる割合，R0 の規則と両方を満たす割合を出す．"""
import sys, importlib.util
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
SP = "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v5"
sp = importlib.util.spec_from_file_location("p16v5", SP + "/dai16_probe_5600b47.py"); P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
rng = np.random.default_rng(41)

def table(n, sep):
    y = rng.integers(0, 4, n)
    out = []
    for k in range(sep.shape[0]):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]
        out.append(P.dprime(P.softmax(Z), y)[0])
    return np.stack(out)

def run(name, seps, sepT, n, reps=80):
    gap, r0, both = [], [], []
    for _ in range(reps):
        Dt = table(n, sepT)
        cs, ps, g = [], [], []
        for sS in seps:
            Ds = table(n, sS)
            cs.append(P.corr(P.dcenter(Ds), P.dcenter(Dt))); ps.append(P.perm_share(Ds, Dt))
            g.append(float((Ds.argmax(0) == Dt.argmax(0)).mean()) - float((Ds.mean(1).argmax() == Dt.argmax(0)).mean()))
        h = bool(np.nanmean(cs) >= 0.3 and np.nanmean(ps) <= 0.2)
        gap.append(np.mean(g)); r0.append(h); both.append(h and np.mean(g) >= 0.1)
    print(f"  {name}（{n} 本）：差の平均 {np.mean(gap):+.3f}（標準偏差 {np.std(gap):.3f}），差が 0.1 以上の割合 {np.mean(np.array(gap) >= 0.1):.2f}，R0 の規則 {np.mean(r0):.2f}，両方 {np.mean(both):.2f}")

col = lambda a: np.array(a)[:, None]
e = np.array([0.5, 0.9, 1.4, 2.0]); es = np.array([0.3, 0.5, 0.8, 1.2])
sq = lambda r, ee: (col(r) + np.sqrt(ee)) ** 2
small = np.array([[0.25, -0.25, 0.0, 0.0], [-0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
inter = np.array([[2.5, 0.3, 0.3, 0.8], [0.3, 2.5, 0.3, 0.8], [0.3, 0.3, 2.5, 0.8]])
for n in (20000, 5000):
    run("E3 2 乗の形（ソースの 1 つは水準が並ぶ）", [sq([0.4, 0.0, -0.4], es), sq([0.3, 0.1, -0.4], es), sq([0.2, 0.2, -0.4], es)], sq([0.35, 0.0, -0.35], es), n)
    run("E3′ 2 乗の形（水準の順ははっきり）", [sq([0.4, 0.0, -0.4], es), sq([0.35, 0.05, -0.4], es), sq([0.3, 0.0, -0.3], es)], sq([0.35, 0.0, -0.35], es), n)
    run("E5 平方根の形", [np.sqrt(np.maximum(0, col(r) + 2 * es)) for r in ([0.5, 0.0, -0.5], [0.4, 0.1, -0.5], [0.3, 0.2, -0.5])], np.sqrt(np.maximum(0, col([0.45, 0.0, -0.45]) + 2 * es)), n)
    run("E2 2 乗の形・上限に当たる", [sq([0.5, 0.0, -0.5], e), sq([0.4, 0.1, -0.5], e), sq([0.3, 0.2, -0.5], e)], sq([0.45, 0.0, -0.45], e), n)
    run("F 掛け算の形 ＋ 小さな得意の違い", [col([1.4, 1.0, 0.6]) * e + small, col([1.2, 1.1, 0.7]) * e + small, col([1.0, 1.3, 0.7]) * e + small], col([1.3, 1.0, 0.7]) * e + small, n)
    run("D 得意な感情が違う", [inter, inter, inter], inter, n)
