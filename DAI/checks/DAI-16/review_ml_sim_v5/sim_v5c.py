"""ML の査読役の検算 3（DAI-16 第 5 版）．R0 の規則を，設計と同じ「3 通りのソースの組の平均」に当てる．
1 回 = ソースの表 3 つ（別々に測る）とターゲットの表 1 つ．相関と並べ替えの割合を 3 つで平均し，「0.3 以上かつ 0.2 以下」を満たすかを見る．
分類器 × 感情の成分の大きさ（絶対値の平均）も出す．"""
import sys, importlib.util
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
SP = "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v5"
sp = importlib.util.spec_from_file_location("p16v5", SP + "/dai16_probe_5600b47.py"); P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
rng = np.random.default_rng(23)

def table(n, sep):
    y = rng.integers(0, 4, n)
    out = []
    for k in range(sep.shape[0]):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]
        out.append(P.dprime(P.softmax(Z), y)[0])
    return np.stack(out)

def run(name, seps, sepT, n, reps=60):
    hit, mc, mag = [], [], []
    for _ in range(reps):
        Dt = table(n, sepT)
        cs, ps = [], []
        for sS in seps:
            Ds = table(n, sS)
            cs.append(P.corr(P.dcenter(Ds), P.dcenter(Dt))); ps.append(P.perm_share(Ds, Dt))
        mc.append(np.nanmean(cs)); hit.append(np.nanmean(cs) >= 0.3 and np.nanmean(ps) <= 0.2); mag.append(np.abs(P.dcenter(Dt)).mean())
    print(f"  {name}（{n} 本）：3 つの平均の相関 {np.mean(mc):+.2f}（標準偏差 {np.std(mc):.2f}），規則を満たした割合 {np.mean(hit):.2f}，成分の大きさ {np.mean(mag):.3f}")

e = np.array([0.5, 0.9, 1.4, 2.0]); col = lambda a: np.array(a)[:, None]
sq = lambda r: (col(r) + np.sqrt(e)) ** 2
small = np.array([[0.25, -0.25, 0.0, 0.0], [-0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
mul = lambda a: col(a) * e
for n in (50000, 20000, 5000, 2000):
    run("A 水準 × 当てやすさ（3 つのソースの水準は別々）", [mul([1.4, 1.0, 0.6]), mul([1.2, 1.1, 0.7]), mul([1.0, 1.3, 0.7])], mul([1.3, 1.0, 0.7]), n)
    run("E (水準 ＋ 当てやすさ) の 2 乗（同）", [sq([0.25, 0.0, -0.25]), sq([0.15, 0.1, -0.25]), sq([0.0, 0.25, -0.25])], sq([0.2, 0.0, -0.2]), n)
    run("E2 水準の差が大きい 2 乗の形", [sq([0.5, 0.0, -0.5]), sq([0.4, 0.1, -0.5]), sq([0.3, 0.2, -0.5])], sq([0.45, 0.0, -0.45]), n)
    run("F 水準 × 当てやすさ ＋ 小さな得意の違い（ソース 3 つとターゲットで同じ違い）", [mul([1.4, 1.0, 0.6]) + small, mul([1.2, 1.1, 0.7]) + small, mul([1.0, 1.3, 0.7]) + small], mul([1.3, 1.0, 0.7]) + small, n)
