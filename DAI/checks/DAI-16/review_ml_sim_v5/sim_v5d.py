"""ML の査読役の検算 4（DAI-16 第 5 版）．水準と感情の当てやすさだけで決まり，どの感情でも分類器の強さの順が同じ表のうち，
足し算とも掛け算とも違う形（2 乗の形，頭打ち，弱い分類器が 0 に張りつく形）で，R0 の規則（3 通りのソースの組の平均）がどれだけ満たされるか．
あわせて，順位で見る量を出す：感情ごとに最も強い分類器がソースとターゲットで合う割合と，ソースで水準が最も高い分類器を 4 感情すべてに当てたときに合う割合．"""
import sys, importlib.util
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
SP = "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v5"
sp = importlib.util.spec_from_file_location("p16v5", SP + "/dai16_probe_5600b47.py"); P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
rng = np.random.default_rng(31)

def table(n, sep):
    y = rng.integers(0, 4, n)
    out = []
    for k in range(sep.shape[0]):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]
        out.append(P.dprime(P.softmax(Z), y)[0])
    return np.stack(out)

def run(name, seps, sepT, n, reps=60):
    hit, mc, mag, ae, al, mx, nrev = [], [], [], [], [], [], []
    for _ in range(reps):
        Dt = table(n, sepT)
        cs, ps, a_e, a_l = [], [], [], []
        for sS in seps:
            Ds = table(n, sS)
            cs.append(P.corr(P.dcenter(Ds), P.dcenter(Dt))); ps.append(P.perm_share(Ds, Dt))
            a_e.append(float((Ds.argmax(0) == Dt.argmax(0)).mean())); a_l.append(float((Ds.mean(1).argmax() == Dt.argmax(0)).mean()))
        mc.append(np.nanmean(cs)); hit.append(np.nanmean(cs) >= 0.3 and np.nanmean(ps) <= 0.2); mag.append(np.abs(P.dcenter(Dt)).mean())
        ae.append(np.mean(a_e)); al.append(np.mean(a_l)); mx.append(Dt.max()); nrev.append(len(set(Dt.argmax(0).tolist())))
    print(f"  {name}（{n} 本）：相関 {np.mean(mc):+.2f}，規則を満たした割合 {np.mean(hit):.2f}，成分の大きさ {np.mean(mag):.3f}，表の最大 {np.mean(mx):.2f}；"
          f"感情ごとの最強の分類器が合う割合 {np.mean(ae):.2f}，水準の最強を全感情に当てた割合 {np.mean(al):.2f}，ターゲットで最強になる分類器の数 {np.mean(nrev):.2f}")

col = lambda a: np.array(a)[:, None]
e = np.array([0.5, 0.9, 1.4, 2.0]); es = np.array([0.3, 0.5, 0.8, 1.2])
sq = lambda r, ee: (col(r) + np.sqrt(ee)) ** 2
small = np.array([[0.25, -0.25, 0.0, 0.0], [-0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
for n in (20000, 5000):
    run("E2 2 乗の形・頭打ちあり（最大の信頼度が上限 2.90 に当たる）", [sq([0.5, 0.0, -0.5], e), sq([0.4, 0.1, -0.5], e), sq([0.3, 0.2, -0.5], e)], sq([0.45, 0.0, -0.45], e), n)
    run("E3 2 乗の形・頭打ちなし（水準の差は大きい）", [sq([0.4, 0.0, -0.4], es), sq([0.3, 0.1, -0.4], es), sq([0.2, 0.2, -0.4], es)], sq([0.35, 0.0, -0.35], es), n)
    run("E4 指数の形・頭打ちなし", [0.5 * np.exp(col(r) + np.log(es) + 0.6) for r in ([0.5, 0.0, -0.5], [0.4, 0.1, -0.5], [0.3, 0.2, -0.5])], 0.5 * np.exp(col([0.45, 0.0, -0.45]) + np.log(es) + 0.6), n)
    run("E5 平方根の形（差が縮む向き）", [np.sqrt(np.maximum(0, col(r) + 2 * es)) for r in ([0.5, 0.0, -0.5], [0.4, 0.1, -0.5], [0.3, 0.2, -0.5])], np.sqrt(np.maximum(0, col([0.45, 0.0, -0.45]) + 2 * es)), n)
    run("H 掛け算の形・弱い分類器が 0 に張りつく", [col([1.3, 0.8, 0.03]) * e, col([1.1, 0.9, 0.03]) * e, col([0.9, 1.2, 0.03]) * e], col([1.2, 0.8, 0.03]) * e, n)
    run("F 掛け算の形 ＋ 小さな得意の違い（同じ）", [col([1.4, 1.0, 0.6]) * e + small, col([1.2, 1.1, 0.7]) * e + small, col([1.0, 1.3, 0.7]) * e + small], col([1.3, 1.0, 0.7]) * e + small, n)
