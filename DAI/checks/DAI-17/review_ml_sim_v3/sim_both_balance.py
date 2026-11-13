"""人工の数：特徴のコサインと分類器の答えのコサインを平均した近さ（both）で引いた上位 100 本が，
どちらの近さの上位 100 本とどれだけ重なるか．2 つのコサインの散らばりの比を変える．MER-DA のデータは読まない．"""
import numpy as np
rng = np.random.default_rng(0)
N, Q, D, L, R, K = 8000, 300, 768, 8, 20, 100
def unit(a): return a / np.linalg.norm(a, axis=1, keepdims=True)
# 特徴：低い次元の構造 R 次元 + 雑音．構造の強さ s_f で上位のコサインの散らばりが変わる
Zs, Zq = rng.normal(size=(N, R)), rng.normal(size=(Q, R))
A = rng.normal(size=(R, D))
for s_f in (0.5, 1.0, 2.0):
    Fs = unit(Zs @ A * s_f / np.sqrt(R) + rng.normal(size=(N, D)) * np.sqrt(1))
    Fq = unit(Zq @ A * s_f / np.sqrt(R) + rng.normal(size=(Q, D)) * np.sqrt(1))
    # 分類器の答え：同じ潜在の一部から作る 8 次元（標準化して長さ 1）
    B = rng.normal(size=(R, L))
    Ls = unit((lambda E: (E - E.mean(0)) / E.std(0))(Zs @ B + 0.5 * rng.normal(size=(N, L))))
    Lq = unit((lambda E: (E - E.mean(0)) / E.std(0))(Zq @ B + 0.5 * rng.normal(size=(Q, L))))
    cf, cl = Fq @ Fs.T, Lq @ Ls.T
    cb = (cf + cl) / 2
    top = lambda C: np.argsort(-C, 1)[:, :K]
    tb, tf, tl = top(cb), top(cf), top(cl)
    ov = lambda a, b: np.mean([len(np.intersect1d(x, y)) / K for x, y in zip(a, b)])
    sd_f = cf.std(1).mean(); sd_l = cl.std(1).mean()
    print(f"特徴の構造の強さ {s_f}：コサインの散らばり（発話ごとの標準偏差の平均）特徴 {sd_f:.3f}，答え {sd_l:.3f}．"
          f"both の上位 100 本と重なる割合：特徴だけ {ov(tb, tf):.2f}，答えだけ {ov(tb, tl):.2f}．"
          f"both の上位の特徴のコサインの平均 {np.take_along_axis(cf, tb, 1).mean():.3f}（特徴だけの上位 {np.take_along_axis(cf, tf, 1).mean():.3f}），"
          f"答えのコサインの平均 {np.take_along_axis(cl, tb, 1).mean():.3f}（答えだけの上位 {np.take_along_axis(cl, tl, 1).mean():.3f}）")
