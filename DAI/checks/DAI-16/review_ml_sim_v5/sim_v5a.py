"""ML の査読役の検算（DAI-16 第 5 版．コミット 5600b47）．人工の数だけ．MER-DA のデータと結果は読まない．
1：でたらめな 3 × 4 の表での R0 の量（設計 137 行の数：標準偏差 約 0.40，0.3 以上の割合 約 0.25，3 通りの平均で基準を満たす割合 約 0.02）．
2：水準と感情の当てやすさだけでできた表（どの感情でも分類器の強さの順が同じ）と，得意な感情が本当に違う表での R0 の規則．
台本の関数（dprime・tukey_fit・noint・dcenter・corr・perm_share）は 5600b47 の版を読み込む．"""
import sys, importlib.util
import numpy as np
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
SP = "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/rev_ml_v5"
sp = importlib.util.spec_from_file_location("p16v5", SP + "/dai16_probe_5600b47.py"); P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

rng = np.random.default_rng(11)
print("---- 1．でたらめな表（R0 の偶然の水準）")
def null_tables(kind, size):
    if kind == "正規":
        return rng.normal(size=size)
    if kind == "一様 0.2〜2":
        return rng.uniform(0.2, 2.0, size=size)
    if kind == "水準と当てやすさ ＋ 雑音":                                    # d′ らしい表：水準 × 当てやすさ ＋ 小さな雑音（負は 0）
        a = rng.uniform(0.5, 1.5, size=size[:-1] + (1,)); b = rng.uniform(0.5, 1.5, size=size[:-2] + (1, size[-1]))
        return np.maximum(0.0, a * b + rng.normal(scale=0.1, size=size))
for kind in ("正規", "一様 0.2〜2", "水準と当てやすさ ＋ 雑音"):
    R = 4000
    A1 = null_tables(kind, (R, 3, 4)); B1 = null_tables(kind, (R, 3, 4))
    c1 = np.array([P.corr(P.dcenter(a), P.dcenter(b)) for a, b in zip(A1, B1)])
    c_add = np.array([P.corr(a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean(), b - b.mean(1, keepdims=True) - b.mean(0, keepdims=True) + b.mean()) for a, b in zip(A1, B1)])
    R3 = 1500
    hit_ind, hit_shared, hit_corr_only = [], [], []
    for _ in range(R3):
        As = null_tables(kind, (3, 3, 4)); Bs = null_tables(kind, (3, 3, 4))
        ci = [P.corr(P.dcenter(a), P.dcenter(b)) for a, b in zip(As, Bs)]; pi = [P.perm_share(a, b) for a, b in zip(As, Bs)]
        hit_ind.append(np.nanmean(ci) >= 0.3 and np.nanmean(pi) <= 0.2)
        cs = [P.corr(P.dcenter(a), P.dcenter(Bs[0])) for a in As]; ps = [P.perm_share(a, Bs[0]) for a in As]      # ターゲットの表は 3 つの組で同じ
        hit_shared.append(np.nanmean(cs) >= 0.3 and np.nanmean(ps) <= 0.2); hit_corr_only.append(np.nanmean(cs) >= 0.3)
    print(f"  {kind}：1 つの単位の相関の標準偏差 {np.nanstd(c1):.3f}（足し算の形だけ引くと {np.nanstd(c_add):.3f}），0.3 以上の割合 {np.nanmean(c1 >= 0.3):.3f}；"
          f"3 通りのソースの組の平均で 2 つの基準を満たす割合：ターゲットの表が組ごとに別 {np.mean(hit_ind):.3f}，同じ {np.mean(hit_shared):.3f}（相関の基準だけなら {np.mean(hit_corr_only):.3f}）")

print("---- 2．信頼度を人工の分類器から測った表での R0 の規則（1 回 = ソースの表 1 つとターゲットの表 1 つ．40 回）")
def draw(n, sep):
    y = rng.integers(0, 4, n)
    Ps = []
    for k in range(sep.shape[0]):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += sep[k][y]
        Ps.append(P.softmax(Z))
    return y, Ps
def run(name, sepS, sepT, n, reps=40):
    ic, ip, ok, mag = [], [], [], []
    for _ in range(reps):
        ys, Ps = draw(n, sepS); yt, Pt = draw(n, sepT)
        Ds = np.stack([P.dprime(p, ys)[0] for p in Ps]); Dt = np.stack([P.dprime(p, yt)[0] for p in Pt])
        ic.append(P.corr(P.dcenter(Ds), P.dcenter(Dt))); ip.append(P.perm_share(Ds, Dt)); ok.append(ic[-1] >= 0.3 and ip[-1] <= 0.2)
        mag.append(np.abs(P.dcenter(Dt)).mean())
    print(f"  {name}（{n} 本）：相関 {np.nanmean(ic):+.2f}（標準偏差 {np.nanstd(ic):.2f}），並べ替えの割合 {np.nanmean(ip):.3f}，規則を満たした割合 {np.mean(ok):.2f}，成分の大きさ {np.mean(mag):.3f}")
e = np.array([0.5, 0.9, 1.4, 2.0]); col = lambda a: np.array(a)[:, None]
inter = np.array([[2.5, 0.3, 0.3, 0.8], [0.3, 2.5, 0.3, 0.8], [0.3, 0.3, 2.5, 0.8]])
mulT = col([1.3, 1.0, 0.7]) * e
small = np.array([[0.25, -0.25, 0.0, 0.0], [-0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
for n in (20000, 2000):
    run("A 水準 × 当てやすさ（順は同じ）", col([1.4, 1.0, 0.6]) * e, mulT, n)
    run("A′ 水準の差が小さい", col([1.1, 1.0, 0.9]) * e, col([1.1, 1.0, 0.9]) * e, n)
    run("B 水準の順が逆", col([1.4, 1.0, 0.6]) * e, col([0.6, 1.0, 1.4]) * e, n)
    run("C 水準 ＋ 当てやすさ", col([0.4, 0.0, -0.4]) + e, col([0.3, 0.0, -0.3]) + e, n)
    run("E (水準 ＋ 当てやすさ) の 2 乗", (col([0.25, 0.0, -0.25]) + np.sqrt(e)) ** 2, (col([0.2, 0.0, -0.2]) + np.sqrt(e)) ** 2, n)
    run("D 得意な感情が違う（同じ）", inter, inter, n)
    run("F 水準 × 当てやすさ ＋ 小さな得意の違い（同じ）", mulT + small, mulT + small, n)
    run("G 同じ形で，得意の違いがソースとターゲットで逆", mulT + small, mulT - small, n)
