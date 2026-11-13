"""感情認識の査読想定の再検査（DAI-16 第 5 版．コミット 5600b47）の検算．人工の数だけを使う（MER-DA のデータと results/ は読まない）．
台本 DAI/exp/dai16_probe.py の関数（dprime・to_w・tukey_fit・noint・dcenter・corr・perm_share・combine・match_shift・fit_wb）をそのまま呼ぶ．

第 5 版は「分類器 × 感情の成分」を，表を「全体の平均 ＋ 行の差 a ＋ 列の差 b ＋ λ a b」で当てはめた残りに替えた．
検算 1：当てはめの式に入る表の例．(i) 掛け算だけの表（どの感情でも分類器の比が同じ）．(ii) λ a b の項で，当てにくい感情だけ分類器の強さの順が逆になる表．
検算 2：対照（台本の noint．当てはめの値を重みにする）と，第 4 版の対照（行の平均 ＋ 列の平均 − 全体の平均）が，次の場合をどう分けるか．
        共通の成分だけ／掛け算だけ／分類器ごとに得意な感情が違う／当てにくい感情だけ強さの順が逆になる（λ a b の形）．
検算 3：R0 の量（当てはめの残りの相関と入れ替えの割合）の，でたらめな表での散らばりと，上の場合での値．
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
import dai16_probe as P  # noqa: E402


def uar(y, pred):
    return 100 * float(np.mean([(pred[y == k] == k).mean() for k in range(4)]))


def add_fit(X):
    """第 4 版の当てはめ：行の平均 ＋ 列の平均 − 全体の平均．"""
    return X.mean(1, keepdims=True) + X.mean(0, keepdims=True) - X.mean()


def lam_of(X):
    g = X.mean(); a, b = X.mean(1) - g, X.mean(0) - g
    den = float((a ** 2).sum() * (b ** 2).sum())
    return (float((a[:, None] * b[None, :] * X).sum() / den) if den > 1e-12 else 0.0), a, b, g


def describe(name, X):
    lam, a, b, g = lam_of(X)
    fac = 1.0 + lam * b                                   # 当てはめの中で，感情 i の分類器どうしの差は a_k × (1 + λ b_i)
    ia = X - add_fit(X)
    share = float(((ia - P.dcenter(X)) ** 2).sum() / max((ia ** 2).sum(), 1e-12))
    print(f"  {name}")
    print("    表（行＝分類器，列＝感情）：", np.round(X, 2).tolist())
    print(f"    λ = {lam:+.2f}，全体の平均の逆数 1/g = {1 / g:+.2f}．感情ごとの倍率 1 + λ b = {np.round(fac, 2).tolist()}（負の感情では強さの順が逆）")
    print(f"    感情ごとに最も強い分類器：{X.argmax(0).tolist()}．当てはめの残りの最大 {np.abs(P.dcenter(X)).max():.3f}，行と列の平均だけを引いた残りの最大 {np.abs(ia).max():.3f}，"
          f"そのうち λ a b の項が占める割合 {share:.2f}")


G, A_, B_ = 1.0, np.array([0.4, 0.0, -0.4]), np.array([-0.5, 0.1, 0.2, 0.2])
T_MUL = np.array([1.4, 1.0, 0.6])[:, None] * np.array([0.5, 1.1, 1.2, 1.2])[None, :]
T_REV = G + A_[:, None] + B_[None, :] + 3.0 * A_[:, None] * B_[None, :]


def sim1():
    print("== 検算 1：当てはめの式（全体の平均 ＋ a ＋ b ＋ λ a b）に入る表")
    describe("(i) 掛け算だけの表", T_MUL)
    describe("(ii) 当てにくい感情 0 だけ，強さの順が逆になる表（λ = 3）", T_REV)


def gen(rng, n, sep):
    y = rng.integers(0, 4, n)
    Ps = []
    for k in range(3):
        Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += np.asarray(sep[k])[y]
        Ps.append(P.softmax(Z))
    return y, Ps


def run_case(rng, sep, reps=8, n=6000, show=False):
    rows, Ds = [], []
    for _ in range(reps):
        yp, Pp = gen(rng, n, sep); yt, Pt = gen(rng, n, sep)
        Lp, Lt = np.stack([P.centered(p) for p in Pp]), np.stack([P.centered(p) for p in Pt])
        D = np.stack([P.dprime(p, yp)[0] for p in Pp]); Ds.append(D)
        W = P.to_w(D)
        q1p, q1t = P.softmax(Lp.sum(0)), P.softmax(Lt.sum(0))
        target = q1p.mean(0)
        u = {}
        for name, form, w in (("prod", "F2", np.ones((3, 4))), ("F2", "F2", W), ("F6", "F6", W), ("F6n", "F6", P.noint(W)), ("F6a", "F6", np.maximum(0.0, add_fit(W))),
                              ("F3", "F3", W), ("F3n", "F3", P.noint(W)), ("F3a", "F3", np.maximum(0.0, add_fit(W)))):
            Sp, St = P.combine(form, Lp, w, q1p), P.combine(form, Lt, w, q1t)
            dl, _ = P.match_shift(Sp, target)
            u[name] = uar(yt, (St + dl).argmax(1))
        rows.append([u["prod"], u["F2"] - u["prod"], u["F6"] - u["prod"], u["F6"] - u["F6n"], u["F6"] - u["F6a"], u["F3"] - u["prod"], u["F3"] - u["F3n"], u["F3"] - u["F3a"]])
    m = np.mean(rows, 0); se = np.std(rows, 0, ddof=1) / np.sqrt(len(rows))
    Dm = np.mean(Ds, 0)
    lam, a, b, g = lam_of(Dm)
    print(f"    測った信頼度の表の平均：{np.round(Dm, 2).tolist()}")
    print(f"    λ = {lam:+.2f}（1/g = {1 / g:+.2f}），1 + λ b = {np.round(1 + lam * b, 2).tolist()}，感情ごとに最も強い分類器 {Dm.argmax(0).tolist()}")
    print(f"    積の UAR {m[0]:.1f}．adj の層：F2 − 積 {m[1]:+.2f}，F6 − 積 {m[2]:+.2f}，F6 − 第 5 版の対照 {m[3]:+.2f}，F6 − 第 4 版の対照 {m[4]:+.2f}，"
          f"F3 − 積 {m[5]:+.2f}，F3 − 第 5 版の対照 {m[6]:+.2f}，F3 − 第 4 版の対照 {m[7]:+.2f}（標準誤差の最大 {se[1:].max():.2f}）")


def sim2():
    rng = np.random.default_rng(21)
    same = lambda v: [v, v, v]  # noqa: E731
    print("== 検算 2：対照が分けるもの（分類器の分かれ方を表のとおりに置く．雑音は 1）")
    print("  (B) 3 つの分類器が同じ作りで，感情 0 だけ分かれ方が 0.4（共通の成分だけ）")
    run_case(rng, same([0.4, 1.5, 1.5, 1.5]))
    print("  (M) 掛け算だけ：分かれ方 = 分類器の水準（1.8，1.2，0.6）× 感情の当てやすさ（0.5，1.1，1.2，1.2）")
    run_case(rng, (np.array([1.8, 1.2, 0.6])[:, None] * np.array([0.5, 1.1, 1.2, 1.2])[None, :]).tolist())
    print("  (D) 分類器ごとに得意な感情が違う（水準は同じ）")
    run_case(rng, [[2.5, 0.5, 0.5, 1.0], [0.5, 2.5, 0.5, 1.0], [0.5, 0.5, 2.5, 1.0]])
    print("  (R) 当てにくい感情 0 だけ，分類器の強さの順が逆（検算 1 の (ii) の表 × 1.3）")
    run_case(rng, (1.3 * T_REV).tolist())
    print("  (R2) 同じ形で，逆になる度合いが小さい（λ = 2 の表 × 1.3）")
    run_case(rng, (1.3 * (G + A_[:, None] + B_[None, :] + 2.0 * A_[:, None] * B_[None, :])).tolist())


def sim3():
    print("== 検算 3：R0 の量（第 5 版：当てはめの残りの相関．第 4 版：行と列の平均だけを引いた残りの相関）")
    rng = np.random.default_rng(22)
    r5, r4 = [], []
    for _ in range(20000):
        A, B = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
        r5.append(P.corr(P.dcenter(A), P.dcenter(B))); r4.append(P.corr(A - add_fit(A), B - add_fit(B)))
    r5, r4 = np.array(r5), np.array(r4)
    print(f"  でたらめな 2 つの表：第 5 版の量の標準偏差 {np.nanstd(r5):.2f}，0.3 以上の割合 {np.nanmean(r5 >= 0.3):.2f}（第 4 版の量は {np.std(r4):.2f}，{np.mean(r4 >= 0.3):.2f}）")
    hit = []
    for _ in range(4000):
        B = rng.normal(size=(3, 4))
        v = [(P.corr(P.dcenter(A), P.dcenter(B)), P.perm_share(A, B)) for A in (rng.normal(size=(3, 4)) for _ in range(3))]
        hit.append(np.mean([x[0] for x in v]) >= 0.3 and np.mean([x[1] for x in v]) <= 0.2)
    print(f"  でたらめな表で，3 通りのソースの表の平均に 2 つの基準を当てたときの当たりの割合 {np.mean(hit):.3f}")

    def rate(make, n=3000):
        c5, c4, hp = [], [], []
        for _ in range(n):
            A, B = make()
            x = P.corr(P.dcenter(A), P.dcenter(B)); c5.append(x); c4.append(P.corr(A - add_fit(A), B - add_fit(B)))
            hp.append(x >= 0.3 and P.perm_share(A, B) <= 0.2)
        return float(np.nanmean(c5)), float(np.mean(c4)), float(np.mean(hp))

    def common_only():
        e = rng.normal(0, 0.30, 4)
        return (rng.uniform(0.6, 1.6, 3)[:, None] + e + rng.normal(0, 0.10, (3, 4)), rng.uniform(0.6, 1.6, 3)[:, None] + e + rng.normal(0, 0.10, (3, 4)))

    def mult_only():
        e = 1.0 + rng.normal(0, 0.30, 4)
        return (rng.uniform(0.6, 1.6, 3)[:, None] * e + rng.normal(0, 0.10, (3, 4)), rng.uniform(0.6, 1.6, 3)[:, None] * e + rng.normal(0, 0.10, (3, 4)))

    def specific():
        g = rng.normal(0, 0.15, (3, 4))
        return (rng.uniform(0.6, 1.6, 3)[:, None] + rng.normal(0, 0.30, 4) + g + rng.normal(0, 0.10, (3, 4)),
                rng.uniform(0.6, 1.6, 3)[:, None] + rng.normal(0, 0.30, 4) + g + rng.normal(0, 0.10, (3, 4)))

    def reversal():
        return (T_REV + rng.normal(0, 0.10, (3, 4)), T_REV + rng.normal(0, 0.10, (3, 4)))

    for label, f in (("共通の成分だけが合う（足し算の形）", common_only), ("共通の成分だけが合う（掛け算の形．水準は表ごとに違う）", mult_only),
                     ("分類器 × 感情の成分が合う（測る揺れ 0.10）", specific), ("当てにくい感情だけ強さの順が逆になる表が，ソースとターゲットで同じ（測る揺れ 0.10）", reversal)):
        a, b, c = rate(f)
        print(f"  {label}：第 5 版の量の平均 {a:+.2f}，第 4 版の量の平均 {b:+.2f}，表 1 組が第 5 版の基準を満たす割合 {c:.2f}")




def sim4():
    """検算 4：λ a b の項が，行と列の平均だけを引いた残りのうちどれだけを占めるか．当てはめた表で，感情によって最も強い分類器が替わる割合．"""
    rng = np.random.default_rng(23)
    print("== 検算 4：人工の信頼度の表（水準 0.6〜1.6，共通の成分の散らばり 0.30）での λ a b の項")
    for label, sg, extra in (("分類器 × 感情の成分がでたらめ（散らばり 0.15）", 0.15, 0.0),
                             ("同じ表に，当てにくい感情ほど分類器の差が縮む項（λ = −1.5 の a b）を足す", 0.15, -1.5)):
        share, flip, flip_big = [], [], []
        for _ in range(5000):
            lev = rng.uniform(0.6, 1.6, 3); e = rng.normal(0, 0.30, 4)
            a, b = lev - lev.mean(), e - e.mean()
            X = lev[:, None] + e[None, :] + rng.normal(0, sg, (3, 4)) + extra * a[:, None] * b[None, :]
            ia = X - add_fit(X); fit = P.tukey_fit(X)
            share.append(float(((ia - P.dcenter(X)) ** 2).sum() / max((ia ** 2).sum(), 1e-12)))
            top = fit.argmax(0)
            flip.append(len(set(top.tolist())) > 1)
            srt = np.sort(fit, 0)
            flip_big.append(len(set(top.tolist())) > 1 and float((srt[-1] - srt[-2]).min()) > 0.05)
        print(f"  {label}：λ a b の項が占める割合の平均 {np.mean(share):.2f}．当てはめた表（対照の重み）で，感情によって最も強い分類器が替わる表の割合 {np.mean(flip):.2f}"
              f"（どの感情でも 1 位と 2 位の差が 0.05 を超えるものに限ると {np.mean(flip_big):.2f}）")


if __name__ == "__main__":
    print("読んだ台本：", P.__file__, "sha256 の先頭", hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest()[:16])
    P.selftest()
    sim1(); sim2(); sim3(); sim4()
