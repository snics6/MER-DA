"""感情認識の査読想定の再検査（DAI-16 第 4 版．コミット 6432e2e）の検算．人工の数だけを使う（MER-DA のデータと results/ は読まない）．
台本 DAI/exp/dai16_probe.py の関数（dprime・to_w・noint・dcenter・corr・perm_share・combine・match_shift・fit_wb）をそのまま呼ぶ．

検算 1：第 3 版の A2 への対応．切片を全ラベルで決める層（ob）と参照 prod_ob が，切片だけで出る分と重みの分を分けるか．
検算 2：第 3 版の A1 への対応（対照）．分類器 × 感情の成分を消した対照（台本の noint）が，共通の成分だけで出る差と，
        分類器ごとに得意な感情が違うときの差を分けるか（F6 と F3．主の層 adj と ob の層）．
検算 3：第 3 版の A1 への対応（R0）．分類器 × 感情の成分の相関（dcenter の相関）と入れ替えの割合（perm_share）が，
        共通の成分だけが合う場合に当たりにならないか．でたらめな表での当たりの割合．
検算 4：感情ごとの残り方（away ÷ home）は，home の信頼度が 0 に近い感情でどれだけ不安定か．
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
import dai16_probe as P  # noqa: E402


def uar(y, pred):
    return 100 * float(np.mean([(pred[y == k] == k).mean() for k in range(4)]))


def gen(rng, n, sep, noise, bias):
    y = rng.integers(0, 4, n)
    Ps = []
    for k in range(3):
        Z = rng.normal(size=(n, 4)) * np.asarray(noise[k]); Z[np.arange(n), y] += np.asarray(sep[k])[y]
        Ps.append(P.softmax(Z + np.asarray(bias)))
    return y, Ps


def layers(form, Lp, Lt, W, q1p, q1t, yp, target):
    """台本の run_unit と同じ手順で，adj の層と ob の層のテストの判定を作る．"""
    Sp, St = P.combine(form, Lp, W, q1p), P.combine(form, Lt, W, q1t)
    dl, _ = P.match_shift(Sp, target)
    bo = P.fit_wb(Sp[None], yp, np.ones(1), 0.0, fit_w=False, fit_b=True)[1]
    return St.argmax(1), (St + dl).argmax(1), (St + bo).argmax(1)


def run_case(rng, sep, noise, bias, reps=8, n=6000):
    names = ("prod", "F6", "F6n", "F3", "F3n")
    rows = []
    for _ in range(reps):
        yp, Pp = gen(rng, n, sep, noise, bias); yt, Pt = gen(rng, n, sep, noise, bias)
        Lp, Lt = np.stack([P.centered(p) for p in Pp]), np.stack([P.centered(p) for p in Pt])
        W = P.to_w(np.stack([P.dprime(p, yp)[0] for p in Pp]))
        q1p, q1t = P.softmax(Lp.sum(0)), P.softmax(Lt.sum(0))
        target = q1p.mean(0)
        u = {}
        for name, form, w in (("prod", "F2", np.ones((3, 4))), ("F6", "F6", W), ("F6n", "F6", P.noint(W)), ("F3", "F3", W), ("F3n", "F3", P.noint(W))):
            pr, pa, po = layers(form, Lp, Lt, w, q1p, q1t, yp, target)
            u[name] = (uar(yt, pr), uar(yt, pa), uar(yt, po))
        rows.append([u["prod"][2] - u["prod"][0],                                             # prod_ob − prod（切片だけで出る分）
                     u["F6"][1] - u["prod"][1], u["F6"][1] - u["F6n"][1], u["F3"][1] - u["prod"][1], u["F3"][1] - u["F3n"][1],     # adj の層
                     u["F6"][2] - u["prod"][2], u["F6"][2] - u["F6n"][2], u["F3"][2] - u["prod"][2], u["F3"][2] - u["F3n"][2]])    # ob の層
    m = np.mean(rows, 0); se = np.std(rows, 0, ddof=1) / np.sqrt(len(rows))
    print(f"    切片だけで出る分（prod_ob − prod）{m[0]:+.2f}")
    print(f"    adj の層：F6 − 積 {m[1]:+.2f}，F6 − 対照 {m[2]:+.2f}，F3 − 積 {m[3]:+.2f}，F3 − 対照 {m[4]:+.2f}")
    print(f"    ob の層 ：F6 − 積 {m[5]:+.2f}，F6 − 対照 {m[6]:+.2f}，F3 − 積 {m[7]:+.2f}，F3 − 対照 {m[8]:+.2f}（標準誤差の最大 {se.max():.2f}）")


def sim12():
    rng = np.random.default_rng(11)
    same = lambda v: [v, v, v]  # noqa: E731
    zero, tilt = np.zeros(4), np.array([-1.0, 0.0, 0.0, 0.0])
    print("== 検算 1・2：切片だけで出る分，共通の成分だけで出る差，分類器ごとに得意な感情が違うときの差")
    print("  (A) 3 つの分類器が同じ作りで，3 つとも感情 0 を答えにくい（切片の偏りだけ）")
    run_case(rng, same([1.0] * 4), same([1.0] * 4), tilt)
    print("  (B) 3 つの分類器が同じ作りで，感情 0 の点だけ雑音が 3 倍（共通の成分だけ）")
    run_case(rng, same([1.5] * 4), same([3.0, 1.0, 1.0, 1.0]), zero)
    print("  (C) 3 つの分類器が同じ作りで，感情 0 だけ分かれ方が 0.4（共通の成分だけ）")
    run_case(rng, same([0.4, 1.5, 1.5, 1.5]), same([1.0] * 4), zero)
    print("  (D) 分類器ごとに得意な感情が違う（分類器 × 感情の成分）")
    run_case(rng, [[2.5, 0.5, 0.5, 1.0], [0.5, 2.5, 0.5, 1.0], [0.5, 0.5, 2.5, 1.0]], same([1.0] * 4), zero)
    print("  (E) (D) に，3 つとも感情 0 を答えにくい偏りを足す")
    run_case(rng, [[2.5, 0.5, 0.5, 1.0], [0.5, 2.5, 0.5, 1.0], [0.5, 0.5, 2.5, 1.0]], same([1.0] * 4), tilt)


def sim3():
    print("== 検算 3：R0 の量（分類器 × 感情の成分の相関 ≥ 0.3 かつ入れ替えの割合 ≤ 0.2 を，表 1 組ごとに当てた割合）")
    rng = np.random.default_rng(12)

    def rate(make, n=3000):
        ic, ip = [], []
        for _ in range(n):
            A, B = make()
            ic.append(P.corr(P.dcenter(A), P.dcenter(B))); ip.append(P.perm_share(A, B))
        ic, ip = np.array(ic), np.array(ip)
        return float(np.mean(ic)), float(np.mean(ip)), float(np.mean((ic >= 0.3) & (ip <= 0.2)))

    def common_only():
        e = rng.normal(0, 0.30, 4)
        return (rng.uniform(0.6, 1.6, 3)[:, None] + e + rng.normal(0, 0.15, (3, 4)), rng.uniform(0.6, 1.6, 3)[:, None] + e + rng.normal(0, 0.15, (3, 4)))

    def inter_only():
        g = rng.normal(0, 0.15, (3, 4))
        return (rng.uniform(0.6, 1.6, 3)[:, None] + rng.normal(0, 0.30, 4) + g, rng.uniform(0.6, 1.6, 3)[:, None] + rng.normal(0, 0.30, 4) + g)

    def inter_noisy():
        g = rng.normal(0, 0.15, (3, 4))
        return (rng.uniform(0.6, 1.6, 3)[:, None] + rng.normal(0, 0.30, 4) + g + rng.normal(0, 0.10, (3, 4)),
                rng.uniform(0.6, 1.6, 3)[:, None] + rng.normal(0, 0.30, 4) + g + rng.normal(0, 0.10, (3, 4)))

    for label, f in (("共通の成分だけが合う", common_only), ("分類器 × 感情の成分だけが合う", inter_only),
                     ("分類器 × 感情の成分が合い，測る揺れ 0.10 が両方に入る", inter_noisy),
                     ("でたらめな 2 つの表", lambda: (rng.normal(size=(3, 4)), rng.normal(size=(3, 4))))):
        a, b, c = rate(f)
        print(f"  {label}：相関の平均 {a:+.2f}，入れ替えの割合の平均 {b:.2f}，表 1 組が当たりになる割合 {c:.2f}")
    # 3 通りのソースの組を平均したときの当たり（でたらめ．同じターゲットの表を 3 つのソースの表と比べる）
    hit = []
    for _ in range(3000):
        B = rng.normal(size=(3, 4))
        v = [(P.corr(P.dcenter(A), P.dcenter(B)), P.perm_share(A, B)) for A in (rng.normal(size=(3, 4)) for _ in range(3))]
        hit.append(np.mean([x[0] for x in v]) >= 0.3 and np.mean([x[1] for x in v]) <= 0.2)
    print(f"  でたらめな表で，3 通りのソースの表の平均に基準を当てたときの当たりの割合 {np.mean(hit):.3f}")


def sim4():
    print("== 検算 4：感情ごとの残り方（away ÷ home）の不安定さ")
    rng = np.random.default_rng(13)
    for label, sep_home, sep_away, n in (("強い分類器（home の分かれ方 1.5，away 0.9．感情あたり 300 本）", 1.5, 0.9, 300),
                                         ("弱い分類器（home の分かれ方 0.15，away 0.10．感情あたり 300 本）", 0.15, 0.10, 300),
                                         ("弱い分類器・少ない件数（同じ分かれ方．感情あたり 40 本）", 0.15, 0.10, 40)):
        r = []
        for _ in range(300):
            y = np.repeat(np.arange(4), n)
            def d(sep):
                Z = rng.normal(size=(len(y), 4)); Z[np.arange(len(y)), y] += sep
                return P.dprime(P.softmax(Z), y)[0]
            home = d(sep_home) + d(sep_home); away = d(sep_away) + d(sep_away)          # 2 つの向きの合計（台本の ret_emo と同じ形）
            r.append(away / np.maximum(home, 1e-12))
        r = np.array(r)
        print(f"  {label}：感情ごとの残り方の中央値 {np.median(r):.2f}，5%〜95% の幅 {np.quantile(r, 0.05):.2f}〜{np.quantile(r, 0.95):.2f}，10 を超える割合 {np.mean(r > 10):.3f}")




def sim5():
    """検算 5：でたらめな 2 つの表の相関の散らばり．行の平均だけを引く量（第 3 版の R0）と，行と列の両方の平均を引く量（第 4 版の R0）．"""
    rng = np.random.default_rng(14)
    r_row, r_both = [], []
    for _ in range(20000):
        A, B = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
        r_row.append(P.corr(A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)))
        r_both.append(P.corr(P.dcenter(A), P.dcenter(B)))
    print("== 検算 5：でたらめな 2 つの表の相関")
    print(f"  行の平均だけを引く量（第 3 版）：標準偏差 {np.std(r_row):.2f}，0.3 以上の割合 {np.mean(np.array(r_row) >= 0.3):.2f}")
    print(f"  行と列の両方の平均を引く量（第 4 版）：標準偏差 {np.std(r_both):.2f}，0.3 以上の割合 {np.mean(np.array(r_both) >= 0.3):.2f}")


if __name__ == "__main__":
    print("読んだ台本：", P.__file__, "sha256 の先頭", hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest()[:16])
    P.selftest()
    sim12(); sim3(); sim4(); sim5()
