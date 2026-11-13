"""感情認識の査読想定の検査（DAI-16 第 3 版）の検算．人工の数だけを使う（MER-DA のデータと results/ は読まない）．
台本 DAI/exp/dai16_probe.py の関数（dprime・to_w・combine・match_shift・fit_wb・mix_rel など）をそのまま呼ぶ．

検算 1：adj の層と kb の層では，切片だけの対照 orc_bias の判定が基準の積 prod と同じになるか．
検算 2：R0 の相関（分類器 × 感情の相性の相関）は，3 つのモダリティに共通の「感情ごとの難しさ」だけで高くなるか．
        逆に，モダリティに固有の成分が完全に同じでも，共通の成分がコーパスで違えば低くなるか．
検算 3：3 つの分類器がまったく同じ作り（モダリティ × 感情の組み合わせの違いが無い）で，1 つの感情だけがどの分類器でも
        当てにくいとき，F6（感情を順に仮定して比べる）・F3（先に疑似ラベル）・F2（感情で共通の重み）は積とどれだけ違うか．
        重みをモダリティで平均した対照（感情ごとの数 4 つだけ）と比べる．
検算 4：仮定 (b) の旗の量（候補とテストの信頼度の差の絶対値の平均）は，ずれが無いときに標本の揺れだけでどれだけ出るか．
        K 本の水準（負を 0 に切った d′ の平均）は，情報の無い分類器でどれだけ正に出るか．
"""
import sys
from pathlib import Path

import numpy as np

import hashlib
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
if os.environ.get("DAI16_PROBE_DIR"):          # 検査の対象のコミット（556a06f）の台本を別の置き場から読む（作業木の台本は検査の途中で書き換わった）
    sys.path.insert(0, os.environ["DAI16_PROBE_DIR"])
import dai16_probe as P  # noqa: E402
import dai_solver as S  # noqa: E402


def uar(y, pred):
    return 100 * float(np.mean([(pred[y == k] == k).mean() for k in range(4)]))


def make(rng, n, sep, noise, shares=None, bias=None):
    """分類器 1 つの確率．点 = 雑音（列ごとの大きさ noise）＋ 本当の感情の列に sep[本当の感情]．"""
    y = rng.choice(4, size=n, p=shares) if shares is not None else rng.integers(0, 4, n)
    Z = rng.normal(size=(n, 4)) * np.asarray(noise)
    Z[np.arange(n), y] += np.asarray(sep)[y]
    if bias is not None:
        Z = Z + np.asarray(bias)
    return y, Z


def sim1():
    print("== 検算 1：adj・kb の層の orc_bias と prod")
    rng = np.random.default_rng(1)
    out = []
    for rep in range(5):
        n = 3000
        yp = rng.integers(0, 4, n); yt = rng.integers(0, 4, n); yk = np.repeat(np.arange(4), 5)
        def probs(y):
            Ps = []
            for k in range(3):
                Z = rng.normal(size=(len(y), 4)); Z[np.arange(len(y)), y] += 1.0; Z[:, 0] -= 1.0   # 3 つとも感情 0 を答えにくい
                Ps.append(P.softmax(Z))
            return Ps
        Pp, Pt, Pk = probs(yp), probs(yt), probs(yk)
        prod = lambda Ps: P.norm_log(sum(P.centered(p) for p in Ps))  # noqa: E731
        Sp, St, Sh = prod(Pp), prod(Pt), prod(Pk)
        target = np.exp(Sp).mean(0)
        b_orc = P.fit_wb(Sp[None], yp, np.ones(1), 0.0, fit_w=False, fit_b=True)[1]
        res = {}
        for name, (a, b, c) in {"prod": (Sp, St, Sh), "orc_bias": (P.norm_log(Sp + b_orc), P.norm_log(St + b_orc), P.norm_log(Sh + b_orc))}.items():
            dl, rs = P.match_shift(a, target)
            bk = P.fit_wb((c + dl)[None], yk, np.ones(1), P.TAU, fit_w=False, fit_b=True)[1]
            res[name] = dict(raw=b.argmax(1), adj=(b + dl).argmax(1), kb=(b + dl + bk).argmax(1), dl=dl)
        out.append([uar(yt, res["orc_bias"]["raw"]) - uar(yt, res["prod"]["raw"]),
                    float((res["orc_bias"]["adj"] != res["prod"]["adj"]).mean()), float((res["orc_bias"]["kb"] != res["prod"]["kb"]).mean()),
                    float(np.abs(res["orc_bias"]["dl"] + (b_orc - b_orc.mean())).max())])
    o = np.array(out)
    print(f"  raw の層：orc_bias − prod の UAR の差の平均 {o[:,0].mean():+.2f} 点")
    print(f"  adj の層：判定が食い違う発話の割合の最大 {o[:,1].max():.5f}．kb の層：同 {o[:,2].max():.5f}")
    print(f"  割合をそろえる切片 ＋ 全ラベルの切片（平均を引いた値）の絶対値の最大 {o[:,3].max():.1e}（0 なら，割合をそろえる操作が全ラベルの切片を打ち消している）")


def prof(D):
    return D - D.mean(1, keepdims=True)


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def dcen(D):
    X = prof(D)
    return X - X.mean(0, keepdims=True)


def sim2():
    print("== 検算 2：R0 の相関と，3 つのモダリティに共通の感情ごとの成分")
    rng = np.random.default_rng(2)
    r_a, r_a2, r_b, r_b2 = [], [], [], []
    for _ in range(2000):
        lev_s, lev_t = rng.uniform(0.6, 1.6, 3), rng.uniform(0.6, 1.6, 3)
        e = rng.normal(0, 0.30, 4)                      # 感情ごとの難しさ（3 つのモダリティに共通）
        # (a) 共通の成分だけが同じ．モダリティに固有の成分はソースとターゲットで無関係
        Ds = lev_s[:, None] + e + rng.normal(0, 0.15, (3, 4))
        Dt = lev_t[:, None] + e + rng.normal(0, 0.15, (3, 4))
        r_a.append(corr(prof(Ds), prof(Dt))); r_a2.append(corr(dcen(Ds), dcen(Dt)))
        # (b) モダリティに固有の成分は完全に同じ．共通の成分はソースとターゲットで無関係（ラベルの決め方の違いに当たる）
        g = rng.normal(0, 0.15, (3, 4))
        Ds = lev_s[:, None] + rng.normal(0, 0.30, 4) + g
        Dt = lev_t[:, None] + rng.normal(0, 0.30, 4) + g
        r_b.append(corr(prof(Ds), prof(Dt))); r_b2.append(corr(dcen(Ds), dcen(Dt)))
    print(f"  (a) 共通の成分だけが持ち越される：設計の相関の平均 {np.mean(r_a):+.2f}（0.3 以上の割合 {np.mean(np.array(r_a) >= 0.3):.2f}）．"
          f"感情ごとの平均も引いた相関の平均 {np.mean(r_a2):+.2f}")
    print(f"  (b) 固有の成分だけが持ち越される：設計の相関の平均 {np.mean(r_b):+.2f}（0.3 以上の割合 {np.mean(np.array(r_b) >= 0.3):.2f}）．"
          f"感情ごとの平均も引いた相関の平均 {np.mean(r_b2):+.2f}")
    # 相関 12 個の値がでたらめのときの散らばり（行ごとに平均を引いた 3×4 の表どうし）
    r0 = [corr(prof(rng.normal(size=(3, 4))), prof(rng.normal(size=(3, 4)))) for _ in range(20000)]
    print(f"  でたらめな 2 つの表の相関：標準偏差 {np.std(r0):.2f}，0.3 以上の割合 {np.mean(np.array(r0) >= 0.3):.2f}")


def run_case(rng, sep, noise, reps=8, n=6000, calib=False):
    rows = []
    for _ in range(reps):
        def gen(n):
            y = rng.integers(0, 4, n)
            Ps = []
            for k in range(3):
                Z = rng.normal(size=(n, 4)) * np.asarray(noise[k]); Z[np.arange(n), y] += np.asarray(sep[k])[y]
                if calib:                               # 正しい事後確率（雑音の大きさが列で同じときだけ使う）
                    Z = Z * np.asarray(sep[k])
                Ps.append(P.softmax(Z))
            return y, Ps
        yp, Pp = gen(n); yt, Pt = gen(n)
        Lp, Lt = np.stack([P.centered(p) for p in Pp]), np.stack([P.centered(p) for p in Pt])
        D = np.stack([P.dprime(p, yp)[0] for p in Pp])
        W = P.to_w(D)
        Wc = np.repeat(W.mean(0, keepdims=True), 3, 0)                 # モダリティで平均した重み（感情ごとの数 4 つ）
        q1p, q1t = P.softmax(Lp.sum(0)), P.softmax(Lt.sum(0))
        target = q1p.mean(0)
        r = {}
        for name, form, w in (("prod", "F2", np.ones((3, 4))), ("F2", "F2", W), ("F3", "F3", W), ("F6", "F6", W), ("F6c", "F6", Wc), ("F3c", "F3", Wc)):
            Sp, St = P.combine(form, Lp, w, q1p), P.combine(form, Lt, w, q1t)
            dl, _ = P.match_shift(Sp, target)
            r[name] = (uar(yt, St.argmax(1)), uar(yt, (St + dl).argmax(1)))
        rows.append([r[k][1] - r["prod"][1] for k in ("F2", "F3", "F6", "F6c", "F3c")] + [r["prod"][1], r["F6"][0] - r["prod"][0]] + list(D.mean(0)) + [np.abs(D - D.mean(0)).max()])
    m = np.mean(rows, 0); se = np.std(rows, 0, ddof=1) / np.sqrt(len(rows))
    print(f"    積の UAR {m[5]:.1f}．adj の層の差（− 積）：F2 {m[0]:+.2f}，F3 {m[1]:+.2f}，F6 {m[2]:+.2f}，モダリティで平均した重みの F6 {m[3]:+.2f}，同 F3 {m[4]:+.2f}"
          f"（標準誤差 {se[:5].max():.2f}）．raw の層の F6 − 積 {m[6]:+.2f}")
    print(f"    感情ごとの信頼度（3 つの分類器の平均）{np.round(m[7:11], 2).tolist()}．分類器の間の差の最大 {m[11]:.2f}")


def sim3():
    print("== 検算 3：3 つの分類器が同じ作りのとき（モダリティ × 感情の組み合わせの違いは無い）")
    rng = np.random.default_rng(3)
    same = lambda v: [v, v, v]  # noqa: E731
    print("  (i) 感情 0 の点だけ，どの分類器でも雑音が 2 倍（ほかは 1）．分かれ方は 4 感情とも 1.5")
    run_case(rng, same([1.5, 1.5, 1.5, 1.5]), same([2.0, 1.0, 1.0, 1.0]))
    print("  (ii) 感情 0 だけ，どの分類器でも分かれ方が 0.4（ほかは 1.5）．雑音は 1")
    run_case(rng, same([0.4, 1.5, 1.5, 1.5]), same([1.0, 1.0, 1.0, 1.0]))
    print("  (iii) 感情 0 の点だけ，どの分類器でも雑音が 3 倍")
    run_case(rng, same([1.5, 1.5, 1.5, 1.5]), same([3.0, 1.0, 1.0, 1.0]))
    print("  (iv) 対照：分類器ごとに得意な感情が違う（共通の難しさは無い）")
    run_case(rng, [[2.5, 0.5, 0.5, 1.0], [0.5, 2.5, 0.5, 1.0], [0.5, 0.5, 2.5, 1.0]], same([1.0, 1.0, 1.0, 1.0]))


def sim4():
    print("== 検算 4：標本の揺れだけで出る量")
    rng = np.random.default_rng(4)
    for label, n_pool, n_test in (("AFEW に近い件数（候補 150，テスト 40／感情）", 150, 40), ("IEMOCAP に近い件数（候補 550，テスト 130／感情）", 550, 130),
                                  ("MOSEI に近い件数（候補 1,300／感情，テスト 300／感情）", 1300, 300)):
        v = []
        for _ in range(60):
            def one(n):
                y = np.repeat(np.arange(4), n)
                return np.stack([P.dprime(P.softmax(make_fixed(rng, y, s)), y)[0] for s in (1.5, 1.0, 0.6)])
            v.append(np.abs(one(n_pool) - one(n_test)).mean())
        print(f"  ずれが無いときの「候補とテストの信頼度の差の絶対値の平均」：{label} で平均 {np.mean(v):.3f}（設計の旗の基準は 0.05）")
    for K in (5, 25):
        lv = []
        for _ in range(4000):
            y = np.repeat(np.arange(4), K)
            lv.append(P.dprime(P.softmax(rng.normal(size=(4 * K, 4))), y)[0].mean())
        print(f"  感情と関係の無い点を出す分類器の K 本の水準（本当は 0）：K = {K} で平均 {np.mean(lv):.2f}，上位 10% {np.quantile(lv, 0.9):.2f}")
    for K in (5, 25):
        lv = []
        for _ in range(4000):
            y = np.repeat(np.arange(4), K)
            lv.append(P.dprime(P.softmax(make_fixed(rng, y, 1.5)), y)[0].mean())
        y = np.repeat(np.arange(4), 20000)
        tr = P.dprime(P.softmax(make_fixed(rng, y, 1.5)), y)[0].mean()
        print(f"  分かれ方 1.5 の分類器の K 本の水準：K = {K} で平均 {np.mean(lv):.2f}，標準偏差 {np.std(lv):.2f}（件数が多いときの値 {tr:.2f}）")


def make_fixed(rng, y, s):
    Z = rng.normal(size=(len(y), 4))
    Z[np.arange(len(y)), y] += s
    return Z


if __name__ == "__main__":
    print("読んだ台本：", P.__file__, "sha256 の先頭", hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest()[:16])
    P.selftest()
    sim1(); sim2(); sim3(); sim4()
