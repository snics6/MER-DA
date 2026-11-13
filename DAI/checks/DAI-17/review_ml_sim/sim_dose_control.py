"""Q4 の対照（ソースの分類器を足す版 MS_eq）の倍率の確かめ（人工の数．MER-DA のデータ・結果は読まない．CPU だけ）．
台本の式：
  MN_eq(λ) = 微調整した 3 つの点の和 ＋ λ ×（近いソースの発話の答え 3 つの点の和）          … 足す項は 3 つ
  MS_eq(λ) = 微調整した 3 つの点の和 ＋ λ ×（ソースの分類器 6 つ = 2 コーパス × 3 モダリティの点の和） … 足す項は 6 つ
問い：近いソースの発話の答えが，ソースの分類器と全く同じ情報しか持たないとき，設計の 3 つの差
  MN_eq_sel − prod，MN_eq_sel − MS_eq_sel，MN_eq_sel − MS_eq_tied（同じ λ の MS_eq）はどうなるか．
作り：「近いソースの発話の答え」を，同じモダリティの 2 つのソースの分類器の点の平均と置く（情報は同じ．項の数だけが 3 と 6 で違う）．
  このとき MN_eq(λ) は MS_eq(λ/2) と同じ式になる．λ の格子は台本と同じ {0.25, 1}．
  ソースの分類器は，微調整した分類器と別の情報を持つが，点が大きく（確信が強すぎ），決まった取り違え（Sad→Neutral，Angry→Happy）を持つ．
層は adj（ラベルを使わずに，候補の上の確率の平均を積の平均に合わせる．dai16_probe.match_shift）．格子の点は別の標本（選ぶ用）の UAR で選ぶ．
"""
import sys
import numpy as np

sys.path.insert(0, sys.argv[1])
import dai17_probe as P17  # noqa: E402
P16, S = P17.P16, P17.S
LAMS = P17.LAMS


def cen(Z):
    return Z - Z.mean(1, keepdims=True)


def make(rng, n, g, acc):
    y = rng.integers(0, 4, n)
    M = [cen(1.2 * np.eye(4)[y] + rng.normal(size=(n, 4))) for _ in range(3)]
    conf = np.array([0, 1, 0, 1])                                              # 決まった取り違え：Sad→Neutral，Angry→Happy
    Ssrc = []
    for _m in range(3):
        yy = np.where(rng.random(n) < acc, y, conf[y])
        shared = rng.normal(size=(n, 4))
        for _s in range(2):
            Ssrc.append(cen(g * (np.eye(4)[yy] + 0.7 * shared + 0.7 * rng.normal(size=(n, 4)))))
    Ssrc = np.stack(Ssrc)                                                      # (6, n, 4)：並びはモダリティごとに 2 つ
    KNN = np.stack([Ssrc[2 * m:2 * m + 2].mean(0) for m in range(3)])          # 同じ情報を 3 つの項にまとめたもの
    return y, np.sum(M, 0), Ssrc.sum(0), KNN.sum(0)


def uar_adj(score_pool, score_test, y_test, target):
    dl, _ = P16.match_shift(P16.norm_log(score_pool), target)
    return 100 * S.uar(y_test, (P16.norm_log(score_test) + dl).argmax(1))


def run(g, acc, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for part in ("select", "eval"):
        yp, Mp, Sp, Np = make(rng, 4000, g, acc)                              # 候補（ラベルは使わない）
        yt, Mt, St, Nt = make(rng, 20000, g, acc)                             # テスト
        target = P16.softmax(Mp).mean(0)
        r = {"prod": uar_adj(Mp, Mt, yt, target)}
        for j, lam in enumerate(LAMS):
            r[f"MN_l{j}"] = uar_adj(Mp + lam * Np, Mt + lam * Nt, yt, target)
            r[f"MS_l{j}"] = uar_adj(Mp + lam * Sp, Mt + lam * St, yt, target)
        for lam in (0.125, 0.5):
            r[f"MS_{lam}"] = uar_adj(Mp + lam * Sp, Mt + lam * St, yt, target)
        out[part] = r
    sel, ev = out["select"], out["eval"]
    jn = max(range(len(LAMS)), key=lambda j: sel[f"MN_l{j}"])
    js = max(range(len(LAMS)), key=lambda j: sel[f"MS_l{j}"])
    return ev, jn, js


print("g：ソースの分類器の点の大きさ．acc：ソースの分類器が本当の感情を見ている割合（残りは決まった取り違え）")
print("UAR（adj．テスト 20,000 本）と，設計の 3 つの差（点）．近いソースの発話の答えの情報は，ソースの分類器と同じ．")
for g, acc in ((1.0, 0.7), (2.0, 0.7), (3.0, 0.7), (3.0, 0.55), (0.5, 0.9)):
    ev, jn, js = run(g, acc, 1)
    mn, ms, tied = ev[f"MN_l{jn}"], ev[f"MS_l{js}"], ev[f"MS_l{jn}"]
    print(f"\n g={g} acc={acc}：prod {ev['prod']:.1f}｜MS_eq λ=0.125 {ev['MS_0.125']:.1f}，0.25 {ev['MS_l0']:.1f}，0.5 {ev['MS_0.5']:.1f}，1 {ev['MS_l1']:.1f}"
          f"｜MN_eq λ=0.25 {ev['MN_l0']:.1f}，1 {ev['MN_l1']:.1f}")
    print(f"   選んだ λ：MN_eq_sel {LAMS[jn]}，MS_eq_sel {LAMS[js]}．"
          f"MN_eq_sel − prod {mn - ev['prod']:+.1f}，MN_eq_sel − MS_eq_sel {mn - ms:+.1f}，MN_eq_sel − MS_eq_tied {mn - tied:+.1f}"
          f"｜項の重みの合計をそろえた対照（MS_eq の λ を半分）との差 {mn - ev[f'MS_{LAMS[jn] / 2}']:+.2f}")
