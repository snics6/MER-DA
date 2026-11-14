"""DAI-19 の感情認識の査読想定の再検査（review_affect_v2.md）の検算．
入力は results/dai16_probe/summary/reliability_src.csv だけ（ソースどうしの値．声・顔・言葉）．
設計の第 2 版で足した腕 wn（w'_s = (A_s/H_t) / (A_s/H_t + A_t/H_s)）と表 extpcn（r'_s = A_s/H_t）を，
計算の前にソースの表から見積もる．GPU は使わない．MER-DA の直下で流す．"""
import itertools
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
d = pd.read_csv("results/dai16_probe/summary/reliability_src.csv")
E = ["d_0", "d_1", "d_2", "d_3"]
d["sum"] = d[E].sum(axis=1)
home = d[d["where"] == "home"].set_index(["source", "modality"])
H = home["sum"]
aw = d[d["where"].str.startswith("away")].copy()
aw["t"] = aw["where"].str[5:]
aw = aw.set_index(["source", "t", "modality"])
A = aw["sum"]
C = ["AFEW", "IEMOCAP", "MELD", "MOSEI"]
M = ["A", "V", "T"]

# 1. q[s->t] = A_{s->t} / H_t（s の分類器が，t 自身で学んだ分類器の何割まで届くか）
print("== 1. q[s->t] = A_{s->t} / H_t（行：分類器のソース s，列：当てた相手 t）")
Q = {}
for m in M:
    tab = pd.DataFrame(index=C, columns=C, dtype=float)
    for s, t in itertools.permutations(C, 2):
        Q[(s, t, m)] = A[(s, t, m)] / H[(t, m)]
        tab.loc[s, t] = Q[(s, t, m)]
    print(m)
    print(tab.round(2))
    # 当てた相手ごとに，届く割合の順位（1 = 最も届く）
    rk = tab.rank(axis=0, ascending=False)
    print("  相手ごとの順位（1 が最も届く）\n", rk)

# 2. 重み w と wn
print("\n== 2. 重み（最初に書いたコーパスの重み）")
rows = []
for s, t in itertools.combinations(C, 2):
    for m in M:
        w = A[(s, t, m)] / (A[(s, t, m)] + A[(t, s, m)])
        wn = Q[(s, t, m)] / (Q[(s, t, m)] + Q[(t, s, m)])
        rows.append(dict(s=s, t=t, m=m, w_s=w, wn_s=wn, H_s=H[(s, m)], H_t=H[(t, m)],
                         A_s=A[(s, t, m)], A_t=A[(t, s, m)], q_s=Q[(s, t, m)], q_t=Q[(t, s, m)]))
W = pd.DataFrame(rows)
print(W.round(3).to_string(index=False))
for col in ["w_s", "wn_s"]:
    big = np.maximum(W[col], 1 - W[col])
    print(col, "大きいほうの重み：最小 %.3f 中央 %.3f 最大 %.3f ／ |w-0.5| の中央 %.3f ／ 0.45〜0.55 に入る組 %d/18" % (
        big.min(), big.median(), big.max(), (W[col] - 0.5).abs().median(), ((W[col] - 0.5).abs() <= 0.05).sum()))

# コーパスごとに，重みが 0.5 を超える数
print("\n コーパスごとの「重み > 0.5」の数（9 個中）と重みの範囲")
for c in C:
    for col in ["w_s", "wn_s"]:
        v = list(W.loc[W.s == c, col]) + list(1 - W.loc[W.t == c, col])
        print(f"  {c:8s} {col:5s} >0.5: {sum(x > 0.5 for x in v)}/9  範囲 {min(v):.2f}〜{max(v):.2f}")

# 3. 重みの大きいソースは，相手の中の値が高いソースか，自分の中の値が高いソースか（a2）
print("\n== 3. 重みの大きいソースと中の値（18 組）")
for col in ["w_s", "wn_s"]:
    big_is_s = W[col] > 0.5
    partner_home_higher = np.where(big_is_s, W.H_t > W.H_s, W.H_s > W.H_t)
    print(f"  {col}: 重みの大きいソースが『相手の中の値が高い』ソース {partner_home_higher.sum()}/18，『自分の中の値が高い』ソース {(~partner_home_higher).sum()}/18")
# wn：重みの大きいソースは，同じ相手に当てた 3 つのソースの中で届く割合が何位か
print("  wn で重みが大きいソースは q の大きいほう（定義どおり）．相手側の q の小ささで決まるかを下の分解で見る")

# 4. 対数の散らばりを「ソースの項 ＋ 当てた相手の項」に分ける（最小二乗．対角が無いので釣り合わない配置）
print("\n== 4. 対数の散らばりの分解（最小二乗の決定係数．12 個の値）")


def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()


for m in M:
    pairs = list(itertools.permutations(C, 2))
    S = np.array([[float(s == c) for c in C[1:]] for s, t in pairs])
    T = np.array([[float(t == c) for c in C[1:]] for s, t in pairs])
    for name, y in [("log A      ", np.log([A[(s, t, m)] for s, t in pairs])),
                    ("log(A/H_t) ", np.log([Q[(s, t, m)] for s, t in pairs]))]:
        print(f"  {m} {name} ソースだけ {r2(y, S):.2f}  相手だけ {r2(y, T):.2f}  両方 {r2(y, np.column_stack([S, T])):.2f}")

# 5. ソースの表だけからの (a) の見積もり：重みの大きいソースが，第 3 のコーパス c で A の大きいソースか
#    （c は単位の仮のターゲットに当たる．A_{s->c} は c の全発話の上の値で，設計 6b (a) の量（c の候補の上の値）に近い）
print("\n== 5. 重みの大きいソースが，第 3 のコーパスに当てた値の大きいソースと一致する数（声・顔・言葉 36 個）")
cnt = {"w_s": 0, "wn_s": 0}
det = []
for _, r in W.iterrows():
    for c in C:
        if c in (r.s, r.t):
            continue
        s_stronger = A[(r.s, c, r.m)] > A[(r.t, c, r.m)]
        for col in cnt:
            ok = (r[col] > 0.5) == s_stronger
            cnt[col] += ok
        det.append(dict(s=r.s, t=r.t, c=c, m=r.m, w_s=round(r.w_s, 2), wn_s=round(r.wn_s, 2),
                        A_s_c=round(A[(r.s, c, r.m)], 2), A_t_c=round(A[(r.t, c, r.m)], 2),
                        w_ok=(r.w_s > 0.5) == s_stronger, wn_ok=(r.wn_s > 0.5) == s_stronger))
D = pd.DataFrame(det)
print(D.to_string(index=False))
print("  一致：", cnt, "/ 36")
print("  モダリティごと\n", D.groupby("m")[["w_ok", "wn_ok"]].sum())
print("  ソースの対ごと\n", D.groupby(["s", "t"])[["w_ok", "wn_ok"]].sum())
far = D[(D.wn_s - 0.5).abs() > 0.05]
print("  wn の重みが 0.45〜0.55 の外の組だけ：", int(far.wn_ok.sum()), "/", len(far))

# 6. 表 ext・extpc・extpcn（eq の腕．v_s = 0.5）の行の水準と，各ソースの分
print("\n== 6. eq の腕の表の水準（4 感情の合計 ÷ 2 ソース）と，最初のコーパスの分")
rows = []
nclip = 0
for _, r in W.iterrows():
    rm = (r.A_s + r.A_t) / (r.H_s + r.H_t)
    rs, rt = r.A_s / r.H_s, r.A_t / r.H_t
    rsc, rtc = min(rs, 1.5), min(rt, 1.5)
    qs, qt = r.q_s, r.q_t
    nclip += (qs > 1.5) + (qt > 1.5)
    ext = (r.H_s * rm, r.H_t * rm)
    pc = (r.H_s * rsc, r.H_t * rtc)
    pcn = (r.H_s * qs, r.H_t * qt)
    rows.append(dict(s=r.s, t=r.t, m=r.m,
                     ext_lv=sum(ext) / 2, pc_lv=sum(pc) / 2, pcn_lv=sum(pcn) / 2,
                     ext_share_s=ext[0] / sum(ext), pc_share_s=pc[0] / sum(pc), pcn_share_s=pcn[0] / sum(pcn),
                     pcn_row_s=pcn[0], A_s=r.A_s, pcn_row_t=pcn[1], A_t=r.A_t,
                     wn_pcn_share_s=r.wn_s * pcn[0] / (r.wn_s * pcn[0] + (1 - r.wn_s) * pcn[1])))
X = pd.DataFrame(rows)
print(X.round(2).to_string(index=False))
print("  r' = A_s/H_t が 1.5 を超える向き：", nclip, "/ 36．最大", round(max(Q.values()), 2))
print("  水準の比 extpcn ÷ ext：最小 %.2f 中央 %.2f 最大 %.2f" % tuple((X.pcn_lv / X.ext_lv).quantile([0, .5, 1])))
print("  行の水準 ÷ 実際に測った away（extpcn）：最小 %.2f 最大 %.2f" % (
    min((X.pcn_row_s / X.A_s).min(), (X.pcn_row_t / X.A_t).min()), max((X.pcn_row_s / X.A_s).max(), (X.pcn_row_t / X.A_t).max())))
hi = np.where(W.H_s > W.H_t, X.pcn_share_s, 1 - X.pcn_share_s)
print("  extpcn：中の値が高いソースの分 最小 %.2f 中央 %.2f 最大 %.2f（0.5 超 %d/18）" % (hi.min(), np.median(hi), hi.max(), (hi > 0.5).sum()))
hi = np.where(W.H_s > W.H_t, X.ext_share_s, 1 - X.ext_share_s)
print("  ext   ：中の値が高いソースの分 最小 %.2f 中央 %.2f 最大 %.2f" % (hi.min(), np.median(hi), hi.max()))
hi = np.where(W.H_s > W.H_t, X.pc_share_s, 1 - X.pc_share_s)
print("  extpc ：中の値が高いソースの分 最小 %.2f 中央 %.2f 最大 %.2f（0.5 超 %d/18）" % (hi.min(), np.median(hi), hi.max(), (hi > 0.5).sum()))

# 7. 感情ごと：中の Neutral と，外から当てた Neutral
print("\n== 7. Neutral（d_0）：中の値と，外から当てた値（3 ソースの平均），その比．ほかの 3 感情の比も")
for m in M:
    for c in C:
        hn = home.loc[(c, m), "d_0"]
        an = np.mean([aw.loc[(s, c, m), "d_0"] for s in C if s != c])
        ho = home.loc[(c, m), E[1:]].sum()
        ao = np.mean([aw.loc[(s, c, m), E[1:]].sum() for s in C if s != c])
        print(f"  {m} {c:8s} 中 Neutral {hn:.2f}  外 {an:.2f}  比 {an / hn:.2f} ｜ ほかの 3 感情：中 {ho:.2f} 外 {ao:.2f} 比 {ao / ho:.2f}")
