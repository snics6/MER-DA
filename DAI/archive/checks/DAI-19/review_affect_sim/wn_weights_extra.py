"""DAI-19 の感情認識の査読想定の再検査（review_affect_v2.md）の検算の追加．
入力は results/dai16_probe/summary/reliability_src.csv だけ．MER-DA の直下で流す．
(1) 設計 6b (a) の見積もりの内訳（MOSEI を含む対，差が 1 割未満の組，第 3 のコーパスごと）．
(2) 第 3 のコーパスごとに，当てた値が最も小さいソース．
(3) 設計 6b (b) の見積もり（第 3 のコーパスでの残り方と，r_s・r'_s・r_m の差の絶対値の平均）．"""
import itertools
import pandas as pd

d = pd.read_csv("results/dai16_probe/summary/reliability_src.csv")
E = ["d_0", "d_1", "d_2", "d_3"]
d["sum"] = d[E].sum(axis=1)
H = d[d["where"] == "home"].set_index(["source", "modality"])["sum"]
aw = d[d["where"].str.startswith("away")].copy()
aw["t"] = aw["where"].str[5:]
A = aw.set_index(["source", "t", "modality"])["sum"]
C = ["AFEW", "IEMOCAP", "MELD", "MOSEI"]
M = "AVT"

rows = []
for s, t in itertools.combinations(C, 2):
    for m in M:
        w = A[(s, t, m)] / (A[(s, t, m)] + A[(t, s, m)])
        qs, qt = A[(s, t, m)] / H[(t, m)], A[(t, s, m)] / H[(s, m)]
        wn = qs / (qs + qt)
        for c in C:
            if c in (s, t):
                continue
            a, b = A[(s, c, m)], A[(t, c, m)]
            rows.append(dict(s=s, t=t, c=c, m=m, mosei=("MOSEI" in (s, t)), close=abs(a - b) / max(a, b) < 0.10,
                             w_ok=(w > 0.5) == (a > b), wn_ok=(wn > 0.5) == (a > b)))
D = pd.DataFrame(rows)
print("(1) 重みの大きいソースが，第 3 のコーパスに当てた値の大きいソースと一致する数（w，wn）")
print("  全体", D.w_ok.sum(), D.wn_ok.sum(), "/", len(D))
print("  MOSEI を含む対", D[D.mosei].w_ok.sum(), D[D.mosei].wn_ok.sum(), "/", D.mosei.sum())
print("  MOSEI を含まない対", D[~D.mosei].w_ok.sum(), D[~D.mosei].wn_ok.sum(), "/", (~D.mosei).sum())
print("  差が 1 割未満の組", D.close.sum(), "／それを除く", D[~D.close].w_ok.sum(), D[~D.close].wn_ok.sum(), "/", (~D.close).sum())
print("  第 3 のコーパスごと\n", D.groupby("c")[["w_ok", "wn_ok"]].agg(["sum", "count"]))

print("(2) 第 3 のコーパスごとに，当てた値（4 感情の合計）と，最も小さいソース")
for m in M:
    for c in C:
        v = {s: round(float(A[(s, c, m)]), 2) for s in C if s != c}
        print("  ", m, c, v, "最も小さい:", min(v, key=v.get))

rows = []
for s, t in itertools.permutations(C, 2):
    for m in M:
        rs = A[(s, t, m)] / H[(s, m)]
        rp = A[(s, t, m)] / H[(t, m)]
        rm = (A[(s, t, m)] + A[(t, s, m)]) / (H[(s, m)] + H[(t, m)])
        for c in C:
            if c in (s, t):
                continue
            tg = A[(s, c, m)] / H[(s, m)]
            rows.append(dict(m=m, 切る_r_s=abs(min(rs, 1.5) - min(tg, 1.5)), 切る_r_dash=abs(rp - min(tg, 1.5)), 切る_r_m=abs(rm - min(tg, 1.5)),
                             切らない_r_s=abs(rs - tg), 切らない_r_dash=abs(rp - tg), 切らない_r_m=abs(rm - tg)))
B = pd.DataFrame(rows)
print("(3) 第 3 のコーパスでの残り方（A_{s->c} ÷ H_s）との差の絶対値の平均（72 個）")
print(B.drop(columns="m").mean().round(3))
print(B.groupby("m").mean().round(3))
