"""DAI-19 の感情認識の査読想定（review_affect.md）の検算．
実験 16 の保存 results/dai16_probe/summary/reliability_src.csv（ソースどうしの値だけ．ターゲットのラベルは入らない）から，
2 つのソースの単位の重み w_s = a_s / (a_s + a_t)，a の対数を「ソースの項 ＋ 当てた相手の項」に分けた割合，
コーパスごとの残り方 r_s = away / home と 1.5 で切られる数を出す．"""
import itertools
import numpy as np
import pandas as pd

d = pd.read_csv("results/dai16_probe/summary/reliability_src.csv")
d["sum"] = d[["d_0", "d_1", "d_2", "d_3"]].sum(axis=1)
H = d[d["where"] == "home"].set_index(["source", "modality"])["sum"]
aw = d[d["where"].str.startswith("away")].copy()
aw["t"] = aw["where"].str[5:]
P = aw.pivot_table(index=["source", "t"], columns="modality", values="sum")
print("home の合計\n", H.unstack().round(2))
print("away の合計（行：ソース，当てた相手）\n", P.round(2))
C = ["AFEW", "IEMOCAP", "MELD", "MOSEI"]
for s, t in itertools.combinations(C, 2):
    print(s, t, {m: round(P.loc[(s, t), m] / (P.loc[(s, t), m] + P.loc[(t, s), m]), 2) for m in "ATV"}, f"(w_{s})")
print("当てた相手ごとの平均\n", aw.groupby(["t", "modality"])["sum"].mean().unstack().round(2))
print("ソースごとの平均\n", aw.groupby(["source", "modality"])["sum"].mean().unstack().round(2))
print("当てた相手ごとの Neutral の d′\n", aw.groupby(["t", "modality"])["d_0"].mean().unstack().round(2))
for m in "ATV":
    x = aw[aw.modality == m].copy()
    x["la"] = np.log(x["sum"])
    g = x["la"].mean()
    se = x.groupby("source")["la"].mean() - g
    te = x.groupby("t")["la"].mean() - g
    tot = ((x.la - g) ** 2).sum()
    print(m, "ソースの項", round((x.source.map(se) ** 2).sum() / tot, 2), "相手の項", round((x.t.map(te) ** 2).sum() / tot, 2))
n = nc = 0
for _, r in aw.iterrows():
    rr = r["sum"] / H[(r.source, r.modality)]
    n += 1
    nc += rr > 1.5
    if rr > 1.5:
        print("1.5 を超える：", r.source, "->", r.t, r.modality, round(rr, 2))
print(nc, "/", n)
