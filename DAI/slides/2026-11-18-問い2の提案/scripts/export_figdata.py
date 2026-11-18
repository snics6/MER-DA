"""スライドの図と本文の数を，結果ファイルから書き出す（2026-11-18 の問い 2 の提案の説明）．
出力：figures/*.csv（pgfplots が読む）と figures/numbers.tex（本文の数．\\N{key} で引く）．
出どころ：
  results/dai20_eval/summary/levels.csv   まとめた答え（提案，積，比べる相手 A），単独の分類器，参照の表（4 コーパス）
  results/dai20_eval/summary/weights.csv  評価の回の提案の重みの表（50 反復の平均）
  results/dai21_eval5/summary/levels.csv  学び直した AVT（積の疑似ラベル，提案の疑似ラベル），K 本だけの AVT，CREMA-D の全部
どちらも provenance.json の dirty が偽であることを確かめる．照合を通した値（原稿 ja_v4 の 4.4 節，11-16 の資料）と一致しなければ止まる．
使い方：cd DAI/slides && make DECK=2026-11-18-問い2の提案 data
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DECK = HERE.parent
DAI = DECK.parents[1]
FIG = DECK / "figures"
FIG.mkdir(exist_ok=True)
C4 = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
C5 = C4 + ["CREMA-D"]
KS = [5, 10, 25]
NUM = {}


def kname(key):
    for a, b in [("25", "twentyfive"), ("10", "onezero"), ("0", "zero"), ("5", "five"), ("1", "one"), ("2", "two"), ("3", "three"), ("4", "four")]:
        key = key.replace(a, b)
    return re.sub(r"[^A-Za-z]", "", key)


def put(key, v, nd=2, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        s = "--"
    else:
        v = float(v)
        if round(v, nd) == 0:
            v = 0.0
        s = f"{v:+.{nd}f}" if (sign and v != 0) else f"{v:.{nd}f}"
        s = s.replace("-", "\\textminus{}")
    NUM[kname(key)] = s


def csv(name, df, src):
    with open(FIG / name, "w") as f:
        f.write(f"# {src}\n")
        df.to_csv(f, index=False)


for d in ["dai20_eval", "dai21_eval5"]:
    p = json.load(open(DAI / "results" / d / "provenance.json"))
    assert p["dirty"] is False, d

L20 = pd.read_csv(DAI / "results/dai20_eval/summary/levels.csv")
L20 = L20[L20.seeds == "all"]
L21 = pd.read_csv(DAI / "results/dai21_eval5/summary/levels.csv")
L21 = L21[L21.seeds == "all"]


def lv(c, K, ver):
    """UAR と 95% 区間．4 コーパスのまとめた答えと単独の分類器は実験 20 の集計，学び直しと CREMA-D は実験 21 の集計から引く．"""
    if c == "CREMA-D" or ver in ("stud_prod", "stud_prop", "none_AVT"):
        t = L21[(L21.c == c) & (L21.K == K) & (L21.version == ver)]
    else:
        t = L20[(L20.c == c) & (L20.K == K) & (L20.version == ver)]
    if len(t) == 0:
        return None
    assert len(t) == 1, (c, K, ver, len(t))
    r = t.iloc[0]
    assert int(r.n_fr) == 50, (c, K, ver, r.n_fr)
    return float(r.uar), float(r.lo), float(r.hi)


# 実験 20 と実験 21 で同じ版の値がそろうこと（4 コーパスの積と提案）
for c in C4:
    for K in KS:
        for ver in ["prod", "prop"]:
            a = L20[(L20.c == c) & (L20.K == K) & (L20.version == ver)].uar.iloc[0]
            b = L21[(L21.c == c) & (L21.K == K) & (L21.version == ver) & (L21.part == "4corpora")].uar.iloc[0]
            assert abs(a - b) < 1e-9, (c, K, ver, a, b)

# ---------- 質問 4：UAR の絶対値の表 ----------
MAIN = [  # (key, 版, 表の名前)
    ("konly", "none_AVT", "K 本だけで学んだ AVT"),
    ("sA", "ret_A", "声の分類器"),
    ("sV", "ret_V", "顔の分類器"),
    ("sT", "ret_T", "言葉の分類器"),
    ("prod", "prod", "積（まとめた答え）"),
    ("prop", "prop", "提案（まとめた答え）"),
    ("stprod", "stud_prod", "積の疑似ラベルで学び直した AVT"),
    ("stprop", "stud_prop", "提案の疑似ラベルで学び直した AVT"),
    ("orc", "M_F6_orc", "参照：全ラベルで測った表"),
]
rows = []
for c in C5:
    for K in KS:
        for key, ver, name in MAIN:
            v = ver
            if c == "CREMA-D" and ver == "none_AVT":
                v = "none_AV"          # CREMA-D は言葉が無いので声と顔をつないだ分類器
            x = lv(c, K, v)
            if x is None:
                assert c == "CREMA-D" and ver == "ret_T", (c, K, ver)
                put(f"u{key}{c}{K}", None)
                continue
            put(f"u{key}{c}{K}", x[0])
            rows.append(dict(c=c, K=K, key=key, version=v, uar=x[0], lo=x[1], hi=x[2]))
U = pd.DataFrame(rows)
csv("abs_uar.csv", U, "UAR（%）と補正しない 95% 区間．50 反復（分割 5 × K 本の引き方 10）の平均．"
    "4 コーパスのまとめた答え・単独の分類器・参照は results/dai20_eval/summary/levels.csv（seeds=all），"
    "学び直した AVT・K 本だけの AVT・CREMA-D は results/dai21_eval5/summary/levels.csv（seeds=all）")

# 図：コーパスごとの 4 系列（K 本だけの AVT，積，提案，提案の疑似ラベルで学び直した AVT）．横軸は K の位置 1,2,3 に少しずらして置く
off = {"konly": -0.15, "prod": -0.05, "prop": 0.05, "stprop": 0.15}
for c in C5:
    t = U[(U.c == c) & U.key.isin(off)].copy()
    t["x"] = t.K.map({5: 1, 10: 2, 25: 3}) + t.key.map(off)
    t["ep"] = t.hi - t.uar
    t["em"] = t.uar - t.lo
    for key in off:
        csv(f"abs_{kname(c)}_{key}.csv", t[t.key == key][["x", "uar", "ep", "em"]],
            f"{c}，{key}．abs_uar.csv から．x は K（1=5，2=10，3=25）に系列ごとのずらしを足した位置")

# 照合を通した値との一致（原稿 ja_v4 4.4 節と 11-16 の資料）
chk = {("AFEW", 5, "prop"): 71.36, ("AFEW", 5, "prod"): 69.23, ("AFEW", 5, "sV"): 70.70, ("AFEW", 5, "sA"): 53.24,
       ("AFEW", 5, "sT"): 45.76, ("AFEW", 10, "sV"): 71.93, ("AFEW", 10, "prod"): 70.88}
for (c, K, key), v in chk.items():
    got = float(U[(U.c == c) & (U.K == K) & (U.key == key)].uar.iloc[0])
    assert abs(round(got, 2) - v) < 1e-9, (c, K, key, got, v)

# 提案 − 積（まとめた答え）：12 条件のうち 11 で正，AFEW +2.13，+3.51，+3.14，ほかは −0.09〜+0.83（15 条件）
dpp = {(c, K): float(U[(U.c == c) & (U.K == K) & (U.key == "prop")].uar.iloc[0] - U[(U.c == c) & (U.K == K) & (U.key == "prod")].uar.iloc[0])
       for c in C5 for K in KS}
assert sum(dpp[(c, K)] > 0 for c in C4 for K in KS) == 11
assert [round(dpp[("AFEW", K)], 2) for K in KS] == [2.13, 3.51, 3.14]
oth = [dpp[(c, K)] for c in C5 if c != "AFEW" for K in KS]
assert (round(min(oth), 2), round(max(oth), 2)) == (-0.09, 0.83)
for (c, K), v in dpp.items():
    put(f"dpp{c}{K}", v, 2, True)
put("dppothmin", min(oth), 2, True); put("dppothmax", max(oth), 2, True)
afew = [dpp[("AFEW", K)] for K in KS]
put("dppafewmin", min(afew), 2, True); put("dppafewmax", max(afew), 2, True)

# 絶対値で読む量：提案（まとめた答え）のコーパスごとの範囲と，積の K = 5 → 25 の上がり幅
for c in C5:
    p = U[(U.c == c) & (U.key == "prop")].uar
    put(f"propmin{c}", p.min()); put(f"propmax{c}", p.max())
    q = U[(U.c == c) & (U.key == "prod")].set_index("K").uar
    put(f"prodgain{c}", q.loc[25] - q.loc[5], 2, True)
    o = U[(U.c == c) & (U.key == "orc")].uar
    if len(o):
        gap = [float(U[(U.c == c) & (U.K == K) & (U.key == "orc")].uar.iloc[0] - U[(U.c == c) & (U.K == K) & (U.key == "prop")].uar.iloc[0]) for K in KS]
        put(f"orcgapmin{c}", min(gap), 2, True); put(f"orcgapmax{c}", max(gap), 2, True)
uu = lambda c, K, k: float(U[(U.c == c) & (U.K == K) & (U.key == k)].uar.iloc[0])
nsk = sum(uu(c, K, "stprop") > uu(c, K, "konly") for c in C4 for K in KS)
assert nsk == 10                      # 原稿 ja_v4 4.6 節「基準の AVT との差は 12 条件のうち 10 で正」
NUM["nstpropkonly"] = str(nsk)
NUM["nstpropkonlycremad"] = str(sum(uu("CREMA-D", K, "stprop") > uu("CREMA-D", K, "konly") for K in KS))
allp = U[U.key == "prop"].uar
put("propallmin", allp.min()); put("propallmax", allp.max())

# ---------- 補足：比べる相手 A（まとめ方）の UAR の絶対値 ----------
A = [("Fone", "prod", "F1 積"), ("Ftwo", "avg", "F2 確率の平均"), ("Fthree", "psel", "F3 別のコーパスで選んだ指数の積"),
     ("Ffour", "conf", "F4 確信度で重みづける平均"), ("Ffive", "M_F2_K", "F5 水準だけの重み"), ("Fsix", "M_F6_K", "F6 全部を K 本で測った表"),
     ("Fseven", "cf_D_l05", "F7 取り違えの表で読み替えて掛ける"), ("Feight", "tcal", "F8 分類器ごとの温度の較正"),
     ("Fnine", "temp", "F9 K 本で合わせたモダリティごとの重み"), ("Ften", "wf1000", "F10 1,001 個の表から K 本で選ぶ"),
     ("prop", "prop", "提案")]
rows = []
for c in C5:
    for K in KS:
        best = -1
        for key, ver, name in A:
            x = lv(c, K, ver)
            if x is None:
                assert c == "CREMA-D" and ver == "psel"
                NUM[kname(f"a{key}{c}{K}")] = "--"
                continue
            NUM[kname(f"a{key}{c}{K}")] = f"{x[0]:.2f}"
            rows.append(dict(c=c, K=K, key=key, version=ver, uar=x[0], lo=x[1], hi=x[2]))
AA = pd.DataFrame(rows)
# 各条件で最も高い版を太字にする印
for (c, K), t in AA.groupby(["c", "K"]):
    top = t.uar.max()
    for _, r in t.iterrows():
        if r.uar == top:
            NUM[kname(f"a{r.key}{c}{K}")] = "\\textbf{" + NUM[kname(f"a{r.key}{c}{K}")] + "}"
csv("abs_A.csv", AA, "比べる相手 A と提案のまとめた答えの UAR（%）．切片をずらした答え．results/dai20_eval/summary/levels.csv と results/dai21_eval5/summary/levels.csv（CREMA-D）")
# 11-16 の資料の表（提案 − 相手）と一致すること：F8・AFEW・K=5 が −2.94，F9・AFEW・K=5 が −3.5，F10・AFEW・K=5 が −0.1
g = lambda c, K, k: float(AA[(AA.c == c) & (AA.K == K) & (AA.key == k)].uar.iloc[0])
assert round(g("AFEW", 5, "prop") - g("AFEW", 5, "Feight"), 2) == -2.94
assert round(g("AFEW", 5, "prop") - g("AFEW", 5, "Fnine"), 1) == -3.5
assert round(g("AFEW", 5, "prop") - g("AFEW", 5, "Ften"), 1) == -0.1
# 提案が A の 11 版の中で最も高かった条件の数（15 条件．CREMA-D は 10 版）
ntop = sum(int(t.loc[t.uar.idxmax(), "key"] == "prop") for _, t in AA.groupby(["c", "K"]))
NUM["ntopprop"] = str(ntop)
tops = sorted((c, K) for (c, K), t in AA.groupby(["c", "K"]) if t.loc[t.uar.idxmax(), "key"] == "prop")
assert tops == [("CREMA-D", 5), ("CREMA-D", 10), ("MOSEI", 5), ("MOSEI", 10), ("MOSEI", 25)], tops   # 本文に名前で書いた条件
assert all(AA[(AA.c == "AFEW") & (AA.K == K)].set_index("key").uar.idxmax() == "Fnine" for K in KS)
E20 = pd.read_csv(DAI / "results/dai20_eval/summary/effects.csv")
e = E20[(E20.a == "prop") & (E20.b == "temp") & (E20.c == "AFEW") & (E20.seeds == "all")]
assert len(e) == 3 and ((e.lo < 0) & (e.hi > 0)).all()     # 本文：AFEW の提案 − F9 の区間はどれも 0 を含む
# 11 版に入れなかった派生の版（F9 の切片も合わせる版，F10 の候補 10・100）を足した 14 版で数え直す
X = {"tempb": "F9 の切片も合わせる版", "wf10": "F10 の候補を 10 にした版", "wf100": "F10 の候補を 100 にした版"}
xs = []
for c in C5:
    for K in KS:
        for ver in X:
            x = lv(c, K, ver)
            if x is not None:
                xs.append(dict(c=c, K=K, key=ver, version=ver, uar=x[0]))
A14 = pd.concat([AA, pd.DataFrame(xs)], ignore_index=True)
ntop14 = sum(int(t.loc[t.uar.idxmax(), "key"] == "prop") for _, t in A14.groupby(["c", "K"]))
NUM["ntopproponefour"] = str(ntop14)
t = A14[(A14.c == "AFEW") & (A14.K == 5)]
r = t.loc[t.uar.idxmax()]
assert r.key == "tempb"
put("tempbAFEWfive", r.uar)
# 提案 − 積（まとめた答え）の区間：4 コーパスは実験 20，CREMA-D は実験 21 の集計
e4 = E20[(E20.a == "prop") & (E20.b == "prod") & (E20.seeds == "all")]
E21 = pd.read_csv(DAI / "results/dai21_eval5/summary/effects.csv")
e5 = E21[(E21.part == "cremad") & (E21.a == "prop") & (E21.b == "prod") & (E21.seeds == "all")]
assert len(e4) == 12 and len(e5) == 3
NUM["npppos"] = str(int((e4.est > 0).sum())); NUM["nppposall"] = str(int((e4.est > 0).sum() + (e5.est > 0).sum()))
NUM["npplo"] = str(int((e4.lo > 0).sum())); NUM["nppposincl"] = str(int(((e4.est > 0) & (e4.lo <= 0)).sum()))
assert (NUM["npppos"], NUM["nppposall"], NUM["npplo"]) == ("11", "14", "7")
# 設定：固定のファイルで n0 = 0 の条件の数と，形を使う（β = 1）条件の数
fz = json.load(open(DAI / "frozen/dai20_frozen.json"))["choice"]
NUM["nnzero"] = str(sum(fz[c][str(K)]["n0"] == 0 for c in C4 for K in KS))
assert all(fz[c][str(K)]["beta"] == 1 for c in C4 for K in KS)
f9 = [g("AFEW", K, "Fnine") - g("AFEW", K, "prop") for K in KS]
put("fninegapminAFEW", min(f9)); put("fninegapmaxAFEW", max(f9))
SHORT = {k: n for k, _, n in A}
for (c, K), t in AA.groupby(["c", "K"]):
    r = t.loc[t.uar.idxmax()]
    NUM[kname(f"best{c}{K}")] = SHORT[r.key]
    NUM[kname(f"bestu{c}{K}")] = f"{r.uar:.2f}"
    NUM[kname(f"bestd{c}{K}")] = ("--" if r.key == "prop" else
                                  f"{g(c, K, 'prop') - r.uar:+.2f}".replace("-", "\\textminus{}"))

# ---------- 質問 1：評価の回の重みの表（50 反復の平均） ----------
W = pd.read_csv(DAI / "results/dai20_eval/summary/weights.csv")
assert (W.n_units == 50).all()
MOD = {"A": "声", "V": "顔", "T": "言葉"}
EMO = ["N", "H", "S", "A"]
for c in C4:
    for K in KS:
        for m in MOD:
            t = W[(W.c == c) & (W.K == K) & (W["mod"] == m)].sort_values("emotion")
            assert list(t.emotion) == [0, 1, 2, 3]
            for e, v in zip(EMO, t.w_prop):
                put(f"w{c}{K}{m}{e}", v)
            put(f"wmean{c}{K}{m}", t.w_prop.mean())
        t = W[(W.c == c) & (W.K == K)]
        assert abs(t.w_prop.mean() - 1) < 1e-6, (c, K, t.w_prop.mean())   # 12 個の平均は 1
assert [round(W[(W.c == "AFEW") & (W.K == 5) & (W["mod"] == m)].w_prop.mean(), 2) for m in "AVT"] == [0.93, 1.17, 0.90]
csv("weights.csv", W[["c", "K", "mod", "emotion", "w_prop"]], "results/dai20_eval/summary/weights.csv の w_prop（提案の重み．50 反復の平均）．mod：A=声，V=顔，T=言葉．emotion：0=N，1=H，2=S，3=A")

# ---------- 書き出し ----------
with open(FIG / "numbers.tex", "w") as f:
    f.write("% scripts/export_figdata.py が書いた．手で直さない．\n")
    for k, v in sorted(NUM.items()):
        f.write(f"\\expandafter\\gdef\\csname NUM@{k}\\endcsname{{{v}}}\n")
print(len(NUM), "numbers;", len(list(FIG.glob('*.csv'))), "csv files; prop top in", ntop, "of 15")
