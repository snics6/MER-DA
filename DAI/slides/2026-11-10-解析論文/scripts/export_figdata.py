"""スライドの図と本文の数を，結果ファイルから書き出す（2026-11-10 の解析論文の資料）．
出力：figures/*.csv（pgfplots が読む）と figures/numbers.tex（本文の数．\\N{key} で引く）．
導入（parts/a_intro.tex）の出どころ：評価の回（コミット ba98974，4 コーパス × 50 単位）の
results/explore_fusion/fusion_cells.csv（AVT の K 本だけの版）と results/dai3/eval/decomp.csv（version four の AVT|start の sum）．
作り方は 2026-11-05-途中経過/scripts/export_figdata.py の「難しさ」と同じ．
使い方：cd DAI/slides && make DECK=2026-11-10-解析論文 data
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DECK = HERE.parent
ROOT = DECK.parents[2]          # MER-DA
FIG = DECK / "figures"
FIG.mkdir(exist_ok=True)
CORPORA = ["AFEW", "IEMOCAP", "MELD", "MOSEI"]
NUM = {}


def put(key, v, nd=1, sign=False):
    key = re.sub(r"[^A-Za-z]", "", key.replace("25", "twentyfive").replace("0", "zero").replace("5", "five")
                 .replace("1", "one").replace("2", "two").replace("3", "three").replace("4", "four"))
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        s = "--"
    else:
        if round(float(v), nd) == 0:
            v = 0.0                                   # −0.0 を書かない
        s = f"{v:+.{nd}f}" if (sign and v != 0) else f"{v:.{nd}f}"
        s = s.replace("-", "\\textminus{}")
    NUM[key] = s


def csv(name, df, src):
    with open(FIG / name, "w") as f:
        f.write(f"# {src}\n")
        df.to_csv(f, index=False)


# ---------- 導入：ほかのコーパスで学んだ分類器と，同じコーパスの全ラベルで学んだ分類器（AVT） ----------
dc = pd.read_csv(ROOT / "results/dai3/eval/decomp.csv")
assert dc.commit.str.startswith("ba98974").all() and not dc.dirty.any()
dc = dc[(dc.version == "four") & (dc.quantity == "sum")]
fu = pd.read_csv(ROOT / "results/explore_fusion/fusion_cells.csv")
rows = []
for i, T in enumerate(CORPORA):
    start = float(fu[(fu["T"] == T) & (fu.K == 0)].avt_none.iloc[0])
    k25 = float(fu[(fu["T"] == T) & (fu.K == 25)].avt_none.iloc[0])
    gap0 = float(dc[(dc["T"] == T) & (dc.item == "AVT|start")].effect.iloc[0])
    gap25 = float(dc[(dc["T"] == T) & (dc.item == "AVT|none|25")].effect.iloc[0])
    upper = start + gap0
    assert abs(upper - (k25 + gap25)) < 0.1, (T, upper, k25 + gap25)   # 2 通りの作り方で上限が一致する
    rows.append(dict(ci=i + 1, T=T, start=start, upper=upper, gap=gap0))
    put(f"istart_{T}", start); put(f"iupper_{T}", upper); put(f"igap_{T}", gap0)
D = pd.DataFrame(rows)
put("igapmin", float(D.gap.min())); put("igapmax", float(D.gap.max()))
csv("intro_drop.csv", D, "AVT の分類器．start = ほかのコーパスで学んだ分類器（ターゲットのラベル 0 本．fusion_cells.csv の avt_none K=0），"
    "upper = 同じコーパスの候補の全ラベルで学んだ重みに，テストの発話のラベルで切片だけを（感情の釣り合いの重みつきで）解き直した分類器（start ＋ decomp.csv version four の AVT|start の sum）．"
    "評価の回 ba98974")

# ---------- 導入：コーパスと感情ごとの発話数（data/README.md の「クラス別の件数と話者群の数」の表．音声 WavLM の件数） ----------
GROUPS = {"IEMOCAP-impro": "iemoimpro", "IEMOCAP-script": "iemoscript", "MOSEI": "mosei", "MELD": "meld",
          "AFEW": "afew", "RAVDESS": "ravdess", "CREMA-D": "cremad"}
readme = (ROOT / "data/README.md").read_text(encoding="utf-8")
sec = readme.split("クラス別の件数と話者群の数は次のとおりである")[1].split("MOSEI の話者群 2,626")[0]
rows = []
for line in sec.splitlines():
    m = re.match(r"\|\s*([A-Za-z\-]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|", line)
    if not m or m.group(1) not in GROUPS:
        continue
    g = GROUPS[m.group(1)]
    spk, neu, hap, sad, ang = (int(x.replace(",", "")) for x in m.groups()[1:])
    rows.append(dict(group=m.group(1), speakers=spk, neu=neu, hap=hap, sad=sad, ang=ang, total=neu + hap + sad + ang))
    for k, v in [("spk", spk), ("neu", neu), ("hap", hap), ("sad", sad), ("ang", ang), ("tot", neu + hap + sad + ang)]:
        NUM[f"c{k}{g}"] = f"{v:,}".replace(",", "{,}")
assert len(rows) == 7, rows
csv("corpora.csv", pd.DataFrame(rows), "data/README.md の「クラス別の件数と話者群の数」の表（音声 WavLM で特徴の取れた発話）を読み込んだもの")

# ---------- 結果の前半：組の設定（E76 の再解析．results/e76_analysis/．同じ 7 群・反復 0〜9 の再推定で確認ではない） ----------
import json
E76 = ROOT / "results/e76_analysis"
def read76(name):
    path = E76 / name
    head = json.loads(path.open(encoding="utf-8").readline().lstrip("# ").strip())
    assert head["commit"].startswith("e39e54a") and head["dirty"] is False, (name, head.get("commit"), head.get("dirty"))
    return pd.read_csv(path, comment="#")
KINDS = ["A", "A(HuBERT)", "V", "T", "AV", "VT", "AT", "AVT"]          # 並び順（memory の決まり）
KEY = {"A": "a", "A(HuBERT)": "ah", "V": "v", "T": "t", "AV": "av", "VT": "vt", "AT": "at", "AVT": "avt"}
kind = read76("e76a_1_kind.csv")
KLAB = ["1", "5", "10", "25", "50", "all"]
rows = []
for k in KINDS:
    d = kind[kind.short == k].set_index("K")
    row = {"kind": KEY[k]}
    for i, KK in enumerate(KLAB):
        r = d.loc[KK]
        row[f"g{i}"] = r.mean_gain; row[f"lo{i}"] = r.lo; row[f"hi{i}"] = r.hi
    rows.append(row)
    r1, ra = d.loc["1"], d.loc["all"]
    put(f"gone_{KEY[k]}", r1.mean_gain); put(f"goneci_{KEY[k]}", r1.lo); put(f"gonecihi_{KEY[k]}", r1.hi)
    put(f"gall_{KEY[k]}", ra.mean_gain); put(f"gallci_{KEY[k]}", ra.lo); put(f"gallcihi_{KEY[k]}", ra.hi)
    NUM[f"gonen{KEY[k]}"] = str(int(r1.n_pairs)); NUM[f"goneup{KEY[k]}"] = str(int(r1.n_up_holm)); NUM[f"galldown{KEY[k]}"] = str(int(ra.n_down_holm))
    assert r1.lo > 0 and ra.hi < 0, k   # 本文の「K=1 で 8 種類すべて区間が 0 より上，全件で 8 種類すべて 0 より下」
W = pd.DataFrame(rows)
# 折れ線用（長い形）と K=1・全件の点
long = []
for i, KK in enumerate(KLAB):
    rec = {"x": i + 1}
    for _, r in W.iterrows():
        rec[r.kind] = r[f"g{i}"]
    long.append(rec)
csv("gain_curve.csv", pd.DataFrame(long), "e76a_1_kind.csv の mean_gain（合併 − ターゲットのみ，種類の組の平均）．x = 1..6 は K = 1, 5, 10, 25, 50, 全件")
# 要約（parts/a0_summary.tex）：声の 3 つの学び方の UAR を K ごとに（ソースのみ・合併・ターゲットのみ）．
# 出どころは E76 と同じ地図 results/fixc_map/audio_wavlm/（合併の C はソースのみの C に固定）．反復 0〜9，IEMOCAP の即興と台本の組を除く 40 組の平均．
# 合併 − ターゲットのみ が e76a_1_kind.csv の mean_gain（A）と一致することを確かめる
FX = ROOT / "results/fixc_map/audio_wavlm"
fx = []
for f in sorted(FX.glob("*.csv")):
    head = json.loads(f.open(encoding="utf-8").readline().lstrip("# ").strip())
    assert head["script"] == "exp/fixc_map.py" and head["dirty"] is False, f
    fx.append(pd.read_csv(f, comment="#"))
fx = pd.concat(fx)
fx = fx[fx.rep.between(0, 9)]
SAME = {("IEMOCAP-impro", "IEMOCAP-script"), ("IEMOCAP-script", "IEMOCAP-impro")}
so = fx[fx.method == "src_only"].groupby(["target", "source"]).uar.mean()
pairs = [(t, sc) for t, sc in so.index if t != sc and (t, sc) not in SAME]
assert len(pairs) == 40
mo = fx[fx.method == "merged"].groupby(["target", "source", "K"]).uar.mean()
to = fx[fx.method == "tgt_only"].groupby(["target", "K"]).uar.mean()
rows = []
dA = kind[kind.short == "A"].set_index("K")
for i, KK in enumerate(KLAB):
    S_ = 100 * np.mean([so[(t, sc)] for t, sc in pairs])
    M_ = 100 * np.mean([mo[(t, sc, KK)] for t, sc in pairs])
    T_ = 100 * np.mean([to[(t, KK)] for t, sc in pairs])
    assert abs((M_ - T_) - dA.loc[KK].mean_gain) < 0.01, (KK, M_ - T_, dA.loc[KK].mean_gain)
    rows.append(dict(x=i + 1, K=KK, src=S_, merge=M_, tgt=T_))
KC = pd.DataFrame(rows)
csv("kcurve_a.csv", KC, "results/fixc_map/audio_wavlm/*.csv（声，合併の C はソースのみの C）．反復 0〜9，40 組（IEMOCAP の即興と台本の組を除く）の平均の UAR（%）．"
    "src = ソースのみ，merge = 合併，tgt = ターゲットのみ．x = 1..6 は K = 1, 5, 10, 25, 50, 全件．merge − tgt は e76a_1_kind.csv の A の mean_gain と 0.01 以内で一致")
put("kcsrc", float(KC.src.iloc[0]))
ends = pd.DataFrame([dict(x=i + 1, kind=r.kind, g=r[f"g{j}"], ep=r[f"hi{j}"] - r[f"g{j}"], em=r[f"g{j}"] - r[f"lo{j}"])
                     for i, r in W.iterrows() for j in [0, 5]])
for j, nm in [(0, "gain_k1.csv"), (5, "gain_all.csv")]:
    e = pd.DataFrame([dict(x=i + 1, kind=r.kind, g=r[f"g{j}"], ep=r[f"hi{j}"] - r[f"g{j}"], em=r[f"g{j}"] - r[f"lo{j}"]) for i, r in W.iterrows()])
    csv(nm, e, f"e76a_1_kind.csv の K={KLAB[j]} の mean_gain と区間（lo, hi）．x は種類の並び（A, A(HuBERT), V, T, AV, VT, AT, AVT）")
gmin = W.loc[W.g0.idxmin()]; gmax = W.loc[W.g0.idxmax()]
NUM["gonemink"] = [k for k in KINDS if KEY[k] == gmin.kind][0]; NUM["gonemaxk"] = [k for k in KINDS if KEY[k] == gmax.kind][0]
put("gonemin", gmin.g0); put("gonemax", gmax.g0)

q4 = read76("e76a_4_kind.csv")
rows = []
for i, k in enumerate(KINDS):
    d = q4[(q4.short == k) & (q4.K == 5)].set_index("quantity")
    rec = dict(x=i + 1, kind=KEY[k])
    for qn, col in [("G_merge", "merge"), ("G_crit", "crit"), ("G_center", "center"), ("G_crit_c", "critc")]:
        r = d.loc[qn]
        rec[col] = r["mean"]; rec[col + "_ep"] = r.hi - r["mean"]; rec[col + "_em"] = r["mean"] - r.lo
        put(f"{col}_{KEY[k]}", r["mean"], 2); put(f"{col}lo_{KEY[k]}", r.lo, 2); put(f"{col}hi_{KEY[k]}", r.hi, 2)
    rec["ratio"] = rec["crit"] / rec["merge"]
    put(f"ratio_{KEY[k]}", rec["ratio"], 2)
    rows.append(rec)
C5 = pd.DataFrame(rows)
# 感度（注に書く）：切片の罰則を 10 倍（l2 = 0.1），中心合わせの平均をプール（候補）だけで取る版
for k in ["A", "A(HuBERT)"]:
    d = q4[(q4.short == k) & (q4.K == 5)].set_index("quantity")
    put(f"critlt_{KEY[k]}", d.loc["G_crit_l2_0.1", "mean"], 2)
for k in KINDS:
    d = q4[(q4.short == k) & (q4.K == 5)].set_index("quantity")
    put(f"centerpool_{KEY[k]}", d.loc["G_center_pool", "mean"], 2)
    r = d.loc["G_crit_c_pool"]; put(f"critcpool_{KEY[k]}", r["mean"], 2); put(f"critcpoollo_{KEY[k]}", r.lo, 2); put(f"critcpoolhi_{KEY[k]}", r.hi, 2)
NUM["critcpoolpos"] = ", ".join(k for k in KINDS if q4[(q4.short == k) & (q4.K == 5) & (q4.quantity == "G_crit_c_pool")].lo.iloc[0] > 0)
NUM["critcpos"] = str(int(((C5.critc - C5.critc_em) > 0).sum()))   # 区間の下端が 0 より上の種類の数（本文では 0 と書く）
assert NUM["critcpos"] == "0"
csv("crit_merge_k5.csv", C5, "e76a_4_kind.csv の K=5．merge = G_merge（合併 − ソースのみ），crit = G_crit（ソースのみ＋K 本で切片だけ直す − ソースのみ），"
    "center = G_center（中心合わせ − ソースのみ．ラベル不要），critc = G_crit_c（中心合わせの後に K 本で切片を直す得）．_ep/_em は区間の上と下までの幅．定義は docs/exploratory-analyses.md 3251〜3252 行")
# 要約のページ（parts/a0_summary.tex）：声・顔・言葉の 3 種類だけを抜き出し，x を 1〜3 に付け直す
S3 = C5[C5.kind.isin(["a", "v", "t"])].copy()
assert list(S3.kind) == ["a", "v", "t"]
S3["x"] = range(1, 4)
csv("sum_crit.csv", S3, "crit_merge_k5.csv の a・v・t の行（要約のページ用．x を 1〜3 に付け直した．値は crit_merge_k5.csv と同じ）")

qs = read76("e76a_3_q_summary.csv")
qs = qs[(qs.rule == "main") & (qs.D_version == "D") & (qs.fit == "3pt")]
rows = []
for i, k in enumerate(KINDS):
    r = qs[qs.short == k].iloc[0]
    rows.append(dict(x=i + 1, kind=KEY[k], q=r.median_q_incl_edge, ep=r.q75 - r.median_q_incl_edge, em=r.median_q_incl_edge - r.q25, ntarget=int(r.n_targets)))
    put(f"q_{KEY[k]}", r.median_q_incl_edge, 1)
Q = pd.DataFrame(rows)
csv("q_kind.csv", Q, "e76a_3_q_summary.csv の rule=main, D_version=D, fit=3pt（K = 1, 5, 10 の当てはめ）．q はターゲットごとの中央値，ep/em は四分位までの幅")
Ks = np.geomspace(1, 50, 80)
qc = pd.DataFrame({"K": Ks, "inrange": (Ks <= 10).astype(int)})
for k in ["V", "A", "T"]:
    qq = float(Q[Q.kind == KEY[k]].q.iloc[0])
    qc[KEY[k]] = 1 / np.sqrt(1 + qq / Ks)
    put(f"qten_{KEY[k]}", 100 / np.sqrt(1 + qq / 10), 0)      # K = 10 で最良の向きの離れ方の何 % か（式の値）
    for KKx, nm in [(1, "one"), (5, "five"), (50, "fifty")]:
        put(f"qf_{KEY[k]}_{nm}", 1 / np.sqrt(1 + qq / KKx), 2)    # 式の値（導出のページの数値例）
csv("q_curve.csv", qc, "当てはめた式 d_T(K)/D = 1/sqrt(1+q/K) に，q_kind.csv の中央値（V，A，T）を入れた曲線．当てはめに使ったのは K = 1, 5, 10（inrange = 1 は K ≤ 10）．10 より先は式の延長")
# 実測の点：e76a_3_dT.csv の rule=main．d_T/D を 6 対で平均し，ターゲットの中央値を取る（K = 1, 5, 10．当てはめに使った K）
dT = read76("e76a_3_dT.csv")
dT = dT[dT["rule"] == "main"].assign(r=lambda x: x.d_T / x.D)
gT = dT.groupby(["short", "target", "K"]).r.mean().reset_index().groupby(["short", "K"]).r.median()
qp = pd.DataFrame({"K": [1, 5, 10]})
for k in ["V", "A", "T"]:
    qp[KEY[k]] = [float(gT[(k, str(KK))]) for KK in [1, 5, 10]]
csv("q_pts.csv", qp, "e76a_3_dT.csv（rule=main）の d_T/D を 6 対で平均し，ターゲットの中央値を取った実測の点（K = 1, 5, 10）")
# 33 ページの補い（2026-11-11 追加．研究者「これは理論？導出も見せて．何が言えるんですか？今回の結果は理論に則った？A，V，T の違いは？」）
# 出どころは同じ e76a_3_*（E76 の 11 節の結果．docs/exploratory-analyses.md 9009〜9010 行）
KK = ["1", "5", "10", "25", "50", "all"]
rows = []
for k in ["A", "V", "T"]:
    rec = {"kind": KEY[k]}
    for KKk in KK:
        v = float(gT[(k, KKk)]); rec[f"r{KKk}"] = v
        put(f"dtr_{KEY[k]}_{ {'1':'one','5':'five','10':'ten','25':'tf','50':'fifty','all':'all'}[KKk]}", v, 2)
    rows.append(rec)
csv("q_ratio_table.csv", pd.DataFrame(rows), "e76a_3_dT.csv（rule=main）の d_T/D を 6 対で平均し，ターゲットの中央値（K = 1〜全件）")
gD = read76("e76a_3_dT_gt_D.csv")
for k in ["A", "V", "T"]:
    for KKk, nm in [("10", "ten"), ("all", "all")]:
        r = gD[(gD.short == k) & (gD.K.astype(str) == KKk)].iloc[0]
        NUM[f"gtd{KEY[k]}{nm}"] = str(int(r.n_gt_D)); NUM[f"gtdn{KEY[k]}"] = str(int(r.n_cells))
qsD = read76("e76a_3_q_summary.csv")
qtD = read76("e76a_3_q_targets.csv")
for k in ["A", "V", "T"]:
    r = qsD[(qsD.short == k) & (qsD["rule"] == "main") & (qsD.D_version == "Dprime") & (qsD.fit == "3pt")].iloc[0]
    put(f"qdp_{KEY[k]}", r.median_q_incl_edge, 1)
    x = qtD[(qtD.short == k) & (qtD["rule"] == "main") & (qtD.D_version == "D") & (qtD.fit == "3pt")]
    put(f"qmin_{KEY[k]}", x.q.min(), 1); put(f"qmax_{KEY[k]}", x.q.max(), 1)
    put(f"qpairratio_{KEY[k]}", float(x.q_pair_max_over_min.median()), 1)
    put(f"dmed_{KEY[k]}", float(x.D_mean.median()), 2)
    dd = dT[dT.short == k]
    put(f"dpd_{KEY[k]}", float((dd.Dprime / dd.D).median()), 2)
    x5 = qtD[(qtD.short == k) & (qtD["rule"] == "main") & (qtD.D_version == "Dprime") & (qtD.fit == "5pt")]
    NUM[f"qloodp{KEY[k]}"] = str(int(x5.M1orM2_better_loko.astype(str).eq("True").sum())); NUM[f"qndp{KEY[k]}"] = str(len(x5))
    rr = dd.assign(rp=dd.d_T / dd.Dprime).groupby(["target", "K"]).rp.mean().reset_index().groupby("K").rp.median()
    put(f"dtrp_{KEY[k]}_ten", float(rr["10"]), 2)
    put(f"qratio_{KEY[k]}", float(x.q.max() / x.q.min()), 0)
# 同じターゲットで言葉と声の q を比べる（D と D′）
for ver, nm in [("D", "d"), ("Dprime", "dp")]:
    qa_ = qtD[(qtD.short == "A") & (qtD["rule"] == "main") & (qtD.D_version == ver) & (qtD.fit == "3pt")].set_index("target").q
    qt_ = qtD[(qtD.short == "T") & (qtD["rule"] == "main") & (qtD.D_version == ver) & (qtD.fit == "3pt")].set_index("target").q
    common = qt_.index.intersection(qa_.index)
    NUM[f"tslower{nm}"] = str(int((qt_[common] > qa_[common]).sum())); NUM[f"tcommon{nm}"] = str(len(common))
    put(f"qmeldt{nm}", float(qt_["MELD"]), 1); put(f"qmelda{nm}", float(qa_["MELD"]), 1)


# ---------- 結果の後半①：世界の設定（E79．向きの損と UAR の単位の判定．main の行） ----------
def judge_counts(path, family):
    u = pd.read_csv(path, comment="#", low_memory=False)
    m = u[(u.family == family) & (u.sensitivity == "main")]
    m = m[m.judgment != "missing"]
    return {j: int((m.judgment == j).sum()) for j in ["pos", "same", "undet", "neg", "inactive"]}, len(m)
P4c, P4n = judge_counts(ROOT / "results/e79_p4/judge/units.csv", "P4")
P53c, P53n = judge_counts(ROOT / "results/e79/judge/units.csv", "P5-3")
P53uc, P53un = judge_counts(ROOT / "results/e79/judge/units.csv", "P5-3u")
P52c, P52n = judge_counts(ROOT / "results/e79/judge/units.csv", "P5-2")
P51c, P51n = judge_counts(ROOT / "results/e79/judge/units.csv", "P5-1")
assert (P4n, P4c["pos"], P4c["neg"]) == (210, 7, 79), (P4n, P4c)          # 記録（docs/exploratory-analyses.md 9693 行）と一致
assert (P53n, P53c["pos"], P53c["neg"]) == (81, 38, 1), (P53n, P53c)      # 同 9584 行
for tag, c, n in [("pf", P4c, P4n), ("pft", P53c, P53n), ("pftu", P53uc, P53un), ("pftwo", P52c, P52n), ("pfone", P51c, P51n)]:
    NUM[f"w{tag}n"] = str(n)
    for j, v in c.items():
        NUM[f"w{tag}{j}"] = str(v)
put("wpfchance", P4n * 0.025, 2); put("wpftchance", P53n * 0.025, 2)
def sens_counts(path, family, sens):
    u = pd.read_csv(path, comment="#", low_memory=False)
    m = u[(u.family == family) & (u.sensitivity == sens) & (u.judgment != "missing")]
    return int((m.judgment == "pos").sum()), int((m.judgment == "neg").sum())
for tag, path, fam_ in [("pf", ROOT / "results/e79_p4/judge/units.csv", "P4"), ("pft", ROOT / "results/e79/judge/units.csv", "P5-3")]:
    for sens, sk in [("delta_half", "half"), ("delta_double", "dbl")]:
        pos, neg = sens_counts(path, fam_, sens)
        NUM[f"w{tag}{sk}pos"] = str(pos); NUM[f"w{tag}{sk}neg"] = str(neg)
# 1 コーパスを抜く設定：検定に残ったセル（族 1〜3）
_f = pd.read_csv(ROOT / "results/dai1/stats/families_flagged.csv")
_f = _f[_f.family.isin([1, 2, 3])]
NUM["locells"] = str(len(_f)); NUM["locellstest"] = str(int((~_f.descriptive.astype(bool)).sum()))
rows = []
for i, (lab, c, n) in enumerate([("p4", P4c, P4n), ("p53", P53c, P53n)]):
    rows.append(dict(y=i + 1, lab=lab, n=n, **{j: 100 * v / n for j, v in c.items()}))
csv("world_dir.csv", pd.DataFrame(rows), "向きの損の単位の判定の割合（%）．p4 = 同じモダリティのラベル無しの 8 手法 − K 本だけ（results/e79_p4/judge/units.csv family=P4，種類の生徒 A・V・T，K=0・5），"
    "p53 = ほかのモダリティを含む疑似ラベル − K 本だけ（results/e79/judge/units.csv family=P5-3，AVT・AV の世界の生徒，K=0・5・25）．sensitivity=main，欠けを除く")
rows = []
for i, (lab, c, n) in enumerate([("p53u", P53uc, P53un), ("p52", P52c, P52n), ("p51", P51c, P51n)]):
    rows.append(dict(y=i + 1, lab=lab, n=n, **{j: 100 * v / n for j, v in c.items()}))
csv("world_uar.csv", pd.DataFrame(rows), "UAR の単位の判定の割合（%）．p53u = 蒸留 − K 本だけ，p52 = 蒸留 − 8 手法の最良（仮のターゲットで選んだ），p51 = 蒸留 − 同じ教師の集合を渡した温度つき蒸留．results/e79/judge/units.csv，sensitivity=main")

# ---------- 結果の後半②：1 コーパスを抜く設定の本番（DAI-1 の評価の回．族 1 = 提案 − K 本だけ，族 3 = 提案 − 同じ情報の相手） ----------
fam = pd.read_csv(ROOT / "results/dai1/stats/families_flagged.csv")
assert fam.commit.str.startswith("ba98974").all() and not fam.dirty.any()
LCORP = ["AFEW", "IEMOCAP", "MELD", "MOSEI"]; LST = ["A", "V", "T", "AVT"]; LK = [0, 5, 25]
for family, tag in ((1, "fone"), (3, "fthree")):
    d = fam[(fam.family == family) & fam.K.isin(LK)]
    for st in LST:
        rows = []
        for ci, T in enumerate(LCORP):
            rec = dict(ci=ci + 1)
            for j, K in enumerate(LK):
                r = d[(d["T"] == T) & (d.student == st) & (d.K == K)].iloc[0]
                rec[f"x{K}"] = ci + 1 + (j - 1) * 0.22
                rec[f"y{K}"] = r.effect; rec[f"ep{K}"] = r.hi - r.effect; rec[f"em{K}"] = r.effect - r.lo
                rec[f"test{K}"] = int(not r.descriptive)
            rows.append(rec)
        csv(f"{tag}_{st}.csv", pd.DataFrame(rows), f"results/dai1/stats/families_flagged.csv family={family} student={st}（提案 − {'K 本だけ' if family == 1 else '同じ情報の相手（多くは温度つき蒸留）'}．x は コーパス ＋ K のずらし）")
    dd = d[d.student.isin(LST)]
    NUM[f"{tag}n"] = str(len(dd)); NUM[f"{tag}inczero"] = str(int(((dd.lo <= 0) & (dd.hi >= 0)).sum()))
    for st in LST:
        e = dd[dd.student == st]
        NUM[f"{tag}pos{st.lower()}"] = str(int((e.effect > 0).sum())); NUM[f"{tag}lo{st.lower()}"] = str(int((e.lo > 0).sum()))
        put(f"{tag}min{st.lower()}", float(e.effect.min()), 1, True); put(f"{tag}max{st.lower()}", float(e.effect.max()), 1, True)
        NUM[f"{tag}locells{st.lower()}"] = "，".join(f"{r['T']}（K = {int(r.K)}）" for _, r in e[e.lo > 0].sort_values(["T", "K"]).iterrows())
    a1 = dd[(dd.student == "A") & (dd["T"] == "MOSEI")]
    NUM[f"{tag}mosein"] = str(int((~a1.descriptive).sum())); put(f"{tag}moseipmax", float(a1.p_holm.max()), 4)
    if family == 1:
        assert a1.p_holm.max() < 0.001                                  #「本番でも，声の分類器は 4 コーパスすべてで上がった」のページの注の「Holm 後 p < 0.001」
    if family == 1:                                                     # 声で検定に残ったほかのコーパス（同じページの注．gate-claim slides-2026-11-10 の 13）
        tA = dd[(dd.student == "A") & (~dd.descriptive.astype(bool)) & (dd["T"] != "MOSEI")]
        assert set(tA["T"]) == {"MELD"} and sorted(tA.K) == [0, 5], tA[["T", "K"]]
        for _, r in tA.iterrows():
            put(f"{tag}meldp_{int(r.K)}", float(r.p_holm), 2)
assert NUM["fonelocellsa"].count("（") == 10
d3 = fam[(fam.family == 3) & fam.K.isin(LK) & fam.student.isin(LST)]
NUM["fthreehineg"] = str(int((d3.hi < 0).sum())); NUM["fthreelopos"] = str(int((d3.lo > 0).sum()))
put("fthreehinegmax", float(d3[d3.hi < 0].effect.abs().max()), 2)
NUM["fthreehinegcells"] = "，".join(f"{r['T']} の {r.student}（K = {int(r.K)}）" for _, r in d3[d3.hi < 0].sort_values(["T", "student", "K"]).iterrows())

# ---------- メタ学習（仮のターゲットでの設定の選択）の結果．results/dai1/stats/desc/counts.csv の ALL の行と select/frozen.json ----------
cnt = pd.read_csv(ROOT / "results/dai1/stats/desc/counts.csv")
assert cnt.commit.str.startswith("ba98974").all() and not cnt.dirty.any()
allr = cnt[cnt["T"] == "ALL"].set_index(["item", "scope"])
NUM["mldefault"] = str(int(allr.loc[("fam4_selected_equals_default", "family_K"), "count"])); NUM["mldefaultof"] = str(int(allr.loc[("fam4_selected_equals_default", "family_K"), "of"]))
NUM["mlkd"] = str(int(allr.loc[("partner_is_kd", "all_K"), "count"])); NUM["mlkdof"] = str(int(allr.loc[("partner_is_kd", "all_K"), "of"]))
fz = json.loads((ROOT / "results/dai1/select/frozen.json").read_text())
mains = [fz[T]["main"][k] for T in ["AFEW", "IEMOCAP", "MELD", "MOSEI"] for k in fz[T]["main"]]
NUM["mlmainone"] = str(sum(m == "one" for m in mains)); NUM["mlmainof"] = str(len(mains))
assert NUM["mldefault"] == "44" and NUM["mldefaultof"] == "48"     # 結果の文の下書き 2026-11-05 の A4 と一致

# ---------- なぜ効くか：下調べ 11（results/dai11_probe/）・12（results/dai12_probe/）．仮のターゲットの回．DAI-3 の余地（評価の回） ----------
for sub in ["dai11_probe", "dai12_probe"]:
    pv = json.loads((ROOT / f"results/{sub}/provenance.json").read_text())
    assert pv["dirty"] is False, sub
P11 = ROOT / "results/dai11_probe"
g11 = pd.read_csv(P11 / "req_vs_gain.csv", comment="#")
assert len(g11) == 280
for st in ["A", "V", "T", "AVT"]:
    csv(f"why_{st}.csv", g11[g11.student == st][["c", "teacher", "K", "delta_uar", "proj_gain", "gain"]],
        f"results/dai11_probe/req_vs_gain.csv student={st}．delta_uar = 出所の候補での UAR − K 本だけの候補での UAR，proj_gain = 出所の対数確率を学び直す分類器の特徴から線形に交差予測した判定の UAR − K 本だけの UAR，gain = 学び直した版 − K 本だけのテストの UAR（30 単位の平均）")
# 要約の④（parts/a0_summary.tex）：説明のための 2 つの例（K = 5，出所 = ほかのモダリティの積 other）
ex = []
for i, (c, st) in enumerate([("IEMOCAP", "V"), ("MOSEI", "A")]):
    r = g11[(g11.c == c) & (g11.student == st) & (g11.teacher == "other") & (g11.K == 5)]
    assert len(r) == 1, (c, st)
    r = r.iloc[0]
    ex.append(dict(x=i + 1, c=c, student=st, delta_uar=r.delta_uar, proj_gain=r.proj_gain, gain=r.gain))
csv("sum_mimic.csv", pd.DataFrame(ex), "results/dai11_probe/req_vs_gain.csv の 2 行（IEMOCAP の V と MOSEI の A，teacher=other，K=5）．"
    "delta_uar = 疑似ラベルの候補での UAR − K 本だけの候補での UAR，proj_gain = 疑似ラベルを学び直す分類器の特徴から線形に交差予測した判定の UAR − K 本だけの UAR，gain = 学び直した版 − K 本だけのテストの UAR（30 単位の平均）")
flag = g11[(g11.delta_uar >= 5) & (g11.gain < 0.5)].assign(frac=lambda d: d.proj_gain / d.delta_uar)
csv("why_flag.csv", flag[["c", "student", "teacher", "K", "delta_uar", "proj_gain", "gain", "frac"]], "req_vs_gain.csv のうち Δ ≥ 5 かつ 得 < 0.5 のセル．frac = 取り込める割合（proj_gain ÷ Δ）")
NUM["wflagn"] = str(len(flag)); NUM["wflagfracbelow"] = str(int((flag.frac < 0.5).sum())); put("wflagfracmax", float(flag.frac.max()), 2)
assert len(flag) == 30 and (flag.frac < 0.5).all()
NUM["wdeltage"] = str(int((g11.delta_uar >= 5).sum())); NUM["wdeltageok"] = str(int(((g11.delta_uar >= 5) & (g11.gain >= 0.5)).sum()))
fv = flag[(flag.c == "IEMOCAP") & (flag.student == "V")]
NUM["wfvn"] = str(len(fv)); put("wfvdmin", float(fv.delta_uar.min())); put("wfvdmax", float(fv.delta_uar.max()))
put("wfvgmin", float(fv.gain.min()), 2, True); put("wfvgmax", float(fv.gain.max()), 2, True)
put("wfvfmin", float(fv.frac.min()), 2, True); put("wfvfmax", float(fv.frac.max()), 2, True)
rc = pd.read_csv(P11 / "rank_corr.csv", comment="#")
rc = rc[rc.kind == "centered_spearman|blk_cm"]
for subset, sk in [("all", "all"), ("student=A", "a"), ("student=V", "v"), ("student=T", "t")]:
    for qn, qk in [("delta_uar", "d"), ("proj_gain", "p")]:
        r = rc[(rc.subset == subset) & (rc.quantity == qn)].iloc[0]
        put(f"rho{qk}{sk}", float(r.rho), 2); put(f"null{qk}{sk}", float(r.null95), 2)
# 顔の IEMOCAP の例（K = 0, 5, 25）：候補の上の UAR（K 本だけ，出所 other）と，テストの UAR（K 本だけ，学び直した版，全ラベルの上限）
c11 = pd.read_csv(P11 / "cells.csv", comment="#")
ci = c11[c11.c == "IEMOCAP"].set_index("key").uar
rows = []
for j, K in enumerate([0, 5, 25]):
    gg = g11[(g11.c == "IEMOCAP") & (g11.student == "V") & (g11.teacher == "other") & (g11.K == K)].iloc[0]
    rows.append(dict(x=j + 1, K=K, pool_s=gg.uar_s, pool_t=gg.uar_t, test_none=ci[f"K{K}_V_none"], test_other=ci[f"K{K}_V_other"], test_oracle=ci["V_oracle"]))
FV = pd.DataFrame(rows)
csv("why_face_iemocap.csv", FV, "results/dai11_probe/req_vs_gain.csv（c=IEMOCAP，student=V，teacher=other の uar_s，uar_t．候補の上）と cells.csv（K*_V_none，K*_V_other，V_oracle．テスト．30 単位の平均）")
put("wfvoracle", float(FV.test_oracle.iloc[0])); put("wfvnonemin", float(FV.test_none.min())); put("wfvnonemax", float(FV.test_none.max()))
# 下調べ 12
P12 = ROOT / "results/dai12_probe"
d12 = pd.read_csv(P12 / "diffs.csv", comment="#", low_memory=False)
r1 = d12[(d12["quantity"] == "nl_minus_lince") & (d12["where"] == "pool") & (d12["subset"] == "all") & d12["teacher"].isin(["enc2", "fuse1", "other", "enc2fuse", "pair"]) & d12["K"].isin([1, 5, 10, 25])]
q3 = d12[(d12["quantity"] == "lab_nl_minus_lince") & (d12["subset"] == "all")]
assert len(r1) == 240 and len(q3) == 60
put("nlmin", float(r1.est.min()), 2, True); put("nlmax", float(r1.est.max()), 2, True); NUM["nlcells"] = str(len(r1))
put("labmin", float(q3.est.min()), 2, True); put("labmax", float(q3.est.max()), 2, True)
cr = pd.read_csv(P12 / "criteria.csv", comment="#", low_memory=False)
r1v = cr[cr["rule"] == "R1"]; NUM["nlgroups"] = str(len(r1v)); NUM["nlnotsay"] = str(int((r1v["verdict"] == "言えない").sum()))
TEACH = {"enc2": 1, "fuse1": 2, "other": 3, "enc2fuse": 4, "pair": 5}
STX = {"A": 0, "V": 1, "T": 2}
r1 = r1.assign(x=lambda d: d["student"].map(STX) * 6 + d["teacher"].map(TEACH))
csv("why_nl.csv", r1[["x", "c", "student", "teacher", "K", "est"]], "results/dai12_probe/diffs.csv quantity=nl_minus_lince（候補の上の UAR．隠れ層つき − 線形），出所 5 つ，K = 1, 5, 10, 25．x = 分類器 × 出所")
q3 = q3.assign(x=lambda d: d["student"].map(STX) * 6 + 3)
csv("why_lab.csv", q3[["x", "c", "student", "K", "est"]], "results/dai12_probe/diffs.csv quantity=lab_nl_minus_lince（候補の全ラベルで学んだ隠れ層つき − 線形）")
q2 = pd.read_csv(P12 / "q2_cells.csv", comment="#")
NUM["qtwoinfo"] = str(int(q2.status.str.startswith("この罰則のかけ方").sum())); NUM["qtwoundet"] = str(int(q2.status.str.startswith("判定できない").sum()))
# 余地を埋めた割合（評価の回．DAI-3．K = 5，声）
ex = pd.read_csv(ROOT / "results/dai3/eval/expl4_corpus_restored.csv")
assert not ex.dirty.any(), "DAI-3 の解析の出力が dirty"
ex = ex[(ex.subset == "all") & (ex.K == 5)]
rows = []
for i, T in enumerate(["AFEW", "IEMOCAP", "MELD", "MOSEI"]):
    gq = ex[ex.corpus == T].set_index("quantity")
    gain, room = float(gq.loc["gain", "A"]), float(gq.loc["room", "A"])
    rows.append(dict(ci=i + 1, T=T, gain=gain, room=room, share=100 * gain / room))
    put(f"shareA_{T}", 100 * gain / room, 0)
csv("why_room.csv", pd.DataFrame(rows), "results/dai3/eval/expl4_corpus_restored.csv subset=all K=5 の声（gain = 提案 − K 本だけ，room = 全ラベルで学んだ重みに切片を当て直した参照 − K 本だけ．評価の回．点の値）")

# ---------- メタ学習（全ソース設定の出発点．E39．検査の仕組みを入れる前の結果．区間なし） ----------
e39 = pd.read_csv(ROOT / "results/explore_relation/e39_tests.csv", comment="#")
E39K = {"audio_wavlm": "a", "audio_hubert": "ah", "video": "v", "fuse-audio-video": "av"}
rows = []
for j, K in enumerate([1, 5, 10, 25]):
    rec = {"x": j + 1, "K": K}
    for kd, kk in E39K.items():
        r = e39[(e39.kind == kd) & (e39.K == K)].iloc[0]
        rec[kk] = r["meta-D1"]
        if K == 5:
            put(f"emd_{kk}", r["meta-D1"], 2, True); put(f"emp_{kk}", r["meta-P4"], 1, True)
            NUM[f"emup{kk}"] = str(int(r["meta>D1"])); NUM[f"emn{kk}"] = str(int(r["targets"]))
    rows.append(rec)
csv("e39_meta.csv", pd.DataFrame(rows), "results/explore_relation/e39_tests.csv の meta-D1（メタ学習で 5 つの値を選んだ出発点 − 等しい比重の出発点．ターゲットの平均．検査前．区間なし）")
sub = e39[e39.kind.isin(E39K)]
assert (sub["meta-D1"] > 0).all()                                     # 記録：D1 を全種類・全 K で上回る
put("emoramin", float(-sub["meta-oracle"].max()), 1); put("emoramax", float(-sub["meta-oracle"].min()), 1)
put("emdmin", float(sub[sub.K == 5]["meta-D1"].min()), 2, True); put("emdmax", float(sub[sub.K == 5]["meta-D1"].max()), 2, True)

# ---------- 補足の表（台本が表の行を書く．手で数を書かない） ----------
def mark(lo, hi):
    return "$^{\\uparrow}$" if lo > 0 else ("$^{\\downarrow}$" if hi < 0 else "")
def fmt(v, nd=1, sign=True):
    if round(v, nd) == 0:
        v = 0.0                                   # −0.0 を書かない
    t = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    if round(v, nd) == 0:
        t = f"{0:.{nd}f}"
    return t.replace("-", "\\textminus{}")
# S4：組の設定の利得（e76a_1_kind.csv）
lines = []
for k in KINDS:
    d = kind[kind.short == k].set_index("K")
    cells = [fmt(d.loc[KK].mean_gain) + mark(d.loc[KK].lo, d.loc[KK].hi) for KK in KLAB]
    lines.append(f"{k} & {int(d.loc['1'].n_pairs)} & " + " & ".join(cells) + " \\\\")
(FIG / "tab_gain.tex").write_text("% e76a_1_kind.csv（合併 − ターゲットのみ．種類の組の平均．↑ 区間 > 0，↓ 区間 < 0）\n" + "\n".join(lines) + "\n")
# S5：1 コーパスを抜く設定の族 1（提案 − K 本だけ）
lines = []
d1 = fam[(fam.family == 1) & fam.K.isin(LK)]
for st in LST:
    cells = []
    for T in LCORP:
        for K in LK:
            r = d1[(d1["T"] == T) & (d1.student == st) & (d1.K == K)].iloc[0]
            cells.append(fmt(r.effect) + mark(r.lo, r.hi) + ("$^{*}$" if not r.descriptive else ""))
    lines.append(f"{st} & " + " & ".join(cells) + " \\\\")
(FIG / "tab_fam1.tex").write_text("% families_flagged.csv family=1（↑ 区間 > 0，↓ 区間 < 0，* 検定に残ったセル）\n" + "\n".join(lines) + "\n")
# S6：標準の手法との比べ（融合の分類器．results/dai2/stats/prop_minus_method.csv．K = 0, 5, 25）
pm = pd.read_csv(ROOT / "results/dai2/stats/prop_minus_method.csv", comment="#")
assert pm.commit.astype(str).str.startswith("ba98974").all() and not pm.dirty.any()
pm = pm[(pm.student == "AVT") & pm.K.isin([0, 5, 25])]
MNAME = [("entmin", "エントロピー最小化"), ("im", "IM"), ("mme", "MME"), ("t3a", "T3A"), ("coral", "CORAL"), ("dann", "DANN"), ("self", "自己学習"),
         ("mapctl", "写像だけの対照"), ("own", "自分の予測を提案の形で"), ("st", "S＋T（ソースと K 本を混ぜる）"), ("lp", "ラベル伝播（K = 5, 25）"),
         ("t3a_b", "T3A（補助の版）"), ("saerens", "割合の補正（Saerens）")]   # xMUDA 型は融合の分類器に無い
lines = []; got = {}
for m, nm in MNAME:
    e = pm[pm.method == m]
    up, dn = int((e.lo > 0).sum()), int((e.hi < 0).sum()); got[m] = up
    lines.append(f"{nm} & {len(e)} & {up} & {dn} \\\\")
assert (got["entmin"], got["im"], got["mme"], got["t3a"], got["coral"], got["dann"], got["self"], got["mapctl"], got["own"], got["st"]) == (9, 8, 8, 8, 5, 4, 4, 4, 4, 3), got
(FIG / "tab_dai2.tex").write_text("% results/dai2/stats/prop_minus_method.csv student=AVT, K=0,5,25（提案 − 手法．区間の下端 > 0 と上端 < 0 のセルの数）\n" + "\n".join(lines) + "\n")

# ---------- 結果の後半：ほかの手法との比べ（LOCO 設定の評価の回．提案 − 手法．K = 0, 5, 25） ----------
# 実験 2 の手法（prop_minus_method.csv）．温度つき蒸留（族 3）は 2026-11-10 の研究者の指示で資料から外した．
# 照合済みの文：融合の 12 条件は DAI/docs/結果の文の下書き_2026-11-05.md の B6．声・顔・言葉の手法の行はこの資料で数えた
pmx = pd.read_csv(ROOT / "results/dai2/stats/prop_minus_method.csv", comment="#")
assert pmx.commit.astype(str).str.startswith("ba98974").all() and not pmx.dirty.any()
pmx = pmx[pmx.K.isin([0, 5, 25])]
CMPROWS = [  # (群, 鍵, 表示名)
    ("same", "entmin", "エントロピー最小化"), ("same", "im", "IM"), ("same", "self", "自己学習"), ("same", "lp", "ラベル伝播（K = 5, 25）"),
    ("same", "t3a", "T3A"), ("same", "own", "自分の予測を提案の形で（対照）"),
    ("src", "st", "S＋T（ソースと K 本を混ぜる）"), ("src", "coral", "CORAL"), ("src", "dann", "DANN"), ("src", "mme", "MME"),
    ("src", "mapctl", "DANN・MME の DA の項を 0 に（対照）"),
    ("other", "xmuda", "xMUDA 型"),
    ("app", "saerens", "割合の補正（Saerens．付録）"),
]
STU = ["A", "V", "T", "AVT"]
lines = []; tally = {}
for r, (g, key, name) in enumerate(CMPROWS):
    lines.append(f"\\cmplab{{{r}}}{{{name}}}")
    for c, st in enumerate(STU):
        e = pmx[(pmx.method == key) & (pmx.student == st)]
        if len(e) == 0:
            lines.append(f"\\cmpna{{{c}}}{{{r}}}"); continue
        up, dn, n = int((e.lo > 0).sum()), int((e.hi < 0).sum()), len(e)
        tally[(key, st)] = (up, dn, n)
        lines.append(f"\\cmpcell{{{c}}}{{{r}}}{{{up}}}{{{dn}}}{{{n}}}")
groups = {}
for r, (g, _, _) in enumerate(CMPROWS):
    groups.setdefault(g, []).append(r)
GNAME = {"same": "同じモダリティのラベル無しの発話を使う", "src": "ソースのデータも使う", "other": "ほかのモダリティの予測を使う", "app": "付録"}
for g, rs in groups.items():
    lines.append(f"\\cmpgroup{{{min(rs)}}}{{{max(rs)}}}{{{GNAME[g]}}}")
(FIG / "cmp_loco.tex").write_text("% results/dai2/stats/prop_minus_method.csv，K = 0, 5, 25．"
                                  "\\cmpcell{列}{行}{提案が上（区間の下端 > 0）}{手法が上（上端 < 0）}{条件の数}\n" + "\n".join(lines) + "\n")
# 照合済みの数と合うか（B6：融合の 12 条件，B5：族 3 の 48 条件で提案が上 1，相手が上 7）
assert tally[("entmin", "AVT")][:2] == (9, 1) and tally[("coral", "AVT")][:2] == (5, 1) and tally[("lp", "AVT")] == (8, 0, 8)
# 声の分類器が下回った（手法が上）条件はどの手法にも無い（タイトルの文）
assert all(tally[(k, "A")][1] == 0 for (_, k, _) in CMPROWS)
NUM["cmpvhalf"] = str(sum(1 for (_, k, _) in CMPROWS if ("" if (k, "V") not in tally else 1) and tally[(k, "V")][0] * 2 > tally[(k, "V")][2]))
NUM["cmpnmethods"] = str(len(CMPROWS))

# ---------- 「向きのずれは，ラベルを足しても少しずつしか直らない」のページ：理論から導いた 1 母数の式の当てはまり（e76a_3_q_targets.csv．声・顔・言葉） ----------
qt = read76("e76a_3_q_targets.csv")
qt5 = qt[(qt["rule"] == "main") & (qt.D_version == "D") & (qt.fit == "5pt") & qt.short.isin(["A", "V", "T"])]
NUM["qfitn"] = str(len(qt5)); NUM["qfitexc"] = str(int(qt5.resid_exceeds.astype(str).eq("True").sum()))
assert NUM["qfitexc"] == "0"
for k in ["A", "V", "T"]:
    x = qt5[qt5.short == k]
    NUM[f"qloo{KEY[k]}"] = str(int(x.M1orM2_better_loko.astype(str).eq("True").sum())); NUM[f"qn{KEY[k]}"] = str(len(x))

# ---------- 全ソース設定：向きの損の減りの値をモダリティごとに並べる（43・44 ページを数値の図にする．2026-11-11 追加．gate-claim 第 5 回で相手の族を差し替えた） ----------
# 量：K 本だけの分類器の向きの損 − 手法の向きの損（6 対の平均．点の値．正は減った）．どれも results/e79/judge/units.csv，sensitivity=main，missing を除く
# 提案（全ソース設定の版 oursx）：P5-3（81 条件）．8 手法：提案と同じ AV・AVT の組・同じ出発点・同じ K 本だけ・同じテストで測った族
#   単独の分類器は P4_uni_world（8 手法），融合の分類器は P4′（6 手法．DANN・MME は融合の分類器で回していない）
# 比べる範囲：提案と同じ（組，分類器，K，コーパス）のうち K = 0・5（P4′ に K = 25 が無いので）．提案の K = 25 は注に書く
pv79 = json.loads((ROOT / "results/e79/judge/provenance.json").read_text())
assert pv79["commit"].startswith("ec0aeb6") and pv79.get("dirty") is False
pv79p4 = json.loads((ROOT / "results/e79_p4/judge/provenance.json").read_text())
assert pv79p4["commit"].startswith("c4d05b6") and pv79p4.get("dirty") is False
u79a = pd.read_csv(ROOT / "results/e79/judge/units.csv", comment="#", low_memory=False)
def dir_units(fam_):
    return u79a[(u79a.family == fam_) & (u79a.metric == "dir") & (u79a.sensitivity == "main") & (u79a.judgment != "missing")].copy()
q5 = dir_units("P5-3"); qu = dir_units("P4_uni_world"); qf = dir_units("P4'")
assert len(q5) == 81 and q5.comparison.eq("none-oursx").all()
assert qu.comparison.str.startswith("none-").all() and qf.comparison.str.startswith("none-").all()
assert qu.comparison.nunique() == 8 and qf.comparison.nunique() == 6
KEYC = ["run", "student", "role", "K", "corpus"]
q8 = pd.concat([qu, qf])
assert not q8.duplicated(KEYC + ["comparison"]).any() and not q5.duplicated(KEYC).any()
cells = q5[q5.K.isin([0, 5])][KEYC]
q8m = q8.merge(cells, on=KEYC)                                           # 提案と同じ条件だけ
q5m = q5[q5.K.isin([0, 5])]
assert set(map(tuple, q8m[KEYC].drop_duplicates().values)) <= set(map(tuple, q5m[KEYC].values))
SX = {"audio_wavlm": (1, "A"), "video": (2, "V"), "text_roberta": (3, "T"), "fuse-audio-video": (4, "AV"), "fuse-audio-video-text": (5, "AVT")}
rng = np.random.default_rng(0)                                           # 点の横のずらし（見た目だけ）
rows = []
for who, q, off in [("m8", q8m, -0.2), ("prop", q5m, 0.2)]:
    for _, r in q.iterrows():
        x, sh = SX[r.student]
        rows.append(dict(who=who, kind=sh, x=x + off + rng.uniform(-0.09, 0.09), y=r.point, lo=r.lo, hi=r.hi, K=int(r.K), corpus=r.corpus,
                         run=r.run, method=r.comparison.split("-", 1)[1], family=r.family, judgment=r.judgment))
DV = pd.DataFrame(rows)
for who in ["m8", "prop"]:
    csv(f"dirval_{who}.csv", DV[DV.who == who], "全ソース設定の向きの損の減り（K 本だけ − 手法．6 対の平均．点の値）．results/e79/judge/units.csv（コミット ec0aeb6）"
        + ("の P4_uni_world（単独の分類器，8 手法）と P4′（融合の分類器，6 手法）" if who == "m8" else "の P5-3（提案 oursx）")
        + "．sensitivity=main，missing を除く．提案と同じ（組，分類器，K，コーパス）の K = 0・5．x は分類器の位置 ＋ 横のずらし（見た目だけ）")
med = []
for st, (xi, sh) in SX.items():
    a = DV[(DV.who == "m8") & (DV.kind == sh)]; b = DV[(DV.who == "prop") & (DV.kind == sh)]
    med.append(dict(kind=sh, x=xi, n_m8=len(a), n_prop=len(b), med_m8=a.y.median(), med_prop=b.y.median(),
                    neg_m8=int((a.y < 0).sum()), pos_prop=int((b.y > 0).sum())))
    NUM[f"dvnprop{sh}"] = str(len(b)); NUM[f"dvposprop{sh}"] = str(int((b.y > 0).sum())); put(f"dvmedprop{sh}", b.y.median(), 1, True)
    NUM[f"dvnm{sh}"] = str(len(a)); NUM[f"dvnegm{sh}"] = str(int((a.y < 0).sum())); put(f"dvmedm{sh}", a.y.median(), 1, True)
    assert a.y.median() < 0 < b.y.median(), sh                          # 見出し「中央値は 8 手法で負，提案で正」
MD = pd.DataFrame(med); MD["xm8"] = MD.x - 0.2; MD["xprop"] = MD.x + 0.2
csv("dirval_med.csv", MD, "dirval_m8.csv・dirval_prop.csv の分類器ごとの中央値（med_m8，med_prop）と数．コミット ec0aeb6")
NUM["dvnmall"] = str(len(q8m)); NUM["dvnpropall"] = str(len(q5m))
NUM["dvnegmall"] = str(int((q8m.point < 0).sum())); NUM["dvpospropall"] = str(int((q5m.point > 0).sum()))
# 同じ条件の対：提案の値 > 手法の値
pr = q8m.merge(q5m[KEYC + ["point"]].rename(columns={"point": "pp"}), on=KEYC)
NUM["dvpairn"] = str(len(pr)); NUM["dvpairwin"] = str(int((pr.pp > pr.point).sum()))
pa = pr[pr.student == "audio_wavlm"]; NUM["dvpairnA"] = str(len(pa)); NUM["dvpairwinA"] = str(int((pa.pp > pa.point).sum()))
q5k = q5[q5.K == 25]; NUM["dvnproptwentyfive"] = str(len(q5k)); NUM["dvposproptwentyfive"] = str(int((q5k.point > 0).sum()))
assert (q5k.point > 0).all()                                             # 注の「27 条件すべてで正」
put("dvmedproptwentyfive", q5k.point.median(), 1, True)
_mm = q8m.groupby(q8m.comparison.str.split("-", n=1).str[1]).point.median()
NUM["dvmethpos"] = str(int((_mm > 0).sum())); put("dvmethmin", _mm.min(), 2, True); put("dvmethmax", _mm.max(), 2, True)
assert (_mm < 0).all()
# 仮のターゲットで選ぶ候補の数（results/e79/select/chosen.csv．提案は oursx・incl，8 手法は none を除いて選ぶ excl）
ch = pd.read_csv(ROOT / "results/e79/select/chosen.csv", comment="#")
co = ch[(ch.selector == "oursx") & (ch.variant == "incl")]
cm = ch[ch.selector.isin(["entmin", "self", "im", "t3a", "lp", "coral", "dann", "mme"]) & (ch.variant == "excl") & (ch.n_cand > 0)]
NUM["dvcandpmin"] = str(int(co.n_cand.min())); NUM["dvcandpmax"] = str(int(co.n_cand.max())); NUM["dvcandpmed"] = str(int(co.n_cand.median()))
NUM["dvcandmmin"] = str(int(cm.n_cand.min())); NUM["dvcandmmax"] = str(int(cm.n_cand.max()))
assert not co.chose_none.astype(bool).any()
# 全ソース設定の提案（oursx）で選ばれた形の内訳（102 の選択．param ＝ 疑似ラベルの出所:重みの付け方）
src = co.param.str.split(":").str[0]; wt = co.param.str.split(":").str[1].str.replace(r"hard.*", "hard", regex=True)
NUM["oxn"] = str(len(co)); NUM["oxprod"] = str(int(src.isin(["ref", "ref0"]).sum()))
NUM["oxpfk"] = str(int((src == "pFK").sum())); NUM["oxagree"] = str(int((src == "agree").sum()))
NUM["oxbal"] = str(int((wt == "bal").sum())); NUM["oxhard"] = str(int((wt == "hard").sum())); NUM["oxraw"] = str(int((wt == "raw").sum()))
assert int(NUM["oxprod"]) + int(NUM["oxpfk"]) + int(NUM["oxagree"]) == len(co) and int(NUM["oxbal"]) + int(NUM["oxhard"]) + int(NUM["oxraw"]) == len(co)
NUM["dvpropav"] = str(int((q5m.run == "fuse-audio-video").sum()))

# ---------- 後半を前半の物差しで読む：提案の UAR の上がり幅を，向きの分の改善と切片の分の改善に分ける（2026-11-10 追加） ----------
# UAR の上がり幅（提案 − K 本だけ）＝ 向きの分の改善（重みはそのままで，切片だけをテストの発話のラベルで解き直した後の UAR の差．
#                                     切片は感情の釣り合いの重みをつけた交差エントロピーで解く）
#                                   ＋ 切片の分の改善（K 本だけの「解き直しで取り戻せる分」− 提案のその分）
# 全ソース設定：results/e79/judge/units.csv の P5-3u（UAR），P5-3_desc_bfix（当て直した UAR），P5-3_desc_ofix（取り戻せる分）．どれも 提案 − K 本だけの点の値
u79 = pd.read_csv(ROOT / "results/e79/judge/units.csv", comment="#", low_memory=False)
K79 = ["run", "student", "role", "K", "corpus"]
def g79(fam):
    x = u79[(u79.family == fam) & (u79.sensitivity == "main")]
    assert not x.duplicated(K79).any() and x.comparison.eq("oursx-none").all(), fam
    return x.set_index(K79).point
m79 = pd.concat([g79("P5-3u").rename("uar"), g79("P5-3_desc_bfix").rename("bfix"), g79("P5-3_desc_ofix").rename("ofix")], axis=1)
assert len(m79) == 81 and m79.notna().all().all()
assert (m79.uar - (m79.bfix - m79.ofix)).abs().max() < 1e-9          # 足し算が成り立つ
m79 = m79.reset_index()
WNAME = [("audio_wavlm", "A"), ("video", "V"), ("text_roberta", "T"), ("fuse-audio-video", "AV"), ("fuse-audio-video-text", "AVT")]
rows = []
for i, (st, sh) in enumerate(WNAME):
    x = m79[m79.student == st]
    tot, dirp, posp = x.uar.mean(), x.bfix.mean(), -x.ofix.mean()
    rows.append(dict(ci=i + 1, kind=sh, n=len(x), total=tot, dir=dirp, pos=posp, share=100 * dirp / tot))
    put(f"wdtot_{sh}", tot, 2, True); put(f"wddir_{sh}", dirp, 2, True); put(f"wdpos_{sh}", posp, 2, True)
    put(f"wdshare_{sh}", 100 * dirp / tot, 0); NUM["wdn" + sh] = str(len(x))
WD = pd.DataFrame(rows)
csv("split_world.csv", WD, "全ソース設定．results/e79/judge/units.csv（sensitivity=main）の P5-3u・P5-3_desc_bfix・P5-3_desc_ofix（提案 − K 本だけ．点の値）の分類器ごとの平均．"
    "total = UAR の上がり幅，dir = 切片を当て直した UAR の差（向きの分の改善），pos = −(取り戻せる分の差)（切片の分の改善）．total = dir + pos．share = 100 dir/total")
put("wdtotall", m79.uar.mean(), 2, True); put("wddirall", m79.bfix.mean(), 2, True); put("wdposall", -m79.ofix.mean(), 2, True)
put("wdshareall", 100 * m79.bfix.mean() / m79.uar.mean(), 0)
NUM["wddirposn"] = str(int((m79.bfix > 0).sum())); NUM["wdn"] = str(len(m79))
put("wdsharemin", WD.share.min(), 0); put("wdsharemax", WD.share.max(), 0)
# LOCO 設定（評価の回）：results/dai3/eval/decomp.csv（version four．全ラベルで学んだ重みに切片を当て直した分類器との UAR の差を，
# 当て直しで減る分 position と残る分 direction に分けたもの）．差の区間：向きの分は expl1.csv の (iv)，UAR の上がり幅は families_flagged.csv の族 1
dd = pd.read_csv(ROOT / "results/dai3/eval/decomp.csv", comment="#")
assert dd.commit.str.startswith("ba98974").all() and not dd.dirty.any()
dd = dd[dd.version == "four"]
def dval(T, item, q):
    x = dd[(dd["T"] == T) & (dd.item == item) & (dd.quantity == q)]
    assert len(x) == 1, (T, item, q)
    return x.iloc[0]
e1 = pd.read_csv(ROOT / "results/dai3/eval/expl1.csv", comment="#")
assert e1.commit.str.startswith("ba98974").all() and not e1.dirty.any()
f1 = fam[fam.family == 1]
rows = []; supp = []
for st in ["A", "V", "T", "AVT"]:
    for K in [5, 25]:
        for i, T in enumerate(CORPORA):
            nd_, pd_ = dval(T, f"{st}|none|{K}", "direction"), dval(T, f"{st}|prop|{K}", "direction")
            np_, pp_ = dval(T, f"{st}|none|{K}", "position"), dval(T, f"{st}|prop|{K}", "position")
            ns_, ps_ = dval(T, f"{st}|none|{K}", "sum"), dval(T, f"{st}|prop|{K}", "sum")
            for a, b, c in [(nd_, np_, ns_), (pd_, pp_, ps_)]:
                assert abs(a.effect + b.effect - c.effect) < 1e-9                 # 向きの分 ＋ 切片の分 ＝ 全ラベルの分類器との差
            iv = e1[(e1["T"] == T) & (e1.student == st) & (e1.K == K)].iloc[0]
            assert abs(iv.iv_effect - (nd_.effect - pd_.effect)) < 1e-6          # (iv) は向きの分の改善そのもの
            tt = f1[(f1["T"] == T) & (f1.student == st) & (f1.K == K)].iloc[0]
            assert abs(tt.effect - (ns_.effect - ps_.effect)) < 0.05, (T, st, K, tt.effect, ns_.effect - ps_.effect)   # 族 1 の声のページの値と同じ量
            rec = dict(ci=i + 1, T=T, st=st, K=K, total=tt.effect, total_lo=tt.lo, total_hi=tt.hi, total_decomp=ns_.effect - ps_.effect,
                       dir=iv.iv_effect, dir_lo=iv.iv_lo, dir_hi=iv.iv_hi, pos=np_.effect - pp_.effect,
                       dir_none=nd_.effect, dir_prop=pd_.effect, pos_none=np_.effect, pos_prop=pp_.effect, check2=iv.check2)
            supp.append(rec)
            if st == "A" and K == 5:
                rows.append(dict(rec, tep=tt.hi - tt.effect, tem=tt.effect - tt.lo,
                                 dep=iv.iv_hi - iv.iv_effect, dem=iv.iv_effect - iv.iv_lo))
LS = pd.DataFrame(rows)
csv("split_loco_a.csv", LS, "LOCO 設定の評価の回，声の分類器，K = 5．total = UAR の上がり幅（families_flagged.csv 族 1 の effect と区間．「本番でも，声の分類器は 4 コーパスすべてで上がった」のページと同じ値．decomp.csv の sum の差 total_decomp とは 0.05 以内），"
    "dir = 向きの分の改善（decomp.csv direction の差 = expl1.csv の iv．区間も iv），pos = 切片の分の改善（decomp.csv position の差．区間なし）．total_decomp = dir + pos")
for _, r in LS.iterrows():
    put(f"lsdir_{r['T']}", r.dir, 1, True); put(f"lspos_{r['T']}", r.pos, 1, True); put(f"lstot_{r['T']}", r.total, 1, True)
    put(f"lsdirnone_{r['T']}", r.dir_none, 1); put(f"lsdirprop_{r['T']}", r.dir_prop, 1)
SP = pd.DataFrame(supp)
sa = SP[SP.st == "A"]
NUM["lsapos"] = str(int((sa.dir_lo > 0).sum())); NUM["lsan"] = str(len(sa))
put("lsadirmin", sa[sa.dir_lo > 0].dir.min(), 1, True); put("lsadirmax", sa[sa.dir_lo > 0].dir.max(), 1, True)
put("lsposmin", LS.pos.min(), 1, True); put("lsposmax", LS.pos.max(), 1, True)
put("lsdirminfive", LS.dir.min(), 1, True); put("lsdirmaxfive", LS.dir.max(), 1, True)
LS["share"] = 100 * LS.dir / LS.total                                   # 族 1 の声のページの上がり幅に占める割合（IEMOCAP は 100% を超える）
for _, r in LS.iterrows():
    put(f"lsshare_{r['T']}", r.share, 0)
put("lssharemin", LS.share.min(), 0)
put("wdposshare", 100 * (-m79.ofix.mean()) / m79.uar.mean(), 0)
assert sa.check2.eq("合格").all()                                             # 声は 4 コーパスとも確かめ 2 に合格（「向き」と呼べる）
# 補足の表：4 分類器 × 4 コーパス × K = 5, 25（向きの分の改善 [区間の印]，切片の分の改善，確かめ 2 の不合格に †）
lines = []
for st in ["A", "V", "T", "AVT"]:
    for K in [5, 25]:
        cells = []
        for T in CORPORA:
            r = SP[(SP.st == st) & (SP.K == K) & (SP["T"] == T)].iloc[0]
            dag = "$^{\\dagger}$" if r.check2 != "合格" else ""
            cells.append(fmt(r.dir) + mark(r.dir_lo, r.dir_hi) + dag + " & " + fmt(r.pos))
        lines.append(f"{st} & {K} & " + " & ".join(cells) + " \\\\")
(FIG / "tab_split_loco.tex").write_text("% results/dai3/eval/decomp.csv（version four）と expl1.csv の iv．各コーパスの左 = 向きの分の改善（↑ 区間 > 0，↓ 区間 < 0），"
                                        "右 = 切片の分の改善（区間なし）．† 確かめ 2 に不合格（「向き」と呼ばない）\n" + "\n".join(lines) + "\n")
NUM["lsfail"] = str(int(SP[["st", "T", "check2"]].drop_duplicates().check2.ne("合格").sum()))
so = SP[(SP.st != "A") & (SP.check2 == "合格")]                               # 顔・言葉・融合で確かめ 2 に合格した条件
NUM["lsothern"] = str(len(so)); NUM["lsotherpos"] = str(int((so.dir_lo > 0).sum())); NUM["lsotherneg"] = str(int((so.dir_hi < 0).sum()))
# 計算の前に決めた説明 1 の判定（verdicts.json）と，K 本で切片だけを直す版が基準を下回った条件の数（expl1.csv の ii）
vj = json.loads((ROOT / "results/dai3/eval/verdicts.json").read_text())
assert vj["expl1"]["verdict"] == "判定できない"
NUM["eonebelow"] = str(int((e1.ii_verdict == "下回る").sum())); NUM["eonen"] = str(len(e1))
put("eonemed", float(e1.ii_effect.median()), 1, True)                     # 32 条件の中央値（結果の文 C1 の「中央値 −1.6」はこの値）

# ---------- なぜ上がるか：中間層（下調べ 13．results/dai13_probe/summary_seeds2/．仮のターゲットの回．声，出所 pair，K = 5） ----------
P13 = ROOT / "results/dai13_probe/summary_seeds2"
pv13 = json.loads((P13 / "provenance.json").read_text())
assert pv13["commit"].startswith("494eaf8") and pv13["dirty"] is False
d13 = pd.read_csv(P13 / "diffs.csv", comment="#", low_memory=False)
d13 = d13[(d13.teacher == "pair") & (d13.subset == "all") & (d13.student == "A")]
assert not d13.duplicated(["c", "K", "quantity"]).any()
def v13(c, K, q):
    x = d13[(d13.c == c) & (d13.K == K) & (d13.quantity == q)]
    assert len(x) == 1, (c, K, q)
    return x.iloc[0]
rows = []
for i, c in enumerate(CORPORA):
    g, l, un = v13(c, 5, "info_gap"), v13(c, 5, "layers_minus_final"), v13(c, 5, "unif_minus_final")
    rows.append(dict(ci=i + 1, c=c, gap=g.est, gap_lo=g.lo, gap_hi=g.hi, lw=l.est, lw_lo=l.lo, lw_hi=l.hi, unif=un.est, unif_lo=un.lo, unif_hi=un.hi,
                     lwp=l.hi - l.est, lwm=l.est - l.lo, unp=un.hi - un.est, unm=un.est - un.lo, share=100 * l.est / g.est))
    put(f"mlgap_{c}", g.est, 1); put(f"mllw_{c}", l.est, 2, True); put(f"mlunif_{c}", un.est, 2, True); put(f"mlshare_{c}", 100 * l.est / g.est, 0)
    put(f"mllwlo_{c}", l.lo, 2, True); put(f"mllwhi_{c}", l.hi, 2, True)
ML = pd.DataFrame(rows)
csv("why_layers.csv", ML, "results/dai13_probe/summary_seeds2/diffs.csv（声，出所 pair，subset=all，K = 5．候補の発話の上の UAR．コミット 494eaf8）．"
    "gap = info_gap（疑似ラベルの UAR − 最終層から線形に真似した UAR），lw = layers_minus_final（25 層の重みつき和 − 最終層），unif = unif_minus_final（25 層の一様平均 − 最終層）")
for c in CORPORA:                                                        # 照合済みの文 3 節：K ∈ {1, 5, 10, 25} の平均どうしの比
    ga = np.mean([v13(c, K, "info_gap").est for K in [1, 5, 10, 25]]); la = np.mean([v13(c, K, "layers_minus_final").est for K in [1, 5, 10, 25]])
    put(f"mlshareavg_{c}", 100 * la / ga, 0)
    lu = v13(c, 5, "lw_minus_unif")
    put(f"mllu_{c}", lu.est, 2, True); put(f"mllulo_{c}", lu.lo, 2, True); put(f"mlluhi_{c}", lu.hi, 2, True)
gI = v13("IEMOCAP", 5, "info_gap"); put("mlgaploIEMOCAP", gI.lo, 1, True); put("mlgaphiIEMOCAP", gI.hi, 1, True)
# 対照 self（自分自身の予測を目標にした版）の入れ子の前の量：AFEW は手順の雑音が正の側（照合済みの文 1.5 節）．K = 5 の値で割り引いた比も出す
d13s = pd.read_csv(P13 / "diffs.csv", comment="#", low_memory=False)
d13s = d13s[(d13s.teacher == "self") & (d13s.subset == "all") & (d13s.student == "A") & (d13s.quantity == "layers_minus_final_pre")]
sA = d13s[d13s.c == "AFEW"].set_index("K").est
put("mlselfAFEW", float(sA.loc[5]), 2, True); put("mlselfminAFEW", float(sA.min()), 2, True); put("mlselfmaxAFEW", float(sA.max()), 2, True)
lA, gA = v13("AFEW", 5, "layers_minus_final").est, v13("AFEW", 5, "info_gap").est
put("mlsharediscAFEW", 100 * (lA - sA.loc[5]) / gA, 0); put("mlsharediscmaxAFEW", 100 * (lA - sA.max()) / gA, 0)
three = ML[ML.c != "IEMOCAP"]
assert (three.lw_lo > 0).all() and not (ML[ML.c == "IEMOCAP"].lw_lo > 0).any()
put("mlsharemin", three.share.min(), 0); put("mlsharemax", three.share.max(), 0)

# ---------- 書き出し ----------
with open(FIG / "numbers.tex", "w") as f:
    f.write("% scripts/export_figdata.py が書いた．手で直さない．\n")
    for k, v in sorted(NUM.items()):
        f.write(f"\\expandafter\\gdef\\csname NUM@{k}\\endcsname{{{v}}}\n")
print(len(NUM), "numbers;", len(list(FIG.glob('*.csv'))), "csv files")
