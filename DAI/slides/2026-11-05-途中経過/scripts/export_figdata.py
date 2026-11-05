"""スライドの図と本文の数を，結果ファイルから書き出す（2026-11-05 の途中経過の資料）．
出力：figures/*.csv（pgfplots が読む）と figures/numbers.tex（本文の数．\N{key} で引く）．
出どころ：評価の回（コミット ba98974，4 コーパス × 50 単位）の集計 results/dai1/stats/families.csv，results/dai2/stats/prop_minus_method.csv，
results/dai3/eval/{decomp.csv, expl4_corpus_restored.csv}，評価の回の後の探索 results/explore_consensus/*.csv，results/explore_fusion/*.csv，
今日の AFEW の途中の値 results/explore_fusion/peek_select_units_AFEW.csv（本番の台本 explore_fusion_select.py の出力がそろったら select_cells.csv に替える）．
使い方：cd DAI/slides/2026-11-05-途中経過 && ../../../.venv/bin/python scripts/export_figdata.py
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
STUDENTS = ["A", "V", "T", "AVT"]
KS = [0, 5, 25]
NUM = {}


def put(key, v, nd=1, sign=False):
    key = re.sub(r"[^A-Za-z]", "", key.replace("0", "zero").replace("5", "five").replace("25", "twentyfive")
                 .replace("1", "one").replace("2", "two").replace("3", "three").replace("4", "four"))
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        s = "--"
    else:
        s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
        s = s.replace("-", "\\textminus{}")
    NUM[key] = s


def csv(name, df, src):
    with open(FIG / name, "w") as f:
        f.write(f"# {src}\n")
        df.to_csv(f, index=False)


# ---------- 実験 1：族 1（提案 − K 本だけ）と族 3（提案 − 同じ情報の相手） ----------
fam = pd.read_csv(ROOT / "results/dai1/stats/families.csv")
assert (fam.commit.str.startswith("ba98974")).all() and not fam.dirty.any()
sidx = {s: i + 1 for i, s in enumerate(STUDENTS)}
for family, tag in ((1, "fam1"), (3, "fam3")):
    d = fam[(fam.family == family) & fam.K.isin(KS)].copy()
    d["xi"] = d.student.map(sidx)
    for T in CORPORA:
        for K in KS:
            e = d[(d["T"] == T) & (d.K == K)][["xi", "student", "effect", "lo", "hi", "p_holm", "n_units"]].sort_values("xi")
            csv(f"{tag}_{T}_K{K}.csv", e, f"results/dai1/stats/families.csv family={family} T={T} K={K}（提案 − {'K 本だけ' if family == 1 else '族 3 の相手'}．UAR ポイント，95% 区間）")
            for _, r in e.iterrows():
                put(f"{tag}_{T}_{r.student}_{K}", r.effect, 1, True)
                put(f"{tag}lo_{T}_{r.student}_{K}", r.lo, 1, True)
                put(f"{tag}hi_{T}_{r.student}_{K}", r.hi, 1, True)
                put(f"{tag}p_{T}_{r.student}_{K}", r.p_holm, 3)
                put(f"{tag}2d_{T}_{r.student}_{K}", r.effect, 2, True)
                put(f"{tag}lo2d_{T}_{r.student}_{K}", r.lo, 2, True)
                put(f"{tag}hi2d_{T}_{r.student}_{K}", r.hi, 2, True)

# ---------- 実験 1：絶対の UAR（K 本だけ，提案の主の版）．results/dai1/stats/desc/levels.csv ----------
lv = pd.read_csv(ROOT / "results/dai1/stats/desc/levels.csv")
assert (lv.commit.str.startswith("ba98974")).all() and not lv.dirty.any()
for s_ in STUDENTS:
    rows = []
    for i, T in enumerate(CORPORA):
        rec = dict(ci=i + 1, T=T)
        for K in (5, 25):
            q = lv[(lv["T"] == T) & (lv.student == s_) & (lv.K == K)].set_index("row")
            rec[f"none{K}"] = q.loc["none", "effect"]; rec[f"prop{K}"] = q.loc["joint|one|main", "effect"]
            put(f"lvlnone_{T}_{s_}_{K}", rec[f"none{K}"], 1); put(f"lvlprop_{T}_{s_}_{K}", rec[f"prop{K}"], 1)
        rows.append(rec)
    csv(f"levels_{s_}.csv", pd.DataFrame(rows), f"results/dai1/stats/desc/levels.csv student={s_}（テスト UAR の 50 単位の平均：none = K 本だけ，joint|one|main = 提案の主の版）")

# ---------- 実験 2：提案 − 標準の DA 手法（AVT の生徒） ----------
pm = pd.read_csv(ROOT / "results/dai2/stats/prop_minus_method.csv")
assert (pm.commit.str.startswith("ba98974")).all() and not pm.dirty.any()
METHODS = ["self", "entmin", "im", "t3a", "lp", "own", "st", "coral", "dann", "mme"]
midx = {m: i + 1 for i, m in enumerate(METHODS)}
d2 = pm[(pm.student == "AVT") & pm.K.isin(KS) & pm.method.isin(METHODS)].copy()
d2["yi"] = d2.method.map(midx)
for T in CORPORA:
    for K in KS:
        e = d2[(d2["T"] == T) & (d2.K == K)][["yi", "method", "uar_prop", "uar_other", "effect", "lo", "hi", "n_units"]].sort_values("yi")
        csv(f"dai2_{T}_K{K}.csv", e, f"results/dai2/stats/prop_minus_method.csv student=AVT T={T} K={K}（提案 − 手法．UAR ポイント，95% 区間．ラベル伝播は K=0 が無い）")
        for _, r in e.iterrows():
            put(f"dai2_{T}_{r.method}_{K}", r.effect, 1, True)
            put(f"dai2lo_{T}_{r.method}_{K}", r.lo, 1, True)
            put(f"dai2hi_{T}_{r.method}_{K}", r.hi, 1, True)
            put(f"dai2uar_{T}_{r.method}_{K}", r.uar_other, 1)
            v = f"{r.effect:+.1f}".replace("-", "\\textminus{}")
            cell = f"\\textbf{{{v}}}" if r.lo > 0 else (f"\\textcolor{{mBad}}{{{v}}}" if r.hi < 0 else v)
            NUM[re.sub(r"[^A-Za-z]", "", f"dai2cell_{T}_{r.method}_{K}".replace("0", "zero").replace("5", "five").replace("2", "two").replace("3", "three"))] = cell
        pr = pm[(pm.student == "AVT") & (pm.K == K) & (pm["T"] == T) & (pm.method == "self")].uar_prop.iloc[0]
        put(f"dai2prop_{T}_{K}", pr, 1)
# 下端が 0 より上のセルの数（手法ごと，12 セル）
for m in METHODS:
    e = d2[d2.method == m]
    put(f"dai2above_{m}", int((e.lo > 0).sum()), 0)
    put(f"dai2n_{m}", int(len(e)), 0)
    put(f"dai2below_{m}", int((e.hi < 0).sum()), 0)

# ---------- 実験 3：位置と向きの損（版 four），直す余地 ----------
dc = pd.read_csv(ROOT / "results/dai3/eval/decomp.csv")
assert (dc.commit.str.startswith("ba98974")).all() and not dc.dirty.any()
dc = dc[(dc.version == "four") & dc.quantity.isin(["position", "direction", "sum"])]
rows = []
for T in CORPORA:
    for kind in STUDENTS:
        for rest in ["start", "none|5", "prop|5", "none|25", "prop|25"]:
            item = f"{kind}|{rest}"
            q = dc[(dc["T"] == T) & (dc.item == item)].set_index("quantity")
            if len(q) == 0:
                continue
            rows.append(dict(T=T, kind=kind, rest=rest, position=q.loc["position", "effect"], direction=q.loc["direction", "effect"],
                             total=q.loc["sum", "effect"], pos_lo=q.loc["position", "lo"], pos_hi=q.loc["position", "hi"],
                             dir_lo=q.loc["direction", "lo"], dir_hi=q.loc["direction", "hi"], n=int(q.loc["sum", "n_units"])))
            tag = rest.replace("|", "")
            put(f"pos_{T}_{kind}_{tag}", q.loc["position", "effect"], 1)
            put(f"dir_{T}_{kind}_{tag}", q.loc["direction", "effect"], 1)
            put(f"tot_{T}_{kind}_{tag}", q.loc["sum", "effect"], 1)
D = pd.DataFrame(rows)
csv("decomp.csv", D, "results/dai3/eval/decomp.csv version=four（W* に切片を当て直した参照との UAR の差．position = 切片の当て直しで減る分，direction = 当て直した後に残る分）")
# 出発点の損（AVT）の積み上げ用
st = D[(D.kind == "AVT") & (D.rest == "start")][["T", "position", "direction", "total"]].copy()
st["ci"] = st["T"].map({c: i + 1 for i, c in enumerate(CORPORA)})
csv("decomp_start_avt.csv", st, "decomp.csv の AVT|start（出発点の損の内訳）")

# 難しさ：AVT の出発点，K=25，上限（W* に切片を当て直した参照）
fu = pd.read_csv(ROOT / "results/explore_fusion/fusion_cells.csv")
diff = []
for i, T in enumerate(CORPORA):
    n0 = fu[(fu["T"] == T) & (fu.K == 0)].avt_none.iloc[0]
    n25 = fu[(fu["T"] == T) & (fu.K == 25)].avt_none.iloc[0]
    tot0 = D[(D["T"] == T) & (D.kind == "AVT") & (D.rest == "start")].total.iloc[0]
    tot25 = D[(D["T"] == T) & (D.kind == "AVT") & (D.rest == "none|25")].total.iloc[0]
    up = n0 + tot0
    assert abs(up - (n25 + tot25)) < 0.1, (T, up, n25 + tot25)
    diff.append(dict(ci=i + 1, T=T, start=n0, k25=n25, upper=up, gap_start=tot0, gap_k25=tot25))
    put(f"start_{T}", n0, 1); put(f"ktf_{T}", n25, 1); put(f"upper_{T}", up, 1); put(f"gapstart_{T}", tot0, 1); put(f"gapk_{T}", tot25, 1)
csv("difficulty.csv", pd.DataFrame(diff), "AVT の生徒：出発点の UAR（fusion_cells.csv の avt_none K=0），K=25 で学び直した UAR（同 K=25），上限 = 出発点 ＋ decomp の AVT|start の sum（K=25 からの和と 0.1 以内で一致）")

ex = pd.read_csv(ROOT / "results/dai3/eval/expl4_corpus_restored.csv")
ex = ex[ex.subset == "all"]
rm = []
for T in CORPORA:
    for K in (5, 25):
        g = ex[(ex.corpus == T) & (ex.K == K)].set_index("quantity")
        for s, col in (("A", "A"), ("V", "V"), ("T", "T_student")):
            gain, room, repro = g.loc["gain", col], g.loc["room", col], g.loc["repro", col]
            rm.append(dict(T=T, K=K, student=s, gain=gain, room=room, repro=repro, share=100 * gain / room))
            put(f"room_{T}_{s}_{K}", room, 1); put(f"gainr_{T}_{s}_{K}", gain, 1, True); put(f"share_{T}_{s}_{K}", 100 * gain / room, 0, True)
            put(f"repro_{T}_{s}_{K}", repro, 0)
csv("room.csv", pd.DataFrame(rm), "results/dai3/eval/expl4_corpus_restored.csv subset=all（gain = 提案 − K 本だけ，room = 参照 − K 本だけ，repro = 再現できる度合い %）")

# ---------- 探索：教師の先回りと得，一致の有無ごとの教師の正解率 ----------
ta = pd.read_csv(ROOT / "results/explore_consensus/teacher_adv_cells.csv")
for K in (5, 25):
    for s in STUDENTS:
        e = ta[(ta.K == K) & (ta.student == s)][["T", "adv_pool", "gain_prop", "gain_kd", "agree", "fix", "harm", "teacher_uar_pool", "none_uar_pool", "n_units"]]
        csv(f"teacher_adv_{s}_K{K}.csv", e, f"results/explore_consensus/teacher_adv_cells.csv student={s} K={K}（adv_pool = 教師の候補での UAR − 生徒（K 本だけ）の候補での UAR，gain_prop = 提案のテストの得．MOSEI は 39 単位）")
        for _, r in e.iterrows():
            put(f"adv_{r['T']}_{s}_{K}", r.adv_pool, 1, True); put(f"agree_{r['T']}_{s}_{K}", 100 * r.agree, 0)
            put(f"tuar_{r['T']}_{s}_{K}", r.teacher_uar_pool, 1); put(f"suar_{r['T']}_{s}_{K}", r.none_uar_pool, 1)
            put(f"fix_{r['T']}_{s}_{K}", 100 * r.fix, 0); put(f"harm_{r['T']}_{s}_{K}", 100 * r.harm, 0)
    e = ta[(ta.K == K) & ta.student.isin(STUDENTS)]
    from scipy.stats import spearmanr
    rho = spearmanr(e.adv_pool, e.gain_prop).correlation
    put(f"rho_{K}", rho, 2)

cs = pd.read_csv(ROOT / "results/explore_consensus/consensus_cells.csv")
for K in (5, 25):
    for s in STUDENTS:
        e = cs[(cs.K == K) & (cs.student == s)]
        w = e.pivot(index="T", columns="group", values=["teacher_acc", "share", "fix", "harm"])
        out = pd.DataFrame({"T": w.index, "ci": [CORPORA.index(t) + 1 for t in w.index],
                            "acc_agree": 100 * w[("teacher_acc", "一致")].values, "acc_dis": 100 * w[("teacher_acc", "不一致")].values,
                            "share_agree": 100 * w[("share", "一致")].values,
                            "harm_agree": 100 * w[("harm", "一致")].values, "harm_dis": 100 * w[("harm", "不一致")].values}).sort_values("ci")
        csv(f"consensus_{s}_K{K}.csv", out, f"results/explore_consensus/consensus_cells.csv student={s} K={K}（教師の正解率 %：生徒以外のモダリティの判定が一致する発話／しない発話）")
        for _, r in out.iterrows():
            put(f"cagree_{r['T']}_{s}_{K}", r.acc_agree, 0); put(f"cdis_{r['T']}_{s}_{K}", r.acc_dis, 0); put(f"cshare_{r['T']}_{s}_{K}", r.share_agree, 0)
            put(f"charma_{r['T']}_{s}_{K}", r.harm_agree, 1); put(f"charmd_{r['T']}_{s}_{K}", r.harm_dis, 1)

# ---------- 探索：積の融合，指数の上限，K 本の選択，LOO の信頼度 ----------
lo = pd.read_csv(ROOT / "results/explore_fusion/loo_cells.csv")
ks = pd.read_csv(ROOT / "results/explore_fusion/kshot_cells.csv")
gr = pd.read_csv(ROOT / "results/explore_fusion/grid_cells.csv")
print("grid_cells columns:", list(gr.columns))
fz = []
for T in CORPORA:
    for K in (5, 25):
        a = lo[(lo["T"] == T) & (lo.K == K)].iloc[0]; b = ks[(ks["T"] == T) & (ks.K == K)].iloc[0]
        fz.append(dict(T=T, K=K, avt=a.avt_joint, prod111=a.prod111, kshot=b.kshot_sel, rule_cl=a.rule_cl, oracle=a.oracle, cl_ok=100 * a.cl_ok, loo_ok=100 * a.loo_ok))
        put(f"favt_{T}_{K}", a.avt_joint, 1); put(f"fprod_{T}_{K}", a.prod111, 1); put(f"fkshot_{T}_{K}", b.kshot_sel, 1)
        put(f"frule_{T}_{K}", a.rule_cl, 1); put(f"foracle_{T}_{K}", a.oracle, 1); put(f"fclok_{T}_{K}", 100 * a.cl_ok, 0)
        put(f"fruled_{T}_{K}", a.rule_cl - a.avt_joint, 1, True); put(f"foracled_{T}_{K}", a.oracle - a.avt_joint, 1, True)
        put(f"fprodd_{T}_{K}", a.prod111 - a.avt_joint, 1, True); put(f"fkshotd_{T}_{K}", b.kshot_sel - a.avt_joint, 1, True)
        put(f"fheavy_{T}_{K}", 100 * b.heavy_match, 0)
csv("fusion.csv", pd.DataFrame(fz), "results/explore_fusion/{loo_cells,kshot_cells}.csv（テスト UAR：AVT の生徒，積（指数 1,1,1），K 本で選んだ指数，塊ごとの LOO の信頼度の指数，指数をテストで選んだ上限）")

# ---------- 今日の途中の値：AFEW の LOO による選択（本番の select_cells.csv があればそれを使う） ----------
sel_cells = ROOT / "results/explore_fusion/select_cells.csv"
if sel_cells.exists():
    sc = pd.read_csv(sel_cells); src = "results/explore_fusion/select_cells.csv（本番の台本の出力）"
else:
    su = pd.read_csv(ROOT / "results/explore_fusion/peek_select_units_AFEW.csv")
    sc = su.groupby(["T", "K"]).agg(n_units=("seed", "size"), avt_joint=("avt_joint", "mean"), rule_cl=("cl_prod_test", "mean"),
                                    select_cl=("select_cl", "mean"), select_loo=("select_loo", "mean"), oracle_select=("oracle_select", "mean"),
                                    oracle_grid=("oracle_grid", "first"), share_prod_cl=("cl_choose_prod", "mean"), share_prod_loo=("loo_choose_prod", "mean"),
                                    cl_est_prod=("cl_est_prod", "mean"), cl_est_avt=("cl_est_avt", "mean")).reset_index()
    src = "results/explore_fusion/peek_select_units_AFEW.csv（保存した held npz と評価の回の識別器から本番と同じ式で出した AFEW の途中の値）"
csv("select.csv", sc, src)
for _, r in sc.iterrows():
    T, K = r["T"], int(r.K)
    put(f"savt_{T}_{K}", r.avt_joint, 1); put(f"srule_{T}_{K}", r.rule_cl, 1); put(f"ssel_{T}_{K}", r.select_cl, 1); put(f"sselloo_{T}_{K}", r.select_loo, 1)
    put(f"soracle_{T}_{K}", r.oracle_select, 1); put(f"sgrid_{T}_{K}", r.oracle_grid, 1); put(f"sshare_{T}_{K}", 100 * r.share_prod_cl, 0)
    put(f"sseld_{T}_{K}", r.select_cl - r.avt_joint, 1, True); put(f"sruled_{T}_{K}", r.rule_cl - r.avt_joint, 1, True)
    put(f"sestp_{T}_{K}", r.cl_est_prod, 1); put(f"sesta_{T}_{K}", r.cl_est_avt, 1); put(f"sn_{T}_{K}", int(r.n_units), 0)
    put(f"soracled_{T}_{K}", r.oracle_select - r.avt_joint, 1, True)
NUM["selcorpora"] = "，".join(sorted(sc["T"].unique(), key=CORPORA.index))

# ---------- 書き出し ----------
with open(FIG / "numbers.tex", "w") as f:
    f.write("% scripts/export_figdata.py が書いた．手で直さない．\n")
    for k, v in sorted(NUM.items()):
        f.write(f"\\expandafter\\gdef\\csname NUM@{k}\\endcsname{{{v}}}\n")
print(len(NUM), "numbers;", len(list(FIG.glob('*.csv'))), "csv files")
