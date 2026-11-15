"""実験 16・17 の K = 5・10・25 の結果を，研究者が示した形の LaTeX の表に書き出す（結果ファイルを読むだけ）．
出力：docs/説明_実験16・17_結果の表.tex（説明の資料から \\input する）"""
import re
import pandas as pd
_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])   # DAI の直下（整理 2026-11-15）

R = _ROOT + "/results"
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
KS = (5, 10, 25)
OLD16 = _ROOT + "/docs/結果/実験16_下調べ_結果の表_2026-11-14.md"
OUT = _ROOT + "/docs/説明/説明_実験16・17_結果の表.tex"


def both(d, f, **kw):
    return pd.concat([pd.read_csv(f"{R}/{d}/summary/{f}", **kw), pd.read_csv(f"{R}/{d}_k10/summary/{f}", **kw)], ignore_index=True)


def num(v, nd=2):
    return f"${v:+.{nd}f}$".replace("-", "-")


def mark(verdict):
    return {"上回る": "$\\uparrow$", "下回る": "$\\downarrow$"}.get(verdict, "")


def cellv(q, ci):
    s = num(q.est) + mark(q.verdict)
    if ci:
        s += f"\\newline{{\\tiny [{q.lo:+.1f}, {q.hi:+.1f}]}}".replace("-", "$-$").replace("+", "$+$")
    return s


def row(x, ci):
    x = x.drop_duplicates("c").set_index("c")
    out, u, d = [], 0, 0
    for c in C:
        if c not in x.index:
            out.append("—")
            continue
        q = x.loc[c]
        out.append(cellv(q, ci))
        u += q.verdict == "上回る"
        d += q.verdict == "下回る"
    return out, f"{u}／{d}"


def tex(s):
    return s.replace("_", "\\_").replace("−", "$-$")


HEAD_RULE = ("\\begin{{longtable}}{{>{{\\raggedright}}p{{0.75cm}}>{{\\raggedright}}p{{5.6cm}}>{{\\raggedright}}p{{2.05cm}}>{{\\raggedright}}p{{2.05cm}}"
             ">{{\\raggedright}}p{{2.05cm}}>{{\\raggedright}}p{{2.05cm}}p{{0.75cm}}}}\n\\caption{{{cap}}}\\\\\n\\toprule\n区分 & 比べ & AFEW & IEMOCAP & MOSEI & MELD & "
             "$\\uparrow$／$\\downarrow$ \\\\\n\\midrule\n\\endfirsthead\n\\toprule\n区分 & 比べ & AFEW & IEMOCAP & MOSEI & MELD & $\\uparrow$／$\\downarrow$ \\\\\n"
             "\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n")
HEAD_LV = ("\\begin{{longtable}}{{>{{\\raggedright}}p{{8.2cm}}rrrr}}\n\\caption{{{cap}}}\\\\\n\\toprule\n版 & AFEW & IEMOCAP & MOSEI & MELD \\\\\n\\midrule\n\\endfirsthead\n"
           "\\toprule\n版 & AFEW & IEMOCAP & MOSEI & MELD \\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n")

L = []
w = L.append
w("% 生成：scratchpad の latex_tables.py（結果ファイルを読むだけ）．手で直さない．")
w("\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\\setlength{\\LTcapwidth}{\\textwidth}")

# ================= 実験 16
lab16 = {}
for line in open(OLD16):
    m = re.match(r"\| (?:R\d|R3b|比べ) \| (.+?)（`([^`]+?) − ([^`]+?)`） \|", line)
    if m:
        lab16.setdefault((m.group(2), m.group(3)), m.group(1))
vlab16 = {}
for line in open(OLD16):
    m = re.match(r"\| (.+?)（`([^`− ]+)`） \|", line)
    if m:
        vlab16.setdefault(m.group(2), m.group(1).strip())


def fix(s):
    s = s.replace("水準 K＋相性ソース", "水準は K 本＋相性はソース").replace("水準K＋相性ソース", "水準は K 本＋相性はソース").replace("K本", "K 本")
    return s.replace("（切片を全ラベルで決めた層）", "")


cr = both("dai16_probe", "criteria.csv")
ef = both("dai16_probe", "effects.csv")
lv = both("dai16_probe", "levels.csv")
r0 = both("dai16_probe", "r0.csv")
MAIN16 = {("M_F6_orc", "prod", "adj"), ("M_F6_mixsel", "prod", "adj")}

w("\\subsection*{実験 16：各版の UAR}")
for K in KS:
    w(HEAD_LV.format(cap=f"実験 16：各版の UAR（K = {K}．ラベルを使わずに切片をずらした答え．K 本だけの融合，提案の融合の生徒，微調整した分類器はずらす前の答え）"))
    for v, lab in vlab16.items():
        x = lv[(lv.version == v) & (lv.K == K)].set_index("c")
        if not len(x):
            continue
        col = "uar_raw" if v.startswith(("none_", "ret_", "joint_")) else "uar_adj"
        w(tex(fix(lab)) + " & " + " & ".join(f"{x.loc[c, col]:.2f}" for c in C) + " \\\\")
    w("\\end{longtable}")

w("\\subsection*{実験 16：判定の規則と比べ}")
w("最後の列は，上回ったコーパスの数／下回ったコーパスの数．区間は主の 2 行（R1 と R2 の先頭）にだけ書いた（ほかの行の区間は \\texttt{docs/実験16・17\\_下調べ\\_結果の表\\_K5・10・25\\_2026-11-14.md}）．\\par")
for K in KS:
    w(HEAD_RULE.format(cap=f"実験 16：判定の規則と比べ（K = {K}）"))
    seen, prev = set(), None
    for r in cr[cr.K == K].itertuples():
        if r.where not in ("adj", "ob"):
            continue
        if r.where == "ob" and not (r.rule == "R1" and r.a in ("M_F6_orc", "M_F3_orc")):
            continue
        key = (r.rule, r.a, r.b, r.where)
        if key in seen:
            continue
        seen.add(key)
        x = ef[(ef.a == r.a) & (ef.b == r.b) & (ef.K == K) & (ef["where"] == r.where)]
        if not len(x):
            continue
        lab = fix(lab16.get((r.a, r.b), f"{r.a} − {r.b}"))
        if r.where == "ob":
            lab += "（切片をそろえた層）"
        elif r.a == "S_F6_mixsel":
            lab = "ソースの分類器の重みづけ − ソースの分類器"
        elif r.rule in ("R3b", "R3"):
            body = lab.split("：", 1)[1]
            head = "見込みで選ぶ − 元の版：" if r.rule == "R3b" else "見込みで選ぶ − 積："
            lab = (head if prev != r.rule else "同・") + body
        elif r.b == "prod" and r.rule in ("R1", "R2"):
            lab += " − 積"
        elif r.b == "S_eq":
            lab += " − ソースの分類器"
        prev = r.rule
        o, ud = row(x, (r.a, r.b, r.where) in MAIN16)
        w(f"{tex(r.rule)} & {tex(lab)} & " + " & ".join(o) + f" & {ud} \\\\")
    w("\\end{longtable}")

w("\\subsection*{実験 16：ソースで測った信頼度は新しいコーパスで通用するか（R0．微調整した声・顔・言葉）}")
SRC = {"home": "ソースの中", "ext": "外で測り直す（主）", "away": "別のソース", "K": "K 本"}
w("\\begin{longtable}{llp{3.0cm}rlrrr}\n\\caption{ソースで測った信頼度の表と，仮のターゲットの全ラベルで測った表の比べ（R0）．規則は「表の形の相関」と「入れ替えたときの相関」で決める（設計どおり）．}\\\\\n\\toprule\n"
  "コーパス & K & 測り方 & 表の形の相関 & 規則 & \\shortstack{感情ごとの\\\\最強が合う} & 水準の相関 & \\shortstack{最強のモダリ\\\\ティが合う} \\\\\n\\midrule\n\\endfirsthead\n\\toprule\n"
  "コーパス & K & 測り方 & 表の形の相関 & 規則 & \\shortstack{感情ごとの\\\\最強が合う} & 水準の相関 & \\shortstack{最強のモダリ\\\\ティが合う} \\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot")
for c in ("AFEW", "IEMOCAP", "MELD", "MOSEI"):
    for K in KS:
        for s in ("home", "ext", "away", "K"):
            q = r0[(r0.c == c) & (r0.K == K) & (r0.fam == "M") & (r0.source == s)]
            if not len(q):
                continue
            q = q.iloc[0]
            w(f"{c} & {K} & {SRC[s]} & {q.intcorr:.3f} & {'満たす' if q.hit else '満たさない'} & {q.emo_agree:.3f} & ${q.levelcorr:+.3f}$ & {q.topok:.3f} \\\\")
    w("\\midrule")
L[-1] = L[-1]  # 最後の midrule はそのまま（bottomrule が続く）
w("\\end{longtable}")

# ================= 実験 17
w("%%SPLIT%%")
cr = both("dai17_probe", "criteria.csv")
ef = both("dai17_probe", "effects.csv")
lv = both("dai17_probe", "levels.csv")
VL17 = [("prod", "積（今の提案）"), ("none_AVT", "K 本だけの融合"), ("joint_AVT", "提案の融合の生徒"), ("ret_A", "微調整した声"), ("ret_V", "微調整した顔"),
        ("ret_T", "微調整した言葉"), ("MN_F6orc_sel", "近いソースの発話を足す・全ラベルの信頼度（参照）"), ("MN_eq_sel", "近いソースの発話を足す・等しい重み"),
        ("MN_F2K_sel", "近いソースの発話を足す・水準は K 本"), ("MN_F6K_sel", "近いソースの発話を足す・感情ごとの信頼度は K 本"),
        ("MS_eq_sel", "ソースの分類器を足す"), ("MS_eq_tied", "ソースの分類器を足す（同じ倍率）"), ("N_eq_sel", "近いソースの発話の答えだけ"),
        ("S_eq", "ソースの分類器だけ"), ("MN_eq_selmod", "足した版・両方の近さ・モダリティごと"), ("MN_eq_selfeat", "足した版・特徴の近さだけ"),
        ("MN_eq_sellogit", "足した版・分類器の答えの近さだけ"), ("MN_eq_seljoint", "足した版・3 つのモダリティを合わせて引く")]
w("\\subsection*{実験 17：各版の UAR（推定時にまとめた答え）}")
for K in KS:
    w(HEAD_LV.format(cap=f"実験 17：各版の UAR（K = {K}．ラベルを使わずに切片をずらした答え．K 本だけの融合，提案の融合の生徒，微調整した分類器はずらす前の答え）"))
    for v, lab in VL17:
        x = lv[(lv.version == v) & (lv.K == K)].set_index("c")
        if not len(x):
            continue
        col = "uar_raw" if v.startswith(("none_", "ret_", "joint_")) else "uar_adj"
        w(tex(lab) + " & " + " & ".join(f"{x.loc[c, col]:.2f}" if pd.notna(x.loc[c, col]) else "—" for c in C) + " \\\\")
    w("\\end{longtable}")
MAIN17 = {("MN_F6orc_sel", "prod"), ("MN_eq_sel", "prod"), ("MN_F2K_sel", "prod"), ("MN_F6K_sel", "prod")}
w("\\subsection*{実験 17：判定の規則と比べ（推定時にまとめた答え）}")
for K in KS:
    w(HEAD_RULE.format(cap=f"実験 17：判定の規則と比べ（K = {K}．推定時にまとめた答え）"))
    for r in cr[(cr.K == K) & (cr["where"] == "adj")].itertuples():
        x = ef[(ef.a == r.a) & (ef.b == r.b) & (ef.K == K) & (ef["where"] == "adj")]
        if not len(x):
            continue
        lab = r.axis + (" − 積" if r.b == "prod" else "")
        o, ud = row(x, (r.a, r.b) in MAIN17)
        w(f"{tex(r.rule)} & {tex(lab)} & " + " & ".join(o) + f" & {ud} \\\\")
    w("\\end{longtable}")

se = both("dai17_probe", "students_effects.csv")
sl = both("dai17_probe", "students_levels.csv")
SL = {"AVT": "融合", "A": "声", "V": "顔", "T": "言葉"}
TL = {"prod_sel": "積", "MN_eq_sel_sel": "等しい重み", "MN_F2K_sel_sel": "水準は K 本", "MN_F6K_sel_sel": "感情ごとの信頼度は K 本", "MS_eq_sel_sel": "ソースの分類器を足す",
      "MS_eq_tied_sel": "ソースの分類器を足す（同じ倍率）", "none": "K 本だけ", "prod": "積（実験 1 の設定のまま）", "MN_eq_sel": "等しい重み（実験 1 の設定のまま）",
      "MN_F2K_sel": "水準は K 本（実験 1 の設定のまま）", "MN_F6K_sel": "感情ごとの信頼度は K 本（実験 1 の設定のまま）", "MS_eq_sel": "ソースの分類器を足す（実験 1 の設定のまま）",
      "MS_eq_tied": "同じ倍率（実験 1 の設定のまま）"}
w("\\clearpage\\subsection*{実験 17：学び直した分類器（生徒）の UAR}")
for K in KS:
    w(HEAD_LV.format(cap=f"実験 17：疑似ラベルごとに学び直した分類器の UAR（K = {K}．設定を選び直した点．「K 本だけ」は疑似ラベルを使わない分類器）"))
    for s in ("AVT", "A", "V", "T"):
        for t in ("none", "prod_sel", "MN_eq_sel_sel", "MN_F2K_sel_sel", "MN_F6K_sel_sel", "MS_eq_sel_sel", "MS_eq_tied_sel"):
            x = sl[(sl.student == s) & (sl.teacher == t) & (sl.K == K)].set_index("c")
            if len(x):
                w(f"{SL[s]} × {tex(TL[t])} & " + " & ".join(f"{x.loc[c, 'uar']:.2f}" for c in C) + " \\\\")
        w("\\midrule")
    w("\\end{longtable}")
w("\\subsection*{実験 17：判定の規則と比べ（学び直した分類器）}")
for K in KS:
    w(HEAD_RULE.format(cap=f"実験 17：学び直した分類器の比べ（K = {K}．R5 は「疑似ラベル X − 積」で，どちらも設定を選び直した点）"))
    for s in ("AVT", "A", "V", "T"):
        for (a, b), _ in se[(se.student == s) & (se.K == K)].groupby(["a", "b"], sort=False):
            if a == "prod_sel" and b == "none":
                continue
            x = se[(se.student == s) & (se.K == K) & (se.a == a) & (se.b == b)]
            isr = b == "prod_sel" and a in ("MN_eq_sel_sel", "MN_F2K_sel_sel", "MN_F6K_sel_sel")
            o, ud = row(x, isr and s == "AVT")
            w(f"{'R5' if isr else '比べ'} & {SL[s]}：{tex(TL.get(a, a))} − {tex(TL.get(b, b))} & " + " & ".join(o) + f" & {ud} \\\\")
        w("\\midrule")
    w("\\end{longtable}")

rc = pd.concat([pd.read_csv(f"{R}/dai17_probe/summary/reweight_criteria.csv", dtype={"K": str}),
                pd.read_csv(f"{R}/dai17_probe_k10/summary/reweight_criteria.csv", dtype={"K": str}).query("K == '10'")], ignore_index=True)
re_ = pd.concat([pd.read_csv(f"{R}/dai17_probe/summary/reweight_effects.csv", dtype={"K": str}),
                 pd.read_csv(f"{R}/dai17_probe_k10/summary/reweight_effects.csv", dtype={"K": str}).query("K == '10'")], ignore_index=True)
ML = {"A": "声", "V": "顔", "T": "言葉", "POE": "3 つの積"}
AL = {"hub": "よく引かれる発話を重く", "rel": "信頼度で重く", "hubrel": "両方", "base": "重みなし", "base_sel": "重みなし", "hub_sel": "よく引かれる発話を重く",
      "rel_sel": "信頼度で重く", "hubrel_sel": "両方"}
w("\\subsection*{実験 17：ソースの発話に重みを付けて学び直す（B）}")
w("K = 0 はターゲットのラベルを使わない出発点，0ob は切片を全ラベルで決めた出発点（参照）．K = 5・10・25 は出発点から K 本で微調整した分類器で，R4 は引き戻しの強さを重みごとに選び直した比べ．\\par")
for K in ("0", "0ob", "5", "10", "25"):
    w(HEAD_RULE.format(cap=f"実験 17：ソースの発話に重みを付けて学び直す（K = {K}）"))
    for r in rc[rc.K == K].drop_duplicates(["modality", "a", "b"]).itertuples():
        x = re_[(re_.K == K) & (re_.modality == r.modality) & (re_.a == r.a) & (re_.b == r.b)]
        if not len(x):
            continue
        isr = r.a.endswith("_sel") and r.b == "base_sel"
        lab = f"{ML[r.modality]}：{AL.get(r.a, r.a)} − {AL.get(r.b, r.b)}" + ("（強さを選び直す）" if r.a.endswith("_sel") and r.b == "base_sel" else ("（選び直す − 元の強さ）" if r.b == "base" and r.a == "base_sel" else ""))
        o, ud = row(x, False)
        w(f"{'R4' if isr else '比べ'} & {tex(lab)} & " + " & ".join(o) + f" & {ud} \\\\")
    w("\\end{longtable}")
w("\\normalsize")
txt = "\n".join(L) + "\n"
a, b = txt.split("%%SPLIT%%\n")
hdr = L[1] + "\n"
open(OUT.replace(".tex", "_16.tex"), "w").write(a + "\\normalsize\n")
open(OUT.replace(".tex", "_17.tex"), "w").write(L[0] + "\n" + hdr + b)
print(len(L))
