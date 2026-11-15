"""実験 18（6 コーパス）の結果のうち，実験 16 の版（提案の手法とその比べ）だけを表に書き出す（結果ファイルを読むだけ．実験 17 の版は出さない）．
計算の後の点検（gate-code）が通ってから流す．出力：docs/結果/実験18_下調べ_結果の表_2026-11-14.md

2026-11-15：照合の第 1 回（checks/DAI-18/claim.md）の指摘で直した．
  - 区間の端が 0.05 より 0 に近いときと，差が丸めると ±1.00 になるときは，境目のどちら側かが読めるよう桁を足す（結果の文の台本 extract_dai18_text.py と同じ規則）．
  - 対照 M_F6n_* の名前を「分類器 × 感情の成分を消した対照」にそろえ，M_F6 − M_F6n の区分を「帰し方」にした（設計 5 節の得の帰し方の比べ）．
  - 3 節に，mix2 と mix3 が同じ答えになることを 1 行足した（数は results/dai18_probe/posthoc/provenance_selcounts.json から読む）．"""
import json

import pandas as pd
_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])   # DAI の直下（整理 2026-11-15）

R = _ROOT + "/results/dai18_probe/summary"
OUT = _ROOT + "/docs/結果/実験18_下調べ_結果の表_2026-11-14.md"
C6 = ["AFEW", "IEMOCAP", "MOSEI", "MELD", "CREMA-D", "RAVDESS"]
KS = (5, 10, 25)
VLAB = [("prod", "積（今の提案の推定時の答え）"), ("none_AVT", "K 本だけの融合（声・顔・言葉）"), ("none_AV", "K 本だけの融合（声・顔）"),
        ("none_A", "K 本だけの声"), ("none_V", "K 本だけの顔"), ("none_T", "K 本だけの言葉"),
        ("M_F6_mixsel", "提案の手法：本体の形・水準は K 本＋相性はソース"), ("M_F6_K", "本体の形・K 本だけ"), ("M_F6_ext", "本体の形・ソースだけ"),
        ("M_F6_orc", "本体の形・全ラベル（参照）"), ("M_F2_orc", "水準だけ・全ラベル（参照）"), ("M_F6n_orc", "分類器 × 感情の成分を消した対照・全ラベル"),
        ("M_F6n_mixsel", "分類器 × 感情の成分を消した対照・水準は K 本＋相性はソース")]
PAIRS = [("R1", "M_F6_orc", "prod", "本体の形・全ラベル − 積（参照）"), ("R2", "M_F6_mixsel", "prod", "提案の手法（水準は K 本＋相性はソース）− 積"),
         ("R2", "M_F6_K", "prod", "本体の形・K 本だけ − 積"), ("R2", "M_F6_ext", "prod", "本体の形・ソースだけ − 積"),
         ("比べ", "M_F2_orc", "prod", "水準だけ・全ラベル − 積"), ("帰し方", "M_F6_orc", "M_F6n_orc", "分類器 × 感情の成分の分（全ラベル）：本体の形 − 対照"),
         ("帰し方", "M_F6_mixsel", "M_F6n_mixsel", "分類器 × 感情の成分の分（水準は K 本＋相性はソース）：提案の手法 − 対照")]


def fest(x):
    """差：小数 2 桁．丸めると ±1.00 になって 1.0 点の境目のどちら側か読めないときは桁を足す．"""
    s = f"{x:+.2f}"
    if abs(float(s)) == 1.0 and abs(x) != 1.0:
        for nd in (3, 4, 5):
            s = f"{x:+.{nd}f}"
            if abs(float(s)) != 1.0:
                break
    return s


def fend(x):
    """区間の端：小数 1 桁．0.05 より 0 に近いときは符号が読めるよう桁を足す．"""
    s = f"{x:+.1f}"
    if abs(x) < 0.05 and x != 0:
        for nd in (2, 3, 4):
            s = f"{x:+.{nd}f}"
            if float(s) != 0.0:
                break
    return s


def cell(q):
    m = {"上回る": "↑", "下回る": "↓"}.get(q.verdict, "")
    return f"{fest(q.est)} [{fend(q.lo)}, {fend(q.hi)}]{m}"


def row(x):
    x = x.drop_duplicates("c").set_index("c")
    out, u, d, u4 = [], 0, 0, 0
    for c in C6:
        if c not in x.index:
            out.append("—")
            continue
        q = x.loc[c]
        out.append(cell(q))
        u += q.verdict == "上回る"
        d += q.verdict == "下回る"
        u4 += (q.verdict == "上回る") and c in C6[:4]
    return out, u, d, u4


lv = pd.read_csv(f"{R}/levels.csv")
ef = pd.read_csv(f"{R}/effects.csv")
sel = pd.read_csv(f"{R}/sel.csv")
L = []
w = L.append
w("# 実験 18 の下調べ（6 コーパス）の結果の表：実験 16 の版（2026-11-14）")
w("")
w("`results/dai18_probe/summary/` から書き出した（結果ファイルを読むだけ）．実験 17 の版（近いソースの発話を引く版）は，研究者の指示（2026-11-14）でこの表から外した．"
  "最初の版は照合（gate-claim）の前に作った（2026-11-14 20:39）．2026-11-15 に，照合の第 1 回（`checks/DAI-18/claim.md`）の指摘で，区間の端の桁の書き方と対照の名前を直して書き出し直した（値は変えていない）．")
w("")
w("- 差は UAR の点（%）の差，括弧は 95% 区間．↑ は「上回る」（区間の下端 > 0 かつ差 ≥ 1.0），↓ は「下回る」．")
w("- 判定：6 コーパスのうち上回るのが 2 以上で「当たり」（実験 16 の版は声しか無い RAVDESS に使えないので 5 コーパス）．元の 4 コーパスだけの数えも並べる．")
w("- **実験 16（4 コーパス）の再現ではない**：元の 4 コーパスの計算でも，ソースに CREMA-D と RAVDESS が入るので，出発点と微調整した分類器が変わる．")
w("- CREMA-D は声と顔，RAVDESS は声だけ．どちらも演技で決まった文を読むコーパスである．")
w("- 区間の端が 0.05 より 0 に近いときは，符号が読めるよう桁を足して書く（例：[-0.02, +0.6]）．")
w("- 「分類器 × 感情の成分を消した対照」（`M_F6n_*`）は，重みの表から分類器と感情の組み合わせに固有の成分だけを消した版である（最初の版では「相性を消した対照」，差を「モダリティ × 感情の分」と書いていた）．"
  "区分「帰し方」の行は，設計 5 節の得の帰し方の比べで，規則の判定は付けない．区分「比べ」の行の ↑↓ は，同じ基準を当てた印で，規則の判定ではない．")
w("")
w("## 1. 各版の UAR")
w("")
for K in KS:
    w(f"### K = {K}")
    w("")
    w("| 版 | " + " | ".join(C6) + " |")
    w("|---|" + "---|" * len(C6))
    for v, lab in VLAB:
        x = lv[(lv.version == v) & (lv.K == K)].set_index("c")
        if not len(x):
            continue
        col = "uar_raw" if v.startswith("none_") else "uar_adj"
        w(f"| {lab} | " + " | ".join(f"{x.loc[c, col]:.2f}" if c in x.index and pd.notna(x.loc[c, col]) else "—" for c in C6) + " |")
    w("")
w("## 2. 判定の規則と比べ（ラベルを使わずに切片をずらした答え）")
w("")
for K in KS:
    w(f"### K = {K}")
    w("")
    w("| 区分 | 比べ | " + " | ".join(C6) + " | ↑／↓ | 元の 4 コーパスの↑ | 判定 |")
    w("|---|---|" + "---|" * len(C6) + "---|---|---|")
    for rule, a, b, lab in PAIRS:
        x = ef[(ef.a == a) & (ef.b == b) & (ef.K == K) & (ef["where"] == "adj")]
        if not len(x):
            continue
        o, u, d, u4 = row(x)
        j = ("当たり" if u >= 2 and d < 2 else ("下回った" if d >= 2 and u < 2 else "外れ")) if rule in ("R1", "R2") else "—"
        w(f"| {rule} | {lab} | " + " | ".join(o) + f" | {u}／{d} | {u4} | {j} |")
    w("")
w("## 3. 提案の手法で選ばれた設定（評価するコーパスを含まない計算で選んだ）")
w("")
w("n0 は K 本で測った水準を等しい重みの側へ戻す強さ（ラベルの本数），β は相性を使うか（1 は使う），戻し先は eq（3 つの平均）か ext（ソースで見込んだ水準）．")
w("")
s = sel[sel.version == "M_F6_mixsel"].copy()
mx = json.load(open(_ROOT + "/results/dai18_probe/posthoc/provenance_selcounts.json"))["mix2_vs_mix3"]
assert mx["n_utterances_diff"] == 0 and set(s[s.chosen == "M_F6_mix2"].n_tied) == {2.0}
w(f"n0 = 0 の点は式の上で戻し先が効かないので，戻し先 eq の点（mix2）と ext の点（mix3）は同じ答えになる（保存した判定の配列 {mx['n_arrays']:,} 個で，2 つの点の判定はすべて同じ）．"
  f"n0 = 0・β = 1 を選んだ {int((s.chosen == 'M_F6_mix2').sum())} 組は，この 2 点が同点で，並びで先の eq になった．")
w("")
G = [(n0, b, p) for n0 in (0.0, 20.0, 80.0, float("inf")) for b in (0.0, 1.0) for p in ("eq", "ext")]
w("| コーパス | " + " | ".join(f"K = {K}" for K in KS) + " |")
w("|---|" + "---|" * len(KS))
for c in C6[:5]:
    cells = []
    for K in KS:
        q = s[(s.c == c) & (s.K == K)]
        if not len(q):
            cells.append("—")
            continue
        i = int(str(q.iloc[0].chosen).replace("M_F6_mix", ""))
        n0, b, p = G[i]
        cells.append(f"n0 = {'∞' if n0 == float('inf') else int(n0)}，β = {int(b)}，戻し先 {p}（{int(q.iloc[0].n_units)} 単位で選んだ）")
    w(f"| {c} | " + " | ".join(cells) + " |")
w("")
# 生徒
try:
    sl = pd.read_csv(f"{R}/students_levels.csv")
    se = pd.read_csv(f"{R}/students_effects.csv")
    SL = {"AVT": "融合（声・顔・言葉）", "AV": "融合（声・顔）", "A": "声", "V": "顔", "T": "言葉"}
    w("## 4. 疑似ラベルで学び直した分類器（生徒）")
    w("")
    w("提案の手法の答えを疑似ラベルにして学び直した分類器と，積を疑似ラベルにして学び直した分類器（今の提案）の比べ．どちらも設定を選び直した点．")
    w("")
    for K in KS:
        w(f"### K = {K}：UAR")
        w("")
        w("| 分類器 | 疑似ラベル | " + " | ".join(C6) + " |")
        w("|---|---|" + "---|" * len(C6))
        for st in ("AVT", "AV", "A", "V", "T"):
            for t, tl in (("none", "なし（K 本だけ）"), ("prod", "積（今の提案）"), ("M_F6_mixsel", "提案の手法")):
                x = sl[(sl.student == st) & (sl.teacher == t) & (sl.K == K)].set_index("c")
                if len(x):
                    w(f"| {SL[st]} | {tl} | " + " | ".join(f"{x.loc[c, 'uar']:.2f}" if c in x.index else "—" for c in C6) + " |")
        w("")
        w(f"### K = {K}：提案の手法の疑似ラベル − 積の疑似ラベル")
        w("")
        w("| 分類器 | " + " | ".join(C6) + " | ↑／↓ |")
        w("|---|" + "---|" * len(C6) + "---|")
        for role, kinds in (("融合", ("AVT", "AV")), ("声", ("A",)), ("顔", ("V",)), ("言葉", ("T",))):
            x = se[se.student.isin(kinds) & (se.a == "M_F6_mixsel") & (se.b == "prod") & (se.K == K)]
            if len(x):
                o, u, d, _ = row(x)
                w(f"| {role} | " + " | ".join(o) + f" | {u}／{d} |")
        w("")
except FileNotFoundError:
    pass
open(OUT, "w").write("\n".join(L) + "\n")
print(OUT, len(L))
