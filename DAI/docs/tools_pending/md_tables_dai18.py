"""実験 18（6 コーパス）の結果のうち，実験 16 の版（提案の手法とその比べ）だけを表に書き出す（結果ファイルを読むだけ．実験 17 の版は出さない）．
計算の後の点検（gate-code）が通ってから流す．出力：DAI/docs/実験18_下調べ_結果の表_2026-11-14.md"""
import pandas as pd

R = "/home/snics/MER-DA/results/dai18_probe/summary"
OUT = "/home/snics/MER-DA/DAI/docs/実験18_下調べ_結果の表_2026-11-14.md"
C6 = ["AFEW", "IEMOCAP", "MOSEI", "MELD", "CREMA-D", "RAVDESS"]
KS = (5, 10, 25)
VLAB = [("prod", "積（今の提案の推定時の答え）"), ("none_AVT", "K 本だけの融合（声・顔・言葉）"), ("none_AV", "K 本だけの融合（声・顔）"),
        ("none_A", "K 本だけの声"), ("none_V", "K 本だけの顔"), ("none_T", "K 本だけの言葉"),
        ("M_F6_mixsel", "提案の手法：本体の形・水準は K 本＋相性はソース"), ("M_F6_K", "本体の形・K 本だけ"), ("M_F6_ext", "本体の形・ソースだけ"),
        ("M_F6_orc", "本体の形・全ラベル（参照）"), ("M_F2_orc", "水準だけ・全ラベル（参照）"), ("M_F6n_orc", "相性を消した対照・全ラベル"),
        ("M_F6n_mixsel", "相性を消した対照・水準は K 本＋相性はソース")]
PAIRS = [("R1", "M_F6_orc", "prod", "本体の形・全ラベル − 積（参照）"), ("R2", "M_F6_mixsel", "prod", "提案の手法（水準は K 本＋相性はソース）− 積"),
         ("R2", "M_F6_K", "prod", "本体の形・K 本だけ − 積"), ("R2", "M_F6_ext", "prod", "本体の形・ソースだけ − 積"),
         ("比べ", "M_F2_orc", "prod", "水準だけ・全ラベル − 積"), ("比べ", "M_F6_orc", "M_F6n_orc", "モダリティ × 感情の分（全ラベル）"),
         ("比べ", "M_F6_mixsel", "M_F6n_mixsel", "モダリティ × 感情の分（水準は K 本＋相性はソース）")]


def cell(q):
    m = {"上回る": "↑", "下回る": "↓"}.get(q.verdict, "")
    return f"{q.est:+.2f} [{q.lo:+.1f}, {q.hi:+.1f}]{m}"


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
w("`results/dai18_probe/summary/` から書き出した（結果ファイルを読むだけ）．実験 17 の版（近いソースの発話を引く版）は，研究者の指示（2026-11-14）でこの表から外した．照合（gate-claim）の前の表である．")
w("")
w("- 差は UAR の点（%）の差，括弧は 95% 区間．↑ は「上回る」（区間の下端 > 0 かつ差 ≥ 1.0），↓ は「下回る」．")
w("- 判定：6 コーパスのうち上回るのが 2 以上で「当たり」（実験 16 の版は声しか無い RAVDESS に使えないので 5 コーパス）．元の 4 コーパスだけの数えも並べる．")
w("- **実験 16（4 コーパス）の再現ではない**：元の 4 コーパスの計算でも，ソースに CREMA-D と RAVDESS が入るので，出発点と微調整した分類器が変わる．")
w("- CREMA-D は声と顔，RAVDESS は声だけ．どちらも演技で決まった文を読むコーパスである．")
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
