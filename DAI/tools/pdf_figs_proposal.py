"""説明の PDF（docs/説明/説明_実験16の案_図解_2026-11-13.tex）の結果の節の図・表・文を，結果ファイルから書き出す（読むだけ）．
下調べ（4 コーパス．実験 16 と K = 10 の追補．照合済み）と，6 コーパスの下調べ（実験 18）の，提案の手法に関わる行だけを使う．
使い方：.venv/bin/python tools/pdf_figs_proposal.py
出力：docs/説明/説明_提案_図表_下調べ.tex，説明_提案_図表_6コーパス.tex，説明_提案_図表_まとめ.tex
（評価の回の計算の後に exp へ移す．計算中は exp にファイルを足さない）"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "説明"
P4 = {5: ROOT / "results/dai16_probe/summary", 25: ROOT / "results/dai16_probe/summary", 10: ROOT / "results/dai16_probe_k10/summary"}
P6 = ROOT / "results/dai18_probe/summary"
KS = (5, 10, 25)
C4 = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
C5 = C4 + ["CREMA-D"]
SERIES = [("M_F6_mixsel", "cBar1", "提案（水準は K 本＋相性はソース）"), ("M_F6_K", "cBar2", "K 本だけで測った表"), ("M_F6_ext", "cBar3", "ソースだけで測った表")]


def sg(v: float, nd: int = 2) -> str:
    return f"${'+' if v >= 0 else '-'}{abs(v):.{nd}f}$"


def cell(q) -> str:
    s = f"{sg(q.est)} {{\\scriptsize[{sg(q.lo, 1)}, {sg(q.hi, 1)}]}}"
    if q.verdict == "上回る":
        return f"\\textbf{{{sg(q.est)}}} {{\\scriptsize[{sg(q.lo, 1)}, {sg(q.hi, 1)}]}} $\\uparrow$"
    if q.verdict == "下回る":
        return s + " $\\downarrow$"
    return s


def rng(vals) -> str:
    vals = list(vals)
    return f"${min(vals):+.2f}$〜${max(vals):+.2f}$"


def one(q) -> str:
    return f"${q.est:+.2f}$ 点，区間 ${q.lo:+.1f}$〜${q.hi:+.1f}$"


def load4():
    ef = pd.concat([pd.read_csv(P4[K] / "effects.csv").query("K == @K") for K in KS], ignore_index=True)
    lv = pd.concat([pd.read_csv(P4[K] / "levels.csv").query("K == @K") for K in KS], ignore_index=True)
    return ef[ef["where"] == "adj"], lv


def load6():
    ef = pd.read_csv(P6 / "effects.csv")
    return ef[ef["where"] == "adj"], pd.read_csv(P6 / "levels.csv"), pd.read_csv(P6 / "students_effects.csv"), pd.read_csv(P6 / "students_levels.csv")


def get(ef, c, K, a, b="prod"):
    q = ef[(ef.c == c) & (ef.K == K) & (ef.a == a) & (ef.b == b)]
    return q.iloc[0] if len(q) else None


def dotplot(panels, corpora, series, ymin, ymax, yu=0.42, gw=1.10) -> str:
    """panels：[(見出し, {(コーパス, 版): 行})]．点は差，縦の線は 95% 区間．"""
    W = gw * len(corpora) + 0.45
    o = [f"\\begin{{tikzpicture}}[x=1cm,y={yu}cm]"]
    for p, (title, data) in enumerate(panels):
        x0 = p * (W + 0.95)
        for t in range(ymin, ymax + 1):
            if t % 2 == 0:
                o.append(f"\\draw[black!12] ({x0:.2f},{t}) -- ({x0 + W:.2f},{t});")
                lab = "0" if t == 0 else (f"$+{t}$" if t > 0 else f"$-{-t}$")
                o.append(f"\\node[font=\\small,anchor=east] at ({x0 - 0.05:.2f},{t}) {{{lab}}};")
        o.append(f"\\draw[black!45,dashed] ({x0:.2f},1) -- ({x0 + W:.2f},1);")
        o.append(f"\\draw[black,line width=0.8pt] ({x0:.2f},0) -- ({x0 + W:.2f},0);")
        o.append(f"\\draw ({x0:.2f},{ymin}) -- ({x0:.2f},{ymax});")
        for i, c in enumerate(corpora):
            gx = x0 + 0.62 + gw * i
            for j, (key, col, _) in enumerate(series):
                q = data.get((c, key))
                if q is None:
                    continue
                x = gx + (j - (len(series) - 1) / 2) * 0.27
                lo, hi = max(q.lo, ymin), min(q.hi, ymax)
                o.append(f"\\draw[{col},line width=1.3pt] ({x:.2f},{lo:.2f}) -- ({x:.2f},{hi:.2f});")
                o.append(f"\\fill[{col}] ({x:.2f},{q.est:.2f}) circle (2.1pt);")
            o.append(f"\\node[font=\\small,rotate=35,anchor=north east,inner sep=1pt] at ({gx + 0.22:.2f},{ymin - 0.35:.2f}) {{{c}}};")
        o.append(f"\\node[font=\\small\\bfseries] at ({x0 + W / 2:.2f},{ymax + 1.1:.2f}) {{{title}}};")
    o.append(f"\\node[font=\\small,rotate=90] at (-0.95,{(ymin + ymax) / 2:.2f}) {{積との差（UAR の点）}};")
    tot = len(panels) * (W + 0.95) - 0.95
    lx = 0.0
    yl = ymin - 5.0
    for key, col, name in series:
        o.append(f"\\fill[{col}] ({lx + 0.12:.2f},{yl:.2f}) circle (2.4pt);")
        o.append(f"\\draw[{col},line width=1.3pt] ({lx + 0.12:.2f},{yl - 0.8:.2f}) -- ({lx + 0.12:.2f},{yl + 0.8:.2f});")
        o.append(f"\\node[font=\\small,anchor=west] at ({lx + 0.3:.2f},{yl:.2f}) {{{name}}};")
        lx += 0.36 * len(name) + 0.9                                         # 名前の長さに合わせて次の項目を置く
    o.append(f"\\node[font=\\small,anchor=west] at (0,{yl - 2.2:.2f}) {{点は差，縦の線は 95\\% の区間．太い横線は差 0，点線は $+1$ 点（基準）．}};")
    o.append("\\end{tikzpicture}")
    return "\n".join(o)


def diff_table(ef, corpora, rows, caption) -> str:
    """rows：[(ラベル, a, b)]．比べごとに 1 つの塊，K ごとに 1 行．"""
    o = ["\\begin{table}[H]\\centering\\small", f"\\caption{{{caption}}}", "\\setlength{\\tabcolsep}{5pt}",
         "\\begin{adjustbox}{max width=\\linewidth}", "\\begin{tabular}{l" + "l" * len(corpora) + "}", "\\toprule", "K & " + " & ".join(corpora) + " \\\\"]
    for lab, a, b in rows:
        o += ["\\midrule", f"\\multicolumn{{{len(corpora) + 1}}}{{l}}{{\\textbf{{{lab}}}}} \\\\"]
        for K in KS:
            cs = [(cell(q) if (q := get(ef, c, K, a, b)) is not None else "—") for c in corpora]
            o.append(f"{K} & " + " & ".join(cs) + " \\\\")
    o += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"]
    return "\n".join(o)


def level_table(lv, corpora, rows, caption) -> str:
    o = ["\\begin{table}[H]\\centering\\small", f"\\caption{{{caption}}}", "\\begin{tabular}{ll" + "c" * len(corpora) + "}", "\\toprule",
         "K & 答えの作り方 & " + " & ".join(corpora) + " \\\\"]
    for K in KS:
        o.append("\\midrule")
        for n, (lab, vers) in enumerate(rows):
            cs = []
            for c in corpora:
                q = lv[(lv.c == c) & (lv.K == K) & lv.version.isin(vers)]
                v = q.iloc[0].uar_adj if len(q) and pd.notna(q.iloc[0].uar_adj) else (q.iloc[0].uar_raw if len(q) else None)
                cs.append(f"{v:.1f}" if v is not None else "—")
            o.append((f"{K}" if n == 0 else "") + f" & {lab} & " + " & ".join(cs) + " \\\\")
    o += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(o)


LEVEL_ROWS = [("積", ["prod"]), ("提案", ["M_F6_mixsel"]), ("全ラベルで測った表（参照）", ["M_F6_orc"]), ("K 本だけで学んだ融合の分類器", ["none_AVT", "none_AV"])]


def panels_of(ef, corpora):
    return [(f"K = {K}", {(c, key): q for c in corpora for key, _, _ in SERIES if (q := get(ef, c, K, key)) is not None}) for K in KS]


def main() -> None:
    e4, l4 = load4()
    e6, l6, se6, sl6 = load6()
    # ---------------- 9 節：下調べ（4 コーパス）
    t = ["\\begin{figure}[H]\\centering", "\\begin{adjustbox}{max width=\\linewidth}", dotplot(panels_of(e4, C4), C4, SERIES, -4, 6), "\\end{adjustbox}",
         "\\caption{下調べ（4 コーパス）．表の測り方 3 通りの，積との差．}", "\\end{figure}", "",
         diff_table(e4, C4, [("提案 − 積", "M_F6_mixsel", "prod"), ("全ラベルで測った表 − 積（参照）", "M_F6_orc", "prod")],
                    "下調べ（4 コーパス）：積との差．括弧は 95\\% の区間．太字と $\\uparrow$ は基準（区間の下端が 0 より上で，差が 1 点以上）を満たした条件．"), "",
         level_table(l4, C4, LEVEL_ROWS, "下調べ（4 コーパス）：UAR（点）．")]
    (OUT / "説明_提案_図表_下調べ.tex").write_text("\n".join(t) + "\n")
    # ---------------- 10 節：6 コーパス
    fus = se6[se6.student.isin(["AVT", "AV"]) & (se6.a == "M_F6_mixsel") & (se6.b == "prod")]
    fpan = [(f"K = {K}", {(c, "fus"): fus[(fus.c == c) & (fus.K == K)].iloc[0] for c in C5 if len(fus[(fus.c == c) & (fus.K == K)])}) for K in KS]
    fser = [("fus", "cBar1", "提案の答えを疑似ラベルにして学び直した融合の分類器 − 積を疑似ラベルにした融合の分類器")]
    rows = ["\\begin{table}[H]\\centering\\small",
            "\\caption{6 コーパス：学び直した融合の分類器の差（提案の答えを疑似ラベルにした分類器 − 積を疑似ラベルにした分類器）．}",
            "\\setlength{\\tabcolsep}{5pt}", "\\begin{adjustbox}{max width=\\linewidth}", "\\begin{tabular}{l" + "l" * len(C5) + "}", "\\toprule",
            "K & " + " & ".join(C5) + " \\\\", "\\midrule"]
    for K in KS:
        rows.append(f"{K} & " + " & ".join(cell(fus[(fus.c == c) & (fus.K == K)].iloc[0]) for c in C5) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"]
    t = ["\\begin{figure}[H]\\centering", "\\begin{adjustbox}{max width=\\linewidth}", dotplot(panels_of(e6, C5), C5, SERIES, -4, 6, gw=0.92), "\\end{adjustbox}",
         "\\caption{6 コーパス．表の測り方 3 通りの，積との差．RAVDESS は声しか無いので，提案を使えない．}", "\\end{figure}", "",
         diff_table(e6, C5, [("提案 − 積", "M_F6_mixsel", "prod"), ("全ラベルで測った表 − 積（参照）", "M_F6_orc", "prod")],
                    "6 コーパス：積との差．印と括弧の意味は 9 節の表と同じ．"), "",
         level_table(l6, C5, LEVEL_ROWS, "6 コーパス：UAR（点）．CREMA-D の融合は声と顔の 2 つ．"), "", "\n".join(rows)]
    (OUT / "説明_提案_図表_6コーパス.tex").write_text("\n".join(t) + "\n")
    # ---------------- 11 節：分かったこと（数は結果ファイルから）
    def col(ef, cs, a, b="prod"):
        return [q for c in cs for K in KS if (q := get(ef, c, K, a, b)) is not None]

    b = []
    a4, a6 = col(e4, ["AFEW"], "M_F6_mixsel"), col(e6, ["AFEW"], "M_F6_mixsel")
    o4, o6 = col(e4, C4[1:], "M_F6_mixsel"), col(e6, C5[1:], "M_F6_mixsel")
    b.append(f"\\item \\textbf{{提案は AFEW で積を上回った．}}差は，4 コーパスの下調べで {rng(q.est for q in a4)} 点，6 コーパスで {rng(q.est for q in a6)} 点で，"
             f"基準を満たしたのは，4 コーパスの下調べで 3 つの K のうち {sum(q.verdict == '上回る' for q in a4)} つ，6 コーパスで {sum(q.verdict == '上回る' for q in a6)} つだった．")
    up4 = [q for q in o4 if q.verdict == "上回る"]
    b.append(f"\\item \\textbf{{ほかのコーパスでは差が小さい．}}AFEW 以外の差は，4 コーパスの下調べで {rng(q.est for q in o4)} 点（{len(o4)} 条件），6 コーパスで {rng(q.est for q in o6)} 点（{len(o6)} 条件）だった．"
             f"差が負だった条件は {sum(q.est < 0 for q in o4 + o6)} 個である．基準を満たしたのは，4 コーパスの下調べの {'，'.join(f'{q.c} の K = {q.K}' for q in up4) if up4 else '0 条件'}"
             f"{' だけ' if up4 else ''}で，6 コーパスでは {sum(q.verdict == '上回る' for q in o6)} 条件だった．")
    cd = col(e6, ["CREMA-D"], "M_F6_mixsel")
    b.append(f"\\item CREMA-D（声と顔）の差は {rng(q.est for q in cd)} 点で，3 つの K とも区間は 0 より上だった（{sum(q.lo > 0 for q in cd)} 条件）．差が 1 点に届かないので，基準は満たさない．")
    kd = [(nm, q) for nm, qs in (("4 コーパスの下調べ", col(e4, C4, "M_F6_K")), ("6 コーパス", col(e6, C5, "M_F6_K"))) for q in qs if q.hi < 0]
    sd4, sd6 = col(e4, ["MELD"], "M_F6_ext"), col(e6, ["MELD"], "M_F6_ext")
    b.append("\\item \\textbf{水準を K 本で測り，相性をソースから持ち込む組み合わせは，片方だけの版の損を避けた．}"
             f"ソースだけで測った表は，MELD で積を下回った（4 コーパスの下調べで {rng(q.est for q in sd4)} 点，6 コーパスで {rng(q.est for q in sd6)} 点．"
             f"基準を満たして下回ったのは {sum(q.verdict == '下回る' for q in sd4 + sd6)} 条件）．"
             + (f"K 本だけで測った表は，{'，'.join(f'{nm}の {q.c} の K = {q.K}（{one(q)}）' for nm, q in kd)}で区間が 0 より下だった．" if kd else "")
             + f"同じ条件での提案の差は {rng([q.est for q in col(e4, ['MELD'], 'M_F6_mixsel') + col(e6, ['MELD'], 'M_F6_mixsel')] + [get(e4 if nm.startswith('4') else e6, q.c, q.K, 'M_F6_mixsel').est for nm, q in kd])} 点で，負の値は無かった．")
    i4, i6 = col(e4, C4, "M_F6_mixsel", "M_F6n_mixsel"), col(e6, C5, "M_F6_mixsel", "M_F6n_mixsel")
    b.append(f"\\item \\textbf{{「どの感情でどのモダリティが強いか」の成分の寄与は小さい．}}提案からこの成分だけを消した版との差は，4 コーパスの下調べで {rng(q.est for q in i4)} 点，"
             f"6 コーパスで {rng(q.est for q in i6)} 点で，基準を満たした条件は {sum(q.verdict == '上回る' for q in i4 + i6)} 個だった．提案と積の差は，主にモダリティごとの重み（水準）から来ている．")
    r4a, r6a = col(e4, ["AFEW"], "M_F6_orc"), col(e6, ["AFEW"], "M_F6_orc")
    r4o, r6o = col(e4, C4[1:], "M_F6_orc"), col(e6, C5[1:], "M_F6_orc")
    b.append(f"\\item \\textbf{{表を全ラベルで測れたとしたときの余地}}（実際には使えない参照）：積との差は，AFEW で {rng(q.est for q in r4a + r6a)} 点，ほかのコーパスで {rng(q.est for q in r4o + r6o)} 点だった．"
             f"AFEW 以外では，表を全ラベルで測っても，積との差は {max(q.est for q in r4o + r6o):.1f} 点以下だった．")
    fa = [fus[(fus.c == "AFEW") & (fus.K == K)].iloc[0] for K in KS]
    fo = [fus[(fus.c == c) & (fus.K == K)].iloc[0] for c in C5[1:] for K in KS]
    fup = [q for q in fo if q.verdict == "上回る"]
    b.append(f"\\item \\textbf{{提案の答えを疑似ラベルにして融合の分類器を学び直すと，差は AFEW で残った}}（6 コーパス）．積を疑似ラベルにした分類器との差は，AFEW で {rng(q.est for q in fa)} 点（3 つの K とも基準を満たす），"
             f"ほかのコーパスで {rng(q.est for q in fo)} 点だった"
             + (f"（基準を満たしたのは {'，'.join(f'{q.c} の K = {q.K}' for q in fup)}）．" if fup else "．"))
    (OUT / "説明_提案_図表_まとめ.tex").write_text("\\begin{itemize}\n" + "\n".join(b) + "\n\\end{itemize}\n")
    print("\n".join(b))


if __name__ == "__main__":
    main()
