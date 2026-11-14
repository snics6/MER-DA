"""DAI-16・17 の下調べの K = 5・10・25 の結果を 1 つの文書の表に書き出す（読むだけ）．"""
import re
import pandas as pd

R = "/home/snics/MER-DA/results"
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
OUT = "/home/snics/MER-DA/DAI/docs/実験16・17_下調べ_結果の表_K5・10・25_2026-11-14.md"
OLD16 = "/home/snics/MER-DA/DAI/docs/実験16_下調べ_結果の表_2026-11-14.md"


def both(d, f, **kw):
    a = pd.read_csv(f"{R}/{d}/summary/{f}", **kw)
    b = pd.read_csv(f"{R}/{d}_k10/summary/{f}", **kw)
    return pd.concat([a, b], ignore_index=True)


def cell(q):
    m = {"上回る": "↑", "下回る": "↓"}.get(q.verdict, "")
    return f"{q.est:+.2f} [{q.lo:+.1f}, {q.hi:+.1f}]{m}"


def cells(x):
    x = x.set_index("c")
    out, u, d = [], 0, 0
    for c in C:
        if c not in x.index:
            out.append("—")
            continue
        q = x.loc[c]
        out.append(cell(q))
        u += q.verdict == "上回る"
        d += q.verdict == "下回る"
    return out, u, d


def judge(rule, u, d):
    if rule in ("R1", "R2", "R3", "R4", "R5"):
        return "当たり" if u >= 2 else ("下回った" if d >= 2 else "外れ")
    return "—"


# DAI-16 の名前（既存の表から読む）
lab16 = {}
for line in open(OLD16):
    m = re.match(r"\| (R\d|R3b|比べ|対照|帰し方)? ?\|? ?(.+?)（`([^`]+?) − ([^`]+?)`）", line)
    if m:
        lab16.setdefault((m.group(3), m.group(4)), re.sub(r"（切片を全ラベルで決めた層）", "", m.group(2)).strip())
vlab16 = {}
for line in open(OLD16):
    m = re.match(r"\| (.+?)（`([^`− ]+)`）", line)
    if m:
        vlab16.setdefault(m.group(2), m.group(1).strip())

L = []
w = L.append
w("# 実験 16・17 の下調べの結果の表（K = 5・10・25．2026-11-14）")
w("")
w("出ている結果をすべて表にした．K = 5・25 は `results/dai16_probe/`・`results/dai17_probe/`，K = 10 は `results/dai16_probe_k10/`・`results/dai17_probe_k10/` の `summary/` から書き出した（台本 scratchpad の `all_tables.py`．結果ファイルを読むだけ）．")
w("照合済みの結果の文：`DAI/docs/実験16_下調べ_結果の文の下書き_2026-11-14.md`，`実験17_下調べ_結果の文の下書き_2026-11-14.md`，`実験16・17_下調べ_K10_結果の文の下書き_2026-11-14.md`．")
w("")
w("- 差は UAR の点（%）の差，括弧は 95% 区間．↑ は「上回る」（区間の下端 > 0 かつ差 ≥ 1.0），↓ は「下回る」（上端 < 0 かつ差 ≤ −1.0）．印の無いものは「言えない」．")
w("- 判定は規則の行だけ：上回るのが 4 コーパスのうち 2 以上で「当たり」，下回るのが 2 以上で「下回った」，ほかは「外れ」．規則でない行は「—」．")
w("- `prod` は今の提案（微調整した声・顔・言葉の確率の積）．`orc` の付く版は全ラベルを使う参照で，実際には使えない．")
w("- K = 10 の 40 本は K = 5 の 20 本を含み，テストも同じなので，K の間は独立な確かめではない．K どうしの差の区間は出していない．")
w("- 層：adj（ラベルを使わずに切片をずらす．主），ob（切片を全ラベルで決める．参照）．")
w("")

# ---------------- DAI-16
cr = both("dai16_probe", "criteria.csv")
ef = both("dai16_probe", "effects.csv")
lv = both("dai16_probe", "levels.csv")
w("## 1. 実験 16（感情ごとのモダリティの信頼度で答えをまとめる）")
w("")
for K in (5, 10, 25):
    w(f"### 1.{(5, 10, 25).index(K) + 1} 判定の規則と比べ（K = {K}）")
    w("")
    w("| 区分 | 比べ | 層 | AFEW | IEMOCAP | MOSEI | MELD | ↑／↓ | 判定 |")
    w("|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for r in cr[cr.K == K].itertuples():
        if r.where not in ("adj", "ob"):
            continue
        key = (r.rule, r.a, r.b, r.where)
        if key in seen:
            continue
        if r.where == "ob" and not (r.rule == "R1" or r.rule == "比べ" and r.b.endswith("n_orc")):
            continue
        seen.add(key)
        x = ef[(ef.a == r.a) & (ef.b == r.b) & (ef.K == K) & (ef["where"] == r.where)]
        if not len(x):
            continue
        o, u, d = cells(x)
        lab = lab16.get((r.a, r.b), "")
        w(f"| {r.rule} | {lab}（`{r.a} − {r.b}`） | {r.where} | " + " | ".join(o) + f" | {u}／{d} | {judge(r.rule, u, d) if r.where == "adj" else "—"} |")
    w("")
vers16 = [v for v in vlab16]
w("### 1.4 UAR の水準（adj の層．K 本だけの分類器・提案の融合の生徒・微調整した分類器は raw）")
w("")
w("| 版 | K | AFEW | IEMOCAP | MOSEI | MELD |")
w("|---|---|---|---|---|---|")
for v in vers16:
    for K in (5, 10, 25):
        x = lv[(lv.version == v) & (lv.K == K)].set_index("c")
        if not len(x):
            continue
        col = "uar_raw" if v.startswith(("none_", "ret_", "joint_")) else "uar_adj"
        vals = [f"{x.loc[c, col]:.2f}" if c in x.index and pd.notna(x.loc[c, col]) else "—" for c in C]
        w(f"| {vlab16[v]}（`{v}`） | {K} | " + " | ".join(vals) + " |")
w("")

# ---------------- DAI-17
cr = both("dai17_probe", "criteria.csv")
ef = both("dai17_probe", "effects.csv")
lv = both("dai17_probe", "levels.csv")
w("## 2. 実験 17（ターゲットの発話ごとに近いソースの発話を引く）")
w("")
for K in (5, 10, 25):
    w(f"### 2.{(5, 10, 25).index(K) + 1} 推定時にまとめた答え（A．K = {K}．adj の層）")
    w("")
    w("| 区分 | 比べ | AFEW | IEMOCAP | MOSEI | MELD | ↑／↓ | 判定 |")
    w("|---|---|---|---|---|---|---|---|")
    for r in cr[(cr.K == K) & (cr["where"] == "adj")].itertuples():
        x = ef[(ef.a == r.a) & (ef.b == r.b) & (ef.K == K) & (ef["where"] == "adj")]
        if not len(x):
            continue
        o, u, d = cells(x)
        w(f"| {r.rule} | {r.axis}（`{r.a} − {r.b}`） | " + " | ".join(o) + f" | {u}／{d} | {judge(r.rule, u, d)} |")
    w("")
main17 = ["prod", "MS_eq_sel", "MS_eq_tied", "MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel", "MN_F6orc_sel", "N_eq_sel", "S_eq",
          "MN_eq_selmod", "MN_eq_selfeat", "MN_eq_sellogit", "MN_eq_seljoint"]
w("### 2.4 推定時の答えの UAR の水準（adj の層．格子の点は選んだ点だけ）")
w("")
w("| 版 | K | AFEW | IEMOCAP | MOSEI | MELD |")
w("|---|---|---|---|---|---|")
for v in main17:
    for K in (5, 10, 25):
        x = lv[(lv.version == v) & (lv.K == K)].set_index("c")
        if not len(x):
            continue
        w(f"| `{v}` | {K} | " + " | ".join(f"{x.loc[c, 'uar_adj']:.2f}" if c in x.index else "—" for c in C) + " |")
w("")

# students
sc = both("dai17_probe", "students_criteria.csv")
se = both("dai17_probe", "students_effects.csv")
sl = both("dai17_probe", "students_levels.csv")
SL = {"AVT": "融合", "A": "声", "V": "顔", "T": "言葉"}
TL = {"MN_eq_sel_sel": "等しい重み", "MN_F2K_sel_sel": "水準は K 本", "MN_F6K_sel_sel": "感情ごとの信頼度は K 本", "MS_eq_sel_sel": "ソースの分類器を足す",
      "MS_eq_tied_sel": "ソースの分類器を足す（同じ倍率）", "prod_sel": "積", "prod": "積（実験 1 の設定の点）", "none": "K 本だけ",
      "MN_eq_sel": "等しい重み（実験 1 の設定の点）", "MN_F2K_sel": "水準は K 本（実験 1 の設定の点）", "MN_F6K_sel": "感情ごとの信頼度は K 本（実験 1 の設定の点）",
      "MS_eq_sel": "ソースの分類器を足す（実験 1 の設定の点）", "MS_eq_tied": "同じ倍率（実験 1 の設定の点）"}
for K in (5, 10, 25):
    w(f"### 2.{(5, 10, 25).index(K) + 5} 生徒の学び直し（C．K = {K}．差は疑似ラベル X の生徒 − 相手の生徒．「_sel」は設定を選び直した点）")
    w("")
    w("| 生徒 | 比べ | AFEW | IEMOCAP | MOSEI | MELD | ↑／↓ | 判定 |")
    w("|---|---|---|---|---|---|---|---|")
    for s in ("AVT", "A", "V", "T"):
        for (a, b), _ in se[(se.student == s) & (se.K == K)].groupby(["a", "b"], sort=False):
            x = se[(se.student == s) & (se.K == K) & (se.a == a) & (se.b == b)]
            o, u, d = cells(x)
            m = sc[(sc.K == K) & (sc.student == s) & (sc.a == a) & (sc.b == b)]
            isrule = len(m) and bool(m.main.iloc[0]) or (b == "prod_sel" and a in ("MN_eq_sel_sel", "MN_F2K_sel_sel", "MN_F6K_sel_sel"))
            w(f"| {SL[s]} | {TL.get(a, a)} − {TL.get(b, b)}（`{a} − {b}`） | " + " | ".join(o) + f" | {u}／{d} | {judge('R5', u, d) if isrule else '—'} |")
    w("")
w("### 2.8 生徒の UAR の水準（選び直した点）")
w("")
w("| 生徒 | 疑似ラベル | K | AFEW | IEMOCAP | MOSEI | MELD |")
w("|---|---|---|---|---|---|---|")
for s in ("AVT", "A", "V", "T"):
    for t in sl[sl.student == s].teacher.unique():
        for K in (5, 10, 25):
            x = sl[(sl.student == s) & (sl.teacher == t) & (sl.K == K)].set_index("c")
            if len(x):
                w(f"| {SL[s]} | `{t}` | {K} | " + " | ".join(f"{x.loc[c, 'uar']:.2f}" if c in x.index else "—" for c in C) + " |")
w("")

# reweight
rc = pd.read_csv(f"{R}/dai17_probe/summary/reweight_criteria.csv", dtype={"K": str})
rk = pd.read_csv(f"{R}/dai17_probe_k10/summary/reweight_criteria.csv", dtype={"K": str})
rc = pd.concat([rc, rk[rk.K == "10"]], ignore_index=True)
re_ = pd.read_csv(f"{R}/dai17_probe/summary/reweight_effects.csv", dtype={"K": str})
rek = pd.read_csv(f"{R}/dai17_probe_k10/summary/reweight_effects.csv", dtype={"K": str})
re_ = pd.concat([re_, rek[rek.K == "10"]], ignore_index=True)
ML = {"A": "声", "V": "顔", "T": "言葉", "POE": "3 つの積（POE）"}
AL = {"hub": "近いソースの発話を重く", "rel": "信頼度で重く", "hubrel": "両方", "base": "重みなし", "base_sel": "重みなし（強さを選び直す）",
      "hub_sel": "近いソースの発話を重く（強さを選び直す）", "rel_sel": "信頼度で重く（強さを選び直す）", "hubrel_sel": "両方（強さを選び直す）"}
w("### 2.9 ソースの発話の重みを変えて学び直す（B）")
w("")
w("K = 0 はターゲットのラベルを使わない回（0ob は切片を全ラベルで決めた参照）．R4 は K の回の `*_sel − base_sel`．")
w("")
w("| K | モダリティ | 比べ | AFEW | IEMOCAP | MOSEI | MELD | ↑／↓ | 判定 |")
w("|---|---|---|---|---|---|---|---|---|")
for K in ("0", "0ob", "5", "10", "25"):
    for r in rc[rc.K == K].itertuples():
        x = re_[(re_.K == K) & (re_.modality == r.modality) & (re_.a == r.a) & (re_.b == r.b)]
        if not len(x):
            continue
        o, u, d = cells(x)
        isrule = r.a.endswith("_sel") and r.b == "base_sel"
        w(f"| {K} | {ML[r.modality]} | {AL.get(r.a, r.a)} − {AL.get(r.b, r.b)}（`{r.a} − {r.b}`） | " + " | ".join(o) + f" | {u}／{d} | {judge('R4', u, d) if isrule else '—'} |")
w("")
open(OUT, "w").write("\n".join(L) + "\n")
print(OUT, len(L))
