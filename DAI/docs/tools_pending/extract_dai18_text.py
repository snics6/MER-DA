"""実験 18 の下調べ（DAI-18．6 コーパス）の結果の文の下書きを，結果ファイルから数値を取り出して書き出す．

出力：DAI/docs/実験18_下調べ_結果の文の下書き_2026-11-14.md（--out で別の場所に書ける）．
読むもの：results/dai18_probe/summary/*.csv・*.json，provenance.json，provenance_log.jsonl，tether.json，frozen_sel.json，
          posthoc/voice_student_prod40.csv（`dai18_posthoc_voice40.py` が書く），logs/dai18_probe.done，
          DAI/checks/DAI-18/code_post.md と docs/seen-data.md（引いた文がそこにあることを確かめるだけ）．
結果ファイルは読むだけで，書き換えない．GPU は使わない．DAI/exp の台本は定数（格子の定義）を読むために読み込むだけ．

文の中の数値と表は，すべてこの台本が結果ファイルから埋める（手で写さない）．
文が言う当たり・外れや数えは `must(...)` で結果ファイルと照らし，合わなければ止まる．
文を直すときは，この台本の中の文を直して流し直す．

使い方：.venv/bin/python DAI/docs/tools_pending/extract_dai18_text.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                    # GPU を使わない
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True                                            # DAI/exp に __pycache__ を書かない

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai18_probe as P18  # noqa: E402   格子の定義（GRID16，SGRID，LAMS，cfgs_of）を読むだけ

R = ROOT / "results" / "dai18_probe"
SM = R / "summary"
OUT = ROOT / "DAI" / "docs" / "実験18_下調べ_結果の文の下書き_2026-11-14.md"
TABLE_DOC = ROOT / "DAI" / "docs" / "実験18_下調べ_結果の表_2026-11-14.md"
DESIGN = ROOT / "DAI" / "docs" / "実験18_下調べ_設計.md"
CODE_POST = ROOT / "DAI" / "checks" / "DAI-18" / "code_post.md"
LEDGER = ROOT / "docs" / "seen-data.md"
C6 = list(P18.CORP6)
ORIG4, NEW2 = C6[:4], C6[4:]
KS = list(P18.KS)
ROLES = ["fusion", "A", "V", "T"]
ROLE_JP = {"fusion": "融合", "A": "声", "V": "顔", "T": "言葉"}
R5_TEACH16, R5_TEACH17 = ["M_F6_mixsel"], ["MN_F2K_sel", "MN_F6K_sel", "MN_eq_sel"]


def must(cond, msg: str) -> None:
    if not cond:
        raise SystemExit(f"文と結果ファイルが合わない：{msg}")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mtime(p: Path) -> str:
    return time.strftime("%H:%M:%S", time.localtime(p.stat().st_mtime))


# ------------------------------------------------------------------ 読み込み

ef = pd.read_csv(SM / "effects.csv")
cr = pd.read_csv(SM / "criteria.csv").fillna({"up": "", "down": "", "new": ""})
lv = pd.read_csv(SM / "levels.csv")
sel = pd.read_csv(SM / "sel.csv")
se = pd.read_csv(SM / "students_effects.csv")
se["role"] = se.student.map(lambda k: "fusion" if len(k) >= 2 else k)
scr = pd.read_csv(SM / "students_criteria.csv").fillna({"up": "", "down": "", "new": ""})
ssel = pd.read_csv(SM / "students_sel.csv")
scond = pd.read_csv(SM / "students_conditions.csv")
conv = json.load(open(SM / "convergence.json"))
chk = json.load(open(SM / "students_checks.json"))
prov = json.load(open(R / "provenance.json"))
plog = [json.loads(x) for x in open(R / "provenance_log.jsonl")]
teth = json.load(open(R / "tether.json"))
fsel = json.load(open(R / "frozen_sel.json"))
ph = pd.read_csv(R / "posthoc" / "voice_student_prod40.csv")
phprov = json.load(open(R / "posthoc" / "provenance_voice40.json"))
done = open(ROOT / "logs" / "dai18_probe.done").read().strip()
code_post = CODE_POST.read_text()
ledger = LEDGER.read_text()

# ------------------------------------------------------------------ 数の書き方


def fest(x: float) -> str:
    """差：小数 2 桁．丸めると ±1.00 になって 1.0 点の境目のどちら側か読めないときは桁を足す．"""
    s = f"{x:+.2f}"
    if abs(float(s)) == 1.0 and abs(x) != 1.0:
        for nd in (3, 4, 5):
            s = f"{x:+.{nd}f}"
            if abs(float(s)) != 1.0:
                break
    return s.replace("-", "−")


def fend(x: float) -> str:
    """区間の端：小数 1 桁．0.05 より 0 に近いときは符号が読めるよう桁を足す．"""
    s = f"{x:+.1f}"
    if abs(x) < 0.05 and x != 0:
        for nd in (2, 3, 4):
            s = f"{x:+.{nd}f}"
            if float(s) != 0.0:
                break
    return s.replace("-", "−")


def ci(q, mark: bool = False) -> str:
    m = {"上回る": "↑", "下回る": "↓"}.get(q.verdict, "") if mark else ""
    return f"{fest(q.est)} [{fend(q.lo)}, {fend(q.hi)}]{m}"


def sedf(q) -> str:
    return f"{q.se:.2f}（{int(q.df)}）" if pd.notna(q.df) else f"{q.se:.2f}（引き直し）"


def sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    sup = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m}×10{sup}"


def jp(names) -> str:
    if isinstance(names, str):
        names = [x for x in names.split(",") if x]
    return "・".join(names)


# ------------------------------------------------------------------ 取り出し

def eq(a, b, c, K, where="adj"):
    x = ef[(ef.a == a) & (ef.b == b) & (ef.c == c) & (ef.K == K) & (ef["where"] == where)]
    must(len(x) == 1, f"effects.csv に {a}−{b} {c} K={K} {where} が {len(x)} 行")
    return x.iloc[0]


def E(a, b, c, K) -> str:
    return ci(eq(a, b, c, K))


def E3(a, b, c) -> str:
    return "，".join(f"K = {K} で {E(a, b, c, K)}" for K in KS)


def sq(role, a, b, c, K):
    x = se[(se.role == role) & (se.a == a) & (se.b == b) & (se.c == c) & (se.K == K)]
    must(len(x) == 1, f"students_effects.csv に {role} {a}−{b} {c} K={K} が {len(x)} 行")
    return x.iloc[0]


def SQ(role, a, b, c, K) -> str:
    return ci(sq(role, a, b, c, K))


def SQ3(role, a, b, c) -> str:
    return "，".join(f"K = {K} で {SQ(role, a, b, c, K)}" for K in KS)


def crow(a, b, K):
    x = cr[(cr.a == a) & (cr.b == b) & (cr.K == K)]
    must(len(x) == 1, f"criteria.csv に {a}−{b} K={K} が {len(x)} 行")
    return x.iloc[0]


def srow(role, a, b, K):
    x = scr[(scr.student == role) & (scr.a == a) & (scr.b == b) & (scr.K == K)]
    must(len(x) == 1, f"students_criteria.csv に {role} {a}−{b} K={K} が {len(x)} 行")
    return x.iloc[0]


def level(c, K, v, col="uar_adj") -> float:
    x = lv[(lv.c == c) & (lv.K == K) & (lv.version == v)]
    must(len(x) == 1, f"levels.csv に {c} K={K} {v} が {len(x)} 行")
    return float(x.iloc[0][col])


def cond(c, K, student, teacher):
    x = scond[(scond.c == c) & (scond.K == K) & (scond.student == student) & (scond.teacher == teacher)]
    must(len(x) == 1, f"students_conditions.csv に {c} K={K} {student} {teacher} が {len(x)} 行")
    return x.iloc[0]


def verdicts(a, b, K, cs, role=None):
    return [(sq(role, a, b, c, K) if role else eq(a, b, c, K)).verdict for c in cs]


def judge(r) -> str:
    return "当たり" if bool(r.hit) else ("下回った" if bool(r.down2) else "外れ")


def ov(r, rule: bool = True) -> str:
    """一覧の表の 1 条件：判定，上回ったコーパスの数／分母，元の 4 コーパスの数え，名前．"""
    n, u, d, u4 = int(r.n_corpora), int(r.n_up), int(r.n_down), int(r.n_up_orig4)
    s = (f"{judge(r)}．" if rule else "") + f"↑ {u}／{n}" + (f"：{jp(r.up)}" if u else "") + f"（元の 4 コーパスの↑ {u4}）"
    if d:
        s += f"．↓ {d}：{jp(r.down)}"
    return s


def trow(x: pd.DataFrame):
    x = x.set_index("c")
    cells, u, d, u4 = [], 0, 0, 0
    for c in C6:
        if c not in x.index:
            cells.append("—")
            continue
        q = x.loc[c]
        cells.append(ci(q, mark=True))
        u += q.verdict == "上回る"
        d += q.verdict == "下回る"
        u4 += (q.verdict == "上回る") and c in ORIG4
    return cells, int(u), int(d), int(u4)


RULES_A = {(r.a, r.b): r.rule for r in cr.itertuples()}
R5_PAIRS = {(t, "prod") for t in R5_TEACH16 + R5_TEACH17}


def eff_line(kind: str, lab: str, a: str, b: str, K: int, role: str | None = None) -> str:
    if role is None:
        x = ef[(ef.a == a) & (ef.b == b) & (ef.K == K) & (ef["where"] == "adj")]
        r = crow(a, b, K) if RULES_A.get((a, b)) in ("R1", "R2") else None
    else:
        x = se[(se.role == role) & (se.a == a) & (se.b == b) & (se.K == K)]
        r = srow(role, a, b, K) if (a, b) in R5_PAIRS else None
    must(len(x) > 0, f"差の行が無い：{role} {a}−{b} K={K}")
    cells, u, d, u4 = trow(x)
    if r is not None:
        must((u, d, u4) == (int(r.n_up), int(r.n_down), int(r.n_up_orig4)), f"数えが criteria と違う：{role} {a}−{b} K={K}")
    return f"| {kind} | {lab} | {K} | " + " | ".join(cells) + f" | {u}／{d} | {u4} | {judge(r) if r is not None else '—'} |"


HEAD = "| 区分 | 比べ | K | " + " | ".join(C6) + " | ↑／↓ | 元の 4 コーパスの↑ | 判定 |"
HSEP = "|---|---|---|" + "---|" * len(C6) + "---|---|---|"

# ------------------------------------------------------------------ 選ばれた点の読み方


def mix_desc(chosen: str) -> str:
    i = int(chosen.replace("M_F6_mix", ""))
    n0, b, p = P18.GRID16[i]
    return f"mix{i}（n0 {'∞' if np.isinf(n0) else int(n0)}，β {int(b)}，{p}）"


def lam_desc(j: int) -> str:
    v = P18.LAMS[j]
    return f"λ {v:g}" + ("（下端）" if j == 0 else ("（上端）" if j == len(P18.LAMS) - 1 else ""))


def cfg_desc(c: str, i: int) -> str:
    scope, k, rel = P18.cfgs_of(c)[i]
    return f"{scope} {k} {rel}"


def sel_desc(c: str, chosen: str) -> str:
    p = chosen.split("_")
    if chosen.startswith("M_F6_mix"):
        return mix_desc(chosen)
    if chosen.startswith("MS_eq_l"):
        return lam_desc(int(p[-1][1:]))
    if chosen.startswith("N_eq_c"):
        return cfg_desc(c, int(p[-1][1:]))
    return f"{cfg_desc(c, int(p[-2][1:]))} {lam_desc(int(p[-1][1:]))}"


def chosen_of(c, K, version):
    x = sel[(sel.c == c) & (sel.K == K) & (sel.version == version)]
    must(len(x) == 1, f"sel.csv に {c} K={K} {version} が {len(x)} 行")
    return x.iloc[0]


def st_point(c, K, student, teacher) -> str:
    x = ssel[(ssel.c == c) & (ssel.K == K) & (ssel.student == student) & (ssel.teacher == teacher)]
    must(len(x) == 1, f"students_sel.csv に {c} K={K} {student} {teacher} が {len(x)} 行")
    return f"{x.iloc[0].rho:g}・{int(x.iloc[0].a)}"


# ------------------------------------------------------------------ 文と結果ファイルの照らし合わせ（文を書く前に確かめる）

# 出どころ
must(sha(DESIGN) == prov["design_sha256"], "設計の sha256 が provenance と違う")
must(all(p["design_sha256"] == prov["design_sha256"] for p in plog), "設計の sha256 が段で違う")
must(len({p["this_script_sha256"] for p in plog}) == 1 and plog[0]["this_script_sha256"] == prov["this_script_sha256"], "台本の sha256 が段で違う")
must(sha(ROOT / "DAI" / "exp" / "dai18_probe.py") == prov["this_script_sha256"], "今の台本の sha256 が provenance と違う")
must([p["stage"] for p in plog] == ["source", "base", "tether", "units", "students", "summary"], "段の並び")
must(all(p["dirty"] is False for p in plog), "dirty")
COMMITS = {p["stage"]: p["commit"][:7] for p in plog}
must(len({COMMITS[s] for s in ("source", "base", "tether", "units", "students")}) == 1, "5 段のコミットが 1 つでない")
TIMES = {p["stage"]: p["time"][11:] for p in plog}
must(fsel["commit"][:7] == COMMITS["students"] and teth["commit"][:7] == COMMITS["tether"], "frozen_sel / tether のコミット")
must({f"{r.c}|{r.K}|{r.version}": r.chosen for r in sel.itertuples()} == fsel["choice"], "sel.csv と frozen_sel.json の選んだ点")
for s in ["判定\n\n通す．設計の格子の 300 単位が base・units・students の 3 段ですべて csv と npz の対でそろい", "指摘なし．",
          "3,068 個の配列のうち 3,067 個がビット単位で一致し，残る 1 個も一致 0.99808", "720 組すべてで実験 1 の段 1 の保存と完全に一致した",
          "156 行の `chosen` は，鍵の集合も値もすべて同じ", "書いた 13 ファイルはすべて `results/dai18_probe/summary/` とバイト単位で同じだった",
          "`n_units` は 18 組すべてで一致した", "`pseudo_targets` も 18 組すべてで一致した", "検査したコミット 02f356e",
          "`DAI/run/dai19_arm.sh`・`DAI/run/dai19_post.sh` の追加だけ", "勾配の最大：base 1.0×10⁻⁶", "止まり（`stalled`）は base・students とも合計 0",
          "base 2,301 MB，units 8,117 MB，students 2,417 MB", "20:26:29 より後に更新されたファイルは 0",
          "units の 300 単位すべてに，種類ごとのソースの一覧（`srcs_*`．T と c を含まない），`sid_test`，`nn_test_*` が保存されている"]:
    must(s in code_post, f"点検の報告に次の文が無い：{s}")
must(code_post.count("### [記録]") == 3, "点検の報告の記録の数")
design = DESIGN.read_text()
for s_ in ["声だけで 40.9%，顔だけで 58.2%，声と顔で 63.6%", "CREMA-D 12 文，RAVDESS 2 文", "この文の発話のうち Neutral は 10.0% で，ほかの 11 文の 25.0% より低い",
           "Neutral：ほかの 3 感情 ＝ 1：2", "calm は Neutral に混ぜていない", "CREMA-D の顔の特徴は 91.3% の発話で 1 区間の平均", "顔が検出されない 1 発話を除いた",
           "声のソースは 2 つから 4 つになり，うち 2 つが演技のコーパス．顔は 2 つから 3 つで，うち 1 つが演技", "K 3 つで合わせて約 70", "約 0.01", "約 0.6",
           "自由度は台本の決まりで 18．真の 23 より控えめ", "選ぶ単位の 1/3〜1/5 の重みで入る", "E12・E13・E77・E78", "主の K は 5", "第 3 版．計算の前"]:
    must(s_ in design, f"設計に次の文が無い：{s_}")
CK = ROOT / "DAI" / "checks" / "DAI-18"
must("## 判定\n\n通す．" in (CK / "probe_v3.md").read_text() and "## 判定\n\n通す．" in (CK / "code_v3.md").read_text(), "probe_v3 / code_v3 の判定")
for f_ in ("review_ml_v2.md", "review_affect_v2.md"):
    t_ = (CK / f_).read_text()
    must("判定：OK" in t_ and "87b1dc7" in t_, f"{f_} の判定と検査した版")
must("**実験 18 の集計の値を開いたのは 2026-11-14 20:39**" in ledger, "台帳に結果を開いた時刻の文が無い")
must(abs(round(conv["gpu_max_mb"]) - 8117) == 0, "GPU の最大が点検の報告と違う")

# 区間の作り方（adj の層と生徒の段）
adj = ef[ef["where"] == "adj"]
for d_ in (adj, se):
    must(set(d_[d_.c.isin(["AFEW", "IEMOCAP", "MELD", "RAVDESS"])].method) == {"jk+seed"}, "jackknife のコーパス")
    must(set(d_[d_.c.isin(["MOSEI", "CREMA-D"])].method) == {"boot_cl+seed"}, "引き直しのコーパス")
    must({c: set(d_[d_.c == c].df.dropna().astype(int)) for c in ("AFEW", "IEMOCAP", "MELD", "RAVDESS")} == {"AFEW": {13}, "IEMOCAP": {3}, "MELD": {7}, "RAVDESS": {18}}, "自由度")

# 判定の機会と当たり
rulesA = cr[cr.rule.isin(["R1", "R2"])]
rules5 = scr[(scr.b == "prod") & scr.a.isin(R5_TEACH16 + R5_TEACH17)]
N_OPP = len(rulesA) + len(rules5)
must((len(rulesA), len(rules5)) == (24, 48), "判定の機会の数")
hitsA, hits5 = rulesA[rulesA.hit], rules5[rules5.hit]
N_HIT = len(hitsA) + len(hits5)
must(all("AFEW" in u.split(",") for u in list(hitsA.up) + list(hits5.up)), "当たりが AFEW を含まない規則がある")
must(all(int(r.n_up_orig4) >= 2 for r in list(hitsA.itertuples()) + list(hits5.itertuples())), "元の 4 コーパスの数えが 2 未満の当たりがある")
must(not rulesA.down2.any() and not rules5.down2.any(), "下回ったが 2 以上の規則がある")

# 生徒の段の数えるコーパスの数
for K in KS:
    for a in R5_TEACH17:
        must([int(srow(ro, a, "prod", K).n_corpora) for ro in ROLES] == [5, 6, 5, 4], f"R5 の分母 {a} K={K}")
    must([int(srow(ro, "M_F6_mixsel", "prod", K).n_corpora) for ro in ROLES] == [5, 5, 5, 4], f"R5 の分母 M_F6_mixsel K={K}")

# 実験 16 の版
for K in KS:
    r = crow("M_F6_mixsel", "prod", K)
    must(not r.hit and r.up == "AFEW" and int(r.n_up_orig4) == 1 and int(r.n_down) == 0, f"M_F6_mixsel−prod K={K}")
    r = crow("M_F6_K", "prod", K)
    must(not r.hit and r.up == "AFEW" and int(r.n_down) == 0, f"M_F6_K−prod K={K}")
    r = crow("M_F6_ext", "prod", K)
    must(not r.hit and r.down == "MELD" and r.up == ("AFEW" if K == 5 else ""), f"M_F6_ext−prod K={K}")
    r = crow("M_F6_orc", "prod", K)
    must(bool(r.hit) and r.up == {5: "AFEW,IEMOCAP,MOSEI,MELD", 10: "AFEW,IEMOCAP", 25: "AFEW,MELD"}[K], f"M_F6_orc−prod K={K}")
    r = crow("M_F6_orc", "M_F6n_orc", K)
    must(r.up == ("AFEW" if K == 5 else "") and int(r.n_down) == 0, f"M_F6_orc−M_F6n_orc K={K}")
    r = crow("M_F6_mixsel", "M_F6n_mixsel", K)
    must(int(r.n_up) == 0 and int(r.n_down) == 0, f"M_F6_mixsel−M_F6n_mixsel K={K}")
    for a, b in (("M_F6_orc", "prod"), ("M_F6_mixsel", "prod")):
        q = eq(a, b, "CREMA-D", K)
        must(q.lo > 0 and q.est < 1.0 and q.verdict == "言えない", f"{a}−{b} CREMA-D K={K}")
    for a, b in (("M_F6_orc", "prod"), ("M_F6_mixsel", "prod"), ("M_F6_K", "prod"), ("M_F6_ext", "prod"), ("M_F6_orc", "M_F6n_orc"),
                 ("M_F6_mixsel", "M_F6n_mixsel"), ("M_F2_orc", "prod"), ("M_F6n_orc", "prod")):
        must(eq(a, b, "CREMA-D", K).verdict == "言えない", f"{a}−{b} CREMA-D K={K} が言えないでない")
        must(len(ef[(ef.a == a) & (ef.b == b) & (ef.c == "RAVDESS")]) == 0, f"{a}−{b} に RAVDESS の行がある")
    r = srow("fusion", "M_F6_mixsel", "prod", K)
    must(r.up == ("AFEW,MELD" if K == 25 else "AFEW") and bool(r.hit) == (K == 25) and int(r.n_down) == 0, f"融合 × M_F6_mixsel K={K}")
    for role in ("A", "V", "T"):
        r = srow(role, "M_F6_mixsel", "prod", K)
        must(int(r.n_up) == 0 and int(r.n_down) == 0, f"{role} × M_F6_mixsel K={K}")
    for role in ("fusion", "A", "V"):
        must(sq(role, "M_F6_mixsel", "prod", "CREMA-D", K).verdict == "言えない", f"{role} × M_F6_mixsel CREMA-D K={K}")
q = eq("M_F6_mixsel", "prod", "MELD", 25)
must(q.lo > 0 and q.est < 1.0, "M_F6_mixsel−prod MELD K=25")
must(len(se[(se.a == "M_F6_mixsel") & (se.c == "RAVDESS")]) == 0 and len(se[(se.role == "fusion") & (se.c == "RAVDESS")]) == 0, "RAVDESS の生徒の行")

# RAVDESS：積は K 本だけの声の分類器と同じ
for K in KS:
    must(level("RAVDESS", K, "prod") == level("RAVDESS", K, "prod", "uar_raw") == level("RAVDESS", K, "none_A", "uar_raw"), f"RAVDESS の積と K 本だけの声 K={K}")
    must(abs(cond("RAVDESS", K, "A", "prod").edge) < 1e-12, f"RAVDESS の疑似ラベルの幅 K={K}")

# 提案の手法の選ばれた点
mixsel = sel[sel.version == "M_F6_mixsel"]
must(len(mixsel) == 15 and set(mixsel.c) == set(C6[:5]) and set(mixsel.n_units) == {40.0}, "M_F6_mixsel の選択の行")
N_MIX2 = int((mixsel.chosen == "M_F6_mix2").sum())
must(all(P18.GRID16[int(x.replace("M_F6_mix", ""))][1] == 1.0 for x in mixsel.chosen), "β = 0 の点を選んだ組がある")
must(P18.GRID16[2] == (0.0, 1.0, "eq") and P18.GRID16[3] == (0.0, 1.0, "ext"), "格子の並び")
must(set(mixsel[mixsel.chosen == "M_F6_mix2"].n_tied) == {2.0} and set(mixsel[mixsel.chosen != "M_F6_mix2"].n_tied) == {1.0}, "同点の数")
units_csv = pd.concat([pd.read_csv(p, usecols=["K", "version", "uar_adj", "T", "c", "fold", "seed"], low_memory=False) for p in sorted((R / "units").glob("*.csv"))])
pv = units_csv[units_csv.version.isin(["M_F6_mix2", "M_F6_mix3"])].pivot_table(index=["T", "c", "fold", "seed", "K"], columns="version", values="uar_adj")
MIX23_N, MIX23_DIFF = len(pv), float((pv.M_F6_mix2 - pv.M_F6_mix3).abs().max())
must(MIX23_DIFF == 0.0, "mix2 と mix3 の UAR が違う単位がある")
for c in C6:
    for v in ("MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel", "MS_eq_sel", "N_eq_sel"):
        x = sel[(sel.c == c) & (sel.version == v)]
        must(len(x) == 3 and set(x.n_units) == {50.0 if c == "RAVDESS" else 40.0}, f"{c} {v} の選ぶ単位の数")
        must(all(c not in t.split(",") for t in x.pseudo_targets) and x.pseudo_targets.nunique() == 1, f"{c} {v} の仮のターゲット")
        must(all(("RAVDESS" in t.split(",")) == False for t in x.pseudo_targets), f"{c} {v} の選ぶ単位に RAVDESS がある")
must(set(sel[sel.c == "RAVDESS"].pseudo_targets.dropna()) == {"AFEW,CREMA-D,IEMOCAP,MELD,MOSEI"}, "RAVDESS の仮のターゲット")
ms = sel[sel.version == "MS_eq_sel"]
N_MS_MIN = int(ms.at_min_lam.astype(str).eq("True").sum())

# 引き戻しの強さ：CREMA-D の融合（声・顔）は 3 モダリティの融合の値
TS = [t for t in C6 if t != "CREMA-D"]
N_OWN_DIFF = 0
for t in TS:
    for K in KS:
        must(teth["tether"][t][f"AV|{K}"] == teth["tether"][t][f"AVT|{K}"], f"tether {t} AV|{K}")
        N_OWN_DIFF += teth["tether"][t][f"AV|{K}|own"] != teth["tether"][t][f"AV|{K}"]

# 声の生徒の選び直し
must(len(ph) == 15 and set(ph.c) == set(C6[:5]) and phprov["reproduce_max_abs_diff"] < 1e-9, "声の生徒の選び直しの表")
must(sha(SM / "students_effects.csv") == phprov["students_effects_sha256"] and sha(SM / "students_sel.csv") == phprov["students_sel_sha256"], "選び直しの入力が今の集計と違う")
must(phprov["script_sha256"] == sha(ROOT / "DAI" / "docs" / "tools_pending" / "dai18_posthoc_voice40.py"), "選び直しの台本が出力の後に変わった")
phc = ph[ph.changed]
must(set(ph.verdict_sel50) == {"言えない"} and set(ph.verdict_sel40) == {"言えない"}, "声の生徒の選び直しで上回る／下回るがある")
must([(r.c, int(r.K)) for r in phc.itertuples()] == [("AFEW", 5), ("CREMA-D", 25)], "点が変わった組")
must(all(abs(r.est_sel50 - r.est_sel40) < 1e-12 for r in ph[~ph.changed].itertuples()), "点が同じ組で値が違う")
for r in ph.itertuples():
    must(abs(r.est_sel50 - sq("A", "M_F6_mixsel", "prod", r.c, r.K).est) < 1e-9, f"選び直しの表の元の値 {r.c} K={r.K}")


def phci(r, which: str) -> str:
    return f"{fest(getattr(r, 'est_' + which))} [{fend(getattr(r, 'lo_' + which))}, {fend(getattr(r, 'hi_' + which))}]"


# 実験 17 の版
for K in KS:
    r = crow("MN_F6orc_sel", "prod", K)
    must(bool(r.hit) and r.up == {5: "AFEW,IEMOCAP,MOSEI,MELD", 10: "AFEW,IEMOCAP,MOSEI", 25: "AFEW,MOSEI,MELD"}[K], f"MN_F6orc_sel K={K}")
    r = crow("MN_eq_sel", "prod", K)
    must(int(r.n_up) == 0 and int(r.n_down) == 0, f"MN_eq_sel K={K}")
    r = crow("MN_F2K_sel", "prod", K)
    must(r.up == ("AFEW,MELD" if K == 25 else "AFEW") and bool(r.hit) == (K == 25), f"MN_F2K_sel K={K}")
    r = crow("MN_F6K_sel", "prod", K)
    must(bool(r.hit) and r.up == ("AFEW,MELD" if K == 25 else "AFEW,IEMOCAP"), f"MN_F6K_sel K={K}")
    for b in ("MS_eq_sel", "MS_eq_tied"):
        r = crow("MN_eq_sel", b, K)
        must(int(r.n_up) == 0 and int(r.n_down) == 0, f"MN_eq_sel−{b} K={K}")
    r = crow("MS_eq_sel", "prod", K)
    must(int(r.n_up) == 0 and int(r.n_down) == 0, f"MS_eq_sel−prod K={K}")
    for a in ("MN_F6orc_sel", "MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel"):
        must(set(verdicts(a, "prod", K, NEW2)) == {"言えない"}, f"{a}−prod の新しい 2 コーパス K={K}")
r17 = rules5[rules5.a.isin(R5_TEACH17)]
must(len(r17) == 36 and int(r17.hit.sum()) == 1, "実験 17 の版の R5 の当たりの数")
r = r17[r17.hit].iloc[0]
must((r.student, r.a, int(r.K), r.up) == ("fusion", "MN_F6K_sel", 25, "AFEW,MELD"), "実験 17 の版の R5 の当たり")
must(int(r17.n_down.sum()) == 0, "実験 17 の版の R5 に下回るがある")

# ------------------------------------------------------------------ 文

L: list[str] = []
w = L.append
PRO, MIX = "prod", "M_F6_mixsel"

w("# 実験 18 の下調べ（DAI-18．6 コーパスで実験 16・17 の主な版を流し直した下調べ）：結果の文の下書き（2026-11-14．第 1 版．照合の前）")
w("")
w("この文書は，台本 `DAI/docs/tools_pending/extract_dai18_text.py` が `results/dai18_probe/` の結果ファイルから数値を取り出して書き出した．"
  "結果ファイルから取る数値と表は，すべて台本が埋めた（手で写していない）．設計・点検の報告・台帳から引いた数値（0 節）は文として引き，引いた文が元の文書にあることを台本が確かめた．"
  "文が言う当たり・外れと数えも，台本が結果ファイルと照らしてから書き出している．文を直すときは台本の中の文を直して流し直す．")
w("")
w("差はすべて UAR（4 感情の再現率の平均．%）の点の差で，角かっこは 95% 区間である．")
w("")
w("## 0. 出どころと申告")
w("")
w(f"- **設計**：`DAI/docs/実験18_下調べ_設計.md`（第 3 版．sha256 {prov['design_sha256'][:8]}…．6 段の記録の `design_sha256` と同じ）．")
w(f"- **台本とコミット**：`DAI/exp/dai18_probe.py`（sha256 {prov['this_script_sha256'][:8]}…．6 段とも同じ値で，今の台本とも同じ）．起動の台本は `DAI/run/dai18_probe.sh`．"
  f"source・base・tether・units・students の 5 段はコミット {COMMITS['source']}，summary の段は {COMMITS['summary']} で流れ，dirty は 6 段とも偽である（`provenance_log.jsonl`）．"
  "関門 1 が検査したコミット 02f356e からこの 2 つのコミットまでの差分は，台本・共通の部品・入力のパスには無く，`DAI/run/dai19_*.sh` の追加だけである（点検の報告 2 節と記録 1）．")
w("- **数値を取り出した台本**：`DAI/docs/tools_pending/extract_dai18_text.py`（この文書の全体を書き出す．結果ファイルは読むだけ）．"
  "計算の後に足した量の台本は `DAI/docs/tools_pending/dai18_posthoc_voice40.py`（下の「計算の後に足した量」）．")
w("- **計算の前の検査**（`DAI/checks/DAI-18/`）：設計の検査の第 3 回 `probe_v3.md` と台本の検査の第 3 回 `code_v3.md` はどちらも「通す」．"
  "査読想定の役の `review_ml_v2.md` と `review_affect_v2.md` はどちらも「判定：OK」で，検査した版はコミット 87b1dc7 の設計と台本である．"
  "その後の直し（`MS_eq_sel` を `MN_eq_sel` と同じ単位で選ぶことなど．設計 10 節）は，`probe_v3.md` と `code_v3.md` が検査した．")
w("- **結果の場所**：`results/dai18_probe/`．設定とコミットの記録は `provenance.json`（summary の段の記録）と `provenance_log.jsonl`（6 段）．"
  "選んだ格子の点は `frozen_sel.json`（`summary/sel.csv` と同じ 156 行であることを台本が確かめた），引き戻しの強さは `tether.json`．")
w(f"- **起動と終了**：2026-11-14 の {TIMES['source'][:5]} に起動し（source の段の記録 {TIMES['source']}），{done[11:]} に終わった（`logs/dai18_probe.done`）．"
  f"段の始まりは base {TIMES['base']}，tether {TIMES['tether']}，units {TIMES['units']}，students {TIMES['students']}，summary {TIMES['summary']} である．"
  f"計算機は研究室の PC の GPU 1 枚（{prov['gpu']}）で，RunPod は使っていない．")
w("- **計算の後の点検**：`DAI/checks/DAI-18/code_post.md`（判定は「通す」．指摘なし，記録 3 件）．報告が確かめた量は次のとおりである．"
  "300 単位が base・units・students の 3 段ですべてそろい，欠け・余分・重複・壊れた npz・非有限の値は 0 である．"
  "`sel.csv` の 156 行は `frozen_sel.json` と同じで，`MS_eq_sel` と `MN_eq_sel` の選ぶ単位（数と仮のターゲット）は 18 組すべてで一致した．"
  "集計の 13 ファイルは CPU で作り直すとバイト単位で同じだった．"
  "生徒の配列を 4 単位で作り直すと，3,068 個のうち 3,067 個がビット単位で一致し，残る 1 個の一致は 0.99808 だった（基準 0.995）．"
  "T と c がどちらも元の 4 コーパスの 120 単位では，言葉と融合の K 本だけの分類器のテストの判定が，720 組すべてで実験 1 の保存と一致した（設計 5b 節 (d)）．"
  "base の勾配の最大は 1.0×10⁻⁶，止まりは base・students とも 0，GPU のメモリの最大は base 2,301 MB，units 8,117 MB，students 2,417 MB だった．")
w(f"- **検算の量**（台本が読んだ値）：`convergence.json` は単位 {conv['n_units']}，割合をそろえる切片の残りの最大 {sci(conv['adj_resid_max'])}，GPU の使用の最大 {conv['gpu_max_mb']:,.0f} MB，"
  f"units の段の所要の合計 {conv['sec_units']:,.0f} 秒．`students_checks.json` は単位 {chk['n_units']}，疑似ラベルの作り直しと units の保存の一致の最小 {chk['teacher_agree_min']:.1f}，勾配の最大 {sci(chk['grad_max'])}．")
w(f"- **結果を開いた時刻**：本体が集計の値を開いたのは 2026-11-14 20:39 で，点検の報告（最後の書き込み {mtime(CODE_POST)}）が出る前である．"
  f"研究者の「実験１８の結果は？」に答えて表の文書 `DAI/docs/実験18_下調べ_結果の表_2026-11-14.md`（最後の書き込み {mtime(TABLE_DOC)}）を作った．"
  "この時刻は `docs/seen-data.md` の DAI-20 の行に書いてある（DAI-18 の行には無い）．点検の時点で，20:26:29 より後に更新された結果ファイルは無かった（点検の報告 2 節．その後に足したのは下の `posthoc/` だけである）．")
w(f"- **計算の後に足した量**：声の生徒の選び直し（設計 5 節 R5 が結果の文に並べると決めた感度）を，`DAI/docs/tools_pending/dai18_posthoc_voice40.py` で "
  f"`results/dai18_probe/posthoc/voice_student_prod40.csv` に出した（{phprov['time']}．記録は同じ置き場の `provenance_voice40.json`）．"
  f"学び直さずに `students/` の保存（格子の全点のテストの判定）だけを読み，CPU で流した．元の選び方の差が `students_effects.csv` の値と合うことを確かめた（違いの最大 {sci(phprov['reproduce_max_abs_diff'])}）．"
  "この量を出したのは結果を開いた後である．")
w("")
w("**申告**（設計 0・5・8・9 節が求めるもの）：")
w("")
w("- **仮のターゲットの回だけの下調べで，評価の回は作っていない**．単位は（評価のターゲット T，仮のターゲット c，分割 5，種 2）の 300 で，値はどれも仮のターゲット c のテストの上で測り，T のデータは使わない．"
  "この文書で「コーパス c の値」と書くものは，c を仮のターゲットにした単位（T は c 以外の 5 つ）から作った値である．実験 16 の版は，c のモダリティが 2 つ以上の単位だけにある．")
w("- **4 コーパスの実験 16・17 の再現ではない**．元の 4 コーパスを評価する単位でも，ソースに CREMA-D と RAVDESS が入るので，出発点（ソースの分類器の重みの平均）と微調整した分類器が変わる．"
  "4 コーパスの結果との違いは，(1) ソースの数と中身（声のソースは 2 つから 4 つになり，うち 2 つが演技のコーパス．顔は 2 つから 3 つで，うち 1 つが演技），(2) 選び方に使う仮のターゲット，"
  "(3) 近いソースの発話の表の 3 つが同時に変わった結果で，どれの効果とも書かない．数えの違いを「仮のターゲットを足したため」とも読まない．この文書では 4 コーパスの実験 16・17 の値と並べた比べを作っていない．")
w("- **6 コーパスの数えには，結果を見た後の 4 コーパスが入る**．この下調べは，実験 16・17 の 4 コーパスの結果（照合済み）を見た後の計算である．"
  "特に，`M_F6_mixsel − prod` と融合の生徒 × `MN_F2K_sel` は，4 コーパスの結果を見てから重く見た版である．")
w("- **CREMA-D と RAVDESS は確認のデータにならない**．この 2 コーパスの UAR は以前の解析（E12・E13・E77・E78 など）で見ている．「見ていないデータでの確認」とは書かない．")
w("- **K は入れ子で，独立な確かめではない**．K = 10 の 40 本は K = 5 の 20 本を含み，K = 25 の 100 本は K = 10 の 40 本を含む．テストの発話・分割・種・ソースの分類器は 3 つの K で同じである．"
  "**K どうしの差の区間は作っていない**ので，K の間の判定の違いを差の証拠として読まない．")
w(f"- **判定の機会と偶然の水準**：R1・R2・R5 の判定の機会は，K 3 つを合わせて {N_OPP} ある（R1 と R2 が {len(rulesA)}，R5 が {len(rules5)}．設計は「約 70」と書いた）．"
  "設計の目安では，1 つの規則が偶然当たる見込みを約 0.01 とすると，どれかが偶然当たる見込みの合計は約 0.6 である（規則どうしが独立なときの値．同じ単位を使うので独立ではない）．"
  f"したがって 1 つの規則の当たりを主張にしない．当たりは {N_OPP} のうち {N_HIT}（R1 の参照が {int((hitsA.rule == 'R1').sum())}，R2 が {int((hitsA.rule == 'R2').sum())}，R5 が {len(hits5)}）で，"
  f"{N_HIT} 件すべてが AFEW を含み，元の 4 コーパスの数えも 2 以上である．")
w("- **本番に進む判断には使わない**．この下調べは 4 コーパスの結果の広がりを見る記述で，出力に `proceed` の列は無い．主の K は 5 である．")
w("- **本体が決めたこと**（研究者の確認を経ていない）：流す版を実験 16・17 で判定に使った主な版に絞ること，実験 17 のコーパスごとの重みを外すこと，層を adj・raw・候補の 3 つに絞ること，CREMA-D と RAVDESS の区間の作り方．"
  "実験 16 の研究者の形（F3）・F4・F5・表の読み替え・ob と kb の層，実験 17 の片方だけの近さの版・自分のコーパスの信頼度の版・コーパスごとの重みは流していない．"
  "したがって実験 16 の R0・R3 と実験 17 の R3・R4 に当たる規則は，この下調べに無い．")
w("- **区間の作り方**（95%．補正しない）：AFEW・IEMOCAP・MELD は塊を全分割から同時に除く jackknife と種の項（自由度 13・3・7），MOSEI と CREMA-D は塊の重みの引き直しと種の項（自由度なし）である．"
  "**RAVDESS は，24 話者を全分割から 1 人ずつ除く jackknife で，自由度は台本の決まりで 18 にした（真の 23 より小さいので，区間は控えめに広く出る）**．"
  "IEMOCAP は自由度 3 で区間が広い．検出できる差の目安は CREMA-D と RAVDESS には無いので，外れた規則には標準誤差と自由度を 4 節に並べる．")
w("- **版の格子の点の選び方**：コーパス c を評価するときは，c を評価のターゲットにした単位（c のデータを含まない）のテストの UAR の平均が最大の点を，K ごとに選んだ．"
  "候補がすべてそろう単位だけで平均を取るので，c が RAVDESS 以外のときは RAVDESS を仮のターゲットにした単位を除く 40 単位，c が RAVDESS のときは 50 単位で選んでいる（`sel.csv` の `n_units` と `pseudo_targets`）．"
  "元の 4 コーパスを評価するときに選択に入る演技のコーパスは CREMA-D だけである．"
  "CREMA-D（分類器 2 つ）と RAVDESS（分類器 1 つ）を評価するときも，分類器が 3 つの単位を含む単位で選んだ点を使う．")
w("- **引き戻しの強さの選び方**（K 本だけの分類器を出発点へ引き戻す強さ）：評価のターゲット T ごとに，T の仮のターゲットの単位のテストの UAR の平均で選ぶ（実験 1 と同じ規則）．"
  "評価するコーパス c 自身のテストが，選ぶ単位の 1/3〜1/5 の重みで入る．積と各版は同じ強さを使うが，UAR の水準と「今の提案 − K 本だけ」の差は甘く出る向きがある．"
  "この選択には，演技のコーパス（CREMA-D・RAVDESS）を仮のターゲットにした単位が入る．")
w(f"- **CREMA-D の融合（声・顔）の引き戻しの強さは，同じ T と K の 3 モダリティの融合で選んだ値である**（声・顔の融合を持つ仮のターゲットは CREMA-D だけなので，元の規則では CREMA-D 自身のテストだけで選ぶことになるため）．"
  f"元の規則で選んだ値は `tether.json` に記録だけ残してあり，{len(TS) * len(KS)} 組（T 5 つ × K 3 つ）のうち {N_OWN_DIFF} 組で使った値と違う（5 節）．")
w("- **生徒の格子の点の選び方**：疑似ラベルの重み ρ と K 本の倍率 a の 12 点を，c を評価のターゲットにした単位で，役（声・顔・言葉・融合）と疑似ラベルごとに選び直した．"
  "CREMA-D の融合（声・顔）の生徒は，3 モダリティの融合の生徒で選んだ点を使う．"
  "**声の生徒では，提案の手法（`M_F6_mixsel`）の疑似ラベルの側は 40 単位，積の疑似ラベルの側は 50 単位で点を選んでおり，単位の数が違う**（RAVDESS を仮のターゲットにした単位に提案の手法の答えが無いため）．"
  "積の側を同じ 40 単位で選び直した値を要点 5 に並べる．")
w("- **CREMA-D と RAVDESS のラベルと刺激**（設計 8 節）：ラベルは演者に指示した感情で，聞き手の判断ではない．"
  "CREMA-D の原論文（Cao ほか 2014）では，人が意図どおりの感情を当てた割合は声だけで 40.9%，顔だけで 58.2%，声と顔で 63.6% である．"
  "どちらも決まった文を読み（CREMA-D 12 文，RAVDESS 2 文），文は学習側とテストにまたがる．言葉のモダリティは使っていない．"
  "CREMA-D では文「It's eleven o'clock.」の発話のうち Neutral が 10.0% で，ほかの 11 文の 25.0% より低いので，文が感情の手がかりになりうる．"
  "RAVDESS では Neutral とほかの 3 感情の件数の比が 1：2 で，話す発話だけを使い，calm は Neutral に混ぜていない．RAVDESS が声だけなのは映像の特徴を作っていないためで，コーパスの性質ではない．"
  "CREMA-D の顔の特徴は 91.3% の発話で 1 区間の平均で，顔が検出されない 1 発話を除いた．これらを確かめる量（設計 8 節の 6）はまだ出していない（6 節）．")
w("- **実験 17 の版の扱い**：研究者は 2026-11-14 に「実験 17 はいったん無視したい」と言った．実験 17 の版（近いソースの発話の答えを足す版）は，規則の結果を要点 7 に短く記録し，表を 3 節に置く．")
w("- **版の名前**：`prod` は「積」（微調整した各モダリティの分類器の確率の積．今の提案の疑似ラベル）．`M_F6_mixsel` は「提案の手法（水準は K 本＋相性はソース）」で，分類器 × 感情の信頼度の表のうち，"
  "分類器ごとの水準（感情の平均）を K 本で測り，水準からの感情ごとの上下（相性）をソースで測った表から取る．`M_F6n_mixsel` は同じ格子の点で分類器 × 感情の成分を消した対照．"
  "`M_F6_orc` は信頼度の表を全ラベルで測った参照，`M_F6n_orc` はその分類器 × 感情の成分を消した対照，`M_F2_orc` は全ラベルで測った水準だけを使う版．"
  "`M_F6_K` は信頼度を K 本だけで測る版，`M_F6_ext` はソースで測った値だけで決める版．"
  "積の疑似ラベルで学び直した分類器を「今の提案」，疑似ラベルで学び直した分類器を設計の語で「生徒」と書く．実験 17 の版の名前は要点 7 に書く．")
w("- **表の読み方**：↑ は「上回る」（区間の下端 > 0 かつ差 ≥ 1.0），↓ は「下回る」（上端 < 0 かつ差 ≤ −1.0）．"
  "「判定」は設計の規則で，数えるコーパス（6．実験 16 の版は RAVDESS を除く 5）のうち上回るのが 2 以上なら「当たり」，それ以外は「外れ」とする．規則でない行は「—」．"
  "生徒の段は役ごとに数えるコーパスが違い，声 6，顔 5，言葉 4，融合 5 である（提案の手法の疑似ラベルの生徒は RAVDESS に無いので，声も 5）．"
  "元の 4 コーパス（AFEW・IEMOCAP・MOSEI・MELD）だけの数えを並べる．差が丸めると ±1.00 になるときと，区間の端が 0.05 より 0 に近いときは，境目のどちら側かが読めるよう桁を足して書く．")
w("")

# ---- 1 節
w("## 1. 要点")
w("")
w("### 規則の当たり・外れの一覧（K = 5・10・25）")
w("")
w("「↑ a／b」は，数える b コーパスのうち a コーパスで上回ったことを表す．名前は上回ったコーパス．得の帰し方と対照の行は規則ではないので，判定の語を付けない．")
w("")
w("| 区分 | 比べ | K = 5（主） | K = 10 | K = 25 |")
w("|---|---|---|---|---|")
OV16 = [("R1（参照）", "全ラベルで測った参照 `M_F6_orc − prod`", "M_F6_orc", PRO, None, True),
        ("R2", "提案の手法 `M_F6_mixsel − prod`", MIX, PRO, None, True),
        ("R2", "K 本だけで測る `M_F6_K − prod`", "M_F6_K", PRO, None, True),
        ("R2", "ソースだけで決める `M_F6_ext − prod`", "M_F6_ext", PRO, None, True)] + \
       [("R5", f"{ROLE_JP[ro]}の生徒：提案の手法の疑似ラベル − 積の疑似ラベル", MIX, PRO, ro, True) for ro in ROLES] + \
       [("得の帰し方", "分類器 × 感情の成分の分（全ラベル）`M_F6_orc − M_F6n_orc`", "M_F6_orc", "M_F6n_orc", None, False),
        ("得の帰し方", "分類器 × 感情の成分の分（提案の手法）`M_F6_mixsel − M_F6n_mixsel`", MIX, "M_F6n_mixsel", None, False)]
OV17 = [("R1（参照）", "`MN_F6orc_sel − prod`", "MN_F6orc_sel", PRO, None, True)] + \
       [("R2", f"`{a} − prod`", a, PRO, None, True) for a in ("MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel")] + \
       [("R5", f"{ROLE_JP[ro]}の生徒 × `{a}`", a, PRO, ro, True) for ro in ROLES for a in R5_TEACH17] + \
       [("得の帰し方", "`MN_eq_sel − MS_eq_sel`", "MN_eq_sel", "MS_eq_sel", None, False), ("得の帰し方", "`MN_eq_sel − MS_eq_tied`", "MN_eq_sel", "MS_eq_tied", None, False),
        ("対照", "`MS_eq_sel − prod`", "MS_eq_sel", PRO, None, False)]
for title, rows_ in (("実験 16 の版", OV16), ("実験 17 の版", OV17)):
    w(f"| **{title}** | | | | |")
    for kind, lab, a, b, ro, rule in rows_:
        w(f"| {kind} | {lab} | " + " | ".join(ov(srow(ro, a, b, K) if ro else crow(a, b, K), rule) for K in KS) + " |")
w("")
w("### 実験 16 の版")
w("")
others = ["IEMOCAP", "MOSEI", "MELD"]
w(f"1. **提案の手法（水準は K 本＋相性はソース．`M_F6_mixsel`）の推定時の答えは，K = 5・10・25 のどれでも，積（`prod`）を上回ったコーパスが AFEW の 1 つで，R2 の規則は 3 つの K とも外れた**．"
  f"AFEW の差は {E3(MIX, PRO, 'AFEW')} である．元の 4 コーパスの数えは 3 つの K とも 1 である．"
  f"IEMOCAP・MOSEI・MELD は 3 つの K とも上回るとは言えなかった（K = 5 は順に {'，'.join(E(MIX, PRO, c, 5) for c in others)}．K = 10・25 の値は 3.1 節が指す表，標準誤差と自由度は 4 節）．"
  f"MELD の K = 25 は {E(MIX, PRO, 'MELD', 25)} で，区間の下端は 0 より上だが差が 1.0 点に届かない．")
w(f"   - **CREMA-D**（新しいコーパス．声と顔）：{E3(MIX, PRO, 'CREMA-D')}．3 つの K とも区間の下端は 0 より上で，差は 1.0 点に届かないので，設計の語では上回るとは言えなかった．RAVDESS は声だけなので，この版が無い．")
w(f"   - **得の帰し方**：分類器 × 感情の成分を消した対照との差 `M_F6_mixsel − M_F6n_mixsel` は，3 つの K のどのコーパスでも上回らなかった（AFEW は {E3(MIX, 'M_F6n_mixsel', 'AFEW')}）．"
  "設計 5 節は，この差が上回るときに限って得を分類器 × 感情の成分に帰すと決めているので，AFEW の差をこの成分の分とは書かない．")
nonmix2 = mixsel[mixsel.chosen != "M_F6_mix2"]
w(f"   - **選ばれた点**（5.1 節）：15 組（5 コーパス × K 3 つ）のうち {N_MIX2} 組が mix2（n0 = 0，β = 1．K 本で測った水準をそのまま使い，相性を使う点）である．"
  f"残る {len(nonmix2)} 組は，" + "，".join(f"{r.c} の K = {int(r.K)} が {mix_desc(r.chosen)}" for r in nonmix2.itertuples()) + "である．β = 0（相性を使わない）の点を選んだ組は無い．")
w("   - **出どころ**：この比べは，4 コーパスの実験 16 の結果を見てから重く見た版である（設計 0 節）．")
w(f"2. **R2 のほかの 2 つの版（信頼度を K 本だけで測る `M_F6_K`，ソースで測った値だけで決める `M_F6_ext`）も，3 つの K とも外れた**．"
  f"`M_F6_K − prod` で上回ったのは，3 つの K とも AFEW の 1 コーパスである（{E3('M_F6_K', PRO, 'AFEW')}）．"
  f"`M_F6_ext − prod` は，K = 5 で AFEW が上回り（{E('M_F6_ext', PRO, 'AFEW', 5)}），MELD が 3 つの K とも下回った（{E3('M_F6_ext', PRO, 'MELD')}）．"
  "K = 10・25 では上回ったコーパスが無い．下回ったコーパスは 3 つの K とも MELD の 1 つである．"
  f"CREMA-D はどちらの版でも上回るとは言えなかった（`M_F6_K − prod` は {E3('M_F6_K', PRO, 'CREMA-D')}．`M_F6_ext − prod` は {E3('M_F6_ext', PRO, 'CREMA-D')}）．")
w(f"3. **全ラベルで測った参照（R1．`M_F6_orc − prod`）は，3 つの K とも規則の上で当たりだった**．"
  f"上回ったのは，K = 5 で {jp(crow('M_F6_orc', PRO, 5).up)} の {int(crow('M_F6_orc', PRO, 5).n_up)} コーパス，K = 10 で {jp(crow('M_F6_orc', PRO, 10).up)} の {int(crow('M_F6_orc', PRO, 10).n_up)} コーパス，"
  f"K = 25 で {jp(crow('M_F6_orc', PRO, 25).up)} の {int(crow('M_F6_orc', PRO, 25).n_up)} コーパスである（値は 3.1 節が指す表）．"
  "この版は，値を測るコーパス自身の全ラベルで信頼度を測るので，実際には使えない参照である．"
  f"CREMA-D は {E3('M_F6_orc', PRO, 'CREMA-D')} で，3 つの K とも区間の下端は 0 より上だが差が 1.0 点に届かず，上回るとは言えなかった．"
  f"得の帰し方では，分類器 × 感情の成分を消した対照との差 `M_F6_orc − M_F6n_orc` が上回ったのは K = 5 の AFEW（{E('M_F6_orc', 'M_F6n_orc', 'AFEW', 5)}）の 1 つで，K = 10・25 ではどのコーパスでも上回らなかった．")
w(f"4. **生徒の段（R5）では，提案の手法の答えを疑似ラベルにして学び直した融合の分類器と，今の提案（積を疑似ラベルにして学び直した融合の分類器）の差は，K = 5・10 では AFEW の 1 コーパスで上回って外れ，"
  f"K = 25 では AFEW と MELD の 2 コーパスで上回って規則の上で当たりになった**．"
  f"AFEW は {SQ3('fusion', MIX, PRO, 'AFEW')}，MELD は {SQ3('fusion', MIX, PRO, 'MELD')} である．"
  "声・顔・言葉の生徒では，3 つの K のどのコーパスでも上回るとは言えなかった（値は 3.1 節が指す表と 3.5 節）．K = 25 の当たりには次の断りが付く．")
w("   - 主の K は 5 で，K = 5 と K = 10 は外れである．3 つの K は入れ子で，テストは同じなので，独立な確かめではない．K どうしの差の区間は作っていないので，K の間の判定の違いを差の証拠として読まない．")
w(f"   - 判定の機会は {N_OPP} あり（0 節），1 つの規則の当たりを主張にしない．")
w(f"   - 当たりの 2 コーパスは AFEW を含み，どちらも元の 4 コーパスである（元の 4 コーパスの数え 2）．CREMA-D の融合（声・顔）の生徒は K = 25 で {SQ('fusion', MIX, PRO, 'CREMA-D', 25)} で，上回るとは言えなかった．")
w(f"   - 推定時の答えの同じ比べ（要点 1）は，K = 25 でも外れである（MELD は {E(MIX, PRO, 'MELD', 25)}）．")
w("   - **得を何に帰すか**：分類器 × 感情の成分を消した対照 `M_F6n_mixsel` を疑似ラベルにした生徒は学び直していない．設計 5 節のとおり，`M_F6_mixsel` の生徒には同じ重みの付け方の対照が無いので，この差を分類器 × 感情の成分の分とは書かない．")
ca = {c: (cond(c, 25, "AVT", MIX), cond(c, 25, "AVT", PRO)) for c in ("AFEW", "MELD")}
w("   - **疑似ラベルの量**（`students_conditions.csv`．融合，K = 25．提案の手法の疑似ラベル 対 積の疑似ラベル）：疑似ラベルの UAR が K 本だけの融合の分類器の UAR を超える幅（候補の K 本以外の発話）は，"
  + "，".join(f"{c} {ca[c][0].edge:.2f} 対 {ca[c][1].edge:.2f}" for c in ca) + "．疑似ラベルの最大の確率の平均は，"
  + "，".join(f"{c} {ca[c][0].maxp:.3f} 対 {ca[c][1].maxp:.3f}" for c in ca) + "．生徒の判定が疑似ラベルの判定と一致した割合は，"
  + "，".join(f"{c} {ca[c][0].mimic:.3f} 対 {ca[c][1].mimic:.3f}" for c in ca) + "（この割合は，GPU と CPU の計算の違いで小数第 3 位が変わりうる．点検の報告の記録 3）．")
w(f"   - **選んだ設定**（5.3 節．疑似ラベルの重み ρ・K 本の倍率 a）：K = 25 の融合の生徒は，AFEW では提案の手法の疑似ラベルの側が {st_point('AFEW', 25, 'AVT', MIX)}，積の疑似ラベルの側が {st_point('AFEW', 25, 'AVT', PRO)}，"
  f"MELD では提案の手法の疑似ラベルの側が {st_point('MELD', 25, 'AVT', MIX)}，積の疑似ラベルの側が {st_point('MELD', 25, 'AVT', PRO)} である．")
w("   - **出どころ**：`M_F6_mixsel` は，4 コーパスの実験 16 の結果を見てから重く見た版である．")
r1, r2 = phc.iloc[0], phc.iloc[1]
w(f"5. **声の生徒の比べ（提案の手法の疑似ラベル − 積の疑似ラベル）は，積の側の格子の点を提案の手法の側と同じ 40 単位で選び直しても，5 コーパス × K 3 つの 15 組のどれでも「上回る」「下回る」に入らなかった**．"
  "計算の前に決めた選び方では，提案の手法の側は 40 単位，積の側は 50 単位で点を選ぶ（0 節）．"
  f"積の側を 40 単位で選び直すと，15 組のうち {len(phc)} 組（{r1.c} の K = {int(r1.K)}，{r2.c} の K = {int(r2.K)}）で点が変わった．"
  f"{r1.c} の K = {int(r1.K)} は，積の側の点が ρ {r1.rho_prod50:g}・a {int(r1.a_prod50)} から ρ {r1.rho_prod40:g}・a {int(r1.a_prod40)} に変わり，差は {phci(r1, 'sel50')} から {phci(r1, 'sel40')} になった．"
  f"{r2.c} の K = {int(r2.K)} は，ρ {r2.rho_prod50:g}・a {int(r2.a_prod50)} から ρ {r2.rho_prod40:g}・a {int(r2.a_prod40)} に変わり，差は {phci(r2, 'sel50')} から {phci(r2, 'sel40')} になった．"
  f"残る {len(ph) - len(phc)} 組は点が変わらず，値も変わらない（`results/dai18_probe/posthoc/voice_student_prod40.csv`）．判定は，計算の前に決めた選び方の値（3.5 節）で行う．")
w("")
w("### 新しい 2 コーパス（CREMA-D，RAVDESS）")
w("")
w(f"6. **新しい 2 コーパスのうち実験 16 の版があるのは CREMA-D で，CREMA-D では R1・R2・R5 と得の帰し方の比べのどれも「上回る」「下回る」に入らなかった**．"
  f"RAVDESS は声だけなので，実験 16 の版が無い．規則の当たりは，どれも元の 4 コーパスの中で成り立った（当たりの {N_HIT} 件の元の 4 コーパスの数えは 2 以上）．")
w(f"   - **CREMA-D**（声と顔．分類器 2 つ）：推定時の答えの差は要点 1〜3 のとおりである．生徒の段の「提案の手法の疑似ラベル − 積の疑似ラベル」は，"
  f"融合（声・顔）が {SQ3('fusion', MIX, PRO, 'CREMA-D')}，声が {SQ3('A', MIX, PRO, 'CREMA-D')}，顔が {SQ3('V', MIX, PRO, 'CREMA-D')} である．"
  f"積の UAR は K = 5・10・25 で {level('CREMA-D', 5, PRO):.2f}・{level('CREMA-D', 10, PRO):.2f}・{level('CREMA-D', 25, PRO):.2f}，"
  f"提案の手法の UAR は {level('CREMA-D', 5, MIX):.2f}・{level('CREMA-D', 10, MIX):.2f}・{level('CREMA-D', 25, MIX):.2f} である（水準は引き戻しの強さの選び方で甘く出る向きがある．0 節）．"
  "格子の点は，分類器が 3 つある元の 4 コーパスの単位で選んだ点を使っている．")
w(f"   - **RAVDESS**（声だけ．分類器 1 つ）：微調整に使う融合の分類器が無いので，微調整した声の分類器は K 本だけの声の分類器と同じで，積は K 本だけの声の分類器と作りの上で同じになる（設計 2 節）．"
  f"`levels.csv` でも，積と K 本だけの声の分類器の UAR は 3 つの K とも同じ値である（K = 5・10・25 で {level('RAVDESS', 5, PRO):.2f}・{level('RAVDESS', 10, PRO):.2f}・{level('RAVDESS', 25, PRO):.2f}）．"
  "複数の分類器の信頼度を比べる実験 16 の版と，融合の生徒は無い．RAVDESS にあるのは実験 17 の版と声の生徒で，値は 3.3〜3.6 節の表にある．")
w("   - この 2 コーパスは以前の解析で UAR を見ており，確認のデータにならない．ラベルは演者に指示した感情で，決まった文を読むコーパスである（0 節）．")
w("")
w("### 実験 17 の版（規則の結果の記録）")
w("")
w("7. **実験 17 の版（近いソースの発話の答えを積に足す版）の規則の結果を記録する**．研究者が「実験 17 はいったん無視したい」と言ったので，解釈は加えない．"
  "版の名前：`MN_eq_sel` は積に近いソースの発話の答えを等しい重みで足す版，`MN_F2K_sel` は水準を K 本で測って重みづける版，`MN_F6K_sel` は感情ごとの信頼度を K 本で測って重みづける版，"
  "`MN_F6orc_sel` は `MN_F6K_sel` と同じ点で信頼度を全ラベルで測った参照，`MS_eq_sel` は積にソースの分類器の答えを等しい重みで足す対照，`MS_eq_tied` は `MN_eq_sel` が選んだのと同じ λ の対照，`N_eq_sel` は近いソースの発話の答えだけの版である．")
c5, c10, c25 = (crow("MN_F6orc_sel", PRO, K) for K in KS)
w(f"   - **R1（参照）**：`MN_F6orc_sel − prod` は 3 つの K とも当たり（上回ったのは K = 5 で {jp(c5.up)}，K = 10 で {jp(c10.up)}，K = 25 で {jp(c25.up)}）．"
  f"CREMA-D は {E3('MN_F6orc_sel', PRO, 'CREMA-D')}，RAVDESS は {E3('MN_F6orc_sel', PRO, 'RAVDESS')} で，どちらも上回るとは言えなかった．")
w(f"   - **R2**：`MN_eq_sel − prod` は 3 つの K とも上回ったコーパスが無く，外れ．"
  f"`MN_F2K_sel − prod` は K = 5・10 で AFEW の 1 コーパス（{E('MN_F2K_sel', PRO, 'AFEW', 5)}，{E('MN_F2K_sel', PRO, 'AFEW', 10)}）で外れ，"
  f"K = 25 で AFEW（{E('MN_F2K_sel', PRO, 'AFEW', 25)}）と MELD（{E('MN_F2K_sel', PRO, 'MELD', 25)}）で当たり．"
  f"`MN_F6K_sel − prod` は 3 つの K とも当たりで，上回ったのは K = 5・10 で AFEW と IEMOCAP，K = 25 で AFEW と MELD である"
  f"（AFEW は {E3('MN_F6K_sel', PRO, 'AFEW')}．IEMOCAP は K = 5 で {E('MN_F6K_sel', PRO, 'IEMOCAP', 5)}，K = 10 で {E('MN_F6K_sel', PRO, 'IEMOCAP', 10)}．MELD は K = 25 で {E('MN_F6K_sel', PRO, 'MELD', 25)}）．"
  "当たりはどれも AFEW を含み，元の 4 コーパスの数えは 2 である．CREMA-D と RAVDESS では，3 つの版のどの K でも上回るとは言えなかった（3.4 節）．")
w(f"   - **得の帰し方**：`MN_eq_sel − MS_eq_sel` と `MN_eq_sel − MS_eq_tied` は，3 つの K のどのコーパスでも上回らなかった．対照 `MS_eq_sel − prod` は，どのコーパスでも上回りも下回りもしなかった．"
  f"`MS_eq_sel` が最小の λ を選んだ組は 18 組のうち {N_MS_MIN} 組である（5.2 節）．"
  "`MN_F2K_sel` と `MN_F6K_sel` には同じ重みの付け方の対照が無いので，この 2 つの版の差はソースの分類器を足した分と区別していない．")
one17 = r17[(~r17.hit) & (r17.n_up == 1)].copy()
one17["o"] = one17.student.map(ROLES.index)
one17 = one17.sort_values(["o", "a", "K"])
w(f"   - **R5（生徒）**：36 の機会（役 4 × 疑似ラベル 3 × K 3）のうち，当たりは融合の生徒 × `MN_F6K_sel` の K = 25 の 1 つ"
  f"（AFEW {SQ('fusion', 'MN_F6K_sel', PRO, 'AFEW', 25)}，MELD {SQ('fusion', 'MN_F6K_sel', PRO, 'MELD', 25)}）で，残る 35 は外れである．"
  f"外れのうち 1 コーパスで上回った組は {len(one17)} あり，"
  + "，".join(f"{ROLE_JP[r.student]} × `{r.a}` の K = {int(r.K)}（{r.up} {SQ(r.student, r.a, PRO, r.up, int(r.K))}）" for r in one17.itertuples())
  + "である．このうち顔 × `MN_F6K_sel` の K = 25 は演技のコーパス CREMA-D で上回ったもので，元の 4 コーパスの数えは 0 である．"
  "`MN_F2K_sel`・`MN_F6K_sel` の生徒にも同じ重みの付け方の対照が無いので，ソースの分類器を足した分と区別していない．")
w("")

# ---- 2 節
w("## 2. そのほかの比べ（規則でない量）")
w("")


def ups(a, b, K, role=None):
    x = se[(se.role == role) & (se.a == a) & (se.b == b) & (se.K == K)] if role else ef[(ef.a == a) & (ef.b == b) & (ef.K == K) & (ef["where"] == "adj")]
    must(len(x) > 0, f"差の行が無い：{role} {a}−{b} K={K}")
    x = x.set_index("c")
    cs = [c for c in C6 if c in x.index]
    return [c for c in cs if x.loc[c].verdict == "上回る"], [c for c in cs if x.loc[c].verdict == "下回る"]


def ups_text(a, b, role=None) -> str:
    out = []
    for K in KS:
        u, d = ups(a, b, K, role)
        if u and d:
            out.append(f"K = {K} で上回ったのは {jp(u)}，下回ったのは {jp(d)}")
        elif u:
            out.append(f"K = {K} で上回ったのは {jp(u)}")
        elif d:
            out.append(f"K = {K} で下回ったのは {jp(d)}（上回ったコーパスは無い）")
        else:
            out.append(f"K = {K} は上回ったコーパスも下回ったコーパスも無い")
    return "．".join(out)


w(f"- **水準だけを全ラベルで測る版 `M_F2_orc − prod`**（3.1 節が指す表）：{ups_text('M_F2_orc', PRO)}．CREMA-D は {E3('M_F2_orc', PRO, 'CREMA-D')} である．")
w(f"- **分類器 × 感情の成分を消した対照（全ラベル）`M_F6n_orc − prod`**（3.2 節）：{ups_text('M_F6n_orc', PRO)}．CREMA-D は {E3('M_F6n_orc', PRO, 'CREMA-D')} である．")
w("- **今の提案 − K 本だけの分類器**（生徒の段の `prod − none`．3.6 節）：引き戻しの強さの選び方のため，この差は甘く出る向きがある（0 節）．")
for ro in ROLES:
    w(f"   - {ROLE_JP[ro]}の生徒：{ups_text(PRO, 'none', ro)}．")
w(f"   - CREMA-D の値：声が {SQ3('A', PRO, 'none', 'CREMA-D')}，顔が {SQ3('V', PRO, 'none', 'CREMA-D')}，融合（声・顔）が {SQ3('fusion', PRO, 'none', 'CREMA-D')} である．")
w(f"   - RAVDESS の値：声が {SQ3('A', PRO, 'none', 'RAVDESS')} である．RAVDESS では積が K 本だけの声の分類器と同じなので，この差は「K 本だけの声の分類器の答えを疑似ラベルにして学び直した声の分類器 − K 本だけの声の分類器」に当たる．")
w(f"- **近いソースの発話の答えだけの版 `N_eq_sel − prod`**（実験 17．3.4 節）：{ups_text('N_eq_sel', PRO)}．")
w(f"- **水準で重みづけた分 `MN_F2K_sel − MN_eq_sel`**（実験 17．3.4 節）：{ups_text('MN_F2K_sel', 'MN_eq_sel')}．")
w("")

# ---- 3 節
w("## 3. 表")
w("")
w("### 3.1 実験 16 の版（既にある表の文書）")
w("")
w(f"実験 16 の版の表は `DAI/docs/実験18_下調べ_結果の表_2026-11-14.md`（sha256 {sha(TABLE_DOC)[:8]}…．書き出しの台本 `DAI/docs/tools_pending/md_tables_dai18.py`）にあり，ここには写さない．"
  "その文書の 1 節が各版の UAR，2 節が判定の規則と比べ（R1・R2，水準だけの版，「モダリティ × 感情の分」＝この文書の分類器 × 感情の成分の分），3 節が提案の手法で選ばれた設定，4 節が生徒の UAR と「提案の手法の疑似ラベル − 積の疑似ラベル」である．"
  "UAR の水準は，引き戻しの強さの選び方で甘く出る向きがある（0 節）．")
w("")
w("### 3.2 実験 16 の版で，表の文書に無い比べ")
w("")
w(HEAD)
w(HSEP)
for K in KS:
    w(eff_line("記述", "相性を消した対照（全ラベル）− 積 `M_F6n_orc − prod`", "M_F6n_orc", PRO, K))
w("")
w("### 3.3 実験 17 の版：UAR（ラベルを使わずに切片をずらした答え）")
w("")
w("水準は，引き戻しの強さの選び方で甘く出る向きがある（0 節）．")
w("")
V17 = [("prod", "積"), ("MN_F6orc_sel", "`MN_F6orc_sel`（全ラベルの信頼度．参照）"), ("MN_eq_sel", "`MN_eq_sel`（等しい重み）"), ("MN_F2K_sel", "`MN_F2K_sel`（水準は K 本）"),
       ("MN_F6K_sel", "`MN_F6K_sel`（感情ごとの信頼度は K 本）"), ("MS_eq_sel", "`MS_eq_sel`（積＋ソースの分類器．対照）"), ("MS_eq_tied", "`MS_eq_tied`（同じ λ の対照）"),
       ("N_eq_sel", "`N_eq_sel`（近いソースの発話の答えだけ）")]
w("| 版 | K | " + " | ".join(C6) + " |")
w("|---|---|" + "---|" * len(C6))
for v, lab in V17:
    for K in KS:
        w(f"| {lab} | {K} | " + " | ".join(f"{level(c, K, v):.2f}" for c in C6) + " |")
w("")
w("### 3.4 実験 17 の版：判定の規則と比べ（ラベルを使わずに切片をずらした答え）")
w("")
w(HEAD)
w(HSEP)
T17 = [("R1", "全ラベルの信頼度 `MN_F6orc_sel − prod`（参照）", "MN_F6orc_sel", PRO), ("R2", "等しい重み `MN_eq_sel − prod`", "MN_eq_sel", PRO),
       ("R2", "水準は K 本 `MN_F2K_sel − prod`", "MN_F2K_sel", PRO), ("R2", "感情ごとの信頼度は K 本 `MN_F6K_sel − prod`", "MN_F6K_sel", PRO),
       ("帰し方", "ソースの発話を引く − ソースの分類器を足す `MN_eq_sel − MS_eq_sel`", "MN_eq_sel", "MS_eq_sel"),
       ("帰し方", "同じ λ の対照との差 `MN_eq_sel − MS_eq_tied`", "MN_eq_sel", "MS_eq_tied"), ("対照", "ソースの分類器を足す `MS_eq_sel − prod`", "MS_eq_sel", PRO),
       ("記述", "水準で重みづけた分 `MN_F2K_sel − MN_eq_sel`", "MN_F2K_sel", "MN_eq_sel"), ("記述", "近いソースの発話の答えだけ `N_eq_sel − prod`", "N_eq_sel", PRO)]
for kind, lab, a, b in T17:
    for K in KS:
        w(eff_line(kind, lab, a, b, K))
w("")
w("### 3.5 生徒の段：規則（R5．疑似ラベル X の生徒 − 積の疑似ラベルの生徒．どちらも設定を選び直した点）")
w("")
w("融合の列は，元の 4 コーパスが声・顔・言葉の融合，CREMA-D が声・顔の融合である．「—」はその生徒が無いことを表す（RAVDESS は声だけ．CREMA-D に言葉は無い．`M_F6_mixsel` の疑似ラベルは RAVDESS に無い）．")
w("")
w(HEAD)
w(HSEP)
TEACH_LAB = {"M_F6_mixsel": "提案の手法 `M_F6_mixsel`", "MN_F2K_sel": "`MN_F2K_sel`", "MN_F6K_sel": "`MN_F6K_sel`", "MN_eq_sel": "`MN_eq_sel`"}
for a in R5_TEACH16 + R5_TEACH17:
    for ro in ROLES:
        for K in KS:
            w(eff_line("R5", f"{ROLE_JP[ro]} × {TEACH_LAB[a]}", a, PRO, K, role=ro))
w("")
w("### 3.6 生徒の段：規則でない比べ")
w("")
w("「今の提案 − K 本だけ」は，積の疑似ラベルで学び直した分類器と K 本だけの分類器の差で，引き戻しの強さの選び方で甘く出る向きがある（0 節）．"
  "`MS_eq_sel − prod`・`MS_eq_tied − prod`（対照の疑似ラベルの生徒 − 積の疑似ラベルの生徒）は `students_effects.csv` にあり，ここには並べない．")
w("")
w(HEAD)
w(HSEP)
for ro in ROLES:
    for K in KS:
        w(eff_line("記述", f"{ROLE_JP[ro]} × 今の提案 − K 本だけ `prod − none`", PRO, "none", K, role=ro))
for b, lab in (("MS_eq_sel", "`MN_eq_sel − MS_eq_sel`"), ("MS_eq_tied", "`MN_eq_sel − MS_eq_tied`")):
    for ro in ROLES:
        for K in KS:
            w(eff_line("帰し方", f"{ROLE_JP[ro]} × {lab}", "MN_eq_sel", b, K, role=ro))
w("")
w("### 表の下の注（3.2〜3.6 節に共通）")
w("")
w("- K どうしの差の区間は作っていない．K の間の判定の違いを差の証拠として読まない．3 つの K は入れ子で，テストは同じである．")
w("- RAVDESS の区間は自由度 18 の jackknife で，控えめに広く出る（0 節）．")
w("- 選ばれた格子の点は 5 節．")
w("")

# ---- 4 節
w("## 4. 外れた規則の標準誤差と自由度")
w("")
w("標準誤差（括弧は自由度．「引き直し」は塊の重みの引き直しで，自由度を持たない）．`effects.csv`（adj の層）と `students_effects.csv` の `se`・`df`．自由度はコーパスで決まり，AFEW 13，IEMOCAP 3，MELD 7，RAVDESS 18 である．")
w("")
w("### 推定時の答え（R2 の外れた規則）")
w("")
w("| 比べ | K | " + " | ".join(C6) + " |")
w("|---|---|" + "---|" * len(C6))
for r in rulesA[~rulesA.hit].itertuples():
    x = adj[(adj.a == r.a) & (adj.b == r.b) & (adj.K == r.K)].set_index("c")
    w(f"| `{r.a} − {r.b}` | {int(r.K)} | " + " | ".join(sedf(x.loc[c]) if c in x.index else "—" for c in C6) + " |")
must(set(rulesA[~rulesA.hit].rule) == {"R2"}, "R1 に外れがある")
w("")
w("### 生徒の段（R5 の外れた規則）")
w("")
w("| 生徒 × 疑似ラベル | K | " + " | ".join(C6) + " |")
w("|---|---|" + "---|" * len(C6))
for a in R5_TEACH16 + R5_TEACH17:
    for ro in ROLES:
        for K in KS:
            if bool(srow(ro, a, PRO, K).hit):
                continue
            x = se[(se.role == ro) & (se.a == a) & (se.b == PRO) & (se.K == K)].set_index("c")
            w(f"| {ROLE_JP[ro]} × `{a}` | {K} | " + " | ".join(sedf(x.loc[c]) if c in x.index else "—" for c in C6) + " |")
w("")

# ---- 5 節
w("## 5. 選ばれた点")
w("")
w("### 5.1 提案の手法 `M_F6_mixsel`（`sel.csv`）")
w("")
w("n0 は K 本で測った水準を戻し先へ寄せる強さ（ラベルの本数），β は相性を使うか（1 は使う），戻し先は eq（分類器の平均）か ext（ソースで見込んだ水準）．16 点から選ぶ．"
  f"どの組も 40 単位で選んだ．n0 = 0 の点は，式の上で戻し先が効かない（水準は K 本で測った値のまま．`dai16_probe.mix_rel`）ので，mix2（eq）と mix3（ext）は同じ答えになる"
  f"（単位の保存の {MIX23_N:,} 行で UAR の差の最大は {MIX23_DIFF:g}）．このため mix2 を選んだ {N_MIX2} 組は同点が 2 つあり，並びで先の mix2 になった（`n_tied`）．"
  "対照 `M_F6n_mixsel` は同じ番号の点で分類器 × 感情の成分を消した版である．")
w("")
w("| 評価するコーパス | K = 5 | K = 10 | K = 25 | 選ぶのに使った仮のターゲット |")
w("|---|---|---|---|---|")
for c in C6[:5]:
    x = [chosen_of(c, K, MIX) for K in KS]
    w(f"| {c} | " + " | ".join(mix_desc(q.chosen) for q in x) + f" | {jp(x[0].pseudo_targets)} |")
w("")
w("### 5.2 実験 17 の推定時の版（`sel.csv`）")
w("")
w("引く範囲 mod はモダリティごと，joint はモダリティを合わせる形．数字は近い発話の本数，none／away は信頼度を使わない／別のソースで測った信頼度を使う．λ は足す答えの倍率（0.0625・0.25・1 の 3 点）．"
  "`MN_F6orc_sel` は `MN_F6K_sel` と同じ点，`MS_eq_tied` は `MN_eq_sel` と同じ λ である．RAVDESS を評価するときは 50 単位（仮のターゲットは元の 4 コーパスと CREMA-D），ほかは 40 単位で選んだ．")
w("")
w("| 評価するコーパス | 版 | K = 5 | K = 10 | K = 25 |")
w("|---|---|---|---|---|")
for c in C6:
    for v in ("MS_eq_sel", "MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel", "N_eq_sel"):
        w(f"| {c} | `{v}` | " + " | ".join(sel_desc(c, chosen_of(c, K, v).chosen) for K in KS) + " |")
w("")
w("### 5.3 生徒（`students_sel.csv`．ρ・a）")
w("")
w("ρ は疑似ラベルの重みの合計 ÷ 候補の本数（0.0625・0.25・1・4），a は K 本の倍率（1・4・16）．融合は，元の 4 コーパスが声・顔・言葉の融合，CREMA-D が声・顔の融合（点は 3 モダリティの融合の生徒で選んだ）．"
  "声の生徒は，積の疑似ラベルの側を 50 単位，提案の手法の側を 40 単位で選んだ（要点 5）．ほかの疑似ラベルと役の点は `students_sel.csv` にある．")
w("")
w("| 評価するコーパス | 生徒 | 疑似ラベル | K = 5 | K = 10 | K = 25 |")
w("|---|---|---|---|---|---|")
ST_OF = {"fusion": lambda c: P18.fus(c), "A": lambda c: "A", "V": lambda c: "V", "T": lambda c: "T"}
for c in C6:
    for ro in ROLES:
        st = ST_OF[ro](c)
        if st is None or st not in P18.kinds_of(c):
            continue
        teachers = [(PRO, "積")] + ([(MIX, "提案の手法")] if len(P18.MODS_OF[c]) >= 2 else []) + ([(t, f"`{t}`") for t in R5_TEACH17] if ro == "fusion" else [])
        for t, tl in teachers:
            w(f"| {c} | {ROLE_JP[ro]} | {tl} | " + " | ".join(st_point(c, K, st, t) for K in KS) + " |")
w("")
w("### 5.4 CREMA-D の融合（声・顔）の引き戻しの強さ（`tether.json`．C′・C_b）")
w("")
w(f"CREMA-D を仮のターゲットにした単位で使った値（評価のターゲット T ごと．同じ T と K の 3 モダリティの融合で選んだ値）と，元の規則（CREMA-D 自身のテストだけで選ぶ）で選んだ記録の値．{len(TS) * len(KS)} 組のうち {N_OWN_DIFF} 組で違う．")
w("")
w("| 評価のターゲット T | K = 5 | K = 10 | K = 25 |")
w("|---|---|---|---|")


def tpair(v) -> str:
    return f"{v[0]:g}・{v[1]:g}"


for t in TS:
    cells = []
    for K in KS:
        used, own = teth["tether"][t][f"AV|{K}"], teth["tether"][t][f"AV|{K}|own"]
        cells.append(f"{tpair(used)}（元の規則 {tpair(own)}{'．違う' if used != own else ''}）")
    w(f"| {t} | " + " | ".join(cells) + " |")
w("")

# ---- 6 節
w("## 6. まだ出していない量")
w("")
w("設計が結果の文に並べる，または計算の後に保存から出すと決めた量のうち，この第 1 版に無いものを一覧にする．"
  "保存から出す量は学び直さずに出せる見込みである（点検の報告 1 節は，単位ごとのソースの一覧 `srcs_*`・`sid_test`・`nn_test_*` が 300 単位すべてに保存されていることを確かめた）．出し方はまだ確かめていない．")
w("")
w("- **設計 8 節の 6（CREMA-D と RAVDESS の確認）**：(1) CREMA-D で文「It's eleven o'clock.」を除いたときの UAR，(2) RAVDESS の強さ（普通・強い）ごとの正解率，(3) RAVDESS の水準に天井の効果があるか（水準そのものは 3.3 節と表の文書にある），"
  "(4) 近いソースの発話のうち演技のコーパスから来た割合，(5) 自然な会話のコーパスだけを仮のターゲットにして選び直したときに選ぶ点が変わるか．")
w("- **設計 5b 節の仮定の表**：(a) ソースで測った信頼度と全ラベルで測った信頼度の，分類器 × 感情の成分の相関，(b) 感情ごとの残り方の比，(e) 実験 16 の仮定の表の (c)〜(f) に当たる量．"
  "(c) の勾配の最大・切片の残り・疑似ラベルの作り直しの一致と，(d) の実験 1 との一致は 0 節に書いた．")
w("- **選択を通した依存の数え**（設計 8 節．評価するコーパス c を除いて選び直すと，生徒の疑似ラベルの版の点が変わる組の数）．")
w("- **4 コーパスの実験 16・17 の値と並べた表**．設計 5 節の読み方（差は 3 つの変化が重なった結果で，どれの効果とも書かない）に従う表はまだ作っていない．")
w("- **利用条件の確認**（設計 9 節）：CREMA-D と RAVDESS の利用条件は原文で確かめていない（論文の段で申告する）．")
w("")
w("集計にあるが，この文書で引いていない量：感情ごとの再現率（`per_emotion.csv`），近いソースの発話の記述（`neighbors.csv`），raw の層と候補の層の差（`effects.csv` の `where` が `raw`・`pool` の行），"
  "生徒の UAR（`students_levels.csv`．実験 16 の版の分は表の文書の 4 節にある），融合の K = 25 以外の疑似ラベルの量（`students_conditions.csv`）．")
w("")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()
text = "\n".join(L) + "\n"
must("、" not in text and "。" not in text, "句読点に「、」「。」がある")
for bad in ("差が無い", "差がない", "再現した", "6 コーパスに広がった", "演技のコーパスを足した効果", "セル", "K で増え", "K で減"):
    must(bad not in text, f"使わないと決めた語がある：{bad}")
Path(args.out).write_text(text)
print(f"書いた：{args.out}（{len(L)} 行）")
