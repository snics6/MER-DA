"""実験 21（DAI-21．提案の評価の回の続き：学び直しを 5 コーパスに足し，CREMA-D を評価するコーパスに足す）の結果の文の下書きを，
結果ファイルから数値を取り出して書き出す．

出力：DAI/docs/実験21_結果の文の下書き_2026-11-15.md（--out で別の場所に書ける）．
読むもの：results/dai21_eval5/summary/*.csv・convergence.json，provenance.json，provenance_log.jsonl，
          students4/*.csv・*.npz と cremad/*.csv・*.npz（作り直しの照合），
          results/dai20_eval/summary/levels.csv・effects.csv・provenance.json と units/*.npz（元の 4 コーパスの推定時の答え・実験 1 の保存の判定・候補のラベル），
          results/dai18_probe/summary/students_effects.csv（設計 0 節が引く実験 18 の値の照合だけ），
          logs/dai21_*.log・dai21_eval5.done，DAI/frozen/dai21_frozen.json・dai20_frozen.json，DAI/run/dai21_eval5.sh，
          設計 DAI/docs/実験21_設計.md・検査と点検の報告 DAI/checks/DAI-21/*.md（13 本）・台帳 docs/seen-data.md，
          実験 18・20 の設計，実験 20 の結果の文，DAI/docs/論文のセッションへの伝言_2026-11-14.md，features/build_manifest_extra.py（文の記号の対応の 1 行），
          本体の作業の覚え書き（memory の dai21-retrain-cremad-2026-11-14.md．句があることだけを見る），
          照合（関門 2）の第 1 回〜第 4 回の報告 DAI/checks/DAI-21/claim.md・claim_v2.md・claim_v3.md・claim_v4.md（判定と指摘の数と，引く文だけを読む．報告の中の値は文の値に使わない），実験 20 の点検の報告 DAI/checks/DAI-20/code_post.md（引いた文の確かめ），
          DAI/docs・DAI/paper・DAI/slides・docs・slides・paper の .md と .tex（この結果の値の文字が出てくる文書を探す）
          （どれも，引いた文がそこにあることを確かめるだけ．報告の中の数は照合に使わない．報告の表の行を引いた 3.2・3.3 節は，報告の値として書く），
          表の文書 DAI/docs/実験21_結果の表_2026-11-15.md と書き出しの台本 md_tables_dai21.py（版の呼び名をそろえる．表の文書が今の集計と合うことと，
          最初に書き出した版から変わったのが注 3 個・桁 8 個・文の 2 行だけであることを確かめる），
          実験 20 の計算の後に使う台本の検査の報告 DAI/checks/DAI-20/code_post_tools.md・code_post_tools_v2.md（起動の台本の検査の記録），
          git の記録（コミットの時刻と差分），
          計算の後の解析（CREMA-D の感度の表）の出力 results/dai21_eval5/posthoc/*.csv・cremad_sentence_provenance.json と，その台本 dai21_posthoc_cremad_sentence.py（sha256 を取るだけ．import しない），
          ログ logs/dai21_posthoc_cremad_sentence.log，台本の検査の報告 DAI/checks/DAI-21/code_post_tools.md・code_post_tools_v2.md（判定と，引く文だけを読む）．
          感度の表の値は，cremad/*.npz の判定と発話の名前から別の書き方で作り直した値と照らしてから使う（話者の重みの引き直しで作る区間と，仮定の表の 7 列は作り直さない）．
結果ファイルは読むだけで，書き換えない（results/ には何も書かない．足した出力のファイルも無い）．GPU は使わない．
DAI/exp の台本は import しない（dai21_eval5.py を文字として読み，監視対象の一覧があることを確かめるだけ）．

文の中の数値と表は，すべてこの台本が結果ファイルから埋める（手で写さない）．
文が言う数えと大小は `must(...)` で結果ファイルと照らし，合わなければ止まる．
集計の値は，単位の保存（判定の配列）から別の書き方で作り直した値と照らしてから使う（照合を恒等式にしない）．
設計・点検の報告・台帳から引く文は，元の文書（台帳はその行）に同じ文があることを確かめる．
文を直すときは，この台本の中の文を直して流し直す．

使い方：.venv/bin/python DAI/docs/tools_pending/extract_dai21_text.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                    # GPU を使わない
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

ROOT = Path("/home/snics/MER-DA")
TOOLS = ROOT / "DAI" / "docs" / "tools_pending"
sys.path.insert(0, str(TOOLS))
import md_tables_dai21 as MT  # noqa: E402   版の呼び名（表の文書と同じ名前を使う）．読み込むだけ

R = ROOT / "results" / "dai21_eval5"
SM, U4DIR, UCDIR = R / "summary", R / "students4", R / "cremad"
R20 = ROOT / "results" / "dai20_eval"
OUT = ROOT / "DAI" / "docs" / "実験21_結果の文の下書き_2026-11-15.md"
TABLE_DOC = ROOT / "DAI" / "docs" / "実験21_結果の表_2026-11-15.md"
TEXT20 = ROOT / "DAI" / "docs" / "実験20_結果の文の下書き_2026-11-15.md"
DESIGN = ROOT / "DAI" / "docs" / "実験21_設計.md"
CK = ROOT / "DAI" / "checks" / "DAI-21"
LEDGER = ROOT / "docs" / "seen-data.md"
FROZEN = ROOT / "DAI" / "frozen" / "dai21_frozen.json"
FROZEN20 = ROOT / "DAI" / "frozen" / "dai20_frozen.json"
SCRIPT21 = ROOT / "DAI" / "exp" / "dai21_eval5.py"
TABLES_PY = TOOLS / "md_tables_dai21.py"
C4 = list(MT.C4)
CD = MT.CD
C5 = C4 + [CD]
KS = list(MT.KS)
ALL12 = [(c, K) for c in C4 for K in KS]
ALL3 = [(CD, K) for K in KS]
ALL15 = ALL12 + ALL3
EMO = ["Neutral", "Happy", "Sad", "Angry"]
FLOOR = 1.0                                                               # 表の印の差の下限（設計 4 節）
SEEDS = {"all": None, "0-1": (0, 1), "2-9": tuple(range(2, 10))}
EXTRA = [t for t, _ in MT.EXTRA]                                          # 疑似ラベルを替えた学び直し 5 つ
EX1 = [m for m, _ in MT.EX1]                                              # 実験 1 の手法のうち積の疑似ラベルを受け取る 3 つ
B5 = [f"b_{m}" for m, _ in MT.EX1 + MT.B_ONLY]                            # 実験 1 の 5 つの比較手法（保存の判定）
A2 = [v for v, _ in MT.A2]                                                # CREMA-D の比べる相手（積を含む 12 個）
ABL = [v for v, _ in MT.ABL]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()


def must(cond, msg: str) -> None:
    if not cond:
        raise SystemExit(f"文と結果ファイルが合わない：{msg}")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout


QUOTED: list[str] = []                                                    # 元の文書から引いた文（使わないと決めた語の検査から除く）


def quote(text: str, s: str, where: str) -> str:
    must(s in text, f"{where}に次の文が無い：{s}")
    QUOTED.append(s)
    return s


# ------------------------------------------------------------------ 読み込み

ef = pd.read_csv(SM / "effects.csv", low_memory=False, dtype={"folds_dropped": str})
for col_ in ("mark", "a_same_as", "b_same_as", "folds_dropped"):
    ef[col_] = ef[col_].astype(object).where(ef[col_].notna(), "").astype(str)
must(ef.same_pred.dtype == bool, "effects.csv の same_pred が真偽でない")
lv = pd.read_csv(SM / "levels.csv", low_memory=False)
for col_ in ("same_as", "same_pred"):
    lv[col_] = lv[col_].astype(object).where(lv[col_].notna(), "").astype(str)
pe = pd.read_csv(SM / "per_emotion.csv", low_memory=False)
cfm = pd.read_csv(SM / "confusion.csv")
sg = pd.read_csv(SM / "student_grid.csv")
wcd = pd.read_csv(SM / "weights_cremad.csv")
glc = pd.read_csv(SM / "grid_levels_cremad.csv")
conv = json.load(open(SM / "convergence.json"))
prov = json.load(open(R / "provenance.json"))
plog = [json.loads(x) for x in open(R / "provenance_log.jsonl")]
frozen = json.load(open(FROZEN))
frozen20 = json.load(open(FROZEN20))
prov20 = json.load(open(R20 / "provenance.json"))
lv20 = pd.read_csv(R20 / "summary" / "levels.csv", low_memory=False)
ef20 = pd.read_csv(R20 / "summary" / "effects.csv", low_memory=False)
ef20["mark"] = ef20["mark"].astype(object).where(ef20["mark"].notna(), "").astype(str)
design = DESIGN.read_text()
ledger = LEDGER.read_text()
code_post = (CK / "code_post.md").read_text()
table_doc = TABLE_DOC.read_text()
text20 = TEXT20.read_text()
src21 = SCRIPT21.read_text()
log_all = (ROOT / "logs" / "dai21_eval5.log").read_text()
done = (ROOT / "logs" / "dai21_eval5.done").read_text().strip()

EF = {(r.a, r.b, r.c, int(r.K), r.seeds): r for r in ef.itertuples()}
LV = {(r.version, r.c, int(r.K), r.seeds): r for r in lv.itertuples()}
EF20 = {(r.a, r.b, r.c, int(r.K), r.seeds): r for r in ef20.itertuples()}
LV20 = {(r.version, r.c, int(r.K), r.seeds): r for r in lv20.itertuples()}

# ------------------------------------------------------------------ 数の書き方


def fest(x: float) -> str:
    """差：小数 2 桁．丸めると ±1.00 か 0.00 になって，1.0 点の境目や符号が読めないときは桁を足す．"""
    s = f"{x:+.2f}"
    if (abs(float(s)) == 1.0 and abs(x) != 1.0) or (float(s) == 0.0 and x != 0.0):
        for nd in (3, 4, 5, 6, 7, 8):
            s = f"{x:+.{nd}f}"
            if abs(float(s)) not in (1.0, 0.0):
                break
    return s.replace("-", "−")


def fend(x: float) -> str:
    """区間の端：小数 1 桁．0.05 より 0 に近いときは符号が読めるよう桁を足す．"""
    s = f"{x:+.1f}"
    if abs(x) < 0.05 and x != 0:
        for nd in (2, 3, 4, 5, 6, 7, 8):
            s = f"{x:+.{nd}f}"
            if float(s) != 0.0:
                break
    return s.replace("-", "−")


def fpt(x: float) -> str:
    """点の値の引き算（区間の無い差）：符号つきの小数 2 桁．0.00 に丸まるときは桁を足す．"""
    s = f"{x:+.2f}"
    if float(s) == 0.0 and x != 0.0:
        for nd in (3, 4, 5):
            s = f"{x:+.{nd}f}"
            if float(s) != 0.0:
                break
    return s.replace("-", "−")


def sci(x: float) -> str:
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    sup = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m}×10{sup}"


def same_counts(q) -> bool:
    """差の両側の版で，塊ごと・感情ごとの正解の数が全部の単位で同じ（集計の same_as が同じ群）．"""
    if not hasattr(q, "a_same_as"):                                        # 実験 20 の集計の行（same_counts の列を持つ）
        return bool(getattr(q, "same_counts", False))
    return q.a_same_as != "" and q.a_same_as == q.b_same_as


def klass(q) -> str:
    """差の読み：pp・nn は区間が 0 を含まず差が 1.0 点以上，p・n は区間が 0 を含まず差が 1.0 点未満，z は区間が 0 を含む，
    same は判定が全部の単位で同じ（same_pred），cnt は判定は違うが塊ごと・感情ごとの正解の数が全部の単位で同じ．"""
    if bool(getattr(q, "same_pred", False)):
        return "same"
    if same_counts(q):
        return "cnt"
    if q.lo > 0:
        return "pp" if q.est >= FLOOR else "p"
    if q.hi < 0:
        return "nn" if q.est <= -FLOOR else "n"
    return "z"


NEAR_TH = 0.1                                                               # 基準：引き直しで作る区間の，0 に近いほうの端が，0 から標準誤差の 0.1 倍以内
NEAR_USED: list = []                                                        # ＊ を付けて書いた区間（書いた順）


def near0(q, lo: str = "lo", hi: str = "hi", se: str = "se", method: str = "method") -> bool:
    """塊の重みの引き直しで作る区間（method が boot で始まる行）で，0 に近いほうの端が，0 から標準誤差の NEAR_TH 倍以内にある．"""
    s_ = getattr(q, se, None)
    return bool(str(getattr(q, method, "")).startswith("boot") and s_ is not None and s_ == s_ and s_ > 0 and min(abs(getattr(q, lo)), abs(getattr(q, hi))) < NEAR_TH * s_)


def ci(q, mark: bool = False, star: bool = True) -> str:
    """差と区間．star：引き直しで作る区間の端が 0 に近いとき，値の後に ＊ を付ける（ほかの文書の文字と比べるときは付けない）．"""
    k = klass(q)
    if k == "same":
        return "0（同じ判定）"
    if k == "cnt":
        return "0（正解の数が同じ）"
    m = {"pp": "↑", "nn": "↓"}.get(k, "") if mark else ""
    s_ = f"{fest(q.est)} [{fend(q.lo)}, {fend(q.hi)}]{m}"
    if star and near0(q):
        NEAR_USED.append(s_)
        s_ += "＊"
    return s_


def lci(r) -> str:
    return f"{r.uar:.2f} [{r.lo:.1f}, {r.hi:.1f}]"


NAME = {"prop": "提案の手法（推定時の答え）", "prod": "積", "stud_prop": "提案の手法（学び直した融合）", "stud_prod": "今の提案（この実験で解いた）",
        "joint_AVT": "今の提案（実験 1 の保存）", "none_AVT": "K 本だけの融合", "none_AV": "K 本だけの融合", "none_A": "K 本だけの声", "none_V": "K 本だけの顔",
        "ret_A": "微調整した声", "ret_V": "微調整した顔", "start_AV": "ソースの融合の分類器の平均（K = 0）", "full_AV": "上限（候補の全ラベルで学んだ融合．参照）",
        "start_A": "ソースの声の分類器の平均（K = 0）", "start_V": "ソースの顔の分類器の平均（K = 0）",
        "full_A": "上限（声）", "full_V": "上限（顔）", "M_F6_orc": "全ラベルで測った表（参照）", "M_F6_ext": "ソースだけで測った表（参考）",
        "stud_prop_sel": "提案の手法の疑似ラベル（疑似ラベルごとに選んだ点）", "stud_prod_sel": "積の疑似ラベル（疑似ラベルごとに選んだ点）"}
NAME.update({f"stud_{t}": f"{lab}の疑似ラベル" for t, lab in MT.EXTRA})
NAME.update({f"b_{m}": lab for m, lab in MT.EX1 + MT.B_ONLY})
NAME.update({f"{m}_prod": f"{lab}（積の疑似ラベル．この実験で解いた）" for m, lab in MT.EX1})
NAME.update({f"{m}_prop": f"{lab}（提案の手法の疑似ラベル）" for m, lab in MT.EX1})
NAME.update({v: lab for v, lab in MT.A2 + MT.ABL})


def nm(v: str) -> str:
    return NAME[v.split("@")[0]]


def nmc(v: str) -> str:
    return f"{nm(v)}（`{v}`）"


# ------------------------------------------------------------------ 取り出し

def q(a: str, b: str, c: str, K: int, seeds: str = "all"):
    must((a, b, c, K, seeds) in EF, f"effects.csv に {a}−{b} {c} K={K} {seeds} が無い")
    return EF[(a, b, c, K, seeds)]


def q20(a: str, b: str, c: str, K: int, seeds: str = "all"):
    must((a, b, c, K, seeds) in EF20, f"実験 20 の effects.csv に {a}−{b} {c} K={K} {seeds} が無い")
    return EF20[(a, b, c, K, seeds)]


def level(v: str, c: str, K: int, seeds: str = "all"):
    must((v, c, K, seeds) in LV, f"levels.csv に {v} {c} K={K} {seeds} が無い")
    return LV[(v, c, K, seeds)]


def fus(c: str) -> str:
    return "none_AV" if c == CD else "none_AVT"


def eget(a, b, seeds: str = "all"):
    """a・b は版の名前か，コーパスから版の名前を返す関数．"""
    return lambda c, K: q(a(c) if callable(a) else a, b(c) if callable(b) else b, c, K, seeds)


def groups(get, pairs) -> dict:
    g = {k: [] for k in ("pp", "p", "z", "n", "nn", "same", "cnt")}
    for c, K in pairs:
        g[klass(get(c, K))].append((c, K))
    return g


def order(pairs):
    return sorted(pairs, key=lambda x: (C5.index(x[0]), x[1]))


def by_corpus(pairs) -> str:
    """条件の名前：「AFEW の K = 5・10・25，MELD の K = 25」．"""
    out = []
    for c in C5:
        ks = [K for c2, K in order(pairs) if c2 == c]
        if ks:
            out.append(f"{c} の K = {'・'.join(map(str, ks))}")
    return "，".join(out)


def vals(pairs, get) -> str:
    """条件の名前と値：「AFEW の K = 5 で +2.13 [+1.0, +3.3]，K = 10 で …」．"""
    out = []
    for c in C5:
        ks = [K for c2, K in order(pairs) if c2 == c]
        if ks:
            out.append(f"{c} の " + "，".join(f"K = {K} で {ci(get(c, K))}" for K in ks))
    return "，".join(out)


def read_short(get, pairs) -> str:
    """差の読み．区間が 0 を含まない条件（符号と 1.0 点の境目で分ける）と，区間が 0 を含み点推定が負の条件は，名前と値を書く．
    区間が 0 を含み点推定が正の条件は名前だけを書く（値は表にある）．"""
    g = groups(get, pairs)
    s = []
    if g["pp"]:
        s.append("区間が 0 を含まず +1.0 点以上：" + vals(g["pp"], get))
    if g["nn"]:
        s.append("区間が 0 を含まず −1.0 点以下：" + vals(g["nn"], get))
    if g["p"]:
        s.append("区間が 0 を含まず 0〜+1.0 点：" + vals(g["p"], get))
    if g["n"]:
        s.append("区間が 0 を含まず −1.0〜0 点：" + vals(g["n"], get))
    if not (g["pp"] or g["nn"] or g["p"] or g["n"]):
        s.append("区間が 0 を含まない条件は無い")
    zneg = [x for x in g["z"] if get(*x).est < 0]
    zpos = [x for x in g["z"] if get(*x).est >= 0]
    if zneg:
        s.append("区間が 0 を含み点推定が負：" + vals(zneg, get))
    if zpos:
        s.append("区間が 0 を含み点推定が正：" + by_corpus(zpos))
    if g["same"]:
        s.append("2 つの版が同じ判定（差は 0）：" + by_corpus(g["same"]))
    if g["cnt"]:
        s.append("2 つの版で判定は違う発話があるが，塊ごと・感情ごとの正解の数が全部の単位で同じ（差は 0）：" + by_corpus(g["cnt"]))
    return "．".join(s) + "．"


def short(pairs) -> str:
    out = []
    for c in C5:
        ks = [K for c2, K in order(pairs) if c2 == c]
        if ks:
            out.append(f"{c} {'・'.join(map(str, ks))}")
    return "，".join(out) if out else "—"


CAT_HEAD = ("| 0 を含まず +1.0 点以上 | 0 を含まず 0〜+1.0 点 | 0 を含む（点推定は正） | 0 を含む（点推定は負） | 0 を含まず −1.0〜0 点 | 0 を含まず −1.0 点以下 | 差が 0（同じ判定か，正解の数が同じ） |")
CAT_SEP = "|---|---|---|---|---|---|---|"


def cat_cells(get, pairs) -> str:
    g = groups(get, pairs)
    zp = [x for x in g["z"] if get(*x).est >= 0]
    zn = [x for x in g["z"] if get(*x).est < 0]
    return " | ".join(short(x) for x in (g["pp"], g["p"], zp, zn, g["n"], g["nn"], g["same"] + g["cnt"]))


def kjoin(xs) -> str:
    return "・".join(str(x) for x in xs)


# ================================================================== 文を書く前の照合

# ---- 出どころ
must([p["stage"] for p in plog] == ["students4", "cremad", "summary"], "provenance_log.jsonl の並び")
must(plog[2] == prov, "provenance.json が provenance_log.jsonl の 3 行目と違う")
must(all(p["dirty"] is False for p in plog), "dirty")
must(len({p["design_sha256"] for p in plog}) == 1 and sha(DESIGN) == prov["design_sha256"], "設計の sha256")
must(len({p["this_script_sha256"] for p in plog}) == 1 and sha(SCRIPT21) == prov["this_script_sha256"], "台本の sha256")
must(all(p["scripts"] == prov["scripts"] for p in plog) and all(sha(ROOT / "DAI" / "exp" / n) == h for n, h in prov["scripts"].items()), "部品の sha256")
must(len({p["frozen_sha256"] for p in plog}) == 1 and sha(FROZEN) == prov["frozen_sha256"], "固定のファイルの sha256")
must(len({p["frozen20_sha256"] for p in plog}) == 1 and sha(FROZEN20) == prov["frozen20_sha256"], "実験 20 の固定のファイルの sha256")
must(all(p["inputs_sha256"] == prov["inputs_sha256"] for p in plog) and len(prov["inputs_sha256"]) == 75, "入力の sha256")
must(all(p["dai20_provenance_sha256"] == sha(R20 / "provenance.json") for p in plog), "実験 20 の由来の記録の sha256")
must(plog[0]["dai1_eval_units_digest"].startswith("600:") and plog[1]["dai1_eval_units_digest"] == "" and prov["dai1_eval_units_digest"] == "", "実験 1 の単位の保存をまとめた値の置き場")
C_S4, C_CD, C_SUM = plog[0]["commit"], plog[1]["commit"], plog[2]["commit"]
must(conv["students4"]["commits"] == [C_S4] and conv["cremad"]["commits"] == [C_CD] and conv["summary_commit"] == C_SUM and len({C_S4, C_CD, C_SUM}) == 3, "3 つの段のコミット")
WATCH = list(prov["consts"]["WATCH"])
must(all(p["consts"] == prov["consts"] for p in plog), "定数の記録")
must('WATCH = tuple(f"DAI/exp/{n}" for n in SCRIPTS) + ("DAI/exp/dai21_eval5.py", "DAI/frozen/dai21_frozen.json", "DAI/frozen/dai20_frozen.json", "DAI/splits", "measure", "features",' in src21,
     "台本の監視対象の一覧")
must(WATCH == [f"DAI/exp/{n}" for n in prov["scripts"]] + ["DAI/exp/dai21_eval5.py", "DAI/frozen/dai21_frozen.json", "DAI/frozen/dai20_frozen.json", "DAI/splits", "measure", "features",
                                                           "data/metadata", "requirements.txt", "DAI/docs/実験21_設計.md"], "監視対象の一覧の中身")
for a_, b_ in ((C_S4, C_CD), (C_CD, C_SUM), (C_S4, C_SUM), (C_S4, "HEAD")):
    must(git("diff", "--stat", a_, b_, "--", *WATCH).strip() == "", f"コミット {a_[:7]} と {b_[:7]} のあいだで計算に関わるファイルが変わっている")
N_BETWEEN = {k: int(git("rev-list", "--count", f"{a_}..{b_}")) for k, (a_, b_) in {"s4_cd": (C_S4, C_CD), "cd_sum": (C_CD, C_SUM), "s4_sum": (C_S4, C_SUM)}.items()}
must(N_BETWEEN["s4_cd"] + N_BETWEEN["cd_sum"] == N_BETWEEN["s4_sum"], "コミットの数")
BETWEEN_FILES = sorted({x for x in git("-c", "core.quotepath=false", "diff", "--name-only", C_S4, C_SUM).strip().split("\n") if x})      # 3 つの段のあいだに変わったファイル
T_S4, T_CD, T_SUM = (p["time"] for p in plog)
must(T_S4 < T_CD < T_SUM < done, "3 つの段と done の時刻の順")


def ctime(commit: str) -> str:
    return git("log", "-1", "--format=%ad", "--date=format:%Y-%m-%d %H:%M:%S", commit).strip()


def last_commit(path: str) -> tuple[str, str]:
    h, t = git("log", "-1", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M:%S", "--", path).strip().split(" ", 1)
    return h, t


D21 = {"設計": "DAI/docs/実験21_設計.md", "台本": "DAI/exp/dai21_eval5.py", "固定のファイル": "DAI/frozen/dai21_frozen.json", "起動の台本": "DAI/run/dai21_eval5.sh"}
D21_LAST = {k: last_commit(p) for k, p in D21.items()}
must(git("status", "--porcelain", "--", *D21.values()).strip() == "", "設計・台本・固定のファイル・起動の台本に，コミットしていない変更がある")
OPEN20 = "2026-11-15 03:21:00"                                            # 実験 20 の結果を開いた時刻（台帳）
must(all(t < OPEN20 for _, t in D21_LAST.values()), f"設計・台本・固定のファイルが，実験 20 の結果を開いた後に変わっている（文を直す）：{D21_LAST}")
must(D21_LAST["設計"][0] == "918ebea" and D21_LAST["固定のファイル"][0] == "07c5d67" and C_S4.startswith("07c5d67"), "設計と固定のファイルの最後のコミット")
must(hashlib.sha256(git("show", "918ebea:DAI/docs/実験21_設計.md").encode()).hexdigest() == prov["design_sha256"], "918ebea の設計の sha256")
must(frozen["commit"].startswith("86c38f1") and D21_LAST["台本"][0] == "86c38f1", "固定のファイルを作ったコミットと台本の最後のコミット")

# ---- 固定のファイル
SGRID = [(float(r), int(a)) for r, a in frozen["sgrid"]]
must(SGRID == [(float(r), int(a)) for r, a in prov["SGRID"]] == [(r, a) for r in (0.0625, 0.25, 1.0, 4.0) for a in (1, 4, 16)], "学び直しの設定の 12 点")
GRID = [(float(a), float(b), p) for a, b, p in prov20["GRID"]]
INF = float("inf")
must(GRID == [(n0, b, pr) for n0 in (0.0, 20.0, 80.0, INF) for b in (0.0, 1.0) for pr in ("eq", "ext")], "提案の格子の 16 点")
for r in glc.itertuples():                                                # CREMA-D の格子の表の n0・β・戻し先が，同じ番号の格子の点と合う
    must((float(r.n0), float(r.beta), r.prior) == GRID[int(r.grid)], f"grid_levels_cremad.csv の点 {r.grid}")


def same_but(i: int, n0=None, beta=None) -> int:
    a, b, p = GRID[i]
    return GRID.index((a if n0 is None else n0, b if beta is None else beta, p))


FZ4 = {(c, K): frozen["four"][c][str(K)] for c, K in ALL12}
FZC = {K: frozen["cremad"][str(K)] for K in KS}
for (c, K), z_ in FZ4.items():
    must(int(z_["prop_grid"]) == int(frozen20["choice"][c][str(K)]["grid"]), f"提案の点が実験 20 の固定のファイルと違う {c} K={K}")
must(all(int(z_["student_main"]) == SGRID.index((1.0, 1)) for z_ in FZ4.values()), "4 コーパスの主の点が ρ = 1，a = 1 でない")
must([SGRID[int(FZC[K]["student_main"])] for K in KS] == [(1.0, 16), (4.0, 1), (4.0, 1)], "CREMA-D の主の点")
must(all(GRID[int(FZC[K]["prop_grid"])] == (0.0, 1.0, "eq") and int(FZC[K]["n_units"]) == 40 and CD not in FZC[K]["corpora_in_units"] for K in KS), "CREMA-D の提案の点と選んだ単位")
must(all(FZC[K]["corpora_in_units"] == ["AFEW", "IEMOCAP", "MELD", "MOSEI", "RAVDESS"] for K in KS), "CREMA-D の設定を選んだ単位に現れたコーパス")
for p_, h_ in frozen["source_sha256"].items():
    must(sha(ROOT / p_) == h_, f"固定のファイルの元（{p_}）が変わっている")
GM = {(c, K): int(FZ4[(c, K)]["student_main"]) for c, K in ALL12} | {(CD, K): int(FZC[K]["student_main"]) for K in KS}        # 主の点
GI = {(c, K): int(FZ4[(c, K)]["prop_grid"]) for c, K in ALL12} | {(CD, K): int(FZC[K]["prop_grid"]) for K in KS}             # 提案の格子の点
GS = {(c, K): int(FZ4[(c, K)]["student_prop_sel"]) for c, K in ALL12} | {(CD, K): int(FZC[K]["student_prop_sel"]) for K in KS}
GPS = {(c, K): int(FZ4[(c, K)]["student_prod_sel"]) for c, K in ALL12}
# 選んだ学び直しの設定の点が，格子の端（ρ = 0.0625・4，a = 1・16）に当たった数を，疑似ラベルごとに数える（固定のファイルの点と，student_grid.csv の ρ・a）
must(all(SGRID[int(r.grid)] == (float(r.rho), int(r.a)) for r in sg.itertuples()), "student_grid.csv の ρ・a が，格子の番号と合わない")


def edge_counts(points: list) -> dict:
    return dict(n=len(points), rho_hi=sum(SGRID[g][0] == 4.0 for g in points), rho_lo=sum(SGRID[g][0] == 0.0625 for g in points), a_edge=sum(SGRID[g][1] in (1, 16) for g in points))


EDGE_MAIN = edge_counts([GM[k_] for k_ in ALL15])                              # 主の点（15 条件）
EDGE_PROP = edge_counts([GS[k_] for k_ in ALL15])                              # 提案の手法の疑似ラベルのために選んだ点（15 条件）
EDGE_PROD4 = edge_counts([GPS[k_] for k_ in ALL12])                            # 積の疑似ラベルのために選んだ点（元の 4 コーパスの 12 条件）
EDGE_PROD15 = edge_counts([GPS[k_] for k_ in ALL12] + [GM[(CD, K)] for K in KS])   # 同上に，CREMA-D の主の点（積の疑似ラベルのために同じ規則で選んだ点）を足した 15 条件
must(EDGE_PROP["rho_hi"] + EDGE_PROD15["rho_hi"] + EDGE_PROP["rho_lo"] + EDGE_PROD15["rho_lo"] + sum(SGRID[GM[k_]][0] in (0.0625, 4.0) for k_ in ALL12) == frozen["n_student_points_at_rho_edge"],
     "ρ が格子の端の点の数が，固定のファイルの記録と合わない")
# 設計 3.1 節の点の記述と固定のファイル
DQ_MAIN4 = quote(design, "4 コーパス：実験 1 が仮のターゲットで選んで固定した点（どのコーパス・K でも ρ ＝ 1，a ＝ 1）．", "設計 3.1 節")
DQ_MAINCD = quote(design, "CREMA-D：実験 18 が，CREMA-D を触らずに残した単位だけで選んだ点（K = 5 は ρ ＝ 1・a ＝ 16，K = 10・25 は ρ ＝ 4・a ＝ 1）．"
                          "この点は，その単位の 3 モダリティの融合の分類器で選んだ値を，声と顔の融合に借りたものである．", "設計 3.1 節")
DQ_PROPCD = quote(design, "設定は，実験 18 が CREMA-D を触らずに残した 40 単位で選んだ点（3 つの K とも n0 ＝ 0，相性を使う）．", "設計 3.3 節")
must(GI[("AFEW", 25)] == 6 and GI[("MELD", 5)] == 7, "設計 3.1 節が書く AFEW の K = 25 と MELD の K = 5 の提案の点")

# ---- 台帳：句がどの行にあるかまで確かめる
LROWS = ledger.split("\n")
row21 = [x for x in LROWS if x.startswith("| DAI-21（")]
row20 = [x for x in LROWS if x.startswith("| DAI-20（実験 16 の提案の手法の評価の回")]
must(len(row21) == 1 and len(row20) == 1, "台帳の DAI-21・DAI-20 の行")
ROW21, ROW20 = row21[0], row20[0]

# ---- 点検の報告
must("## 判定\n\n通す．格子の全部の単位がそろい，重複・NaN・無限大は 0 で，台本・設計・固定のファイルは検査を通したものと同じであり，"
     "保存した W・b からの判定の作り直しと，水準・差・区間・印の独立な作り直しがすべて保存の集計と一致した．残る指摘は記録だけである．" in code_post, "点検の報告の判定")
N_REC = code_post.count("### [記録 ")
must(N_REC == 4, "点検の報告の記録の数")
must("この報告には UAR と差の値を書かない．" in code_post, "点検の報告が書く量の文")

# ---- 集計の表の作り：行の数，差の対，印，区間の作り方
P4 = sorted(set(zip(ef[ef.part == "4corpora"].a, ef[ef.part == "4corpora"].b)))
PC = sorted(set(zip(ef[ef.part == "cremad"].a, ef[ef.part == "cremad"].b)))
must(len(P4) == 41 and len(PC) == 78 and len(ef) == 3 * (41 * 12 + 78 * 3) == 2178 and not ef.duplicated(["a", "b", "c", "K", "seeds"]).any() and set(ef.seeds) == set(SEEDS), "effects.csv の行と差の対")
must(set(ef[ef.part == "4corpora"].c) == set(C4) and set(ef[ef.part == "cremad"].c) == {CD}, "effects.csv の part とコーパス")
must(all((a, b, c, K, s) in EF for a, b in P4 for c, K in ALL12 for s in SEEDS) and all((a, b, CD, K, s) in EF for a, b in PC for K in KS for s in SEEDS), "差の対の欠け")
V4 = sorted(set(lv[lv.part == "4corpora"].version))
VC = sorted(set(lv[lv.part == "cremad"].version))
must(len(V4) == 24 and len(VC) == 53 and len(lv) == 3 * (24 * 12 + 53 * 3) == 1341 and not lv.duplicated(["c", "K", "version", "seeds"]).any() and set(lv.seeds) == set(SEEDS), "levels.csv の行と版")
must(len(pe) == 4 * (24 * 12 + 53 * 3) == 1788 and bool(pe.recall.notna().all()), "per_emotion.csv の行")
N_INT = dict(ef=len(ef), ef_all=int((ef.seeds == "all").sum()), lv=len(lv), pe=len(pe))
must(N_INT["ef_all"] == 726, "差の区間の数")
DQ_NINT = quote(design, "**区間の数**：差の対は 4 コーパスで 41 個（× 12），CREMA-D で 78 個（× 3），計 726 個．種で分けた値（種 0・1 と種 2〜9）を入れるとその 3 倍．K の 3 つは同じテストと入れ子の K 本を共有する．", "設計 4 節")
DQ_MARK = quote(design, "**表の印**（↑・↓．区間が 0 を含まず，差が 1.0 点以上）は読みやすさのためで，判定ではない．印の無い行は「同じ水準」を意味しない．", "設計 4 節")
mk_ = np.where((ef.lo > 0) & (ef.est >= FLOOR), "↑", np.where((ef.hi < 0) & (ef.est <= -FLOOR), "↓", ""))      # 印を設計 4 節の文から作り直す
must(bool((mk_ == ef["mark"].to_numpy()).all()), "印の列が設計 4 節の文と合わない")
must(bool(((ef.lo <= ef.est + 1e-12) & (ef.est <= ef.hi + 1e-12)).all()) and bool(((lv.lo <= lv.uar + 1e-12) & (lv.uar <= lv.hi + 1e-12)).all()), "区間が点推定を挟まない行がある")
METH = {c: sorted(set(ef[ef.c == c].method)) for c in C5}
must(METH == {"AFEW": ["jk+seed"], "IEMOCAP": ["jk+seed"], "MOSEI": ["boot_cl+seed"], "MELD": ["jk+seed"], CD: ["boot_cl+seed"]}, f"区間の作り方：{METH}")
must({c: sorted(set(lv[lv.c == c].method)) for c in C5} == METH, "水準の区間の作り方")
DF = {c: sorted(set(ef[ef.c == c].df.dropna().astype(int))) for c in C5}
must(DF == {"AFEW": [13], "IEMOCAP": [3], "MOSEI": [], "MELD": [7], CD: []}, f"自由度：{DF}")
FD = {c: sorted(set(ef[ef.c == c].folds_dropped)) for c in C5}
must(FD == {"AFEW": ["0"], "IEMOCAP": ["0,1,2,3,4"], "MOSEI": [""], "MELD": [""], CD: [""]}, f"jackknife で除かれる分割：{FD}")
must(bool((ef[ef.c == "IEMOCAP"].v_seed == 0).all()), "IEMOCAP の種の項")
NFR = {s_: sorted(set(ef[ef.seeds == s_].n_fr)) for s_ in SEEDS}
must(NFR == {"all": [50], "0-1": [10], "2-9": [40]} and {s_: sorted(set(lv[lv.seeds == s_].n_fr)) for s_ in SEEDS} == NFR, f"単位の数：{NFR}")

# ---- 単位の保存（判定の配列）から，水準 1,341 行と差 2,178 行の点推定，同じ判定・同じ数の印，混同行列，感情ごとの全体の再現率を作り直す
#      版の名前と保存の配列の対応は，集計の台本を読み込まずに，設計 3 節の表と固定のファイルから書き直した


def uar_pct(y: np.ndarray, pred: np.ndarray) -> float:
    return 100.0 * float(np.mean([(pred[y == e] == e).mean() for e in range(4) if (y == e).any()]))


UM: dict = {}        # (コーパス, K, 版) → (分割, K 本の引き方) ごとの UAR
PH: dict = {}        # 同 → 判定の並びの sha256（全部の単位をつないだもの）
CH: dict = {}        # 同 → 塊ごと・感情ごとの正解の数の sha256
CONF: dict = {}      # 同 → 混同行列（全部の単位の合計）
TQ: dict = {}        # (コーパス, K, 疑似ラベル) → 疑似ラベルの UAR（候補の上）と，学び直した分類器が疑似ラベルと判定が一致した割合（候補の上．主の点）
REPRO: dict = {}     # (コーパス, K, 手法) → 積の疑似ラベルで学び直した判定が実験 1 の保存と一致する割合
N_TEST: dict = {}
N_CL: dict = {}
NEST_BAD = 0
LEV14 = GRID.index((INF, 1.0, "eq"))
must(LEV14 == 14, "格子の点 14")


def add(c, K, v, f, r, y, inv, ncl, pred):
    pred = pred.astype(int)
    must(len(pred) == len(y), f"判定の長さがテストと違う：{c} K={K} {v} f{f} r{r}")
    UM.setdefault((c, K, v), np.full((5, 10), np.nan))[f, r] = uar_pct(y, pred)
    PH.setdefault((c, K, v), hashlib.sha256()).update(np.ascontiguousarray(pred.astype(np.int8)).tobytes())
    ok = pred == y
    cnt = np.zeros((ncl, 4), np.int64)
    np.add.at(cnt, (inv[ok], y[ok]), 1)
    CH.setdefault((c, K, v), hashlib.sha256()).update(cnt.tobytes())
    np.add.at(CONF.setdefault((c, K, v), np.zeros((4, 4))), (y, pred), 1)


def teacher_keys(i: int) -> dict:
    """疑似ラベルの名前 → まとめ方の版の名前（設計 3.1 節の表）．"""
    return {"prod": "prod", "prop": f"M_F6_mix{i}", "noprof": f"M_F6_mix{same_but(i, beta=0.0)}", "noint": f"M_F6n_mix{i}", "levelfree": f"M_F6_mix{LEV14}", "konly": "M_F6_K", "tcal": "tcal"}


def teacher_check(c, K, f, r, zs, zt, ypool, kpos):
    """zs：学び直しの保存，zt：まとめ方の保存（候補の上の判定 pred_pool_*．K 本を除いた候補の並び）．"""
    nk = np.setdiff1d(np.arange(len(ypool)), kpos)
    for t_, key in teacher_keys(GI[(c, K)]).items():
        tp = zt[f"pred_pool_K{K}_{key}"].astype(int)
        must(len(tp) == len(nk), f"候補の上の判定の長さ：{c} K={K} {t_}")
        sp = zs[f"predpool_K{K}_stud_{t_}_g{GM[(c, K)]}"].astype(int)
        d = TQ.setdefault((c, K, t_), dict(t=np.full((5, 10), np.nan), m=np.full((5, 10), np.nan), sp=np.full((5, 10), np.nan)))
        d["t"][f, r], d["m"][f, r], d["sp"][f, r] = uar_pct(ypool[nk], tp), float((sp[nk] == tp).mean()), uar_pct(ypool[nk], sp[nk])


F4 = sorted(U4DIR.glob("*.npz"))
FC = sorted(UCDIR.glob("*.npz"))
must(len(F4) == 200 and len(FC) == 50 and len(sorted(U4DIR.glob("*.csv"))) == 200 and len(sorted(UCDIR.glob("*.csv"))) == 50, "単位の csv と npz の数")
for p_ in F4:
    head_, r_ = p_.stem.rsplit("_r", 1)
    head_, f_ = head_.rsplit("_f", 1)
    c = head_.split("_", 1)[0]
    must(head_ == f"{c}_ALL" and c in C4, f"単位の名前：{p_.name}")
    f, r = int(f_), int(r_)
    z, z20 = np.load(p_, allow_pickle=False), np.load(R20 / "units" / p_.name, allow_pickle=False)
    y = z["y_test"].astype(int)
    must(bool((z20["y_test"].astype(int) == y).all()) and bool((z20["sid_test"].astype(str) == z["sid_test"].astype(str)).all()) and bool((z20["cl_test"] == z["cl_test"]).all()),
         f"実験 20 の単位のテストと違う：{p_.name}")
    _, inv = np.unique(z["cl_test"], return_inverse=True)
    ncl = int(inv.max()) + 1
    N_CL.setdefault((c, f), set()).add(ncl)                                  # テストの塊の数（分割ごと．K 本の引き方で変わらないはず）
    N_TEST.setdefault(c, {})[(f, r)] = np.bincount(y, minlength=4)
    ypool = z20["y_pool"].astype(int)
    kp_ = []
    for K in KS:
        i, gm = GI[(c, K)], GM[(c, K)]
        names = {"stud_prop": (z, f"pred_K{K}_stud_prop_g{gm}"), "stud_prod": (z, f"pred_K{K}_stud_prod_g{gm}"),
                 "stud_prop_sel": (z, f"pred_K{K}_stud_prop_g{GS[(c, K)]}"), "stud_prod_sel": (z, f"pred_K{K}_stud_prod_g{GPS[(c, K)]}"),
                 **{f"stud_{t_}": (z, f"pred_K{K}_stud_{t_}_g{gm}") for t_ in EXTRA},
                 **{f"{m_}_{t_}": (z, f"pred_K{K}_{m_}_{t_}") for m_ in EX1 for t_ in ("prod", "prop")},
                 "prop": (z20, f"pred_adj_K{K}_M_F6_mix{i}"), "prod": (z20, f"pred_adj_K{K}_prod"),
                 **{v: (z20, f"pred_raw_K{K}_{v}") for v in ["none_AVT", "joint_AVT"] + B5}}
        for v, (src_, kk) in names.items():
            add(c, K, v, f, r, y, inv, ncl, src_[kk])
        kpos = z20[f"kpos_K{K}"].astype(int)
        kp_.append(set(kpos.tolist()))
        teacher_check(c, K, f, r, z, z20, ypool, kpos)
        for m_, a_, b_ in [("joint", f"pred_K{K}_stud_prod_g{gm}", f"pred_raw_K{K}_joint_AVT")] + [(m_, f"pred_K{K}_{m_}_prod", f"pred_raw_K{K}_b_{m_}") for m_ in EX1]:
            REPRO.setdefault((c, K, m_), np.full((5, 10), np.nan))[f, r] = float((z[a_].astype(int) == z20[b_].astype(int)).mean())
    NEST_BAD += int(not (kp_[0] <= kp_[1] <= kp_[2]) or [len(x) for x in kp_] != [4 * K for K in KS])
CD_FACTS = dict(n_test={}, spk={}, sent_test=set(), sent_pool=set(), spk_overlap=0, srcs=set(), bad_sid=0, ieo_neutral={}, neutral={})
# 計算の後の解析（CREMA-D の感度の表）を，単位の npz から別の書き方で作り直すための材料
SENT12 = ["DFA", "IEO", "IOM", "ITH", "ITS", "IWL", "IWW", "MTI", "TAI", "TIE", "TSI", "WSI"]      # 発話の名前の 2 番目の欄（文の記号）
INT4 = ["XX", "LO", "MD", "HI"]                                           # 発話の名前の 4 番目の欄（強さ）
EMO_CODE = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3}
BRK_V = ["prop", "prod", "stud_prop", "stud_prod", "none_AV", "tcal", "none_A", "ret_A", "none_V", "ret_V", "start_A", "start_V", "start_AV", "full_A", "full_V", "full_AV"]   # 内訳の版
SCOR: dict = {}      # (K, 版) → {(分割, 引き方): 文 × 感情の正解の数 (12, 4)}
STOT: dict = {}      # (分割, 引き方) → 文 × 感情の発話の数 (12, 4)
UMN: dict = {}       # (K, 版) → 文 IEO を除いたテストの UAR（分割, 引き方）
ICOR: dict = {}      # (K, 版) → 強さ × 感情の正解の数（50 単位の合計）
ITOT = np.zeros((4, 4))                                                   # 強さ × 感情の発話の数（50 単位の合計）
KIEO: dict = {}      # (K, 分割, 引き方) → K 本の中の文 IEO の本数（感情ごと）
ASM_U: dict = {}     # K → 単位ごとの仮定の量
DCAP = float(np.sqrt(2.0) * sps.norm.ppf(0.98) - 1e-9)                    # d′ の上限（AUC を 0.98 で切る）
SRC_TAB: list = []   # 単位ごとの（ソースで測った表 ext，ソースの中で測った表 home，残り方）
TOP_ORC: list = []   # （単位，K）ごとの，候補の全ラベルで測った表で行の平均が最大の分類器（0 は声，1 は顔）


def pcorr(a_: np.ndarray, b_: np.ndarray) -> float:
    return float(np.corrcoef(a_.ravel(), b_.ravel())[0, 1]) if a_.std() > 1e-12 and b_.std() > 1e-12 else float("nan")


def rowc(X: np.ndarray) -> np.ndarray:
    return X - X.mean(1, keepdims=True)


for p_ in FC:
    head_, r_ = p_.stem.rsplit("_r", 1)
    head_, f_ = head_.rsplit("_f", 1)
    must(head_ == f"{CD}_ALL", f"単位の名前：{p_.name}")
    f, r = int(f_), int(r_)
    z = np.load(p_, allow_pickle=False)
    y = z["y_test"].astype(int)
    cl, sid, sidp, clp = z["cl_test"].astype(str), z["sid_test"].astype(str), z["sid_pool"].astype(str), z["cl_pool"].astype(str)
    _, inv = np.unique(cl, return_inverse=True)
    ncl = int(inv.max()) + 1
    N_TEST.setdefault(CD, {})[(f, r)] = np.bincount(y, minlength=4)
    must(all(x == "CRE_" + s.split("_")[0] for x, s in zip(cl, sid)) and all(x == "CRE_" + s.split("_")[0] for x, s in zip(clp, sidp)), "CREMA-D の塊が話者でない")
    CD_FACTS["n_test"][(f, r)] = len(y)
    CD_FACTS["spk"].setdefault(f, set()).update(cl.tolist())
    CD_FACTS["sent_test"] |= {s.split("_")[1] for s in sid}
    CD_FACTS["sent_pool"] |= {s.split("_")[1] for s in sidp}
    CD_FACTS["spk_overlap"] += len(set(cl.tolist()) & set(clp.tolist()))
    CD_FACTS["srcs"] |= {tuple(sorted(z[k_].astype(str).tolist())) for k_ in ("srcs_A", "srcs_V", "srcs_AV")}
    CD_FACTS["bad_sid"] += int("1076_MTI_SAD_XX" in set(sid.tolist()) | set(sidp.tolist()))
    if r == 0:
        ieo = np.array([s.split("_")[1] == "IEO" for s in sid])
        CD_FACTS["ieo_neutral"][f], CD_FACTS["neutral"][f] = int((ieo & (y == 0)).sum()), int((y == 0).sum())
        CD_FACTS.setdefault("ieo", {})[f] = int(ieo.sum())
    ypool = z["y_pool"].astype(int)
    if r == 0:                                                               # 候補（K 本を引く前のラベル付きの発話）の中の，文 IEO の発話の感情ごとの数
        CD_FACTS.setdefault("pool_ieo", {})[f] = np.bincount(ypool[np.array([s_.split("_")[1] == "IEO" for s_ in sidp])], minlength=4)
    parts_ = [s_.split("_") for s_ in sid]
    must(all(len(q_) == 4 and q_[1] in SENT12 and q_[3] in INT4 and EMO_CODE[q_[2]] == yy_ for q_, yy_ in zip(parts_, y)), f"発話の名前の形と感情 {p_.name}")
    must(all((q_[3] == "XX") == (q_[1] != "IEO" or q_[2] == "NEU") for q_ in parts_), f"強さと文の関係（文を直す） {p_.name}")
    sidx, iidx = np.array([SENT12.index(q_[1]) for q_ in parts_]), np.array([INT4.index(q_[3]) for q_ in parts_])
    keep = sidx != SENT12.index("IEO")                                     # 文 IEO を除いたテスト
    tot_ = np.zeros((12, 4))
    np.add.at(tot_, (sidx, y), 1)
    STOT[(f, r)] = tot_
    np.add.at(ITOT, (iidx, y), 1)
    De_, Dh_, ret_ = z["D_M_ext"].astype(float), z["D_M_home"].astype(float), z["ret_M"].astype(float)
    must(De_.shape == (2, 4) and Dh_.shape == (2, 4) and ret_.shape == (2,), f"ソースで測った表の形 {p_.name}")
    SRC_TAB.append((De_, Dh_, ret_))
    TOP_ORC += [int(z[f"D_M_orc_K{K}"].astype(float).mean(1).argmax()) for K in KS]
    kp_ = []
    for K in KS:
        i, gm = GI[(CD, K)], GM[(CD, K)]
        names = {v: f"pred_adj_K{K}_{v}" for v in A2 + ["M_F6_orc", "M_F6_ext"]}
        names.update({f"{v}@raw": f"pred_raw_K{K}_{v}" for v in A2})
        names.update({"prop": f"pred_adj_K{K}_M_F6_mix{i}", "prop@raw": f"pred_raw_K{K}_M_F6_mix{i}", "prop_noprof": f"pred_adj_K{K}_M_F6_mix{same_but(i, beta=0.0)}",
                      "prop_noprof_sel": f"pred_adj_K{K}_M_F6_mix{int(FZC[K]['noprof_sel'])}", "prop_noint": f"pred_adj_K{K}_M_F6n_mix{i}",
                      "prop_n0zero": f"pred_adj_K{K}_M_F6_mix{same_but(i, n0=0.0)}", "prop_home": f"pred_adj_K{K}_M_F6h_mix{i}", "prop_levelfree": f"pred_adj_K{K}_M_F6_mix{LEV14}",
                      "stud_prod": f"pred_K{K}_stud_prod_g{gm}", "stud_prop": f"pred_K{K}_stud_prop_g{gm}", "stud_prop_sel": f"pred_K{K}_stud_prop_g{GS[(CD, K)]}",
                      **{f"stud_{t_}": f"pred_K{K}_stud_{t_}_g{gm}" for t_ in EXTRA},
                      **{f"none_{k_}": f"pred_raw_K{K}_none_{k_}" for k_ in ("A", "V", "AV")}, "ret_A": f"pred_raw_K{K}_ret_A", "ret_V": f"pred_raw_K{K}_ret_V",
                      **{f"full_{k_}": f"pred_raw_K0_full_{k_}" for k_ in ("A", "V", "AV")}, **{f"start_{k_}": f"pred_raw_K0_start_{k_}" for k_ in ("A", "V", "AV")}})
        for v, kk in names.items():
            add(CD, K, v, f, r, y, inv, ncl, z[kk])
        kpos = z[f"kpos_K{K}"].astype(int)
        kp_.append(set(kpos.tolist()))
        teacher_check(CD, K, f, r, z, z, ypool, kpos)
        # 文ごとの正解の数，文 IEO を除いたテストの UAR，強さごとの正解の数
        for v, kk in names.items():
            pr_ = z[kk].astype(int)
            ok_ = pr_ == y
            cor_ = np.zeros((12, 4))
            np.add.at(cor_, (sidx[ok_], y[ok_]), 1)
            SCOR.setdefault((K, v), {})[(f, r)] = cor_
            UMN.setdefault((K, v), np.full((5, 10), np.nan))[f, r] = uar_pct(y[keep], pr_[keep])
            if v in BRK_V:
                np.add.at(ICOR.setdefault((K, v), np.zeros((4, 4))), (iidx[ok_], y[ok_]), 1)
        # K 本の中の文 IEO の本数（感情ごと）
        must(bool((np.bincount(ypool[kpos], minlength=4) == K).all()) and bool((np.array([EMO_CODE[s_.split("_")[2]] for s_ in sidp[kpos]]) == ypool[kpos]).all()), f"K 本の感情の数と名前 {p_.name} K={K}")
        KIEO[(K, f, r)] = np.bincount(ypool[kpos][np.array([s_.split("_")[1] == "IEO" for s_ in sidp[kpos]])], minlength=4)
        # 仮定の量（相関・水準・最も強いモダリティ・上限の割合．成分の相関と入れ替えの割合は作り直さない）
        Dk_, Do_ = z[f"D_M_K_K{K}"].astype(float), z[f"D_M_orc_K{K}"].astype(float)
        a_ = {}
        for r2_, D_ in (("home", Dh_), ("ext", De_), ("K", Dk_)):
            a_[f"profcorr_M_{r2_}"], a_[f"emocorr_M_{r2_}"] = pcorr(rowc(D_), rowc(Do_)), pcorr(D_.mean(0), Do_.mean(0))
            a_[f"topok_M_{r2_}"] = float(D_.mean(1).argmax() == Do_.mean(1).argmax())
        for r2_, D_ in (("home", Dh_), ("ext", De_), ("K", Dk_), ("orc", Do_)):
            a_[f"level_M_{r2_}_A"], a_[f"level_M_{r2_}_V"] = float(D_[0].mean()), float(D_[1].mean())
        a_["ret_A"], a_["ret_V"] = float(ret_[0]), float(ret_[1])
        a_["dk_ceiling_share"], a_["dk_zero_share"] = float((Dk_ >= DCAP).mean()), float((Dk_ <= 0).mean())
        a_["profcorr_row_A"], a_["profcorr_row_V"] = pcorr(rowc(De_)[0], rowc(Do_)[0]), pcorr(rowc(De_)[1], rowc(Do_)[1])
        ASM_U.setdefault(K, []).append(a_)
    NEST_BAD += int(not (kp_[0] <= kp_[1] <= kp_[2]) or [len(x) for x in kp_] != [4 * K for K in KS])
must(NEST_BAD == 0, f"K 本が入れ子でない単位がある：{NEST_BAD}")
must(all(len(x) == 1 for x in N_CL.values()), "テストの塊の数が K 本の引き方で違う")
MELD_CL = [next(iter(N_CL[("MELD", f)])) for f in range(5)]
must(MELD_CL[4] == 2 and min(MELD_CL[:4]) > 2, f"MELD の分割ごとの塊の数（文を直す）：{MELD_CL}")
must(sorted({v for c, K, v in UM if c != CD}) == V4 and sorted({v for c, K, v in UM if c == CD}) == VC and not any(np.isnan(x).any() for x in UM.values()), "作り直した版の一覧")


def fold_mean(M: np.ndarray, seeds) -> float:
    """分割ごとに K 本の引き方の平均を取り，分割で平均する．"""
    return float((M if seeds is None else M[:, list(seeds)]).mean(1).mean())


DEV_LV = max(abs(fold_mean(UM[(r.c, int(r.K), r.version)], SEEDS[r.seeds]) - r.uar) for r in lv.itertuples())
DEV_EF = max(abs(fold_mean(UM[(r.c, int(r.K), r.a)] - UM[(r.c, int(r.K), r.b)], SEEDS[r.seeds]) - r.est) for r in ef.itertuples())
must(DEV_LV < 1e-9 and DEV_EF < 1e-9, f"集計の点推定が，単位の保存から作り直した値と違う：水準 {DEV_LV:.2e}，差 {DEV_EF:.2e}")
# 同じ判定の印（差の表：両側の判定の並びが全部の単位で同じ）と，水準の表の群（same_pred：判定が同じ版の群．same_as：塊ごと・感情ごとの正解の数が同じ版の群）
must(all(bool(r.same_pred) == (PH[(r.c, int(r.K), r.a)].hexdigest() == PH[(r.c, int(r.K), r.b)].hexdigest()) for r in ef.itertuples()), "effects.csv の same_pred が，判定の並びから作り直した印と違う")
for col_, H_ in (("same_pred", PH), ("same_as", CH)):
    for c, K in ALL15:
        vs = V4 if c != CD else VC
        grp: dict = {}
        for v in vs:
            grp.setdefault(H_[(c, K, v)].hexdigest(), []).append(v)
        for g_ in grp.values():
            lab = {getattr(level(v, c, K), col_) for v in g_}
            must((len(g_) == 1 and lab == {""}) or (len(g_) > 1 and len(lab) == 1 and lab != {""} and next(iter(lab)) in g_), f"levels.csv の {col_} が作り直した群と違う：{c} K={K} {g_} {lab}")
        for sn in ("0-1", "2-9"):
            must(all(getattr(level(v, c, K, sn), col_) == getattr(level(v, c, K), col_) for v in vs), f"K 本の引き方で分けた行の {col_}")
must(all((r.a_same_as, r.b_same_as) == (level(r.a, r.c, int(r.K)).same_as, level(r.b, r.c, int(r.K)).same_as) for r in ef.itertuples()), "effects.csv の a_same_as・b_same_as")
# 正解の数が同じで，判定が違う版（点検の報告の記録 4）
CNT_ONLY = sorted((c, K, v) for (c, K, v) in UM if level(v, c, K).same_as != "" and level(v, c, K).same_pred == "")
must(CNT_ONLY == sorted([("MELD", 10, "b_kd"), ("MELD", 10, "kd_prod"), ("MELD", 25, "joint_AVT"), ("MELD", 25, "stud_prod"), ("MOSEI", 25, "b_kd"), ("MOSEI", 25, "kd_prod")]),
     f"正解の数が同じで判定が違う版（文を直す）：{CNT_ONLY}")
must(all(klass(r) != "cnt" or (abs(r.est) < 1e-12 and abs(r.se) < 1e-12) for r in ef.itertuples()) and all(not bool(r.same_pred) or (abs(r.est) < 1e-12 and abs(r.se) < 1e-12) for r in ef.itertuples()),
     "同じ判定・同じ数の行の差が 0 でない")
# 混同行列と，感情ごとの全体の再現率
must(len(cfm) == 16 * (6 * 12 + 5 * 3), "confusion.csv の行")
for (c, K, v), g_ in cfm.groupby(["c", "K", "version"]):
    M_ = CONF[(c, int(K), v)]
    must(len(g_) == 16 and all(M_[int(r.true), int(r.pred)] == r.n for r in g_.itertuples()), f"confusion.csv が判定から数え直した数と違う {c} K={K} {v}")
DEV_PE = max(abs(100 * CONF[(r.c, int(r.K), r.version)][int(r.emotion), int(r.emotion)] / CONF[(r.c, int(r.K), r.version)][int(r.emotion)].sum() - r.recall_pooled) for r in pe.itertuples())
must(DEV_PE < 1e-9 and all(CONF[(r.c, int(r.K), r.version)][int(r.emotion)].sum() == r.n for r in pe.itertuples()), f"per_emotion.csv の全体の再現率：{DEV_PE:.2e}")
N_PER_SEED = {c: sum(N_TEST[c][(f, 0)] for f in range(5)) for c in C5}
for c in C5:
    must(all((sum(N_TEST[c][(f, r)] for f in range(5)) == N_PER_SEED[c]).all() for r in range(10)), f"テストの発話の数が K 本の引き方で違う {c}")
# CREMA-D の単位の事実
must(sum(CD_FACTS["n_test"][(f, 0)] for f in range(5)) == 4899 == int(N_PER_SEED[CD].sum()), "CREMA-D のテストの発話の数")
CD_SPK = [len(CD_FACTS["spk"][f]) for f in range(5)]
must(len(set().union(*CD_FACTS["spk"].values())) == 91 == sum(CD_SPK) and min(CD_SPK) == 18 and max(CD_SPK) == 19, f"CREMA-D の話者の数：{CD_SPK}")
must(CD_FACTS["spk_overlap"] == 0 and CD_FACTS["bad_sid"] == 0 and CD_FACTS["srcs"] == {tuple(sorted(C4))}, "CREMA-D の話者の重なり・除いた発話・ソース")
must(len(CD_FACTS["sent_test"]) == 12 and CD_FACTS["sent_test"] == CD_FACTS["sent_pool"], "CREMA-D の文の数")
IEO_NEU, CD_NEU, IEO_ALL = sum(CD_FACTS["ieo_neutral"].values()), sum(CD_FACTS["neutral"].values()), sum(CD_FACTS["ieo"].values())

# ---- 単位の csv：検算の量，学び直しの格子，疑似ラベルの量
CS4 = pd.concat([pd.read_csv(p_, low_memory=False) for p_ in sorted(U4DIR.glob("*.csv"))], ignore_index=True)
CSC = pd.concat([pd.read_csv(p_, low_memory=False) for p_ in sorted(UCDIR.glob("*.csv"))], ignore_index=True)
must(len(CS4) == 200 * 108 and len(CSC) == 50 * 276 and not CS4.duplicated(["name", "K", "version", "grid"]).any() and not CSC.duplicated(["name", "K", "version", "grid"]).any(), "単位の csv の行")
must(set(CS4.commit) == {C_S4} and set(CSC.commit) == {C_CD} and set(CS4.dirty.astype(str)) == {"False"} and set(CSC.dirty.astype(str)) == {"False"}, "単位の csv のコミット")
HOSTS, GPUS = sorted(set(CS4.host) | set(CSC.host)), sorted(set(CS4.gpu) | set(CSC.gpu))
must(len(HOSTS) == 1 and GPUS == [prov["gpu"]], "計算機の記録")
for d_ in (CS4, CSC):
    d_["c"] = d_.name.str.split("_ALL_").str[0]
    d_["fold"] = d_.name.str.extract(r"_f(\d+)_r")[0].astype(int)
    d_["seed"] = d_.name.str.extract(r"_r(\d+)$")[0].astype(int)
HEAD4, HEADC = [x for x in CS4.columns if x not in ("c", "fold", "seed")], [x for x in CSC.columns if x not in ("c", "fold", "seed")]
must(not any(("mem" in x.lower()) or ("gpu_max" in x.lower()) for x in HEAD4) and "gpu_max_mb" in HEADC, "GPU のメモリの記録の有無（文を直す）")
CK4, CKC = CS4[CS4.version == "_check"], CSC[CSC.version == "_check"]
must(len(CK4) == 600 and len(CKC) == 150, "検算の行の数")
s4_, cd_ = conv["students4"], conv["cremad"]
must(s4_["n_units"] == 200 and cd_["n_units"] == 50, "単位の数")
must(abs(float(CK4.grad.max()) - s4_["grad_max"]) < 1e-15 and int(CK4.stalled.sum()) == s4_["stalled"] == 0 and s4_["grad_max"] <= 1e-5, "学び直しの勾配と止まり")
must(abs(float(CKC.grad.max()) - cd_["grad_max"]) < 1e-15 and int(CKC.stalled.sum()) == cd_["stalled"] == 0 and cd_["grad_max"] <= 1e-5, "CREMA-D の勾配と止まり")
must(float(CK4.agree_min.min()) == s4_["agree_min"] == 1.0 and float(CK4[["agree_prod", "agree_prop"]].min().min()) == 1.0, "疑似ラベルの作り直しの一致")
must(abs(float(CK4.joint_repro.min()) - s4_["joint_repro_min"]) < 1e-15 and abs(float(CK4.joint_repro.mean()) - s4_["joint_repro_mean"]) < 1e-15, "今の提案の再現の割合")
BELOW = {m_: int((CK4[f"{m_}_repro"] < 0.995).sum()) for m_ in ["joint"] + EX1}
must(BELOW == {"joint": s4_["n_joint_repro_below"], **{m_: s4_[f"n_{m_}_repro_below"] for m_ in EX1}} and set(BELOW.values()) == {0} and s4_["joint_repro_ok"] is True, f"基準 0.995 に満たない単位：{BELOW}")
must(all(abs(float(CK4[f"{m_}_repro"].min()) - s4_[f"{m_}_repro_min"]) < 1e-15 for m_ in EX1), "実験 1 の 3 手法の再現の割合")
must(prov["consts"]["AGREE_MIN"] == 0.995 and prov["consts"]["GRAD_TOL"] == 1e-5, "基準の定数")
# 判定の配列から数え直した一致の割合が，単位の csv の値と同じ
for m_ in ["joint"] + EX1:
    for c, K in ALL12:
        g_ = CK4[(CK4.c == c) & (CK4.K == K)]
        must(max(abs(REPRO[(c, K, m_)][int(r.fold), int(r.seed)] - getattr(r, f"{m_}_repro")) for r in g_.itertuples()) < 1e-12, f"{m_} の一致の割合が判定の配列から数え直した値と違う {c} K={K}")
LT1 = {m_: {(c, K): int((REPRO[(c, K, m_)] < 1.0).sum()) for c, K in ALL12 if (REPRO[(c, K, m_)] < 1.0).any()} for m_ in ["joint"] + EX1}       # 一致の割合が 1 未満の（単位，K）の数
must(sum(LT1["joint"].values()) == int((CK4.joint_repro < 1.0).sum()) and sum(LT1["kd"].values()) == int((CK4.kd_repro < 1.0).sum()) and LT1["agree"] == {} and LT1["mix"] == {},
     f"一致の割合が 1 未満の単位：{LT1}")
must(all(c in ("MOSEI", "MELD") for m_ in LT1 for c, _ in LT1[m_]), "AFEW・IEMOCAP に一致の割合が 1 未満の単位がある（文を直す）")
must(abs(float(CSC.adj_resid.max()) - cd_["adj_resid_max"]) < 1e-15 and int(CSC[CSC.version == "wf1000"].wf_n_capped.fillna(0).sum()) == cd_["wf_n_capped"] == 0, "CREMA-D の切片のずらしの残りと上限")
GPU_CD_MAX = float(CKC.gpu_max_mb.max())
SEC4, SECC = float(CS4.groupby("name").sec.first().sum()), float(CSC.groupby("name").sec.first().sum())
must(bool((CK4.prop_grid.astype(int).to_numpy() == np.array([GI[(c, int(K))] for c, K in zip(CK4.c, CK4.K)])).all())
     and bool((CK4.student_main.astype(int).to_numpy() == np.array([GM[(c, int(K))] for c, K in zip(CK4.c, CK4.K)])).all())
     and bool((CKC.prop_grid.astype(int).to_numpy() == np.array([GI[(CD, int(K))] for K in CKC.K])).all())
     and bool((CKC.student_main.astype(int).to_numpy() == np.array([GM[(CD, int(K))] for K in CKC.K])).all()), "検算の行の主の点・提案の点が固定のファイルと違う")
# 学び直しの格子の表を単位の csv から作り直す
ST = pd.concat([CS4[CS4.version.astype(str).str.startswith("stud_")], CSC[CSC.version.astype(str).str.startswith("stud_")]], ignore_index=True)
g_ = ST.groupby(["c", "K", "version", "grid"])[["uar", "mimic", "teacher_uar"]].agg(["mean", "count"])
g_.columns = [f"{a}_{b}" for a, b in g_.columns]
m_ = sg.merge(g_.reset_index(), on=["c", "K", "version", "grid"], suffixes=("", "_my"))
must(len(m_) == len(sg) == 435 and all(float((m_[f"{k_}_mean"] - m_[f"{k_}_mean_my"]).abs().max()) < 1e-9 and set(m_[f"{k_}_count"]) == {50} for k_ in ("uar", "mimic", "teacher_uar")),
     "student_grid.csv が単位の csv から作り直した平均と違う")
SG = {(r.c, int(r.K), r.version, int(r.grid)): r for r in sg.itertuples()}
# 疑似ラベルの UAR と，疑似ラベルと判定が一致した割合を，判定の配列から数え直した値と比べる（主の点．列の定義の確かめ）
DEV_TQ = 0.0
for (c, K, t_), d_ in TQ.items():
    g_ = ST[(ST.c == c) & (ST.K == K) & (ST.version == f"stud_{t_}") & (ST.grid == GM[(c, K)])]
    must(len(g_) == 50, f"単位の csv の学び直しの行 {c} K={K} {t_}")
    for r in g_.itertuples():
        DEV_TQ = max(DEV_TQ, abs(d_["t"][int(r.fold), int(r.seed)] - r.teacher_uar), abs(d_["m"][int(r.fold), int(r.seed)] - r.mimic), abs(UM[(c, K, f"stud_{t_}")][int(r.fold), int(r.seed)] - r.uar))
    s_ = SG[(c, K, f"stud_{t_}", GM[(c, K)])]
    must(abs(d_["t"].mean() - s_.teacher_uar_mean) < 1e-9 and abs(d_["m"].mean() - s_.mimic_mean) < 1e-9 and abs(UM[(c, K, f"stud_{t_}")].mean() - s_.uar_mean) < 1e-9, f"student_grid.csv の主の点 {c} K={K} {t_}")
must(DEV_TQ < 1e-9, f"疑似ラベルの UAR・判定が一致した割合・学び直した分類器の UAR が，判定の配列から数え直した値と違う：{DEV_TQ:.2e}")
must('mimic=float((pp[nk] == teach[t_][nk].argmax(1)).mean())' in src21 and 'teacher_uar=100 * S.uar(yp[nk], teach[t_][nk].argmax(1))' in src21, "台本の mimic と teacher_uar の定義の行")
# CREMA-D の格子の表を単位の csv から作り直す
g_ = CSC[CSC.version.astype(str).str.fullmatch(r"M_F6_mix\d+")].copy()
g_["grid"] = g_.version.str.replace("M_F6_mix", "").astype(int)
gmn = g_.groupby(["K", "grid"]).uar_adj.mean()
must(len(glc) == 48 and all(abs(gmn[(int(r.K), int(r.grid))] - r.uar_adj) < 1e-9 and bool(r.frozen) == (int(r.grid) == GI[(CD, int(r.K))]) for r in glc.itertuples()), "grid_levels_cremad.csv")
must(len(wcd) == 24 and set(wcd.n_units) == {50} and float((wcd.groupby("K")[["w_prop", "w_orc", "w_K", "w_ext"]].mean() - 1.0).abs().max().max()) < 1e-9, "weights_cremad.csv の重みの平均が 1 でない")

# ---- 計算の後の解析（CREMA-D の感度の表）：出力の csv を読み，由来の記録を確かめ，単位の npz から別の書き方で作り直した値と照らす
PHDIR = R / "posthoc"
PH_PY = TOOLS / "dai21_posthoc_cremad_sentence.py"
phv = json.load(open(PHDIR / "cremad_sentence_provenance.json"))


def ph_read(name: str, strs=()) -> pd.DataFrame:
    d_ = pd.read_csv(PHDIR / name, low_memory=False, dtype={"folds_dropped": str})
    for col_ in strs:
        d_[col_] = d_[col_].astype(object).where(d_[col_].notna(), "").astype(str)
    return d_


nl = ph_read("cremad_no_ieo_levels.csv", ("same_as", "same_pred"))
ne = ph_read("cremad_no_ieo_effects.csv", ("mark", "mark_all", "a_same_as", "b_same_as"))
npe = ph_read("cremad_no_ieo_per_emotion.csv")
nrm = ph_read("cremad_no_ieo_removed.csv")
sce = ph_read("cremad_sentence_cluster_effects.csv", ("mark", "mark_speaker"))
scl = ph_read("cremad_sentence_cluster_levels.csv", ("same_pred",))
bys = ph_read("cremad_by_sentence.csv")
byi = ph_read("cremad_by_intensity.csv")
ksh = ph_read("cremad_kshot_ieo_units.csv")
asm = ph_read("cremad_assumptions.csv")
PH_FILES = {"cremad_no_ieo_levels.csv": nl, "cremad_no_ieo_effects.csv": ne, "cremad_no_ieo_per_emotion.csv": npe, "cremad_no_ieo_removed.csv": nrm, "cremad_sentence_cluster_effects.csv": sce,
            "cremad_sentence_cluster_levels.csv": scl, "cremad_by_sentence.csv": bys, "cremad_by_intensity.csv": byi, "cremad_kshot_ieo_units.csv": ksh, "cremad_assumptions.csv": asm}
# 由来の記録
must(phv["script_sha256"] == sha(PH_PY) and phv["script_sha256"].startswith("997a098b") and phv["commit"].startswith("f485755") and phv["dirty_tracked"] is False and phv["script_committed_clean"] is True
     and phv["allow_partial"] is False, "計算の後の解析の台本の sha256 とコミット")
must(phv["n_units"] == 50 and phv["unit_commits"] == [C_CD] and phv["frozen_sha256"] == prov["frozen_sha256"] and phv["frozen20_sha256"] == prov["frozen20_sha256"] and phv["design_sha256"] == prov["design_sha256"]
     and phv["dai21_provenance_sha256"] == sha(R / "provenance.json") and phv["n_versions"] == 53 and phv["n_pairs"] == 78, "計算の後の解析が読んだ単位・固定のファイル・設計")
must(phv["summary_sha256"] == {n_: sha(SM / n_) for n_ in phv["summary_sha256"]} and set(phv["summary_sha256"]) == {"levels.csv", "effects.csv", "per_emotion.csv", "confusion.csv", "convergence.json"},
     "計算の後の解析が照らした集計が，今の集計と違う")
must(all(phv["modules_sha256"][n_] == prov["scripts"][n_] for n_ in prov["scripts"]) and phv["modules_sha256"]["dai21_eval5.py"] == prov["this_script_sha256"] and len(phv["modules_sha256"]) == 14, "計算の後の解析が読み込んだ部品")
must({n_: len(d_) for n_, d_ in PH_FILES.items()} == {n_: phv["outputs"][n_] for n_ in PH_FILES} and phv["outputs"]["cremad_by_sentence_confusion.csv"] == 9216 and len(phv["outputs"]) == 11
     and len(list(PHDIR.glob("*"))) == 12, "計算の後の解析の出力の行の数と本数")
T_PH = phv["time"]
PH_COMMITS = [x.split(" ", 1) for x in git("log", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M:%S", "--", "DAI/docs/tools_pending/dai21_posthoc_cremad_sentence.py").strip().split("\n")]
must([h_ for h_, _ in PH_COMMITS] == ["f485755", "dbede81"] and PH_COMMITS[0][1] < T_PH and git("status", "--porcelain", "--", "DAI/docs/tools_pending/dai21_posthoc_cremad_sentence.py").strip() == "",
     f"計算の後の解析の台本のコミット：{PH_COMMITS}")
must(all(time_ >= T_PH[:16] for time_ in [__import__("time").strftime("%Y-%m-%d %H:%M", __import__("time").localtime(x_.stat().st_mtime)) for x_ in PHDIR.glob("*.csv")]), "計算の後の解析の出力が，由来の記録より前に書かれている")
LOG_PH = (ROOT / "logs" / "dai21_posthoc_cremad_sentence.log").read_text()
must(LOG_PH.rstrip().endswith("EXIT=0") and "Traceback" not in LOG_PH, "計算の後の解析のログ")
# 検査の報告（判定と，検査した版）
PT1, PT2 = (CK / "code_post_tools.md").read_text(), (CK / "code_post_tools_v2.md").read_text()
must("## 判定\n\n直してから進む．" in PT1 and "コミット dbede81" in PT1.split("\n")[0] and "sha256 21087a39c6a50a8717333775426f83d30cd431a6502c31c1972495bfee67a9e2" in PT1
     and hashlib.sha256(git("show", "dbede81:DAI/docs/tools_pending/dai21_posthoc_cremad_sentence.py").encode()).hexdigest().startswith("21087a39"), "計算の後の解析の台本の検査（第 1 回）")
must("## 判定\n\n通す．前の報告の指摘 1〜7 はすべて直っており，直しの前の出力の値は 1 つも変わっていない．" in PT2 and "コミット f485755" in PT2.split("\n")[0] and f"sha256 {phv['script_sha256']}" in PT2,
     "計算の後の解析の台本の検査（第 2 回）")
N_PT1, N_PT2 = PT1.count("\n### ["), PT2.count("\n### [")
must(N_PT1 == 7 and N_PT2 == 2, f"計算の後の解析の台本の検査の指摘の数：{N_PT1}，{N_PT2}")
# 全発話の値が集計と同じ（出力の *_all・*_speaker の列）
NE = {(r.a, r.b, int(r.K), r.seeds): r for r in ne.itertuples()}
SCE = {(r.a, r.b, int(r.K), r.seeds): r for r in sce.itertuples()}
NL = {(r.version, int(r.K), r.seeds): r for r in nl.itertuples()}
SCL = {(r.version, int(r.K), r.seeds): r for r in scl.itertuples()}
must(set(NE) == set(SCE) == {(a, b, K, s_) for a, b in PC for K in KS for s_ in SEEDS} and set(NL) == set(SCL) == {(v, K, s_) for v in VC for K in KS for s_ in SEEDS}, "計算の後の解析の差の対と版")
DEV_PH_ALL = max(max(abs(r.est_all - q(a, b, CD, K, s_).est), abs(r.lo_all - q(a, b, CD, K, s_).lo), abs(r.hi_all - q(a, b, CD, K, s_).hi)) for (a, b, K, s_), r in NE.items())
DEV_PH_SPK = max(max(abs(r.est_speaker - q(a, b, CD, K, s_).est), abs(r.lo_speaker - q(a, b, CD, K, s_).lo), abs(r.hi_speaker - q(a, b, CD, K, s_).hi), abs(r.est - r.est_speaker)) for (a, b, K, s_), r in SCE.items())
DEV_PH_LV = max(max(abs(r.uar_all - level(v, CD, K, s_).uar), abs(SCL[(v, K, s_)].uar - level(v, CD, K, s_).uar)) for (v, K, s_), r in NL.items())
must(DEV_PH_ALL < 1e-9 and DEV_PH_SPK < 1e-9 and DEV_PH_LV < 1e-9, f"計算の後の解析の全発話の値が集計と違う：{DEV_PH_ALL:.2e}，{DEV_PH_SPK:.2e}，{DEV_PH_LV:.2e}")
must(all(abs(r.d_est - (r.est - r.est_all)) < 1e-9 and r.d_lo <= r.d_est + 1e-12 <= r.d_hi + 2e-12 for r in ne.itertuples()) and all(abs(r.d_uar - (r.uar - r.uar_all)) < 1e-9 for r in nl.itertuples()),
     "「除いた値 − 全発話の値」の列")
for d_ in (ne, sce):                                                       # 印を設計 4 節の文から作り直す
    mk_ = np.where((d_.lo > 0) & (d_.est >= FLOOR), "↑", np.where((d_.hi < 0) & (d_.est <= -FLOOR), "↓", ""))
    must(bool((mk_ == d_["mark"].to_numpy()).all()), "計算の後の解析の印の列")
must(set(ne.method) == {"boot_cl+seed"} and set(ne.d_method) == {"boot_cl+seed"} and set(sce.method) == {"jk+seed"} and set(sce.df) == {9} and set(sce.n_clusters) == {12} and set(sce.df_n12) == {11}, "計算の後の解析の区間の作り方")
# (1) 文 IEO を除いたテストの点推定：水準 477 行と差 702 行
DEV_NL = max(abs(fold_mean(UMN[(int(r.K), r.version)], SEEDS[r.seeds]) - r.uar) for r in nl.itertuples())
DEV_NE = max(abs(fold_mean(UMN[(int(r.K), r.a)] - UMN[(int(r.K), r.b)], SEEDS[r.seeds]) - r.est) for r in ne.itertuples())
must(DEV_NL < 1e-9 and DEV_NE < 1e-9, f"文を除いたテストの点推定が，単位の npz から作り直した値と違う：{DEV_NL:.2e}，{DEV_NE:.2e}")
# (2) 除いた発話の数（別々の発話．感情ごと）
N_SENT = {s_: sum(STOT[(f, 0)][j].sum() for f in range(5)) for j, s_ in enumerate(SENT12)}        # 文ごとのテストの別々の発話の数
N_SENT_E = {s_: sum(STOT[(f, 0)][j] for f in range(5)) for j, s_ in enumerate(SENT12)}
RM_E = N_SENT_E["IEO"].astype(int)                                           # 除いた発話（感情ごと）
KEEP_E = (N_PER_SEED[CD] - N_SENT_E["IEO"]).astype(int)                      # 残った発話（感情ごと）
must(all((sum(STOT[(f, r)] for f in range(5)) == sum(STOT[(f, 0)] for f in range(5))).all() for r in range(10)), "文ごとの発話の数が K 本の引き方で違う")
g_ = nrm.groupby("emotion")[["n_all", "n_removed", "n_kept"]].sum() / 10
must([int(x) for x in g_.n_removed] == RM_E.tolist() and [int(x) for x in g_.n_kept] == KEEP_E.tolist() and [int(x) for x in g_.n_all] == N_PER_SEED[CD].astype(int).tolist()
     and RM_E.tolist() == [phv["removed"]["distinct"][str(e)] for e in range(4)] and int(RM_E.sum()) == 910 == IEO_ALL, f"除いた発話の数：{RM_E}")
must(all(float(r.n_per_seed) == KEEP_E[int(r.emotion)] and float(r.n_all) == 10 * N_PER_SEED[CD][int(r.emotion)] for r in npe.itertuples()), "文を除いた感情ごとの再現率の件数")
# (3) 文を塊にした区間：文を 1 つずつ全部の分割から除く jackknife ＋ K 本の引き方の項．自由度は ⌊文の実効の数⌋ − 1
G_EFF_SENT = float(sum(N_SENT.values()) ** 2 / sum(x ** 2 for x in N_SENT.values()))
DF_SENT = int(np.floor(G_EFF_SENT)) - 1
must(abs(G_EFF_SENT - phv["sentence_cluster_check"]["g_eff_sentence"]) < 1e-9 and DF_SENT == 9 == phv["sentence_cluster_check"]["df_main"] and bool((np.abs(sce.g_eff_sentence - G_EFF_SENT) < 1e-9).all())
     and {s_: int(n_) for s_, n_ in N_SENT.items()} == phv["sentence_cluster_check"]["n_per_sentence_distinct"], f"文の実効の数：{G_EFF_SENT}")
TQ9, TQ11 = float(sps.t.ppf(0.975, DF_SENT)), float(sps.t.ppf(0.975, 11))


def uar4(cor_: np.ndarray, tot_: np.ndarray) -> float:
    return 100.0 * float((cor_ / tot_).mean())


# 版ごとに，文を 1 つ除いたときの UAR を先に作る：JKU[(K, 版)][0] は全発話，[1 + g] は文 g を除いたテスト（分割 × 引き方）
JKU: dict = {}
for (K, v), d_ in SCOR.items():
    M_ = np.full((13, 5, 10), np.nan)
    for (f, r), cor_ in d_.items():
        t_ = STOT[(f, r)]
        must(bool((t_.sum(1) > 0).all()), f"テストに無い文がある：分割 {f} 引き方 {r}")
        M_[0, f, r] = uar4(cor_.sum(0), t_.sum(0))
        for g in range(12):
            M_[1 + g, f, r] = uar4(cor_.sum(0) - cor_[g], t_.sum(0) - t_[g])
    must(not np.isnan(M_).any(), f"文を除いた UAR の欠け：K={K} {v}")
    JKU[(K, v)] = M_
must(max(float(np.abs(JKU[(K, v)][0] - UM[(CD, K, v)]).max()) for (K, v) in JKU) < 1e-9, "文ごとの正解の数から作った UAR が，判定から作った UAR と違う")


def jk_sentence(K: int, a: str, b: str, seeds) -> tuple:
    """文を塊にした区間（差 a − b）．戻り：(点推定, 標準誤差)．文を 1 つずつ全部の分割から除く jackknife の分散に，K 本の引き方の項を足す．"""
    rs = list(range(10)) if seeds is None else list(seeds)
    D_ = (JKU[(K, a)] - JKU[(K, b)])[:, :, rs]                               # (13, 分割, 引き方)
    th_ = D_[1:].mean(2).mean(1)                                             # 文 g を除いたときの値
    v_jk = float(11 / 12 * ((th_ - th_.mean()) ** 2).sum())
    v_seed = float((D_[0].var(1, ddof=1) / len(rs)).sum() / 25)
    return float(D_[0].mean(1).mean()), float(np.sqrt(v_jk + v_seed))


DEV_JK = 0.0
for (a, b, K, s_), r in SCE.items():
    e_, se_ = jk_sentence(K, a, b, SEEDS[s_])
    DEV_JK = max(DEV_JK, abs(e_ - r.est), abs(se_ - r.se), abs(e_ - TQ9 * se_ - r.lo), abs(e_ + TQ9 * se_ - r.hi), abs(e_ - TQ11 * se_ - r.lo_n12), abs(e_ + TQ11 * se_ - r.hi_n12))
must(DEV_JK < 1e-8, f"文を塊にした区間が，単位の npz から作り直した値と違う：{DEV_JK:.2e}")
DEV_JKL = 0.0                                                                # 水準（版ごとの UAR）の文を塊にした区間
for (v, K, s_), r in SCL.items():
    rs = list(range(10)) if SEEDS[s_] is None else list(SEEDS[s_])
    D_ = JKU[(K, v)][:, :, rs]
    th_ = D_[1:].mean(2).mean(1)
    e_, se_ = float(D_[0].mean(1).mean()), float(np.sqrt(11 / 12 * ((th_ - th_.mean()) ** 2).sum() + (D_[0].var(1, ddof=1) / len(rs)).sum() / 25))
    DEV_JKL = max(DEV_JKL, abs(e_ - r.uar), abs(se_ - r.se), abs(e_ - TQ9 * se_ - r.lo), abs(e_ + TQ9 * se_ - r.hi), abs(e_ - TQ11 * se_ - r.lo_n12), abs(e_ + TQ11 * se_ - r.hi_n12))
must(DEV_JKL < 1e-8 and set(scl.method) == {"jk+seed"} and set(scl.df) == {9}, f"水準の文を塊にした区間が，単位の npz から作り直した値と違う：{DEV_JKL:.2e}")
# 文を除いたテストの感情ごとの再現率（単位ごとの再現率の平均と，50 単位の数を足してから割った値）．全発話の列は集計の per_emotion.csv と同じ
J_IEO = SENT12.index("IEO")
DEV_NPE = 0.0
for r in npe.itertuples():
    K, v, e = int(r.K), r.version, int(r.emotion)
    U_ = np.full((5, 10), np.nan)
    for (f, r2), cor_ in SCOR[(K, v)].items():
        t_ = STOT[(f, r2)]
        U_[f, r2] = 100 * (cor_.sum(0) - cor_[J_IEO])[e] / (t_.sum(0) - t_[J_IEO])[e]
    pooled_ = 100 * sum((c_.sum(0) - c_[J_IEO])[e] for c_ in SCOR[(K, v)].values()) / sum((STOT[k_].sum(0) - STOT[k_][J_IEO])[e] for k_ in STOT)
    pe_ = pe[(pe.c == CD) & (pe.K == K) & (pe.version == v) & (pe.emotion == e)]
    must(len(pe_) == 1, f"per_emotion.csv の行：{v} K={K} {e}")
    DEV_NPE = max(DEV_NPE, abs(fold_mean(U_, None) - r.recall), abs(pooled_ - r.recall_pooled), abs(r.recall_all - float(pe_.recall.iloc[0])), abs(r.lo_all - float(pe_.lo.iloc[0])), abs(r.hi_all - float(pe_.hi.iloc[0])),
                  abs(r.d_recall - (r.recall - r.recall_all)))
must(DEV_NPE < 1e-9 and len(npe) == 53 * 3 * 4, f"文を除いたテストの感情ごとの再現率が，単位の npz から作り直した値と違う：{DEV_NPE:.2e}")
# (4) 文ごと・強さごとの内訳（50 単位の数を足してから割った値）
DEV_BS = 0.0
for r in bys.itertuples():
    j = SENT12.index(r.sentence)
    cor_ = sum(SCOR[(int(r.K), r.version)][k_][j] for k_ in STOT)
    tot_ = sum(STOT[k_][j] for k_ in STOT)
    DEV_BS = max(DEV_BS, abs(100 * float((cor_ / tot_).mean()) - r.uar_pooled), max(abs(100 * cor_[e] / tot_[e] - getattr(r, f"recall_pooled_{e}")) for e in range(4)),
                 abs(tot_.sum() - r.n), abs(tot_.sum() / 10 - r.n_per_seed))
DEV_BI = 0.0
for r in byi.itertuples():
    i_, e = INT4.index(r.intensity), int(r.emotion)
    DEV_BI = max(DEV_BI, abs(100 * ICOR[(int(r.K), r.version)][i_, e] / ITOT[i_, e] - r.recall_pooled), abs(ITOT[i_, e] - r.n))
must(DEV_BS < 1e-9 and DEV_BI < 1e-9 and sorted(set(bys.version)) == sorted(BRK_V) == sorted(set(byi.version)) and len(bys) == 12 * 3 * 16 and len(byi) == 13 * 3 * 16
     and phv["consts"]["breakdown_versions"] == BRK_V, f"文ごと・強さごとの内訳が，単位の npz から作り直した値と違う：{DEV_BS:.2e}，{DEV_BI:.2e}")
must(bool((ITOT[1:, 0] == 0).all()) and bool((ITOT[1:, 1:] == 910).all()) and all(N_SENT_E[s_].astype(int).tolist() in ([91, 91, 91, 91], [90, 91, 91, 91], [90, 90, 90, 90], [90, 90, 89, 90]) for s_ in SENT12 if s_ != "IEO"),
     "強さと文ごとの発話の数（文を直す）")
# (5) K 本の中の文 IEO の本数と，単位ごとの値
must(len(ksh) == 150 and all(int(r.n_ieo) == int(KIEO[(int(r.K), int(r.fold), int(r.seed))].sum()) and [int(getattr(r, f"n_ieo_{e}")) for e in range(4)] == KIEO[(int(r.K), int(r.fold), int(r.seed))].tolist()
                             and int(r.n_kshot) == 4 * int(r.K) for r in ksh.itertuples()), "K 本の中の文 IEO の本数")
DEV_KS = 0.0
for r in ksh.itertuples():
    K, f, r2 = int(r.K), int(r.fold), int(r.seed)
    for a, b in (("prop", "prod"), ("stud_prop", "stud_prod"), ("stud_prop", "none_AV"), ("stud_prod", "none_AV"), ("prop", "none_AV")):
        DEV_KS = max(DEV_KS, abs(UM[(CD, K, a)][f, r2] - UM[(CD, K, b)][f, r2] - getattr(r, f"d_{a}__{b}")), abs(UMN[(K, a)][f, r2] - UMN[(K, b)][f, r2] - getattr(r, f"d_noieo_{a}__{b}")))
    for v in BRK_V:
        DEV_KS = max(DEV_KS, abs(UM[(CD, K, v)][f, r2] - getattr(r, f"uar_{v}")), abs(UMN[(K, v)][f, r2] - getattr(r, f"uar_noieo_{v}")))
must(DEV_KS < 1e-9, f"K 本の表の単位ごとの値が，単位の npz から作り直した値と違う：{DEV_KS:.2e}")
# (6) 仮定の表：相関・水準・最も強いモダリティ・上限の割合の 23 列（成分の相関と入れ替えの割合，テストとのずれは作り直さない）
ASM = {int(r.K): r for r in asm.itertuples()}
ASM_COLS = sorted(ASM_U[5][0])
must(len(ASM_COLS) == 23 and abs(DCAP - phv["assumptions"]["dk_cap"]) < 1e-9 and all(len(ASM_U[K]) == 50 for K in KS), "仮定の量の作り直しの列")
DEV_ASM = max(abs(float(np.nanmean([u_[c_] for u_ in ASM_U[K]])) - float(getattr(ASM[K], c_))) for K in KS for c_ in ASM_COLS)
must(DEV_ASM < 1e-9 and set(asm.n_units) == {50} and set(asm.check_row_assumption_columns) == {"無い"}, f"仮定の表が，単位の npz から作り直した値と違う：{DEV_ASM:.2e}")
must(len(SRC_TAB) == 50 and all(np.array_equal(x_[0], SRC_TAB[0][0]) and np.array_equal(x_[1], SRC_TAB[0][1]) and np.array_equal(x_[2], SRC_TAB[0][2]) for x_ in SRC_TAB)
     and float(np.abs(SRC_TAB[0][0] - SRC_TAB[0][1] * SRC_TAB[0][2][:, None]).max()) < 1e-12, "ソースで測った表が単位で違うか，ext が home × 残り方でない（文を直す）")
must(int(SRC_TAB[0][0].mean(1).argmax()) == 1 and int(SRC_TAB[0][1].mean(1).argmax()) == 0 and len(TOP_ORC) == 150 and set(TOP_ORC) == {0}, "行の平均が最大の分類器（文を直す）")
ASM_NOT = phv["assumptions"]["not_made"]
must(set(ASM_NOT) == {"levelcorr_M_*", "excess_AV", "sdratio_M", "*_away", "profcorr_noV"} and len(phv["assumptions"]["from_npz"]) == 30 and set(phv["assumptions"]["from_npz"]) == set(asm.columns) - {"c", "K", "n_units", "check_row_assumption_columns"},
     "仮定の表の作った量と作っていない量")
print(f"計算の後の解析の照合：文を除いた点推定 {max(DEV_NL, DEV_NE):.1e}，感情ごと {DEV_NPE:.1e}，文を塊にした区間 {max(DEV_JK, DEV_JKL):.1e}，内訳 {max(DEV_BS, DEV_BI):.1e}，K 本の表 {DEV_KS:.1e}，仮定の表 {DEV_ASM:.1e}", flush=True)

# ---- ログ
N_SKIP = log_all.count("済み")
N_HEAD = log_all.count("=== dai21 ")
must(N_SKIP == 0 and N_HEAD == 3 and log_all.rstrip().endswith("EXIT=0") and not (ROOT / "logs" / "dai21_eval5.failed").exists() and "Traceback" not in log_all and "Warning" not in log_all, "ログの終わり方")
must("selftest：通った" in log_all and "check3：通った" in log_all and "check18：通った" in log_all, "検算の段のログ")
T_HEAD = dict(re.findall(r"=== dai21 (\w+) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", log_all))
must(set(T_HEAD) == {"students4", "cremad", "summary"}, "段の見出しの時刻")
LOG3 = (ROOT / "logs" / "dai21_check3.log").read_text()
LOG18 = (ROOT / "logs" / "dai21_check18.log").read_text()
c3 = [x.split() for x in LOG3.strip().split("\n")[1:-1]]
must(len(c3) == 4 and all(x[2] == "38" and float(x[3]) == 1.0 and x[5] == "38" for x in c3) and sorted({int(x[1]) for x in c3}) == [5, 25] and len({x[0] for x in c3}) == 2, f"check3 のログ：{c3}")
c18 = {int(x.split()[0]): x.split() for x in LOG18.strip().split("\n")[1:4]}
must(set(c18) == {5, 10, 25} and all(c18[K][1] == "39" for K in KS) and float(c18[5][2]) == float(c18[10][2]) == 1.0 and c18[5][4] == c18[10][4] == "39", f"check18 のログ：{c18}")
C18_MIN25, C18_WORST25, C18_EXACT25, C18_DEV = float(c18[25][2]), c18[25][3], int(c18[25][4]), max(float(c18[K][5]) for K in KS)
must(C18_MIN25 >= 0.995, "check18 の一致の割合")

# ---- 実験 20 の集計と重なる版（元の 4 コーパスの推定時の答えと，実験 1 の保存の判定）：点推定は同じ．MOSEI の区間だけ乱数の鍵の違いで少し違う
BOTH = ["prop", "prod", "none_AVT", "joint_AVT"] + B5
DEV20_UAR = max(abs(level(v, c, K).uar - LV20[(v, c, K, "all")].uar) for v in BOTH for c, K in ALL12)
DEV20_END = {c: max(max(abs(level(v, c, K).lo - LV20[(v, c, K, "all")].lo), abs(level(v, c, K).hi - LV20[(v, c, K, "all")].hi)) for v in BOTH for K in KS) for c in C4}
DEV20_EF = max(abs(q("prop", "joint_AVT", c, K, s).est - q20("prop", "joint_AVT", c, K, s).est) for c, K in ALL12 for s in SEEDS)
DEV20_EFEND = {c: max(max(abs(q("prop", "joint_AVT", c, K, s).lo - q20("prop", "joint_AVT", c, K, s).lo), abs(q("prop", "joint_AVT", c, K, s).hi - q20("prop", "joint_AVT", c, K, s).hi))
                      for K in KS for s in SEEDS) for c in C4}
must(DEV20_UAR < 1e-9 and DEV20_EF < 1e-9 and all(DEV20_END[c] < 1e-9 and DEV20_EFEND[c] < 1e-9 for c in ("AFEW", "IEMOCAP", "MELD")) and 0 < DEV20_END["MOSEI"] < 0.1 and 0 < DEV20_EFEND["MOSEI"] < 0.02,
     f"実験 20 の集計との照合：{DEV20_UAR:.2e} {DEV20_END} {DEV20_EF:.2e} {DEV20_EFEND}")
must(not any((a, b) == ("prop", "prod") for a, b in P4) and all(("prop", "prod", c, K, s) in EF20 for c, K in ALL12 for s in SEEDS), "元の 4 コーパスの「提案の手法 − 積」の置き場（文を直す）")
BOTH_EF = sorted((a, b) for a, b in P4 if (a, b, "AFEW", 5, "all") in EF20)
must(BOTH_EF == [("prop", "joint_AVT")], f"2 つの集計の両方にある差の対（文を直す）：{BOTH_EF}")

# ---- 表の文書：今の集計から書き出し直すと，今ある文書と同じになる
with tempfile.TemporaryDirectory() as td_:
    tmp_ = Path(td_) / "tables.md"
    subprocess.run([sys.executable, str(TABLES_PY), "--out", str(tmp_)], capture_output=True, text=True, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    must(tmp_.read_text() == table_doc, "表の文書が，今の集計から書き出した表と違う")
must(git("status", "--porcelain", "--", "DAI/docs/実験21_結果の表_2026-11-15.md", "DAI/docs/tools_pending/md_tables_dai21.py").strip() == "", "表の文書か表の台本に，コミットしていない変更がある")
TD_HIST = [x.split(" ", 1) for x in git("log", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M:%S", "--", "DAI/docs/実験21_結果の表_2026-11-15.md").strip().split("\n")]
must(len(TD_HIST) >= 2 and TD_HIST[-1][0] == "d75cd76", f"表の文書のコミット：{TD_HIST}")
TD_FIRST, TD_LAST = TD_HIST[-1], TD_HIST[0]                                # 最初に書き出した版と，今の版（本体が書き出し直した版）
TD_CELLS = [x.strip() for ln in table_doc.split("\n") if ln.startswith("|") for x in ln.strip("|").split("|")]
NOTE_SAME, NOTE_CNT = "（同じ判定）", "（正解の数は同じ．判定は違う）"
TD_ZERO = [x for x in TD_CELLS if re.search(r"\[[+-]0\.0, |, [+-]0\.0\]", x)]                 # 区間の端が ±0.0 の欄は，差がちょうど 0 の欄だけ
must(all(x in ("+0.00 [+0.0, +0.0]" + NOTE_SAME, "+0.00 [+0.0, +0.0]" + NOTE_CNT) for x in TD_ZERO), "表の文書に，符号の読めない区間の端か，注の無い差が 0 の欄が残っている")
TD_ZERO_SAME, TD_ZERO_CNT = sum(x.endswith(NOTE_SAME) for x in TD_ZERO), sum(x.endswith(NOTE_CNT) for x in TD_ZERO)
TD_SHOWN0 = [("stud_prod", "joint_AVT"), ("kd_prod", "b_kd"), ("agree_prod", "b_agree"), ("mix_prod", "b_mix")]       # 表の文書に出る差のうち，差がちょうど 0 の条件を持つ対（4 コーパス）
must(TD_ZERO_SAME == sum(klass(q(a, b, c, K)) == "same" for a, b in TD_SHOWN0 for c, K in ALL12) + sum(klass(q("prop", "prop_n0zero", CD, K)) == "same" for K in KS)
     and TD_ZERO_CNT == sum(klass(q(a, b, c, K)) == "cnt" for a, b in TD_SHOWN0 for c, K in ALL12) == 3, f"表の文書の差が 0 の欄の数：{TD_ZERO_SAME}・{TD_ZERO_CNT}")
must(not any(re.search(r"^[+-]0\.0+ \[|\[[+-]0\.0+, |, [+-]0\.0+\]", x) and x not in TD_ZERO for x in TD_CELLS), "表の文書に，0 に丸まったままの差か区間の端が残っている")
# 表の文書：最初に書き出した版と欄ごと・行ごとに比べ，変わったのが，注を足した欄，桁を足した差，文の 2 行だけであること
OLD_TBL = git("show", f"{TD_FIRST[0]}:DAI/docs/実験21_結果の表_2026-11-15.md")
MIX_OLD = "今の提案：この実験で解いた値 − 実験 1 の保存（4 コーパス．再現の確かめ）／− K 本だけの融合（CREMA-D）"     # 最初の版で，再現の検算と性能の差が 1 行に並んでいた行
SPLIT_A = "今の提案 − K 本だけの融合（元の 4 コーパスの今の提案は実験 1 の保存，CREMA-D はこの実験で解いた値）"
SPLIT_B = "再現の確かめ：今の提案をこの実験で解いた値 − 実験 1 の保存（元の 4 コーパスだけ．性能の差ではない）"


def rows_of(text_: str) -> list:
    return [[y.strip() for y in x.strip().strip("|").split("|")] for x in text_.split("\n") if x.startswith("|")]


def cell_change(a_: str, b_: str) -> str:
    """最初の版の欄 a_ と今の版の欄 b_ の違い：同じなら空，注が付いたなら note，桁が足されたなら digit．ほかの違いなら止まる．"""
    if a_ == b_:
        return ""
    if a_ == "+0.00 [+0.0, +0.0]" and b_ == a_ + NOTE_CNT:
        return "note"
    ma_, mb_ = re.match(r"([+-]0\.00) (\[.*)$", a_), re.match(r"([+-]\d+\.\d+) (\[.*)$", b_)
    must(ma_ is not None and mb_ is not None and ma_.group(2) == mb_.group(2) and 0.0 < abs(float(mb_.group(1))) <= 0.005 + 1e-12 and mb_.group(1)[0] == ma_.group(1)[0], f"表の文書の欄が，差の桁と注のほかで変わった：{a_} → {b_}")
    return "digit"


o_all, n_all = rows_of(OLD_TBL), rows_of(table_doc)
o_mix, n_a, n_b = [r for r in o_all if r[0] == MIX_OLD], [r for r in n_all if r[0] == SPLIT_A], [r for r in n_all if r[0] == SPLIT_B]
n_j = [r for r in n_all if r[0] == "今の提案 − K 本だけの融合" and len(r) == 6]                # 3 節の表の同じ差の行（元の 4 コーパス）
must(len(o_mix) == 3 and len(n_a) == 3 and len(n_b) == 3 and len(n_j) == 3 and not any(r[0] == MIX_OLD for r in n_all), "表の文書の，分けた行の数")
o_rows, n_rows = [r for r in o_all if r[0] != MIX_OLD], [r for r in n_all if r[0] not in (SPLIT_A, SPLIT_B)]
must(len(o_rows) == len(n_rows) and len(n_all) == len(o_all) + 3, "表の文書の表の行の数")
TBL_CH = []
for ro_, rn_ in zip(o_rows, n_rows):
    must(len(ro_) == len(rn_), "表の文書の列の数が変わった")
    TBL_CH += [cell_change(a_, b_) for a_, b_ in zip(ro_, rn_)]
for ro_, ra_, rb_, rj_ in zip(o_mix, n_a, n_b, n_j):                        # 分けた 2 行：欄の値は，元の行の欄と，3 節の表の同じ差の行の欄と同じ
    must(len(ro_) == 7 and len(ra_) == 7 and len(rb_) == 7 and ro_[1] == ra_[1] == rb_[1] == rj_[1], "表の文書の，分けた行の K")
    must(ra_[6] == ro_[6] and rb_[6] == "—" and ra_[2:6] == rj_[2:6], f"表の文書の，分けた行の欄：{ra_}")
    TBL_CH += [cell_change(a_, b_) for a_, b_ in zip(ro_[2:6], rb_[2:6])]
TBL_NOTE, TBL_DIG = TBL_CH.count("note"), TBL_CH.count("digit")
must(TBL_NOTE == 3 and TBL_DIG == 8, f"表の文書で変わった欄の数：注 {TBL_NOTE}，桁 {TBL_DIG}")
o_text = [x for x in OLD_TBL.split("\n") if not x.startswith("|")]
n_text = [x for x in table_doc.split("\n") if not x.startswith("|")]
TBL_GONE, TBL_NEW = [x for x in o_text if x not in n_text], [x for x in n_text if x not in o_text]
TD_K29_OLD = "引き方 0・1 は下調べと同じ K 本，引き方 2〜9 はこの実験で初めて使う K 本である．テストの発話はどちらも同じである．"
TD_K29 = "引き方 0・1 は下調べと同じ K 本，引き方 2〜9 は評価の回で初めて使う K 本である（元の 4 コーパスは実験 20 と同じ単位，CREMA-D はこの実験の単位）．"
TD_DIGIT = ("差がちょうど 0 で区間の両端も 0 の欄は，全部の単位で両側の正解の数が同じ条件である．判定まで同じなら「（同じ判定）」，判定が違えば「（正解の数は同じ．判定は違う）」と書く．")
TD_EXPL = "この表の値は，手法と設定を選んだときに見たテストと同じテストの発話の上の値で，探索の結果として読む（設計 0 節）．CREMA-D は，元の 4 コーパスと並ぶ独立な確かめとして数えない．"
must(len(TBL_GONE) == 2 and TBL_GONE[0].startswith("- 数の桁：") and TBL_GONE[1] == TD_K29_OLD and len(TBL_NEW) == 3 and set(TBL_NEW[1:]) == {TD_EXPL, TD_K29 + "テストの発話はどちらも同じである．"}
     and TBL_NEW[0].startswith("- 数の桁：") and TBL_NEW[0].endswith(TD_DIGIT) and "差が 0.00 に丸まるときは" in TBL_NEW[0], f"表の文書の文の行が，決まった行のほかで変わった：{TBL_GONE}，{TBL_NEW}")
TD_SHIFT = "ずらす量は，ラベル無しの候補の発話の上で，その版の確率の平均が，積の確率の平均と同じになるように決める"
must(TD_SHIFT in table_doc and TD_SHIFT in text20, "表の文書と実験 20 の結果の文の，切片のずらしの定義の文")
TD_DASH = "部品を外した版と，全ラベルで測った表・ソースだけで測った表は，切片をずらした答えだけを集計している．「切片をずらす前」の欄の「—」は，その値を集計していないことを表す．"
must(TD_DASH in table_doc, "表の文書の「—」の注（点検の報告の記録 3）")
TD_SAMEPRED = [x for x in table_doc.split("\n") if x.endswith("と同じ判定") and x.startswith("- ")]
SP_LV = lv[(lv.seeds == "all") & (lv.same_pred != "") & (lv.same_pred != lv.version)]
must(len(TD_SAMEPRED) == len(SP_LV) and "## 注：判定が全部の単位で同じ版" in table_doc, "表の文書の「同じ判定」の注の行の数")

print(f"照合：水準 {DEV_LV:.1e}，差 {DEV_EF:.1e}，再現率 {DEV_PE:.1e}，疑似ラベルの量 {DEV_TQ:.1e}，実験 20 との区間のずれ {DEV20_END['MOSEI']:.3f}・{DEV20_EFEND['MOSEI']:.4f}", flush=True)

# ---- 計算の前の検査（関門 1 と査読想定の役）：どの報告がどの版を見たか
REP = {n: (CK / f"{n}.md").read_text() for n in ("stats", "leak", "leak_v2", "compare", "compare_v2", "define", "review_ml", "review_ml_v2", "review_affect", "code", "code_v2", "code_v3")}


def verdict(n: str) -> str:
    m_ = re.search(r"## 判定\s*\n+\s*(通す|直してから通す|止める)．", REP[n])
    must(m_ is not None, f"{n}.md の判定が読めない")
    return m_.group(1)


VERD = {n: verdict(n) for n in ("stats", "leak", "leak_v2", "compare", "compare_v2", "define", "code", "code_v2", "code_v3")}
must(VERD == {"stats": "通す", "define": "通す", "leak": "直してから通す", "compare": "直してから通す", "code": "止める", "leak_v2": "通す", "compare_v2": "通す", "code_v2": "通す", "code_v3": "通す"}, f"検査の判定：{VERD}")
must("\n判定：条件つき OK\n" in REP["review_ml"] and "\n判定：OK\n" in REP["review_ml_v2"] and "\n判定：OK\n" in REP["review_affect"], "査読想定の役の判定")
for n, s_ in (("stats", "対象：`DAI/docs/実験21_設計.md` 第 1 版（コミット 9b961fa），台本の下書き `DAI/docs/tools_pending/dai21_eval5.py`（sha256 8011ef0ee5a68146dba57e93a3010b30a88dd8cb3bebec202cacdc1e187c40fd を確かめた）"),
              ("define", "対象：コミット 9b961fa の設計 `DAI/docs/実験21_設計.md`（第 1 版）"),
              ("review_affect", "対象：設計 `DAI/docs/実験21_設計.md` 第 1 版と台本の下書き `DAI/docs/tools_pending/dai21_eval5.py`（どちらもコミット 9b961fa．作業木はコミットと同じ）"),
              ("leak", "対象：`DAI/docs/実験21_設計.md` 第 1 版（コミット 9b961fa"), ("compare", "対象：`DAI/docs/実験21_設計.md` 第 1 版（コミット 9b961fa）"),
              ("review_ml", "検査した版：設計 `DAI/docs/実験21_設計.md` 第 1 版（コミット 9b961fa"),
              ("code", "# 関門 1：実装と出力（DAI-21，2026-11-14，コミット 9b961fa．台本 `DAI/docs/tools_pending/dai21_eval5.py` sha256 8011ef0e…40fd）"),
              ("leak_v2", "対象：コミット 6eeb53c（22:56）の `DAI/docs/実験21_設計.md`"), ("compare_v2", "対象：コミット 6eeb53c の設計 第 2 版 `DAI/docs/実験21_設計.md`"),
              ("review_ml_v2", "検査した版：設計 `DAI/docs/実験21_設計.md` 第 2 版（コミット 6eeb53c"),
              ("code_v2", "# 関門 1：実装と出力の再検査（DAI-21 第 2 版，2026-11-14，コミット 6eeb53c．台本 `DAI/docs/tools_pending/dai21_eval5.py` sha256 e77bb0cb…56dd）"),
              ("code_v3", "# 関門 1：実装と出力の差分の再検査（DAI-21 第 3 版，2026-11-14，コミット f518545．台本 `DAI/docs/tools_pending/dai21_eval5.py` sha256 48f77ef0…0c86）")):
    must(s_ in REP[n], f"{n}.md が検査した版の文")
must(not any((CK / f_).exists() for f_ in ("stats_v2.md", "define_v2.md", "review_affect_v2.md", "leak_v3.md", "compare_v3.md", "review_ml_v3.md", "code_v4.md")), "計算の前の検査に，この台本が知らない版がある")
must(prov["this_script_sha256"] in REP["code_v3"] and prov["this_script_sha256"] in code_post and prov["this_script_sha256"].endswith("5c86"), "流した台本の sha256 が code_v3.md の台本と違う")
must("48f77ef0…0c86" in REP["code_v3"].split("\n")[0] and "48f77ef0…0c86" in code_post[code_post.index("## 2. 台本の sha256 とコミット"):code_post.index("## 3. 検算の量")], "報告の sha256 の略し方（文を直す）")
must(not any("dai21_eval5.sh" in REP[n_] or "起動の台本" in REP[n_] for n_ in REP if n_ != "code_v3"), "DAI-21 の計算の前の報告に，起動の台本の検査の記録がある（文を直す）")
# 起動の台本は，実験 20 の計算の後に使う台本の検査（DAI/checks/DAI-20/code_post_tools.md・code_post_tools_v2.md）が，下書きの版を検査した
CK20 = ROOT / "DAI" / "checks" / "DAI-20"
TOOLS1, TOOLS2 = (CK20 / "code_post_tools.md").read_text(), (CK20 / "code_post_tools_v2.md").read_text()
RUN_PATH, RUN_OLD = ROOT / "DAI" / "run" / "dai21_eval5.sh", "DAI/docs/tools_pending/dai21_eval5.sh"
RUN_SHA = sha(RUN_PATH)
RUN_SHA1 = hashlib.sha256(git("show", f"79fa614:{RUN_OLD}").encode()).hexdigest()
must(hashlib.sha256(git("show", f"1f5b65e:{RUN_OLD}").encode()).hexdigest() == RUN_SHA and f"- `dai21_eval5.sh`：{RUN_SHA}" in TOOLS2 and RUN_SHA.startswith("933bf6a4"), "流した起動の台本が，code_post_tools_v2.md が検査した版と違う")
must(f"- `DAI/docs/tools_pending/dai21_eval5.sh`：{RUN_SHA1}" in TOOLS1 and RUN_SHA1.startswith("ebce1974") and RUN_SHA1 != RUN_SHA, "code_post_tools.md が検査した起動の台本の版")
must("## 判定\n\n直してから進む．" in TOOLS1 and "## 判定\n\n通す．" in TOOLS2 and "### 3. `dai21_eval5.sh`" in TOOLS1 and "## D．起動の台本（依頼の (e)）" in TOOLS2, "計算の後に使う台本の検査の判定と節")
must(git("log", "-1", "--format=%h", "--", "DAI/run/dai21_eval5.sh").strip() == "07c5d67" and git("status", "--porcelain", "--", "DAI/run/dai21_eval5.sh").strip() == "", "起動の台本の最後のコミット")
T1_SNAP = quote(TOOLS1, "### [I] 起動の台本：入力の照合（`snap`）に漏れがある（`dai21_eval5.sh`）", "code_post_tools.md")
T1_FRZ = quote(TOOLS1, "### [記録] 起動の台本：freeze の段が無いことと，ログの追記", "code_post_tools.md")
_ = quote(TOOLS2, "| 起動の台本の記録の指摘 | 入力の置き場が無ければ止まる．照らす入力を足した．freeze を先に済ませる前提を冒頭に書いた．起動のたびに前のログを消す | 下の D | 合う |", "code_post_tools_v2.md")
T_TOOLS = {n_: git("log", "--format=%ad", "--date=format:%Y-%m-%d %H:%M:%S", "--", f"DAI/checks/DAI-20/{n_}").strip().split("\n")[-1] for n_ in ("code_post_tools.md", "code_post_tools_v2.md")}
must(all(t_ < "2026-11-15 03:20:35" for t_ in T_TOOLS.values()), f"計算の後に使う台本の検査の報告が，起動より後にコミットされている（文を直す）：{T_TOOLS}")
# 計算の前の検査と査読想定の役が，結果の文・表の書き方について残した求め（報告の文のまま引く．この文書のどこで対応したかは 0 節の一覧）
RQ_SRC = {
    "S2": ("stats", "CREMA-D の表で印の無い行を「差が言えない行」と読むと，区間が 0 を含まない行を取りこぼす．"),
    "S3": ("stats", "引き直しの分散は塊の数 n について (n − 1)/n 倍に縮むので，SE は約 3% 小さく出る．この 2 つは同じ向きである．"),
    "S3b": ("stats", "実験 20 の設計 156 行の IEMOCAP・MELD・jackknife の列の読み方も写す"),
    "S5a": ("stats", "補正しない 95% の区間なので，差がすべて 0 でも片側 2.5% の割合で 0 をまたがない行が出うる．"),
    "S5b": ("stats", "3 つの K で同じ向きの区間が並んでも 3 回の独立な一致ではない．"),
    "S5c": ("stats", "`same_as` の立った行と作りの上で 0 になる行は独立な比べとして数えない"),
    "S6": ("stats", "区間の無い記述の値．テストの上の最良の点は 12 個の最大で上に偏る．主の点との差を選びの証拠にしない"),
    "S7": ("stats", "2 つの違いは `stud_prod − joint_AVT` の行と `joint_repro` で示す"),
    "S8": ("stats", "種 0・1 と種 2〜9 の値の違いには区間が無い"),
    "S38": ("stats", "区間が 0 を含まなくなるのは AFEW と CREMA-D だけで，IEMOCAP・MOSEI・MELD は区間が 0 をまたぐ見込みである．"),
    "L3": ("leak", "種 2〜9 の表は K 本の重なりだけを除く表で，テストの重なりは除かない"),
    "L2": ("leak", "CREMA-D の値は実験 18 の値と独立な確かめにならず，5 つ目のコーパスとして数えると，評価に足すコーパスを結果を見て選んだ形になる．"),
    "L4": ("leak", "出発点を今のままにした判断がその試しを見た後かどうかを 1 文書く．"),
    "LV3": ("leak_v2", "`weights_cremad.csv` の `w_orc` は候補のラベルを使う参照の列である．"),
    "C1": ("compare", "書き方：論文で `stud_prop` と実験 1 の 5 手法を並べるときは，(1) 提案の疑似ラベルはソースで測った表と K 本の作り直しを使い，相手は使わない，(2) 提案の形は同じテストの発話の上の値を見た後に決め，相手は実験 1 で固定した，"
                       "(3) 提案の (ρ, a) は相手の 1 つ（`joint_AVT`）のために選んだ点を借りた，の 3 つを表の注に書く．(1) と (2) は提案に有利な向き，(3) は不利な向きである．"),
    "C5": ("compare", "5 コーパスを 1 つの表に並べると，コーパスの違いに選び方の違いが重なる．とくに Q1 の主の差（`stud_prop − stud_prod`）は，4 コーパスでは ρ = 1，CREMA-D の K = 10・25 では ρ = 4 で測る．"),
    "C9a": ("compare", "CREMA-D の学び直しの結果を「既にある学び直しの手法との比べ」として書かない"),
    "C9b": ("compare", "4 コーパスの学び直しの表には，実験 20 の `others.csv` の実験 2 の手法の水準（単位の数 50 と 15，対の差ではない）を並べるか，並べないと書く．"),
    "C10": ("compare", "疑似ラベルごとに最大の点を取って並べるとオラクルの比べになる．"),
    "C16a": ("compare", "実験 18 の規則で選んだ点は，提案の疑似ラベルで ρ の上端 4 が 15 のうち 13，積の疑似ラベルで 15 のうち 8 である"),
    "C16b": ("compare", "`stud_prop_sel` と CREMA-D の主の点（K = 10・25）は格子の上端にあり，格子が狭いと，ρ を大きく取りたい側（提案の疑似ラベル）が不利になる．"),
    "C16c": ("compare", "`freeze` の段で，選んだ点が格子の端（ρ = 0.0625・4，a = 1・16）に当たった (コーパス, K, 疑似ラベル) の数を固定のファイルに書き，表の注に写す．"),
    "C11": ("compare", "`stud_* − none_*` の差には，K 本だけの融合のために選んだ引き戻しの強さの上乗せが学び直しに不利な向きに入る．実験 1 と同じ作りで，両方の疑似ラベルに同じに入る．"),
    "C12": ("compare", "同点の規則（実験 18 の学び直しの点は `np.nanargmax` で番号の小さい点．`dai18_probe.py:846`）で点が決まった (コーパス, K) の数は確かめていない"),
    "C14": ("compare", "推定時の答えの除去実験の内訳を，学び直した分類器の差の内訳として読まない"),
    "CV4a": ("compare_v2", "提案の追加の情報の分は，`stud_prop − stud_konly` と `stud_prop − stud_noprof` で測れる．"),
    "CV4b": ("compare_v2", "`noprof` は n0 = 0 の (T, K) ではソースの表を使わない．`tcal` はソースの表を使わず，K 本で合わせた重みを使う．"),
    "CV4c": ("compare_v2", "足した版どうしの比べと，足した版と `stud_prop` の比べには，選んだ分の上乗せが入らない．`stud_prod` との比べにだけ，積の側の上乗せが入る．"),
    "CV4d": ("compare_v2", "`stud_prop` も積の点を借りるので，「学び直しの形の違いの分」（`stud_prop − <手法>_prop`）は，両側とも積のために選んだ点を借りた差になる．"),
    "D2": ("define", "したがって「提案の疑似ラベルのために選んだ点」は，この実験の提案の疑似ラベルのために選んだ点ではない．"),
    "D4": ("define", "分類器が 2 つなので，分類器 × 感情の成分は 3 次元で，対照はそのうち Tukey の 1 次元を残して 2 次元を消す（3 つのときは 6 次元のうち 5 次元を消す）"),
    "D4b": ("define", "`prop` − `prop_noint` の値を 4 コーパスと CREMA-D で並べるときに効く．"),
    "D6a": ("define", "`mimic`（学び直した分類器の判定が疑似ラベルの判定と一致する割合）は記述の量で，主張に使わない．"),
    "D6b": ("define", "学び直しの差は重みの表と切片のずらしの両方を含む．"),
    "RA2": ("review_affect", "この偏りは K 本を使うすべての版（K 本だけの分類器，積，提案，学び直し）に入る．声の K 本の水準（提案が声に付ける重みの材料）を上げるか下げるかは，計算の前には分からない．"),
    "RA3": ("review_affect", "分類器が 2 つなので，相関は 8 個の数の相関になり偶然でも大きく振れる．表の数そのもの（声の行と顔の行の 4 つずつ）を並べる．"),
    "RA4": ("review_affect", "結果の文の段で，実験 18 の設計 8 節の記録と合わせて引き継ぐ．"),
    "RA4b": ("review_affect", "設定が CREMA-D に合わなかった場合の感度は，`student_grid.csv`（学び直しの 12 点）と CREMA-D の単位の保存（提案の格子 16 点の判定）で読める．"),
    "RA5a": ("review_affect", "RAVDESS はこの実験の計算に入らないが，設定を選んだ単位に入るので，利用条件の確認は論文の段に残る．"),
    "RA5d": ("review_affect", "配布元の README は「made available under the Open Database License」と書く（https://github.com/CheyneyComputerScience/CREMA-D ．今回開いて確かめた）．"),
    "RA5e": ("review_affect", "実験 18 の設計 9 節の「記憶による．未確認」は，CREMA-D については確かめた状態になった．"),
    "RA5b": ("review_affect", "配布元の README はこの発話を「error picture, no audio, very short」と書く．"),
    "RA5c": ("review_affect", "CREMA-D のラベルが演者に指示した感情であること，顔の特徴の 91.3% が 1 区間の平均であることは，実験 18 の設計 8 節の 1〜6 に記録がある．この設計の 8 節が引くのは文の重なりだけなので，結果の文の段で同じ節から引き継ぐ．"),
    "RM2": ("review_ml", "主の比べで差が 0 以下のとき，その差からは「提案の疑似ラベルが悪い」とは言えない（設定が積の側のために選ばれているため）．選択を含まない材料は 12 点の表（`student_grid.csv` と保存の判定）である．"),
    "RM5": ("review_ml", "設計 0 節の 1 が引く実験 18 の学び直しの差（+1.81 など）は，出どころの値を開いていないので照合していない．"),
    "RM10a": ("review_ml", "4 コーパスの `stud_prop_sel` も，ソースに CREMA-D・RAVDESS が入る実験 18 の単位で選んだ値である．"),
    "RM10b": ("review_ml", "CREMA-D で，提案と同じテストの上で先に値を見た版（積，K 本だけの表，ソースだけの表，全ラベルの表，選んだ点，2 通りの疑似ラベルの生徒）と，見ていない版（確率の平均，確信度の平均，温度の較正，K 本で合わせた重み，重みの表を探す版）の区別が設計に無い．"),
    "RM11": ("review_ml", "CREMA-D の顔の引き戻しの強さは 0.001（14 点の格子の端），学び直しの ρ は K ＝ 10・25 で 4（格子の端）である．どの方法にも同じ値が入るので比べの公平さには関わらない．"),
    "K1": ("code_v2", "提案の部品を外した版 6 つのうち，`prop_n0zero`・`prop_home`・`prop_noprof_sel` は疑似ラベルにしていない"),
    "KV4": ("code_v2", "`agree`・`mix` が残した疑似ラベルの本数（`make_G` が書く `n_kept`）は csv に残らない（保存から作り直せる）"),
    "KW4": ("code_v3", "`same_pred` は `same_as` と同じく全部の単位で付ける印で，種で分けた行（`0-1`・`2-9`）にも全部の単位の印が入る．"),
}
RQ = {k_: quote(REP[n_], s_, f"{n_}.md") for k_, (n_, s_) in RQ_SRC.items()}
RQ_FILE = {k_: f"`{n_}.md`" for k_, (n_, _) in RQ_SRC.items()}
design18 = (ROOT / "DAI" / "docs" / "実験18_下調べ_設計.md").read_text()
design20 = (ROOT / "DAI" / "docs" / "実験20_設計.md").read_text()
E18_LABEL = quote(design18, "ラベルは演者に指示した感情で，聞き手の判断ではない．", "実験 18 の設計 8 節")
E18_FACE = quote(design18, "CREMA-D の顔の特徴は 91.3% の発話で 1 区間の平均である．", "実験 18 の設計 8 節")
E18_LIC = quote(design18, "利用条件（論文の段で申告する．原文で確かめる）：CREMA-D は ODbL，RAVDESS は CC BY-NC-SA 4.0（記憶による．未確認）．", "実験 18 の設計")
# 設計 0 節の 1 が引く実験 18 の学び直しの差（K = 5）を，実験 18 の集計と照らす（ML の査読想定は照合していないと記録した）
se18 = pd.read_csv(ROOT / "results" / "dai18_probe" / "summary" / "students_effects.csv", low_memory=False)
E18V = {c: float(se18[(se18.c == c) & (se18.K == 5) & (se18.student == ("AV" if c == CD else "AVT")) & (se18.a == "M_F6_mixsel") & (se18.b == "prod")].est.iloc[0]) for c in C5}
must([f"{E18V[c]:+.2f}" for c in C5] == ["+1.81", "+0.23", "+0.19", "+0.66", "+0.49"], f"設計 0 節の 1 が引く実験 18 の値：{E18V}")

# ================================================================== 文

L: list[str] = []
w = L.append


def gdesc(i: int) -> str:
    n0, b, p = GRID[i]
    return f"点 {i}（n0 = {'∞' if n0 == INF else int(n0)}，β = {int(b)}，{p}）"


def sdesc(g: int) -> str:
    return f"点 {g}（ρ = {SGRID[g][0]:g}，a = {SGRID[g][1]}）"


def cnt(get, pairs, kinds) -> int:
    g = groups(get, pairs)
    return sum(len(g[k]) for k in kinds)


def rng_of(get, pairs) -> str:
    xs = [get(c, K).est for c, K in pairs]
    return f"{fest(min(xs))}〜{fest(max(xs))}"


def read_multi(opps: list, mk) -> str:
    """CREMA-D で，相手の一覧を通した読み．mk(v) は差の対 (a, b) を返す．区間が 0 を含み点推定が正の条件は名前だけを書く．"""
    items = [(v, K, q(*mk(v), CD, K)) for v in opps for K in KS]

    def fmt(sel, with_val: bool) -> str:
        out = []
        for v in opps:
            ks = [(K, r) for v2, K, r in sel if v2 == v]
            if ks:
                out.append(f"{nmc(v)}の " + ("，".join(f"K = {K} で {ci(r)}" for K, r in ks) if with_val else f"K = {kjoin(K for K, _ in ks)}"))
        return "；".join(out)

    s = []
    for k, head in (("pp", "区間が 0 を含まず +1.0 点以上："), ("nn", "区間が 0 を含まず −1.0 点以下："), ("p", "区間が 0 を含まず 0〜+1.0 点："), ("n", "区間が 0 を含まず −1.0〜0 点：")):
        sel = [x for x in items if klass(x[2]) == k]
        if sel:
            s.append(head + fmt(sel, True))
    zneg = [x for x in items if klass(x[2]) == "z" and x[2].est < 0]
    zpos = [x for x in items if klass(x[2]) == "z" and x[2].est >= 0]
    if zneg:
        s.append("区間が 0 を含み点推定が負：" + fmt(zneg, True))
    if zpos:
        s.append("区間が 0 を含み点推定が正：" + fmt(zpos, False))
    same = [x for x in items if klass(x[2]) == "same"]
    if same:
        s.append("2 つの版が同じ判定（差は 0）：" + fmt(same, False))
    return "．".join(s) + "．"


def n_multi(opps: list, mk, kinds) -> int:
    return sum(klass(q(*mk(v), CD, K)) in kinds for v in opps for K in KS)


def brk(qs: list) -> dict:
    """区間の読みの内訳：hi は区間が 0 を含まず正，lo は 0 を含まず負，z は 0 を含む（zn はそのうち点推定が負），zero は差がちょうど 0．"""
    k = [klass(x) for x in qs]
    return dict(n=len(qs), hi=sum(x in ("pp", "p") for x in k), lo=sum(x in ("nn", "n") for x in k), z=sum(x == "z" for x in k),
                zn=sum(x == "z" and y.est < 0 for x, y in zip(k, qs)), zero=sum(x in ("same", "cnt") for x in k))


S_PP = eget("stud_prop", "stud_prod")
S_J = eget("stud_prop", "joint_AVT")

# ---- 照合（関門 2）の第 1 回の報告：判定と指摘の数だけを読む（報告の中の数は，文の値に使わない）
claim1 = (CK / "claim.md").read_text()
must("## 判定\n\n直してから通す．下書きの数値・区間・読みの区分・同じ判定の扱いはすべて結果ファイルと一致し" in claim1 and "第 1 版．820 行．コミット 8994e38" in claim1, "照合の第 1 回の判定と対象")
N_CLAIM1 = claim1.count("\n### [")
must(N_CLAIM1 == 10 and claim1.count("- 重さ：記録する") == 10, f"照合の第 1 回の指摘の数：{N_CLAIM1}")
V1_TEXT = git("show", "8994e38:DAI/docs/実験21_結果の文の下書き_2026-11-15.md")           # 照合の第 1 回が見た第 1 版
claim2 = (CK / "claim_v2.md").read_text()                                 # 照合の第 2 回：判定，新しい指摘の数，引く 2 文だけを読む
must("## 判定\n\n通す．第 1 回の指摘 10 件はどれも直っており，直しで値は変わっていない．" in claim2 and "第 2 版．838 行．コミット a020495" in claim2, "照合の第 2 回の判定と対象")
C2_NEW = quote(claim2, "新しい指摘 3 件は，どれも論文のセッションへの伝言の書き方についての「記録する」で，結果の文の下書きには直す点が無い．", "claim_v2.md")
N_CLAIM2 = claim2.count("\n### [")
must(N_CLAIM2 == 3, f"照合の第 2 回の新しい指摘の数：{N_CLAIM2}")
C2_OK = quote(claim2, "は，結果の文としては足りる．", "claim_v2.md")
C2_STOP = quote(claim2, "論文の本文や発表資料に CREMA-D の値を入れる段で，感度の表がまだ無ければ，その段の照合で「止める」に当たる．", "claim_v2.md")
V2_TEXT = git("show", "a020495:DAI/docs/実験21_結果の文の下書き_2026-11-15.md")           # 照合の第 2 回が見た第 2 版
claim3 = (CK / "claim_v3.md").read_text()                                 # 照合の第 3 回：判定，指摘の数，引く文だけを読む（報告の中の値は，文の値に使わない）
must("## 判定\n\n直してから通す．足した値・数え・区間はすべて出力の csv と一致した．" in claim3 and "第 3 版．1,393 行．コミット 8187af5" in claim3, "照合の第 3 回の判定と対象")
N_CLAIM3 = claim3.count("\n### [")
must(N_CLAIM3 == 5 and claim3.count("- 重さ：記録する") == 5, f"照合の第 3 回の指摘の数：{N_CLAIM3}")
C3_OK = quote(claim3, "足した値・数え・区間はすべて出力の csv と一致した．", "claim_v3.md")
C3_BOOT = quote(claim3, "書き手が作り直していないと申告した話者の引き直しの区間も，別の乱数で独立に作り直すとモンテカルロの誤差の範囲で一致し，文を塊にした区間は 1 桁目まで一致した．", "claim_v3.md")
C3_SE = quote(claim3, "標準誤差の比（私の値 ÷ csv）は，文を除いたテストの差で 0.983〜1.010，「除いた値 − 全発話の値」で 0.985〜1.048 だった．", "claim_v3.md")
C3_FLIP = quote(claim3, "別の乱数で 2,000 組を引いて作り直すと，[−0.338, +0.005] で 0 を含んだ（`check_v3_sensitivity_out.txt`）．", "claim_v3.md")
C3_KEEP = quote(claim3, "ほかの 3 個（上端 −0.025・−0.050，下端 +0.010）は，私の作り直しでも 0 を含まなかった．", "claim_v3.md")
C3_DIR = quote(claim3, "懸念が予想するのは，K 本で学ぶと文 `IEO` の Neutral の再現率が下がる向きである．", "claim_v3.md")
V3_TEXT = git("show", "8187af5:DAI/docs/実験21_結果の文の下書き_2026-11-15.md")           # 照合の第 3 回が見た第 3 版
claim4 = (CK / "claim_v4.md").read_text()                                 # 照合の第 4 回：判定，残る点の数，引く文だけを読む（報告の中の値は，文の値に使わない）
must("## 判定\n\n通す．第 3 回の指摘 5 件はどれも直っていて，第 3 版の値は 1 つも変わっていない．" in claim4 and "第 4 版（1,417 行．コミット 8f46972）" in claim4, "照合の第 4 回の判定と対象")
N_CLAIM4 = claim4.count("\n### [")
must(N_CLAIM4 == 2 and claim4.count("- 重さ：記録する") == 2, f"照合の第 4 回の残る点の数：{N_CLAIM4}")
C4_OK = quote(claim4, "第 3 回の指摘 5 件はどれも直っていて，第 3 版の値は 1 つも変わっていない．", "claim_v4.md")
C4_MATCH = quote(claim4, "足した Neutral の再現率の表と要点 16 の文，「＊」の基準と数と付け方は，結果ファイルから作り直した値と一致した．", "claim_v4.md")
C4_USE = quote(claim4, "これも照合（関門 2）の前の使用である．", "claim_v4.md")
V4_TEXT = git("show", "8f46972:DAI/docs/実験21_結果の文の下書き_2026-11-15.md")           # 照合の第 4 回が見た第 4 版
must("第 4 版．照合の第 3 回の直しを入れた" in V4_TEXT.split("\n")[0] and len(V4_TEXT.split("\n")) - 1 == 1417, "照合の第 4 回が見た第 4 版（コミット 8f46972）の題と行の数")
must("第 3 版．CREMA-D の感度の表を足した．照合の第 3 回の前" in V3_TEXT.split("\n")[0] and len(V3_TEXT.split("\n")) - 1 == 1393, "照合の第 3 回が見た第 3 版（コミット 8187af5）の題と行の数")

# ---- 冒頭
w("# 実験 21（DAI-21．提案の手法の評価の回の続き：学び直し 5 コーパスと CREMA-D の評価の回）：結果の文の下書き（2026-11-15．第 4 版．照合済み：gate-claim 第 4 回 `claim_v4.md` 通す．記録 2 件を入れた）")
w("")
w("この文書は，台本 `DAI/docs/tools_pending/extract_dai21_text.py` が `results/dai21_eval5/` の結果ファイルから数値を取り出して書き出した．"
  "結果ファイルから取る数値と表は，すべて台本が埋めた（手で写していない）．設計・点検の報告・台帳から引いた文は，引いた文が元の文書にあることを台本が確かめた．"
  "文が言う数えと大小は，台本が結果ファイルと照らしてから書き出している．集計の点推定は，台本が単位の保存（判定の配列）から別の書き方で作り直した値と照らした（0 節）．文を直すときは，台本の中の文を直して流し直す．"
  f"第 1 版（コミット 8994e38）への照合の第 1 回（関門 2．`DAI/checks/DAI-21/claim.md`）の判定は「直してから通す」だった．報告は，下書きの数値・区間・読みの区分・同じ判定の扱いがすべて結果ファイルと一致したと書き，指摘は {N_CLAIM1} 件（どれも重さは「記録する」）だった．"
  f"その {N_CLAIM1} 件の直しを入れた第 2 版（コミット a020495）への照合の第 2 回（`DAI/checks/DAI-21/claim_v2.md`）の判定は「通す」だった．報告は，第 1 回の指摘 {N_CLAIM1} 件がどれも直り，直しで値が変わっていないと書いている．"
  f"新しい指摘は {N_CLAIM2} 件で，報告は「{C2_NEW}」と書いている．"
  "第 3 版（コミット 8187af5）は，照合を通った第 2 版に，設計 4 節が予告した CREMA-D の感度の表（計算の後の解析．`results/dai21_eval5/posthoc/`）の値と読み方を足した版である．"
  "足したのは，要点 13〜18，2.7 節，3.3 節の仮定の表，0 節の「計算の後の解析」で，要点 7・10，2.6 節，3.3 節，4 節にあった「感度の表はまだ出していない」という文を直した．第 2 版の値と表は変えていない．"
  f"第 3 版への照合の第 3 回（`DAI/checks/DAI-21/claim_v3.md`）の判定は「直してから通す」だった．報告は「{C3_OK}」と書き，指摘は {N_CLAIM3} 件（どれも重さは「記録する」）だった．"
  f"第 4 版は，その {N_CLAIM3} 件の直しを入れた版である．"
  "直したのは，要点 16（設計の懸念に直接当たる量である，文「It's eleven o'clock.」の Neutral の再現率の並びを足し，太字の文の範囲を狭めた），要点 14・15（引き直しの乱数に依りうる区間の注），"
  "要点 17（K 本の引き方を 2 つに分けた並びを外した），要点 18 の 1 文，0 節の「照合の前に値を使った先」の (5) である（変更の一覧は 6 節）．第 3 版の値は変えていない．"
  f"**第 4 版（コミット 8f46972）への照合の第 4 回（`DAI/checks/DAI-21/claim_v4.md`）の判定は「通す」だった**．報告は「{C4_OK}{C4_MATCH}」と書き，残る点は {N_CLAIM4} 件（どちらも重さは「記録する」）だった．"
  f"この文書は，照合を通った第 4 版に，その {N_CLAIM4} 件（要点 16 の上限の声の分類器についての文と，0 節の (5) の本体の申告）を入れ，題と，照合の状態を書いた文を「照合済み」に合わせたものである．ほかの文と値は変えていない（変更の一覧は 6 節）．"
  "論文のセッションへの伝言の 0 節の 10 は，照合が通った後に，CREMA-D の値を論文や資料に入れるときは感度の表と一緒に書く，という項に書き換えられた（0 節）．")
w("")
w("値は UAR（4 感情の再現率の平均．%．でたらめに答えると 25%）で，差はすべて UAR の点の差である．角かっこは 95% 区間で，補正していない．")
w("")
# 引き直しで作る区間（集計の差の区間のうち，MOSEI と CREMA-D）のうち，端が 0 に近い区間を全部数える
BOOT_ROWS = [r for r in ef.itertuples() if str(r.method).startswith("boot") and klass(r) not in ("same", "cnt")]
must({r.c for r in BOOT_ROWS} == {"MOSEI", CD} and all(str(r.method).startswith("jk") for r in ef.itertuples() if r.c in ("AFEW", "IEMOCAP", "MELD"))
     and max(max(abs(r.lo - (r.est - 1.96 * r.se)), abs(r.hi - (r.est + 1.96 * r.se))) for r in BOOT_ROWS) < 1e-9
     and max(max(abs(r.lo - (r.est - 1.96 * r.se)), abs(r.hi - (r.est + 1.96 * r.se))) for r in ef20.itertuples() if str(r.method).startswith("boot") and r.se > 0) < 1e-9,
     "引き直しで作る区間のコーパスと，区間が 点推定 ± 1.96 × 標準誤差 であること")
NEAR_EF = [r for r in BOOT_ROWS if near0(r)]


def n_boot(c: str, allseeds: bool, near: bool) -> int:
    return sum(r.c == c and (r.seeds == "all") == allseeds for r in (NEAR_EF if near else BOOT_ROWS))


NEAR_MOSEI = [r for r in NEAR_EF if r.c == "MOSEI" and r.seeds == "all"]
must([(r.a, r.b, int(r.K)) for r in NEAR_MOSEI] == [("stud_prop", "stud_prod", 10), ("stud_prop", "stud_konly", 25)], f"MOSEI の，端が 0 に近い区間（文を直す）：{[(r.a, r.b, int(r.K)) for r in NEAR_MOSEI]}")
w("**「＊」の印**（第 4 版で足した）：MOSEI と CREMA-D の区間は，塊の重みの引き直しで作る区間で，乱数を使う（0 節の「区間の作り方」）．"
  f"この文書は，引き直しで作る区間のうち，0 に近いほうの端が 0 から標準誤差の {NEAR_TH} 倍以内にある区間の値の後に「＊」を付ける．"
  "**＊ の付いた区間は，区間の端が 0 に近く，0 を含むかどうかが引き直しの乱数の組に依りうる**（基準は要点 14・15 と同じである．照合の第 3 回の指摘を受けて，第 4 版で決めた）．"
  "jackknife で作る区間（AFEW・IEMOCAP・MELD の区間と，CREMA-D の文を塊にした区間）は乱数を使わないので，印を付けない．"
  f"集計の差の区間（`effects.csv`）のうち，この基準に当たるのは，K 本の引き方 0〜9 の行では，MOSEI が {n_boot('MOSEI', True, False)} 個のうち {n_boot('MOSEI', True, True)} 個（"
  + "，".join(f"`{r.a} − {r.b}` の K = {int(r.K)}" for r in NEAR_MOSEI) + f"），CREMA-D が {n_boot(CD, True, False)} 個のうち {n_boot(CD, True, True)} 個（名前は要点 15）である．"
  f"K 本の引き方で分けた行では，MOSEI が {n_boot('MOSEI', False, False)} 個のうち {n_boot('MOSEI', False, True)} 個，CREMA-D が {n_boot(CD, False, False)} 個のうち {n_boot(CD, False, True)} 個である（両側が同じ判定の行と，正解の数が同じ行は数えていない）．"
  "この文書が値を書いた区間のうち，この基準に当たるものには，全部 ＊ が付いている（台本が確かめた．要点 1〜12 では ＠要点1〜12の＊＠ 箇所）．"
  "区間が 0 を含まない条件の数えと名前は，＊ の有無で変えていない．")
w("")
DQ_KIND = quote(design, "**種類**：本番の評価の回の追加．判定の規則は置かず，結果を表で見る（実験 20 と同じ扱い）．", "設計 0 節")
DQ_KS = quote(design, "K は 5・10・25（同じ扱いで並べる）．", "設計 2 節")
w(f"**この実験には判定の規則が無く，主の K も無い**（設計 0 節：「{DQ_KIND}」．設計 2 節：「{DQ_KS}」）．差は次の 3 つに分けて読む．"
  "(1) 区間が 0 を含まず，差が 1.0 点以上．(2) 区間が 0 を含まず，差が 1.0 点未満．(3) 区間が 0 を含む．"
  "表の印（↑・↓）は (1) に付く．印は表を読みやすくするための印で，判定ではない．(3) の差は，2 つの版が同じ水準であることを意味しない．K どうしの差の区間は作っていないので，K の間の値の違いを差の証拠として読まない．")
w("")
w("**設計 0 節の事実の記録**（内部の記録の 6 項．設計の文のまま）：")
w("")
D0 = [quote(design, "学び直しを足すと決めたのは，実験 18 の結果（提案の答えを疑似ラベルにして学び直した融合の分類器 − 積を疑似ラベルにした融合の分類器が，K = 5 で AFEW +1.81，IEMOCAP +0.23，MOSEI +0.19，MELD +0.66，CREMA-D +0.49）と，"
                    "実験 18 の値を実験 1・2 の評価の回の値と並べた表（22:01）を見た後である．", "設計 0 節"),
      quote(design, "CREMA-D を評価するコーパスに足す案が承認されたのは 20:55 で，実験 18 の集計（CREMA-D の値を含む）を開いた 20:39 の後である．CREMA-D は，元の 4 コーパスと並ぶ独立な確かめとして数えない．", "設計 0 節"),
      quote(design, "話者の分割は固定なので，どの種でも，テストの発話は実験 18・16・1 のテストと同じ発話である．種で変わるのは K 本だけである．種 0・1 は K 本も下調べと同じである．「種 2〜9 でも同じ向き」を再現の確認として書かない．", "設計 0 節"),
      quote(design, "CREMA-D の種 0・1 の単位は，実験 18 の「RAVDESS を触らずに残した単位」とソース（4 コーパス）・K 本が同じである．違うのは引き戻しの強さの選び方だけで，K = 25 では強さも同じ値になる．", "設計 0 節"),
      quote(design, "CREMA-D のソースは，元の 4 コーパスの全発話で学んだ分類器である．元の 4 コーパスの評価のテストの発話も，その学習に入る．5 コーパスの結果は互いに独立ではない．Q1 はコーパスごとの記述として読む．", "設計 0 節"),
      quote(design, "1〜5 から，この実験の値は，手法を選んだときに見たテストと同じテストの上の値であり，探索の結果として扱う．", "設計 0 節")]
for i_, s_ in enumerate(D0, 1):
    w(f"{i_}. {s_}")
w("")
w(f"1 の項が引く実験 18 の値（K = 5 の 5 つ）は，この台本が実験 18 の集計（`results/dai18_probe/summary/students_effects.csv`）と照らし，同じ値であることを確かめた．計算の前の ML の査読想定（`review_ml.md`）は「{RQ['RM5']}」と記録していた．")
w("")
w("設計の文の「種」は，K 本の引き方の番号（台本の `seed`．0〜9）である．この文書では「K 本の引き方」と書く．「条件」は，コーパスと K の組を指す．"
  "元の 4 コーパスの 12 条件は 4 コーパス × 同じテストを共有する 3 つの K，5 コーパスの 15 条件は 5 コーパス × 同じテストを共有する 3 つの K で，独立な 12 個・15 個の測定ではない．")
w("")

# ---- 0 節
w("## 0. 出どころと申告")
w("")
w(f"- **設計**：`DAI/docs/実験21_設計.md`（第 3 版．計算の前．sha256 {prov['design_sha256'][:8]}…．3 つの段の記録の `design_sha256` と同じで，今のファイルとも同じ．最後のコミットは {D21_LAST['設計'][0]}（{D21_LAST['設計'][1][5:16]}））．")
must(all(x.startswith("DAI/checks/DAI-20/") or x in ("DAI/docs/tools_pending/extract_dai20_text.py", "DAI/docs/tools_pending/md_tables_dai20.py", "DAI/docs/tools_pending/md_tables_dai21.py",
                                                    "DAI/docs/実験20_結果の文の下書き_2026-11-15.md", "DAI/docs/実験20_結果の表_2026-11-15.md", "DAI/docs/実験の地図.md", "docs/seen-data.md")
         for x in BETWEEN_FILES), f"3 つの段のあいだに変わったファイル（文を直す）：{BETWEEN_FILES}")
w(f"- **台本とコミット**：`DAI/exp/dai21_eval5.py`（sha256 {prov['this_script_sha256'][:8]}…．3 つの段の記録で同じ値で，今の台本とも同じ）．起動の台本は `DAI/run/dai21_eval5.sh`．"
  f"**学び直しの段（students4．元の 4 コーパスの 200 単位）はコミット {C_S4[:7]}，CREMA-D の段（cremad．50 単位）はコミット {C_CD[:7]}，集計の段（summary）はコミット {C_SUM[:7]} で流れた**．"
  "dirty は 3 つの段とも偽である（`provenance_log.jsonl` の 3 行）．`provenance.json` は集計の段の記録で，そこに書いてあるコミットは集計のコミットである．"
  "単位のコミットは `provenance_log.jsonl` の 1・2 行目，`summary/convergence.json` の `students4.commits`・`cremad.commits`，単位の csv の `commit` の列（学び直しの 200 単位の全行で 1 種類，CREMA-D の 50 単位の全行で 1 種類）にある．"
  f"コミットが段で違うのは，計算のあいだに本体がほかのファイルをコミットしたためである．学び直しの段と CREMA-D の段のあいだに {N_BETWEEN['s4_cd']} コミット，CREMA-D の段と集計の段のあいだに {N_BETWEEN['cd_sum']} コミットがある．"
  "そのあいだに変わったファイルは，実験 20 の検査の報告（`DAI/checks/DAI-20/`），実験 20 の結果の文・表の文書と実験の地図（`DAI/docs/`），表と文を書き出す台本（`DAI/docs/tools_pending/` の `extract_dai20_text.py`・`md_tables_dai20.py`・`md_tables_dai21.py`），台帳（`docs/seen-data.md`）である．"
  f"**計算に関わるファイル（台本と読み込む部品の計 {len(prov['scripts']) + 1} 本，2 つの固定のファイル，`DAI/splits`，`measure`，`features`，`data/metadata`，`requirements.txt`，設計．`provenance.json` の `consts.WATCH` の一覧）の差分は，"
  "学び直しの段と CREMA-D の段のあいだ，CREMA-D の段と集計の段のあいだ，学び直しの段と今の HEAD のあいだのどれでも 0 行である**（この台本が `git diff` で確かめた）．"
  f"読み込む部品 {len(prov['scripts'])} 本と入力 {len(prov['inputs_sha256'])} 個の sha256 は 3 つの段の記録で同じで，部品は今のファイルとも同じである．")
DQ_TETH = quote(design, "出発点への L2 正則化の強さは，実験 18 が CREMA-D を触らずに残した単位だけで選んだ値（融合は，その単位の 3 モダリティの融合の値を借りる．CREMA-D のテストで選ばない）．", "設計 3.3 節")
DQ_RAV = quote(design, "設定を選んだ実験 18 の単位では，声のソースに RAVDESS が入る（固定のファイルの `corpora_in_units` に残る）．この実験の CREMA-D の単位のソースは 4 コーパスで，RAVDESS は入らない．", "設計 8 節")
w(f"- **固定のファイル**：`DAI/frozen/dai21_frozen.json`（sha256 {prov['frozen_sha256'][:8]}…．3 つの段の記録で同じ値で，今のファイルとも同じ．ファイルの中の記録では {frozen['time']} にコミット {frozen['commit'][:7]} の上で作り，コミット {D21_LAST['固定のファイル'][0]} に入れた）．"
  f"学び直しの設定（ρ・a）の主の点は，元の 4 コーパスの 12 条件とも{sdesc(GM[('AFEW', 5)])}，CREMA-D は K = 5 が{sdesc(GM[(CD, 5)])}，K = 10・25 が{sdesc(GM[(CD, 10)])}である（設計 3.1 節の記述と同じ）．"
  f"提案の手法の設定は，元の 4 コーパスが実験 20 の固定のファイル（sha256 {prov['frozen20_sha256'][:8]}…）の点，CREMA-D が 3 つの K とも{gdesc(GI[(CD, 5)])}である（設計 3.3 節：「{DQ_PROPCD}」）．"
  f"固定のファイルの記録では，CREMA-D の点は 40 単位で選んであり，その単位に現れたコーパス（`corpora_in_units`）は {'・'.join(FZC[5]['corpora_in_units'])} で，CREMA-D は現れない．"
  f"RAVDESS が現れることについて，設計 8 節は「{DQ_RAV}」と書いている．固定のファイルの元の 5 つのファイル（実験 18 の選んだ設定 3 つ，実験 20 の固定のファイル，実験 1 の固定の設定）の sha256 は，今のファイルと同じである．CREMA-D の引き戻しの強さについて，設計 3.3 節は「{DQ_TETH}」と書いている（値は 2.6 節）．選んだ点の一覧は 2.6 節．")
w("- **結果の場所**：`results/dai21_eval5/`．集計は `summary/`（7 個の csv と `convergence.json`），単位の保存は `students4/`（元の 4 コーパスの 200 単位の csv と npz．学び直した分類器の判定と重み）と `cremad/`（CREMA-D の 50 単位の csv と npz），"
  "設定とコミットの記録は `provenance.json`・`provenance_log.jsonl`．検算 `check18` の出力は `results/dai21_eval5_check18/`，ログは `logs/dai21_*.log`．")
DISP20 = [(c, K) for c, K in ALL12 if ci(q("prop", "joint_AVT", c, K), star=False) != ci(q20("prop", "joint_AVT", c, K), star=False)]
must(all(c == "MOSEI" for c, _ in DISP20), "実験 20 の集計と書いた桁で違う条件が MOSEI のほかにある")
w(f"- **実験 20 の集計との使い分け**（点検の報告の記録 1）：この文書の値は，1 つの例外を除いて，実験 21 の集計（`results/dai21_eval5/summary/`）から引く．"
  "**例外は，元の 4 コーパスの「提案の手法（推定時の答え）− 積」で，実験 21 の集計に無いので，実験 20 の集計（`results/dai20_eval/summary/effects.csv`）から引く**（要点 6 と 2.3 節．引いた箇所にそのつど書く）．"
  f"2 つの集計の両方にある版は，元の 4 コーパスの {len(BOTH)} 版（提案の手法の推定時の答え，積，K 本だけの融合，今の提案の実験 1 の保存，実験 1 の 5 つの比較手法）で，UAR の点推定は 2 つの集計で同じである（ずれの最大 {sci(DEV20_UAR) if DEV20_UAR > 0 else '0'}）．"
  f"区間は，AFEW・IEMOCAP・MELD（jackknife）では同じで，MOSEI（塊の重みの引き直し）では乱数の鍵が実験ごとに違うので少し違う（水準の区間の端のずれの最大 {DEV20_END['MOSEI']:.3f} 点）．"
  f"両方の集計にある差の対は「提案の手法（推定時の答え）− 今の提案（実験 1 の保存）」（`prop − joint_AVT`）の 1 つで，点推定は同じ，MOSEI の区間の端のずれの最大は {DEV20_EFEND['MOSEI']:.4f} 点である（K 本の引き方で分けた行を含む）．"
  "この文書は，この対と上の版の区間を実験 21 の集計から引く．"
  + (f"この文書の桁で書くと，実験 20 の集計の値と書いた数字が違う条件は {by_corpus(DISP20)} である．" if DISP20 else "この文書の桁で書くと，この対の 12 条件の値は，実験 20 の集計から引いた値と同じ数字になる．"))
LQ_START = quote(ROW21, "実験 20 の点検が通った後，2026-11-15 03:20:35 に tmux `mer-da:dai21` で起動した（台本の sha256 48f77ef0…，設計は 918ebea，固定のファイルは 07c5d67）", "台帳の DAI-21 の行")
LQ_END = quote(ROW21, "2026-11-15 05:17:11 に終わった（`logs/dai21_eval5.done`．学び直し 200 単位，CREMA-D 50 単位，集計．所要 1 時間 57 分）", "台帳の DAI-21 の行")
must(done == "2026-11-15 05:17:11" and prov["this_script_sha256"].startswith("48f77ef0"), "logs/dai21_eval5.done の時刻と台本の sha256")
must(T_HEAD["students4"] <= T_S4 and T_HEAD["cremad"] <= T_CD and T_HEAD["summary"] <= T_SUM, "段の見出しの時刻と由来の記録の時刻")
w(f"- **起動と終了**：台帳の DAI-21 の行は「{LQ_START}」「{LQ_END}」と書いている．ログと記録の時刻は，学び直しの段の始まりが {T_HEAD['students4']}，CREMA-D の段の始まりが {T_HEAD['cremad']}，集計の段の始まりが {T_HEAD['summary']}"
  f"（`logs/dai21_eval5.log` の見出し．`provenance_log.jsonl` の 3 行の時刻は {T_S4[11:]}・{T_CD[11:]}・{T_SUM[11:]}），終わりが {done}（`logs/dai21_eval5.done`）である．"
  f"単位の所要の合計は，学び直しの 200 単位が {SEC4:,.0f} 秒（{SEC4 / 60:.1f} 分），CREMA-D の 50 単位が {SECC:,.0f} 秒（{SECC / 60:.1f} 分）である（単位の csv の `sec`）．"
  f"計算機は研究室の PC（`{HOSTS[0]}`）の GPU 1 枚（{GPUS[0]}）で，250 単位とも同じ計算機である．ログに「済み（飛ばす）」の行は {N_SKIP} 行で，段の見出しは 3 つの段に 1 回ずつなので，途中から流し直していない．ログの最後は `EXIT=0` である．"
  "学び直しの段の前に，検算の 3 つの段（selftest・check3・check18）が同じ起動の中で流れて通った（3.1 節）．")
# ---- 計算の前の検査：台本と設計の版
OLD_PY = "DAI/docs/tools_pending/dai21_eval5.py"                         # 計算の前の台本の置き場（実験 20 の計算のあいだ）
must(all(hashlib.sha256(git("show", f"{h_}:{OLD_PY}").encode()).hexdigest() == prov["this_script_sha256"] for h_ in ("f518545", "918ebea")), "f518545・918ebea の台本の sha256")
must(git("diff", "--stat", "f518545", "918ebea", "--", OLD_PY).strip() == "" and git("diff", "--stat", "f518545", "918ebea", "--", "DAI/docs/実験21_設計.md").strip() != "", "f518545 と 918ebea のあいだの差分")
DQ_G1 = quote(design, "第 1 版への判定：統計「通す」，定義「通す」，感情認識の査読想定「OK」，漏れ「直してから通す」，比較「直してから通す」，ML の査読想定「条件つき OK」，台本「止める」", "設計 10 節")
DQ_G2 = quote(design, "第 2 版への判定（再検査した 4 体）：漏れ「通す」（`leak_v2.md`），比較「通す」（`compare_v2.md`），ML の査読想定「OK」（`review_ml_v2.md`），台本「通す」（`code_v2.md`）．残った指摘はどれも記録で，第 3 版に入れた．", "設計 10 節")
DQ_G3 = quote(design, "第 3 版の台本の差分への判定：台本「通す」（`code_v3.md`．台本の sha256 48f77ef0…）．記録の指摘 3 件（`same_as` の例のコーパス，積を渡した版の役，まとめた値の置き場）は設計の文に入れた．台本は変えていない．", "設計 10 節")
KQ_RUN = quote(REP["code_v3"], "起動の台本の下書きは今回の検査の範囲（台本 .py の差分）の外なので，検査していない．", "code_v3.md")
KQ_MOVE = quote(REP["leak_v2"], "台本を `DAI/exp/` へ移した後の sha256 が同じであることを `provenance.json` で確かめる", "leak_v2.md")
w(f"- **計算の前の検査**（`DAI/checks/DAI-21/`）：設計 10 節は「{DQ_G1}」「{DQ_G2}」「{DQ_G3}」と書いている．台本が報告のファイルで確かめたこと：第 1 版（コミット 9b961fa．台本の下書きの sha256 8011ef0e…）を見たのは 7 体で，"
  f"判定は `stats.md`「{VERD['stats']}」，`define.md`「{VERD['define']}」，`review_affect.md`「判定：OK」，`leak.md`「{VERD['leak']}」，`compare.md`「{VERD['compare']}」，`review_ml.md`「判定：条件つき OK」，`code.md`「{VERD['code']}」である．"
  f"第 2 版（コミット 6eeb53c．台本の sha256 e77bb0cb…）を見たのは 4 体で，`leak_v2.md`「{VERD['leak_v2']}」，`compare_v2.md`「{VERD['compare_v2']}」，`review_ml_v2.md`「判定：OK」，`code_v2.md`「{VERD['code_v2']}」である．"
  f"第 3 版は，台本の差分を `code_v3.md`（コミット f518545．台本の sha256 48f77ef0…）が見て「{VERD['code_v3']}」とした．"
  "`code_v3.md`（題）と `code_post.md`（2 節）は，台本の sha256 を「48f77ef0…0c86」と略して書いている．2 つの報告の本文にある sha256 の全体の末尾は 5c86 で，略した末尾の 4 桁と合わない（頭の 8 桁は合う）．"
  f"**統計の検査・定義の検査・感情認識の査読想定が見たのは第 1 版だけで，第 2 版で足した計算（疑似ラベルを替えた学び直し 5 つ，実験 1 の 3 手法に提案の手法の疑似ラベルを渡した版など）と，第 3 版で足した集計（疑似ラベルだけを替えた差 3 つ，判定が同じ版の印）を再検査していない．"
  "漏れの検査・比較の検査・ML の査読想定は第 2 版まで見て，第 3 版を見ていない．第 3 版を見たのは台本の検査だけで，見たのは台本の差分である**（`DAI/checks/DAI-21/` に，これより後の版の報告は無い）．"
  "設計の文の最後の直し（コミット 918ebea．`code_v3.md` の記録 3 件を設計の文に入れた．台本は f518545 から変わっていない）は，その後に検査を受けていない．"
  f"**起動の台本（`DAI/run/dai21_eval5.sh`）は，DAI-21 の検査の報告の範囲の外で（`code_v3.md`：「{KQ_RUN}」），実験 20 の計算の後に使う台本の検査（`DAI/checks/DAI-20/code_post_tools.md`・`code_post_tools_v2.md`）が，下書きの版（`DAI/docs/tools_pending/dai21_eval5.sh`）を検査した**．"
  f"第 1 回（`code_post_tools.md`．コミット 79fa614 の版．sha256 {RUN_SHA1[:8]}…．3 本の台本の全体への判定は「直してから進む」）は，3 節で，sha256 の照合・実験 20 が済んだことの確かめ・段の順序・本番の守り・失敗の印を「合う」とし，"
  f"起動の台本への指摘を記録の 2 件にした（「{T1_SNAP.lstrip('# ')}」「{T1_FRZ.lstrip('# ')}」）．その直しを入れた版（コミット 1f5b65e．sha256 {RUN_SHA[:8]}…）を第 2 回（`code_post_tools_v2.md`．判定は「通す」）が検査し，"
  f"前回の指摘との対応の表の「起動の台本の記録の指摘」の行に，直しを「入力の置き場が無ければ止まる．照らす入力を足した．freeze を先に済ませる前提を冒頭に書いた．起動のたびに前のログを消す」，結果を「合う」と書いている．**流した起動の台本の sha256 は {RUN_SHA[:8]}… で，第 2 回が検査した版と同じである**（コミット 07c5d67 で `DAI/run/` へ移した．この台本が，git の記録から作り直した sha256 と今のファイルの sha256 を比べて確かめた）．"
  "2 つの報告がコミットされたのは，計算の起動（03:20:35）より前である．"
  f"漏れの検査は「{KQ_MOVE}」と求めていた．流した台本の sha256（3 つの段の記録の `this_script_sha256`）は，`code_v3.md` が見た台本（コミット f518545・918ebea の `DAI/docs/tools_pending/dai21_eval5.py`）の sha256 と同じである（この台本が git の記録から作り直して確かめた）．"
  "実験 1 の評価の回の単位の保存 600 個の名前と sha256 をまとめた値は，`provenance_log.jsonl` の学び直しの段の行にだけある（`provenance.json` では空．設計 6 節）．")
CQ_BEFORE =quote(code_post, "台本（23:15），固定のファイル（03:10:03．中の `commit` は 86c38f1），起動（03:20:35）は，台帳の実験 20 の結果を開いた時刻（03:21）より前である．", "点検の報告")
w(f"- **計算の後の点検**：`DAI/checks/DAI-21/code_post.md`（判定は「通す」．記録 {N_REC} 件）．報告は UAR と差の値を書かず，数・一致の割合・ずれの大きさ・sha256・時刻だけを書いている．記録 {N_REC} 件の中身を，この台本が結果ファイルから数え直した値で書く．")
w("   1. **MOSEI の区間は，実験 20 の集計と実験 21 の集計で，同じ版でも少し違う**（乱数の鍵が違う）．点推定は同じである．この文書がどちらの集計から引くかは，上の「実験 20 の集計との使い分け」に書いた．")
LQ_GPU = quote(ROW21, "学び直しの段の GPU のメモリの最大は記録なし（CREMA-D の段は最大 1,962 MB）", "台帳の DAI-21 の行")
CQ_GPU = quote(code_post, "再検査の試走（`code_v2.md`）の見積もりは MOSEI で 2.9 GB，MELD で 4.4 GB である．", "点検の報告")
must(f"{GPU_CD_MAX:,.0f}" == "1,962", "CREMA-D の段の GPU のメモリの最大")
w(f"   2. **学び直しの段に，GPU のメモリの記録が無い**．学び直しの段の単位の csv の {len(HEAD4)} 列に，メモリの量の項目が無い．CREMA-D の段の単位の csv にはあり（`gpu_max_mb`），50 単位の最大は {GPU_CD_MAX:,.0f} MB である．"
  f"台帳は「{LQ_GPU}」と書いている．点検の報告によると，{CQ_GPU.rstrip('．')}（計算の前の試走の値で，今回の計算の値ではない）．")
w(f"   3. **表の文書の CREMA-D の UAR の表の「—」の欄**．表の文書には，表の上に「{TD_DASH}」の注がある．")
w("   4. **正解の数が同じで，判定が違う版が 6 つある**．集計の `same_as`（塊ごと・感情ごとの正解の数が全部の単位で同じ版の印）が付き，`same_pred`（判定そのものが全部の単位で同じ版の印）が付かない版は，"
  "MELD の K = 10 と MOSEI の K = 25 の温度つき蒸留（実験 1 の保存 `b_kd` と，この実験で解いた `kd_prod`），MELD の K = 25 の今の提案（実験 1 の保存 `joint_AVT` と，この実験で解いた `stud_prod`）である（この台本が判定の配列から印を作り直して確かめた）．"
  "この文書は，この 3 組を「同じ判定」と書かず，「正解の数が同じ」と書く．表の文書も，この 3 組の差の欄（解いた版 − 実験 1 の保存）に「（正解の数は同じ．判定は違う）」の注を付けている（2.1 節）．")
JR, KR = LT1["joint"], LT1["kd"]
w(f"- **検算の量**（`summary/convergence.json`．この台本が単位の csv から数え直して同じ値になることを確かめた．値の一覧は 3.1 節）：学び直しの段は，単位 {s4_['n_units']}，勾配の最大 {sci(s4_['grad_max'])}，止まり {s4_['stalled']}，"
  f"疑似ラベルの作り直しの一致の最小 {s4_['agree_min']:.6f}，今の提案の再現（積の疑似ラベル・主の点で学び直した判定が，実験 1 の保存の判定と一致する割合）の最小 {s4_['joint_repro_min']:.6f}（0.995 未満の（単位，K）は {BELOW['joint']}）．"
  f"この割合が 1 未満の（単位，K）は 600 組のうち {sum(JR.values())} 組（{'，'.join(f'{c} の K = {K} で {n}' for (c, K), n in sorted(JR.items(), key=lambda x: (C4.index(x[0][0]), x[0][1])))}）である．"
  f"温度つき蒸留の再現の最小は {s4_['kd_repro_min']:.6f}（1 未満は {sum(KR.values())} 組：{'，'.join(f'{c} の K = {K} で {n}' for (c, K), n in sorted(KR.items(), key=lambda x: (C4.index(x[0][0]), x[0][1])))}），"
  "一致した発話だけの疑似ラベルと，ラベルと疑似ラベルを混ぜる方法の再現は 600 組とも 1 である．AFEW と IEMOCAP では，4 つの再現の割合が全部の（単位，K）で 1 である．"
  f"CREMA-D の段は，単位 {cd_['n_units']}，勾配の最大 {sci(cd_['grad_max'])}，止まり {cd_['stalled']}，切片のずらしの残りの最大 {sci(cd_['adj_resid_max'])}，重みの表を探す版で切片の反復が上限に達した表の数 {cd_['wf_n_capped']} である．")
CQ_CI = quote(code_post, "水準 48 行と差 48 行（部分 × 種の組ごとに 8 行ずつ），感情ごとの再現率 12 行で，`lo`・`hi`・`se` のずれの最大は 7.1e-15，再現率は 1.4e-14 だった．", "点検の報告")
CQ_W = quote(code_post, "単位の `D_M_K_K*`・`D_M_orc_K*`・`D_M_ext` から 4 つの表を作り直し，ずれの最大 2.2e-16．", "点検の報告")
N_GRP = 24 * 12 + 53 * 3
w(f"- **この台本の照合**（どれも読むだけ．CPU）：(a) 集計の水準 {len(lv):,} 行と差 {len(ef):,} 行の点推定を，単位の保存の判定の配列から「単位ごとに UAR を作り，分割ごとに K 本の引き方の平均を取り，分割で平均する」形で作り直した．"
  "読んだのは，学び直しの 200 個と CREMA-D の 50 個の npz と，実験 20 の単位の 200 個の npz（元の 4 コーパスの推定時の答えと，実験 1 の保存の判定）である．"
  f"集計とのずれの最大は，水準 {sci(DEV_LV)}，差 {sci(DEV_EF)} である．版の名前と保存の配列の対応は，集計の台本を読み込まずに，設計 3 節の表と固定のファイルから書き直した．"
  f"(b) 判定が同じ版の印（`same_pred`）と，正解の数が同じ版の印（`same_as`）を判定の配列から作り直し，水準の表の {N_GRP} 組（コーパス，K，版）の群と，差の表の {len(ef):,} 行の印に一致した．"
  f"(c) 混同行列（{len(cfm) // 16} 組 × 16）と，感情ごとの全体の再現率（{len(pe):,} 行）を数え直し，集計と一致した（再現率のずれの最大 {sci(DEV_PE)}）．"
  f"(d) 表の印（↑・↓）を設計 4 節の文から作り直し，`effects.csv` の全行で印の列と一致した．"
  f"(e) 学び直しの設定の格子の表（`student_grid.csv` の {len(sg)} 行）と，CREMA-D の提案の格子の表（`grid_levels_cremad.csv` の {len(glc)} 行）を単位の csv から作り直し，一致した．"
  f"疑似ラベルの UAR（`teacher_uar`）と，学び直した分類器が疑似ラベルと判定が一致した割合（`mimic`）は，候補の上の判定と候補のラベルから数え直し，主の点の全部の単位で単位の csv の値と一致した（ずれの最大 {sci(DEV_TQ)}）．"
  "(f) 検算の量を単位の csv から数え直し，`convergence.json` と一致した．今の提案と実験 1 の 3 手法の再現の割合は，判定の配列からも数え直して単位の csv の値と一致した．"
  "(g) 表の文書は，今の集計から書き出し直した表と 1 字も違わない（2.1 節）．"
  f"区間（両端と標準誤差）と CREMA-D の重みの表は，この台本では作り直していない（点検の報告：「{CQ_CI}」「{CQ_W}」）．")
LQ_OPEN = quote(ROW21, "本体が結果を開いたのは点検の後の 2026-11-15 05:28（`DAI/docs/tools_pending/md_tables_dai21.py` で `DAI/docs/実験21_結果の表_2026-11-15.md` を書き出して読んだ）", "台帳の DAI-21 の行")
LQ_OPEN20 = quote(ROW21, "本体が実験 20 の結果を開いたのは起動の後の 03:21 である．それより後に設計・台本・固定のファイルは変えていない．", "台帳の DAI-21 の行")
DQ_VER = quote(design, "第 1 版から第 3 版は，どれも実験 20 の結果を開く前に書いた．実験 20 の結果を開いた時刻を台帳に残し，それより後にこの設計か台本を変えたら，変えた箇所を一覧にする．", "設計 0 節")
RUN_SH = (ROOT / "DAI" / "run" / "dai21_eval5.sh").read_text()
must('[ "$(sha256sum DAI/exp/dai21_eval5.py | cut -d\' \' -f1)" = "${SHA21:?}" ] || fail "dai21_eval5.py の sha256 が違う"' in RUN_SH, "起動の台本が台本の sha256 を照らす行")
must(T_HEAD["students4"] > OPEN20, "学び直しの段の始まりが，実験 20 の結果を開いた時刻より前（文を直す）")
w(f"- **結果を開いた時刻**：台帳の DAI-21 の行は「{LQ_OPEN}」と書いている．")
w(f"- **実験 20 の結果を開いた時刻との前後**：設計 0 節は「{DQ_VER}」と決めている．台帳の DAI-21 の行は「{LQ_OPEN20}」と書いている．"
  "最後のコミットは，" + "，".join(f"{k}（`{D21[k]}`）が {h}（{t[5:16]}）" for k, (h, t) in D21_LAST.items())
  + "で，どれも 2026-11-15 03:21 より前であり，作業ツリーにも変更が無い（この台本が git の記録で確かめた）．変えた箇所は無いので，一覧は無い．"
  f"点検の報告も「{CQ_BEFORE}」と書いている．**起動（03:20:35）の後，検算の 3 つの段が流れ，学び直しの段が始まったのは {T_HEAD['students4'][11:]} で，実験 20 の結果を開いた時刻（03:21）より後である**．"
  "起動の台本は，検査を通した台本の sha256 を受け取り，今の台本の sha256 と違えば止まる作りである．設計・台本・固定のファイルの sha256 は，3 つの段の記録で同じである．")
# ---- 計算の後の解析（CREMA-D の感度の表）：台帳の行，台本の検査の文
row_ph = [x for x in LROWS if x.startswith("| DAI-21 の計算の後の解析：CREMA-D の感度の表")]
must(len(row_ph) == 1, "台帳の，計算の後の解析の行")
ROW_PH = row_ph[0]
LQ_PH_SEEN = quote(ROW_PH, "この解析の出力の値は，台本の検査（gate-code）の前には本体は読んでいない（台本を書いた担当は，行の数・照合のずれ・除いた発話の数だけを報告した）", "台帳の計算の後の解析の行")
LQ_PH_PLAN = quote(ROW_PH, "表の中身は，実験 21 の設計 4 節末が計算の前に予告したものである（除く文，文を塊にした区間，内訳，仮定の表）．版と差の対は集計の段と同じ 78 対で，結果を見てから選んだ対は無い．"
                           "自由度の決め方（塊の数 12 から 11 とするか，実効の数 10.55 から 9 とするか）と，仮定の表に足した 2 列は，結果を開いた後に決めた点である．", "台帳の計算の後の解析の行")
LQ_PH_READ = quote(ROW_PH, "出力は 2026-11-15 07:50 に書いた（12 本）．本体が出力の値を読んだのは，第 2 回の検査が通った後の 07:59 である", "台帳の計算の後の解析の行")
DQ_LATER = quote(design, "**計算の後に保存から出す表**（設計に予告しておく）：CREMA-D で文「It's eleven o'clock.」を除いたテストの UAR と差（この文は Neutral の発話が少なく，K 本に入ると声の分類器の手がかりになりうる），"
                         "文を塊にした区間（12 塊）の感度，CREMA-D の文ごと・強さごとの内訳，仮定の表．", "設計 4 節")
PT2_J = quote(PT2, "通す．前の報告の指摘 1〜7 はすべて直っており，直しの前の出力の値は 1 つも変わっていない．", "code_post_tools_v2.md")
PT2_BOOT = quote(PT2, "点推定・標準誤差・両端のずれの最大は 2.6e-14 だった．", "code_post_tools_v2.md")
PT2_ASM = quote(PT2, "私の書き方（Tukey の当てはめを自分で書いた）で 13 列を作り直して一致．", "code_post_tools_v2.md")
G1 = quote(PT2, "「除いた値 − 全発話の値」の区間（`d_lo`・`d_hi`）は同じ組の上の対応のある区間で，「除いても変わらない／変わる」はこの区間で書く．2 つの区間が重なるかどうかで書かない．", "code_post_tools_v2.md")
G1B = quote(PT2, "この表が測るのは「水準と差が，テストの文 `IEO` の 910 本（Neutral 91，ほかの感情 273 ずつ）に頼っているか」である．K 本に文 `IEO` が入ることの影響（学習側）は測っていない．", "code_post_tools_v2.md")
G2 = quote(PT2, "話者と学習した分類器を固定し，テストの 12 文を同じ文の集まりから取り直したときの揺れと，K 本の引き方の揺れを含む．テストの話者の揺れは含まない．学習側にも同じ 12 文があるので，学習に現れない文への区間ではない", "code_post_tools_v2.md")
G2B = quote(PT2, "話者の区間と文の区間を足し合わせた区間は出していない．", "code_post_tools_v2.md")
G3 = quote(PT2, "文 `IEO` の Happy・Sad・Angry には強さ LO・MD・HI があり，ほかの文には XX しかない．文の違いと強さの違いは分けられない．", "code_post_tools_v2.md")
G4 = quote(PT2, "しかし引き方は本数以外の点でも違うので，並べた結果は記述に留め，因果として書かない．K 本から文 `IEO` を除いて学び直す試しは範囲の外で，設計の懸念はこの表で「解けた」とは書けない．", "code_post_tools_v2.md")
G5 = quote(PT2, "分類器が 2 つなので，`topok` はでたらめでも 0.5，`intcorr`・`intperm` は自由に動く 2 個の数の上の量である．3 分類器の 4 コーパスの値と同じ尺度で並べない．", "code_post_tools_v2.md")
G5B = quote(PT2, "`emocorr_M_*`・`profcorr_row_*`：4 個の数の相関．`profcorr_M_*` は 8 個の数で，各行の和が 0 なので自由な値は 6 個である．", "code_post_tools_v2.md")
G5C = quote(PT2, "仮定の表の `ret_A`・`ret_V` は残り方であり，内訳の版の名前とは別のものである．", "code_post_tools_v2.md")
PH_REPORTS = sorted(x.name for x in CK.glob("*.md") if "dai21_posthoc_cremad_sentence" in x.read_text())
must(PH_REPORTS == ["claim_v3.md", "code_post_tools.md", "code_post_tools_v2.md"], f"計算の後の解析を見た報告（文を直す）：{PH_REPORTS}")
must(PH_COMMITS[0][1] < T_PH and T_PH.startswith("2026-11-15 07:50") and PH_COMMITS[0][1] > "2026-11-15 05:28:00" and phv["consts"]["N_BOOT"] == 2000, "計算の後の解析の時刻")
w(f"- **計算の後の解析（CREMA-D の感度の表．第 3 版で足し，第 4 版で直した）**：設計 4 節は，計算の後に保存から出す表を予告していた（「{DQ_LATER}」）．"
  f"この 4 種類の表を，CREMA-D の 50 単位の保存（npz の判定と，ソースで測った表．単位のコミットは {C_CD[:7]}）から，学び直さずに数え直して出した．"
  f"台本は `DAI/docs/tools_pending/dai21_posthoc_cremad_sentence.py`（sha256 {phv['script_sha256'][:8]}…．コミット {PH_COMMITS[0][0]}，{PH_COMMITS[0][1][5:]}．今のファイルと同じ），"
  f"出力は `results/dai21_eval5/posthoc/` の 12 本（csv 11 本と，由来の記録 `cremad_sentence_provenance.json`．記録の時刻は {T_PH}，コミットは {phv['commit'][:7]}，dirty は偽，所要 {phv['seconds']:.0f} 秒），"
  "ログは `logs/dai21_posthoc_cremad_sentence.log`（最後は `EXIT=0`）である．"
  f"由来の記録は，台本，読み込んだ部品 {len(phv['modules_sha256'])} 本，固定のファイル 2 本，設計，集計の {len(phv['summary_sha256'])} ファイル，実験 21 の `provenance.json` の sha256 と，出力の csv {len(phv['outputs'])} 本の行の数を持つ．"
  "この台本が，今のファイルから作り直した sha256 と行の数が，記録と同じであることを確かめた．")
w(f"   - **台本の検査**（関門 1 の台本の検査役．`DAI/checks/DAI-21/`）：第 1 回（`code_post_tools.md`．コミット {PH_COMMITS[1][0]} の版）の判定は「直してから進む」（指摘 {N_PT1} 件）で，"
  f"直した版（コミット {PH_COMMITS[0][0]}）への第 2 回（`code_post_tools_v2.md`）の判定は「{PT2_J}」だった．第 2 回の新しい指摘は {N_PT2} 件で，どちらも重さは「記録する」である．"
  "第 2 回の報告は，結果の文に値を入れるときの注意を 5 項書いている．この文書は，その 5 項を要点 13〜18 に引いた．"
  "この解析を見た報告は，この 2 本と，照合の第 3 回（`claim_v3.md`．次の項）の 3 本である（`DAI/checks/DAI-21/` の報告のうち，計算の後の解析の台本の名前が出てくるもの）．統計・漏れ・比較・定義の検査と査読想定の役は，この解析を見ていない．")
w(f"   - **照合の第 3 回**（関門 2．`claim_v3.md`．第 3 版を見た）：判定は「直してから通す」で，報告は「{C3_OK}{C3_BOOT}」と書いている．"
  f"指摘は {N_CLAIM3} 件で，どれも重さは「記録する」である：設計の懸念に直接当たる量（文 `IEO` の Neutral の再現率）を引いていないこと，引き直しの乱数に依る区間，要点 17 の分け方，要点 18 の 1 文，照合の前の使用の申告．"
  "第 4 版は，この 5 件を直した．検算の台本と出力は `DAI/checks/DAI-21/claim_run/`（`check_v3_*`）にある．この台本は，報告の中の値を文の値に使っていない（引いた文は，引用として書いた）．")
w(f"   - **照合の第 4 回**（関門 2．`claim_v4.md`．第 4 版を見た）：判定は「通す」で，報告は「{C4_OK}{C4_MATCH}」と書いている．"
  f"残る点は {N_CLAIM4} 件で，どちらも重さは「記録する」である：要点 16 で，上限の声の分類器でも同じ向きであることを「懸念が予想する向き」の文と並べて書くこと，0 節の (5) に訂正の会話を書くこと．"
  "この文書は，この 2 件を入れた．検算の台本と出力は `DAI/checks/DAI-21/claim_run/`（`check_v4_*`）にある．")
w(f"   - **台帳の記録**（`docs/seen-data.md` の「DAI-21 の計算の後の解析」の行．本体の申告）：「{LQ_PH_SEEN}」「{LQ_PH_PLAN}」「{LQ_PH_READ}」"
  f"値を読んだ時刻は本体の申告で，この台本からは確かめられない．由来の記録の時刻（{T_PH[11:]}）が，直した版のコミット（{PH_COMMITS[0][1][11:]}）より後であることは，この台本が確かめた．")
w("   - **設計が計算の前に予告したのは，表の種類である．次の点は，結果を開いた後（2026-11-15 05:28 より後）に決まった**：文を塊にした区間の自由度（9），文ごと・強さごとの内訳に入れる 16 版，"
  "K 本の中の文の本数と単位ごとの値の表，「除いた値 − 全発話の値」の区間，仮定の表に入れる量（どれも，計算の後の解析の台本と，その検査の指摘で決まった）．"
  "この文書が決めた並べ方（2.7 節の差の表で，要点 1〜10 が読みを書いた差を先に並べる順序）も，結果を開いた後に決めた．"
  "第 4 版で足した，文 `IEO` とほかの 11 文の感情ごとの再現率の表（要点 16，2.7 節）と，区間の端が 0 に近いと数える基準（要点 14）は，照合の第 3 回の指摘を受けて，結果を開いた後に決めた．"
  "第 3 版にあった，K 本の引き方を文の本数で 2 つに分けた並びは，第 4 版で外した（6 節）．")
w(f"   - **この台本の照合（感度の表）**：出力の csv の値は，単位の npz の判定から，計算の後の解析の台本を読み込まずに，別の書き方で作り直した値と照らした．"
  f"(h) 文を除いたテストの点推定：水準 {len(nl)} 行と差 {len(ne)} 行（ずれの最大 {sci(max(DEV_NL, DEV_NE))}），感情ごとの再現率 {len(npe)} 行（{sci(DEV_NPE)}），除いた発話の数．"
  f"(i) 文を塊にした区間：差 {len(sce)} 行と水準 {len(scl)} 行の点推定・標準誤差・自由度 9 と 11 の両端を，「文を 1 つずつ全部の分割から除く jackknife の分散 ＋ K 本の引き方の項」で作り直した（ずれの最大 {sci(max(DEV_JK, DEV_JKL))}）．"
  f"文の実効の数（{G_EFF_SENT:.3f}）と自由度（{DF_SENT}）も作り直した．"
  f"(j) 文ごとの内訳 {len(bys)} 行（4 感情の平均と，感情ごとの再現率の 4 列）と，強さごとの内訳 {len(byi)} 行（{sci(max(DEV_BS, DEV_BI))}）．(k) K 本の中の文の本数と，単位ごとの UAR と差（{len(ksh)} 行．{sci(DEV_KS)}）．"
  f"(l) 仮定の表の {len(phv['assumptions']['from_npz'])} 列のうち {len(ASM_COLS)} 列（水準，残り方，最も強いモダリティの一致，共通の感情の成分の相関，行の中の差の相関，行ごとの相関，上限の割合，0 以下の割合．{sci(DEV_ASM)}）．"
  f"(m) 出力の全発話の列（`*_all`・`*_speaker`）が集計の値と同じこと（ずれの最大 {sci(max(DEV_PH_ALL, DEV_PH_SPK, DEV_PH_LV))}），「除いた値 − 全発話の値」の点推定の列が 2 つの列の引き算であること，印の列．"
  f"**作り直していないもの**：文を除いたテストの区間と，「除いた値 − 全発話の値」の区間（話者の重みの引き直し {phv['consts']['N_BOOT']:,} 回で作る．乱数を使うので，この台本では作り直していない．"
  f"第 2 回の検査は，別の書き方で 78 個を作り直した結果を「{PT2_BOOT}」と書いている．照合の第 3 回は，別の乱数で区間を作り直し，「{C3_SE}」と書いている．乱数に依りうる区間は要点 14・15 に書いた），"
  f"仮定の表の残りの {len(phv['assumptions']['from_npz']) - len(ASM_COLS)} 列（成分の相関，入れ替えの割合，テストとのずれ．第 2 回の検査は，仮定の表について「{PT2_ASM}」と書いている），"
  "文ごとの取り違えの数（`cremad_by_sentence_confusion.csv`．この文書は引いていない）．")
# 論文のセッションへの伝言：結果を開いた後のコミットで，実験 21 の値を「照合中」の印つきで書いた（照合の前に値を使った先）
NOTE_P = "DAI/docs/論文のセッションへの伝言_2026-11-14.md"
NOTE_NOW = (ROOT / NOTE_P).read_text()
NOTE_AFTER = [x.split(" ", 1) for x in git("log", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M:%S", "--", NOTE_P).strip().split("\n") if x.split(" ", 1)[1] > "2026-11-15 05:28:00"][::-1]
must([h_ for h_, _ in NOTE_AFTER] == ["c6d5e2e", "e97b618", "8772450", "45a7501", "6687708"], f"伝言の，結果を開いた後のコミット（文を直す）：{NOTE_AFTER}")
NOTE_C3 = git("show", f"8772450:{NOTE_P}")                                  # 感度の表を出す前の版
NOTE_C4 = git("show", f"45a7501:{NOTE_P}")                                  # 感度の表を出した後，照合の第 3 回・第 4 回のあいだの版
NQ10 = quote(NOTE_C4, "感度の表は 11-15 の 07:50 に出した（`results/dai21_eval5/posthoc/`．台本の検査は `DAI/checks/DAI-21/code_post_tools_v2.md` で通過）．値と読み方は，実験 21 の結果の文の第 3 版に入れて照合に回す．"
                       "**その照合が通るまで，CREMA-D の値を本文に入れない**（通ったら，この項を書き換える）．感度の表は，文をテストから除いたときの値と，区間の塊を文に替えた値である．"
                       "文が K 本に入ることの影響（設計の懸念）は測っていないので，「懸念が解けた」とは書かない．", "論文のセッションへの伝言")
NOTE_L10 = [x for x in NOTE_C4.split("\n") if NQ10 in x]
NQ10B = quote(NOTE_NOW, "**CREMA-D の値を論文や資料に入れるときは，感度の表と一緒に書く**", "論文のセッションへの伝言")
NQ10C = quote(NOTE_NOW, "CREMA-D の値を本文に入れてよい．そのとき次の 4 点を落とさない．", "論文のセッションへの伝言")
NOTE_L10B = [x for x in NOTE_NOW.split("\n") if NQ10B in x]
must(NQ10B not in NOTE_C4 and NQ10 not in NOTE_NOW and git("diff", "--numstat", "45a7501", "6687708", "--", NOTE_P).split()[:2] == ["1", "1"] and len(NOTE_L10B) == 1 and NQ10C in NOTE_L10B[0]
     and re.search(r"\d\.\d\d", NOTE_L10B[0]) is None and all(f"({x_})" in NOTE_L10B[0] for x_ in "abcd") and "(e)" not in NOTE_L10B[0], "伝言の，照合の第 4 回の後の差分（文を直す）")
must(NQ10 not in NOTE_C3 and git("diff", "--numstat", "8772450", "45a7501", "--", NOTE_P).split()[:2] == ["1", "1"] and len(NOTE_L10) == 1 and re.search(r"\d\.\d\d", NOTE_L10[0]) is None,
     "伝言の，感度の表を出した後の差分（文を直す）")
NOTE_C2 = git("show", f"e97b618:{NOTE_P}")                                  # 照合の第 2 回が見た版
NOTE_ADDED = ["照合済み（`DAI/checks/DAI-21/claim_v2.md`．11-15 朝）", "**どの文も，どの差についての文かを添えて写す**", "上の 2 つは「別のまとめ方との差」と「別の疑似ラベルとの差」に限った一覧である．",
              "**CREMA-D の値を論文や資料に入れる前に，設計が予告した感度の表を出す**"]
must(all(x in NOTE_C3 and x not in NOTE_C2 for x in NOTE_ADDED) and all(x in NOTE_NOW for x in NOTE_ADDED[:3]), "伝言の，照合の第 2 回の後に足された文（文を直す）")
NOTE_C1 = git("show", f"c6d5e2e:{NOTE_P}")
NOTE_VALS = [ci(S_PP(c, K), star=False) for c, K in ALL15] + [ci(q("prop", "prod", CD, K), star=False) for K in KS]      # 伝言の文字と比べるので，＊ は付けない
must("照合中" in NOTE_C1 and all(x in NOTE_C1 for x in NOTE_VALS) and all(x in NOTE_C2 for x in NOTE_VALS) and all(x in NOTE_C3 for x in NOTE_VALS) and all(x in NOTE_C4 for x in NOTE_VALS) and all(x in NOTE_NOW for x in NOTE_VALS), "伝言に書いてある実験 21 の値が，この文書の値と違う")
NOTE_LOW = [fest(q("stud_prop", f"stud_{t}", c, K).est) for t in EXTRA for c, K in ALL15 if klass(q("stud_prop", f"stud_{t}", c, K)) in ("nn", "n")]
NOTE_LOW_OK = all(x in NOTE_NOW for x in NOTE_LOW)                           # 今の伝言が，低かった条件の全部の点推定を持つか
must(len(NOTE_LOW) == 6, "区間が 0 を含まずに低かった条件の数")
# この結果の値の文字が出てくる文書を探す（AFEW の K = 10 の値と区間）
PROBE = ci(S_PP("AFEW", 10), star=False)
SCAN = sorted({str(x.relative_to(ROOT)) for d_ in ("DAI/docs", "DAI/paper", "DAI/slides", "docs", "slides", "paper") if (ROOT / d_).exists()
               for x in (ROOT / d_).rglob("*") if x.suffix in (".md", ".tex") and x.is_file() and x.resolve() != OUT.resolve() and PROBE in x.read_text(errors="ignore")})
must({"DAI/docs/実験21_結果の表_2026-11-15.md", NOTE_P} <= set(SCAN), f"値の文字が出てくる文書：{SCAN}")
r3_ = NE[("prop", "prod", 5, "all")]
PROBE3 = (f"{fest(r3_.est)} [{r3_.lo:+.2f}, {r3_.hi:+.2f}]").replace("-", "−")                # 文を除いたテストの prop − prod の K = 5（要点 13 と同じ書き方）
SCAN3 = sorted({str(x.relative_to(ROOT)) for d_ in ("DAI/docs", "DAI/paper", "DAI/slides", "docs", "slides", "paper") if (ROOT / d_).exists()
                for x in (ROOT / d_).rglob("*") if x.suffix in (".md", ".tex") and x.is_file() and x.resolve() != OUT.resolve() and PROBE3 in x.read_text(errors="ignore")})
MEMO = Path("/home/snics/.claude/projects/-home-snics-MER-DA/memory/dai21-retrain-cremad-2026-11-14.md")
MEMO_OK = MEMO.exists() and "2026-11-15 05:40 の追記" in MEMO.read_text()
w(f"- **照合の前に値を使った先**：照合（関門 2）の前に，この結果の値は次の場所で使われている．(1) 表の文書 `DAI/docs/実験21_結果の表_2026-11-15.md`（書き出しの台本 `md_tables_dai21.py`．最初の書き出しはコミット {TD_FIRST[0]}，{TD_FIRST[1][5:16]}．"
  f"今の版はコミット {TD_LAST[0]}，{TD_LAST[1][5:16]} で，sha256 {sha(TABLE_DOC)[:8]}…．値は最初の版から変わっていない．2.1 節）．"
  "(2) 本体が 2026-11-15 05:35 ごろに，研究者へ会話で主な値を報告した（学び直した融合の差の 5 コーパス × 3 つの K の表，K = 5 の UAR の表，学び直した融合 − 推定時の答えの範囲，CREMA-D の推定時の答え，提案の手法が低かった条件，"
  "温度つき蒸留に同じ疑似ラベルを渡した差，今の提案の再現の最小）．(3) 本体の作業の覚え書き（memory の `dai21-retrain-cremad-2026-11-14.md`）の「2026-11-15 05:40 の追記」に，要点の値が書いてある．"
  "覚え書きの文のうち，区間の読みがこの文書の値と合わなかった 2 つ（温度つき蒸留に同じ疑似ラベルを渡した差の MOSEI の K = 25 と，学び直した融合 − 推定時の答えの CREMA-D の K = 10 は，区間が 0 を含まない）は，本体が直した．"
  "(2)・(3) は本体のセッションの申告で，この台本からは確かめられない"
  + ("（覚え書きに「2026-11-15 05:40 の追記」の句があることだけを見た）．" if MEMO_OK else "．")
  + f"(4) 論文のセッションへの伝言（`{NOTE_P}`）の 0 節．コミット c6d5e2e（{NOTE_AFTER[0][1][5:16]}）で，実験 21 の値（学び直した融合の「提案の手法の疑似ラベル − 積の疑似ラベル」の 15 個の値と区間，CREMA-D の推定時の答えの「提案の手法 − 積」の 3 個の値と区間）を，「照合中」の印を付けて書いた．"
  + f"コミット e97b618（{NOTE_AFTER[1][1][5:16]}）で，本体が，提案の手法が低かった条件を全部挙げる形に直した．"
  + f"**照合の第 2 回の後のコミット 8772450（{NOTE_AFTER[2][1][5:16]}）で，本体が伝言をもう一度直した**．直した中身は 4 つである：0 節の状態の表の実験 21 の行を「照合済み」にした，「落とさないこと」の各項にどの差についての文かを書いた，"
  + "低かった条件の場所の案内を足した，CREMA-D の値を論文や資料に入れる前に感度の表を出すという項を足した（この台本は，この 4 つの文が 8772450 の版の伝言にあり，e97b618 の版に無いことを確かめた）．"
  + "18 個の値と区間の書き方は変わっていない．この台本は，c6d5e2e・e97b618・8772450・45a7501 の版と今の版の伝言にあるこの 18 個の値と区間が，この文書の値と同じ文字であることを確かめた（この文書が区間の後に付けた ＊ の印は，除いて比べた）．"
  + f"**感度の表を出した後のコミット 45a7501（{NOTE_AFTER[3][1][5:16]}）で，本体が伝言の 0 節の 10 を書き換えた**．変わったのは 1 行で，足された文は「{NQ10}」だった．その行に，感度の表の値（小数 2 桁の数）は無かった．"
  + f"**照合の第 4 回が通った後のコミット 6687708（{NOTE_AFTER[4][1][5:16]}）で，本体が伝言の 0 節の 10 をもう一度書き換えた**．変わったのは 1 行で，今の項は「{NQ10B}」で始まり，「{NQ10C}」と書いて，落とさない 4 点（(a)〜(d)）を並べている．"
  + "4 点は，文を除いても主の 2 つの差の変化の区間が 0 を含むこと，この表が測ったのはテストの側だけであること，声の分類器でのこの文の Neutral の再現率の並びが懸念の予想する向きで原因は言えないこと，文を塊にした区間の読み方である．"
  + "その行に，小数 2 桁の数は無い．18 個の値と区間の書き方は，この版でも変わっていない．"
  + ("今の版の伝言は，区間が 0 を含まずに提案の手法の疑似ラベルが低かった 6 条件（要点 5）の点推定を全部持つ．" if NOTE_LOW_OK else "今の版の伝言には，区間が 0 を含まずに提案の手法の疑似ラベルが低かった 6 条件（要点 5）の点推定のうち，無いものがある．")
  + f"**ほかの文書**：この台本が，AFEW の K = 10 の値と区間の文字（{PROBE}）を，`DAI/docs`・`DAI/paper`・`DAI/slides`・`docs`・`slides`・`paper` の .md と .tex のファイルから探すと，この文書のほかに見つかるのは "
  + "，".join(f"`{x}`" for x in SCAN) + " である．値を別の桁や別の形で写した文書は，この探し方では見つからない．検査の報告（`DAI/checks/`）は探していない．"
  + "(5) **感度の表の値**（第 3 版で足した値）：台帳によると，本体が出力の値を読んだのは 07:59 である（上の「計算の後の解析」）．"
  + "伝言には，感度の表の値が書かれていない（感度の表を出した後に変わったのは上の 1 行だけで，その前の版は表を出す前の版である）．"
  + "**本体の申告**（照合の第 3 回の指摘 5 を受けて書く．会話の中身は，この台本からは確かめられない）：本体は 2026-11-15 の 08:00 ごろ，研究者へ会話で感度の表の値を伝えた．"
  + "伝えたのは，提案の手法 − 積，学び直した融合の提案の手法の疑似ラベル − 積の疑似ラベル，学び直した融合 − K 本だけの融合の 3 つについての，全発話の値・文を除いた値・変化とその区間，"
  + "提案の手法の中の学び直した融合 − 推定時の答えの K = 10 の 2 つの区間，文を塊にした区間の 1 例である．"
  + "08:40 ごろには，第 3 版の数え（231 個のうち 4 個，12 個）と，`start_A` の文 `IEO` の UAR が 12 文の中で最も低いこと（「この文が当てにくい並びは K 本を使う前からある」という読み）を伝えた．"
  + "**この読みは，照合の第 3 回の指摘 1 で，範囲が広すぎるとされた（要点 16 を第 4 版で直した）．本体は，2026-11-15 の 08:50 ごろと 09:10 ごろ，研究者へ会話で訂正を伝えた**．"
  + "伝えたのは，文 `IEO` の Neutral の再現率の並びは設計の懸念が予想する向きで，前の読みは範囲が広すぎたこと，声の分類器・提案の手法・積の Neutral の再現率の値，顔の分類器では向きが逆であること，端が 0 に近い区間の数である．"
  + "本体は，ほかの文書には感度の表の値を書いていない．08:00 ごろと 08:40 ごろの会話は，照合（関門 2）が通る前の使用である．"
  + f"訂正の 2 回の会話について，照合の第 4 回の報告は「{C4_USE}」と書いている（照合が通ったことを記録したコミット 6687708 の時刻は {NOTE_AFTER[4][1][11:16]} で，09:10 ごろの会話は，その後である）．"
  + f"この台本が，文を除いたテストの `prop − prod` の K = 5 の値と区間の文字（{PROBE3}）を，上と同じ場所の .md と .tex のファイルから探すと，この文書のほかに"
  + ("は見つからない" if not SCAN3 else "見つかるのは " + "，".join(f"`{x}`" for x in SCAN3) + " である") + "（検査の報告 `DAI/checks/` は探していない）．")
w("")
w("**申告**（設計 0・3・4・8・9 節が求めるもの．冒頭の 6 項に加えて）：")
w("")
DQ_ADD = quote(design, "足した計算：疑似ラベルを替えた学び直し 5 つ，実験 1 の 3 手法に提案の疑似ラベルを渡した版と再現の検算，CREMA-D の出発点と上限の判定，学び直した分類器の重みと候補の上の判定の保存．", "設計 11 節")
DQ_R1 = quote(design, "20:55「CREMA-Dも追加したいです」→ 本体の案（CREMA-D を評価するコーパスにだけ足す．ソースは今の 4 コーパス，声と顔．元の 4 コーパスの計算とソースは変えない）に「その案でよい」．22 時ごろ「CREMA-Dも本番の形で測って．」", "設計 0 節")
DQ_R2 = quote(design, "本体が「いまの評価の回（実験 20）には学び直しが入っていない．学び直しを 5 コーパスぶん足す」と推し，「よい．足して．」", "設計 0 節")
DQ_R3 = quote(design, "「重みを保存しよう．」（学び直した分類器の重み．前半の解析の道具を学び直した分類器にも当てるため）", "設計 0 節")
w(f"- **研究者の決定と指示，本体が決めたこと**（設計 0 節）：研究者の決定と指示は次の 3 つである．「{DQ_R1}」「{DQ_R2}」「{DQ_R3}」"
  f"学び直しを足すことは，本体が推して研究者が承認した．検査の指摘で第 2 版に足した計算は，設計 11 節に「{DQ_ADD}」とある（重みの保存は研究者の指示）．疑似ラベルを替えた学び直し 5 つの中身と，実験 1 の 3 手法に提案の手法の疑似ラベルを渡した版は，本体が検査の指摘を受けて決めた．")
DQ_MAIN = quote(design, "この決め方は提案の側に不利な向きに働く（12 点から選んだ分の得が，積の側にだけ付く）．一方，学び直しを足すと決めたのは同じテストの上の値を見た後で，こちらは提案に有利な向きである．2 つを合わせた向きは決まらない．", "設計 3.1 節")
DQ_MAIN0 = quote(design, "**主の点**：積の疑似ラベルのために選ばれた点を，どの疑似ラベルにも使う．同じ設定で疑似ラベルだけを替えるので，差は疑似ラベルの違いだけになる．", "設計 3.1 節")
w(f"- **学び直しの設定（主の点）と，偏りの向き**（設計 3.1 節）：「{DQ_MAIN0}」主の点は，{DQ_MAIN4.rstrip('．')}，{DQ_MAINCD.rstrip('．')}．設計は偏りの向きを「{DQ_MAIN}」と書いている．")
DQ_Q2B = quote(design, "3 手法の設定は，実験 1 がその手法に積の疑似ラベルを渡して選んだ点である．選んだ分の上乗せは積の側にだけ付くので，疑似ラベルの違いの分は提案の疑似ラベルに不利な向きに偏る．共訓練と共正則化は積を受け取らない方法なので，この内訳は作れない．", "設計 4 節")
w(f"- **実験 1 の 3 手法に提案の手法の疑似ラベルを渡した版の設定**（設計 4 節）：「{DQ_Q2B}」")
DQ_SEL = quote(design, "**`stud_prop_sel` の点の選択には，評価するコーパスのテストのラベルが間接に入る**（実験 18 が学び直しの点を選んだ単位では，提案の疑似ラベルの格子の点が，評価するコーパスを仮のターゲットに含む単位で選ばれている）．"
                       "**`stud_prod_sel` の点の選択には入らない**（積の疑似ラベルには選ぶ格子の点が無い．CREMA-D では主の点と同じ選び方である）．差の片側にだけこの経路があるので，この差は記述だけに使い，主張に使わない．主の点にはこの経路が無い．", "設計 3.1 節")
w(f"- **設定を疑似ラベルごとに選んだ点（記述の版）**（設計 3.1 節）：「{DQ_SEL}」")
w("- **選ぶデータと候補の数**（設計 4 節の表のまま）：")
w("")
w("   | 行 | 設定を選んだデータ | 候補の数 |")
w("   |---|---|---|")
for s_ in ("| 学び直し（提案）の主の点 | 借りた（積の疑似ラベルのために選ばれた点） | 選んでいない |",
           "| 今の提案，実験 1 の 5 手法 | 実験 1 の仮のターゲットの回（自分のための点） | 12，12・9・9・4・5 |",
           "| CREMA-D の主の点，提案の点，引き戻しの強さ | 実験 18 の，CREMA-D を触らずに残した単位 | 12，16，14 |",
           "| 疑似ラベルを替えた学び直し 5 つ | 借りた（主の点） | 選んでいない |",
           "| 実験 1 の 3 手法に提案の疑似ラベルを渡した版（`kd_prop`・`agree_prop`・`mix_prop`） | 借りた（実験 1 がその手法に積の疑似ラベルを渡して選んだ点．提案の疑似ラベルに不利な向き） | 選んでいない |",
           "| `stud_prop_sel`・`stud_prod_sel` | 実験 18 の規則で疑似ラベルごとに選んだ点．`stud_prop_sel` にだけ 3.1 節の間接の経路がある（提案に有利な向き） | 12 |",
           "| 提案の疑似ラベル | ソースで測った表と，K 本の 5 組を 1 組ずつ抜いて作り直した分類器を使う（相手が受け取らない計算と資産） | — |"):
    w("   " + quote(design, s_, "設計 4 節の表"))
w("")
w("- **K は入れ子で，テストは同じである**．K = 10 の K 本（40 本）は K = 5 の K 本（20 本）を含み，K = 25 の K 本（100 本）は K = 10 の K 本を含む（台本が，元の 4 コーパスは実験 20 の単位の npz，CREMA-D はこの実験の単位の npz の `kpos_K*` で確かめた）．"
  "テストの発話・分割・K 本の引き方は 3 つの K で同じである．**K どうしの差の区間は作っていない**．")
DQ_CI = quote(design, "区間は実験 20 と同じ（`dai1_stats.effect_stats`．95%．補正しない）．水準と感情ごとの再現率にも同じ規則の区間を付ける．", "設計 4 節")
DQ_CNT = quote(design, "CREMA-D の区間の半幅は約 0.3 点の見込みなので，0.3〜1.0 点の差は，区間が 0 を含まなくても印が付かない．印の数を数えて主張に使わない．"
                       "コーパスをまたいだ数え上げを検定の結果として書かない（5 コーパスの符号検定の p 値の最小は 0.0625 で，コーパスは独立でもない）．", "設計 4 節")
w(f"- **区間は 95% で，補正していない**（設計 4 節：「{DQ_CI}」）．差の区間の数について，設計 4 節は「{DQ_NINT}」と書いている．`effects.csv` は {N_INT['ef']:,} 行で，うち K 本の引き方 0〜9 の行は {N_INT['ef_all']} 行である．"
  f"このほかに，水準の区間が {N_INT['lv']:,} 個（`levels.csv`），感情ごとの再現率の区間が {N_INT['pe']:,} 個（`per_emotion.csv`）ある．どれも補正していない．差が全部 0 でも偶然の印は付きうる．設計 4 節：「{DQ_CNT}」"
  f"統計の検査（`stats.md`）は「{RQ['S5a']}」「{RQ['S5b']}」「{RQ['S5c']}」と書いている．")
DQ20_JK1 = quote(design20, "IEMOCAP の区間は 5 セッションの jackknife（自由度 3）である．MELD の分割 4 は塊が 2 つしか無い．", "実験 20 の設計 5 節")
DQ20_JK2 = quote(design20, "jackknife のコーパス（AFEW・IEMOCAP・MELD）では，外れた分割の種の揺れが `v_cluster` の側に入る（IEMOCAP は全部の分割が外れるので `v_seed` は 0 になる）．", "実験 20 の設計 5 節")
CP20_AFEW = quote((ROOT / "DAI" / "checks" / "DAI-20" / "code_post.md").read_text(), "AFEW では，ある塊を除くと分割 0 のテストが無効になり，その分割の種の揺れが `v_cluster` の側に入る．", "実験 20 の点検の報告")
DQ_NOISE = quote(design, "**区間が含まない揺れ**：ソースの取り方，学習側の発話，手法と設定の選び方，重みの表の引き方，CREMA-D の文（12 文を全話者が読むので，話者と文が交差する．区間は「同じ 12 文を読む新しい話者」への区間である）．"
                         "CREMA-D の種 0・1 の区間は，種の項の自由度が小さいので狭めに出る．", "設計 4 節")
w(f"- {DQ_NOISE}（設計 4 節の文のまま）")
w(f"- **区間の作り方**（`effects.csv` の `method`・`df`・`folds_dropped`）：AFEW・IEMOCAP・MELD は，塊を全分割から同時に除く jackknife と K 本の引き方の項（自由度は順に {DF['AFEW'][0]}・{DF['IEMOCAP'][0]}・{DF['MELD'][0]}），"
  "MOSEI と CREMA-D は，塊の重みの引き直しと K 本の引き方の項（自由度を持たない）である．CREMA-D の塊は話者（91 人）である（設計 2 節．台本が単位の npz で，塊の名前が発話の名前の話者の番号と同じことを確かめた）．"
  f"jackknife で除かれる分割は，AFEW の全行で「{FD['AFEW'][0]}」，IEMOCAP の全行で「{FD['IEMOCAP'][0]}」である．IEMOCAP は全部の分割が除かれ，K 本の引き方の項（`v_seed`）は全行で 0 である．IEMOCAP は自由度 3 で区間が広い．"
  f"統計の検査（`stats.md`）は，CREMA-D の区間について「{RQ['S3']}」と書いている（「この 2 つ」は，K 本の引き方の項の自由度が小さいことと，この縮みである）．K 本の引き方の項の割合は 3.2 節に並べた．"
  f"統計の検査は「{RQ['S3b']}」ことも求めていた．実験 20 の設計 5 節：「{DQ20_JK1}」「{DQ20_JK2}」実験 20 の計算の後の点検（`DAI/checks/DAI-20/code_post.md`）：「{CP20_AFEW}」"
  "この実験の集計でも，IEMOCAP は全部の分割が除かれて `v_seed` が全行で 0 で，AFEW は全行で分割 0 が除かれる（上の値）．**IEMOCAP の `v_seed` が 0 であることは，区間が K 本の引き方の揺れを含まないことを意味しない．"
  "実験 20 の設計の文のとおり，除かれた分割の K 本の引き方の揺れは塊の項（`v_cluster`）に入る**．"
  f"MELD のテストの塊の数は，分割 0〜4 の順に {'・'.join(map(str, MELD_CL))} 個で，分割 4 は 2 個である（この台本が単位の npz で数えた）．")
DQ_SAME1 = quote(design, "`same_as` の印は，全部の単位で塊ごと・感情ごとの正解の数が同じ版に付く．正解の数が同じでも，誤りの行き先が違えば判定は違う", "設計 4 節")
DQ_SAME2 = quote(design, "判定そのものが全部の単位で同じ版には `same_pred` の印を付ける．「同じ判定」と書くのは `same_pred` の付いた版だけにする．", "設計 4 節")
DQ_SAME3 = quote(design, "作りの上で同じになる版：CREMA-D の提案は n0 ＝ 0 なので，相性を外した版は水準だけの重みと同じ判定になり，n0 を外した版は提案と同じになる．相性を使わない点を選び直した版は，選ばれた点が n0 ＝ 0 なら相性を外した版と同じになる．"
                         "学び直しの 2 つの点が同じ番号の (コーパス, K) では，`stud_prop_sel` は `stud_prop` と同じになる．", "設計 4 節")
w(f"- **「同じ判定」と「正解の数が同じ」**（設計 4 節）：「{DQ_SAME1}」「{DQ_SAME2}」この文書の「同じ判定」は，2 つの版の判定が全部の単位の全部のテストの発話で同じこと（`same_pred`）を指す．"
  "「正解の数が同じ」は，判定は違う発話があるが，塊ごと・感情ごとの正解の数が全部の単位で同じこと（`same_as` が付き `same_pred` が付かない）を指す．どちらも差は 0 になる．同じ群の版は独立な比べとして数えない．"
  f"設計 4 節：「{DQ_SAME3}」")
DQ_LIC = quote(design, "CREMA-D の利用条件は Open Database License（配布元の README）．", "設計 8 節")
DQ_CD1 = quote(design, "CREMA-D は，俳優が決まった 12 文を演技で読むコーパスである．同じ文が学習側とテストにまたがる．**文の重ならない分割での測り直しは範囲の外で，CREMA-D の値は「同じ 12 文が両側に出る条件」での値である．**", "設計 8 節")
DQ_CD2 = quote(design, "CREMA-D の設定（引き戻しの強さ，提案の点，学び直しの点）は，実験 18 の単位で選んだ値である．提案の点と学び直しの点を選んだ単位の仮のターゲットは元の 4 コーパス（分類器が 3 つ）で，演技のコーパスは入らない．"
                       "声の引き戻しの強さを選んだ単位にだけ，RAVDESS を仮のターゲットにした単位が入る．", "設計 8 節")
DQ_CD3 = quote(design, "**CREMA-D には，学び直しの標準の相手（実験 1 の 5 手法，実験 2 の手法）を入れない**．それぞれの設定を CREMA-D のために選ぶ規則と，仮のターゲットの回の計算が新しく要る．CREMA-D の相手は，積，K 本だけの融合，積を疑似ラベルにした融合，別のまとめ方，上限である．", "設計 8 節")
must('"IEO": "It\'s eleven o\'clock."' in (ROOT / "features" / "build_manifest_extra.py").read_text(), "文の記号 IEO の対応")
w(f"- **CREMA-D の事情**（設計 8 節）：「{DQ_CD1}」「{DQ_CD2}」「{DQ_CD3}」台本が単位の npz で確かめたこと：テストは 5 分割の合計で {int(N_PER_SEED[CD].sum()):,} 発話，話者は 91 人（分割ごとに {min(CD_SPK)}〜{max(CD_SPK)} 人）で，"
  "50 単位のどれでも，テストの話者は同じ単位の候補に現れない．ソースは 50 単位とも元の 4 コーパスである．文は 12 種類で，12 種類ともテストと候補の両方に出る．"
  f"文「It's eleven o'clock.」（発話の名前の記号 `IEO`）のテストの発話は {IEO_ALL} 本で，うち Neutral は {IEO_NEU} 本である（テストの Neutral は全部で {CD_NEU:,} 本）．声と顔がそろわない 1 発話（`1076_MTI_SAD_XX`）は，テストにも候補にも無い．"
  f"感情認識の査読想定（`review_affect.md`）は，実験 18 の設計の記録を結果の文に引き継ぐよう求めた（「{RQ['RA5c']}」）．実験 18 の設計 8 節：「{E18_LABEL}」「{E18_FACE}」"
  f"使わない 1 発話について，同じ報告は「{RQ['RA5b']}」と書いている．利用条件について，実験 18 の設計は「{E18_LIC}」と書いていた．**CREMA-D の利用条件は，その後に確かめてある**．同じ報告は「{RQ['RA5d']}」「{RQ['RA5e']}」と書き，この実験の設計 8 節は「{DQ_LIC}」と書いている．"
  f"まだ確かめていないのは RAVDESS の利用条件だけである（同じ報告：「{RQ['RA5a']}」）．")
DQ_OUT1 = quote(design, "**声・顔・言葉の単独の分類器の学び直しは範囲の外**．", "設計 8 節")
DQ_OUT2 = quote(design, "**引き戻しの強さの感度は範囲の外**．強さは K 本だけの融合のために選んだ値を，どの疑似ラベルにも共通に使う．", "設計 8 節")
DQ_OUT3 = quote(design, "1 組のエンコーダ，凍結した特徴の上の線形の分類器に限る．話者と内容の分離についての記録は実験 20 の設計 9 節と同じ．", "設計 8 節")
DQ_OUT4 = quote(design, "性別ごとの内訳は AFEW と IEMOCAP でしか出せない（MOSEI・MELD・CREMA-D の発話一覧には性別の列が無い）．", "設計 8 節")
w(f"- **範囲**（設計 8 節）：「{DQ_OUT1}」「{DQ_OUT2}」「{DQ_OUT3}」「{DQ_OUT4}」話者と内容の分離，コーパスごとの事情は，実験 20 の結果の文（`DAI/docs/実験20_結果の文の下書き_2026-11-15.md`）の 0 節に書いてある．")
DQ_SEEN1 = quote(design, "設計の前に本体が見た値：実験 16・18 の結果（提案の推定時の答えと学び直しの表．CREMA-D の値を含む），実験 1・2 の評価の回の集計（22:01 に実験 18 と並べた），実験 1〜3・14 の評価の回の結果（以前の解析），"
                         "DAI-19 の軽い試しの値（19:50．CREMA-D を含む 6 コーパスの全ラベルで，出発点の UAR を測った），MER-DA の以前の解析での CREMA-D の UAR．", "設計 9 節")
DQ_SEEN2 = quote(design, "実験 20 の結果は見ていない（計算中）．", "設計 9 節")
DQ_SEEN3 = quote(design, "試走で見たのは，行の数・所要・一致の割合・勾配だけである．試走の出力のうち UAR を含むもの（本体の scratchpad と，検査役の `DAI/checks/DAI-21/code_run/`）は，中を開かずに消した．", "設計 9 節")
LQ_START0 = quote(ROW21, "学び直しの出発点を実験 1 と同じ「ソースの融合の分類器の等しい重みの平均」のままにした判断は，DAI-19 の軽い試し（2026-11-14 19:50 ごろ．CREMA-D を含む 6 コーパスの全ラベルで出発点の UAR を測った）を見た後である（この設計を書いたのは同じ日の 22 時台）", "台帳の DAI-21 の行")
must("**出発点についての申告**（漏れの検査 `leak.md` の L1 が求めた 1 文．2026-11-15 に本体が書いた）：" + LQ_START0 in ROW21, "台帳の申告の文の前置き")
w(f"- **見たデータ**（設計 9 節）：「{DQ_SEEN1}」「{DQ_SEEN2}」「{DQ_SEEN3}」"
  f"計算の前の漏れの検査（`leak.md`）は「{RQ['L4']}」と求めていた．設計にこの文は無い．台帳の DAI-21 の行は，本体の申告として「{LQ_START0}」と書いている（2026-11-15 に本体が足した文．この台本は，台帳の行にこの文があることだけを確かめた）．")
w("")
RQ_WHERE = {
    "S2": "冒頭の 3 つの読み分け．CREMA-D の差は区間で読んだ（要点 7〜10，表の下の注）", "S3": "0 節の「区間の作り方」，3.2 節", "S3b": "0 節の「区間の作り方」（実験 20 の設計と点検の報告の文を引いた）", "S5a": "0 節の「区間は 95% で，補正していない」", "S5b": "同上．要点の冒頭",
    "S5c": "0 節の「同じ判定」と「正解の数が同じ」", "S6": "要点 11，2.6 節", "S7": "要点 1 の「今の提案（実験 1 の保存）との差」", "S8": "要点 1 の「K 本の引き方で分けた値」，表の下の注",
    "S38": "3.2 節（計算の前の見込みと結果）", "L2": "要点 8，冒頭の 6 項の 2", "L3": "要点 1 の「K 本の引き方で分けた値」", "L4": "0 節の「見たデータ」（台帳の本体の申告を引いた）", "LV3": "2.6 節",
    "C1": "要点 2", "C5": "要点 1，4 節", "C9a": "要点 10，4 節", "C9b": "並べていない（要点 2，4 節）", "C10": "要点 11（疑似ラベルごとの最大の点どうしの差は書いていない）", "C16a": "2.6 節（この台本が固定のファイルから数え直した）", "C16b": "要点 1，2.6 節",
    "C16c": "2.6 節（ρ と a の端の数を疑似ラベルごとに書いた）", "C11": "要点 2・10",
    "C12": "数えていない（4 節）", "C14": "要点 5・9，4 節", "CV4a": "要点 5", "CV4b": "要点 5", "CV4c": "要点 5", "CV4d": "要点 3", "D2": "要点 11", "D4": "要点 9", "D4b": "要点 9", "D6a": "要点 12", "D6b": "要点 1",
    "RA2": "要点 13・14・16・17，2.7 節（文を除いたテストの表と，文ごとの Neutral の再現率の並びを，計算の後の解析の出力から書いた．K 本の側の影響は測っていない）", "RA3": "2.6 節（表の数を並べた）．要点 18，3.3 節（相関は，計算の後の解析の仮定の表にある）", "RA4": "0 節の「固定のファイル」「CREMA-D の事情」", "RA4b": "2.6 節",
    "RA5a": "0 節の「CREMA-D の事情」", "RA5d": "同上", "RA5e": "同上", "RA5b": "同上", "RA5c": "同上", "RM2": "要点 1", "RM5": "冒頭（この台本が実験 18 の集計と照らした）", "RM10a": "要点 11", "RM10b": "要点 8", "RM11": "2.6 節",
    "K1": "要点 5，4 節", "KV4": "4 節", "KW4": "表の下の注"}
must(set(RQ_WHERE) == set(RQ), "求めの一覧と対応の一覧がそろわない")
w("**計算の前の検査と査読想定の役が，結果の文・表の書き方について残した求めと，この文書の対応**．報告を全部読み，設計の第 3 版に同じ文が無いか，一部だけ入った求めと読み方を，報告の文のまま一覧にした．"
  "設計に入った求め（Q1 の読み方，再現の基準，選ぶデータの表，区間が含まない揺れ，記述の版の経路，文の重なりなど）は，設計の文として 0 節と要点に引いた．起動の台本と，台本を移した後の sha256 についての記録は，上の「計算の前の検査」に書いた．")
w("")
w("| 報告 | 求め・読み方（報告の文のまま） | この文書の対応 |")
w("|---|---|---|")
for k_ in RQ_SRC:
    w(f"| {RQ_FILE[k_]} | {RQ[k_].replace('|', '｜')} | {RQ_WHERE[k_]} |")
w("")
DQ_TCAL = quote(design, "分類器ごとの温度の較正の疑似ラベルは，答えのまとめ方だけでなく確率の鋭さも積や提案と違う（温度が確率の鋭さを変える）．「学び直し（提案）− 学び直し（温度の較正）」の差には，まとめ方の違いと鋭さの違いの両方が入る．", "設計 3.1 節")
w("**版の名前**（表の文書 `DAI/docs/実験21_結果の表_2026-11-15.md` と，実験 20 の結果の文の呼び名を使う．括弧は台本の名前）：")
w("")
w("- **提案の手法**：感情ごとの信頼度の表で分類器の答えをまとめる手法（実験 20 の設計 3 節）．**推定時の答え**（`prop`）は，まとめた答えそのものである．**学び直した融合**（`stud_prop`）は，その答えを疑似ラベルにして融合の分類器を学び直したものである．"
  "融合の分類器は，元の 4 コーパスでは声・顔・言葉の特徴をつないだ分類器，CREMA-D では声・顔の特徴をつないだ分類器である．")
w("- **積**（`prod`）：微調整した分類器の確率を同じ重みで掛けた，推定時の答え．**今の提案**：積を疑似ラベルにして学び直した融合の分類器（実験 1 の手法）．この実験で解いた版（`stud_prod`）と，実験 1 の評価の回の保存の判定（`joint_AVT`．元の 4 コーパスだけ）がある．"
  "**K 本だけの融合**（元の 4 コーパスは `none_AVT`，CREMA-D は `none_AV`）：K 本のラベルだけで学んだ融合の分類器．")
w("- **学び直しの形**：実験 1 の今の提案と同じ式と同じ解き方で，疑似ラベルだけを替える（設計 3.1 節）．設定（疑似ラベルの重み ρ と K 本の倍率 a）は主の点で，積の疑似ラベルのために選ばれた点を，どの疑似ラベルにも使う．")
w("- **疑似ラベルを替えた学び直し 5 つ**（主の点．設計 3.1 節の表）：" + "，".join(f"{lab}（`stud_{t}`）" for t, lab in MT.EXTRA) + "．"
  "相性を外した表は，提案の手法と同じ n0・戻し先で，相性を使わない表である．分類器どうしの水準の違いを外した表は，分類器の水準をそろえ，相性だけを使う表である．"
  f"分類器ごとの温度の較正について，設計 3.1 節は「{DQ_TCAL}」と書いている．")
w("- **実験 1 の比較手法**（元の 4 コーパス．実験 1 の評価の回の保存の判定）：" + "，".join(f"{lab}（`b_{m}`）" for m, lab in MT.EX1 + MT.B_ONLY) + "．"
  "初めの 3 つは積の疑似ラベルを受け取る方法で，この実験は，実験 1 が固定した設定のまま，積の疑似ラベルを渡した版（`kd_prod`・`agree_prod`・`mix_prod`．実験 1 の保存の判定と同じになるかの検算に使う）と，提案の手法の疑似ラベルを渡した版（`kd_prop`・`agree_prop`・`mix_prop`）を解いた．")
w(f"- **切片の扱い**：「切片をずらした答え」は，ラベルを使わずに切片をずらした答えである．{TD_SHIFT}（積はずらす前と後で同じ答えになる）．「切片をずらす前」（版の名前に `@raw` を付ける）は，そのままの答えである．表の文書と同じ定義である．"
  "推定時の答え（提案の手法，別のまとめ方，部品を外した版）は，断りが無ければ切片をずらした答えである．学び直した 1 つの分類器（学び直した融合，K 本だけの融合，実験 1 の比較手法）の判定は，切片をずらしていない．疑似ラベルは，どれも同じ切片のずらしを当てた確率である（設計 3.1 節）．")
w("- **CREMA-D の比べる相手**（声と顔の 2 つの分類器の答えの別のまとめ方．設計 3.3 節の 4）：" + "，".join(f"{lab}（`{v}`）" for v, lab in MT.A2[1:]) + "．"
  "**提案の部品を 1 つ外した版**（設計 3.3 節の 5）：" + "，".join(f"{lab}（`{v}`）" for v, lab in MT.ABL) + "．"
  f"**参照と参考**：{nmc('M_F6_orc')}，{nmc('M_F6_ext')}，{nmc('full_AV')}，{nmc('start_AV')}．")
w("")

# ---- 1 節
w("## 1. 要点")
w("")
w("この節は，区間が 0 を含まない条件と，点推定が負の条件を，名前と値で書く．区間が 0 を含み点推定が正の条件は，名前だけを書く．全部の値は，表の文書と 2 節の表にある．条件の名前は「コーパスの K」で書く．"
  "15 条件は，5 コーパス × 同じテストを共有する 3 つの K で，独立な 15 個の測定ではない（元の 4 コーパスだけの差は 12 条件）．区間は補正していない．条件の数を数えて主張にはしない．"
  "学び直した融合の分類器の値は，断りが無ければ主の点の値である．")
w("")
w("### 学び直した融合の分類器の UAR（主の点．50 単位）")
w("")
w("| コーパス | K | 積の疑似ラベル（今の提案．`stud_prod`） | 提案の手法の疑似ラベル（`stud_prop`） | 提案の手法の疑似ラベル − 積の疑似ラベル |")
w("|---|---|---|---|---|")
for c, K in ALL15:
    w(f"| {c} | {K} | {lci(level('stud_prod', c, K))} | {lci(level('stud_prop', c, K))} | {ci(S_PP(c, K), mark=True)} |")
w("")
w("### 問い 1（設計の Q1）：提案の手法の答えを疑似ラベルにして学び直した融合の分類器は，積を疑似ラベルにして学び直した融合の分類器（今の提案）より UAR が高いか")
w("")
g1 = groups(S_PP, ALL15)
AF3, CD3 = [("AFEW", K) for K in KS], ALL3
MID9 = [(c, K) for c in ("IEMOCAP", "MOSEI", "MELD") for K in KS]
must(set(g1["pp"]) == set(AF3) and set(CD3) <= set(g1["p"]) and not g1["nn"] and not g1["n"] and not g1["same"] and not g1["cnt"], "要点 1 の読み")
MID_P = [x for x in g1["p"] if x in MID9]
MID_Z = [x for x in g1["z"] if x in MID9]
MID_ZN = [x for x in MID_Z if S_PP(*x).est < 0]
must(len(MID_P) + len(MID_Z) == 9 and len(MID_P) > 0 and len(MID_ZN) > 0, "要点 1 の 9 条件の読み")
NEAR1 = [(c, K) for c, K in ALL15 if near0(S_PP(c, K))]
must(NEAR1 == [("MOSEI", 10)] and NEAR1[0] in MID_Z, f"要点 1 の差のうち，区間の端が 0 に近いもの（文を直す）：{NEAR1}")
NEAR1_TXT = (f"**{vals(NEAR1, S_PP)} の区間は，区間の端が 0 に近く，引き直しの乱数の組に依りうる**（＊ の印．MOSEI の区間は塊の重みの引き直しで作る．"
             f"0 に近いほうの端が 0 から標準誤差の {NEAR_TH} 倍以内という，要点 15 と同じ基準に当たる．この条件は，上の「区間が 0 を含む条件」に入れてある）．")
w(f"1. **学び直した融合の分類器で，提案の手法の疑似ラベル − 積の疑似ラベル（`stud_prop − stud_prod`．主の点．同じ計算の中で疑似ラベルだけを替えた差）は，AFEW では 3 つの K とも区間が 0 を含まず 1.0 点以上で（{vals(AF3, S_PP)}），"
  f"CREMA-D では 3 つの K とも区間が 0 を含まず 1.0 点未満だった（{vals(CD3, S_PP)}）**．"
  f"IEMOCAP・MOSEI・MELD の 9 条件（3 コーパス × 同じテストを共有する 3 つの K）の点推定は {rng_of(S_PP, MID9)} 点で，区間が 0 を含まない条件は {vals(MID_P, S_PP)} である（どれも 1.0 点未満）．"
  f"区間が 0 を含む条件は {by_corpus(MID_Z)} で，このうち点推定が負の条件は {vals(MID_ZN, S_PP)} である．"
  f"{NEAR1_TXT}"
  f"設計は，この差の偏りの向きを「{DQ_MAIN}」と書いている（0 節）．値は，手法を選んだときに見たテストと同じテストの上の値である（冒頭の 6 項）．CREMA-D は，元の 4 コーパスと並ぶ独立な確かめとして数えない（冒頭の 6 項の 2）．5 コーパスの結果は互いに独立ではない（同 5）．"
  f"比較の検査（`compare.md`）は，5 コーパスを並べるときの読み方を「{RQ['C5']}」と書いている（主の点は，元の 4 コーパスが ρ = 1・a = 1，CREMA-D の K = 5 が ρ = 1・a = 16，K = 10・25 が ρ = 4・a = 1 である）．"
  f"定義の検査（`define.md`）は「{RQ['D6b']}」と書いている（疑似ラベルは，どれも同じ切片のずらしを当てた確率である）．"
  f"ML の査読想定（`review_ml.md`）は「{RQ['RM2']}」と書いている．点推定が負の {by_corpus(MID_ZN)} は，この読み方に当たる．"
  f"実験 18 の規則で疑似ラベルごとに選ぶと，提案の手法の疑似ラベルは 15 条件のうち {EDGE_PROP['rho_hi']} 条件で ρ の上端（4）を選び，積の疑似ラベルは {EDGE_PROD15['rho_hi']} 条件で選ぶ（2.6 節）．主の点の ρ は，元の 4 コーパスが 1，CREMA-D が K = 5 で 1，K = 10・25 で 4 である．"
  f"比較の検査（`compare.md`）は「{RQ['C16b']}」と書いている．")
gJ = groups(eget("stud_prod", "joint_AVT"), ALL12)
J_REST = gJ["z"]
must(len(gJ["same"]) + len(gJ["cnt"]) + len(J_REST) == 12 and len(gJ["cnt"]) == 1, "stud_prod − joint_AVT の読み（文を直す）")
J_MAX = max(abs(q("stud_prod", "joint_AVT", c, K).est) for c, K in J_REST)
must(J_MAX < 0.01, "stud_prod − joint_AVT の点推定")
DQ_REPRO = quote(design, "単位 × K の最小が 0.995 以上なら，「− 今の提案（実験 1 の保存）」の差を「− 学び直し（積）」と同じ重みで読む．", "設計 4 節")
DEV_J = max(abs(S_PP(c, K).est - S_J(c, K).est) for c, K in ALL12)
must(all(klass(S_PP(c, K)) == klass(S_J(c, K)) for c, K in ALL12) and DEV_J < 0.01, "stud_prop − joint_AVT の読みが stud_prop − stud_prod と違う（文を直す）")
w(f"   - **今の提案（実験 1 の保存）との差**（`stud_prop − joint_AVT`．元の 4 コーパス）：設計 4 節は「{DQ_REPRO}」と決めている．今の提案の再現の最小は {s4_['joint_repro_min']:.6f} で，0.995 未満の（単位，K）は無い（0 節）．"
  f"12 条件の読みの区分は `stud_prop − stud_prod` と同じで，2 つの差の点推定のずれの最大は {DEV_J:.4f} 点である．値は表の文書の 1 節にある．"
  f"統計の検査（`stats.md`）は「{RQ['S7']}」ことを求めた．`stud_prod − joint_AVT`（今の提案をこの実験で解いた版 − 実験 1 の保存）は，12 条件のうち {len(gJ['same'])} 条件で同じ判定，"
  f"{len(gJ['cnt'])} 条件（{by_corpus(gJ['cnt'])}）で正解の数が同じ（判定は違う発話がある），残る {len(J_REST)} 条件（{by_corpus(J_REST)}）は区間が 0 を含み，点推定の絶対値は {J_MAX:.4f} 点以下である．")
T20_SEED = quote(text20, "MOSEI の種で分けた値のうち種 0・1 の区間は，種の項の自由度が小さいので狭めに出る．", "実験 20 の結果の文")
w("   - **K 本の引き方で分けた値**（`stud_prop − stud_prod`）：引き方 0・1 は下調べ（元の 4 コーパスは実験 16 と K = 10 の追補，CREMA-D は実験 18）と同じ K 本（コーパスごとに 10 単位），引き方 2〜9 は下調べと違う K 本（40 単位）である．"
  "テストの発話は，どちらも下調べと同じ発話である．2 つは同じテストの上の値で，独立な 2 回の測定ではない．CREMA-D の引き方 0・1 の区間は狭めに出る（0 節）．MOSEI も CREMA-D と同じ作り方の区間で，実験 20 の結果の文は"
  f"「{T20_SEED}」と書いている．統計の検査（`stats.md`）は「{RQ['S8']}」，漏れの検査（`leak.md`）は「{RQ['L3']}」と書いている．値の表は表の文書の 1 節にある．")
w(f"      - 引き方 0・1：{read_short(eget('stud_prop', 'stud_prod', '0-1'), ALL15)}")
w(f"      - 引き方 2〜9：{read_short(eget('stud_prop', 'stud_prod', '2-9'), ALL15)}")
PE = {(r.c, int(r.K), r.version, int(r.emotion)): r for r in pe.itertuples()}


def emo_diff(a: str, b: str, c: str, e: int) -> list:
    return [PE[(c, K, a, e)].recall - PE[(c, K, b, e)].recall for K in KS]


em_txt, em_pos, em_neg = [], [], []
for c in C5:
    parts = []
    for e in range(4):
        d = emo_diff("stud_prop", "stud_prod", c, e)
        parts.append(f"{EMO[e]} {fpt(min(d))}〜{fpt(max(d))}")
        if all(x > 0 for x in d):
            em_pos.append((c, e))
        if all(x < 0 for x in d):
            em_neg.append((c, e))
    em_txt.append(f"{c} が " + "，".join(parts))


def emo_list(xs) -> str:
    return "，".join(f"{c} の {'・'.join(EMO[e] for c2, e in xs if c2 == c)}" for c in C5 if any(c2 == c for c2, _ in xs)) if xs else "無い"


w("   - **感情ごとの再現率**（2.5 節．`stud_prop − stud_prod` の点の値の引き算．感情ごとの差の区間は作っていない）：3 つの K の値の幅は，" + "；".join(em_txt) + " である．"
  f"3 つの K とも引き算が正の感情は，{emo_list(em_pos)}．3 つの K とも負の感情は，{emo_list(em_neg)}．区間が無いので，符号と大きさを並べるだけにとどめる．")
w("")
w("### 問い 2（設計の Q2．記述）：学び直した融合の分類器の UAR は，既にある方法（K 本だけの融合，実験 1 の 5 つの比較手法）と並べてどの高さか（元の 4 コーパス）")
w("")
B6 = ["none_AVT"] + B5
N_LOW2 = sum(cnt(eget("stud_prop", v), ALL12, ("nn", "n")) for v in B6)
must(N_LOW2 == 0 and all(cnt(eget("stud_prop", v), ALL12, ("same", "cnt")) == 0 for v in B6), "要点 2 の数え（文を直す）")
BK2 = brk([q("stud_prop", v, c, K) for v in B6 for c, K in ALL12])
BK2I = brk([q("stud_prop", v, "IEMOCAP", K) for v in B6 for K in KS])
must(BK2["n"] == 72 and BK2["lo"] == 0 and BK2["zero"] == 0 and BK2["hi"] + BK2["z"] == 72 and BK2I["n"] == 18, f"要点 2 の内訳：{BK2}")
w(f"2. **提案の手法の疑似ラベルで学び直した融合（`stud_prop`）と，K 本だけの融合・実験 1 の 5 つの比較手法（実験 1 の評価の回の保存の判定）との差では，相手 {len(B6)} 個 × 12 条件の {BK2['n']} 個の区間（補正していない）のうち，"
  f"区間が 0 を含まず `stud_prop` が高いのは {BK2['hi']} 個，区間が 0 を含むのは {BK2['z']} 個（うち点推定が負 {BK2['zn']} 個．IEMOCAP は {BK2I['n']} 個のうち {BK2I['z']} 個が 0 を含む）で，区間が 0 を含まずに `stud_prop` が低かった条件は無い**．"
  "区間が 0 を含む差は，2 つの版が同じ水準であることを意味しない．低かった条件が無いことを，損が無いことの証拠として読まない．IEMOCAP は自由度 3 で区間が広い．"
  "差は，同じ 200 単位の上の対の差である．実験 1 の 5 つの比較手法の設定は，それぞれの方法が自分のために選んだ点で（候補の数は 12・9・9・4・5），`stud_prop` の主の点は，積の疑似ラベルのために選ばれた点を借りた点である（0 節の表）．"
  "この差は，疑似ラベルの違いと学び直しの形の違いの両方を含む（設計 1 節の Q2．内訳は要点 3）．`stud_prop` の判定も相手の判定も，切片をずらしていない．"
  f"比較の検査（`compare.md`）は，この並べ方に 3 つの注を求めた：「{RQ['C1']}」K 本だけの融合との差について，同じ報告は「{RQ['C11']}」と書いている．"
  f"標準の手法（実験 2 の手法）と `stud_prop` は，この文書では並べていない（同じ報告：「{RQ['C9b']}」．実験 2 の手法の UAR は，実験 20 の表の文書の 5 節にある）．"
  "相手ごとの読みは次のとおりである（12 条件の値と UAR は表の文書の 3 節）．")
for v in B6:
    w(f"   - **{nmc(v)}**：{read_short(eget('stud_prop', v), ALL12)}")
DEV_ADD = max(max(abs(q("stud_prop", f"b_{m}", c, K).est - q("stud_prop", f"{m}_prop", c, K).est - q(f"{m}_prop", f"b_{m}", c, K).est),
                  abs(q(f"{m}_prop", f"b_{m}", c, K).est - q(f"{m}_prop", f"{m}_prod", c, K).est - q(f"{m}_prod", f"b_{m}", c, K).est)) for m in EX1 for c, K in ALL12)
DEV_REP = max(abs(q(f"{m}_prod", f"b_{m}", c, K).est) for m in EX1 for c, K in ALL12)
must(DEV_ADD < 1e-9 and DEV_REP < 0.01, f"要点 3 の足し算：{DEV_ADD:.2e}，{DEV_REP:.2e}")
EXN = dict(MT.EX1)


def multi4(mk, kinds) -> list:
    """3 手法 × 12 条件のうち，読みが kinds に入る (手法, コーパス, K)．"""
    return [(m, c, K) for m in EX1 for c, K in ALL12 if klass(q(*mk(m), c, K)) in kinds]


def multi4_txt(sel, mk) -> str:
    out = []
    for m in EX1:
        x = [(c, K) for m2, c, K in sel if m2 == m]
        if x:
            out.append(f"{EXN[m]}の " + vals(x, lambda c, K: q(*mk(m), c, K)))
    return "；".join(out)


MK_A = lambda m: (f"{m}_prop", f"{m}_prod")       # noqa: E731
MK_B = lambda m: ("stud_prop", f"{m}_prop")       # noqa: E731
MK_C = lambda m: (f"{m}_prop", f"b_{m}")          # noqa: E731
A_NEG, A_POS = multi4(MK_A, ("nn", "n")), multi4(MK_A, ("pp", "p"))
B_NEG, B_POS = multi4(MK_B, ("nn", "n")), multi4(MK_B, ("pp", "p"))
must(len(A_NEG) > 0 and len(B_NEG) > 0, "要点 3 の，区間が 0 を含まず負の条件が無い（文を直す）")
BKA = brk([q(*MK_A(m), c, K) for m in EX1 for c, K in ALL12])
BKB = brk([q(*MK_B(m), c, K) for m in EX1 for c, K in ALL12])
must(BKA["lo"] == len(A_NEG) and BKB["lo"] == len(B_NEG) and BKA["zero"] == 0 and BKB["zero"] == 0 and BKA["hi"] + BKA["z"] + BKA["lo"] == 36 and BKB["hi"] + BKB["z"] + BKB["lo"] == 36, "要点 3 の内訳")
w(f"3. **温度つき蒸留・一致した発話だけの疑似ラベル・ラベルと疑似ラベルを混ぜる方法の 3 つについて，`stud_prop` − 実験 1 の手法を，設計 4 節の読み方で分けた**．"
  "`stud_prop − <手法>（実験 1 の保存）` ＝（同じ疑似ラベルで学び直しの形を替えた差：`stud_prop − <手法>_prop`）＋（同じ手法の中で疑似ラベルだけを替えた差：`<手法>_prop − <手法>_prod`）＋（この実験で解き直した版と実験 1 の保存のずれ：`<手法>_prod − <手法>（実験 1 の保存）`）で，"
  f"点推定はこの足し算が成り立つ（台本が確かめた）．解き直した版と実験 1 の保存のずれの点推定の絶対値は，3 手法 × 12 条件のどれでも {DEV_REP:.4f} 点以下である．共訓練と共正則化は積を受け取らない方法なので，この内訳は作れない（設計 4 節）．値は表の文書の 3 節にある．")
w(f"   - **同じ手法の中で疑似ラベルだけを替えた差**（`<手法>_prop − <手法>_prod`．どちらもこの実験で解いた．両側は同じ手法・同じ設定で，違いは疑似ラベルだけである．設計 4 節の名前は「疑似ラベルの違いの分」）："
  f"区間が 0 を含まずに提案の手法の疑似ラベルが低かった条件は，{multi4_txt(A_NEG, MK_A)} である．3 手法 × 12 条件の 36 個の区間（補正していない）のうち，ほかの {36 - len(A_NEG)} 個は，区間が 0 を含まず提案の手法の疑似ラベルが高い {BKA['hi']} 個と，区間が 0 を含む {BKA['z']} 個（うち点推定が負 {BKA['zn']} 個）である．"
  f"設計 4 節は，この差の偏りの向きを「{DQ_Q2B.split('．')[0]}．{DQ_Q2B.split('．')[1]}．」と書いている．"
  "一方，提案の手法の疑似ラベルは，形と設定を同じテストの発話の上の値を見た後に決めた手法の答えである（冒頭の 6 項の 6）．比較の検査（`compare.md`）は，これを提案に有利な向きと書いている（要点 2 に引いた文の (2)）．"
  "不利な向きと有利な向きの 2 つを合わせた向きは決まらない．")
for m in EX1:
    w(f"      - {EXN[m]}：{read_short(eget(*MK_A(m)), ALL12)}")
w(f"   - **同じ疑似ラベルで学び直しの形を替えた差**（`stud_prop − <手法>_prop`．両側は同じ提案の手法の疑似ラベルを受け取る．違いは，学び直しの形と，それぞれの形の設定である．設計 4 節の名前は「学び直しの形の違いの分」）："
  f"区間が 0 を含まずに `stud_prop` が低かった条件は，{multi4_txt(B_NEG, MK_B)} である．3 手法 × 12 条件の 36 個の区間（補正していない）のうち，ほかの {36 - len(B_NEG)} 個は，区間が 0 を含まず `stud_prop` が高い {BKB['hi']} 個と，区間が 0 を含む {BKB['z']} 個（うち点推定が負 {BKB['zn']} 個）である．"
  f"比較の再検査（`compare_v2.md`）は「{RQ['CV4d']}」と書いている．"
  "両側が同じ提案の手法の疑似ラベルを受け取るので，疑似ラベルの形と設定を同じテストの上の値を見た後に決めたことによる有利な向きは，この差の両側に入る．")
for m in EX1:
    w(f"      - {EXN[m]}：{read_short(eget(*MK_B(m)), ALL12)}")
DQ_AFEW = quote(design, "AFEW のテストは 5 分割とも 158 発話なので，1 発話でも判定が違うと一致の割合は 0.9937 になり基準を下回る．AFEW ではこの基準は「完全に一致」と同じ意味である．", "設計 4 節")
DEV_C = max(abs(q(*MK_C(m), c, K).est - q(*MK_A(m), c, K).est) for m in EX1 for c, K in ALL12)
C_DIFF = [(m, c, K) for m in EX1 for c, K in ALL12 if klass(q(*MK_C(m), c, K)) != klass(q(*MK_A(m), c, K))]
must(C_DIFF == [] and DEV_C < 0.01, f"提案の疑似ラベルの版 − 保存の読みが，この実験の中の差と違う（文を直す）：{C_DIFF}")
w(f"   - **再現の基準**（設計 4 節）：3 手法とも，積の疑似ラベルを渡した版の判定が実験 1 の保存の判定と一致する割合に，0.995 未満の（単位，K）は無い（最小は，温度つき蒸留 {s4_['kd_repro_min']:.6f}，ほかの 2 つは 1）．"
  f"設計 4 節の基準では，「提案の手法の疑似ラベルを渡した版 − 実験 1 の保存」（`<手法>_prop − <手法>（実験 1 の保存）`）も並べられる．この差の 36 条件の読みの区分は，上の「同じ手法の中で疑似ラベルだけを替えた差」と同じで，点推定のずれの最大は {DEV_C:.4f} 点である．"
  f"AFEW について，設計 4 節は「{DQ_AFEW}」と書いている．AFEW では 4 つの再現の割合が全部 1 である（0 節）．")
w("4. **実験 1 の保存どうしの差**（設計 4 節が並べると決めた差．実験 1 の評価の回の保存の判定から数えた値で，この実験で解いた分類器の値ではない）．"
  f"今の提案 − K 本だけの融合（`joint_AVT − none_AVT`）：{read_short(eget('joint_AVT', 'none_AVT'), ALL12)}"
  f"提案の手法（推定時の答え）− 今の提案（`prop − joint_AVT`．実験 20 の結果の文の要点 4 と同じ対）：{read_short(eget('prop', 'joint_AVT'), ALL12)}"
  "実験 1 の 5 つの比較手法 − 今の提案の読みの一覧は 2.2 節にある．")
w("")
w("### 疑似ラベルを替えた学び直し 5 つ（5 コーパス．主の点）")
w("")


def multi5(mk, kinds) -> list:
    return [(t, c, K) for t in EXTRA for c, K in ALL15 if klass(q(*mk(t), c, K)) in kinds]


def multi5_txt(sel, mk) -> str:
    out = []
    for t in EXTRA:
        x = [(c, K) for t2, c, K in sel if t2 == t]
        if x:
            out.append(f"{nmc('stud_' + t)}との差の " + vals(x, lambda c, K: q(*mk(t), c, K)))
    return "；".join(out)


MK_P = lambda t: ("stud_prop", f"stud_{t}")       # noqa: E731
MK_X = lambda t: (f"stud_{t}", "stud_prod")       # noqa: E731
P_NEG = multi5(MK_P, ("nn", "n"))
must(len(P_NEG) > 0 and {t for t, _, _ in P_NEG} >= {"tcal", "konly"}, "要点 5 の，提案の手法の疑似ラベルが低い条件（文を直す）")
BKP = brk([q(*MK_P(t), c, K) for t in EXTRA for c, K in ALL15])
must(BKP["n"] == 75 and BKP["lo"] == len(P_NEG) and BKP["zero"] == 0 and BKP["hi"] + BKP["z"] + BKP["lo"] == 75, "要点 5 の内訳")
NEAR5P = [(t, c, K) for t in EXTRA for c, K in ALL15 if near0(q(*MK_P(t), c, K))]
NEAR5X = [(t, c, K) for t in EXTRA for c, K in ALL15 if near0(q(*MK_X(t), c, K))]
must(("konly", "MOSEI", 25) in NEAR5P and all(c in ("MOSEI", CD) for _, c, _ in NEAR5P + NEAR5X), f"要点 5 の差のうち，区間の端が 0 に近いもの（文を直す）：{NEAR5P}，{NEAR5X}")
NEAR5_TXT = (f"**次の区間は，区間の端が 0 に近く，引き直しの乱数の組に依りうる**（＊ の印．0 に近いほうの端が 0 から標準誤差の {NEAR_TH} 倍以内という，要点 15 と同じ基準に当たる）："
             f"提案の手法の疑似ラベル − 別の疑似ラベルでは，{multi5_txt(NEAR5P, MK_P)}"
             + (f"．別の疑似ラベル − 積の疑似ラベルでは，{multi5_txt(NEAR5X, MK_X)}" if NEAR5X else "") + "．この区間の読み（区間が 0 を含むか）は，下の読みのままにしてある．")
w(f"5. **提案の手法の疑似ラベル − 別の疑似ラベル（`stud_prop − stud_<別の疑似ラベル>`．同じ主の点で，疑似ラベルだけを替えた差）で，区間が 0 を含まずに提案の手法の疑似ラベルが低かった条件は，{multi5_txt(P_NEG, MK_P)} である**．"
  f"別の疑似ラベル 5 つ × 15 条件の 75 個の区間（補正していない）のうち，ほかの {75 - len(P_NEG)} 個は，区間が 0 を含まず提案の手法の疑似ラベルが高い {BKP['hi']} 個と，区間が 0 を含む {BKP['z']} 個（うち点推定が負 {BKP['zn']} 個）である．"
  f"{NEAR5_TXT}"
  f"分類器ごとの温度の較正との差について，設計 3.1 節は「{DQ_TCAL.split('．')[1]}．」と書いている．別の疑似ラベルごとの読みを下に並べる（値と UAR は表の文書の 2 節．読みの一覧は 2.2 節）．"
  "下の差に，設計は部品の名前を付けていない．この文書も「〜の分」と呼ばない．"
  f"比較の再検査（`compare_v2.md`）は，K 本だけで測った表との差と相性を外した表との差について「{RQ['CV4a']}」と書いている（「提案の追加の情報の分」は報告の呼び名である．この文書はこの名前を使わず，差の両側の版の名前で書く）．"
  f"同じ報告は「{RQ['CV4b']}」「{RQ['CV4c']}」とも書いている．比較の検査（`compare.md`）の案：「{RQ['C14']}」．"
  f"台本の再検査（`code_v2.md`）によると，{RQ['K1']}．")
for t, lab in MT.EXTRA:
    w(f"   - **{lab}（`stud_{t}`）**")
    w(f"      - 提案の手法の疑似ラベル − {lab}の疑似ラベル：{read_short(eget(*MK_P(t)), ALL15)}")
    w(f"      - {lab}の疑似ラベル − 積の疑似ラベル：{read_short(eget(*MK_X(t)), ALL15)}")
w("")
w("### 提案の手法の中の，学び直した融合 − 推定時の答え（5 コーパス）")
w("")
S_SP = eget("stud_prop", "prop")
g6 = groups(S_SP, ALL15)
G6_NZ = g6["pp"] + g6["p"] + g6["n"] + g6["nn"]
must(len(G6_NZ) >= 1 and not g6["same"] and not g6["cnt"], "要点 6 の読み（文を直す）")
w(f"6. **提案の手法の中で，学び直した融合 − 推定時の答え（`stud_prop − prop`）は，{vals(G6_NZ, S_SP)} で区間が 0 を含まず，残る {15 - len(G6_NZ)} 条件は区間が 0 を含んだ**．15 条件の点推定は {rng_of(S_SP, ALL15)} 点である．"
  f"{read_short(S_SP, ALL15)}推定時の答えは，推定のときに声・顔・言葉（CREMA-D は声・顔）の分類器を動かしてまとめた答えで，切片をずらした答えである．学び直した融合は 1 つの分類器で，切片をずらしていない．")
w(f"   - **学び直した融合 − 積（推定時の答え）**（`stud_prop − prod`）：{read_short(eget('stud_prop', 'prod'), ALL15)}")
w("   - 推定時の答えどうしの「提案の手法 − 積」は，元の 4 コーパスが実験 20 の集計，CREMA-D が実験 21 の集計にある（2.3 節に並べた）．")
w("")

w("### 問い 3（設計の Q3）：CREMA-D で，提案の手法の推定時の答えは，積や別のまとめ方より UAR が高いか（声と顔の 2 つの分類器）")
w("")
C_PP = eget("prop", "prod")
must(all(klass(C_PP(CD, K)) == "p" for K in KS), "要点 7 の読み（文を直す）")
HW_PP = [(C_PP(CD, K).hi - C_PP(CD, K).lo) / 2 for K in KS]
DQ_HW = quote(design, "CREMA-D の区間の半幅は約 0.3 点の見込みなので，0.3〜1.0 点の差は，区間が 0 を含まなくても印が付かない．", "設計 4 節")
w(f"7. **CREMA-D の推定時の答えで，提案の手法 − 積（どちらも切片をずらした答え）は，3 つの K とも区間が 0 を含まず 1.0 点未満だった（{vals(ALL3, C_PP)}）**．差が 1.0 点未満なので，表の印は付かない．"
  f"設計 4 節は計算の前に「{DQ_HW}」と書いている．この差の区間の半幅は，K = 5・10・25 の順に {'・'.join(f'{x:.2f}' for x in HW_PP)} 点である．"
  "CREMA-D は，元の 4 コーパスと並ぶ独立な確かめとして数えない（冒頭の 6 項の 2）．CREMA-D の値は，同じ 12 文が学習側とテストの両方に出る条件での値で，設定は実験 18 の単位で選んだ値である（0 節）．"
  "K 本の引き方 0・1 の単位は，実験 18 の単位とソース・K 本が同じである（冒頭の 6 項の 4）．"
  "設計 4 節が計算の前に予告した，文「It's eleven o'clock.」を除いたテストの値と，文を塊にした区間は，計算の後の解析で出した（要点 13〜15，2.7 節）．"
  "この項と要点 8〜10 の CREMA-D の値は，テストの全発話の上の値で，区間は話者を塊にした区間である．")
w(f"   - **切片をずらす前どうし**（`prop@raw − prod@raw`）：{read_short(eget('prop@raw', 'prod@raw'), ALL3)}")
w(f"   - **提案の手法の中の，切片をずらした答え − ずらす前**（`prop − prop@raw`）：{read_short(eget('prop', 'prop@raw'), ALL3)}")
w(f"   - **K 本の引き方で分けた値**（どちらも切片をずらした答え．値の表は 2.3 節）：引き方 0・1（10 単位）：{read_short(eget('prop', 'prod', '0-1'), ALL3)}引き方 2〜9（40 単位）：{read_short(eget('prop', 'prod', '2-9'), ALL3)}"
  "2 つは同じテストの上の値で，独立な 2 回の測定ではない．引き方 0・1 の区間は狭めに出る（0 節）．")
em7 = [f"{EMO[e]} {fpt(min(emo_diff('prop', 'prod', CD, e)))}〜{fpt(max(emo_diff('prop', 'prod', CD, e)))}" for e in range(4)]
w("   - **感情ごとの再現率**（2.5 節．提案の手法 − 積の点の値の引き算．感情ごとの差の区間は作っていない）：3 つの K の値の幅は，" + "，".join(em7) + " である．")
DQ20_WF = quote(design20, "3 つの候補の数は入れ子の表なので，独立な 3 つの相手として数えない．", "実験 20 の設計 4 節")
DQ_2CLS = quote(design, "**分類器が 2 つであることの読み方**：K 本で量を合わせる相手は，自由に動かせる値の数が 4 コーパスより少ない（重みの表を探す版では 8 個が 4 個になる）．相手の位置を，4 コーパスと並べて同じ性質として読まない．", "設計 3.3 節")
DQ_PSEL = quote(design, "別のコーパスで選んだ指数の積は，CREMA-D には選んだ指数が無いので入れない．", "設計 3.3 節")
DQ_WF = quote(design, "重みの表を探す版（候補の数 10・100・1,000．表は Dirichlet(1, 1) を感情ごとに引いて 2 倍．種 `[20261135, 1]`．表 0 は全部 1）", "設計 3.3 節")
must(prov["rng_seed"] == 20261135 and "u = ua[:m + 1]" in src21, "重みの表を探す版の種と，候補の取り方")
OPP = A2[1:]
MKA = lambda v: ("prop", v)                       # noqa: E731
MKR = lambda v: ("prop@raw", f"{v}@raw")          # noqa: E731
MKM = lambda v: ("prop", f"{v}@raw")              # noqa: E731
must(len(OPP) == 11 and all(n_multi(OPP, mk_, ("nn", "n", "same")) == 0 for mk_ in (MKA, MKR, MKM)), "要点 8 の数え（文を直す）")
ZN8 = [(v, K) for v in OPP for K in KS if klass(q("prop", v, CD, K)) == "z" and q("prop", v, CD, K).est < 0]
must(len(ZN8) > 0, "要点 8 の点推定が負の条件が無い（文を直す）")
BK8 = brk([q("prop", v, CD, K) for v in OPP for K in KS])
must(BK8["n"] == 33 and BK8["lo"] == 0 and BK8["zero"] == 0 and BK8["zn"] == len(ZN8), f"要点 8 の内訳：{BK8}")
w(f"8. **CREMA-D で，提案の手法 − 別のまとめ方（相手 {len(OPP)} 個．どちらも切片をずらした答え）の {BK8['n']} 個の区間（相手 {len(OPP)} 個 × 同じテストを共有する 3 つの K．補正していない）のうち，"
  f"区間が 0 を含まず提案の手法が高いのは {BK8['hi']} 個，区間が 0 を含むのは {BK8['z']} 個（うち点推定が負 {BK8['zn']} 個）で，区間が 0 を含まずに提案の手法が低かった条件は無い**．"
  "区間が 0 を含む差は，2 つの版が同じ水準であることを意味しない．"
  "点推定が負の条件は，" + "，".join(f"{nmc(v)}の K = {K} で {ci(q('prop', v, CD, K))}" for v, K in ZN8) + " である．"
  f"相手ごとの読み：{read_multi(OPP, MKA)}値と UAR は表の文書の 4 節にある．"
  f"設計 3.3 節は相手の読み方を「{DQ_2CLS}」と書いている．{DQ_PSEL}（設計 3.3 節）"
  f"**重みの表を乱数で探す版（候補 10・100・1,000）の値は，種を 1 つに決めて引いた表の組に条件づけた値で，表の引き方の揺れは出していない**（設計 3.3 節：「{DQ_WF}」．候補 10 の版は，全部 1 の表と，引いた表のうち表 1〜10 から選ぶ）．"
  f"実験 20 の設計 4 節は「{DQ20_WF}」と決めている．"
  f"ML の査読想定（`review_ml.md`）は「{RQ['RM10b']}」と書いている．漏れの検査（`leak.md`）は「{RQ['L2']}」と書いている．"
  "CREMA-D を評価するコーパスに足すと決めたのは，実験 18 で CREMA-D の同じテストの上の値を見た後である（冒頭の 6 項の 2）．この節の値は，その前後を含めて，探索の結果として読む．")
w(f"   - **切片をずらす前どうし**（`prop@raw − 相手@raw`．{len(OPP) * 3} 個の区間）：{read_multi(OPP, MKR)}")
w(f"   - **提案の手法（切片をずらした答え）− 相手（切片をずらす前）**（相手に切片のずらしを当てない判定との差．{len(OPP) * 3} 個の区間）：{read_multi(OPP, MKM)}")
w("   - **相手 − 積**（どちらも切片をずらした答え）の読みの一覧は 2.2 節にある．")
DEV9 = max(max(abs(q("prop", "prod", CD, K).est - q("prop", "prop_noprof", CD, K).est - q("prop_noprof", "prod", CD, K).est),
               abs(q("prop", "prod", CD, K).est - q("prop", "prop_levelfree", CD, K).est - q("prop_levelfree", "prod", CD, K).est)) for K in KS)
must(DEV9 < 1e-9, f"要点 9 の足し算：{DEV9:.2e}")
must(all(PH[(CD, K, "prop_n0zero")].hexdigest() == PH[(CD, K, "prop")].hexdigest() and PH[(CD, K, "prop_noprof")].hexdigest() == PH[(CD, K, "M_F2_K")].hexdigest() for K in KS)
     and [PH[(CD, K, "prop_noprof_sel")].hexdigest() == PH[(CD, K, "prop_noprof")].hexdigest() for K in KS] == [True, True, False]
     and [int(FZC[K]["noprof_sel"]) for K in KS] == [0, 0, 4] and GRID[0] == (0.0, 0.0, "eq") and GRID[4] == (20.0, 0.0, "eq"), "CREMA-D の作りの上で同じ判定になる版（文を直す）")
must(all(klass(q("prop", "prop_n0zero", CD, K)) == "same" for K in KS), "prop − prop_n0zero")
w("9. **CREMA-D で，提案の手法 − 部品を 1 つ外した版**（どちらも切片をずらした答え．値は表の文書の 4 節）．CREMA-D の提案の点は 3 つの K とも n0 = 0 である．"
  "点推定は，提案の手法 − 積 ＝（提案の手法 − 相性を外す）＋（相性を外す − 積）＝（提案の手法 − 水準の違いを外す）＋（水準の違いを外す − 積）の足し算が成り立つ（台本が確かめた）．外す順序で値が変わるので，2 つの順序を並べる．"
  f"この項の差は推定時の答えの差である（比較の検査 `compare.md` の案：「{RQ['C14']}」）．")
w(f"   - **提案の手法 − {nmc('prop_noprof')}**（両側の違いは，相性を使うかどうかだけである）：{read_short(eget('prop', 'prop_noprof'), ALL3)}"
  f"相性を外した版は，K 本で測った水準だけで重みを決める版で，3 つの K とも{nmc('M_F2_K')}と同じ判定である．**相性を外す − 積**：{read_short(eget('prop_noprof', 'prod'), ALL3)}")
w(f"   - **提案の手法 − {nmc('prop_levelfree')}**（対照は{gdesc(LEV14)}で，2 つの分類器の水準を K 本で測った水準の平均にそろえ，相性はソースから取る）：{read_short(eget('prop', 'prop_levelfree'), ALL3)}"
  f"**水準の違いを外す − 積**：{read_short(eget('prop_levelfree', 'prod'), ALL3)}")
w(f"   - **提案の手法 − {nmc('prop_noint')}**（同じ格子の点で，重みの表からこの成分だけを消した対照との差）：{read_short(eget('prop', 'prop_noint'), ALL3)}"
  f"定義の検査（`define.md`）は，CREMA-D のこの対照について「{RQ['D4']}」「{RQ['D4b']}」と書いている．"
  f"**分類器 × 感情の成分を消す − 相性を外す**：{read_short(eget('prop_noint', 'prop_noprof'), ALL3)}")
w(f"   - **提案の手法 − {nmc('prop_home')}**（相性を，残り方を掛けないソースの中の値から取る版との差）：{read_short(eget('prop', 'prop_home'), ALL3)}")
w(f"   - **提案の手法 − {nmc('prop_noprof_sel')}**（相性を使わない点から，提案の手法と同じ規則で実験 18 の単位が選んだ点との差）：選ばれた点は，K = 5・10 が{gdesc(0)}で，相性を外した版と同じ判定である．"
  f"K = 25 は{gdesc(4)}で，両側は相性と n0 の 2 つが違う．値は K = 25 で {ci(q('prop', 'prop_noprof_sel', CD, 25))} である（K = 5・10 は「提案の手法 − 相性を外す」と同じ値）．")
w(f"   - **提案の手法 − {nmc('prop_n0zero')}**：提案の点が n0 = 0 なので，3 つの K とも提案の手法と同じ判定である（差は 0）．")
w(f"   - **参照と参考**：{nmc('M_F6_orc')} − 積：{read_short(eget('M_F6_orc', 'prod'), ALL3)}{nmc('M_F6_ext')} − 積：{read_short(eget('M_F6_ext', 'prod'), ALL3)}"
  f"提案の手法 − ソースだけで測った表：{read_short(eget('prop', 'M_F6_ext'), ALL3)}全ラベルで測った表は，評価するコーパスの候補のうち K 本を除いた発話の全ラベルで信頼度を測る版で，実際には使えない．提案の手法 − 全ラベルで測った表の対の差は集計に無い．"
  "UAR の点の値の引き算（全ラベルで測った表 − 提案の手法．対の差ではない．区間は無い）は，K = 5・10・25 の順に "
  + "・".join(fpt(level("M_F6_orc", CD, K).uar - level("prop", CD, K).uar) for K in KS) + " 点である．")
SING = {K: max(("ret_A", "ret_V"), key=lambda v: level(v, CD, K).uar) for K in KS}
must(set(SING.values()) == {"ret_A"} and not any("ret_" in a or "ret_" in b for a, b in PC), "CREMA-D の単独の分類器（文を直す）")
w("10. **CREMA-D で，学び直した 1 つの分類器と並べる**（学び直した 1 つの分類器の判定は，切片をずらしていない．UAR は表の文書の 4 節）．"
  f"提案の手法（推定時の答え）− K 本だけの融合（`prop − none_AV`）：{read_short(eget('prop', 'none_AV'), ALL3)}"
  f"提案の手法（推定時の答え）− 今の提案（`prop − stud_prod`）：{read_short(eget('prop', 'stud_prod'), ALL3)}"
  f"今の提案 − K 本だけの融合（`stud_prod − none_AV`）：{read_short(eget('stud_prod', 'none_AV'), ALL3)}"
  f"提案の手法（学び直した融合）− K 本だけの融合（`stud_prop − none_AV`）：{read_short(eget('stud_prop', 'none_AV'), ALL3)}"
  f"今の提案 − 積（`stud_prod − prod`）：{read_short(eget('stud_prod', 'prod'), ALL3)}"
  f"CREMA-D には，学び直しの標準の相手（実験 1 の 5 手法，実験 2 の手法）を入れていない（0 節）．CREMA-D の学び直しの値は，既にある学び直しの手法との比べではない（比較の検査 `compare.md` の求め：「{RQ['C9a']}」）．"
  f"K 本だけの融合との差について，同じ報告は「{RQ['C11']}」と書いている．"
  "この項の差の，文「It's eleven o'clock.」を除いたテストの値と，文を塊にした区間は，要点 13〜15 と 2.7 節にある．"
  "単独の分類器（微調整した声・顔）と提案の手法の対の差は集計に無い．UAR の値が大きいほうの単独の分類器は 3 つの K とも声で，UAR の点の値の引き算（提案の手法 − 微調整した声．対の差ではない．区間は無い）は，K = 5・10・25 の順に "
  + "・".join(fpt(level("prop", CD, K).uar - level("ret_A", CD, K).uar) for K in KS) + " 点である．"
  f"上限（候補の全ラベルで学んだ融合．参照）の UAR は {level('full_AV', CD, 5).uar:.2f}，ソースの融合の分類器の平均（K = 0）の UAR は {level('start_AV', CD, 5).uar:.2f} である（どちらも K に依らない判定）．")
must(all(abs(level(v, CD, 5).uar - level(v, CD, K).uar) < 1e-9 for v in ("full_AV", "start_AV") for K in KS), "上限と K = 0 の UAR が K で違う")
w("")
w("### 記述の版：設定を疑似ラベルごとに選んだ点（設計 3.1 節．記述だけに使う）")
w("")
SELP = eget("stud_prop_sel", lambda c: "stud_prod" if c == CD else "stud_prod_sel")
SAME_P = [k_ for k_ in ALL15 if GS[k_] == GM[k_]]
SAME_D = [k_ for k_ in ALL12 if GPS[k_] == GM[k_]]
must(all((PH[(c, K, "stud_prop_sel")].hexdigest() == PH[(c, K, "stud_prop")].hexdigest()) == ((c, K) in SAME_P) for c, K in ALL15)
     and all((PH[(c, K, "stud_prod_sel")].hexdigest() == PH[(c, K, "stud_prod")].hexdigest()) == ((c, K) in SAME_D) for c, K in ALL12), "記述の版と主の点が同じ判定になる条件")
BOTH_SAME = [k_ for k_ in SAME_P if k_ in SAME_D or k_[0] == CD]
must(all(abs(SELP(*k_).est - S_PP(*k_).est) < 1e-12 for k_ in BOTH_SAME), "両側が主の点と同じ条件の差")
DQ_SEL2 = quote(design, "AFEW の K = 25 と MELD の K = 5 では，実験 18 が学び直しの点を選んだときの疑似ラベル（まとめ方の格子の点 2）と，この実験の提案の疑似ラベル（実験 20 の固定の点．AFEW の K = 25 は点 6，MELD の K = 5 は点 7）が違う．"
                        "この 2 つの条件の `stud_prop_sel` は，別の疑似ラベルのために選ばれた (ρ, a) で学ぶ．", "設計 3.1 節")
DQ_GRID12 = quote(design, "12 点すべての UAR（`student_grid.csv`．区間は付けない．テストで最良の点の値は 12 個の最大なので上に偏る）．", "設計 3.1 節")
w(f"11. **実験 18 の規則で疑似ラベルごとに選んだ点どうしの差（元の 4 コーパスは `stud_prop_sel − stud_prod_sel`，CREMA-D は `stud_prop_sel − stud_prod`）**：{read_short(SELP, ALL15)}"
  "この差は記述だけに使い，主張に使わない．提案の手法の疑似ラベルの点の選択にだけ，評価するコーパスのテストのラベルが間接に入るからである（0 節の設計 3.1 節の文．提案の手法に有利な向き）．CREMA-D の積の疑似ラベルの点は，主の点と同じ選び方の点である．"
  f"提案の手法の疑似ラベルの点が主の点と同じ条件は {by_corpus(SAME_P)}，積の疑似ラベルの点が主の点と同じ条件は {by_corpus(SAME_D)} で，これらの条件では，その版は主の点の版と同じ判定である．"
  f"両側とも主の点と同じ条件（{by_corpus(BOTH_SAME)}）の値は，要点 1 の値と同じである．設計 3.1 節：「{DQ_SEL2}」"
  f"定義の検査（`define.md`）：「{RQ['D2']}」ML の査読想定（`review_ml.md`）：「{RQ['RM10a']}」選んだ点の一覧は 2.6 節にある．")
G12 = {}
for c, K in ALL15:
    for v in ("stud_prod", "stud_prop"):
        xs = {g: SG[(c, K, v, g)].uar_mean for g in range(12)}
        G12[(c, K, v)] = dict(main=xs[GM[(c, K)]], best=max(xs.values()), arg=max(xs, key=xs.get), lo=min(xs.values()))
GAP = {v: [G12[(c, K, v)]["best"] - G12[(c, K, v)]["main"] for c, K in ALL15] for v in ("stud_prod", "stud_prop")}
w(f"   - **設定の 12 点の UAR**（`student_grid.csv`．50 単位の平均．設計 3.1 節：「{DQ_GRID12}」）：12 点の UAR の最大 − 主の点の UAR（点の値の引き算．区間は無い）は，15 条件で，積の疑似ラベルが {fpt(min(GAP['stud_prod']))}〜{fpt(max(GAP['stud_prod']))} 点，"
  f"提案の手法の疑似ラベルが {fpt(min(GAP['stud_prop']))}〜{fpt(max(GAP['stud_prop']))} 点である．最大は，テストの上で 12 個から選んだ値で，実際には使えない．"
  f"統計の検査（`stats.md`）：「{RQ['S6']}」比較の検査（`compare.md`）：「{RQ['C10']}」この文書は，疑似ラベルごとの最大の点どうしの差を書かない．条件ごとの値は 2.6 節にある．")
w("")
w("### 学び直しでの 3 つの量の並び（探索の記述．設計の文書に無い並べ方）")
w("")
NOTE = ROOT / "DAI" / "docs" / "論文のセッションへの伝言_2026-11-14.md"
NQ = quote(NOTE.read_text(), "**3 条件の当てはめ**：疑似ラベルが当たるようになった幅（提案の答え − 積）と，学び直した融合の分類器の上がり幅の関係．学び直した分類器が疑似ラベルをどれだけ再現したか．実験 21 の出力から集計する．", "論文のセッションへの伝言")
NQ1 = quote(NOTE.read_text(), "(b) 8 節の 3 条件を，学び直しに当てる．", "論文のセッションへの伝言")
NOTE_FIRST = git("log", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M:%S", "--", "DAI/docs/論文のセッションへの伝言_2026-11-14.md").strip().split("\n")[-1].split(" ", 1)
NOTE_V1 = git("show", f"{NOTE_FIRST[0]}:DAI/docs/論文のセッションへの伝言_2026-11-14.md")
must(NQ in NOTE_V1 and NQ1 in NOTE_V1 and NOTE_FIRST[1] < "2026-11-15 03:20:00", "伝言の最初の版に，引いた文が無い（文を直す）")
must(git("status", "--porcelain", "--", "DAI/docs/論文のセッションへの伝言_2026-11-14.md").strip() == "", "伝言に，コミットしていない変更がある")


def tq(c, K, t_, k_):
    return float(TQ[(c, K, t_)][k_].mean())


DT = {k_: tq(*k_, "prop", "t") - tq(*k_, "prod", "t") for k_ in ALL15}
must(all(abs((TQ[(c, K, "prop")]["t"] - TQ[(c, K, "prod")]["t"]).mean() - DT[(c, K)]) < 1e-9 for c, K in ALL15), "疑似ラベルの UAR の差")
SIGN_SAME = [k_ for k_ in ALL15 if (DT[k_] > 0) == (S_PP(*k_).est > 0)]
NEG_T = [k_ for k_ in ALL15 if DT[k_] < 0]
MIM = [tq(*k_, t_, "m") for k_ in ALL15 for t_ in ("prod", "prop")]
MIM_D = [tq(*k_, "prop", "m") - tq(*k_, "prod", "m") for k_ in ALL15]
BOTH_POS = [k_ for k_ in ALL15 if DT[k_] > 0 and S_PP(*k_).est > 0]
BOTH_NEG = [k_ for k_ in ALL15 if DT[k_] < 0 and S_PP(*k_).est < 0]
must(len(SIGN_SAME) == 15 and len(NEG_T) >= 1 and len(BOTH_POS) + len(BOTH_NEG) == 15 and BOTH_NEG == NEG_T, "要点 12 の符号の並び（文を直す）")
must("def fit_students(X, Xt, W0, b0, cw, cb, gk, n: int, own, teachers: dict, specs: list, label: str):" in src21 and 'fit_students(u.Xp["AVT"], u.Xt["AVT"]' in src21
     and "fit_students(u.Xp[u.fus], u.Xt[u.fus]" in src21 and "uar_pool=100 * S.uar(yp[nk], pp[nk])" in src21, "台本の学び直しの入力と，候補の上で測る行")
w("12. **この項は，設計の文書に無い並べ方で，結果を開いた後に並べた探索の記述である．仕組みについての主張には使わない**．"
  f"研究者が 2026-11-14 の夜に決めたことの記録（`DAI/docs/論文のセッションへの伝言_2026-11-14.md` の 1 節の 3）に「{NQ1}」とあり，同じ文書の 10.4 節の 2 に「{NQ}」とある．"
  f"どちらの文も，伝言の最初の版（コミット {NOTE_FIRST[0]}，{NOTE_FIRST[1][5:16]}．計算の起動より前）からある．「3 条件」の当てはめそのものは，この文書ではしていない．並べたのは，その材料になる 3 つの量である．量の定義と並べ方は，結果を開いた後に，この台本で決めた．"
  "並べる量は 3 つである．(a) **疑似ラベルの UAR**（単位の csv の `teacher_uar`）：疑似ラベルの判定（確率が最大の感情）の UAR を，K 本を除いた候補の発話の上で，候補のラベルで測った値．候補のラベルは，疑似ラベルを作るのにも学び直しにも使っていない（測るためだけに使う）．"
  "(b) **学び直した融合の分類器の UAR**（`uar`）：テストの発話の上の UAR．(c) **疑似ラベルと判定が一致した割合**（`mimic`）：K 本を除いた候補の発話のうち，学び直した融合の分類器の判定が，疑似ラベルの判定と同じだった発話の割合．"
  "列の定義は，台本の該当の行を読み，判定の配列から数え直した値が単位の csv の値と一致することで確かめた（0 節）．(a) と (c) は候補の発話の上の量で，(b) はテストの発話の上の量である．"
  "**候補の発話（K 本を除く）は，学び直した融合の分類器が疑似ラベルで学んだ入力である．(a)・(c) と，2.4 節の候補の上の UAR は，学習データの上の量で，(b) だけが学習に使っていない発話の上の量である**（台本の `fit_students` は候補の全部の発話を学習に使い，`save_students` は同じ発話の上で測る）．"
  "値は 50 単位の平均で，(a) の差は平均どうしの引き算である（区間は無い．丸める前の値から引いた）．"
  f"定義の検査（`define.md`）：「{RQ['D6a']}」")
w("")
w("   | コーパス | K | (a) 疑似ラベルの UAR：積 | (a) 提案の手法 | (a) の差（提案の手法 − 積） | (b) の差（`stud_prop − stud_prod`．要点 1 と同じ値） | (c) 判定が一致した割合：積 | (c) 提案の手法 |")
w("   |---|---|---|---|---|---|---|---|")
for c, K in ALL15:
    w(f"   | {c} | {K} | {tq(c, K, 'prod', 't'):.2f} | {tq(c, K, 'prop', 't'):.2f} | {fpt(DT[(c, K)])} | {ci(S_PP(c, K))} | {tq(c, K, 'prod', 'm'):.3f} | {tq(c, K, 'prop', 'm'):.3f} |")
w("")
w(f"   - (a) の差は 15 条件で {fpt(min(DT.values()))}〜{fpt(max(DT.values()))} 点，(b) の差の点推定は {rng_of(S_PP, ALL15)} 点である．(a) の差が負の条件は {'，'.join(f'{c} の K = {K}（{fpt(DT[(c, K)])}）' for c, K in NEG_T)}で，"
  f"この条件は (b) の差の点推定も負である（要点 1）．(a) の差と (b) の差の点推定は，{len(BOTH_POS)} 条件でどちらも正，{len(BOTH_NEG)} 条件（{by_corpus(BOTH_NEG)}）でどちらも負である．"
  "どちらの差も正の条件が多いので，符号が同じ条件の数は，2 つの差の結びつきの強さを表さない．15 条件は 5 コーパス × 同じテストを共有する 3 つの K で，独立な 15 個の測定ではない．この数えを，検定の結果として読まない．")
w(f"   - (c) は 15 条件 × 2 つの疑似ラベルで {min(MIM):.3f}〜{max(MIM):.3f} で，同じ条件の中の提案の手法 − 積の差は " + f"{min(MIM_D):+.4f}〜{max(MIM_D):+.4f}".replace("-", "−") + " である．(c) は，学び直した融合の分類器が学習に使った発話の上で，学習に使った疑似ラベルとの一致を測った量である．")
w("   - 疑似ラベルを替えた学び直し 5 つを含む 7 通りの疑似ラベルの同じ 3 つの量と，学び直した融合の分類器の候補の上の UAR は，2.4 節に並べた．(a) と (b) の差の大きさの関係を説明する量（学び直した分類器の切片の分と向きの分など）は，まだ出していない（4 節）．")
w("")

# ---- 要点 13〜18：CREMA-D の感度の表（計算の後の解析）


def fend2(x: float) -> str:
    """区間の端：小数 2 桁（区間どうしを並べる表で使う）．0.00 に丸まるときは符号が読めるよう桁を足す．"""
    s = f"{x:+.2f}"
    if float(s) == 0.0 and x != 0:
        for nd in (3, 4, 5, 6, 7, 8):
            s = f"{x:+.{nd}f}"
            if float(s) != 0.0:
                break
    return s.replace("-", "−")


def ci2(r, e: str = "est", lo: str = "lo", hi: str = "hi") -> str:
    """差と区間（端は小数 2 桁）．両側が同じ判定の行は「0（同じ判定）」．"""
    if bool(r.same_pred):
        return "0（同じ判定）"
    s_ = f"{fest(getattr(r, e))} [{fend2(getattr(r, lo))}, {fend2(getattr(r, hi))}]"
    cols_ = {("est", "lo", "hi"): ("se", "method"), ("d_est", "d_lo", "d_hi"): ("d_se", "d_method"), ("est", "lo_speaker", "hi_speaker"): ("se_speaker", "method_speaker")}.get((e, lo, hi))
    if cols_ and near0(r, lo, hi, *cols_):                                   # 引き直しで作る区間の端が 0 に近いとき（文を塊にした区間は jackknife なので付かない）
        NEAR_USED.append(s_)
        s_ += "＊"
    return s_


def kc(r, e: str = "est", lo: str = "lo", hi: str = "hi") -> str:
    """読みの区分（klass と同じ分け方．列の名前を選べる）．"""
    if bool(r.same_pred):
        return "same"
    x, a_, b_ = getattr(r, e), getattr(r, lo), getattr(r, hi)
    return ("pp" if x >= FLOOR else "p") if a_ > 0 else (("nn" if x <= -FLOOR else "n") if b_ < 0 else "z")


def pc(a: str, b: str) -> str:
    return f"`{a} − {b}`"


PN_FIX = {"tempb": "K 本で合わせたモダリティごとの重み（切片も合わせる）"}


def pn1(v: str) -> str:
    base = v.split("@")[0]
    must(base in NAME, f"版の呼び名が無い：{v}")
    return PN_FIX.get(base, NAME[base]) + ("［切片をずらす前］" if v.endswith("@raw") else "")


def pn(a: str, b: str) -> str:
    return f"{pn1(a)} − {pn1(b)}（{pc(a, b)}）"


def klist(items) -> str:
    """（a, b, K）の一覧を「`a − b` の K = 5・10」の形で書く．"""
    out, seen = [], []
    for a, b, _ in items:
        if (a, b) not in seen:
            seen.append((a, b))
            out.append(f"{pc(a, b)} の K = {kjoin(K for a2, b2, K in items if (a2, b2) == (a, b))}")
    return "，".join(out)


ALLP = [(a, b, K) for a, b in PC for K in KS]                               # 集計の段の 78 対 × 3 つの K
NEA = {x: NE[x + ("all",)] for x in ALLP}                                   # 文を除いたテストの差（K 本の引き方 0〜9）
SCA = {x: SCE[x + ("all",)] for x in ALLP}                                  # 文を塊にした区間
NSAME = [x for x in ALLP if bool(NEA[x].same_pred)]
must(NSAME == [("prop", "prop_n0zero", K) for K in KS] and all(bool(NEA[x].same_pred) == bool(SCA[x].same_pred) == bool(q(x[0], x[1], CD, x[2]).same_pred) for x in ALLP), f"両側が同じ判定の差（文を直す）：{NSAME}")
LIVE = [x for x in ALLP if x not in NSAME]
must(len(ALLP) == 234 and len(LIVE) == 231 and not any(NEA[x].a_same_as != "" and NEA[x].a_same_as == NEA[x].b_same_as for x in LIVE)
     and not any(NEA[x].lo == 0 or NEA[x].hi == 0 or NEA[x].d_lo == 0 or NEA[x].d_hi == 0 or SCA[x].lo == 0 or SCA[x].hi == 0 for x in LIVE), "感度の表の行の数と，端がちょうど 0 の区間")
KEY8 = [("prop", "prod"), ("stud_prop", "stud_prod"), ("stud_prop", "prop"), ("prop", "none_AV"), ("prop", "stud_prod"), ("stud_prod", "none_AV"), ("stud_prop", "none_AV"), ("stud_prod", "prod")]   # 要点 1・6・7・10 の差
KEY11 = [("prop", v) for v in OPP]                                           # 要点 8 の差
KEY9 = [("prop", "prop_noprof"), ("prop_noprof", "prod"), ("prop", "prop_levelfree"), ("prop_levelfree", "prod"), ("prop", "prop_noint"), ("prop_noint", "prop_noprof"), ("prop", "prop_home"),
        ("prop", "prop_noprof_sel"), ("prop", "prop_n0zero"), ("M_F6_orc", "prod"), ("M_F6_ext", "prod"), ("prop", "M_F6_ext")]                                                       # 要点 9 の差
KEYRAW = [("prop@raw", "prod@raw"), ("prop", "prop@raw")] + [("prop@raw", f"{v}@raw") for v in OPP] + [("prop", f"{v}@raw") for v in OPP]      # 要点 7・8 の下位の項の差（切片をずらす前の版との差）
KEYOPP = [(v, "prod") for v in OPP]                                           # 2.2 節の「相手 − 積」
REST = [x for x in PC if x not in KEY8 + KEY11 + KEY9 + KEYRAW + KEYOPP]
must(all(x in PC for x in KEY8 + KEY11 + KEY9 + KEYRAW + KEYOPP) and len(set(KEY8 + KEY11 + KEY9 + KEYRAW + KEYOPP)) == 8 + 11 + 12 + 24 + 11 and len(REST) == 12, f"2.7 節の差の並べ方：{REST}")
KEY19 = [(a, b, K) for a, b in KEY8 + KEY11 for K in KS]                     # 要点 1・6・7・8・10 の本文が読みを書いた差（19 対 × 3 つの K）
D_NZ = [x for x in LIVE if NEA[x].d_lo > 0 or NEA[x].d_hi < 0]               # 「除いた値 − 全発話の値」の区間が 0 を含まない
D_NZ19 = [x for x in D_NZ if x in KEY19]
KCH = [x for x in LIVE if kc(NEA[x]) != kc(NEA[x], "est_all", "lo_all", "hi_all")]     # 読みの区分が全発話の値と違う
KCH19 = [x for x in KCH if x in KEY19]
D_RNG = [NEA[x].d_est for x in LIVE]
D_RNG19 = [NEA[x].d_est for x in KEY19]


def nz(k: str) -> bool:
    return k in ("pp", "p", "n", "nn")


KCH_TO_Z = [x for x in KCH if nz(kc(NEA[x], "est_all", "lo_all", "hi_all")) and kc(NEA[x]) == "z"]
KCH_TO_NZ = [x for x in KCH if kc(NEA[x], "est_all", "lo_all", "hi_all") == "z" and nz(kc(NEA[x]))]
KCH_FLOOR = [x for x in KCH if nz(kc(NEA[x], "est_all", "lo_all", "hi_all")) and nz(kc(NEA[x]))]
must(len(KCH_TO_Z) + len(KCH_TO_NZ) + len(KCH_FLOOR) == len(KCH) and all((NEA[x].est > 0) == (NEA[x].est_all > 0) for x in KCH_FLOOR), "読みの区分の違いの分け方（文を直す）")
N_PP, N_SS = [("prop", "prod", K) for K in KS], [("stud_prop", "stud_prod", K) for K in KS]
must(all(kc(NEA[x]) == "p" for x in N_PP + N_SS) and not any(x in D_NZ for x in N_PP + N_SS) and all(kc(SCA[x]) == "p" for x in N_PP + N_SS), "要点 13・15 の太字の読み（文を直す）")
must(sum(RM_E) == IEO_ALL and RM_E[0] == 91 and set(RM_E[1:].tolist()) == {273}, "除いた発話の数（文を直す）")
must(PROBE3 == ci2(NEA[("prop", "prod", 5)]), "0 節で探した値の文字が，要点 13 の書き方と違う")
# 引き直しで作る区間の端が 0 に近い区間（照合の第 3 回の指摘 2）．区間は 点推定 ± 1.96 × 標準誤差
Z95 = 1.96
must(max(max(abs(NEA[x].d_lo - (NEA[x].d_est - Z95 * NEA[x].d_se)), abs(NEA[x].d_hi - (NEA[x].d_est + Z95 * NEA[x].d_se)), abs(SCA[x].lo_speaker - (SCA[x].est - Z95 * SCA[x].se_speaker)),
             abs(SCA[x].hi_speaker - (SCA[x].est + Z95 * SCA[x].se_speaker))) for x in LIVE) < 1e-9, "引き直しで作る区間が，点推定 ± 1.96 × 標準誤差でない（文を直す）")


def near_d(x) -> float:
    """「除いた値 − 全発話の値」の区間の 0 に近いほうの端が，0 から標準誤差の何倍離れているか．"""
    return min(abs(NEA[x].d_lo), abs(NEA[x].d_hi)) / NEA[x].d_se


def near_s(x) -> float:
    """話者を塊にした区間（全発話）の 0 に近いほうの端が，0 から標準誤差の何倍離れているか．"""
    return min(abs(SCA[x].lo_speaker), abs(SCA[x].hi_speaker)) / SCA[x].se_speaker


NEAR_D = [x for x in LIVE if near_d(x) < NEAR_TH]
NEAR_S = [x for x in LIVE if near_s(x) < NEAR_TH]
NEAR_N = [x for x in LIVE if near0(NEA[x])]                                   # 文を除いたテストの区間の端が 0 に近いもの
must(NEAR_D == [x for x in LIVE if near0(NEA[x], "d_lo", "d_hi", "d_se", "d_method")] and NEAR_S == [x for x in LIVE if near0(SCA[x], "lo_speaker", "hi_speaker", "se_speaker", "method_speaker")]
     and NEAR_S == [x for x in LIVE if near0(q(x[0], x[1], CD, x[2]))] and len(NEAR_S) == n_boot(CD, True, True), "端が 0 に近い区間の 2 つの数え方が合わない")
KCH_NEAR = [x for x in KCH if x in NEAR_S or x in NEAR_N]
must(NEAR_D == [("stud_prod", "prod", 5)] and NEAR_D[0] in D_NZ and NEA[NEAR_D[0]].d_hi < 0 and f"{NEA[NEAR_D[0]].d_hi:+.3f}" == "-0.003" and f"{NEA[NEAR_D[0]].d_lo:+.2f}" == "-0.33",
     f"「除いた値 − 全発話の値」の区間の端が 0 に近いもの（文を直す）：{NEAR_D}")
NPE = {(r.version, int(r.K), int(r.emotion)): r for r in npe.itertuples()}


def emo_rng(a: str, b: str, col: str) -> str:
    return "，".join(f"{EMO[e]} {fpt(min(getattr(NPE[(a, K, e)], col) - getattr(NPE[(b, K, e)], col) for K in KS))}〜{fpt(max(getattr(NPE[(a, K, e)], col) - getattr(NPE[(b, K, e)], col) for K in KS))}" for e in range(4))


w("### CREMA-D の感度の表（設計 4 節が予告した表．計算の後の解析．要点 1・6〜10 の CREMA-D の値への感度）")
w("")
w("**この節の値は，計算の後の解析（0 節）の出力 `results/dai21_eval5/posthoc/` の値である．第 3 版にあった値は，照合（関門 2）の第 3 回で出力の csv と一致した（判定は「直してから通す」）．"
  "第 4 版で足した値（要点 16 の感情ごとの再現率，要点 14・15 の端が 0 に近い区間の数え）と直した文は，照合の第 4 回で結果ファイルから作り直した値と一致した（判定は「通す」）**．"
  "照合の第 4 回の後に足したのは，要点 16 の上限の声の分類器についての文である（再現率の値は第 4 版と同じである．候補の発話の中の文 `IEO` の Neutral の割合は，この仕上げで単位の npz から数えた）．"
  "「文 `IEO`」は，文「It's eleven o'clock.」を指す（発話の名前の記号）．「全発話の値」は，要点 1〜12 と同じ値である（テストの全発話の上の値．区間は話者を塊にした区間）．"
  "区間は 95% で，補正していない．この節と 2.7 節は区間どうしを並べるので，区間の端を小数 2 桁で書く（5 節）．"
  "設計が計算の前に予告したのは表の種類で，自由度，内訳の版，並べ方は，結果を開いた後に決まった（0 節）．この節の値も，要点 1〜12 と同じく探索の結果として扱う．CREMA-D は，元の 4 コーパスと並ぶ独立な確かめとして数えない．")
w("")
w(f"13. **CREMA-D のテストから文 `IEO` の発話（{IEO_ALL} 本）を除いても，推定時の答えの提案の手法 − 積（`prop − prod`）と，学び直した融合の提案の手法の疑似ラベル − 積の疑似ラベル（`stud_prop − stud_prod`）は，"
  "3 つの K とも区間が 0 を含まず 1.0 点未満だった**（文を除いたテストの値．`prop − prod`：" + "，".join(f"K = {K} で {ci2(NEA[('prop', 'prod', K)])}" for K in KS)
  + "．`stud_prop − stud_prod`：" + "，".join(f"K = {K} で {ci2(NEA[('stud_prop', 'stud_prod', K)])}" for K in KS) + "）．"
  "全発話の値は，`prop − prod` が " + "，".join(f"K = {K} で {ci2(q('prop', 'prod', CD, K))}" for K in KS) + "，`stud_prop − stud_prod` が " + "，".join(f"K = {K} で {ci2(q('stud_prop', 'stud_prod', CD, K))}" for K in KS)
  + " である（要点 7・1 と同じ値を，区間の端を小数 2 桁にして書いた）．"
  "**「除いた値 − 全発話の値」の区間は，この 6 個のどれも 0 を含む**（`prop − prod`：" + "，".join(f"K = {K} で {ci2(NEA[('prop', 'prod', K)], 'd_est', 'd_lo', 'd_hi')}" for K in KS)
  + "．`stud_prop − stud_prod`：" + "，".join(f"K = {K} で {ci2(NEA[('stud_prop', 'stud_prod', K)], 'd_est', 'd_lo', 'd_hi')}" for K in KS) + "）．"
  "この区間は，同じ話者の引き直しの上で 2 つの値の引き算を作った，対応のある区間である．**区間が 0 を含むことは，文を除いても差が変わらないことを意味しない**．"
  "この区間からは，文を除くと差が変わるという証拠が得られなかった，という読みに留める．"
  f"台本の検査（`code_post_tools_v2.md`）は，読み方を「{G1}」「{G1B}」と書いている．"
  f"除いた発話は Neutral {RM_E[0]} 本，Happy {RM_E[1]} 本，Sad {RM_E[2]} 本，Angry {RM_E[3]} 本で，"
  f"残ったテストは Neutral {KEEP_E[0]:,} 本，Happy {KEEP_E[1]:,} 本，Sad {KEEP_E[2]:,} 本，Angry {KEEP_E[3]:,} 本の計 {int(KEEP_E.sum()):,} 本である（K 本の引き方 1 つぶん．5 分割の合計．この台本が単位の npz の発話の名前で数えた）．"
  "**この表が測るのは，テストの側で，値が文 `IEO` の発話に頼っているかどうかだけである．K 本と候補に文 `IEO` が入ることの影響（学習の側）は測っていない**（要点 17）．"
  "文を除いたテストの区間は，全発話の区間と同じ作り方（話者の重みの引き直しと，K 本の引き方の項）である．"
  "この項と要点 14 は，差についての読みである．版ごとの UAR そのものの「除いた値 − 全発話の値」は，2.7 節に並べた．")
w(f"   - **要点 1・6・7・8・10 の本文が読みを書いた CREMA-D の差**（{len(KEY8) + len(KEY11)} 対 × 同じテストを共有する 3 つの K ＝ {len(KEY19)} 個．2.7 節の差の表の (i)・(ii)．"
  "要点 7・8 の下位の項にある，切片をずらす前の版との差は，次の項に分けた）：「除いた値 − 全発話の値」の区間が 0 を含まないのは " + f"{len(D_NZ19)} 個"
  + ("（" + "，".join(f"{pn(a, b)}の K = {K} で {ci2(NEA[(a, b, K)], 'd_est', 'd_lo', 'd_hi')}．全発話 {ci2(q(a, b, CD, K))}，文を除いたテスト {ci2(NEA[(a, b, K)])}" for a, b, K in D_NZ19) + "．補正していない）" if D_NZ19 else "（補正していない）")
  + f"で，残る {len(KEY19) - len(D_NZ19)} 個は 0 を含む．"
  + ("この 1 個の区間の上端（−0.003）は 0 に近く，引き直しの乱数を替えると 0 を含みうる（要点 14）．" if D_NZ19 == NEAR_D else "")
  + f"点推定の変化（除いた値 − 全発話の値）は {fpt(min(D_RNG19))}〜{fpt(max(D_RNG19))} 点である．"
  f"読みの区分（区間が 0 を含むか．差が 1.0 点以上か）が全発話の値と違うのは {len(KCH19)} 個である："
  + "，".join(f"{pn(a, b)}の K = {K}（全発話 {ci2(q(a, b, CD, K))}，文を除いたテスト {ci2(NEA[(a, b, K)])}，除いた値 − 全発話の値 {ci2(NEA[(a, b, K)], 'd_est', 'd_lo', 'd_hi')}）" for a, b, K in KCH19)
  + "．区分の違いは，差が変わったことの証拠ではない（この " + str(len(KCH19)) + " 個の「除いた値 − 全発話の値」の区間は"
  + ("どれも 0 を含む" if not any(x in D_NZ for x in KCH19) else "，上に挙げたものを除いて 0 を含む") + "）．文を除いたテストは，発話が " + str(IEO_ALL) + " 本少ない．")


def dgrp(pairs: list) -> str:
    """差の組について，「除いた値 − 全発話の値」の区間が 0 を含まない数と，点推定の変化の範囲を書く．"""
    xs = [(a, b, K) for a, b in pairs for K in KS if (a, b, K) in LIVE]
    hit = [x for x in xs if x in D_NZ]
    same_ = 3 * len(pairs) - len(xs)
    return (f"{len(pairs)} 対 × 同じテストを共有する 3 つの K ＝ {3 * len(pairs)} 個" + (f"（うち両側が同じ判定の {same_} 個を除く {len(xs)} 個）" if same_ else "") + "のうち，「除いた値 − 全発話の値」の区間が 0 を含まないものは"
            + (f" {len(hit)} 個（" + "，".join(f"{pn(a, b)}の K = {K} で {ci2(NEA[(a, b, K)], 'd_est', 'd_lo', 'd_hi')}．全発話 {ci2(q(a, b, CD, K))}，文を除いたテスト {ci2(NEA[(a, b, K)])}" for a, b, K in hit) + "．補正していない）で，残りは 0 を含む"
               if hit else "無い（補正していない）")
            + f"．点推定の変化は {fpt(min(NEA[x].d_est for x in xs))}〜{fpt(max(NEA[x].d_est for x in xs))} 点である．")


must(sum(len([x for x in D_NZ if (x[0], x[1]) in g_]) for g_ in (KEY8 + KEY11, KEYRAW, KEY9, KEYOPP, REST)) == len(D_NZ), "「除いた値 − 全発話の値」の区間が 0 を含まない差の組分け")
w("   - **要点 7・8 の下位の項が読みを書いた，切片をずらす前の版との差**（2.7 節の差の表の (iv)）：" + dgrp(KEYRAW))
w("   - **要点 9 の差**（部品を 1 つ外した版と，参照・参考．2.7 節の差の表の (iii)）：" + dgrp(KEY9) + "**2.2 節の「相手 − 積」**（同 (v)）：" + dgrp(KEYOPP) + "**集計の段の残りの差**（同 (vi)）：" + dgrp(REST))
w("   - **感情ごとの再現率**（文を除いたテスト．2.7 節．点の値の引き算で，感情ごとの差の区間は作っていない）：提案の手法 − 積（推定時の答え）の 3 つの K の値の幅は，" + emo_rng("prop", "prod", "recall")
  + "（全発話では " + emo_rng("prop", "prod", "recall_all") + "）．学び直した融合の提案の手法の疑似ラベル − 積の疑似ラベルは，" + emo_rng("stud_prop", "stud_prod", "recall")
  + "（全発話では " + emo_rng("stud_prop", "stud_prod", "recall_all") + "）である．")
w(f"14. **集計の段の差の全部（78 対 × 同じテストを共有する 3 つの K ＝ {len(ALLP)} 個）のうち，両側が同じ判定の {len(NSAME)} 個を除く {len(LIVE)} 個で，「除いた値 − 全発話の値」の区間が 0 を含まないのは {len(D_NZ)} 個だった（補正していない）**："
  + "，".join(f"{pn(a, b)}の K = {K} で {ci2(NEA[(a, b, K)], 'd_est', 'd_lo', 'd_hi')}（全発話 {ci2(q(a, b, CD, K))}，文を除いたテスト {ci2(NEA[(a, b, K)])}）" for a, b, K in D_NZ) + "．"
  f"**この {len(D_NZ)} 個のうち，{pc(*NEAR_D[0][:2])} の K = {NEAR_D[0][2]} は，区間の上端（−0.003）が 0 から標準誤差の {near_d(NEAR_D[0]):.2f} 倍しか離れていない．"
  f"この区間は話者の重みの引き直し（{phv['consts']['N_BOOT']:,} 組）で作る区間で，照合の第 3 回の検算は「{C3_FLIP}」と書いている．「0 を含まない {len(D_NZ)} 個」のうちこの 1 個は，引き直しの乱数の組に依る**．"
  f"同じ報告は「{C3_KEEP}」と書いている．"
  f"この文書は，区間の 0 に近いほうの端が，0 から標準誤差の {NEAR_TH} 倍以内にある区間を「端が 0 に近い区間」と数える．"
  f"この基準は第 4 版で決めた（照合の第 3 回の作り直しでは，「{C3_SE}」標準誤差が 4.8% 違うと，区間の端は標準誤差の約 0.09 倍動く）．"
  f"{len(LIVE)} 個の「除いた値 − 全発話の値」の区間のうち，端が 0 に近い区間は，上の {len(NEAR_D)} 個だけである（0 を含む {len(LIVE) - len(D_NZ)} 個の中には無い）．"
  f"{len(LIVE)} 個の区間は，同じテストと同じ単位の上の値で，独立ではない．補正していないので，0 を含まない区間は偶然にも出うる．この数を検定の結果として読まない．"
  f"点推定の変化（除いた値 − 全発話の値）は，{len(LIVE)} 個で {fpt(min(D_RNG))}〜{fpt(max(D_RNG))} 点である．"
  f"読みの区分が全発話の値と違うのは {len(KCH)} 個である．全発話では区間が 0 を含まず，文を除いたテストでは 0 を含むものが {len(KCH_TO_Z)} 個（{klist(KCH_TO_Z)}），"
  f"全発話では 0 を含み，文を除いたテストでは 0 を含まないものが {len(KCH_TO_NZ)} 個（{klist(KCH_TO_NZ)}），"
  f"どちらも 0 を含まず，1.0 点の境目のどちら側かだけが違うものが {len(KCH_FLOOR)} 個（{klist(KCH_FLOOR)}）である．"
  + (f"この {len(KCH)} 個の「除いた値 − 全発話の値」の区間は，どれも 0 を含む．" if not any(x in D_NZ for x in KCH) else f"この {len(KCH)} 個のうち，「除いた値 − 全発話の値」の区間が 0 を含まないのは {sum(x in D_NZ for x in KCH)} 個である．")
  + f"文を除いたテストの {len(LIVE)} 個の区間（話者の重みの引き直しで作る）のうち，端が 0 に近い区間は {len(NEAR_N)} 個である（2.7 節の表の ＊）．"
  + f"読みの区分が全発話の値と違う {len(KCH)} 個のうち，全発話の区間か，文を除いたテストの区間の端が 0 に近いものは {len(KCH_NEAR)} 個（{klist(KCH_NEAR)}）で，この {len(KCH_NEAR)} 個の区分の違いは，引き直しの乱数にも依りうる．"
  + "区分の違いは，差が変わったことの証拠ではない．値は 2.7 節の表にある．"
  f"両側が同じ判定の {len(NSAME)} 個（{klist(NSAME)}）は，文を除いても差は 0 である．")
SP_NZ = [x for x in LIVE if nz(kc(SCA[x], "est", "lo_speaker", "hi_speaker"))]
SE_NZ = [x for x in LIVE if nz(kc(SCA[x]))]
SP_ONLY, SE_ONLY = [x for x in SP_NZ if x not in SE_NZ], [x for x in SE_NZ if x not in SP_NZ]
DF_CH = [x for x in LIVE if nz(kc(SCA[x])) != nz(kc(SCA[x], "est", "lo_n12", "hi_n12"))]
SE_RATIO = [SCA[x].se / SCA[x].se_speaker for x in LIVE]
must(any(x in KEY19 for x in SP_ONLY + SE_ONLY), "要点 15 の，読みを書いた差に入るもの（文を直す）")
NEAR_S_NZ, NEAR_S_Z = [x for x in NEAR_S if x in SP_NZ], [x for x in NEAR_S if x not in SP_NZ]
NEAR_S12 = [x for x in SP_ONLY + SE_ONLY if x in NEAR_S]
must(len(NEAR_S) > 0 and len(NEAR_S12) > 0 and len(NEAR_S_NZ) + len(NEAR_S_Z) == len(NEAR_S), "話者を塊にした区間の端が 0 に近いもの（文を直す）")
must(all(nz(kc(SCA[x], "est", "lo_speaker", "hi_speaker")) == nz(klass(q(x[0], x[1], CD, x[2]))) for x in LIVE), "話者を塊にした区間の読み")
w(f"15. **文を塊にした区間（12 文．自由度 {DF_SENT}）でも，`prop − prod` と `stud_prop − stud_prod` の区間は，3 つの K とも 0 を含まなかった**（全発話の値．`prop − prod`：" + "，".join(f"K = {K} で {ci2(SCA[('prop', 'prod', K)])}" for K in KS)
  + "．`stud_prop − stud_prod`：" + "，".join(f"K = {K} で {ci2(SCA[('stud_prop', 'stud_prod', K)])}" for K in KS) + "．点推定は要点 7・1 と同じで，区間だけが違う）．"
  f"文を塊にした区間は，「{G2}」（台本の検査 `code_post_tools_v2.md` が決めた言い方）．{G2B}"
  "作り方は，全発話のテストの上で，文を 1 つずつ全部の分割から除く jackknife の分散に，K 本の引き方の項を足す形である．"
  f"自由度は，jackknife で区間を作るコーパス（AFEW・IEMOCAP・MELD）と同じ規則（⌊塊の実効の数⌋ − 1）で，文の実効の数 {G_EFF_SENT:.2f} から {DF_SENT} とした．"
  f"文の実効の数が 12 より小さいのは，文 `IEO` の発話（{IEO_ALL} 本）がほかの文（{min(int(n_) for s_, n_ in N_SENT.items() if s_ != 'IEO')}〜{max(int(n_) for s_, n_ in N_SENT.items() if s_ != 'IEO')} 本）より多いためである．"
  "台帳は，自由度の決め方を，結果を開いた後に決めた点として書いている（0 節）．塊の数から決めた自由度 11 の区間（参考）は，csv の `lo_n12`・`hi_n12` にある．"
  f"**{len(LIVE)} 個の差（両側が同じ判定の {len(NSAME)} 個を除く）のうち，区間が 0 を含まないのは，話者を塊にした区間で {len(SP_NZ)} 個，文を塊にした区間で {len(SE_NZ)} 個である（補正していない）．"
  f"0 を含むかどうかが 2 つの区間で違うのは {len(SP_ONLY) + len(SE_ONLY)} 個で，話者を塊にした区間だけが 0 を含まないものが {len(SP_ONLY)} 個（{klist(SP_ONLY)}），"
  f"文を塊にした区間だけが 0 を含まないものが {len(SE_ONLY)} 個（{klist(SE_ONLY)}）である**．"
  f"この数は，同じテストと同じ単位の上の {len(LIVE)} 個の区間の数えで，検定の結果として読まない．"
  f"**話者を塊にした区間は引き直しで作るので，端が 0 に近い区間（要点 14 の基準：0 に近いほうの端が，0 から標準誤差の {NEAR_TH} 倍以内）は，0 を含むかどうかが引き直しの乱数の組に依りうる**．"
  f"{len(LIVE)} 個の話者を塊にした区間のうち，端が 0 に近い区間は {len(NEAR_S)} 個で，0 を含まないものが {len(NEAR_S_NZ)} 個（{klist(NEAR_S_NZ)}），0 を含むものが {len(NEAR_S_Z)} 個（{klist(NEAR_S_Z)}）である．"
  f"上の {len(SP_ONLY) + len(SE_ONLY)} 個のうち，話者を塊にした区間の端が 0 に近いものは {len(NEAR_S12)} 個（{klist(NEAR_S12)}）で，この {len(NEAR_S12)} 個が「2 つの区間で違う」に入るかどうかは，引き直しの乱数に依りうる．"
  f"「話者を塊にした区間で {len(SP_NZ)} 個」という数も，この {len(NEAR_S)} 個のぶんだけ乱数に依りうる．この台本は，乱数を替えた作り直しをしていない．"
  f"この {len(NEAR_S)} 個は全発話の値の区間で，要点 1〜12 と 2 節が読みを書いた区間と同じ区間である．要点 1〜12 と 2 節でこの区間の値を書いた箇所には，＊ の印を付けた（冒頭．読みは変えていない）．"
  "元の 4 コーパスでは，MOSEI の区間が引き直しで作る区間で，同じ基準に当たる区間は要点 1・5 に書いた．文を塊にした区間は乱数を使わないので，この揺れは無い．"
  f"0 を含むかどうかが 2 つの区間で違う {len(SP_ONLY) + len(SE_ONLY)} 個のうち，要点 1・6・7・8・10 の本文が読みを書いた {len(KEY19)} 個（要点 13 の初めの下位の項と同じ {len(KEY19)} 個）に入るのは，" + "，".join(f"{pn(a, b)}の K = {K}（話者を塊にした区間 {ci2(q(a, b, CD, K))}，文を塊にした区間 {ci2(SCA[(a, b, K)])}）" for a, b, K in SP_ONLY + SE_ONLY if (a, b, K) in KEY19)
  + "である．"
  f"標準誤差の比（文を塊にした区間 ÷ 話者を塊にした区間）は，{len(LIVE)} 個で {min(SE_RATIO):.2f}〜{max(SE_RATIO):.2f}（中央の値 {float(np.median(SE_RATIO)):.2f}）で，{sum(x < 1 for x in SE_RATIO)} 個で 1 未満である．"
  "2 つの区間は別の揺れを測っている．文を塊にした区間の幅を，話者を塊にした区間の幅の検算として読まない．塊は 12 個で，jackknife の分散の推定は粗い．"
  f"自由度 11 の区間（参考）に替えると，0 を含むかどうかが変わるのは {len(DF_CH)} 個である（{klist(DF_CH)}）．"
  "K 本の引き方で分けた行と，版ごとの UAR の文を塊にした区間は csv にある（UAR の一部は 2.7 節）．")
BYS = {(r.sentence, int(r.K), r.version): r for r in bys.itertuples()}
BYI = {(r.intensity, int(r.emotion), int(r.K), r.version): r for r in byi.itertuples()}
KIND0 = ["start_A", "start_V", "start_AV", "full_A", "full_V", "full_AV"]      # K に依らない判定
KDEP = ["none_A", "ret_A", "none_V", "ret_V", "none_AV", "tcal", "prod", "prop", "stud_prod", "stud_prop"]
must(sorted(KIND0 + KDEP) == sorted(BRK_V) and all(abs(float(BYS[(s_, 5, v)].uar_pooled) - float(BYS[(s_, K, v)].uar_pooled)) < 1e-12 for v in KIND0 for s_ in SENT12 for K in KS)
     and all(abs(float(BYI[(i_, e, 5, v)].recall_pooled) - float(BYI[(i_, e, K, v)].recall_pooled)) < 1e-12 for v in KIND0 for (i_, e, K2, v2) in BYI if K2 == 5 and v2 == v for K in KS), "内訳の版と，K に依らない版の値")


def s_ieo(v: str, K: int) -> float:
    return float(BYS[("IEO", K, v)].uar_pooled)


def s_oth(v: str, K: int) -> list:
    return [float(BYS[(s_, K, v)].uar_pooled) for s_ in SENT12 if s_ != "IEO"]


def s_rank(v: str, K: int) -> int:
    return 1 + sum(x > s_ieo(v, K) for x in s_oth(v, K))


def s_gap(v: str, K: int) -> float:
    return s_ieo(v, K) - float(np.mean(s_oth(v, K)))


def i_rec(v: str, K: int, i_: str, e: int) -> float:
    return float(BYI[(i_, e, K, v)].recall_pooled)


FUS6 = ["none_AV", "tcal", "prod", "prop", "stud_prod", "stud_prop"]
must(s_rank("start_A", 5) == 12 and s_rank("start_AV", 5) == 12 and all(s_gap(v, K) < 0 for v in ("none_A", "ret_A") for K in KS), "要点 16 の文ごとの並び（文を直す）")


def e_ieo(v: str, K: int, e: int) -> float:
    """文 IEO の，感情 e の再現率（50 単位の数を足してから割った値）．"""
    return float(getattr(BYS[("IEO", K, v)], f"recall_pooled_{e}"))


def e_oth(v: str, K: int, e: int) -> list:
    return [float(getattr(BYS[(s_, K, v)], f"recall_pooled_{e}")) for s_ in SENT12 if s_ != "IEO"]


def e_mean(v: str, K: int, e: int) -> float:
    """ほかの 11 文の，文ごとの再現率の平均．"""
    return float(np.mean(e_oth(v, K, e)))


def e_rank(v: str, K: int, e: int) -> int:
    return 1 + sum(x > e_ieo(v, K, e) for x in e_oth(v, K, e))


def e_gap(v: str, K: int, e: int) -> float:
    return e_ieo(v, K, e) - e_mean(v, K, e)


def e3_gap(v: str, K: int) -> float:
    """Happy・Sad・Angry の引き算の平均．"""
    return float(np.mean([e_gap(v, K, e) for e in (1, 2, 3)]))


VK_ALL = [(v, 5) for v in KIND0] + [(v, K) for v in KDEP for K in KS]
# UAR の引き算は，4 感情の再現率の引き算の平均である．文 IEO の Happy・Sad・Angry の値は，強さ LO・MD・HI の値の平均と同じ（強さごとの内訳との突き合わせ）
must(max(abs(s_gap(v, K) - (e_gap(v, K, 0) + 3 * e3_gap(v, K)) / 4) for v, K in VK_ALL) < 1e-9
     and max(abs(e_ieo(v, K, e) - float(np.mean([i_rec(v, K, i_, e) for i_ in ("LO", "MD", "HI")]))) for v, K in VK_ALL for e in (1, 2, 3)) < 1e-9, "感情ごとの再現率と，UAR・強さごとの内訳の突き合わせ")
NS0 = {s_: int(N_SENT_E[s_][0]) for s_ in SENT12}                            # 文ごとの Neutral の発話の数（K 本の引き方 1 つぶん）
NS3 = sorted({int(N_SENT_E[s_][e]) for s_ in SENT12 if s_ != "IEO" for e in (1, 2, 3)})
must(NS0["IEO"] == 91 and sorted({n_ for s_, n_ in NS0.items() if s_ != "IEO"}) == [90, 91] and NS3 == [89, 90, 91] and all(int(N_SENT_E["IEO"][e]) == 273 for e in (1, 2, 3))
     and all(int(getattr(BYS[(s_, 5, "start_A")], f"n_{e}")) == 10 * int(N_SENT_E[s_][e]) for s_ in SENT12 for e in range(4)), "文ごと・感情ごとの発話の数（文を直す）")
VOICE_K = [(v, K) for v in ("none_A", "ret_A") for K in KS]
FACE4 = ["start_V", "none_V", "ret_V", "full_V"]
FUS_VK = [(v, K) for v in FUS6 for K in KS]
FUS_LOW = [x for x in FUS_VK if e_gap(*x, 0) < 0]
FUS_HIGH = [x for x in FUS_VK if e_gap(*x, 0) >= 0]
must(all(e_gap(v, K, 0) < 0 for v, K in VOICE_K) and e_gap("start_A", 5, 0) > 0 and e_gap("full_A", 5, 0) < 0 and e_gap("full_AV", 5, 0) < 0 and e_gap("start_AV", 5, 0) > 0
     and all(e_gap(v, K, 0) > 0 for v in FACE4 for K in ([5] if v in KIND0 else KS)) and all(e_gap("prop", K, 0) < 0 and e_gap("prod", K, 0) < 0 for K in KS), "要点 16 の Neutral の再現率の並び（文を直す）")
must(e3_gap("start_A", 5) < 0 and all(abs(e_gap(v, K, 0)) > abs(e3_gap(v, K)) for v, K in VOICE_K), "要点 16 の UAR の引き算の中身（文を直す）")
E18_IEO = quote(design18, "CREMA-D では文「It's eleven o'clock.」が Neutral で 91 本，Happy・Sad・Angry で各 273 本読まれ，この文の発話のうち Neutral は 10.0% で，ほかの 11 文の 25.0% より低い．", "実験 18 の設計 8 節")
POOL_IEO_NEU = [100 * CD_FACTS["pool_ieo"][f][0] / CD_FACTS["pool_ieo"][f].sum() for f in range(5)]          # 候補の中の文 IEO の発話のうち Neutral の割合（分割ごと）
must(len(CD_FACTS["pool_ieo"]) == 5 and all(9.5 < x < 10.5 for x in POOL_IEO_NEU) and sum(int(CD_FACTS["pool_ieo"][f].sum()) for f in range(5)) == 4 * IEO_ALL
     and abs(100 * RM_E[0] / IEO_ALL - 10.0) < 1e-9 and e_gap("full_A", 5, 0) < 0 and e_rank("full_A", 5, 0) == 12, f"候補の中の文 IEO の Neutral の割合（文を直す）：{POOL_IEO_NEU}")
RA_CUE = quote(REP["review_affect"], "K = 5 の多くの単位では，ラベル付きの発話の中でこの文は Neutral 以外にしか現れないので，声の分類器と声顔の融合の分類器は「この文なら Neutral ではない」を学びうる．", "review_affect.md")
DQ_CUE = quote(design, "この文は Neutral の発話が少なく，K 本に入ると声の分類器の手がかりになりうる", "設計 4 節")


def k3(fn) -> str:
    return "・".join(f"{fn(K):.2f}" for K in KS)


def vk_names(xs) -> str:
    out = []
    for v in dict.fromkeys(v for v, _ in xs):
        out.append(f"`{v}` の K = {kjoin(K for v2, K in xs if v2 == v)}")
    return "，".join(out)


LO_MIN = {e: sum(i_rec(v, K, "LO", e) == min(i_rec(v, K, i_, e) for i_ in INT4) for v in BRK_V for K in ([5] if v in KIND0 else KS)) for e in (1, 2, 3)}
N_VK = len(KIND0) + 3 * len(KDEP)
G3_GAPS = [s_gap(v, K) for v in FUS6 for K in KS]
w(f"16. **文ごと・強さごとの内訳（記述．区間は無い．2.7 節）**．内訳の値は，50 単位の正解の数と発話の数を足してから割った再現率と，その 4 感情の平均である（csv の `recall_pooled`・`uar_pooled`）．"
  "要点 1〜15 の UAR（単位ごとに UAR を作ってから平均する）とは作りが違う．内訳に入れた版は，計算の後の解析の台本が決めた 16 版である（" + "，".join(f"{pn1(v)}（`{v}`）" for v in BRK_V) + "）．"
  "ソースの分類器の平均（K = 0）と上限は K に依らない判定である．ソースの分類器の平均（K = 0）は，CREMA-D のラベルを使っていない．")
w(f"   - **文ごと**：ソースの声の分類器の平均（K = 0．`start_A`）の値は，文 `IEO` が {s_ieo('start_A', 5):.2f} で，12 文の中で最も低い（ほかの 11 文は {min(s_oth('start_A', 5)):.2f}〜{max(s_oth('start_A', 5)):.2f}）．"
  f"ソースの顔の分類器の平均（`start_V`）は文 `IEO` が {s_ieo('start_V', 5):.2f}（ほかの 11 文は {min(s_oth('start_V', 5)):.2f}〜{max(s_oth('start_V', 5)):.2f}．12 文の中で高いほうから {s_rank('start_V', 5)} 番目），"
  f"ソースの融合の分類器の平均（`start_AV`）は {s_ieo('start_AV', 5):.2f}（ほかの 11 文は {min(s_oth('start_AV', 5)):.2f}〜{max(s_oth('start_AV', 5)):.2f}．12 文の中で最も低い）である．"
  "**`start_A` は CREMA-D のラベルを使っていないので，`start_A` で文 `IEO` の UAR が 12 文の中で最も低いことは，K 本から学んだものではない．"
  "K 本で学んだ版の並びのうち，どれだけが K 本から来たかは，この表からは分からない**．"
  f"文 `IEO` の値 − ほかの 11 文の値の平均（点の値の引き算）は，`start_A` が {fpt(s_gap('start_A', 5))} 点で，K 本で学んだ声の分類器では，K = 5・10・25 の順に，K 本だけの声（`none_A`）が "
  + "・".join(fpt(s_gap("none_A", K)) for K in KS) + " 点，微調整した声（`ret_A`）が " + "・".join(fpt(s_gap("ret_A", K)) for K in KS) + " 点である（12 文の中の順位は，高いほうから `none_A` が "
  + "・".join(str(s_rank("none_A", K)) for K in KS) + " 番目，`ret_A` が " + "・".join(str(s_rank("ret_A", K)) for K in KS) + " 番目）．"
  "UAR の引き算は，4 感情の再現率の引き算の平均である．UAR の引き算の大きさだけからは，感情ごとの中身は分からない．設計の懸念に直接当たる Neutral の再現率は，次の項に分けて書く．"
  f"融合と推定時の答えの 6 版（{'・'.join(f'`{v}`' for v in FUS6)}）では，UAR の引き算は 3 つの K で {fpt(min(G3_GAPS))}〜{fpt(max(G3_GAPS))} 点である．"
  f"文 `IEO` の発話は {IEO_ALL} 本，ほかの文は {min(int(n_) for s_, n_ in N_SENT.items() if s_ != 'IEO')}〜{max(int(n_) for s_, n_ in N_SENT.items() if s_ != 'IEO')} 本である（K 本の引き方 1 つぶん）．")
w(f"   - **文 `IEO` の Neutral の再現率（設計の懸念に直接当たる量．2.7 節の表．第 4 版で足した）**：設計 4 節の懸念は「{DQ_CUE}」である．"
  f"計算の前の感情認識の査読想定（`review_affect.md`）は「{RA_CUE}」と書いていた．照合の第 3 回（`claim_v3.md`）は「{C3_DIR}」と書いている．"
  "Neutral の発話は，どの文でも強さが XX である．このため，文 `IEO` とほかの 11 文の Neutral の再現率の比べには，強さの違いは入らない（Happy・Sad・Angry の比べには入る）．"
  f"Neutral の発話は，文 `IEO` が {NS0['IEO']} 本，ほかの文が 1 文に 90〜91 本である（K 本の引き方 1 つぶん）．「ほかの 11 文の平均」は，11 文の文ごとの再現率の平均である．"
  "**K 本で学んだ声の分類器では，3 つの K とも，文 `IEO` の Neutral の再現率が，ほかの 11 文の平均より低い**：K = 5・10・25 の順に，"
  + f"K 本だけの声（`none_A`）は文 `IEO` が {k3(lambda K: e_ieo('none_A', K, 0))}，ほかの 11 文の平均が {k3(lambda K: e_mean('none_A', K, 0))}（12 文の中の順位は，高いほうから {kjoin(e_rank('none_A', K, 0) for K in KS)} 番目），"
  + f"微調整した声（`ret_A`）は文 `IEO` が {k3(lambda K: e_ieo('ret_A', K, 0))}，ほかの 11 文の平均が {k3(lambda K: e_mean('ret_A', K, 0))}（順位は {kjoin(e_rank('ret_A', K, 0) for K in KS)} 番目）である．"
  + f"**CREMA-D のラベルを使っていない声の分類器（`start_A`）では，文 `IEO` の Neutral の再現率は {e_ieo('start_A', 5, 0):.2f} で，ほかの 11 文の平均 {e_mean('start_A', 5, 0):.2f} より低くない**"
  + f"（ほかの 11 文は {min(e_oth('start_A', 5, 0)):.2f}〜{max(e_oth('start_A', 5, 0)):.2f}．順位は {e_rank('start_A', 5, 0)} 番目）．"
  "**この並びは，設計の懸念が予想する向きである．区間は無く，K 本から文 `IEO` を除いて学び直してもいないので，この並びの原因は言えない**．"
  "**上限の声の分類器（`full_A`）は，ラベル付きの候補の発話の全ラベルで学ぶので，K 本の引き方の偏りを持たない．"
  f"それでも，文 `IEO` の Neutral の再現率は {e_ieo('full_A', 5, 0):.2f} で，ほかの 11 文の平均 {e_mean('full_A', 5, 0):.2f} より低い（順位は {e_rank('full_A', 5, 0)} 番目）**．"
  f"ただし，上限の版が学習に使う候補の発話にも，この文の Neutral が少ないというコーパスの偏りがある（実験 18 の設計 8 節：「{E18_IEO}」"
  "この台本が単位の npz で数えると，候補の発話の中の文 `IEO` の発話のうち Neutral は，"
  + (f"5 つの分割のどれでも {POOL_IEO_NEU[0]:.1f}%" if len({f"{x:.1f}" for x in POOL_IEO_NEU}) == 1 else f"5 つの分割で {min(POOL_IEO_NEU):.1f}〜{max(POOL_IEO_NEU):.1f}%") + " である）．"
  "**したがって，この表からは，K 本に入った文 `IEO` の発話から学んだ分，学習データの全体にあるこの偏りから学んだ分，この文の Neutral が声で当てにくい分を，分けられない**．"
  f"顔の分類器（{'・'.join(f'`{v}`' for v in FACE4)}）では，どの版・どの K でも，文 `IEO` の Neutral の再現率はほかの 11 文の平均より高い．"
  f"融合と推定時の答え（K に依る {len(FUS6)} 版 × 3 つの K ＝ {len(FUS_VK)} 個）では，文 `IEO` の Neutral の再現率がほかの 11 文の平均より低いものが {len(FUS_LOW)} 個で，"
  + (f"低くないものは {len(FUS_HIGH)} 個（{vk_names(FUS_HIGH)}）である．" if FUS_HIGH else "低くないものは無い．")
  + f"提案の手法（推定時の答え．`prop`）は文 `IEO` が {k3(lambda K: e_ieo('prop', K, 0))}，ほかの 11 文の平均が {k3(lambda K: e_mean('prop', K, 0))} で，積（`prod`）は {k3(lambda K: e_ieo('prod', K, 0))} と {k3(lambda K: e_mean('prod', K, 0))} である．"
  + f"ソースの融合の分類器の平均（`start_AV`．CREMA-D のラベルを使っていない）は {e_ieo('start_AV', 5, 0):.2f} と {e_mean('start_AV', 5, 0):.2f}，上限の融合（`full_AV`）は {e_ieo('full_AV', 5, 0):.2f} と {e_mean('full_AV', 5, 0):.2f} である．"
  "同じテストの上の値で，版と K は独立ではない．この数えを検定の結果として読まない．"
  "**UAR の引き算の中身**（文 `IEO` − ほかの 11 文の平均．点の値の引き算）：Neutral の引き算は，`start_A` が " + fpt(e_gap("start_A", 5, 0)) + " 点，`none_A` が " + "・".join(fpt(e_gap("none_A", K, 0)) for K in KS)
  + " 点，`ret_A` が " + "・".join(fpt(e_gap("ret_A", K, 0)) for K in KS) + " 点である．Happy・Sad・Angry の引き算の平均は，`start_A` が " + fpt(e3_gap("start_A", 5)) + " 点，`none_A` が "
  + "・".join(fpt(e3_gap("none_A", K)) for K in KS) + " 点，`ret_A` が " + "・".join(fpt(e3_gap("ret_A", K)) for K in KS) + " 点である（この 3 感情の引き算には，文 `IEO` にだけ強さ LO・MD・HI があることが入る）．"
  "UAR の引き算は，この 2 つから（Neutral の引き算 ＋ 3 × 3 感情の引き算の平均）÷ 4 で決まる．`start_A` では Neutral の引き算は正で，UAR の引き算が負になるのは，Happy・Sad・Angry の引き算が負であるためである．"
  "K 本で学んだ声の分類器では，3 つの K とも Neutral の引き算が負で，その絶対値は，Happy・Sad・Angry の引き算の平均の絶対値より大きい．")
w(f"   - **強さごと**：文 `IEO` の Happy・Sad・Angry の発話には強さ LO・MD・HI が付き（感情 × 強さごとに {int(ITOT[1, 1] // 10)} 本），文 `IEO` の Neutral と，ほかの 11 文の発話は XX である（この台本が発話の名前で確かめた）．"
  "このため，Happy・Sad・Angry の XX の値はほかの 11 文の値で，LO・MD・HI の値は文 `IEO` の値である．"
  f"台本の検査（`code_post_tools_v2.md`）は「{G3}」と書いている．"
  f"提案の手法（推定時の答え）の K = 5 では，Angry の再現率は XX {i_rec('prop', 5, 'XX', 3):.2f}，LO {i_rec('prop', 5, 'LO', 3):.2f}，MD {i_rec('prop', 5, 'MD', 3):.2f}，HI {i_rec('prop', 5, 'HI', 3):.2f} である．"
  f"16 版（K に依る 10 版は 3 つの K）の {N_VK} 個の並びのうち，LO の再現率が 4 つの強さの中で最も低いのは，Happy で {LO_MIN[1]} 個，Sad で {LO_MIN[2]} 個，Angry で {LO_MIN[3]} 個である（区間は無い．同じテストの上の値で，独立な {N_VK} 個ではない）．")
KN = {K: np.array([KIEO[(K, f, r)] for f in range(5) for r in range(10)]) for K in KS}             # (50 単位, 4 感情)
must([int(KN[K].sum(1).min()) for K in KS] == [0, 3, 11] and [int(KN[K].sum(1).max()) for K in KS] == [7, 14, 27] and int((KN[5].sum(1) == 0).sum()) == 1, "K 本の中の文の本数が，台本の検査の報告の数と違う（文を直す）")
w(f"17. **K 本の中の文 `IEO` の本数（記述．2.7 節）**．設計 4 節は，文 `IEO` を除く理由を「{DQ_CUE}」と書いている．K 本（感情ごとに K 本．計 4K 本）の中の文 `IEO` の発話は，50 単位で，"
  + "，".join(f"K = {K} が {int(KN[K].sum(1).min())}〜{int(KN[K].sum(1).max())} 本（平均 {KN[K].sum(1).mean():.2f} 本）" for K in KS) + f"である．K = 5 で 1 本も無い単位は {int((KN[5].sum(1) == 0).sum())} 個である．"
  "Neutral の K 本に文 `IEO` が 1 本も無い単位は，K = 5・10・25 の順に " + "・".join(str(int((KN[K][:, 0] == 0).sum())) for K in KS) + " 個である"
  f"（テストでは，文 `IEO` の発話は Neutral の {int(N_PER_SEED[CD][0]):,} 本のうち {RM_E[0]} 本，Happy・Sad・Angry の {int(N_PER_SEED[CD][1:].min()):,}〜{int(N_PER_SEED[CD][1:].max()):,} 本のうち {RM_E[1]} 本ずつである）．"
  f"計算の前の感情認識の査読想定（`review_affect.md`）は「{RA_CUE}」と書いていた．"
  "K 本の中の文 `IEO` の発話が，Neutral 以外の感情にだけある単位（Neutral の K 本に無い単位から，1 本も無い単位を除いた数）は，K = 5・10・25 の順に 50 単位のうち " + "・".join(str(int(((KN[K][:, 0] == 0) & (KN[K].sum(1) > 0)).sum())) for K in KS) + " 個である．"
  "K 本で学んだ声の分類器で，文 `IEO` の Neutral の再現率がほかの 11 文の平均より低い並びは，要点 16 に書いた．"
  f"**K 本の中の本数は記述で，文 `IEO` が K 本に入ることの影響を測った値ではない**．台本の検査（`code_post_tools_v2.md`）は「{G4}」と書いている．"
  "この文書は，K 本の中の本数と，単位ごとの UAR や差を並べていない（第 3 版にあった，K 本の引き方を本数で 2 つに分けた並びは，照合の第 3 回の指摘 3 を受けて外した．6 節）．K 本から文 `IEO` を除いて学び直す試しは，していない．")
ASM_K = {K: ASM[K] for K in KS}


def a3(col: str, nd: int = 2) -> str:
    return "・".join(f"{float(getattr(ASM_K[K], col)):.{nd}f}".replace("-", "−") for K in KS)


def a1(col: str, nd: int = 2) -> str:
    must(all(abs(float(getattr(ASM_K[K], col)) - float(getattr(ASM_K[5], col))) < 1e-12 for K in KS), f"K に依らないはずの量が K で違う：{col}")
    return f"{float(getattr(ASM_K[5], col)):.{nd}f}".replace("-", "−")


must(all(float(getattr(ASM_K[K], "topok_M_ext")) == 0.0 and float(getattr(ASM_K[K], "topok_M_home")) == 1.0 for K in KS), "最も強いモダリティの一致（文を直す）")
w("18. **仮定の表（CREMA-D．3.3 節）：ソースで測った表と，K 本で測った表を，候補の全ラベルで測った表（参照）と比べた量**．"
  f"**分類器が 2 つ（声と顔）なので，分類器が 3 つの元の 4 コーパスの値（実験 20 の結果の文の 3.2 節）と同じ尺度で並べない**（台本の検査 `code_post_tools_v2.md`：「{G5}」「{G5B}」）．基準は置かない．値は 50 単位の平均である．"
  "ソースの中で測った表（`home`）と，それに残り方を掛けた表（`ext`）は，それぞれ，50 単位のどれでも同じ値になる（ソースだけから作るので，単位に依らない．ソースは 50 単位とも元の 4 コーパスである．この台本が確かめた）．"
  "`home` と `ext` は互いに違う表である．K 本で測った表と，全ラベルで測った表は，単位ごとに違う．")
w(f"   - **水準**（d′ の行の平均）：ソースの中で測った水準は声 {a1('level_M_home_A')}・顔 {a1('level_M_home_V')}，残り方は声 {a1('ret_A', 3)}・顔 {a1('ret_V', 3)} で，ソースで測った表（残り方を掛けた表）の水準は声 {a1('level_M_ext_A')}・顔 {a1('level_M_ext_V')} である．"
  f"K 本で測った水準は，K = 5・10・25 の順に，声が {a3('level_M_K_A')}，顔が {a3('level_M_K_V')} で，全ラベルで測った水準は，声が {a3('level_M_orc_A')}，顔が {a3('level_M_orc_V')} である．"
  "**ソースで測った表（残り方を掛けた表）は顔の水準のほうが高く，全ラベルで測った表は 50 単位 × 3 つの K の全部で声の水準のほうが高い**．ソースの中で測った表（残り方を掛ける前）は，声の水準のほうが高い．"
  f"このため「最も強いモダリティの一致」は，ソースで測った表が {a1('topok_M_ext')}，ソースの中で測った表が {a1('topok_M_home')} になる．K 本で測った表は，K = 5・10・25 の順に {a3('topok_M_K')} である．"
  "この量は 2 つの分類器から 1 つを選ぶ量なので，でたらめに選んでも 0.5 になる．CREMA-D の提案の点は 3 つの K とも n0 = 0 である（要点 9．相性を外した版は，K 本で測った水準だけで重みを決める版と同じ判定になる）．")
w(f"   - **相性と，共通の感情の成分**（ソースで測った表 `ext` と，全ラベルで測った表）：K = 5・10・25 の順に，行の中の差の相関は {a3('profcorr_M_ext')}，行ごとの相関は声が {a3('profcorr_row_A')}，顔が {a3('profcorr_row_V')}，共通の感情の成分の相関は {a3('emocorr_M_ext')} である．"
  f"K 本で測った表と，全ラベルで測った表では，行の中の差の相関が {a3('profcorr_M_K')}，共通の感情の成分の相関が {a3('emocorr_M_K')} である．"
  "行ごとの相関と共通の感情の成分の相関は 4 個の数の相関，行の中の差の相関は 8 個の数（自由な値は 6 個）の相関で，偶然でも大きく振れる．")
w(f"   - **成分の相関と，入れ替えの割合**（声の行の 4 個の数のうち，自由に動く 2 個の数の上の量）：K = 5・10・25 の順に，ソースで測った表は，成分の相関が {a3('intcorr_M_ext')}，入れ替えの割合が {a3('intperm_M_ext', 3)} で，"
  f"K 本で測った表は，成分の相関が {a3('intcorr_M_K')}，入れ替えの割合が {a3('intperm_M_K', 3)} である．")
w(f"   - **そのほか**：テストとのずれ（全ラベルで測った表と，テストで測った d′ の差の絶対値の平均）は {a3('drift_M', 3)}，K 本で測った 8 個の d′ のうち上限（{DCAP:.2f}）に達した割合は {a3('dk_ceiling_share', 4)}，0 以下だった割合は {a3('dk_zero_share', 4)} である．")
w("   - **作っていない量**：水準の相関（2 つの分類器では 2 点の相関になり，±1 にしかならない），一致の余りと広がりの比（単位の npz から作れるが，作っていない），別のソースに当てた値で測った量（CREMA-D の npz に表が無い），"
  "顔の行を除いた相関（声の行だけになり，行ごとの相関の声と同じ）．"
  f"**名前の区別**：仮定の表の「残り方」（csv の列の名前は `ret_A`・`ret_V`．ソースで測った値）は，内訳の表の版「微調整した声・顔」（版の名前が同じ `ret_A`・`ret_V`．K 本で微調整した分類器）とは別のものである（台本の検査：「{G5C}」）．"
  f"感情認識の査読想定（`review_affect.md`）は，計算の前に「{RQ['RA3']}」と書いていた．表の数そのものは 2.6 節に並べてある．"
  "成分の相関，入れ替えの割合，テストとのずれの 7 列は，この台本では作り直していない（0 節）．")
w("")

# ---- 2 節
w("## 2. 表")
w("")
w("### 2.1 表の文書（既にある表）")
w("")
DQ_UNITS = quote(design, "分割 5 × 種 10 ＝ 1 コーパス 50，計 200（実験 20 と同じ単位）", "設計 2 節")
w(f"UAR と差の表は `DAI/docs/実験21_結果の表_2026-11-15.md`（sha256 {sha(TABLE_DOC)[:8]}…．書き出しの台本 `DAI/docs/tools_pending/md_tables_dai21.py`）にあり，ここには写さない．"
  "その文書の 1 節が，実験 20 と合わせた 5 コーパスのまとめ（K 本だけの融合，積，今の提案，提案の手法の推定時の答えと学び直した融合の UAR，提案の手法との差，K 本の引き方で分けた値），"
  "2 節が疑似ラベルを替えた学び直し，3 節が実験 1 の比較手法との並び（元の 4 コーパス），4 節が CREMA-D の推定時の答え，5 節が記述の版（設定を疑似ラベルごとに選んだ点，設定の 12 点の UAR）である．"
  "この台本は，表の文書が今の集計から書き出し直した表と 1 字も違わないことを確かめた（0 節）．")
w("")
w(f"**表の文書は，最初の書き出しの後に，本体が {len(TD_HIST) - 1} 回書き出し直した**（最初の版はコミット {TD_FIRST[0]}，{TD_FIRST[1][5:16]}．その後のコミットは {'・'.join(h_ for h_, _ in TD_HIST[-2::-1])}．今の版は {TD_LAST[0]}，{TD_LAST[1][5:16]}）．"
  "この台本は，最初の版と今の版を欄ごと・行ごとに比べ，次の直しだけが入っていることを確かめた．値と印は変わっていない．"
  f"(1) 差がちょうど 0 で区間の両端も 0 の欄のうち，判定が違う {TBL_NOTE} 個に「{NOTE_CNT}」の注が付いた（MELD の K = 25 の今の提案，MELD の K = 10 と MOSEI の K = 25 の温度つき蒸留の，この実験で解いた版 − 実験 1 の保存）．"
  f"(2) 差が 0.00 に丸まっていた {TBL_DIG} 個の欄に，符号が読めるように桁が足された．(3) 文の行は 2 行が変わり，1 行が増えた．変わったのは，桁の決まりの行（差が 0.00 に丸まるときの桁と，差がちょうど 0 の欄の注の説明が入った）と，K 本の引き方の説明の行である．"
  f"増えたのは，1 節のまとめの表の下の注「{TD_EXPL}」である（照合の第 1 回の指摘 10）．"
  "(4) 1 節の「提案の手法との差」の表で，再現の検算（元の 4 コーパス）と性能の差（CREMA-D）が 1 行に並んでいた行が，「今の提案 − K 本だけの融合」の行と「再現の確かめ」の行の 2 行に分かれた（同じ指摘）．表の行は 3 行増えた．"
  "分かれた行の欄の値は，元の行の欄と，3 節の表の「今の提案 − K 本だけの融合」の行の欄と同じである．"
  f"K 本の引き方の説明は「{TD_K29}」になった．元の 4 コーパスの単位は実験 20 と同じ単位である（設計 2 節：「{DQ_UNITS}」）．この文書は，引き方 2〜9 の K 本を「下調べと違う K 本」と書く．")
w("")
w(f"今の表の文書と，この文書の書き方の対応を記録する．(a) 表の文書は負の符号を「-」，この文書は「−」と書く．(b) 差が丸めると ±1.00 か 0.00 になるときと，区間の端が 0.05 より 0 に近いときに桁を足す決まりは，2 つの文書で同じである．今の表の文書に，0 に丸まったままの差と区間の端は残っていない（差がちょうど 0 の欄を除く）．"
  f"(c) 表の文書の「+0.00 [+0.0, +0.0]」の欄は，差がちょうど 0 の欄で，{TD_ZERO_SAME + TD_ZERO_CNT} 個ある．うち {TD_ZERO_SAME} 個に「{NOTE_SAME}」，{TD_ZERO_CNT} 個に「{NOTE_CNT}」が付く．"
  "この文書は，前者を「0（同じ判定）」，後者を「0（正解の数が同じ）」と書く．表の文書の注の「正解の数は同じ」は，塊ごと・感情ごとの正解の数が全部の単位で同じことを指す（この台本が，その 3 つの条件で集計の `same_as` が同じ群であることを確かめた）．"
  "(d) 切片のずらしの定義は，この文書と表の文書で同じ文である（0 節の「版の名前」）．")
w("")
w("### 2.2 読みの一覧")
w("")
w("欄の中は「コーパス K」で，その読みに入る条件を並べた．「0 を含まず」は区間が 0 を含まないことを表す．一覧は値を読みやすくするためのもので，判定ではない．欄の中の条件の数を数えて主張に使わない．値は表の文書と要点にある．"
  "最後の列は，差がちょうど 0 の条件で，2 つの版が同じ判定の条件と，判定は違う発話があるが塊ごと・感情ごとの正解の数が全部の単位で同じ条件である（どちらの条件かは要点と 2.1 節）．区間は補正していない．"
  "5 コーパスの行には 15 個，元の 4 コーパスの行には 12 個，CREMA-D の行には 3 個の区間がある．同じコーパスの 3 つの K は同じテストを共有する．"
  "重みの表を乱数で探す版の行は，引いた表の組に条件づけた値で，表の引き方の揺れは出していない（要点 8）．")
w("")
SHOWN: set = set()


def cat_table(title: str, rows: list, pairs: list, note: str = "") -> None:
    """rows：(a, b, 行の名前)．a・b は版の名前か，コーパスから版の名前を返す関数．"""
    w(f"**{title}**")
    w("")
    if note:
        w(note)
        w("")
    w("| 差 " + CAT_HEAD)
    w("|---" + CAT_SEP)
    for a, b, lab in rows:
        w(f"| {lab} | {cat_cells(eget(a, b), pairs)} |")
        for c, _ in pairs:
            SHOWN.add((a(c) if callable(a) else a, b(c) if callable(b) else b, "cd" if c == CD else "4"))
    w("")


LAB5 = {t: lab for t, lab in MT.EXTRA}
cat_table("学び直した融合（提案の手法の疑似ラベル．`stud_prop`）− 相手：5 コーパス", [
    ("stud_prop", "stud_prod", "− 積の疑似ラベルで学び直した融合（今の提案．`stud_prod`）"), ("stud_prop", fus, "− K 本だけの融合"), ("stud_prop", "prop", "− 提案の手法の推定時の答え（`prop`）"),
    ("stud_prop", "prod", "− 積（推定時の答え．`prod`）")] + [("stud_prop", f"stud_{t}", f"− {LAB5[t]}の疑似ラベル（`stud_{t}`）") for t in EXTRA], ALL15)
cat_table("疑似ラベルを替えた学び直し − 積の疑似ラベルで学び直した融合（`stud_<別の疑似ラベル> − stud_prod`）：5 コーパス", [(f"stud_{t}", "stud_prod", f"{LAB5[t]}（`stud_{t}`）") for t in EXTRA], ALL15)
cat_table("学び直した融合（提案の手法の疑似ラベル．`stud_prop`）− 実験 1 の手法：元の 4 コーパス", [("stud_prop", "joint_AVT", "− 今の提案（実験 1 の保存．`joint_AVT`）")]
          + [("stud_prop", v, f"− {nm(v)}（実験 1 の保存．`{v}`）") for v in B5] + [("stud_prop", f"{m}_prop", f"− {EXN[m]}に提案の手法の疑似ラベルを渡した版（`{m}_prop`）") for m in EX1], ALL12)
cat_table("実験 1 の 3 手法の疑似ラベルを替えた版と，実験 1 の保存どうしの差：元の 4 コーパス",
          [(f"{m}_prop", f"{m}_prod", f"{EXN[m]}：提案の手法の疑似ラベル − 積の疑似ラベル（どちらもこの実験で解いた）") for m in EX1]
          + [(f"{m}_prop", f"b_{m}", f"{EXN[m]}：提案の手法の疑似ラベルの版 − 実験 1 の保存") for m in EX1]
          + [(f"{m}_prod", f"b_{m}", f"{EXN[m]}：積の疑似ラベルの版 − 実験 1 の保存（解き直しのずれ）") for m in EX1]
          + [("stud_prod", "joint_AVT", "今の提案：この実験で解いた版 − 実験 1 の保存（解き直しのずれ）")]
          + [(v, "joint_AVT", f"{nm(v)} − 今の提案（実験 1 の保存どうし）") for v in B5]
          + [("joint_AVT", "none_AVT", "今の提案 − K 本だけの融合（実験 1 の保存どうし）"), ("prop", "joint_AVT", "提案の手法の推定時の答え − 今の提案（実験 1 の保存）")], ALL12)
cat_table("記述の版：疑似ラベルごとに選んだ点どうしの差", [("stud_prop_sel", "stud_prod_sel", "元の 4 コーパス：`stud_prop_sel − stud_prod_sel`")], ALL12,
          "提案の手法の疑似ラベルの点の選択にだけ，評価するコーパスのテストのラベルが間接に入る（設計 3.1 節）．記述だけに使う．")
cat_table("記述の版：CREMA-D", [("stud_prop_sel", "stud_prod", "CREMA-D：`stud_prop_sel − stud_prod`")], ALL3)
cat_table("CREMA-D：提案の手法（推定時の答え）− 相手（どちらも切片をずらした答え）", [("prop", v, f"− {nm(v)}（`{v}`）") for v in A2 + ABL + ["M_F6_ext"]], ALL3)
cat_table("CREMA-D：提案の手法 − 相手（どちらも切片をずらす前）", [("prop@raw", f"{v}@raw", f"− {nm(v)}（`{v}@raw`）") for v in A2], ALL3)
cat_table("CREMA-D：提案の手法（切片をずらした答え）− 相手（切片をずらす前）", [("prop", f"{v}@raw", f"− {nm(v)}（`{v}@raw`）") for v in A2[1:]] + [("prop", "prop@raw", "− 提案の手法（切片をずらす前．`prop@raw`）")], ALL3,
          "相手に切片のずらしを当てない判定との差である．")
cat_table("CREMA-D：相手 − 積（どちらも切片をずらした答え），そのほかの差", [(v, "prod", f"{nm(v)}（`{v}`）− 積") for v in A2[1:] + ["prop_noprof", "prop_levelfree", "M_F6_orc", "M_F6_ext"]]
          + [("prop_noint", "prop_noprof", "分類器 × 感情の成分を消す − 相性を外す"), ("prop", "none_AV", "提案の手法（推定時の答え）− K 本だけの融合"), ("prop", "stud_prod", "提案の手法（推定時の答え）− 今の提案"),
             ("stud_prod", "none_AV", "今の提案 − K 本だけの融合"), ("stud_prod", "prod", "今の提案 − 積（推定時の答え）")], ALL3)
must({(a, b) for a, b, p_ in SHOWN if p_ == "4"} == set(P4) and {(a, b) for a, b, p_ in SHOWN if p_ == "cd"} == set(PC), "読みの一覧に出していない差の対がある")
w(f"上の一覧には，集計の差の対の全部（元の 4 コーパス {len(P4)} 個，CREMA-D {len(PC)} 個）が出ている（台本が確かめた）．")
w("")
w("### 2.3 推定時の答え：提案の手法 − 積（どちらも切片をずらした答え．K 本の引き方で分けた値を含む）")
w("")
w("**元の 4 コーパスの値は実験 20 の集計（`results/dai20_eval/summary/effects.csv`），CREMA-D の値は実験 21 の集計から引いた**（0 節の「実験 20 の集計との使い分け」）．元の 4 コーパスの値は，実験 20 の結果の文の要点 1 と同じ値である．"
  "引き方 0・1 は下調べと同じ K 本（10 単位），引き方 2〜9 は下調べと違う K 本（40 単位）である．テストの発話はどちらも下調べと同じである．")
w("")
w("| コーパス | K | 引いた集計 | K 本の引き方 0〜9（50 単位） | 引き方 0・1 | 引き方 2〜9 |")
w("|---|---|---|---|---|---|")
for c, K in ALL15:
    g_ = q20 if c != CD else q
    w(f"| {c} | {K} | {'実験 20' if c != CD else '実験 21'} | {ci(g_('prop', 'prod', c, K), mark=True)} | {ci(g_('prop', 'prod', c, K, '0-1'), mark=True)} | {ci(g_('prop', 'prod', c, K, '2-9'), mark=True)} |")
w("")
T20_ROWS = [x for x in text20.split("\n") if x.startswith("| ")]
must(all(any(x.startswith(f"| {c} | {K} | ") and x.endswith(f"| {ci(q20('prop', 'prod', c, K), mark=True)} |") for x in T20_ROWS) for c, K in ALL12), "元の 4 コーパスの提案の手法 − 積が，実験 20 の結果の文の表の値と違う")
w("### 2.4 疑似ラベルごとの 3 つの量（探索の記述．主の点．50 単位の平均）")
w("")
w("要点 12 の 3 つの量を，7 通りの疑似ラベルについて並べた．(a) は疑似ラベルの判定の UAR（K 本を除いた候補の発話の上．候補のラベルで測る），(b) は学び直した融合の分類器のテストの上の UAR（区間つき．`levels.csv`），"
  "(c) は，K 本を除いた候補の発話のうち，学び直した融合の分類器の判定が疑似ラベルの判定と同じだった割合である．「候補の上の UAR」は，学び直した融合の分類器の UAR を，(a) と同じ候補の発話の上で測った値である（単位の csv の `uar_pool`）．"
  "候補の発話（K 本を除く）は，学び直した融合の分類器が疑似ラベルで学んだ入力なので，(a)・(c)・候補の上の UAR は学習データの上の量である．(b) だけが，学習に使っていない発話の上の量である．"
  "(a)・(c)・候補の上の UAR に区間は無い．結果を開いた後に並べた記述で，仕組みについての主張には使わない．")
w("")
w("| コーパス | K | 疑似ラベル | (a) 疑似ラベルの UAR（候補の上） | 学び直した融合の UAR（候補の上） | (b) 学び直した融合の UAR（テストの上） | (c) 判定が一致した割合 |")
w("|---|---|---|---|---|---|---|")
TLAB = {"prod": "積", "prop": "**提案の手法**"} | {t: lab for t, lab in MT.EXTRA}
UP = ST.groupby(["c", "K", "version", "grid"]).uar_pool.mean()
for c, K in ALL15:
    for t in ["prod", "prop"] + EXTRA:
        up = float(UP[(c, K, f"stud_{t}", GM[(c, K)])])
        must(abs(up - tq(c, K, t, "sp")) < 1e-9, f"候補の上の UAR が判定の配列から数え直した値と違う {c} K={K} {t}")
        w(f"| {c} | {K} | {TLAB[t]} | {tq(c, K, t, 't'):.2f} | {up:.2f} | {lci(level('stud_' + t, c, K))} | {tq(c, K, t, 'm'):.3f} |")
w("")
w("### 2.5 感情ごとの再現率（学び直した融合の分類器と，CREMA-D の推定時の答え）")
w("")
w("`per_emotion.csv` の再現率（%）と 95% 区間（補正していない）．差の区間は作っていない．件数は，K 本の引き方 1 つぶんのテストの発話の数（5 分割の合計）で，3 つの K と全部の版で同じである．集計の件数の列（`n`）は，K 本の引き方 10 個について足した数である．")
w("")
w("| コーパス | Neutral | Happy | Sad | Angry | 計 |")
w("|---|---|---|---|---|---|")
for c in C5:
    must(all(float(PE[(c, 5, "stud_prop", e)].n_per_seed) == N_PER_SEED[c][e] and float(PE[(c, 5, "stud_prop", e)].n) == 10 * N_PER_SEED[c][e] for e in range(4)), f"per_emotion.csv の件数 {c}")
    w(f"| {c} | " + " | ".join(f"{int(N_PER_SEED[c][e]):,}" for e in range(4)) + f" | {int(N_PER_SEED[c].sum()):,} |")
w("")
w("| コーパス | K | 版 | Neutral | Happy | Sad | Angry |")
w("|---|---|---|---|---|---|---|")


def pe_row(c, K, v, lab):
    w(f"| {c} | {K} | {lab} | " + " | ".join(f"{PE[(c, K, v, e)].recall:.2f} [{PE[(c, K, v, e)].lo:.1f}, {PE[(c, K, v, e)].hi:.1f}]" for e in range(4)) + " |")


for c, K in ALL15:
    pe_row(c, K, "stud_prod", "学び直した融合：積の疑似ラベル（`stud_prod`）")
    pe_row(c, K, "stud_prop", "学び直した融合：提案の手法の疑似ラベル（`stud_prop`）")
for K in KS:
    pe_row(CD, K, "prod", "推定時の答え：積（`prod`）")
    pe_row(CD, K, "prop", "推定時の答え：提案の手法（`prop`）")
w("")
w("### 2.6 選んだ設定の点と，設定を変えた値（記述）")
w("")
EDGE = [g for d_ in list(FZ4.values()) + list(FZC.values()) for k_, g in d_.items() if k_.startswith("student_")]
must(len(EDGE) == frozen["n_student_points"] == 42 and sum(SGRID[int(g)][0] in (0.0625, 4.0) for g in EDGE) == frozen["n_student_points_at_rho_edge"], "固定のファイルの，ρ が格子の端の点の数")
w("**選んだ点**（固定のファイル `DAI/frozen/dai21_frozen.json`）．提案の格子の点は (n0，β，戻し先) の 16 点の番号，学び直しの設定の点は (ρ，a) の 12 点の番号である．")
w("")
w("| コーパス | K | 提案の格子の点 | 学び直しの主の点 | 疑似ラベルごとに選んだ点：提案の手法の疑似ラベル（`stud_prop_sel`） | 積の疑似ラベル（`stud_prod_sel`） |")
w("|---|---|---|---|---|---|")
for c, K in ALL15:
    w(f"| {c} | {K} | {gdesc(GI[(c, K)])} | {sdesc(GM[(c, K)])} | {sdesc(GS[(c, K)])} | {sdesc(GPS[(c, K)]) if c != CD else '主の点と同じ選び方'} |")
w("")
TETH = {k_: [tuple(FZC[K]["tether"][k_]) for K in KS] for k_ in ("A", "V", "AV")}
must(all(TETH["V"][i_][0] == 0.001 for i_ in range(3)) and [SGRID[GM[(CD, K)]][0] for K in KS] == [1.0, 4.0, 4.0], "CREMA-D の顔の引き戻しの強さと ρ")
RHO_LT = sum(SGRID[GM[k_]][0] < SGRID[GS[k_]][0] for k_ in ALL15)
RHO_EQ = sum(SGRID[GM[k_]][0] == SGRID[GS[k_]][0] for k_ in ALL15)
must(RHO_LT + RHO_EQ == 15, "主の点の ρ が，提案の手法の疑似ラベルのために選んだ点の ρ より大きい条件がある（文を直す）")
must(EDGE_PROP["rho_hi"] == 13 and EDGE_PROD15["rho_hi"] == 8, f"ρ の上端の点の数が，比較の検査の報告の数と違う（文を直す）：{EDGE_PROP}，{EDGE_PROD15}")
w(f"固定のファイルの記録では，上の表の学び直しの設定の点（主の点と，疑似ラベルごとに選んだ点の計 {frozen['n_student_points']} 個）のうち，ρ が格子の端（0.0625 か 4）の点は {frozen['n_student_points_at_rho_edge']} 個である．"
  "**選んだ点が格子の端に当たった数を，疑似ラベルごとに数えた**（この台本が，固定のファイルの点の番号と `student_grid.csv` の ρ・a から数えた．格子は ρ が 0.0625・0.25・1・4，a が 1・4・16）．"
  f"主の点（積の疑似ラベルのために選ばれた点．15 条件）は，ρ が上端（4）の点が {EDGE_MAIN['rho_hi']} 個（CREMA-D の K = 10・25），下端（0.0625）の点が {EDGE_MAIN['rho_lo']} 個，a が端（1 か 16）の点が {EDGE_MAIN['a_edge']} 個である．"
  f"提案の手法の疑似ラベルのために選んだ点（`stud_prop_sel`．15 条件）は，ρ が上端の点が {EDGE_PROP['rho_hi']} 個，下端の点が {EDGE_PROP['rho_lo']} 個，a が端の点が {EDGE_PROP['a_edge']} 個である．"
  f"積の疑似ラベルのために選んだ点（元の 4 コーパスの `stud_prod_sel` の 12 条件）は，ρ が上端の点が {EDGE_PROD4['rho_hi']} 個，下端の点が {EDGE_PROD4['rho_lo']} 個，a が端の点が {EDGE_PROD4['a_edge']} 個である．"
  f"これに CREMA-D の主の点（積の疑似ラベルのために同じ規則で選んだ点．3 条件）を足した 15 条件では，ρ が上端の点が {EDGE_PROD15['rho_hi']} 個，a が端の点が {EDGE_PROD15['a_edge']} 個である．"
  f"計算の前の比較の検査（`compare.md`）は「{RQ['C16a']}」「{RQ['C16b']}」と書き，「{RQ['C16c']}」と求めていた．固定のファイルが持つのは ρ の端の数の合計だけで，疑似ラベルごとの数と a の端の数は，この文書で数えた．"
  f"主の点の ρ が，提案の手法の疑似ラベルのために実験 18 の規則で選んだ点の ρ より小さい条件は，15 条件のうち {RHO_LT} 条件で，残る {RHO_EQ} 条件は同じ ρ である（主の点の ρ のほうが大きい条件は無い）．"
  f"CREMA-D の引き戻しの強さ（固定のファイルの `tether`．K = 5・10・25 の順）は，声が {'・'.join(f'({a:g}, {b:g})' for a, b in TETH['A'])}，顔が {'・'.join(f'({a:g}, {b:g})' for a, b in TETH['V'])}，声と顔の融合が {'・'.join(f'({a:g}, {b:g})' for a, b in TETH['AV'])} である．"
  f"ML の査読想定（`review_ml.md`）は「{RQ['RM11']}」と書いている．")
w("")
w("**設定の 12 点の UAR**（学び直した融合の分類器．`student_grid.csv`．50 単位の平均．区間は無い）．「最大」は，テストの上で 12 個から選んだ値で，上に偏る．実際には使えない．主の点との差を，選び方の根拠にしない（要点 11）．")
w("")
w("| コーパス | K | 積の疑似ラベル：主の点 | 12 点の最小 | 12 点の最大（点） | 提案の手法の疑似ラベル：主の点 | 12 点の最小 | 12 点の最大（点） |")
w("|---|---|---|---|---|---|---|---|")
for c, K in ALL15:
    a_, b_ = G12[(c, K, "stud_prod")], G12[(c, K, "stud_prop")]
    w(f"| {c} | {K} | {a_['main']:.2f} | {a_['lo']:.2f} | {a_['best']:.2f}（{a_['arg']}） | {b_['main']:.2f} | {b_['lo']:.2f} | {b_['best']:.2f}（{b_['arg']}） |")
w("")
w(f"**CREMA-D の提案の格子 16 点の UAR**（切片をずらした答え．`grid_levels_cremad.csv`．50 単位の平均．区間は無い．「固定」は固定のファイルの点）．テストの上の値で，点を選ぶためには使っていない．"
  f"感情認識の査読想定（`review_affect.md`）は「{RQ['RA4b']}」と書いている．")
w("")
w("| 点 | n0 | β | 戻し先 | K = 5 | K = 10 | K = 25 |")
w("|---|---|---|---|---|---|---|")
GL = {(int(r.K), int(r.grid)): r for r in glc.itertuples()}
for g in range(16):
    n0, b, pr = GRID[g]
    w(f"| {g} | {'∞' if n0 == INF else int(n0)} | {int(b)} | {pr} | " + " | ".join(f"{GL[(K, g)].uar_adj:.2f}{'（固定）' if bool(GL[(K, g)].frozen) else ''}" for K in KS) + " |")
w("")
must(all(abs(GL[(K, GI[(CD, K)])].uar_adj - level("prop", CD, K).uar) < 1e-9 and abs(GL[(K, 12)].uar_adj - level("prod", CD, K).uar) < 1e-9 for K in KS), "CREMA-D の格子の表の固定の点と点 12 の UAR")
w("点 12（n0 = ∞，β = 0）の UAR は積の UAR と同じで，固定の点の UAR は提案の手法の UAR と同じである（台本が確かめた）．n0 = 0 の点は，戻し先が eq でも ext でも同じ値である．")
must(all(abs(GL[(K, g)].uar_adj - GL[(K, g + 1)].uar_adj) < 1e-9 for K in KS for g in (0, 2)), "n0 = 0 の点の UAR が戻し先で違う（文を直す）")
w("")
w(f"**CREMA-D で使った重みの表**（`weights_cremad.csv`．50 単位の平均．表ごとに 8 個の重みの平均が 1 になる大きさ）．提案の手法の重みの表と，K 本だけで測った表，ソースだけで測った表，全ラベルで測った表（参照）の重みである．"
  f"漏れの再検査（`leak_v2.md`）：「{RQ['LV3']}」感情認識の査読想定（`review_affect.md`）は「{RQ['RA3']}」と書いている．この節は，表の数そのものを並べる．ソースで測った表の当てはまりを測る量（仮定の表．相関を含む）は，計算の後の解析で出した（要点 18，3.3 節）．")
w("")
w("| K | 表 | 声：Neutral | Happy | Sad | Angry | 顔：Neutral | Happy | Sad | Angry |")
w("|---|---|---|---|---|---|---|---|---|---|")
WC = {(int(r.K), r.mod, int(r.emotion)): r for r in wcd.itertuples()}
must(sorted(set(wcd["mod"])) == ["A", "V"], "weights_cremad.csv の分類器")
for K in KS:
    for col, lab in (("w_prop", "提案の手法"), ("w_K", "K 本だけで測った表"), ("w_ext", "ソースだけで測った表"), ("w_orc", "全ラベルで測った表（参照）")):
        w(f"| {K} | {lab} | " + " | ".join(f"{getattr(WC[(K, m_, e)], col):.2f}" for m_ in ("A", "V") for e in range(4)) + " |")
w("")
w("### 2.7 CREMA-D の感度の表（計算の後の解析．`results/dai21_eval5/posthoc/`）")
w("")
w("第 3 版にあった表の値は，照合の第 3 回で出力の csv と一致した．第 4 版で足した「文 `IEO` と，ほかの 11 文の感情ごとの再現率」の表の値と ＊ の付け方は，照合の第 4 回で結果ファイルと一致した（判定は「通す」）．"
  "第 3 版にあった「K 本の引き方を，文 `IEO` の本数で 2 つに分けた並び」の表は，第 4 版で外した（6 節）．読み方は要点 13〜17 に書いた．区間は 95% で，補正していない．この節は，区間の端を小数 2 桁で書く（5 節）．「全発話」は，テストの全発話の上の値で，要点 1〜12 と表の文書と同じ値である．「文 `IEO`」は，文「It's eleven o'clock.」を指す．")
w("")
w("**除いた発話の数**（K 本の引き方 1 つぶん．5 分割の合計．`cremad_no_ieo_removed.csv`．この台本が単位の npz の発話の名前で数え直した）")
w("")
w("| | Neutral | Happy | Sad | Angry | 計 |")
w("|---|---|---|---|---|---|")
w("| テストの全発話 | " + " | ".join(f"{int(x):,}" for x in N_PER_SEED[CD]) + f" | {int(N_PER_SEED[CD].sum()):,} |")
w("| 除いた発話（文 `IEO`） | " + " | ".join(f"{int(x):,}" for x in RM_E) + f" | {int(RM_E.sum()):,} |")
w("| 残った発話 | " + " | ".join(f"{int(x):,}" for x in KEEP_E) + f" | {int(KEEP_E.sum()):,} |")
w("")
w("**差**（`cremad_no_ieo_effects.csv`・`cremad_sentence_cluster_effects.csv`．K 本の引き方 0〜9．集計の段の 78 対の全部）．列は左から，全発話の差と話者を塊にした区間（集計の `effects.csv` と同じ値），"
  f"同じ差の文を塊にした区間（自由度 {DF_SENT}．点推定は左の列と同じ），文 `IEO` を除いたテストの差と区間（話者を塊にした区間），除いた値 − 全発話の値と，その対応のある区間である．"
  "(i)・(ii) は要点 1・6・7・8・10 の本文が読みを書いた差，(iii) は要点 9 の差，(iv) は要点 7・8 の下位の項が読みを書いた差（切片をずらす前の版との差），(v) は 2.2 節に読みを並べた「相手 − 積」，"
  "(vi) は集計の段の残りの差（どれも学び直した融合の差）である（(vi) の並びは版の名前の順）．版の名前に `@raw` が付く版は，切片をずらす前の答えである．")
w("")


def tab27(head: str, pairs: list) -> None:
    w(head)
    w("")
    w("| 差 | K | 全発話：差 [話者を塊にした区間] | 全発話：文を塊にした区間 | 文を除いたテスト：差 [区間] | 除いた値 − 全発話の値 [区間] |")
    w("|---|---|---|---|---|---|")
    for a, b in pairs:
        for K in KS:
            x = (a, b, K)
            se_ = "—" if x in NSAME else f"[{fend2(SCA[x].lo)}, {fend2(SCA[x].hi)}]"
            w(f"| {pn(a, b)} | {K} | {ci2(q(a, b, CD, K))} | {se_} | {ci2(NEA[x])} | {ci2(NEA[x], 'd_est', 'd_lo', 'd_hi')} |")
    w("")


tab27(f"(i) 要点 1・6・7・10 の差（{len(KEY8)} 対）", KEY8)
tab27(f"(ii) 要点 8 の差：提案の手法 − 別のまとめ方（{len(KEY11)} 対．どちらも切片をずらした答え）", KEY11)
tab27(f"(iii) 要点 9 の差：部品を 1 つ外した版と，参照・参考（{len(KEY9)} 対）", KEY9)
tab27(f"(iv) 要点 7・8 の下位の項の差：切片をずらす前の版との差（{len(KEYRAW)} 対）", KEYRAW)
tab27(f"(v) 2.2 節の「相手 − 積」（{len(KEYOPP)} 対．どちらも切片をずらした答え）", KEYOPP)
must(all(a.startswith("stud_") for a, _ in REST), f"集計の段の残りの差（文を直す）：{REST}")
tab27(f"(vi) 集計の段の残りの差（{len(REST)} 対．疑似ラベルを替えた学び直しの差，疑似ラベルごとに選んだ点の差，学び直した融合 − 積）", REST)
NLA = {(v, K): NL[(v, K, "all")] for v in VC for K in KS}
SLA = {(v, K): SCL[(v, K, "all")] for v in VC for K in KS}
must(all(abs(NLA[(v, 5)].uar - NLA[(v, K)].uar) < 1e-12 and abs(NLA[(v, 5)].d_uar - NLA[(v, K)].d_uar) < 1e-12 and abs(SLA[(v, 5)].lo - SLA[(v, K)].lo) < 1e-12 and abs(SLA[(v, 5)].hi - SLA[(v, K)].hi) < 1e-12
         for v in KIND0 for K in KS), "K に依らない版の UAR が K で違う")
LVD_NZ = [(v, K) for v in VC for K in KS if NLA[(v, K)].d_lo > 0 or NLA[(v, K)].d_hi < 0]
LVD = [NLA[(v, K)].d_uar for v in VC for K in KS]
LVD_V = sorted({v for v, _ in LVD_NZ})
must(all(v in KIND0 for v in LVD_V) and all((v, K) in LVD_NZ for v in LVD_V for K in KS), f"UAR の「除いた値 − 全発話の値」の区間が 0 を含まない版（文を直す）：{LVD_NZ}")
ROWS_LV = [(v, 5) for v in KIND0] + [(v, K) for v in KDEP for K in KS]
LVD_NEAR = [(v, K) for v in VC for K in KS if near0(NLA[(v, K)], "d_lo", "d_hi", "d_se", "d_method")]      # UAR の変化の区間の端が 0 に近いもの


def dlv(v: str, K: int) -> str:
    n_ = NLA[(v, K)]
    s_ = f"{fest(n_.d_uar)} [{fend2(n_.d_lo)}, {fend2(n_.d_hi)}]"
    if (v, K) in LVD_NEAR:
        NEAR_USED.append(s_)
        s_ += "＊"
    return s_


w(f"**版ごとの UAR**（`cremad_no_ieo_levels.csv`・`cremad_sentence_cluster_levels.csv`．内訳に入れた 16 版）．K に依らない版（ソースの分類器の平均と上限）は 1 行にし，K = 5 の行の値を書いた．"
  "K に依らない版の UAR と文を塊にした区間は 3 つの K の行で同じで，話者を塊にした区間の端は，引き直しの乱数が行ごとに違うので，3 つの K の行で少し違う．"
  f"版ごとの UAR の「除いた値 − 全発話の値」の区間が 0 を含まないのは，集計の段の 53 版のうち {len(LVD_V)} 版（{'，'.join(f'{pn1(v)}（`{v}`）' for v in LVD_V)}）である"
  f"（補正していない．どちらも K に依らない版で，3 つの K の行とも 0 を含まない．残る版は，3 つの K の行とも 0 を含む）．集計の段の 53 版 × 3 つの K の {len(LVD)} 行の点推定の変化は {fpt(min(LVD))}〜{fpt(max(LVD))} 点である．"
  "UAR の水準の変化と，差の変化は別の量である（要点 13・14 は差の変化を読んでいる）．"
  f"この区間も話者の重みの引き直しで作る．{len(LVD)} 行のうち，端が 0 に近い区間（冒頭の基準）は {len(LVD_NEAR)} 行"
  + ("（" + "，".join(f"`{v}` の K = {K}" for v, K in LVD_NEAR) + "）" if LVD_NEAR else "") + "で，"
  + ("この表の行には無い．" if not any(x in LVD_NEAR for x in ROWS_LV) else "この表の行では ＊ を付けた．"))
w("")
w("| 版 | K | 全発話：UAR [話者を塊にした区間] | 全発話：文を塊にした区間 | 文を除いたテスト：UAR [区間] | 除いた値 − 全発話の値 [区間] |")
w("|---|---|---|---|---|---|")
for v, K in ROWS_LV:
    a_, s2_, n_ = level(v, CD, K), SLA[(v, K)], NLA[(v, K)]
    w(f"| {pn1(v)}（`{v}`） | {'K に依らない' if v in KIND0 else K} | {a_.uar:.2f} [{a_.lo:.2f}, {a_.hi:.2f}] | [{s2_.lo:.2f}, {s2_.hi:.2f}] | {n_.uar:.2f} [{n_.lo:.2f}, {n_.hi:.2f}] | {dlv(v, K)} |")
w("")
w("**感情ごとの再現率**（文を除いたテスト．`cremad_no_ieo_per_emotion.csv`．再現率と 95% 区間．丸かっこは全発話の再現率で，2.5 節と同じ値である．差の区間は作っていない）．"
  f"件数は，Neutral {KEEP_E[0]:,} 本，Happy {KEEP_E[1]:,} 本，Sad {KEEP_E[2]:,} 本，Angry {KEEP_E[3]:,} 本である（K 本の引き方 1 つぶん）．")
w("")
w("| K | 版 | Neutral | Happy | Sad | Angry |")
w("|---|---|---|---|---|---|")
for K in KS:
    for v, lab in (("prod", "推定時の答え：積（`prod`）"), ("prop", "推定時の答え：提案の手法（`prop`）"), ("stud_prod", "学び直した融合：積の疑似ラベル（`stud_prod`）"), ("stud_prop", "学び直した融合：提案の手法の疑似ラベル（`stud_prop`）")):
        must(all(abs(NPE[(v, K, e)].recall_all - PE[(CD, K, v, e)].recall) < 1e-9 for e in range(4)), f"感情ごとの再現率の全発話の列 {v} K={K}")
        w(f"| {K} | {lab} | " + " | ".join(f"{NPE[(v, K, e)].recall:.2f} [{NPE[(v, K, e)].lo:.1f}, {NPE[(v, K, e)].hi:.1f}]（{NPE[(v, K, e)].recall_all:.2f}）" for e in range(4)) + " |")
w("")
w("**文ごとの内訳：文 `IEO` と，ほかの 11 文**（`cremad_by_sentence.csv` の `uar_pooled`．50 単位の正解の数と発話の数を足してから割った再現率の，4 感情の平均．区間は無い）．「順位」は，12 文の中で値が高いほうから数えた文 `IEO` の順位である．")
w("")
w("| 版 | K | 文 `IEO` | ほかの 11 文の最小〜最大 | ほかの 11 文の平均 | 文 `IEO` − ほかの 11 文の平均 | 順位 |")
w("|---|---|---|---|---|---|---|")
for v, K in ROWS_LV:
    w(f"| {pn1(v)}（`{v}`） | {'K に依らない' if v in KIND0 else K} | {s_ieo(v, K):.2f} | {min(s_oth(v, K)):.2f}〜{max(s_oth(v, K)):.2f} | {float(np.mean(s_oth(v, K))):.2f} | {fpt(s_gap(v, K))} | {s_rank(v, K)} |")
w("")
w("**文ごとの内訳：声の分類器の 12 文の値**（同じ csv．発話の数は K 本の引き方 1 つぶん）")
w("")
w("| 文（記号） | 発話の数 | `start_A` | `none_A`：K = 5 | K = 10 | K = 25 | `ret_A`：K = 5 | K = 10 | K = 25 |")
w("|---|---|---|---|---|---|---|---|---|")
for s_ in SENT12:
    must(int(BYS[(s_, 5, "start_A")].n_per_seed) == int(N_SENT[s_]), f"文ごとの発話の数 {s_}")
    w(f"| {s_} | {int(N_SENT[s_])} | {float(BYS[(s_, 5, 'start_A')].uar_pooled):.2f} | " + " | ".join(f"{float(BYS[(s_, K, v)].uar_pooled):.2f}" for v in ("none_A", "ret_A") for K in KS) + " |")
w("")
w("**文ごとの内訳：文 `IEO` と，ほかの 11 文の感情ごとの再現率**（`cremad_by_sentence.csv` の `recall_pooled_0`〜`recall_pooled_3`．50 単位の正解の数と発話の数を足してから割った再現率．区間は無い．第 4 版で足した）．"
  "「11 文」は，ほかの 11 文の文ごとの再現率の平均である．Neutral の丸かっこは，11 文の最小〜最大である．「順位」は，12 文の中で Neutral の再現率が高いほうから数えた文 `IEO` の順位である．"
  f"Neutral の発話は，どの文も強さが XX で，文 `IEO` が {NS0['IEO']} 本，ほかの文が 1 文に 90〜91 本である（K 本の引き方 1 つぶん）．"
  "Happy・Sad・Angry は，文 `IEO` の発話に強さ LO・MD・HI が付き（感情ごとに 273 本），ほかの文は XX である（1 文に 89〜91 本）．この 3 感情では，文の違いと強さの違いは分けられない．"
  "区間は無く，K 本から文 `IEO` を除いて学び直してもいないので，この並びの原因は言えない（要点 16）．")
w("")
w("| 版 | K | Neutral：文 `IEO` | 11 文（最小〜最大） | 順位 | Happy：文 `IEO` | 11 文 | Sad：文 `IEO` | 11 文 | Angry：文 `IEO` | 11 文 |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for v, K in ROWS_LV:
    w(f"| {pn1(v)}（`{v}`） | {'K に依らない' if v in KIND0 else K} | {e_ieo(v, K, 0):.2f} | {e_mean(v, K, 0):.2f}（{min(e_oth(v, K, 0)):.2f}〜{max(e_oth(v, K, 0)):.2f}） | {e_rank(v, K, 0)} | "
      + " | ".join(f"{e_ieo(v, K, e):.2f} | {e_mean(v, K, e):.2f}" for e in (1, 2, 3)) + " |")
w("")
w(f"**強さごとの内訳**（`cremad_by_intensity.csv` の `recall_pooled`．50 単位の正解の数と発話の数を足してから割った再現率．区間は無い）．XX は，Neutral では 12 文の発話（{int(ITOT[0, 0] // 10):,} 本），"
  f"Happy・Sad・Angry ではほかの 11 文の発話（{int(ITOT[0, 1] // 10)}・{int(ITOT[0, 2] // 10)}・{int(ITOT[0, 3] // 10)} 本）である．LO・MD・HI は文 `IEO` の発話（感情 × 強さごとに {int(ITOT[1, 1] // 10)} 本）である．")
w("")
w("| 版 | K | Neutral：XX | Happy：XX | LO | MD | HI | Sad：XX | LO | MD | HI | Angry：XX | LO | MD | HI |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for v, K in ROWS_LV:
    w(f"| {pn1(v)}（`{v}`） | {'K に依らない' if v in KIND0 else K} | {i_rec(v, K, 'XX', 0):.2f} | " + " | ".join(f"{i_rec(v, K, i_, e):.2f}" for e in (1, 2, 3) for i_ in INT4) + " |")
w("")
w("**K 本の中の文 `IEO` の本数**（`cremad_kshot_ieo_units.csv`．50 単位．この台本が単位の npz の K 本の発話の名前で数え直した）")
w("")
w("| K | K 本の数 | 文 `IEO` の本数：最小〜最大 | 平均 | 感情ごとの平均（Neutral・Happy・Sad・Angry） | 文 `IEO` が 1 本も無い単位 | Neutral の K 本に文 `IEO` が無い単位 |")
w("|---|---|---|---|---|---|---|")
for K in KS:
    w(f"| {K} | {4 * K} | {int(KN[K].sum(1).min())}〜{int(KN[K].sum(1).max())} | {KN[K].sum(1).mean():.2f} | " + "・".join(f"{x:.2f}" for x in KN[K].mean(0)) + f" | {int((KN[K].sum(1) == 0).sum())} | {int((KN[K][:, 0] == 0).sum())} |")
w("")
w("### 表の下の注（2 節に共通）")
w("")
w(f"- ↑・↓ は，区間が 0 を含まず差が 1.0 点以上の欄に付く（設計 4 節：「{DQ_MARK}」）．CREMA-D の差は 1.0 点未満のものが多く，区間が 0 を含まなくても印が付かない．CREMA-D の行は区間で読む．")
w(f"- 区間は 95% で，補正していない．集計の差の区間は，K 本の引き方 0〜9 の行が {N_INT['ef_all']} 個，K 本の引き方で分けた行を入れると {N_INT['ef']:,} 個ある．水準の区間（`levels.csv` {N_INT['lv']:,} 個）と，感情ごとの再現率の区間（`per_emotion.csv` {N_INT['pe']:,} 個）は，別に数える．")
w("- K の 3 つは入れ子の K 本と同じテストの上の値である．K どうしの差の区間は作っていない．K 本の引き方 0・1 の値と 2〜9 の値の違いにも，区間は無い．")
w(f"- 「＊」は，塊の重みの引き直しで作る区間（MOSEI と CREMA-D）のうち，0 に近いほうの端が 0 から標準誤差の {NEAR_TH} 倍以内にある区間の値に付く（冒頭）．この区間が 0 を含むかどうかは，引き直しの乱数の組に依りうる．"
  "2.7 節では，全発話の区間，文を除いたテストの区間，「除いた値 − 全発話の値」の区間に付く．")
w(f"- 「同じ判定」は，2 つの版の判定が全部の単位で同じこと（集計の `same_pred`）を指す．台本の差分の検査（`code_v3.md`）：「{RQ['KW4']}」")
w("- AFEW の jackknife では分割 0 が除かれる塊があり，IEMOCAP では全部の分割が除かれる（0 節）．MOSEI の区間は実験 21 の集計の値で，実験 20 の集計の値と少し違う（0 節）．")
w("- 2.7 節の表は，計算の後の解析の値である．第 3 版にあった値は照合の第 3 回で，第 4 版で足した感情ごとの再現率の表は照合の第 4 回で，結果ファイルと一致した．区間の端は小数 2 桁で書いた．文を除いたテストの区間と「除いた値 − 全発話の値」の区間は話者を塊にした区間で，文を塊にした区間は全発話のテストの上の区間である．"
  "文ごと・強さごとの内訳は，50 単位の数を足してから割った値で，区間は無い．"
  "文を除いたテストの区間，「除いた値 − 全発話の値」の区間，話者を塊にした区間は，話者の重みの引き直しで作る．端が 0 に近い区間は，0 を含むかどうかが引き直しの乱数に依りうる（要点 14・15）．")
w("")

# ---- 3 節
w("## 3. 検算と仮定")
w("")
w("### 3.1 検算（設計 5 節の表）")
w("")
w("| 項目（設計 5 節） | 値 | 止める基準 |")
w("|---|---|---|")
w(f"| まとめ方の計算が実験 16・20 と同じ（`check3`） | `logs/dai21_check3.log`：実験 16 の下調べの 2 単位 × K = 5・25 の 4 行とも，38 の版の判定が保存と一致する割合の最小 1.0（38 版とも完全に一致）．「check3：通った」 | 0.995 未満で止める |")
w(f"| CREMA-D の単位の計算が実験 18 と同じ（`check18`） | `logs/dai21_check18.log`：K = 5・10 は 39 の版とも一致の割合 1.0．K = 25 は最小 {C18_MIN25:.6f}（`{C18_WORST25}`）で，39 版のうち {C18_EXACT25} 版が完全に一致．確率のずれの最大 {sci(C18_DEV)}．「check18：通った」 | 0.995 未満で止める |")
w(f"| 疑似ラベルの作り直し | 7 通りの疑似ラベルの判定が実験 20 の保存の判定と一致する割合の最小 {s4_['agree_min']:.6f}（元の 4 コーパスの 600 組の（単位，K）） | 0.995 未満で止める |")
w(f"| 今の提案と実験 1 の 3 手法の再現 | 積の疑似ラベルで学び直した判定が実験 1 の保存と一致する割合の最小：今の提案 {s4_['joint_repro_min']:.6f}（平均 {s4_['joint_repro_mean']:.6f}），温度つき蒸留 {s4_['kd_repro_min']:.6f}，一致した発話だけの疑似ラベル {s4_['agree_repro_min']:.6f}，"
  f"ラベルと疑似ラベルを混ぜる方法 {s4_['mix_repro_min']:.6f}．0.995 未満の（単位，K）は 4 つとも 0．1 未満の（単位，K）は今の提案 {sum(JR.values())} 組，温度つき蒸留 {sum(KR.values())} 組（内訳は 0 節） | 設計 4 節の基準（0.995） |")
w(f"| 最適化 | 勾配の最大：学び直しの段 {sci(s4_['grad_max'])}，CREMA-D の段 {sci(cd_['grad_max'])}．止まり：どちらも 0 | 1×10⁻⁵ を超えたら止める |")
w(f"| 固定のファイル | 3 つの段の記録で sha256 が同じ（{prov['frozen_sha256'][:8]}…）．今のファイルとも同じ．元の 5 つのファイルの sha256 も今と同じ．CREMA-D の点を選んだ単位の数 40．選び直しの一致は固定のファイルを作る段が確かめる（この台本は流し直していない） | 違えば止める |")
w(f"| 続きから流す条件 | 途中から流し直していない（ログの「済み」の行 {N_SKIP}）．単位のコミットは，学び直しの段 1 種類（{C_S4[:7]}），CREMA-D の段 1 種類（{C_CD[:7]}）．2 つのコミットと集計のコミットのあいだで，計算に関わるファイルの差分は 0 行 | 計算に関わるファイルに差があれば止める |")
w(f"| CREMA-D の切片のずらしの残り | 候補の上の確率の平均と目標の差の最大 {sci(cd_['adj_resid_max'])} | — |")
w(f"| CREMA-D の切片の反復 | 重みの表を探す版で，上限までに収束しなかった表の数 {cd_['wf_n_capped']} | 記録 |")
w("| 人工の数での検算（`selftest`） | `logs/dai21_selftest.log` に「selftest：通った」 | — |")
must("selftest：通った" in (ROOT / "logs" / "dai21_selftest.log").read_text(), "selftest のログ")
w("")
w("### 3.2 区間についての記録")
w("")
S_ROWS = {c: quote(REP["stats"], s_, "stats.md") for c, s_ in (
    ("AFEW", "| AFEW | jackknife，自由度 13 | 0.032 / 0.055 / 0.039 | 1.18 / 1.29 / 1.09 |"), ("IEMOCAP", "| IEMOCAP | jackknife，自由度 3 | 0.020 / 0.027 / 0.023 | 0.67 / 0.94 / 0.72 |"),
    ("MOSEI", "| MOSEI | 塊の引き直し | 0.133 / 0.123 / 0.099 | 0.51 / 0.50 / 0.41 |"), ("MELD", "| MELD | jackknife，自由度 7 | 0.114 / 0.067 / 0.084 | 1.57 / 0.66 / 0.63 |"),
    ("CREMA-D", "| CREMA-D | 話者の引き直し | 0.019 / 0.017 / 0.017 | 0.31 / 0.30 / 0.33 |"))}
S28 = quote(REP["stats"], "「10 種の見込み」は塊の項をそのまま使い，種の項を 2/10 に縮めた概算である", "stats.md")
S38B = quote(REP["stats"], "CREMA-D の推定時の答えの差（提案 − 積，adj）の半幅の見込みは 0.40 / 0.29 / 0.37 点である．", "stats.md")
NZ1 = g1["pp"] + g1["p"]
w(f"**区間の半幅：計算の前の見込みと結果**（`stud_prop − stud_prod`．K = 5 / 10 / 25．点）．見込みは，計算の前の統計の検査（`stats.md`）が実験 18 の保存の判定から出した値で（「{S28}」），差の推定値ではない．結果は，この実験の区間の（上端 − 下端）÷ 2 である．")
w("")
w("| コーパス | 見込み（`stats.md`） | 結果 |")
w("|---|---|---|")
for c in C5:
    w(f"| {c} | {S_ROWS[c].strip('| ').split(' | ')[-1]} | " + " / ".join(f"{(S_PP(c, K).hi - S_PP(c, K).lo) / 2:.2f}" for K in KS) + " |")
w("")
w(f"同じ報告は，CREMA-D の推定時の答えについて「{S38B}」と書いていた．結果の半幅は {' / '.join(f'{x:.2f}' for x in HW_PP)} 点である（要点 7）．"
  f"同じ報告は，実験 18 の値と同じ大きさの差を前提に「{RQ['S38']}」と書いていた．結果では，`stud_prop − stud_prod` の区間が 0 を含まない条件は {by_corpus(NZ1)} である（要点 1）．")
w("")
w("**区間の 2 つの項**（`effects.csv` の `v_cluster`・`v_seed`．標準誤差の 2 乗 ＝ 塊の項 ＋ K 本の引き方の項．塊の重みの引き直しで区間を作る MOSEI と CREMA-D の行）．K 本の引き方の項の割合は次のとおりである．")
w("")
w("| 差 | コーパス | K = 5 | K = 10 | K = 25 |")
w("|---|---|---|---|---|")
for a, b, c, lab in (("stud_prop", "stud_prod", "MOSEI", "学び直した融合：提案の手法の疑似ラベル − 積の疑似ラベル"), ("stud_prop", "stud_prod", CD, "同上"), ("prop", "prod", CD, "推定時の答え：提案の手法 − 積")):
    sh = []
    for K in KS:
        r = q(a, b, c, K)
        must(abs(r.se ** 2 - (r.v_cluster + r.v_seed)) < 1e-9, f"標準誤差の 2 乗が 2 つの項の和でない {a} {b} {c} K={K}")
        sh.append(f"{100 * r.v_seed / (r.v_cluster + r.v_seed):.0f}%")
    w(f"| {lab} | {c} | " + " | ".join(sh) + " |")
w("")
w("### 3.3 仮定を測る表")
w("")
D7A = quote(REP["define"], "| 主の比べ（`stud_prop` − `stud_prod`）は疑似ラベルだけが違う | 2 つの学び直しの重みの K 本の部分，合計，出発点，引き戻し | 作りの上で同じ（台本 207 行で同じ `gk`・`mval(ρ, n)`・`W0`・`b0`・`cw`・`cb`） | 作りで成り立つ（測らない） |", "define.md")
D7B = quote(REP["define"], "| 実験 18 が選んだ CREMA-D の設定（引き戻し，提案の点，学び直しの点）が 4 ソースの計算に合う | なし（CREMA-D のテストを使わずには測れない） | — | 測らない．限界（8 節 120 行）に書いてある |", "define.md")
w("設計 4 節が「計算の後に保存から出す表」に挙げた仮定の表は，CREMA-D について，計算の後の解析で出した（この節の最後の表．読み方は要点 18）．"
  "計算の前の定義の検査（`define.md`）は，仮定の表の案に次の 2 行を挙げていた（報告の表の行のまま．行の番号は第 1 版の台本と設計の番号）．")
w("")
w("| 仮定 | 測る量 | 値 | 扱い |")
w("|---|---|---|---|")
w(D7A)
w(D7B)
w("")
w("この文書が並べた記述の値のうち，仮定に関わるものは，疑似ラベルの UAR（2.4 節），CREMA-D の重みの表と格子 16 点の UAR，設定の 12 点の UAR（2.6 節）である．")
w("")
w("**CREMA-D の仮定の表**（`cremad_assumptions.csv`．計算の後の解析．50 単位の平均．基準は置かない）．量の呼び名は，実験 20 の結果の文の 3.2 節と同じである．"
  "「比べた表」は，候補の全ラベルで測った表（参照）と比べた表である．ソースの中で測った表（`home`）は残り方を掛ける前の表，ソースで測った表（`ext`）は残り方を掛けた表である．`home` と `ext` は互いに違う表で，それぞれ，50 単位のどれでも同じ値になる（単位に依らない）．"
  "**分類器が 2 つ（声と顔）なので，分類器が 3 つの元の 4 コーパスの値（実験 20 の結果の文の 3.2 節）と同じ尺度で並べない**（要点 18）．"
  "「最も強いモダリティの一致」は，でたらめに選んでも 0.5 になる．「成分の相関」「入れ替えの割合」は，自由に動く 2 個の数の上の量である．"
  "「行ごとの相関」「共通の感情の成分の相関」は 4 個の数の相関，「行の中の差の相関」は 8 個の数（自由な値は 6 個）の相関である．「作り直し」は，この台本が単位の npz から作り直して照らした列に ○ を付けた（0 節）．")
w("")
w("| 量 | 比べた表・分類器 | csv の列 | K = 5 | K = 10 | K = 25 | 作り直し |")
w("|---|---|---|---|---|---|---|")
ASM_ROWS = []
for r2_, lab_ in (("home", "ソースの中で測った表（`home`）"), ("ext", "ソースで測った表（`ext`）"), ("K", "K 本で測った表")):
    ASM_ROWS += [("成分の相関", lab_, f"intcorr_M_{r2_}", 2), ("入れ替えの割合", lab_, f"intperm_M_{r2_}", 3), ("行の中の差の相関", lab_, f"profcorr_M_{r2_}", 2),
                 ("共通の感情の成分の相関", lab_, f"emocorr_M_{r2_}", 2), ("最も強いモダリティの一致", lab_, f"topok_M_{r2_}", 2)]
ASM_ROWS += [("行ごとの相関", "ソースで測った表（`ext`）：声", "profcorr_row_A", 2), ("行ごとの相関", "ソースで測った表（`ext`）：顔", "profcorr_row_V", 2)]
for r2_, lab_ in (("home", "ソースの中で測った表（`home`）"), ("ext", "ソースで測った表（`ext`）"), ("K", "K 本で測った表"), ("orc", "全ラベルで測った表（参照）")):
    ASM_ROWS += [("水準（d′ の行の平均）", f"{lab_}：声", f"level_M_{r2_}_A", 2), ("水準（d′ の行の平均）", f"{lab_}：顔", f"level_M_{r2_}_V", 2)]
ASM_ROWS += [("残り方", "声", "ret_A", 3), ("残り方", "顔", "ret_V", 3), ("テストとのずれ", "全ラベルで測った表と，テストで測った d′", "drift_M", 3),
             ("上限の割合", "K 本で測った 8 個の d′", "dk_ceiling_share", 4), ("0 以下の割合", "K 本で測った 8 個の d′", "dk_zero_share", 4)]
must(sorted(c_ for _, _, c_, _ in ASM_ROWS) == sorted(phv["assumptions"]["from_npz"]) and len(ASM_ROWS) == 30, "仮定の表の行と csv の列")
for lab1_, lab2_, col_, nd_ in ASM_ROWS:
    w(f"| {lab1_} | {lab2_} | `{col_}` | " + " | ".join(f"{float(getattr(ASM_K[K], col_)):.{nd_}f}".replace("-", "−") for K in KS) + f" | {'○' if col_ in ASM_COLS else '—'} |")
w("")
w("作っていない量は，水準の相関，一致の余り，広がりの比，別のソースに当てた値で測った量，顔の行を除いた相関である（理由は要点 18）．仮定の表の「残り方」（`ret_A`・`ret_V`）は，2.7 節の内訳の版「微調整した声・顔」（`ret_A`・`ret_V`）とは別のものである．")
w("")

# ---- 4 節
w("## 4. 限界と，まだ出していない量")
w("")
w("### 限界（設計 0・8 節と 0 節の申告）")
w("")
w("- この実験の値は，手法を選んだときに見たテストと同じテストの上の値で，探索の結果として扱う（冒頭の 6 項）．学び直しを足すと決めたのは，実験 18 の学び直しの結果を見た後である．CREMA-D を評価するコーパスに足す案の承認は，実験 18 の CREMA-D の値を見た後である．"
  "テストの発話は，どの K 本の引き方でも，下調べのテストと同じ発話である．テストの発話が下調べと重ならないコーパスでの値は，この実験には無い．")
w("- 5 コーパスの結果は互いに独立ではない．CREMA-D は，元の 4 コーパスと並ぶ独立な確かめとして数えない．15 条件は 5 コーパス × 同じテストを共有する 3 つの K である．")
w("- 学び直しの設定（主の点）は，積の疑似ラベルのために選ばれた点を借りた点である．この決め方は提案の手法の疑似ラベルに不利な向きで，学び直しを足すと決めた時点は提案の手法に有利な向きである．2 つを合わせた向きは決まらない（設計 3.1 節）．"
  "主の点は，元の 4 コーパスと CREMA-D で違う（ρ・a）．5 コーパスの差の違いには，コーパスの違いに設定の違いが重なる（要点 1）．")
w("- 実験 1 の比較手法との差（要点 2）は，疑似ラベルの違いと学び直しの形の違いの両方を含む．相手の設定は自分のために選んだ点で，提案の手法の疑似ラベルだけがソースで測った表と K 本の作り直しを使う．"
  "内訳（要点 3）のうち，疑似ラベルだけを替えた差は，設定を積の疑似ラベルのために選んだぶん，提案の手法の疑似ラベルに不利な向きに偏る（設計 4 節）．"
  "一方，提案の手法の疑似ラベルの形と設定は同じテストの発話の上の値を見た後に決めたので，こちらは提案の手法に有利な向きである．2 つを合わせた向きは決まらない．共訓練と共正則化には内訳が無い．標準の手法（実験 2 の手法）とは並べていない．")
w("- 疑似ラベルを替えた学び直し 5 つの差（要点 5）には，設計が部品の名前を付けていない．分類器ごとの温度の較正との差には，まとめ方の違いと確率の鋭さの違いの両方が入る（設計 3.1 節）．"
  "推定時の答えの部品を外した版の内訳（要点 9，実験 20）を，学び直した融合の差の内訳として読まない．提案の部品を外した版 6 つのうち 3 つは，疑似ラベルにしていない．")
w("- CREMA-D の値は，同じ 12 文が学習側とテストの両方に出る条件での値である．文の重ならない分割での測り直しは範囲の外である．CREMA-D の設定は実験 18 の単位で選んだ値で，その単位の仮のターゲットは分類器が 3 つの元の 4 コーパスである．"
  "分類器が 2 つなので，相手の位置を元の 4 コーパスと並べて同じ性質として読まない．CREMA-D には学び直しの標準の相手が無く，CREMA-D の学び直しの値は，既にある学び直しの手法との比べではない．"
  "CREMA-D のラベルは演者に指示した感情である（実験 18 の設計 8 節）．")
w("- CREMA-D の感度の表（要点 13〜18，2.7 節，3.3 節）は，テストから文「It's eleven o'clock.」の発話を除いたときの値と，区間の塊を話者から文に替えたときの値である．"
  "**K 本と候補に同じ文が入ることの影響は測っていない．K 本からこの文を除いて学び直す試しは範囲の外で，設計が挙げた懸念（この文が K 本に入ると声の分類器の手がかりになりうる）は，この表では解けていない**．"
  "K 本で学んだ声の分類器では，この文の Neutral の再現率がほかの 11 文の平均より低く，CREMA-D のラベルを使っていない声の分類器ではそうなっていない．この並びは，懸念が予想する向きである（要点 16，2.7 節．区間は無く，原因は言えない）．"
  "候補の全ラベルで学んだ上限の声の分類器でも同じ向きで，K 本から学んだ分，学習データの全体にある偏りから学んだ分，この文の Neutral が声で当てにくい分は，この表からは分けられない．"
  "文を塊にした区間は，テストの話者の揺れを含まず，学習に現れない文への区間でもない．話者の揺れと文の揺れを合わせた区間は無い．塊は 12 個で，自由度の決め方は結果を開いた後に決まった．"
  "文ごと・強さごとの内訳には，区間が無い．文の違いと強さの違いは分けられない．仮定の表は分類器が 2 つの上の量で，元の 4 コーパスの値と同じ尺度で並べない．"
  "設計が計算の前に予告したのは表の種類で，内訳の版と並べ方は結果を開いた後に決まった．"
  "文を除いたテストの区間と話者を塊にした区間は引き直しで作るので，端が 0 に近い区間は，0 を含むかどうかが引き直しの乱数に依りうる（要点 14・15）．"
  "感度の表の値は，照合の第 3 回と第 4 回で結果ファイルと照らされ，一致した．第 4 回の判定は「通す」である．")
w("- 重みの表を乱数で探す版の値は，種を 1 つに決めて引いた表の組に条件づけた値である．表の引き方の揺れは区間に入っていない．")
w("- 区間は補正していない．要点 1〜12 の区間は，ソースの取り方，学習側の発話，手法と設定の選び方，重みの表の引き方，CREMA-D の文の取り方の揺れを含まない（CREMA-D の文を塊にした区間は，要点 15 に別に並べた）．IEMOCAP は自由度 3 で区間が広い．CREMA-D の K 本の引き方 0・1 の区間は狭めに出る．K どうしの差の区間は無い．")
w("- 記述の版（設定を疑似ラベルごとに選んだ点）は，提案の手法の疑似ラベルの点の選択にだけ，評価するコーパスのテストのラベルが間接に入る．記述だけに使う．設定の 12 点の UAR の最大は，テストの上で選んだ値で上に偏る．")
w("- 学び直しでの 3 つの量の並び（要点 12，2.4 節）は，設計の文書に無い並べ方で，結果を開いた後に並べた探索の記述である．疑似ラベルの UAR と判定が一致した割合は候補の発話の上の量で，学び直した融合の UAR はテストの発話の上の量である．仕組みについての主張には使わない．")
w("- 計算の前の検査のうち，統計の検査・定義の検査・感情認識の査読想定は，第 2 版で足した計算と第 3 版で足した集計を再検査していない．漏れの検査・比較の検査・ML の査読想定は第 3 版を見ていない．起動の台本は，DAI-21 の検査の報告ではなく，実験 20 の計算の後に使う台本の検査が検査した（0 節）．")
w("- 1 組のエンコーダ，凍結した特徴の上の線形の分類器に限る．学び直すのは融合の分類器だけで，声・顔・言葉の単独の分類器の学び直しと，引き戻しの強さの感度は範囲の外である（設計 8 節）．IEMOCAP は即興だけである．")
w("- 元の 4 コーパスの MOSEI の区間は，実験 20 の集計と実験 21 の集計で少し違う．この文書は実験 21 の集計から引いた（0 節）．")
w("")
w("### まだ出していない量")
w("")
w(f"設計 4 節は，計算の後に保存から出す表を予告していた：「{DQ_LATER}」**この 4 種類の表は，CREMA-D について，計算の後の解析で出した**（0 節．要点 13〜18，2.7 節，3.3 節）．"
  f"照合の第 2 回（`DAI/checks/DAI-21/claim_v2.md`）は，表が出る前の版について「{C2_STOP}」と記録していた．感度の表は，その後の {T_PH[:16]} に出し，第 3 版と第 4 版に入れた．"
  "照合の第 3 回の判定は「直してから通す」で，その直しを入れた第 4 版への照合の第 4 回の判定は「通す」だった．"
  f"論文のセッションへの伝言の 0 節の 10 は，照合が通った後に「{NQ10B}」という項に書き換えられ，「{NQ10C}」と書いている（0 節）．次の量は，出していない．")
w("")
w(f"- **K 本と候補から文「It's eleven o'clock.」を除いて学び直した値**．学び直しが要る．範囲の外で，出していない．感度の表は，この文をテストから除いた値である（要点 13・17）．"
  f"感情認識の査読想定（`review_affect.md`）は，この文について「{RQ['RA2']}」と書いている．")
w("- **CREMA-D で，話者の揺れと文の揺れを合わせた区間**．話者を塊にした区間と，文を塊にした区間を，別々に出した（要点 15）．2 つを合わせた区間は出していない．文の重ならない分割での測り直しは範囲の外である．")
w("- **CREMA-D の仮定の表のうち，一致の余りと広がりの比**．単位の npz から作れるが，作っていない（要点 18）．")
w("- **学び直した融合の分類器の，切片の分と向きの分**．実験 20 の推定時の答えについて出した分け方（実験 20 の結果の文の要点 9）を，学び直した分類器に当てた値である．学び直した分類器の重み（W・b）は単位の npz に保存してある．今回は出していない．"
  "学び直した分類器に切片のずらしを当てた値と，確率の較正も出していない．")
w("- **重みの表を乱数で探す版の，表の引き方の揺れ**（CREMA-D）．値は引いた表の組に条件づけた値である（要点 8）．")
w("- **感情ごとの再現率の対の差と区間**．`per_emotion.csv` は版ごとの再現率と区間を持つ．感情ごとの差の区間は作っていない（単位の保存の判定から作れる）．")
w(f"- **標準の手法（実験 2 の手法）と，学び直した融合（`stud_prop`）の並び**．並べていない（要点 2）．実験 2 の手法の UAR は実験 20 の集計（`others.csv`）にあり，単位の数は 50 と 15 で，対の差ではない．")
w("- **CREMA-D で，提案の手法と単独の分類器，提案の手法と全ラベルで測った表の対の差**．集計の差の一覧に無い（要点 9・10 は点の値の引き算だけ）．")
w("- **設定の 12 点の UAR の区間と，主の点のほかの点での差の区間**．`student_grid.csv` は平均だけを持つ（単位の保存の判定から作れる）．")
w(f"- **一致した発話だけの疑似ラベルと，ラベルと疑似ラベルを混ぜる方法が，疑似ラベルを残した発話の本数**．台本の再検査（`code_v2.md`）：「{RQ['KV4']}」")
w("- **発話ごとの対の比べ**（どの発話の判定が変わったか）．判定は保存してある．出していない．")
w("- **提案の格子の別の点の疑似ラベルでの学び直し**．学び直しが要る．設計の範囲の外にも挙がっていない．")
w(f"- **実験 18 が学び直しの点を選んだときに，同点の規則で点が決まった条件の数**．比較の検査（`compare.md`）：「{RQ['C12']}」この文書でも数えていない．")
w("- **K どうしの差**．区間は作っていない（0 節）．")
w("- **性別ごとの内訳**．AFEW と IEMOCAP でしか出せない（設計 8 節）．出していない．")
w("- **学び直しの段の GPU のメモリの最大**．記録が無く，後から出せない（0 節）．")
w("")
w("集計にあるが，この文書で引いていない量：K 本の引き方で分けた値のうち，`stud_prop − stud_prod` と CREMA-D の提案の手法 − 積のほかの差（`effects.csv` の `seeds` が `0-1`・`2-9` の行），K 本の引き方で分けた水準（`levels.csv`），"
  "学び直した融合と CREMA-D の提案の手法・積のほかの版の感情ごとの再現率（`per_emotion.csv`），混同行列（`confusion.csv`），設定の 12 点の点ごとの UAR と判定が一致した割合（`student_grid.csv`）．"
  "計算の後の解析の出力（`results/dai21_eval5/posthoc/`）にあるが，この文書で引いていない量：K 本の引き方で分けた行（`seeds` が `0-1`・`2-9`），自由度 11 の区間（`lo_n12`・`hi_n12`），"
  "2.7 節の 16 版のほかの版の UAR と感情ごとの再現率，ほかの 11 文の 1 文ごとの感情ごとの再現率（`cremad_by_sentence.csv`．この文書は，11 文の平均と，Neutral の最小〜最大を書いた），"
  "文ごとの取り違えの数（`cremad_by_sentence_confusion.csv`），単位ごとの UAR と差（`cremad_kshot_ieo_units.csv` の `uar_*`・`d_*`・`uar_noieo_*`・`d_noieo_*`）．")
w("")

# ---- 5 節
w("## 5. 数値の桁の決まり")
w("")
w("- UAR は小数 2 桁，UAR の区間の端は小数 1 桁．差は符号つきの小数 2 桁，差の区間の端は符号つきの小数 1 桁で，「差 [下端, 上端]」の形で書く．負の符号は「−」．")
w("- 差の区間の端が 0.05 より 0 に近いときは，区間が 0 を含むかどうかが読めるように桁を足す（例：+0.01）．差が丸めると ±1.00 か 0.00 になるときも，1.0 点の境目と符号が読めるように桁を足す．表の文書も同じ決まりで桁を足している．")
w("- 点の値の引き算（区間の無い差．疑似ラベルの UAR の差，感情ごとの再現率の引き算，単独の分類器との引き算，設定の 12 点の最大 − 主の点など）は，丸める前の値どうしを引いてから 1 回で丸める．表に並べた丸めた値どうしを引いた値と，0.01 違うことがある．")
w("- 読みの 3 つの分け方は，丸める前の値で決めている：区間の下端 > 0（または上端 < 0）で「区間が 0 を含まない」，差の絶対値 ≥ 1.0 で「1.0 点以上」．")
w("- 2 つの版の判定が全部の単位で同じとき，その差は「0（同じ判定）」と書く．判定は違う発話があるが，塊ごと・感情ごとの正解の数が全部の単位で同じとき，その差は「0（正解の数が同じ）」と書く．")
w("- 割合は小数 3 桁（一致の割合は小数 6 桁），検算の量は 2 桁の指数の形で書く．")
w(f"- 引き直しで作る区間（MOSEI と CREMA-D）のうち，0 に近いほうの端が 0 から標準誤差の {NEAR_TH} 倍以内にある区間は，値の後に「＊」を付ける（冒頭）．読みの 3 つの分け方と，区間が 0 を含まない条件の数えは，＊ の有無で変えない．")
w("- CREMA-D の感度の表（要点 13〜15 と 2.7 節）は，同じ差の区間どうしを並べるので，区間の端を符号つきの小数 2 桁で書く（0.00 に丸まるときは桁を足す）．同じ全発話の値を，要点 1〜12 と表の文書は区間の端を小数 1 桁で書いている．"
  "2.7 節の感情ごとの再現率の区間の端は，2.5 節と同じ小数 1 桁である．仮定の表（3.3 節）は，相関と水準を小数 2 桁，残り方・入れ替えの割合・テストとのずれを小数 3 桁，上限の割合と 0 以下の割合を小数 4 桁で書く．")
w("")
w("## 6. 変更の記録")
w("")
N_TRI_V1 = len(re.findall(r"[+−]?\d+\.\d+ \[[+−]?\d+\.\d+, [+−]?\d+\.\d+\]", V1_TEXT))
must(N_TRI_V1 == 706, f"第 1 版の「値 [下端, 上端]」の組の数：{N_TRI_V1}")
must(git("log", "--format=%h", "--diff-filter=A", "--", "DAI/docs/実験21_結果の文の下書き_2026-11-15.md").strip() == "4f1cbe0", "この文書の最初のコミット")
w("- 第 1 版（2026-11-15）：照合（gate-claim）の前の版．最初に書き出した後（コミット 4f1cbe0）に，本体が台帳と表の文書を直したので，同じ第 1 版のまま次の直しを入れて書き出し直した（コミット 8994e38）．結果ファイルから取った値と表は変えていない．")
w("   1. 表の文書についての記述（2.1 節，0 節の点検の記録 4，5 節）を，書き出し直した表の文書に合わせた．最初の書き出しでは，注の無い差が 0 の欄 3 個，K 本の引き方の説明の文，差が 0.00 に丸まる欄 8 個を，表の文書との違いとして書いていた．"
  "この 3 点は表の文書の側で直ったので，直ったことと，最初の版から変わったのが注 3 個・桁 8 個・文の 2 行だけであること（台本が欄ごとに比べた）を書いた．")
w("   2. 計算の前の漏れの検査が求めた 1 文（DAI-19 の軽い試しを見た前後）を，台帳の DAI-21 の行に本体が足した申告の文として 0 節の「見たデータ」に引いた．4 節の「まだ出していない量」から，この項を外した．")
w("   3. 0 節の「照合の前に値を使った先」に，本体の申告（2026-11-15 05:35 ごろの研究者への会話での報告の中身，覚え書きの 2 つの文を直したこと）を足した．")
w("   4. 0 節の「計算の前の検査」の起動の台本についての文を直した．最初の書き出しは「起動の台本を検査した記録は，計算の前の報告に無い」と書いていた．起動の台本の下書きの版は，実験 20 の計算の後に使う台本の検査（`DAI/checks/DAI-20/code_post_tools.md`・`code_post_tools_v2.md`）が検査していた．"
  "流した起動の台本の sha256 が，第 2 回が検査した版と同じであることを台本が確かめた．4 節の限界の同じ記述も直した．")
w("")
w(f"- 第 2 版（2026-11-15）：照合の第 1 回（`DAI/checks/DAI-21/claim.md`．判定は「直してから通す」．報告は数値の食い違いを挙げていない．指摘 {N_CLAIM1} 件）の直しを入れた．"
  f"結果ファイルから取った値は第 1 版から変えていない．第 1 版（コミット 8994e38）の「値 [下端, 上端]」の形の {N_TRI_V1} 組は，この版に全部残っている（台本が確かめた）．")
w("   1. 0 節の起動の台本の文で，流した起動の台本の sha256 の値が埋まっていなかった箇所を直した．書き出した文に，埋めていない書式が残っていれば台本が止まるようにした（指摘 1）．")
w("   2. 0 節の「照合の前に値を使った先」に，論文のセッションへの伝言（コミット c6d5e2e．「照合中」の印つきで実験 21 の値を書いた）を足した．伝言の値がこの文書の値と同じ文字であることを台本が確かめる．"
  "この結果の値の文字が出てくる文書を台本が探し，見つかった文書を書くようにした．伝言の最後のコミットが結果を開く前であることを求めていた台本の照合を外した（指摘 2）．")
w("   3. 2.6 節に，選んだ学び直しの設定の点が格子の端に当たった数を，ρ と a の両方について疑似ラベルごとに書いた（台本が固定のファイルと `student_grid.csv` から数えた）．格子が狭いと提案の手法の疑似ラベルの側が不利になるという比較の検査の文を，要点 1 と 2.6 節に引いた．"
  "0 節の「求めと，この文書の対応」の一覧に，この項を足した（指摘 3）．")
w("   4. 要点 3 の「同じ手法の中で疑似ラベルだけを替えた差」と 4 節の限界に，提案の手法に有利な向き（提案の手法の疑似ラベルの形と設定を，同じテストの上の値を見た後に決めた）と，2 つを合わせた向きが決まらないことを書いた．"
  "「同じ疑似ラベルで学び直しの形を替えた差」には，この有利な向きが両側に入ることを書いた（指摘 4）．")
w("   5. 0 節の「区間の作り方」に，統計の検査が求めた区間の読み方を足した：IEMOCAP の K 本の引き方の項が 0 でも，その揺れは除かれた分割を通して塊の項に入ること（実験 20 の設計と点検の報告の文），MELD の分割 4 の塊が 2 個であること（台本が単位の npz で数えた）．一覧にもこの項を足した（指摘 5）．")
w("   6. 0 節の「CREMA-D の事情」で，CREMA-D の利用条件が確かめてあること（感情認識の査読想定の報告と設計 8 節）を引き，まだ確かめていないのは RAVDESS だけと書いた（指摘 6）．")
w("   7. 要点 12 の「15 条件のどれでも符号は同じ」を，「どちらも正の条件の数と，どちらも負の条件の数」に書き換えた．候補の発話が学び直した分類器の学習データで，疑似ラベルの UAR・判定が一致した割合・候補の上の UAR が学習データの上の量であることを，要点 12 と 2.4 節に書いた（指摘 7）．")
w("   8. 要点 2 と 8 の太字の文に，区間の読みの内訳（区間が 0 を含まず高い数，区間が 0 を含む数，そのうち点推定が負の数．要点 2 は IEMOCAP の内訳も）を同じ文で書いた．要点 3 と 5 の「ほかの…個」にも同じ内訳を書いた．数は台本が集計から数えた（指摘 8）．")
w("   9. 要点 7 と 10 に，設計 4 節が予告した CREMA-D の感度の表（文「It's eleven o'clock.」を除いた値，文を塊にした区間）がまだ出ていないことを書いた．4 節に，論文で CREMA-D の値を使う前に出す量として残すことを書いた（指摘 9）．")
w("   10. 要点 5 で，「〜の分」と呼ばないと書いた直後に置いた比較の再検査の引用に，引用の中の名前が報告の呼び名で，この文書は使わないことを書いた．2.1 節を，本体が書き出し直した表の文書（まとめの表の下の注，1 行に並んでいた 2 種類の差を 2 行に分けた）に合わせた（指摘 10）．")
w("")
N_TRI_V2 = len(re.findall(r"[+−]?\d+\.\d+ \[[+−]?\d+\.\d+, [+−]?\d+\.\d+\]", V2_TEXT))
w(f"- 第 2 版への追記（2026-11-15）：照合の第 2 回（`DAI/checks/DAI-21/claim_v2.md`．判定は「通す」．新しい指摘 {N_CLAIM2} 件は，論文のセッションへの伝言の書き方についての記録）の後に，次の 3 点を直した．ほかの文と値は変えていない．"
  f"照合の第 2 回が見た版（コミット a020495）の「値 [下端, 上端]」の形の {N_TRI_V2} 組は，この版に全部残っている（台本が確かめた）．")
w("   1. 題と冒頭を「照合済み」にし，照合の第 2 回の判定と新しい指摘の数を書いた．")
w("   2. 0 節の「照合の前に値を使った先」の伝言の記述に，照合の第 2 回の後に本体が伝言をもう一度直したこと（コミット 8772450）と，直した中身 4 つを足した．伝言の 18 個の値と区間の書き方が変わっていないことを台本が確かめる．")
w("   3. 4 節の，設計が予告した CREMA-D の感度の表の文を直した．照合の第 2 回が見た版は「論文で CREMA-D の値を使う前に出す量として残す．いつ出すかは決まっていない」と書いていた．本体の予定（論文や資料で CREMA-D の値を使う前に出す）と，照合の第 2 回の記録を書いた．")
w("")

V2B_TEXT = git("show", "3fd9f2a:DAI/docs/実験21_結果の文の下書き_2026-11-15.md")           # 照合済みにした第 2 版（題と冒頭，照合の後の記録 2 つを足した版）
N_TRI_V2B = len(re.findall(r"[+−]?\d+\.\d+ \[[+−]?\d+\.\d+, [+−]?\d+\.\d+\]", V2B_TEXT))
must("第 2 版．照合済み：gate-claim 第 2 回 `claim_v2.md` 通す" in V2B_TEXT.split("\n")[0] and N_TRI_V2B == N_TRI_V2, f"照合済みにした第 2 版（コミット 3fd9f2a）の題と，値と区間の組の数：{N_TRI_V2B}")
N_TRI_V3 = len(re.findall(r"[+−]?\d+\.\d+ \[[+−]?\d+\.\d+, [+−]?\d+\.\d+\]", V3_TEXT))
must(N_TRI_V3 == 1617, f"照合の第 3 回が見た第 3 版の「値 [下端, 上端]」の組の数：{N_TRI_V3}")
w(f"- 第 3 版（2026-11-15．コミット 8187af5）：設計 4 節が予告した CREMA-D の感度の表（計算の後の解析．`results/dai21_eval5/posthoc/`）の値と読み方を足した．照合の第 3 回が見た版である．"
  f"第 2 版の値と表は変えていない．照合済みにした第 2 版（コミット 3fd9f2a）の「値 [下端, 上端]」の形の {N_TRI_V2B} 組は，第 3 版に全部残っていた．第 3 版で増えた組は {N_TRI_V3 - N_TRI_V2B} 組で，どれも要点 13〜15 と 2.7 節にあった．"
  "増えた組は，文を除いたテストの値，「除いた値 − 全発話の値」，文を塊にした区間と，それらと並べるために区間の端を小数 2 桁で書き直した全発話の値である．")
w("   1. 題と冒頭を第 3 版にし，足した箇所と，足した値がまだ照合を受けていないことを書いた．")
w("   2. 0 節に「計算の後の解析」を足した：台本と出力の出どころ，台本の検査の 2 回の判定，台帳の記録，結果を開いた後に決まった点，この台本が単位の npz から作り直して照らした量と，作り直していない量．")
w("   3. 0 節の「照合の前に値を使った先」に，伝言のコミット 45a7501（感度の表を出した後に，本体が 0 節の 10 を書き換えた）と，感度の表の値についての (5) を足した．")
w("   4. 要点 13〜18 を足した：文を除いたテストの差（13），集計の段の差の全部での「除いた値 − 全発話の値」の区間（14），文を塊にした区間（15），文ごと・強さごとの内訳（16），K 本の中の文の本数（17），仮定の表（18）．")
w("   5. 2.7 節を足した（除いた発話の数，78 対の差の表，16 版の UAR，文を除いたテストの感情ごとの再現率，文ごとの内訳 2 つ，強さごとの内訳，K 本の中の文の本数，K 本の引き方を文の本数で 2 つに分けた並び）．3.3 節に CREMA-D の仮定の表を足した．")
w("   6. 要点 7・10，2.6 節，3.3 節，4 節にあった「感度の表はまだ出していない」という文を直した．4 節の限界に感度の表の限界を足した．4 節の「まだ出していない量」から，出した 4 種類の表を外し，"
  "学び直しが要る試し，話者と文を合わせた区間，仮定の表の作っていない量を足した．0 節の求めの一覧の 2 行（文「It's eleven o'clock.」についての求めと，表の相関についての求め）の対応を直した．")
w("   7. 2 節の表の下の注と 5 節に，感度の表の区間の端の桁と，内訳に区間が無いことを足した．")
w("")
w(f"- 第 4 版（2026-11-15）：照合の第 3 回（`DAI/checks/DAI-21/claim_v3.md`．判定は「直してから通す」．報告は，足した値・数え・区間がすべて出力の csv と一致したと書いている．指摘 {N_CLAIM3} 件．どれも重さは「記録する」）の直しを入れた．"
  f"第 3 版の値は変えていない．第 3 版（コミット 8187af5）の「値 [下端, 上端]」の形の {N_TRI_V3:,} 組は，この版に全部残っている（台本が確かめた）．＠第4版で増えた組＠")
w("   1. 要点 16 の太字の文の範囲を狭めた．第 3 版は，UAR（4 感情の平均）の並びについて「この並びは，K 本から学んだものではない」と書いていた．"
  "第 4 版は，CREMA-D のラベルを使っていない声の分類器（`start_A`）の UAR についてだけそう書き，K 本で学んだ版の並びのどれだけが K 本から来たかは分からないと書いた．"
  "「K 本で学んだ声の分類器の引き算の絶対値は `start_A` より小さい」という文を外した．"
  "設計の懸念に直接当たる量として，文「It's eleven o'clock.」の Neutral の再現率を，この文とほかの 11 文に分けて要点 16 に書き，2.7 節に 16 版の表（Happy・Sad・Angry の再現率も並べた）を足した．"
  "K 本で学んだ声の分類器ではこの文の Neutral の再現率がほかの 11 文の平均より低く，`start_A` ではそうなっていないこと，この並びが懸念の予想する向きであること，区間が無く原因は言えないことを書いた．"
  "4 節の限界と，0 節の求めの一覧の 1 行にも，この並びを足した．値は出力の csv から取り，単位の npz から作り直した値と照らした（指摘 1）．")
w("   2. 要点 14 に，「除いた値 − 全発話の値」の区間が 0 を含まない 4 個のうち 1 個（今の提案 − 積の K = 5）は区間の端が 0 に近く，照合の第 3 回の検算では別の乱数で 0 を含んだことを書いた（要点 13 の下位の項にも 1 文足した）．"
  "区間の端が 0 に近いと数える基準（0 に近いほうの端が，0 から標準誤差の 0.1 倍以内）を決め，要点 15 に，話者を塊にした区間のうち端が 0 に近いものの数と名前，0 を含むかどうかが 2 つの区間で違う 12 個のうちの数を書いた．"
  "2 節の表の下の注と 4 節の限界にも 1 文ずつ足した（指摘 2）．")
TIES = []                                                                    # 外した並びの，分けた境目で本数が同じだった分割の数（第 3 版に書いた数を，記録のために数え直す）
for K in KS:
    t_ = 0
    for f in range(5):
        od_ = sorted(range(10), key=lambda r_: (int(KIEO[(K, f, r_)].sum()), r_))
        t_ += int(KIEO[(K, f, od_[4])].sum() == KIEO[(K, f, od_[5])].sum())
    TIES.append(t_)
must(f"5 分割のうち {kjoin(TIES)} 個である" in V3_TEXT, f"外した並びの，境目で本数が同じ分割の数が第 3 版の文と違う：{TIES}")
w("   3. 要点 17 と 2.7 節から，K 本の引き方を文の本数で 2 つに分けた並び（文と表）を外した．K 本の中の文の本数の表は残した．"
  f"外した理由は 4 つである：分け方を結果を開いた後に決めた，区間が無い，分けた境目で本数が同じ分割がある（K = 5・10・25 の順に 5 分割のうち {kjoin(TIES)} 個），"
  "分ける量（K 本の中の文の総数）が設計の懸念の量（Neutral の K 本にこの文が入るか）と違う（指摘 3）．")
w("   4. 要点 18 と 3.3 節の「ソースで測った 2 つの表は，50 単位で同じ値である」を，`home` と `ext` のそれぞれが 50 単位のどれでも同じ値になる（互いには違う表である）という文に直した（指摘 4）．")
w("   5. 0 節の「照合の前に値を使った先」の (5) に，本体の申告（08:00 ごろと 08:40 ごろに研究者へ会話で伝えた中身，要点 16 の読みの訂正を研究者に伝えること，ほかの文書には書いていないこと）を書いた．"
  "感度の表の値の文字をほかの文書から探した結果も書いた（指摘 5）．")
w("   6. 題と冒頭，照合の状態を書いた文（要点 13〜18 の前書き，2.7 節の見出しと前書き，2 節の表の下の注，3.3 節の表の見出し，4 節）を第 4 版に合わせた．"
  "0 節の「計算の後の解析」に照合の第 3 回の項を足し，この解析を見た報告を 3 本にした．")
w("   7. 第 4 版への追記（本体の指示）：指摘 2 の基準を，引き直しで作る区間の全部（元の 4 コーパスでは MOSEI．CREMA-D は全部）に当てた．基準に当たる区間の値の後に ＊ の印を付け，冒頭，2 節の表の下の注，5 節に印の説明を書いた．"
  "要点 1 に MOSEI の K = 10（提案の手法の疑似ラベル − 積の疑似ラベル），要点 5 に MOSEI の K = 25（提案の手法の疑似ラベル − K 本だけで測った表の疑似ラベル）と，同じ項の CREMA-D の区間について，"
  "「区間の端が 0 に近く，引き直しの乱数の組に依りうる」という注を書いた．要点 14 に，文を除いたテストの区間で端が 0 に近いものの数を足した．"
  "台本は，集計の引き直しの区間のうち基準に当たるものを全部数え，要点 1〜12 が値を書いた箇所の全部に ＊ が付いていること，基準に当たらない区間には付いていないことを確かめる．"
  "読みと値は変えていない．＊ が付いた箇所は，要点 1〜12 で ＠要点1〜12の＊＠ 箇所，文書の全体で ＠全体の＊＠ 箇所である．")
w("")
w(f"- 第 4 版の仕上げ（2026-11-15）：第 4 版（コミット 8f46972）への照合の第 4 回（`DAI/checks/DAI-21/claim_v4.md`．判定は「通す」．残る点 {N_CLAIM4} 件．どちらも重さは「記録する」）の後に，次を直した．"
  "ほかの文と値は変えていない．照合の第 4 回が見た第 4 版の「値 [下端, 上端]」の形の 1,623 組は，この版に全部残っていて，増えた組は無い（台本が確かめた）．")
w("   1. 題と冒頭を「照合済み」にし，照合の第 4 回の判定と，残る点の数を書いた．0 節の「計算の後の解析」に，照合の第 4 回の項を足した．")
w("   2. 要点 16 の「この並びは，設計の懸念が予想する向きである」の直後に，上限の声の分類器（`full_A`）についての文を置いた（照合の第 4 回の記録 1）．"
  "上限の声の分類器は候補の全ラベルで学ぶので K 本の引き方の偏りを持たないが，それでも文「It's eleven o'clock.」の Neutral の再現率がほかの 11 文の平均より低いこと，"
  "候補の発話にもこの文の Neutral が少ないという偏りがあること（実験 18 の設計 8 節の文を引き，単位の npz で数えた割合を書いた），"
  "この表からは，K 本に入ったこの文の発話から学んだ分，学習データの全体の偏りから学んだ分，この文の Neutral が声で当てにくい分を分けられないことを書いた．"
  "第 4 版では，上限の声の分類器の値を，太字の文の後の別の文として書いていた．値は同じである．4 節の限界にも 1 文足した．")
w("   3. 0 節の (5) に，本体の申告を足した：本体が 08:50 ごろと 09:10 ごろに研究者へ会話で訂正を伝えたことと，その中身（照合の第 4 回の記録 2）．")
w("   4. 0 節の「照合の前に値を使った先」の (4) に，伝言のコミット 6687708（照合の第 4 回が通った後に，本体が 0 節の 10 を「CREMA-D の値を入れるときは感度の表と一緒に書く」という項に書き換えた）を足した．"
  "伝言の版ごとの照合を，その版の文で比べる形に直した（8772450・45a7501 の版にあった文は，その版から引く）．")
w("   5. 照合の状態を書いた文（要点 13〜18 の前書き，2.7 節の前書き，2 節の表の下の注，4 節）を，照合の第 4 回が通ったことに合わせた．"
  "4 節の「まだ出していない量」の前書きの，伝言の引用を今の伝言の文に替えた．")
w("")

# ---- 書き出す前の検査（句読点，使わないと決めた語，符号の読めない区間の端）
text = "\n".join(L) + "\n"
# 第 1 版（照合の第 1 回が見た版）にあった「値 [下端, 上端]」の組が，第 2 版に全部残っていること（値を変えていないことの確かめ）
TRI = re.compile(r"[+−]?\d+\.\d+ \[[+−]?\d+\.\d+, [+−]?\d+\.\d+\]")
tri_old, tri_new = TRI.findall(V1_TEXT), TRI.findall(text)
gone_ = [x for x in set(tri_old) if tri_old.count(x) > tri_new.count(x)]
must(len(tri_old) == N_TRI_V1 and gone_ == [], f"第 1 版にあった値と区間の組が減っている：{gone_[:5]}（第 1 版 {len(tri_old)} 組，この版 {len(tri_new)} 組）")
tri_v2 = TRI.findall(V2_TEXT)
gone2_ = [x for x in set(tri_v2) if tri_v2.count(x) > tri_new.count(x)]
must(len(tri_v2) == N_TRI_V2 and gone2_ == [], f"照合の第 2 回が見た版の値と区間の組が減っている：{gone2_[:5]}（{len(tri_v2)} 組，この版 {len(tri_new)} 組）")
tri_v2b = TRI.findall(V2B_TEXT)
gone3_ = [x for x in set(tri_v2b) if tri_v2b.count(x) > tri_new.count(x)]
must(len(tri_v2b) == N_TRI_V2B and gone3_ == [] and len(tri_new) > len(tri_v2b), f"照合済みにした第 2 版の値と区間の組が減っている：{gone3_[:5]}（{len(tri_v2b)} 組，この版 {len(tri_new)} 組）")
tri_v3 = TRI.findall(V3_TEXT)
gone4_ = [x for x in set(tri_v3) if tri_v3.count(x) > tri_new.count(x)]
must(len(tri_v3) == N_TRI_V3 and gone4_ == [], f"照合の第 3 回が見た第 3 版の値と区間の組が減っている：{gone4_[:5]}（{len(tri_v3)} 組，この版 {len(tri_new)} 組）")
must(text.count("＠第4版で増えた組＠") == 1, "増えた組の数を入れる場所")
tri_v4 = TRI.findall(V4_TEXT)
gone5_ = [x for x in set(tri_v4) if tri_v4.count(x) > tri_new.count(x)]
must(len(tri_v4) == 1623 and gone5_ == [] and len(tri_new) == len(tri_v4), f"照合の第 4 回が見た第 4 版と，値と区間の組が違う：{gone5_[:5]}（{len(tri_v4)} 組，この版 {len(tri_new)} 組）")
# 増えた組が，要点 13〜18 の節と 2.7 節にだけあること（ほかの節の値と区間の組は，照合済みにした第 2 版と同じ数）
i13_, i2_, i27_, inote_ = (text.index(x) for x in ("\n### CREMA-D の感度の表（設計 4 節が予告した表．", "\n## 2. 表\n", "\n### 2.7 CREMA-D の感度の表（", "\n### 表の下の注（2 節に共通）\n"))
must(i13_ < i2_ < i27_ < inote_, "感度の表の節の並び")
tri_out_ = TRI.findall(text[:i13_] + text[i2_:i27_] + text[inote_:])
extra_ = sorted(x for x in set(tri_out_) for _ in range(tri_out_.count(x) - tri_v2b.count(x)))      # 感度の表の節のほかで増えた組：0 節の (5) の探した値の文字と，要点 1・5 の ＊ の注に書いた値
NOTE_STAR = [TRI.search(ci(S_PP(c, K), star=False)).group(0) for c, K in NEAR1] + [TRI.search(ci(q(*mk_(t), c, K), star=False)).group(0) for sel_, mk_ in ((NEAR5P, MK_P), (NEAR5X, MK_X)) for t, c, K in sel_]
must(extra_ == sorted([PROBE3] + NOTE_STAR) and not [x for x in set(tri_v2b) if tri_v2b.count(x) > tri_out_.count(x)],
     f"感度の表の節のほかに，値と区間の組が増えている：{len(tri_out_)} 組（照合済みにした第 2 版 {len(tri_v2b)} 組）．増えた組：{extra_}")
# ＊ の印：値を書いた区間のうち，引き直しで作る区間で端が 0 に近いものの全部に付いていて，ほかには付いていないこと
STAR = re.compile(r"\][↑↓]?＊")
must(len(STAR.findall(text)) == len(NEAR_USED) and len(NEAR_USED) > 0, f"＊ の印の数（{len(STAR.findall(text))}）が，＊ を付けて書いた区間の数（{len(NEAR_USED)}）と違う")
i1_ = text.index("\n## 1. 要点\n")
must(i1_ < i13_, "要点 1〜12 の範囲")
R12 = text[i1_:i13_]                                                          # 要点 1〜12（1 節の初めから，感度の表の節の前まで）
flag_: dict = {}
for r in list(ef.itertuples()) + list(ef20.itertuples()):                     # 書きうる差の行（実験 21 と実験 20 の集計の全部の行）
    if klass(r) not in ("same", "cnt"):
        flag_.setdefault(f"{fest(r.est)} [{fend(r.lo)}, {fend(r.hi)}]", set()).add(near0(r))
AMB = sorted(s_ for s_, f_ in flag_.items() if len(f_) == 2 and s_ in R12)   # 同じ文字で，基準に当たる行と当たらない行があるもの
N_STAR_OK = 0
for s_, f_ in flag_.items():
    if len(f_) == 2 or s_ not in R12:
        continue
    n_all_, n_star_ = R12.count(s_), len(re.findall(re.escape(s_) + r"[↑↓]?＊", R12))
    must(n_star_ == (n_all_ if f_ == {True} else 0), f"要点 1〜12 の ＊ の印が，基準と合わない：{s_}（書いた箇所 {n_all_}，＊ {n_star_}，基準に当たるか {f_}）")
    N_STAR_OK += n_star_
must(AMB == [] and N_STAR_OK == len(STAR.findall(R12)) and N_STAR_OK > 0, f"要点 1〜12 の ＊ の印の照合：文字が重なる行 {AMB}，照合した ＊ {N_STAR_OK}，要点 1〜12 の ＊ {len(STAR.findall(R12))}")
must(text.count("＠要点1〜12の＊＠") == 2 and text.count("＠全体の＊＠") == 1, "＊ の数を入れる場所")
text = text.replace("＠要点1〜12の＊＠", str(N_STAR_OK)).replace("＠全体の＊＠", str(len(NEAR_USED)))
i14_, i16_ = text.index("\n14. **集計の段の差の全部"), text.index("\n16. **文ごと・強さごとの内訳")
must(i13_ < i14_ < i16_ < i2_ and len(TRI.findall(text[i16_:i2_])) == 0, "要点 16〜18 に，値と区間の組がある（6 節の文を直す）")
must(len(tri_new) - len(tri_v3) == 1 + len(NOTE_STAR), f"第 4 版で増えた値と区間の組の数（6 節の文を直す）：{len(tri_new) - len(tri_v3)}")
text = text.replace("＠第4版で増えた組＠", f"この版で増えた組は {1 + len(NOTE_STAR)} 組で，0 節の (5) に書いた，ほかの文書から探した値の文字の 1 組（要点 13 にある値と同じ値）と，"
                    f"要点 1・5 の ＊ の注に書いた {len(NOTE_STAR)} 組（集計の値で，読みは変えていない）である．")
# 感度の表を足す前の行（照合済みにした第 2 版の行）のうち，この版に無い行を数える（直した文の数の確かめ）
old_lines_, new_set_ = V2B_TEXT.split("\n"), set(text.split("\n"))
CHANGED_OLD = [x for x in old_lines_ if x not in new_set_]
print(f"照合済みにした第 2 版の {len(old_lines_)} 行のうち，この版に無い行（直した行）：{len(CHANGED_OLD)} 行．値と区間の組：{len(tri_v2b)} → {len(tri_new)}")
must("、" not in text and "。" not in text, "句読点に「、」「。」がある")
body = text
for s_ in sorted(set(QUOTED), key=len, reverse=True):                     # 元の文書から引いた文は，語の検査から除く
    body = body.replace(s_, "")
for ok_ in ("再現率", "今の提案の再現", "3 手法の再現", "温度つき蒸留の再現", "混ぜる方法の再現", "4 つの再現の割合", "再現の基準", "再現の検算", "再現の割合", "再現の確かめ"):
    body = body.replace(ok_, "")                                          # 「再現率」は指標の名前．「今の提案の再現」「再現の基準」は設計 4・5 節の検算の名前．「再現の確かめ」は表の文書の行の名前
for bad in ("セル", "当たり", "外れ", "有意", "勝", "負け", "差が無い", "差がない", "差は無い", "差はない", "K で増え", "K で減", "K が大きいほど", "K とともに",
            "再現", "確認", "効い", "効く", "効か", "効き", "上回", "下回", "生徒", "教師", "先生", "出発点", "明らか", "顕著", "劇的"):
    must(bad not in body, f"使わないと決めた語がある：{bad}（…{body[max(0, body.find(bad) - 40):body.find(bad) + 30]}…）")
m_ = re.search(r"\[[+−-]0\.0, |, [+−-]0\.0\]", text.replace("「+0.00 [+0.0, +0.0]」", ""))      # 表の文書の欄の書き方を引いた 1 か所を除く
m2_ = re.search(r"\{[A-Za-z_][A-Za-z0-9_\[\]:.'()]*\}", text)
must(m2_ is None, f"埋めていない書式がある（…{text[max(0, (m2_.start() if m2_ else 0) - 60):(m2_.end() if m2_ else 0) + 20]}…）")
must(m_ is None, f"符号の読めない区間の端がある（…{text[max(0, (m_.start() if m_ else 0) - 120):(m_.end() if m_ else 0) + 20]}…）")
Path(args.out).write_text(text)
print(f"書いた：{args.out}（{len(L)} 行）")
