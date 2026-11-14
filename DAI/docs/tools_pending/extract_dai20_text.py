"""実験 20（DAI-20．提案の手法の評価の回．4 コーパス）の結果の文の下書きを，結果ファイルから数値を取り出して書き出す．

出力：DAI/docs/実験20_結果の文の下書き_2026-11-15.md（--out で別の場所に書ける）．
読むもの：results/dai20_eval/summary/*.csv・convergence.json，provenance.json，provenance_log.jsonl，units/*.csv・*.npz（作り直しの照合），
          posthoc/intercept_direction.csv・intercept_levels.csv・intercept_provenance.json（`dai20_posthoc_intercept.py` が書いた），
          logs/dai20_eval.log・dai20_eval.done・dai20_selftest.log，DAI/frozen/dai20_frozen.json，
          設計 DAI/docs/実験20_設計.md・点検の報告 DAI/checks/DAI-20/*.md・台帳 docs/seen-data.md・DAI/docs/論文のセッションへの伝言_2026-11-14.md
          （引いた文がそこにあることを確かめるだけ．code_post_tools_v3.md は，あれば判定を引く），
          表の文書 DAI/docs/実験20_結果の表_2026-11-15.md と書き出しの台本 md_tables_dai20.py（版の呼び名をそろえる．表の文書が今の集計と合うことを確かめる），
          git の記録（コミットの時刻と差分．表の文書の前の版．実験 21 の設計・台本・固定のファイルは，最後のコミットの時刻だけを見る）．
結果ファイルは読むだけで，書き換えない（results/dai20_eval/ には何も書かない．足した出力のファイルも無い）．GPU は使わない．
DAI/exp の台本は import しない（dai20_eval.py を文字として読み，監視対象の一覧があることを確かめるだけ）．

文の中の数値と表は，すべてこの台本が結果ファイルから埋める（手で写さない）．
文が言う数えと大小は `must(...)` で結果ファイルと照らし，合わなければ止まる．
集計の値は，単位の保存（units の csv と npz）から別の書き方で作り直した値と照らしてから使う（照合を恒等式にしない）．
設計・点検の報告・台帳から引く文は，元の文書（台帳はその行）に同じ文があることを確かめる．
文を直すときは，この台本の中の文を直して流し直す．

使い方：.venv/bin/python DAI/docs/tools_pending/extract_dai20_text.py
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
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                    # GPU を使わない
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path("/home/snics/MER-DA")
TOOLS = ROOT / "DAI" / "docs" / "tools_pending"
sys.path.insert(0, str(TOOLS))
import md_tables_dai20 as MT  # noqa: E402   版の呼び名（表の文書と同じ名前を使う）．読み込むだけ

R = ROOT / "results" / "dai20_eval"
SM, PHDIR, UDIR = R / "summary", R / "posthoc", R / "units"
OUT = ROOT / "DAI" / "docs" / "実験20_結果の文の下書き_2026-11-15.md"
TABLE_DOC = ROOT / "DAI" / "docs" / "実験20_結果の表_2026-11-15.md"
DESIGN = ROOT / "DAI" / "docs" / "実験20_設計.md"
DESIGN16 = ROOT / "DAI" / "docs" / "実験16_下調べ_設計.md"
CK = ROOT / "DAI" / "checks" / "DAI-20"
LEDGER = ROOT / "docs" / "seen-data.md"
FROZEN = ROOT / "DAI" / "frozen" / "dai20_frozen.json"
SCRIPT20 = ROOT / "DAI" / "exp" / "dai20_eval.py"
POSTHOC_PY = TOOLS / "dai20_posthoc_intercept.py"
TABLES_PY = TOOLS / "md_tables_dai20.py"
C4 = list(MT.C4)
KS = list(MT.KS)
ALL12 = [(c, K) for c in C4 for K in KS]
EMO = ["Neutral", "Happy", "Sad", "Angry"]
MODS = ["A", "V", "T"]
MOD_JP = {"A": "声", "V": "顔", "T": "言葉"}
FLOOR = 1.0                                                               # 表の印の差の下限（設計 5 節）
SEEDS = {"all": None, "0-1": (0, 1), "2-9": tuple(range(2, 10)), "0-2": (0, 1, 2)}


def must(cond, msg: str) -> None:
    if not cond:
        raise SystemExit(f"文と結果ファイルが合わない：{msg}")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mtime(p: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


QUOTED: list[str] = []                                                    # 元の文書から引いた文（使わないと決めた語の検査から除く）


def quote(text: str, s: str, where: str) -> str:
    must(s in text, f"{where}に次の文が無い：{s}")
    QUOTED.append(s)
    return s


# ------------------------------------------------------------------ 読み込み

ef = pd.read_csv(SM / "effects.csv", low_memory=False, dtype={"folds_dropped": str})
for col_ in ("mark", "a_same_as", "b_same_as", "folds_dropped"):
    ef[col_] = ef[col_].fillna("").astype(str)
lv = pd.read_csv(SM / "levels.csv", low_memory=False)
lv["same_as"] = lv["same_as"].fillna("").astype(str)
pe = pd.read_csv(SM / "per_emotion.csv", low_memory=False)
ot = pd.read_csv(SM / "others.csv", low_memory=False)
gl = pd.read_csv(SM / "grid_levels.csv")
eg = pd.read_csv(SM / "est_grid.csv")
wt = pd.read_csv(SM / "weights.csv")
cfm = pd.read_csv(SM / "confusion.csv")
cfit = pd.read_csv(SM / "competitor_fit.csv")
asm = pd.read_csv(SM / "assumptions.csv")
stab = pd.read_csv(SM / "source_tables.csv")
conv = json.load(open(SM / "convergence.json"))
prov = json.load(open(R / "provenance.json"))
plog = [json.loads(x) for x in open(R / "provenance_log.jsonl")]
phd = pd.read_csv(PHDIR / "intercept_direction.csv")
phl = pd.read_csv(PHDIR / "intercept_levels.csv")
phprov = json.load(open(PHDIR / "intercept_provenance.json"))
frozen = json.load(open(FROZEN))
design = DESIGN.read_text()
design16 = DESIGN16.read_text()
ledger = LEDGER.read_text()
code_post = (CK / "code_post.md").read_text()
tools_v1 = (CK / "code_post_tools.md").read_text()
tools_v2 = (CK / "code_post_tools_v2.md").read_text()
table_doc = TABLE_DOC.read_text()
dai_rules = (ROOT / "DAI" / "CLAUDE.md").read_text()
log_eval = (ROOT / "logs" / "dai20_eval.log").read_text()
done = (ROOT / "logs" / "dai20_eval.done").read_text().strip()

EF = {(r.a, r.b, r.c, int(r.K), r.seeds): r for r in ef.itertuples()}
LV = {(r.version, r.c, int(r.K), r.seeds): r for r in lv.itertuples()}
PHD = {(r.version, r.quantity, r.layer, r.c, int(r.K), r.seeds): r for r in phd.itertuples()}
PHL = {(r.version, r.layer, r.c, int(r.K), r.seeds): r for r in phl.itertuples()}

# ------------------------------------------------------------------ 数の書き方


def fest(x: float) -> str:
    """差：小数 2 桁．丸めると ±1.00 か 0.00 になって，1.0 点の境目や符号が読めないときは桁を足す．"""
    s = f"{x:+.2f}"
    if (abs(float(s)) == 1.0 and abs(x) != 1.0) or (float(s) == 0.0 and x != 0.0):
        for nd in (3, 4, 5, 6):
            s = f"{x:+.{nd}f}"
            if abs(float(s)) not in (1.0, 0.0):
                break
    return s.replace("-", "−")


def fend(x: float) -> str:
    """区間の端：小数 1 桁．0.05 より 0 に近いときは符号が読めるよう桁を足す．"""
    s = f"{x:+.1f}"
    if abs(x) < 0.05 and x != 0:
        for nd in (2, 3, 4, 5):
            s = f"{x:+.{nd}f}"
            if float(s) != 0.0:
                break
    return s.replace("-", "−")


def klass(q) -> str:
    """差の読み：pp・nn は区間が 0 を含まず差が 1.0 点以上，p・n は区間が 0 を含まず差が 1.0 点未満，z は区間が 0 を含む，same は正解の数が全部の単位で同じ．"""
    if bool(getattr(q, "same_counts", False)):
        return "same"
    if q.lo > 0:
        return "pp" if q.est >= FLOOR else "p"
    if q.hi < 0:
        return "nn" if q.est <= -FLOOR else "n"
    return "z"


def ci(q, mark: bool = False) -> str:
    if klass(q) == "same":
        return "0（同じ数）"
    m = {"pp": "↑", "nn": "↓"}.get(klass(q), "") if mark else ""
    return f"{fest(q.est)} [{fend(q.lo)}, {fend(q.hi)}]{m}"


def lci(r) -> str:
    return f"{r.uar:.2f} [{r.lo:.1f}, {r.hi:.1f}]"


def sci(x: float) -> str:
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    sup = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m}×10{sup}"


def nm(v: str) -> str:
    return MT.NAME.get(v.split("@")[0], v.split("@")[0])


def nmc(v: str) -> str:
    return f"{nm(v)}（`{v}`）"


# ------------------------------------------------------------------ 取り出し

def q(a: str, b: str, c: str, K: int, seeds: str = "all"):
    must((a, b, c, K, seeds) in EF, f"effects.csv に {a}−{b} {c} K={K} {seeds} が無い")
    return EF[(a, b, c, K, seeds)]


def level(v: str, c: str, K: int, seeds: str = "all"):
    must((v, c, K, seeds) in LV, f"levels.csv に {v} {c} K={K} {seeds} が無い")
    return LV[(v, c, K, seeds)]


def pq(v: str, quantity: str, layer: str, c: str, K: int, seeds: str = "all"):
    must((v, quantity, layer, c, K, seeds) in PHD, f"intercept_direction.csv に {v} {quantity} {layer} {c} K={K} {seeds} が無い")
    return PHD[(v, quantity, layer, c, K, seeds)]


def eget(a: str, b: str, seeds: str = "all"):
    return lambda c, K: q(a, b, c, K, seeds)


def pget(v: str, quantity: str, layer: str, seeds: str = "all"):
    return lambda c, K: pq(v, quantity, layer, c, K, seeds)


def groups(get, pairs=None) -> dict:
    g = {k: [] for k in ("pp", "p", "z", "n", "nn", "same")}
    for c, K in (ALL12 if pairs is None else pairs):
        g[klass(get(c, K))].append((c, K))
    return g


def order(pairs):
    return sorted(pairs, key=lambda x: (C4.index(x[0]), x[1]))


def by_corpus(pairs) -> str:
    """条件の名前：「AFEW の K = 5・10・25，MELD の K = 25」．"""
    out = []
    for c in C4:
        ks = [K for c2, K in order(pairs) if c2 == c]
        if ks:
            out.append(f"{c} の K = {'・'.join(map(str, ks))}")
    return "，".join(out)


def vals(pairs, get) -> str:
    """条件の名前と値：「AFEW の K = 5 で +2.13 [+1.0, +3.3]，K = 10 で …」．"""
    out = []
    for c in C4:
        ks = [K for c2, K in order(pairs) if c2 == c]
        if ks:
            out.append(f"{c} の " + "，".join(f"K = {K} で {ci(get(c, K))}" for K in ks))
    return "，".join(out)


def read_short(get, pairs=None) -> str:
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
        s.append("2 つの版の正解の数が全部の単位で同じ（差は 0）：" + by_corpus(g["same"]))
    return "．".join(s) + "．"


def short(pairs) -> str:
    out = []
    for c in C4:
        ks = [K for c2, K in order(pairs) if c2 == c]
        if ks:
            out.append(f"{c} {'・'.join(map(str, ks))}")
    return "，".join(out) if out else "—"


CAT_HEAD = ("| 0 を含まず +1.0 点以上 | 0 を含まず 0〜+1.0 点 | 0 を含む（点推定は正） | 0 を含む（点推定は負） | 0 を含まず −1.0〜0 点 | 0 を含まず −1.0 点以下 | 正解の数が同じ |")
CAT_SEP = "|---|---|---|---|---|---|---|"


def cat_cells(get) -> str:
    g = groups(get)
    zp = [x for x in g["z"] if get(*x).est >= 0]
    zn = [x for x in g["z"] if get(*x).est < 0]
    return " | ".join(short(x) for x in (g["pp"], g["p"], zp, zn, g["n"], g["nn"], g["same"]))


# ================================================================== 文を書く前の照合

# ---- 出どころ
must([p["stage"] for p in plog] == ["units", "summary"], "provenance_log.jsonl の並び")
must(plog[1] == prov, "provenance.json が provenance_log.jsonl の 2 行目と違う")
must(all(p["dirty"] is False for p in plog), "dirty")
must(sha(DESIGN) == prov["design_sha256"] == plog[0]["design_sha256"], "設計の sha256")
must(sha(SCRIPT20) == prov["this_script_sha256"] == plog[0]["this_script_sha256"], "台本の sha256")
must(plog[0]["scripts"] == prov["scripts"] and all(sha(ROOT / "DAI" / "exp" / n) == h for n, h in prov["scripts"].items()), "部品の sha256")
must(sha(FROZEN) == prov["frozen_sha256"] == plog[0]["frozen_sha256"], "固定のファイルの sha256")
UNIT_COMMIT, SUM_COMMIT = plog[0]["commit"], plog[1]["commit"]
must(conv["unit_commits"] == [UNIT_COMMIT] and conv["summary_commit"] == SUM_COMMIT and UNIT_COMMIT != SUM_COMMIT, "単位と集計のコミット")
N_BETWEEN = int(git("rev-list", "--count", f"{UNIT_COMMIT}..{SUM_COMMIT}"))
SRC20 = SCRIPT20.read_text()
must('("DAI/frozen/dai20_frozen.json", "DAI/splits", "measure", "features", "data/metadata", "requirements.txt",' in SRC20, "台本の監視対象の一覧")
WATCH = [f"DAI/exp/{n}" for n in prov["scripts"]] + ["DAI/frozen/dai20_frozen.json", "DAI/splits", "measure", "features", "data/metadata", "requirements.txt", "DAI/docs/実験20_設計.md"]
must(git("diff", "--stat", UNIT_COMMIT, SUM_COMMIT, "--", *WATCH).strip() == "", "単位と集計のコミットのあいだで計算に関わるファイルが変わっている")
must(git("diff", "--stat", UNIT_COMMIT, "HEAD", "--", *WATCH).strip() == "", "単位のコミットから今までに計算に関わるファイルが変わっている")


def ctime(commit: str) -> str:
    return git("log", "-1", "--format=%ad", "--date=format:%Y-%m-%d %H:%M:%S", commit).strip()


T_UNITS, T_SUM = plog[0]["time"], plog[1]["time"]
must(T_UNITS < T_SUM < done, "units・summary・done の時刻の順")
GRID = [(float(a), float(b), p) for a, b, p in prov["GRID"]]
must(GRID == [(n0, b, pr) for n0 in (0.0, 20.0, 80.0, float("inf")) for b in (0.0, 1.0) for pr in ("eq", "ext")], "16 点の格子")


def same_but(i: int, n0=None, beta=None) -> int:
    a, b, p = GRID[i]
    return GRID.index((a if n0 is None else n0, b if beta is None else beta, p))


FZ = {(c, K): frozen["choice"][c][str(K)] for c, K in ALL12}
for (c, K), z_ in FZ.items():
    must(GRID[z_["grid"]] == (float(z_["n0"]), float(z_["beta"]), z_["prior"]), f"固定のファイルの点と格子 {c} K={K}")
# 設計 3 節の表と固定のファイル
for c in C4:
    line = [x for x in design.split("\n") if x.startswith(f"| {c} | n0 = ")]
    must(len(line) == 1, f"設計 3 節の表に {c} の行が {len(line)} 行")
    cells = [x.strip() for x in line[0].strip("|").split("|")][1:]
    must(len(cells) == 3, f"設計 3 節の表の {c} の列")
    for K, cell in zip(KS, cells):
        m_ = re.fullmatch(r"n0 = (\d+)，β = (\d)(?:，戻し先 (\w+))?", cell)
        must(m_ is not None, f"設計 3 節の表の欄が読めない：{cell}")
        z_ = FZ[(c, K)]
        must(float(m_.group(1)) == z_["n0"] and float(m_.group(2)) == z_["beta"] and (m_.group(3) is None or m_.group(3) == z_["prior"]), f"設計 3 節の表と固定のファイル {c} K={K}")
        must((m_.group(3) is None) == (z_["n0"] == 0), f"設計 3 節の表の戻し先の書き方 {c} K={K}")
N0_20 = [k_ for k_ in ALL12 if FZ[k_]["n0"] == 20.0]
N0_0 = [k_ for k_ in ALL12 if FZ[k_]["n0"] == 0.0]
must(len(N0_20) + len(N0_0) == 12 and all(FZ[k_]["beta"] == 1.0 for k_ in ALL12), "固定の点の n0 と β")
PRIOR_EXT = [k_ for k_ in N0_20 if FZ[k_]["prior"] == "ext"]
NPSEL_DIFF = [k_ for k_ in ALL12 if FZ[k_]["noprof_sel"] != same_but(FZ[k_]["grid"], beta=0.0)]      # 相性を外す点を選び直すと，同じ n0・戻し先の点と違う点になる条件
must(all(GRID[FZ[k_]["noprof_sel"]][1] == 0.0 for k_ in ALL12), "選び直した点の β")
must(all(len(x["corpora_in_units"]) == 3 and x["T"] not in x["corpora_in_units"] and x["n_units"] == 30 for x in frozen["selfcheck"]) and len(frozen["selfcheck"]) == 12,
     "固定のファイルの選びに使った単位")

# ---- 台帳：句がどの行にあるかまで確かめる
LROWS = ledger.split("\n")
row20 = [x for x in LROWS if x.startswith("| DAI-20（実験 16 の提案の手法の評価の回")]
rowph = [x for x in LROWS if x.startswith("| DAI-20 の計算の後の解析")]
row21 = [x for x in LROWS if x.startswith("| DAI-21（")]
must(len(row20) == 1 and len(rowph) == 1 and len(row21) == 1, "台帳の DAI-20・DAI-21 の行")
ROW20, ROWPH, ROW21 = row20[0], rowph[0], row21[0]

# ---- 点検の報告と検査の報告
must("## 判定\n\n通す．200 単位と集計の 12 ファイルが設計の格子どおりにそろい，欠けと重複は 0 件だった．" in code_post, "点検の報告の判定")
must("水準 8 個（" in code_post and "差 11 個（" in code_post, "点検の報告が作り直した水準と差の数")
must("この報告には UAR と差の値を書かない．書くのは数，一致の割合，ずれの大きさ，sha256，時刻だけである．" in code_post, "点検の報告が書く量の文")
N_REC = code_post.count("### [記録]")
must(N_REC == 5, "点検の報告の記録の数")
for f_, s_ in (("stats_v4.md", "## 判定\n\n通す．"), ("define_v4.md", "## 判定\n通す．"), ("code_v3.md", "## 判定\n\n通す．"), ("compare_v3.md", "## 判定\n通す．"), ("leak.md", "## 判定\n\n通す．"),
               ("review_ml_v3.md", "判定：OK"), ("review_affect_v3.md", "判定：OK")):
    must(s_ in (CK / f_).read_text(), f"{f_} の判定")
must(not any((CK / f_).exists() for f_ in ("compare_v4.md", "leak_v2.md", "leak_v3.md", "leak_v4.md", "review_ml_v4.md", "review_affect_v4.md", "stats_v5.md", "define_v5.md", "code_v4.md")),
     "計算の前の検査に，この台本が知らない版がある")
must("設計 `DAI/docs/実験20_設計.md` 第 3 版" in (CK / "review_ml_v3.md").read_text() and "設計 `DAI/docs/実験20_設計.md` 第 3 版" in (CK / "review_affect_v3.md").read_text(), "査読想定の役が検査した版")
must("2d1b5480" in (CK / "code_v3.md").read_text() and prov["this_script_sha256"].startswith("2d1b5480"), "code_v3 が検査した台本の sha256")

# ---- 計算の後の解析（切片の分と向きの分）の出どころ
must(phprov["script_sha256"] == sha(POSTHOC_PY), "計算の後の解析の台本が，出力の後に変わった")
must(phprov["dirty"] is False and phprov["allow_partial"] is False and phprov["key"] == "T" and phprov["grid"] is None and phprov["n_units"] == 200, "計算の後の解析の記録")
must(phprov["frozen_sha256"] == prov["frozen_sha256"], "計算の後の解析が読んだ固定のファイル")
must(phprov["summary_sha256"] == {n: sha(SM / n) for n in ("levels.csv", "effects.csv", "grid_levels.csv")}, "計算の後の解析が照らした集計が，今の集計と違う")
must(all(phprov["modules_sha256"][n] == prov["scripts"][n] for n in phprov["modules_sha256"]), "計算の後の解析が読み込んだ部品")
PH_COMMITS = [x.split(" ", 1) for x in git("log", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M:%S", "--", "DAI/docs/tools_pending/dai20_posthoc_intercept.py").strip().split("\n")]
must(len(PH_COMMITS) == 3 and all(t_ < done for _, t_ in PH_COMMITS), "計算の後の解析の台本のコミットが，計算の終わりより後にある")
PH_LAST, PH_FIRST = PH_COMMITS[0][0], PH_COMMITS[-1][0]
PH_SHAS = {h_: hashlib.sha256(git("show", f"{h_}:DAI/docs/tools_pending/dai20_posthoc_intercept.py").encode()).hexdigest() for h_, _ in PH_COMMITS}
must(PH_SHAS[PH_LAST] == phprov["script_sha256"], "最後にコミットした台本が，出力の記録の sha256 と違う")
PH_CHECKED = [h_ for h_ in PH_SHAS if f"`dai20_posthoc_intercept.py`：{PH_SHAS[h_]}" in tools_v2]
must(len(PH_CHECKED) == 1 and PH_CHECKED[0] == PH_COMMITS[1][0], "code_post_tools_v2.md が検査した台本の版")
must(phprov["script_sha256"] not in tools_v2 and phprov["script_sha256"] not in tools_v1, "流した台本の sha256 が検査の報告にある（文を直す）")
must("## 判定\n\n通す．" in tools_v2 and "## 判定\n\n直してから進む．" in tools_v1, "計算の後に使う台本の検査の判定")
N_TOOL_REC = tools_v2.count("### [記録 ")
must(N_TOOL_REC == 5, "code_post_tools_v2.md の記録の数")
TB_COMMITS = [x.split(" ", 1) for x in git("log", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M:%S", "--", "DAI/docs/tools_pending/md_tables_dai20.py").strip().split("\n")]
TB_SHAS = {h_: hashlib.sha256(git("show", f"{h_}:DAI/docs/tools_pending/md_tables_dai20.py").encode()).hexdigest() for h_, _ in TB_COMMITS}
must(sha(TABLES_PY) == TB_SHAS[TB_COMMITS[0][0]], "表の台本が，最後のコミットと違う")
TB_CHECKED = [h_ for h_ in TB_SHAS if f"`md_tables_dai20.py`：{TB_SHAS[h_]}" in tools_v2]
must(len(TB_CHECKED) == 1, "code_post_tools_v2.md が検査した表の台本の版")
TB_AFTER = [h_ for h_, _ in TB_COMMITS[:[h_ for h_, _ in TB_COMMITS].index(TB_CHECKED[0])]][::-1]      # 検査の後のコミット（古い順）
must(len(TB_AFTER) >= 1, "表の台本が，検査した版から変わっていない（文を直す）")
N_TOOL_REC_PH = tools_v2.count("該当箇所：`dai20_posthoc_intercept.py")
must(N_TOOL_REC_PH == 4 and tools_v2.count("該当箇所：`md_tables_dai20.py`") == 1, "code_post_tools_v2.md の記録の内訳")
must("five recorded notes applied" in git("log", "-1", "--format=%s", PH_LAST), "最後のコミットの題")
LQ_PHRUN = quote(ROWPH, "検査 `code_post_tools_v2.md` が「通す」とした版は sha256 7efb23c1…（コミット 1f5b65e）である．その報告の記録 5 件を入れた版（sha256 156ec36a…．コミット 7b0de5e）を，"
                        "2026-11-15 03:21 に実験 20 の単位に起動し，03:37 に出力を書いた", "台帳の計算の後の解析の行")
PHRE = "記録 5 件の直しの差分は，流した後に gate-code に再検査を頼んだ（報告は `DAI/checks/DAI-20/code_post_tools_v3.md`）"
LQ_PHRE = quote(ROWPH, PHRE, "台帳の計算の後の解析の行") if PHRE in ROWPH else None      # 再検査の結果を台帳に書き足すと文が変わるので，今の文があるときだけ引く
must("code_post_tools_v3.md" in ROWPH, "台帳の計算の後の解析の行に，再検査の報告の名前が無い")
must(PH_SHAS[PH_CHECKED[0]].startswith("7efb23c1") and PH_CHECKED[0] == "1f5b65e" and phprov["script_sha256"].startswith("156ec36a") and PH_LAST == "7b0de5e"
     and phprov["time"].startswith("2026-11-15 03:37"), "台帳の計算の後の解析の行が，検査の報告と出力の記録に合わない")
V3 = CK / "code_post_tools_v3.md"                                        # 記録 5 件の直しの差分の再検査（頼んである．報告ができていれば判定を引く）
if V3.exists():
    v3_text = V3.read_text()
    m_ = re.search(r"## 判定\s*\n+\s*(.+?)．", v3_text)
    must(m_ is not None, "code_post_tools_v3.md の判定が読めない")
    V3_VERDICT = quote(v3_text, m_.group(1), "code_post_tools_v3.md")
    V3_COVERS = [n_ for n_, h_ in (("流した台本", phprov["script_sha256"]), ("今の表の台本", sha(TABLES_PY))) if h_ in v3_text]
    V3_TEXT = (f"再検査の報告 `code_post_tools_v3.md` の判定は「{V3_VERDICT}」である"
               + (f"（報告が対象に挙げる sha256 は，{'と'.join(V3_COVERS)}の sha256 と同じである）" if V3_COVERS else "（報告に，流した台本の sha256 の全体は書かれていない）"))
else:
    V3_VERDICT, V3_COVERS = None, []
    V3_TEXT = "再検査の報告 `code_post_tools_v3.md` は，この文書を書き出した時点ではまだ無い（gate-code に頼んである）"

# ---- 表の印を，設計 5 節の文（下端 > 0 かつ差 ≥ 1.0 点に ↑，上端 < 0 かつ差 ≤ −1.0 点に ↓）から作り直して，結果ファイルの印と比べる
must("区間の下端 > 0 かつ差 ≥ 1.0 点に ↑，上端 < 0 かつ差 ≤ −1.0 点に ↓ を付ける" in design, "設計 5 節の印の文")
for d_ in (ef, phd):
    mk_ = np.where((d_.lo > 0) & (d_.est >= FLOOR), "↑", np.where((d_.hi < 0) & (d_.est <= -FLOOR), "↓", ""))
    must(bool((mk_ == d_["mark"].fillna("").astype(str).to_numpy()).all()), "印の列が設計 5 節の文と合わない")
    must(bool(((d_.lo <= d_.est + 1e-12) & (d_.est <= d_.hi + 1e-12)).all()), "区間が点推定を挟まない行がある")
must(bool((ef[ef.same_counts].est.abs() < 1e-12).all()) and bool((ef[ef.same_counts].se.abs() < 1e-12).all()), "正解の数が同じ行の差が 0 でない")
must(len(ef) == 2772 and ef.groupby(["a", "b"]).ngroups == 77 and set(ef.seeds) == {"all", "0-1", "2-9"} and not ef.duplicated(["a", "b", "c", "K", "seeds"]).any(), "effects.csv の行")
must(len(lv) == 1176 and lv.version.nunique() == 49 and set(lv.seeds) == {"all", "0-2"}, "levels.csv の行")
must(set(ef[ef.c.isin(["AFEW", "IEMOCAP", "MELD"])].method) == {"jk+seed"} and set(ef[ef.c == "MOSEI"].method) == {"boot_cl+seed"}, "区間の作り方")
DF = {c: sorted(set(ef[ef.c == c].df.dropna().astype(int))) for c in C4}
must(DF == {"AFEW": [13], "IEMOCAP": [3], "MOSEI": [], "MELD": [7]}, f"自由度：{DF}")
FD = {c: sorted(set(ef[ef.c == c].folds_dropped)) for c in C4}
must(FD == {"AFEW": ["0"], "IEMOCAP": ["0,1,2,3,4"], "MOSEI": [""], "MELD": [""]}, f"jackknife で除かれる分割：{FD}")
must(bool((ef[ef.c == "IEMOCAP"].v_seed == 0).all()), "IEMOCAP の種の項")
NFR = {s_: sorted(set(ef[ef.seeds == s_].n_fr)) for s_ in ("all", "0-1", "2-9")}
must(NFR == {"all": [50], "0-1": [10], "2-9": [40]}, f"差の単位の数：{NFR}")

# ---- 集計の値を，単位の csv から別の書き方で作り直して比べる（水準 1,176 行と差 2,772 行の点推定）
UFILES = sorted(UDIR.glob("*.csv"))
must(len(UFILES) == 200 and len(sorted(UDIR.glob("*.npz"))) == 200, "units の csv と npz の数")
UCOLS = ["K", "version", "uar_raw", "uar_adj", "T", "c", "fold", "seed", "tag", "commit", "dirty", "sec", "host", "gpu", "teacher_dev", "grad", "stalled", "adj_resid",
         "recheck_agree_min", "eq_agree", "psel_agree", "wf_n_capped", "n_unused", "dai1_commits"]
HEADER = list(pd.read_csv(UFILES[0], nrows=0).columns)
must(not any(("mem" in x.lower()) or ("gb" in x.lower()) or ("vram" in x.lower()) for x in HEADER) and not any("mem" in k_.lower() for k_ in list(conv) + list(prov)), "GPU のメモリの記録がある（文を直す）")
U = pd.concat([pd.read_csv(p_, usecols=UCOLS, low_memory=False) for p_ in UFILES], ignore_index=True)
must(len(U) == 200 * 840 and not U.duplicated(["T", "fold", "seed", "K", "version"]).any(), "単位の csv の行")
must(set(U.commit) == {UNIT_COMMIT} and set(U.dirty.astype(str)) == {"False"} and set(U.tag) == {"eval"} and bool((U["T"] == U["c"]).all()), "単位の csv のコミット")
must(set(U.groupby("T").apply(lambda d: d[["fold", "seed"]].drop_duplicates().shape[0], include_groups=False)) == {50}, "コーパスごとの単位の数")
HOSTS, GPUS, DAI1_COMMITS = sorted(set(U.host)), sorted(set(U.gpu)), sorted(set(U.dai1_commits))
must(len(HOSTS) == 1 and GPUS == [prov["gpu"]] and len(DAI1_COMMITS) == 1, "計算機の記録")
UG = {k_: g_.set_index("version") for k_, g_ in U.groupby(["T", "K", "fold", "seed"])}
g_ = U[U.version.astype(str).str.fullmatch(r"M_F6_mix\d+")].copy()
g_["grid"] = g_.version.str.replace("M_F6_mix", "").astype(int)
GM = g_.groupby(["T", "K", "grid"]).uar_adj.mean()                          # 格子の点ごとの UAR（切片をずらした答え．50 単位の平均）
PLAIN = ("none_AVT", "joint_AVT", "ret_A", "ret_V", "ret_T", "b_kd", "b_agree", "b_mix", "b_cotrain", "b_coreg")
MY_PICK = {}
for c, K in ALL12:
    own = GM[c, K]
    oth = sum(GM[c2, K] for c2 in C4 if c2 != c) / 3.0                       # ほかの 3 コーパスの平均（コーパス 1 つを 1 票）
    MY_PICK[(c, K)] = dict(loto=int(oth.idxmax()), oracle=int(own.idxmax()))
    x_ = gl[(gl.c == c) & (gl.K == K)].set_index("grid")
    must(len(x_) == 16 and int(x_.frozen.sum()) == 1 and int(x_.loto.sum()) == 1 and int(x_.oracle.sum()) == 1, f"grid_levels.csv の印 {c} K={K}")
    must(int(x_.index[x_.frozen][0]) == FZ[(c, K)]["grid"] and int(x_.index[x_.loto][0]) == MY_PICK[(c, K)]["loto"] and int(x_.index[x_.oracle][0]) == MY_PICK[(c, K)]["oracle"],
         f"選び方の 3 通りの点が，単位から作り直した点と違う {c} K={K}")
    must(float((x_.uar_adj - own).abs().max()) < 1e-9 and float((x_.uar_adj_others - oth).abs().max()) < 1e-9, f"grid_levels.csv の UAR {c} K={K}")


def unit_name(c: str, K: int, v: str) -> tuple[str, str]:
    """集計の版の名前 → 単位の csv の版の名前と列（集計の台本の対応を，設計 5 節の表から書き直したもの）．"""
    i = FZ[(c, K)]["grid"]
    b = v.split("@")[0]
    m_ = {"prop": f"M_F6_mix{i}", "prop_noprof": f"M_F6_mix{same_but(i, beta=0.0)}", "prop_noprof_sel": f"M_F6_mix{FZ[(c, K)]['noprof_sel']}", "prop_n0zero": f"M_F6_mix{same_but(i, n0=0.0)}",
          "prop_levelfree": f"M_F6_mix{GRID.index((float('inf'), 1.0, 'eq'))}", "prop_noint": f"M_F6n_mix{i}", "prop_home": f"M_F6h_mix{i}",
          "prop_loto": f"M_F6_mix{MY_PICK[(c, K)]['loto']}", "oracle_grid": f"M_F6_mix{MY_PICK[(c, K)]['oracle']}"}
    return m_.get(b, b), ("uar_raw" if (v.endswith("@raw") or b in PLAIN) else "uar_adj")


UCACHE: dict = {}


def unit_uar(c: str, K: int, v: str) -> pd.Series:
    """(分割, K 本の引き方) ごとの UAR．"""
    if (c, K, v) not in UCACHE:
        ver, col = unit_name(c, K, v)
        s_ = pd.Series({(f, r): float(UG[(c, K, f, r)].at[ver, col]) for f in range(5) for r in range(10)})
        must(bool(s_.notna().all()), f"単位の csv に {ver} {col} の欠けがある {c} K={K}")
        UCACHE[(c, K, v)] = s_
    return UCACHE[(c, K, v)]


def fold_mean(s_: pd.Series, seeds) -> float:
    """分割ごとに K 本の引き方の平均を取り，分割で平均する．"""
    if seeds is not None:
        s_ = s_[[r in seeds for _, r in s_.index]]
    return float(s_.groupby(level=0).mean().mean())


DEV_LV = max(abs(fold_mean(unit_uar(r.c, int(r.K), r.version), SEEDS[r.seeds]) - r.uar) for r in lv.itertuples())
DEV_EF = max(abs(fold_mean(unit_uar(r.c, int(r.K), r.a) - unit_uar(r.c, int(r.K), r.b), SEEDS[r.seeds]) - r.est) for r in ef.itertuples())
must(DEV_LV < 1e-9 and DEV_EF < 1e-9, f"集計の点推定が，単位の csv から作り直した値と違う：水準 {DEV_LV:.2e}，差 {DEV_EF:.2e}")

# ---- 検算の量を単位の csv から数え直す
CKR = U[U.version == "_check"]
C20 = U[U.version == "_check20"]
must(len(CKR) == 600 and len(C20) == 600, "検算の行の数")
TD_MAX = float(CKR.teacher_dev.max())
TD_ROW = CKR.loc[CKR.teacher_dev.idxmax()]
TD_BY = {c: (float(CKR[CKR["T"] == c].teacher_dev.min()), float(CKR[CKR["T"] == c].teacher_dev.max())) for c in C4}
TD_OVER = int((CKR.teacher_dev > 5e-5).sum())
must(abs(TD_MAX - conv["teacher_dev_max"]) < 1e-15 and TD_MAX <= 1e-4, "作り直しの差の最大")
must(abs(float(CKR.grad.max()) - conv["grad_max"]) < 1e-15 and conv["grad_max"] <= 1e-5 and int(CKR.stalled.sum()) == conv["stalled"] == 0, "勾配の最大と止まり")
must(abs(float(U.adj_resid.max()) - conv["adj_resid_max"]) < 1e-15, "切片のずらしの残り")
must(float(C20.recheck_agree_min.min()) == conv["recheck_agree_min"] == 1.0 and float(C20.eq_agree.min()) == conv["eq_agree_min"] == 1.0 and float(C20.psel_agree.min()) == conv["psel_agree_min"] == 1.0,
     "足した版の計算の道の一致")
WF_CAPPED = int(U[U.version.isin(["wf10", "wf100", "wf1000"])].wf_n_capped.fillna(0).sum())
must(WF_CAPPED == 0, "切片の反復が上限までに収束しなかった表がある（文を直す）")
CF_UNUSED = int(U[U.version == "cf_D_l05"].n_unused.sum())
SEC_TOTAL = float(U.groupby(["T", "fold", "seed"]).sec.first().sum())
must(abs(SEC_TOTAL - conv["sec_total"]) < 1e-6 and conv["n_units"] == 200, "所要の合計と単位の数")
must(conv["others_B_dev_max"] < 1e-6 and len(ot[ot.group == "B"]) == 84 and bool(ot[ot.group == "B"].uar_here.notna().all()), "実験 1 の集計から写した値")
N_WARN = log_eval.count("RuntimeWarning: Mean of empty slice")
N_SKIP = log_eval.count("済み")
N_UNITS_HEAD = log_eval.count("=== dai20 units")
must(N_SKIP == 0 and N_UNITS_HEAD == 1 and log_eval.rstrip().endswith("EXIT=0") and not (ROOT / "logs" / "dai20_eval.failed").exists(), "ログの終わり方")
must("Traceback" not in log_eval, "ログに例外がある")
must("selftest：通った" in (ROOT / "logs" / "dai20_selftest.log").read_text(), "selftest のログ")

# ---- 提案の手法と積の判定を npz から読み，UAR・混同行列・感情ごとの再現率を数え直す
CONF_MY = {(c, K, v): np.zeros((4, 4)) for c, K in ALL12 for v in ("prop", "prod")}
NPZ_DEV, PROD_DIFF, NEST_BAD, N_TEST = 0.0, 0, 0, {}
for p_ in sorted(UDIR.glob("*.npz")):
    head_, r_ = p_.stem.rsplit("_r", 1)
    head_, f_ = head_.rsplit("_f", 1)
    c = head_.split("_", 1)[0]
    must(head_ == f"{c}_ALL", f"単位の名前：{p_.name}")
    z_ = np.load(p_, allow_pickle=False)
    y_ = z_["y_test"].astype(int)
    if not N_TEST:                                                         # 最初の単位で，候補の全ラベルに合わせた切片の層（ob）の判定がある版と無い版を確かめる
        i0 = FZ[(c, 5)]["grid"]
        NO_OB = ["avg", "conf", "psel", "tcal", "temp", "tempb", "wf10", "wf100", "wf1000", f"M_F6h_mix{i0}"]
        must(all(f"pred_adj_K5_{v}" in z_.files and f"pred_ob_K5_{v}" not in z_.files for v in NO_OB), "ob の層の判定が無いはずの版")
        must(all(f"pred_ob_K5_{v}" in z_.files for v in ("prod", f"M_F6_mix{i0}", f"M_F6n_mix{i0}", "M_F2_K", "M_F6_K", "cf_D_l05", "M_F6_ext", "M_F8", "M_F6_orc")), "ob の層の判定があるはずの版")
    N_TEST.setdefault(c, {})[(int(f_), int(r_))] = np.bincount(y_, minlength=4)
    for K in KS:
        i = FZ[(c, K)]["grid"]
        for v, kk, col in (("prop", f"pred_adj_K{K}_M_F6_mix{i}", "uar_adj"), ("prop@raw", f"pred_raw_K{K}_M_F6_mix{i}", "uar_raw"), ("prod", f"pred_raw_K{K}_prod", "uar_raw")):
            pr_ = z_[kk].astype(int)
            uar_ = 100.0 * float(np.mean([(pr_[y_ == e] == e).mean() for e in range(4) if (y_ == e).any()]))
            NPZ_DEV = max(NPZ_DEV, abs(uar_ - float(UG[(c, K, int(f_), int(r_))].at[kk.split(f"_K{K}_", 1)[1], col])))
            if (c, K, v) in CONF_MY:
                np.add.at(CONF_MY[(c, K, v)], (y_, pr_), 1)
        PROD_DIFF += int((z_[f"pred_adj_K{K}_prod"] != z_[f"pred_raw_K{K}_prod"]).sum())
    kp_ = [set(z_[f"kpos_K{K}"].tolist()) for K in KS]                      # K 本の入れ子：K = 5 の K 本 ⊂ K = 10 の K 本 ⊂ K = 25 の K 本
    NEST_BAD += int(not (kp_[0] <= kp_[1] <= kp_[2]) or [len(x) for x in kp_] != [4 * K for K in KS])
must(NPZ_DEV < 1e-9, f"npz の判定から数え直した UAR が単位の csv と違う：{NPZ_DEV:.2e}")
must(PROD_DIFF == 0, "積の判定が，切片をずらす前と後で違う発話がある（文を直す）")
must(NEST_BAD == 0, f"K 本が入れ子でない単位がある：{NEST_BAD}")
for (c, K, v), M_ in CONF_MY.items():
    x_ = cfm[(cfm.c == c) & (cfm.K == K) & (cfm.version == v)]
    must(len(x_) == 16 and all(M_[int(r.true), int(r.pred)] == r.n for r in x_.itertuples()), f"confusion.csv が npz から数え直した数と違う {c} K={K} {v}")
    for e in range(4):
        r = pe[(pe.c == c) & (pe.K == K) & (pe.version == v) & (pe.emotion == e)]
        must(len(r) == 1 and abs(100 * M_[e, e] / M_[e].sum() - float(r.recall_pooled.iloc[0])) < 1e-9 and M_[e].sum() == float(r.n.iloc[0]), f"per_emotion.csv の全体の再現率 {c} K={K} {v} {e}")
N_PER_SEED = {c: sum(N_TEST[c][(f, 0)] for f in range(5)) for c in C4}       # K 本の引き方 1 つぶんの，テストの発話の数（5 分割の合計）
for c in C4:
    must(all((sum(N_TEST[c][(f, r)] for f in range(5)) == N_PER_SEED[c]).all() for r in range(10)), f"テストの発話の数が K 本の引き方で違う {c}")
    for e in range(4):
        must(float(pe[(pe.c == c) & (pe.K == 5) & (pe.version == "prop") & (pe.emotion == e)].n_per_seed.iloc[0]) == N_PER_SEED[c][e], f"per_emotion.csv の n_per_seed {c} {e}")
must(bool(pe.recall.notna().all()) and len(pe) == 2352, "per_emotion.csv の行")

# ---- 計算の後の解析：全体（切片をずらした答えどうし）が集計の「版 − 積」と合う．足し算が成り立つ
PH_VERS = [v for v in ("prop", "prop_noprof", "prop_noprof_sel", "prop_noint", "prop_n0zero", "prop_levelfree", "prop_loto", "oracle_grid", "M_F2_K", "M_F6_K", "cf_D_l05",
                       "M_F6_ext", "M_F8", "M_F6_orc")]
must(sorted(set(phd.version) - {"prod"}) == sorted(PH_VERS) and set(phd.seeds) == {"all", "0-1", "2-9"} and len(phd) == phprov["n_rows"], "intercept_direction.csv の版")
PH_DEV_TOT, PH_DEV_ADD, n_tot = 0.0, 0.0, 0
for v in PH_VERS:
    for c, K in ALL12:
        for sn in ("all", "0-1", "2-9"):
            d_ = pq(v, "direction", "ob", c, K, sn)
            for lay in ("adj", "raw", "kb"):
                t_, i_ = pq(v, "total", lay, c, K, sn), pq(v, "intercept", lay, c, K, sn)
                PH_DEV_ADD = max(PH_DEV_ADD, abs(t_.est - i_.est - d_.est))
            if (v, "prod", c, K, sn) in EF:
                e_, t_ = EF[(v, "prod", c, K, sn)], pq(v, "total", "adj", c, K, sn)
                PH_DEV_TOT = max(PH_DEV_TOT, abs(e_.est - t_.est), abs(e_.lo - t_.lo), abs(e_.hi - t_.hi))
                n_tot += 1
must(PH_DEV_ADD < 1e-9 and PH_DEV_TOT < 1e-6 and n_tot == phprov["total_vs_summary"]["n"] == 432, f"計算の後の解析と集計の照合：{PH_DEV_ADD:.2e}，{PH_DEV_TOT:.2e}，{n_tot}")
RAW_EST_DEV = max(abs(pq("prop", "total", "raw", c, K).est - q("prop@raw", "prod@raw", c, K).est) for c, K in ALL12)
RAW_END_DEV = {c: max(max(abs(pq("prop", "total", "raw", c, K).lo - q("prop@raw", "prod@raw", c, K).lo), abs(pq("prop", "total", "raw", c, K).hi - q("prop@raw", "prod@raw", c, K).hi)) for K in KS) for c in C4}
must(RAW_EST_DEV < 1e-9 and all(RAW_END_DEV[c] < 1e-6 for c in C4 if c != "MOSEI") and RAW_END_DEV["MOSEI"] < 0.01, f"切片をずらす前どうしの全体：{RAW_EST_DEV:.2e}，{RAW_END_DEV}")
for c, K in ALL12:                                                         # 水準：切片をずらした答えと，ずらす前の UAR が集計と合う
    for v, lay, sv in (("prop", "adj", "prop"), ("prop", "raw", "prop@raw"), ("prod", "adj", "prod"), ("prod", "raw", "prod@raw")):
        must(abs(PHL[(v, lay, c, K, "all")].uar - level(sv, c, K).uar) < 1e-9, f"intercept_levels.csv の水準 {v} {lay} {c} K={K}")

# ---- 表の文書：今の集計から書き出し直すと，今ある文書と同じになる
with tempfile.TemporaryDirectory() as td_:
    tmp_ = Path(td_) / "tables.md"
    subprocess.run([sys.executable, str(TABLES_PY), "--out", str(tmp_)], capture_output=True, text=True, check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    must(tmp_.read_text() == table_doc, "表の文書が，今の集計から書き出した表と違う")
# 表の文書：桁を足す前の版（最初にコミットした版）と比べ，変わった値が，符号の読めなかった区間の端と，±1.00 に丸まっていた差だけであること
ZERO_END = re.compile(r"\[[+-]0\.0, |, [+-]0\.0\]")
must(not ZERO_END.search(table_doc), "表の文書に符号の読めない区間の端が残っている")
TD_COMMITS = git("log", "--format=%h", "--", "DAI/docs/実験20_結果の表_2026-11-15.md").strip().split("\n")
must(len(TD_COMMITS) >= 2 and TD_COMMITS[-1] == "a618df8", f"表の文書のコミット：{TD_COMMITS}")
OLD_TBL = git("show", f"{TD_COMMITS[-1]}:DAI/docs/実験20_結果の表_2026-11-15.md")
must(git("status", "--porcelain", "--", "DAI/docs/実験20_結果の表_2026-11-15.md").strip() == "", "表の文書に，コミットしていない変更がある")
NUMRE = re.compile(r"[+-]\d+\.\d+")
o_rows = [x.split("|") for x in OLD_TBL.split("\n") if x.startswith("|")]
n_rows = [x.split("|") for x in table_doc.split("\n") if x.startswith("|")]
must(len(o_rows) == len(n_rows), "表の文書の表の行の数が変わった")
TBL_ENDS, TBL_EST = 0, 0
for ro_, rn_ in zip(o_rows, n_rows):
    must(len(ro_) == len(rn_), "表の文書の列の数が変わった")
    for a_, b_ in zip(ro_, rn_):
        if a_ == b_:
            continue
        na_, nb_ = NUMRE.findall(a_), NUMRE.findall(b_)
        must(len(na_) == 3 and len(nb_) == 3 and a_.strip()[-1] == b_.strip()[-1] and NUMRE.sub("#", a_) == NUMRE.sub("#", b_), f"表の文書の欄が，数の桁のほかで変わった：{a_} → {b_}")
        if na_[0] != nb_[0]:
            must(abs(float(na_[0])) == 1.0 and f"{float(nb_[0]):+.2f}" == na_[0], f"表の文書の差が変わった：{a_} → {b_}")
            TBL_EST += 1
        for x_, y_ in zip(na_[1:], nb_[1:]):
            if x_ != y_:
                must(x_ in ("+0.0", "-0.0") and abs(float(y_)) <= 0.05 and y_[0] == x_[0], f"表の文書の区間の端が変わった：{a_} → {b_}")   # 0.05 より 0 に近い値を 2 桁以上で書いた欄（丸めて 0.05 になる欄を含む）
                TBL_ENDS += 1
o_text = [x for x in OLD_TBL.split("\n") if not x.startswith("|")]
n_text = [x for x in table_doc.split("\n") if not x.startswith("|")]
TBL_NEW_LINES = [x for x in n_text if x not in o_text]
must(all(x in n_text for x in o_text) and len(TBL_NEW_LINES) == 1 and TBL_NEW_LINES[0].startswith("- 数の桁："), f"表の文書の文の行が，桁の注のほかで変わった：{TBL_NEW_LINES}")
must(TBL_ENDS > 0, "表の文書に，桁を足した区間の端が無い（文を直す）")

# ---- 実験 21 の設計・台本・固定のファイル・起動の台本が，実験 20 の結果を開いた後に変わっていないこと（git の記録だけを見る．結果の置き場は開かない）
D21 = ["DAI/docs/実験21_設計.md", "DAI/exp/dai21_eval5.py", "DAI/frozen/dai21_frozen.json", "DAI/run/dai21_eval5.sh"]
D21_LAST = {f_: git("log", "-1", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M:%S", "--", f_).strip().split(" ", 1) for f_ in D21}
must(all(len(v_) == 2 and v_[1] < "2026-11-15 03:21:00" for v_ in D21_LAST.values()) and git("status", "--porcelain", "--", *D21).strip() == "",
     f"実験 21 の設計・台本・固定のファイルが，実験 20 の結果を開いた後に変わっている（文を直す）：{D21_LAST}")
# 格子の点 12（n0 = ∞，β = 0）は積と同じ判定になる（設計 3 節）：単位の csv の UAR で確かめる
must(GRID[12] == (float("inf"), 0.0, "eq") and GRID[14] == (float("inf"), 1.0, "eq"), "格子の点 12・14")
P12_DEV = max(abs(float(UG[k_].at["M_F6_mix12", "uar_adj"]) - float(UG[k_].at["prod", "uar_adj"])) for k_ in UG)
must(P12_DEV < 1e-9, f"格子の点 12 の UAR が積と違う：{P12_DEV:.2e}")

# ================================================================== 文

L: list[str] = []
w = L.append
A_OPP = [v for v, _ in MT.A_LAB[1:]]                                       # 比べる相手 A のうち積を除く 12 個
ABL = [v for v, _ in MT.ABL_LAB]
B_OPP = [v for v, _ in MT.B_LAB]
must(len(A_OPP) == 12 and len(ABL) == 6 and len(B_OPP) == 7 and MT.A_LAB[0][0] == "prod", "表の台本の版の一覧")
INF = float("inf")


def gdesc(i: int) -> str:
    n0, b, p = GRID[i]
    return f"点 {i}（n0 = {'∞' if n0 == INF else int(n0)}，β = {int(b)}，{p}）"


def nv(name: str, val: float) -> str:
    return f"{name}{'' if name.endswith('）') else ' '}{val:.2f}"


def kjoin(xs) -> str:
    return "・".join(str(x) for x in xs)


# ---- 冒頭
w("# 実験 20（DAI-20．提案の手法の評価の回．4 コーパス）：結果の文の下書き（2026-11-15．第 1 版．照合の前．台帳と表の文書の直しを反映した）")
w("")
w("この文書は，台本 `DAI/docs/tools_pending/extract_dai20_text.py` が `results/dai20_eval/` の結果ファイルから数値を取り出して書き出した．"
  "結果ファイルから取る数値と表は，すべて台本が埋めた（手で写していない）．設計・点検の報告・台帳から引いた文は，引いた文が元の文書にあることを台本が確かめた．"
  "文が言う数えと大小は，台本が結果ファイルと照らしてから書き出している．集計の点推定は，台本が単位の保存から別の書き方で作り直した値と照らした（0 節）．文を直すときは，台本の中の文を直して流し直す．"
  "最初に書き出した後に，本体が台帳（計算の後の解析の行）と表の文書（区間の端と差の桁）を直した．この版は，その直しを反映して書き出し直した第 1 版である（6 節）．")
w("")
w("値は UAR（4 感情の再現率の平均．%．でたらめに答えると 25%）で，差はすべて UAR の点の差である．角かっこは 95% 区間で，補正していない．")
w("")
w("**この実験には判定の規則が無く，主の K も無い**（研究者の指示．設計 0 節）．K = 5・10・25 を同じ扱いで並べる．差は次の 3 つに分けて読む．"
  "(1) 区間が 0 を含まず，差が 1.0 点以上．(2) 区間が 0 を含まず，差が 1.0 点未満．(3) 区間が 0 を含む．"
  "表の印（↑・↓）は (1) に付く．印は表を読みやすくするための印で，判定ではない．(3) の差は，2 つの版が同じ水準であることを意味しない．K どうしの差の区間は作っていないので，K の間の値の違いを差の証拠として読まない．")
w("")
w("**設計 0 節の事実の記録**（設計 5 節が，結果の文書の冒頭に写すと決めた 4 項．設計の文のまま）：")
w("")
D0 = [quote(design, "評価のターゲット T のテストの発話は，5 分割のすべてで，下調べ（実験 16）で T を仮のターゲットにした単位のテストと同じ発話である（重なり 100%）．種 0・1 の単位は，K 本と候補も下調べと同じである．"
                    "違うのはソースの数（3 つと 2 つ）と，それによる分類器だけである．手法は，下調べの値（K = 5 で提案 − 積が AFEW +2.70，IEMOCAP +0.55，MOSEI +0.32，MELD +0.01）を見た後に選んだ．", "設計 0 節"),
      quote(design, "4 コーパスの評価の回の出力は，実験 1〜3・14 の解析で既に見ている．提案の手法の案（実験 16）は，評価の回を見た後に出た．"
                    "とくに，2026-11-05 の評価の回の探索（`results/explore_fusion/loo_cells.csv`．台帳 247 行）で，「K 本の 1 本抜きの信頼度で決めた，感情に共通の指数の積」を同じテストの上で測っている"
                    "（K = 5 で積との差が AFEW +4.9，MOSEI −1.3）．この版は，提案の部品「水準は K 本」と，比べる相手 A の「水準だけの重み」にほぼ当たる．", "設計 0 節"),
      quote(design, "下調べでは，実験 16 の設計が置いた「本番に進む目安」（K = 5 で 2 コーパス以上で積を上回る）を満たさなかった．研究者の判断で進む．", "設計 0 節"),
      quote(design, "上の 1・2 から，この評価の回の値は，手法を選んだときに見たテストと同じテストの上の値であり，探索の結果として扱う．", "設計 0 節")]
for i_, s_ in enumerate(D0, 1):
    w(f"{i_}. {s_}")
w("")
w("設計の文の「種」は，K 本の引き方の番号（台本の `seed`．0〜9）である．この文書では「K 本の引き方」と書く．")
w("")

# ---- 0 節
w("## 0. 出どころと申告")
w("")
w(f"- **設計**：`DAI/docs/実験20_設計.md`（第 5 版．計算の前．sha256 {prov['design_sha256'][:8]}…．単位の計算と集計の 2 つの記録の `design_sha256` と同じで，今のファイルとも同じ）．")
w(f"- **台本とコミット**：`DAI/exp/dai20_eval.py`（sha256 {prov['this_script_sha256'][:8]}…．2 つの記録で同じ値で，今の台本とも同じ）．起動の台本は `DAI/run/dai20_eval.sh`．"
  f"**単位の計算（units）はコミット {UNIT_COMMIT[:7]}，集計（summary）はコミット {SUM_COMMIT[:7]} で流れた**．dirty はどちらも偽である（`provenance_log.jsonl` の 2 行）．"
  f"`provenance.json` は集計の記録で，そこに書いてあるコミットは集計のコミットである．単位のコミットは `provenance_log.jsonl` の 1 行目，`summary/convergence.json` の `unit_commits`，単位の csv の `commit` の列（200 単位の全行で 1 種類）にある．"
  f"2 つのコミットのあいだには {N_BETWEEN} コミットがあり，計算に関わるファイル（台本と読み込む部品の計 {len(prov['scripts'])} 本，固定のファイル，`DAI/splits`，`measure`，`features`，`data/metadata`，`requirements.txt`，設計）の差分は 0 行である"
  "（この台本が `git diff` で確かめた．単位のコミットから今の HEAD までの差分も 0 行）．読み込む部品の sha256 は 2 つの記録で同じで，今のファイルとも同じである．")
w(f"- **固定のファイル**：`DAI/frozen/dai20_frozen.json`（sha256 {prov['frozen_sha256'][:8]}…．2 つの記録で同じ値で，今のファイルとも同じ）．"
  "選んだ点（12 条件の n0・β・戻し先）は設計 3 節の表と同じである（台本が設計の表の文字と照らした）．"
  f"固定のファイルの記録では，どの点も，評価するコーパスを含まない下調べの 30 単位で選んである（`selfcheck` の 12 行）．固定の点が n0 = 20 の条件は {by_corpus(N0_20)} の {len(N0_20)} つで，"
  f"うち戻し先が ext の条件は {by_corpus(PRIOR_EXT)} である．残る {len(N0_0)} 条件は n0 = 0（点 2）である．β は 12 条件とも 1 である．")
w("- **結果の場所**：`results/dai20_eval/`．集計は `summary/`（11 個の csv と `convergence.json`），単位の保存は `units/`（200 単位の csv と npz），設定とコミットの記録は `provenance.json`・`provenance_log.jsonl`，"
  "実験 1 の評価の回の単位の保存の sha256 は `dai1_eval_units_sha256.txt`，計算の後の解析の出力は `posthoc/`．")
LQ_RUN = quote(ROW20, "計算は 2026-11-14 21:25:31 に起動し，2026-11-15 03:07:46 に終わった", "台帳の DAI-20 の行")
must(done == "2026-11-15 03:07:46", "logs/dai20_eval.done の時刻")
w(f"- **起動と終了**：台帳の DAI-20 の行は「{LQ_RUN}」と書いている．記録の時刻は，単位の計算の始まりが {T_UNITS}，集計の始まりが {T_SUM}（`provenance_log.jsonl`），終わりが {done}（`logs/dai20_eval.done`）である．"
  f"単位の所要の合計は {SEC_TOTAL:,.0f} 秒（{SEC_TOTAL / 3600:.2f} 時間）である．計算機は研究室の PC（`{HOSTS[0]}`）の GPU 1 枚（{GPUS[0]}）で，200 単位とも同じ計算機である．"
  f"ログに「済み（飛ばす）」の行は {N_SKIP} 行で，単位の計算の見出しは {N_UNITS_HEAD} 回だけなので，途中から流し直していない．ログの最後は `EXIT=0` である．")
LQ_CHK = quote(ROW20, "stats 通す（v4），leak 通す（第 2 版），compare 通す（v3），define 通す（v4），code 通す（`code_v3.md`．コミット 604ca3e の台本 sha256 2d1b5480…），"
                      "ML の査読想定 OK と感情認識の査読想定 OK（どちらも第 3 版に対して．その後の第 4・5 版の変更は，この 2 体の軽微な指摘とほかの関門の指摘への対応で，stats・define・code が再検査した）", "台帳の DAI-20 の行")
must("足した量（`tcal`，`psel`，提案の adj − 相手の raw）の定義は一意" in (CK / "define_v4.md").read_text(), "define_v4.md が検査した量")
KQ_LEAK = quote((CK / "leak.md").read_text(), "対象：`DAI/docs/実験20_設計.md` 第 2 版（コミット d952b6e．作業木の変更なし）．台本 `DAI/exp/dai20_eval.py` と固定のファイル `DAI/frozen/dai20_frozen.json` はまだ無い", "leak.md")
must("対象：`DAI/docs/実験20_設計.md` 第 3 版（作業ツリーの版．コミット前）" in (CK / "compare_v3.md").read_text(), "compare_v3.md が検査した版")
DQ_V4 = quote(design, "比べる相手：分類器ごとの温度の較正（`tcal`）を足した．別のコーパスで選んだ指数の積に切片のずらしを当てた版（`psel`）を A に入れた．", "設計 11 節")
w(f"- **計算の前の検査**（`DAI/checks/DAI-20/`）：台帳の DAI-20 の行は，起動の前（21:26）の状態を「{LQ_CHK}」と書いている．"
  "台本が確かめたこと：`stats_v4.md`・`define_v4.md`・`code_v3.md`・`compare_v3.md`・`leak.md` の判定は「通す」，`review_ml_v3.md`・`review_affect_v3.md` は「判定：OK」で，査読想定の 2 役が検査した設計は第 3 版である．"
  f"設計の第 4 版は比べる相手を足した（設計 11 節：「{DQ_V4}」）．**第 4・5 版を再検査したのは統計（`stats_v4.md`）・定義（`define_v4.md`）・台本（`code_v2.md`・`code_v3.md`）の検査で，"
  "比較の公平さの検査（最後の報告は `compare_v3.md`．第 3 版），漏れの検査（`leak.md`），査読想定の 2 役は，第 4・5 版を再検査していない**（`DAI/checks/DAI-20/` に，これより後の版の報告は無い）．"
  f"**漏れの検査が検査したのは設計の第 2 版である**（`leak.md`：「{KQ_LEAK}」）．比べる相手，部品を外した版，設定の選び方の 3 通りを足した第 3 版より後の設計と，台本・固定のファイルは，漏れの検査を受けていない．")
CQ_WARN = quote(code_post, "警告は記録用の配列（感情の対ごとの AUC の表）の対角から出る．", "点検の報告")
CQ_JK = quote(code_post, "AFEW では，ある塊を除くと分割 0 のテストが無効になり，その分割の種の揺れが `v_cluster` の側に入る．", "点検の報告")
LQ_GPU = quote(ROW20, "GPU のメモリの最大は記録なし（試走で 6.4 GB）", "台帳の DAI-20 の行")
CQ_SID = quote(code_post, "200 単位すべてで，npz の `sid_test` が実験 1 の評価の回の保存（`results/dai1/eval/units/<名前>_p1.npz` の `sample_test`）と並びまで同じだった", "点検の報告")
w(f"- **計算の後の点検**：`DAI/checks/DAI-20/code_post.md`（判定は「通す」．記録 {N_REC} 件）．報告は UAR と差の値を書かず，数・一致の割合・ずれの大きさ・sha256・時刻だけを書いている．"
  f"報告によると，{CQ_SID}．記録 {N_REC} 件の中身を，この台本が結果ファイルとログから数え直した値で書く．")
w(f"   1. **GPU のメモリの最大が記録されていない**．単位の csv の {len(HEADER)} 列にも，`convergence.json`・`provenance.json` にも，メモリの量の項目が無い．台帳は「{LQ_GPU}」と書いている．")
w(f"   2. **警告 `RuntimeWarning: Mean of empty slice` が {N_WARN} 件ある**（`logs/dai20_eval.log`）．報告によると，{CQ_WARN.rstrip('．')}（対角の AUC は定義されない値）．")
w(f"   3. **作り直した積と実験 1 の保存の積の差の最大は {sci(TD_MAX)} で，止める基準 1×10⁻⁴ の {100 * TD_MAX / 1e-4:.0f}% である**（{TD_ROW['T']} の K = {int(TD_ROW.K)}，分割 {int(TD_ROW.fold)}，K 本の引き方 {int(TD_ROW.seed)} の単位）．"
  "コーパスごとの（単位，K）の最大は，" + "，".join(f"{c} {sci(TD_BY[c][1])}" for c in C4) + f" である．5×10⁻⁵ を超えた（単位，K）は 600 組のうち {TD_OVER} 組である．"
  f"保存の確率から作り直した提案の手法の判定は，600 組すべてで元の関数の判定と一致した（一致の割合の最小 {conv['recheck_agree_min']:.6f}）．")
w(f"   4. **`provenance.json` のコミット（集計．{SUM_COMMIT[:7]}）は，単位のコミット（{UNIT_COMMIT[:7]}）と違う**．2 つのあいだで計算に関わるファイルの差分は 0 行である（上の「台本とコミット」）．")
w(f"   5. **jackknife で除かれる分割**：`effects.csv` の `folds_dropped` は，AFEW の全行で「{FD['AFEW'][0]}」，IEMOCAP の全行で「{FD['IEMOCAP'][0]}」，MELD と MOSEI の全行で空である．"
  f"報告は「{CQ_JK}」と書いている．IEMOCAP は全部の分割が除かれ，`v_seed`（K 本の引き方の項）は全行で 0 である．K 本の引き方の項の割合は MOSEI の行でだけ読む（設計 5 節）．")
w(f"- **検算の量**（`summary/convergence.json`．この台本が単位の csv から数え直して同じ値になることを確かめた）：単位 {conv['n_units']}，作り直した積と保存の積の差の最大 {sci(conv['teacher_dev_max'])}，"
  f"勾配の最大 {sci(conv['grad_max'])}，止まり {conv['stalled']}，切片のずらしの残りの最大 {sci(conv['adj_resid_max'])}，作り直した提案の手法の判定の一致の最小 {conv['recheck_agree_min']:.6f}，"
  f"等しい重みの表と積の一致の最小 {conv['eq_agree_min']:.6f}，指数の積の作り直しの一致の最小 {conv['psel_agree_min']:.6f}，実験 1 の集計から写した水準との差の最大 {sci(conv['others_B_dev_max'])}．値の一覧は 3.1 節．")
w(f"- **この台本の照合**（どれも読むだけ．CPU）：(a) 集計の水準 {len(lv):,} 行と差 {len(ef):,} 行の点推定を，単位の csv の UAR から「分割ごとに K 本の引き方の平均を取り，分割で平均する」形で作り直した．"
  f"集計とのずれの最大は，水準 {sci(DEV_LV)}，差 {sci(DEV_EF)} である．版の名前と格子の点の対応は，集計の台本を読み込まずに，設計 5 節の表と固定のファイルから書き直した．"
  f"(b) 提案の手法と積の判定を 200 個の npz から読み，UAR・混同行列・感情ごとの全体の再現率を数え直した．単位の csv の UAR とのずれの最大は {sci(NPZ_DEV) if NPZ_DEV > 0 else '0'} で，`confusion.csv` と `per_emotion.csv` の数は数え直した数と同じである．"
  "積の判定は，切片をずらす前と後で，200 単位 × K 3 つの全部の発話で同じである．"
  "(c) 選び方の 3 通りの点（固定の点，ほかの 3 コーパスで選んだ点，自分のテストで最良の点）を単位の csv から選び直し，`grid_levels.csv` の印と同じ点になった．"
  "(d) 表の印（↑・↓）を設計 5 節の文から作り直し，`effects.csv` と計算の後の解析の出力の全行で印の列と一致した．"
  f"(e) 計算の後の解析の「全体」（切片をずらした答えどうし）{n_tot} 行は，集計の「版 − 積」の点推定と区間の両端に一致した（ずれの最大 {sci(PH_DEV_TOT)}）．"
  "(f) 表の文書は，今の集計から書き出し直した表と 1 字も違わない．桁を足す前の版から変わった値は，区間の端と差の桁だけである（2.1 節）．区間（両端と標準誤差）は，この台本では作り直していない（点検の報告が水準 8 個と差 11 個を作り直して一致を報告している）．")
LQ_OPEN = quote(ROW20, "本体が結果を開いたのは点検の後の 2026-11-15 03:21", "台帳の DAI-20 の行")
LQ_TBL = quote(ROW20, "`DAI/docs/tools_pending/md_tables_dai20.py` で `DAI/docs/実験20_結果の表_2026-11-15.md` を書き出して読んだ", "台帳の DAI-20 の行")
w(f"- **結果を開いた時刻**：台帳の DAI-20 の行は「{LQ_OPEN}（{LQ_TBL}）」と書いている．")
LQ21 = quote(ROW21, "実験 20 の結果を開いたのは起動の後の 03:21 である．", "台帳の DAI-21 の行")
w(f"- **照合の前に値を使った先**：照合（関門 2）の前に，この結果の値は次の場所で使われている．"
  f"(1) 表の文書 `DAI/docs/実験20_結果の表_2026-11-15.md`（sha256 {sha(TABLE_DOC)[:8]}…．書き出しの台本 `md_tables_dai20.py`．03:21 に書き出し，その後に桁を足して書き出し直した．2.1 節）．"
  "(2) 計算の後の解析（下の項）．集計の `levels.csv`・`effects.csv`・`grid_levels.csv` を照合と点の対応に読む．"
  "(3) 2026-11-15 03:25 ごろ，本体が研究者へ会話で主な値を報告した（提案の手法 − 積，別のまとめ方との比べ，部品ごとの寄与，学び直した分類器との比べ）．"
  "(4) 03:40 ごろ，本体が研究者へ会話で切片の分と向きの分の値を報告した．"
  "(5) 本体の作業の覚え書き（memory の `dai20-eval-run-2026-11-14.md`）に要点の値を書いた．"
  "(3)〜(5) は本体のセッションの申告で，この台本からは確かめられない．"
  "実験 21（DAI-21）の設計・台本・固定のファイル・起動の台本は，実験 20 の結果を開いた後に変わっていない．最後のコミットは，"
  + "，".join(f"`{f_.rsplit('/', 1)[1]}` が {v_[0]}（{v_[1][5:16]}）" for f_, v_ in D21_LAST.items())
  + f" で，どれも 2026-11-15 03:21 より前であり，作業ツリーにも変更が無い（この台本が git の記録で確かめた）．台帳の DAI-21 の行は「{LQ21}」と書いている．")
LQ_PHDEF = quote(ROWPH, "定義（向きの分 ＝ 候補の全ラベルに合わせた切片を両方に足した答えどうしの差．切片の分 ＝ 全体 − 向きの分）と版の対応は，実験 20 の結果を開く前に台本として決めた", "台帳の計算の後の解析の行")
NOTE_PAPER = (ROOT / "DAI" / "docs" / "論文のセッションへの伝言_2026-11-14.md").read_text()
must("## 1. 研究者が決めたこと（今日）" in NOTE_PAPER, "論文のセッションへの伝言の 1 節")
NQ1 = quote(NOTE_PAPER, "10 節に，前半の解析の道具を使った解析を入れる．", "論文のセッションへの伝言の 1 節の 3")
NQ2 = quote(NOTE_PAPER, "(a) 提案の gain を「切片の分」と「向きの分」に分ける．", "論文のセッションへの伝言の 1 節の 3")
must(f"3. **{NQ1}**{NQ2}" in NOTE_PAPER, "論文のセッションへの伝言の 1 節の 3 の文")
TB_NOTE = ("1 回目は検査の記録の直し，2 回目は区間の端と差に桁を足す直しである．" if TB_AFTER == ["7b0de5e", "73fd3aa"] else "")
w(f"- **計算の後の解析（切片の分と向きの分）**：設計の文書に無い解析で，**探索の解析として扱う**．台本は `DAI/docs/tools_pending/dai20_posthoc_intercept.py`（冒頭に定義），出力は `results/dai20_eval/posthoc/`．"
  f"この解析を入れることは研究者が決めた．研究者が 2026-11-14 の夜に決めたことの記録（`DAI/docs/論文のセッションへの伝言_2026-11-14.md` の 1 節の 3）に「{NQ1}{NQ2}」とある．"
  f"台帳の行は「{LQ_PHDEF}」と書いている．台本の 3 つのコミット（{'，'.join(f'{h_} {t_[5:16]}' for h_, t_ in reversed(PH_COMMITS))}）は，どれも計算の終わり（{done[5:16]}）より前である．"
  f"**検査と起動**：`code_post_tools.md`（コミット {PH_FIRST} の版．判定は「直してから進む」）の指摘を直した版（コミット {PH_CHECKED[0]}．sha256 {PH_SHAS[PH_CHECKED[0]][:8]}…）を `code_post_tools_v2.md` が検査し，"
  f"判定は「通す」（記録 {N_TOOL_REC} 件．うち {N_TOOL_REC_PH} 件がこの台本，1 件が表の台本）だった．"
  f"**流した台本は，その記録の直しを入れた後の版（コミット {PH_LAST}．sha256 {phprov['script_sha256'][:8]}…．今の台本と同じ）である**．台帳の行は「{LQ_PHRUN}」{f'「{LQ_PHRE}」' if LQ_PHRE else ''}と書いている．"
  f"出力の記録（`intercept_provenance.json`）では，書いた時刻は {phprov['time']}，HEAD は {phprov['commit'][:7]}，dirty は偽である．台帳の起動の時刻（03:21）は，結果を開いた時刻（03:21）と同じ分である．"
  f"**{V3_TEXT}**．"
  f"表の文書を書き出した `md_tables_dai20.py`（今の sha256 {sha(TABLES_PY)[:8]}…．コミット {TB_COMMITS[0][0]}）は，`code_post_tools_v2.md` が検査した版（コミット {TB_CHECKED[0]}）の後に {len(TB_AFTER)} 回直してある"
  f"（コミット {'・'.join(TB_AFTER)}）．{TB_NOTE}"
  + ("今の版は，再検査の報告が対象に挙げた版である．" if "今の表の台本" in V3_COVERS else "")
  + "この台本は，表の文書が今の集計から書き出し直した表と同じことと，桁を足す前の版から変わった値が区間の端と差の桁だけであることを確かめた（2.1 節）．"
  f"台本の中の照合（出力の記録）：足し合わせ「全体 ＝ 切片の分 ＋ 向きの分」のずれの最大 {sci(phprov['additivity_dev_max'])}，集計の水準 {phprov['levels_vs_summary']['n']} 行とのずれの最大 {sci(phprov['levels_vs_summary']['dev_max'])}，"
  f"集計の差 {phprov['total_vs_summary']['n']} 行との点推定のずれの最大 {sci(phprov['total_vs_summary']['est_dev_max'])}・区間の端のずれの最大 {sci(phprov['total_vs_summary']['ci_dev_max'])}．")
w("")
w("**申告**（設計 0・3・4・5・9・10 節が求めるもの．冒頭の 4 項に加えて）：")
w("")
DQ_BODY = quote(design, "比べる相手 A の中身（4 節の表の 2〜9 行目），提案の部品を外した版（5 節），K = 10 も流すこと，実験 1 の 5 つの比較手法との対の差を出すこと", "設計 0 節")
DQ_BODY0 = quote(design, "研究者の明示の確認を経ていない．検査の指摘で足したものを含む", "設計 0 節")
w(f"- **本体が決めたこと**（{DQ_BODY0}）：{DQ_BODY}（設計 0 節）．計算の後の解析（切片の分と向きの分）は，研究者が入れると決めた解析で，"
  "定義（向きの分を測るのに，候補の全ラベルに合わせた切片の判定を使うこと）と版の対応は，本体が台本として決めた．")
must(quote(dai_rules, "手法は評価に使ったのと同じ 4 コーパスの上で作った", "DAI/CLAUDE.md") != "", "")
DQ_SEL1 = quote(design, "下調べの全体（約 200 の版を，この実験と同じテストの発話の上で比べた）", "設計 4 節")
DQ_SEL2 = quote(design, "T を含まない下調べの 30 単位のテスト", "設計 4 節")
w(f"- **手法は，評価に使ったのと同じ 4 コーパスの上で作った**（`DAI/CLAUDE.md` の 2026-11-04 の決定）．設計 4 節の「選ぶデータの表」では，提案の手法の形を選んだデータは「{DQ_SEL1}」，"
  f"設定（n0・β・戻し先）を選んだデータは「{DQ_SEL2}」である．手法を作るのに使っていないコーパスでの値は，この実験には無い．")
DQ_2SRC = quote(design, "設定（n0・β・戻し先）は，ソースが 2 つの下調べで選んだ点で，ソースが 3 つの評価の回に合うとは限らない．", "設計 9 節")
w(f"- **設定は，ソースが 2 つの下調べで選んだ点である**（設計 9 節：「{DQ_2SRC}」）．")
w("- **K は入れ子で，テストは同じである**．K = 10 の K 本（40 本）は K = 5 の K 本（20 本）を含み，K = 25 の K 本（100 本）は K = 10 の K 本を含む（台本が 200 単位の npz の `kpos_K*` で確かめた）．テストの発話・分割・K 本の引き方は 3 つの K で同じである．"
  "**K どうしの差の区間は作っていない**（設計 5 節）．")
DQ_NINT = quote(design, "区間の数（差の対は 77 個で，T 4 × K 3 を掛けると 924 個．種で分けた値を入れるとその 3 倍）", "設計 5 節")
DQ_CHANCE = quote(design, "差が全部 0 でも偶然の印は付きうる．", "設計 5 節")
w(f"- **区間は 95% で，補正していない**．差の対は 77 個で，4 コーパス × K 3 つを掛けると 924 個，K 本の引き方で分けた値を入れると {len(ef):,} 個である（`effects.csv` の行の数．設計 5 節：「{DQ_NINT}」）．"
  f"{DQ_CHANCE}印の数を数えて主張に使わない．コーパスをまたいだ数え上げを検定の結果として書かない（設計 5 節）．")
DQ_VAR = quote(design, "含むのは，テストの塊の取り方と K 本の抜き方（種）である．含まないのは，ソースの取り方，学習側の発話，手法と設定の選び方，重みの表を探す版の 1,001 個の表の引き方（全単位で共通の表を使う）である．"
                       "IEMOCAP の区間は 5 セッションの jackknife（自由度 3）である．MELD の分割 4 は塊が 2 つしか無い．", "設計 5 節")
DQ_S01 = quote(design, "MOSEI の種で分けた値のうち種 0・1 の区間は，種の項の自由度が小さいので狭めに出る", "設計 5 節")
w(f"- **区間が含む揺れと含まない揺れ**（設計 5 節の文のまま）：{DQ_VAR}{DQ_S01}．")
w(f"- **区間の作り方**（`effects.csv` の `method`・`df`）：AFEW・IEMOCAP・MELD は，塊を全分割から同時に除く jackknife と K 本の引き方の項（自由度は順に {DF['AFEW'][0]}・{DF['IEMOCAP'][0]}・{DF['MELD'][0]}），"
  "MOSEI は塊の重みの引き直しと K 本の引き方の項（自由度を持たない）である．塊は，IEMOCAP がセッション，AFEW が俳優と作品の塊，MELD が話者の群，MOSEI が話者の代わりの塊（多くは動画 ID）である（設計 2 節）．"
  "IEMOCAP は自由度 3 で区間が広い．")
DQ_BIAS = quote(design, "A の相手のうち上の 2 行目の 3 つと，別のコーパスで選んだ指数の積の raw の判定（下調べの `prod_sel` と同じ式）は，提案を選んだ下調べで同じテストの発話の上で比べた版である．"
                        "このため，これらとの差は，提案に有利な向きに偏る．", "設計 4 節")
w(f"- **比べる相手 A との差は，提案の手法に有利な向きに偏る**．設計 4 節：「{DQ_BIAS}」「上の 2 行目の 3 つ」は，{nmc('M_F2_K')}，{nmc('M_F6_K')}，{nmc('cf_D_l05')}である．"
  "候補の多い相手は選ぶデータ（K 本）の上で有利になり，選ぶデータが少ないと選んだ表がテストで合わなくなる（設計 4 節）．")
DQ_B = quote(design, "B の判定は切片をずらしていない（提案だけが積の平均に合わせてある）．", "設計 4 節")
DQ_C = quote(design, "C との対の差は作らない．C の UAR には区間を付けない", "設計 4 節")
DQ_C2 = quote(design, "水準の区間どうしの重なりは差の検定ではない．", "設計 4 節")
w(f"- **比べる相手 B・C は種類が違う**．B・C は学び直した 1 つの分類器で，提案の手法は 3 つの分類器の答えの組み合わせである（推定時に 3 つの分類器を動かす）．{DQ_B}"
  f"{DQ_C}（写す元の表が持つのは「今の提案 − 手法」の差の区間だけである）．{DQ_C2}提案の手法の答えを疑似ラベルにした学び直しは，この実験では行っていない（設計 9 節）．")
DQ_LOTO = quote(design, "この選び方は，T のテストの上の成績を使わない．ただし，ほかのターゲットの単位では T がソースで，T の全発話（T を評価するときにテストになる発話を含む）のラベルが，ソースの分類器とソースで測った表に使われる．", "設計 3 節")
w(f"- **設定の選び方 (b)（ほかの 3 コーパスの評価の回で選んだ点）は参考である**．設計 3 節：「{DQ_LOTO}」選び方 (c)（自分のテストで最良の点）は実際には使えず，固定の点との差は作りの上で 0 以上になる．")
DQ_SPK = quote(design, "(1) MOSEI の塊の多くは人物ではなく動画 ID で，同じ人物の別の動画が学習側に残りうる（MOSEI の UAR は高めに出る向き）．(2) K 本で水準を測る 5 組は感情ごとに並びの順で配り，話者でも対話でも分けないので，K 本の水準は高めに出る．"
                       "(3) 作り直した分類器は，抜いた組の発話の特徴を，ラベル無しの候補として学習に使う．", "設計 9 節")
DQ_IEM = quote(design, "筋書き（場面の指示）は全セッションで共通で，同じ筋書きの発話が K 本・候補とテストの両方に入る．言葉の分類器の K 本の水準が筋書きの手がかりで高く出ると，提案は言葉を重くする．", "設計 9 節")
DQ_MELD = quote(design, "MELD はテストに出る話（season，episode）の 92〜98% が候補にも出る", "設計 9 節")
DQ_CORP = quote(design16, "感情ラベルの決め方はコーパスで違う（IEMOCAP の Happy は興奮を含む．MOSEI の Neutral は 6 つの感情がどれも弱い発話）．MOSEI は話者の代わりにチャンネル（無ければ動画）で分ける．"
                          "MELD の顔は話者でない人の顔でありうる．AFEW の言葉は Whisper の書き起こしで，声と言葉が同じ音から来る．", "実験 16 の設計 8 節")
w(f"- **話者と内容の分離**（設計 9 節）：テストと学習に使う発話の話者は分かれている．次の 3 点は，テストには及ばず，提案の手法と K 本を使う相手に同じく入る．{DQ_SPK}"
  f"IEMOCAP の即興では，{DQ_IEM}{DQ_MELD}（実験 1 の設計 1 節の記録）．")
w(f"- **コーパスごとの事情**（実験 16 の設計 8 節から引き継ぐ）：{DQ_CORP}")
DQ_ENC = quote(design, "1 組のエンコーダ，凍結した特徴の上の線形の分類器に限る．", "設計 9 節")
DQ_4C = quote(design, "4 コーパスだけ．CREMA-D と RAVDESS は入れない（比べる相手の結果が無い）．IEMOCAP は即興だけ．", "設計 9 節")
DQ_STACK = quote(design, "ターゲットの K 本で 12 入力 × 4 出力の第 2 段を学ぶ stacking は入れない．重みは 48 個で，K = 5 の 20 本では決まらない．数を減らした版（モダリティごとの重み 3 つ，感情ごとに和を固定した 12 個）を A に入れた．", "設計 9 節")
w(f"- **範囲**（設計 9 節）：{DQ_ENC}{DQ_4C}{DQ_STACK}")
DQ_SEEN = quote(design, "この設計を書くために評価の回の出力は開いていない．", "設計 10 節")
LQ_2201 = quote(ROW20, "実験 20 の設計・台本・固定のファイルはコミット済みで計算中であり，この値で変えたものは無い", "台帳の DAI-20 の行")
must("**2026-11-14 22:01**" in ROW20, "台帳の 22:01 の記録")
DQ_18 = quote(design, "第 3 版・台本・固定のファイルは，実験 18 の結果を開く前にコミットした（bfbd939，20:27）．実験 18 の集計の値は 2026-11-14 20:39 に開いた", "設計 10 節")
DQ_18B = quote(design, "どれも検査役の指摘による変更で，実験 18 の値を理由にした変更は無い．", "設計 10 節")
w(f"- **見たデータ**（設計 10 節）：設計の前に本体が見た値は，実験 16・17 の下調べと K = 10 の追補の結果，実験 1〜3・14 の評価の回の結果（以前の解析）などである．{DQ_SEEN}"
  f"{DQ_18}．その後に台本に入れた変更について，設計は「{DQ_18B}」と書いている．"
  f"計算中の 2026-11-14 22:01 に，本体は実験 1・2 の評価の回の集計の値を開いて実験 18 の値と並べた．台帳は「{LQ_2201}」と書いている．")
DQ_EXP = quote(design, "下調べの値から，提案 − 積は AFEW で +2〜+3 点，IEMOCAP・MOSEI・MELD で 0〜+1 点と見込む．", "設計 5 節")
w(f"- **計算の前の見込み**（設計 5 節の記録）：「{DQ_EXP}」結果と並べた値は要点 1 にある．見込みを作った下調べの値と今回の値は，同じテストの発話（K 本の引き方 0・1 では同じ K 本）の上の値である．")
w("")
w("**版の名前**（表の文書 `DAI/docs/実験20_結果の表_2026-11-15.md` の呼び名を使う．括弧は台本の名前）：")
w("")
w(f"- **提案の手法**（`prop`）：感情ごとの信頼度の表で声・顔・言葉の答えをまとめる手法（設計 3 節）．分類器ごとの水準（信頼度の行の平均）を K 本で測り，水準からの感情ごとの差（相性）をソースで測った表から取る．"
  "設定は固定のファイルの点である．**積**（`prod`）：微調整した声・顔・言葉の分類器の確率を同じ重みで掛けた，推定時の答え（今の提案が疑似ラベルに使う答え）．"
  "**今の提案**（`joint_AVT`）：積を疑似ラベルにして学び直した融合の分類器．")
w("- **切片の扱い**：「切片をずらした答え」（台本の `adj`）は，ラベルを使わずに，候補の発話の上の確率の平均が積の確率の平均と同じになるように切片をずらしてから判定した答えである（設計 3 節の 6）．"
  "「切片をずらす前」（`raw`．版の名前に `@raw` を付ける）は，ずらさずに判定した答えである．積は，切片をずらす前と後で判定が同じである．")
w("- **比べる相手 A**（同じ 3 つの分類器の答えの別のまとめ方．設計 4 節）：" + "，".join(nmc(v) for v in A_OPP) + "．")
w("- **提案の部品を 1 つ外した版**（設計 5 節）：" + "，".join(nmc(v) for v in ABL) + "．")
w("- **比べる相手 B**（学び直した 1 つの分類器．実験 1 の評価の回の保存の判定）：" + "，".join(nmc(v) for v in B_OPP) + "．単独の分類器は " + "，".join(nmc(v) for v, _ in MT.SINGLE_LAB) + "．")
w("- **比べる相手 C**（標準の手法．実験 2 の評価の回の集計から写した UAR）：" + "，".join(f"{lab}（`{m}`）" for m, lab in MT.C_LAB.items() if (ot.method == m).any()) + "．")
w("- **参照・参考・設定の選び方**：" + "，".join(nmc(v) for v, _ in MT.REF_LAB) + "．")
w("")

# ---- 1 節
w("## 1. 要点")
w("")
w("この節は，区間が 0 を含まない条件と，点推定が負の条件を，名前と値で書く．区間が 0 を含み点推定が正の条件は，名前だけを書く．12 条件の全部の値は，表の文書と 2 節の表にある．"
  "条件の名前は「コーパスの K」で書く．条件の数を数えて主張にはしない．")
w("")
w("### 提案の手法と積の UAR（切片をずらした答え．50 単位）")
w("")
w("| コーパス | K | 提案の手法 | 積 | 提案の手法 − 積 |")
w("|---|---|---|---|---|")
for c, K in ALL12:
    w(f"| {c} | {K} | {lci(level('prop', c, K))} | {lci(level('prod', c, K))} | {ci(q('prop', 'prod', c, K), mark=True)} |")
w("")
w("### 問い 1（設計の Q1）：提案の手法の答えは，積より UAR が高いか")
w("")
G1 = groups(eget("prop", "prod"))
must(G1["pp"] == [("AFEW", K) for K in KS] and not G1["n"] and not G1["nn"] and not G1["same"], "提案の手法 − 積の読み")
REST9 = [k_ for k_ in ALL12 if k_[0] != "AFEW"]
E9 = [q("prop", "prod", c, K).est for c, K in REST9]
must(all(abs(x) < FLOOR for x in E9), "AFEW 以外の差が 1.0 点以上")
Z1NEG = [k_ for k_ in G1["z"] if q("prop", "prod", *k_).est < 0]
w(f"1. **提案の手法 − 積（どちらも切片をずらした答え）は，AFEW では 3 つの K とも区間が 0 を含まず 1.0 点以上で（{vals(G1['pp'], eget('prop', 'prod'))}），"
  f"IEMOCAP・MOSEI・MELD の 9 条件では点推定が {fest(min(E9))}〜{fest(max(E9))} 点で 1.0 点に届かなかった**．"
  f"9 条件のうち，区間が 0 を含まない条件は {vals(G1['p'], eget('prop', 'prod'))} である．"
  f"区間が 0 を含む条件は {by_corpus(G1['z'])} で，"
  + (f"このうち点推定が負の条件は {vals(Z1NEG, eget('prop', 'prod'))} である．" if Z1NEG else "点推定はどれも正である．"))
w(f"   - **切片をずらす前どうし**（`prop@raw − prod@raw`）：{read_short(eget('prop@raw', 'prod@raw'))}")
w(f"   - **提案の手法の中の，切片をずらした答え − ずらす前**（`prop − prop@raw`）：{read_short(eget('prop', 'prop@raw'))}")
w(f"   - **K 本の引き方で分けた値**（どちらも切片をずらした答え）：引き方 0・1 は下調べと同じ K 本（コーパスごとに {NFR['0-1'][0]} 単位），引き方 2〜9 は下調べと違う K 本（{NFR['2-9'][0]} 単位）である．"
  "テストの発話は，どちらも下調べと同じ発話である．2 つは同じテストの上の値で，独立な 2 回の測定ではない．MOSEI の引き方 0・1 の区間は狭めに出る（0 節）．")
w(f"      - 引き方 0・1：{read_short(eget('prop', 'prod', '0-1'))}")
w(f"      - 引き方 2〜9：{read_short(eget('prop', 'prod', '2-9'))}")
EXPECT = {"AFEW": (2.0, 3.0), "IEMOCAP": (0.0, 1.0), "MOSEI": (0.0, 1.0), "MELD": (0.0, 1.0)}
IN_ = [k_ for k_ in ALL12 if EXPECT[k_[0]][0] <= q("prop", "prod", *k_).est <= EXPECT[k_[0]][1]]
ABOVE = [k_ for k_ in ALL12 if q("prop", "prod", *k_).est > EXPECT[k_[0]][1]]
BELOW = [k_ for k_ in ALL12 if q("prop", "prod", *k_).est < EXPECT[k_[0]][0]]
w(f"   - **計算の前の見込みと並べた値**：設計の見込みは AFEW で +2〜+3 点，IEMOCAP・MOSEI・MELD で 0〜+1 点だった（0 節）．点推定が見込みの幅の中にある条件は {by_corpus(IN_)}，"
  + (f"幅より大きい条件は {by_corpus(ABOVE)}，" if ABOVE else "") + (f"幅より小さい条件は {by_corpus(BELOW)} " if BELOW else "")
  + "である．見込みを作った下調べの値と今回の値は，同じテストの発話（引き方 0・1 では同じ K 本）の上の値である．")
PE = {(r.version, r.c, int(r.K), int(r.emotion)): r for r in pe.itertuples()}


def edif(c: str, K: int, e: int) -> float:
    return float(PE[("prop", c, K, e)].recall - PE[("prod", c, K, e)].recall)


def erange(c: str, e: int) -> str:
    d_ = [edif(c, K, e) for K in KS]
    return f"{EMO[e]} {fest(min(d_))}〜{fest(max(d_))}"


EPOS = {c: [EMO[e] for e in range(4) if all(edif(c, K, e) > 0 for K in KS)] for c in C4}
ENEG = {c: [EMO[e] for e in range(4) if all(edif(c, K, e) < 0 for K in KS)] for c in C4}
w("   - **感情ごとの再現率**（2.6 節．提案の手法 − 積の点の値の引き算．感情ごとの差の区間は作っていない）：3 つの K の値の幅は，"
  + "；".join(f"{c} が " + "，".join(erange(c, e) for e in range(4)) for c in C4) + " である．"
  "3 つの K とも引き算が正の感情は，" + "，".join(f"{c} の {kjoin(EPOS[c])}" for c in C4 if EPOS[c]) + "．"
  + ("3 つの K とも負の感情は，" + "，".join(f"{c} の {kjoin(ENEG[c])}" for c in C4 if ENEG[c]) + "．" if any(ENEG.values()) else "3 つの K とも負の感情は無い．")
  + "区間が無いので，符号と大きさを並べるだけにとどめる．")
w("")
w("### 問い 2（設計の Q2）：提案の手法は，既にあるまとめ方より UAR が高いか．差は，水準・行の中の差・分類器 × 感情の成分のどれに当たるか")
w("")
NEG_ADJ = [(v, c, K) for v in A_OPP for c, K in ALL12 if klass(q("prop", v, c, K)) in ("n", "nn")]
must(NEG_ADJ == [("tcal", "AFEW", 5)], f"切片をずらした答えどうしで，区間が 0 を含まずに提案の手法が低い条件：{NEG_ADJ}")
ZNEG1 = [(v, c, K) for v in A_OPP for c, K in ALL12 if klass(q("prop", v, c, K)) == "z" and q("prop", v, c, K).est <= -FLOOR]
must(ZNEG1 and all(c == "AFEW" for _, c, _ in ZNEG1), f"区間が 0 を含み点推定が −1.0 点以下の条件：{ZNEG1}")


def opp_vals(triples, a_of=lambda v: "prop", b_of=lambda v: v) -> str:
    out = []
    for v in A_OPP + B_OPP + ABL:
        ps = [(c, K) for v2, c, K in triples if v2 == v]
        if ps:
            out.append(f"{nm(v)}（`{v}`）との差の {vals(ps, eget(a_of(v), b_of(v)))}")
    return "；".join(out)


w(f"2. **提案の手法 − 別のまとめ方（比べる相手 A のうち積を除く 12 個．どちらも切片をずらした答え）は，相手とコーパスで符号と区間の読みが違う**．差は「提案の手法 − 相手」で，負の値は提案の手法の UAR が低いことを表す．"
  f"**区間が 0 を含まずに提案の手法が低かった条件は 1 つで，{opp_vals(NEG_ADJ)} である**．"
  f"区間が 0 を含み，点推定が −1.0 点以下の条件は，どれも AFEW にある（{opp_vals(ZNEG1)}）．"
  "相手ごとの読みを下に並べる．12 条件の値と UAR は表の文書の 2 節，読みの一覧は 2.2 節にある．相手のうち 3 つとの差は提案の手法に有利な向きに偏る（0 節）．")
for K in KS:
    must(abs(q("prop", "psel", "AFEW", K).est - q("prop", "prod", "AFEW", K).est) < 1e-12 and level("psel", "AFEW", K).same_as == "prod", f"AFEW の psel と積 K={K}")
    must(all(level("psel", c, K).same_as != "prod" for c in C4[1:]), f"AFEW 以外の psel と積 K={K}")
must(all(level("prop_noprof", c, K).same_as == "M_F2_K" for c, K in N0_0) and all(level("prop_noprof", c, K).same_as != "M_F2_K" for c, K in N0_20), "相性を外した版と水準だけの重みが同じ条件")
DQ_PSEL = quote(design, "AFEW では選ばれた指数が (1, 1, 1) なので，作りの上で積と同じ判定になる", "設計 4 節")
DQ_WF = quote(design, "3 つの候補の数は入れ子の表なので，独立な 3 つの相手として数えない．", "設計 4 節")
DQ_CF = quote(design, "実験 15 で，声の分類器の疑似ラベルを直す用途のために仮のターゲットで選んだ値をそのまま使う", "設計 4 節")
DQ_TCAL = quote(design, "`tcal` の原典（Guo ほか 2017）との違いは，感情ごとに等しい重みの対数尤度で合わせること，1 へ引く罰則を置くこと，重み 0 を許すことである．", "設計 4 節")
DQ_WFO = quote(design, "原典との違い：足す点が対数確率であること，検証データが 20〜100 本であること（原典は 383 本），候補の数は原典に書かれていないこと，同点の表を平均するので使う表が候補に無い表になりうること．", "設計 4 節")
DQ_CONF = quote(design, "原典は確信度の量を定めていない．最大の確率で測るのはこの設計の選択である", "設計 4 節")
DQ_SCALE = quote(design, "点の全体の倍率が積と違う相手では，まとめ方とは別の理由で判定が変わりうる．", "設計 4 節")
DQ_SAME = quote(design, "同じ群の版は独立な比べとして数えない．", "設計 5 節")
NOTE_A = {
    "psel": f"{DQ_PSEL}（設計 4 節）．AFEW の値は提案の手法 − 積と同じである．",
    "M_F2_K": f"固定の点が n0 = 0 の {len(N0_0)} 条件では，この相手は，相性を外した版（`prop_noprof`）と正解の数が全部の単位で同じである（設計 5 節）．",
    "cf_D_l05": f"縮める強さ 0.5 は，{DQ_CF}（設計 4 節）．使わなかった分類器（対角の平均が 0.25 以下）の数は，200 単位 × K 3 つの合計で {CF_UNUSED} である．",
    "wf10": f"候補の数 10・100・1,000 の 3 つについて，設計 4 節は「{DQ_WF}」と決めている．値の良い候補の数を後から選ばない．Kaya ほか 2017 の型をもとにした版で，設計 4 節は「{DQ_WFO}」と書いている．",
    "tcal": f"温度の較正（temperature scaling）を分類器ごとに当てた形である．{DQ_TCAL.rstrip('．')}（設計 4 節）．",
    "conf": f"確信度で重みづける後段の融合の型で，{DQ_CONF}（設計 4 節）．",
}
for v in A_OPP:
    w(f"   - **{nmc(v)}**：{read_short(eget('prop', v))}" + (NOTE_A[v] if v in NOTE_A else ""))
NEG_RAW = [(v, c, K) for v in A_OPP for c, K in ALL12 if klass(q("prop@raw", f"{v}@raw", c, K)) in ("n", "nn")]
NEG_MIX = [(v, c, K) for v in A_OPP for c, K in ALL12 if klass(q("prop", f"{v}@raw", c, K)) in ("n", "nn")]
must(len(NEG_RAW) == 4 and len(NEG_MIX) == 1, f"切片をずらす前の相手との差で，区間が 0 を含まずに提案の手法が低い条件：{NEG_RAW}，{NEG_MIX}")
w(f"   - **切片の扱いを変えた組を並べる理由**（設計 4 節）：{DQ_SCALE}切片をずらす前の判定は，重みの全体の倍率に依らない．")
w(f"   - **切片をずらす前どうし**（`prop@raw − 相手@raw`．2.2 節の 2 つ目の表）：区間が 0 を含まずに提案の手法が低かった条件は 4 つで，"
  f"{opp_vals(NEG_RAW, lambda v: 'prop@raw', lambda v: v + '@raw')} である．")
w(f"   - **提案の手法（切片をずらした答え）− 相手（切片をずらす前）**（相手が原典の形のままのときの差．2.2 節の 3 つ目の表）：区間が 0 を含まずに提案の手法が低かった条件は 1 つで，"
  f"{opp_vals(NEG_MIX, lambda v: 'prop', lambda v: v + '@raw')} である．")
w("   - **相手 − 積**（切片をずらした答えどうし）の読みの一覧は 2.3 節にある．")

# 部品を 1 つ外した版（設計 5 節の表の名前で，差ごとに書く）
for c, K in ALL12:
    for sn in ("all", "0-1", "2-9"):
        t_ = q("prop", "prod", c, K, sn).est
        must(abs(t_ - q("prop", "prop_noprof", c, K, sn).est - q("prop_noprof", "prod", c, K, sn).est) < 1e-9, f"先に相性を外す順序の足し算 {c} K={K} {sn}")
        must(abs(t_ - q("prop", "prop_levelfree", c, K, sn).est - q("prop_levelfree", "prod", c, K, sn).est) < 1e-9, f"先に水準を外す順序の足し算 {c} K={K} {sn}")
        must(abs(q("prop", "prop_noprof", c, K, sn).est - q("prop", "prop_noint", c, K, sn).est - q("prop_noint", "prop_noprof", c, K, sn).est) < 1e-9, f"提案の手法 − 相性を外すの足し算 {c} K={K} {sn}")
AF = [("AFEW", K) for K in KS]
MO = [("MOSEI", K) for K in KS]
P_NP, NP_PD, P_LF, LF_PD = eget("prop", "prop_noprof"), eget("prop_noprof", "prod"), eget("prop", "prop_levelfree"), eget("prop_levelfree", "prod")
must(all(klass(NP_PD(*k_)) == "pp" and klass(P_LF(*k_)) == "pp" for k_ in AF), "AFEW の「相性を外す − 積」と「提案の手法 − 水準の違いを外す」")
AF_SMALL = [P_NP(*k_).est for k_ in AF] + [LF_PD(*k_).est for k_ in AF]
must(all(0 < x <= 0.6 for x in AF_SMALL), "AFEW の「提案の手法 − 相性を外す」と「水準の違いを外す − 積」")
must(all(NP_PD(*k_).est < 0 and P_LF(*k_).est < 0 for k_ in MO), "MOSEI の「相性を外す − 積」と「提案の手法 − 水準の違いを外す」の符号")
must(all(klass(P_NP(*k_)) == "p" and klass(LF_PD(*k_)) == "p" for k_ in MO), "MOSEI の「提案の手法 − 相性を外す」と「水準の違いを外す − 積」")
EN = [P_NP(*k_).est for k_ in ALL12]
must(all(0 < x < FLOOR for x in EN), "提案の手法 − 相性を外すの符号と大きさ")
N20_EQ = [k_ for k_ in N0_20 if FZ[k_]["prior"] == "eq"]
must(sorted(N20_EQ + PRIOR_EXT) == sorted(N0_20) and N20_EQ and PRIOR_EXT, "n0 = 20 の条件の戻し先")
DQ_ORDER = quote(design, "重みは max(0, 水準 ＋ 相性) で作るので", "設計 5 節")
DQ_P12 = quote(design, "格子の点 12（n0 = ∞，β = 0）は積と同じ判定になる", "設計 3 節")


def k3(get, ks) -> str:
    return kjoin(fest(get(*k_).est) for k_ in ks)


w("3. **提案の手法 − 積を，設計 5 節の 2 つの順序で分けた**（2.4 節の表）．先に相性を外す順序は，提案の手法 − 積 ＝（提案の手法 − 相性を外す）＋（相性を外す − 積）である．"
  "先に水準を外す順序は，提案の手法 − 積 ＝（提案の手法 − 水準の違いを外す）＋（水準の違いを外す − 積）である．"
  f"{DQ_ORDER}，2 つの部品の寄与は足し算にならず，外す順序で値が変わる（設計 5 節）．点推定は，どちらの順序でも，2 つの差を足すと提案の手法 − 積になる（台本が確かめた）．"
  f"**AFEW では，「相性を外す − 積」と「提案の手法 − 水準の違いを外す」が，3 つの K とも区間が 0 を含まず 1.0 点以上だった**（順に {k3(NP_PD, AF)} と {k3(P_LF, AF)}）．"
  f"AFEW の「提案の手法 − 相性を外す」と「水準の違いを外す − 積」の点推定は {fest(min(AF_SMALL))}〜{fest(max(AF_SMALL))} 点だった．"
  f"**MOSEI では，「相性を外す − 積」と「提案の手法 − 水準の違いを外す」の点推定が 3 つの K とも負で（順に {k3(NP_PD, MO)} と {k3(P_LF, MO)}），"
  f"「提案の手法 − 相性を外す」と「水準の違いを外す − 積」は 3 つの K とも区間が 0 を含まず正で 1.0 点未満だった**（順に {k3(P_NP, MO)} と {k3(LF_PD, MO)}）．"
  "差ごとの読みを下に並べる．かぎかっこの名前は設計 5 節の表の名前である．両側の違いが 1 つの部品だけでない条件には設計の名前を使わず，何が重なっているかを書く．")
w(f"   - **提案の手法 − {nmc('prop_noprof')}**（設計 5 節の名前は「行の中の差を使った分」．どの分類器にも共通の感情ごとの成分を含む．12 条件とも，両側の違いは β だけである）："
  f"12 条件とも点推定は正で 1.0 点未満である（{fest(min(EN))}〜{fest(max(EN))}）．{read_short(P_NP)}")
w(f"   - **相性を外す − 積**（先に相性を外す順序の後ろの差）：固定の点が n0 = 0 の {len(N0_0)} 条件（{by_corpus(N0_0)}）では，相性を外した版は K 本で測った水準だけで重みを決める版（水準だけの重みと同じ判定）で，"
  f"この差は設計 5 節の「水準の分」である．{read_short(NP_PD, N0_0)}"
  f"固定の点が n0 = 20 の {len(N0_20)} 条件では，相性を外した版は，K 本で測った水準を戻し先へ寄せた水準を使う．この差には，K 本の水準と戻し先が重なっている．"
  f"戻し先が 3 つの分類器の平均の条件は {vals(N20_EQ, NP_PD)} である．"
  f"戻し先がソースで測った水準の条件（ソースの水準の重みは n0 ÷ (n ＋ n0)．K = 5 で 0.5，K = 10 で 1/3．設計 5 節）は {vals(PRIOR_EXT, NP_PD)} である．")
w(f"   - **提案の手法 − {nmc('prop_noint')}**（設計 5 節の名前は「分類器 × 感情の成分の分」．12 条件とも同じ格子の点で，重みの表からこの成分だけを消した対照との差）：{read_short(eget('prop', 'prop_noint'))}")
w(f"   - **分類器 × 感情の成分を消す − 相性を外す**（設計 5 節の名前は「当てはめに残る成分の分」．中身は，どの分類器にも共通の感情ごとの倍率，水準と感情の掛け算の項，重みを 0 に切ったときに生じる水準の違いである）："
  f"{read_short(eget('prop_noint', 'prop_noprof'))}")
w(f"   - **提案の手法 − {nmc('prop_levelfree')}**（先に水準を外す順序の前の差）：対照は格子の点 14（n0 = ∞，β = 1，eq）で，3 つの水準を K 本で測った水準の平均にそろえ，相性はソースから取る．"
  "K 本の水準の平均は残るので，この対照も K 本を使う（設計 5 節）．"
  f"固定の点が n0 = 0 の {len(N0_0)} 条件では，両側の違いは，分類器ごとの水準を K 本の値のまま使うか，平均にそろえるかだけで，この差は設計 5 節の「相性を使ったうえでの，分類器どうしの水準の違いの分」である．{read_short(P_LF, N0_0)}"
  f"固定の点が n0 = 20 の {len(N0_20)} 条件では，提案の手法の水準が戻し先へ寄せてある．戻し先が 3 つの分類器の平均の条件は，水準の違いをラベル 20 本ぶんの強さで平均へ寄せた版と，平均にそろえた版の差で，{vals(N20_EQ, P_LF)} である．"
  f"戻し先がソースで測った水準の条件は，K 本の水準の違いとソースの水準の違いが重なった差で，{vals(PRIOR_EXT, P_LF)} である．")
w(f"   - **水準の違いを外す − 積**（先に水準を外す順序の後ろの差．設計 5 節の名前は「水準の違いなしでの相性の分」）：12 条件とも，両側は格子の点 14 と，積と同じ判定になる点 12 で（設計 3 節：「{DQ_P12}」），違いは β だけである．"
  f"{read_short(LF_PD)}")
must(sorted(NPSEL_DIFF) == sorted([("AFEW", 25), ("MELD", 10), ("MELD", 25)]), f"相性を外す点を選び直すと違う点になる条件：{NPSEL_DIFF}")
w(f"   - **提案の手法 − {nmc('prop_noprof_sel')}**（相性を使わない 8 点から，提案の手法と同じ規則で下調べが選んだ点との差）：選び直した点が，同じ n0・戻し先で β = 0 の点と違う条件は {by_corpus(NPSEL_DIFF)} である．"
  f"この 3 条件の差には，β の違いと n0・戻し先の違いが重なっている．値は {vals(NPSEL_DIFF, eget('prop', 'prop_noprof_sel'))} である．残る 9 条件は「提案の手法 − 相性を外す」と同じ点である．")
must(all(q("prop", "prop_n0zero", *k_).est < 0 for k_ in N0_20) and all(klass(q("prop", "prop_n0zero", *k_)) == "same" for k_ in N0_0), "水準を戻さない版との差")
w(f"   - **提案の手法 − {nmc('prop_n0zero')}**（同じ β・戻し先で n0 = 0 の点との差）：固定の点が n0 = 20 の 4 条件だけで差が出る．4 条件とも点推定は負である（{vals(N0_20, eget('prop', 'prop_n0zero'))}）．"
  f"残る {len(N0_0)} 条件は同じ点である．")
w(f"   - **提案の手法 − {nmc('prop_home')}**（設計 5 節の名前は「残り方で相性を縮めた分」．相性を，残り方を掛けないソースの中の値から取る版との差．12 条件とも同じ格子の点）：{read_short(eget('prop', 'prop_home'))}")
w("")
w("### 問い 3（設計の Q3．記述）：学び直した 1 つの分類器と並べた UAR の高さ")
w("")
NEG_B = [(v, c, K) for v in B_OPP for c, K in ALL12 if klass(q("prop", v, c, K)) in ("n", "nn")]
must(not NEG_B, f"B との差で，区間が 0 を含まずに提案の手法が低い条件：{NEG_B}")
GJ = groups(eget("prop", "joint_AVT"))
w(f"4. **提案の手法（推定時の答え．切片をずらした答え）− 今の提案（`joint_AVT`．学び直した融合の分類器）は，区間が 0 を含まない条件が {vals(GJ['pp'] + GJ['p'], eget('prop', 'joint_AVT'))} で，"
  f"残る {len(GJ['z'])} 条件は区間が 0 を含んだ**．比べる相手 B の 7 つとの差で，区間が 0 を含まずに提案の手法が低かった条件は無い．"
  "B の判定は切片をずらしていない．提案の手法は学び直す前の答えで，推定時に 3 つの分類器を動かす（0 節）．相手ごとの読みは次のとおりである（12 条件の値と UAR は表の文書の 4 節）．")
for v in B_OPP:
    w(f"   - **{nmc(v)}**：{read_short(eget('prop', v))}")
OC = ot[ot.group == "C"]
must(set(OC.n_units.astype(int)) == {50, 15} and sorted(set(OC[OC.n_units == 15].method)) == ["dann", "mapctl", "mme"] and len(OC) == 156, "標準の手法の行")
M50 = [m for m in MT.C_LAB if ((OC.method == m) & (OC.n_units == 50)).any()]
M15 = [m for m in MT.C_LAB if ((OC.method == m) & (OC.n_units == 15)).any()]
must(len(M50) == 10 and len(M15) == 3, "標準の手法の数")
C_HI, C_NONE = [], []
for c, K in ALL12:
    p50, p15 = level("prop", c, K).uar, level("prop", c, K, "0-2").uar
    x50 = OC[(OC.K == K) & (OC.c == c) & (OC.n_units == 50)].set_index("method").uar
    x15 = OC[(OC.K == K) & (OC.c == c) & (OC.n_units == 15)].set_index("method").uar
    must(len(x50) == 10 and len(x15) == 3, f"標準の手法の行 {c} K={K}")
    must(all(x15[m] < p15 for m in M15), f"15 単位の上で，UAR の値が提案の手法以上の手法がある {c} K={K}（文を直す）")
    hi_ = [m for m in M50 if x50[m] > p50]
    must(all(x50[m] < p50 for m in M50 if m not in hi_), f"50 単位の上で，UAR の値が提案の手法と同じ手法がある {c} K={K}（文を直す）")
    if hi_:
        C_HI.append(f"{c} の K = {K}（提案の手法 {p50:.2f}）：" + "，".join(nv(MT.C_LAB[m], x50[m]) for m in hi_))
    else:
        C_NONE.append((c, K))
w(f"5. **標準の手法（比べる相手 C．実験 2 の評価の回の集計から写した UAR）は，UAR を並べるだけである**．提案の手法との対の差は作っていない．C の UAR には区間が無い．UAR の高低は差の検定ではない（0 節）．"
  f"50 単位（K 本の引き方 0〜9）で測った手法は {len(M50)} 個（{'，'.join(MT.C_LAB[m] for m in M50)}），15 単位（引き方 0〜2）で測った勾配法の手法は {len(M15)} 個（{'，'.join(MT.C_LAB[m] for m in M15)}）である．"
  "15 単位の手法は，提案の手法の同じ 15 単位の上の UAR と並べる．xMUDA 型には融合の分類器が無いので，並べる値が無い．手法ごとの値は表の文書の 5 節にある．")
w(f"   - **50 単位の上**：UAR の値が提案の手法より高い手法がある条件と，その手法は次のとおりである．{'；'.join(C_HI)}．"
  f"残る条件（{by_corpus(C_NONE)}）では，{len(M50)} 個の手法の UAR の値は，どれも提案の手法の値より低い．")
w(f"   - **15 単位の上**：12 条件とも，勾配法の {len(M15)} 個の手法の UAR の値は，提案の手法の同じ 15 単位の上の UAR の値より低い．")
w("")
w("### 参照と参考，設定の点")
w("")
GO = groups(eget("M_F6_orc", "prod"))
must(len(GO["pp"]) + len(GO["p"]) == 12, "参照 − 積の読み")
EO = [q("M_F6_orc", "prod", *k_).est for k_ in ALL12]
OP = [level("M_F6_orc", *k_).uar - level("prop", *k_).uar for k_ in ALL12]
must(all(x > 0 for x in OP), "参照の UAR が提案の手法の UAR 以下の条件がある")
w(f"6. **参照の版（全ラベルで測った表．`M_F6_orc`．評価するコーパスの候補のうち K 本を除いた発話の全ラベルで信頼度を測る．実際には使えない）− 積は，12 条件とも区間が 0 を含まず正だった**（{fest(min(EO))}〜{fest(max(EO))}）．"
  f"{read_short(eget('M_F6_orc', 'prod'))}提案の手法 − 参照の対の差は集計に無い．UAR の点の値の引き算（参照 − 提案の手法．区間は無い）は，12 条件とも正で {fest(min(OP))}〜{fest(max(OP))} 点である．")
w(f"   - **参考の版：ソースだけで測った表（`M_F6_ext`．ソースのラベルだけで作る表．K 本を使わない）− 積**：{read_short(eget('M_F6_ext', 'prod'))}")
w(f"   - **提案の手法 − ソースだけで測った表**（`prop − M_F6_ext`）：{read_short(eget('prop', 'M_F6_ext'))}")
w(f"   - **参考の版：ソースどうしで学んだ重み（`M_F8`．ソースのラベルだけで作る重み．K 本を使わない）− 積**：{read_short(eget('M_F8', 'prod'))}")
w(f"   - **提案の手法 − ソースどうしで学んだ重み**（`prop − M_F8`）：{read_short(eget('prop', 'M_F8'))}")
LOTO_DIFF = [k_ for k_ in ALL12 if MY_PICK[k_]["loto"] != FZ[k_]["grid"]]
ORA_DIFF = [k_ for k_ in ALL12 if MY_PICK[k_]["oracle"] != FZ[k_]["grid"]]
must(all(MY_PICK[k_]["loto"] == 2 for k_ in LOTO_DIFF) and set(LOTO_DIFF) <= set(N0_20), "ほかの 3 コーパスで選んだ点")
must(all(klass(q("prop_loto", "prop", *k_)) == "same" for k_ in ALL12 if k_ not in LOTO_DIFF) and all(klass(q("oracle_grid", "prop", *k_)) == "same" for k_ in ALL12 if k_ not in ORA_DIFF), "同じ点の条件の差")
must(all(q("oracle_grid", "prop", *k_).est >= -1e-12 for k_ in ALL12), "自分のテストで最良の点 − 固定の点が負")
w(f"7. **設定の点**（格子 16 点．表の文書の 6 節に点と UAR）：ほかの 3 コーパスの評価の回で選んだ点（`prop_loto`）が固定の点と違う条件は {by_corpus(LOTO_DIFF)} で，"
  f"どれも固定の点が n0 = 20 の条件で，選ばれた点は{gdesc(2)}である．差（ほかの 3 コーパスで選んだ点 − 固定の点）は {vals(LOTO_DIFF, eget('prop_loto', 'prop'))} である．残る {12 - len(LOTO_DIFF)} 条件は同じ点である．"
  f"自分のテストで最良の点（`oracle_grid`．実際には使えない．差は作りの上で 0 以上）が固定の点と違う条件は {by_corpus(ORA_DIFF)} で，"
  f"差（最良の点 − 固定の点）は {vals(ORA_DIFF, eget('oracle_grid', 'prop'))} である．選ばれた点は表の文書の 6 節にある．"
  f"固定の点が n0 = 20 の 4 条件（{by_corpus(N0_20)}）では，n0 = 0 の点の UAR の点推定が固定の点より高い（要点 3 の「提案の手法 − 水準を戻さない」）．選び方 (b) は参考である（0 節）．")
w("")
w("### 計算の後の解析（探索）：提案の手法 − 積を，切片の分と向きの分に分ける")
w("")
GI = groups(pget("prop", "intercept", "adj"))
GD = groups(pget("prop", "direction", "ob"))
must(len(GI["z"]) == 12, "切片の分（切片をずらした答えどうし）の読み")
EI = [pq("prop", "intercept", "adj", *k_).est for k_ in ALL12]
must(not GD["n"] and not GD["nn"], "向きの分の読み")
w("8. **この解析は設計に無く，計算の後に足した探索の解析である**（0 節）．単位の保存には，版ごとに切片の扱いを変えた判定がある．"
  "「向きの分」を測る層（台本の `ob`）は，提案の手法と積のそれぞれに，**ラベル付きの候補の発話（K 本以外）の全ラベルで，対数尤度（感情ごとに等しい重み）が最大になるように合わせた切片**を足して判定した答えである．"
  "合わせるのは候補の発話の対数尤度で，テストの UAR が最大になる切片ではない．候補の全ラベルを使うので，実際には使えない参照である．"
  "定義は，向きの分 ＝ 提案の手法（この層）− 積（この層），切片の分 ＝ 全体 − 向きの分 である（全体は，切片をずらした答えどうしの提案の手法 − 積）．"
  "台本の定義では，2 つの版の切片を同じ規則で合わせるので，この層の差を，重みの向きの違いによる差として読む．切片の分は，全体から向きの分を引いた残りである．")
w(f"   - **切片の分**（切片をずらした答えどうしの全体から）：12 条件とも区間が 0 を含んだ．点推定は {fest(min(EI))}〜{fest(max(EI))} 点である（値は 2.5 節）．")
w(f"   - **向きの分**：{read_short(pget('prop', 'direction', 'ob'))}")
w(f"   - **切片をずらす前どうしの全体から分けた切片の分**：{read_short(pget('prop', 'intercept', 'raw'))}")
w("   - ほかの版（部品を外した版，比べる相手 A の 3 つ，参照，参考）の向きの分と切片の分は 2.5 節にある．残り方を掛けない版（`prop_home`）は，この層の判定の保存が無いので分けられない．"
  "比べる相手 A のうち，確率の平均，確信度で重みづける平均，別のコーパスで選んだ指数の積，温度の較正，K 本で合わせた重み，重みの表を探す版も，この層の判定の保存が無い．")
w("")

# ---- 2 節
w("## 2. 表")
w("")
w("### 2.1 表の文書（既にある表）")
w("")
w(f"UAR と差の表は `DAI/docs/実験20_結果の表_2026-11-15.md`（sha256 {sha(TABLE_DOC)[:8]}…．書き出しの台本 `DAI/docs/tools_pending/md_tables_dai20.py`）にあり，ここには写さない．"
  "その文書の 1 節が提案の手法と積（K 本の引き方で分けた値を含む），2 節が比べる相手 A（UAR，提案の手法 − 相手の 3 通りの組，相手 − 積），3 節が部品を 1 つ外した版，4 節が比べる相手 B と単独の分類器，"
  "5 節が比べる相手 C，6 節が参照・参考・設定の点である．この台本は，表の文書が今の集計から書き出し直した表と同じであることを確かめた（0 節）．")
w("")
w(f"表の文書は，本体が 2026-11-15 に書き出し直した（コミット {TD_COMMITS[0]}）．区間の端が 0.05 より 0 に近い欄と，差が丸めると ±1.00 になる欄に，符号と 1.0 点の境目が読めるように桁を足してある．"
  f"書き出し直す前の版（コミット {TD_COMMITS[-1]}）と欄ごとに比べると，変わった値は，区間の端 {TBL_ENDS} か所と差 {TBL_EST} か所だけである．ほかの値，印，行の数は変わっていない．文の行は，桁の決まりの 1 行が増えただけである．"
  "表の文書に，符号の読めない区間の端は残っていない．表の文書は負の符号を「-」，この文書は「−」と書く．この文書は，差が丸めると 0.00 になるときにも桁を足す（表の文書は足さない）．")
w("")
w("### 2.2 提案の手法 − 相手の読みの一覧")
w("")
w("欄の中は「コーパス K」で，その読みに入る条件を並べた．「0 を含まず」は区間が 0 を含まないことを表す．一覧は値を読みやすくするためのもので，判定ではない．欄の中の条件の数を数えて主張に使わない．値は表の文書と要点にある．"
  f"「正解の数が同じ」は，2 つの版の正解の数が全部の単位で同じ条件である．{DQ_SAME.rstrip('．')}（設計 5 節）．")
w("")
for title, a_of, b_of, opps in (("比べる相手 A：どちらも切片をずらした答え（`prop − 相手`）", lambda v: "prop", lambda v: v, ["prod"] + A_OPP),
                                ("比べる相手 A：どちらも切片をずらす前（`prop@raw − 相手@raw`）", lambda v: "prop@raw", lambda v: f"{v}@raw", ["prod"] + A_OPP),
                                ("比べる相手 A：提案の手法（切片をずらした答え）− 相手（切片をずらす前）（`prop − 相手@raw`）", lambda v: "prop", lambda v: f"{v}@raw", A_OPP),
                                ("比べる相手 B：提案の手法（推定時の答え．切片をずらした答え）− 相手（`prop − 相手`）", lambda v: "prop", lambda v: v, B_OPP),
                                ("提案の部品を 1 つ外した版：提案の手法 − 外した版", lambda v: "prop", lambda v: v, ABL)):
    w(f"**{title}**")
    w("")
    w("| 相手 " + CAT_HEAD)
    w("|---" + CAT_SEP)
    for v in opps:
        w(f"| {nm(v)} | {cat_cells(eget(a_of(v), b_of(v)))} |")
    w("")
w("### 2.3 相手 − 積，参照・参考 − 積の読みの一覧（どちらも切片をずらした答え）")
w("")
w("| 版 − 積 " + CAT_HEAD)
w("|---" + CAT_SEP)
for v in A_OPP + ["prop_noprof", "prop_noprof_sel", "prop_levelfree", "M_F6_orc", "M_F6_ext", "M_F8", "prop_loto", "oracle_grid"]:
    w(f"| {nm(v)} − 積 | {cat_cells(eget(v, 'prod'))} |")
w("")
w("### 2.4 提案の手法 − 積の分け方（設計 5 節の 2 つの順序．どちらも切片をずらした答え）")
w("")
w("先に相性を外す順序：提案の手法 − 積 ＝（提案の手法 − 相性を外す）＋（相性を外す − 積）．先に水準を外す順序：提案の手法 − 積 ＝（提案の手法 − 水準の違いを外す）＋（水準の違いを外す − 積）．"
  "点推定は，どちらの順序でも足すと提案の手法 − 積になる（台本が確かめた）．区間の端は足し算にならない．"
  "設計 5 節の名前（「行の中の差を使った分」「水準の分」「相性を使ったうえでの，分類器どうしの水準の違いの分」「水準の違いなしでの相性の分」）が当てはまる条件と，別の量が重なる条件は，要点 3 に書いた．"
  "固定の点が n0 = 20 の条件では，「相性を外す − 積」と「提案の手法 − 水準の違いを外す」に，戻し先（3 つの平均，またはソースで測った水準）が重なる．")
w("")
w("| コーパス | K | 固定の点 | 提案の手法 − 積 | 提案の手法 − 相性を外す | 相性を外す − 積 | 提案の手法 − 水準の違いを外す | 水準の違いを外す − 積 |")
w("|---|---|---|---|---|---|---|---|")
for c, K in ALL12:
    w(f"| {c} | {K} | {gdesc(FZ[(c, K)]['grid'])} | " + " | ".join(ci(q(a, b, c, K), mark=True) for a, b in (("prop", "prod"), ("prop", "prop_noprof"), ("prop_noprof", "prod"), ("prop", "prop_levelfree"), ("prop_levelfree", "prod"))) + " |")
w("")
w("「提案の手法 − 相性を外す」は，さらに 2 つの差に分かれる（設計 5 節）：提案の手法 − 分類器 × 感情の成分を消す（「分類器 × 感情の成分の分」）と，分類器 × 感情の成分を消す − 相性を外す（「当てはめに残る成分の分」）．")
w("")
w("| コーパス | K | 提案の手法 − 分類器 × 感情の成分を消す | 分類器 × 感情の成分を消す − 相性を外す |")
w("|---|---|---|---|")
for c, K in ALL12:
    w(f"| {c} | {K} | {ci(q('prop', 'prop_noint', c, K), mark=True)} | {ci(q('prop_noint', 'prop_noprof', c, K), mark=True)} |")
w("")
w("### 2.5 切片の分と向きの分（計算の後の解析．探索）")
w("")
w("`results/dai20_eval/posthoc/intercept_direction.csv` の値（K 本の引き方 0〜9）．版 − 積（どちらも切片をずらした答え）＝ 切片の分 ＋ 向きの分．"
  "向きの分は，版と積のそれぞれに，候補の発話（K 本以外）の全ラベルで対数尤度が最大になるように合わせた切片を足した答えどうしの差である（要点 8）．"
  "区間は，集計の「版 − 積」と同じ乱数の鍵で作ってある（台本の冒頭の説明）．印は読みやすさのための印で，判定ではない．"
  f"切片をずらす前どうしの「全体」は，この出力と集計（`prop@raw − prod@raw`）とで点推定が同じで，区間の端は，塊の重みを引き直す MOSEI でだけ違う（K 本の引き方 0〜9 の値で最大 {RAW_END_DEV['MOSEI']:.4f} 点．乱数の鍵が違うため）．"
  "この文書は，切片をずらす前どうしの提案の手法 − 積を集計から引いている（要点 1）．")
w("")
PH_LAB = {"prop": "**提案の手法**", "prop_loto": "ほかの 3 コーパスで選んだ点", "oracle_grid": "自分のテストで最良の点"}
for title, quantity, layer in (("向きの分", "direction", "ob"), ("切片の分（切片をずらした答えどうしの全体から）", "intercept", "adj")):
    w(f"**{title}**")
    w("")
    w("| 版 − 積 | K | " + " | ".join(C4) + " |")
    w("|---|---|" + "---|" * len(C4))
    for v in PH_VERS:
        for K in KS:
            w(f"| {PH_LAB.get(v, nm(v))} | {K} | " + " | ".join(ci(pq(v, quantity, layer, c, K), mark=True) for c in C4) + " |")
    w("")
w("**候補の全ラベルに合わせた切片の答え − 切片をずらした答え**（同じ版の中の差．`intercept_gap`）と，その層の UAR（`intercept_levels.csv`）")
w("")
w("| 版 | K | " + " | ".join(f"{c}：差 | {c}：全ラベルに合わせた切片の答えの UAR" for c in C4) + " |")
w("|---|---|" + "---|---|" * len(C4))
for v in ("prop", "prod"):
    for K in KS:
        w(f"| {nm(v)} | {K} | " + " | ".join(f"{ci(pq(v, 'intercept_gap', 'adj', c, K), mark=True)} | {PHL[(v, 'ob', c, K, 'all')].uar:.2f}" for c in C4) + " |")
w("")
w("### 2.6 感情ごとの再現率（提案の手法と積．切片をずらした答え）")
w("")
w("`per_emotion.csv` の `recall`（分割ごとに K 本の引き方の平均を取り，分割で平均した値）と 95% 区間．「引き算」は 2 つの点の値の引き算で，区間は作っていない．"
  "件数は，K 本の引き方 1 つぶんのテストの発話の数（5 分割の合計．別々の発話の数）である．")
w("")
w("| コーパス | K | 版 | " + " | ".join(EMO) + " |")
w("|---|---|---|" + "---|" * 4)
for c in C4:
    w(f"| {c} | — | テストの件数 | " + " | ".join(f"{int(N_PER_SEED[c][e]):,}" for e in range(4)) + " |")
    for K in KS:
        for v in ("prop", "prod"):
            w(f"| {c} | {K} | {nm(v)} | " + " | ".join(f"{PE[(v, c, K, e)].recall:.1f} [{PE[(v, c, K, e)].lo:.1f}, {PE[(v, c, K, e)].hi:.1f}]" for e in range(4)) + " |")
        w(f"| {c} | {K} | 引き算 | " + " | ".join(fest(edif(c, K, e)) for e in range(4)) + " |")
w("")
w("### 表の下の注（2 節に共通）")
w("")
w("- ↑・↓ は，区間が 0 を含まず差が 1.0 点以上の欄に付く．表を読みやすくするための印で，判定ではない．印の無い欄は「同じ水準」を意味しない．")
w(f"- 区間は 95% で，補正していない．差の対は 77 個，条件は 12 個，K 本の引き方で分けた値を入れると {len(ef):,} 個の区間がある．")
w("- K の 3 つは入れ子の K 本と同じテストの上の値である．K どうしの差の区間は作っていない．")
w("- AFEW の jackknife では分割 0 が除かれる塊があり，IEMOCAP では全部の分割が除かれる（0 節）．K 本の引き方の項の割合は MOSEI の行でだけ読む．")
w("")

# ---- 3 節
w("## 3. 検算と仮定")
w("")
w("### 3.1 検算（設計 6 節の表）")
w("")
CQ_REPRO = quote((CK / "code_v2.md").read_text(), "3 単位それぞれで，実験 16 の保存（K = 5・25）と同じ名前の配列 2,176 個（うち判定 2,084）と，K = 10 の追補の配列 1,104 個（うち判定 1,042）が，すべてビット単位で一致した．", "code_v2.md")
w("| 項目（設計 6 節） | 値 | 止める基準 |")
w("|---|---|---|")
w(f"| 作り直しが実験 1 と合う | K 本を全部使って作り直した積と，実験 1 の保存の積の差の最大 {sci(TD_MAX)}（{TD_ROW['T']} の K = {int(TD_ROW.K)}）．コーパスごとの最大："
  + "，".join(f"{c} {sci(TD_BY[c][1])}" for c in C4) + f"．5×10⁻⁵ を超えた（単位，K）は 600 組のうち {TD_OVER} 組 | 1×10⁻⁴ 以下 |")
w(f"| 最適化が解けている | 勾配の最大 {sci(conv['grad_max'])}，止まり {conv['stalled']} | 1×10⁻⁵ 以下 |")
w(f"| 同じ台本を仮のターゲットの回に当てると，実験 16 の保存と同じ判定になる | 台本の検査（`code_v2.md`）の報告：「{CQ_REPRO}」この台本は流し直していない | 台本の検査で確かめる |")
w(f"| 足した版の計算の道 | 保存の確率から作り直した提案の手法の判定と元の関数の判定の一致の最小 {conv['recheck_agree_min']:.6f}．等しい重みの表を同じ道に通した判定と積の一致の最小 {conv['eq_agree_min']:.6f}．"
  f"作り直した指数の積の判定（切片をずらす前）と実験 1 の判定の一致の最小 {conv['psel_agree_min']:.6f}（どれも 600 組の最小） | 1 未満なら数を書く |")
w(f"| 切片の反復 | 重みの表を探す版で，上限（3,000 回）までに収束しなかった表の数 {WF_CAPPED} | 記録 |")
w(f"| 固定のファイル | 単位の計算と集計の記録で sha256 が同じ（{prov['frozen_sha256'][:8]}…）．今のファイルとも同じ | 違えば止める |")
w("| 等しい重みの表（人工の数） | `logs/dai20_selftest.log` に「selftest：通った」 | — |")
w(f"| 写した値 | 比べる相手 B の 7 つ × 12 条件 ＝ 84 行．実験 1 の集計の水準との差の最大 {sci(conv['others_B_dev_max'])} | 1×10⁻⁶ 以下．84 行 |")
w(f"| 続きから流す条件 | 途中から流し直していない（ログの「済み」の行 {N_SKIP}）．単位のコミットは 1 種類（{UNIT_COMMIT[:7]}） | 計算に関わるファイルに差があれば止める |")
w(f"| 切片のずらしの残り | 候補の上の確率の平均と目標の差の最大 {sci(conv['adj_resid_max'])} | — |")
w("")
w("### 3.2 仮定を測る表（設計 6 節．基準は置かない．コーパス × K ごとに並べ，K をまたいで平均しない）")
w("")
w("**相性はソースからターゲットへ持ち越せるか**（`assumptions.csv`．ソースで測った表 `ext` と，評価するコーパスの候補の全ラベルで測った表を比べた量．単位の平均）．"
  "「成分の相関」は，表から「全体の平均 ＋ 行の差 ＋ 列の差 ＋ λ × 行の差 × 列の差」の当てはめを引いた残り（分類器 × 感情の成分）の相関．"
  "「入れ替えの割合」は，感情の並びを 24 通りに入れ替えたとき，成分の相関が元の並びの値以上になる割合（元の並びを含む）．「行の中の差の相関」は，表から行の平均を引いた値（相性）の相関．"
  "「共通の感情の成分の相関」は列の平均の相関．「水準の相関」は行の平均（3 つの値）の相関．「最も強いモダリティの一致」は，行の平均が最大のモダリティが一致した単位の割合．")
w("")
w("| コーパス | K | 成分の相関 | 入れ替えの割合 | 行の中の差の相関 | 共通の感情の成分の相関 | 水準の相関 | 最も強いモダリティの一致 | 行ごとの相関（声・顔・言葉） | 顔の行を除いた相関 |")
w("|---|---|---|---|---|---|---|---|---|---|")
AS = {(r.c, int(r.K)): r for r in asm.itertuples()}
must(len(AS) == 12, "assumptions.csv の行")
for c, K in ALL12:
    r = AS[(c, K)]
    w(f"| {c} | {K} | {r.intcorr_M_ext:.2f} | {r.intperm_M_ext:.2f} | {r.profcorr_M_ext:.2f} | {r.emocorr_M_ext:.2f} | {r.levelcorr_M_ext:.2f} | {r.topok_M_ext:.2f} | "
      f"{r.profcorr_row_A:.2f}・{r.profcorr_row_V:.2f}・{r.profcorr_row_T:.2f} | {r.profcorr_noV:.2f} |")
w("")
w("**K 本で測った表と，d′ の読み**（K 本で測った表と，全ラベルで測った表を比べた量．単位の平均）．"
  "d′ は AUC を 0.02〜0.98 に切ってから作るので上限は 2.90 である．「上限の割合」「0 以下の割合」は，K 本で測った 12 個の d′ のうち，上限に達した割合と 0 以下だった割合．")
w("")
w("| コーパス | K | 成分の相関 | 行の中の差の相関 | 水準の相関 | 最も強いモダリティの一致 | 上限の割合 | 0 以下の割合 |")
w("|---|---|---|---|---|---|---|---|")
for c, K in ALL12:
    r = AS[(c, K)]
    w(f"| {c} | {K} | {r.intcorr_M_K:.2f} | {r.profcorr_M_K:.2f} | {r.levelcorr_M_K:.2f} | {r.topok_M_K:.2f} | {r.dk_ceiling_share:.3f} | {r.dk_zero_share:.3f} |")
w("")
w("**水準**（d′ の行の平均．声・顔・言葉の順．単位の平均）：K 本で測った水準，候補の全ラベルで測った水準，ソースで測った表 `ext` の水準．")
w("")
w("| コーパス | K | K 本 | 全ラベル | ソース（ext） |")
w("|---|---|---|---|---|")
for c, K in ALL12:
    r = AS[(c, K)]
    w(f"| {c} | {K} | " + " | ".join("・".join(f"{getattr(r, f'level_M_{s_}_{m}'):.2f}" for m in MODS) for s_ in ("K", "orc", "ext")) + " |")
w("")
w("**そのほかの仮定**：「テストとのずれ」は，候補の全ラベルで測った表と，テストで測った d′ の差の絶対値の平均．「一致の余り」は，本当の感情を固定したときに 2 つの分類器の判定が一致した割合から，"
  "独立なら見込まれる割合を引いた値（声と顔，声と言葉，顔と言葉．作りの上で正に出る）．「広がりの比」は，同じ感情の中の点の標準偏差の，3 つの分類器の最大 ÷ 最小．")
w("")
w("| コーパス | K | テストとのずれ | 一致の余り（声と顔・声と言葉・顔と言葉） | 広がりの比 |")
w("|---|---|---|---|---|")
for c, K in ALL12:
    r = AS[(c, K)]
    w(f"| {c} | {K} | {r.drift_M:.3f} | {r.excess_AV:.3f}・{r.excess_AT:.3f}・{r.excess_VT:.3f} | {r.sdratio_M:.2f} |")
w("")
w("**残り方は感情で共通か**（`source_tables.csv`．評価するコーパスごとに 1 つ．残り方 ＝ 別のソースに当てた d′ の合計 ÷ ソースの中の d′ の合計．提案の手法は，モダリティごとに 1 つの残り方を使う）")
w("")
w("| 評価するコーパス | モダリティ | 使った残り方（感情に共通） | " + " | ".join(f"{e} の残り方" for e in EMO) + " |")
w("|---|---|---|" + "---|" * 4)
must(len(stab) == 12 and bool((stab["T"] == stab["c"]).all()), "source_tables.csv の行")
for c in C4:
    for m in MODS:
        r = stab[(stab["T"] == c) & (stab["mod"] == m)].iloc[0]
        w(f"| {c} | {MOD_JP[m]} | {r.ret:.3f} | " + " | ".join(f"{getattr(r, f'ret_emo_{e}'):.3f}" for e in range(4)) + " |")
w("")
w("**2 ソースで選んだ点が 3 ソースでも同じ意味を持つか**（`est_grid.csv`・`grid_levels.csv`）：K 本での見込み（入れ子．テストのラベルを使わない）で，固定の点が格子 16 点の中で何位か（同点は同じ順位．n0 = 0 の点は戻し先が式に入らないので，eq と ext が同点になる）．"
  "右の 3 列は，テストの UAR（切片をずらした答え）の，格子 16 点の最小と最大，固定の点の値である．")
w("")
w("| コーパス | K | 固定の点 | 見込みの対数尤度の順位 | 見込みの UAR の順位 | 見込みの UAR | 格子の UAR の最小〜最大 | 固定の点の UAR |")
w("|---|---|---|---|---|---|---|---|")
for c, K in ALL12:
    r = eg[(eg.c == c) & (eg.K == K) & eg.frozen]
    must(len(r) == 1 and int(r.grid.iloc[0]) == FZ[(c, K)]["grid"], f"est_grid.csv の固定の点 {c} K={K}")
    r = r.iloc[0]
    x_ = gl[(gl.c == c) & (gl.K == K)]
    w(f"| {c} | {K} | {gdesc(int(r.grid))} | {int(r.rank_ll)} | {int(r.rank_uar)} | {r.est_uar:.2f} | {x_.uar_adj.min():.2f}〜{x_.uar_adj.max():.2f} | {float(x_[x_.frozen].uar_adj.iloc[0]):.2f} |")
w("")
w("**使った重み**（`weights.csv` の `w_prop`．提案の手法の重みの表の，50 単位の平均．12 個の平均が 1 になる大きさ．全ラベルの表・K 本だけの表・ソースだけの表の重みは同じファイルの `w_orc`・`w_K`・`w_ext`）")
w("")
w("| コーパス | K | モダリティ | " + " | ".join(EMO) + " |")
w("|---|---|---|" + "---|" * 4)
WT = {(r.c, int(r.K), r.mod, int(r.emotion)): r for r in wt.itertuples()}
must(len(WT) == 144 and set(wt.n_units) == {50}, "weights.csv の行")
for c, K in ALL12:
    for m in MODS:
        w(f"| {c} | {K} | {MOD_JP[m]} | " + " | ".join(f"{WT[(c, K, m, e)].w_prop:.2f}" for e in range(4)) + " |")
w("")
w("**比べる相手 A の合わせた量**（`competitor_fit.csv`．50 単位の平均）：温度の較正と K 本で合わせた重みは，合わせた重み（声・顔・言葉）と，重みが範囲の端（0 か 5）に達した割合．"
  "重みの表を探す版は，等しい重みの表だけが選ばれた単位の割合（その単位では積と同じ判定），最大に並んだ表の数の平均，選んだ表の K 本の上の UAR（候補の中の最大の値で，上に偏る．設計 4 節），テストの UAR（切片をずらした答え）．")
w("")
w("| コーパス | K | 温度の較正：重み／端の割合 | K 本で合わせた重み：重み／端の割合 | 同上（切片も合わせる）：重み／端の割合 | 候補 10：等しい重みだけの割合／並んだ数／K 本の UAR／テストの UAR | 候補 100 | 候補 1,000 |")
w("|---|---|---|---|---|---|---|---|")
CF = {(r.c, int(r.K), r.version): r for r in cfit.itertuples()}
must(len(CF) == 84, "competitor_fit.csv の行")
for c, K in ALL12:
    cells = []
    for v in ("tcal", "temp", "tempb"):
        r = CF[(c, K, v)]
        cells.append(f"{r.w_A:.2f}・{r.w_V:.2f}・{r.w_T:.2f}／{r.w_bound:.3f}")
    for v in ("wf10", "wf100", "wf1000"):
        r = CF[(c, K, v)]
        must(abs(r.uar_adj - level(v, c, K).uar) < 1e-9, f"competitor_fit.csv の UAR {c} K={K} {v}")
        cells.append(f"{r.eq_only_share:.2f}／{r.wf_n_tied:.2f}／{r.wf_uar_k:.2f}／{r.uar_adj:.2f}")
    w(f"| {c} | {K} | " + " | ".join(cells) + " |")
w("")
MS = {K: q("prop", "prod", "MOSEI", K) for K in KS}
must(all(abs(x.se ** 2 - (x.v_cluster + x.v_seed)) < 1e-9 for x in MS.values()), "MOSEI の分散の 2 つの項")
w("**MOSEI の区間の 2 つの項**（提案の手法 − 積．`effects.csv` の `v_cluster`・`v_seed`．標準誤差の 2 乗 ＝ 塊の項 ＋ K 本の引き方の項）：K 本の引き方の項の割合は，"
  + "，".join(f"K = {K} で {100 * MS[K].v_seed / (MS[K].v_cluster + MS[K].v_seed):.0f}%" for K in KS) + " である．")
w("")

# ---- 4 節
w("## 4. 限界と，まだ出していない量")
w("")
w("### 限界（設計 9 節と 0 節の申告）")
w("")
w("- この評価の回の値は，手法を選んだときに見たテストと同じテストの上の値で，探索の結果として扱う（冒頭の 4 項）．テストの発話が下調べと重ならないコーパスでの値は，この実験には無い．")
w("- 手法は評価に使ったのと同じ 4 コーパスの上で作った．設定はソースが 2 つの下調べで選んだ点である．固定の点が n0 = 20 の 4 条件では，n0 = 0 の点の UAR の点推定が固定の点より高かった（要点 3・7）．")
w("- 設計 5 節の部品の名前（「水準の分」など）は，固定の点が n0 = 20 の 4 条件では，戻し先が重なった差に当たる（要点 3）．")
w("- 比べる相手 A のうち 3 つ（水準だけの重み，K 本だけで測った表，取り違えの表で読み替えて掛ける）との差は，提案の手法に有利な向きに偏る．")
w("- 区間は補正していない．区間はソースの取り方，学習に使う発話，手法と設定の選び方の揺れを含まない．IEMOCAP は自由度 3 で区間が広い．K どうしの差の区間は無い．")
w("- 比べる相手 B・C は学び直した 1 つの分類器で，提案の手法は学び直す前の 3 つの分類器の答えの組み合わせである．C は UAR を並べただけで，対の差も区間も無い．")
w("- 1 組のエンコーダ，凍結した特徴の上の線形の分類器，4 コーパス，IEMOCAP は即興だけである．話者と内容の分離とコーパスごとの事情は 0 節に書いた．")
w("- 計算の後の解析（切片の分と向きの分）は設計に無い探索である．流した台本は，検査（`code_post_tools_v2.md`）が通した版に記録の直しを入れた後の版で，"
  + ("直しの差分の再検査は gate-code に頼んであり，報告はまだ無い．" if V3_VERDICT is None else f"直しの差分の再検査（`code_post_tools_v3.md`）の判定は「{V3_VERDICT}」である．")
  + "向きの分を測る層は候補の全ラベルを使う参照である．")
w("- 計算の前の検査のうち，比較の公平さ・漏れ・査読想定の 2 役は，比べる相手を足した第 4・5 版を再検査していない（0 節）．")
w("")
w("### まだ出していない量")
w("")
w("- **感情ごとの再現率の対の差と区間**．`per_emotion.csv` は版ごとの再現率と区間を持つ．提案の手法 − 積の感情ごとの差の区間は作っていない（単位の保存の判定から作れる）．")
w("- **提案の手法 − 参照（全ラベルで測った表）の対の差**．集計の差の一覧に無い．")
w("- **比べる相手 C との対の差と，C の UAR の区間**．実験 2 は実験 1 と同じ単位の上にあるので，実験 2 の単位の保存から作れる（設計 4 節）．作っていない．")
w("- **K どうしの差**．区間を作らない決まりである（設計 5 節）．")
w("- **性別ごとの内訳**．AFEW と IEMOCAP でしか出せない（設計 9 節）．出していない．")
w("- **切片をずらすと判定が変わる発話の割合**（設計 4 節が下調べの 12 単位で並べた量）．評価の回の単位では数えていない（単位の保存の判定から数えられる）．")
w("- **残り方を掛けない版と，候補の全ラベルに合わせた切片の層の保存が無い相手の，切片の分と向きの分**（要点 8）．")
w("- **提案の手法の答えを疑似ラベルにした学び直し**．この実験の範囲の外である（設計 9 節）．")
w("- **GPU のメモリの最大**．記録が無く，後から出せない（0 節）．")
w("")
w("集計にあるが，この文書で引いていない量：K 本で決めた切片を足した層（`intercept_direction.csv` の `kb`），K 本の引き方で分けた値のうち提案の手法 − 積のほかの差（`effects.csv` の `seeds` が `0-1`・`2-9` の行），"
  "単独の分類器と比べる相手の感情ごとの再現率（`per_emotion.csv`），積の混同行列（`confusion.csv`），ソースで測った表の値（`source_tables.csv` の `ext_*`・`home_*`），"
  "ソースの中の値・別のソースに当てた値で測った仮定の量（`assumptions.csv` の `*_M_home`・`*_M_away`）．")
w("")

# ---- 末尾
w("## 5. 数値の桁の決まり")
w("")
w("- UAR は小数 2 桁，UAR の区間の端は小数 1 桁．差は符号つきの小数 2 桁，差の区間の端は符号つきの小数 1 桁で，「差 [下端, 上端]」の形で書く．負の符号は「−」．")
w("- 差の区間の端が 0.05 より 0 に近いときは，区間が 0 を含むかどうかが読めるように桁を足す（例：+0.01）．差が丸めると ±1.00 か 0.00 になるときも，1.0 点の境目と符号が読めるように桁を足す．"
  "表の文書も同じ決まりで桁を足している（差が 0.00 になるときを除く）．")
w("- 読みの 3 つの分け方は，丸める前の値で決めている：区間の下端 > 0（または上端 < 0）で「区間が 0 を含まない」，差の絶対値 ≥ 1.0 で「1.0 点以上」．")
w("- 正解の数が全部の単位で同じ 2 つの版の差は「0（同じ数）」と書く．正解の数で比べているので，判定そのものが同じとは限らない．")
w("- 相関と割合は小数 2〜3 桁，検算の量は 2 桁の指数の形で書く．")
w("")
w("## 6. 変更の記録")
w("")
w("- 第 1 版（2026-11-15）：照合（gate-claim）の前の版．最初に書き出した後（コミット 73fd3aa）に，本体が台帳と表の文書を直したので，同じ第 1 版のまま次の直しを入れて書き出し直した．")
w("   1. 0 節の計算の後の解析の段を，直した台帳（検査が通した版，流した版，起動と出力の時刻，差分の再検査を頼んだこと）に合わせた．台帳との食い違いの記述は，台帳が直ったので消した．"
  "再検査の報告（`code_post_tools_v3.md`）は，ファイルがあれば判定を引き，無ければ「まだ無い」と書く作りにした．")
w("   2. 0 節の「照合の前に値を使った先」に，本体の申告（研究者への会話での報告 2 回，作業の覚え書き）と，実験 21 の設計・台本・固定のファイルが結果を開いた後に変わっていないこと（git の記録）を足した．")
w("   3. 計算の後の解析を入れると決めた出どころを，`DAI/docs/論文のセッションへの伝言_2026-11-14.md` の 1 節の 3 にした．")
w("   4. 2.1 節の「桁を足した値の表」を外した．表の文書が桁を足して書き出し直されたためである．代わりに，書き出し直す前の版から変わった値が区間の端と差の桁だけであることを台本が確かめる．")
w("   5. 要点（1 節）を縮めた．相手ごとに 12 条件の値を文で並べるのをやめ，区間が 0 を含まない条件と点推定が負の条件を名前と値で書き，残りは名前だけを書いて表を指す．標準の手法は，UAR の値が提案の手法より高い手法がある条件だけを挙げる．")
w("   6. 部品を外した版の差を，設計 5 節の表の名前で差ごとに書いた．最初の書き出しで付けていたまとめの呼び名（差を水準と相性の 2 つにまとめる呼び名）をやめた．"
  "両側の違いが 1 つの部品だけでない条件（固定の点が n0 = 20 の条件）には設計の名前を使わず，重なっている量を書いた．2.4 節の表の見出しも差の名前にした．")
w("")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()
text = "\n".join(L) + "\n"
must("、" not in text and "。" not in text, "句読点に「、」「。」がある")
body = text
for s_ in sorted(set(QUOTED), key=len, reverse=True):                     # 元の文書から引いた文は，語の検査から除く
    body = body.replace(s_, "")
body = body.replace("再現率", "")                                          # 「再現率」は指標の名前（K 本の引き方 2〜9 の値を「再現」と書かない決まりの検査から除く）
for bad in ("セル", "当たり", "外れ", "有意", "勝", "負け", "差が無い", "差がない", "差は無い", "差はない", "K で増え", "K で減", "K が大きいほど", "K とともに", "切片を最良", "最良にそろえ",
            "再現", "確認", "効い", "効く", "効か", "効き", "上回", "下回", "生徒", "教師", "先生", "出発点", "明らか", "顕著", "劇的"):
    must(bad not in body, f"使わないと決めた語がある：{bad}")
must(not re.search(r"\[[+−-]0\.0, |, [+−-]0\.0\]", text), "符号の読めない区間の端がある")
for bad in ("水準に関わる項", "相性に関わる項"):
    must(bad not in body, f"やめた呼び名がある：{bad}")
Path(args.out).write_text(text)
print(f"書いた：{args.out}（{len(L)} 行）")
