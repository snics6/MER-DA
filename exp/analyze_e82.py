"""【本番の感度】E82（CPU）：P4 の手法の C′ を選び直した版の P4・P4′ の判定を，E79 の関数で出す（E79 の台本は変えない）。

設計は `docs/exploratory-analyses.md` の「E82．P4 の手法の出発点への罰則を選び直す」の 2・3 節．入力は E77（`exp/compare_dai.py`），E78（`exp/compare_dai2.py`），
E82（`exp/s43_penalty.py`．entmin・IM・自己学習の C′ を E39 の値の 3 倍・10 倍にした行）の出力で，学び直しをしない．GPU を使わない。

2 つの版（--grid）を，E79 の `--families p4` の段（delta → select → measure → judge）でそれぞれ流す：
  fixed    ：E77・E78 の行だけ（E79 の P4 の判定そのもの．E79 の出力があれば compare の段で --fixed-from に渡して読むだけでもよい）
  reselect ：entmin・IM・自己学習の候補を「E77 の倍率 1 の行 ∪ E82 の倍率 3・10 の行」に替える．同点は倍率の小さい方（1 → 3 → 10），倍率が同じ
             候補の間は E79 の 'method|pnorm|meff' の文字列の順（設計の 2 節）．ほかの手法と規則（excl の変種，none を除いて選ぶ，区間，Holm の族）は E79 のまま
比べは 2 つ（設計の 2 節．関門 1 の define D10・compare）：(a) E79 と同じ「none（E39 の C′）− 選んだ手法」（judge の段），
(b) 対照「none（選んだ手法と同じ C′）− 選んだ手法」（judge_b の段．reselect の版だけ．倍率 1 の単位と K=0 では (a) と同じ）．
「働かなかった」は同じ C′ の none との予測の食い違いで判定する（(a)・(b) で共通の材料）。

E79 の関数への差し込み（reselect の版だけ．`install`）：E79 は 3 つ目の波を E80 の段 c の場所（大域の E80C，波の名前 e80c）から読む．
E82 の行はこの枠に置く（ここでは E80C を E82 の場所だけにする．本物の段 c の行は読まない．P4 の判定には使わない行）．
  - `E80C_METHODS` ＝ entmin・im・self（E77 の none との照合は none の行で E79 のとおり行い，候補は 3 手法の行だけ）
  - `sel_defs`：entmin・self・im の族に，E77 の行と同じ判定の関数で e80c の波の行を足す（im は E79 と同じく m_eff ∈ {100, 500, 2000, n}）
  - `check_e80c_prov`：E82 の provenance の照合（実験 E82，実行の名前，K ＝ 0・5，倍率 3・10，dirty で流していないか）
  - `diagnostics`：E79 の診断に，e80c の波から選んだ設定の「働かなかった」の材料（選んだ設定と同じ C′ の none との予測の食い違い．種 0 と種の平均）を
    E79 と同じ式で足す（E79 の診断は e77 以外の波の行を E78 の表で探すので，足さないと E82 から選んだ単位が「働かなかった」に替わらない）
  - `choose`：E82 の行を含む候補だけ，同点は倍率の小さい方（倍率が同じなら E79 の文字列の順）．E82 の行を含まない族は E79 の choose のまま
  - `read_rows`：E82 の場所の表の倍率の C′ の none（method none，param `-C3x`・`-C10x`）を method `none_cp` に読み替える（E79 は e80c の波の none を
    1 つの鍵に 1 行と決めて E77 の none と照らすので，E39 の C′ の none だけを none として見せる）
  - `key_table`：none_cp の鍵を測る鍵に足す（比べ (b) の相手）
  - `Store`（judge_b の段だけ）：E79 の判定の compare が手法の鍵のすぐ後に読む none の鍵を，直前の手法の鍵の倍率の none_cp に替える（`StoreB`）
  出力の表では波の名前 e80c を e82 と書き直す（compare の段）。

段（--stage．--grid ごとに `<out>/<grid>/<段>/` に書く．既にあれば止まる．E79 の段の間の照合（δ・chosen.csv・fixed.csv のコミット）もそのまま）
  delta   ：E79 と同じ δ（E77 の仮のターゲットの none だけから）．reselect では E82 の `_ps_*_none.npz` を E77 と照らす（`e82_ps_none_check.csv`）．
            --e79-delta を渡すと E79 の delta.csv と sha256 が同じかを確かめる
  select・measure・judge：E79 の段そのもの（差し込みの後）．judge は比べ (a)
  judge_b ：reselect の版の measure の出力から比べ (b) の判定を作る（E79 の判定の段を StoreB で流す）
  compare ：`judgments.csv`（3 手法 × 族 × K × 感度 × 版（固定，(a)，(b)）ごとの判定の数），`judgment_units.csv`（単位ごとの 3 つの判定と変わったか），
            `statements.csv`（筋 4 の文の鍵：族の全手法（E79 の statements.csv）と手法ごと（E79 の規則を 1 手法の組み合わせに当てる）．固定・(a)・(b)），
            `decision.csv`（設計の 3 節の書き方の規則：手法 × 族ごとに tag main・subset all の鍵で，(a) が同じ／(a) が変わり (b) も同じ向き／
            (a) が変わり (b) は同じ．unseen_only の鍵と (2′) の same_combos は記述の列．限定を外せるのは entmin・IM・自己学習だけ），
            `candidates.csv`（候補の数），`oracle_dir_both.csv`・`oracle_dir_summary.csv`（2 つの版を格子の大きさを添えて），
            `multipliers.csv`・`multiplier_units.csv`（倍率ごとに選ばれた単位の数（（実行，生徒，K，手法）ごとにも），10 倍の数と過半か（母数は手法が選ばれた単位），
            同点に複数の倍率が並んだ単位の数，同点で倍率の小さい方を選んだ単位の数，固定の版と同じ設定か）
画面には行の数と場所だけを出す（評価の値は出さない）。

    ./.venv/bin/python exp/analyze_e82.py --stage delta --grid reselect --e79-delta results/e79/delta/delta.csv
    ./.venv/bin/python exp/analyze_e82.py --stage select --grid reselect      （chosen.csv の sha256 を文書に書いてコミットしてから measure）
    ./.venv/bin/python exp/analyze_e82.py --stage measure --grid reselect
    ./.venv/bin/python exp/analyze_e82.py --stage judge --grid reselect
    ./.venv/bin/python exp/analyze_e82.py --stage judge_b --grid reselect
    ./.venv/bin/python exp/analyze_e82.py --stage compare --fixed-from results/e79
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                         # GPU を使わない（E79 と同じ）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
import analyze_e79 as A  # noqa: E402

SLOT = "e80c"                                                                  # E79 の関数が 3 つ目の波を読む名前（E82 の行をここに置く）
E82_METHODS = ("entmin", "im", "self")
E82_MULTS = (3.0, 10.0)
GRIDS = ("fixed", "reselect")
P4_FAMS = ("P4", "P4'")
MULT_RE = re.compile(r"C([0-9.]+)x$")
STAGES = ("delta", "select", "measure", "judge", "judge_b", "compare")
NONE_CP = "none_cp"                                                            # E82 の倍率の C′ の none（param -C3x・-C10x）を読み手の中で呼ぶ名前
ORIG = {"sel_defs": A.sel_defs, "check_e80c_prov": A.check_e80c_prov, "diagnostics": A.diagnostics, "E80C_METHODS": A.E80C_METHODS,
        "choose": A.choose, "read_rows": A.read_rows, "key_table": A.key_table, "Store": A.Store}
RULE_TEXT = {True: "3 手法のすべてで文の鍵が固定の版と同じ：筋 4 の文を罰則の限定なしで書き，E82 を感度の表に載せる",
             False: "文の鍵が変わった手法がある：筋 4 の文に「E39 の罰則で出発点につないだとき」と限定を付け，選び直した版の文の鍵と判定を並べて書く"}


def mult_of(wave, method, pnorm):
    """選んだ行の倍率（E77 の行は 1，E82 の行は param の C<倍率>x．none と候補なしは NaN）。"""
    if not isinstance(method, str) or method in ("none", A.EMPTY) or wave not in ("e77", SLOT, "e82"):
        return np.nan
    if wave == "e77":
        return 1.0
    m = MULT_RE.search(str(pnorm))
    if m is None:
        raise SystemExit(f"E82 の行の param に倍率の印が無い：{method} {pnorm}")
    return float(m.group(1))


# ---------------------------------------------------------------- E79 の関数への差し込み

def choose_e82(cs, none_score):
    """E79 の choose（analyze_e79.py 570〜580 行）の同点の決め方だけを変えたもの（2026-11-02 の本体の決定）：仮のターゲットの点数がちょうど同じ
    候補が並んだら倍率の小さい方（1 → 3 → 10）を選び，倍率が同じ候補の間は E79 と同じ 'method|pnorm|meff' の文字列の順の先頭．
    候補に E82 の行（e80c の波）が無ければ E79 の choose をそのまま呼ぶ（E82 の行が入るのは entmin・im・self の族だけ）。"""
    if cs.empty or not (cs.wave == SLOT).any():
        return ORIG["choose"](cs, none_score)
    cs = cs.assign(_k=cs.method + "|" + cs.pnorm + "|" + cs.meff,
                   _m=[mult_of(w, m, p) for w, m, p in zip(cs.wave, cs.method, cs.pnorm)]).sort_values(["_m", "_k"])
    if cs._m.isna().any():
        raise SystemExit(f"E82 の行を含む候補に倍率の分からない行がある：{sorted(set(cs[cs._m.isna()].method))}")
    best = cs.score.max()
    tie = int((cs.score == best).sum())
    ex = cs[cs.score == best].iloc[0]
    inc = ex if best > none_score else None
    return ex, inc, tie


def read_rows_e82(path):
    """E79 の read_rows．E82 の場所の表では，倍率の C′ の none の行（method none，param が '-' でない）を method `none_cp` に読み替える．
    E79 は e80c の波の none を（生徒, K）などで 1 行と決めて E77 の none と照らす（analyze_e79.py 606・1470 行）ので，E39 の C′ の none だけを none として見せる。"""
    d = ORIG["read_rows"](path)
    if any(Path(path).resolve().parent == Path(x).resolve() for x in A.E80C.values()) and len(d) and "method" in d.columns:
        m = (d.method == "none") & (d.param.astype(str) != "-")
        if m.any():
            d.loc[m, "method"] = NONE_CP
    return d


def key_table_e82(sel, fixed, run, T, tab=None, extra=()):
    """E79 の key_table に，倍率の C′ の none（none_cp）の鍵を足す（比べ (b) の相手．ブートストラップの複製を作るため）。"""
    ks = list(ORIG["key_table"](sel, fixed, run, T, tab, extra))
    t = (tab or {}).get(SLOT)
    if t is not None:
        x = t[(t.method == NONE_CP) & t.K.isin(A.KS)][["student", "K", "pnorm", "meff"]].drop_duplicates()
        ks = sorted(set(ks) | {(r.student, r.K, SLOT, NONE_CP, r.pnorm, r.meff, "-") for r in x.itertuples(index=False)})
    return ks


class StoreB(ORIG["Store"]):
    """比べ (b)：「none（選んだ手法と同じ C′）− 選んだ手法」．E79 の判定の compare は単位の群ごとに，手法の鍵を読んだすぐ後に none の鍵
    （e77・none・'-'）を読む（analyze_e79.py 2084〜2085 行）．直前に読んだ手法の鍵の倍率が 1 でなければ，none の鍵を同じ倍率の none_cp に替えて返す．
    倍率が 1 の行（E77・E78 の行）と K=0（倍率の none は E39 の none と同じ）では (a) と同じ値になる（compare の段で確かめる）。"""

    last = None

    def __init__(self, mdir, runs):
        super().__init__(mdir, runs)
        self._last, self.n_redirect = {}, 0
        StoreB.last = self

    def get(self, run, T, st, K, wave, meth, pn, meff, md):
        u = (run, T, st, K)
        if meth == "none" and wave == "e77" and pn == "-":
            m = self._last.get(u, 1.0)
            if np.isfinite(m) and m != 1.0:
                self.n_redirect += 1
                return super().get(run, T, st, K, SLOT, NONE_CP, f"-C{m:g}x", "0", "-")
            return super().get(run, T, st, K, wave, meth, pn, meff, md)
        m = mult_of(wave, meth, pn) if meth in E82_METHODS else 1.0
        self._last[u] = 1.0 if not np.isfinite(m) else m
        return super().get(run, T, st, K, wave, meth, pn, meff, md)


def sel_defs_e82(role, kept=(), noj_new=((), ())):
    """E79 の sel_defs に，entmin・self・im の族の E82 の行（e80c の波．E77 の行と同じ判定の関数）を足す。"""
    D = ORIG["sel_defs"](role, kept, noj_new)
    for m in E82_METHODS:
        D[m] = list(D[m]) + [(SLOT, fn) for w, fn in D[m] if w == "e77"]
    D["groupB"] = [x for k in A.GROUP_B if k in D for x in D[k]]
    return D


def check_e82_prov(run, args, issues, kept_sha=None):
    """E82 の provenance.json と，同じコミットの provenance_log.jsonl の記録を確かめる（E79 の check_e80c_prov の代わり）。"""
    d = A.E80C.get(run)
    if d is None:
        return None
    f = Path(d) / "provenance.json"
    if not f.exists():
        A.fail_or_note(args, issues, check="e82_provenance_missing", run=run, path=str(f))
        return None
    p = json.loads(f.read_text())
    recs = [("provenance.json", p)]
    lg = Path(d) / "provenance_log.jsonl"
    if lg.exists():
        recs += [(f"provenance_log.jsonl:{i}", e) for i, e in enumerate(json.loads(x) for x in lg.read_text().splitlines() if x.strip())
                 if e.get("commit") == p.get("commit")]
    for where, e in recs:
        ks = [str(k) for k in (e.get("KS") or [])]
        mults = sorted(float(m) for m in (e.get("mults") or []))
        issues.append({"check": "e82_provenance", "run": run, "where": where, "ks": ks, "mults": mults, "dirty": e.get("dirty"),
                       "allow_dirty": e.get("allow_dirty")})
        if e.get("experiment") != "E82" or e.get("run") != run:
            raise SystemExit(f"{f}（{where}）は E82 の {run} の出力でない（{e.get('experiment')} {e.get('run')}）")
        if ks != list(A.P4_KS):
            A.fail_or_note(args, issues, check="e82_ks_mismatch", run=run, where=where, ks=ks, expected=list(A.P4_KS))
        if mults != list(E82_MULTS):
            A.fail_or_note(args, issues, check="e82_mults_mismatch", run=run, where=where, mults=mults, expected=list(E82_MULTS))
        if e.get("dirty") or e.get("allow_dirty"):
            A.fail_or_note(args, issues, check="e82_dirty_or_allow_dirty", run=run, where=where)
    return {"path": str(d), "commit": p.get("commit"), "dirty": p.get("dirty"), "script_sha256": p.get("script_sha256"),
            "compare_dai_sha256": p.get("compare_dai_sha256"), "KS": p.get("KS"), "mults": p.get("mults"), "none_ks": p.get("none_ks"),
            "nc_seeds": p.get("nc_seeds"), "n_records_checked": len(recs)}


def chosen_row(sel, run, T, st, K, selname, variant):
    """E79 の diagnostics の中の chosen_row と同じ（analyze_e79.py 1721〜1723 行）。"""
    ch = sel[(sel.run == run) & (sel.target == T) & (sel.student == st) & (sel.K == K) & (sel.selector == selname) & (sel.variant == variant)]
    return None if ch.empty or ch.method.iloc[0] in (A.EMPTY, None) or pd.isna(ch.method.iloc[0]) else ch.iloc[0]


def diagnostics_e82(args, run, T, d77, d78, tab, reps, sel, issues, diag_rows):
    """E79 の診断のあとに，e80c の波（E82）から選んだ entmin・im・self の設定の「働かなかった」の材料を，E79 と同じ式
    （analyze_e79.py 1745〜1763 行：none とのテストの予測の食い違いの割合．判定と同じ種の平均と，種のある手法の種 0）で足す．
    比べる none は選んだ設定と同じ C′ の none（E82 の none_cp．設計の 2 節）．倍率 1 の行は E79 のまま E39 の none と比べる。"""
    n0 = len(diag_rows)
    ORIG["diagnostics"](args, run, T, d77, d78, tab, reps, sel, issues, diag_rows)
    t82 = tab.get(SLOT)
    d82 = A.E80C.get(run)
    if t82 is None or d82 is None:
        return
    t77 = tab["e77"]
    have = {(x["rep"], x["student"], x["K"], x["what"]) for x in diag_rows[n0:]}
    n_add = 0
    for r in reps:
        P82 = np.load(Path(d82) / f"{T}_r{r}_pred.npy", mmap_mode="r")
        a77, a82 = t77[t77.rep == r], t82[t82.rep == r]
        for st in a77.student.unique():
            for K in A.KS:
                n7 = a77[(a77.student == st) & (a77.method == "none") & (a77.K == K)].row_id
                if not len(n7):
                    continue
                for meth in E82_METHODS:
                    for v in ("excl", "ns_excl"):
                        ch = chosen_row(sel, run, T, st, K, meth, v)
                        if ch is None or ch.method == "none" or ch.wave != SLOT:
                            continue
                        m_ = mult_of(ch.wave, ch.method, ch.param)
                        nm = a82[(a82.student == st) & (a82.method == NONE_CP) & (a82.K == K) & (a82.pnorm == f"-C{m_:g}x")].row_id
                        if len(nm) != 1:
                            raise SystemExit(f"{run} {T} 反復 {r} {st} K={K}：倍率 {m_:g} の none の行が {len(nm)} 行（1 行のはず）")
                        pn_m = P82[int(nm.iloc[0])]                            # 同じ C′ の none の予測
                        x = a82[(a82.student == st) & (a82.method == ch.method) & (a82.K == K) & (a82.pnorm == ch.param) & (a82.meff == ch.meff)]
                        xs = x[x.seed == 0] if ch.method in A.SEEDED else x
                        if v == "ns_excl" and ch.method in A.SEEDED - A.NC_SEEDED:
                            x = xs
                        add = []
                        if len(x):
                            add.append((f"pred_disagree_vs_none|{meth}|{v}", float(np.mean([(P82[int(i)] != pn_m).mean() for i in x.row_id])),
                                        {"n_rows": int(len(x)), "none_param": f"-C{m_:g}x"}))
                        if ch.method in A.SEEDED and len(xs):
                            add.append((f"pred_disagree_vs_none_s0|{meth}|{v}", float((P82[int(xs.row_id.iloc[0])] != pn_m).mean()),
                                        {"none_param": f"-C{m_:g}x"}))
                        for what, val, kw in add:
                            if (r, st, K, what) in have:                     # E79 の診断が同じ材料を書いていたら止める（二重に数えない）
                                raise SystemExit(f"E79 の診断が {what} を既に書いている（{run} {T} 反復 {r} {st} K={K}）")
                            diag_rows.append({"run": run, "target": T, "rep": r, "student": st, "K": K, "what": what, "value": val, **kw, "source": "e82"})
                            n_add += 1
    issues.append({"check": "e82_inactive_material_added", "run": run, "target": T, "n": n_add})


def install(grid, e82):
    """版ごとに E79 の関数をそろえる（fixed は E79 のまま．E80C は空にして本物の段 c も読まない）。"""
    A.E80C.clear()
    (A.sel_defs, A.check_e80c_prov, A.diagnostics, A.E80C_METHODS, A.choose, A.read_rows, A.key_table,
     A.Store) = (ORIG[k] for k in ("sel_defs", "check_e80c_prov", "diagnostics", "E80C_METHODS", "choose", "read_rows", "key_table", "Store"))
    if grid == "reselect":
        A.E80C.update(e82)
        A.E80C_METHODS = E82_METHODS
        A.sel_defs = sel_defs_e82
        A.check_e80c_prov = check_e82_prov
        A.diagnostics = diagnostics_e82
        A.choose = choose_e82
        A.read_rows = read_rows_e82
        A.key_table = key_table_e82


def parse_e82(args):
    e = dict(x.split("=", 1) for x in (args.e82 or []))
    out = {}
    for r in args.runs:
        d = Path(e.get(r, ROOT / "results" / f"e82_{r}"))
        if not d.exists():
            if (args.stage == "compare" or args.grid == "reselect") and not args.allow_incomplete:
                raise SystemExit(f"E82 の {r} の出力 {d} が無い（--e82 {r}=場所）")
            continue
        out[r] = d
    return out


def precheck_e82(args, dirs, stage, issues):
    """reselect の版で，E82 の出力がターゲットごとにあるか（無いと E82 の候補が黙って抜け，固定の版と同じ選択になる）。開かずに場所だけを見る。"""
    for run in dirs:
        d = A.E80C.get(run)
        targets = [t for t in (args.targets or A.groups_for(run)) if t in A.groups_for(run)]
        for T in targets:
            for nm in ((f"{T}_pseudo.csv",) if stage == "select" else (f"{T}_pseudo.csv", f"{T}.csv", f"{T}_samples.csv")):
                if d is None or not (Path(d) / nm).exists():
                    A.fail_or_note(args, issues, check="e82_file_missing", run=run, target=T, file=nm)


# ---------------------------------------------------------------- delta の段の追加の照合

def check_ps_none(args, dirs, e82, out):
    """E82 の仮のターゲットの none（`<T>_ps_<c>_r<r>_none.npz`）が E77 と同じか（鍵ごとの対数の確率の差の最大．E79 の TOL_LOGP）。"""
    rows, issues = [], []
    for run, (d77, _) in dirs.items():
        d82 = e82.get(run)
        if d82 is None:
            continue
        for f in sorted(Path(d82).glob("*_ps_*_none.npz")):
            g = Path(d77) / f.name
            if not g.exists():
                A.fail_or_note(args, issues, check="e82_ps_none_without_e77", run=run, file=f.name)
                continue
            za, zb = np.load(f), np.load(g)
            same_te = bool(np.array_equal(za["te"], zb["te"]) and np.array_equal(za["y"], zb["y"]))
            kb = {str(k): i for i, k in enumerate(zb["keys"])}
            v, n, miss = 0.0, 0, 0
            for i, k in enumerate(za["keys"]):
                if str(k) not in kb:
                    miss += 1
                    continue
                v = max(v, float(np.abs(za[f"lp{i}"] - zb[f"lp{kb[str(k)]}"]).max()))
                n += 1
            ok = same_te and miss == 0 and v <= A.TOL_LOGP
            rows.append({"run": run, "file": f.name, "same_te_y": same_te, "n_keys": n, "n_unmatched": miss, "absdiff_max": v, "ok": ok})
            if not ok:
                A.fail_or_note(args, issues, check="e82_ps_none_mismatch", run=run, file=f.name, absdiff_max=v, n_unmatched=miss)
    pd.DataFrame(rows).to_csv(out / "e82_ps_none_check.csv", index=False)
    return {"n_files": len(rows), "n_bad": int(sum(not r["ok"] for r in rows)), "absdiff_max": max((r["absdiff_max"] for r in rows), default=None),
            "issues": issues}


# ---------------------------------------------------------------- compare の段

def adjust_nc_seed0(S_):
    """E79 の summarize の種の感度の書き直し（analyze_e79.py 2427〜2445 行と同じ規則）を，手法ごとの文の鍵にも当てる。"""
    if S_.empty or "subset" not in S_.columns:
        return S_
    for base, alt in (("main", "nc_seed0"), ("main+excl_mosei_afew", "nc_seed0+excl_mosei_afew")):
        for fam in P4_FAMS:
            for sub in ("all", "unseen_only"):
                m0 = (S_.tag == base) & (S_.family == fam) & (S_.subset == sub) & ~S_.rule.isin(["chance"])
                m1 = (S_.tag == alt) & (S_.family == fam) & (S_.subset == sub) & ~S_.rule.isin(["chance"])
                if not m0.any() or not m1.any():
                    continue
                r0, r1 = S_.loc[m0, "rule"].iloc[0], S_.loc[m1, "rule"].iloc[0]
                if r0 != r1:
                    i0 = S_.index[m0][0]
                    note = f"entmin・IM の種 0 だけの判定（{alt}）では {r1}：{S_.loc[m1, 'statement_key'].iloc[0]}"
                    if r0 == "(3)":
                        S_.loc[i0, "rule"] = "(4)"
                        S_.loc[i0, "statement_key"] = (("(5) 筋の 4 を「単モダリティの特徴の空間では」と限る．" if fam == "P4'" else "")
                                                       + "ラベル無しの手法が向きの損を減らした証拠は無かった（区間の幅と判定できない数を添える）")
                        note = "種の平均と種 0 で文の鍵が食い違うので (3) にせず (4) の文にする．" + note
                    S_.loc[i0, "notes"] = " / ".join(x for x in (str(S_.loc[i0, "notes"]) if pd.notna(S_.loc[i0, "notes"]) else "", note) if x)
    return S_


def statements_of(C, methods=None):
    """combos から筋 4 の文の鍵を作る（E79 の statements_p4 と種の感度の書き直し）．methods を渡すとその手法の組み合わせだけに当てる。"""
    Cm = C if methods is None else C[C.method.isin(methods)]
    rows = []
    for tag, g in Cm.groupby("tag"):
        rows += A.statements_p4(tag, g, Cm)
    S_ = pd.DataFrame(rows)
    if S_.empty:
        return S_
    S_ = adjust_nc_seed0(S_.reset_index(drop=True))
    return S_[S_.rule != "chance"]


def read_grid(d, judge="judge"):
    d = Path(d)
    need = {"chosen": d / "select" / "chosen.csv", "units": d / judge / "units.csv", "combos": d / judge / "combos.csv",
            "statements": d / judge / "statements.csv"}
    miss = [str(v) for v in need.values() if not v.exists()]
    if miss:
        raise SystemExit(f"{d} に {miss} が無い（その版の select と {judge} を先に流す）")
    dt = {"K": str, "param": str, "meff": str}
    G = {"chosen": pd.read_csv(need["chosen"], dtype=dt), "units": pd.read_csv(need["units"], dtype={"K": str}),
         "combos": pd.read_csv(need["combos"], dtype={"K": str}), "statements": pd.read_csv(need["statements"], dtype={"K": str})}
    od = d / judge / "oracle_dir.csv"
    G["oracle"] = pd.read_csv(od, dtype={"K": str}) if od.exists() else pd.DataFrame()
    return G, {k: A.sha256(v) for k, v in need.items()}


def p4_units(U):
    m = U.family.isin(P4_FAMS) & U.comparison.isin([f"none-{x}" for x in E82_METHODS]) & U.metric.isin(["dir", "dir_rev"])
    return U[m].assign(method=lambda d: d.comparison.str.split("-", n=1).str[1])


def tie_info(args, dirs, e82, chosen):
    """選んだ単位ごとに，同点の候補の倍率（E79 の選択と同じ規則で仮のターゲットの値から作り直す．値は書かない）と，作り直した選択が chosen.csv と同じか。"""
    out = []
    for run, (d77, _) in dirs.items():
        if run not in e82:
            continue
        P = A.load_pseudo(run, d77, None)
        P = P[P.K.isin(A.P4_KS) & (P.seed <= 0)]
        ch = chosen[(chosen.run == run) & chosen.selector.isin(E82_METHODS) & chosen.K.isin(A.P4_KS) & chosen.role.isin(["kind", "fused"])]
        kk = ["wave", "method", "pnorm", "meff"]
        for r in ch.itertuples(index=False):
            PK = P[(P.eval_target == r.target) & (P.student == r.student) & (P.K == r.K)]
            PV = PK[PK.pseudo != "IEMOCAP-script"] if r.variant.startswith("ns") else PK
            none = PV[(PV.method == "none") & (PV.wave == "e77")]
            n_units = none.groupby(["pseudo", "rep"]).ngroups
            C = PV[A.cand_mask(PV, A.sel_defs(r.role)[r.selector])]
            row = {"run": run, "target": r.target, "student": r.student, "K": r.K, "selector": r.selector, "variant": r.variant}
            if C.empty:
                out.append({**row, "n_tie_recomputed": 0, "tie_mults": "", "recomputed_matches_chosen": r.method == A.EMPTY})
                continue
            sc = C.groupby(kk).uar.agg(score="mean", n="size").reset_index().merge(
                C.drop_duplicates(kk + ["pseudo", "rep"]).groupby(kk).size().rename("nu").reset_index())
            sc = sc[(sc.nu == n_units) & (sc.n == n_units)]
            ns = float(none.uar.mean()) if len(none) else np.nan
            ex, inc, tie = A.choose(sc, ns)                                    # reselect の規則（choose_e82．同点は倍率の小さい方）
            ex0, inc0, _ = ORIG["choose"](sc, ns)                              # E79 の文字列の順だけで決めたら
            pick = ex if r.variant.endswith("excl") else inc
            pick0 = ex0 if r.variant.endswith("excl") else inc0
            best = sc.score.max()
            tm = sorted({mult_of(w, m, p) for w, m, p in zip(*(sc[sc.score == best][c] for c in ("wave", "method", "pnorm")))})
            if pick is None:
                same = r.method in ("none", A.EMPTY)
            else:
                same = (pick.wave, pick.method, pick.pnorm, pick.meff) == (r.wave, r.method, r.param, r.meff)
            mr = mult_of(r.wave, r.method, r.param)
            m0 = mult_of(pick0.wave, pick0.method, pick0.pnorm) if pick0 is not None else np.nan
            out.append({**row, "n_tie_recomputed": tie, "tie_mults": ";".join(f"{x:g}" for x in tm), "tie_multi_mult": len(tm) > 1,
                        "tie_with_lower_mult": bool(np.isfinite(mr) and any(x < mr for x in tm)),
                        "mult_string_order": m0, "tie_chose_lower_mult": bool(np.isfinite(mr) and np.isfinite(m0) and mr < m0),
                        "recomputed_matches_chosen": bool(same)})
    return pd.DataFrame(out)


STRENGTH = {"(3)": 0, "(4)": 1, "(2')": 2, "(2)": 3, "(1)": 4}                 # 筋 4 を狭める強さ（E79 の (1)，(2)，(2′)，(4) の順．(3) は言い換えない）
SCOPE_NOTE = ("限定を外せるのは entmin・IM・自己学習だけ．T3A・LP・DANN・MME は C′ を選び直していないので，その文には「E39 の罰則で出発点につないだとき」を残す"
              "（CORAL は C′ を格子で選んでいる）")
BOTH4_NOTE = "固定の版と選び直した版がともに (4)：書けるのは「約 1.5δ_d より大きな減りは無かった」までで，「罰則に依らない」とは書かない"
CATEGORY = {"same_a": "比べ (a) の鍵が固定の版と同じ：罰則の限定なしで書き，E82 を感度の表に載せる",
            "limit": "比べ (a) の鍵が変わり，比べ (b) も同じ向きに変わった：「E39 の罰則で出発点につないだとき」と限定を付け，選び直した版の鍵と判定を並べる",
            "k_label": "比べ (a) の鍵が変わったが比べ (b) の鍵は固定の版と同じ：変わった分は罰則を弱めて K 本のラベルで向きが動いた分．限定なしで書き，内訳を添える",
            "opposite": "比べ (a) と (b) の鍵が逆の向きに変わった：設計の 3 節の規則に無い場合（本体が判断する）",
            "missing": "鍵が無い（その族の組み合わせが無いか全部が欠け）"}
CAT_ORDER = ["limit", "opposite", "k_label", "same_a", "missing"]              # 手法と全体のまとめで先に立てる順


def direction(r_new, r_old):
    if not isinstance(r_new, str) or not isinstance(r_old, str):
        return np.nan
    return int(np.sign(STRENGTH.get(r_new, np.nan) - STRENGTH.get(r_old, np.nan)))


def category_of(rf, ra, rb):
    if not all(isinstance(x, str) for x in (rf, ra, rb)):
        return "missing"
    if ra == rf:
        return "same_a"
    da, db = direction(ra, rf), direction(rb, rf)
    if rb == rf:
        return "k_label"
    return "limit" if db == da else "opposite"


def stage_compare(args, dirs, e82, out):
    fsrc = args.fixed_from or Path(args.out) / "fixed"
    F, hf = read_grid(fsrc)
    Ra, hra = read_grid(Path(args.out) / "reselect")
    Rb, hrb = read_grid(Path(args.out) / "reselect", "judge_b")
    G = {"fixed": F, "a": Ra, "b": Rb}                                          # 固定の版では比べ (a) と (b) は同じ（全部が倍率 1）
    # 1．判定の数（3 手法 × 族 × K × 感度 × 版）と単位ごとの比べ
    parts = []
    for gname, g_ in G.items():
        x = p4_units(g_["units"])
        c = x.groupby(["family", "method", "K", "sensitivity", "metric"]).judgment.value_counts().unstack(fill_value=0)
        for j in ("pos", "same", "neg", "undet", "inactive", "missing"):
            if j not in c.columns:
                c[j] = 0
        c = c.rename(columns={j: f"n_{j}" for j in c.columns}).reset_index()
        c["n_units"] = c[[k for k in c.columns if k.startswith("n_")]].sum(axis=1)
        parts.append(c.assign(grid={"fixed": "fixed", "a": "reselect (a) none(E39)", "b": "reselect (b) none(同じ C′)"}[gname]))
    pd.concat(parts, ignore_index=True).to_csv(out / "judgments.csv", index=False)
    key = ["family", "method", "run", "student", "K", "corpus", "metric", "sensitivity", "variant"]
    U = {k_: p4_units(g_["units"])[key + ["judgment", "point"]] for k_, g_ in G.items()}
    UJ = U["fixed"].rename(columns={"judgment": "judgment_fixed", "point": "_pf"}).merge(
        U["a"].rename(columns={"judgment": "judgment_a", "point": "_pa"}), on=key, how="outer", indicator="merge_fixed_a").merge(
        U["b"].rename(columns={"judgment": "judgment_b", "point": "_pb"}), on=key, how="outer")
    UJ["changed_a"] = UJ.judgment_fixed != UJ.judgment_a
    UJ["changed_b"] = UJ.judgment_fixed != UJ.judgment_b
    same_ab = (UJ.judgment_a == UJ.judgment_b) & ((UJ._pa == UJ._pb) | (UJ._pa.isna() & UJ._pb.isna()))
    UJ["same_a_b"] = same_ab
    n_ab_k0 = int((~same_ab[UJ.K == "0"]).sum())                               # 検算：K=0 では倍率の none ＝ E39 の none なので (a) と (b) は同じ
    UJ.drop(columns=["_pf", "_pa", "_pb"]).to_csv(out / "judgment_units.csv", index=False)
    other = [m for m in A.GROUP_B if m not in E82_METHODS]                    # ほかの 5 手法の判定は 3 つで同じはず（検算）
    ko = ["family", "comparison", "run", "student", "K", "corpus", "metric", "sensitivity", "variant"]
    om = {k_: g_["units"][g_["units"].family.isin(P4_FAMS) & g_["units"].comparison.isin([f"none-{m}" for m in other])][ko + ["judgment", "point"]]
          for k_, g_ in G.items()}
    OJ = om["fixed"].merge(om["a"], on=ko, how="outer", suffixes=("_f", "_a")).merge(om["b"].rename(columns={"judgment": "judgment_b", "point": "point_b"}),
                                                                                     on=ko, how="outer")
    n_other_diff = int(((OJ.judgment_f != OJ.judgment_a) | (OJ.judgment_a != OJ.judgment_b)).sum())
    n_other_point_diff = int((((OJ.point_a != OJ.point_b) & ~(OJ.point_a.isna() & OJ.point_b.isna()))).sum())
    # 2．筋 4 の文の鍵：族（E79 の statements.csv）と手法ごと（E79 の規則を 1 手法の組み合わせに当てる）
    srows, longs = [], []
    for gname, g_ in G.items():
        fam = g_["statements"]
        fam = fam[fam.family.isin(P4_FAMS) & (fam.rule != "chance")].assign(scope="family")
        re_fam = statements_of(g_["combos"]).assign(scope="family")             # 検算：E79 の statements.csv を combos から作り直せるか
        chk = fam[["tag", "family", "subset", "rule"]].merge(re_fam[["tag", "family", "subset", "rule"]], on=["tag", "family", "subset"],
                                                             how="outer", suffixes=("", "_re"))
        srows.append({"grid": gname, "n_family_rows": len(fam), "n_recomputed_rows": len(re_fam), "n_rule_mismatch": int((chk.rule != chk.rule_re).sum())})
        per = [statements_of(g_["combos"], [m]).assign(scope=m) for m in E82_METHODS]
        longs.append(pd.concat([fam] + per, ignore_index=True)[["scope", "tag", "family", "subset", "rule", "statement_key", "combos"]].assign(grid=gname))
    recheck = pd.DataFrame(srows)
    recheck.to_csv(out / "statements_recheck.csv", index=False)
    L = pd.concat(longs, ignore_index=True)
    ks = ["scope", "tag", "family", "subset"]
    ST = None
    for gname in G:
        x = L[L.grid == gname].drop(columns="grid").rename(columns={c: f"{c}_{gname}" for c in ("rule", "statement_key", "combos")})
        ST = x if ST is None else ST.merge(x, on=ks, how="outer")
    for gname in ("a", "b"):
        ST[f"same_rule_{gname}"] = ST.rule_fixed == ST[f"rule_{gname}"]
        ST[f"same_combos_{gname}"] = ST.combos_fixed.fillna("") == ST[f"combos_{gname}"].fillna("")
        ST[f"dir_{gname}"] = [direction(r1, r0) for r1, r0 in zip(ST[f"rule_{gname}"], ST.rule_fixed)]
    ST["category"] = [category_of(rf, ra, rb) for rf, ra, rb in zip(ST.rule_fixed, ST.rule_a, ST.rule_b)]
    ST.to_csv(out / "statements.csv", index=False)
    # 3．論文の書き方の規則（設計の 3 節．判断は tag main・subset all の手法ごとの鍵．unseen_only と same_combos は記述）
    D = ST[(ST.tag == "main") & ST.scope.isin(E82_METHODS)]
    drows = []
    for m in E82_METHODS:
        for fam in P4_FAMS:
            g = D[(D.scope == m) & (D.family == fam)]
            ga, gu = g[g.subset == "all"], g[g.subset == "unseen_only"]
            if ga.empty:
                continue
            r = ga.iloc[0]
            drows.append({"method": m, "family": fam, "rule_fixed": r.rule_fixed, "rule_a": r.rule_a, "rule_b": r.rule_b, "dir_a": r.dir_a, "dir_b": r.dir_b,
                          "category": r.category, "category_text": CATEGORY[r.category],
                          "same_combos_a_desc": r.same_combos_a, "same_combos_b_desc": r.same_combos_b,
                          "unseen_only_rules_desc": (f"fixed {gu.rule_fixed.iloc[0]} / a {gu.rule_a.iloc[0]} / b {gu.rule_b.iloc[0]}" if len(gu) else ""),
                          "both4_note": BOTH4_NOTE if (r.rule_fixed == "(4)" and r.rule_a == "(4)") else ""})
    DD = pd.DataFrame(drows)
    cats = list(DD.category) if len(DD) else ["missing"]
    worst = next(c for c in CAT_ORDER if c in cats)
    per_m = {m: next((c for c in CAT_ORDER if c in set(DD[DD.method == m].category)), "missing") for m in E82_METHODS}
    fam_main = ST[(ST.tag == "main") & (ST.subset == "all") & (ST.scope == "family")]
    DD = pd.concat([DD, pd.DataFrame([{"method": f"*{m}*", "category": c, "category_text": CATEGORY[c]} for m, c in per_m.items()]),
                    pd.DataFrame([{"method": "*all3*", "category": worst, "category_text": CATEGORY[worst],
                                   "writing_rule": (CATEGORY["same_a"] if worst == "same_a" else
                                                    "手法ごとの category に従う（" + "，".join(f"{m}：{c}" for m, c in per_m.items()) + "）"),
                                   "scope_note": SCOPE_NOTE,
                                   "family_level_same_rule_a_desc": bool(len(fam_main) and fam_main.same_rule_a.all()),
                                   "family_level_same_rule_b_desc": bool(len(fam_main) and fam_main.same_rule_b.all()),
                                   "n_other_method_units_judgment_differ": n_other_diff, "n_units_a_b_differ_K0": n_ab_k0}])], ignore_index=True)
    # 4．倍率ごとに選ばれた単位の数（excl・ns_excl が判定に使う版．incl は記述）．過半の母数は手法が選ばれた単位（__none__ を除く）
    ch = Ra["chosen"]
    ch = ch[ch.selector.isin(E82_METHODS) & ch.K.isin(A.P4_KS) & ch.role.isin(["kind", "fused"])].copy()
    ch["family"] = np.where(ch.role == "kind", "P4", "P4'")
    ch["mult"] = [mult_of(w, m, p) for w, m, p in zip(ch.wave, ch.method, ch.param)]
    ch["wave"] = ch.wave.replace({SLOT: "e82"})
    fc = F["chosen"][["run", "target", "student", "K", "selector", "variant", "wave", "method", "param", "meff", "n_cand"]]
    ch = ch.merge(fc, on=["run", "target", "student", "K", "selector", "variant"], how="left", suffixes=("", "_fixed"))
    ch["same_as_fixed"] = (ch.method == ch.method_fixed) & (ch.param == ch.param_fixed) & (ch.meff == ch.meff_fixed) & (ch.wave == ch.wave_fixed)
    T_ = tie_info(args, dirs, e82, Ra["chosen"])
    if len(T_):
        ch = ch.merge(T_, on=["run", "target", "student", "K", "selector", "variant"], how="left")
    keep = [c for c in ("run", "target", "student", "role", "family", "K", "selector", "variant", "wave", "method", "param", "meff", "mult", "n_tie",
                        "n_tie_recomputed", "tie_mults", "tie_multi_mult", "tie_with_lower_mult", "mult_string_order", "tie_chose_lower_mult",
                        "recomputed_matches_chosen", "n_cand", "n_cand_fixed", "wave_fixed", "param_fixed", "meff_fixed", "same_as_fixed") if c in ch.columns]
    ch[keep].to_csv(out / "multiplier_units.csv", index=False)

    def count(g):
        n = len(g)
        m = g.mult
        nc = int(m.notna().sum())                                              # 母数：手法が選ばれた単位（__none__ を除く．関門 1 の define D1）
        n10 = int((m == 10).sum())
        flag = lambda c_: g[c_].fillna(False).astype(bool) if c_ in g else pd.Series(False, index=g.index)  # noqa: E731
        tl, tm_, tc = flag("tie_with_lower_mult"), flag("tie_multi_mult"), flag("tie_chose_lower_mult")
        n10s = int(((m == 10) & ~tl).sum())
        return pd.Series({"n_units": n, "n_chosen": nc, "n_mult1": int((m == 1).sum()), "n_mult3": int((m == 3).sum()), "n_mult10": n10,
                          "n_none_or_empty": n - nc, "frac_mult10": n10 / nc if nc else np.nan, "majority_mult10": bool(nc and n10 > nc / 2),
                          "n_tie_multi_mult": int(tm_.sum()), "n_tie_chose_lower_mult": int(tc.sum()),
                          "n_tie_with_lower_mult": int(tl.sum()), "n_mult10_strict": n10s, "majority_mult10_strict": bool(nc and n10s > nc / 2),
                          "n_same_as_fixed": int(g.same_as_fixed.sum())})
    MM = pd.concat([ch.groupby(["run", "student", "K", "selector", "variant"]).apply(count, include_groups=False).reset_index().assign(level="run×student×K×method"),
                    ch.groupby(["family", "selector", "K", "variant"]).apply(count, include_groups=False).reset_index().assign(level="family×K"),
                    ch.groupby(["selector", "variant"]).apply(count, include_groups=False).reset_index().assign(level="method"),
                    ch.groupby(["variant"]).apply(count, include_groups=False).reset_index().assign(level="all3")], ignore_index=True)
    MM["note"] = "単位（評価のターゲット）は仮のターゲットを共有するので，10 倍の選択の数は独立な選択の数ではない（関門 1 の stats S7）"
    MM.to_csv(out / "multipliers.csv", index=False)
    ex = MM[(MM.level == "method") & (MM.variant == "excl")].set_index("selector")
    grp = MM[(MM.level == "run×student×K×method") & (MM.variant == "excl")]
    a3 = MM[(MM.level == "all3") & (MM.variant == "excl")]
    for c_ in ("majority_mult10", "majority_mult10_strict"):
        DD[f"{c_}_excl"] = DD.method.map(lambda m, c_=c_: bool(ex[c_].get(m.strip("*"))) if m.strip("*") in ex.index else np.nan)
        DD.loc[DD.method == "*all3*", f"{c_}_excl"] = bool(a3[c_].iloc[0]) if len(a3) else np.nan
    DD["n_groups_majority_mult10_excl"] = DD.method.map(
        lambda m: f"{int(grp[grp.selector == m.strip('*')].majority_mult10.sum())}/{int((grp.selector == m.strip('*')).sum())}" if m.strip("*") in E82_METHODS else "")
    DD["grid_note"] = np.where(DD.majority_mult10_excl.astype(str) == "True", "過半が 10 倍：格子がまだ足りないことを限界に書く", "")
    DD.to_csv(out / "decision.csv", index=False)
    # 5．候補の数（固定の版と選び直した版．entmin 4 → 12，IM・自己学習 12 → 36 のはず）
    cand = []
    for gname, g_ in (("fixed", F), ("reselect", Ra)):
        c = g_["chosen"]
        c = c[c.selector.isin(list(A.GROUP_B)) & (c.variant == "excl") & c.K.isin(A.P4_KS) & c.role.isin(["kind", "fused"])].copy()
        c["family"] = np.where(c.role == "kind", "P4", "P4'")
        c["n_cand"] = pd.to_numeric(c.n_cand, errors="coerce")
        cand.append(c.groupby(["family", "selector", "K"]).n_cand.agg(n_cand_min="min", n_cand_median="median", n_cand_max="max").reset_index().assign(grid=gname))
    CC = pd.concat(cand, ignore_index=True)
    CC.to_csv(out / "candidates.csv", index=False)
    # 6．oracle_dir（格子の中で向きの損が最も小さい設定）：2 つの版を格子の大きさを添えて並べる（選び直した版は候補が 3 倍で上に偏る）
    ors = []
    for gname, g_ in (("fixed", F), ("reselect", Ra)):
        o = g_["oracle"]
        if o.empty:
            continue
        c = g_["chosen"]
        c = c[(c.variant == "excl")].assign(corpus=lambda d: d.target.map(A.corp), n_cand=lambda d: pd.to_numeric(d.n_cand, errors="coerce"))
        gs = c.groupby(["run", "corpus", "student", "K", "selector"]).n_cand.median().rename("grid_size").reset_index().rename(columns={"selector": "method"})
        ors.append(o.merge(gs, on=["run", "corpus", "student", "K", "method"], how="left").assign(grid=gname))
    if ors:
        O = pd.concat(ors, ignore_index=True)
        O["note"] = "格子のどの設定でも，はそれぞれの格子について書く．選び直した版は候補が 3 倍なので最大が上に偏る"
        O.to_csv(out / "oracle_dir_both.csv", index=False)
        O.assign(at=O.at_least_delta_d.astype(str) == "True").groupby(["grid", "method", "K"]).agg(
            n_units=("at", "size"), n_at_least_delta_d=("at", "sum"), grid_size_median=("grid_size", "median")).reset_index().to_csv(
            out / "oracle_dir_summary.csv", index=False)
    n_mis = int((~ch.recomputed_matches_chosen.astype(bool)).sum()) if "recomputed_matches_chosen" in ch else -1
    outs = [f.name for f in sorted(out.glob("*.csv"))]
    return {"reselect_a_inputs_sha256": hra, "reselect_b_inputs_sha256": hrb, "fixed_inputs_sha256": hf, "fixed_from": str(fsrc),
            "outputs_sha256": {nm: A.sha256(out / nm) for nm in outs}, "n_units_rows": len(UJ),
            "n_unmatched_units": int((UJ["merge_fixed_a"] != "both").sum()), "n_other_method_units_judgment_differ": n_other_diff,
            "n_other_method_units_point_differ_a_b": n_other_point_diff, "n_units_a_b_differ_K0": n_ab_k0,
            "statement_recheck_mismatch": int(recheck.n_rule_mismatch.sum()), "n_choice_recompute_mismatch": n_mis, "n_multiplier_units": len(ch)}


# ---------------------------------------------------------------- 入口

def write_e82_prov(d, args, e82, extra):
    prov = {"script": "exp/analyze_e82.py", "experiment": "E82", "stage": args.stage, "grid": getattr(args, "grid", None), "argv": sys.argv[1:],
            "commit": A.git("rev-parse", "HEAD"), "dirty_files": A.git("status", "--porcelain", "--untracked-files=no").splitlines(),
            "script_sha256": A.sha256(Path(__file__)), "script_tracked": bool(A.git("ls-files", "exp/analyze_e82.py")),
            "analyze_e79_sha256": A.sha256(Path(A.__file__)), "started": args._t0_str, "seconds": round(time.time() - args._t0, 1),
            "e82_dirs": {k: str(v) for k, v in e82.items()}, "slot": SLOT, "e82_methods": E82_METHODS, "e82_mults": E82_MULTS,
            "patched": ["E80C", "E80C_METHODS", "sel_defs", "check_e80c_prov", "diagnostics", "choose", "read_rows", "key_table"]
            + (["Store（比べ (b)）"] if args.stage == "judge_b" else []) if getattr(args, "grid", None) == "reselect" else [],
            "allow_incomplete": args.allow_incomplete, "skip_commit_check": args.skip_commit_check, "allow_dirty": args.allow_dirty, **extra}
    (Path(d) / "e82_provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--grid", choices=GRIDS, help="delta・select・measure・judge で必須：fixed（E79 のまま）か reselect（E82 の行を足す）")
    ap.add_argument("--runs", nargs="+", default=list(A.RUNS), choices=A.RUNS)
    ap.add_argument("--e77", nargs="+", help="名前=場所（既定 results/e77_<名前>）")
    ap.add_argument("--e78", nargs="+", help="名前=場所（既定 results/e78_<名前>）")
    ap.add_argument("--e82", nargs="+", help="名前=場所（既定 results/e82_<名前>）")
    ap.add_argument("--targets", nargs="+", help="試し用．評価のターゲットを限る")
    ap.add_argument("--reps", nargs="+", type=int, default=list(A.REPS))
    ap.add_argument("--pseudo-reps", nargs="+", type=int, default=list(A.PSEUDO_REPS))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e82_analysis")
    ap.add_argument("--e79-delta", type=Path, help="delta：E79 の delta.csv（sha256 が同じかを確かめる）")
    ap.add_argument("--fixed-from", type=Path, help="compare：固定の版の出力の場所（select/ と judge/ がある E79 の --out．既定 <out>/fixed）")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--no-bias-fixed", dest="bias_fixed", action="store_false", help="試し用．切片だけを直した UAR を計算しない")
    ap.add_argument("--allow-incomplete", action="store_true", help="試し用．欠けと食い違いを止めずに記録する")
    ap.add_argument("--skip-commit-check", action="store_true", help="試し用．δ・chosen.csv・fixed.csv のコミットの確認を飛ばす")
    ap.add_argument("--allow-dirty", action="store_true", help="試し用．exp/・measure/ に未コミットの変更があっても流す")
    args = ap.parse_args()
    if args.stage != "compare" and args.grid is None:
        ap.error("--grid を指定する（fixed か reselect）")
    args.families, args.e80c, args.e80_kept = "p4", None, ROOT / "results" / "e80" / "kept.csv"   # E79 の --families p4（kept.csv は読まない）
    args._t0, args._t0_str = time.time(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
    e82 = parse_e82(args)
    dirs = A.parse_dirs(args)
    install("reselect" if args.stage == "compare" else args.grid, e82)
    roots = [d for pair in dirs.values() for d in pair if d is not None] + list(A.E80C.values())
    A.check_dirty(args)
    base = Path(args.out)
    if args.stage == "compare":
        out = A.prepare_out(base, "compare")
        A.ALLOW["compare"] = A.ALLOW["select"]                                 # 仮のターゲットの行と provenance だけを開く（評価の出力と data/ は開かない）
        A.install_guard("compare", roots)
        A.guard_selftest("compare", [str(d) for d in roots])
        res = stage_compare(args, dirs, e82, out)
        write_e82_prov(out, args, e82, res)
        print(f"比べ：単位 {res['n_units_rows']} 行（相手の無い行 {res['n_unmatched_units']}），倍率の表の単位 {res['n_multiplier_units']}，"
              f"ほかの 5 手法で判定の違う単位 {res['n_other_method_units_judgment_differ']}（(a) と (b) で点の値が違う単位 {res['n_other_method_units_point_differ_a_b']}），"
              f"K=0 で (a) と (b) の違う単位 {res['n_units_a_b_differ_K0']}，文の鍵の作り直しの食い違い {res['statement_recheck_mismatch']}，"
              f"選択の作り直しの食い違い {res['n_choice_recompute_mismatch']}（{out}）", flush=True)
        return 0
    args.out = base / args.grid                                                # E79 の関数は args.out/<段> を読み書きする
    out = A.prepare_out(args.out, args.stage)
    A.install_guard(args.stage, roots)
    if args.stage in A.ALLOW:
        probe = A.guard_selftest(args.stage, [str(d) for d in roots])
        print(f"見張り：{args.stage} の段は入力の場所で {A.ALLOW[args.stage]} だけを開く（{Path(probe).name} を開けないことを確かめた）", flush=True)
    extra, issues = {}, []
    if args.grid == "reselect" and args.stage in ("select", "measure"):
        precheck_e82(args, dirs, args.stage, issues)
    if args.stage == "judge_b":                                                # 比べ (b)：none を選んだ手法と同じ C′ の none に替えて E79 の判定の段を流す
        if args.grid != "reselect":
            raise SystemExit("judge_b は --grid reselect だけ（固定の版では (b) は (a) と同じ）")
        A.Store = StoreB
    {"delta": A.stage_delta, "select": A.stage_select, "measure": A.stage_measure, "judge": A.stage_judge,
     "judge_b": A.stage_judge}[args.stage](args, dirs, out)
    if args.stage == "judge_b":
        extra["n_none_redirected"] = StoreB.last.n_redirect if StoreB.last is not None else None
        print(f"比べ (b)：none の鍵を同じ C′ の none に替えた回数 {extra['n_none_redirected']}", flush=True)
    if args.stage == "delta":
        h = A.sha256(out / "delta.csv")
        extra["delta_sha256"] = h
        if args.e79_delta:
            h79 = A.sha256(args.e79_delta)
            extra["e79_delta"] = {"path": str(args.e79_delta), "sha256": h79, "same": h79 == h}
            if h79 != h:
                A.fail_or_note(args, issues, check="delta_differs_from_e79", e79=h79[:16], e82=h[:16])
        other = base / ("fixed" if args.grid == "reselect" else "reselect") / "delta" / "delta.csv"
        if other.exists():
            extra["other_grid_delta_same"] = A.sha256(other) == h
        if args.grid == "reselect":
            r = check_ps_none(args, dirs, e82, out)
            issues += r.pop("issues")
            extra["e82_ps_none"] = r
            print(f"E82 の仮のターゲットの none：npz {r['n_files']} 本，食い違い {r['n_bad']}，対数の確率の差の最大 {r['absdiff_max']}", flush=True)
    extra["e82_issues"] = issues
    write_e82_prov(out, args, e82, extra)
    if issues:
        pd.DataFrame(issues).to_csv(out / "e82_checks.csv", index=False)
    print(f"E82 の記録：{len(issues)} 件（{out}）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
