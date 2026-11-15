"""DAI-11 の全部のセルの値と区間を，結果ファイルから Markdown の表に書き出す（設計 4.6 節の「どの規則も，満たしたかどうかによらず全部のセルの値と区間を表に出す」）。

python exp/table_dai11_cells.py > "docs/結果/実験11_下調べ_全セルの表_2026-11-09.md"

読むファイル：results/dai11_probe/diffs.csv，criteria.csv，req_cells.csv，rank_corr.csv，cells.csv，select/enc2_grid.csv．
書き換えはしない．
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D11 = ROOT / "results" / "dai11_probe"
STUDENTS = ("A", "V", "T", "AVT")
KS_ALL = (0, 1, 5, 10, 25)
PAIRS = (("enc2", "none"), ("self", "none"), ("other", "none"), ("fuse1", "none"), ("enc2fuse", "none"),
         ("enc2", "fuse1"), ("enc2fuse", "fuse1"), ("fuse1", "self"), ("enc2fuse", "enc2"), ("enc2", "other"), ("enc2fuse", "other"),
         ("enc2", "self"), ("enc2none", "none"), ("none", "noneloco"),
         ("enc2", "noneloco"), ("self", "noneloco"), ("other", "noneloco"), ("fuse1", "noneloco"), ("enc2fuse", "noneloco"))
SETS = ("none", "noneloco", "enc2none", "self", "enc2", "fuse1", "other", "enc2fuse")
REQ_COLS = ("n_over_d", "uar_s", "uar_t", "delta_uar", "delta_uar_bal", "agree", "comp2", "kappa", "phi", "uar_proj", "proj_gain", "r2", "lam_rel", "share_cell")
D8_COLS = ("d8_agree_none_projbal", "d8_agree_student_projbal", "d8_agree_none_projbal_t", "d8_agree_student_projbal_t",
           "d8_agree_student_teacherbal", "d8_agree_student_teacher", "d8_agree_none_proj", "d8_agree_student_proj", "uar_projbal", "uar_projbal_t", "uar_student_pool")


def fmt(v, nd=1):
    if pd.isna(v):
        return ""
    return f"{v:.{nd}f}"


def cell(r, pre=""):
    return f"{fmt(r[pre + 'est'], 2)} [{fmt(r[pre + 'lo'], 2)}, {fmt(r[pre + 'hi'], 2)}]"


def table(d, title, pre=""):
    ks = [k for k in KS_ALL if k in set(d.K)]
    out = [f"#### {title}", "", "| c | " + " | ".join(f"K={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for c in sorted(d.c.unique()):
        row = [c]
        for k in ks:
            x = d[(d.c == c) & (d.K == k)]
            row.append(cell(x.iloc[0], pre) if len(x) == 1 else "")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def wide(d, cols, title, nd=3, keys=("c", "K")):
    out = [f"#### {title}", "", "| " + " | ".join(keys) + " | " + " | ".join(cols) + " |", "|" + "---|" * (len(keys) + len(cols))]
    for _, r in d.sort_values(list(keys)).iterrows():
        out.append("| " + " | ".join(str(r[k]) for k in keys) + " | " + " | ".join(fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    print("# 実験 11 の下調べ：全部のセルの値と区間（付録．2026-11-09）\n")
    print("表は `exp/table_dai11_cells.py` が `results/dai11_probe/diffs.csv`，`criteria.csv`，`req_cells.csv`，`rank_corr.csv`，`cells.csv`，`select/enc2_grid.csv` から書き出した．"
          "差の表の値は「a − b」のテストの UAR の差のポイントで，点推定 [95% 区間] である．UAR の区間は `dai1_stats.effect_stats`（(f, r) を単位，3 つの T の数を平均），"
          "WAR・WAF・macro-F1 の区間は分割ごとの平均の 5 つの値の t の区間（自由度 4）である（設計 4.1 節）．K = 0 の K 本だけの版は種に依らず同じ判定になるので，K = 0 の独立な単位は分割の 5 個である．\n")
    d = pd.read_csv(D11 / "diffs.csv")
    print("## 1. 差と区間（UAR．学び直す分類器 × 比べ × 部分集合）\n")
    for s in STUDENTS:
        print(f"### 学び直す分類器 {s}\n")
        x = d[d["student"] == s]
        for a, b in PAIRS:
            y = x[(x.a == a) & (x.b == b)]
            if len(y) == 0:
                continue
            for sub in ["all"] + sorted(set(y.subset) - {"all"}):
                print(table(y[y.subset == sub], f"{a} − {b}（部分集合 {sub}）"))
    print("## 2. 差と区間（WAR・WAF・macro-F1．部分集合 all．学び直す分類器 × 比べ）\n")
    for s in STUDENTS:
        print(f"### 学び直す分類器 {s}\n")
        x = d[(d["student"] == s) & (d.subset == "all")]
        for a, b in PAIRS:
            y = x[(x.a == a) & (x.b == b)]
            if len(y) == 0:
                continue
            for met, name in (("war", "WAR"), ("waf", "WAF"), ("mf1", "macro-F1")):
                print(table(y, f"{a} − {b}（{name}）", pre=f"{met}_"))
    c9 = pd.read_csv(D11 / "criteria.csv")
    print("## 3. criteria.csv の全部の行（R1〜R4，SEL の数え上げ）\n")
    cols = [c for c in c9.columns if c not in ("c", "student", "rule")]
    print("| c | student | rule | " + " | ".join(cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for _, r in c9.iterrows():
        print(f"| {r.c} | {r.student} | {r.rule} | " + " | ".join("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r[cols]) + " |")
    print()
    rq = pd.read_csv(D11 / "req_cells.csv")
    print("## 4. 要件の量（候補のうち K 本を除いた発話の上．30 単位の平均．`req_cells.csv`）\n")
    print("Δ = `delta_uar` = 出所の UAR − none の UAR（ポイント）．`comp2`，`kappa`，`phi`，`agree` は割合．`share_cell` は Δ の平均が 0.5 未満のセルでは空である．\n")
    for s in STUDENTS:
        for t in ("self", "enc2", "fuse1", "other", "enc2fuse"):
            y = rq[(rq.student == s) & (rq.teacher == t)]
            if len(y) == 0:
                continue
            print(wide(y, [c for c in REQ_COLS if c in y.columns], f"学び直す分類器 {s}，出所 {t}"))
    print("## 5. D8 の一致率（学び直した版・none と，射影・出所の判定の一致．候補のうち K 本を除いた発話の上）\n")
    for s in STUDENTS:
        for t in ("enc2", "fuse1", "other", "enc2fuse"):
            y = rq[(rq.student == s) & (rq.teacher == t)]
            if len(y) == 0:
                continue
            print(wide(y, [c for c in D8_COLS if c in y.columns], f"学び直す分類器 {s}，出所 {t}"))
    rc = pd.read_csv(D11 / "rank_corr.csv")
    print("## 6. rank_corr.csv の全部の行（R3．中心を引いた順位相関，偶然の水準，部分相関，数え上げ，D8 の割合）\n")
    cols = [c for c in rc.columns]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for _, r in rc.iterrows():
        print("| " + " | ".join("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r[cols]) + " |")
    print()
    ce = pd.read_csv(D11 / "cells.csv")
    print("## 7. テストの指標の 30 単位の平均（UAR・WAR・WAF・macro-F1・感情ごとの再現率．`cells.csv`．部分集合 all）\n")
    print("感情は 0=Neutral，1=Happy，2=Sad，3=Angry．感情ごとの主張はしない（設計 4.4 節）．\n")
    ce = ce[~ce.key.str.contains(r"\|sub_")].copy()
    for s in STUDENTS:
        for c in sorted(ce.c.unique()):
            rows = []
            for K in KS_ALL:
                for st in SETS:
                    k = f"K{K}_{s}_{st}"
                    x = ce[(ce.c == c) & (ce.key == k)]
                    if len(x) == 1:
                        rows.append((K, st, x.iloc[0]))
            x = ce[(ce.c == c) & (ce.key == f"{s}_oracle")]
            if len(x) == 1:
                rows.append(("全ラベル", "oracle", x.iloc[0]))
            if not rows:
                continue
            print(f"#### 学び直す分類器 {s}，c = {c}\n")
            print("| K | 版 | UAR | WAR | WAF | macro-F1 | rec Neutral | rec Happy | rec Sad | rec Angry |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            for K, st, r in rows:
                print(f"| {K} | {st} | " + " | ".join(fmt(r[c_], 2) for c_ in ("uar", "war", "waf", "mf1", "rec0", "rec1", "rec2", "rec3")) + " |")
            print()
    g = pd.read_csv(D11 / "select" / "enc2_grid.csv")
    print("## 8. 第 2 エンコーダの K 本だけの版の引き戻しの選択（`select/enc2_grid.csv`．選ばれた点と，格子の最良との差）\n")
    sel = g[g.selected == True].copy()  # noqa: E712
    print("| T | kind | K | Cw | Cb | mean_uar（T の 30 単位のテストの平均） | grid_oracle_gap_per_TkindK |")
    print("|---|---|---|---|---|---|---|")
    for _, r in sel.sort_values(["T", "kind", "K"]).iterrows():
        print(f"| {r['T']} | {r['kind']} | {r['K']} | {r['Cw']} | {r['Cb']} | {fmt(r['mean_uar'], 2)} | {fmt(r['grid_oracle_gap_per_TkindK'], 2)} |")
    print()


if __name__ == "__main__":
    main()
