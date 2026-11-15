"""DAI-12 の全部のセルの値と区間を，結果ファイルから Markdown の表に書き出す（設計 4.5 節の「どの規則も，満たしたかどうかによらず全部のセルの値と区間を表に出す」）。

python DAI/exp/table_dai12_cells.py > "DAI/docs/実験12_下調べ_全セルの表_2026-11-09.md"

読むファイル：results/dai12_probe/diffs.csv，criteria.csv，cells_nl.csv，cells_lab.csv，q2_cells.csv，se_summary.csv，halves_units.csv．
書き換えはしない．丸めは結果ファイルの値から 1 回だけ行う（A25）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D12 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "dai12_probe"
MODS = ("A", "V", "T")
MOD_NAME = {"A": "音声 A", "V": "顔 V", "T": "言葉 T"}
KS_ALL = (0, 1, 5, 10, 25)
TEACHERS = ("self", "enc2", "fuse1", "other", "enc2fuse", "pair", "pair_shuf")
POOL_Q = (("nl_minus_lince", "非線形 nl − 線形 lince（主の量）"), ("nl_minus_lince_fixed", "固定の版の nl − lince"),
          ("nl_minus_lince0", "nl − 初期値 0 の線形 lince0（記述）"), ("nl_gain", "nl − none"), ("lince_gain", "lince − none"), ("lince0_gain", "lince0 − none"),
          ("ridge_gain", "リッジ − none"), ("delta_uar", "出所 − none（Δ）"),
          ("read_nl_minus_nl", "4×4 の読み出しの後の nl − nl"), ("read_nl_minus_read_lince", "読み出しの後の nl − 読み出しの後の lince"),
          ("shuf_nl", "nl(pair) − nl(pair_shuf)（ラベルを通らない分．記述）"), ("shuf_lince", "lince(pair) − lince(pair_shuf)（ラベルを通らない分．記述）"))
TEST_Q = (("nl_vs_lince", "テスト：nl − lince"), ("nlhid_vs_lince", "テスト：入れ子の前の隠れ層の版 − lince（記述）"), ("nl_vs_lince0", "テスト：nl − lince0"), ("nl_vs_none", "テスト：nl − none"), ("nl_vs_noneloco", "テスト：nl − none_loco"),
          ("nlfix_vs_lincefix", "テスト：固定の版 nl − lince"), ("lince_vs_none", "テスト：lince − none"), ("nl_vs_ret11", "テスト：nl − DAI-11 の線形の版（作り方の違いを含む）"))
CELL_COLS = ("n_over_d", "n_half0", "n_half1", "uar_s", "uar_t", "delta_uar", "agree", "uar_ridge", "r2_ridge", "lam_rel", "uar_lince", "uar_lince_fixed", "uar_lince0", "uar_nl", "uar_nl_fixed", "nl_minus_lince", "nl_minus_lince_fixed", "nl_minus_lince0", "lince_minus_ridge", "lince_below_ridge", "ce_ridge", "ce_lince", "ce_lince0", "ce_nl", "agree_lince_t", "agree_nl_t", "agree_nl_lince", "uar_read_nl", "uar_read_lince", "A_min_sv", "A_cond", "pair_comp_kappa", "share_ridge", "share_lince", "share_nl", "lince0_gain", "uar_nl_hidden", "ce_nl_hidden", "nl_used_hidden_h0", "nl_used_hidden_h1", "uar_nl_h0", "uar_nl_h1", "uar_lince_h0", "uar_lince_h1", "nl_conv", "lince_conv", "lince0_conv", "nl_edge_wd_low_h0", "nl_edge_wd_low_h1", "nl_edge_wd_high_h0", "nl_edge_wd_high_h1", "nl_edge_step_low_h0", "nl_edge_step_low_h1", "nl_edge_step_high_h0", "nl_edge_step_high_h1", "lince_edge_wd_low_h0", "lince_edge_wd_low_h1", "lince_edge_wd_high_h0", "lince_edge_wd_high_h1", "lince_edge_step_low_h0", "lince_edge_step_low_h1", "lince_edge_step_high_h0", "lince_edge_step_high_h1", "nl_hidden_edge_wd_low_h0", "nl_hidden_edge_wd_low_h1", "nl_hidden_edge_wd_high_h0", "nl_hidden_edge_wd_high_h1", "nl_hidden_edge_step_low_h0", "nl_hidden_edge_step_low_h1", "nl_hidden_edge_step_high_h0", "nl_hidden_edge_step_high_h1", "ce_nl_hidden_h0", "ce_nl_hidden_h1", "lince0_edge_wd_low_h0", "lince0_edge_wd_low_h1", "lince0_edge_wd_high_h0", "lince0_edge_wd_high_h1", "lince0_edge_step_low_h0", "lince0_edge_step_low_h1", "lince0_edge_step_high_h0", "lince0_edge_step_high_h1", "nl_conv_h0", "nl_conv_h1", "lince_conv_h0", "lince_conv_h1", "ce_nl_h0", "ce_nl_h1", "ce_lince_h0", "ce_lince_h1")
A_COLS = tuple(f"A_{y}_{k}" for y in range(4) for k in range(4))


def fmt(v, nd=2):
    if pd.isna(v):
        return ""
    if isinstance(v, (bool,)):
        return str(v)
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def cell(r):
    return f"{fmt(r['est'])} [{fmt(r['lo'])}, {fmt(r['hi'])}]"


def table(d, title):
    ks = [k for k in KS_ALL if k in set(d.K)]
    out = [f"#### {title}", "", "| c | " + " | ".join(f"K={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for c in sorted(d.c.unique()):
        row = [c]
        for k in ks:
            x = d[(d.c == c) & (d.K == k)]
            row.append(cell(x.iloc[0]) if len(x) == 1 else "")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def wide(d, cols, title, nd=3, keys=("c", "K")):
    cols = [c for c in cols if c in d.columns]
    out = [f"#### {title}", "", "| " + " | ".join(keys) + " | " + " | ".join(cols) + " |", "|" + "---|" * (len(keys) + len(cols))]
    for _, r in d.sort_values(list(keys)).iterrows():
        out.append("| " + " | ".join(str(r[k]) for k in keys) + " | " + " | ".join(fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    Dd = pd.read_csv(D12 / "diffs.csv")
    Cr = pd.read_csv(D12 / "criteria.csv")
    Rc = pd.read_csv(D12 / "cells_nl.csv")
    Lc = pd.read_csv(D12 / "cells_lab.csv")
    Q2 = pd.read_csv(D12 / "q2_cells.csv")
    Se = pd.read_csv(D12 / "se_summary.csv")
    Hv = pd.read_csv(D12 / "halves_units.csv")
    print("# 実験 12 の下調べ：全部のセルの値と区間（付録．2026-11-09）\n")
    print("表は `DAI/exp/table_dai12_cells.py` が `results/dai12_probe/diffs.csv`，`criteria.csv`，`cells_nl.csv`，`cells_lab.csv`，`q2_cells.csv`，`se_summary.csv`，`halves_units.csv` から書き出した．"
          "差の表の値は「a − b」の UAR の差のポイントで，点推定 [95% 区間] である．候補の上の差（「候補の上の差」の節と部分集合の節，Q3 の節）は候補のうち K 本を除いた発話の上の値で，区間は `effect_stats_pool`（分割をまたいで塊の重みを 1 組だけ引く引き直し．AFEW・IEMOCAP・MELD は塊の jackknife）．"
          "テストの差（「テストの差」の節）の区間は `dai1_stats.effect_stats`．どちらも (f, r) を単位にし 3 つの T の数を平均する．値は結果ファイルから 1 回だけ丸めた（小数 2 桁）．\n")
    sec = 1
    print(f"## {sec}. 候補の上の差（学び直す分類器 × 出所）\n"); sec += 1
    for q, title in POOL_Q:
        dq = Dd[(Dd["where"] == "pool") & (Dd["quantity"] == q) & (Dd["subset"] == "all")]
        if len(dq) == 0:
            continue
        print(f"### {sec - 1}.{POOL_Q.index((q, title)) + 1} {title}\n")
        for m in MODS:
            for t_ in TEACHERS:
                d = dq[(dq.student == m) & (dq.teacher == t_)]
                if len(d):
                    print(table(d, f"{MOD_NAME[m]} × 出所 `{t_}`"))
    print(f"## {sec}. 候補の上の差の部分集合（nl − lince，固定の版，nl − none）\n"); sec += 1
    for q in ("nl_minus_lince", "nl_minus_lince_fixed", "nl_gain"):
        for sub in sorted(set(Dd[(Dd["where"] == "pool") & (Dd["subset"] != "all")]["subset"])):
            dq = Dd[(Dd["where"] == "pool") & (Dd["quantity"] == q) & (Dd["subset"] == sub)]
            for m in MODS:
                for t_ in TEACHERS:
                    d = dq[(dq.student == m) & (dq.teacher == t_)]
                    if len(d):
                        print(table(d, f"{q}，部分集合 `{sub}`，{MOD_NAME[m]} × 出所 `{t_}`"))
    print(f"## {sec}. 候補の全ラベルの分類器（Q3）：非線形 − 線形\n"); sec += 1
    for q in ("lab_nl_minus_lince", "lab_nl_minus_lince_fixed"):
        dq = Dd[(Dd["where"] == "pool") & (Dd["quantity"] == q) & (Dd["subset"] == "all")]
        for m in MODS:
            d = dq[dq.student == m]
            if len(d):
                print(table(d, f"{q}，{MOD_NAME[m]}"))
    print(wide(Lc, [c for c in Lc.columns if c not in ("c", "student", "K")], "候補の全ラベルの分類器のセルの平均", nd=2, keys=("c", "student", "K")))
    print(f"## {sec}. テストの差（Q4．記述）\n"); sec += 1
    for q, title in TEST_Q:
        dq = Dd[(Dd["where"] == "test") & (Dd["quantity"] == q) & (Dd["subset"] == "all")]
        if len(dq) == 0:
            continue
        print(f"### {title}\n")
        for m in MODS:
            for t_ in TEACHERS:
                d = dq[(dq.student == m) & (dq.teacher == t_)]
                if len(d):
                    print(table(d, f"{MOD_NAME[m]} × 出所 `{t_}`"))
                    if "war_est" in d.columns:
                        print(wide(d, ["war_est", "war_lo", "war_hi", "waf_est", "waf_lo", "waf_hi", "mf1_est", "mf1_lo", "mf1_hi"], f"同じ比べの WAR・WAF・macro-F1（分割の t）", nd=2))
    for sub in sorted(set(Dd[(Dd["where"] == "test") & (Dd["subset"] != "all")]["subset"])):
        dq = Dd[(Dd["where"] == "test") & (Dd["subset"] == sub) & (Dd["quantity"].isin(("nl_vs_lince", "nl_vs_none")))]
        for q in ("nl_vs_lince", "nl_vs_none"):
            for m in MODS:
                for t_ in TEACHERS:
                    d = dq[(dq.student == m) & (dq.teacher == t_) & (dq.quantity == q)]
                    if len(d):
                        print(table(d, f"テストの部分集合 `{sub}`，{q}，{MOD_NAME[m]} × 出所 `{t_}`"))
    print(f"## {sec}. 数え上げ（criteria.csv の全行）\n"); sec += 1
    cols = [c for c in Cr.columns if c not in ("c", "student", "rule", "teacher", "quantity")]
    print(wide(Cr, cols, "criteria.csv", nd=2, keys=("c", "student", "rule", "teacher", "quantity")))
    print(f"## {sec}. セルの量（cells_nl.csv．30 単位の平均）\n"); sec += 1
    for m in MODS:
        for t_ in TEACHERS:
            d = Rc[(Rc.student == m) & (Rc.teacher == t_)]
            if len(d):
                print(wide(d, CELL_COLS, f"{MOD_NAME[m]} × 出所 `{t_}`", nd=3))
    print(f"## {sec}. 出所の感情ごとの平均の予測の行列 A（行 = 本当の感情，列 = 出所の確率．cells_nl.csv）\n"); sec += 1
    for m in MODS:
        for t_ in TEACHERS:
            d = Rc[(Rc.student == m) & (Rc.teacher == t_)]
            if len(d):
                print(wide(d, A_COLS, f"{MOD_NAME[m]} × 出所 `{t_}`", nd=3))
    print(f"## {sec}. Q2：DAI-11 で Δ ≥ 5 なのに得 < 0.5 だった 30 セルとの照合（q2_cells.csv）\n"); sec += 1
    print(wide(Q2, [c for c in Q2.columns if c not in ("c", "student", "teacher", "K")], "q2_cells.csv", nd=2, keys=("c", "student", "teacher", "K")))
    print(f"## {sec}. 標準誤差の分布（se_summary.csv）と半分の本数（halves_units.csv の要約）\n"); sec += 1
    print(wide(Se, [c for c in Se.columns if c not in ("c", "quantity")], "候補の上の差の標準誤差（セルごと）の中央値と 90% 点", nd=3, keys=("c", "quantity")))
    hs = Hv.groupby("c").agg(n_half0_min=("n_half0", "min"), n_half0_max=("n_half0", "max"), n_half1_min=("n_half1", "min"), n_half1_max=("n_half1", "max"),
                             **{f"h{h}_e{e}_min": (f"h{h}_e{e}", "min") for h in (0, 1) for e in range(4)}).reset_index()
    print(wide(hs, [c for c in hs.columns if c != "c"], "半分の本数（単位ごとの最小と最大）と感情ごとの本数の最小", nd=0, keys=("c",)))


if __name__ == "__main__":
    main()
