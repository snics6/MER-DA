"""DAI-13 の全部のセルの値と区間を，集計の出力から Markdown の表に書き出す（設計 4.5 節の「どの規則も，満たしたかどうかによらず全部のセルの値を付録に出す」のため）．

python exp/table_dai13_cells.py results/dai13_probe/summary_seeds2 > "docs/結果/実験13_下調べ_全セルの表_2026-11-10.md"

読むファイル（引数の集計のフォルダの下）：diffs.csv，criteria.csv，cells_lw.csv，cells_lab.csv，cells_layers.csv，alpha_summary.csv，v_init.csv，se_summary.csv，summary_meta.json．
書き換えはしない．丸めは結果ファイルの値から 1 回だけ行う（A25）．DAI-12 の `table_dai12_cells.py` と同じ形．
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "dai13_probe" / "summary_seeds2"
KS_ALL = (0, 1, 5, 10, 25)
TEACHERS = ("pair", "pair_shuf", "self", "labels")
POOL_Q = (("layers_minus_final", "層の重みつき和 lw − 最終層 final（主の量．入れ子の後）"),
          ("layers_minus_final_pre", "入れ子の前の lw − final（R1pre・R1c）"),
          ("layers_minus_final_fixed", "固定の点（wd 1e−4，500 ステップ）の lw − final"),
          ("lw_minus_finallw", "lw − final_lw（経路と束の分け方をそろえた比較）"),
          ("lwpre_minus_finallw", "入れ子の前の lw − final_lw"),
          ("finallw_minus_final", "final_lw − final（経路と束の分け方の違い）"),
          ("lw_minus_unif", "lw − unif（層の重みを学んだ効果）"),
          ("unif_minus_final", "unif − final（中間層を一様に足した効果）"),
          ("lw_gain", "lw − none（学び直しの得）"), ("final_gain", "final − none"),
          ("info_gap", "特徴に無い情報の量（出所の UAR − 線形に読んだ UAR）"), ("delta_uar", "出所 − none（Δ）"),
          ("shuf_lw", "lw(pair) − lw(pair_shuf)（ラベルを通らない分．記述）"), ("shuf_final", "final(pair) − final(pair_shuf)（同）"))
READ_Q = tuple((f"read_{kind}_wd{w}_minus_{kind}", f"Q4：4×4 の読み出しの後の {kind} − {kind}（wd の番号 {w}）") for kind in ("lw", "final") for w in range(3)) + \
         tuple((f"read_{kind}_wd{w}_minus_none", f"Q4：読み出しの後の {kind} − none（wd の番号 {w}）") for kind in ("lw", "final") for w in range(3))
KSHOT_Q = tuple((f"kshot_layers_minus_final_wd{w}", f"Q5：K 本で学んだ層の重みの分類器 kshot_lw − kshot_final（wd の番号 {w}）") for w in range(2)) + \
          tuple((f"lw_minus_kshot_lw_wd{w}", f"Q5：lw − kshot_lw（wd の番号 {w}）") for w in range(2)) + \
          tuple((f"kshot_lw_wd{w}_minus_none", f"Q5：kshot_lw − none（wd の番号 {w}）") for w in range(2))
LAB_Q = (("lab_layers_minus_final", "Q2：全ラベルで学んだ lw_lab − final_lab（入れ子の後）"), ("lab_layers_minus_final_pre", "Q2：入れ子の前"),
         ("lab_layers_minus_final_fixed", "Q2：固定の点"), ("lab_fixalpha_minus_final", "Q3：α を pair の解に固定した全ラベルの分類器 − final_lab"),
         ("lab_lw_minus_unif", "Q2：lw_lab − unif_lab"), ("lab_unif_minus_final", "Q2：unif_lab − final_lab"))
TEST_Q = (("lw_vs_final_test", "テスト：lw − final"), ("lw_vs_none_test", "テスト：lw − none"), ("final_vs_none_test", "テスト：final − none"),
          ("unif_vs_final_test", "テスト：unif − final"))
CELL_COLS = ("n_pool", "n_nk", "uar_s", "uar_t", "delta_uar", "uar_final", "uar_lw", "layers_minus_final", "uar_lw_pre", "layers_minus_final_pre", "uar_unif",
             "unif_minus_final", "lw_minus_unif", "uar_finallw", "finallw_minus_final", "lw_minus_finallw", "uar_final_fixed", "uar_lw_fixed", "layers_minus_final_fixed",
             "lw_gain", "final_gain", "info_gap", "ce_lw", "ce_final", "ce_unif", "ce_lw_pre", "agree_lw_final", "agree_finallw_final",
             "final_vs_dai12_lince_frac", "final_vs_dai12_lince_uar_absdiff", "lw_conv", "final_conv", "unif_conv", "finallw_conv",
             "lw_loss_above_final_h0", "lw_loss_above_final_h1", "lw_used_layers_h0", "lw_used_layers_h1", "alpha_max_h0", "alpha_max_h1", "alpha_argmax_h0", "alpha_argmax_h1",
             "alpha_final_h0", "alpha_final_h1", "alpha_entropy_h0", "alpha_entropy_h1", "psi_var_h0", "psi_var_h1", "n_units")
EDGE_COLS = tuple(f"{arm}_{ax}_h{h}" for arm in ("lw", "final", "unif", "finallw", "lw_pre") for ax in ("edge_wd_low", "edge_wd_high", "edge_step_low", "edge_step_high") for h in (0, 1)) + \
            tuple(f"{arm}_{q}_h{h}" for arm in ("lw", "final", "unif", "finallw") for q in ("wd", "step") for h in (0, 1))
READ_COLS = tuple(f"uar_read_{kind}_wd{w}" for kind in ("lw", "final") for w in range(3)) + tuple(f"uar_kshot_{kind}_wd{w}" for kind in ("lw", "final") for w in range(2)) + \
            tuple(f"kshot_{q}_wd{w}" for q in ("alpha_max", "alpha_argmax", "alpha_final", "alpha_entropy", "alpha_mean_layer") for w in range(2))


def fmt(v, nd=2):
    if pd.isna(v):
        return ""
    if isinstance(v, bool):
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


def layer_table(d, col, title):
    ks = [k for k in KS_ALL if k in set(d.K)]
    out = [f"#### {title}", "", "| c | 層 | " + " | ".join(f"K={k}" for k in ks) + " |", "|---|---|" + "---|" * len(ks)]
    for c in sorted(d.c.unique()):
        for l in sorted(d.layer.unique()):
            row = [c, str(l)]
            for k in ks:
                x = d[(d.c == c) & (d.layer == l) & (d.K == k)]
                row.append(fmt(x[col].iloc[0]) if len(x) == 1 else "")
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def pool_section(Dd, qs, sec, title):
    print(f"## {sec}. {title}\n")
    for i, (q, t) in enumerate(qs):
        dq = Dd[(Dd["where"] == "pool") & (Dd["quantity"] == q) & (Dd["subset"] == "all")]
        if len(dq) == 0:
            continue
        print(f"### {sec}.{i + 1} {t}（`{q}`）\n")
        for t_ in TEACHERS:
            d = dq[dq.teacher == t_]
            if len(d):
                print(table(d, f"出所 `{t_}`"))


def main():
    Dd = pd.read_csv(S / "diffs.csv")
    Cr = pd.read_csv(S / "criteria.csv")
    Rc = pd.read_csv(S / "cells_lw.csv")
    Lc = pd.read_csv(S / "cells_lab.csv")
    Ly = pd.read_csv(S / "cells_layers.csv")
    Al = pd.read_csv(S / "alpha_summary.csv")
    Vi = pd.read_csv(S / "v_init.csv")
    Se = pd.read_csv(S / "se_summary.csv")
    meta = json.loads((S / "summary_meta.json").read_text(encoding="utf-8"))
    n_seeds = meta.get("n_seeds", meta.get("seeds", ""))
    print("# 実験 13 の下調べ：全部のセルの値と区間（付録）\n")
    print(f"表は `exp/table_dai13_cells.py` が `{S.relative_to(ROOT) if S.is_relative_to(ROOT) else S}/` の `diffs.csv`，`criteria.csv`，`cells_lw.csv`，`cells_lab.csv`，`cells_layers.csv`，"
          f"`alpha_summary.csv`，`v_init.csv`，`se_summary.csv` から書き出した（種の数 {n_seeds}，行の台本の sha256 {str(meta.get('script_sha_rows', ''))[:8]}…，集計の台本 {str(meta.get('script_sha_summarize', ''))[:8]}…，単位 {meta.get('n_units', '')}）．"
          "差の表の値は「a − b」の候補の上の UAR の差のポイントで，点推定 [95% 区間] である．候補の上の差の区間は `effect_stats_pool`（候補のうち K 本を除いた発話を塊で数える引き直し）に，"
          "層の重みの量については v_init（初期値の揺れの見積もり）を足したもの（設計 4.1 節．`v_init_applied` の列が真の行）．テストの差（「テストの差」の節）の区間は `effect_stats`．"
          "値は結果ファイルから 1 回だけ丸めた．学び直す分類器は音声（A）だけである．\n")
    sec = 1
    pool_section(Dd, POOL_Q, sec, "候補の上の差（層の重みの腕 × 出所）"); sec += 1
    print(f"## {sec}. 候補の上の差の部分集合（主の量，入れ子の前，lw − none）\n"); sec += 1
    for q in ("layers_minus_final", "layers_minus_final_pre", "lw_gain"):
        for sub in sorted(set(Dd[(Dd["where"] == "pool") & (Dd["subset"] != "all")]["subset"])):
            dq = Dd[(Dd["where"] == "pool") & (Dd["quantity"] == q) & (Dd["subset"] == sub)]
            for t_ in TEACHERS:
                d = dq[dq.teacher == t_]
                if len(d):
                    print(table(d, f"`{q}`，部分集合 `{sub}`，出所 `{t_}`"))
    pool_section(Dd, LAB_Q, sec, "候補の全ラベルの分類器（Q2・Q3）"); sec += 1
    print(wide(Lc, [c for c in Lc.columns if c not in ("c", "student", "K")], "候補の全ラベルの分類器のセルの平均（cells_lab.csv）", nd=2, keys=("c", "K")))
    pool_section(Dd, READ_Q, sec, "Q4：K 本のラベルでの 4×4 の読み出し（記述）"); sec += 1
    pool_section(Dd, KSHOT_Q, sec, "Q5：層の重みを K 本のラベルで学ぶ分類器（記述）"); sec += 1
    print(f"## {sec}. テストの差（記述）\n"); sec += 1
    for q, t in TEST_Q:
        dq = Dd[(Dd["where"] == "test") & (Dd["quantity"] == q) & (Dd["subset"] == "all")]
        if len(dq) == 0:
            continue
        print(f"### {t}（`{q}`）\n")
        for t_ in TEACHERS:
            d = dq[dq.teacher == t_]
            if len(d):
                print(table(d, f"出所 `{t_}`"))
    print(f"## {sec}. 数え上げ（criteria.csv の全行）\n"); sec += 1
    cols = [c for c in Cr.columns if c not in ("c", "student", "rule", "teacher", "quantity")]
    print(wide(Cr, cols, "criteria.csv", nd=3, keys=("c", "rule", "teacher", "quantity")))
    print(f"## {sec}. セルの量（cells_lw.csv．単位の平均）\n"); sec += 1
    for t_ in TEACHERS:
        d = Rc[Rc.teacher == t_]
        if len(d):
            print(wide(d, CELL_COLS, f"出所 `{t_}`：量", nd=3))
            print(wide(d, EDGE_COLS, f"出所 `{t_}`：格子の端と選んだ点", nd=3))
            print(wide(d, READ_COLS, f"出所 `{t_}`：Q4・Q5 の値", nd=3))
    print(f"## {sec}. 層ごとの線形に読める分（cells_layers.csv．25 層 × K）\n"); sec += 1
    print(layer_table(Ly, "uar_layer_ridge_pair", "層ごとのリッジの UAR（出所 pair の目標）"))
    print(layer_table(Ly, "uar_layer_ridgelab", "層ごとのリッジの UAR（全ラベル）"))
    print(layer_table(Ly, "r2_layer_ridge_pair", "層ごとのリッジの R²（出所 pair の目標）"))
    print(f"## {sec}. 層の重み α の平均（alpha_summary.csv．w0〜w24 は softmax 後の重み，w_final は添字 24）\n"); sec += 1
    print(wide(Al, [f"w{i}" for i in range(25)] + ["argmax_layer", "w_final", "w_max"], "alpha_summary.csv", nd=3, keys=("c", "kind", "teacher", "K")))
    print(f"## {sec}. v_init（初期値の揺れの見積もり）と標準誤差の分布\n"); sec += 1
    print(wide(Vi, ["v_init", "n_folds", "d_init_mean"], "v_init.csv", nd=4, keys=("c", "teacher", "K")))
    print(wide(Se, [c for c in Se.columns if c not in ("c", "quantity")], "候補の上の差の標準誤差（セルごと）の中央値と範囲", nd=3, keys=("c", "quantity")))


if __name__ == "__main__":
    main()
