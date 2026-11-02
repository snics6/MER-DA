"""diag.py の出力（out/*.csv）を（世界 × 生徒 × K）ごとの表にまとめる．UAR は 100 倍した点で書く。

単位（セル）は（評価のターゲット T, 仮のターゲットのコーパス c）．平均とセルの間の標準偏差（SD）を出す．
出力：out/table1_levels.csv … out/table6_rank.csv，out/teacher_levels.csv，out/tables.md（報告に貼る表）

    ./.venv/bin/python "research_notes/改善案_2026-11-02/diagnostic/report.py"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
WORLD_ORDER = {"fuse-audio-video-text": 0, "fuse-audio-video": 1}
WNAME = {"fuse-audio-video-text": "AVT", "fuse-audio-video": "AV"}
SNAME = {"audio_wavlm": "音声", "video": "顔", "text_roberta": "言葉", "fuse-audio-video-text": "融合", "fuse-audio-video": "融合"}
ST_ORDER = {"audio_wavlm": 0, "video": 1, "text_roberta": 2, "fuse-audio-video-text": 3, "fuse-audio-video": 3}
KEY = ["world", "student", "K"]


def order(df):
    df = df.copy()
    df["_w"] = df.world.map(WORLD_ORDER)
    df["_s"] = df.student.map(ST_ORDER)
    df["_k"] = df.K.astype(int)
    return df.sort_values(["_w", "_s", "_k"]).drop(columns=["_w", "_s", "_k"])


def ms(x, scale=100.0, dig=1):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] * scale
    if not len(x):
        return "—"
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return f"{x.mean():.{dig}f} ({sd:.{dig}f})" if np.isfinite(sd) else f"{x.mean():.{dig}f}"


def frac(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return f"{(x > 0).mean():.2f}" if len(x) else "—"


def agg(df, fn_cols):
    rows = []
    for (w, st, K), g in df.groupby(KEY):
        r = {"world": w, "student": st, "K": K, "n_cells": len(g)}
        for name, fn in fn_cols.items():
            r[name] = fn(g)
        rows.append(r)
    return order(pd.DataFrame(rows))


def num(df, fn_cols):
    """数値の版（CSV 用）：平均と SD。"""
    rows = []
    for (w, st, K), g in df.groupby(KEY):
        r = {"world": w, "student": st, "K": K, "n_cells": len(g)}
        for name, fn in fn_cols.items():
            v = np.asarray(fn(g), float)
            v = v[np.isfinite(v)]
            r[name + "_mean"] = float(v.mean()) if len(v) else np.nan
            r[name + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            r[name + "_n"] = int(len(v))
        rows.append(r)
    return order(pd.DataFrame(rows))


def md(df, cols, header):
    lines = ["| 世界 | 生徒 | K | " + " | ".join(header) + " |", "|" + "---|" * (3 + len(header))]
    for _, r in df.iterrows():
        lines.append(f"| {WNAME[r.world]} | {SNAME[r.student]} | {r.K} | " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    L = pd.read_csv(OUT / "lopo_cells.csv", dtype={"K": str})
    S = pd.read_csv(OUT / "spread_cells.csv", dtype={"K": str})
    R = pd.read_csv(OUT / "rank_units.csv", dtype={"K": str})
    TC = pd.read_csv(OUT / "teacher_cells.csv", dtype={"K": str})
    F = pd.read_csv(OUT / "full_select.csv", dtype={"K": str})
    fam = {f: L[L.family == f].set_index(["world", "T", "corpus", "student", "K"]) for f in L.family.unique()}
    base = fam["oursx"].reset_index()
    J = base[["world", "T", "corpus", "student", "K", "none", "best_kept_teacher_uar", "n_kept_teachers"]].copy()
    for f in ("oursx", "oursx77", "kd", "kd77", "groupB"):
        x = fam[f].reindex(base.set_index(["world", "T", "corpus", "student", "K"]).index)
        for c in ("oracle_excl", "oracle_incl", "val_excl", "val_incl", "sel_incl", "oracle_teacher_uar", "sel_excl_teacher_uar",
                  "heldout_rank_pct", "rho_select_vs_heldout", "same_as_full", "n_cand",
                  "oracle_cf_excl", "oracle_cf_incl"):
            J[f"{f}__{c}"] = x[c].to_numpy()
    J.to_csv(OUT / "joined_cells.csv", index=False)

    # 1. 水準
    t1 = {"none": lambda g: g.none, "best_kept_teacher": lambda g: g.best_kept_teacher_uar,
          "oursx_oracle": lambda g: g["oursx__oracle_excl"], "oursx_lopo": lambda g: g["oursx__val_incl"],
          "kd_oracle": lambda g: g["kd__oracle_excl"], "kd_lopo": lambda g: g["kd__val_incl"],
          "groupB_oracle": lambda g: g["groupB__oracle_excl"], "groupB_lopo": lambda g: g["groupB__val_incl"],
          "oursx_oracle_cf": lambda g: g["oursx__oracle_cf_excl"], "kd_oracle_cf": lambda g: g["kd__oracle_cf_excl"],
          "groupB_oracle_cf": lambda g: g["groupB__oracle_cf_excl"]}
    num(J, t1).to_csv(OUT / "table1_levels.csv", index=False)
    T1 = agg(J, {k: (lambda fn: lambda g: ms(fn(g)))(v) for k, v in t1.items()})
    t1c = {k + "_minus_none": (lambda fn: lambda g: fn(g) - g.none)(v) for k, v in t1.items() if k != "none"}
    num(J, t1c).to_csv(OUT / "table1c_minus_none.csv", index=False)
    T1c = agg(J, {**{k: (lambda fn: lambda g: ms(fn(g)))(v) for k, v in t1c.items()},
                  **{k + "_pos": (lambda fn: lambda g: frac(fn(g)))(v) for k, v in t1c.items()}})
    # 教師ごとの水準
    tl = TC.groupby(["world", "K", "teacher"]).uar.agg(mean=lambda x: 100 * x.mean(), sd=lambda x: 100 * x.std(ddof=1), n="size").reset_index()
    tl.to_csv(OUT / "teacher_levels.csv", index=False)
    # 教師が kept に入った割合（生徒 × K の単位で）
    # 2. 選択の取りこぼし
    t2 = {}
    for f in ("oursx", "kd", "groupB", "oursx77", "kd77"):
        t2[f"{f}_gap_incl"] = (lambda f_: lambda g: g[f"{f_}__oracle_incl"] - g[f"{f_}__val_incl"])(f)
        t2[f"{f}_gap_excl"] = (lambda f_: lambda g: g[f"{f_}__oracle_excl"] - g[f"{f_}__val_excl"])(f)
        t2[f"{f}_gapcf_incl"] = (lambda f_: lambda g: g[f"{f_}__oracle_cf_incl"] - g[f"{f_}__val_incl"])(f)
    num(J, t2).to_csv(OUT / "table2_selection_gap.csv", index=False)
    T2 = agg(J, {**{k: (lambda fn: lambda g: ms(fn(g)))(v) for k, v in t2.items()},
                 "oursx_none_frac": lambda g: f"{(g['oursx__sel_incl'] == 'none|-|0').mean():.2f}",
                 "kd_none_frac": lambda g: f"{(g['kd__sel_incl'] == 'none|-|0').mean():.2f}",
                 "groupB_none_frac": lambda g: f"{(g['groupB__sel_incl'] == 'none|-|0').mean():.2f}"})
    # 3. 生徒と教師
    t3 = {"oracle_minus_teacher": lambda g: g["oursx__oracle_excl"] - g["oursx__oracle_teacher_uar"],
          "lopo_minus_teacher": lambda g: g["oursx__val_excl"] - g["oursx__sel_excl_teacher_uar"],
          "oracle_minus_bestkept": lambda g: g["oursx__oracle_excl"] - g.best_kept_teacher_uar,
          "lopo_incl_minus_bestkept": lambda g: g["oursx__val_incl"] - g.best_kept_teacher_uar,
          "teacher_of_lopo_minus_none": lambda g: g["oursx__sel_excl_teacher_uar"] - g.none,
          "bestkept_minus_none": lambda g: g.best_kept_teacher_uar - g.none}
    num(J, t3).to_csv(OUT / "table3_student_vs_teacher.csv", index=False)
    T3 = agg(J, {**{k: (lambda fn: lambda g: ms(fn(g)))(v) for k, v in t3.items()},
                 **{k + "_pos": (lambda fn: lambda g: frac(fn(g)))(v) for k, v in t3.items()}})
    # 4. oursx と KD
    t4 = {"oracle_diff": lambda g: g["oursx__oracle_excl"] - g["kd__oracle_excl"],
          "lopo_incl_diff": lambda g: g["oursx__val_incl"] - g["kd__val_incl"],
          "lopo_excl_diff": lambda g: g["oursx__val_excl"] - g["kd__val_excl"],
          "oracle_cf_diff": lambda g: g["oursx__oracle_cf_excl"] - g["kd__oracle_cf_excl"],
          "oracle_diff77": lambda g: g["oursx77__oracle_excl"] - g["kd77__oracle_excl"],
          "lopo_incl_diff77": lambda g: g["oursx77__val_incl"] - g["kd77__val_incl"]}
    num(J, t4).to_csv(OUT / "table4_oursx_vs_kd.csv", index=False)
    T4 = agg(J, {**{k: (lambda fn: lambda g: ms(fn(g)))(v) for k, v in t4.items()},
                 **{k + "_pos": (lambda fn: lambda g: frac(fn(g)))(v) for k, v in t4.items()}})
    # 5. 格子の中の広がり
    sp_cols = ["grid_range", "main_range_teacher", "main_range_wgt", "main_range_meff", "opt_range_teacher", "opt_range_wgt", "opt_range_meff",
               "bestof_range_teacher", "bestof_range_wgt", "bestof_range_meff"]
    sh_cols = ["share_teacher", "share_wgt", "share_meff", "share_interaction"]
    out5 = []
    for f in ("oursx", "oursx77"):
        Sf = S[S.family == f]
        t5 = {c: (lambda c_: lambda g: g[c_])(c) for c in sp_cols + sh_cols + ["n_teachers"]}
        n5 = num(Sf, t5).assign(family=f)
        out5.append(n5)
    pd.concat(out5).to_csv(OUT / "table5_grid_spread.csv", index=False)
    S1 = S[S.family == "oursx"]
    T5 = agg(S1, {**{c: (lambda c_: lambda g: ms(g[c_]))(c) for c in sp_cols},
                  **{c: (lambda c_: lambda g: f"{g[c_].mean():.2f}")(c) for c in sh_cols},
                  "n_teachers": lambda g: f"{g.n_teachers.mean():.1f}"})
    S2 = S[S.family == "oursx77"]
    T5b = agg(S2, {**{c: (lambda c_: lambda g: ms(g[c_]))(c) for c in sp_cols},
                   **{c: (lambda c_: lambda g: f"{g[c_].mean():.2f}")(c) for c in sh_cols}})
    # 最適点の（重みの付け方, 総重み, 教師）の分布
    opt = S1.groupby(["world", "K", "opt_wgt"]).size().unstack(fill_value=0)
    opt.to_csv(OUT / "table5_opt_wgt_counts.csv")
    optm = S1.groupby(["world", "K", "opt_meff"]).size().unstack(fill_value=0)
    optm.to_csv(OUT / "table5_opt_meff_counts.csv")
    optt = S1.groupby(["world", "K", "opt_teacher"]).size().unstack(fill_value=0)
    optt.to_csv(OUT / "table5_opt_teacher_counts.csv")
    # 6. 順位の一致
    r6 = []
    for f in ("oursx", "kd", "groupB", "oursx77"):
        Rf = R[R.family == f]
        a = Rf.groupby(KEY)[["rho_between_corpus_mean", "rho_between_single_rep", "rho_within_group_single_rep"]].mean()
        Lf = L[L.family == f]
        b = Lf.groupby(KEY).agg(rho_select_vs_heldout=("rho_select_vs_heldout", "mean"), heldout_rank_pct=("heldout_rank_pct", "mean"),
                                same_as_full=("same_as_full", "mean"), n_cand=("n_cand", "mean"))
        r6.append(a.join(b).reset_index().assign(family=f))
    R6 = pd.concat(r6)
    R6.to_csv(OUT / "table6_rank.csv", index=False)

    # 表 0：要点（1 行に主な数値．平均だけ）
    def mean_of(fn):
        return lambda g: f"{100 * np.nanmean(np.asarray(fn(g), float)):.1f}"
    T0 = agg(J, {"none": mean_of(lambda g: g.none),
                 "teach": mean_of(lambda g: g.best_kept_teacher_uar - g.none),
                 "ox_or": mean_of(lambda g: g["oursx__oracle_excl"] - g.none),
                 "ox_lo": mean_of(lambda g: g["oursx__val_incl"] - g.none),
                 "gap": mean_of(lambda g: g["oursx__oracle_incl"] - g["oursx__val_incl"]),
                 "gapcf": mean_of(lambda g: g["oursx__oracle_cf_incl"] - g["oursx__val_incl"]),
                 "kdgapcf": mean_of(lambda g: g["kd__oracle_cf_incl"] - g["kd__val_incl"]),
                 "bgapcf": mean_of(lambda g: g["groupB__oracle_cf_incl"] - g["groupB__val_incl"]),
                 "ox_t": mean_of(lambda g: g["oursx__oracle_excl"] - g.best_kept_teacher_uar),
                 "kdcf": mean_of(lambda g: g["oursx__oracle_cf_excl"] - g["kd__oracle_cf_excl"]),
                 "kdlo": mean_of(lambda g: g["oursx__val_incl"] - g["kd__val_incl"])})
    sh = S1.groupby(KEY)[sh_cols].mean()
    names = {"share_teacher": "教師", "share_wgt": "重み", "share_meff": "総重み", "share_interaction": "交互作用"}
    T0["main"] = [f"{names[sh.loc[(w, st, K)].idxmax()]} {sh.loc[(w, st, K)].max():.2f}" for w, st, K in zip(T0.world, T0.student, T0.K)]
    rr = R[R.family == "oursx"].groupby(KEY)[["rho_between_corpus_mean", "rho_within_group_single_rep", "rho_between_single_rep"]].mean()
    T0["rho"] = [f"{rr.loc[(w, st, K)].rho_between_corpus_mean:.2f} / {rr.loc[(w, st, K)].rho_between_single_rep:.2f} / "
                 f"{rr.loc[(w, st, K)].rho_within_group_single_rep:.2f}" for w, st, K in zip(T0.world, T0.student, T0.K)]
    T0.to_csv(OUT / "table0_summary.csv", index=False)

    # 表（markdown）
    parts = []
    parts.append("### 表 0．要点（UAR の点．セルの平均）\n")
    parts.append(md(T0, ["n_cells", "none", "teach", "ox_or", "ox_lo", "gap", "gapcf", "kdgapcf", "bgapcf", "ox_t", "kdcf", "kdlo", "main", "rho"],
                    ["セル", "none", "残した教師の最良 − none", "oursx 最良 − none", "oursx LOPO − none", "oursx 取りこぼし", "oursx 取りこぼし（交差）",
                     "KD 取りこぼし（交差）", "群 B 取りこぼし（交差）", "oursx 最良 − 残した教師の最良", "oursx − KD（交差オラクル）",
                     "oursx − KD（LOPO）", "oursx の格子で分散の最大の要因", "順位相関：コーパス間（5 反復）/ コーパス間（1 反復）/ 同じ群の反復間"]))
    parts.append("")
    parts.append("### 表 1．水準（UAR の点．平均（セルの間の SD））\n")
    parts.append(md(T1, ["n_cells", "none", "best_kept_teacher", "oursx_oracle", "oursx_oracle_cf", "oursx_lopo", "kd_oracle", "kd_oracle_cf", "kd_lopo",
                         "groupB_oracle", "groupB_oracle_cf", "groupB_lopo"],
                    ["セル", "none", "残した教師の最良", "oursx 格子の最良", "oursx 交差オラクル", "oursx LOPO", "KD 格子の最良", "KD 交差オラクル", "KD LOPO",
                     "群 B 最良", "群 B 交差オラクル", "群 B LOPO"]))
    parts.append("\n### 表 1c．none との差（セルごとの差の平均（SD），右は正の割合）\n")
    parts.append(md(T1c, ["best_kept_teacher_minus_none", "oursx_oracle_minus_none", "oursx_lopo_minus_none", "oursx_lopo_minus_none_pos",
                          "kd_oracle_minus_none", "kd_lopo_minus_none", "kd_lopo_minus_none_pos", "groupB_oracle_minus_none",
                          "groupB_lopo_minus_none", "groupB_lopo_minus_none_pos"],
                     ["残した教師の最良", "oursx 最良", "oursx LOPO", "正", "KD 最良", "KD LOPO", "正", "群 B 最良", "群 B LOPO", "正"]))
    parts.append("\n### 表 1b．教師そのものの UAR（世界 × K × 教師．平均（SD），セルの数）\n")
    tl2 = tl.copy()
    tl2["v"] = [f"{m:.1f} ({s:.1f}) n={n}" for m, s, n in zip(tl2["mean"], tl2["sd"], tl2["n"])]
    pv = tl2.pivot_table(index=["world", "K"], columns="teacher", values="v", aggfunc="first")
    pv = pv.reset_index()
    pv["_w"] = pv.world.map(WORLD_ORDER)
    pv["_k"] = pv.K.astype(int)
    pv = pv.sort_values(["_w", "_k"]).drop(columns=["_w", "_k"])
    tcols = [c for c in ["pF", "pFK", "poe0", "ref", "ref0", "agree", "ds", "mlp", "poolC", "other_audio_wavlm", "other_video", "other_text_roberta"]
             if c in pv.columns]
    lines = ["| 世界 | K | " + " | ".join(tcols) + " |", "|" + "---|" * (2 + len(tcols))]
    for _, r in pv.iterrows():
        lines.append(f"| {WNAME[r.world]} | {r.K} | " + " | ".join(str(r[c]) if isinstance(r[c], str) else "—" for c in tcols) + " |")
    parts.append("\n".join(lines))
    parts.append("\n### 表 2．選択の取りこぼし（格子のオラクル − LOPO で選んだ値．点．平均（SD））\n")
    parts.append(md(T2, ["oursx_gap_incl", "oursx_gapcf_incl", "oursx_gap_excl", "oursx_none_frac", "kd_gap_incl", "kd_gapcf_incl", "kd_none_frac",
                         "groupB_gap_incl", "groupB_gapcf_incl", "groupB_none_frac", "oursx77_gap_incl", "oursx77_gapcf_incl", "kd77_gap_incl"],
                    ["oursx（none 込み）", "oursx 交差オラクル", "oursx（none 抜き）", "oursx で none の割合", "KD（none 込み）", "KD 交差オラクル",
                     "KD で none の割合", "群 B（none 込み）", "群 B 交差オラクル", "群 B で none の割合", "oursx77（none 込み）", "oursx77 交差オラクル",
                     "KD77（none 込み）"]))
    parts.append("\n### 表 3．生徒と教師の差（点．平均（SD），右は正の割合）\n")
    parts.append(md(T3, ["oracle_minus_teacher", "oracle_minus_teacher_pos", "lopo_minus_teacher", "lopo_minus_teacher_pos",
                         "oracle_minus_bestkept", "oracle_minus_bestkept_pos", "lopo_incl_minus_bestkept", "lopo_incl_minus_bestkept_pos",
                         "teacher_of_lopo_minus_none", "bestkept_minus_none"],
                    ["格子の最良 − その教師", "正", "LOPO（none 抜き）− その教師", "正", "格子の最良 − 残した教師の最良", "正",
                     "LOPO（none 込み）− 残した教師の最良", "正", "LOPO の教師 − none", "残した教師の最良 − none"]))
    parts.append("\n### 表 4．oursx − KD（点．平均（SD），右は正の割合）\n")
    parts.append(md(T4, ["oracle_diff", "oracle_diff_pos", "oracle_cf_diff", "oracle_cf_diff_pos", "lopo_incl_diff", "lopo_incl_diff_pos", "lopo_excl_diff", "lopo_excl_diff_pos",
                         "oracle_diff77", "lopo_incl_diff77"],
                    ["格子の最良どうし", "正", "交差オラクルどうし", "正", "LOPO（none 込み）どうし", "正", "LOPO（none 抜き）どうし", "正", "最良どうし（E77 の教師）",
                     "LOPO どうし（E77 の教師）"]))
    parts.append("\n### 表 5．oursx の格子（残した教師）の中の広がり（点．平均（SD））と分散の割合\n")
    parts.append(md(T5, ["n_teachers", "grid_range", "main_range_teacher", "main_range_wgt", "main_range_meff", "opt_range_teacher", "opt_range_wgt",
                         "opt_range_meff", "bestof_range_teacher", "bestof_range_wgt", "bestof_range_meff",
                         "share_teacher", "share_wgt", "share_meff", "share_interaction"],
                    ["教師の数", "格子の幅", "主効果の幅：教師", "重み", "総重み", "最適点で動かす：教師", "重み", "総重み",
                     "残りを最良にした幅：教師", "重み", "総重み", "分散の割合：教師", "重み", "総重み", "交互作用"]))
    for nm, f_ in (("重みの付け方", "table5_opt_wgt_counts.csv"), ("総重み", "table5_opt_meff_counts.csv"), ("教師", "table5_opt_teacher_counts.csv")):
        cc = pd.read_csv(OUT / f_)
        cc["world"] = cc.world.map(WNAME)
        parts.append(f"\n格子の最良の点の{nm}（セル × 生徒の数）\n")
        parts.append("| " + " | ".join(cc.columns) + " |\n|" + "---|" * len(cc.columns) + "\n" +
                     "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in cc.itertuples(index=False)))
    parts.append("\n### 表 5b．同じ広がり（E77 の教師 3 つの格子）\n")
    parts.append(md(T5b, ["grid_range", "main_range_teacher", "main_range_wgt", "main_range_meff", "opt_range_teacher", "opt_range_wgt", "opt_range_meff",
                          "bestof_range_teacher", "bestof_range_wgt", "bestof_range_meff", "share_teacher", "share_wgt", "share_meff", "share_interaction"],
                    ["格子の幅", "主効果の幅：教師", "重み", "総重み", "最適点で動かす：教師", "重み", "総重み", "残りを最良にした幅：教師", "重み",
                     "総重み", "分散の割合：教師", "重み", "総重み", "交互作用"]))
    parts.append("\n### 表 6．順位の一致（Spearman の順位相関．平均）\n")
    R6m = order(R6)
    lines = ["| 世界 | 生徒 | K | 族 | 候補の数 | 別のコーパスどうし（5 反復の平均） | 別のコーパスどうし（1 反復） | 同じ群の別の反復（1 反復） |"
             " LOPO の選ぶ点数と抜いた値 | 抜いた所での LOPO の選択の順位（0＝最良，1＝最下位） | 全部で選んだものと同じ割合 |",
             "|" + "---|" * 11]
    for _, r in R6m.iterrows():
        lines.append(f"| {WNAME[r.world]} | {SNAME[r.student]} | {r.K} | {r.family} | {r.n_cand:.0f} | {r.rho_between_corpus_mean:.2f} | "
                     f"{r.rho_between_single_rep:.2f} | {r.rho_within_group_single_rep:.2f} | {r.rho_select_vs_heldout:.2f} | "
                     f"{r.heldout_rank_pct:.2f} | {r.same_as_full:.2f} |")
    parts.append("\n".join(lines))
    (OUT / "tables.md").write_text("\n".join(parts) + "\n")
    print("書いた：", OUT / "tables.md")


if __name__ == "__main__":
    main()
