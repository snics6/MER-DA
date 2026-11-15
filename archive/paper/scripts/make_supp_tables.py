#!/usr/bin/env python3
"""補足資料の表（supp/tables.tex）を台帳の key から作る．数値は \\NUM{key} で書く（手で写さない）．

使い方: python3 scripts/make_supp_tables.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
L = json.loads((ROOT / "results" / "numbers.json").read_text(encoding="utf-8"))
OUT = ROOT / "supp" / "tables.tex"

MOD_METHODS = [("e39", "Start"), ("self", "ST"), ("hard", "HPL"), ("entmin", "EM"), ("kd", "KD"), ("gd", "GD"),
               ("cotrain", "CoT"), ("coreg", "CoR"), ("saerens", "PA"), ("massbal", "RB"), ("ours", "Ours"), ("oursx", "Ours-ext")]
FUS_METHODS = [("e39", "Start"), ("self", "ST"), ("entmin", "EM"), ("kd", "KD"), ("cotrain-fused", "CoT-F"), ("cotrain-prod", "CoT-P"),
               ("coreg-sum", "CoR"), ("saerens", "PA"), ("massbal", "RB"), ("ours", "Ours"), ("oursx", "Ours-ext")]
MOD_ROWS = [("avt-a", "AVT", "A"), ("avt-v", "AVT", "V"), ("avt-t", "AVT", "T"),
            ("w2-a", "AVT (2nd)", "A"), ("w2-v", "AVT (2nd)", "V"), ("w2-t", "AVT (2nd)", "T")]
FUS_ROWS = [("avt-f", "AVT"), ("av-f", "AV"), ("vt-f", "VT"), ("at-f", "AT"), ("w2-f", "AVT (2nd)")]
STAT_ROWS = [("avt-a", "AVT", "A"), ("avt-v", "AVT", "V"), ("avt-t", "AVT", "T"), ("avt-f", "AVT", "Fused"),
             ("av-f", "AV", "Fused"), ("vt-f", "VT", "Fused"), ("at-f", "AT", "Fused"),
             ("w2-a", "AVT (2nd)", "A"), ("w2-v", "AVT (2nd)", "V"), ("w2-t", "AVT (2nd)", "T"), ("w2-f", "AVT (2nd)", "Fused")]


def num(key):
    if key not in L:
        raise KeyError(f"make_supp_tables: key '{key}' が台帳に無い")
    return f"\\NUM{{{key}}}"


def main():
    lines = []
    # 表 S2：モダリティの生徒
    cols = "ll" + "r" * len(MOD_METHODS)
    lines += [r"\begin{table*}[t]", r"\caption{UAR (\%) of the tuned competitors and of our methods for the single-modality students, averaged over targets and ten iterations. ST: self-training; HPL: hard pseudo-labels; EM: entropy minimization; KD: distillation with a temperature; GD: generalized distillation; CoT: co-training; CoR: co-regularization; PA: prior adjustment by EM; RB: output rebalancing.}",
              r"\label{tab:supp-mod}", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule",
              "Config. & Student, $K$ & " + " & ".join(n for _, n in MOD_METHODS) + r" \\", r"\midrule"]
    for ws, cfg, st in MOD_ROWS:
        for K in (0, 5):
            lines.append(f"{cfg if K == 0 else ''} & {st}, {K} & " + " & ".join(num(f"tab-{ws}-k{K}-{m}") for m, _ in MOD_METHODS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    # 表 S3：融合の生徒
    cols = "ll" + "r" * len(FUS_METHODS)
    lines += [r"\begin{table*}[t]", r"\caption{UAR (\%) of the tuned competitors and of our methods for the fused students. CoT-F and CoT-P: co-training with the fused or the product view; other abbreviations as in Table~\ref{tab:supp-mod}.}",
              r"\label{tab:supp-fused}", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule",
              "Config. & $K$ & " + " & ".join(n for _, n in FUS_METHODS) + r" \\", r"\midrule"]
    for ws, cfg in FUS_ROWS:
        for K in (0, 5):
            lines.append(f"{cfg if K == 0 else ''} & {K} & " + " & ".join(num(f"tab-{ws}-k{K}-{m}") for m, _ in FUS_METHODS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    # 表 S4：検定
    lines += [r"\begin{table}[t]", r"\caption{Ours-ext minus the strongest tuned competitor over target and iteration pairs: mean (points), effect size $d_z$, Holm-corrected $p$ of the Wilcoxon signed-rank test, and the number of targets with a positive mean.}",
              r"\label{tab:supp-stats}", r"\centering\footnotesize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
              r" & & \multicolumn{4}{c}{$K=0$} & \multicolumn{4}{c}{$K=5$} \\ \cmidrule(lr){3-6}\cmidrule(l){7-10}",
              r"Config. & Stud. & Mean & $d_z$ & $p$ & Pos. & Mean & $d_z$ & $p$ & Pos. \\", r"\midrule"]
    for ws, cfg, st in STAT_ROWS:
        cells = []
        for K in (0, 5):
            b = f"st-oursx-best-{ws}-k{K}"
            p = (r"$<$0.001" if L[f"{b}-pholmlt001"]["value"] == 1 else num(f"{b}-pholm"))
            cells += [num(f"{b}-mean"), num(f"{b}-dz"), p, num(f"{b}-tgtpos")]
        lines.append(f"{cfg} & {st} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("% 自動生成（scripts/make_supp_tables.py）．手で編集しない．\n" + "\n".join(lines), encoding="utf-8")
    print(f"make_supp_tables: {OUT.relative_to(ROOT)} を書いた")


if __name__ == "__main__":
    main()
