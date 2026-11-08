"""DAI-8・DAI-9 の全部のセルの値と区間を，結果ファイルから Markdown の表に書き出す（設計 4 節の「全部のセルの値と区間を表に出す」）。

python DAI/exp/table_dai89_cells.py > "DAI/docs/実験8・9_下調べ_全セルの表_2026-11-08.md"
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D8, D9 = ROOT / "results" / "dai8_probe", ROOT / "results" / "dai9_probe"


def cell(r):
    return f"{r.effect:.1f} [{r.lo:.1f}, {r.hi:.1f}]"


def table(d, title):
    ks = sorted(d.K.unique())
    out = [f"#### {title}", "", "| c | " + " | ".join(f"K={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for c in sorted(d.c.unique()):
        row = [c]
        for k in ks:
            x = d[(d.c == c) & (d.K == k)]
            row.append(cell(x.iloc[0]) if len(x) == 1 else "")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def main():
    print("# 実験 8・9 の下調べ：全部のセルの値と区間（付録．2026-11-08）\n")
    print("表は `DAI/exp/table_dai89_cells.py` が `results/dai8_probe/diffs.csv`，`diag.csv` と `results/dai9_probe/diffs.csv`，`criteria.csv` から書き出した．"
          "値は UAR の差のポイントで，点推定 [95% 区間] である．区間の作り方は結果の文の下書きの冒頭に書いた．\n")
    d8 = pd.read_csv(D8 / "diffs.csv")
    print("## DAI-8：比べ（空間 × 比べ × 部分集合）\n")
    for s in ("AVT", "A", "V", "T"):
        print(f"### 空間 {s}\n")
        x = d8[d8["space"] == s]
        for cmp in sorted(x["compare"].unique()):
            for sub in sorted(x[x["compare"] == cmp]["subset"].unique()):
                y = x[(x["compare"] == cmp) & (x["subset"] == sub)]
                print(table(y, f"{cmp}（部分集合 {sub}）"))
    dg = pd.read_csv(D8 / "diag.csv")
    print("## DAI-8：仮定の表と診断の記述（設計 3.3 節．単位の平均）\n")
    cols = ["n_iter_done", "hist_at_rule", "moved", "moved_at10", "ratio", "beta_flip_last", "entropy", "pi_err", "tr_sinv_s", "n_k_min",
            "pool_uar", "pool_uar_none", "pool_uar_at10", "agree_none_test", "keep_init"]
    for s in ("AVT", "A", "V", "T"):
        keys = [f"K{K}_{s}_em_a0.5_b1.0" for K in (0, 1, 5, 10, 25)]
        x = dg[dg.key.isin(keys)].copy()
        x["K"] = x.key.str.split("_").str[0]
        print(f"### 空間 {s}，固定の点の EM（α = 0.5，β = 1）\n")
        cs = [c for c in cols if c in x.columns]
        print("| c | K | " + " | ".join(cs) + " |")
        print("|---|---|" + "---|" * len(cs))
        for _, r in x.sort_values(["c", "key"]).iterrows():
            print(f"| {r.c} | {r.K} | " + " | ".join("" if pd.isna(r[c]) else f"{r[c]:.3f}" for c in cs) + " |")
        print()
    d9 = pd.read_csv(D9 / "diffs.csv")
    print("## DAI-9：比べ（τ × 比べ）\n")
    for cmp in sorted(d9["compare"].unique()):
        print(table(d9[d9["compare"] == cmp], cmp))
    c9 = pd.read_csv(D9 / "criteria.csv")
    print("## DAI-9：criteria.csv の全部の列\n")
    cs = [c for c in c9.columns if c != "c"]
    print("| 列 | " + " | ".join(c9.c) + " |")
    print("|---|" + "---|" * len(c9))
    for col in cs:
        print(f"| {col} | " + " | ".join("" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in c9[col]) + " |")


if __name__ == "__main__":
    main()
