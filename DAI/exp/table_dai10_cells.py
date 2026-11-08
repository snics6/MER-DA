"""DAI-10 の全部のセルの値と区間を，結果ファイルから Markdown の表に書き出す（設計 4 節の「全部のセルの値と区間を表に出す」）。

python DAI/exp/table_dai10_cells.py > "DAI/docs/実験10_下調べ_全セルの表_2026-11-08.md"
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_dai89_cells import table  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
D10 = ROOT / "results" / "dai10_probe"


def main():
    print("# 実験 10 の下調べ：全部のセルの値と区間（付録．2026-11-08）\n")
    print("表は `DAI/exp/table_dai10_cells.py` が `results/dai10_probe/diffs.csv`，`criteria.csv`，`diag_units.csv` から書き出した．値は UAR の差のポイントで，点推定 [95% 区間] である．\n")
    d = pd.read_csv(D10 / "diffs.csv")
    for s in ("AVT", "A", "V", "T"):
        print(f"## 空間 {s}\n")
        x = d[d["space"] == s]
        for cmp in sorted(x["compare"].unique()):
            for sub in sorted(x[x["compare"] == cmp]["subset"].unique()):
                print(table(x[(x["compare"] == cmp) & (x["subset"] == sub)], f"{cmp}（部分集合 {sub}）"))
    g = pd.read_csv(D10 / "diag_units.csv")
    e = g[g.key.str.endswith("_emo_eigs_a0.5")].copy()
    e["space"] = e.key.str.split("_").str[0]
    cols = ["eigs_1", "eigs_2", "eigs_3", "eigs_null_mean_1", "eigs_null_mean_2", "eigs_null_mean_3", "eigs_null_summax", "n_over_d"]
    m = e.groupby(["space", "c"])[cols].mean().reset_index()
    print("## 縮みの係数（A⁻¹B の固有値 λ．単位の平均．縮み 1 − 0.5λ は表を読むときに計算する）\n")
    print("| 空間 | c | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for _, r in m.iterrows():
        print(f"| {r.space} | {r.c} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    k = g[g.key.str.match(r"^K\d+_")].copy()
    k["K"] = k.key.str.split("_").str[0]
    k["space"] = k.key.str.split("_").str[1]
    cols2 = ["cos_ldaK_ldaO", "cos_ldaT_ldaO", "cos_emfromK_ldaO", "cos_emKmet_ldaO", "cos_emAmet_ldaO", "lam_max_sub", "emKmet_moved", "emAmet_moved"]
    m2 = k.groupby(["space", "c", "K"])[cols2].mean().reset_index()
    print("\n## 向きの近さ，引き算の版の λ_max，EM の動いた割合（単位の平均）\n")
    print("| 空間 | c | K | " + " | ".join(cols2) + " |")
    print("|---|---|---|" + "---|" * len(cols2))
    for _, r in m2.iterrows():
        print(f"| {r.space} | {r.c} | {r.K} | " + " | ".join(f"{r[c]:.3f}" for c in cols2) + " |")
    c9 = pd.read_csv(D10 / "criteria.csv")
    print("\n## criteria.csv の全部の列\n")
    print("| 列 | " + " | ".join(c9.c) + " |")
    print("|---|" + "---|" * len(c9))
    for col in [x for x in c9.columns if x != "c"]:
        print(f"| {col} | " + " | ".join("" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in c9[col]) + " |")


if __name__ == "__main__":
    main()
