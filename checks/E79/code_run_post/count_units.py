"""E79 p4 / E82 の units.csv の数え上げ（読むだけ）。"""
import sys
import pandas as pd
f = sys.argv[1]
U = pd.read_csv(f, dtype={"K": str}, low_memory=False)
key = ["family", "comparison", "run", "student", "K", "corpus", "metric", "sensitivity", "variant"]
dup = U.duplicated(key, keep=False)
print("rows", len(U), "dup rows on key", int(dup.sum()))
# P4 主の族
for fam in ["P4", "P4'"]:
    x = U[(U.family == fam) & (U.sensitivity == "main") & (U.metric == "dir")]
    print(fam, "main dir units", len(x), "combos", x.groupby(["comparison", "student", "K"]).ngroups,
          "judgment", x.judgment.value_counts().to_dict())
    print("  comparisons", sorted(x.comparison.unique()))
    print("  per student/K corpora", x.groupby(["student", "K"]).corpus.nunique().to_dict())
    print("  per comparison n", x.groupby("comparison").size().to_dict())
    m = x[x.judgment == "missing"]
    print("  missing", m.groupby(["comparison", "K"]).size().to_dict())
# 感度ごとの P4 の行数
x = U[U.family.isin(["P4", "P4'"])]
print(x.groupby(["family", "sensitivity", "metric", "variant"]).size().to_string())
# NaN
for c in ["point", "lo", "hi", "delta"]:
    print(c, "nan", int(U[c].isna().sum()), "inf", int((U[c].abs() == float("inf")).sum()))
print("nan point by judgment", U[U.point.isna()].judgment.value_counts().to_dict())
print("nan lo by judgment", U[U.lo.isna()].judgment.value_counts().to_dict())
print("n_boot_valid", U.n_boot_valid.describe().to_dict())
