"""chosen.csv・fixed.csv・candidates.csv の格子の数え上げ（読むだけ）。"""
import sys
import pandas as pd
d = sys.argv[1]
C = pd.read_csv(f"{d}/select/chosen.csv", dtype={"K": str, "param": str, "meff": str}, low_memory=False)
print("chosen", C.shape, list(C.columns))
k = ["run", "target", "student", "K", "selector", "variant"]
print("dup", int(C.duplicated(k).sum()))
print("selectors", C.selector.value_counts().to_dict())
print("variants", C.variant.value_counts().to_dict())
print("K", C.K.value_counts().to_dict())
print("role", C.groupby(["run", "role"]).student.unique().to_dict())
print("targets per run", C.groupby("run").target.unique().apply(sorted).to_dict())
emp = C[C.method == "__none__"]
print("empty", len(emp), emp.groupby(["selector", "K"]).size().to_dict())
# 期待：(run, target, student) × K{0,5,25} × selector × variant の全組
g = C.groupby(["run", "target", "student"]).size()
print("per (run,target,student) rows", g.value_counts().to_dict())
gg = C.groupby(["run", "role", "selector"]).apply(lambda x: (x.K.nunique(), sorted(x.variant.unique())), include_groups=False)
print(gg.to_string())
# ns 版は MOSEI・MELD だけ
print("ns targets", sorted(C[C.variant.str.startswith("ns")].target.unique()))
F = pd.read_csv(f"{d}/select/fixed.csv", dtype={"K": str}, low_memory=False)
print("fixed", F.shape, list(F.columns)[:20])
kf = [c for c in ("run", "target", "student", "K", "name", "fixed", "seed") if c in F.columns]
print("fixed key", kf, "dup", int(F.duplicated(kf).sum()))
col = "name" if "name" in F.columns else F.columns[4]
print(F.groupby([col]).size().to_dict())
print("nan per col", {c: int(v) for c, v in F.isna().sum().items() if v})
print("chosen nan per col", {c: int(v) for c, v in C.isna().sum().items() if v})
