"""E79 p4 と E82 reselect の units.csv の鍵の差（読むだけ）。"""
import pandas as pd
key = ["family", "comparison", "run", "student", "K", "corpus", "metric", "sensitivity", "variant"]
A = pd.read_csv("results/e79_p4/judge/units.csv", dtype={"K": str}, low_memory=False, usecols=key + ["judgment"])
for nm in ("judge", "judge_b"):
    B = pd.read_csv(f"results/e82_analysis/reselect/{nm}/units.csv", dtype={"K": str}, low_memory=False, usecols=key + ["judgment"])
    print(nm, "rows", len(B), "dup", int(B.duplicated(key).sum()))
    M = A.merge(B, on=key, how="outer", indicator=True, suffixes=("_f", "_r"))
    print(M._merge.value_counts().to_dict())
    lo = M[M._merge == "left_only"]
    print(" left_only (E79 だけ) family/comparison:", lo.groupby(["family", "comparison"]).size().to_dict())
    ro = M[M._merge == "right_only"]
    print(" right_only (E82 だけ):", ro.groupby(["family", "comparison"]).size().to_dict())
