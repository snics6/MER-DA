"""DAI-3 validate の合否（設計 1.4 節）を check1_units.csv・check2_units.csv から作り直し，summary.json と frozen.json と照合する。"""
import json, hashlib, numpy as np, pandas as pd
from pathlib import Path
R = Path("/home/snics/MER-DA/results/dai3")
d1 = pd.read_csv(R / "validate/check1_units.csv"); d2 = pd.read_csv(R / "validate/check2_units.csv"); cv = pd.read_csv(R / "validate/convergence.csv")
summ = json.load(open(R / "validate/summary.json")); fz = json.load(open(R / "frozen.json"))
print("frozen validity == summary.json:", fz["validity"] == summ, "| validate_sha256 ok:", fz["_meta"]["validate_sha256"] == hashlib.sha256(open(R / "validate/summary.json", "rb").read()).hexdigest())
# 格子：単位 120 × 生徒 4 × W の種類 6 × 版 7
g = d1.groupby(["T", "c", "fold", "seed", "student", "wtype", "version"]).size()
print("check1 rows", len(d1), "dup keys", int((g > 1).sum()), "units", d1.groupby(["T", "c", "fold", "seed"]).ngroups, "wtypes", sorted(d1.wtype.unique()))
print("check1 invalid (version not computable) rows:", int((~d1.valid).sum()), "by c/version:", d1[~d1.valid].groupby(["c", "version"]).size().to_dict())
v = d1[d1.valid]
print("check1 nonfinite diff:", int((~np.isfinite(v["diff"])).sum()), "negative diff (opt worse than b*):", int((v["diff"] < -1e-9).sum()))
bad = []
v = v.assign(ver=np.where(v.version == "four", "four", "pair"))
for (c, s, wt, ver), g in v.groupby(["c", "student", "wtype", "ver"]):
    k = f"{c}|{s}|{wt}|{ver}"; med = g["diff"].median(); ref = summ["check1"][k]
    if not np.isclose(med, ref["median_diff"]) or (med <= 1.0) != ref["passed"] or len(g) != ref["n"]: bad.append(k)
print("check1 keys", len(summ["check1"]), "rebuilt mismatches", bad, "passed", sum(x["passed"] for x in summ["check1"].values()))
print("  failed:", {k: round(x["median_diff"], 3) for k, x in summ["check1"].items() if not x["passed"]})
# 確かめ 2
g2 = d2.groupby(["T", "c", "fold", "seed", "student", "angle"]).size()
print("check2 rows", len(d2), "dup", int((g2 > 1).sum()), "valid False", int((~d2.valid).sum()), "nonfinite", int((~np.isfinite(d2[["uar_drop", "dir_increase"]])).sum().sum()))
bad2 = []
for (c, s), g in d2[d2.valid].groupby(["c", "student"]):
    use = g[g.uar_drop >= 1.0]; ratio = use.dir_increase.sum() / use.uar_drop.sum() if len(use) else np.nan
    ver = "判定できない" if len(use) < 10 else ("合格" if ratio >= 0.7 else "不合格")
    ref = summ["check2"][f"{c}|{s}"]
    if ver != ref["verdict"] or len(use) != ref["n_usable"] or not np.isclose(ratio, ref["ratio"]): bad2.append((c, s))
print("check2 rebuilt mismatches", bad2, "verdicts", {k: (x["verdict"], x["n_usable"], x["n_all"]) for k, x in summ["check2"].items()})
print("validate convergence: rows", len(cv), "n_over_tol", int(cv.n_over_tol.sum()), "stalled", int(cv.stalled.sum()), "finite False", int((~cv.finite.astype(bool)).sum()), "grad_max max", cv.grad_max.max())
print("check1 grad_max max:", d1.grad_max.max())
