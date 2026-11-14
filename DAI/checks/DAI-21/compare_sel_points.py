"""gate-compare（DAI-21，2026-11-14）の確かめ：選んだ設定の点だけを並べる（読むだけ．UAR の列は出さない）．
使い方：./.venv/bin/python DAI/checks/DAI-21/compare_sel_points.py
"""
import json
import pandas as pd

R = "/home/snics/MER-DA"
d = pd.read_csv(f"{R}/results/dai18_probe/summary/students_sel.csv")
q = d[(d.role == "fusion") & d.teacher.isin(["prod", "M_F6_mixsel"])][["c", "K", "student", "teacher", "rho", "a"]]
print(q.to_string(index=False))
for t in ("prod", "M_F6_mixsel"):
    s = q[q.teacher == t]
    print(t, "rho=4（格子の上端）", int((s.rho == 4).sum()), "/", len(s), " a=16（上端）", int((s.a == 16).sum()), "/", len(s))
p = q.pivot_table(index=["c", "K"], columns="teacher", values=["rho", "a"])
same = ((p[("rho", "prod")] == p[("rho", "M_F6_mixsel")]) & (p[("a", "prod")] == p[("a", "M_F6_mixsel")]))
print("実験 18 の規則で積と提案の疑似ラベルが同じ点を選んだ (c, K)：", int(same.sum()), "/", len(same))
fs = json.load(open(f"{R}/results/dai18_probe/frozen_sel.json"))["choice"]
f20 = json.load(open(f"{R}/DAI/frozen/dai20_frozen.json"))["choice"]
fz1 = json.load(open(f"{R}/results/dai1/select/frozen.json"))
for T in ("AFEW", "IEMOCAP", "MOSEI", "MELD"):
    for K in ("5", "10", "25"):
        j = fz1[T]["select"][f"AVT|{K}|joint|one"]
        print(T, K, "実験 1 の学び直しの点", (j["rho"], j["a"]), "実験 20 の提案の点", f20[T][K]["grid"], "実験 18 の提案の点", fs[f"{T}|{K}|M_F6_mixsel"])
