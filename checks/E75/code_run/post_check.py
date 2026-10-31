import json, glob, os
from pathlib import Path
import numpy as np, pandas as pd
R = Path("/home/snics/MER-DA/results/e75"); Q = Path("/home/snics/MER-DA/checks/E75/code_run/rerun_527a4ec")
p = json.loads((R / "provenance.json").read_text())
print({k: p.get(k) for k in ("commit", "dirty", "script_status", "script_sha256", "started", "ended", "argv", "n_gaps", "n_checks", "n_checks_failed")})
print("部品の状態", {k: v["status"] for k, v in p["modules"].items() if v["status"] != "clean"}, "部品の数", len(p["modules"]))
print("settings", {k: p["settings"][k] for k in ("kinds", "items", "n_perm", "n_boot", "n_boot_targets", "seed")})
print("code_snapshot", p.get("code_snapshot"))
w = {x["file"]: x["rows"] for x in p["written"]}
q = {x["file"]: x["rows"] for x in json.loads((Q / "provenance.json").read_text())["written"]}
print("書いたファイル", len(w), "；再検査の版と行の数が違うファイル", {f: (w.get(f), q.get(f)) for f in set(w) | set(q) if w.get(f) != q.get(f)})
on_disk = sorted(os.path.basename(f) for f in glob.glob(str(R / "*.csv")))
print("ディスクの CSV", len(on_disk), "provenance に無い", sorted(set(on_disk) - set(w)), "provenance にあってディスクに無い", sorted(set(w) - set(on_disk)))
C = pd.read_csv(R / "checks.csv", comment="#"); print("検算", len(C), "外れ", int((~C.ok).sum()))
newc = C[C.what.str.contains("C が違う組の数|の和|学び直しの違いの分")]
print(newc[["item", "kind", "what", "value", "ok", "detail"]].to_string())
G = pd.read_csv(R / "gaps.csv", comment="#"); Gq = pd.read_csv(Q / "gaps.csv", comment="#")
print("gaps", len(G), "再検査の版と同じか", G[["item", "kind", "what", "expected", "found"]].equals(Gq[["item", "kind", "what", "expected", "found"]]))
print(pd.read_csv(R / "records.csv", comment="#").to_string())
# 重複：主な出力の識別子
keys = {"e75_1_pairs.csv": ["kind", "target", "source", "K"], "e75_4_pairs.csv": ["rule", "kind", "target", "source", "K"],
        "e75_5_pairs.csv": ["kind", "target", "source", "K"], "e75_6_pairs.csv": ["kind", "target", "source"],
        "e75_7_rows.csv": ["kind", "clf", "target", "source", "K", "rep", "pair"], "e75_8_pairs.csv": ["kind", "target", "source"],
        "e75_9_pairs.csv": ["kind", "target", "source"], "e75_10_pairs.csv": ["kind", "target", "source", "K"],
        "e75_11_targets.csv": ["kind", "target"], "e75_11_boot.csv": ["kind", "target", "b"], "e75_13_pairs.csv": ["kind", "target", "source"],
        "e75_12_within_target.csv": ["kind", "K", "target"], "e75_8_shrinkage.csv": ["kind", "group", "half"]}
for f, k in keys.items():
    d = pd.read_csv(R / f, comment="#", dtype={"K": str, "source": str})
    print(f"重複 {f}: {int(d.duplicated(k).sum())}／{len(d)}")
# 格子：種類 × 組 × K
P1 = pd.read_csv(R / "e75_1_pairs.csv", comment="#", dtype={"K": str})
print("項目 1 の組 × K", P1.groupby("kind").size().to_dict(), "反復の数", sorted(P1.n_rep.unique()))
P4 = pd.read_csv(R / "e75_4_pairs.csv", comment="#", dtype={"K": str})
print("項目 4 の規則", P4.groupby("kind").rule.nunique().to_dict())
# 値の範囲：UAR 系と割合
P6 = pd.read_csv(R / "e75_6_pairs.csv", comment="#")
u = P6[[c for c in P6 if "uar" in c]]
print("項目 6 の UAR 系の範囲", float(u.min().min()), float(u.max().max()), "割合の範囲", P6.frac_recovered_joint_model.min(), P6.frac_recovered_joint_model.max(), "success", bool(P6.joint_success.all()))
print("項目 1 の得の範囲（UAR の差）", float(P1["mean"].min()), float(P1["mean"].max()))
# 再検査の版との数値の一致（項目 6・10 以外は同じ台本の部分なので一致するはず）
diffs = {}
for f in sorted(set(w) & set(q)):
    if not f.startswith("e75_"):
        continue
    a = pd.read_csv(R / f, comment="#"); b = pd.read_csv(Q / f, comment="#")
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        diffs[f] = "列か行が違う"; continue
    na, nb = a.select_dtypes("number"), b.select_dtypes("number")
    m = float(np.nanmax(np.abs(na.to_numpy(float) - nb.to_numpy(float)))) if na.size else 0.0
    nanpat = bool((na.isna().to_numpy() != nb.isna().to_numpy()).any())
    if m > 1e-9 or nanpat:
        diffs[f] = (m, nanpat)
print("再検査の版と数値が違うファイル", diffs)
