"""DAI-3 の detect（設計 1.1 節）：frozen.json の detect の各セルを se_rows.csv と実験 1 の固定のファイルから作り直す．
効果の見込み（detect_expected.json）は値を画面に出さず，記述に回すかの真偽との一致だけを数える。"""
import json, hashlib, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as st
ROOT = Path("/home/snics/MER-DA"); R = ROOT / "results/dai3"
fz = json.load(open(R / "frozen.json")); fz1 = json.load(open(ROOT / "results/dai1/select/frozen.json"))
ex = json.load(open(R / "detect_expected.json"))
se = pd.read_csv(R / "detect/se_rows.csv")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = fz["_meta"]
print("meta:", {k: m[k] for k in ("commit", "dirty", "targets", "encoder")}, "partial" in str(m))
print("dai1_frozen_sha ok:", m["dai1_frozen_sha256"] == sha(ROOT / "results/dai1/select/frozen.json"), "| expected sha ok:", m["detect_expected_sha256"] == sha(R / "detect_expected.json"))
print("script sha at d199583 ok:", {n: m["script_sha256"][n] == sha(ROOT / "DAI/exp" / n) for n in m["script_sha256"]})
print("se_rows: dup", int(se.duplicated(["X", "cell", "T"]).sum()), "cells per X", se.groupby("X").cell.nunique().to_dict(), "T per cell", se.groupby(["X", "cell"]).size().unique().tolist(),
      "nonfinite v_jk", int((~np.isfinite(se.v_jk)).sum()), "nonfinite v_seed_all", int((~np.isfinite(se.v_seed_all)).sum()))
geff = {}
for c in ["AFEW", "IEMOCAP", "MOSEI", "MELD"]:
    n = pd.read_csv(ROOT / f"DAI/splits/{c}.csv", keep_default_na=False).cluster.value_counts().to_numpy(); geff[c] = n.sum() ** 2 / (n ** 2).sum()
def crit(X, kind, mm):
    q = 1 - 0.05 / mm / 2
    if geff[X] >= 20: return st.t.ppf(q, 49) if kind == "refit" else st.norm.ppf(q)
    return st.t.ppf(q, max(1, int(np.floor(geff[X])) - 1))
bad = []; nd = {}; ns = {}
for X, D in fz["detect"].items():
    Rr = fz1[X]["n_seeds_eval"]; delta = fz1[X]["detect"]["delta_same"]
    if D["n_seeds_eval"] != Rr or D["delta_same"] != delta: bad.append((X, "R/delta"))
    if not np.isclose(D["crit_first_family5"], crit(X, "refit", 24)): bad.append((X, "crit5"))
    for key, cell in D["cells"].items():
        fam = key.split("|")[0]; kind = "refit" if fam in ("5", "e1iii", "e1iv") else "pred"
        g = se[(se.X == X) & (se.cell == key)]
        vcl = np.mean([max(0, a - (b - c)) for a, b, c in zip(g.v_jk, g.v_seed_all, g.v_seed) if np.isfinite(a)])
        vs = np.mean(g.v_seed_all) * 2 / Rr; s_ = np.sqrt(vcl + vs)
        if not np.isclose(s_, cell["se"], rtol=1e-9): bad.append((X, key, "se"))
        if not np.isclose(cell["halfwidth"], crit(X, kind, 1) * cell["se"]): bad.append((X, key, "hw"))
        if cell["same_level_possible"] != bool(cell["halfwidth"] < delta): bad.append((X, key, "same"))
        if fam == "5":
            if not np.isclose(cell["detectable_first"], crit(X, kind, 24) * cell["se"]): bad.append((X, key, "det"))
            e = ex["expected_effect"][X][key]
            if cell["descriptive"] != bool((not np.isfinite(e)) or cell["detectable_first"] > abs(e)): bad.append((X, key, "desc"))
    nd[X] = len(D["descriptive_cells"]); ns[X] = len(D["same_level_cells"])
    print(X, "cells", len(D["cells"]), "family5", sum(k.startswith("5|") for k in D["cells"]), "crit_first_family5 =", round(D["crit_first_family5"], 2),
          "| halfwidth/δ min over cells =", round(min(c["halfwidth"] for c in D["cells"].values()) / delta, 2),
          "| expected NaN:", sum(not np.isfinite(v) for v in ex["expected_effect"][X].values()))
print("problems", bad[:10], len(bad), "descriptive", nd, "same_level", ns)
print("oracle tether keys", {T: len(v) for T, v in fz["oracle_tether"].items()}, "oracle at edge", {T: sum(x["at_edge"] for x in v.values()) for T, v in fz["oracle_convergence"].items()},
      "oracle grad over / stalled", {T: (sum(x["n_grad_over"] for x in v.values()), sum(x["n_stalled"] for x in v.values()), set(x["n"] for x in v.values())) for T, v in fz["oracle_convergence"].items()})
print("scenario_folds", fz["scenario_folds"], "| count_rule n_dropped", fz["count_rule"]["n_dropped"])
