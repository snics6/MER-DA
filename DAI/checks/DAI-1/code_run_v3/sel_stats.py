"""選ぶ段と統計の台本を，絞った設定の仮のターゲットの出力で動かす（UAR は出さない）。引数：置き場"""
import sys, json, subprocess
import numpy as np, pandas as pd
from pathlib import Path
import dai1, dai1_select as SEL, dai1_stats as ST, dai_data as DD
out = Path(sys.argv[1])
before = subprocess.run(["git", "status", "--porcelain", "DAI/frozen"], cwd=dai1.ROOT, capture_output=True, text=True).stdout
SEL.main(out, ["MOSEI"])
after = subprocess.run(["git", "status", "--porcelain", "DAI/frozen"], cwd=dai1.ROOT, capture_output=True, text=True).stdout
fz = json.load(open(out / "select" / "frozen.json"))
F = fz["MOSEI"]
print("DAI/frozen が変わらない", before == after)
print("選んだ鍵の数", len(F["select"]), "手法ごと", pd.Series([k.split("|")[2] for k in F["select"]]).value_counts().to_dict())
print("相手", pd.Series([v["method"] + "|" + v["version"] for v in F["partner"].values()]).value_counts().to_dict())
print("主の版", F["main"], "同点", F["tie_rule_used_family"], F["tie_rule_used_extra"], "格子の端", F.get("grid_edge_cells"))
# coreg と cotrain の選んだ点の鍵
print("coreg の鍵の例", {k: v for k, v in F["select"].items() if "|coreg|" in k and k.startswith("AVT|5")})
# collect の照合
y_of, cl_of = ST.label_maps("AFEW")
units = [(f, r) for f in range(DD.N_FOLDS) for r in (0, 1)]
cells, mism, total = ST.collect(out, "pseudo", "MOSEI", "AFEW", F, units, y_of, cl_of)
print("collect の照合 食い違い", mism, "/", total, "セルの数", len(cells))
# 相手を coreg に替えた固定のファイルで unit_preds が通るか（全生徒・K）
F2 = json.loads(json.dumps(F))
for k in F2["partner"]:
    F2["partner"][k] = {"method": "coreg", "version": "cr", "pseudo_uar": 0}
cells2, _, _ = ST.collect(out, "pseudo", "MOSEI", "AFEW", F2, units, y_of, cl_of, check_rows=False)
print("相手が coreg の版でも集められた", len(cells2))
g_eff, _ = ST.g_eff_of("AFEW")
rng = np.random.default_rng(0)
for fam, (A, B, ks) in ST.FAMS.items():
    for s in DD.STUDENTS:
        for K in ks:
            e = ST.effect_stats(cells[(s, K, A)], cells[(s, K, B)], g_eff, rng)
    print("族", fam, "の区間を作れた（最後のセル）", e["method"], "identical", e["identical"])
tw = json.load(open(out / "pseudo" / "MOSEI_tether_w.json"))
ee = ST.expected_effects(out, "MOSEI", tw)
print("expected_effects の鍵", len(ee), "非有限", int(sum(not np.isfinite(v) for v in ee.values())))
