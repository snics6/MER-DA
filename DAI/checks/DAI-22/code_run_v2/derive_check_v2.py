"""gate-code 再検査（DAI-22 第 2 版）：要約の新しい量を，実験 14 の部品で独立に作り直して差だけを出す．値は出さない．
- delta_minus_delta_<s>：Δ_AVT − Δ_s を，実験 14 の npz の y・cl・nk・J_prop・none__K__s から D14.cell で Δ_s を作って引いた値と比べる
- MELD nonshared の uar_pseudo：実験 14 の npz と D14.cell(subset="nonshared", K_nk) で作り直す
- npz の新しい鍵：nonshared・marker・sex が実験 14 の npz と同じ，start は K に依らない，clamped の数
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai14 as D14  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai22  # noqa: E402

O = Path(sys.argv[1])
c = pd.read_csv(O / "summary" / "conditions.csv")
worst = {}
for T in ("AFEW", "IEMOCAP", "MELD", "MOSEI"):
    U = dai22.load_units(O, T)
    U14 = {k: dict(np.load(dai22.OUT14 / "units" / f"{T}_f{k[0]}_r{k[1]}.npz", allow_pickle=False)) for k in U}
    for k in U:
        for key in ("nonshared", "marker", "sex", "y", "half"):
            assert (np.asarray(U[k][key]).astype(str) == np.asarray(U14[k][key]).astype(str)).all(), (T, k, key)
    g_eff, _ = ST.g_eff_of(T)
    for K in dai22.KS:
        for subset in (["all", "nonshared"] if T == "MELD" else ["all"]):
            for s in "AVT":
                d_s = D14.combo_stats([D14.cell(U14, f"tgt__K{K}__prop_one__true", "y", subset, K),
                                       D14.cell(U14, f"none__K{K}__{s}", "y", subset, K)], [1, -1], g_eff, np.random.default_rng(0))["effect"]
                row = c[(c["T"] == T) & (c.K == K) & (c.subset == subset)]
                got = float(row[row.quantity == f"delta_minus_delta_{s}"].effect.iloc[0])
                dl = float(row[row.quantity == "delta"].effect.iloc[0])
                worst["dmd"] = max(worst.get("dmd", 0), abs(got - (dl - d_s)))
            up = D14.combo_stats([D14.cell(U14, f"tgt__K{K}__prop_one__true", "y", subset, K)], [1], g_eff, np.random.default_rng(0))["effect"]
            got = float(row[row.quantity == "uar_pseudo"].effect.iloc[0])
            worst[f"pseudo_{subset}"] = max(worst.get(f"pseudo_{subset}", 0), abs(got - up))
    n_ns = [int((a["nk__K5"].astype(bool) & a["nonshared"].astype(bool)).sum()) for a in U.values()]
    n_cl = [int(a[f"clamped__K{K}"].sum()) for a in U.values() for K in dai22.KS]
    print(T, "units", len(U), "nonshared∩nk (K=5) min", min(n_ns), "clamped total", sum(n_cl),
          "start dtype", U[sorted(U)[0]]["start"].dtype, "pca", U[sorted(U)[0]]["pca"].shape, "keys", len(U[sorted(U)[0]]))
print("max abs diff:", worst)
