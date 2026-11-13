"""select_points（台本 v1 の写し）に人工の単位の表を渡す．どの格子の点も単位の中では同じ UAR（差は 0）で，仮のターゲットの水準だけが違うとき，何が選ばれるか．"""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, "/home/snics/MER-DA/DAI/checks/DAI-18/review_ml_sim")
import v1_copy as V
rows = []
level = {"AFEW": 40, "IEMOCAP": 45, "MOSEI": 42, "MELD": 38, "CREMA-D": 55, "RAVDESS": 30}
rng = np.random.default_rng(0)
for T, c, f, r in V.all_units():
    for K in V.KS:
        base = level[c] + rng.normal(0, 0.01)
        CF = V.cfgs_of(c)
        vs = [f"MN_{fm}_c{i}_l{j}" for fm in ("eq", "F2K", "F6K", "F6orc") for i in range(len(CF)) for j in range(3)]
        vs += [f"N_eq_c{i}" for i in range(len(CF))] + [f"MS_eq_l{j}" for j in range(3)]
        if len(V.MODS_OF[c]) >= 2:
            vs += [f"M_F6_mix{i}" for i in range(16)]
        for v in vs:
            rows.append(dict(T=T, c=c, fold=f, seed=r, K=K, version=v, uar_adj=base))
d = pd.DataFrame(rows)
ch, cr = V.select_points(d)
s = pd.DataFrame(cr)
s = s[(s.K == 5) & s.version.isin(["MN_eq_sel", "N_eq_sel", "MN_F2K_sel"])]
s["joint"] = s.chosen.str.extract(r"_c(\d)").astype(int) >= 4
print(s[["c", "version", "chosen", "joint", "n_tied", "n_cands"]].to_string(index=False))
