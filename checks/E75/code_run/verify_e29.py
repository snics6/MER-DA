import numpy as np, pandas as pd
S = "/home/snics/MER-DA/results/explore_relation/snapshots/20261011-1524_e29"
e = pd.read_csv(f"{S}/e29_pairs.csv", dtype={"K": str}); T = pd.read_csv(f"{S}/e29_tests.csv")
A = "/home/snics/MER-DA/checks/E75/code_run/runA"
p5 = pd.read_csv(f"{A}/e75_5_pairs.csv", dtype={"K": str}); p8 = pd.read_csv(f"{A}/e75_8_pairs.csv")
z = pd.read_csv(f"{A}/e75_8_p29_3.csv")
k = ["kind", "target", "source", "K"]
j = p5.merge(e[k + ["share_pos", "L_crit", "L_disc", "share_loss"]], on=k)
print("割合（今の順序）と E29 の share_pos の差の最大", float(np.nanmax(np.abs(j.share_f - j.share_pos))), "行", len(j))
j8 = p8.merge(e[e.K == "5"][["kind", "target", "source", "L_crit", "L_disc", "share_loss"]], on=["kind", "target", "source"], suffixes=("", "_e29"))
print("項目 8 の L_crit・L_disc（生）と E29 の差の最大", float(np.nanmax(np.abs(np.r_[j8.L_crit - j8.L_crit_e29, j8.L_disc_raw - j8.L_disc]))), "行", len(j8))
print("損の割合（生）と E29 share_loss の差の最大", float(np.nanmax(np.abs(j8.share_loss_raw - j8.share_loss))))
t3 = T[T.test == "P29-3"][["group", "rho"]]
zz = z[(z.order == "f") & (z.dir_loss == "raw")][["kind", "rho"]]
print(t3.merge(zz, left_on="group", right_on="kind", how="outer").assign(diff=lambda d: (d.rho_x - d.rho_y).abs())[["group", "kind", "diff"]].to_string())
