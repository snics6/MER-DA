"""gate-code 第 2 回の検算．UAR の水準と差は画面に出さない．"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA/DAI"); sys.path.insert(0, str(ROOT / "exp"))
import dai16_probe as P16, dai_solver as S
O = Path(sys.argv[1])
e = pd.read_csv(O / "select_strongest_effects.csv"); l = pd.read_csv(O / "select_strongest_levels.csv")
pe = pd.read_csv(O / "select_strongest_per_emotion.csv"); pk = pd.read_csv(O / "select_strongest_picks.csv")
print("行数", len(e), len(l), len(pe), len(pk))
print("重複", e.duplicated(["a", "b", "c", "K", "seeds"]).sum(), l.duplicated(["c", "K", "version", "seeds"]).sum(),
      pe.duplicated(["c", "K", "version", "emotion"]).sum(), pk.duplicated(["T", "fold", "seed", "K"]).sum())
print("effects の seeds", e.seeds.value_counts().to_dict(), "levels の seeds", l.seeds.value_counts().to_dict())
print("main の行", e[e.main].groupby(["a", "b"]).size().to_dict())
print("n_fr", sorted(e.n_fr.unique()), sorted(l.n_fr.unique()))
print("NaN（effects）", {k: int(v) for k, v in e.isna().sum().items() if v}, "NaN（levels，n_ を除く）", {k: int(v) for k, v in l.drop(columns=[c for c in l if c.startswith("n_")]).isna().sum().items() if v})
print("NaN（per_emotion）", int(pe.recall.isna().sum()))
# 感情ごとの再現率の平均と水準の一致（all）
la = l[l.seeds == "all"].set_index(["c", "K", "version"]).uar
pm = pe.groupby(["c", "K", "version"]).recall.mean()
print("4 感情の再現率の平均 − 水準 の最大の絶対値", float((pm - la.loc[pm.index]).abs().max()))
# 選びの独立な作り直し
MODS = "AVT"; mism = {k: 0 for k in ("sel_Kuar", "sel_test_raw", "sel_test_adj", "n_tie_Kuar")}; adjdiff = 0; ntie_test = 0
pkk = pk.set_index(["T", "fold", "seed", "K"])
for p in sorted((ROOT / "results/dai20_eval/units").glob("*_ALL_f*_r*.npz")):
    T, _, f, r = p.stem.split("_"); f, r = int(f[1:]), int(r[1:])
    z = np.load(p); yt, yp = z["y_test"].astype(int), z["y_pool"].astype(int)
    for K in (5, 10, 25):
        row = pkk.loc[(T, f, r, K)]
        yk = yp[z[f"kpos_K{K}"]]
        uk = np.array([S.uar(yk, z[f"pcf_{m}_K{K}"].argmax(1)) for m in MODS])
        dk = z[f"D_M_K_K{K}"].mean(1)
        cand = [j for j in range(3) if abs(uk[j] - uk.max()) < 1e-12]
        best = max(cand, key=lambda j: (dk[j], -j))
        mism["sel_Kuar"] += MODS[best] != row.sel_Kuar_mod
        mism["n_tie_Kuar"] += (len(cand) > 1) != bool(row.sel_Kuar_tie)
        LMt = [P16.centered(z[f"p_test_ret_{m}_K{K}"].astype(np.float64)) for m in MODS]
        LMp = [P16.centered(z[f"p_pool_ret_{m}_K{K}"].astype(np.float64)) for m in MODS]
        tgt = np.exp(P16.norm_log(sum(LMp))).mean(0)
        sh = []
        for j in range(3):
            dl, _ = P16.match_shift(LMp[j], tgt); sh.append((LMt[j] + dl).argmax(1))
        rawj = [z[f"pred_raw_K{K}_ret_{m}"].astype(int) for m in MODS]
        ur = [S.uar(yt, q) for q in rawj]; ua = [S.uar(yt, q) for q in sh]
        mism["sel_test_raw"] += MODS[int(np.argmax(ur))] != row.sel_test_raw_mod
        mism["sel_test_adj"] += MODS[int(np.argmax(ua))] != row.sel_test_adj_mod
        ntie_test += int(row.sel_test_raw_tie) + int(row.sel_test_adj_tie)
print("選びの作り直しと台本の違い", mism, "sel_test の同点（raw と adj の合計）", ntie_test)
print("同点の数（picks）", {c: int(pk[c].sum()) for c in pk if c.endswith("_tie")})
# levels の n_ の和
nn = l[l.version.str.startswith("sel")]
print("n_A+n_V+n_T の範囲", (nn.n_A + nn.n_V + nn.n_T).min(), (nn.n_A + nn.n_V + nn.n_T).max())
# 集計と同じ名前の水準の照合（prop・prop@raw・prod の 0-2）
L0 = pd.read_csv(ROOT / "results/dai20_eval/summary/levels.csv")
m_ = l[l.version.isin(["prop", "prop@raw", "prod"])].merge(L0, on=["c", "K", "version", "seeds"], suffixes=("", "_0"))
print("集計の levels.csv との照合（行数，est・lo・hi のずれの最大）", len(m_), float(np.max(np.abs(m_[["uar", "lo", "hi"]].to_numpy() - m_[["uar_0", "lo_0", "hi_0"]].to_numpy()))))
E0 = pd.read_csv(ROOT / "results/dai20_eval/summary/effects.csv")
pe0 = pd.read_csv(ROOT / "results/dai20_eval/summary/per_emotion.csv")
m2 = pe[pe.version.isin(["prop", "prop@raw", "prod"])].merge(pe0, on=["c", "K", "version", "emotion"], suffixes=("", "_0"))
print("集計の per_emotion.csv との照合", len(m2), float(np.nanmax(np.abs(m2[["recall", "lo", "hi"]].to_numpy() - m2[["recall_0", "lo_0", "hi_0"]].to_numpy()))))
# 同じ判定になる対
print("same_counts が真の対の数", int(e[e.seeds == "all"].same_counts.sum()), "／", int((e.seeds == "all").sum()))
