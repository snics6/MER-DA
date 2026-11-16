"""gate-code の検算（dai20_select_strongest）．UAR の水準と差は画面に出さない．"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA/DAI")
sys.path.insert(0, str(ROOT / "exp"))
import dai16_probe as P16
import dai_solver as S
U = ROOT / "results/dai20_eval/units"
MODS = "AVT"; KS = (5, 10, 25)
# 1. 単位の csv のコミット
com = set(); dirty = set()
for p in sorted(U.glob("*.csv")):
    d = pd.read_csv(p, usecols=["commit", "dirty"], low_memory=False)
    com |= set(map(str, d.commit.unique())); dirty |= set(map(str, d.dirty.unique()))
print("units commit", com, "dirty", dirty)
stat = dict(prod_adj_ne_raw=0, ret_raw_ne_argmax=0, avg_adj_recon_ne=0, n=0, tieK_uar=0, tie_d={"K": 0, "ext": 0, "orc": 0},
            near_d={"K": 0, "ext": 0, "orc": 0}, kuar_agree_dK=0, frac_change_adj={m: [] for m in MODS}, adj_resid_max=0.0)
ext_per_T = {}
for p in sorted(U.glob("*_ALL_f*_r*.npz")):
    T = p.stem.split("_")[0]
    z = np.load(p, allow_pickle=False)
    yt, yp = z["y_test"].astype(int), z["y_pool"].astype(int)
    ext_per_T.setdefault(T, set()).add(int(np.argmax(z["SR_M_ext"].mean(1))))
    for K in KS:
        stat["n"] += 1
        stat["prod_adj_ne_raw"] += int((z[f"pred_adj_K{K}_prod"] != z[f"pred_raw_K{K}_prod"]).sum())
        Pt = [z[f"p_test_ret_{m}_K{K}"].astype(np.float64) for m in MODS]
        Pp = [z[f"p_pool_ret_{m}_K{K}"].astype(np.float64) for m in MODS]
        for k, m in enumerate(MODS):
            stat["ret_raw_ne_argmax"] += int((Pt[k].argmax(1) != z[f"pred_raw_K{K}_ret_{m}"]).sum())
        LMt = np.stack([P16.centered(x) for x in Pt]); LMp = np.stack([P16.centered(x) for x in Pp])
        target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)
        # 作り直しの照合：add_extra の avg の adj の判定
        dl, _ = P16.match_shift(P16.lg(np.mean(Pp, 0)), target)
        stat["avg_adj_recon_ne"] += int(((P16.lg(np.mean(Pt, 0)) + dl).argmax(1) != z[f"pred_adj_K{K}_avg"]).sum())
        # 単独の分類器を積の候補の上の平均に合わせた判定（ret_m@adj）
        for k, m in enumerate(MODS):
            dl, rs = P16.match_shift(LMp[k], target)
            stat["adj_resid_max"] = max(stat["adj_resid_max"], rs)
            stat["frac_change_adj"][m].append(float(((LMt[k] + dl).argmax(1) != z[f"pred_raw_K{K}_ret_{m}"]).mean()))
        # K 本の上の交差の UAR で選ぶ
        kpos = z[f"kpos_K{K}"]; yk = yp[kpos]
        ku = np.array([S.uar(yk, z[f"pcf_{m}_K{K}"].astype(np.float64).argmax(1)) for m in MODS])
        if (np.isclose(ku, ku.max(), atol=1e-12)).sum() > 1:
            stat["tieK_uar"] += 1
        dK = z[f"D_M_K_K{K}"].mean(1)
        if int(np.argmax(ku)) == int(np.argmax(dK)):
            stat["kuar_agree_dK"] += 1
        for nm, s in (("K", dK), ("ext", z["SR_M_ext"].mean(1)), ("orc", z[f"D_M_orc_K{K}"].mean(1))):
            ss = np.sort(s)[::-1]
            stat["tie_d"][nm] += int(np.isclose(ss[0], ss[1], rtol=0, atol=1e-12))
            stat["near_d"][nm] += int(ss[0] - ss[1] < 1e-3)
print({k: v for k, v in stat.items() if k != "frac_change_adj"})
print("sel_ext の選び方が T の中で何通りか", {T: len(v) for T, v in ext_per_T.items()})
fc = {m: (np.min(v), np.median(v), np.max(v)) for m, v in stat["frac_change_adj"].items()}
print("ret_m の判定のうち，積の平均に合わせる切片のずらしで変わる割合（最小，中央，最大）", fc)
