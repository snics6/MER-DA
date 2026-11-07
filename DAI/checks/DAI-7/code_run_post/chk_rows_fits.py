"""DAI-7 計算の後の点検 (4) の続き：rows/ の単位の行を fits/ の点数と判定から作り直す（偶然の水準 d⁰ は除く）．
値は出さず，一致・不一致の数と最大の差の桁だけを出す．"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/home/snics/MER-DA/results/dai7_probe_v2")
MODS = "AVT"
PAIRS = ("AV", "AT", "VT")
BLK = {"A": slice(0, 3), "V": slice(3, 6), "T": slice(6, 9)}
SETS3 = ("none", "ret", "joint", "joint_a1", "oracle", "joint_a1_rho0.0625", "joint_a1_rho0.25", "joint_a1_rho1.0", "joint_a1_rho4.0",
         "tt_poe_none", "tt_start_AVT", "tt_fused_AVT", "tt_ownret")
KS = (0, 1, 5, 10, 25)


def mag(x):
    return "0" if x == 0 else f"< 1e{int(np.floor(np.log10(x))) + 1}"


def dstat(a, b):
    a = a - a.mean(0); b = b - b.mean(0); n = len(a)
    Saa, Sbb, Sab = a.T @ a / n, b.T @ b / n, a.T @ b / n
    Saa = Saa + 1e-6 * np.trace(Saa) / 3 * np.eye(3); Sbb = Sbb + 1e-6 * np.trace(Sbb) / 3 * np.eye(3)
    return np.trace(np.linalg.solve(Saa, Sab) @ np.linalg.solve(Sbb, Sab.T)) / 3


units = sorted(p.name[: -len("_units.csv")] for p in (OUT / "rows").glob("*_units.csv"))
mx = {"uar": 0.0, "rec": 0.0, "d": 0.0, "phi": 0.0, "Dmean": 0.0, "z_ret_fused": 0.0}
bad = {k: 0 for k in mx}
nc = {k: 0 for k in mx}
same_joint, n_joint = 0, 0
d0_same, d0_n = 0, 0
for u in units:
    z = np.load(OUT / "fits" / f"{u}.npz")
    R = pd.read_csv(OUT / "rows" / f"{u}_units.csv").set_index(["K", "set", "pair"])
    A = pd.read_csv(OUT / "rows" / f"{u}_uar.csv").set_index(["K", "set"])
    y = z["test_y"].astype(int)
    for K in KS:
        for s in SETS3:
            a = A.loc[(K, s)]
            preds = {q: z[f"K{K}_{s}_pred_{q}"].astype(int) for q in (*MODS, "prod", "blk", "full", "lda_A", "lda_V", "lda_T")}
            for q, p in preds.items():
                v = 100 * np.mean([np.mean(p[y == e] == e) for e in range(4)])
                dd = abs(v - a[f"uar_{q}"]); mx["uar"] = max(mx["uar"], dd); bad["uar"] += dd > 1e-9; nc["uar"] += 1
                for e in range(4):
                    dd = abs(np.mean(p[y == e] == e) - a[f"rec_{q}_e{e}"]); mx["rec"] = max(mx["rec"], dd); bad["rec"] += dd > 1e-9; nc["rec"] += 1
            Z = z[f"K{K}_{s}_z"].astype(np.float64)
            for a_, b_ in PAIRS:
                r = R.loc[(K, s, a_ + b_)]
                for e in range(4):
                    ix = y == e
                    dv = dstat(Z[ix, BLK[a_]], Z[ix, BLK[b_]])
                    # fits の点数は float32 で保存した．行の d_e は float64 の点数から作った
                    dd = abs(dv - r[f"d_e{e}"]); mx["d"] = max(mx["d"], dd); bad["d"] += dd > 1e-4; nc["d"] += 1
                Dm = np.nanmean([r[f"d_e{e}"] - r[f"d0_e{e}"] for e in range(4)])
                dd = abs(Dm - r["D"]); mx["Dmean"] = max(mx["Dmean"], dd); bad["Dmean"] += dd > 1e-12; nc["Dmean"] += 1
                vals = []
                pa, pb = preds[a_], preds[b_]
                for e in range(4):
                    ix = y == e
                    ea, eb = (pa[ix] == e).astype(float), (pb[ix] == e).astype(float)
                    if ix.sum() >= 15 and ea.std() > 0 and eb.std() > 0:
                        vals.append(np.corrcoef(ea, eb)[0, 1])
                dd = abs(np.mean(vals) - r["phi"]); mx["phi"] = max(mx["phi"], dd); bad["phi"] += dd > 1e-12; nc["phi"] += 1
        # joint と joint_a1 の点数が同じか（設計 3.2 節の「固定のファイルの実際」）
        n_joint += 1
        same_joint += bool(np.array_equal(z[f"K{K}_joint_z"], z[f"K{K}_joint_a1_z"]))
        # 共通の乱数：d⁰ の行が組で違うか（点数が違えば違う．同じ点数なら同じはず）→ joint と joint_a1
        for p in PAIRS:
            d0_n += 1
            d0_same += all(R.loc[(K, "joint", p), f"d0_e{e}"] == R.loc[(K, "joint_a1", p), f"d0_e{e}"] for e in range(4))
        # ret と tt_fused_AVT は同じ最適化の問（設計 3.2 節．数値の確かめ）
        dz = float(np.max(np.abs(z[f"K{K}_ret_z"].astype(float) - z[f"K{K}_tt_fused_AVT_z"].astype(float))))
        sc = float(np.max(np.abs(z[f"K{K}_ret_z"].astype(float))))
        mx["z_ret_fused"] = max(mx["z_ret_fused"], dz / sc); nc["z_ret_fused"] += 1; bad["z_ret_fused"] += dz / sc > 1e-2
for k in mx:
    print(f"{k}：比べた {nc[k]}，許す差を超えた数 {int(bad[k])}，最大の差 {mag(mx[k])}")
print("許す差：uar・rec 1e-9，d_e 1e-4（fits の点数は float32），D の平均・phi 1e-12，ret と tt_fused_AVT の点数の相対の差 1e-2")
print(f"joint と joint_a1 の点数が同じ (単位, K)：{same_joint} / {n_joint}")
print(f"joint と joint_a1 の d⁰ が同じ (単位, K, 対)：{d0_same} / {d0_n}")
