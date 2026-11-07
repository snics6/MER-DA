"""DAI-7 計算の後の点検：保存済みの値の再現．実験 1 の仮のターゲットの回が保存した単独の分類器のテストの UAR（p1・p3 の CSV）を，
この下調べが W・b を当て直して出した UAR（rows/*_uar.csv）と突き合わせる．値は出さず，一致・不一致の数と最大の差の桁だけを出す．"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA")
OUT = ROOT / "results" / "dai7_probe_v2"
UD = ROOT / "results" / "dai1" / "pseudo" / "units"
F = json.load(open(ROOT / "DAI" / "frozen" / "dai1_frozen.json"))
MODS = "AVT"


def mag(x):
    return "0" if x == 0 else f"< 1e{int(np.floor(np.log10(x))) + 1}"


units = sorted(p.name[: -len("_uar.csv")] for p in (OUT / "rows").glob("*_uar.csv"))
res = {k: [0, 0, 0.0, 0] for k in ("none", "oracle", "joint_a1", "tt_ownret", "joint_a1_rho1.0")}   # 比べた，不一致，最大の差，見つからない
for u in units:
    T = u.split("_")[0]
    A = pd.read_csv(OUT / "rows" / f"{u}_uar.csv").set_index(["K", "set"])
    p1 = pd.read_csv(UD / f"{u}_p1.csv")
    p3 = pd.read_csv(UD / f"{u}_p3.csv")
    teth = {}
    for s in np.load(UD / f"{u}_p3.npz")["tether"]:
        kind, K, cw, cb = str(s).split("|")
        teth[(kind, int(K))] = (float(cw), float(cb))

    def put(key, ours, saved):
        r = res[key]
        if len(saved) != 1:
            r[3] += 1
            return
        d = abs(float(ours) - float(saved.iloc[0]))
        r[0] += 1; r[1] += d > 1e-6; r[2] = max(r[2], d)

    for m in MODS:
        cw, cb = teth[(m, 25)]
        sv = p1[(p1.method == "oracle") & (p1.kind == m) & np.isclose(p1.Cw, cw) & np.isclose(p1.Cb, cb)].uar
        put("oracle", A.loc[(0, "oracle"), f"uar_{m}"], sv)
        for K in (1, 5, 10, 25):
            cw, cb = teth[(m, K)]
            sv = p1[(p1.method == "none") & (p1.kind == m) & (p1.K == K) & np.isclose(p1.Cw, cw) & np.isclose(p1.Cb, cb)].uar
            put("none", A.loc[(K, "none"), f"uar_{m}"], sv)
        for K in (0, 1, 5, 10, 25):
            ver = F[T]["main"][str(K)]
            rho = float(F[T]["select"][f"{m}|{K}|joint_a1|{ver}"]["rho"])
            sv = p3[(p3.method == "joint") & (p3.student == m) & (p3.K == K) & np.isclose(p3.rho, rho) & np.isclose(p3.a, 1) & (p3.teacher == ver)].uar
            put("joint_a1", A.loc[(K, "joint_a1"), f"uar_{m}"], sv)
            sv = p3[(p3.method == "joint") & (p3.student == m) & (p3.K == K) & np.isclose(p3.rho, 1) & np.isclose(p3.a, 1) & (p3.teacher == ver)].uar
            put("joint_a1_rho1.0", A.loc[(K, "joint_a1_rho1.0"), f"uar_{m}"], sv)
            sv = p3[(p3.method == "tt") & (p3.student == m) & (p3.K == K) & np.isclose(p3.rho, 1) & np.isclose(p3.a, 1) & (p3.teacher == f"ret_{m}")].uar
            put("tt_ownret", A.loc[(K, "tt_ownret"), f"uar_{m}"], sv)
for k, (n, b, mx, miss) in res.items():
    print(f"{k}：比べた {n}，差が 1e-6 を超えた {int(b)}，最大の差 {mag(mx)}，保存の行が 1 つに決まらない {miss}")
