"""DAI-7 計算の後の点検：tether_weakening の作り直しとログの警告．値は出さない．"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA"); OUT = ROOT / "results/dai7_probe_v2"; UD = ROOT / "results/dai1/pseudo/units"
F = json.load(open(ROOT / "DAI/frozen/dai1_frozen.json"))
TWu = pd.read_csv(OUT / "tether_weakening_units.csv").set_index(["T", "c", "fold", "seed", "K", "mod"])
TWc = pd.read_csv(OUT / "tether_weakening.csv").set_index(["c", "K", "mod"])
cwmax = 1.0
bad = n = 0
for p in sorted(UD.glob("*_p3.npz")):
    u = p.name[:-len("_p3.npz")]; T, c, fs, rs = u.split("_")
    teth = {(s.split("|")[0], int(s.split("|")[1])): float(s.split("|")[2]) for s in map(str, np.load(p)["tether"])}
    npool = len(np.load(OUT / "fits" / f"{u}.npz")["pool_sample_id"])
    for K in (1, 5, 10, 25):
        ver = F[T]["main"][str(K)]
        for m in "AVT":
            r = TWu.loc[(T, c, int(fs[1:]), int(rs[1:]), K, m)]
            rho = float(F[T]["select"][f"{m}|{K}|joint_a1|{ver}"]["rho"]); cw = teth[(m, K)]
            exp = dict(n_pool=npool, rho_joint_a1=rho, weakening_joint_a1=(4*K + rho*npool)/(4*K), weakening_rho1=(4*K + npool)/(4*K), C_none=cw, tw_range=cwmax/cw)
            for k, v in exp.items():
                n += 1; bad += not np.isclose(r[k], v, rtol=1e-12, atol=0)
print(f"tether_weakening_units：比べた {n}，不一致 {bad}")
cols = ["n_pool", "rho_joint_a1", "weakening_joint_a1", "weakening_rho1", "C_none", "tw_range"]
g = TWu.reset_index().groupby(["c", "K", "mod"])[cols].mean()
d = np.abs(g[cols].to_numpy() - TWc.loc[g.index, cols].to_numpy())
print(f"tether_weakening（単位の表の平均）：比べた {d.size}，不一致 {int((d > 1e-9 * np.maximum(1, np.abs(g[cols].to_numpy()))).sum())}")
L = (ROOT / "logs/dai7_probe_v2.log").read_text().splitlines()
pat = re.compile(r"(Error|Exception|Traceback|Warning|warn|nan|inf|Killed|OOM|out of memory|やり直)", re.I)
hits = [(i + 1, l[:80]) for i, l in enumerate(L) if pat.search(l)]
print("ログの行数", len(L), "警告・例外らしい行", len(hits))
for i, l in hits:
    print(" ", i, l)
print("単位の行の数", sum(1 for l in L if re.match(r"^[A-Z]+_[A-Z]+_f\d_r\d：", l)), "始まり・終わりの行", [l[:40] for l in L if l.startswith(("開始", "終了"))])
