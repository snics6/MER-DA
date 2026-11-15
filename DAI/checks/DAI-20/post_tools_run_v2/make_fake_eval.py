"""dai20_posthoc_intercept.py の既定の道（--key T）を試すための作り物の評価の回の置き場を作る．
units/：評価の回と同じ名前（{T}_ALL_f{f}_r{r}）と同じ鍵（y_test，cl_test，pred_{adj,raw,kb,ob}_K{K}_{版}）の小さい npz と csv．
summary/：dai20_eval.stage_summary と同じ作り（同じ cells の作り方，同じ乱数の鍵，同じ種の分け方）で作った levels.csv・effects.csv と，grid_levels.csv．
値は作り物．"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1_stats as ST  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai20_eval as E  # noqa: E402

OUT = Path(sys.argv[1])
U, S = OUT / "units", OUT / "summary"
U.mkdir(parents=True, exist_ok=True); S.mkdir(parents=True, exist_ok=True)
NCL, NPER = 6, 40
BASES = ["prod", "M_F6_K", "M_F2_K", "cf_D_l05", "M_F6_ext", "M_F8", "M_F6_orc"] + [f"M_F6_mix{i}" for i in range(E.NG)] + [f"M_F6n_mix{i}" for i in range(E.NG)]
LAY = ("adj", "raw", "kb", "ob")
acc = {b: 0.35 + 0.25 * ((hash(b) % 1000) / 1000.0) for b in BASES}
acc = {b: 0.35 + 0.02 * (j % 13) for j, b in enumerate(BASES)}
frozen = E.load_frozen()
LOTO = {c: 3 for c in E.CORP}
ORAC = {c: 14 for c in E.CORP}
for ci, T in enumerate(E.CORP):
    for f in range(5):
        for r in range(10):
            rng = np.random.default_rng([ci, f, r])
            y = np.tile(np.arange(4), NCL * NPER // 4)
            rng.shuffle(y)
            cl = np.repeat([f"{T}_f{f}_g{j}" for j in range(NCL)], NPER)
            arr = dict(y_test=y.astype(np.int8), cl_test=cl)
            for K in E.KS:
                for b in BASES:
                    for li, lay in enumerate(LAY):
                        a = acc[b] + 0.03 * li + 0.01 * (K / 25)
                        hit = rng.random(len(y)) < a
                        arr[f"pred_{lay}_K{K}_{b}"] = np.where(hit, y, (y + rng.integers(1, 4, len(y))) % 4).astype(np.int8)
            name = f"{T}_ALL_f{f}_r{r}"
            np.savez_compressed(U / f"{name}.npz", **arr)
            pd.DataFrame([dict(T=T, c=T, fold=f, seed=r, K=5, version="prod", commit="fake")]).to_csv(U / f"{name}.csv", index=False)

# ---- 集計の段と同じ作り（dai20_eval.stage_summary 428〜543 行の該当の部分）
gr = []
pick = {}
for c in E.CORP:
    for K in E.KS:
        i_fz, j_np = frozen[c][K]
        pick[(c, K)] = dict(prop=i_fz, prop_noprof=E.same_but(i_fz, beta=0.0), prop_noprof_sel=j_np, prop_n0zero=E.same_but(i_fz, n0=0.0), prop_loto=LOTO[c],
                            oracle_grid=ORAC[c], prop_levelfree=E.GRID.index((float("inf"), 1.0, "eq")))
        for i in range(E.NG):
            gr.append(dict(c=c, K=K, grid=i, n0=E.GRID[i][0], beta=E.GRID[i][1], prior=E.GRID[i][2], uar_adj=50.0, uar_adj_others=50.0,
                           frozen=bool(i == i_fz), loto=bool(i == LOTO[c]), oracle=bool(i == ORAC[c])))
pd.DataFrame(gr).to_csv(S / "grid_levels.csv", index=False)
cells = {}
for p in sorted(U.glob("*.npz")):
    T, _, f, r = p.stem.split("_")
    f, r = int(f[1:]), int(r[1:])
    z = np.load(p, allow_pickle=False)
    yt = z["y_test"].astype(int)
    for K in E.KS:
        i = pick[(T, K)]["prop"]
        names = {v: f"pred_adj_K{K}_{v}" for v in ("prod", "M_F2_K", "M_F6_K", "cf_D_l05") + E.REF_VERS + E.ORC_VERS}
        names.update({f"{v}@raw": f"pred_raw_K{K}_{v}" for v in ("prod", "M_F2_K", "M_F6_K", "cf_D_l05")})
        names.update({v: f"pred_adj_K{K}_M_F6_mix{j}" for v, j in pick[(T, K)].items()})
        names["prop@raw"] = f"pred_raw_K{K}_M_F6_mix{i}"
        names["prop_noint"] = f"pred_adj_K{K}_M_F6n_mix{i}"
        for v, kk in names.items():
            P12._add(cells.setdefault((T, K, v), {}), (f, r), ST.counts(yt, z[kk].astype(int), z["cl_test"]), f"{p.stem} {v}")
P12._finish(cells)
geff = {c: ST.g_eff_of(c)[0] for c in E.CORP}


def zero(A):
    return {k_: (x[0], np.zeros_like(x[1]), x[2]) for k_, x in A.items()}


def sub(A, seeds):
    return A if seeds is None else {k_: x for k_, x in A.items() if k_[1] in seeds}


lrows = []
for (c, K, v), dd in sorted(cells.items()):
    for sn in ("all", "0-2"):
        ds = sub(dd, E.SEED_SETS[sn])
        es = ST.effect_stats(ds, zero(ds), geff[c], E.seed_rng("L", c, K, v, sn))
        lrows.append(dict(c=c, K=K, version=v, seeds=sn, uar=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], same_as=""))
pd.DataFrame(lrows).to_csv(S / "levels.csv", index=False)
effs = []
for a in ("prop", "prop_noprof", "prop_noprof_sel", "prop_loto", "oracle_grid", "prop_levelfree", "M_F2_K", "M_F6_K", "cf_D_l05", "M_F6_orc", "M_F6_ext", "M_F8"):
    for c in E.CORP:
        for K in E.KS:
            A, B = cells[(c, K, a)], cells[(c, K, "prod")]
            for sn, ss in E.SEED_SETS.items():
                As, Bs = sub(A, ss), sub(B, ss)
                if sn == "0-2" or not As or (sn != "all" and len(As) == len(A)):
                    continue
                es = ST.effect_stats(As, Bs, geff[c], E.seed_rng("A", a, "prod", c, K, sn))
                effs.append(dict(a=a, b="prod", c=c, K=K, seeds=sn, est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], method=es["method"]))
pd.DataFrame(effs).to_csv(S / "effects.csv", index=False)
print("作り物の置き場", OUT, "単位", len(list(U.glob('*.npz'))), "水準", len(lrows), "差", len(effs))
