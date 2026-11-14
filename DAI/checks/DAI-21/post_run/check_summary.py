"""DAI-21 計算の後の点検 2：summary の表の行・鍵・対の数・NaN 列，same_as/same_pred の作り直し，水準と差の点の推定の独立な作り直し，
区間の作り直し（同じ鍵），実験 20 の集計との一致，混同行列・感情ごとの再現率・格子・重みの作り直し（読むだけ）．値は出さず，数とずれだけを出す．"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1_stats as ST  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai18_probe as P18  # noqa: E402
import dai20_eval as P20  # noqa: E402

R = ROOT / "results" / "dai21_eval5"
SD = R / "summary"
OUT20 = ROOT / "results" / "dai20_eval"
fz = json.load(open(ROOT / "DAI" / "frozen" / "dai21_frozen.json"))
f20 = json.load(open(ROOT / "DAI" / "frozen" / "dai20_frozen.json"))["choice"]
C4 = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
CD = "CREMA-D"
KS = (5, 10, 25)
EXTRA_T = ("noprof", "noint", "levelfree", "konly", "tcal")
EX1 = ("kd", "agree", "mix")
GRID = P16.GRID
B_VERS = P20.B_VERS
A2 = ("prod", "avg", "conf", "M_F2_K", "M_F6_K", "cf_D_l05", "tcal", "temp", "tempb", "wf10", "wf100", "wf1000")
SEEDS = {"all": None, "0-1": (0, 1), "2-9": tuple(range(2, 10))}
out = {}


def uar(y, p):
    return 100.0 * np.mean([np.mean(p[y == c] == c) for c in range(4) if (y == c).any()])


# ---------------------------------------------------------------- 1. 行・鍵・NaN 列
files = {"levels.csv": ["part", "c", "K", "version", "seeds"], "effects.csv": ["part", "a", "b", "c", "K", "seeds"],
         "per_emotion.csv": ["part", "c", "K", "version", "emotion"], "confusion.csv": ["c", "K", "version", "true", "pred"],
         "student_grid.csv": ["part", "c", "K", "version", "grid"], "weights_cremad.csv": ["c", "K", "mod", "emotion"], "grid_levels_cremad.csv": ["c", "K", "grid"]}
tab = {}
for fn, key in files.items():
    d = pd.read_csv(SD / fn, keep_default_na=True, low_memory=False)
    tab[fn] = d
    out[fn] = dict(rows=len(d), dup_keys=int(d.duplicated(key).sum()), all_nan_cols=[c for c in d.columns if d[c].isna().all()],
                   inf_cells=int(np.isinf(d.select_dtypes(include=[np.number]).to_numpy()).sum()))
L, E = tab["levels.csv"], tab["effects.csv"]
for nm, d, cols in (("levels.csv", L, ["uar", "lo", "hi", "se"]), ("effects.csv", E, ["est", "lo", "hi", "se"])):
    out[nm]["nan_by_col"] = {c: int(d[c].isna().sum()) for c in cols}
    out[nm]["by_part_seeds"] = {f"{p}|{s}": int(n) for (p, s), n in d.groupby(["part", "seeds"]).size().items()}
    out[nm]["methods"] = {f"{p}|{m}": int(n) for (p, m), n in d.groupby(["part", "method"]).size().items()}
out["effects.csv"]["n_pairs"] = {p: int(g[["a", "b"]].drop_duplicates().shape[0]) for p, g in E.groupby("part")}
out["effects.csv"]["pairs_per_cK"] = {p: sorted(set(g.groupby(["c", "K"]).size().tolist())) for p, g in E[E.seeds == "all"].groupby("part")}
out["levels.csv"]["n_versions_per_cK"] = {p: sorted(set(g.groupby(["c", "K"]).size().tolist())) for p, g in L[L.seeds == "all"].groupby("part")}
out["levels.csv"]["n_fr"] = {f"{p}|{s}": sorted(set(g.n_fr.tolist())) for (p, s), g in L.groupby(["part", "seeds"])}

# ---------------------------------------------------------------- 2. 単位の判定を集める（台本と別に書いた名前の対応）
preds = {}   # (c, K, v) -> {(f, r): (y, pred, cl)}
for p in sorted((R / "students4").glob("*.npz")):
    T, rest = p.stem.split("_ALL_")
    f, r = int(rest.split("_r")[0][1:]), int(rest.split("_r")[1])
    z, z20 = np.load(p), np.load(OUT20 / "units" / p.name)
    y, cl = z["y_test"].astype(int), z["cl_test"]
    for K in KS:
        q = fz["four"][T][str(K)]
        i = int(f20[T][str(K)]["grid"])
        gm, gs, gps = q["student_main"], q["student_prop_sel"], q["student_prod_sel"]
        m = {"stud_prop": z[f"pred_K{K}_stud_prop_g{gm}"], "stud_prod": z[f"pred_K{K}_stud_prod_g{gm}"],
             "stud_prop_sel": z[f"pred_K{K}_stud_prop_g{gs}"], "stud_prod_sel": z[f"pred_K{K}_stud_prod_g{gps}"]}
        m.update({f"stud_{t}": z[f"pred_K{K}_stud_{t}_g{gm}"] for t in EXTRA_T})
        m.update({f"{a}_{t}": z[f"pred_K{K}_{a}_{t}"] for a in EX1 for t in ("prod", "prop")})
        m.update({"prop": z20[f"pred_adj_K{K}_M_F6_mix{i}"], "prod": z20[f"pred_adj_K{K}_prod"]})
        m.update({v: z20[f"pred_raw_K{K}_{v}"] for v in ("none_AVT", "joint_AVT") + B_VERS})
        for v, a in m.items():
            preds.setdefault((T, K, v), {})[(f, r)] = (y, a.astype(int), cl)
for p in sorted((R / "cremad").glob("*.npz")):
    rest = p.stem.split("_ALL_")[1]
    f, r = int(rest.split("_r")[0][1:]), int(rest.split("_r")[1])
    z = np.load(p)
    y, cl = z["y_test"].astype(int), z["cl_test"]
    for K in KS:
        q = fz["cremad"][str(K)]
        i, gm, gs = q["prop_grid"], q["student_main"], q["student_prop_sel"]
        m = {v: z[f"pred_adj_K{K}_{v}"] for v in A2 + ("M_F6_orc", "M_F6_ext")}
        m.update({f"{v}@raw": z[f"pred_raw_K{K}_{v}"] for v in A2})
        m.update({"prop": z[f"pred_adj_K{K}_M_F6_mix{i}"], "prop@raw": z[f"pred_raw_K{K}_M_F6_mix{i}"],
                  "prop_noprof": z[f"pred_adj_K{K}_M_F6_mix{P20.same_but(i, beta=0.0)}"], "prop_noprof_sel": z[f"pred_adj_K{K}_M_F6_mix{q['noprof_sel']}"],
                  "prop_noint": z[f"pred_adj_K{K}_M_F6n_mix{i}"], "prop_n0zero": z[f"pred_adj_K{K}_M_F6_mix{P20.same_but(i, n0=0.0)}"],
                  "prop_home": z[f"pred_adj_K{K}_M_F6h_mix{i}"], "prop_levelfree": z[f"pred_adj_K{K}_M_F6_mix{GRID.index((float('inf'), 1.0, 'eq'))}"],
                  "stud_prod": z[f"pred_K{K}_stud_prod_g{gm}"], "stud_prop": z[f"pred_K{K}_stud_prop_g{gm}"], "stud_prop_sel": z[f"pred_K{K}_stud_prop_g{gs}"]})
        m.update({f"stud_{t}": z[f"pred_K{K}_stud_{t}_g{gm}"] for t in EXTRA_T})
        m.update({f"none_{k}": z[f"pred_raw_K{K}_none_{k}"] for k in ("A", "V", "AV")})
        m.update({"ret_A": z[f"pred_raw_K{K}_ret_A"], "ret_V": z[f"pred_raw_K{K}_ret_V"]})
        m.update({f"full_{k}": z[f"pred_raw_K0_full_{k}"] for k in ("A", "V", "AV")})
        m.update({f"start_{k}": z[f"pred_raw_K0_start_{k}"] for k in ("A", "V", "AV")})
        for v, a in m.items():
            preds.setdefault((CD, K, v), {})[(f, r)] = (y, a.astype(int), cl)
part_of = lambda c: "cremad" if c == CD else "4corpora"  # noqa: E731
Lall = L[L.seeds == "all"]
out["cells"] = dict(n_mine=len(preds), n_levels_all=len(Lall),
                    same_set=set(preds) == {(r.c, int(r.K), r.version) for r in Lall.itertuples()})

# ---------------------------------------------------------------- 3. same_pred・same_as の作り直し（全部の版）
sp_mine, sa_mine = {}, {}
for c, K in sorted({(c, K) for c, K, _ in preds}):
    vs = sorted(v for (c2, K2, v) in preds if (c2, K2) == (c, K))
    units = sorted(preds[(c, K, vs[0])])
    full = {v: np.concatenate([preds[(c, K, v)][u][1] for u in units]) for v in vs}
    cnt = {}
    for v in vs:
        h = []
        for u in units:
            y, p_, cl = preds[(c, K, v)][u]
            names, inv = np.unique(cl, return_inverse=True)
            cor = np.zeros((len(names), 4)); np.add.at(cor, (inv[p_ == y], y[p_ == y]), 1)
            h.append(cor.tobytes())
        cnt[v] = hashlib.sha256(b"".join(h)).hexdigest()
    for v in vs:
        grp = [w for w in vs if np.array_equal(full[w], full[v])]
        sp_mine[(c, K, v)] = set(grp) if len(grp) > 1 else set()
        grp2 = [w for w in vs if cnt[w] == cnt[v]]
        sa_mine[(c, K, v)] = set(grp2) if len(grp2) > 1 else set()
bad_sp, bad_sa, n_sp, n_sa = 0, 0, 0, 0
for r in Lall.itertuples():
    k = (r.c, int(r.K), r.version)
    s_ = "" if pd.isna(r.same_pred) else str(r.same_pred)
    a_ = "" if pd.isna(r.same_as) else str(r.same_as)
    n_sp += bool(s_); n_sa += bool(a_)
    if (s_ == "") != (len(sp_mine[k]) == 0) or (s_ and s_ not in sp_mine[k]):
        bad_sp += 1
    if (a_ == "") != (len(sa_mine[k]) == 0) or (a_ and a_ not in sa_mine[k]):
        bad_sa += 1
# 種で分けた行も同じ印か
Ls = L[L.seeds != "all"].merge(Lall[["part", "c", "K", "version", "same_pred", "same_as"]], on=["part", "c", "K", "version"], suffixes=("", "_all"))
seed_rows_mark_diff = int(((Ls.same_pred.fillna("") != Ls.same_pred_all.fillna("")) | (Ls.same_as.fillna("") != Ls.same_as_all.fillna(""))).sum())
ebad = 0
for r in E.itertuples():
    eq = np.array_equal(np.concatenate([preds[(r.c, int(r.K), r.a)][u][1] for u in sorted(preds[(r.c, int(r.K), r.a)])]),
                        np.concatenate([preds[(r.c, int(r.K), r.b)][u][1] for u in sorted(preds[(r.c, int(r.K), r.b)])]))
    ebad += bool(eq) != bool(r.same_pred)
# same_as が付き same_pred が付かない版（正解の数は同じで判定が違う）
sa_not_sp = [f"{c}|{K}|{v}" for (c, K, v) in sa_mine if sa_mine[(c, K, v)] and not sp_mine[(c, K, v)]]
out["marks"] = dict(levels_same_pred_mismatch=bad_sp, levels_same_as_mismatch=bad_sa, n_levels_with_same_pred=n_sp, n_levels_with_same_as=n_sa,
                    seed_rows_mark_differs_from_all=seed_rows_mark_diff, effects_same_pred_mismatch=ebad, n_effects_same_pred_true=int(E.same_pred.astype(bool).sum()),
                    n_same_as_without_same_pred=len(sa_not_sp), same_as_without_same_pred=sa_not_sp,
                    effects_same_pred_true_pairs=sorted({f"{r.part}|{r.a}-{r.b}" for r in E[E.same_pred.astype(bool) & (E.seeds == "all")].itertuples()}))

# ---------------------------------------------------------------- 4. 点の推定の独立な作り直し（全部の行）
def theta(cell_a, cell_b, seeds):
    us = sorted(u for u in cell_a if seeds is None or u[1] in seeds)
    folds = sorted({f for f, _ in us})
    d = {f: [uar(*cell_a[u][:2]) - (uar(*cell_b[u][:2]) if cell_b is not None else 0.0) for u in us if u[0] == f] for f in folds}
    return float(np.mean([np.mean(v) for v in d.values()])), len(us)


cache = {}
lv_dev, lv_nfr_bad = 0.0, 0
for r in L.itertuples():
    est, n = theta(preds[(r.c, int(r.K), r.version)], None, SEEDS[r.seeds])
    lv_dev = max(lv_dev, abs(est - r.uar)); lv_nfr_bad += n != r.n_fr
ef_dev, ef_nfr_bad = 0.0, 0
for r in E.itertuples():
    est, n = theta(preds[(r.c, int(r.K), r.a)], preds[(r.c, int(r.K), r.b)], SEEDS[r.seeds])
    ef_dev = max(ef_dev, abs(est - r.est)); ef_nfr_bad += n != r.n_fr
out["recompute_points"] = dict(levels_rows=len(L), levels_max_abs_dev=lv_dev, levels_nfr_mismatch=lv_nfr_bad,
                               effects_rows=len(E), effects_max_abs_dev=ef_dev, effects_nfr_mismatch=ef_nfr_bad)

# ---------------------------------------------------------------- 5. 区間の作り直し（ST.effect_stats，同じ鍵．標本の行）
geff = {c: ST.g_eff_of(c)[0] for c in C4} | {CD: P18.g_eff_of(CD)}


def cells_of(c, K, v, seeds):
    return {u: ST.counts(y, p_, cl) for u, (y, p_, cl) in preds[(c, K, v)].items() if seeds is None or u[1] in seeds}


def zero(A):
    return {k: (x[0], np.zeros_like(x[1]), x[2]) for k, x in A.items()}


rng0 = np.random.default_rng(7)
iv_dev, n_iv = 0.0, 0
pickL = pd.concat([g.sample(min(len(g), 8), random_state=1) for _, g in L.groupby(["part", "seeds"])])
for r in pickL.itertuples():
    A = cells_of(r.c, int(r.K), r.version, SEEDS[r.seeds])
    es = ST.effect_stats(A, zero(A), geff[r.c], P16.row_rng("dai21", "L", r.part, r.c, int(r.K), r.version, r.seeds))
    iv_dev = max(iv_dev, abs(es["lo"] - r.lo), abs(es["hi"] - r.hi), abs(es["se"] - r.se)); n_iv += 1
pickE = pd.concat([g.sample(min(len(g), 8), random_state=2) for _, g in E.groupby(["part", "seeds"])])
for r in pickE.itertuples():
    A, B = cells_of(r.c, int(r.K), r.a, SEEDS[r.seeds]), cells_of(r.c, int(r.K), r.b, SEEDS[r.seeds])
    es = ST.effect_stats(A, B, geff[r.c], P16.row_rng("dai21", "A", r.part, r.a, r.b, r.c, int(r.K), r.seeds))
    iv_dev = max(iv_dev, abs(es["lo"] - r.lo), abs(es["hi"] - r.hi), abs(es["se"] - r.se)); n_iv += 1
# 感情ごとの再現率の区間（標本）
PE = tab["per_emotion.csv"]
pe_dev, n_pe = 0.0, 0
for r in pd.concat([g.sample(6, random_state=3) for _, g in PE.groupby("part")]).itertuples():
    A = cells_of(r.c, int(r.K), r.version, None)
    m = np.zeros(4); m[int(r.emotion)] = 1.0
    Ai = {k: (x[0], x[1] * m, x[2]) for k, x in A.items()}
    es = ST.effect_stats(Ai, zero(A), geff[r.c], P16.row_rng("dai21", "E", r.part, r.c, int(r.K), r.version, int(r.emotion)))
    pe_dev = max(pe_dev, abs(4 * es["effect"] - r.recall), abs(4 * es["lo"] - r.lo), abs(4 * es["hi"] - r.hi)); n_pe += 1
out["recompute_intervals"] = dict(n_rows=n_iv, max_abs_dev_lo_hi_se=iv_dev, per_emotion_rows=n_pe, per_emotion_max_abs_dev=pe_dev)

# ---------------------------------------------------------------- 6. 感情ごとの再現率（全部の単位の合計）と混同行列の作り直し
rp_dev = 0.0
for r in PE.itertuples():
    cor = tot = 0.0
    for y, p_, _ in preds[(r.c, int(r.K), r.version)].values():
        tot += (y == r.emotion).sum(); cor += ((y == r.emotion) & (p_ == r.emotion)).sum()
    rp_dev = max(rp_dev, abs(100 * cor / max(tot, 1) - r.recall_pooled), abs(tot - r.n))
CF = tab["confusion.csv"]
cf_dev = 0.0
for (c, K, v), g in CF.groupby(["c", "K", "version"]):
    M = np.zeros((4, 4))
    for y, p_, _ in preds[(c, int(K), v)].values():
        np.add.at(M, (y, p_), 1)
    cf_dev = max(cf_dev, float(np.abs(M[g.true.to_numpy(), g.pred.to_numpy()] - g.n.to_numpy()).max()))
out["per_emotion_confusion"] = dict(per_emotion_rows=len(PE), recall_pooled_and_n_max_abs_dev=rp_dev, confusion_cells=int(CF[["c", "K", "version"]].drop_duplicates().shape[0]),
                                    confusion_versions=sorted(CF.version.unique()), confusion_max_abs_dev=cf_dev)

# ---------------------------------------------------------------- 7. 格子（student_grid）・CREMA-D の格子の水準・重みの表
SG = tab["student_grid.csv"]
uc = []
for sub in ("students4", "cremad"):
    for p in sorted((R / sub).glob("*.csv")):
        d = pd.read_csv(p, low_memory=False)
        d = d[d.version.astype(str).str.startswith("stud_")].assign(c=p.stem.split("_ALL_")[0])
        uc.append(d[["c", "K", "version", "grid", "uar", "mimic", "teacher_uar"]])
uc = pd.concat(uc)
g2 = uc.groupby(["c", "K", "version", "grid"]).agg(uar_mean=("uar", "mean"), uar_count=("uar", "count"), mimic_mean=("mimic", "mean"), tu=("teacher_uar", "mean")).reset_index()
mm = g2.merge(SG, on=["c", "K", "version", "grid"], how="outer", indicator=True)
sg_dev = float(np.nanmax(np.abs(np.r_[mm.uar_mean_x - mm.uar_mean_y, mm.mimic_mean_x - mm.mimic_mean_y, mm.tu - mm.teacher_uar_mean])))
out["student_grid"] = dict(rows=len(SG), merge_only_one_side=int((mm._merge != "both").sum()), max_abs_dev=sg_dev, counts=sorted(set(SG.uar_count.tolist())))
# 学び直しの格子の UAR を判定の配列から独立に（標本：主の点と 12 点の一部）
gd = 0.0
for p in sorted((R / "students4").glob("MELD_ALL_f2_r*.npz"))[:3] + sorted((R / "cremad").glob("*_f3_r*.npz"))[:3]:
    z = np.load(p); c_ = pd.read_csv(p.with_suffix(".csv"), low_memory=False)
    y = z["y_test"].astype(int)
    for K in KS:
        for it in z[f"stud_items_K{K}"]:
            it = str(it)
            if not it.startswith("stud_"):
                continue
            t_, g_ = it[5:].rsplit("_g", 1)
            row = c_[(c_.K == K) & (c_.version == f"stud_{t_}") & (c_.grid == int(g_))]
            gd = max(gd, abs(uar(y, z[f"pred_K{K}_{it}"].astype(int)) - float(row.uar.iloc[0])))
out["student_grid"]["unit_uar_from_pred_max_abs_dev"] = gd
GL = tab["grid_levels_cremad.csv"]
gl_dev, gl_frozen_bad = 0.0, 0
for r in GL.itertuples():
    vals = [uar(y, p_) for (y, p_, _) in [(np.load(p)["y_test"].astype(int), np.load(p)[f"pred_adj_K{r.K}_M_F6_mix{r.grid}"].astype(int), None)
                                           for p in sorted((R / "cremad").glob("*.npz"))]]
    gl_dev = max(gl_dev, abs(np.mean(vals) - r.uar_adj))
    gl_frozen_bad += bool(r.frozen) != (int(fz["cremad"][str(int(r.K))]["prop_grid"]) == int(r.grid))
out["grid_levels_cremad"] = dict(rows=len(GL), max_abs_dev_from_pred=gl_dev, frozen_flag_bad=gl_frozen_bad, n_frozen_true=int(GL.frozen.sum()))
WC = tab["weights_cremad.csv"]
wsum = {K: [] for K in KS}
for p in sorted((R / "cremad").glob("*.npz")):
    z = np.load(p)
    for K in KS:
        Dk, Do, De = z[f"D_M_K_K{K}"], z[f"D_M_orc_K{K}"], z["D_M_ext"]
        wsum[K].append(np.stack([P16.to_w(P16.mix_rel(Dk, De, 4 * K, *GRID[fz["cremad"][str(K)]["prop_grid"]])), P16.to_w(Do), P16.to_w(Dk), P16.to_w(De)]))
wd = 0.0
for r in WC.itertuples():
    W = np.mean(wsum[int(r.K)], 0); j = "AV".index(r.mod)
    wd = max(wd, abs(W[0, j, r.emotion] - r.w_prop), abs(W[1, j, r.emotion] - r.w_orc), abs(W[2, j, r.emotion] - r.w_K), abs(W[3, j, r.emotion] - r.w_ext))
out["weights_cremad"] = dict(rows=len(WC), max_abs_dev=float(wd), n_units=sorted(set(WC.n_units.tolist())))

# ---------------------------------------------------------------- 8. 実験 20 の集計との一致（4 コーパス，seeds = all）
L20 = pd.read_csv(OUT20 / "summary" / "levels.csv")
L20 = L20[L20.seeds == "all"]
m = Lall[Lall.part == "4corpora"].merge(L20, on=["c", "K", "version"], suffixes=("", "_20"))
out["vs_dai20"] = dict(versions=sorted(m.version.unique()), n_rows=len(m),
                       max_abs_dev={c: float(np.abs(m[c] - m[f"{c}_20"]).max()) for c in ("uar", "lo", "hi", "se")},
                       n_fr_equal=bool((m.n_fr == m.n_fr_20).all()), method_equal=bool((m.method == m.method_20).all()))
json.dump(out, sys.stdout, indent=1, ensure_ascii=False, default=str)
print()
