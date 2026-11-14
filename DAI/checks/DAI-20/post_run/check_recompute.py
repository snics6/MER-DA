"""DAI-20 計算の後の点検：保存（npz）からの作り直し．results/dai20_eval は読むだけ．値は画面に出さない（ずれと一致の数だけ）．
1. 全部の単位で，npz の判定から UAR を数え直し，単位の csv の uar_adj・uar_raw と比べる．
2. 全部の単位で，テストの発話の並び（sid_test）が実験 1 の評価の回の保存（p1 の sample_test）と同じか．
3. 数個の (コーパス, K, 版) について，水準と差を npz から作り直す（塊ごとの数は自前で数え，点の値は自前で平均する．区間は dai1_stats.effect_stats に同じ鍵の乱数で渡す）．
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1_stats as ST  # noqa: E402
import dai16_probe as P16  # noqa: E402

R = ROOT / "results" / "dai20_eval"
U, SM = R / "units", R / "summary"
D1 = ROOT / "results" / "dai1" / "eval" / "units"
CORP = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
KS = (5, 10, 25)
GRID = P16.GRID
fz = json.load(open(ROOT / "DAI" / "frozen" / "dai20_frozen.json"))["choice"]
gl = pd.read_csv(SM / "grid_levels.csv")
lv = pd.read_csv(SM / "levels.csv")
ef = pd.read_csv(SM / "effects.csv")
pe = pd.read_csv(SM / "per_emotion.csv")
cfm = pd.read_csv(SM / "confusion.csv")


def uar(y, p):
    return 100.0 * float(np.mean([(p[y == k] == k).mean() for k in range(4) if (y == k).any()]))


# ---- 1・2
nver, dmax, nmis, sidbad = 0, 0.0, 0, []
Z = {}
for p in sorted(U.glob("*.npz")):
    z = np.load(p, allow_pickle=False)
    d = pd.read_csv(p.with_suffix(".csv"), low_memory=False)
    y = z["y_test"].astype(int)
    for r in d.itertuples():
        for lay, col in (("adj", r.uar_adj), ("raw", r.uar_raw)):
            k = f"pred_{lay}_K{r.K}_{r.version}"
            if k in z.files:
                u = uar(y, z[k].astype(int))
                if not np.isfinite(col):
                    nmis += 1
                    continue
                dmax = max(dmax, abs(u - col)); nver += 1
    s1 = np.load(D1 / f"{p.stem}_p1.npz", allow_pickle=False)["sample_test"]
    if not (len(s1) == len(z["sid_test"]) and (s1.astype(str) == z["sid_test"].astype(str)).all()):
        sidbad.append(p.stem)
    if p.stem.split("_")[0] and p.stem.endswith(("_r0", "_r1", "_r2", "_r3", "_r4", "_r5", "_r6", "_r7", "_r8", "_r9")):
        pass
print(f"1. 判定から数え直した UAR と csv の比べ：{nver} 組，ずれの最大 {dmax:.2e}．csv が NaN で npz に判定がある組 {nmis}")
print(f"2. テストの発話の並びが実験 1 の p1 と違う単位：{sidbad}（{len(list(U.glob('*.npz')))} 単位を照合）")


# ---- 3
def pick_of(c, K):
    i, j = int(fz[c][str(K)]["grid"]), int(fz[c][str(K)]["noprof_sel"])
    s = gl[(gl.c == c) & (gl.K == K)]
    a, b, pr = GRID[i]
    return dict(prop=i, prop_noprof=GRID.index((a, 0.0, pr)), prop_n0zero=GRID.index((0.0, b, pr)), prop_noprof_sel=j,
                prop_loto=int(s[s.loto].grid.iloc[0]), oracle_grid=int(s[s.oracle].grid.iloc[0]), prop_levelfree=14)


PLAIN = ("none_AVT", "joint_AVT", "ret_A", "ret_V", "ret_T", "b_kd", "b_agree", "b_mix", "b_cotrain", "b_coreg")


def key_of(c, K, v):
    pk = pick_of(c, K)
    if v in pk:
        return f"pred_adj_K{K}_M_F6_mix{pk[v]}"
    if v == "prop@raw":
        return f"pred_raw_K{K}_M_F6_mix{pk['prop']}"
    if v == "prop_noint":
        return f"pred_adj_K{K}_M_F6n_mix{pk['prop']}"
    if v == "prop_home":
        return f"pred_adj_K{K}_M_F6h_mix{pk['prop']}"
    if v in PLAIN:
        return f"pred_raw_K{K}_{v}"
    if v.endswith("@raw"):
        return f"pred_raw_K{K}_{v[:-4]}"
    return f"pred_adj_K{K}_{v}"


def cells_of(c, K, v):
    out = {}
    for f in range(5):
        for r in range(10):
            z = np.load(U / f"{c}_ALL_f{f}_r{r}.npz", allow_pickle=False)
            y, pr, cl = z["y_test"].astype(int), z[key_of(c, K, v)].astype(int), z["cl_test"]
            names = np.array(sorted(set(cl.tolist())))
            cor, tot = np.zeros((len(names), 4)), np.zeros((len(names), 4))
            pos = {n: i for i, n in enumerate(names)}
            for yy, pp, cc in zip(y, pr, cl):
                tot[pos[cc], yy] += 1
                cor[pos[cc], yy] += pp == yy
            out[(f, r)] = (names, cor, tot)
    return out


def point(A, B=None):
    def u(x):
        c, t = x[1].sum(0), x[2].sum(0)
        return 100.0 * np.mean(c[t > 0] / t[t > 0])
    d = {}
    for (f, r), x in A.items():
        d.setdefault(f, []).append(u(x) - (u(B[(f, r)]) if B is not None else 0.0))
    return float(np.mean([np.mean(v) for v in d.values()]))


def sub(A, seeds):
    return A if seeds is None else {k: x for k, x in A.items() if k[1] in seeds}


SEEDS = {"all": None, "0-1": (0, 1), "2-9": tuple(range(2, 10)), "0-2": (0, 1, 2)}
geff = {c: ST.g_eff_of(c)[0] for c in CORP}
zero = lambda A: {k: (x[0], np.zeros_like(x[1]), x[2]) for k, x in A.items()}  # noqa: E731
cache = {}


def cell(c, K, v):
    if (c, K, v) not in cache:
        cache[(c, K, v)] = cells_of(c, K, v)
    return cache[(c, K, v)]


worst = dict(lev_pt=0.0, lev_ci=0.0, eff_pt=0.0, eff_ci=0.0, emo=0.0, conf=0.0)
nlev = neff = 0
lev_jobs = [("AFEW", 5, "prop", "all"), ("AFEW", 5, "prod", "0-2"), ("IEMOCAP", 10, "wf100", "all"), ("MOSEI", 25, "prop", "all"), ("MOSEI", 25, "b_kd", "all"),
            ("MELD", 5, "prop_noprof", "all"), ("MELD", 10, "prop_loto", "all"), ("IEMOCAP", 25, "tcal@raw", "all")]
for c, K, v, sn in lev_jobs:
    A = sub(cell(c, K, v), SEEDS[sn])
    es = ST.effect_stats(A, zero(A), geff[c], P16.row_rng("dai20", "L", c, K, v, sn))
    row = lv[(lv.c == c) & (lv.K == K) & (lv.version == v) & (lv.seeds == sn)].iloc[0]
    worst["lev_pt"] = max(worst["lev_pt"], abs(point(A) - row.uar), abs(es["effect"] - row.uar))
    worst["lev_ci"] = max(worst["lev_ci"], abs(es["lo"] - row.lo), abs(es["hi"] - row.hi), abs(es["se"] - row.se))
    nlev += 1
eff_jobs = [("prop", "prod", "AFEW", 5, "all"), ("prop", "prod", "AFEW", 5, "0-1"), ("prop", "prod", "MOSEI", 25, "2-9"), ("prop", "wf100", "IEMOCAP", 10, "all"),
            ("prop", "b_kd", "MOSEI", 25, "all"), ("prop", "prop_noprof", "MELD", 5, "all"), ("prop_loto", "prop", "MELD", 10, "all"),
            ("prop@raw", "tcal@raw", "IEMOCAP", 25, "all"), ("prop_levelfree", "prod", "MELD", 25, "all"), ("prop_noint", "prop_noprof", "AFEW", 25, "all"),
            ("prop", "joint_AVT", "MELD", 10, "0-1")]
for a, b, c, K, sn in eff_jobs:
    A, B = sub(cell(c, K, a), SEEDS[sn]), sub(cell(c, K, b), SEEDS[sn])
    es = ST.effect_stats(A, B, geff[c], P16.row_rng("dai20", "A", a, b, c, K, sn))
    row = ef[(ef.a == a) & (ef.b == b) & (ef.c == c) & (ef.K == K) & (ef.seeds == sn)].iloc[0]
    worst["eff_pt"] = max(worst["eff_pt"], abs(point(A, B) - row.est), abs(es["effect"] - row.est))
    worst["eff_ci"] = max(worst["eff_ci"], abs(es["lo"] - row.lo), abs(es["hi"] - row.hi), abs(es["se"] - row.se))
    same = all((A[k][1] == B[k][1]).all() for k in A)
    if bool(same) != bool(row.same_counts):
        print("same_counts の食い違い", a, b, c, K, sn)
    neff += 1
# 感情ごとの再現率（全体で合わせた値と区間）
for c, K, v in (("AFEW", 5, "prop"), ("MOSEI", 10, "prod")):
    A = cell(c, K, v)
    for i in range(4):
        row = pe[(pe.c == c) & (pe.K == K) & (pe.version == v) & (pe.emotion == i)].iloc[0]
        cor = sum(x[1].sum(0) for x in A.values()); tot = sum(x[2].sum(0) for x in A.values())
        m = np.zeros(4); m[i] = 1
        es = ST.effect_stats({k: (x[0], x[1] * m, x[2]) for k, x in A.items()}, zero(A), geff[c], P16.row_rng("dai20", "E", c, K, v, i))
        worst["emo"] = max(worst["emo"], abs(100 * cor[i] / tot[i] - row.recall_pooled), abs(tot[i] - row.n), abs(4 * es["effect"] - row.recall), abs(4 * es["lo"] - row.lo))
# 混同行列
for c, K, v in (("AFEW", 5, "prop"), ("MELD", 25, "prod")):
    M = np.zeros((4, 4))
    for f in range(5):
        for r in range(10):
            z = np.load(U / f"{c}_ALL_f{f}_r{r}.npz", allow_pickle=False)
            np.add.at(M, (z["y_test"].astype(int), z[key_of(c, K, v)].astype(int)), 1)
    s = cfm[(cfm.c == c) & (cfm.K == K) & (cfm.version == v)]
    worst["conf"] = max(worst["conf"], float(max(abs(M[t, q] - n) for t, q, n in zip(s.true, s.pred, s.n))))
print(f"3. 水準 {nlev} 個，差 {neff} 個を作り直した．ずれの最大：{ {k: f'{v_:.2e}' for k, v_ in worst.items()} }")
# 直接の判定の一致（同じ数の印の確かめ）
for c, K, a, b in (("AFEW", 10, "prop_noprof", "M_F2_K"), ("IEMOCAP", 5, "prop_n0zero", "prop"), ("AFEW", 5, "psel", "prod"), ("MELD", 5, "prop_noprof", "M_F2_K")):
    eq = np.mean([(np.load(U / f"{c}_ALL_f{f}_r{r}.npz")[key_of(c, K, a)] == np.load(U / f"{c}_ALL_f{f}_r{r}.npz")[key_of(c, K, b)]).mean() for f in range(5) for r in range(10)])
    sa = lv[(lv.c == c) & (lv.K == K) & (lv.seeds == "all")].set_index("version").same_as.fillna("")
    print(f"  {c} K={K} {a} と {b}：判定が同じ発話の割合 {eq:.4f}，levels の same_as の群が同じ {(sa[a] or a) == (sa[b] or b)}")
