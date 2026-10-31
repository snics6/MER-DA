"""gate-code（E79）：judge の units.csv の点の値と区間を，台本の関数を使わずに作り直して照合する。また選択を素朴に作り直す。

使い方：python verify_point.py <E79 の out> <run> <E77 の場所> <E78 の場所>
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, load_any  # noqa: E402
from explore_pairs import ba  # noqa: E402
from explore_synergy import d_o  # noqa: E402

out, run, d77, d78 = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
U = pd.read_csv(out / "judge" / "units.csv", dtype={"K": str})
sel = pd.read_csv(out / "select" / "chosen.csv", dtype={"K": str, "param": str, "meff": str})
keys = pd.read_csv(out / "measure" / f"keys_{run}.csv", dtype={"K": str, "param": str, "meff": str})
Dt = pd.read_csv(out / "measure" / f"D_{run}.csv")
SEEDED = {"dann", "mme", "mapctl", "xmuda", "cot", "cotctl", "cotfix", "cotzero"}


def point(T, st, K, wave, meth, pn, meff, md):
    t = pd.read_csv((d77 if wave == "e77" else d78) / f"{T}.csv", dtype={"K": str, "param": str, "meff": str})
    pnorm = t.param.str.replace(r"s\d+$", "", regex=True) if meth in SEEDED else t.param
    x = t[(t.student == st) & (t.K == K) & (t.method == meth) & (t.meff == meff) & (pnorm == pn)]
    if md == "s0":
        x = x[x.param.str.endswith("s0")]
    df, _ = load_any(T, run)
    yall = df.label_4.to_numpy().astype(int)
    us, ds = [], []
    for r, g in x.groupby("rep"):
        te = np.load((d77 if wave == "e77" else d78) / f"{T}_r{r}_te.npy")
        y = yall[te]
        L = np.load((d77 if wave == "e77" else d78) / f"{T}_r{r}_logp.npy", mmap_mode="r")
        D = Dt[(Dt.target == T) & (Dt.student == st) & (Dt.rep == r)].sort_values("pair").D.to_numpy()
        uu, dd = [], []
        for i in g.row_id:
            Li = np.asarray(L[i], float)
            pred = Li.argmax(1)
            uu.append(100 * np.mean([(pred[y == k] == k).mean() for k in range(4)]))
            dd.append(100 * np.mean([ba(D[p], 0) - ba(dv, 0) for p, (dv, _) in enumerate(d_o(Li, y))]))
        us.append(np.mean(uu))
        ds.append(np.mean(dd))
    return np.mean(us), np.mean(ds), len(x)


def ci(T, kx, ky, metric):
    z = np.load(out / "measure" / f"boot_{run}_{T}.npz")
    kk = keys[keys.target == T].set_index(["student", "K", "wave", "method", "param", "meff", "seedmode"]).key_id
    a = z["uar" if metric == "uar" else "dirloss"][:, kk[kx]]
    b = z["uar" if metric == "uar" else "dirloss"][:, kk[ky]]
    return (a - b) if metric == "uar" else (b - a)


worst = 0.0
for cmp_, fam in (("none-dann", "P4"), ("none-entmin", "P4"), ("none-coral", "P4"), ("none-entmin", "P4'"), ("oursx-kd", "P5-1"),
                  ("none-oursx", "P5-3"), ("oursx-cot", "P5-4")):
    x = U[(U.run == run) & (U.family == fam) & (U.comparison == cmp_) & (U.sensitivity == "main") & (U.K == "5")]
    for r in x.itertuples():
        Ts = [t for t in GROUP_SPEC if GROUP_SPEC[t][0] == r.corpus and (out / "measure" / f"boot_{run}_{t}.npz").exists()]
        pts, reps = [], []
        if not Ts:
            continue
        meth = cmp_.split("-")[1] if cmp_.startswith("none") else "oursx"
        for T in Ts:
            vv = "excl" if fam.startswith("P4") else "incl"
            s = sel[(sel.run == run) & (sel.target == T) & (sel.student == r.student) & (sel.K == "5") & (sel.selector == meth) & (sel.variant == vv)].iloc[0]
            md = "agg" if s.method in SEEDED else "-"
            kx = (r.student, "5", s.wave, s.method, s.param, s.meff, md if s.method != "none" else "-")
            if cmp_.startswith("none"):
                ky = (r.student, "5", "e77", "none", "-", "0", "-")
                ux, dx, nx = point(T, *kx)
                uy, dy, ny = point(T, *ky)
                pts.append(dy - dx)
                reps.append(ci(T, kx, ky, "dir"))
            else:
                other = {"P5-1": "kd", "P5-4": "cot"}[fam]
                s2 = sel[(sel.run == run) & (sel.target == T) & (sel.student == r.student) & (sel.K == "5") & (sel.selector == other) & (sel.variant == "incl")].iloc[0]
                md2 = "agg" if s2.method in SEEDED else "-"
                ky = (r.student, "5", s2.wave, s2.method, s2.param, s2.meff, md2 if s2.method != "none" else "-")
                ux, dx, _ = point(T, *kx)
                uy, dy, _ = point(T, *ky)
                pts.append(ux - uy)
                reps.append(ci(T, kx, ky, "uar"))
        p = np.mean(pts)
        b = np.mean(np.stack(reps), 0)
        b = b[np.isfinite(b)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        e = max(abs(p - r.point), abs(lo - r.lo), abs(hi - r.hi))
        worst = max(worst, e)
        print(fam, cmp_, r.student, r.corpus, f"点 {p:.6f} vs {r.point:.6f}  区間の差 {abs(lo - r.lo):.2e} {abs(hi - r.hi):.2e}")
print("最大の差", worst)

# 選択を素朴に作り直す（none を除く版．E77 の方法の族）
P = pd.concat([pd.read_csv(d / f"{T}_pseudo.csv", dtype={"K": str, "param": str, "meff": str}).assign(wave=w, T=T)
               for w, d in (("e77", d77), ("e78", d78)) for T in sorted({f.name[:-11] for f in d.glob("*_pseudo.csv")})])
bad = 0
n = 0
for (T, st, K), g in P.groupby(["T", "student", "K"]):
    for meth in ("entmin", "self", "t3a", "dann", "mme"):
        gm = g[g.method == meth]
        if gm.empty:
            continue
        pn = gm.param.str.replace(r"s\d+$", "", regex=True) if meth in SEEDED else gm.param
        sc = gm.assign(p=pn + "|" + gm.meff).groupby("p").uar.mean()
        best = sc[sc == sc.max()].index.sort_values()[0]
        s = sel[(sel.run == run) & (sel.target == T) & (sel.student == st) & (sel.K == K) & (sel.selector == meth) & (sel.variant == "excl")]
        n += 1
        if len(s) != 1 or f"{s.param.iloc[0]}|{s.meff.iloc[0]}" != best:
            bad += 1
            print("選択が違う", T, st, K, meth, best, s[["param", "meff"]].to_dict("records"))
print("選択の照合：", n, "件のうち違う", bad)
