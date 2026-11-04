"""gate-code（DAI-1 code_v5）：MOSEI をテストにした区間（塊の引き直しと種の項）を，設計 9.3 節の文から書き直した手続きで作り直し，
試走の出力（T = AFEW，c = MOSEI）の se と比べる．乱数は別なので，一致は乱数の誤差の範囲（B = 2,000 で分散の相対誤差は約 3%）で見る。
成績の値は画面に出さず，se の比と θ の差だけを出す。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/snics/MER-DA")
TRIAL = Path(sys.argv[1])
T, c = "AFEW", "MOSEI"
UN = ROOT / "results/dai1/pseudo/units"
FZ = json.load(open(ROOT / "DAI/frozen/dai1_frozen.json"))[T]
lab = pd.read_csv(ROOT / f"DAI/splits/{c}.csv", keep_default_na=False, dtype={"cluster": str}).set_index("sample_id")
rng = np.random.default_rng(12345)


def nm(v):
    f = float(v); return str(int(f)) if f.is_integer() else repr(f)


units = {}
for f in range(5):
    for r in (0, 1):
        z1, z3 = np.load(UN / f"{T}_{c}_f{f}_r{r}_p1.npz"), np.load(UN / f"{T}_{c}_f{f}_r{r}_p3.npz")
        sid = z1["sample_test"].astype(str)
        units[(f, r)] = (z1, z3, lab.loc[sid].label_4.to_numpy().astype(int), lab.loc[sid].cluster.to_numpy())


def pred(u, s, K, kind):
    z1, z3, y, cl = u
    if kind == "none":
        cw, cb = FZ["tether"][f"{s}|{K}"]
        it = [str(x) for x in z1[f"p1_{s}_items"]]
        return z1[f"p1_{s}_pred"][it.index(f"{K}|{cw}|{cb}")].astype(int)
    v = FZ["main"][str(K)]; j = FZ["select"][f"{s}|{K}|joint|{v}"]
    items = [json.loads(x) for x in z3[f"p3_{s}_K{K}_items"]]
    h = [i for i, it in enumerate(items) if it["method"] == "joint" and it["teacher"] == v and nm(it["rho"]) == nm(j["rho"]) and nm(it["a"]) == nm(j["a"])]
    return z3[f"p3_{s}_K{K}_pred"][h[0]].astype(int)


def uar_w(y, p, w):
    """発話の重み w（引き直しの回数）での UAR。感情が欠ければ NaN，30 本未満でも NaN。"""
    tot = np.array([w[y == k].sum() for k in range(4)]); cor = np.array([w[(y == k) & (p == k)].sum() for k in range(4)])
    if tot.sum() < 30 or (tot == 0).any():
        return np.nan
    return 100 * np.mean(cor / tot)


def boot(s, K, mode, B=2000):
    ths = np.zeros(B); nval = np.zeros(B)
    d_fr = {f: [] for f in range(5)}
    for f in range(5):
        cl = units[(f, 0)][3]; names = np.unique(cl); idx = np.searchsorted(names, cl)
        draws = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=B)   # (B, G)
        per_seed = []
        for r in (0, 1):
            u = units[(f, r)]; y = u[2]
            pa = pred(u, s, K, "prop" if mode == "diff" else "none")
            pb = pred(u, s, K, "none") if mode == "diff" else None
            d_fr[f].append(uar_w(y, pa, np.ones(len(y))) - (uar_w(y, pb, np.ones(len(y))) if pb is not None else 0))
            vals = []
            for b in range(B):
                w = draws[b][idx].astype(float)
                vals.append(uar_w(y, pa, w) - (uar_w(y, pb, w) if pb is not None else 0))
            per_seed.append(vals)
        dm = np.nanmean(np.array(per_seed), 0)
        ok = np.isfinite(dm); ths[ok] += dm[ok]; nval[ok] += 1
    ths = ths / np.maximum(nval, 1); ths = ths[nval > 0]
    theta = np.mean([np.mean(v) for v in d_fr.values()])
    vseed = sum(np.var(v, ddof=1) / 2 for v in d_fr.values()) / 25
    return theta, np.sqrt(np.var(ths, ddof=1) + vseed)


L = pd.read_csv(TRIAL / "levels.csv"); L = L[(L["T"] == T) & (L.c == c)]
O = pd.read_csv(TRIAL / "outside_families.csv"); O = O[(O["T"] == T) & (O.c == c)]
for s in ("A", "AVT"):
    for K in (1, 10):
        th, se = boot(s, K, "level", B=400)
        row = L[(L.student == s) & (L.K == K) & (L.row == "none")].iloc[0]
        print(f"水準 none {s} K={K}: θ の差 {abs(th - row.effect):.2e}，se の比 {se / row.se:.3f}，区間の幅 = 2×1.96×se か {abs((row.hi - row.lo) - 2 * 1.96 * row.se):.1e}", flush=True)
        th, se = boot(s, K, "diff", B=400)
        row = O[(O.student == s) & (O.K == K) & (O.family == 1)].iloc[0]
        print(f"族 1 型の差 {s} K={K}: θ の差 {abs(th - row.effect):.2e}，se の比 {se / row.se:.3f}", flush=True)
