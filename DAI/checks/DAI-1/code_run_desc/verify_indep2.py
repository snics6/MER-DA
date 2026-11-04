"""gate-code（DAI-1 code_v5）：dai1_desc.py の出力の一部を，台本の関数を使わずに保存の npz と DAI/splits のラベルから作り直す。
使い方：python verify_indep.py <desc の出力の置き場> <T>（pseudo の段の出力）
区間は設計 9.3 節の文から書き直した jackknife（G_eff < 20 のコーパスだけ）。成績の値は画面に出さず，差の最大だけを出す。"""
import json, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st

ROOT = Path("/home/snics/MER-DA")
DEST = Path(sys.argv[1]); T = sys.argv[2]
UNITS = ROOT / "results/dai1/pseudo/units"
FZ = json.load(open(ROOT / "DAI/frozen/dai1_frozen.json"))[T]
RHO = (0.0625, 0.25, 1.0, 4.0); A = (1, 4, 16); COREG = (0.0, 0.0625, 0.25, 1.0, 4.0)
TETHER = [(cw, cb) for cw in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0) for cb in (1.0, 10.0)]


def nm(v):
    f = float(v); return str(int(f)) if f.is_integer() else repr(f)


def uar(y, p):
    ks = [k for k in range(4) if (y == k).any()]
    return 100.0 * np.mean([(p[y == k] == k).mean() for k in ks])


def rec(y, p, k):
    return 100.0 * (p[y == k] == k).mean()


class U:
    def __init__(s, c, f, r, lab):
        b = UNITS / f"{T}_{c}_f{f}_r{r}"
        s.p1, s.p3 = np.load(f"{b}_p1.npz"), np.load(f"{b}_p3.npz")
        sid = s.p1["sample_test"].astype(str)
        L = lab.set_index("sample_id").loc[sid]
        s.y = L.label_4.to_numpy().astype(int); s.cl = L.cluster.astype(str).to_numpy(); s.dl = L.dialogue.astype(str).to_numpy()
        s.sid, s.f, s.r = sid, f, r

    def pred(s, spec):
        t = spec[0]
        if t == "start":
            return s.p1[f"p1_{spec[1]}_start_pred"].astype(int)
        if t == "none":
            _, k, K, cw, cb = spec
            it = [str(x) for x in s.p1[f"p1_{k}_items"]]
            return s.p1[f"p1_{k}_pred"][it.index(f"{K}|{cw}|{cb}")].astype(int)
        if t == "tgtonly":
            it = [str(x) for x in s.p1[f"p1_{spec[1]}_tgt_items"]]
            return s.p1[f"p1_{spec[1]}_tgt_pred"][it.index(f"{spec[2]}|{spec[3]}")].astype(int)
        if t == "kbias":
            it = [str(x) for x in s.p1[f"p1_{spec[1]}_kbias_items"]]
            return s.p1[f"p1_{spec[1]}_kbias_pred"][it.index(f"{spec[2]}|{spec[3]}")].astype(int)
        if t == "p3":
            _, key, st_, K, want = spec
            items = [json.loads(x) for x in s.p3[f"{key}_{st_}_K{K}_items"]]
            h = [j for j, it in enumerate(items) if all(nm(it.get(a)) == nm(b) if not isinstance(b, str) else it.get(a) == b for a, b in want.items())]
            assert len(h) == 1, (spec, len(h))
            return s.p3[f"{key}_{st_}_K{K}_pred"][h[0]].astype(int)
        if t == "rolesep":
            _, st_, K, v, rho, a = spec
            ja = [(vv, rr) for vv in ("sel", "one") for rr in RHO].index((v, float(rho)))
            if K == 0:
                return s.p3[f"p3r_{st_}_K0_pred_a"][ja].astype(int)
            return s.p3[f"p3r_{st_}_K{K}_pred_b"][ja * 3 + A.index(int(float(a)))].astype(int)
        if t == "ct1":
            return s.p3[f"p3c_K{spec[2]}_{spec[1]}_pred"][RHO.index(float(spec[3]))].astype(int)
        if t == "coreg":
            return s.p3[f"p3g_K{spec[2]}_{spec[1]}_pred"][COREG.index(float(spec[3]))].astype(int)
        raise ValueError(spec)


def spec_of(s_, K, method, ver, p):
    p = {k: v for k, v in p.items() if k != "pseudo_uar"}
    if method in ("joint", "joint_a1"):
        return ("p3", "p3", s_, K, dict(method="joint", teacher=ver, rho=p["rho"], a=p["a"]))
    if method in ("kd", "agree", "mix"):
        return ("p3", "p3", s_, K, dict(method=method, teacher=ver, **p))
    if method == "rolesep":
        return ("rolesep", s_, K, ver, p["rho"], p["a"])
    if method == "cotrain":
        return ("p3", "p3", s_, K, dict(method="cotrain", teacher="ct", rho=p["rho"])) if s_ == "AVT" else ("ct1", s_, K, p["rho"])
    if method == "coreg":
        return ("coreg", s_, K, p["rho"])
    if method == "tgtonly":
        return ("tgtonly", s_, K, nm(p["C"]) if False else p["C"])
    if method == "kbias":
        return ("kbias", s_, K, p["Cb"])
    if method == "joint_scratch":
        return ("p3", "p3z", s_, K, dict(method="joint_scratch", teacher=ver, rho=1.0, a=1, C=p["C"]))
    raise ValueError(method)


def sel(s_, K, m, v):
    return FZ["select"][f"{s_}|{K}|{m}|{v}"]


def none_spec(s_, K):
    if K == 0:
        return ("start", s_)
    cw, cb = FZ["tether"][f"{s_}|{K}"]
    return ("none", s_, K, cw, cb)


def g_eff(c):
    n = pd.read_csv(ROOT / f"DAI/splits/{c}.csv", keep_default_na=False).cluster.value_counts().to_numpy().astype(float)
    return n.sum() ** 2 / (n ** 2).sum()


def jk(units, fA, fB, ge, mask=None, clkey="cl", stat=uar):
    """設計 9.3 節の jackknife（G_eff < 20）．fA，fB：単位 → 判定．mask：単位 → 残す印。"""
    def vals(u, keep):
        y = u.y[keep]
        return y, fA(u)[keep], fB(u)[keep]
    folds = sorted({u.f for u in units})
    d = {f: [] for f in folds}
    for u in units:
        m = np.ones(len(u.y), bool) if mask is None else mask(u)
        y, a, b = vals(u, m)
        d[u.f].append(stat(y, a) - stat(y, b))
    theta = np.mean([np.mean(d[f]) for f in folds])
    R = {f: len(d[f]) for f in folds}
    s2 = {f: (np.var(d[f], ddof=1) if R[f] > 1 else 0.0) for f in folds}
    cls = sorted({c for u in units for c in getattr(u, clkey)[(np.ones(len(u.y), bool) if mask is None else mask(u))]})
    th, dropped = [], set()
    for g in cls:
        dg = {}
        for f in folds:
            v = []
            for u in [x for x in units if x.f == f]:
                m = np.ones(len(u.y), bool) if mask is None else mask(u)
                cl = getattr(u, clkey)
                if (cl[m] == g).any():
                    keep = m & (cl != g)
                    y = u.y[keep]
                    if len(y) >= 30 and all((y == k).any() for k in range(4)):
                        v.append(stat(y, fA(u)[keep]) - stat(y, fB(u)[keep]))
                else:
                    y, a, b = vals(u, m)
                    v.append(stat(y, a) - stat(y, b))
            if v:
                dg[f] = np.mean(v)
            else:
                dropped.add(f)
        th.append(np.mean(list(dg.values())))
    th = np.array(th); G = len(cls)
    vjk = (G - 1) / G * ((th - th.mean()) ** 2).sum()
    vs = sum(s2[f] / R[f] for f in folds if f not in dropped) / len(folds) ** 2
    se = np.sqrt(vjk + vs)
    df = max(1, int(np.floor(ge)) - 1)
    q = st.t.ppf(0.975, df)
    return dict(effect=theta, lo=theta - q * se, hi=theta + q * se, se=se)


NAN = {"n": 0}


def mx(cur, *vals):
    for v in vals:
        if not np.isfinite(v):
            NAN["n"] += 1
        else:
            cur = max(cur, v)
    return cur


def main():
    out = {}
    for c in [x for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ("AFEW", "IEMOCAP", "MOSEI", "MELD")) if x != T]:
        lab = pd.read_csv(ROOT / f"DAI/splits/{c}.csv", keep_default_na=False, dtype={"cluster": str, "dialogue": str})
        units = [U(c, f, r, lab) for f in range(5) for r in (0, 1)]
        ge = g_eff(c)
        L = pd.read_csv(DEST / "levels.csv"); L = L[L.c == c]
        R_ = pd.read_csv(DEST / "recall.csv"); R_ = R_[R_.c == c]
        AB = pd.read_csv(DEST / "ablation.csv"); AB = AB[AB.c == c]
        OF = pd.read_csv(DEST / "outside_families.csv"); OF = OF[OF.c == c]
        diffs = {"level_theta": 0.0, "level_ci": 0.0, "recall_theta": 0.0, "recall_ci": 0.0, "abl_theta": 0.0, "out_ci": 0.0,
                 "n_level": 0, "n_recall": 0, "n_abl": 0, "n_out": 0}
        for s_ in ("A", "V", "T", "AVT"):
            for K in (0, 1, 5, 10, 25):
                v = FZ["main"][str(K)]
                part = FZ["partner"][f"{s_}|{K}"]
                rows = {"none": none_spec(s_, K),
                        f"joint|{v}|main": spec_of(s_, K, "joint", v, sel(s_, K, "joint", v)),
                        f"rolesep|{v}|main": spec_of(s_, K, "rolesep", v, sel(s_, K, "rolesep", v)),
                        "partner": spec_of(s_, K, part["method"], part["version"], sel(s_, K, part["method"], part["version"])),
                        "joint|one|rho=1|a=1 (default)": ("p3", "p3", s_, K, dict(method="joint", teacher="one", rho=1.0, a=1))}
                if K > 0:
                    rows["kbias"] = spec_of(s_, K, "kbias", "-", sel(s_, K, "kbias", "-"))
                    rows["tgtonly"] = spec_of(s_, K, "tgtonly", "-", sel(s_, K, "tgtonly", "-"))
                for name, spec in rows.items():
                    mine = jk(units, lambda u, sp=spec: u.pred(sp), lambda u: np.full(len(u.y), -1), ge) if ge < 20 else None
                    th = np.mean([np.mean([uar(u.y, u.pred(spec)) for u in units if u.f == f]) for f in range(5)])
                    row = L[(L.student == s_) & (L.K == K) & (L.row == name)]
                    assert len(row) == 1, (c, s_, K, name, len(row))
                    diffs["level_theta"] = mx(diffs["level_theta"], abs(th - row.effect.iloc[0])); diffs["n_level"] += 1
                    if mine is not None:
                        diffs["level_ci"] = mx(diffs["level_ci"], abs(mine["lo"] - row.lo.iloc[0]), abs(mine["hi"] - row.hi.iloc[0]))
                    # 感情ごとの再現率（族の行だけ）
                    if name in ("none", f"joint|{v}|main", "partner") and K in (0, 5):
                        rn = {"none": "none", f"joint|{v}|main": "prop", "partner": "partner"}[name]
                        for k in range(4):
                            stat = lambda y, p, k=k: rec(y, p, k)
                            thk = np.mean([np.mean([rec(u.y, u.pred(spec), k) for u in units if u.f == f]) for f in range(5)])
                            rr = R_[(R_.kind == "level") & (R_.student == s_) & (R_.K == K) & (R_.row == rn) & (R_.emotion == k)]
                            assert len(rr) == 1
                            diffs["recall_theta"] = mx(diffs["recall_theta"], abs(thk - rr.effect.iloc[0])); diffs["n_recall"] += 1
                            if ge < 20:
                                mk = jk(units, lambda u, sp=spec: u.pred(sp), lambda u: np.full(len(u.y), -1), ge, stat=stat)
                                diffs["recall_ci"] = mx(diffs["recall_ci"], abs(mk["lo"] - rr.lo.iloc[0]), abs(mk["hi"] - rr.hi.iloc[0]))
                # 除去実験の d（総量をそろえる操作）と h（学び直し）の θ
                d1 = ("p3", "p3", s_, K, dict(method="joint", teacher="one", rho=1.0, a=1))
                for rid, b in (("d", ("p3", "p3", s_, K, dict(method="kd", teacher="one", temp=1, rho=1.0))),
                               ("h", ("p3", "p3", s_, K, dict(method="tt", teacher="poe_none", rho=1.0, a=1)))):
                    th = np.mean([np.mean([uar(u.y, u.pred(d1)) - uar(u.y, u.pred(b)) for u in units if u.f == f]) for f in range(5)])
                    row = AB[(AB.student == s_) & (AB.K == K) & (AB.row == rid)]
                    diffs["abl_theta"] = mx(diffs["abl_theta"], abs(th - row.effect.iloc[0])); diffs["n_abl"] += 1
                # 族の外：K = 1，10 の族 1 の区間
                if K in (1, 10) and ge < 20:
                    a, b = rows[f"joint|{v}|main"], rows["none"]
                    mine = jk(units, lambda u: u.pred(a), lambda u: u.pred(b), ge)
                    row = OF[(OF.student == s_) & (OF.K == K) & (OF.family == 1)]
                    diffs["out_ci"] = mx(diffs["out_ci"], abs(mine["lo"] - row.lo.iloc[0]), abs(mine["hi"] - row.hi.iloc[0]),
                                          abs(mine["effect"] - row.effect.iloc[0])); diffs["n_out"] += 1
        # 取りこぼし：提案（joint|one）の格子 12 点の θ と最良 − 選んだ，2 組の水準
        RG = pd.read_csv(DEST / "regret.csv"); RG = RG[RG.c == c]
        dreg = 0.0; nreg = 0
        halves = {}
        for u in units:
            g = np.zeros(len(u.y), int)
            for k in range(4):
                ix = np.flatnonzero(u.y == k)
                order = ix[np.lexsort((u.sid[ix], u.cl[ix]))]
                g[order] = np.arange(len(order)) % 2
            halves[(u.f, u.r)] = g
        okf = [f for f in range(5) if all(len(u.y[halves[(u.f, u.r)] == h]) >= 30 and all((u.y[halves[(u.f, u.r)] == h] == k).any() for k in range(4))
                                          for u in units if u.f == f for h in (0, 1))]
        for s_ in ("A", "AVT"):
            for K in (5, 25):
                for ver in ("sel", "one"):
                    pts = [("p3", "p3", s_, K, dict(method="joint", teacher=ver, rho=r, a=a)) for r in RHO for a in A]
                    ch = sel(s_, K, "joint", ver)
                    isel = [i for i, p in enumerate(pts) if float(p[4]["rho"]) == float(ch["rho"]) and int(p[4]["a"]) == int(float(ch["a"]))][0]
                    th = np.array([np.mean([np.mean([uar(u.y, u.pred(p)) for u in units if u.f == f]) for f in range(5)]) for p in pts])
                    best = isel if th[isel] >= th.max() else int(np.argmax(th))
                    def thh(p, h):
                        return np.mean([np.mean([uar(u.y[halves[(u.f, u.r)] == h], u.pred(p)[halves[(u.f, u.r)] == h]) for u in units if u.f == f]) for f in okf])
                    tA = np.array([thh(p, 0) for p in pts]); tB = np.array([thh(p, 1) for p in pts])
                    null = np.mean([tB[np.argmax(tA)] - tB[isel], tA[np.argmax(tB)] - tA[isel]])
                    row = RG[(RG.student == s_) & (RG.K == K) & (RG.method == "joint") & (RG.version == ver)]
                    dreg = mx(dreg, abs((th[best] - th[isel]) - row.effect.iloc[0]), abs(null - row.null_level.iloc[0])); nreg += 1
        diffs["regret"] = dreg; diffs["n_regret"] = nreg
        # 部分集合（9.4 節）
        S = pd.read_csv(DEST / "sensitivity.csv"); S = S[S.c == c]
        sub = pd.read_csv(ROOT / "DAI/frozen/subsets.csv", dtype=str)
        dsub, nsub = 0.0, 0
        if c in ("AFEW", "IEMOCAP", "MELD"):
            if c == "AFEW":
                bad = set(sub[(sub.corpus == "AFEW") & (sub.subset == "asr_suspect")].sample_id)
                variants = {"asr_ok": (lambda u: ~np.isin(u.sid, list(bad)), "cl", ("T", "AVT"))}
            elif c == "IEMOCAP":
                mk = dict(zip(sub[sub.subset == "marker"].sample_id, sub[sub.subset == "marker"].value))
                variants = {"marker": (lambda u: np.array([mk.get(x) == "marker" for x in u.sid]), "cl", ("A", "V", "T", "AVT")),
                            "nomarker": (lambda u: np.array([mk.get(x) == "nomarker" for x in u.sid]), "cl", ("A", "V", "T", "AVT"))}
            else:
                fc = dict(zip(sub[sub.subset == "face_compete"].sample_id, sub[sub.subset == "face_compete"].value))
                variants = {"face_single": (lambda u: np.array([fc.get(x) == "single" for x in u.sid]), "cl", ("V", "AVT")),
                            "dialogue": (None, "dl", ("A", "V", "T", "AVT"))}
            for vname, (mf, ck, studs) in variants.items():
                for s_ in studs:
                    for K in (0, 5, 25):
                        a = spec_of(s_, K, "joint", FZ["main"][str(K)], sel(s_, K, "joint", FZ["main"][str(K)]))
                        b = none_spec(s_, K)
                        nf = {f: int((mf(u) if mf else np.ones(len(u.y), bool)).sum()) for u in units if u.r == 0 for f in [u.f]}
                        badf = {f for f, n in nf.items() if n < 30}
                        uu = [u for u in units if u.f not in badf]
                        row = S[(S.subset == vname) & (S.family == 1) & (S.student == s_) & (S.K == K)]
                        assert len(row) == 1, (vname, s_, K)
                        if len(uu) == 0:
                            continue
                        mine = jk(uu, lambda u: u.pred(a), lambda u: u.pred(b), ge, mask=mf, clkey=ck)
                        dsub = mx(dsub, abs(mine["effect"] - row.effect.iloc[0]), abs(mine["lo"] - row.lo.iloc[0]), abs(mine["hi"] - row.hi.iloc[0]))
                        nsub += 1
        diffs["subsets"] = dsub; diffs["n_subsets"] = nsub
        # 出発点の C の曲線：選んだ C の点の判定が出発点と同じか（再構成の食い違い）
        SC = pd.read_csv(DEST / "start_C_curve.csv"); SC = SC[SC.c == c]
        diffs["start_C_recon_mismatch_max"] = int(SC.recon_mismatch.max())
        diffs["nan_diffs"] = NAN["n"]; NAN["n"] = 0
        out[c] = diffs
        print(c, json.dumps({k: (round(v, 9) if isinstance(v, float) else v) for k, v in diffs.items()}), flush=True)
    json.dump(out, open(DEST.parent / f"verify_indep_{T}_{'_'.join(out)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
