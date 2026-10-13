"""【探索的】E31：論文 2 の比較実験の解析（設定の選び方 3 規則，主な比較，予言 R1〜R11 のうち段 1 で判定できるもの）。新しい学習なし。

規則は `docs/exploratory-analyses.md` E31「段 1 の解析の細部」。入力は `results/compare_<kind>/*.csv`，
地図（`results/map_<kind>/`：ターゲットのみ・ソースのみ・合併の C の交差検証，`results/fixc_map/<kind>/`：C を固定した合併）。

    ./.venv/bin/python exp/explore_compare.py [--kinds …]

出力：`results/explore_relation/e31_pairs.csv`（組 × K × 手法と選び方の UAR．点），`e31_tests.csv`（予言の判定），
`e31_main.csv`（主な比較），`e31_summary.csv`（種類 × K の平均）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
REPS = list(range(10))
OUT = ROOT / "results" / "explore_relation"
DEF_CP, DEF_CB, DEF_ALPHA = 0.01, "8", 0.5
CB_ORDER = {"8": 0, "2": 1, "free": 2}


def corpus(g: str) -> str:
    return GROUP_SPEC[g][0]


def same(a: str, b: str) -> bool:
    return corpus(a) == corpus(b)


# ---------------------------------------------------------------- 読み込み

def load_compare(kind: str) -> pd.DataFrame:
    fs = sorted((ROOT / "results" / f"compare_{kind}").glob("*.csv"))
    d = pd.concat([pd.read_csv(f, dtype={"K": str, "cb": str, "variant": str}) for f in fs], ignore_index=True)
    d["cb"], d["variant"] = d.cb.fillna(""), d.variant.fillna("")
    d["uar"], d["cv_uar"] = 100 * d.uar, 100 * d.cv_uar
    return d[d.rep.isin(REPS)]


def load_map(kind: str, targets) -> pd.DataFrame:
    out = []
    for t in targets:
        for path, tag in ((ROOT / "results" / f"map_{kind}" / f"{t}.csv", "cv"), (ROOT / "results" / "fixc_map" / kind / f"{t}.csv", "fix")):
            d = pd.read_csv(path, comment="#", dtype={"K": str, "source": str}, usecols=["target", "source", "method", "K", "rep", "uar"])
            d["K"], d["source"], d["src"] = d.K.fillna(""), d.source.fillna(""), tag
            out.append(d[d.rep.isin(REPS)])
    m = pd.concat(out, ignore_index=True)
    m["uar"] = 100 * m.uar
    return m


# ---------------------------------------------------------------- 選び方

class Family:
    """手法の族：行の絞り込み，設定の列，既定，同点の順。"""

    def __init__(self, name, method, keys, cond, default, tie_key):
        self.name, self.method, self.keys, self.cond, self.default, self.tie_key = name, method, keys, cond, default, tie_key

    def rows(self, d):
        x = d[(d.method == self.method)]
        x = x[self.cond(x)].copy()
        x["setting"] = list(zip(*[x[k] for k in self.keys])) if len(self.keys) > 1 else list(x[self.keys[0]])
        return x


def ht_tie(s):
    """同点の順：|log10(C′/0.01)|，C′，C_b（8，2，自由）。s は C′ か（C′, C_b）。"""
    cp, cb = s if isinstance(s, tuple) else (s, DEF_CB)
    return (round(abs(np.log10(cp / DEF_CP)), 9), cp, CB_ORDER[cb])


FAMILIES = [
    Family("P1", "HT", ["cp"], lambda x: (x.cen == 0) & (x.cb == "free"), DEF_CP, ht_tie),
    Family("P2", "HT", ["cp"], lambda x: (x.cen == 1) & (x.cb == "free"), DEF_CP, ht_tie),
    Family("P3", "HT", ["cp", "cb"], lambda x: (x.cen == 1) & x.cb.isin(["2", "8"]), (DEF_CP, DEF_CB), ht_tie),
    Family("P3c", "HT", ["cp"], lambda x: (x.cen == 1) & (x.cb == "8"), DEF_CP, ht_tie),
    Family("HT24", "HT", ["cen", "cp", "cb"], lambda x: x.cen.isin([0, 1]), (1, DEF_CP, DEF_CB),
           lambda s: ((0 if s[0] == 1 else 1),) + ht_tie((s[1], s[2]))),
    Family("O1", "O1", ["alpha"], lambda x: x.alpha.notna(), DEF_ALPHA, lambda a: (round(abs(a - DEF_ALPHA), 9), a)),
]
P4FAM = Family("P4", "HT_ALL", ["cp", "cb"], lambda x: (x.cen == 1) & x.cb.isin(["2", "8"]), (DEF_CP, DEF_CB), ht_tie)


def pick(scores: dict, fam: Family):
    """scores：設定 → 値。最大の設定（同点は既定，次に fam.tie_key の順）。"""
    best = max(scores.values())
    tied = [s for s, v in scores.items() if abs(v - best) <= 1e-12]
    if fam.default in tied:
        return fam.default
    return sorted(tied, key=fam.tie_key)[0]


def select_family(x: pd.DataFrame, fam: Family) -> pd.DataFrame:
    """1 ソースの族：組 × K ごとに 既定・メタ検証・K 本の交差検証・オラクルの UAR。"""
    pm = x.groupby(["target", "source", "K", "setting"]).uar.mean()
    tab = pm.unstack("setting")
    out = []
    pairs = tab.index.to_frame(index=False)
    for (t, s, K), row in tab.iterrows():
        rec = {"target": t, "source": s, "K": K}
        rec[f"{fam.name}_def"] = row.get(fam.default, np.nan)
        rec[f"{fam.name}_orc"] = row.max()
        # メタ検証：S をほかのターゲット T′ に使った組（T・S のコーパス・T のコーパスを除く）
        tp = pairs[(pairs.source == s) & (pairs.K == K)].target
        tp = [u for u in tp if u != t and not same(u, s) and not same(u, t)]
        if not tp:
            raise ValueError(f"メタ検証に使える T′ が無い: {s} → {t}，K={K}")
        sc = tab.loc[[(u, s, K) for u in tp]].mean(0)
        ch = pick(sc.dropna().to_dict(), fam)
        rec[f"{fam.name}_meta"], rec[f"{fam.name}_meta_set"] = row[ch], str(ch)
        out.append(rec)
    R = pd.DataFrame(out)
    # K 本の交差検証：反復ごとに選ぶ
    cvv = []
    for (t, s, K, r), g in x.groupby(["target", "source", "K", "rep"]):
        sc = g.dropna(subset=["cv_uar"])
        if len(sc):
            ch = pick(dict(zip(sc.setting, sc.cv_uar)), fam)
        else:
            ch = fam.default
        cvv.append({"target": t, "source": s, "K": K, "rep": r, "u": float(g[g.setting.apply(lambda v: v == ch)].uar.iloc[0])})
    C = pd.DataFrame(cvv).groupby(["target", "source", "K"]).u.mean().rename(f"{fam.name}_cv").reset_index()
    return R.merge(C, on=["target", "source", "K"])


def select_p4(d: pd.DataFrame, ht: pd.DataFrame) -> pd.DataFrame:
    """P4：ターゲット × K ごとに 既定・メタ検証（1 ソースの疑似組で代用）・交差検証・オラクル。"""
    x = P4FAM.rows(d)
    tab = x.groupby(["target", "K", "setting"]).uar.mean().unstack("setting")
    h = P4FAM.rows(ht)
    h = h[~h.apply(lambda r: same(r.source, r.target), axis=1)]
    hp = h.groupby(["target", "source", "K", "setting"]).uar.mean().reset_index()
    out = []
    for (t, K), row in tab.iterrows():
        q = hp[(hp.K == K) & ~hp.target.map(lambda u: same(u, t)) & ~hp.source.map(lambda u: same(u, t))]
        sc = q.groupby("setting").uar.mean().to_dict()
        ch = pick(sc, P4FAM)
        out.append({"target": t, "K": K, "P4_def": row.get(P4FAM.default, np.nan), "P4_orc": row.max(),
                    "P4_meta": row[ch], "P4_meta_set": str(ch)})
    R = pd.DataFrame(out)
    cvv = []
    for (t, K, r), g in x.groupby(["target", "K", "rep"]):
        sc = g.dropna(subset=["cv_uar"])
        ch = pick(dict(zip(sc.setting, sc.cv_uar)), P4FAM) if len(sc) else P4FAM.default
        cvv.append({"target": t, "K": K, "u": float(g[g.setting.apply(lambda v: v == ch)].uar.iloc[0])})
    C = pd.DataFrame(cvv).groupby(["target", "K"]).u.mean().rename("P4_cv").reset_index()
    return R.merge(C, on=["target", "K"])


# ---------------------------------------------------------------- 組の表

def pair_table(kind: str) -> pd.DataFrame:
    d = load_compare(kind)
    targets = sorted(d.target.unique())
    mp = load_map(kind, targets)
    T = None
    for fam in FAMILIES:
        S = select_family(fam.rows(d), fam)
        T = S if T is None else T.merge(S, on=["target", "source", "K"], how="outer")
    ht = d[d.method == "HT"]
    for cen in (0, 1):
        for cb in ("8", "free"):
            v = ht[(ht.cen == cen) & (ht.cp == DEF_CP) & (ht.cb == cb)].groupby(["target", "source", "K"]).uar.mean()
            T = T.merge(v.rename(f"HT_c{cen}_{cb}").reset_index(), on=["target", "source", "K"], how="left")
    h3 = d[d.method == "H3"].groupby(["target", "source", "K"]).uar.mean().rename("H3").reset_index()
    T = T.merge(h3, on=["target", "source", "K"], how="left")
    for variant in ("none", "src_mean", "tgt_mean"):
        v = d[(d.method == "O6") & (d.variant == variant)].groupby(["target", "K"]).uar.mean().rename(f"O6_{variant}").reset_index()
        T = T.merge(v, on=["target", "K"], how="left")
    o7 = d[d.method == "O7"].groupby(["target", "K"]).uar.mean().rename("O7").reset_index()
    T = T.merge(o7, on=["target", "K"], how="left")
    T = T.merge(select_p4(d, ht), on=["target", "K"], how="left")
    # 地図の基準
    b1 = mp[(mp.src == "cv") & (mp.method == "tgt_only")].groupby(["target", "K"]).uar.mean().rename("B1").reset_index()
    b2 = mp[(mp.src == "cv") & (mp.method == "src_only")].groupby(["target", "source"]).uar.mean().rename("B2").reset_index()
    b3 = mp[(mp.src == "cv") & (mp.method == "merged")].groupby(["target", "source", "K"]).uar.mean().rename("B3").reset_index()
    b4 = mp[(mp.src == "fix") & (mp.method == "merged")].groupby(["target", "source", "K"]).uar.mean().rename("B4").reset_index()
    T = T.merge(b1, on=["target", "K"], how="left").merge(b2, on=["target", "source"], how="left")
    T = T.merge(b3, on=["target", "source", "K"], how="left").merge(b4, on=["target", "source", "K"], how="left")
    T.insert(0, "kind", kind)
    T["same"] = [same(a, b) for a, b in zip(T.target, T.source)]
    return T


# ---------------------------------------------------------------- 判定

def wil(x, alternative="two-sided"):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or np.allclose(x, 0):
        return np.nan
    return float(stats.wilcoxon(x, alternative=alternative).pvalue)


def tci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def holm(ps):
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    ok = ~np.isnan(ps)
    idx = np.argsort(np.where(ok, ps, np.inf))
    m, run = ok.sum(), 0.0
    for rank, i in enumerate(idx[:m]):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        out[i] = run
    return out


def unsafe(P: pd.DataFrame, col: str) -> pd.Series:
    return P[col] < np.maximum(P.B1, P.B2) - 1


def judge(P: pd.DataFrame):
    rows = []
    X = P[~P.same]
    for kind in X.kind.unique():
        A = X[X.kind == kind]
        aud = kind in ("audio_wavlm", "audio_hubert")
        k1, k5, k10, k25 = (A[A.K == K] for K in ("1", "5", "10", "25"))
        # R1
        db = (k1.HT_c1_8 - k1.HT_c0_8).mean()
        df = (k1.HT_c1_free - k1.HT_c0_free).mean()
        rows.append({"kind": kind, "R": "R1", "value": f"縛る {db:+.2f}，自由 {df:+.2f}", "holds": (db >= 1 and df < 0.5) if aud else np.nan})
        # R2
        d5, d10, d1 = (g.P3_def - g.H3 for g in (k5, k10, k1))
        p5, p10 = wil(d5, "greater"), wil(d10, "greater")
        ok = d5.mean() >= 1 and p5 < 0.05 and d10.mean() >= 1 and p10 < 0.05 and abs(d1.mean()) <= 0.5
        rows.append({"kind": kind, "R": "R2", "value": f"K5 {d5.mean():+.2f}(p={p5:.3g}) K10 {d10.mean():+.2f}(p={p10:.3g}) K1 {d1.mean():+.2f}",
                     "holds": ok if aud else np.nan})
        # R3（ターゲットが単位）
        g = k5.groupby("target").agg(P4=("P4_def", "first"), P3m=("P3_def", "mean"), P3b=("P3_def", "max"))
        r3 = (g.P4 - g.P3m).mean()
        rows.append({"kind": kind, "R": "R3", "value": f"P4 − P3 平均 {r3:+.2f}（記述：P4 − 最良 {(g.P4 - g.P3b).mean():+.2f}，n={len(g)}）",
                     "holds": (r3 >= 3) if aud else np.nan})
        # R6 前半
        f25 = (k25.P3_def > k25.B1).mean()
        rows.append({"kind": kind, "R": "R6a", "value": f"K25 で P3 > ターゲットのみ {f25:.2f}（n={len(k25)}）", "holds": (f25 >= 0.7) if aud else np.nan})
        # R7
        L = A[A.K.isin(["1", "5", "10"])]
        mc, om = (L.P3c_meta - L.P3c_cv).mean(), (L.P3c_orc - L.P3c_meta).mean()
        per = "，".join(f"K{K} {(L[L.K == K].P3c_meta - L[L.K == K].P3c_cv).mean():+.2f}/{(L[L.K == K].P3c_orc - L[L.K == K].P3c_meta).mean():.2f}"
                       for K in ("1", "5", "10"))
        rows.append({"kind": kind, "R": "R7", "value": f"メタ − 交差 {mc:+.2f}，オラクル − メタ {om:.2f}（{per}）",
                     "holds": (mc >= 0.5 and om <= 1) if aud else np.nan})
        # R8
        a6, a7 = (k5.P3_def - k5.O6_tgt_mean).mean(), (k5.P3_def - k5.O7).mean()
        rows.append({"kind": kind, "R": "R8", "value": f"P3 − O6 {a6:+.2f}，P3 − O7 {a7:+.2f}", "holds": (a6 >= 2 and a7 >= 0) if aud else np.nan})
        # R10
        u1, u5 = unsafe(k1, "P3_def").mean(), unsafe(k5, "P3_def").mean()
        rows.append({"kind": kind, "R": "R10", "value": f"K1 {u1:.3f}，K5 {u5:.3f}（n={len(k1)}）",
                     "holds": (u1 <= 0.05 and u5 <= 0.075) if aud else (u1 <= 0.2 and u5 <= 0.2)})
    return pd.DataFrame(rows)


MAIN_PAIR = [("P3_def", "B3"), ("P3_def", "B4"), ("P3_def", "H3"), ("P3_def", "P1_def"),
             ("P3_meta", "B3"), ("P3_meta", "B4"), ("P3_meta", "H3"), ("P3_meta", "P1_def")]
MAIN_TGT = [("P4_def", "P3_srcmean"), ("P4_def", "P3_srcbest"), ("P4_meta", "P3_srcmean"), ("P4_meta", "P3_srcbest")]


def main_comparisons(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    X = P[~P.same]
    for kind in X.kind.unique():
        for K in sorted(X.K.unique(), key=lambda k: (k == "all", float(k) if k != "all" else 0)):
            A = X[(X.kind == kind) & (X.K == K)]
            G = A.groupby("target").agg(P4_def=("P4_def", "first"), P4_meta=("P4_meta", "first"),
                                        P3_srcmean=("P3_def", "mean"), P3_srcbest=("P3_def", "max"))
            fam = []
            for a, b in MAIN_PAIR:
                dd = (A[a] - A[b]).to_numpy()
                fam.append({"unit": "pair", "a": a, "b": b, "dd": dd, "safe_a": 1 - unsafe(A, a).mean()})
            for a, b in MAIN_TGT:
                fam.append({"unit": "target", "a": a, "b": b, "dd": (G[a] - G[b]).to_numpy(), "safe_a": np.nan})
            ps = [wil(f["dd"]) for f in fam]
            for f, p, ph in zip(fam, ps, holm(ps)):
                lo, hi = tci(f["dd"])
                rows.append({"kind": kind, "K": K, "unit": f["unit"], "a": f["a"], "b": f["b"], "n": int(np.sum(~np.isnan(f["dd"]))),
                             "diff": np.nanmean(f["dd"]), "lo": lo, "hi": hi, "wins": int(np.sum(f["dd"] > 0)),
                             "p": p, "p_holm": ph, "safe_a": f["safe_a"]})
    return pd.DataFrame(rows)


COLS = ["B1", "B2", "B3", "B4", "H3", "P1_def", "P2_def", "P3_def", "P3_meta", "P3_cv", "P3_orc", "HT24_meta", "HT24_orc",
        "P4_def", "P4_meta", "P4_cv", "P4_orc", "O1_meta", "O1_cv", "O1_orc", "O6_tgt_mean", "O6_none", "O7"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    args = ap.parse_args()
    kinds = [k for k in args.kinds if (ROOT / "results" / f"compare_{k}").exists() and any((ROOT / "results" / f"compare_{k}").glob("*.csv"))]
    P = pd.concat([pair_table(k) for k in kinds], ignore_index=True)
    P.to_csv(OUT / "e31_pairs.csv", index=False)
    T, Mn = judge(P), main_comparisons(P)
    T.to_csv(OUT / "e31_tests.csv", index=False)
    Mn.to_csv(OUT / "e31_main.csv", index=False)
    S = P[~P.same].groupby(["kind", "K"])[[c for c in COLS if c in P]].mean().reset_index()
    S.to_csv(OUT / "e31_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(T.to_string(index=False))
    print(S.round(2).to_string(index=False))
    print(Mn.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
