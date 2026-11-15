"""【探索的】E25（手の比較）と E26（複数のソースを混ぜる）の解析。新しい学習はしない。

予言と解析の細部は `docs/exploratory-analyses.md` E25・E26 の 1 節（計算の前に書いた）。
入力は `exp/interventions.py` と `exp/multisource.py` の出力（C を固定した地図で解いたもの），
式の λ と位置による損は C を固定した地図の E15 の出力，ターゲットのみ・ソースのみは C を固定した地図。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_interventions.py
    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_interventions.py --stage transfer

出力：`OUT/e25_pairs.csv`（組 × K の手ごとの UAR と λ・位置による損），`OUT/e25_tests.csv`，`OUT/e25_summary.csv`，
`OUT/e26_targets.csv`（ターゲット × K × 混ぜ方），`OUT/e26_tests.csv`，`OUT/e26_summary.csv`．
`--stage transfer`（E25 の 3 節）：`exp/transfer_scores.py` の点数から手 7 の d・o を測り，C を固定した段階 B の
合併と比べる．出力は `OUT/e25t_rows.csv`・`OUT/e25t_pairs.csv`・`OUT/e25t_summary.csv`・`OUT/e25t_tests.csv`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
from explore_pairs import ba  # noqa: E402
from explore_synergy import KS, O_MIN, PAIRS, d_o, load_map  # noqa: E402
from pair_table import OUT, holm  # noqa: E402

MAIN = {"audio_wavlm": "W7", "audio_hubert": "Hu7", "video": "V", "fuse-audio-video": "AV"}
TEXT = {"text_roberta": "T", "fuse-audio-text": "AT", "fuse-video-text": "VT", "fuse-audio-video-text": "AVT"}
E15_DIR = {**{k: ROOT / "results/fixc_analysis/explore_relation" for k in MAIN},
           **{k: ROOT / "results/fixc_text/explore_relation" for k in TEXT}}
REPS = range(10)
HANDS = {"1 ふつうの合併": "h1", "2 中心合わせ＋ソースのみ": "h2", "3 中心合わせ＋合併": "h3",
         "4 ソースのみ＋切片直し": "h4", "5 中心合わせ＋ソースのみ＋切片直し": "h5",
         "6 ターゲット専用の切片つき合併": "h6", "7 仮説転移": "h7", "8 CORAL＋合併": "h8"}
H = list(HANDS.values())


def read_dir(d: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str}) for f in sorted(d.glob("*.csv"))],
                     ignore_index=True)


def spear(x: pd.Series, y: pd.Series, alt: str) -> tuple[float, float, int]:
    ok = x.notna() & y.notna()
    r, p = stats.spearmanr(x[ok], y[ok], alternative=alt)
    return float(r), float(p), int(ok.sum())


# ---------------------------------------------------------------- E25

def e25_pairs(kind: str) -> pd.DataFrame:
    d = read_dir(ROOT / "results" / f"interventions_{kind}")
    d = d[d.rep.isin(REPS)]
    d["K"] = d.K.fillna("")
    d["hand"] = d.method.map({**HANDS, "0 ソースのみ（学び直し）": "src"})
    per = d.groupby(["target", "source", "K", "hand"]).uar.mean().unstack("hand")
    noK = per.xs("", level="K")[["src", "h2"]]
    withK = per.drop(index="", level="K").drop(columns=["src", "h2"])
    w = withK.reset_index().merge(noK.reset_index(), on=["target", "source"])
    mp = load_map(kind)
    mp = mp[mp.rep.isin(REPS)]
    w = w.merge(mp.groupby(["target", "K"]).t.mean().reset_index(), on=["target", "K"])
    lam = pd.read_csv(E15_DIR[kind] / "e15_lambda.csv", comment="#", dtype={"K": str})
    lam = lam[lam.kind == kind][["target", "source", "K", "lam_pred_rep09"]].rename(columns={"lam_pred_rep09": "lam"})
    pos = pd.read_csv(E15_DIR[kind] / "e15_pairs.csv", comment="#", dtype={"K": str})
    pos = pos[(pos.kind == kind) & (pos.K == "5")][["target", "source", "loss_offset"]]
    w = w.merge(lam, on=["target", "source", "K"], how="left").merge(pos, on=["target", "source"], how="left")
    w.insert(0, "kind", kind)
    for h in H[1:]:
        w[f"{h}_minus_h1"] = w[h] - w.h1
    w["h2_minus_src"] = w.h2 - w.src
    w["best"] = w[H].idxmax(axis=1)
    return w


def e25_tests(w: pd.DataFrame, kind: str) -> list[dict]:
    rows = []
    for K in KS:
        g = w[w.K == K]
        tests = [("I1", "loss_offset", "h2_minus_src", "greater"), ("I2", "lam", "h6_minus_h1", "less"),
                 ("I3", "lam", "h4_minus_h1", "less"), ("I4", "lam", "h3_minus_h1", "less")]
        res = []
        for name, x, y, alt in tests:
            r, p, n = spear(g[x], g[y], alt)
            res.append({"kind": kind, "K": K, "test": name, "x": x, "y": y, "alt": alt, "rho": r, "p": p, "n": n})
        ph = holm(np.array([t["p"] for t in res]))
        for t, q in zip(res, ph):
            t["p_holm_ref"] = float(q)
        rows += res
    return rows


def e25_summary(w: pd.DataFrame, kind: str) -> list[dict]:
    rows = []
    for K, g in w.groupby("K"):
        row = {"kind": kind, "K": K, "n_pairs": len(g), "t": g.t.mean(), "src": g.src.mean()}
        for h in H:
            row[h] = g[h].mean()
            row[f"{h}_minus_h1"] = (g[h] - g.h1).mean()
            row[f"n_best_{h}"] = int((g.best == h).sum())
            row[f"n_gt_h1_{h}"] = int((g[h] > g.h1).sum())
        rows.append(row)
    return rows


# ---------------------------------------------------------------- E26

def e26_targets(kind: str) -> pd.DataFrame:
    d = read_dir(ROOT / "results" / f"multisource_{kind}")
    d = d[d.rep.isin(REPS)]
    g = (d.groupby(["target", "K", "variant"])
         .agg(uar=("uar", "mean"), src_only=("uar_src_only", "mean"), n_sources=("n_sources", "first"),
              sources=("sources", "first"), best1=("best1", "first"), best1_m=("best1_m", "first"),
              mean1_m=("mean1_m", "first"), C=("C", "first"))
         .reset_index())
    mp = load_map(kind)
    mp = mp[mp.rep.isin(REPS)]
    g = g.merge(mp.groupby(["target", "K"]).t.mean().reset_index(), on=["target", "K"])
    g.insert(0, "kind", kind)
    g["minus_best1"] = g.uar - g.best1_m
    g["minus_mean1"] = g.uar - g.mean1_m
    return g


def e26_tests(g: pd.DataFrame, kind: str) -> list[dict]:
    rows = []
    for K in KS:
        for v, col, alt in [("全ソース", "minus_mean1", "greater"), ("全ソース", "minus_best1", "less"),
                            ("上位2つ（分析用）", "minus_best1", "greater"),
                            ("上位2つ（平均的な良さ）", "minus_best1", "greater"),
                            ("上位2つ（平均的な良さ）", "minus_mean1", "greater")]:
            x = g[(g.K == K) & (g.variant == v)][col].to_numpy()
            p = float(stats.ttest_1samp(x, 0, alternative=alt).pvalue)
            ok = (x > 0) if alt == "greater" else (x < 0)
            rows.append({"kind": kind, "K": K, "variant": v, "quantity": col, "alt": alt, "mean": float(x.mean()),
                         "n_targets": len(x), "n_as_predicted": int(ok.sum()), "p_paired_t_ref": p})
    return rows


# ---------------------------------------------------------------- E25 の 3 節：仮説転移の d・o

def only(d: pd.DataFrame, kind: str) -> pd.DataFrame:
    return d[d.kind == kind]


def lam_obs(o_s: pd.Series, o_x: pd.Series) -> float:
    v = o_s.abs() >= O_MIN
    return 1 - float((np.sign(o_s[v]) * o_x[v]).sum() / o_s[v].abs().sum()) if v.any() else np.nan


def transfer(kinds: list[str]) -> int:
    rows = []
    for kind in kinds:
        for f in sorted((ROOT / "results" / f"transfer_scores_{kind}").glob("*.npz")):
            z, t = np.load(f), f.stem
            reps = sorted(int(k.split("_r")[1]) for k in z.files if k.startswith("test_idx_r"))
            srcs = sorted({k[2:].rsplit("_r", 1)[0] for k in z.files if k.startswith("s_")})
            for r in reps:
                y = z[f"y_test_r{r}"]
                for s in srcs:
                    ds = d_o(z[f"s_{s}_r{r}"], y)
                    for K in KS:
                        dh = d_o(z[f"h7_{s}_K{K}_r{r}"], y)
                        for (k, j), (d_s, o_s), (d_h, o_h) in zip(PAIRS, ds, dh):
                            rows.append({"kind": kind, "target": t, "source": s, "K": K, "rep": r, "pair": f"{k}-{j}",
                                         "d_s": d_s, "o_s": o_s, "d_h": d_h, "o_h": o_h})
    R = pd.DataFrame(rows)
    key = ["kind", "target", "source", "K", "rep", "pair"]
    Mg = pd.concat([only(pd.read_csv(E15_DIR[k] / "e15b_rows.csv", dtype={"K": str}), k) for k in kinds])
    Mg = Mg[Mg.rep.isin(REPS)].rename(columns={"d_s": "d_s_merged_run", "o_s": "o_s_merged_run"})
    R = R.merge(Mg, on=key, how="left")
    diff = (R.d_s - R.d_s_merged_run).abs().max()
    print(f"ソースのみの d の，段階 B との差の最大：{diff:.2e}（行 {len(R)}，合併の行が無いもの {int(R.d_m.isna().sum())}）")
    for x in ("h", "m"):
        R[f"pos_{x}"] = (ba(R.d_s, R[f"o_{x}"]) - ba(R.d_s, R.o_s)) * 100
        R[f"dir_{x}"] = (ba(R[f"d_{x}"], R[f"o_{x}"]) - ba(R.d_s, R[f"o_{x}"])) * 100
    R.to_csv(OUT / "e25t_rows.csv", index=False)
    agg = []
    for (kind, t, s, K), g in R.groupby(["kind", "target", "source", "K"]):
        agg.append({"kind": kind, "target": t, "source": s, "K": K,
                    "lam_h": lam_obs(g.o_s, g.o_h), "lam_m": lam_obs(g.o_s, g.o_m),
                    "abs_o_s": g.o_s.abs().mean(), "abs_o_h": g.o_h.abs().mean(), "abs_o_m": g.o_m.abs().mean(),
                    "d_s": g.d_s.mean(), "dchg_h": (g.d_h - g.d_s).mean(), "dchg_m": (g.d_m - g.d_s).mean(),
                    "pos_h": g.pos_h.mean(), "dir_h": g.dir_h.mean(), "pos_m": g.pos_m.mean(), "dir_m": g.dir_m.mean()})
    A = pd.DataFrame(agg)
    lam = pd.concat([only(pd.read_csv(E15_DIR[k] / "e15_lambda.csv", comment="#", dtype={"K": str}), k)
                     for k in kinds])
    A = A.merge(lam[["kind", "target", "source", "K", "lam_pred_rep09"]], on=["kind", "target", "source", "K"], how="left")
    A.to_csv(OUT / "e25t_pairs.csv", index=False)
    S = (A.groupby(["kind", "K"])[["lam_h", "lam_m", "abs_o_s", "abs_o_h", "abs_o_m", "d_s", "dchg_h", "dchg_m",
                                   "pos_h", "dir_h", "pos_m", "dir_m"]].mean().reset_index())
    S["n_pairs"] = A.groupby(["kind", "K"]).size().to_numpy()
    S["frac_lam_h_gt_m"] = A.assign(v=A.lam_h > A.lam_m).groupby(["kind", "K"]).v.mean().to_numpy()
    S["rho_lam_h_pred"] = [stats.spearmanr(g.lam_h, g.lam_pred_rep09, nan_policy="omit")[0]
                           for _, g in A.groupby(["kind", "K"])]
    S["rho_lam_m_pred"] = [stats.spearmanr(g.lam_m, g.lam_pred_rep09, nan_policy="omit")[0]
                           for _, g in A.groupby(["kind", "K"])]
    S.to_csv(OUT / "e25t_summary.csv", index=False)
    tests = []
    for kind in kinds:
        a = A[A.kind == kind].set_index("K")
        m = S[S.kind == kind].set_index("K")
        w = lambda x, y, alt: float(stats.wilcoxon(x, y, alternative=alt, nan_policy="omit").pvalue)  # noqa: E731
        tests += [
            {"kind": kind, "test": "T1", "K": "5", "value_h": m.loc["5", "lam_h"], "value_m": m.loc["5", "lam_m"],
             "holds": bool(m.loc["5", "lam_h"] >= 0.8 and m.loc["5", "lam_h"] > m.loc["5", "lam_m"]),
             "p_wilcoxon_ref": w(a.loc["5", "lam_h"], a.loc["5", "lam_m"], "greater")},
            {"kind": kind, "test": "T2", "K": "1<5<10, 10", "value_h": m.loc["10", "dchg_h"], "value_m": m.loc["10", "dchg_m"],
             "holds": bool(m.loc["1", "dchg_h"] < m.loc["5", "dchg_h"] < m.loc["10", "dchg_h"]
                           and m.loc["10", "dchg_h"] > m.loc["10", "dchg_m"]),
             "p_wilcoxon_ref": w(a.loc["10", "dchg_h"], a.loc["10", "dchg_m"], "greater")},
            {"kind": kind, "test": "T3", "K": "5（位置），1・10（向き）", "value_h": m.loc["5", "pos_h"], "value_m": m.loc["5", "pos_m"],
             "holds": bool(m.loc["5", "pos_h"] > m.loc["5", "pos_m"] and m.loc["1", "dir_h"] <= m.loc["1", "dir_m"]
                           and m.loc["10", "dir_h"] > m.loc["10", "dir_m"]),
             "p_wilcoxon_ref": w(a.loc["5", "pos_h"], a.loc["5", "pos_m"], "greater")},
        ]
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "e25t_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string(index=False))
    print(S.round(3).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["e25_26", "transfer"], default="e25_26")
    ap.add_argument("--kinds", nargs="+", default=list({**MAIN, **TEXT}))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "transfer":
        return transfer(args.kinds)
    pd.set_option("display.width", 250)
    pairs, t25, s25 = [], [], []
    for kind in {**MAIN, **TEXT}:
        w = e25_pairs(kind)
        pairs.append(w)
        t25 += e25_tests(w, kind)
        s25 += e25_summary(w, kind)
    P, T, S = pd.concat(pairs), pd.DataFrame(t25), pd.DataFrame(s25)
    P.to_csv(OUT / "e25_pairs.csv", index=False)
    T.to_csv(OUT / "e25_tests.csv", index=False)
    S.to_csv(OUT / "e25_summary.csv", index=False)
    print("E25 検定\n", T.round(4).to_string(index=False))
    cols = ["kind", "K", "n_pairs", "t", "src"] + H
    print("E25 手ごとの UAR（組の平均）\n", S[cols].round(4).to_string(index=False))
    print("E25 最良になった組の数\n", S[["kind", "K"] + [f"n_best_{h}" for h in H]].to_string(index=False))

    tg, t26 = [], []
    for kind in MAIN:
        g = e26_targets(kind)
        tg.append(g)
        t26 += e26_tests(g, kind)
    G, T6 = pd.concat(tg), pd.DataFrame(t26)
    G.to_csv(OUT / "e26_targets.csv", index=False)
    T6.to_csv(OUT / "e26_tests.csv", index=False)
    S6 = (G.groupby(["kind", "K", "variant"])
          .agg(n_targets=("target", "size"), uar=("uar", "mean"), best1_m=("best1_m", "mean"),
               mean1_m=("mean1_m", "mean"), src_only=("src_only", "mean"), t=("t", "mean"),
               n_sources=("n_sources", "mean"))
          .reset_index())
    S6.to_csv(OUT / "e26_summary.csv", index=False)
    print("E26 検定\n", T6.round(4).to_string(index=False))
    print("E26 まとめ\n", S6.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
