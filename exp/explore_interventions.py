"""【探索的】E25（手の比較）と E26（複数のソースを混ぜる）の解析。新しい学習はしない。

予言と解析の細部は `docs/exploratory-analyses.md` E25・E26 の 1 節（計算の前に書いた）。
入力は `exp/interventions.py` と `exp/multisource.py` の出力（C を固定した地図で解いたもの），
式の λ と位置による損は C を固定した地図の E15 の出力，ターゲットのみ・ソースのみは C を固定した地図。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_interventions.py

出力：`OUT/e25_pairs.csv`（組 × K の手ごとの UAR と λ・位置による損），`OUT/e25_tests.csv`，`OUT/e25_summary.csv`，
`OUT/e26_targets.csv`（ターゲット × K × 混ぜ方），`OUT/e26_tests.csv`，`OUT/e26_summary.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
from explore_synergy import KS, load_map  # noqa: E402
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
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
