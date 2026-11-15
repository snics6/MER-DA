"""【探索的】E35 の追加：ソースごとの向きの縮め方 ρ を振った等しい重みの下書きと，ソースを 1 つずつ抜いた適応後の見込みによる ρ の選び方（CPU）。

規則は `docs/exploratory-analyses.md` E35 の追加（設計と予言 Q35-3・Q35-4 は計算の前に書いた）。
ソース s の向き W_s(ρ) = ((1 − ρ)Σ_s + ρ τ_s I)⁻¹ m_sᵀ（Σ_s は感情の平均を引いた残りの標本共分散，τ_s はその固有値の平均）を，
評価するコーパスの発話でラベル無しの尺度（`explore_source_combine.scale`）にそろえて等しい重みで足す（E32 の 5 節の equal と同じ束ね方）。
ターゲットでの d̄（テストのラベルを使う仕組みの確かめ）と，ソースコーパスを 1 つずつ抜いて仮のターゲットにした適応後の UAR の見込み
（E35 と同じ式．e は E13 の own_d で割り 0.999 で切る．q は E28-9 の q_dir）を ρ ごとに出す。

    ./.venv/bin/python exp/explore_draft_shrink.py [--kinds …] [--targets …]

出力：`results/explore_relation/e35s_curve.csv`（種類 × ターゲット × ρ），`e35s_targets.csv`，`e35s_summary.csv`，`e35s_tests.csv`，
`e35s_provenance.json`
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import explore_compare as EC  # noqa: E402
import explore_source_combine as SC  # noqa: E402
import explore_source_value as SV  # noqa: E402
from data import GROUP_SPEC, load_any  # noqa: E402

RHOS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.97, 1.0]
KS_LOSS = (1, 5, 10, 25)


def shrunk_dirs(Z: np.ndarray, y: np.ndarray) -> dict:
    """ρ → (4, 次元) の縮めた向き ((1 − ρ)Σ + ρτI)⁻¹ m_kᵀ。"""
    m = np.stack([Z[y == k].mean(0) for k in range(4)])
    R = Z - m[y]
    lam, V = np.linalg.eigh(R.T @ R / (len(y) - 4))
    lam = np.clip(lam, 0.0, None)
    tau = float(lam.mean())
    P = V.T @ m.T                                                       # (次元, 4)
    return {rho: (V @ (P / ((1 - rho) * lam + rho * tau)[:, None])).T for rho in RHOS}


def draft_scores(Z: np.ndarray, comps: list[np.ndarray]) -> np.ndarray:
    Ls = [Z @ W.T for W in comps]
    return sum(L / SC.scale(L) for L in Ls) / len(Ls)


def adapt_uar(own: np.ndarray, e: np.ndarray, q: float) -> float:
    """適応後の UAR の見込み（K ∈ KS_LOSS の平均．E35 と同じ模型）。"""
    return float(np.mean([SV.bridge(SV.d_star(own, e, q, K) / (2 * SV.jitter(K, True))) for K in KS_LOSS]))


def run_target(kind: str, t: str, qd: pd.Series, own_of: dict) -> list[dict]:
    t0 = time.perf_counter()
    zm = np.load(ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz")
    srcs = [str(s) for s in zm["src_names"]]
    reps = [r for r in EC.REPS if f"s_all_r{r}" in zm.files]
    data = {s: load_any(s, kind) for s in srcs}
    Xall = np.concatenate([X for _, X in data.values()])
    mu, sd = Xall.mean(0), Xall.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    del Xall
    Zs = {s: (X - mu) / sd for s, (_, X) in data.items()}
    ys = {s: df["label_4"].to_numpy() for s, (df, _) in data.items()}
    W = {s: shrunk_dirs(Zs[s], ys[s]) for s in srcs}
    # ターゲットのテストでの d̄（反復で平均）
    dfT, XT = load_any(t, kind)
    yT = dfT["label_4"].to_numpy()
    d_t = {rho: [] for rho in RHOS}
    for r in reps:
        idx, y = zm[f"test_idx_r{r}"], zm[f"y_test_r{r}"]
        if not np.array_equal(yT[idx], y):
            raise ValueError(f"テストのラベルが合わない: {kind} {t} r{r}")
        Z = (XT[idx] - mu) / sd
        for rho in RHOS:
            d_t[rho].append(SC.d_pairs(draft_scores(Z, [W[s][rho] for s in srcs]), y))
    # ソースを 1 つずつ抜いた見込み（ターゲットのラベルは使わない）
    loco_u = {rho: [] for rho in RHOS}
    loco_e = {rho: [] for rho in RHOS}
    for c in srcs:
        rest = [s for s in srcs if s != c and GROUP_SPEC[s][0] != GROUP_SPEC[c][0]]
        own = own_of[c]
        for rho in RHOS:
            d = SC.d_pairs(draft_scores(Zs[c], [W[s][rho] for s in rest]), ys[c])
            e = np.clip(d / own, 0.0, 0.999)
            loco_u[rho].append(adapt_uar(own, e, float(qd[c])))
            loco_e[rho].append(float(e.mean()))
    rows = [{"kind": kind, "target": t, "rho": rho, "d_target": float(np.mean(d_t[rho], axis=0).mean()),
             "loco_uar": float(np.mean(loco_u[rho])), "loco_e": float(np.mean(loco_e[rho])), "n_sources": len(srcs)}
            for rho in RHOS]
    print(f"  {kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return rows


def per_target(C: pd.DataFrame, e35: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (kind, t), g in C.groupby(["kind", "target"]):
        g = g.set_index("rho")
        rs, rse, rb = g.loco_uar.idxmax(), g.loco_e.idxmax(), g.d_target.idxmax()
        ref = e35[(e35.kind == kind) & (e35.target == t)].set_index("draft").d_mean
        out.append({"kind": kind, "target": t, "rho_star": rs, "rho_star_e": rse, "rho_best": rb,
                    "d_star": g.d_target[rs], "d_star_e": g.d_target[rse], "d_best": g.d_target[rb],
                    "d_rho1": g.d_target[1.0], "d_rho_min": g.d_target[RHOS[0]],
                    "d_equal": float(ref["equal"]), "d_pooled": float(ref["pooled"]),
                    "log_gap": abs(np.log10(rs) - np.log10(rb))})
    return pd.DataFrame(out)


def summarize(T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    srows, trows = [], []
    for kind, g in T.groupby("kind"):
        row = {"kind": kind, "targets": len(g), "rho_best_median": float(g.rho_best.median()),
               "rho_star_median": float(g.rho_star.median()), "log_gap_median": float(g.log_gap.median())}
        for c in ("d_star", "d_star_e", "d_best", "d_rho1", "d_rho_min", "d_equal", "d_pooled"):
            row[c] = float(g[c].median())
        for a in ("d_star", "d_star_e", "d_best", "d_rho1"):
            diff = g[a] - g.d_equal
            row[f"{a}-equal"] = float(diff.median())
            row[f"{a}_wins"] = int((diff > 0).sum())
        srows.append(row)
        diff = g.d_star - g.d_equal
        trows += [{"kind": kind, "test": "Q35-3", "value": float(g.rho_best.median()), "pass": bool(g.rho_best.median() >= 0.3)},
                  {"kind": kind, "test": "Q35-4", "value": float(diff.median()), "wins": int((diff > 0).sum()),
                   "targets": len(g), "pass": bool(diff.median() > 0)}]
    R = pd.DataFrame(trows)
    for test in ("Q35-3", "Q35-4"):
        h = R[(R.test == test) & R.kind.isin(["audio_wavlm", "audio_hubert"])]
        R.loc[R.test == test, "verdict"] = "支持" if len(h) == 2 and h["pass"].all() else "支持されない"
    return pd.DataFrame(srows), R


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=SV.MAIN)
    ap.add_argument("--targets", nargs="+")
    args = ap.parse_args()
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/explore_draft_shrink.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "exp", "measure")),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "RHOS": RHOS, "KS_LOSS": list(KS_LOSS), "reps": list(EC.REPS)}
    rows = []
    for kind in args.kinds:
        e13 = SV.load_e13(kind)
        own_of = {g: h.groupby("pair").own_d.mean().reindex(SV.PAIR_NAMES).to_numpy() for g, h in e13.groupby("target")}
        qd = pd.read_csv(SV.FIX / "e28t9_targets.csv").query("kind == @kind").set_index("target").q_dir
        targets = [t for t in sorted(qd.index) if (ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz").exists()]
        if args.targets:
            targets = [t for t in targets if t in args.targets]
        for t in targets:
            rows += run_target(kind, t, qd, own_of)
    C = pd.DataFrame(rows)
    T = per_target(C, pd.read_csv(SV.OUT / "e35_targets.csv"))
    S, R = summarize(T)
    prov["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    sfx = "" if not args.targets and args.kinds == SV.MAIN else "_partial"
    C.to_csv(SV.OUT / f"e35s_curve{sfx}.csv", index=False)
    T.to_csv(SV.OUT / f"e35s_targets{sfx}.csv", index=False)
    S.to_csv(SV.OUT / f"e35s_summary{sfx}.csv", index=False)
    R.to_csv(SV.OUT / f"e35s_tests{sfx}.csv", index=False)
    (SV.OUT / f"e35s_provenance{sfx}.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 250)
    print(S.round(3).T.to_string())
    print(R.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
