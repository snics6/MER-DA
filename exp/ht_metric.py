"""【探索的】E38（GPU）：ラベル無しのターゲットの散らばりで罰則の計量を作った仮説転移。白色化の度合い ρ の K の法則はメタ学習。

規則は `docs/exploratory-analyses.md` E38（設計と予言 R38-1〜3 は計算の前に書いた）。評価・候補・中心・K 本・下書き（P4）・学習器は
E31 の P4 と同じ。反復ごとに，ターゲットのテスト以外の発話（ラベル無し）の x̃（下書きの統計で標準化した中心合わせ後の特徴）の共分散
Σ から M_ρ = (1 − ρ)Σ/τ + ρI を作り，特徴を x′ = x̃ M_ρ^{−1/2}，下書きの重みを W_s M_ρ^{1/2} に直して（点数は同じ）解く。
ρ = 1 は P4 と同じ。ターゲット（role=target，反復 0〜9）と，T のソースコーパス c を仮のターゲットにした分（role=pseudo．T と c の
コーパスの群を除くソースの合併を下書き，C = 0.01，反復 0〜4）を解く。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/ht_metric.py --kind audio_wavlm [--target …]
    ./.venv/bin/python exp/ht_metric.py --analyze

出力：`results/ht_metric_<kind>/<ターゲット>.csv`（role, meta_for, eval, rho, rep, K, uar），`provenance.json`，`provenance_log.jsonl`。
判定は `results/explore_relation/e38_targets.csv`・`e38_tests.csv`
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402

RHOS = (1.0, 0.97, 0.9, 0.8, 0.7, 0.5, 0.3)
RHO_INF = (0.3, 0.5, 0.7, 0.8, 0.9, 0.97)
K0S = (1, 3, 10, 30, 100)
PSEUDO_REPS = [0, 1, 2, 3, 4]
C_PSEUDO = 0.01
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"


def ht_metric(ctx, r, src, C_s, add):
    """反復 r で，ρ ごとに計量を変えた仮説転移（K ∈ ctx.ks をまとめて解く）。"""
    tgt = ctx.tgt
    dev, dt = src.mu.device, src.mu.dtype
    pool = np.flatnonzero(~ctx.te[r])
    Zp = (M.rows(tgt.X, pool) - ctx.ct[r] - src.mu) / src.sd
    Zp = Zp - Zp.mean(0)
    lam, V = torch.linalg.eigh(Zp.T @ Zp / len(pool))
    lam = lam.clamp(min=0.0)
    tau = lam.mean()
    used = np.unique(np.concatenate([ctx.idx[(r, K)] for K in ctx.ks]))
    Xu = (M.rows(tgt.X, used) - ctx.ct[r] - src.mu) / src.sd
    yu = M.labels(tgt.y[used])
    jobs, masks = CB.jobs_for(ctx, [r], {int(i): k for k, i in enumerate(used)}, len(used), False)
    zero = torch.zeros(Xu.shape[1], dtype=dt, device=dev)
    one = torch.ones(Xu.shape[1], dtype=dt, device=dev)
    for rho in RHOS:
        ev = (1 - rho) * lam / tau + rho
        A = (V * ev.rsqrt()) @ V.T                                   # M^{-1/2}
        Mh = (V * ev.sqrt()) @ V.T                                   # M^{1/2}
        anchor = logreg.Fitted(zero, one, {C_s: src.W[C_s] @ Mh}, {C_s: src.b[C_s]}, {C_s: 0}, {C_s: 0.0})
        fits = logreg.fit_batch(Xu @ A, yu, masks, (CB.DEF_CP,), anchor=anchor, anchor_c=C_s, stats=(zero, one),
                                Cb=[CB.DEF_CB] * len(masks))
        X_of = lambda r_, ii, A=A: ((M.rows(tgt.X, ii) - ctx.ct[r] - src.mu) / src.sd) @ A  # noqa: E731
        CB.collect(ctx, jobs, fits, (CB.DEF_CP,), X_of, None, lambda rr, K, c, u, cv, L, rho=rho, **kw: add(rr, K, rho, u), {})


def evaluate(role, meta_for, ev, src, C, reps, ks, rows):
    ctx = CB.Ctx(ev, reps, ks, center="ind")

    def add(r, K, rho, u):
        rows.append({"role": role, "meta_for": meta_for, "eval": ev.name, "rho": rho, "rep": r, "K": K, "uar": u})
    for r in reps:
        ht_metric(ctx, r, src, C, add)


def run_target(kind, T, data, reps, ks, out, cache):
    t0 = time.perf_counter()
    rows = []
    corp = lambda g: GROUP_SPEC[g][0]  # noqa: E731
    pool_T = [s for s in data if corp(s) != corp(T)]
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])

    def draft(pool, C):
        key = (tuple(sorted(pool)), C)
        if key not in cache:
            Xp = torch.cat([data[s].X for s in pool])
            yp = M.labels(np.concatenate([data[s].y for s in pool]))
            cache[key] = logreg.fit(Xp - Xp.mean(0), yp, (C,))
            del Xp
        return cache[key]

    evaluate("target", T, data[T], draft(pool_T, C_T), C_T, reps, ks, rows)
    t1 = time.perf_counter()
    for c in pool_T:
        pool_c = [s for s in pool_T if corp(s) != corp(c)]
        evaluate("pseudo", T, data[c], draft(pool_c, C_PSEUDO), C_PSEUDO, PSEUDO_REPS, ks, rows)
    print(f"  {kind} {T}：ターゲット {t1 - t0:.0f} 秒，仮のターゲット {len(pool_T)} 個 {time.perf_counter() - t1:.0f} 秒", flush=True)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def snap(x: float) -> float:
    return min(RHOS, key=lambda v: abs(v - x))


def law(rho_inf: float, k0: float, K: int) -> float:
    return snap(1 - (1 - rho_inf) * K / (K + k0))


def analyze() -> int:
    tabs = []
    for kind in MAIN:
        fs = sorted((ROOT / "results" / f"ht_metric_{kind}").glob("*.csv"))
        if not fs:
            continue
        a = pd.concat([pd.read_csv(f, dtype={"K": str}) for f in fs])
        ps = a[a.role == "pseudo"].groupby(["meta_for", "rho", "K"]).uar.mean()
        tg = a[(a.role == "target") & a.rep.isin(EC.REPS)].groupby(["eval", "rho", "K"]).uar.mean() * 100
        for T in sorted(a.meta_for.unique()):
            P = ps.loc[T].unstack("K")                                            # ρ × K
            best_law = max(itertools.product(RHO_INF, K0S),
                           key=lambda g: np.mean([P.loc[law(g[0], g[1], int(K)), K] for K in ("1", "5", "10", "25")]))
            rho_const = P[["1", "5", "10", "25"]].mean(1).idxmax()
            Tt = tg.loc[T].unstack("K")
            for K in ("1", "5", "10", "25"):
                rl = law(*best_law, int(K))
                tabs.append({"kind": kind, "target": T, "K": K, "rho_inf": best_law[0], "K0": best_law[1], "rho_law": rl,
                             "rho_perK": float(P[K].idxmax()), "rho_const": float(rho_const), "rho_oracle": float(Tt[K].idxmax()),
                             "P4": float(Tt.loc[1.0, K]), "law": float(Tt.loc[rl, K]), "perK": float(Tt.loc[P[K].idxmax(), K]),
                             "const": float(Tt.loc[rho_const, K]), "oracle": float(Tt[K].max())})
    D = pd.DataFrame(tabs)
    rows = []
    for (kind, K), g in D.groupby(["kind", "K"]):
        row = {"kind": kind, "K": K, "targets": len(g), "P4": float(g.P4.mean())}
        for m in ("law", "perK", "const", "oracle"):
            diff = g[m] - g.P4
            row[f"{m}-P4"] = float(diff.mean())
            row[f"{m}>P4"] = int((diff > 0).sum())
        row["rho_oracle_median"] = float(g.rho_oracle.median())
        row["rho_law_median"] = float(g.rho_law.median())
        rows.append(row)
    R = pd.DataFrame(rows)
    for kind in ("audio_wavlm", "audio_hubert"):
        g = R[R.kind == kind].set_index("K")
        if set(g.index) >= {"1", "5", "10", "25"}:
            R.loc[R.kind == kind, "R38-1"] = bool(g.loc["10", "law-P4"] > 0 and g.loc["25", "law-P4"] > 0)
            R.loc[R.kind == kind, "R38-2"] = bool(g.loc["1", "law-P4"] >= -0.3 and g.loc["5", "law-P4"] >= -0.3)
            R.loc[R.kind == kind, "R38-3"] = bool(g.loc["25", "rho_oracle_median"] < g.loc["1", "rho_oracle_median"])
    D.to_csv(OUT_REL / "e38_targets.csv", index=False)
    R.to_csv(OUT_REL / "e38_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print(D.groupby("kind")[["rho_inf", "K0"]].agg(lambda s: s.value_counts().idxmax()).to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        return analyze()
    out = args.out_dir or ROOT / "results" / f"ht_metric_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/ht_metric.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "pseudo_reps": PSEUDO_REPS,
            "rhos": list(RHOS), "rho_inf": list(RHO_INF), "K0": list(K0S), "C_pseudo": C_PSEUDO, "cp": CB.DEF_CP, "cb": CB.DEF_CB,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    cache: dict = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, args.reps, args.ks, out, cache)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
