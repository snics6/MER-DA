"""【探索的】E50（GPU）：ラベル無しで選ぶ既存の能動学習の方法（E46）を，生徒を E39 の学習器に替えて測り直す。

規則は `docs/exploratory-analyses.md` E50（設計と予言は計算の前に書いた）。生徒と設定は E49 と同じ（E40 の生徒．感情の重みは present_weights．
判定には E40 の random の組の設定を使い，teach の組の ρ_M の生徒でも解いて残す）。選び方は E46 と同じ作りで，下書きの確率だけ E39 の下書きのものに替える：
  margin・badge       ：E39 の下書きの確率で選ぶ
  margin_p4・badge_p4 ：P4 の出発点の確率で選ぶ（E46 と同じ K 本．生徒だけ替わる）
  coreset・typiclust  ：特徴だけで選ぶ（E46 と同じ K 本）
比べる相手は E49 の random_pool・herding_pred・random_true・LLL・C:0000・C:0001（同じ生徒）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/al_student39.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/al_student39.py --analyze

出力：`results/al_student39_<kind>/<ターゲット>.csv`（policy, rho_M, cb, cp, rep, K, uar），`<ターゲット>_sel.csv`，`provenance.json`，
`provenance_log.jsonl`。判定は `results/explore_relation/e50_targets.csv`・`e50_tests.csv`
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from belief_check import knn  # noqa: E402
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_baselines import K_TYP, order_badge, order_coreset, order_margin, pick_typiclust  # noqa: E402
from teach_student39 import Student, solve_students  # noqa: E402

POLICIES = ["margin", "badge", "margin_p4", "badge_p4", "coreset", "typiclust"]
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"


def run_target(kind, T, data, reps, ks, out, cache):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    sel = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"]).loc[(kind, T)]
    rhos_m = sorted({float(sel.t_rho_M), float(sel.r_rho_M)})
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {p: {} for p in POLICIES}
    srows, rows, students = [], [], {}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, float(sel.gamma), float(sel.rho_s), rhos_m)
        students[r] = st
        P39 = torch.softmax(st.L0, 1).float()
        P4 = torch.softmax(src_pc.logits(Xc, C_T), 1).float()
        X = st.Xstd.float()                                                    # x̃（E46 と同じ座標）
        gen = lambda name: torch.Generator().manual_seed(zlib.crc32(f"e46|{name}|{T}|{r}".encode()))  # noqa: E731
        orders = {"margin": order_margin(P39), "badge": order_badge(X, P39, 4 * kmax, gen("badge")),
                  "margin_p4": order_margin(P4), "badge_p4": order_badge(X, P4, 4 * kmax, gen("badge")),
                  "coreset": order_coreset(X, 4 * kmax)}
        Xn = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
        nb = knn(X, K_TYP)
        typ = 1.0 / (1.0 - (Xn[:, None, :] * Xn[nb]).sum(2)).mean(1).clamp(min=1e-6)
        for K in ks:
            n = 4 * int(K)
            picks = {p: orders[p][:n] for p in orders}
            picks["typiclust"] = pick_typiclust(X, typ, n, gen(f"typiclust{K}"))
            for p, pp in picks.items():
                pp = np.array(pp)
                if len(np.unique(pp)) != n:
                    raise ValueError(f"選んだ本数がおかしい: {p} r{r} K{K}")
                idx[p][(r, K)] = pool[pp]
                cnt = np.bincount(yq[pp], minlength=4)
                srows.append({"target": T, "policy": p, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                              "min_count": int(cnt.min()), "max_count": int(cnt.max())})
    solve_students(T, ev, ctx, reps, ks, students, idx, rhos_m, C_T, rows)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN) -> int:
    sel = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"])
    e49 = pd.read_csv(OUT_REL / "e49_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    e46 = pd.read_csv(OUT_REL / "e46_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    ref = ["random_pool", "herding_pred", "herding_pred_p4", "random_true", "LLL", "C:0000", "C:0001"]
    tabs = []
    for kind in kinds:
        d = ROOT / "results" / f"al_student39_{kind}"
        for x in sorted(f for f in d.glob("*.csv") if not f.stem.endswith("_sel")):
            T = x.stem
            a = pd.read_csv(x, dtype={"K": str, "cb": str})
            a = a[a.rep.isin(EC.REPS)]
            u = a.groupby(["policy", "rho_M", "cb", "cp", "K"]).uar.mean() * 100
            s = sel.loc[(kind, T)]
            rk = (float(s.r_rho_M), f"{float(s.r_cb):g}", float(s.r_cp))
            for K in ("1", "5", "10", "25"):
                row = {"kind": kind, "target": T, "K": K}
                for p in POLICIES:
                    row[p] = float(u.loc[(p,) + rk + (K,)])
                for p in ref:
                    row[p] = float(e49.loc[(kind, T, K), p])
                for p in ("margin", "coreset", "typiclust", "badge"):
                    row[f"P4:{p}"] = float(e46.loc[(kind, T, K), p])
                tabs.append(row)
    D = pd.DataFrame(tabs)
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for x in POLICIES + ref:
            row[x] = float(g[x].mean())
        for p in POLICIES + ["herding_pred", "C:0001"]:
            diff = g[p] - g.random_pool
            row[f"{p}-random_pool"] = float(diff.mean())
            row[f"{p}>random_pool"] = int((diff > 0).sum())
        for p in POLICIES:
            row[f"C:0001-{p}"] = float((g["C:0001"] - g[p]).mean())
        for p in ("margin", "coreset", "typiclust", "badge"):
            row[f"{p}-P4:{p}"] = float((g[p] - g[f"P4:{p}"]).mean())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in ("audio_wavlm", "audio_hubert"):
        k5 = R[(R.kind == kind) & (R.K == "5")]
        if len(k5):
            k5 = k5.iloc[0]
            R.loc[R.kind == kind, "Z1"] = bool(max(k5[f"{p}-random_pool"] for p in ("margin", "coreset", "typiclust", "badge")) < 1.0)
            R.loc[R.kind == kind, "Z2"] = bool(k5["coreset-random_pool"] < 0)
    D.to_csv(OUT_REL / "e50_targets.csv", index=False)
    R.to_csv(OUT_REL / "e50_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in R if c in ("Z1", "Z2")]).round(2).T.to_string())
    vc = [c for c in ("Z1", "Z2") if c in R]
    if vc:
        print(R.dropna(subset=vc).drop_duplicates("kind")[["kind"] + vc].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=MAIN)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds)
    out = args.out_dir or ROOT / "results" / f"al_student39_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    prov = {"script": "exp/al_student39.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "policies": POLICIES,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, args.reps, args.ks, out, cache)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
