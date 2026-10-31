"""E26：複数のソースを混ぜる（GPU）。混ぜ方と予言は `docs/exploratory-analyses.md` E26。

混ぜ方：全ソース，上位 2 つ（分析用．1 つずつ混ぜたときの合併の UAR で選ぶ），上位 2 つ（平均的な良さ）。
テストと K 本は地図と同じ。同じコーパスの群（IEMOCAP の即興と台本）は混ぜるソースに含めない。
C は，混ぜるソース全体で学んだソースのみの分類器の C（`exp/map.py` の `train`．交差検証で選ぶ）に固定する。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/multisource.py --kind audio_wavlm [--reps 0 … 9]

出力：`results/multisource_<kind>/<ターゲット>.csv`（行＝混ぜ方 × K × 反復）。終わったターゲットは飛ばす。
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
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import logreg  # noqa: E402
import map as M  # noqa: E402
import shift  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from explore_synergy import load_map  # noqa: E402

KS = ("1", "5", "10")


def rankings(kind: str, groups: list[str], reps) -> dict:
    """（ターゲット，K）ごとの候補の並び：分析用（1 つずつの合併の UAR），平均的な良さ（ほかのターゲットでの得）。"""
    mp = load_map(kind)
    mp = mp[mp.rep.isin(reps)]
    single = mp.groupby(["target", "source", "K"]).m.mean()
    gain = mp.assign(g=mp.m - mp.t).groupby(["target", "source", "K"]).g.mean()
    out = {}
    for t in groups:
        cands = [s for s in groups if s != t and GROUP_SPEC[s][0] != GROUP_SPEC[t][0]]
        for K in KS:
            oracle = sorted(cands, key=lambda s: -single[(t, s, K)])
            avg = sorted(cands, key=lambda s: -np.mean([gain[(o, s, K)] for o in groups
                                                        if o not in (t, s) and (o, s, K) in gain.index]))
            out[(t, K)] = {"cands": cands, "上位2つ（分析用）": oracle[:2], "上位2つ（平均的な良さ）": avg[:2],
                           "best1": oracle[0], "best1_m": float(single[(t, oracle[0], K)]),
                           "mean1_m": float(np.mean([single[(t, s, K)] for s in cands]))}
    return out


def run_target(kind: str, tgt: M.Group, data: dict, rank: dict, reps) -> pd.DataFrame:
    split = {}
    for r in reps:
        te = M.test_mask(tgt, r)
        order = M.kshot_order(tgt, np.flatnonzero(~te), r)
        split[r] = (te, {K: M.take_k(order, K) for K in KS})
    nt = len(tgt.y)
    rows = []
    # 混ぜ方ごとのソースの組（全ソースは K によらない．上位 2 つは K ごと）
    sets = {}
    for K in KS:
        info = rank[(tgt.name, K)]
        sets.setdefault(tuple(sorted(info["cands"])), set()).add(("全ソース", K))
        for v in ("上位2つ（分析用）", "上位2つ（平均的な良さ）"):
            sets.setdefault(tuple(sorted(info[v])), set()).add((v, K))
    for srcs, uses in sets.items():
        t0 = time.perf_counter()
        Xs = torch.cat([data[s].X for s in srcs])
        ys = np.concatenate([data[s].y for s in srcs])
        ps = np.concatenate([data[s].person for s in srcs])
        tr = M.train(Xs, ys, ps, np.ones(len(ys), bool), M.SRC_CV_SEED)          # 混ぜるソース全体のソースのみ
        C = tr.C
        X = torch.cat([tgt.X, Xs])
        y = M.labels(np.r_[tgt.y, ys])
        jobs = [(r, K) for r in reps for K in sorted({K for _, K in uses})]
        masks = []
        for r, K in jobs:
            m = np.r_[np.zeros(nt, bool), np.ones(len(ys), bool)]
            m[split[r][1][K]] = True
            masks.append(m)
        fits = logreg.fit_batch(X, y, masks, (C,))
        for (r, K), f in zip(jobs, fits):
            te = split[r][0]
            Xte, y_te = M.rows(tgt.X, te), tgt.y[te]
            u = shift.uar(y_te, f.logits(Xte, C).cpu().numpy().argmax(1))
            u_src = shift.uar(y_te, tr.final.logits(Xte, C).cpu().numpy().argmax(1))
            info = rank[(tgt.name, K)]
            for v, KK in uses:
                if KK != K:
                    continue
                rows.append({"target": tgt.name, "K": K, "rep": r, "variant": v, "sources": "+".join(srcs),
                             "n_sources": len(srcs), "n_src": len(ys), "C": C, "uar": u, "uar_src_only": u_src,
                             "best1": info["best1"], "best1_m": info["best1_m"], "mean1_m": info["mean1_m"]})
        print(f"  {kind} {tgt.name} ← {'+'.join(srcs)}：{time.perf_counter() - t0:.1f} 秒", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / f"multisource_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/multisource.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    groups = groups_for(args.kind)
    data = {g: M.load(g, args.kind) for g in groups}
    rank = rankings(args.kind, groups, args.reps)
    for t in args.target or groups:
        path = out / f"{t}.csv"
        if path.exists():
            print(f"{args.kind} {t}：済み（飛ばす）", flush=True)
            continue
        t0 = time.perf_counter()
        d = run_target(args.kind, data[t], data, rank, args.reps)
        with open(path, "w") as f:
            f.write("# " + json.dumps(prov, ensure_ascii=False) + "\n")
            d.to_csv(f, index=False)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
