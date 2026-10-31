"""E28 の 9 節：ターゲットのみの分類器（K 本だけ）を地図と同じ条件で学び直し，テストの点数を保存する（GPU，小さい）。

C は地図（C を固定した地図でも同じ値）のターゲットのみの行の C．標準化は地図と同じく K 本の行の統計。
確かめとして UAR を地図の値と並べる。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/tgt_scores.py --kind audio_wavlm

出力：`results/tgt_scores_<kind>/<ターゲット>.npz`（`test_idx_r{r}`・`y_test_r{r}`・`t_K{K}_r{r}`），`<ターゲット>.csv`，`provenance.json`
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
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
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402

KS = ("1", "5", "10")


def tgt_rows(kind: str, target: str) -> dict:
    d = pd.read_csv(map_dir(kind) / f"{target}.csv", comment="#", dtype={"K": str, "source": str},
                    usecols=["method", "K", "rep", "C", "uar"])
    d = d[d.method == "tgt_only"]
    return {(r.K, int(r.rep)): (float(r.C), float(r.uar)) for r in d.itertuples()}


def run_target(kind: str, tgt: M.Group, reps, out: Path) -> pd.DataFrame:
    ref = tgt_rows(kind, tgt.name)
    yt = M.labels(tgt.y)
    nt = len(tgt.y)
    split, saved, rows = {}, {}, []
    for r in reps:
        te = M.test_mask(tgt, r)
        order = M.kshot_order(tgt, np.flatnonzero(~te), r)
        split[r] = (te, {K: M.take_k(order, K) for K in KS})
        saved[f"test_idx_r{r}"] = np.flatnonzero(te)
        saved[f"y_test_r{r}"] = tgt.y[te]
    jobs = defaultdict(list)
    for r in reps:
        for K in KS:
            C, _ = ref[(K, r)]
            m = np.zeros(nt, bool)
            m[split[r][1][K]] = True
            jobs[C].append((r, K, m))
    fits = {}
    for C, js in jobs.items():
        used = np.unique(np.concatenate([np.flatnonzero(m) for _, _, m in js]))
        Xu, yu = M.rows(tgt.X, used), yt[torch.as_tensor(used, device=yt.device)]
        fb = logreg.fit_batch(Xu, yu, [m[used] for _, _, m in js], (C,))
        for (r, K, _), f in zip(js, fb):
            fits[(r, K)] = (f, C)
    for r in reps:
        te = split[r][0]
        Xte, y_te = M.rows(tgt.X, te), tgt.y[te]
        for K in KS:
            f, C = fits[(r, K)]
            L = f.logits(Xte, C).cpu().numpy()
            saved[f"t_K{K}_r{r}"] = L.astype(np.float32)
            rows.append({"target": tgt.name, "rep": r, "K": K, "C": C, "uar": shift.uar(y_te, L.argmax(1)),
                         "uar_map": ref[(K, r)][1], "n_iter": int(f.n_iter[C]), "grad": float(f.grad[C])})
    np.savez_compressed(out / f"{tgt.name}.npz", **saved)
    d = pd.DataFrame(rows)
    d.to_csv(out / f"{tgt.name}.csv", index=False)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / f"tgt_scores_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    (out / "provenance.json").write_text(json.dumps({
        "script": "exp/tgt_scores.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "map_dir": str(map_dir(args.kind)), "K": list(KS), "reps": args.reps}, ensure_ascii=False, indent=1))
    for t in args.target or groups_for(args.kind):
        if (out / f"{t}.npz").exists():
            print(f"{args.kind} {t}：済み（飛ばす）", flush=True)
            continue
        t0 = time.perf_counter()
        d = run_target(args.kind, M.load(t, args.kind), args.reps, out)
        diff = (d.uar - d.uar_map).abs()
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒．地図の UAR との差 最大 {diff.max():.1e}，"
              f"0.001 を超えた条件 {int((diff > 1e-3).sum())}／{len(d)}", flush=True)


if __name__ == "__main__":
    main()
