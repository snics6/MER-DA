"""E25 の手 7（仮説転移）を学び直し，ターゲットのテストでの点数を保存する（GPU）。

目的は `docs/exploratory-analyses.md` E25 の 2 節の読み（仮説転移では位置のずれがほぼ全部直り，
向きは K とともに良くなる）を，E15 の段階 B と同じ d・o の測り方で確かめること。
学び方は `exp/interventions.py` の手 7 と同じ：ソースのみの分類器（C ＝ 同じ反復のソースのみの C）を出発点にし，
ターゲットの K 本だけで，罰則 ‖W − W_s‖²/(2C) をかけて解く。テストと K 本は地図と同じ。
確かめとして，UAR を `results/interventions_<kind>/` の手 7 の値と並べて残す。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/transfer_scores.py --kind audio_wavlm [--target …] [--source …]

出力：`results/transfer_scores_<kind>/<ターゲット>.npz`（`test_idx_r{r}`・`y_test_r{r}`，
`s_{ソース}_r{r}`：ソースのみの点数，`h7_{ソース}_K{K}_r{r}`：仮説転移の点数），`<ターゲット>.csv`（UAR と
手 7 の UAR），`provenance.json`。終わったターゲットは飛ばす（`--overwrite` で作り直す）。
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
import logreg  # noqa: E402
import map as M  # noqa: E402
import shift  # noqa: E402
from data import groups_for  # noqa: E402
from interventions import map_rows  # noqa: E402
from pair_table import map_dir  # noqa: E402

KS = ("1", "5", "10")
HAND7 = "7 仮説転移"


def hand7_ref(kind: str, target: str) -> dict:
    d = pd.read_csv(ROOT / "results" / f"interventions_{kind}" / f"{target}.csv", comment="#",
                    dtype={"K": str, "source": str})
    d = d[d.method == HAND7]
    return {(r.source, r.K, int(r.rep)): float(r.uar) for r in d.itertuples()}


def run_target(kind: str, tgt: M.Group, sources: list[M.Group], reps, out: Path) -> pd.DataFrame:
    ref, iv = map_rows(kind, tgt.name), hand7_ref(kind, tgt.name)
    Xt, yt = tgt.X, M.labels(tgt.y)
    nt = len(tgt.y)
    split, saved, rows = {}, {}, []
    for r in reps:
        te = M.test_mask(tgt, r)
        order = M.kshot_order(tgt, np.flatnonzero(~te), r)
        split[r] = (te, {K: M.take_k(order, K) for K in KS}, np.unique(tgt.person[te]))
        saved[f"test_idx_r{r}"] = np.flatnonzero(te)
        saved[f"y_test_r{r}"] = tgt.y[te]
    for s in sources:
        t0 = time.perf_counter()
        groups = {}
        for r in reps:                                        # ソースの行と C が同じ反復を束ねる（interventions.py と同じ）
            keep = ~np.isin(s.person, split[r][2])
            C = ref[("src_only", s.name, "", r)][0]
            groups.setdefault((keep.tobytes(), C), []).append(r)
        for (kb, C), rs in groups.items():
            keep = np.frombuffer(kb, dtype=bool)
            src = logreg.fit(M.rows(s.X, keep), M.labels(s.y[keep]), (C,))
            jobs = [(r, K) for r in rs for K in KS]
            masks = []
            for r, K in jobs:
                m = np.zeros(nt, bool)
                m[split[r][1][K]] = True
                masks.append(m)
            fits = dict(zip(jobs, logreg.fit_batch(Xt, yt, masks, (C,), anchor=src, anchor_c=C)))
            for r in rs:
                te = split[r][0]
                Xte, y_te = M.rows(Xt, te), tgt.y[te]
                L = src.logits(Xte, C).cpu().numpy()
                saved[f"s_{s.name}_r{r}"] = L.astype(np.float32)
                base = {"target": tgt.name, "source": s.name, "rep": r, "C": C}
                rows.append({**base, "K": "", "method": "src_only", "uar": shift.uar(y_te, L.argmax(1)),
                             "uar_ref": ref[("src_only", s.name, "", r)][1]})
                for K in KS:
                    f = fits[(r, K)]
                    L7 = f.logits(Xte, C).cpu().numpy()
                    saved[f"h7_{s.name}_K{K}_r{r}"] = L7.astype(np.float32)
                    rows.append({**base, "K": K, "method": "transfer", "uar": shift.uar(y_te, L7.argmax(1)),
                                 "uar_ref": iv.get((s.name, K, r), np.nan),
                                 "n_iter": int(f.n_iter[C]), "grad": float(f.grad[C])})
        print(f"  {kind} {tgt.name} ← {s.name}：{time.perf_counter() - t0:.1f} 秒（束 {len(groups)}）", flush=True)
    np.savez_compressed(out / f"{tgt.name}.npz", **saved)
    d = pd.DataFrame(rows)
    d.to_csv(out / f"{tgt.name}.csv", index=False)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--source", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / f"transfer_scores_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    (out / "provenance.json").write_text(json.dumps({
        "script": "exp/transfer_scores.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "map_dir": str(map_dir(args.kind)), "K": list(KS), "reps": args.reps,
        "solver": {"tol": logreg.TOL, "device": str(logreg.DEVICE)}}, ensure_ascii=False, indent=1))
    t0 = time.perf_counter()
    groups = groups_for(args.kind)
    data = {g: M.load(g, args.kind) for g in groups}
    print(f"{args.kind}：読み込み {time.perf_counter() - t0:.0f} 秒", flush=True)
    for t in args.target or groups:
        if (out / f"{t}.npz").exists() and not args.overwrite:
            print(f"{args.kind} {t}：済み（飛ばす）", flush=True)
            continue
        t1 = time.perf_counter()
        d = run_target(args.kind, data[t], [data[s] for s in (args.source or groups) if s != t], args.reps, out)
        diff = (d.uar - d.uar_ref).abs()
        print(f"{args.kind} {t}：{time.perf_counter() - t1:.0f} 秒．手 7（または地図）の UAR との差 最大 {diff.max():.2e}，"
              f"0.001 を超えた条件 {int((diff > 1e-3).sum())}／{len(d)}", flush=True)


if __name__ == "__main__":
    main()
