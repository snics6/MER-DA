"""E15 の段階 B：合併とソースのみの分類器を地図と同じ条件で学び直し，ターゲットのテストでの点数を保存する。

目的は `docs/exploratory-analyses.md` E15 の 6 節（段階 B）。地図（`exp/map.py`）は UAR しか残して
いないので，合併の分類器の向き d と位置 o を感情の対ごとに測れない。ここでは地図と同じ

- テストの袋（`map.test_mask`），K 本（`map.kshot_order`・`take_k`．K をまたいで入れ子），
- ソースの行（同じコーパスの組はテストの人物を外す），
- 学習器（`logreg.fit_batch`．学習データで標準化，感情を均等にする重み），
- C（**地図で選ばれた値に固定**．交差検証はしない）

で解き直し，テストの点数（4 感情の対数オッズ）を保存する。W について狭義凸なので，地図と同じ C なら
地図と同じ分類器になる（温め直しの有無は解に影響しない）。確かめとして UAR を地図の値と並べて残す。

同じ（ターゲット，ソース）の学習データ（反復 × K）は，C が同じものを束ねて一度に解く。

    ./.venv/bin/python exp/merged_scores.py --kind audio_wavlm --target CREMA-D --source RAVDESS   # 試走
    ./.venv/bin/python exp/merged_scores.py --kind audio_wavlm                                      # 本番

出力：`results/merged_scores_<kind>/<ターゲット>.npz`（`test_idx_r{r}`・`y_test_r{r}`，
`m_{ソース}_K{K}_r{r}`・`s_{ソース}_r{r}`：テストの点数），`<ターゲット>.csv`（条件ごとの C，
UAR，地図の UAR），`provenance.json`。終わったターゲットは飛ばす（`--overwrite` で作り直す）。
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
from pair_table import OUT_ROOT, map_dir  # noqa: E402

KS = ("1", "5", "10")
REPS = tuple(range(10))


def map_rows(kind: str, target: str) -> dict:
    """地図の (method, source, K, rep) → (C, uar)。"""
    d = pd.read_csv(map_dir(kind) / f"{target}.csv", comment="#",
                    dtype={"K": str, "source": str}, usecols=["method", "source", "K", "rep", "C", "uar"])
    d["K"], d["source"] = d.K.fillna(""), d.source.fillna("")
    return {(r.method, r.source, r.K, int(r.rep)): (float(r.C), float(r.uar)) for r in d.itertuples()}


def run_target(kind: str, tgt: M.Group, sources: list[M.Group], reps, out: Path) -> pd.DataFrame:
    ref = map_rows(kind, tgt.name)
    saved, rows = {}, []
    split = {}
    for r in reps:
        te = M.test_mask(tgt, r)
        order = M.kshot_order(tgt, np.flatnonzero(~te), r)
        split[r] = (te, {K: M.take_k(order, K) for K in KS}, np.unique(tgt.person[te]))
        saved[f"test_idx_r{r}"] = np.flatnonzero(te)
        saved[f"y_test_r{r}"] = tgt.y[te]
    nt = len(tgt.y)
    for s in sources:
        t0 = time.perf_counter()
        X = torch.cat([tgt.X, s.X])
        y = M.labels(np.r_[tgt.y, s.y])
        jobs = defaultdict(list)                      # C → [(名前, method, K, rep, mask)]
        src_seen = {}
        for r in reps:
            te, idx, persons = split[r]
            keep = ~np.isin(s.person, persons)
            src_mask = np.r_[np.zeros(nt, bool), keep]
            C_s, _ = ref[("src_only", s.name, "", r)]
            key = (C_s, keep.tobytes())
            if key not in src_seen:                   # ソースのみは反復に依らないことが多い（同じ学習データは 1 回だけ解く）
                src_seen[key] = f"s_{s.name}_r{r}"
                jobs[C_s].append((src_seen[key], "src_only", "", r, src_mask))
            for K in KS:
                m = src_mask.copy()
                m[idx[K]] = True
                C_m, _ = ref[("merged", s.name, K, r)]
                jobs[C_m].append((f"m_{s.name}_K{K}_r{r}", "merged", K, r, m))
        fitted = {}
        for C, js in jobs.items():
            fits = logreg.fit_batch(X, y, [j[4] for j in js], (C,))
            for j, f in zip(js, fits):
                fitted[j[0]] = (f, C, j)
        for r in reps:                                # テストでの点数と UAR
            te, _, persons = split[r]
            keep = ~np.isin(s.person, persons)
            Xte, yte = M.rows(tgt.X, te), tgt.y[te]
            C_s, _ = ref[("src_only", s.name, "", r)]
            names = [(src_seen[(C_s, keep.tobytes())], "src_only", "")] + \
                    [(f"m_{s.name}_K{K}_r{r}", "merged", K) for K in KS]
            for name, method, K in names:
                f, C, _ = fitted[name]
                L = f.logits(Xte, C).cpu().numpy()
                key = f"s_{s.name}_r{r}" if method == "src_only" else name
                saved[key] = L.astype(np.float32)
                rows.append({"target": tgt.name, "source": s.name, "method": method, "K": K, "rep": r,
                             "C": C, "uar": shift.uar(yte, L.argmax(1)),
                             "uar_map": ref[(method, s.name, K, r)][1],
                             "n_iter": int(f.n_iter[C]), "grad": float(f.grad[C])})
        print(f"  {kind} {tgt.name} ← {s.name}：{time.perf_counter() - t0:.1f} 秒"
              f"（問 {sum(len(v) for v in jobs.values())}）", flush=True)
    np.savez_compressed(out / f"{tgt.name}.npz", **saved)
    d = pd.DataFrame(rows)
    d.to_csv(out / f"{tgt.name}.csv", index=False)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--source", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(REPS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = args.out_dir or OUT_ROOT / f"merged_scores_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    (out / "provenance.json").write_text(json.dumps({
        "script": "exp/merged_scores.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "K": list(KS), "reps": args.reps, "solver": {"tol": logreg.TOL, "device": str(logreg.DEVICE)}},
        ensure_ascii=False, indent=1))
    groups = groups_for(args.kind)
    targets = args.target or groups
    data = {g: M.load(g, args.kind) for g in groups}
    for t in targets:
        if (out / f"{t}.npz").exists() and not args.overwrite:
            print(f"{args.kind} {t}：済み（飛ばす）", flush=True)
            continue
        t0 = time.perf_counter()
        srcs = [data[s] for s in (args.source or groups) if s != t]
        d = run_target(args.kind, data[t], srcs, args.reps, out)
        diff = (d.uar - d.uar_map).abs()
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒．地図の UAR との差 最大 {diff.max():.2e}，"
              f"0.001 を超えた条件 {int((diff > 1e-3).sum())}／{len(d)}", flush=True)


if __name__ == "__main__":
    main()
