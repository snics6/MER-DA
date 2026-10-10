"""E28 の 1 節：仮説転移を，ソースの統計で標準化し，罰則 C′ と切片の扱いを振って学び直す（GPU）。

予言は `docs/exploratory-analyses.md` E28 の 1.2 節（計算の前に書いた）。下書きは同じ反復のソースのみの分類器
（C を固定した地図の C）。テストと K 本は地図と同じ。設定：

    free_C{0.001,0.01,0.1,1}   切片は自由，罰則 C′ を振る
    free_Csrc                  切片は自由，C′ ＝ ソースの C（E25 の手 7 と同じ C．標準化だけが違う）
    fixed_C{0.01,0.1}          切片をソースの切片に固定
    cb{0.5,2,8}_C0.01          切片に罰則 ‖b − b_s‖²/(2 C_b S) をかける

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/transfer_grid.py --kind audio_wavlm [--target …] [--source …]

出力：`results/transfer_grid_<kind>/<ターゲット>.npz`（`test_idx_r{r}`・`y_test_r{r}`・`s_{ソース}_r{r}`：ソースのみの点数，
`{設定}_{ソース}_K{K}_r{r}`：仮説転移の点数），`<ターゲット>.csv`（設定ごとの UAR，手 7 の UAR，切片の動き，重みの動き），
`provenance.json`。終わったターゲットは飛ばす（`--overwrite` で作り直す）。
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
from data import groups_for  # noqa: E402
from interventions import map_rows  # noqa: E402
from pair_table import map_dir  # noqa: E402
from transfer_scores import hand7_ref  # noqa: E402

KS = ("1", "5", "10")
FREE_CS = (0.001, 0.01, 0.1, 1.0)
FIXED_CS = (0.01, 0.1)
CB_LIST = (0.5, 2.0, 8.0, 32.0, 128.0)
CB_C = 0.01


def cfg_name(mode: str, c: float, cb: float | None = None) -> str:
    if mode == "cb":
        return f"cb{cb:g}_C{c:g}"
    return f"{mode}_C{c:g}"


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
        for r in reps:                                        # ソースの行と C が同じ反復を束ねる
            keep = ~np.isin(s.person, split[r][2])
            C = ref[("src_only", s.name, "", r)][0]
            groups.setdefault((keep.tobytes(), C), []).append(r)
        for (kb, C_s), rs in groups.items():
            keep = np.frombuffer(kb, dtype=bool)
            src = logreg.fit(M.rows(s.X, keep), M.labels(s.y[keep]), (C_s,))
            stats = (src.mu, src.sd)
            jobs = [(r, K) for r in rs for K in KS]
            masks = []
            for r, K in jobs:
                m = np.zeros(nt, bool)
                m[split[r][1][K]] = True
                masks.append(m)
            fits: dict[str, dict] = {}                         # 設定名 → {(r, K): (Fitted, C)}
            cs = FREE_CS if C_s in FREE_CS else FREE_CS + (C_s,)
            fb = logreg.fit_batch(Xt, yt, masks, cs, anchor=src, anchor_c=C_s, stats=stats)
            for c in FREE_CS:
                fits[cfg_name("free", c)] = {j: (f, c) for j, f in zip(jobs, fb)}
            fits["free_Csrc"] = {j: (f, C_s) for j, f in zip(jobs, fb)}
            fx = logreg.fit_batch(Xt, yt, masks, FIXED_CS, anchor=src, anchor_c=C_s, stats=stats, fix_b=True)
            for c in FIXED_CS:
                fits[cfg_name("fixed", c)] = {j: (f, c) for j, f in zip(jobs, fx)}
            for cb in CB_LIST:
                fc = logreg.fit_batch(Xt, yt, masks, (CB_C,), anchor=src, anchor_c=C_s, stats=stats, Cb=cb)
                fits[cfg_name("cb", CB_C, cb)] = {j: (f, CB_C) for j, f in zip(jobs, fc)}
            Ws, bs = src.W[C_s], src.b[C_s]
            for r in rs:
                te = split[r][0]
                Xte, y_te = M.rows(Xt, te), tgt.y[te]
                L = src.logits(Xte, C_s).cpu().numpy()
                saved[f"s_{s.name}_r{r}"] = L.astype(np.float32)
                base = {"target": tgt.name, "source": s.name, "rep": r, "C_src": C_s}
                rows.append({**base, "K": "", "config": "src_only", "uar": shift.uar(y_te, L.argmax(1)),
                             "uar_ref": ref[("src_only", s.name, "", r)][1]})
                for K in KS:
                    for name, d in fits.items():
                        f, c = d[(r, K)]
                        Lh = f.logits(Xte, c).cpu().numpy()
                        saved[f"{name}_{s.name}_K{K}_r{r}"] = Lh.astype(np.float32)
                        db = (f.b[c] - bs).cpu().numpy()
                        rows.append({**base, "K": K, "config": name, "C_prime": c,
                                     "uar": shift.uar(y_te, Lh.argmax(1)),
                                     "uar_ref": iv.get((s.name, K, r), np.nan) if name == "free_Csrc" else np.nan,
                                     "n_iter": int(f.n_iter[c]), "grad": float(f.grad[c]),
                                     "dW_norm": float((f.W[c] - Ws).norm()), "db_norm": float(np.linalg.norm(db)),
                                     **{f"db{k}": float(db[k]) for k in range(4)}})
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
    out = args.out_dir or ROOT / "results" / f"transfer_grid_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    (out / "provenance.json").write_text(json.dumps({
        "script": "exp/transfer_grid.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "map_dir": str(map_dir(args.kind)), "K": list(KS), "reps": args.reps,
        "configs": {"free": FREE_CS, "fixed": FIXED_CS, "cb": CB_LIST, "cb_C": CB_C},
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
        chk = d.dropna(subset=["uar_ref"])
        src_diff = (chk[chk.config == "src_only"].uar - chk[chk.config == "src_only"].uar_ref).abs().max()
        h7 = chk[chk.config == "free_Csrc"]
        print(f"{args.kind} {t}：{time.perf_counter() - t1:.0f} 秒．ソースのみの地図との差 最大 {src_diff:.1e}．"
              f"free_Csrc − 手 7（K 本で標準化）の UAR：平均 {(h7.uar - h7.uar_ref).mean():+.4f}", flush=True)


if __name__ == "__main__":
    main()
