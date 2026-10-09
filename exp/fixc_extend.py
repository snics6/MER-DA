"""C を固定した地図を，交差検証なしの合併で延ばす（`docs/researchvision.md` 3.7 節の段 1）。

交差検証の地図（`exp/map.py`）は，合併の C を交差検証で選ぶので時間がかかる（言葉を含む 4 種類を 30 回にすると約 67 時間）。
仕組みの解析の土台は「合併の C ＝ 同じ反復のソースのみの C」の地図（`docs/researchvision.md` 6.2 節）なので，
合併はその C で 1 回だけ解けばよい。ここでは `exp/map.py` と同じ

- テストの袋（`test_mask`），K 本（`kshot_order`・`take_k`），ソースの行（同じコーパスの組はテストの人物を外す），
- ターゲットのみ（交差検証で C を選ぶ）とソースのみ（交差検証で C を選ぶ．ソースごとに 1 回学んで使い回す），

で学び，合併だけを C ＝ ソースのみの C で解く。行は地図と同じ列で `results/fixc_map/<kind>/<ターゲット>.csv` に足す
（合併の行は `C_rule` が `src`，交差検証の列は空，`test_uar_C<C>` だけが入る．`exp/fixc_map.py` で作った行と同じ意味）。
テストでの点数（ソースのみと合併）は `OUT_ROOT/merged_scores_<kind>/<ターゲット>.npz` に足す（`exp/merged_scores.py` と同じ鍵）。

    MERDA_OUT_ROOT=results/fixc_text ./.venv/bin/python exp/fixc_extend.py --kind text_roberta --reps 5 6 … 29
    ./.venv/bin/python exp/fixc_extend.py --kind text_roberta --reps 0 1 2 3 4 --out-map /tmp/x --out-scores /tmp/y   # 確かめ

終わった反復は飛ばす（地図の CSV にその反復の行がそろっていれば）。
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
from pair_table import OUT_ROOT  # noqa: E402

KS = ("1", "5", "10")


def merged_row(base: dict, s: str, K: str, n_src: int, n_tgt: int, C: float, uar: float, sec: float, it: int) -> dict:
    row = {**base, "source": s, "method": "merged", "K": K, "n_src": n_src, "n_tgt": n_tgt,
           "C": C, "C_rule": "src", "uar": uar, "C_oracle": np.nan, "uar_oracle": np.nan,
           "sec": sec, "n_iter": it, "n_loop": it, "init": "none", "cached": False}
    for c in M.C_GRID:
        row[f"cv_uar_C{c:g}"] = np.nan
        row[f"test_uar_C{c:g}"] = uar if np.isclose(c, C) else np.nan
    return row


def run_rep(tgt: M.Group, sources: list[M.Group], r: int, Ks, kind: str, src_cache: dict) -> tuple[list[dict], dict]:
    te = M.test_mask(tgt, r)
    pool = np.flatnonzero(~te)
    order = M.kshot_order(tgt, pool, r)
    X_test, y_test = M.rows(tgt.X, te), tgt.y[te]
    test_persons = np.unique(tgt.person[te])
    base = {"target": tgt.name, "rep": r, "kind": kind, "n_test": int(te.sum())}
    rows, saved = [], {f"test_idx_r{r}": np.flatnonzero(te), f"y_test_r{r}": y_test}
    src = {}
    for s in sources:                                         # ② ソースのみ（地図と同じ）
        k = ~np.isin(s.person, test_persons)
        key = (s.name, tuple(np.intersect1d(test_persons, s.person)))
        cached = key in src_cache
        if not cached:
            src_cache[key] = M.train(M.rows(s.X, k), s.y[k], s.person[k], np.ones(int(k.sum()), bool), M.SRC_CV_SEED)
        t = src_cache[key]
        src[s.name] = (k, t)
        rows.append({**base, "source": s.name, "method": "src_only", "K": "", "n_src": int(k.sum()), "n_tgt": 0,
                     **M.score(t, X_test, y_test, cached)})
        saved[f"s_{s.name}_r{r}"] = t.final.logits(X_test, t.C).cpu().numpy().astype(np.float32)
    idx = {K: M.take_k(order, K) for K in Ks}
    for K in Ks:                                              # ① ターゲットのみ（地図と同じ）
        i = idx[K]
        t = M.train(M.rows(tgt.X, i), tgt.y[i], tgt.person[i], np.ones(len(i), bool), r)
        rows.append({**base, "source": "", "method": "tgt_only", "K": K, "n_src": 0, "n_tgt": len(i),
                     **M.score(t, X_test, y_test, False)})
    nt = len(tgt.y)
    for s in sources:                                         # ③ 合併（C ＝ ソースのみの C．交差検証なし）
        k, ts = src[s.name]
        X = torch.cat([tgt.X, s.X])
        y = M.labels(np.r_[tgt.y, s.y])
        masks = []
        for K in Ks:
            m = np.r_[np.zeros(nt, bool), k]
            m[idx[K]] = True
            masks.append(m)
        t0 = time.perf_counter()
        fits = logreg.fit_batch(X, y, masks, (ts.C,))
        sec = (time.perf_counter() - t0) / len(Ks)
        for K, f in zip(Ks, fits):
            L = f.logits(X_test, ts.C).cpu().numpy()
            rows.append(merged_row(base, s.name, K, int(k.sum()), len(idx[K]), ts.C, shift.uar(y_test, L.argmax(1)),
                                   sec, int(f.n_iter[ts.C])))
            saved[f"m_{s.name}_K{K}_r{r}"] = L.astype(np.float32)
    return rows, saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--reps", nargs="+", type=int, required=True)
    ap.add_argument("--K", nargs="+", default=list(KS))
    ap.add_argument("--out-map", type=Path)
    ap.add_argument("--out-scores", type=Path)
    args = ap.parse_args()
    out_map = args.out_map or ROOT / "results" / "fixc_map" / args.kind
    out_sc = args.out_scores or OUT_ROOT / f"merged_scores_{args.kind}"
    out_map.mkdir(parents=True, exist_ok=True)
    out_sc.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/fixc_extend.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rule": "merged の C ＝ 同じ反復の src_only の C（交差検証なし）", "K": args.K}
    groups = groups_for(args.kind)
    data = {g: M.load(g, args.kind) for g in groups}
    cols = list(M.COLUMNS)
    src_cache: dict = {}
    t_all = time.perf_counter()
    for t in groups:
        path = out_map / f"{t}.csv"
        n_src = len(groups) - 1
        done = set()
        if path.exists():
            prev = pd.read_csv(path, comment="#", dtype={"K": str, "source": str})
            prev = prev[prev.K.isin(args.K) | (prev.method == "src_only")]
            cnt = prev.groupby("rep").size()
            done = set(int(r) for r, c in cnt.items() if c >= n_src + len(args.K) * (1 + n_src))
        reps = [r for r in args.reps if r not in done]
        if not reps:
            print(f"{args.kind} {t}：済み（飛ばす）", flush=True)
            continue
        new_path = not path.exists()
        with open(path, "a") as f:
            f.write("# " + json.dumps(prov, ensure_ascii=False) + "\n")
            if new_path:
                pd.DataFrame(columns=cols).to_csv(f, index=False)
        npz = out_sc / f"{t}.npz"
        saved_all = dict(np.load(npz)) if npz.exists() else {}
        for r in reps:
            t0 = time.perf_counter()
            rows, saved = run_rep(data[t], [data[s] for s in groups if s != t], r, args.K, args.kind, src_cache)
            pd.DataFrame(rows, columns=cols).to_csv(path, mode="a", header=False, index=False)
            saved_all.update(saved)
            np.savez_compressed(npz, **saved_all)
            print(f"{args.kind} {t} r={r}：{len(rows)} 行 {time.perf_counter() - t0:.1f} 秒（累計 {time.perf_counter() - t_all:.0f} 秒）",
                  flush=True)
    print(f"終わり：{out_map}，{out_sc}", flush=True)


if __name__ == "__main__":
    main()
