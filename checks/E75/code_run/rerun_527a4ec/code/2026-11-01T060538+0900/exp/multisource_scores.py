"""E28 の 7 節：全ソースを混ぜた合併を学び直し，テストの点数と，式の λ に要る量を保存する（GPU）。

混ぜ方は E26 の「全ソース」（ターゲット以外，同じコーパスの群を除く）。C は混ぜたソース全体のソースのみの C（交差検証，E26 と同じ）。
テストと K 本は地図と同じ。反復 0〜9，K ∈ {1, 5, 10}。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/multisource_scores.py --kind audio_wavlm

出力：`results/multisource_scores_<kind>/<ターゲット>.npz`（`test_idx_r{r}`・`y_test_r{r}`・`s_all_r{r}`・`m_all_K{K}_r{r}`：テストの点数，
`p_src_train`：混ぜたソースのみの分類器の学習データでの確率，`p_tgt_all`：同じ分類器のターゲット全発話での確率，
`src_names`・`src_y`），`<ターゲット>.csv`（UAR と E26 の UAR），`provenance.json`。終わったターゲットは飛ばす。
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

KS = ("1", "5", "10")


def e26_ref(kind: str, target: str) -> dict:
    f = ROOT / "results" / f"multisource_{kind}" / f"{target}.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f, comment="#", dtype={"K": str})
    d = d[d.variant == "全ソース"]
    return {(r.K, int(r.rep)): float(r.uar) for r in d.itertuples()}


def run_target(kind: str, tgt: M.Group, data: dict, reps, out: Path) -> pd.DataFrame:
    srcs = [s for s in data if s != tgt.name and GROUP_SPEC[s][0] != GROUP_SPEC[tgt.name][0]]
    Xs = torch.cat([data[s].X for s in srcs])
    ys = np.concatenate([data[s].y for s in srcs])
    ps = np.concatenate([data[s].person for s in srcs])
    tr = M.train(Xs, ys, ps, np.ones(len(ys), bool), M.SRC_CV_SEED)
    C = tr.C
    src = tr.final
    ref = e26_ref(kind, tgt.name)
    nt = len(tgt.y)
    X = torch.cat([tgt.X, Xs])
    y = M.labels(np.r_[tgt.y, ys])
    split, saved, rows = {}, {}, []
    for r in reps:
        te = M.test_mask(tgt, r)
        order = M.kshot_order(tgt, np.flatnonzero(~te), r)
        split[r] = (te, {K: M.take_k(order, K) for K in KS})
        saved[f"test_idx_r{r}"] = np.flatnonzero(te)
        saved[f"y_test_r{r}"] = tgt.y[te]
    jobs = [(r, K) for r in reps for K in KS]
    masks = []
    for r, K in jobs:
        m = np.r_[np.zeros(nt, bool), np.ones(len(ys), bool)]
        m[split[r][1][K]] = True
        masks.append(m)
    fits = dict(zip(jobs, logreg.fit_batch(X, y, masks, (C,))))
    saved["p_src_train"] = torch.softmax(src.logits(Xs, C), 1).cpu().numpy().astype(np.float32)
    saved["p_tgt_all"] = torch.softmax(src.logits(tgt.X, C), 1).cpu().numpy().astype(np.float32)
    saved["src_names"] = np.array(srcs)
    saved["src_y"] = ys
    for r in reps:
        te = split[r][0]
        Xte, y_te = M.rows(tgt.X, te), tgt.y[te]
        Ls = src.logits(Xte, C).cpu().numpy()
        saved[f"s_all_r{r}"] = Ls.astype(np.float32)
        rows.append({"target": tgt.name, "rep": r, "K": "", "method": "src_all", "C": C, "n_src": len(ys),
                     "sources": "+".join(srcs), "uar": shift.uar(y_te, Ls.argmax(1))})
        for K in KS:
            f = fits[(r, K)]
            Lm = f.logits(Xte, C).cpu().numpy()
            saved[f"m_all_K{K}_r{r}"] = Lm.astype(np.float32)
            rows.append({"target": tgt.name, "rep": r, "K": K, "method": "merged_all", "C": C, "n_src": len(ys),
                         "sources": "+".join(srcs), "uar": shift.uar(y_te, Lm.argmax(1)), "uar_e26": ref.get((K, r), np.nan),
                         "n_iter": int(f.n_iter[C]), "grad": float(f.grad[C])})
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
    out = args.out_dir or ROOT / "results" / f"multisource_scores_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    (out / "provenance.json").write_text(json.dumps({
        "script": "exp/multisource_scores.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "K": list(KS), "reps": args.reps}, ensure_ascii=False, indent=1))
    groups = groups_for(args.kind)
    data = {g: M.load(g, args.kind) for g in groups}
    for t in args.target or groups:
        if (out / f"{t}.npz").exists():
            print(f"{args.kind} {t}：済み（飛ばす）", flush=True)
            continue
        t0 = time.perf_counter()
        d = run_target(args.kind, data[t], data, args.reps, out)
        chk = d.dropna(subset=["uar_e26"])
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒．E26 の UAR との差 最大 "
              f"{(chk.uar - chk.uar_e26).abs().max() if len(chk) else float('nan'):.2e}", flush=True)


if __name__ == "__main__":
    main()
