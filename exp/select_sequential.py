"""【探索的】E34 の追加（GPU）：本数をそろえながら何回かに分けて選ぶ herding（seq_herding）と，同じ手順ででたらめに選ぶ seq_random。

規則は `docs/exploratory-analyses.md` E34「E34 の追加」（予言 Q1・Q2 は計算の前に書いた）。評価・候補・中心・下書き・学習器は E34
（`exp/select_labelfree.py`）と同じ。ラベルを 1 回に 4 本ずつ付け，各回の 1 本ごとに「本数が最も少ない感情 c」を選び，いまの
分類器が c と予測する候補から 1 本を選ぶ（herding：ラベルが c と分かった選択済み＋候補の x̃ の平均を，予測が c の群の平均に近づける）。
4 本ごとに仮説転移を解き直して予測を更新する。選ぶ順番は予算によらないので 4·max(K) 本まで選び，各 K は先頭の 4K 本を使う。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/select_sequential.py --kind audio_wavlm --target IEMOCAP-impro

出力：`results/select_seq_<kind>/<ターゲット>.csv`（method=HT_SEQ, source=選び方, rep, K, uar），`.npz`，`<ターゲット>_sel.csv`
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
import compare_b1 as CB  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from select_labelfree import ht_block_present, present_weights  # noqa: E402

STRATS = ("seq_herding", "seq_random")


def fit_reps(ctx, idx_by_rep: dict, src, C_s) -> dict:
    """反復ごとの K 本（選んだ発話）で P4 の既定の仮説転移を 1 回の計算で解き，反復 → 分類器を返す。"""
    tgt = ctx.tgt
    blocks, ys, masks, off, reps = [], [], [], 0, list(idx_by_rep)
    for r in reps:
        ii = np.asarray(idx_by_rep[r])
        blocks.append(M.rows(tgt.X, ii) - ctx.ct[r]); ys.append(tgt.y[ii])
        off += len(ii)
    n = off
    off = 0
    for r in reps:
        m = np.zeros(n, bool); m[off:off + len(idx_by_rep[r])] = True
        masks.append(m); off += len(idx_by_rep[r])
    X, yX = torch.cat(blocks), M.labels(np.concatenate(ys))
    weights = []
    for m in masks:
        w = torch.zeros(n, dtype=logreg.DTYPE, device=CB.DEV)
        mt = torch.as_tensor(m, device=CB.DEV)
        w[mt] = present_weights(yX[mt])
        weights.append(w)
    fits = logreg.fit_batch(X, yX, masks, (CB.DEF_CP,), anchor=src, anchor_c=C_s, stats=(src.mu, src.sd),
                            Cb=[CB.DEF_CB] * len(masks), weights=weights)
    return dict(zip(reps, fits))


def run_target(kind, tgt, data, reps, ks, out):
    ctx = CB.Ctx(tgt, reps, ks, center="ind")
    pool_src = [s for s in data if s != tgt.name and GROUP_SPEC[s][0] != GROUP_SPEC[tgt.name][0]]
    rows, saved, sel_rows = [], {}, []
    for r in reps:
        saved[f"test_idx_r{r}"] = np.flatnonzero(ctx.te[r])
        saved[f"y_test_r{r}"] = tgt.y[ctx.te[r]]
    t0 = time.perf_counter()
    Xp = torch.cat([data[s].X for s in pool_src])
    yp = M.labels(np.concatenate([data[s].y for s in pool_src]))
    C_p = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{tgt.name}.csv", dtype={"K": str}).C.iloc[0])
    src_pc = logreg.fit(Xp - Xp.mean(0), yp, (C_p,))
    del Xp
    kmax = max(int(k) for k in ks)
    pools, Xcs, Xts = {}, {}, {}
    for r in reps:
        pools[r] = np.flatnonzero(~ctx.te[r])
        Xcs[r] = M.rows(tgt.X, pools[r]) - ctx.ct[r]
        Xts[r] = (Xcs[r] - src_pc.mu) / src_pc.sd
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    order = {}
    for st in STRATS:
        state = {}
        for r in reps:
            L = src_pc.logits(Xcs[r], C_p)
            state[r] = {"pred": L.argmax(1).cpu().numpy(), "prob": torch.softmax(L, 1).cpu().numpy(), "sel": [],
                        "cnt": np.zeros(4, int), "sum": torch.zeros(4, Xts[r].shape[1], dtype=Xts[r].dtype, device=CB.DEV),
                        "rng": np.random.default_rng(zlib.crc32(f"seq|{kind}|{tgt.name}|{r}".encode()))}
        for rnd in range(kmax):
            for r in reps:
                S = state[r]
                yq = tgt.y[pools[r]]
                taken = np.zeros(len(pools[r]), bool); taken[S["sel"]] = True
                for _ in range(4):
                    c = int(np.argmin(S["cnt"]))
                    cand = np.flatnonzero((S["pred"] == c) & ~taken)
                    if len(cand) == 0:
                        rest = np.flatnonzero(~taken)
                        cand = rest[np.argsort(-S["prob"][rest, c])[:max(1, len(rest) // 4)]]
                    if st == "seq_random":
                        i = int(S["rng"].choice(cand))
                    else:
                        grp = np.flatnonzero(S["pred"] == c)
                        mu = Xts[r][torch.as_tensor(grp if len(grp) else cand, device=CB.DEV)].mean(0)
                        m = int(S["cnt"][c])
                        Xc_ = Xts[r][torch.as_tensor(cand, device=CB.DEV)]
                        score = (((S["sum"][c] + Xc_) / (m + 1) - mu) ** 2).sum(1)
                        i = int(cand[int(torch.argmin(score))])
                    S["sel"].append(i); taken[i] = True
                    yi = int(yq[i]); S["cnt"][yi] += 1; S["sum"][yi] += Xts[r][i]
            fits = fit_reps(ctx, {r: pools[r][state[r]["sel"]] for r in reps}, src_pc, C_p)
            for r in reps:
                L = fits[r].logits(Xcs[r], CB.DEF_CP)
                state[r]["pred"], state[r]["prob"] = L.argmax(1).cpu().numpy(), torch.softmax(L, 1).cpu().numpy()
        order[st] = {r: state[r]["sel"] for r in reps}
    t_sel = time.perf_counter() - t1

    t2 = time.perf_counter()
    for st in STRATS:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = {}
        for r in reps:
            yq = tgt.y[pools[r]]
            for K in ks:
                pp = np.array(order[st][r][:4 * int(K)])
                sub.idx[(r, K)] = pools[r][pp]
                cnt = np.bincount(yq[pp], minlength=4)
                sel_rows.append({"target": tgt.name, "strategy": st, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                                 "min_count": int(cnt.min()), "max_count": int(cnt.max()),
                                 **{f"n_{c}": int(cnt[c]) for c in range(4)}})

        def add(r, K, c, u, cv, L, cen=0, cb="", st=st):
            rows.append({"target": tgt.name, "source": st, "method": "HT_SEQ", "cen": cen, "cp": c, "cb": cb,
                         "rep": r, "K": K, "uar": u})
            saved[f"{st}_K{K}_r{r}"] = L.astype(np.float32)
        ht_block_present(sub, reps, src_pc, C_p, add)
    print(f"  {kind} {tgt.name}：準備 {t_prep:.1f} 秒，選ぶ {t_sel:.1f} 秒，仮説転移 {time.perf_counter() - t2:.1f} 秒", flush=True)
    np.savez_compressed(out / f"{tgt.name}.npz", **saved)
    pd.DataFrame(rows).to_csv(out / f"{tgt.name}.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(out / f"{tgt.name}_sel.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / f"select_seq_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.npz").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/select_sequential.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "K": args.ks, "reps": args.reps, "strategies": STRATS, "cp": CB.DEF_CP, "cb": CB.DEF_CB}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, data[t], data, args.reps, args.ks, out)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
