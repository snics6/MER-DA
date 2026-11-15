"""【探索的】E37 の追加（GPU）：GLISTER との比較。teach（E37），d_first（目的は d，利得は 1 次の近似），glister（感情を均等にした
対数尤度，GLISTER と同じ 1 次の近似）を，同じ学習器・同じ予算・同じ歩幅の候補とメタ学習で比べる。

規則は `docs/exploratory-analyses.md` E37 の追加「GLISTER との比較」（設計と予言 G1・G2 は計算の前に書いた）。学習器の模型・候補・
自分自身の点数を動かさない扱い・仮のターゲットの作りは E37（`exp/select_teach.py`）と同じ。random と herding も同じ台本で解く。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/select_glister.py --kind audio_wavlm [--target …]
    ./.venv/bin/python exp/select_glister.py --analyze

出力：`results/select_glister_<kind>/<ターゲット>.csv`（role, eval, method, step, rep, K, uar），`provenance.json`，`provenance_log.jsonl`。
判定は `results/explore_relation/e37g_targets.csv`・`e37g_tests.csv`
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
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_herding import herd  # noqa: E402
from select_labelfree import ht_block_present  # noqa: E402
from select_teach import C_PSEUDO, CS, PAIRS, PSEUDO_REPS, teach_order  # noqa: E402

METHODS = ("teach", "d_first", "glister")
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"


def _prep(Xt, y, L0):
    X = Xt.float()
    Km = X @ X.T
    Km.fill_diagonal_(0.0)
    yt = torch.as_tensor(np.array(y), device=X.device)
    onehot = torch.nn.functional.one_hot(yt, 4).float()
    return Km, yt, onehot, L0.float().clone(), torch.ones(len(y), dtype=torch.bool, device=X.device)


def dbar(L: torch.Tensor, yt: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    """未選択の候補での 6 対の d の平均（微分できる形．teach の目的と同じ定義）。"""
    out = 0.0
    for k, l in PAIRS:
        z = L[:, k] - L[:, l]
        a, b = z[(yt == k) & avail], z[(yt == l) & avail]
        out = out + (a.mean() - b.mean()) / torch.sqrt((a.var(unbiased=False) + b.var(unbiased=False)) / 2)
    return out / len(PAIRS)


def d_first_order(Xt, y, L0, c, n_steps) -> list[int]:
    Km, yt, onehot, L, avail = _prep(Xt, y, L0)
    order = []
    for step in range(n_steps):
        cls = step % 4
        R = onehot - torch.softmax(L, 1)
        Lg = L.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad(dbar(Lg, yt, avail), [Lg])
        gain = (R * (Km @ g)).sum(1)                       # ⟨∇_L d̄, ΔL_i⟩ / c
        gain[~((yt == cls) & avail)] = -float("inf")
        i = int(torch.argmax(gain))
        order.append(i)
        avail[i] = False
        L = L + c * Km[:, i:i + 1] * R[i:i + 1, :]
    return order


def glister_order(Xt, y, L0, c, n_steps) -> list[int]:
    Km, yt, onehot, L, avail = _prep(Xt, y, L0)
    R0 = onehot - torch.softmax(L, 1)                      # 学習の勾配は下書きで 1 回だけ（GLISTER と同じ）
    order = []
    for step in range(n_steps):
        cls = step % 4
        cnt = torch.stack([((yt == k) & avail).sum() for k in range(4)]).float().clamp(min=1.0)
        wv = avail.float() / (4 * cnt[yt])                 # 感情を均等にした保留データの重み
        RV = onehot - torch.softmax(L, 1)                  # 先読みした模型での保留データの残差
        gain = (R0 * (Km @ (wv[:, None] * RV))).sum(1)     # 学習の勾配と保留データの勾配の内積
        gain[~((yt == cls) & avail)] = -float("inf")
        i = int(torch.argmax(gain))
        order.append(i)
        avail[i] = False
        L = L + c * Km[:, i:i + 1] * R0[i:i + 1, :]
    return order


ORDER = {"teach": lambda Xt, y, L0, c, n: teach_order(Xt, y, L0, c, n)[0], "d_first": d_first_order, "glister": glister_order}


def evaluate(role, meta_for, ev, src, C, reps, ks, rows):
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {(m, c): {} for m in METHODS for c in CS}
    if role == "target":
        idx[("herding", 0.0)], idx[("random", 0.0)] = {}, {}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        L0 = src.logits(Xc, C)
        Xt = (Xc - src.mu) / src.sd
        for m in METHODS:
            for c in CS:
                order = ORDER[m](Xt, yq, L0, c, 4 * kmax)
                for K in ks:
                    pp = np.array(order[: 4 * int(K)])
                    if len(np.unique(pp)) != len(pp) or (np.bincount(yq[pp], minlength=4) != int(K)).any():
                        raise ValueError(f"選んだ本数がおかしい: {m} c={c} r{r} K{K}")
                    idx[(m, c)][(r, K)] = pool[pp]
        if role == "target":
            Rt = torch.nn.functional.one_hot(torch.as_tensor(np.array(yq), device=CB.DEV), 4).to(Xt.dtype) - torch.softmax(L0, 1)
            ordh = {}
            for k in range(4):                              # E33 の追加と同じ herding
                cand = np.flatnonzero(yq == k)
                ci = torch.as_tensor(cand, device=CB.DEV)
                R, X = Rt[ci], Xt[ci]
                RX = (R[:, :, None] * X[:, None, :]).reshape(len(cand), -1)
                vr, vx = float(R.var(0, unbiased=False).sum()), float(RX.var(0, unbiased=False).sum())
                a = np.sqrt(vx / vr) if vr > 0 else 1.0
                ordh[k] = cand[herd(torch.cat([a * R, RX], 1), kmax)]
            pos_of = {int(i): j for j, i in enumerate(pool)}
            for K in ks:
                kk = int(K)
                idx[("herding", 0.0)][(r, K)] = pool[np.concatenate([ordh[k][:kk] for k in range(4)])]
                idx[("random", 0.0)][(r, K)] = pool[[pos_of[int(i)] for i in ctx.idx[(r, K)]]]
    for (m, c), ii in idx.items():
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = ii

        def add(r, K, cc, u, cv, L, cen=0, cb="", m=m, c=c):
            rows.append({"role": role, "meta_for": meta_for, "eval": ev.name, "method": m, "step": c, "rep": r, "K": K, "uar": u})
        ht_block_present(sub, reps, src, C, add)


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


def analyze() -> int:
    tabs = []
    for kind in MAIN:
        fs = sorted((ROOT / "results" / f"select_glister_{kind}").glob("*.csv"))
        if not fs:
            continue
        a = pd.concat([pd.read_csv(f, dtype={"K": str}) for f in fs])
        pseudo = a[a.role == "pseudo"].groupby(["meta_for", "method", "step"]).uar.mean()
        tgt = a[(a.role == "target") & a.rep.isin(EC.REPS)].groupby(["eval", "method", "step", "K"]).uar.mean() * 100
        for T in sorted(a.meta_for.unique()):
            for K in ("1", "5", "10", "25"):
                row = {"kind": kind, "target": T, "K": K,
                       "random": float(tgt.loc[(T, "random", 0.0, K)]), "herding": float(tgt.loc[(T, "herding", 0.0, K)])}
                for m in METHODS:
                    star = float(pseudo.loc[(T, m)].idxmax())
                    row[f"{m}_step"] = star
                    row[m] = float(tgt.loc[(T, m, star, K)])
                    row[f"{m}_oracle"] = float(max(tgt.loc[(T, m, c, K)] for c in CS))
                tabs.append(row)
    D = pd.DataFrame(tabs)
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"]):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for a_, b_ in (("teach", "glister"), ("teach", "d_first"), ("d_first", "glister"), ("glister", "herding"),
                       ("teach", "herding"), ("glister", "random")):
            diff = g[a_] - g[b_]
            row[f"{a_}-{b_}"] = float(diff.mean())
            row[f"{a_}>{b_}"] = int((diff > 0).sum())
        for m in METHODS:
            row[f"mean_{m}"] = float(g[m].mean())
        row["mean_herding"], row["mean_random"] = float(g.herding.mean()), float(g.random.mean())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in ("audio_wavlm", "audio_hubert"):
        m = (R.kind == kind) & R.K.isin(["5", "10"])
        R.loc[R.kind == kind, "G1"] = bool((R[m]["teach-glister"] > 0).all()) if m.sum() == 2 else np.nan
        R.loc[R.kind == kind, "G2"] = bool((R[m]["teach-d_first"] > 0).all()) if m.sum() == 2 else np.nan
    D.to_csv(OUT_REL / "e37g_targets.csv", index=False)
    R.to_csv(OUT_REL / "e37g_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
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
    out = args.out_dir or ROOT / "results" / f"select_glister_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/select_glister.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "pseudo_reps": PSEUDO_REPS,
            "methods": list(METHODS), "steps": list(CS), "C_pseudo": C_PSEUDO,
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
