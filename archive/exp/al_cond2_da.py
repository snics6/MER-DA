"""【探索的】E61-C（GPU）：能動 DA の代表（CLUE・TQS 型）を，条件 2（見本なし．下書きの状態から 1 回で 4K 本を選ぶ）・E39 の生徒で測る（E50 と同じ手順）。

規則は `docs/exploratory-analyses.md` E61（設計と予言は計算の前に書いた）。生徒・設定・座標・評価・候補・中心は E50（`exp/al_student39.py`）と同じ
（E40 の生徒．感情の重みは present_weights．判定には E40 の random の組の設定を使い，teach の組の ρ_M の生徒でも解いて残す）．
  clue ：下書き（E39）の確率のエントロピーを重みにした k-means（4K 群）の中心に最も近い発話（Prabhu ほか 2021．K ごとに解き直す．E57 の実装）
  tqs  ：ソースの分類器の言い切りの票のエントロピーと，下書きの確率のエントロピーの順位の和が大きい順（Fu ほか 2021 の委員会と不確かさ．ドメインらしさは入れない．K で入れ子）
比べる相手は E49 の random_pool・LLL・C:0001 と E50 の margin・badge・coreset・typiclust（同じ生徒．判定で読む）．

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/al_cond2_da.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/al_cond2_da.py --analyze

出力：`results/al_cond2_da_<kind>/<ターゲット>.csv`（policy, rho_M, cb, cp, rep, K, uar），`<ターゲット>_sel.csv`，`provenance.json`，`provenance_log.jsonl`．
判定は `results/explore_relation/e61c_targets.csv`・`e61c_tests.csv`
"""

from __future__ import annotations

import argparse
import json
import os
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
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from seeded_cond3 import clue_batch, entropy_t  # noqa: E402
from teach_stack import softmax_t  # noqa: E402
from teach_student39 import Student, solve_students  # noqa: E402

KINDS = ["audio_wavlm", "video"]
POLICIES = ["clue", "tqs"]
OUT_REL = ROOT / "results" / "explore_relation"


def rank(v):
    return torch.argsort(torch.argsort(v)).float()


def run_target(kind, T, data, f, Cs, reps, ks, out, cache):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    sel = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"]).loc[(kind, T)]
    rhos_m = sorted({float(sel.t_rho_M), float(sel.r_rho_M)})
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    idx = {p: {} for p in POLICIES}
    srows, rows, students = [], [], {}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, float(sel.gamma), float(sel.rho_s), rhos_m)
        students[r] = st
        P39 = torch.softmax(st.L0, 1).float()
        X = st.Xstd.float()                                                    # x̃（E46・E50 と同じ座標）
        ent = entropy_t(P39)
        Ps = [softmax_t(f[s].logits(Xc, Cs[s])) for s in S]
        votes = torch.as_tensor(np.stack([np.bincount(row, minlength=4) for row in np.stack(Ps, 1).argmax(2)]) / len(S),
                                dtype=torch.float32, device=CB.DEV)
        order_tqs = torch.argsort(rank(entropy_t(votes)) + rank(ent), descending=True).tolist()
        gen = lambda name: torch.Generator().manual_seed(zlib.crc32(f"e61c|{name}|{T}|{r}".encode()))  # noqa: E731
        for K in ks:
            n = 4 * int(K)
            picks = {"tqs": order_tqs[:n], "clue": clue_batch(X, ent, n, gen(f"clue{K}"))}
            for p, pp in picks.items():
                pp = np.array(pp)
                if len(pp) != n or len(np.unique(pp)) != n:
                    raise ValueError(f"選んだ本数がおかしい: {p} r{r} K{K}")
                idx[p][(r, K)] = pool[pp]
                cnt = np.bincount(yq[pp], minlength=4)
                srows.append({"target": T, "policy": p, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                              "min_count": int(cnt.min()), "max_count": int(cnt.max())})
    solve_students(T, ev, ctx, reps, ks, students, idx, rhos_m, C_T, rows)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=KINDS) -> int:
    sel = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"])
    e49 = pd.read_csv(OUT_REL / "e49_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    e50 = pd.read_csv(OUT_REL / "e50_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    ref49, ref50 = ["random_pool", "LLL", "C:0001"], ["margin", "badge", "coreset", "typiclust"]
    tabs = []
    for kind in kinds:
        d = ROOT / "results" / f"al_cond2_da_{kind}"
        for x in sorted(f_ for f_ in d.glob("*.csv") if not f_.stem.endswith("_sel")):
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
                for p in ref49:
                    row[p] = float(e49.loc[(kind, T, K), p])
                for p in ref50:
                    row[p] = float(e50.loc[(kind, T, K), p])
                tabs.append(row)
    D = pd.DataFrame(tabs)
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for x in POLICIES + ref49 + ref50:
            row[x] = float(g[x].mean())
        for p in POLICIES + ref50 + ["LLL", "C:0001"]:
            diff = g[p] - g.random_pool
            row[f"{p}-random_pool"] = float(diff.mean())
            row[f"{p}>random_pool"] = int((diff > 0).sum())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in kinds:                                                            # 予言 B6（docs/exploratory-analyses.md E61）
        sub = R[R.kind == kind].set_index("K")
        if "1" in sub.index and "5" in sub.index:
            R.loc[R.kind == kind, "B6"] = bool(max(sub.loc[K, f"{p}-random_pool"] for p in POLICIES for K in ("1", "5")) < 1.0)
    D.to_csv(OUT_REL / "e61c_targets.csv", index=False)
    R.to_csv(OUT_REL / "e61c_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in R if c == "B6"]).round(2).T.to_string())
    if "B6" in R:
        print(R.dropna(subset=["B6"]).drop_duplicates("kind")[["kind", "B6"]].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MERDA_GPU_FRACTION", "1.0")))
    out = args.out_dir or ROOT / "results" / f"al_cond2_da_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    f, Cs, _, _ = CM.prepare(args.kind, data)
    prov = {"script": "exp/al_cond2_da.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "reps": args.reps, "ks": args.ks, "policies": POLICIES, "src_C": Cs,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, f, Cs, args.reps, args.ks, out, cache)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
