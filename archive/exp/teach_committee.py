"""【探索的】E45（GPU）：ソースの票を癖でまとめた見込み q（E44 の C と B）を言い切る形で teach に入れる（設定 1 の段 3 のうち，q を直さない版）。

規則は `docs/exploratory-analyses.md` E45（設計と予言は計算の前に書いた）。E43 と同じく E41 の PPP の 3 か所に見込みを入れる
（`teach_ablate.ablate_order` の belief）。見込みは選ぶ間ずっと固定（ラベルが付いても直さない）：
  C     ：E44 の C（まとめ方を仮想ターゲットで学ぶ．λ は E44 が仮想ターゲットで選んだ値）の最大の感情を言い切る
  B     ：E44 の B（確率のまま癖で直す）の最大の感情を言い切る
  hardp ：下書きの予測 p の最大の感情を言い切る（ラベル無しで作れる比べる相手．当たり方は p と同じ）
評価・候補・中心・下書き・学習器・手順は E41・E43 と同じ。random_true・random_pool・LLL・PPP は E41 の値を使う。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/teach_committee.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/teach_committee.py --analyze

出力：`results/teach_committee_<kind>/<ターゲット>.csv`（policy, rep, K, uar），`<ターゲット>_q.csv`（各 q の候補の上の UAR．E44 との照合），
`<ターゲット>_sel.csv`，`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e45_targets.csv`・`e45_tests.csv`
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
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import ht_block_present  # noqa: E402
from select_teach import CS, pname  # noqa: E402
from teach_ablate import ablate_order  # noqa: E402

VARIANTS = ["C", "B", "hardp"]
MAIN = ["audio_wavlm", "audio_hubert"]
OUT_REL = ROOT / "results" / "explore_relation"


def run_target(kind, T, data, f, Cs, P, Y, reps, ks, out):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    H, Hs = CM.habits(P, Y, S, S)
    e44 = pd.read_csv(ROOT / "results" / f"committee_{kind}" / f"{T}.csv")
    lam = float(e44[e44.variant == "C"].lam.iloc[0])                                 # E44 が仮想ターゲットで選んだ λ
    Fall, Aall, yall, wall = CM.pool_data(P, Y, S, S)
    pool_C = CM.Pool(S).fit(Fall, Aall, yall, wall, lam)
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    Xp = torch.cat([data[s].X for s in S])
    yp = M.labels(np.concatenate([data[s].y for s in S]))
    src = logreg.fit(Xp - Xp.mean(0), yp, (C_T,))
    del Xp
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {f"{v}_{pname(c)}": {} for v in VARIANTS for c in CS}
    rows, srows, qrows = [], [], []
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        L0 = src.logits(Xc, C_T)
        X = ((Xc - src.mu) / src.sd).float()
        Ps = [torch.softmax(f[s].logits(Xc, Cs[s]), 1).cpu().numpy() for s in S]
        Fq = torch.as_tensor(np.stack([np.log(np.clip(p, 1e-12, None)) for p in Ps], 1), dtype=logreg.DTYPE, device=logreg.DEVICE)
        Aq = torch.ones(len(yq), len(S), dtype=logreg.DTYPE, device=logreg.DEVICE)
        Q = {"C": pool_C.predict(Fq, Aq), "B": CM.agg_B(Ps, [Hs[s] for s in S]),
             "hardp": torch.softmax(L0, 1).cpu().numpy()}
        bel = {}
        for v, q in Q.items():
            a = q.argmax(1)
            qrows.append({"target": T, "variant": v, "rep": r, "uar": CB.uar(yq, a), "acc": float((a == yq).mean())})
            oh = torch.nn.functional.one_hot(torch.as_tensor(a, device=X.device), 4).float()
            bel[v] = lambda L, P, oh=oh: oh
        Km = X @ X.T
        Km.fill_diagonal_(0.0)
        for v, fb in bel.items():
            for c in CS:
                order, turns, fills = ablate_order(Km, yq, L0, c, 4 * kmax, "PPP", belief=fb)
                pn = f"{v}_{pname(c)}"
                for K in ks:
                    n = 4 * int(K)
                    pp = np.array(order[:n])
                    if len(np.unique(pp)) != n:
                        raise ValueError(f"選んだ本数がおかしい: {pn} r{r} K{K}")
                    idx[pn][(r, K)] = pool[pp]
                    cnt = np.bincount(yq[pp], minlength=4)
                    srows.append({"target": T, "policy": pn, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                                  "min_count": int(cnt.min()), "max_count": int(cnt.max()),
                                  "turn_match": float((yq[pp] == np.array(turns[:n])).mean()), "filled": int(sum(fills[:n]))})
        del Km
    for pn in idx:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[pn]

        def add(r, K, c, u, cv, L, cen=0, cb="", pn=pn):
            rows.append({"target": T, "policy": pn, "rep": r, "K": K, "uar": u})
        ht_block_present(sub, reps, src, C_T, add)
    pd.DataFrame(qrows).to_csv(out / f"{T}_q.csv", index=False)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN) -> int:
    star = pd.read_csv(OUT_REL / "e37_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"]).policy_star
    e41 = pd.read_csv(OUT_REL / "e41_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    e43 = pd.read_csv(OUT_REL / "e43_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    tabs = []
    for kind in kinds:
        d = ROOT / "results" / f"teach_committee_{kind}"
        fs = [x for x in sorted(d.glob("*.csv")) if not x.stem.endswith(("_sel", "_q"))]
        if not fs:
            continue
        a = pd.concat([pd.read_csv(x, dtype={"K": str}) for x in fs])
        a = a[a.rep.isin(EC.REPS)]
        u = a.groupby(["target", "policy", "K"]).uar.mean() * 100
        q = pd.concat([pd.read_csv(x) for x in sorted(d.glob("*_q.csv"))])
        q = q[q.rep.isin(a.rep.unique())]
        e44 = pd.concat([pd.read_csv(x) for x in sorted((ROOT / "results" / f"committee_{kind}").glob("*.csv"))
                         if not x.stem.endswith(("_src", "_meta", "_habit"))])
        e44 = e44.assign(variant=e44.variant.replace({"p": "hardp"})).set_index(["target", "variant", "rep"]).uar
        chk = float((q.set_index(["target", "variant", "rep"]).uar - e44.reindex(q.set_index(["target", "variant", "rep"]).index)).abs().max() * 100)
        qm = q.groupby(["target", "variant"]).uar.mean() * 100
        for T in sorted(a.target.unique()):
            cs = float(star.loc[(kind, T)].split("_c")[1])
            for K in ("1", "5", "10", "25"):
                b = e41.loc[(kind, T, K)]
                row = {"kind": kind, "target": T, "K": K, "step": cs, "chk_e44_max": chk}
                for x in ("random_true", "random_pool", "LLL", "PPP"):
                    row[x] = float(b[x])
                for x in ("nbr", "noisy0.25"):
                    row[f"e43_{x}"] = float(e43.loc[(kind, T, K), x])
                for v in VARIANTS:
                    vals = {c: float(u.loc[(T, f"{v}_{pname(c)}", K)]) for c in CS}
                    row[v] = vals[cs]
                    row[f"{v}_oracle"] = max(vals.values())
                    row[f"q_{v}"] = float(qm.loc[(T, v)])
                tabs.append(row)
    D = pd.DataFrame(tabs)
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        gain = float((g.LLL - g.random_true).mean())
        row = {"kind": kind, "K": K, "targets": len(g), "random_true": float(g.random_true.mean()), "random_pool": float(g.random_pool.mean()),
               "LLL": float(g.LLL.mean()), "PPP": float(g.PPP.mean()), "chk_e44_max": float(g.chk_e44_max.max())}
        for v in VARIANTS:
            row[f"q:{v}"] = float(g[f"q_{v}"].mean())
            row[v] = float(g[v].mean())
            row[f"share:{v}"] = float((g[v] - g.random_true).mean()) / gain if gain else np.nan
            row[f"-pool:{v}"] = float((g[v] - g.random_pool).mean())
            row[f">pool:{v}"] = int((g[v] > g.random_pool).sum())
            row[f"oracle-main:{v}"] = float((g[f"{v}_oracle"] - g[v]).mean())
        row["C-hardp"] = float((g.C - g.hardp).mean())
        row["C>hardp"] = int((g.C > g.hardp).sum())
        row["e43_nbr-pool"] = float((g.e43_nbr - g.random_pool).mean())
        row["e43_noisy0.25-pool"] = float((g["e43_noisy0.25"] - g.random_pool).mean())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in MAIN:
        k5 = R[(R.kind == kind) & (R.K == "5")]
        if len(k5):
            k5 = k5.iloc[0]
            R.loc[R.kind == kind, "T1"] = bool(k5["-pool:C"] > 0)
            R.loc[R.kind == kind, "T2"] = bool(k5["C-hardp"] > 0)
            R.loc[R.kind == kind, "T3"] = bool(k5["-pool:hardp"] <= 0)
    D.to_csv(OUT_REL / "e45_targets.csv", index=False)
    R.to_csv(OUT_REL / "e45_tests.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["kind", "K", "random_true", "random_pool", "LLL", "PPP"] + [f"{p}{v}" for v in VARIANTS for p in ("q:", "", "share:", "-pool:", ">pool:", "oracle-main:")] + \
        ["C-hardp", "C>hardp", "e43_nbr-pool", "e43_noisy0.25-pool", "chk_e44_max"]
    print(R[cols].round(2).T.to_string())
    vc = [c for c in ("T1", "T2", "T3") if c in R]
    if vc:
        print(R.drop_duplicates("kind")[["kind"] + vc].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=MAIN)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds)
    out = args.out_dir or ROOT / "results" / f"teach_committee_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    f, Cs, P, Y = CM.prepare(args.kind, data)
    prov = {"script": "exp/teach_committee.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "variants": VARIANTS, "steps": list(CS), "src_C": Cs,
            "cp": CB.DEF_CP, "cb": CB.DEF_CB, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, f, Cs, P, Y, args.reps, args.ks, out)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
