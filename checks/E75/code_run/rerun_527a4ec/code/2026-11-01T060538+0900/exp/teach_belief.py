"""【探索的】E43（GPU）：teach の (a)(b)(c) に入れる感情の見込み q がどれだけ正確なら，teach の得が戻るか。

規則は `docs/exploratory-analyses.md` E43（設計と予言は計算の前に書いた）。E41 の PPP（3 か所すべてに予測 p を入れた版）の p を，
正確さを段階的に変えた見込み q に替える（`teach_ablate.ablate_order` の belief）。評価・候補・中心・下書き・学習器・手順は E41 と同じ
（1 本選ぶごとに正解が見え，模型の点数の更新は正解の残差で行う）。q の作り方（どれも正解のラベルを使う「答えを一部知る見込み」）：
  tilt β  ：q = softmax(L + β e_y)。その時点の点数を正解の方向へ β だけ寄せる。q − p は常に正解の向きを指す（楽観の側）
  noisy ρ ：各候補の感情 ỹ を，確率 ρ で正解，残りは下書きの確率 p₀ から引いた感情にし，q = e_ỹ（選ぶ間は固定）。外れた ỹ は自信をもって外す（悲観の側）
  nbr     ：近い 10 本の正解の数と自分の p を足して 11 で割る（E42 の近所から作れる見込みの上限の目安）
LLL（q = 正解）・PPP（q = p）・random_true・random_pool は E41 の値（同じ計算・同じ GPU）を使う。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/teach_belief.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/teach_belief.py --analyze

出力：`results/teach_belief_<kind>/<ターゲット>.csv`（policy, rep, K, uar），`<ターゲット>_q.csv`（各 q の候補の上の UAR と正解率．下書きの状態），
`<ターゲット>_sel.csv`，`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e43_targets.csv`・`e43_tests.csv`
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
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from belief_check import knn  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import ht_block_present  # noqa: E402
from select_teach import CS, pname  # noqa: E402
from teach_ablate import ablate_order  # noqa: E402

TILTS = (0.25, 0.5, 1.0, 2.0)
RHOS = (0.25, 0.5, 0.75, 0.9)
K_NBR = 10
VARIANTS = [f"tilt{b:g}" for b in TILTS] + [f"noisy{r:g}" for r in RHOS] + ["nbr"]
MAIN = ["audio_wavlm", "audio_hubert"]
OUT_REL = ROOT / "results" / "explore_relation"


def beliefs(yq: np.ndarray, L0: torch.Tensor, X: torch.Tensor, seed: str) -> dict:
    """各 q の belief(L, P)。noisy の ỹ は ρ で入れ子（同じ乱数で，ρ が大きいほど正解の候補が増える）。"""
    yt = torch.as_tensor(yq, device=L0.device)
    oh = torch.nn.functional.one_hot(yt, 4).float()
    P0 = torch.softmax(L0.float(), 1).cpu().numpy()
    g = np.random.default_rng(zlib.crc32(seed.encode()))
    u1, u2 = g.random(len(yq)), g.random(len(yq))
    samp = (P0.cumsum(1) < u2[:, None]).sum(1).clip(max=3)                            # 下書きの確率から引いた感情
    out = {f"tilt{b:g}": (lambda L, P, b=b: torch.softmax(L + b * oh, 1)) for b in TILTS}
    for r in RHOS:
        yn = torch.as_tensor(np.where(u1 < r, yq, samp), device=L0.device)
        Q = torch.nn.functional.one_hot(yn, 4).float()
        out[f"noisy{r:g}"] = lambda L, P, Q=Q: Q
    V = oh[knn(X, K_NBR)].sum(1)
    out["nbr"] = lambda L, P, V=V: (V + P) / (K_NBR + 1)
    return out


def evaluate(kind, ev, src, C, reps, ks, rows, srows, qrows):
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {f"{v}_{pname(c)}": {} for v in VARIANTS for c in CS}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        L0 = src.logits(Xc, C)
        X = ((Xc - src.mu) / src.sd).float()
        bel = beliefs(yq, L0, X, f"e43|{ev.name}|{r}")
        Lf = L0.float()
        P0 = torch.softmax(Lf, 1)
        yt = torch.as_tensor(yq, device=Lf.device)
        for v, f in [("p", lambda L, P: P)] + list(bel.items()):                       # 下書きの状態での q の正確さ
            Q = f(Lf, P0).argmax(1)
            qrows.append({"target": ev.name, "variant": v, "rep": r, "uar": CB.uar(yq, Q.cpu().numpy()),
                          "acc": float((Q == yt).float().mean())})
        Km = X @ X.T
        Km.fill_diagonal_(0.0)
        for v, f in bel.items():
            for c in CS:
                order, turns, fills = ablate_order(Km, yq, L0, c, 4 * kmax, "PPP", belief=f)
                pn = f"{v}_{pname(c)}"
                for K in ks:
                    n = 4 * int(K)
                    pp = np.array(order[:n])
                    if len(np.unique(pp)) != n:
                        raise ValueError(f"選んだ本数がおかしい: {pn} r{r} K{K}")
                    idx[pn][(r, K)] = pool[pp]
                    cnt = np.bincount(yq[pp], minlength=4)
                    srows.append({"target": ev.name, "policy": pn, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                                  "min_count": int(cnt.min()), "max_count": int(cnt.max()),
                                  "turn_match": float((yq[pp] == np.array(turns[:n])).mean()), "filled": int(sum(fills[:n]))})
        del Km
    for pn in idx:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[pn]

        def add(r, K, c, u, cv, L, cen=0, cb="", pn=pn):
            rows.append({"target": ev.name, "policy": pn, "rep": r, "K": K, "uar": u})
        ht_block_present(sub, reps, src, C, add)


def run_target(kind, T, data, reps, ks, out):
    corp = lambda g: GROUP_SPEC[g][0]  # noqa: E731
    pool_T = [s for s in data if corp(s) != corp(T)]
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    Xp = torch.cat([data[s].X for s in pool_T])
    yp = M.labels(np.concatenate([data[s].y for s in pool_T]))
    src = logreg.fit(Xp - Xp.mean(0), yp, (C_T,))
    del Xp
    rows, srows, qrows = [], [], []
    evaluate(kind, data[T], src, C_T, reps, ks, rows, srows, qrows)
    pd.DataFrame(qrows).to_csv(out / f"{T}_q.csv", index=False)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN) -> int:
    star = pd.read_csv(OUT_REL / "e37_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"]).policy_star
    e41 = pd.read_csv(OUT_REL / "e41_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    tabs = []
    for kind in kinds:
        d = ROOT / "results" / f"teach_belief_{kind}"
        fs = [f for f in sorted(d.glob("*.csv")) if not f.stem.endswith(("_sel", "_q"))]
        if not fs:
            continue
        a = pd.concat([pd.read_csv(f, dtype={"K": str}) for f in fs])
        a = a[a.rep.isin(EC.REPS)]
        if sorted(a.rep.unique()) != list(EC.REPS):
            print(f"注意：{kind} の反復が E41 と違う（{sorted(a.rep.unique())}）。E41 の値は反復 0〜9 の平均")
        u = a.groupby(["target", "policy", "K"]).uar.mean() * 100
        q = pd.concat([pd.read_csv(f) for f in sorted(d.glob("*_q.csv"))])
        q = q[q.rep.isin(a.rep.unique())].groupby(["target", "variant"]).uar.mean() * 100
        for T in sorted(a.target.unique()):
            cs = float(star.loc[(kind, T)].split("_c")[1])
            for K in ("1", "5", "10", "25"):
                b = e41.loc[(kind, T, K)]
                row = {"kind": kind, "target": T, "K": K, "step": cs, "q_p": float(q.loc[(T, "p")])}
                for x in ("random_true", "random_pool", "LLL", "PPP", "LLL_oracle", "PPP_oracle"):
                    row[x] = float(b[x])
                for v in VARIANTS:
                    vals = {c: float(u.loc[(T, f"{v}_{pname(c)}", K)]) for c in CS}
                    row[v] = vals[cs]
                    row[f"{v}_oracle"] = max(vals.values())
                    row[f"q_{v}"] = float(q.loc[(T, v)])
                tabs.append(row)
    D = pd.DataFrame(tabs)
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        gain = float((g.LLL - g.random_true).mean())
        row = {"kind": kind, "K": K, "targets": len(g), "random_true": float(g.random_true.mean()),
               "random_pool": float(g.random_pool.mean()), "teach_gain": gain}
        for v in ["PPP"] + VARIANTS + ["LLL"]:
            qv = {"PPP": g.q_p, "LLL": pd.Series([100.0])}.get(v, g.get(f"q_{v}"))
            row[f"q:{v}"] = float(qv.mean())
            row[f"uar:{v}"] = float(g[v].mean())
            row[f"share:{v}"] = float((g[v] - g.random_true).mean()) / gain if gain else np.nan
            row[f"-pool:{v}"] = float((g[v] - g.random_pool).mean())
            row[f">pool:{v}"] = int((g[v] > g.random_pool).sum())
            row[f"oracle-main:{v}"] = float((g[f"{v}_oracle"] - g[v]).mean())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in MAIN:                                                                # 判定は K=5 の W7・Hu7
        k5 = R[(R.kind == kind) & (R.K == "5")]
        if not len(k5):
            continue
        k5 = k5.iloc[0]
        R.loc[R.kind == kind, "C1"] = bool(k5["-pool:nbr"] <= 0)
        for fam, vs in (("tilt", [f"tilt{b:g}" for b in TILTS]), ("noisy", [f"noisy{r:g}" for r in RHOS])):
            ok = [k5[f"q:{v}"] for v in vs if k5[f"share:{v}"] >= 0.5]
            R.loc[R.kind == kind, f"C2_{fam}_qmin"] = min(ok) if ok else np.nan
            R.loc[R.kind == kind, f"C2_{fam}"] = bool(not ok or min(ok) >= 75)
    D.to_csv(OUT_REL / "e43_targets.csv", index=False)
    R.to_csv(OUT_REL / "e43_tests.csv", index=False)
    pd.set_option("display.width", 250)
    for kind, g in R.groupby("kind", sort=False):
        print(f"== {kind}（random_true {g.random_true.round(2).tolist()}，random_pool {g.random_pool.round(2).tolist()}，K = {g.K.tolist()}）")
        vs = ["PPP"] + VARIANTS + ["LLL"]
        tab = pd.DataFrame([{"policy": v, "q_uar": g[f"q:{v}"].iloc[0],
                             **{f"share_K{k}": s for k, s in zip(g.K, g[f"share:{v}"])},
                             **{f"-pool_K{k}": s for k, s in zip(g.K, g[f"-pool:{v}"])},
                             "K5_>pool": int(g.loc[g.K == "5", f">pool:{v}"].iloc[0]),
                             "K5_oracle-main": float(g.loc[g.K == "5", f"oracle-main:{v}"].iloc[0])} for v in vs])
        print(tab.round(2).to_string(index=False))
    vc = [c for c in R.columns if c.startswith("C")]
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
    out = args.out_dir or ROOT / "results" / f"teach_belief_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/teach_belief.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "variants": VARIANTS, "steps": list(CS),
            "k_nbr": K_NBR, "cp": CB.DEF_CP, "cb": CB.DEF_CB, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, args.reps, args.ks, out)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
