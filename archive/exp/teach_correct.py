"""【探索的】E48（GPU）：ラベルが付くたびに見込み q を直しながら，教える選び方で選ぶ（条件 2 の段 3）。

規則は `docs/exploratory-analyses.md` E48（設計と予言は計算の前に書いた）。E45 の C の版（q の最大の感情の 1-hot を E41 の PPP の 3 か所に入れる）を，
ラベルが EVERY 本付くたび（感情の番が 1 巡するたび）に E47 の直し方で q を直して入れ直す版に変える。直し方のつまみは E47 が仮のターゲットで選んだ値。
版の名前は「まとめ方:直し方の組」（組は hab・rel・spr・mix の 0／1．sel は E47 の仮のターゲットの UAR が最も高い組をターゲットごとに選ぶ）：
  C:0000 ：直さない（E45 の C と同じ順になることを確かめる）
  C:0001 ：④ 学び直した分類器と混ぜるだけ（E47 で上がり幅のほぼすべてを占めた直し方）
  C:sel  ：C の 16 通りから，そのターゲットの仮のターゲットで UAR が最も高い組（組が C:0001 と同じなら計算を使い回す）
評価・候補・中心・出発点・学習器・手順・歩幅（E37 の値）は E45 と同じ。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/teach_correct.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/teach_correct.py --analyze

出力：`results/teach_correct_<kind>/<ターゲット>.csv`（policy, rep, K, uar），`<ターゲット>_q.csv`（q を直すたびの，まだラベルの無い候補での q の UAR），
`<ターゲット>_sel.csv`（選んだ発話の感情ごとの本数と使った組），`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e48_targets.csv`・`e48_tests.csv`
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
from belief_correct import AGGS, CORR, Task, knob_key  # noqa: E402
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import ht_block_present, present_weights  # noqa: E402
from teach_ablate import ablate_order  # noqa: E402

VARIANTS = ["C:0000", "C:0001", "C:sel"]
EVERY = 4
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"


def fit_pm(task: Task, R: np.ndarray) -> np.ndarray:
    """付いたラベル R で学び直した仮説転移 4（E47 の `Task.fit_pm` と同じ設定）の候補の上の確率。"""
    X = task.Xc[torch.as_tensor(R, device=task.Xc.device)]
    yX = M.labels(task.y[R])
    w = torch.zeros(len(yX), dtype=logreg.DTYPE, device=CB.DEV)
    w[:] = present_weights(yX)
    fit = logreg.fit_batch(X, yX, [np.ones(len(R), bool)], (CB.DEF_CP,), anchor=task.src, anchor_c=task.C_anchor,
                           stats=(task.src.mu, task.src.sd), Cb=[CB.DEF_CB], weights=[w])[0]
    return torch.softmax(fit.logits(task.Xc, CB.DEF_CP), 1).cpu().numpy()


class Live:
    """ラベルが EVERY 本付くたびに q を直し，その最大の感情の 1-hot を返す見込み（`ablate_order` の belief．order を受け取る）。"""

    def __init__(self, task: Task, agg: str, combo: str, knobs: dict, dev):
        self.task, self.agg, self.combo, self.knobs, self.dev = task, agg, combo, knobs, dev
        self.m, self.log = -1, []
        self._set(task.q0[agg], np.array([], int))

    def _set(self, q, R):
        self.oh = torch.nn.functional.one_hot(torch.as_tensor(q.argmax(1), device=self.dev), 4).float()
        self.m = len(R)
        self.log.append({"m": self.m, **self.task.score(q, R)})

    def __call__(self, L, P, order):
        m = len(order)
        if m > self.m and m % EVERY == 0:
            R = np.array(order)
            pm = fit_pm(self.task, R) if self.combo[3] == "1" else None
            self._set(self.task.corrected(self.agg, self.combo, R, m, pm, self.knobs), R)
        return self.oh


def pick_combo(T, kind, agg) -> str:
    """E47 で，そのターゲットの仮のターゲット（T のソースを 1 つずつ）の UAR（仮のターゲット・順・m の平均）が最も高い組。"""
    ps = pd.read_csv(ROOT / "results" / f"belief_correct_{kind}" / f"{T}_pseudo.csv")
    ps = ps[(ps["agg"] == agg) & ps["corr"].str.startswith("combo")]
    g = ps.assign(combo=ps["corr"].str[5:]).groupby("combo").uar.mean()
    return str(g.idxmax())


def run_target(kind, T, data, f, Cs, P, Y, reps, ks, variants, out):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    H, Hs = CM.habits(P, Y, S, S)
    e44 = pd.read_csv(ROOT / "results" / f"committee_{kind}" / f"{T}.csv")
    lam = float(e44[e44.variant == "C"].lam.iloc[0])
    pool_C = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), lam)
    e47 = pd.read_csv(ROOT / "results" / f"belief_correct_{kind}" / f"{T}.csv")
    knobs = {agg: {knob_key(c, agg): float(e47[f"knob_{agg}:{c}"].iloc[0]) for c in CORR} for agg in AGGS}
    combos = {}
    for v in variants:
        agg, cb = v.split(":")
        combos[v] = (agg, pick_combo(T, kind, agg) if cb == "sel" else cb)
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    Xp = torch.cat([data[s].X for s in S])
    yp = M.labels(np.concatenate([data[s].y for s in S]))
    src = logreg.fit(Xp - Xp.mean(0), yp, (C_T,))
    del Xp
    star = pd.read_csv(OUT_REL / "e37_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"]).policy_star
    c_main = float(star.loc[(kind, T)].split("_c")[1])
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {v: {} for v in variants}
    rows, srows, qrows = [], [], []
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        X = ((Xc - src.mu) / src.sd).float()
        Ps = [torch.softmax(f[s].logits(Xc, Cs[s]), 1).cpu().numpy() for s in S]
        task = Task(Ps, S, H, Hs, pool_C, yq, X, Xc, src, C_T)
        L0 = src.logits(Xc, C_T)
        Km = X @ X.T
        Km.fill_diagonal_(0.0)
        done = {}
        for v, (agg, combo) in combos.items():
            if (agg, combo) not in done:
                live = Live(task, agg, combo, knobs[agg], X.device)
                done[(agg, combo)] = (ablate_order(Km, yq, L0, c_main, 4 * kmax, "PPP", belief=live, belief_order=True), live.log)
            (order, turns, fills), log = done[(agg, combo)]
            qrows += [{"target": T, "policy": v, "combo": combo, "rep": r, **x} for x in log]
            for K in ks:
                n = 4 * int(K)
                pp = np.array(order[:n])
                if len(np.unique(pp)) != n:
                    raise ValueError(f"選んだ本数がおかしい: {v} r{r} K{K}")
                idx[v][(r, K)] = pool[pp]
                cnt = np.bincount(yq[pp], minlength=4)
                srows.append({"target": T, "policy": v, "combo": combo, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                              "min_count": int(cnt.min()), "max_count": int(cnt.max()),
                              "turn_match": float((yq[pp] == np.array(turns[:n])).mean()), "filled": int(sum(fills[:n]))})
        del Km
    for v in idx:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[v]

        def add(r, K, c, u, cv, L, cen=0, cb="", v=v):
            rows.append({"target": T, "policy": v, "combo": combos[v][1], "rep": r, "K": K, "uar": u})
        ht_block_present(sub, reps, src, C_T, add)
    pd.DataFrame(qrows).to_csv(out / f"{T}_q.csv", index=False)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN) -> int:
    e46 = pd.read_csv(OUT_REL / "e46_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    base = ["random_true", "LLL", "random_pool", "herding_pred", "herding_all", "random_pred", "margin", "coreset", "typiclust", "badge", "e45_C"]
    tabs, qtabs = [], []
    for kind in kinds:
        d = ROOT / "results" / f"teach_correct_{kind}"
        fs = [x for x in sorted(d.glob("*.csv")) if not x.stem.endswith(("_sel", "_q"))]
        if not fs:
            continue
        a = pd.concat([pd.read_csv(x, dtype={"K": str}) for x in fs])
        a = a[a.rep.isin(EC.REPS)]
        u = a.groupby(["target", "policy", "K"]).uar.mean() * 100
        q = pd.concat([pd.read_csv(x) for x in sorted(d.glob("*_q.csv"))])
        q = q[q.rep.isin(a.rep.unique()) & q.m.isin([0, 4, 20, 40, 100])]
        qtabs.append(q.groupby(["target", "policy", "m"]).uar.mean().mul(100).rename("q_uar").reset_index().assign(kind=kind))
        sel = pd.concat([pd.read_csv(x, dtype={"combo": str}) for x in sorted(d.glob("*_sel.csv"))]).drop_duplicates(["target", "policy"]).set_index(["target", "policy"]).combo
        for T in sorted(a.target.unique()):
            for K in ("1", "5", "10", "25"):
                row = {"kind": kind, "target": T, "K": K}
                for x in base:
                    row[x] = float(e46.loc[(kind, T, K), x])
                for v in a.policy.unique():
                    row[v] = float(u.loc[(T, v, K)])
                    row[f"combo_{v}"] = sel.loc[(T, v)]
                tabs.append(row)
    D = pd.DataFrame(tabs)
    Qd = pd.concat(qtabs)
    pols = [v for v in VARIANTS if v in D]
    rival = ["herding_pred", "random_pool", "margin", "coreset", "typiclust", "badge"]
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for x in ["random_true", "LLL", "random_pool", "herding_pred", "e45_C"] + pols:
            row[x] = float(g[x].mean())
        for v in pols:
            for x in rival + ["C:0000"]:
                if x == v:
                    continue
                row[f"{v}-{x}"] = float((g[v] - g[x]).mean())
                row[f"{v}>{x}"] = int((g[v] > g[x]).sum())
        row["chk_e45"] = float((g["C:0000"] - g.e45_C).abs().max())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in ("audio_wavlm", "audio_hubert"):
        k5 = R[(R.kind == kind) & (R.K == "5")]
        if len(k5):
            k5 = k5.iloc[0]
            R.loc[R.kind == kind, "V1"] = bool(k5["C:sel-C:0000"] > 0)
            R.loc[R.kind == kind, "V2"] = bool(k5["C:sel-herding_pred"] > 0)
            k25 = R[(R.kind == kind) & (R.K == "25")].iloc[0]
            R.loc[R.kind == kind, "V3"] = bool(k25["C:sel-C:0000"] > k5["C:sel-C:0000"])
    D.to_csv(OUT_REL / "e48_targets.csv", index=False)
    R.to_csv(OUT_REL / "e48_tests.csv", index=False)
    Qd.to_csv(OUT_REL / "e48_q.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in R if c in ("V1", "V2", "V3")]).round(2).T.to_string())
    print("\n== q の UAR（まだラベルの無い候補．m 本付いた時点．ターゲットの平均）")
    print(Qd.groupby(["kind", "policy", "m"]).q_uar.mean().unstack("m").round(2).to_string())
    print("\n== C:sel が選んだ組（ターゲットの数）")
    print(D.drop_duplicates(["kind", "target"]).groupby("kind")["combo_C:sel"].value_counts().to_string())
    vc = [c for c in ("V1", "V2", "V3") if c in R]
    if vc:
        print(R.dropna(subset=vc).drop_duplicates("kind")[["kind"] + vc].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=MAIN)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds)
    out = args.out_dir or ROOT / "results" / f"teach_correct_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    f, Cs, P, Y = CM.prepare(args.kind, data)
    prov = {"script": "exp/teach_correct.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "variants": args.variants, "every": EVERY,
            "src_C": Cs, "cp": CB.DEF_CP, "cb": CB.DEF_CB, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, f, Cs, P, Y, args.reps, args.ks, args.variants, out)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
