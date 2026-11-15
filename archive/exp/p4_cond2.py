"""【探索的】E54（GPU）：P4 の生徒（仮説転移 4 の既定）で，条件 2 の選び方を A（WavLM）・V・T（言葉）でそろえて比べる。

規則は `docs/exploratory-analyses.md` E54（設計と予言は計算の前に書いた）。生徒・評価・候補・中心は E41・E45・E46・E48 と同じ（P4 の既定．
感情の重みは選んだ中にある感情だけ）。教える選び方の歩幅は E37 がメタ学習した値（T は `e37_targets_text.csv`），模型の座標は x̃（P4 の出発点の統計で標準化）。
見込み q は E44 の C（λ は A・V は E44 の値，T は E53 と同じく仮のターゲットを 1 つずつ外して選ぶ）。④ で混ぜる生徒は P4 の生徒（E47・E48 と同じ），
τ は A・V は E47 がターゲットごとに選んだ値，T は 40（E53 と同じ）。HuBERT は使わない（研究者の指示）ので，E52 の pqo は行わない。
  random_true・random_pool ：E41 と同じ
  herding_pred             ：P4 の出発点の予測の群の中で herding（E34 と同じ）
  margin・badge・coreset・typiclust ：E46 と同じ
  LLL      ：条件 1 の教える選び方（E37・E41 と同じ）
  q:fixed  ：q を直さずに入れた教える選び方（E45 の C と同じ）
  q:cycle  ：q を ④ で直す（E48 の C:0001 と同じ）
  pq:cycle・pq:count ：仲間 {P4 の出発点 p, q} の見込み（E52 と同じ作り．仮のターゲット g の上の p は g のコーパスを除くソースの合併（C = 0.01））を ④ で直す．
                      番は 0→3 の順か，付いた本当のラベルが最も少ない感情

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/p4_cond2.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/p4_cond2.py --analyze

出力：`results/p4_cond2_<kind>/<ターゲット>.csv`（policy, rep, K, uar），`<ターゲット>_q.csv`，`<ターゲット>_sel.csv`，`<ターゲット>_meta.csv`，
`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e54_targets.csv`・`e54_tests.csv`・`e54_belief.csv`・`e54_checks.csv`
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
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from belief_check import knn  # noqa: E402
from belief_correct import Task  # noqa: E402
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_baselines import K_TYP, order_badge, order_coreset, order_margin, pick_typiclust  # noqa: E402
from select_herding import herd  # noqa: E402
from select_labelfree import fill, ht_block_present  # noqa: E402
from teach_ablate import ablate_order  # noqa: E402
from teach_correct import fit_pm  # noqa: E402
from teach_stack import Live, c_pool_on, fit_stack, logp, score, softmax_t  # noqa: E402
from text_cond2 import choose_lam  # noqa: E402

KINDS = ["audio_wavlm", "video", "text_roberta"]
POLICIES = ["random_true", "random_pool", "herding_pred", "margin", "badge", "coreset", "typiclust",
            "LLL", "q:fixed", "q:cycle", "pq:cycle", "pq:count"]
TEACHY = ["LLL", "q:fixed", "q:cycle", "pq:cycle", "pq:count"]
C_PSEUDO = 0.01
TAU_TEXT = 40.0
OUT_REL = ROOT / "results" / "explore_relation"


def step_of(kind, T):
    f = "e37_targets_text.csv" if kind == "text_roberta" else "e37_targets.csv"
    st = pd.read_csv(OUT_REL / f, dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"]).policy_star
    return float(st.loc[(kind, T)].split("_c")[1])


def run_target(kind, T, data, f, Cs, P, Y, reps, ks, out, cache):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    H, Hs = CM.habits(P, Y, S, S)
    mrows = []
    if kind == "text_roberta":
        lam, sc = choose_lam(S, P, Y)
        mrows += [{"target": T, "belief": "q", "pseudo": t, "lam": l, "uar": u} for l, us in sc.items() for t, u in zip(S, us)]
        tau = TAU_TEXT
    else:
        lam = float(pd.read_csv(ROOT / "results" / f"committee_{kind}" / f"{T}.csv").query("variant == 'C'").lam.iloc[0])
        tau = float(pd.read_csv(ROOT / "results" / f"belief_correct_{kind}" / f"{T}.csv")["knob_C:mix"].iloc[0])
    pool_q = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), lam)
    mem = {}
    for g in S:                                                                 # 仮のターゲット g の上の仲間（P4 型の出発点と E44 の C）
        pool_g = [s for s in S if CM.corp(s) != CM.corp(g)]
        src_g = M5.pool_info(data, pool_g, C_PSEUDO, cache)[0]
        mem[g] = {"p": softmax_t(src_g.logits(data[g].X - data[g].X.mean(0), C_PSEUDO)), "q": c_pool_on(g, S, P, Y, lam)}
    pool_pq, lam_pq = fit_stack(T, "pq", S, mem, Y, mrows)
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    src = M5.pool_info(data, S, C_T, cache)[0]
    c_t = step_of(kind, T)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {p_: {} for p_ in POLICIES}
    rows, srows, qrows = [], [], []
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        L0 = src.logits(Xc, C_T)
        Xs = (Xc - src.mu) / src.sd
        X = Xs.float()
        Ps = [softmax_t(f[s].logits(Xc, Cs[s])) for s in S]
        task = Task(Ps, S, H, Hs, pool_q, yq, X, Xc, src, C_T)                   # E48 と同じ（q0["C"] が E44 の C）
        p = softmax_t(L0)
        q0 = {"q": task.q0["C"]}
        F = torch.as_tensor(np.stack([logp(p), logp(q0["q"])], 1), dtype=logreg.DTYPE, device=logreg.DEVICE)
        q0["pq"] = pool_pq.predict(F, torch.ones(len(yq), 2, dtype=logreg.DTYPE, device=logreg.DEVICE))
        for nm, v in (("p", p), ("q", q0["q"]), ("pq", q0["pq"])):
            qrows.append({"target": T, "policy": nm, "rep": r, "m": 0, **score(v, np.array([], int), yq)})
        Km = X @ X.T
        Km.fill_diagonal_(0.0)
        pm_fn = lambda R, task=task: fit_pm(task, R)  # noqa: E731
        orders = {"LLL": ablate_order(Km, yq, L0, c_t, 4 * kmax, "LLL")}
        ohq = torch.nn.functional.one_hot(torch.as_tensor(q0["q"].argmax(1), device=X.device), 4).float()
        orders["q:fixed"] = ablate_order(Km, yq, L0, c_t, 4 * kmax, "PPP", belief=lambda L, P_: ohq)
        for v in ("q:cycle", "pq:cycle", "pq:count"):
            b, turn = v.split(":")
            live = Live(q0[b], yq, tau, pm_fn, X.device)
            orders[v] = ablate_order(Km, yq, L0, c_t, 4 * kmax, "PPP", belief=live, belief_order=True, turn=turn)
            qrows += [{"target": T, "policy": v, "rep": r, **x} for x in live.log]
        del Km
        Pf = torch.softmax(L0, 1).float()
        gen = lambda name: torch.Generator().manual_seed(zlib.crc32(f"e46|{name}|{T}|{r}".encode()))  # noqa: E731
        al = {"margin": order_margin(Pf), "coreset": order_coreset(X, 4 * kmax), "badge": order_badge(X, Pf, 4 * kmax, gen("badge"))}
        Xn = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
        nb = knn(X, K_TYP)
        typ = 1.0 / (1.0 - (Xn[:, None, :] * Xn[nb]).sum(2)).mean(1).clamp(min=1e-6)
        pos_of = {int(i): j for j, i in enumerate(pool)}
        perm = np.random.default_rng(zlib.crc32(f"e41pool|{T}|{r}".encode())).permutation(len(pool))
        pred = L0.argmax(1).cpu().numpy()
        groups = [np.flatnonzero(pred == g) for g in range(4)]
        for K in ks:
            kk = int(K)
            picks = {p_: np.array(orders[p_][0][:4 * kk]) for p_ in TEACHY}
            picks.update({p_: np.array(al[p_][:4 * kk]) for p_ in al})
            picks["typiclust"] = np.array(pick_typiclust(X, typ, 4 * kk, gen(f"typiclust{K}")))
            picks["random_true"] = np.array([pos_of[int(i)] for i in ctx.idx[(r, K)]])
            picks["random_pool"] = perm[:4 * kk]
            hord = [list(G[herd(Xs[torch.as_tensor(G, device=CB.DEV)], min(len(G), 4 * kk))]) if len(G) else [] for G in groups]
            picks["herding_pred"] = np.array(fill(groups, [o[:kk] for o in hord], kk, lambda j, ps, hord=hord: hord[j][len(ps)]))
            for p_, pp in picks.items():
                if len(pp) != 4 * kk or len(np.unique(pp)) != len(pp):
                    raise ValueError(f"選んだ本数がおかしい: {p_} r{r} K{K}")
                idx[p_][(r, K)] = pool[pp]
                cnt = np.bincount(yq[pp], minlength=4)
                srows.append({"target": T, "policy": p_, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                              "min_count": int(cnt.min()), "max_count": int(cnt.max())})
    for p_ in POLICIES:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[p_]

        def add(r, K, c, u, cv, L, cen=0, cb="", p_=p_):
            rows.append({"target": T, "policy": p_, "rep": r, "K": K, "uar": u})
        ht_block_present(sub, reps, src, C_T, add)
    meta = pd.DataFrame(mrows)
    meta["chosen"] = [(b == "q" and l == lam) or (b == "pq" and l == lam_pq) for b, l in zip(meta.belief, meta.lam)]
    meta.to_csv(out / f"{T}_meta.csv", index=False)
    pd.DataFrame(qrows).to_csv(out / f"{T}_q.csv", index=False)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=KINDS) -> int:
    e46 = pd.read_csv(OUT_REL / "e46_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    e48 = pd.read_csv(OUT_REL / "e48_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    e37t = pd.read_csv(OUT_REL / "e37_targets_text.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    e53 = pd.read_csv(OUT_REL / "e53_targets.csv", dtype={"K": str}).set_index(["target", "K"])
    tabs, btabs, checks = [], [], []
    for kind in kinds:
        d = ROOT / "results" / f"p4_cond2_{kind}"
        for x in sorted(f for f in d.glob("*.csv") if not f.stem.endswith(("_sel", "_q", "_meta"))):
            T = x.stem
            a = pd.read_csv(x, dtype={"K": str})
            a = a[a.rep.isin(EC.REPS)]
            u = a.groupby(["policy", "K"]).uar.mean() * 100
            q = pd.read_csv(d / f"{T}_q.csv")
            q = q[q.rep.isin(EC.REPS) & (q.m == 0)].groupby("policy").uar.mean() * 100
            btabs.append({"kind": kind, "target": T, "p": float(q["p"]), "q": float(q["q"]), "pq": float(q["pq"])})
            for K in ("1", "5", "10", "25"):
                row = {"kind": kind, "target": T, "K": K, **{p_: float(u.loc[(p_, K)]) for p_ in POLICIES}}
                tabs.append(row)
                ck = {"kind": kind, "target": T, "K": K}
                if kind == "text_roberta":
                    ref = {"random_true": float(e37t.loc[(kind, T, K), "random"]), "LLL": float(e37t.loc[(kind, T, K), "meta"]),
                           "random_pool": float(e53.loc[(T, K), "P4:random_pool"])}
                else:
                    ref = {k_: float(e46.loc[(kind, T, K), k_]) for k_ in ("random_true", "random_pool", "LLL", "herding_pred", "margin",
                                                                          "badge", "coreset", "typiclust")}
                    ref["q:cycle"] = float(e48.loc[(kind, T, K), "C:0001"])
                    if kind == "audio_wavlm":
                        ref["q:fixed"] = float(e46.loc[(kind, T, K), "e45_C"])
                for k_, v in ref.items():
                    ck[k_] = abs(row[k_] - v)
                checks.append(ck)
    D, Bt, Ck = pd.DataFrame(tabs), pd.DataFrame(btabs), pd.DataFrame(checks)
    lf = ["herding_pred", "margin", "badge", "coreset", "typiclust", "q:fixed", "q:cycle", "pq:cycle", "pq:count"]
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for x in POLICIES:
            row[x] = float(g[x].mean())
        for x in lf + ["random_true", "LLL"]:
            diff = g[x] - g.random_pool
            row[f"{x}-random_pool"] = float(diff.mean())
            row[f"{x}>random_pool"] = int((diff > 0).sum())
        for x in ("q:fixed", "q:cycle", "pq:cycle", "pq:count"):
            row[f"{x}-herding_pred"] = float((g[x] - g.herding_pred).mean())
        row["pq:cycle-q:cycle"] = float((g["pq:cycle"] - g["q:cycle"]).mean())
        row["pq:count-pq:cycle"] = float((g["pq:count"] - g["pq:cycle"]).mean())
        tests.append(row)
    R = pd.DataFrame(tests)
    k5 = R[R.K == "5"].set_index("kind")
    if "text_roberta" in k5.index:
        R["Q1"] = bool(k5.loc["text_roberta", "q:fixed-random_pool"] > 0)
    ok = [k for k in ("audio_wavlm", "video") if k in k5.index]
    if ok:
        R["Q2"] = bool(all(k5.loc[k, "pq:cycle-q:cycle"] <= 0 for k in ok))
    R["Q3"] = bool(all(k5.loc[k, "pq:count-pq:cycle"] <= 0 for k in k5.index))
    D.to_csv(OUT_REL / "e54_targets.csv", index=False)
    R.to_csv(OUT_REL / "e54_tests.csv", index=False)
    Bt.to_csv(OUT_REL / "e54_belief.csv", index=False)
    Ck.to_csv(OUT_REL / "e54_checks.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in ("Q1", "Q2", "Q3") if c in R]).set_index(["kind", "K"]).round(2).T.to_string())
    print("\n== 見込みの当たり方（候補の上の UAR．直す前．ターゲットの平均）")
    print(Bt.groupby("kind")[["p", "q", "pq"]].mean().round(2).to_string())
    print("\n== 既存の値との照合（差の絶対値の最大．ポイント）")
    print(Ck.drop(columns=["target", "K"]).groupby("kind").max().round(3).to_string())
    print(R[[c for c in ("Q1", "Q2", "Q3") if c in R]].iloc[:1].to_string(index=False))
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
    out = args.out_dir or ROOT / "results" / f"p4_cond2_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    f, Cs, P, Y = CM.prepare(args.kind, data)
    prov = {"script": "exp/p4_cond2.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "policies": POLICIES, "c_pseudo": C_PSEUDO,
            "tau_text": TAU_TEXT, "src_C": Cs, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, f, Cs, P, Y, args.reps, args.ks, out, cache)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
