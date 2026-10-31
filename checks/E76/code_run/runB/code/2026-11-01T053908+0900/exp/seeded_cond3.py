"""【探索的】E57（GPU）：各感情に 1 本ずつラベルのある見本から始めて，残りのどの発話にラベルを付けるかを選ぶ（問い 5 の条件 3）。

規則は `docs/exploratory-analyses.md` E57（設計と予言は計算の前に書いた）。見本は E31 の K=1 の 4 本（各感情からランダムに 1 本）。
予算は見本を含めて 4K 本（K=5 なら見本 4 本＋選ぶ 16 本）。生徒・評価・候補・中心は E54 と同じ（P4 の既定の仮説転移 4．感情の重みは選んだ中にある感情だけ）。
生徒を使う選び方は，それまでに付いたラベルで生徒を学び直して次のまとまりを選ぶ（`teach_correct.fit_pm`．評価の生徒と同じ）。
まとまりの大きさは 20 本（K=5）までは 4 本（1 巡），40 本（K=10）までは 10 本，100 本（K=25）までは 20 本（計算を軽くするため．K=5 までは 4 本ごと）。
  seed_random ：見本＋ランダム（E41 の random_pool と同じ順から見本を除いて続ける）
  margin      ：生徒の確率の 1 位と 2 位の差が小さい 4 本
  badge       ：勾配の埋め込みに k-means++ の種まきで 4 本（E46 と同じ作りを，ラベルの無い候補の上で）
  coreset     ：見本から始める k-center の貪欲法（生徒を使わない）
  typiclust   ：候補を 4K 個の群に分け，見本の入っていない大きい群から典型的な発話を選ぶ（K ごとに解き直す）
  clue        ：生徒の確率のエントロピーを重みにした k-means（4 群）の中心に近い 4 本（Prabhu ほか 2021）
  tqs         ：ソースの分類器の言い切りの票のエントロピーと生徒の確率のエントロピーの順位の和が大きい 4 本（Fu ほか 2021 の委員会と不確かさ．ドメインらしさは入れない）
  q:cycle     ：教える選び方（見本で模型を更新してから選ぶ）．見込みは E44 の C を ④ で直す（E54 と同じ）
  pq:cycle    ：同じく，見込みは {P4 の出発点, q}（E54 と同じ）
  LLL         ：条件 1 の教える選び方（見本から．上限の目安）

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/seeded_cond3.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/seeded_cond3.py --analyze

出力：`results/seeded_cond3_<kind>/<ターゲット>.csv`（policy, rep, K, uar），`<ターゲット>_sel.csv`，`provenance.json`，`provenance_log.jsonl`。
判定は `results/explore_relation/e57_targets.csv`・`e57_tests.csv`
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
from p4_cond2 import C_PSEUDO, TAU_TEXT, step_of  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_baselines import K_TYP, KM_ITERS, kmeans, order_badge  # noqa: E402
from select_labelfree import ht_block_present  # noqa: E402
from teach_ablate import ablate_order  # noqa: E402
from teach_correct import fit_pm  # noqa: E402
from teach_stack import Live, c_pool_on, fit_stack, logp, softmax_t  # noqa: E402
from text_cond2 import choose_lam  # noqa: E402

KINDS = ["audio_wavlm", "video", "text_roberta"]
POLICIES = ["seed_random", "margin", "badge", "coreset", "typiclust", "clue", "tqs", "q:cycle", "pq:cycle", "LLL"]


def batch_size(m: int) -> int:
    """付いたラベルが m 本のときに次に選ぶまとまりの大きさ。"""
    return 4 if m < 20 else (10 if m < 40 else 20)
OUT_REL = ROOT / "results" / "explore_relation"


def entropy_t(P):
    return -(P * P.clamp(min=1e-12).log()).sum(1)


def rounds(seeds, n_total, refit, pick_batch):
    """見本から始め，batch_size 本ごとに生徒を学び直して pick_batch(生徒の確率, ラベルの無い候補の位置, 本数) で選ぶ。"""
    order = [int(i) for i in seeds]
    while len(order) < n_total:
        P = torch.as_tensor(refit(np.array(order)), dtype=torch.float32, device=CB.DEV)
        un = np.setdiff1d(np.arange(len(P)), order)
        b = min(batch_size(len(order)), n_total - len(order))
        order += [int(i) for i in pick_batch(P, un, b)]
    return order


def clue_batch(X, w, k, g):
    """エントロピーを重みにした k-means（k 群）の中心に最も近い発話（候補の中の位置．重複なし）。"""
    n = len(X)
    pr = (w / w.sum()).cpu()
    cen = [int(torch.multinomial(pr, 1, generator=g))]
    d2 = ((X - X[cen[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        p_ = (w * d2).clamp(min=0).cpu()
        p_[cen] = 0
        i = int(torch.multinomial(p_ / p_.sum(), 1, generator=g))
        cen.append(i)
        d2 = torch.minimum(d2, ((X - X[i]) ** 2).sum(1))
    Cm = X[cen].clone()
    for _ in range(KM_ITERS):
        lab = torch.cdist(X, Cm).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                Cm[j] = (w[m, None] * X[m]).sum(0) / w[m].sum().clamp(min=1e-12)
    picks = []
    D = torch.cdist(Cm, X)
    for j in range(k):
        d = D[j].clone()
        if picks:
            d[torch.as_tensor(picks, device=d.device)] = float("inf")
        picks.append(int(torch.argmin(d)))
    return picks[: min(k, n)]


def coreset_seeded(X, seeds, n_total):
    order = [int(i) for i in seeds]
    dmin = torch.stack([((X - X[i]) ** 2).sum(1) for i in order]).min(0).values
    while len(order) < n_total:
        i = int(torch.argmax(dmin))
        order.append(i)
        dmin = torch.minimum(dmin, ((X - X[i]) ** 2).sum(1))
    return order


def typiclust_seeded(X, typ, seeds, n_total, g):
    """n_total 個の群に分け，見本の入っていない群を大きい順にたどって最も典型的な発話を選ぶ（足りなければ典型的な順に補う）。"""
    lab = kmeans(X, n_total, g)
    covered = {int(lab[i]) for i in seeds}
    sizes = torch.bincount(lab, minlength=n_total)
    out = [int(i) for i in seeds]
    for j in torch.argsort(sizes, descending=True).tolist():
        if len(out) >= n_total:
            break
        if j in covered or sizes[j] == 0:
            continue
        m = torch.nonzero(lab == j).flatten()
        out.append(int(m[torch.argmax(typ[m])]))
    taken = set(out)
    out += [i for i in torch.argsort(typ, descending=True).tolist() if i not in taken][: n_total - len(out)]
    return out[:n_total]


def run_target(kind, T, data, f, Cs, P, Y, reps, ks, out, cache):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    H, Hs = CM.habits(P, Y, S, S)
    mrows = []
    if kind == "text_roberta":
        lam, _ = choose_lam(S, P, Y)
        tau = TAU_TEXT
    else:
        lam = float(pd.read_csv(ROOT / "results" / f"committee_{kind}" / f"{T}.csv").query("variant == 'C'").lam.iloc[0])
        tau = float(pd.read_csv(ROOT / "results" / f"belief_correct_{kind}" / f"{T}.csv")["knob_C:mix"].iloc[0])
    pool_q = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), lam)
    mem = {}
    for g in S:
        pool_g = [s for s in S if CM.corp(s) != CM.corp(g)]
        src_g = M5.pool_info(data, pool_g, C_PSEUDO, cache)[0]
        mem[g] = {"p": softmax_t(src_g.logits(data[g].X - data[g].X.mean(0), C_PSEUDO)), "q": c_pool_on(g, S, P, Y, lam)}
    pool_pq, _ = fit_stack(T, "pq", S, mem, Y, mrows)
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    src = M5.pool_info(data, S, C_T, cache)[0]
    c_t = step_of(kind, T)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    n_total = 4 * kmax
    idx = {p_: {} for p_ in POLICIES}
    srows = []
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        pos_of = {int(i): j for j, i in enumerate(pool)}
        seeds = [pos_of[int(i)] for i in ctx.idx[(r, "1")]]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        L0 = src.logits(Xc, C_T)
        X = ((Xc - src.mu) / src.sd).float()
        Ps = [softmax_t(f[s].logits(Xc, Cs[s])) for s in S]
        task = Task(Ps, S, H, Hs, pool_q, yq, X, Xc, src, C_T)
        p0 = softmax_t(L0)
        q0 = {"q": task.q0["C"]}
        F = torch.as_tensor(np.stack([logp(p0), logp(q0["q"])], 1), dtype=logreg.DTYPE, device=logreg.DEVICE)
        q0["pq"] = pool_pq.predict(F, torch.ones(len(yq), 2, dtype=logreg.DTYPE, device=logreg.DEVICE))
        refit = lambda R, task=task: fit_pm(task, R)  # noqa: E731
        gen = lambda name: torch.Generator().manual_seed(zlib.crc32(f"e57|{name}|{T}|{r}".encode()))  # noqa: E731
        votes = torch.as_tensor(np.stack([np.bincount(row, minlength=4) for row in np.stack(Ps, 1).argmax(2)]) / len(S),
                                dtype=torch.float32, device=CB.DEV)
        vote_ent = entropy_t(votes)
        orders = {}
        perm = np.random.default_rng(zlib.crc32(f"e41pool|{T}|{r}".encode())).permutation(len(pool))
        orders["seed_random"] = seeds + [int(i) for i in perm if int(i) not in set(seeds)][: n_total - 4]
        orders["margin"] = rounds(seeds, n_total, refit, lambda P_, un, b: un[torch.argsort(
            (lambda t: t[:, 0] - t[:, 1])(P_[torch.as_tensor(un, device=P_.device)].topk(2, 1).values)).cpu().numpy()[:b]])
        g_b = gen("badge")
        orders["badge"] = rounds(seeds, n_total, refit, lambda P_, un, b: un[np.array(order_badge(
            X[torch.as_tensor(un, device=X.device)], P_[torch.as_tensor(un, device=P_.device)], b, g_b))])
        g_c = gen("clue")
        orders["clue"] = rounds(seeds, n_total, refit, lambda P_, un, b: un[np.array(clue_batch(
            X[torch.as_tensor(un, device=X.device)], entropy_t(P_[torch.as_tensor(un, device=P_.device)]), b, g_c))])

        def tqs_pick(P_, un, b):
            u = torch.as_tensor(un, device=P_.device)
            sc = torch.argsort(torch.argsort(vote_ent[u])).float() + torch.argsort(torch.argsort(entropy_t(P_[u]))).float()
            return un[torch.argsort(sc, descending=True).cpu().numpy()[:b]]
        orders["tqs"] = rounds(seeds, n_total, refit, tqs_pick)
        orders["coreset"] = coreset_seeded(X, seeds, n_total)
        Km = X @ X.T
        Km.fill_diagonal_(0.0)
        orders["LLL"] = ablate_order(Km, yq, L0, c_t, n_total, "LLL", init=seeds)[0]
        for v, b_ in (("q:cycle", "q"), ("pq:cycle", "pq")):
            live = Live(q0[b_], yq, tau, refit, X.device)
            orders[v] = ablate_order(Km, yq, L0, c_t, n_total, "PPP", belief=live, belief_order=True, init=seeds)[0]
        del Km
        Xn = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
        nb = knn(X, K_TYP)
        typ = 1.0 / (1.0 - (Xn[:, None, :] * Xn[nb]).sum(2)).mean(1).clamp(min=1e-6)
        for K in ks:
            n = 4 * int(K)
            picks = {p_: np.array(orders[p_][:n]) for p_ in orders}
            picks["typiclust"] = np.array(typiclust_seeded(X, typ, seeds, n, gen(f"typiclust{K}")))
            for p_, pp in picks.items():
                if len(pp) != n or len(np.unique(pp)) != n or not set(seeds) <= set(pp.tolist()):
                    raise ValueError(f"選んだ本数か見本がおかしい: {p_} r{r} K{K}")
                idx[p_][(r, K)] = pool[pp]
                cnt = np.bincount(yq[pp], minlength=4)
                srows.append({"target": T, "policy": p_, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                              "min_count": int(cnt.min()), "max_count": int(cnt.max())})
    rows = []
    for p_ in POLICIES:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[p_]

        def add(r, K, c, u, cv, L, cen=0, cb="", p_=p_):
            rows.append({"target": T, "policy": p_, "rep": r, "K": K, "uar": u})
        ht_block_present(sub, reps, src, C_T, add)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=KINDS) -> int:
    e46 = pd.read_csv(OUT_REL / "e46_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    e37t = pd.read_csv(OUT_REL / "e37_targets_text.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    tabs = []
    for kind in kinds:
        d = ROOT / "results" / f"seeded_cond3_{kind}"
        for x in sorted(f for f in d.glob("*.csv") if not f.stem.endswith("_sel")):
            T = x.stem
            a = pd.read_csv(x, dtype={"K": str})
            a = a[a.rep.isin(EC.REPS)]
            u = a.groupby(["policy", "K"]).uar.mean() * 100
            for K in sorted(a.K.unique(), key=int):
                row = {"kind": kind, "target": T, "K": K, **{p_: float(u.loc[(p_, K)]) for p_ in POLICIES}}
                ref = float(e37t.loc[(kind, T, K), "random"]) if kind == "text_roberta" else float(e46.loc[(kind, T, K), "random_true"])
                row["random_true"] = ref
                if K == "1":
                    row["chk_k1"] = abs(row["seed_random"] - ref)
                tabs.append(row)
    D = pd.DataFrame(tabs)
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for x in POLICIES + ["random_true"]:
            row[x] = float(g[x].mean())
        for x in POLICIES[1:] + ["random_true"]:
            diff = g[x] - g.seed_random
            row[f"{x}-seed_random"] = float(diff.mean())
            row[f"{x}>seed_random"] = int((diff > 0).sum())
        if K == "1":
            row["chk_k1"] = float(g.chk_k1.max())
        tests.append(row)
    R = pd.DataFrame(tests)
    k5 = R[R.K == "5"].set_index("kind")
    if len(k5):
        R["SD1"] = bool((k5["margin-seed_random"] > 0).sum() >= 2)
        R["SD2"] = bool((k5["pq:cycle-seed_random"] > 0).sum() >= 2)
        R["SD3"] = bool((k5["LLL-seed_random"] >= 4).all())
    D.to_csv(OUT_REL / "e57_targets.csv", index=False)
    R.to_csv(OUT_REL / "e57_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in ("SD1", "SD2", "SD3") if c in R]).set_index(["kind", "K"]).round(2).T.to_string())
    print("\n== K=5 のターゲットごと（seed_random との差）")
    k5d = D[D.K == "5"].set_index(["kind", "target"])
    print(k5d[POLICIES[1:]].sub(k5d.seed_random, axis=0).round(2).to_string())
    print(R[[c for c in ("SD1", "SD2", "SD3") if c in R]].iloc[:1].to_string(index=False))
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
    out = args.out_dir or ROOT / "results" / f"seeded_cond3_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    f, Cs, P, Y = CM.prepare(args.kind, data)
    prov = {"script": "exp/seeded_cond3.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "policies": POLICIES, "batch": "4 (<20), 10 (<40), 20",
            "src_C": Cs, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
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
