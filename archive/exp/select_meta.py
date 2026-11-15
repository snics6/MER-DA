"""【探索的】E36（GPU）：ラベル無しの選び方の族 π(η, β) を，ソースコーパスを仮のターゲットにした適応後の UAR で選ぶ（メタ学習）。

規則は `docs/exploratory-analyses.md` E36（設計と予言 M1・M2 は計算の前に書いた）。評価・候補・中心・下書き・学習器・感情の重みは
E34（`exp/select_labelfree.py`）と同じ（P4 の既定）。予算は 4K 本で，選んだ発話にだけラベルが付く。
π(η, β)：下書きの予測の群 k に n_k ∝ π̂_k^(1−η) 本を割り当て（最大剰余法．足りない分は E34 の `fill` と同じく群を回って補う），
群の中で ψ = [x̃ ; β c (p ⊗ x̃)] を herding する。(η, β) = (1, 0) は E34 の herding_pred と同じ。
ターゲット T では 9 通りを反復 0〜9 で解き（role=target），T のソースコーパス c ごとに，T と c のコーパスの群を除くソースの合併
（C = 0.01）を下書きにして c で 9 通りを反復 0〜4 で解く（role=pseudo．T のラベルは使わない）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/select_meta.py --kind audio_wavlm [--target …]
    ./.venv/bin/python exp/select_meta.py --analyze        # 判定（CPU．select_lf_・select_herdfeat_ が要る）

出力：`results/select_meta_<kind>/<ターゲット>.csv`（role, eval, policy, rep, K, uar），`<ターゲット>_sel.csv`（感情の欠け・純度），
`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e36_targets.csv`・`e36_tests.csv`
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
from select_labelfree import fill, ht_block_present  # noqa: E402

ETAS, BETAS = (0.0, 0.5, 1.0), (0.0, 1.0, 2.0)
POLICIES = [(e, b) for e in ETAS for b in BETAS]
PSEUDO_REPS = [0, 1, 2, 3, 4]
C_PSEUDO = 0.01
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"


def pname(e: float, b: float) -> str:
    return f"e{e:g}_b{b:g}"


def first_quota(sizes: np.ndarray, K: int, eta: float) -> np.ndarray:
    """群ごとの最初の割り当て（最大剰余法）。η = 1 なら群の大きさによらず K 本ずつ（0⁰ = 1）。"""
    w = (sizes / max(sizes.sum(), 1)) ** (1 - eta)
    share = 4 * K * w / w.sum()
    n = np.floor(share).astype(int)
    for j in np.argsort(-(share - n), kind="stable")[: 4 * K - n.sum()]:
        n[j] += 1
    return n


def psi(Xg: torch.Tensor, Pg: torch.Tensor, beta: float) -> torch.Tensor:
    if beta == 0:
        return Xg
    B = (Pg[:, :, None] * Xg[:, None, :]).reshape(len(Xg), -1)
    c = torch.sqrt(Xg.var(0, unbiased=False).sum() / B.var(0, unbiased=False).sum().clamp(min=1e-30))
    return torch.cat([Xg, beta * c * B], 1)


def select_all(ctx, r, src, C, ks) -> tuple[dict, dict]:
    """方針ごと・K ごとに選んだ候補の位置（pool の中の番号）と，予測の群・pool。"""
    tgt = ctx.tgt
    pool = np.flatnonzero(~ctx.te[r])
    Xc = M.rows(tgt.X, pool) - ctx.ct[r]
    L = src.logits(Xc, C)
    pred = L.argmax(1).cpu().numpy()
    P = torch.softmax(L, 1)
    Xt = (Xc - src.mu) / src.sd
    groups = [np.flatnonzero(pred == g) for g in range(4)]
    sizes = np.array([len(G) for G in groups], dtype=float)
    kmax = max(int(K) for K in ks)
    orders = {}
    for beta in BETAS:
        orders[beta] = []
        for G in groups:
            if not len(G):
                orders[beta].append([])
                continue
            gi = torch.as_tensor(G, device=CB.DEV)
            orders[beta].append(list(G[herd(psi(Xt[gi], P[gi], beta), min(len(G), 4 * kmax))]))
    sel = {}
    for eta, beta in POLICIES:
        od = orders[beta]
        for K in ks:
            kk = int(K)
            q = first_quota(sizes, kk, eta)
            picks = [od[j][: min(int(q[j]), len(od[j]))] for j in range(4)]
            pp = np.array(fill(groups, picks, kk, lambda j, ps: od[j][len(ps)]))
            if len(pp) != 4 * kk or len(np.unique(pp)) != len(pp):
                raise ValueError(f"選んだ本数がおかしい: {pname(eta, beta)} r{r} K{K} {len(pp)}")
            sel[(pname(eta, beta), K)] = pp
    return sel, {"pool": pool, "pred": pred}


def evaluate(role, meta_for, ev, src, C, reps, ks, rows, sel_rows):
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    idx = {pname(e, b): {} for e, b in POLICIES}
    for r in reps:
        sel, info = select_all(ctx, r, src, C, ks)
        yq = ev.y[info["pool"]]
        for (pn, K), pp in sel.items():
            idx[pn][(r, K)] = info["pool"][pp]
            cnt = np.bincount(yq[pp], minlength=4)
            sel_rows.append({"role": role, "meta_for": meta_for, "eval": ev.name, "policy": pn, "rep": r, "K": K,
                             "missing": int((cnt == 0).sum()), "min_count": int(cnt.min()), "max_count": int(cnt.max()),
                             "purity": float((yq[pp] == info["pred"][pp]).mean())})
    for pn in idx:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[pn]

        def add(r, K, c, u, cv, L, cen=0, cb="", pn=pn):
            rows.append({"role": role, "meta_for": meta_for, "eval": ev.name, "policy": pn, "rep": r, "K": K, "uar": u})
        ht_block_present(sub, reps, src, C, add)


def run_target(kind, T, data, reps, ks, out, cache):
    t0 = time.perf_counter()
    rows, sel_rows = [], []
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

    evaluate("target", T, data[T], draft(pool_T, C_T), C_T, reps, ks, rows, sel_rows)
    t1 = time.perf_counter()
    for c in pool_T:
        pool_c = [s for s in pool_T if corp(s) != corp(c)]
        evaluate("pseudo", T, data[c], draft(pool_c, C_PSEUDO), C_PSEUDO, PSEUDO_REPS, ks, rows, sel_rows)
    print(f"  {kind} {T}：ターゲット {t1 - t0:.0f} 秒，仮のターゲット {len(pool_T)} 個 {time.perf_counter() - t1:.0f} 秒", flush=True)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(out / f"{T}_sel.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze() -> int:
    tabs, tests = [], []
    for kind in MAIN:
        fs = [f for f in sorted((ROOT / "results" / f"select_meta_{kind}").glob("*.csv")) if not f.stem.endswith("_sel")]
        if not fs:
            continue
        a = pd.concat([pd.read_csv(f, dtype={"K": str}) for f in fs])
        pseudo = a[a.role == "pseudo"].groupby(["meta_for", "policy"]).uar.mean()
        tgt = a[(a.role == "target") & a.rep.isin(EC.REPS)].groupby(["eval", "policy", "K"]).uar.mean() * 100
        for T in sorted(a.meta_for.unique()):
            star = pseudo.loc[T].idxmax()
            lf = pd.read_csv(ROOT / "results" / f"select_lf_{kind}" / f"{T}.csv", dtype={"K": str})
            hf = pd.read_csv(ROOT / "results" / f"select_herdfeat_{kind}" / f"{T}.csv", dtype={"K": str})
            for K in ("1", "5", "10", "25"):
                row = {"kind": kind, "target": T, "K": K, "policy_star": star, "meta": float(tgt.loc[(T, star, K)]),
                       "oracle": float(tgt.loc[T].xs(K, level="K").max()),
                       "policy_oracle": str(tgt.loc[T].xs(K, level="K").idxmax()),
                       "e1_b0": float(tgt.loc[(T, "e1_b0", K)])}
                for st in ("random_pred", "herding_pred", "herding_all", "random_true"):
                    row[st] = float(lf[(lf.source == st) & (lf.K == K)].uar.mean() * 100)
                row["herding_feat"] = float(hf[(hf.source == "herding_feat") & (hf.K == K)].uar.mean() * 100)
                tabs.append(row)
    D = pd.DataFrame(tabs)
    for (kind, K), g in D.groupby(["kind", "K"]):
        row = {"kind": kind, "K": K, "targets": len(g), "check_e1b0_vs_herding_pred": float((g.e1_b0 - g.herding_pred).abs().max())}
        for b in ("herding_pred", "random_pred", "random_true", "herding_feat"):
            diff = g.meta - g[b]
            row[f"meta-{b}"] = float(diff.mean())
            row[f"meta>{b}"] = int((diff > 0).sum())
        row["oracle-random_pred"] = float((g.oracle - g.random_pred).mean())
        gap = (g.herding_feat - g.random_pred).mean()
        row["recovered"] = float((g.meta - g.random_pred).mean() / gap) if gap > 0 else np.nan
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in ("audio_wavlm", "audio_hubert"):
        m1 = (R.kind == kind) & R.K.isin(["5", "10"])
        m2 = (R.kind == kind) & R.K.isin(["1", "5", "10"])
        R.loc[R.kind == kind, "M1"] = bool((R[m1]["meta-herding_pred"] > 0).all()) if m1.sum() == 2 else np.nan
        R.loc[R.kind == kind, "M2"] = bool((R[m2]["meta-random_pred"] > 0).all()) if m2.sum() == 3 else np.nan
    D.to_csv(OUT_REL / "e36_targets.csv", index=False)
    R.to_csv(OUT_REL / "e36_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print(D.groupby("kind").policy_star.value_counts().to_string())
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
    out = args.out_dir or ROOT / "results" / f"select_meta_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/select_meta.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "pseudo_reps": PSEUDO_REPS,
            "policies": [pname(e, b) for e, b in POLICIES], "C_pseudo": C_PSEUDO, "cp": CB.DEF_CP, "cb": CB.DEF_CB,
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
