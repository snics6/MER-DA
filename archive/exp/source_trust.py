"""【探索的】E55（GPU）：ソースの発話ごとの信頼度で信頼度の低い発話を外して出発点を作り直すと，ターゲットの UAR は上がるか（問い 6 の最初の実験）。

規則は `docs/exploratory-analyses.md` E55（設計と予言は計算の前に書いた）。
信頼度：ソース群 s の発話 x（ラベル y）について，s と別のコーパスのソース群 s′ の分類器 f_{s′}（E44 と同じ．各群の全件で学ぶ）が x を y と言う確率の平均。
ターゲット T のソースは T と同じコーパスの群を除く群 S で，信頼度に使う s′ も S の中に限る（T の情報は使わない）。
各ソース群 × 感情の中で信頼度の下位 r％ を外し，残りの合併で P4 の出発点（合併の C は T の地図の値．E41 と同じ）を作り直す。
外す割合 r ∈ {0, 10, 25, 50, 75} は，S の群を 1 つずつ仮のターゲット c にして選ぶ（c のコーパスを除く群だけで信頼度と出発点を作る．合併の C は 0.01，
K ∈ {1, 5}，反復 0〜4 の UAR の平均．E39・E40 の仮のターゲットと同じ作り）。比べる相手：全部使う版（r = 0．今の P4），同じ数をランダムに外す版，
テストのラベルで選んだ最良の r（オラクル）。生徒・評価・K 本は E41 の random_true と同じ（P4 の既定の仮説転移 4．K 本は E31 と同じ）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/source_trust.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/source_trust.py --analyze

出力：`results/source_trust_<kind>/<ターゲット>.csv`（policy, r, rep, K, uar），`<ターゲット>_pseudo.csv`（pseudo, r, rep, K, uar），
`<ターゲット>_trust.csv`（ソース群 × 感情ごとの信頼度の要約），`provenance.json`，`provenance_log.jsonl`。
判定は `results/explore_relation/e55_targets.csv`・`e55_tests.csv`・`e55_trust.csv`
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
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import ht_block_present  # noqa: E402

RS = (0, 10, 25, 50, 75)
META_KS = ("1", "5")
PSEUDO_REPS = [0, 1, 2, 3, 4]
C_PSEUDO = 0.01
KINDS = ["audio_wavlm", "video", "text_roberta"]
OUT_REL = ROOT / "results" / "explore_relation"


def trust_of(s, members, P, Y):
    """ソース群 s の各発話の信頼度（members のうち s と別のコーパスの群の分類器が，発話のラベルに出す確率の平均）。使える群が無ければ None。"""
    others = [o for o in members if o != s and CM.corp(o) != CM.corp(s)]
    if not others:
        return None
    y = np.asarray(Y[s])
    return np.mean([P[(o, s)][np.arange(len(y)), y] for o in others], 0)


def keep_mask(s, r, tr, Y, rand_key=None):
    """群 s の中で，感情ごとに信頼度の下位 r％ を外した残りの印。rand_key を渡すと同じ数をランダムに外す。"""
    y = np.asarray(Y[s])
    keep = np.ones(len(y), bool)
    if r == 0 or tr is None:
        return keep
    for k in range(4):
        idx = np.flatnonzero(y == k)
        n_drop = int(np.floor(len(idx) * r / 100))
        if n_drop == 0:
            continue
        if rand_key is None:
            drop = idx[np.argsort(tr[idx], kind="stable")[:n_drop]]
        else:
            drop = np.random.default_rng(zlib.crc32(f"e55rand|{rand_key}|{s}|{k}|{r}".encode())).choice(idx, n_drop, replace=False)
        keep[drop] = False
    return keep


def fit_draft(data, members, r, C, P, Y, cache, rand_key=None):
    """members の群の合併から，信頼度（members の中で測る）の下位 r％ を外して P4 の出発点を学ぶ。"""
    key = (tuple(sorted(members)), r, C, rand_key)
    if key in cache:
        return cache[key]
    Xs, ys = [], []
    for s in members:
        m = keep_mask(s, r, trust_of(s, members, P, Y), Y, rand_key)
        Xs.append(data[s].X[torch.as_tensor(np.flatnonzero(m), device=data[s].X.device)])
        ys.append(np.asarray(Y[s])[m])
    Xp = torch.cat(Xs)
    src = logreg.fit(Xp - Xp.mean(0), M.labels(np.concatenate(ys)), (C,))
    del Xp, Xs
    cache[key] = src
    return src


def evaluate(ev, src, C, reps, ks):
    """E41 の random_true と同じ（E31 の K 本．P4 の既定の仮説転移 4）。(rep, K) → UAR。"""
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    out = {}

    def add(r, K, c, u, cv, L, cen=0, cb=""):
        out[(r, K)] = u
    ht_block_present(ctx, reps, src, C, add)
    return out


def run_target(kind, T, data, P, Y, reps, ks, out, cache):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    prows = []
    for c in S:                                                                  # 仮のターゲットで r を選ぶ
        Sc = [s for s in S if CM.corp(s) != CM.corp(c)]
        for r in RS:
            res = evaluate(data[c], fit_draft(data, Sc, r, C_PSEUDO, P, Y, cache), C_PSEUDO, PSEUDO_REPS, list(META_KS))
            prows += [{"target": T, "pseudo": c, "r": r, "rep": rp, "K": K, "uar": u} for (rp, K), u in res.items()]
    pdf = pd.DataFrame(prows)
    r_star = int(pdf.groupby("r").uar.mean().idxmax())
    rows = []
    for r in RS:
        for (rp, K), u in evaluate(data[T], fit_draft(data, S, r, C_T, P, Y, cache), C_T, reps, ks).items():
            rows.append({"target": T, "policy": "trust", "r": r, "rep": rp, "K": K, "uar": u, "r_star": r_star})
        if r > 0:
            for (rp, K), u in evaluate(data[T], fit_draft(data, S, r, C_T, P, Y, cache, rand_key=T), C_T, reps, ks).items():
                rows.append({"target": T, "policy": "random", "r": r, "rep": rp, "K": K, "uar": u, "r_star": r_star})
    trows = []
    for s in S:
        tr = trust_of(s, S, P, Y)
        y = np.asarray(Y[s])
        for k in range(4):
            m = y == k
            trows.append({"target": T, "source": s, "emotion": k, "n": int(m.sum()),
                          "trust_mean": float(tr[m].mean()) if tr is not None else np.nan,
                          "trust_q25": float(np.quantile(tr[m], 0.25)) if tr is not None else np.nan,
                          "acc_others": float((np.mean([P[(o, s)] for o in S if o != s and CM.corp(o) != CM.corp(s)], 0).argmax(1)[m] == k).mean())
                          if tr is not None else np.nan})
    pdf.to_csv(out / f"{T}_pseudo.csv", index=False)
    pd.DataFrame(trows).to_csv(out / f"{T}_trust.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=KINDS) -> int:
    tabs, trs = [], []
    for kind in kinds:
        d = ROOT / "results" / f"source_trust_{kind}"
        for x in sorted(f for f in d.glob("*.csv") if not f.stem.endswith(("_pseudo", "_trust"))):
            T = x.stem
            a = pd.read_csv(x, dtype={"K": str})
            a = a[a.rep.isin(EC.REPS)]
            r_star = int(a.r_star.iloc[0])
            u = a.groupby(["policy", "r", "K"]).uar.mean() * 100
            ps = pd.read_csv(d / f"{T}_pseudo.csv", dtype={"K": str})
            pk = ps.groupby(["K", "r"]).uar.mean()
            for K in sorted(a.K.unique(), key=int):
                row = {"kind": kind, "target": T, "K": K, "r_star": r_star, "all": float(u.loc[("trust", 0, K)]),
                       "meta": float(u.loc[("trust", r_star, K)]),
                       "rand_at_star": float(u.loc[("random", r_star, K)]) if r_star > 0 else float(u.loc[("trust", 0, K)]),
                       "oracle": float(max(u.loc[("trust", r, K)] for r in RS))}
                for r in RS:
                    row[f"trust_r{r}"] = float(u.loc[("trust", r, K)])
                    if r > 0:
                        row[f"rand_r{r}"] = float(u.loc[("random", r, K)])
                if K in META_KS:
                    row["r_star_K"] = int(pk.loc[K].idxmax())
                tabs.append(row)
            t = pd.read_csv(d / f"{T}_trust.csv")
            trs.append(t.assign(kind=kind))
    D = pd.DataFrame(tabs)
    Tr = pd.concat(trs).drop_duplicates(["kind", "source", "emotion", "target"])
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for x in ["all", "meta", "rand_at_star", "oracle"] + [f"trust_r{r}" for r in RS] + [f"rand_r{r}" for r in RS if r > 0]:
            row[x] = float(g[x].mean())
        for a_, b_ in [("meta", "all"), ("meta", "rand_at_star"), ("oracle", "all")] + [(f"trust_r{r}", f"rand_r{r}") for r in RS if r > 0] + \
                      [(f"trust_r{r}", "all") for r in RS if r > 0]:
            diff = g[a_] - g[b_]
            row[f"{a_}-{b_}"] = float(diff.mean())
            row[f"{a_}>{b_}"] = int((diff > 0).sum())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in kinds:
        k1, k5 = R[(R.kind == kind) & (R.K == "1")], R[(R.kind == kind) & (R.K == "5")]
        if len(k1) and len(k5):
            k1, k5 = k1.iloc[0], k5.iloc[0]
            R.loc[R.kind == kind, "SV1"] = bool(k1["meta-all"] > 0 and k5["meta-all"] > 0)
            R.loc[R.kind == kind, "SV2"] = bool(k1["meta-rand_at_star"] > 0 and k5["meta-rand_at_star"] > 0)
            R.loc[R.kind == kind, "SV3"] = bool(k1["meta-all"] > k5["meta-all"])
    D.to_csv(OUT_REL / "e55_targets.csv", index=False)
    R.to_csv(OUT_REL / "e55_tests.csv", index=False)
    Tr.to_csv(OUT_REL / "e55_trust.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in ("SV1", "SV2", "SV3") if c in R]).set_index(["kind", "K"]).round(2).T.to_string())
    print("\n== 選ばれた r（ターゲットの数）")
    print(D.drop_duplicates(["kind", "target"]).groupby("kind").r_star.value_counts().unstack(fill_value=0).to_string())
    print("\n== 信頼度の平均（ソース群 × 感情．ターゲットをまたいで平均）")
    print(Tr.groupby(["kind", "source", "emotion"]).trust_mean.mean().unstack("emotion").round(3).to_string())
    vc = [c for c in ("SV1", "SV2", "SV3") if c in R]
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
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds)
    out = args.out_dir or ROOT / "results" / f"source_trust_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    f, Cs, P, Y = CM.prepare(args.kind, data)
    prov = {"script": "exp/source_trust.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "rs": RS, "meta_ks": META_KS, "pseudo_reps": PSEUDO_REPS,
            "c_pseudo": C_PSEUDO, "src_C": Cs, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, P, Y, args.reps, args.ks, out, cache)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
