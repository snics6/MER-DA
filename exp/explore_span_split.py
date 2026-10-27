"""【探索的】E68 の下調べ（GPU）：疑似ラベルによるアンカーの動きを，K 本の張る空間の中と外に分けて学び直す．仮のターゲットだけ。

問い（2026-10-28．研究者の承認．`docs/exploratory-analyses.md` E68．設計と予言は計算の前に書いた）
  E66 の B（疑似ラベルで動かしたアンカー W_Q から K 本だけで学び直す）の得は，K 本の特徴 z が張る空間の外の成分から来るか（E67 の命題 4）．
  - 反復と K ごとに Δ = W_Q − W0 を K 本の z の張る空間への射影 Δ_in と残り Δ_out に分ける．切片は b_Q のまま
  - アンカー full（W_Q），in（W0 + Δ_in），out（W0 + Δ_out）と，対照 start（W0, b0）から，K 本だけで学び直す（h = h_b = 1）
  生徒・教師モデル・分割・仮のターゲット・解き方は E66（`exp/explore_move_then_learn.py`）と同じ．評価に使うターゲットのテストは使わない

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/explore_span_split.py --world fuse-audio-video [--target …] [--pseudo-reps …] [--out-dir …]
    ./.venv/bin/python exp/explore_span_split.py --analyze [--results-dir …]
出力：`results/span_split_<世界>/<T>.csv`（pseudo, rep, student, variant, meff, K, uar, uar_anchor, r_in），`provenance.json`，`provenance_log.jsonl`．
      判定は `results/explore_relation/e68_check.csv`・`e68_gain.csv`・`e68_tests.csv`
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
import explore_move_then_learn as MTL  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
import multimodal_distill as MD  # noqa: E402
from data import groups_for  # noqa: E402
from distill_belief import solve_all, uar_of  # noqa: E402
from geom_e0 import settings  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import present_weights  # noqa: E402

WORLDS = MTL.WORLDS
BELIEF, MEFF = "pF", (500, 2000)
KS_L = MTL.KS_L
PSEUDO_REPS = list(MTL.PSEUDO_REPS)
OUT_REL = ROOT / "results" / "explore_relation"
KEY = ["world", "target", "pseudo", "rep", "student"]


# ---------------------------------------------------------------- 計算（GPU）

def refit_one(anc, Z, yq, idx, cp, cb):
    """中心 anc（Fitted．C_FIT の解）から，K 本（idx）だけで学び直す（罰則 C′，C_b．E66 の h = h_b = 1）。"""
    dev = Z.device
    X = Z[torch.as_tensor(idx, device=dev)]
    yt = M.labels(yq[idx])
    zero = torch.zeros(Z.shape[1], dtype=Z.dtype, device=dev)
    f = logreg.fit_batch(X, yt, [np.ones(len(idx), bool)], (cp,), anchor=anc, anchor_c=MD.C_FIT,
                         stats=(zero, torch.ones_like(zero)), Cb=[cb], weights=[present_weights(yt)])[0]
    return f.W[cp], f.b[cp]


def fitted(W, b):
    zero = torch.zeros(W.shape[1], dtype=W.dtype, device=W.device)
    return logreg.Fitted(zero, torch.ones_like(zero), {MD.C_FIT: W}, {MD.C_FIT: b}, {MD.C_FIT: 0}, {MD.C_FIT: 0.0})


def split(delta, Zk):
    """Δ の各行を，Zk の行が張る空間への射影と残りに分ける。"""
    U, S, _ = torch.linalg.svd(Zk.T, full_matrices=False)
    U = U[:, S > S.max() * 1e-8]
    d_in = delta @ U @ U.T
    return d_in, delta - d_in


def solve_split(world, stu, yq, yte, label_sets, rows, tag):
    b1 = MD.beliefs1(stu, world)
    for m in MD.WORLDS[world]:
        s = stu[m]
        st, rho_m, cp, cb, Z, Zte = s["st"], s["rho_m"], s["cp"], s["cb"], s["Z"], s["Zte"]
        a0 = st.anchor(rho_m)
        W0, b0 = a0.W[MD.C_FIT], a0.b[MD.C_FIT]
        base = {**tag, "student": m}
        for K in KS_L:
            W, b = refit_one(a0, Z, yq, label_sets[K], cp, cb)
            rows.append({**base, "variant": "start", "meff": 0, "K": K, "uar": uar_of(W, b, Zte, yte),
                         "uar_anchor": uar_of(W0, b0, Zte, yte), "r_in": np.nan})
        sol = solve_all(Z, yq, {BELIEF: b1[BELIEF]}, {"0": label_sets["0"]}, [(BELIEF, e) for e in MEFF], st, rho_m,
                        MD.C_FIT, cp, cb)
        for (_, e, _), (Wq, bq) in sol.items():
            delta = Wq - W0
            for K in KS_L:
                Zk = Z[torch.as_tensor(label_sets[K], device=Z.device)]
                d_in, d_out = split(delta, Zk)
                r_in = float((d_in ** 2).sum() / (delta ** 2).sum())
                for name, Wa in (("full", Wq), ("in", W0 + d_in), ("out", W0 + d_out)):
                    W, b = refit_one(fitted(Wa, bq), Z, yq, label_sets[K], cp, cb)
                    rows.append({**base, "variant": name, "meff": e, "K": K, "uar": uar_of(W, b, Zte, yte),
                                 "uar_anchor": uar_of(Wa, bq, Zte, yte), "r_in": r_in})


def run_target(world, T, data_f, data_m, out, cache):
    """E66 の run_target と同じ生徒・分割で解く。"""
    S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
    sets = {k: settings(k, T) for k in MD.WORLDS[world] + [world]}
    rows = []
    for c in S:
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c:
            continue
        info_c = {k: M5.pool_info(data_m[k] if k != world else data_f, pool_c, MD.C_FIT, cache[k]) for k in sets}
        evc = data_f[c]
        ctxc = CB.Ctx(evc, PSEUDO_REPS, list(KS_L), center="ind")
        for r in PSEUDO_REPS:
            pool = np.flatnonzero(~ctxc.te[r])
            te = np.flatnonzero(ctxc.te[r])
            pos_of = {int(i): j for j, i in enumerate(pool)}
            label_sets = {"0": np.array([], int), **{K: np.array([pos_of[int(i)] for i in ctxc.idx[(r, K)]]) for K in KS_L}}
            stu = MD.make_students(world, c, data_f, data_m, pool_c, pool, te, sets, info_c)
            solve_split(world, stu, evc.y[pool], evc.y[te], label_sets, rows, {"pseudo": c, "rep": r})
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def _meta(results):
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"script": "exp/explore_span_split.py", "commit": git("rev-parse", "--short", "HEAD"), "results": str(results)}


def write(df, name, meta):
    with open(OUT_REL / name, "w") as f:
        f.write("# " + json.dumps(meta, ensure_ascii=False) + "\n")
        df.to_csv(f, index=False)


def load(results, prefix):
    out = []
    for w in WORLDS:
        for p in sorted((results / f"{prefix}_{w}").glob("*.csv")):
            out.append(pd.read_csv(p, dtype={"K": str}).assign(world=w, target=p.stem))
    return pd.concat(out, ignore_index=True)


def analyze(results=None) -> int:
    results = results or ROOT / "results"
    meta = _meta(results)
    X = load(results, "span_split")
    # 確かめ：full と start が E66 の同じ問題と一致するか
    Y = load(results, "move_then_learn")
    Y = Y[(Y.hold == 1) & (Y.hold_b == 1) & (Y.K != "0")]
    ref_full = Y[(Y.center == "moved") & (Y.belief == BELIEF) & Y.meff.isin(MEFF)].set_index(KEY + ["meff", "K"]).uar
    ref_start = Y[Y.center == "start"].set_index(KEY + ["K"]).uar
    xf = X[X.variant == "full"].set_index(KEY + ["meff", "K"]).uar
    xs = X[X.variant == "start"].set_index(KEY + ["K"]).uar
    chk = []
    for name, x, y in (("full", xf, ref_full), ("start", xs, ref_start)):
        j = x.to_frame("x").join(y.rename("y"), how="inner")
        chk.append(dict(what=name, n=len(j), n_x=len(x), max_abs_diff=float((j.x - j.y).abs().max()) if len(j) else np.nan))
    chk = pd.DataFrame(chk)
    write(chk, "e68_check.csv", meta)
    print("== 確かめ（E66 との差）\n" + chk.to_string(index=False))
    base = X[X.variant == "start"].set_index(KEY + ["K"]).uar
    V = X[X.variant != "start"].copy()
    V["gain"] = V.uar.values - base.reindex(pd.MultiIndex.from_frame(V[KEY + ["K"]])).values
    V["gain_anchor"] = V.uar_anchor.values - X[X.variant == "start"].set_index(KEY + ["K"]).uar_anchor.reindex(
        pd.MultiIndex.from_frame(V[KEY + ["K"]])).values
    G = V.groupby(["student", "meff", "variant", "K"]).agg(gain=("gain", "mean"), gain_anchor=("gain_anchor", "mean"),
                                                           r_in=("r_in", "mean"), n=("gain", "size")).reset_index()
    write(G, "e68_gain.csv", meta)
    tests = []
    for s, g in G[G.meff == 2000].groupby("student"):
        g = g.set_index(["variant", "K"]).gain
        q1 = {K: g[("out", K)] / g[("full", K)] for K in ("5", "10", "25")}
        tests.append(dict(rule="Q1", student=s, detail=" ".join(f"K={K}:{v:.2f}" for K, v in q1.items()),
                          ok=bool(all(v >= 0.7 for v in q1.values()))))
        q2 = dict(in5=g[("in", "5")], in25=g[("in", "25")], full25=g[("full", "25")])
        tests.append(dict(rule="Q2", student=s, detail=f"in K=5 {q2['in5']:+.2f}, K=25 {q2['in25']:+.2f}, full K=25 {q2['full25']:+.2f}",
                          ok=bool(q2["in25"] < q2["in5"] and q2["in25"] <= 0.3 * q2["full25"])))
    tests = pd.DataFrame(tests)
    write(tests, "e68_tests.csv", meta)
    print("\n== 判定（pF・m=2000）\n" + tests.to_string(index=False))
    print("\n== 得（start との差，UAR の割合．0.01 = 1 ポイント．仮のターゲットの平均）と r_in")
    print(G.pivot_table(index=["student", "meff", "variant"], columns="K", values="gain").round(2).to_string())
    print(G[G.variant == "full"].pivot_table(index=["student", "meff"], columns="K", values="r_in").round(2).to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=WORLDS)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--pseudo-reps", nargs="+", type=int, help="試走用．既定は 0〜4")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--results-dir", type=Path, help="--analyze が span_split_* と move_then_learn_* を読む場所（既定は results/）")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.results_dir)
    global PSEUDO_REPS
    if args.pseudo_reps is not None:
        PSEUDO_REPS = args.pseudo_reps
    world = args.world
    out = args.out_dir or ROOT / "results" / f"span_split_{world}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(world)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{world}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data_f = {g: M.load(g, world) for g in groups}
    data_m = {m: {g: MD.sub_group(data_f[g], a, b) for g in groups} for m, a, b in MD.blocks(world)}
    prov = {"script": "exp/explore_span_split.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "exp", "measure")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "targets": todo, "map_dir": str(map_dir(world)), "world": world, "mods": MD.WORLDS[world], "pseudo_reps": PSEUDO_REPS,
            "c_fit": MD.C_FIT, "belief": BELIEF, "meff": MEFF,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache = {k: {} for k in MD.WORLDS[world] + [world]}
    for t in todo:
        t0 = time.perf_counter()
        run_target(world, t, data_f, data_m, out, cache)
        print(f"{world} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
