"""【探索的】E71 の下調べ（GPU）：アンカーと教師モデルごとの解を，方向の区間ごとの精度で平均してから K 本で学び直す．仮のターゲットだけ。

問い（2026-10-28．研究者の承認．`docs/exploratory-analyses.md` E71．設計と予言は計算の前に書いた）
  E70 の学習法の候補の前提を確かめる．見方（生徒のアンカー a0，融合の教師モデル pF の疑似ラベルで解いた θ_pF，
  別のモダリティの単独の教師モデルの疑似ラベルで解いた θ_t）を，候補の z の共分散の固有方向の区間ごとに，
  誤差の小さいものほど重く平均する．誤差は (a) W*（仮のターゲットの候補の全部のラベルの解）との差（oracle），
  (b) 見方どうしの食い違いから三角測量で推したもの（est．ラベルを使わない），(c) 同じ重み（mean）．
  中心そのもの（K=0）と，中心から K 本だけで学び直した解（K ≥ 1）の UAR を比べる
  生徒・教師モデル・分割・仮のターゲット・解き方は E68（`exp/explore_span_split.py`）と同じ．評価に使うターゲットのテストは使わない

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/explore_view_fusion.py --world fuse-audio-video [--target …] [--pseudo-reps …] [--out-dir …]
    ./.venv/bin/python exp/explore_view_fusion.py --analyze [--results-dir …]
出力：`results/view_fusion_<世界>/<T>.csv`（pseudo, rep, student, variant, K, uar），`<T>_bins.csv`（区間 × 見方の E・Ê・重み・W* の揺れ），
      `<T>_pairs.csv`（区間 × 見方の組の食い違い D と本当の誤差の共分散 C），`provenance.json`，`provenance_log.jsonl`．
      判定は `results/explore_relation/e71_check.csv`・`e71_uar.csv`・`e71_tests.csv`・`e71_bins.csv`・`e71_corr.csv`

誤差の大きさ：W の感情の平均を除き，区間 g で Σ_{k∈g} s_k ‖(W_x − W_y) v_k‖²（v_k, s_k は候補の z の共分散の固有方向と分散）．
切片は ‖b_x − b_y‖²（感情の平均を除く）を区間 0 とする．D_st = E_s + E_t を全部の組で最小 2 乗に解いて Ê_t を出す
"""

from __future__ import annotations

import argparse
import itertools
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
import explore_span_split as SS  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
import multimodal_distill as MD  # noqa: E402
from data import groups_for  # noqa: E402
from distill_belief import solve_all, uar_of  # noqa: E402
from geom_e0 import CB_STAR, fit_star, settings  # noqa: E402
from pair_table import map_dir  # noqa: E402

WORLDS = MTL.WORLDS
BELIEF, MEFF, MEFF_E68 = SS.BELIEF, 2000, SS.MEFF
KS_L = MTL.KS_L
KS_ALL = ("0",) + tuple(KS_L)
PSEUDO_REPS = list(MTL.PSEUDO_REPS)
OUT_REL = ROOT / "results" / "explore_relation"
KEY = SS.KEY
RANK_EDGES = (4, 16, 64, 256)                 # 区間 1：順位 1〜4，2：5〜16，3：17〜64，4：65〜256，5：257 以降．区間 0 は切片
NB = len(RANK_EDGES) + 2
FLOOR = 0.1                                   # Ê の下限は「区間の D の平均の半分」の FLOOR 倍
TESTS_K = ("0", "5", "25")


# ---------------------------------------------------------------- 方向と誤差の大きさ

class Basis:
    """候補の z の共分散の固有方向（分散の大きい順）と，区間の番号。"""

    def __init__(self, Z):
        Zc = Z - Z.mean(0, keepdim=True)
        s, V = torch.linalg.eigh(Zc.T @ Zc / Z.shape[0])
        order = torch.argsort(s, descending=True)
        self.s, self.V = s[order].clamp_min(0), V[:, order]
        rank = torch.arange(Z.shape[1], device=Z.device)
        g = torch.ones_like(rank)
        for e in RANK_EDGES:
            g = g + (rank >= e).long()
        self.g = g

    def coords(self, W, b):
        """感情の平均を除いた (W, b) を，固有方向の座標 (k × d) と切片 (k) で返す。"""
        Wc, bc = W - W.mean(0, keepdim=True), b - b.mean()
        return Wc @ self.V, bc

    def inner(self, x, y):
        """区間ごとの Σ s_k ⟨x_k, y_k⟩（区間 0 は切片の内積）。長さ NB。"""
        (Px, bx), (Py, by) = x, y
        per = self.s * (Px * Py).sum(0)
        out = torch.zeros(NB, dtype=per.dtype, device=per.device)
        out.index_add_(0, self.g, per)
        out[0] = (bx * by).sum()
        return out

    def fuse(self, views, w):
        """views（(W, b) の並び）を，区間ごとの重み w（見方 × NB．区間ごとに和 1）で平均する。"""
        P = sum(wt[self.g][None, :] * (W @ self.V) for (W, _), wt in zip(views, w))
        b = sum(wt[0] * bv for (_, bv), wt in zip(views, w))
        return P @ self.V.T, b


def diff(x, y):
    return x[0] - y[0], x[1] - y[1]


def hat(D, names):
    """D[(s, t)]（区間ごとの食い違い）から，D_st = E_s + E_t を最小 2 乗で解いた Ê（見方 × NB）。"""
    pairs = list(D)
    A = torch.zeros(len(pairs), len(names), dtype=logreg.DTYPE)
    for i, (s, t) in enumerate(pairs):
        A[i, names.index(s)] = A[i, names.index(t)] = 1.0
    Dm = torch.stack([D[p].cpu() for p in pairs])             # 組 × NB
    E = torch.linalg.lstsq(A, Dm).solution                      # 見方 × NB
    floor = FLOOR * Dm.mean(0) / 2
    return torch.maximum(E, floor[None, :])


def norm_w(prec):
    return prec / prec.sum(0, keepdim=True)


def star(Z, yq, speaker, seed):
    """W*：候補の全部のラベルで 0 への正則化（C′ は話者の半分で選ぶ）と，話者の半分ずつの解。"""
    Ws, bs, c_star, _ = fit_star(Z, yq, speaker, None, None, seed)
    yt = M.labels(yq)
    zero = torch.zeros(Z.shape[1], dtype=Z.dtype, device=Z.device)
    fold = M.speaker_folds(pd.Series(np.asarray(speaker)), k=2, seed=seed).to_numpy()
    h = logreg.fit_batch(Z, yt, [fold == 0, fold != 0], (c_star,), stats=(zero, torch.ones_like(zero)), Cb=[CB_STAR, CB_STAR])
    return (Ws, bs), [(f.W[c_star], f.b[c_star]) for f in h]


# ---------------------------------------------------------------- 計算（GPU）

def solve_fusion(world, stu, yq, yte, label_sets, speaker, seed, rows, brows, prows, tag):
    b1 = MD.beliefs1(stu, world)
    for m in MD.WORLDS[world]:
        s = stu[m]
        st, rho_m, cp, cb, Z, Zte = s["st"], s["rho_m"], s["cp"], s["cb"], s["Z"], s["Zte"]
        a0 = st.anchor(rho_m)
        W0, b0 = a0.W[MD.C_FIT], a0.b[MD.C_FIT]
        base = {**tag, "student": m}
        rows.append({**base, "variant": "start", "K": "0", "uar": uar_of(W0, b0, Zte, yte)})
        for K in KS_L:
            W, b = SS.refit_one(a0, Z, yq, label_sets[K], cp, cb)
            rows.append({**base, "variant": "start", "K": K, "uar": uar_of(W, b, Zte, yte)})
        sol = solve_all(Z, yq, {BELIEF: b1[BELIEF]}, {"0": label_sets["0"]}, [(BELIEF, e) for e in MEFF_E68], st, rho_m,
                        MD.C_FIT, cp, cb)
        others = [k for k in MD.WORLDS[world] if k != m]
        sol_o = solve_all(Z, yq, {k: stu[k]["P"] for k in others}, {"0": label_sets["0"]}, [(k, MEFF) for k in others], st,
                          rho_m, MD.C_FIT, cp, cb)
        views = {"a0": (W0, b0), "pF": sol[(BELIEF, MEFF, "0")], **{k: sol_o[(k, MEFF, "0")] for k in others}}
        names = list(views)
        bs = Basis(Z)
        (Wst, bst), halves = star(Z, yq, speaker, seed)
        cst = bs.coords(Wst, bst)
        err = {v: bs.coords(*diff(views[v], (Wst, bst))) for v in names}
        cw = {v: bs.coords(*views[v]) for v in names}
        E = torch.stack([bs.inner(err[v], err[v]) for v in names]).cpu()
        D = {}
        for a, c in itertools.combinations(names, 2):
            dd = (cw[a][0] - cw[c][0], cw[a][1] - cw[c][1])
            D[(a, c)] = bs.inner(dd, dd)
            prows.append({**base, "s": a, "t": c, **{f"D_{g}": float(D[(a, c)][g]) for g in range(NB)},
                          **{f"C_{g}": float(x) for g, x in enumerate(bs.inner(err[a], err[c]))}})
        Eh = hat(D, names)
        hd = bs.coords(*diff(*halves))
        noise = (bs.inner(hd, hd) / 4).cpu()
        w = {"mean": torch.full_like(E, 1.0 / len(names)), "oracle": norm_w(1.0 / E.clamp_min(1e-12)), "est": norm_w(1.0 / Eh)}
        for i, v in enumerate(names):
            brows.append({**base, "view": v, **{f"E_{g}": float(E[i, g]) for g in range(NB)},
                          **{f"Eh_{g}": float(Eh[i, g]) for g in range(NB)},
                          **{f"wo_{g}": float(w["oracle"][i, g]) for g in range(NB)},
                          **{f"we_{g}": float(w["est"][i, g]) for g in range(NB)},
                          **{f"noise_{g}": float(noise[g]) for g in range(NB)},
                          **{f"star_{g}": float(bs.inner(cst, cst)[g]) for g in range(NB)}})
        centers = {"pF": views["pF"], **{f"t_{k}": views[k] for k in others}}
        vlist = [views[v] for v in names]
        for name, wt in w.items():
            centers[name] = bs.fuse(vlist, wt.to(Z.device))
        for name, (Wa, ba) in centers.items():
            rows.append({**base, "variant": name, "K": "0", "uar": uar_of(Wa, ba, Zte, yte)})
            for K in KS_L:
                W, b = SS.refit_one(SS.fitted(Wa, ba), Z, yq, label_sets[K], cp, cb)
                rows.append({**base, "variant": name, "K": K, "uar": uar_of(W, b, Zte, yte)})


def run_target(world, T, data_f, data_m, out, cache):
    """E68 の run_target と同じ生徒・分割で解く。"""
    S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
    sets = {k: settings(k, T) for k in MD.WORLDS[world] + [world]}
    rows, brows, prows = [], [], []
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
            solve_fusion(world, stu, evc.y[pool], evc.y[te], label_sets, evc.speaker.iloc[pool], 1000 + r, rows, brows, prows,
                         {"pseudo": c, "rep": r})
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)
    pd.DataFrame(brows).to_csv(out / f"{T}_bins.csv", index=False)
    pd.DataFrame(prows).to_csv(out / f"{T}_pairs.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def _meta(results):
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"script": "exp/explore_view_fusion.py", "commit": git("rev-parse", "--short", "HEAD"), "results": str(results)}


def write(df, name, meta):
    with open(OUT_REL / name, "w") as f:
        f.write("# " + json.dumps(meta, ensure_ascii=False) + "\n")
        df.to_csv(f, index=False)


def load(results, suffix=""):
    out = []
    for w in WORLDS:
        for p in sorted((results / f"view_fusion_{w}").glob(f"*{suffix}.csv")):
            stem = p.stem[: len(p.stem) - len(suffix)] if suffix else p.stem
            if not suffix and ("_bins" in p.stem or "_pairs" in p.stem):
                continue
            out.append(pd.read_csv(p, dtype={"K": str}).assign(world=w, target=stem))
    return pd.concat(out, ignore_index=True)


def analyze(results=None) -> int:
    results = results or ROOT / "results"
    meta = _meta(results)
    X = load(results)
    Y = SS.load(results, "span_split")
    chk = []
    for name, yv in (("pF", "full"), ("start", "start")):
        x = X[(X.variant == name) & (X.K != "0")].set_index(KEY + ["K"]).uar
        y = Y[(Y.variant == yv) & ((Y.meff == MEFF) | (yv == "start"))].set_index(KEY + ["K"]).uar
        j = x.to_frame("x").join(y.rename("y"), how="inner")
        chk.append(dict(what=name, n=len(j), n_x=len(x), max_abs_diff=float((j.x - j.y).abs().max()) if len(j) else np.nan))
    chk = pd.DataFrame(chk)
    write(chk, "e71_check.csv", meta)
    print("== 確かめ（E68 との差）\n" + chk.to_string(index=False))
    U = X.groupby(["student", "variant", "K"]).uar.mean().unstack("K")[list(KS_ALL)] * 100
    write(U.reset_index(), "e71_uar.csv", meta)
    tests = []
    for s in sorted(X.student.unique()):
        u = U.loc[s]
        room = {K: u.loc["oracle", K] - u.loc["pF", K] for K in TESTS_K}
        estg = {K: u.loc["est", K] - u.loc["pF", K] for K in TESTS_K}
        vs_mean = {K: u.loc["est", K] - u.loc["mean", K] for K in TESTS_K}
        s1 = all(v >= 0.5 for v in room.values())
        tests.append(dict(rule="S1", student=s, detail=" ".join(f"K={K}:{v:+.2f}" for K, v in room.items()), ok=bool(s1)))
        tests.append(dict(rule="S2", student=s, detail=" ".join(f"K={K}:{estg[K]:+.2f}/{room[K]:+.2f}" for K in TESTS_K),
                          ok=bool(s1 and all(estg[K] >= 0.5 * room[K] for K in TESTS_K)) if s1 else None))
        tests.append(dict(rule="S3", student=s, detail=" ".join(f"K={K}:{v:+.2f}" for K, v in vs_mean.items()),
                          ok=bool(all(v > 0 for v in vs_mean.values()))))
    tests = pd.DataFrame(tests)
    write(tests, "e71_tests.csv", meta)
    print("\n== 判定（UAR のポイント．仮のターゲットの平均）\n" + tests.to_string(index=False))
    print("\n== UAR（ポイント）\n" + U.round(2).to_string())
    B = load(results, "_bins")
    rows = []
    for g in range(NB):
        t = B.assign(bin=g, E=B[f"E_{g}"], ratio_hat=B[f"Eh_{g}"] / B[f"E_{g}"], noise_rel=B[f"noise_{g}"] / B[f"E_{g}"],
                     w_oracle=B[f"wo_{g}"], w_est=B[f"we_{g}"])
        rows.append(t.groupby(["student", "view", "bin"])[["E", "ratio_hat", "noise_rel", "w_oracle", "w_est"]].median().reset_index())
    Bt = pd.concat(rows, ignore_index=True)
    write(Bt, "e71_bins.csv", meta)
    print("\n== 区間ごと（中央値）：本当の誤差 E，Ê/E，W* の揺れ/E，重み\n" + Bt.round(3).to_string(index=False))
    P = load(results, "_pairs")
    Ek = B.set_index(KEY + ["view"])
    rows = []
    for g in range(NB):
        es = Ek[f"E_{g}"].reindex(pd.MultiIndex.from_frame(P[KEY + ["s"]])).values
        et = Ek[f"E_{g}"].reindex(pd.MultiIndex.from_frame(P[KEY + ["t"]])).values
        rows.append(P.assign(bin=g, rho=P[f"C_{g}"] / np.sqrt(es * et)).groupby(["student", "s", "t", "bin"])["rho"].median().reset_index())
    Ct = pd.concat(rows, ignore_index=True)
    write(Ct, "e71_corr.csv", meta)
    print("\n== 見方どうしの誤差の相関（中央値）\n" + Ct.pivot_table(index=["student", "s", "t"], columns="bin", values="rho").round(2).to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=WORLDS)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--pseudo-reps", nargs="+", type=int, help="試走用．既定は 0〜4")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--results-dir", type=Path, help="--analyze が view_fusion_* と span_split_* を読む場所（既定は results/）")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.results_dir)
    global PSEUDO_REPS
    if args.pseudo_reps is not None:
        PSEUDO_REPS = args.pseudo_reps
    world = args.world
    out = args.out_dir or ROOT / "results" / f"view_fusion_{world}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(world)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{world}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data_f = {g: M.load(g, world) for g in groups}
    data_m = {m: {g: MD.sub_group(data_f[g], a, b) for g in groups} for m, a, b in MD.blocks(world)}
    prov = {"script": "exp/explore_view_fusion.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "exp", "measure")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "targets": todo, "map_dir": str(map_dir(world)), "world": world, "mods": MD.WORLDS[world], "pseudo_reps": PSEUDO_REPS,
            "c_fit": MD.C_FIT, "belief": BELIEF, "meff": MEFF, "rank_edges": RANK_EDGES, "floor": FLOOR,
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
