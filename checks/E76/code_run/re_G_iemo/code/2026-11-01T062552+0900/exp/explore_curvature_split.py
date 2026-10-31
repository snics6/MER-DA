"""【探索的】E69 の下調べ（GPU）：疑似ラベルによるアンカーの動きを，K 本の損失の 2 階微分の固有方向で分けて学び直す．仮のターゲットだけ。

問い（2026-10-28．研究者の承認．`docs/exploratory-analyses.md` E69．設計と予言は計算の前に書いた）
  学び直しで残るアンカーの動きは (P + H)⁻¹ P Δa（P は正則化の精度，H は K 本の損失の 2 階微分）で，
  H v = λ P v の固有方向ごとに 1/(1 + λ) が残る（2 次の近似）．B の得は λ ≤ 1 の成分から来るか．
  - 反復と K ごとに，a0 = (W0, b0) から K 本で学び直した θ_s で H を求め，Δa = (W_Q − W0, b_Q − b0) を λ ≤ 1（low）と λ > 1（high）に分ける
  - アンカー full（a0 + Δa），low，high と，対照 start（a0）から，K 本だけで学び直す（h = h_b = 1）
  - 重みの水準：θ_full − θ_s と予言 (P + H)⁻¹ P Δa の相対誤差，λ の区間ごとの Δa の割合と残った割合（実測と予言）
  生徒・教師モデル・分割・仮のターゲット・解き方は E68（`exp/explore_span_split.py`）と同じ．評価に使うターゲットのテストは使わない

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/explore_curvature_split.py --world fuse-audio-video [--target …] [--pseudo-reps …] [--out-dir …]
    ./.venv/bin/python exp/explore_curvature_split.py --analyze [--results-dir …]
出力：`results/curv_split_<世界>/<T>.csv`（pseudo, rep, student, variant, meff, K, uar, uar_anchor，full の行に rel_err・sh_*・km_*・kp_*），
      `provenance.json`，`provenance_log.jsonl`．判定は `results/explore_relation/e69_check.csv`・`e69_gain.csv`・`e69_weights.csv`・`e69_tests.csv`

数理（K 本 n = 4K，感情 k = 4，生徒の座標 z．目的関数に S を掛けると Σ w_i nll_i + ‖W − W_a‖²/(2C′) + ‖b − b_a‖²/(2C_b)）
  出力 l_i = W z_i + b の 2 階微分は w_i (diag π_i − π_i π_iᵀ) = M_i²．G の i 番目の塊を M_i J_i P^{-1/2} とすると P^{-1/2} H P^{-1/2} = GᵀG で，
  G Gᵀ の (i, j) の塊は (C′ z_i·z_j + C_b) M_i M_j（nk 次）．その固有値 λ_j > 0 と固有ベクトル V_j から u_j = Gᵀ V_j / √λ_j．
  Δ の係数は c_j = V_jᵀ g(Δ) / √λ_j（g(Δ)_i = M_i (ΔW z_i + Δb)），Σ_j a_j u_j を元の座標へ戻すと ΔW = C′ qᵀ Z，Δb = C_b Σ_i q_i
  （q_i = M_i v_i，v = Σ_j (a_j/√λ_j) V_j）．M_i 1 = 0 なので，戻した成分は感情の平均が 0 になる
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
import explore_span_split as SS  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
import multimodal_distill as MD  # noqa: E402
from data import groups_for  # noqa: E402
from distill_belief import solve_all, uar_of  # noqa: E402
from geom_e0 import settings  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import present_weights  # noqa: E402

WORLDS = MTL.WORLDS
BELIEF, MEFF = SS.BELIEF, SS.MEFF
KS_L = MTL.KS_L
PSEUDO_REPS = list(MTL.PSEUDO_REPS)
OUT_REL = ROOT / "results" / "explore_relation"
KEY = SS.KEY
LAM_CUT = 1.0                                         # low は λ ≤ 1（学び直しで半分以上が残る方向）
EDGES = (0.1, 1.0, 10.0)                              # 区間 0：λ = 0，1：(0, 0.1]，2：(0.1, 1]，3：(1, 10]，4：10 より大
NB = len(EDGES) + 2


# ---------------------------------------------------------------- 2 階微分の固有方向

class Curv:
    """θ_s = (Ws, bs) での K 本（Zk, 重み w）の 2 階微分について，H v = λ P v の λ > 0 の固有方向（P = diag(1/C′, 1/C_b)）。"""

    def __init__(self, Ws, bs, Zk, w, cp, cb):
        n, k = Zk.shape[0], Ws.shape[0]
        pi = torch.softmax(Zk @ Ws.T + bs, 1)
        Sig = w[:, None, None] * (torch.diag_embed(pi) - pi[:, :, None] * pi[:, None, :])
        e, U = torch.linalg.eigh(Sig)
        self.Mi = U @ torch.diag_embed(e.clamp_min(0).sqrt()) @ U.transpose(1, 2)   # (n, k, k)．M_i² = w_i Σ_i
        Kmat = cp * Zk @ Zk.T + cb
        GG = torch.einsum("ij,iab,jbc->iajc", Kmat, self.Mi, self.Mi).reshape(n * k, n * k)
        lam, V = torch.linalg.eigh((GG + GG.T) / 2)
        keep = lam > lam.max() * 1e-10
        self.lam, self.V = lam[keep], V[:, keep]
        self.Zk, self.cp, self.cb, self.n, self.k = Zk, cp, cb, n, k

    def coef(self, dW, db):
        """Δ = (dW, db) の各固有方向の係数 c_j（P^{1/2} の座標）。"""
        g = torch.einsum("iab,ib->ia", self.Mi, self.Zk @ dW.T + db).reshape(-1)
        return (self.V.T @ g) / self.lam.sqrt()

    def back(self, a):
        """Σ_j a_j u_j を元の座標 (dW, db) へ戻す。"""
        v = (self.V @ (a / self.lam.sqrt())).view(self.n, self.k)
        q = torch.einsum("iab,ib->ia", self.Mi, v)
        return self.cp * q.T @ self.Zk, self.cb * q.sum(0)

    def bins(self):
        b = torch.full_like(self.lam, 1, dtype=torch.long)
        for e in EDGES:
            b = b + (self.lam > e).long()
        return b


def centered(dW, db):
    return dW - dW.mean(0, keepdim=True), db - db.mean()


def pdot(x, y, cp, cb):
    return float((x[0] * y[0]).sum() / cp + (x[1] * y[1]).sum() / cb)


def weight_stats(cv, da, dth):
    """重みの水準の記述：予言の相対誤差，λ の区間ごとの Δa の割合（sh），残った割合の実測（km）と予言（kp）。"""
    cp, cb = cv.cp, cv.cb
    da, dth = centered(*da), centered(*dth)
    ca, ct = cv.coef(*da), cv.coef(*dth)
    lam = cv.lam
    pred_rm = cv.back(ca * lam / (1 + lam))
    pred = (da[0] - pred_rm[0], da[1] - pred_rm[1])
    err = (dth[0] - pred[0], dth[1] - pred[1])
    out = {"rel_err": (pdot(err, err, cp, cb) / pdot(dth, dth, cp, cb)) ** 0.5}
    tot = pdot(da, da, cp, cb)
    ra_rng, rt_rng = cv.back(ca), cv.back(ct)
    ra = (da[0] - ra_rng[0], da[1] - ra_rng[1])
    rt = (dth[0] - rt_rng[0], dth[1] - rt_rng[1])
    s0 = pdot(ra, ra, cp, cb)
    out.update(sh_0=s0 / tot, km_0=pdot(rt, ra, cp, cb) / s0 if s0 > 1e-12 * tot else np.nan, kp_0=1.0)
    bn = cv.bins()
    for j in range(1, NB):
        m = bn == j
        sj = float((ca[m] ** 2).sum())
        out[f"sh_{j}"] = sj / tot
        out[f"km_{j}"] = float((ca[m] * ct[m]).sum()) / sj if sj > 1e-12 * tot else np.nan
        out[f"kp_{j}"] = float((ca[m] ** 2 / (1 + lam[m])).sum()) / sj if sj > 1e-12 * tot else np.nan
    out["share_low"] = 1 - float((ca[lam > LAM_CUT] ** 2).sum()) / tot
    return out


# ---------------------------------------------------------------- 計算（GPU）

def solve_curv(world, stu, yq, yte, label_sets, rows, tag):
    b1 = MD.beliefs1(stu, world)
    for m in MD.WORLDS[world]:
        s = stu[m]
        st, rho_m, cp, cb, Z, Zte = s["st"], s["rho_m"], s["cp"], s["cb"], s["Z"], s["Zte"]
        a0 = st.anchor(rho_m)
        W0, b0 = a0.W[MD.C_FIT], a0.b[MD.C_FIT]
        base = {**tag, "student": m}
        curv, th_s = {}, {}
        for K in KS_L:
            W, b = SS.refit_one(a0, Z, yq, label_sets[K], cp, cb)
            th_s[K] = (W, b)
            idx = label_sets[K]
            curv[K] = Curv(W, b, Z[torch.as_tensor(idx, device=Z.device)], present_weights(M.labels(yq[idx])).to(Z.device, Z.dtype), cp, cb)
            rows.append({**base, "variant": "start", "meff": 0, "K": K, "uar": uar_of(W, b, Zte, yte),
                         "uar_anchor": uar_of(W0, b0, Zte, yte)})
        sol = solve_all(Z, yq, {BELIEF: b1[BELIEF]}, {"0": label_sets["0"]}, [(BELIEF, e) for e in MEFF], st, rho_m,
                        MD.C_FIT, cp, cb)
        for (_, e, _), (Wq, bq) in sol.items():
            dW, db = Wq - W0, bq - b0
            for K in KS_L:
                cv = curv[K]
                ca = cv.coef(dW, db)
                hW, hb = cv.back(torch.where(cv.lam > LAM_CUT, ca, torch.zeros_like(ca)))
                for name, (Wa, ba) in (("full", (Wq, bq)), ("low", (Wq - hW, bq - hb)), ("high", (W0 + hW, b0 + hb))):
                    W, b = SS.refit_one(SS.fitted(Wa, ba), Z, yq, label_sets[K], cp, cb)
                    row = {**base, "variant": name, "meff": e, "K": K, "uar": uar_of(W, b, Zte, yte),
                           "uar_anchor": uar_of(Wa, ba, Zte, yte)}
                    if name == "full":
                        row.update(weight_stats(cv, (dW, db), (W - th_s[K][0], b - th_s[K][1])))
                    rows.append(row)


def run_target(world, T, data_f, data_m, out, cache):
    """E68 の run_target と同じ生徒・分割で解く。"""
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
            solve_curv(world, stu, evc.y[pool], evc.y[te], label_sets, rows, {"pseudo": c, "rep": r})
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def _meta(results):
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"script": "exp/explore_curvature_split.py", "commit": git("rev-parse", "--short", "HEAD"), "results": str(results)}


def write(df, name, meta):
    with open(OUT_REL / name, "w") as f:
        f.write("# " + json.dumps(meta, ensure_ascii=False) + "\n")
        df.to_csv(f, index=False)


def analyze(results=None) -> int:
    results = results or ROOT / "results"
    meta = _meta(results)
    X = SS.load(results, "curv_split")
    # 確かめ：full と start が E68 の同じ問題と一致するか
    Y = SS.load(results, "span_split")
    chk = []
    for name in ("full", "start"):
        k = KEY + (["meff", "K"] if name == "full" else ["K"])
        x = X[X.variant == name].set_index(k).uar
        y = Y[Y.variant == name].set_index(k).uar
        j = x.to_frame("x").join(y.rename("y"), how="inner")
        chk.append(dict(what=name, n=len(j), n_x=len(x), max_abs_diff=float((j.x - j.y).abs().max()) if len(j) else np.nan))
    chk = pd.DataFrame(chk)
    write(chk, "e69_check.csv", meta)
    print("== 確かめ（E68 との差）\n" + chk.to_string(index=False))
    base = X[X.variant == "start"].set_index(KEY + ["K"])
    V = X[X.variant != "start"].copy()
    ix = pd.MultiIndex.from_frame(V[KEY + ["K"]])
    V["gain"] = V.uar.values - base.uar.reindex(ix).values
    V["gain_anchor"] = V.uar_anchor.values - base.uar_anchor.reindex(ix).values
    G = V.groupby(["student", "meff", "variant", "K"]).agg(gain=("gain", "mean"), gain_anchor=("gain_anchor", "mean"),
                                                           n=("gain", "size")).reset_index()
    write(G, "e69_gain.csv", meta)
    F = V[V.variant == "full"]
    wcols = ["rel_err", "share_low"] + [f"{p}_{j}" for p in ("sh", "km", "kp") for j in range(NB)]
    Wt = F.groupby(["student", "meff", "K"])[wcols].median().reset_index()
    write(Wt, "e69_weights.csv", meta)
    tests = []
    for s, g in G[G.meff == 2000].groupby("student"):
        g = g.set_index(["variant", "K"]).gain
        w = Wt[(Wt.meff == 2000) & (Wt.student == s)].set_index("K").rel_err
        r1 = {K: w[K] for K in ("5", "10", "25")}
        tests.append(dict(rule="R1", student=s, detail=" ".join(f"K={K}:{v:.2f}" for K, v in r1.items()),
                          ok=bool(all(v <= 0.25 for v in r1.values()))))
        r2 = {K: g[("low", K)] / g[("full", K)] for K in ("5", "10", "25")}
        tests.append(dict(rule="R2", student=s, detail=" ".join(f"K={K}:{v:.2f}" for K, v in r2.items()),
                          ok=bool(all(v >= 0.7 for v in r2.values()))))
    tests = pd.DataFrame(tests)
    write(tests, "e69_tests.csv", meta)
    print("\n== 判定（pF・m=2000）\n" + tests.to_string(index=False))
    print("\n== 得（start との差，UAR のポイント．仮のターゲットの平均）")
    print((G.pivot_table(index=["student", "meff", "variant"], columns="K", values="gain") * 100).round(2).to_string())
    print("\n== 重みの水準（中央値．m=2000）")
    print(Wt[Wt.meff == 2000].round(3).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=WORLDS)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--pseudo-reps", nargs="+", type=int, help="試走用．既定は 0〜4")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--results-dir", type=Path, help="--analyze が curv_split_* と span_split_* を読む場所（既定は results/）")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.results_dir)
    global PSEUDO_REPS
    if args.pseudo_reps is not None:
        PSEUDO_REPS = args.pseudo_reps
    world = args.world
    out = args.out_dir or ROOT / "results" / f"curv_split_{world}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(world)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{world}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data_f = {g: M.load(g, world) for g in groups}
    data_m = {m: {g: MD.sub_group(data_f[g], a, b) for g in groups} for m, a, b in MD.blocks(world)}
    prov = {"script": "exp/explore_curvature_split.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "exp", "measure")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "targets": todo, "map_dir": str(map_dir(world)), "world": world, "mods": MD.WORLDS[world], "pseudo_reps": PSEUDO_REPS,
            "c_fit": MD.C_FIT, "belief": BELIEF, "meff": MEFF, "lam_cut": LAM_CUT, "edges": EDGES,
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
