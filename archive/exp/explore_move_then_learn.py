"""【探索的】E66 の下調べ（GPU）：先生で出発点を動かし，K 本ではそこから学び直す．仮のターゲットだけ。

問い（2026-10-28．研究者の承認．`docs/exploratory-analyses.md` E66．設計と予言は計算の前に書いた）
  E65 で，別のモダリティの先生の弱いラベルは K とともに UAR を上げなくなった．弱いラベルの項は，先生の解のまわりで
  2 次まで展開すると，罰則の中心を先生の解へ近づけ，生徒を保つ強さを m に比例して増やす．保つ強さのほうを外すと，
  先生の価値（ターゲットのラベル何本ぶんか）は K によらず一定になるか．
  - K=0：先生の弱いラベルで出発点を学び直す（E60 の K=0 の生徒と同じ問題．`solve_all` をそのまま使う）
  - K≥1：弱いラベルを損失から外し，K=0 の生徒 (W_Q, b_Q) を罰則の中心にして，K 本だけで学び直す．
         重みの罰則は C′/h（h ∈ {1, 2, 4}．h=1 は出発点と同じ強さ），切片の罰則は C_b/h_b（h_b ∈ {1, 4, 16}）
         予言（P1〜P5）は h_b = 1（切片の罰則は出発点のまま）で判定する．h_b = 4, 16 は，試走（AV の AFEW の反復 0）で
         K=1 の得が今の手法より低かったのを見て足した探索の版（切片を K 本で学び直す損．第 I 部の 1/(2K)）
  - 対照：出発点そのもの（中心 W0．h = h_b = 1 は E60 の none と同じ問題）
  生徒・先生・分割・仮のターゲットは E60（`exp/multimodal_distill.py`）の仮のターゲットの部分と同じ．
  評価に使うターゲットのテストは使わない．今の手法（弱いラベルとラベルを同時に学ぶ）の値は E60 の `<T>_pseudo.csv` を使う

    ./.venv/bin/python exp/explore_move_then_learn.py --background        （計算の前の予言．E60 と E65 の結果だけ．数秒）
    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/explore_move_then_learn.py --world fuse-audio-video [--target …] [--pseudo-reps …] [--out-dir …]
    ./.venv/bin/python exp/explore_move_then_learn.py --analyze [--results-dir …]
出力：`results/move_then_learn_<世界>/<T>.csv`（pseudo, rep, student, center, belief, meff, hold, hold_b, K, uar．K=0 の行は (h, h_b) ごとに同じ値を書く），
      `provenance.json`，`provenance_log.jsonl`．予言は `results/explore_relation/e66_background.csv`・`e66_pred.csv`・`e66_pred_rows.csv`，
      判定は `e66_check.csv`・`e66_gain.csv`・`e66_value.csv`・`e66_selection.csv`
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
import explore_teacher_value as E  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
import multimodal_distill as MD  # noqa: E402
from data import groups_for  # noqa: E402
from distill_belief import solve_all, uar_of  # noqa: E402
from geom_e0 import settings  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import present_weights  # noqa: E402

WORLDS = ["fuse-audio-video", "fuse-audio-video-text"]
BEL, MEFF, HOLDS = ["p", "pF", "poe", "agree"], (100, 500, 2000), (1, 2, 4)
HOLDS_B = (1, 4, 16)                                                 # 切片の罰則 C_b/h_b（1 だけが予言の版）
TEACHERS = ["pF", "poe", "agree"]                                    # 選び方の候補（E60 の単独の生徒と同じ．p は対照）
KS_L = ("1", "5", "10", "25")
KS = ("0",) + KS_L
PSEUDO_REPS = [0, 1, 2, 3, 4]
OUT_REL = ROOT / "results" / "explore_relation"
KEY = ["world", "target", "pseudo", "rep", "student"]


# ---------------------------------------------------------------- 計算（GPU）

def refit_from(anc, Z, yq, label_sets, cp, cb):
    """中心 anc（Fitted．C_FIT の解）から，弱いラベルなしで K 本だけ学び直す．戻りは {(h, h_b, K): (W, b)}（罰則 C′/h，C_b/h_b）。"""
    rows_all = np.unique(np.concatenate([label_sets[K] for K in KS_L]))
    dev = Z.device
    X = Z[torch.as_tensor(rows_all, device=dev)]
    yt = M.labels(yq[rows_all])
    pos = {int(i): j for j, i in enumerate(rows_all)}
    weights, keys = [], []
    for K in KS_L:
        it = torch.as_tensor([pos[int(i)] for i in label_sets[K]], device=dev)
        w = torch.zeros(len(rows_all), dtype=logreg.DTYPE, device=dev)
        w[it] = present_weights(yt[it])
        for hb in HOLDS_B:
            weights.append(w)
            keys.append((hb, K))
    zero = torch.zeros(Z.shape[1], dtype=Z.dtype, device=dev)
    Cs = tuple(cp / h for h in HOLDS)
    fits = logreg.fit_batch(X, yt, [np.ones(len(rows_all), bool)] * len(keys), Cs, anchor=anc, anchor_c=MD.C_FIT,
                            stats=(zero, torch.ones_like(zero)), Cb=[cb / hb for hb, _ in keys], weights=weights)
    return {(h, hb, K): (f.W[c], f.b[c]) for (hb, K), f in zip(keys, fits) for h, c in zip(HOLDS, Cs)}


def solve_moved(world, stu, yq, yte, label_sets, rows, tag):
    b1 = MD.beliefs1(stu, world)
    for m in MD.WORLDS[world]:
        s = stu[m]
        st, rho_m, cp, cb, Z, Zte = s["st"], s["rho_m"], s["cp"], s["cb"], s["Z"], s["Zte"]
        a0 = st.anchor(rho_m)
        u0 = uar_of(a0.W[MD.C_FIT], a0.b[MD.C_FIT], Zte, yte)
        base = {**tag, "student": m}
        rows += [{**base, "center": "start", "belief": "none", "meff": 0, "hold": h, "hold_b": hb, "K": "0", "uar": u0}
                 for h in HOLDS for hb in HOLDS_B]
        for (h, hb, K), (W, b) in refit_from(a0, Z, yq, label_sets, cp, cb).items():
            rows.append({**base, "center": "start", "belief": "none", "meff": 0, "hold": h, "hold_b": hb, "K": K,
                         "uar": uar_of(W, b, Zte, yte)})
        bel = {"p": s["P"], **b1}
        sol = solve_all(Z, yq, bel, {"0": label_sets["0"]}, [(b, e) for b in BEL for e in MEFF], st, rho_m, MD.C_FIT, cp, cb)
        zero = torch.zeros(Z.shape[1], dtype=Z.dtype, device=Z.device)
        for (b, e, _), (Wq, bq) in sol.items():
            uq = uar_of(Wq, bq, Zte, yte)
            rows += [{**base, "center": "moved", "belief": b, "meff": e, "hold": h, "hold_b": hb, "K": "0", "uar": uq}
                     for h in HOLDS for hb in HOLDS_B]
            anc = logreg.Fitted(zero, torch.ones_like(zero), {MD.C_FIT: Wq}, {MD.C_FIT: bq}, {MD.C_FIT: 0}, {MD.C_FIT: 0.0})
            for (h, hb, K), (W, bb) in refit_from(anc, Z, yq, label_sets, cp, cb).items():
                rows.append({**base, "center": "moved", "belief": b, "meff": e, "hold": h, "hold_b": hb, "K": K,
                             "uar": uar_of(W, bb, Zte, yte)})


def run_target(world, T, data_f, data_m, out, cache):
    """E60 の run_target の仮のターゲットの部分と同じ生徒・分割で解く。"""
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
            solve_moved(world, stu, evc.y[pool], evc.y[te], label_sets, rows, {"pseudo": c, "rep": r})
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 計算の前の予言（CPU）

def uar_at(ks, us, k):
    """学習曲線の本数 k での UAR（log k について線形に補う）．曲線の端を超えたら端の値と印 1。"""
    if not np.isfinite(k):
        return np.nan, 0
    if k >= ks[-1]:
        return float(us[-1]), 1
    if k <= ks[0]:
        return float(us[0]), -1
    lk = np.log(ks)
    i = int(np.searchsorted(lk, np.log(k))) - 1
    f = (np.log(k) - lk[i]) / (lk[i + 1] - lk[i])
    return float(us[i] + f * (us[i + 1] - us[i])), 0


def e60_pseudo(worlds):
    out = []
    for w in worlds:
        for f in sorted((ROOT / "results" / f"multimodal_distill_{w}").glob("*_pseudo.csv")):
            d = pd.read_csv(f, dtype={"K": str})
            out.append(d[d.student.isin(E.SINGLE)].assign(world=w, target=f.name[: -len("_pseudo.csv")]))
    return pd.concat(out, ignore_index=True)


def background() -> int:
    """E60 の仮のターゲットの値と E65 の本数から，対照 p の損・先生だけの情報の分・本数が一定のときの得を出す。"""
    rows = []
    for scope, worlds in (("AV+AVT", WORLDS), ("all4", E.WORLDS)):
        d = e60_pseudo(worlds)
        base = d[d.belief == "none"].set_index(KEY + ["K"]).uar
        x = d[d.belief.isin(["p", "pF"])].copy()
        x["gain"] = 100 * (x.uar.values - base.reindex(pd.MultiIndex.from_frame(x[KEY + ["K"]])).values)
        g = x.pivot_table(index=KEY + ["meff", "K"], columns="belief", values="gain").reset_index()
        g["info"] = g.pF - g.p
        for col in ("p", "pF", "info"):
            t = g.groupby(["student", "meff", "K"])[col].mean().reset_index().rename(columns={col: "value"})
            rows.append(t.assign(scope=scope, what=f"gain_{col}"))
        a = d[(d.belief == "none") | (d.belief.isin(["p", "pF"]) & (d.meff == 2000))]
        t = (a.groupby(["student", "belief", "K"]).uar.mean() * 100).reset_index().rename(columns={"uar": "value", "belief": "meff"})
        rows.append(t.assign(scope=scope, what="uar_abs"))
    Bg = pd.concat(rows, ignore_index=True)

    D = pd.read_csv(OUT_REL / "e65_teacher_value.csv", comment="#", dtype={"K": str, "meff": str})
    D = D[D.single & D.world.isin(WORLDS) & D.belief.isin(TEACHERS)]
    k0 = D[D.K == "0"].set_index(KEY + ["belief", "meff"])
    curves, pr = {}, []
    for x in D[D.K != "0"].itertuples():
        key = (x.world, x.target, x.pseudo, x.rep, x.student, x.belief, x.meff)
        if key not in k0.index or x.flag_none != 0 or not np.isfinite(k0.loc[key, "n_mod"]):
            continue
        ck = (x.student, x.pseudo)
        if ck not in curves:
            curves[ck] = E.curve(*ck)
        u, fl = uar_at(*curves[ck], x.k_none + k0.loc[key, "n_mod"])
        pr.append(dict(zip(KEY + ["belief", "meff"], key), K=x.K, pred=100 * (u - x.uar_none), joint=100 * (x.uar - x.uar_none), cens=fl))
    P = pd.DataFrame(pr)
    S = P.groupby(["student", "belief", "meff", "K"]).agg(pred=("pred", "mean"), joint=("joint", "mean"), n=("pred", "size"),
                                                         cens=("cens", lambda v: float((v == 1).mean()))).reset_index()
    z = k0.reset_index()
    z = z[(z.flag_none == 0) & np.isfinite(z.n_mod) & (z.n_start > 0)]
    z = z.assign(price_ratio=(z.n_start + z.n_mod) / z.n_start)
    R = z.groupby(["student", "belief", "meff"]).agg(n_start=("n_start", "median"), n_mod=("n_mod", "median"),
                                                    price_ratio=("price_ratio", "median")).reset_index()
    meta = _meta()
    OUT_REL.mkdir(parents=True, exist_ok=True)
    (OUT_REL / "e66_background.csv").write_text(f"# {meta}\n" + Bg.to_csv(index=False))
    (OUT_REL / "e66_pred.csv").write_text(f"# {meta}\n" + S.merge(R, on=["student", "belief", "meff"], how="left").to_csv(index=False))
    (OUT_REL / "e66_pred_rows.csv").write_text(f"# {meta}\n" + P.to_csv(index=False))
    pd.set_option("display.width", 250)
    for scope in ("AV+AVT", "all4"):
        print(f"== {scope}：出発点との差（ポイント．単独の生徒，仮のターゲットの平均）")
        t = Bg[(Bg.scope == scope) & Bg.what.str.startswith("gain")]
        print(t.pivot_table(index=["student", "what", "meff"], columns="K", values="value")[list(KS)].round(2).to_string())
        t = Bg[(Bg.scope == scope) & (Bg.what == "uar_abs")]
        print(t.pivot_table(index=["student", "meff"], columns="K", values="value")[list(KS)].round(2).to_string(), "\n")
    print("== 予言（AV+AVT．本数が K=0 の値のまま一定なら．pred = 動かしてから学ぶ，joint = 今の手法の実測．同じ行）")
    print(S.pivot_table(index=["student", "belief", "meff"], columns="K", values=["pred", "joint"])
          .round(2).to_string())
    print("\n== 動かした出発点の本数の比（(n_start + n_mod)/n_start，K=0 の中央値）")
    print(R.round(2).to_string(index=False))
    return 0


# ---------------------------------------------------------------- 判定（CPU）

def _meta():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return json.dumps({"script": "exp/explore_move_then_learn.py", "commit": commit, "worlds": WORLDS})


def load(results):
    out = []
    for w in WORLDS:
        for f in sorted((results / f"move_then_learn_{w}").glob("*.csv")):
            out.append(pd.read_csv(f, dtype={"K": str}).assign(world=w, target=f.stem))
    return pd.concat(out, ignore_index=True)


def consistency(X, J):
    """E60 と同じ問題（出発点の h = h_b = 1，K=0 の動かした出発点）の UAR が一致するか。"""
    one = (X.hold == 1) & (X.hold_b == 1)
    a = X[(X.center == "start") & one].set_index(KEY + ["K"]).uar
    b = J[J.belief == "none"].set_index(KEY + ["K"]).uar
    c = X[(X.center == "moved") & (X.K == "0") & one].set_index(KEY + ["belief", "meff"]).uar
    j0 = J[(J.K == "0") & (J.belief != "none")].set_index(KEY + ["belief", "meff"]).uar
    out = []
    for name, x, y in (("start", a, b), ("moved_K0", c, j0)):
        y = y.reindex(x.index)
        diff = (x - y).abs()
        out.append(dict(check=name, n=int(len(x)), missing=int(y.isna().sum()), max_abs=float(diff.max()), n_diff=int((diff > 1e-9).sum())))
    return pd.DataFrame(out)


def base_of(X):
    return X[(X.center == "start") & (X.hold == 1) & (X.hold_b == 1)].set_index(KEY + ["K"]).uar


def gains(X, J):
    base = base_of(X)
    X = X.copy()
    X["gain"] = 100 * (X.uar.values - base.reindex(pd.MultiIndex.from_frame(X[KEY + ["K"]])).values)
    jb = J[J.belief == "none"].set_index(KEY + ["K"]).uar
    J = J[J.belief != "none"].copy()
    J["joint"] = 100 * (J.uar.values - jb.reindex(pd.MultiIndex.from_frame(J[KEY + ["K"]])).values)
    return X, J


def values(X):
    """E65 と同じ読み方で，動かしてから学んだ生徒の本数 n_mod = K'(u) − K'(u_出発点) を出す。"""
    curves, out = {}, []
    base = base_of(X)
    for x in X[X.center == "moved"].itertuples():
        ck = (x.student, x.pseudo)
        if ck not in curves:
            curves[ck] = E.curve(*ck)
        ks, us = curves[ck]
        u0 = float(base.loc[(x.world, x.target, x.pseudo, x.rep, x.student, x.K)])
        k0, f0 = E.k_equiv(ks, us, u0)
        k1, f1 = E.k_equiv(ks, us, float(x.uar))
        out.append(dict(student=x.student, belief=x.belief, meff=x.meff, hold=x.hold, hold_b=x.hold_b, K=x.K,
                        n_mod=(k1 - k0) if (f0 == 0 and f1 == 0) else np.nan, cens_hi=float(f1 == 1)))
    V = pd.DataFrame(out)
    return V.groupby(["student", "belief", "meff", "hold", "hold_b", "K"]).agg(
        n_mod=("n_mod", "median"), cens_hi=("cens_hi", "mean"), n=("n_mod", "size")).reset_index()


def selection(X, pre=True):
    """仮のターゲットを 1 つ抜いて残りで選び，抜いた仮のターゲットで当たりを比べる（E65 と同じ）．
    A は候補を K ごとに，B は K によらず 1 つ選ぶ．「使わない」（出発点 h = h_b = 1）も候補．
    pre=True は予言の版（先生 × m × h，h_b = 1），False は探索の版（h_b も選び，K≥1 で学び直さない版も候補に入れる）。"""
    c0 = X[(X.center == "start") & (X.hold == 1) & (X.hold_b == 1)].assign(cand="none")
    c1 = X[(X.center == "moved") & X.belief.isin(TEACHERS) & ((X.hold_b == 1) if pre else True)].copy()
    c1["cand"] = c1.belief + "|" + c1.meff.astype(str) + "|" + c1.hold.astype(str) + "|" + c1.hold_b.astype(str)
    parts = [c0, c1]
    if not pre:
        z = X[(X.center == "moved") & X.belief.isin(TEACHERS) & (X.K == "0") & (X.hold == 1) & (X.hold_b == 1)]
        parts += [z.assign(K=K, cand=z.belief + "|" + z.meff.astype(str) + "|norefit") for K in KS]
    H = pd.concat(parts, ignore_index=True)
    out = []
    for (w, T, s), g in H.groupby(["world", "target", "student"]):
        cs = sorted(g.pseudo.unique())
        if len(cs) < 2:
            continue
        for c in cs:
            tr, te = g[g.pseudo != c], g[g.pseudo == c]
            m_tr = tr.groupby(["K", "cand"]).uar.mean()
            pick_a = {K: m_tr.loc[K].idxmax() for K in KS}
            pick_b = tr.groupby("cand").uar.mean().idxmax()
            te_m = te.groupby(["K", "cand"]).uar.mean()
            for K in KS:
                out.append(dict(world=w, target=T, student=s, held_out=c, K=K, pick_a=pick_a[K], pick_b=pick_b,
                                uar_a=te_m.get((K, pick_a[K]), np.nan), uar_b=te_m.get((K, pick_b), np.nan),
                                uar_none=te_m.get((K, "none"), np.nan)))
    return pd.DataFrame(out).assign(version="pre" if pre else "explore")


def analyze(results=None) -> int:
    results = results or ROOT / "results"
    X = load(results)
    J = e60_pseudo(WORLDS)
    J = J.merge(X[KEY].drop_duplicates(), on=KEY)                           # 計算した（世界, T, 仮のターゲット, 反復, 生徒）だけ
    meta = _meta()
    chk = consistency(X, J)
    X, J = gains(X, J)
    G = X.groupby(["student", "center", "belief", "meff", "hold", "hold_b", "K"]).gain.mean().reset_index()
    GJ = J.groupby(["student", "belief", "meff", "K"]).joint.mean().reset_index()
    G = G.merge(GJ, on=["student", "belief", "meff", "K"], how="left")
    V = values(X)
    L = pd.concat([selection(X, True), selection(X, False)], ignore_index=True)
    pr = pd.read_csv(OUT_REL / "e66_pred_rows.csv", comment="#", dtype={"K": str})
    mv = X[(X.center == "moved") & (X.hold == 1) & (X.hold_b == 1)][KEY + ["belief", "meff", "K", "gain"]]
    mp = pr.merge(mv, on=KEY + ["belief", "meff", "K"])
    PM = mp.groupby(["student", "belief", "meff", "K"]).agg(pred=("pred", "mean"), moved=("gain", "mean"), joint=("joint", "mean"),
                                                          n=("gain", "size")).reset_index()
    E65 = pd.read_csv(OUT_REL / "e65_selection.csv", comment="#", dtype={"K": str})
    E65 = E65[E65.world.isin(WORLDS) & E65.single]
    # 判定（予言の版：h = h_b = 1．P5 は h_b = 1 の候補）
    G1 = G[(G.hold == 1) & (G.hold_b == 1)]
    V1 = V[(V.hold == 1) & (V.hold_b == 1)]
    res = []
    for s in sorted(X.student.unique()):
        g = G1[(G1.student == s) & (G1.center == "moved") & (G1.belief == "pF") & (G1.meff == 2000)].set_index("K").gain
        res.append(dict(rule="P1", student=s, detail=" ".join(f"{g[K]:+.2f}" for K in KS_L), ok=bool(all(g[K] > 0 for K in KS_L))))
        v = V1[(V1.student == s) & (V1.belief == "pF") & (V1.meff == 2000)].set_index("K").n_mod
        res.append(dict(rule="P2", student=s, detail=f"K=0 {v['0']:.1f}，K=25 {v['25']:.1f}，比 {v['25'] / v['0']:.2f}",
                        ok=bool(v["25"] >= 0.5 * v["0"])))
        p = PM[(PM.student == s) & (PM.belief == "pF") & (PM.meff == 2000)].set_index("K")
        res.append(dict(rule="P3", student=s, detail=" ".join(f"{p.moved[K] - p.pred[K]:+.2f}" for K in KS_L),
                        ok=bool(all(abs(p.moved[K] - p.pred[K]) <= 0.5 for K in KS_L))))
        q = G1[(G1.student == s) & (G1.center == "moved") & (G1.belief == "p") & G1.K.isin(KS_L)]
        res.append(dict(rule="P4", student=s, detail=f"最小 {q.gain.min():+.2f}", ok=bool(q.gain.min() >= -0.2)))
    sa = L[L.version == "pre"].groupby("K")[["uar_a", "uar_b", "uar_none"]].mean() * 100
    res.append(dict(rule="P5", student="単独の生徒", detail=" ".join(f"{sa.uar_b[K] - sa.uar_a[K]:+.2f}" for K in KS),
                    ok=bool(all(sa.uar_b[K] >= sa.uar_a[K] - 0.1 for K in KS))))
    C = pd.DataFrame(res)
    for name, df in (("e66_check", pd.concat([chk.assign(rule="consistency"), C], ignore_index=True)), ("e66_gain", G),
                     ("e66_value", V), ("e66_selection", L), ("e66_pred_match", PM)):
        (OUT_REL / f"{name}.csv").write_text(f"# {meta}\n" + df.to_csv(index=False))
    pd.set_option("display.width", 250)
    print("== E60 と同じ問題の一致\n" + chk.to_string(index=False))
    print("\n== 出発点との差（ポイント．単独の生徒，仮のターゲットの平均．m=2000）．moved = 動かしてから学ぶ（h, h_b ごと），joint = 今の手法（E60）")
    t = G[(G.center == "moved") & (G.belief.isin(["p", "pF"])) & (G.meff == 2000)]
    print(t.pivot_table(index=["student", "belief", "hold", "hold_b"], columns="K", values="gain")[list(KS)].round(2).to_string())
    t = GJ[GJ.belief.isin(["p", "pF"]) & (GJ.meff == 2000)]
    print(t.pivot_table(index=["student", "belief"], columns="K", values="joint")[list(KS)].round(2).to_string())
    print("\n== 予言との比較（pF・m=2000・h = h_b = 1．予言を計算できた行）")
    t = PM[(PM.belief == "pF") & (PM.meff == 2000)]
    print(t.pivot_table(index="student", columns="K", values=["pred", "moved", "joint"]).round(2).to_string())
    print("\n== 先生の本数（中央値．pF・m=2000）")
    t = V[(V.belief == "pF") & (V.meff == 2000)]
    print(t.pivot_table(index=["student", "hold", "hold_b"], columns="K", values="n_mod")[list(KS)].round(1).to_string())
    print("\n== 選び方（抜いた仮のターゲットの UAR − 出発点，ポイント）．pre = 予言の版，explore = h_b と学び直さない版も候補，E65 = 今の手法")
    se = L[L.version == "explore"].groupby("K")[["uar_a", "uar_b", "uar_none"]].mean() * 100
    e = E65.groupby("K")[["uar_a", "uar_b", "uar_none"]].mean() * 100
    print(pd.DataFrame({"pre A": sa.uar_a - sa.uar_none, "pre B": sa.uar_b - sa.uar_none,
                        "explore A": se.uar_a - se.uar_none, "explore B": se.uar_b - se.uar_none,
                        "E65 A": e.uar_a - e.uar_none, "E65 B": e.uar_b - e.uar_none}).loc[list(KS)].round(2).to_string())
    print("\n== 判定\n" + C.to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=WORLDS)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--pseudo-reps", nargs="+", type=int, help="試走用．既定は 0〜4")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--results-dir", type=Path, help="--analyze が move_then_learn_* を読む場所（既定は results/）")
    args = ap.parse_args()
    if args.background:
        return background()
    if args.analyze:
        return analyze(args.results_dir)
    global PSEUDO_REPS
    if args.pseudo_reps is not None:
        PSEUDO_REPS = args.pseudo_reps
    world = args.world
    out = args.out_dir or ROOT / "results" / f"move_then_learn_{world}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(world)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{world}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data_f = {g: M.load(g, world) for g in groups}
    data_m = {m: {g: MD.sub_group(data_f[g], a, b) for g in groups} for m, a, b in MD.blocks(world)}
    prov = {"script": "exp/explore_move_then_learn.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "exp", "measure")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "targets": todo, "map_dir": str(map_dir(world)), "world": world, "mods": MD.WORLDS[world], "pseudo_reps": PSEUDO_REPS,
            "c_fit": MD.C_FIT, "beliefs": BEL, "meff": MEFF, "holds": HOLDS, "holds_b": HOLDS_B, "agree_low": MD.AGREE_LOW,
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
