"""【探索的】E40（GPU）：教える選び方（teach．E37）と E39 の仮説転移の学習器を組み合わせ，教師の歩幅と学習器の設定を一緒にメタ学習する。

規則は `docs/exploratory-analyses.md` E40（設計と予言 J1〜J3 は計算の前に書いた）。評価・候補・中心・予算（感情ごとに K 本）・仮のターゲットの
作りは E37・E39 と同じ。下書きは E39 の作り方で，(γ, ρ_s) はターゲットごとに E39 が選んだ値（`e39_targets.csv`）に固定する。学習器は E39 と
同じ仮説転移（ラベル無しのターゲットの散らばりから作る罰則の計量 M_ρM で座標を直す）で，ρ_M × C′ × C_b の 45 通り。教師は E37 の
`teach_order` に，計量で直した座標 x′ = x̃ M_ρM^{−1/2} と E39 の下書きの点数を渡す（歩幅 c 4 通り × ρ_M 3 通り．同じ ρ_M の学習器と組む）。
random（E31 と同じ K 本）も同じ学習器の 45 通りで解く。ターゲット（role=target，反復 0〜9）と，T のソースコーパス c′ を仮のターゲットにした分
（role=pseudo．T と c′ のコーパスの群を除くソースで下書きを作り，P4 型の合併の C = 0.01，反復 0〜4．c′ の候補のラベルを使う）を解く。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/teach_joint.py --kind audio_wavlm [--target …]
    ./.venv/bin/python exp/teach_joint.py --check audio_wavlm IEMOCAP-impro   # random の重なる組を E39 と照合（CPU）
    ./.venv/bin/python exp/teach_joint.py --analyze                           # 判定（CPU）

出力：`results/teach_joint_<kind>/<ターゲット>.csv`（role, meta_for, eval, policy, c, rho_M, cb, cp, rep, K, uar），
`<ターゲット>_model.csv`（教師の模型の d），`provenance.json`，`provenance_log.jsonl`。
判定は `results/explore_relation/e40_targets.csv`・`e40_tests.csv`・`e40_checks.csv`
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
import combine_drafts as CD  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
import select_teach as ST  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402

CS = ST.CS                                   # 教師の歩幅（E37 と同じ）
RHOS_M = (1.0, 0.97, 0.9)
CPS = (0.003, 0.01, 0.03, 0.1, 0.3)
CBS = (0.5, 2.0, 8.0)
PSEUDO_REPS = [0, 1, 2, 3, 4]
C_PSEUDO = 0.01
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"
T_KEYS = ["c", "rho_M", "cb", "cp"]          # teach の組
R_KEYS = ["rho_M", "cb", "cp"]               # random の組
CHECK_TOL = 0.1                              # E39 との照合：設定と K ごとの反復平均の差の絶対値の上限（点）


def evaluate(role, meta_for, ev, pool, data, C, reps, ks, gamma, rho_s, rows, mrows, cache):
    src_pc, lin, n_of = M5.pool_info(data, pool, C, cache)
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    dev, dt = src_pc.mu.device, src_pc.mu.dtype
    zero = torch.zeros(ev.X.shape[1], dtype=dt, device=dev)
    one = torch.ones(ev.X.shape[1], dtype=dt, device=dev)
    for r in reps:
        nontest = np.flatnonzero(~ctx.te[r])
        Xu = M.rows(ev.X, nontest) - ctx.ct[r]                                 # 候補（ラベル無しとしても使う．中心合わせ後の生の座標）
        ref_scale = CD.pair_scale(src_pc.logits(Xu, C))
        comps = {}
        for s in pool:                                                         # E39（meta5.evaluate）と同じ計算
            A, a0 = lin[(s, rho_s)]
            sc = CD.pair_scale(Xu @ A.T + a0)
            comps[(s, rho_s)] = (A / sc, a0 / sc)
        w = {s: float(n_of[s]) ** gamma for s in pool}
        tot = sum(w.values())
        A = sum(w[s] / tot * comps[(s, rho_s)][0] for s in pool)
        a0 = sum(w[s] / tot * comps[(s, rho_s)][1] for s in pool)
        g = ref_scale / CD.pair_scale(Xu @ A.T + a0)
        A, a0 = g * A, g * a0
        W_std, b_std = A * src_pc.sd, a0 + src_pc.mu @ A.T                    # 下書きを標準化した座標で
        Xstd = (Xu - src_pc.mu) / src_pc.sd
        Zp = Xstd - Xstd.mean(0)
        lam, V = torch.linalg.eigh(Zp.T @ Zp / len(nontest))
        lam = lam.clamp(min=0.0)
        tau = lam.mean()
        del Zp
        mats = {}
        for rho_m in RHOS_M:
            e_ = (1 - rho_m) * lam / tau + rho_m
            mats[rho_m] = ((V * e_.rsqrt()) @ V.T, (V * e_.sqrt()) @ V.T)
        L0 = Xu @ A.T + a0                                                     # 下書きの点数（どの座標でも同じ）
        yq = ev.y[nontest]

        def solve(sub, used, Am, Mh, emit):
            """K 本（sub.idx）で学習器の C′ × C_b を束ねて解き，テストの UAR を emit に渡す。"""
            jobs, masks = CB.jobs_for(sub, [r], {int(i): k for k, i in enumerate(used)}, len(used), False)
            Xs_used = ((M.rows(ev.X, used) - ctx.ct[r] - src_pc.mu) / src_pc.sd) @ Am
            anchor = logreg.Fitted(zero, one, {C: W_std @ Mh}, {C: b_std}, {C: 0}, {C: 0.0})
            fits = logreg.fit_batch(Xs_used, M.labels(ev.y[used]), masks * len(CBS), CPS, anchor=anchor, anchor_c=C,
                                    stats=(zero, one), Cb=[cb for cb in CBS for _ in masks])
            X_of = lambda r_, ii: ((M.rows(ev.X, ii) - ctx.ct[r] - src_pc.mu) / src_pc.sd) @ Am  # noqa: E731
            for k, cb in enumerate(CBS):
                CB.collect(sub, jobs, fits[k * len(masks):(k + 1) * len(masks)], CPS, X_of, None,
                           lambda rr, K, cp, u, cv, L, cb=cb: emit(rr, K, cp, u, cb), {})

        for rho_m, (Am, Mh) in mats.items():
            def emit_r(rr, K, cp, u, cb, rho_m=rho_m):
                rows.append({"role": role, "meta_for": meta_for, "eval": ev.name, "policy": "random", "c": np.nan,
                             "rho_M": rho_m, "cb": f"{cb:g}", "cp": cp, "rep": rr, "K": K, "uar": u})
            used = np.unique(np.concatenate([ctx.idx[(r, K)] for K in ctx.ks]))
            solve(ctx, used, Am, Mh, emit_r)
            Xt = Xstd @ Am                                                     # 学習器が解く座標（ρ_M = 1 なら E37 の教師と同じ）
            for c in CS:
                order, trace = ST.teach_order(Xt, yq, L0, c, 4 * kmax)
                sub = CB.Ctx.__new__(CB.Ctx)
                sub.__dict__.update(ctx.__dict__)
                sub.idx = {}
                for K in ks:
                    pp = np.array(order[: 4 * int(K)])
                    if len(np.unique(pp)) != len(pp) or (np.bincount(yq[pp], minlength=4) != int(K)).any():
                        raise ValueError(f"選んだ本数がおかしい: c={c} rho_M={rho_m} r{r} K{K}")
                    sub.idx[(r, K)] = nontest[pp]
                    mrows.append({"role": role, "meta_for": meta_for, "eval": ev.name, "c": c, "rho_M": rho_m, "rep": r, "K": K,
                                  "model_d": trace[4 * int(K) - 1]})

                def emit_t(rr, K, cp, u, cb, c=c, rho_m=rho_m):
                    rows.append({"role": role, "meta_for": meta_for, "eval": ev.name, "policy": "teach", "c": c,
                                 "rho_M": rho_m, "cb": f"{cb:g}", "cp": cp, "rep": rr, "K": K, "uar": u})
                solve(sub, nontest[np.array(order)], Am, Mh, emit_t)
            del Xt


def run_target(kind, T, data, reps, ks, out, cache, sel39, pseudo_reps):
    t0 = time.perf_counter()
    rows, mrows = [], []
    corp = lambda g: GROUP_SPEC[g][0]  # noqa: E731
    pool_T = [s for s in data if corp(s) != corp(T)]
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    gamma, rho_s = float(sel39.gamma), float(sel39.rho_s)
    evaluate("target", T, data[T], pool_T, data, C_T, reps, ks, gamma, rho_s, rows, mrows, cache)
    t1 = time.perf_counter()
    for c in (pool_T if pseudo_reps else []):
        pool_c = [s for s in pool_T if corp(s) != corp(c)]
        evaluate("pseudo", T, data[c], pool_c, data, C_PSEUDO, pseudo_reps, ks, gamma, rho_s, rows, mrows, cache)
    print(f"  {kind} {T}：ターゲット {t1 - t0:.0f} 秒，仮のターゲット {len(pool_T)} 個 {time.perf_counter() - t1:.0f} 秒", flush=True)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)
    pd.DataFrame(mrows).to_csv(out / f"{T}_model.csv", index=False)


# ---------------------------------------------------------------- 照合と判定（CPU）

def sel39_table() -> pd.DataFrame:
    e39 = pd.read_csv(OUT_REL / "e39_targets.csv", dtype={"K": str, "cb": str})
    return e39.drop_duplicates(["kind", "target"]).set_index(["kind", "target"])


def check(kind: str, T: str, out: Path | None = None) -> dict:
    """random の組のうち E39 の候補と重なるもの（C′ ∈ {0.01, 0.03, 0.1}，C_b ∈ {2, 8}）を，E39 の同じ (γ, ρ_s) の値と比べる。"""
    sel = sel39_table().loc[(kind, T)]
    a = pd.read_csv((out or ROOT / "results" / f"teach_joint_{kind}") / f"{T}.csv", dtype={"K": str, "cb": str})
    a = a[a.policy == "random"]
    b = pd.read_csv(ROOT / "results" / f"meta5_{kind}" / f"{T}.csv", dtype={"K": str, "cb": str})
    b = b[(b.gamma == float(sel.gamma)) & (b.rho_s == float(sel.rho_s))]
    keys = ["role", "eval", "rho_M", "cb", "cp", "rep", "K"]
    m = a.merge(b, on=keys, suffixes=("_j", "_b"))
    if m.empty:
        raise ValueError(f"{kind} {T}：E39 と重なる行が無い")
    m["diff"] = (m.uar_j - m.uar_b) * 100
    g = m.groupby(["role", "eval", "rho_M", "cb", "cp", "K"])["diff"].mean().abs()
    return {"kind": kind, "target": T, "rows": len(m), "same_rows": int((m["diff"].abs() < 1e-9).sum()),
            "max_row_diff": float(m["diff"].abs().max()), "max_mean_diff": float(g.max())}


def analyze() -> int:
    sel39 = sel39_table()
    e39 = pd.read_csv(OUT_REL / "e39_targets.csv", dtype={"K": str, "cb": str}).set_index(["kind", "target", "K"])
    e37 = pd.read_csv(OUT_REL / "e37_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"])
    tabs, checks = [], []
    for kind in MAIN:
        for f in sorted((ROOT / "results" / f"teach_joint_{kind}").glob("*.csv")):
            if f.stem.endswith("_model"):
                continue
            T = f.stem
            checks.append(check(kind, T))
            a = pd.read_csv(f, dtype={"K": str, "cb": str})
            a["cbv"] = a.cb.astype(float)
            ps = a[a.role == "pseudo"]
            pt = ps[ps.policy == "teach"].groupby(T_KEYS).uar.mean()
            pr = ps[ps.policy == "random"].groupby(R_KEYS).uar.mean()
            st, sr = pt.idxmax(), pr.idxmax()
            plug_c = pt.xs(sr, level=R_KEYS).idxmax()                          # 差し込み版：random の学習器のまま歩幅だけ選ぶ
            tg = a[(a.role == "target") & a.rep.isin(EC.REPS)]
            tt = tg[tg.policy == "teach"].groupby(T_KEYS + ["K"]).uar.mean() * 100
            tr = tg[tg.policy == "random"].groupby(R_KEYS + ["K"]).uar.mean() * 100
            s7 = pd.read_csv(ROOT / "results" / f"select_teach_{kind}" / f"{T}.csv", dtype={"K": str})
            s7 = s7[(s7.role == "target") & (s7.policy == e37.loc[(kind, T), "policy_star"]) & s7.rep.isin(EC.REPS)]
            e37k = s7.groupby("K").uar.mean() * 100
            for K in ("1", "5", "10", "25"):
                tk, rk = tt.xs(K, level="K"), tr.xs(K, level="K")
                ot, orr = tk.idxmax(), rk.idxmax()
                ref = e39.loc[(kind, T, K)]
                tabs.append({"kind": kind, "target": T, "K": K, "gamma": float(sel39.loc[(kind, T), "gamma"]),
                             "rho_s": float(sel39.loc[(kind, T), "rho_s"]),
                             **{f"t_{k}": v for k, v in zip(T_KEYS, st)}, **{f"r_{k}": v for k, v in zip(R_KEYS, sr)},
                             "plug_c": plug_c, "teach": float(tk.loc[st]), "random": float(rk.loc[sr]),
                             "plug": float(tk.loc[(plug_c,) + tuple(sr)]), "teach_oracle": float(tk.max()), "random_oracle": float(rk.max()),
                             **{f"to_{k}": v for k, v in zip(T_KEYS, ot)}, **{f"ro_{k}": v for k, v in zip(R_KEYS, orr)},
                             "E37": float(e37k[K]), "E39": float(ref.meta), "D1": float(ref.D1), "P4": float(ref.P4)})
    D = pd.DataFrame(tabs)
    for col in ("t_cb", "r_cb", "to_cb", "ro_cb"):
        D[col + "_v"] = D[col].astype(float)
    pairs = {"teach-random": ("teach", "random"), "teach-E37": ("teach", "E37"), "teach-plug": ("teach", "plug"),
             "teach-E39": ("teach", "E39"), "teach-D1": ("teach", "D1"), "teach-P4": ("teach", "P4"), "random-E39": ("random", "E39"),
             "teach_oracle-teach": ("teach_oracle", "teach"), "random_oracle-random": ("random_oracle", "random")}
    rows = []
    for (kind, K), g in D.groupby(["kind", "K"]):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for m in ("teach", "random", "plug", "E37", "E39", "D1", "P4", "teach_oracle", "random_oracle"):
            row[m] = float(g[m].mean())
        for name, (x, y) in pairs.items():
            diff = g[x] - g[y]
            row[name] = float(diff.mean())
            row[f"{name}_wins"] = int((diff > 0).sum())
        rows.append(row)
    R = pd.DataFrame(rows)
    R["K"] = pd.Categorical(R.K, ["1", "5", "10", "25"], ordered=True)
    R = R.sort_values(["kind", "K"]).reset_index(drop=True)
    sel = D.drop_duplicates(["kind", "target"])
    J3 = pd.DataFrame([{"kind": kind, "targets": len(g), "cp_teach_gt": int((g.t_cp > g.r_cp).sum()),
                        "cp_teach_lt": int((g.t_cp < g.r_cp).sum()), "cb_teach_gt": int((g.t_cb_v > g.r_cb_v).sum()),
                        "cb_teach_lt": int((g.t_cb_v < g.r_cb_v).sum())} for kind, g in sel.groupby("kind")]).set_index("kind")
    for kind in ("audio_wavlm", "audio_hubert"):
        m = (R.kind == kind) & R.K.isin(["5", "10"])
        if m.sum() == 2:
            R.loc[R.kind == kind, "J1"] = bool((R[m]["teach-random"] > 0).all())
            R.loc[R.kind == kind, "J2"] = bool((R[m]["teach-E37"] > 0).all())
        if kind in J3.index:
            j = J3.loc[kind]
            half = int(j.targets) // 2
            R.loc[R.kind == kind, "J3"] = bool(j.cp_teach_gt > half and j.cb_teach_gt <= half)
    Cks = pd.DataFrame(checks)
    D.to_csv(OUT_REL / "e40_targets.csv", index=False)
    R.to_csv(OUT_REL / "e40_tests.csv", index=False)
    Cks.to_csv(OUT_REL / "e40_checks.csv", index=False)
    J3.to_csv(OUT_REL / "e40_j3.csv")
    pd.set_option("display.width", 250)
    cols = ["kind", "K", "targets", "teach", "random", "E37", "E39", "teach-random", "teach-E37", "teach-plug", "teach-E39",
            "random-E39", "teach_oracle-teach"]
    print(R[cols].round(2).to_string(index=False))
    print(J3.to_string())
    print(R.drop_duplicates("kind")[["kind"] + [c for c in ("J1", "J2", "J3") if c in R]].to_string(index=False))
    print("E39 との照合（設定と K ごとの反復平均の差の最大）：", round(float(Cks.max_mean_diff.max()), 4), f"（{len(Cks)} ターゲット）")
    for k in ("t_c", "t_rho_M", "t_cp", "t_cb", "r_rho_M", "r_cp", "r_cb"):
        print(k, sel.groupby("kind")[k].value_counts().unstack(fill_value=0).to_dict("index"))
    return 0


def posthoc() -> int:
    """【結果を見た後の確かめ】仮のターゲットの値で，(1) 教師の歩幅 c を固定して teach の (ρ_M, C_b, C′) を選び直したときの C′・C_b，
    (2) C′ を random で選ばれた値に固定して teach の (c, ρ_M, C_b) を選び直したときの C_b を，random で選ばれた値と比べる。"""
    D = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"])
    rows = []
    for (kind, T), r in D.iterrows():
        a = pd.read_csv(ROOT / "results" / f"teach_joint_{kind}" / f"{T}.csv", dtype={"K": str, "cb": str})
        ps = a[(a.role == "pseudo") & (a.policy == "teach")]
        row = {"kind": kind, "target": T, "t_c": r.t_c, "t_cp": r.t_cp, "r_cp": r.r_cp, "r_cb": float(r.r_cb)}
        for c in CS:
            g = ps[ps.c == c].groupby(["rho_M", "cb", "cp"]).uar.mean().idxmax()
            row[f"cp_at_c{c:g}"], row[f"cb_at_c{c:g}"] = g[2], float(g[1])
        h = ps[ps.cp == r.r_cp].groupby(["c", "rho_M", "cb"]).uar.mean().idxmax()
        row["cb_at_r_cp"] = float(h[2])
        rows.append(row)
    X = pd.DataFrame(rows)
    X.to_csv(OUT_REL / "e40_posthoc.csv", index=False)
    for kind, g in X.groupby("kind", sort=False):
        s = {f"c={c:g}": (int((g[f"cp_at_c{c:g}"] > g.r_cp).sum()), int((g[f"cb_at_c{c:g}"] > g.r_cb).sum())) for c in CS}
        print(kind, len(g), "（C′ が random より大きい数，C_b が random より大きい数）", s,
              "| C′ を random の値に固定したときの C_b（大きい，小さい，同じ）：",
              (int((g.cb_at_r_cp > g.r_cb).sum()), int((g.cb_at_r_cp < g.r_cb).sum()), int((g.cb_at_r_cp == g.r_cb).sum())),
              "| 選ばれた C′／c：", sorted((g.t_cp / g.t_c).round(1).unique().tolist()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--pseudo-reps", nargs="*", type=int, default=PSEUDO_REPS)
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--check", nargs=2, metavar=("KIND", "TARGET"))
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--posthoc", action="store_true", help="結果を見た後の確かめ（--analyze の後に CPU で）")
    args = ap.parse_args()
    if args.analyze:
        return analyze()
    if args.posthoc:
        return posthoc()
    if args.check:
        res = check(*args.check, out=args.out_dir)
        print(json.dumps(res, ensure_ascii=False))
        return 0 if res["max_mean_diff"] <= CHECK_TOL else 1
    out = args.out_dir or ROOT / "results" / f"teach_joint_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    sel39 = sel39_table()
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/teach_joint.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "pseudo_reps": args.pseudo_reps,
            "grid": {"c": CS, "rho_M": RHOS_M, "cp": CPS, "cb": CBS}, "C_pseudo": C_PSEUDO,
            "draft": {t: {"gamma": float(sel39.loc[(args.kind, t), "gamma"]), "rho_s": float(sel39.loc[(args.kind, t), "rho_s"])}
                      for t in todo},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    cache: dict = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, args.reps, args.ks, out, cache, sel39.loc[(args.kind, t)], args.pseudo_reps)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
