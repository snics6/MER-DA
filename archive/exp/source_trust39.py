"""【探索的】E56（GPU）：信頼度の低いソースの発話を外してから E39 の出発点を作ると，E39（全部使う）を上回るか（問い 6．E55 の E39 版）。

規則は `docs/exploratory-analyses.md` E56（設計と予言は計算の前に書いた）。信頼度と外し方は E55 と同じ（`source_trust.trust_of`・`keep_mask`）。
外した残りから，E39 の出発点（ソース群ごとの閉じた形の向きを尺度をそろえて N_s^γ の重みで足す．`meta5.pool_info` と同じ計算を残りの発話だけで行う）と，
尺度と標準化の基準にする P4 型の合併を作り直す。生徒は E40 と同じ作りの E39 の学習器で，5 つの値（γ, ρ_s, ρ_M, C_b, C′）は E39 がターゲットごとに選んだ値に固定する
（`e39_targets.csv`，T は `e39_text_targets.csv`．random の K 本で選んだもの）。外す割合 r ∈ {0, 10, 25, 50, 75, 90} は，S の群を 1 つずつ仮のターゲット c にして選ぶ
（c のコーパスを除く群だけで信頼度と出発点を作る．合併の C は 0.01，K ∈ {1, 5}，反復 0〜4．E40 の仮のターゲットと同じ作り）。T は仮のターゲットで信頼度を測れない
（E55）ので，r ごとの値を記述として出す。評価・K 本は E39 と同じ（E31 の K 本）。比べる相手：r = 0（E39 そのもの．E39 の値と一致することを確かめる），同じ数をランダムに外す版，
E55 の P4 ＋信頼度。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/source_trust39.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/source_trust39.py --analyze

出力：`results/source_trust39_<kind>/<ターゲット>.csv`（policy, r, rho_M, cb, cp, rep, K, uar），`<ターゲット>_pseudo.csv`，`provenance.json`，`provenance_log.jsonl`。
判定は `results/explore_relation/e56_targets.csv`・`e56_tests.csv`
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
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from source_trust import keep_mask, trust_of  # noqa: E402
from teach_student39 import Student, solve_students  # noqa: E402

RS = (0, 10, 25, 50, 75, 90)
META_KS = ("1", "5")
PSEUDO_REPS = [0, 1, 2, 3, 4]
C_PSEUDO = 0.01
KINDS = ["audio_wavlm", "video", "text_roberta"]
OUT_REL = ROOT / "results" / "explore_relation"


def settings(kind, T):
    f = "e39_text_targets.csv" if kind == "text_roberta" else "e39_targets.csv"
    s = pd.read_csv(OUT_REL / f, dtype={"K": str}).query("kind == @kind").drop_duplicates("target").set_index("target").loc[T]
    return float(s.gamma), float(s.rho_s), float(s.rho_M), float(s.cb), float(s.cp)


def pool_info_kept(data, members, C, r, rho_s, P, Y, cache, rand_key=None):
    """`meta5.pool_info` と同じ計算を，各群の信頼度の下位 r％ を外した残りの発話だけで行う（信頼度は members の中で測る）。"""
    key = (tuple(sorted(members)), C, r, rho_s, rand_key)
    if key in cache:
        return cache[key]
    keep = {s: np.flatnonzero(keep_mask(s, r, trust_of(s, members, P, Y), Y, rand_key)) for s in members}
    rows = lambda s: data[s].X[torch.as_tensor(keep[s], device=data[s].X.device)]  # noqa: E731
    Xp = torch.cat([rows(s) for s in members])
    yp = M.labels(np.concatenate([np.asarray(Y[s])[keep[s]] for s in members]))
    src_pc = logreg.fit(Xp - Xp.mean(0), yp, (C,))
    mu_p, sd_p = Xp.mean(0), Xp.std(0, unbiased=False)
    sd_p = torch.where(sd_p > 1e-12, sd_p, torch.ones_like(sd_p))
    del Xp
    lin = {}
    for s in members:
        Z = (rows(s) - mu_p) / sd_p
        yt = torch.as_tensor(np.asarray(Y[s])[keep[s]], device=Z.device)
        m = torch.stack([Z[yt == k].mean(0) for k in range(4)])
        R = Z - m[yt]
        lam, V = torch.linalg.eigh(R.T @ R / (len(yt) - 4))
        lam = lam.clamp(min=0.0)
        mt = m - Z.mean(0)
        Pm = V.T @ mt.T
        W = (V @ (Pm / ((1 - rho_s) * lam + rho_s * lam.mean())[:, None])).T
        lin[(s, rho_s)] = (W / sd_p, -0.5 * (mt * W).sum(1))
        del Z, R, V
    out = (src_pc, lin, {s: len(keep[s]) for s in members})
    cache[key] = out
    return out


def eval_students(name, ev, members, C, info, st_set, reps, ks, rows, tag):
    """E31 の K 本で，info（pool_info_kept の戻り値）から作った E39 の生徒を解き，E39 が選んだ (ρ_M, C_b, C′) の行だけ rows に足す。"""
    gamma, rho_s, rho_M, cb, cp = st_set
    src_pc, lin, n_of = info
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    students = {}
    for r_ in reps:
        pool = np.flatnonzero(~ctx.te[r_])
        Xc = M.rows(ev.X, pool) - ctx.ct[r_]
        students[r_] = Student(ev, Xc, src_pc, lin, n_of, members, C, gamma, rho_s, [rho_M])
    out = []
    solve_students(ev.name, ev, ctx, reps, ks, students, {name: ctx.idx}, [rho_M], C, out)
    for x in out:
        if x["cb"] == f"{cb:g}" and abs(x["cp"] - cp) < 1e-12:
            rows.append({**x, **tag})


def run_target(kind, T, data, P, Y, reps, ks, out, cache):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    st_set = settings(kind, T)
    rho_s = st_set[1]
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    prows = []
    if kind != "text_roberta":                                                   # 仮のターゲットで r を選ぶ（T は測れないので行わない）
        for c in S:
            Sc = [s for s in S if CM.corp(s) != CM.corp(c)]
            for r in RS:
                eval_students("trust", data[c], Sc, C_PSEUDO, pool_info_kept(data, Sc, C_PSEUDO, r, rho_s, P, Y, cache), st_set,
                              PSEUDO_REPS, list(META_KS), prows, {"target": T, "pseudo": c, "r": r})
        pdf = pd.DataFrame(prows)
        r_star = int(pdf.groupby("r").uar.mean().idxmax())
        pdf.to_csv(out / f"{T}_pseudo.csv", index=False)
    else:
        r_star = -1
    rows = []
    for r in RS:
        eval_students("trust", data[T], S, C_T, pool_info_kept(data, S, C_T, r, rho_s, P, Y, cache), st_set, reps, ks, rows,
                      {"r": r, "r_star": r_star})
        if r > 0:
            eval_students("random", data[T], S, C_T, pool_info_kept(data, S, C_T, r, rho_s, P, Y, cache, rand_key=T), st_set, reps, ks,
                          rows, {"r": r, "r_star": r_star})
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=KINDS) -> int:
    e55 = pd.read_csv(OUT_REL / "e55_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    tabs = []
    for kind in kinds:
        f39 = "e39_text_targets.csv" if kind == "text_roberta" else "e39_targets.csv"
        e39 = pd.read_csv(OUT_REL / f39, dtype={"K": str}).query("kind == @kind").set_index(["target", "K"])
        d = ROOT / "results" / f"source_trust39_{kind}"
        for x in sorted(f for f in d.glob("*.csv") if not f.stem.endswith("_pseudo")):
            T = x.stem
            a = pd.read_csv(x, dtype={"K": str})
            a = a[a.rep.isin(EC.REPS)]
            r_star = int(a.r_star.iloc[0])
            u = a.groupby(["policy", "r", "K"]).uar.mean() * 100
            for K in sorted(a.K.unique(), key=int):
                row = {"kind": kind, "target": T, "K": K, "r_star": r_star, "all": float(u.loc[("trust", 0, K)]),
                       "oracle": float(max(u.loc[("trust", r, K)] for r in RS)),
                       "chk_e39": abs(float(u.loc[("trust", 0, K)]) - float(e39.loc[(T, K), "meta"])),
                       "P4": float(e55.loc[(kind, T, K), "all"]), "P4_trust": float(e55.loc[(kind, T, K), "meta"]),
                       "P4_oracle": float(e55.loc[(kind, T, K), "oracle"])}
                if r_star >= 0:
                    row["meta"] = float(u.loc[("trust", r_star, K)])
                    row["rand_at_star"] = float(u.loc[("random", r_star, K)]) if r_star > 0 else row["all"]
                for r in RS:
                    row[f"trust_r{r}"] = float(u.loc[("trust", r, K)])
                    if r > 0:
                        row[f"rand_r{r}"] = float(u.loc[("random", r, K)])
                tabs.append(row)
    D = pd.DataFrame(tabs)
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        row = {"kind": kind, "K": K, "targets": len(g), "chk_e39": float(g.chk_e39.max())}
        cols = ["all", "oracle", "P4", "P4_trust", "P4_oracle"] + [f"trust_r{r}" for r in RS] + [f"rand_r{r}" for r in RS if r > 0]
        if "meta" in g and g.meta.notna().all():
            cols += ["meta", "rand_at_star"]
        for x in cols:
            row[x] = float(g[x].mean())
        pairs = [("oracle", "all"), ("all", "P4"), ("P4_trust", "P4")] + [(f"trust_r{r}", "all") for r in RS if r > 0] + \
                [(f"trust_r{r}", f"rand_r{r}") for r in RS if r > 0]
        if "meta" in cols:
            pairs = [("meta", "all"), ("meta", "rand_at_star"), ("meta", "P4_trust"), ("meta", "P4")] + pairs
        for a_, b_ in pairs:
            diff = g[a_] - g[b_]
            row[f"{a_}-{b_}"] = float(diff.mean())
            row[f"{a_}>{b_}"] = int((diff > 0).sum())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in ("audio_wavlm", "video"):
        k1, k5 = R[(R.kind == kind) & (R.K == "1")], R[(R.kind == kind) & (R.K == "5")]
        if len(k1) and len(k5) and "meta-all" in R:
            k1, k5 = k1.iloc[0], k5.iloc[0]
            R.loc[R.kind == kind, "SW1"] = bool(k1["meta-all"] > 0 and k5["meta-all"] > 0)
            R.loc[R.kind == kind, "SW2"] = bool(k1["meta-rand_at_star"] > 0 and k5["meta-rand_at_star"] > 0)
            R.loc[R.kind == kind, "SW3"] = bool(k5["meta-P4_trust"] > 0)
    D.to_csv(OUT_REL / "e56_targets.csv", index=False)
    R.to_csv(OUT_REL / "e56_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in ("SW1", "SW2", "SW3") if c in R]).set_index(["kind", "K"]).round(2).T.to_string())
    print("\n== 選ばれた r（ターゲットの数．T は選ばない）")
    print(D.drop_duplicates(["kind", "target"]).groupby("kind").r_star.value_counts().unstack(fill_value=0).to_string())
    vc = [c for c in ("SW1", "SW2", "SW3") if c in R]
    if vc:
        print(R.dropna(subset=vc).drop_duplicates("kind")[["kind"] + vc].to_string(index=False))
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
    out = args.out_dir or ROOT / "results" / f"source_trust39_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    f, Cs, P, Y = CM.prepare(args.kind, data)
    prov = {"script": "exp/source_trust39.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
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
