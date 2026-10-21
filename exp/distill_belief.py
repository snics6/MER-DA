"""【探索的】E59（GPU）：見込みの蒸留——ソースの票の見込みを，選ぶためでなく候補全体の弱いラベルとして生徒に流し込む（生徒は E39 の学習器）。

規則は `docs/exploratory-analyses.md` E59（設計と予言は計算の前に書いた）。E58 の確かめ 3 を，反復 0〜9・3 種類・仮のターゲットでつまみを選ぶ形で測り直す。
候補（テスト以外）の全発話に，見込み B の確率を重みにした 4 つのラベルを付ける（感情ごとの重みの総量をそろえ，全体の総重みを m_eff 本ぶんにする）．
付いた本当のラベル（E31 の K 本．感情の重みは選んだ中にある感情だけ）と合わせて，E39 の出発点から学び直す（罰則・座標は E39 の設定）。
  見込み B：q（E44 の C），pq（{出発点 p, q} の積み上げ．E52），p（出発点の予測そのもの．対照：感情の釣り合いを取り直す効果だけ）
  m_eff ∈ {20, 100, 500}．(B ∈ {q, pq}, m_eff) は，T のソースを 1 つずつ仮のターゲットにして選ぶ（K ∈ {1, 5}，反復 0〜4 の UAR の平均．「蒸留しない」も候補に入れる）
  K ∈ {0, 1, 5, 10, 25}（K=0 はラベル無し）

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/distill_belief.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/distill_belief.py --analyze

出力：`results/distill_belief_<kind>/<ターゲット>.csv`（belief, meff, rep, K, uar），`<ターゲット>_pseudo.csv`，`<ターゲット>_meta.csv`，`provenance.json`，`provenance_log.jsonl`。
判定は `results/explore_relation/e59_targets.csv`・`e59_tests.csv`
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
import meta5 as M5  # noqa: E402
from data import groups_for  # noqa: E402
from geom_e0 import settings  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import present_weights  # noqa: E402
from teach_stack import c_pool_on, draft_logits, fit_stack, logp, softmax_t  # noqa: E402
from teach_student39 import Student  # noqa: E402
from text_cond2 import choose_lam  # noqa: E402

KINDS = ["audio_wavlm", "video", "text_roberta"]
BELIEFS = ["p", "q", "pq"]
MEFFS = (20, 100, 500)
KS = ("0", "1", "5", "10", "25")
META_KS = ("1", "5")
PSEUDO_REPS = [0, 1, 2, 3, 4]
C_PSEUDO = 0.01
CANDS = [("none", 0)] + [(b, m) for b in ("q", "pq") for m in MEFFS]
OUT_REL = ROOT / "results" / "explore_relation"


def weak_rows(B, meff):
    """見込み B（n×4）から，4 つのラベルの行の重み（長さ 4n．行 k·n + i が発話 i の感情 k）。感情ごとの総量をそろえ，全体を meff にする。"""
    Bt = torch.as_tensor(B, dtype=logreg.DTYPE, device=logreg.DEVICE)
    w = (Bt / Bt.sum(0).clamp(min=1e-6)).T.reshape(-1)
    return w / w.sum() * meff


def solve_all(Z, yq, beliefs, label_sets, combos, st, rho_m, C, cp, cb):
    """1 回の束ねた解きで，(見込み, m_eff, ラベルの組) の全部の問題を解く。戻りは {(belief, meff, key): (W, b)}。"""
    n, d = Z.shape
    dev = Z.device
    X = torch.cat([Z.repeat(4, 1), Z])
    y = torch.cat([torch.arange(4, device=dev).repeat_interleave(n), M.labels(yq)])
    yt = M.labels(yq)
    weights, keys = [], []
    for key, idx in label_sets.items():
        wt = torch.zeros(n, dtype=logreg.DTYPE, device=dev)
        if len(idx):
            it = torch.as_tensor(np.asarray(idx), device=dev)
            wt[it] = present_weights(yt[it])
        for b, meff in combos:
            w4 = weak_rows(beliefs[b], meff) if meff > 0 else torch.zeros(4 * n, dtype=logreg.DTYPE, device=dev)
            w = torch.cat([w4, wt])
            if float(w.sum()) == 0.0:
                continue                                                      # ラベル無しで蒸留もしない問題は出発点そのもの
            weights.append(w)
            keys.append((b, meff, key))
    zero = torch.zeros(d, dtype=Z.dtype, device=dev)
    fits = logreg.fit_batch(X, y, [np.ones(len(y), bool)] * len(weights), (cp,), anchor=st.anchor(rho_m), anchor_c=C,
                            stats=(zero, torch.ones_like(zero)), Cb=[cb] * len(weights), weights=weights)
    return {k: (f.W[cp], f.b[cp]) for k, f in zip(keys, fits)}


def uar_of(W, b, Zte, yte):
    return CB.uar(yte, (Zte @ W.T + b).argmax(1).cpu().numpy())


def run_target(kind, T, data, f, Cs, P, Y, reps, out, cache):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    gamma, rho_s, rho_m, cb, cp, _ = settings(kind, T)
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    if kind == "text_roberta":
        lam44, _ = choose_lam(S, P, Y)
    else:
        lam44 = float(pd.read_csv(ROOT / "results" / f"committee_{kind}" / f"{T}.csv").query("variant == 'C'").lam.iloc[0])
    pool_q = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), lam44)
    mem = {g: {"p": softmax_t(draft_logits(data, S, g, gamma, rho_s, cache)), "q": c_pool_on(g, S, P, Y, lam44)} for g in S}
    pool_pq, lam_pq = fit_stack(T, "pq", S, mem, Y, [])
    prows = []
    for c in S:                                                                  # 仮のターゲットで (見込み, m_eff) を選ぶ
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c:
            continue
        Gc = [g for g in S if g != c]
        pool_pq_c, _ = fit_stack(T, "pq", Gc, mem, Y, []) if len(Gc) >= 2 else (None, None)
        src_c, lin_c, n_c = M5.pool_info(data, pool_c, C_PSEUDO, cache)
        evc = data[c]
        ctxc = CB.Ctx(evc, PSEUDO_REPS, list(META_KS), center="ind")
        for r in PSEUDO_REPS:
            pool = np.flatnonzero(~ctxc.te[r])
            yq = evc.y[pool]
            Xc = M.rows(evc.X, pool) - ctxc.ct[r]
            stc = Student(evc, Xc, src_c, lin_c, n_c, pool_c, C_PSEUDO, gamma, rho_s, [rho_m])
            Am = stc.mats[rho_m][0]
            Z = stc.Xstd @ Am
            te = np.flatnonzero(ctxc.te[r])
            Zte = ((M.rows(evc.X, te) - ctxc.ct[r] - stc.src.mu) / stc.src.sd) @ Am
            bel = {"p": mem[c]["p"][pool], "q": mem[c]["q"][pool]}
            if pool_pq_c is not None:
                F = torch.as_tensor(np.stack([logp(bel["p"]), logp(bel["q"])], 1), dtype=logreg.DTYPE, device=logreg.DEVICE)
                bel["pq"] = pool_pq_c.predict(F, torch.ones(len(pool), 2, dtype=logreg.DTYPE, device=logreg.DEVICE))
            pos_of = {int(i): j for j, i in enumerate(pool)}
            label_sets = {K: np.array([pos_of[int(i)] for i in ctxc.idx[(r, K)]]) for K in META_KS}
            combos = [cd for cd in CANDS if cd[0] in bel or cd[0] == "none"]
            sol = solve_all(Z, yq, bel, label_sets, [("q" if b == "none" else b, m) for b, m in combos], stc, rho_m, C_PSEUDO, cp, cb)
            for (b, m, K), (W, bb) in sol.items():
                name = "none" if m == 0 else b
                prows.append({"target": T, "pseudo": c, "belief": name, "meff": m, "rep": r, "K": K, "uar": uar_of(W, bb, Zte, evc.y[te])})
    pdf = pd.DataFrame(prows)
    sc = pdf.groupby(["belief", "meff"]).uar.mean()
    sc = sc[[k for k in sc.index if (k[0] == "none") or (k[0] in ("q", "pq"))]]
    b_star, m_star = sc.idxmax()
    pdf.to_csv(out / f"{T}_pseudo.csv", index=False)
    pd.DataFrame([{"target": T, "belief": b, "meff": m, "uar_pseudo": u, "chosen": (b, m) == (b_star, m_star)} for (b, m), u in sc.items()]
                 ).to_csv(out / f"{T}_meta.csv", index=False)
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ["1", "5", "10", "25"], center="ind")
    rows = []
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, gamma, rho_s, [rho_m])
        Am, Mh = st.mats[rho_m]
        Z = st.Xstd @ Am
        te = np.flatnonzero(ctx.te[r])
        yte = ev.y[te]
        Zte = ((M.rows(ev.X, te) - ctx.ct[r] - st.src.mu) / st.src.sd) @ Am
        Ps = [softmax_t(f[s].logits(Xc, Cs[s])) for s in S]
        tens = lambda cc: (torch.as_tensor(np.stack(cc, 1), dtype=logreg.DTYPE, device=logreg.DEVICE),  # noqa: E731
                           torch.ones(len(yq), len(cc), dtype=logreg.DTYPE, device=logreg.DEVICE))
        p = softmax_t(st.L0)
        q = pool_q.predict(*tens([logp(x) for x in Ps]))
        bel = {"p": p, "q": q, "pq": pool_pq.predict(*tens([logp(p), logp(q)]))}
        pos_of = {int(i): j for j, i in enumerate(pool)}
        label_sets = {"0": np.array([], int), **{K: np.array([pos_of[int(i)] for i in ctx.idx[(r, K)]]) for K in ("1", "5", "10", "25")}}
        combos = [("q", 0)] + [(b, m) for b in BELIEFS for m in MEFFS]
        sol = solve_all(Z, yq, bel, label_sets, combos, st, rho_m, C_T, cp, cb)
        rows.append({"target": T, "rep": r, "belief": "none", "meff": 0, "K": "0", "uar": uar_of(st.W_std @ Mh, st.b_std, Zte, yte)})
        for (b, m, K), (W, bb) in sol.items():
            rows.append({"target": T, "rep": r, "belief": "none" if m == 0 else b, "meff": m, "K": K, "uar": uar_of(W, bb, Zte, yte)})
        for b, u in (("acc_p", p), ("acc_q", q), ("acc_pq", bel["pq"])):
            rows.append({"target": T, "rep": r, "belief": b, "meff": -1, "K": "-", "uar": CB.uar(yq, u.argmax(1))})
    df = pd.DataFrame(rows)
    df["b_star"], df["m_star"] = b_star, m_star
    df.to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=KINDS) -> int:
    tabs = []
    for kind in kinds:
        d = ROOT / "results" / f"distill_belief_{kind}"
        for x in sorted(f_ for f_ in d.glob("*.csv") if not f_.stem.endswith(("_pseudo", "_meta"))):
            T = x.stem
            a = pd.read_csv(x, dtype={"K": str})
            a = a[a.rep.isin(EC.REPS)]
            b_star, m_star = a.b_star.iloc[0], int(a.m_star.iloc[0])
            u = a[a.meff >= 0].groupby(["belief", "meff", "K"]).uar.mean() * 100
            acc = a[a.meff == -1].groupby("belief").uar.mean() * 100
            for K in KS:
                row = {"kind": kind, "target": T, "K": K, "b_star": b_star, "m_star": m_star, "e39": float(u.loc[("none", 0, K)]),
                       "meta": float(u.loc[("none", 0, K)] if m_star == 0 else u.loc[(b_star, m_star, K)]),
                       "p_ctrl": float(u.loc[("p", m_star if m_star > 0 else 100, K)]),
                       "belief_uar_p": float(acc["acc_p"]), "belief_uar_q": float(acc["acc_q"]), "belief_uar_pq": float(acc["acc_pq"])}
                for b in BELIEFS:
                    for m in MEFFS:
                        row[f"{b}{m}"] = float(u.loc[(b, m, K)])
                tabs.append(row)
    D = pd.DataFrame(tabs)
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        row = {"kind": kind, "K": K, "targets": len(g), "e39": float(g.e39.mean()), "meta": float(g.meta.mean()), "p_ctrl": float(g.p_ctrl.mean()),
               "meta-e39": float((g.meta - g.e39).mean()), "meta>e39": int((g.meta > g.e39).sum()),
               "p_ctrl-e39": float((g.p_ctrl - g.e39).mean()), "meta-p_ctrl": float((g.meta - g.p_ctrl).mean()),
               "oracle-e39": float((g[[f"{b}{m}" for b in ("q", "pq") for m in MEFFS]].max(1) - g.e39).mean())}
        for b in BELIEFS:
            for m in MEFFS:
                row[f"{b}{m}-e39"] = float((g[f"{b}{m}"] - g.e39).mean())
        for b in ("p", "q", "pq"):
            row[f"belief_uar_{b}"] = float(g[f"belief_uar_{b}"].mean())
        tests.append(row)
    R = pd.DataFrame(tests)
    r = R.set_index(["kind", "K"])
    for kind in kinds:
        if (kind, "5") in r.index:
            R.loc[R.kind == kind, "DS1"] = bool(r.loc[(kind, "1"), "meta-e39"] > 0 and r.loc[(kind, "5"), "meta-e39"] > 0)
            R.loc[R.kind == kind, "DS2"] = bool(r.loc[(kind, "0"), "q100-e39"] >= 0.5)
            R.loc[R.kind == kind, "DS3"] = bool(r.loc[(kind, "5"), "meta-e39"] < 0.5)
    D.to_csv(OUT_REL / "e59_targets.csv", index=False)
    R.to_csv(OUT_REL / "e59_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in ("DS1", "DS2", "DS3") if c in R]).set_index(["kind", "K"]).round(2).T.to_string())
    print("\n== 選ばれた (見込み, m_eff)")
    print(D.drop_duplicates(["kind", "target"]).groupby("kind").apply(lambda g: g.apply(lambda x: f"{x.b_star}{x.m_star}", axis=1).value_counts().to_dict()).to_string())
    print("\n== K=5 のターゲットごと（meta − e39，p_ctrl − e39）")
    k5 = D[D.K == "5"].set_index(["kind", "target"])
    print(pd.DataFrame({"meta-e39": k5.meta - k5.e39, "p_ctrl-e39": k5.p_ctrl - k5.e39, "q100-e39": k5.q100 - k5.e39}).round(2).to_string())
    vc = [c for c in ("DS1", "DS2", "DS3") if c in R]
    print(R.drop_duplicates("kind")[["kind"] + vc].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds)
    out = args.out_dir or ROOT / "results" / f"distill_belief_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    f, Cs, P, Y = CM.prepare(args.kind, data)
    prov = {"script": "exp/distill_belief.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "reps": args.reps, "beliefs": BELIEFS, "meffs": MEFFS, "ks": KS, "meta_ks": META_KS,
            "pseudo_reps": PSEUDO_REPS, "c_pseudo": C_PSEUDO, "src_C": Cs, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, f, Cs, P, Y, args.reps, out, cache)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
