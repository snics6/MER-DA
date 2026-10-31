"""【探索的】E35 の追加の段 2（GPU）：縮めた閉じた形の向き W_s(ρ) を等しい重みで足した下書きからの仮説転移。

規則は `docs/exploratory-analyses.md` E35 の追加の段 2（予言 Q35-5・Q35-6 は計算の前に書いた）。テスト・K 本・仮説転移・尺度のそろえ方・
等しい重みは E32 の 5 節の段 2（`combine_drafts.py`）と同じで，ソースごとの分類器だけを取り替える：P4 のソースの全発話の平均と標準偏差で
標準化した座標で W_s(ρ) = ((1 − ρ)Σ_s + ρ τ_s I)⁻¹ m̃_sᵀ，切片 b_k = −½ m̃_kᵀ W_k（m̃_s はソースの全体の平均を引いた感情の平均）。
ρ はターゲットごとに `results/explore_relation/e35s_targets.csv` の rho_star（star）と rho_best（best．テストのラベルで選んだ上限）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/combine_shrink.py --kind audio_wavlm [--target …] [--reps …] [--out-dir …]
    ./.venv/bin/python exp/combine_shrink.py --analyze        # 判定（CPU．combine_<kind>/ と compare_<kind>/ が要る）

出力：`results/combine_shrink_<kind>/<ターゲット>.csv`（method=HT_SHR, source=star|best, rho, cen, cp, cb, rep, K, uar），`.npz`，
`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e35g_targets.csv`・`e35g_tests.csv`
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
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import combine_drafts as CD  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402

RHO_TAB = ROOT / "results" / "explore_relation" / "e35s_targets.csv"
OUT_REL = ROOT / "results" / "explore_relation"
DRAFTS = ("star", "best")
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]


def shrunk(Z: torch.Tensor, y: np.ndarray, rho: float) -> tuple[torch.Tensor, torch.Tensor]:
    """標準化した座標の (W (4, d), b (4,))。E35 の追加の `explore_draft_shrink.shrunk_dirs` と同じ向き。"""
    yt = torch.as_tensor(np.array(y), device=Z.device)
    m = torch.stack([Z[yt == k].mean(0) for k in range(4)])
    R = Z - m[yt]
    lam, V = torch.linalg.eigh(R.T @ R / (len(y) - 4))
    lam = lam.clamp(min=0.0)
    mt = m - Z.mean(0)
    W = (V @ ((V.T @ mt.T) / ((1 - rho) * lam + rho * lam.mean())[:, None])).T
    return W, -0.5 * (mt * W).sum(1)


def run_target(kind, tgt, data, reps, ks, out, rhos: dict):
    ctx = CB.Ctx(tgt, reps, ks, center="ind")
    pool = [s for s in data if s != tgt.name and GROUP_SPEC[s][0] != GROUP_SPEC[tgt.name][0]]
    zf = ROOT / "results" / f"multisource_scores_{kind}" / f"{tgt.name}.npz"
    if zf.exists() and [str(s) for s in np.load(zf)["src_names"]] != pool:     # Pod には npz を置かない（研究室PCで全ターゲットを確かめた）
        raise ValueError(f"ソースが E35 と合わない: {pool} 対 {list(np.load(zf)['src_names'])}")
    rows, saved = [], {}
    for r in reps:
        saved[f"test_idx_r{r}"] = np.flatnonzero(ctx.te[r])
        saved[f"y_test_r{r}"] = tgt.y[ctx.te[r]]
    t0 = time.perf_counter()
    Xp = torch.cat([data[s].X for s in pool])
    yp = M.labels(np.concatenate([data[s].y for s in pool]))
    C_p = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{tgt.name}.csv", dtype={"K": str}).C.iloc[0])
    src_pc = logreg.fit(Xp - Xp.mean(0), yp, (C_p,))                    # P4 の下書き（尺度と標準化の基準．combine_drafts と同じ）
    mu_p, sd_p = Xp.mean(0), Xp.std(0, unbiased=False)                  # E35 の追加の座標（ソースの全発話の平均と標準偏差）
    sd_p = torch.where(sd_p > 1e-12, sd_p, torch.ones_like(sd_p))
    del Xp
    lin = {}                                                            # (draft, ソース) → 中心合わせの座標の 1 次関数 (A, a0)
    for s in pool:
        Zs = (data[s].X - mu_p) / sd_p
        for draft, rho in rhos.items():
            W, b = shrunk(Zs, data[s].y, rho)
            lin[(draft, s)] = (W / sd_p, b)
        del Zs
    t_fit = time.perf_counter() - t0

    def adder(draft):
        def add(r, K, c, u, cv, L, cen=0, cb=""):
            rows.append({"target": tgt.name, "source": draft, "rho": rhos[draft], "method": "HT_SHR", "cen": cen, "cp": c,
                         "cb": cb, "rep": r, "K": K, "uar": u})
            saved[f"{draft}_K{K}_r{r}"] = L.astype(np.float32)
        return add

    t1 = time.perf_counter()
    for r in reps:
        Xu = M.rows(tgt.X, np.flatnonzero(~ctx.te[r])) - ctx.ct[r]     # テスト以外の発話（ラベルは使わない）
        ref_scale = CD.pair_scale(src_pc.logits(Xu, C_p))
        for draft in rhos:
            comps = {}
            for s in pool:
                A, a0 = lin[(draft, s)]
                sc = CD.pair_scale(Xu @ A.T + a0)
                comps[s] = (A / sc, a0 / sc)
            A = sum(comps[s][0] for s in pool) / len(pool)
            a0 = sum(comps[s][1] for s in pool) / len(pool)
            g = ref_scale / CD.pair_scale(Xu @ A.T + a0)
            A, a0 = g * A, g * a0
            mu, sd = src_pc.mu, src_pc.sd
            anchor = logreg.Fitted(mu, sd, {C_p: A * sd}, {C_p: a0 + mu @ A.T}, {C_p: 0}, {C_p: 0.0})
            CB.ht_block(ctx, [r], anchor, C_p, 1, adder(draft), cps=(CB.DEF_CP,), cbs=(CB.DEF_CB,), with_cv=False)
    print(f"  {kind} {tgt.name}：下書き {t_fit:.1f} 秒，仮説転移 {time.perf_counter() - t1:.1f} 秒（ソース {len(pool)}，"
          + "，".join(f"{d} ρ={v}" for d, v in rhos.items()) + "）", flush=True)
    np.savez_compressed(out / f"{tgt.name}.npz", **saved)
    pd.DataFrame(rows).to_csv(out / f"{tgt.name}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze() -> int:
    out = []
    for kind in MAIN:
        fs = sorted((ROOT / "results" / f"combine_shrink_{kind}").glob("*.csv"))
        if not fs:
            continue
        a = pd.concat([pd.read_csv(f, dtype={"K": str}) for f in fs])
        a = a[a.rep.isin(EC.REPS)]
        T = a.pivot_table(index=["target", "K"], columns="source", values="uar", aggfunc="mean") * 100
        e = pd.concat([pd.read_csv(f, dtype={"K": str}) for f in sorted((ROOT / "results" / f"combine_{kind}").glob("*.csv"))])
        T["equal"] = e[(e.source == "equal") & e.rep.isin(a.rep.unique())].groupby(["target", "K"]).uar.mean() * 100
        c = pd.concat([pd.read_csv(ROOT / "results" / f"compare_{kind}" / f"{t}.csv", dtype={"K": str, "cb": str})
                       for t in T.index.get_level_values("target").unique()])
        p4 = c[(c.method == "HT_ALL") & (c.cen == 1) & (c.cp == 0.01) & (c.cb == "8") & c.rep.isin(a.rep.unique())]
        T["P4"] = p4.groupby(["target", "K"]).uar.mean() * 100
        T = T.reset_index()
        T.insert(0, "kind", kind)
        out.append(T)
    T = pd.concat(out, ignore_index=True)
    rows = []
    for (kind, K), g in T.groupby(["kind", "K"]):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for a_, b_ in (("star", "equal"), ("star", "P4"), ("best", "equal"), ("equal", "P4")):
            diff = g[a_] - g[b_]
            row[f"{a_}-{b_}"] = float(diff.mean())
            row[f"{a_}-{b_}_wins"] = int((diff > 0).sum())
            row[f"{a_}-{b_}_p"] = float(stats.wilcoxon(diff, alternative="greater").pvalue) if diff.abs().sum() > 0 else np.nan
        rows.append(row)
    R = pd.DataFrame(rows)
    for kind in ("audio_wavlm", "audio_hubert"):
        m = (R.kind == kind) & R.K.isin(["5", "10"])
        R.loc[R.kind == kind, "Q35-5"] = bool((R[m]["star-equal"] >= -0.5).all()) if m.sum() == 2 else np.nan
        R.loc[R.kind == kind, "Q35-6"] = bool((R[m]["star-P4"] > 0).all()) if m.sum() == 2 else np.nan
    T.to_csv(OUT_REL / "e35g_targets.csv", index=False)
    R.to_csv(OUT_REL / "e35g_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
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
    out = args.out_dir or ROOT / "results" / f"combine_shrink_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    rt = pd.read_csv(RHO_TAB).query("kind == @args.kind").set_index("target")
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if t in rt.index and not (out / f"{t}.npz").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/combine_shrink.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "cp": CB.DEF_CP, "cb": CB.DEF_CB,
            "rho_table": str(RHO_TAB.relative_to(ROOT)), "device": str(CB.DEV),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, data[t], data, args.reps, args.ks, out,
                   {"star": float(rt.loc[t, "rho_star"]), "best": float(rt.loc[t, "rho_best"])})
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
