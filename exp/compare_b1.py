"""E31 段 1：論文 2 の比較実験（GPU）。設計は `docs/researchvision.md` 3.8 節，実装の細部は `docs/exploratory-analyses.md` E31。

手法：P1〜P3（1 ソースの仮説転移．中心合わせ 2 × 罰則 C′ 4 × 切片 3），P4（全ソースの下書きの仮説転移．同じ 24 設定），
H3（帰納の中心合わせ＋合併），O1（重みつき合併．α 5 通り），O6（最近傍セントロイド．3 通り），O7（メタ初期値）。
P1〜P4 と O1 では K 本の交差検証の UAR も残す。テストと K 本は地図と同じ。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/compare_b1.py --kind audio_wavlm [--target …] [--reps …] [--ks …]

出力：`results/compare_<kind>/<ターゲット>.csv`（method, source, cen, cp, cb, alpha, variant, rep, K, uar, cv_uar），
`<ターゲット>.npz`（主な設定のテストの点数），`provenance.json`。終わったターゲットは飛ばす。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import logreg  # noqa: E402
import map as M  # noqa: E402
import shift  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from interventions import map_rows  # noqa: E402
from pair_table import map_dir  # noqa: E402
from split import speaker_folds  # noqa: E402

KS = ("1", "5", "10", "25")
CPS = (0.001, 0.01, 0.1, 1.0)
CBS = (2.0, 8.0, None)
ALPHAS = (0.1, 0.25, 0.5, 0.75, 0.9)
DEF_CP, DEF_CB = 0.01, 8.0
O7_STEPS_IN, O7_LR_IN, O7_ITERS, O7_LR_OUT, O7_BATCH, O7_QUERY = 10, 0.5, 300, 1e-2, 4, 256
DEV = logreg.DEVICE


def cbname(cb):
    return "free" if cb is None else f"{cb:g}"


def uar(y, pred):
    return shift.uar(np.asarray(y), np.asarray(pred))


class Ctx:
    """ターゲットごとの分割（テスト，K 本，交差検証の袋，帰納の中心）。"""

    def __init__(self, tgt: M.Group, reps, ks):
        self.tgt, self.reps, self.ks = tgt, list(reps), list(ks)
        self.te, self.idx, self.persons, self.ct, self.cvf = {}, {}, {}, {}, {}
        for r in self.reps:
            te = M.test_mask(tgt, r)
            order = M.kshot_order(tgt, np.flatnonzero(~te), r)
            self.te[r], self.persons[r] = te, np.unique(tgt.person[te])
            self.ct[r] = M.rows(tgt.X, np.flatnonzero(~te)).mean(0)
            for K in self.ks:
                idx = M.take_k(order, K)
                self.idx[(r, K)] = idx
                f = speaker_folds(pd.Series(tgt.person[idx]), k=3, seed=r).to_numpy()
                ok = all(len(np.unique(tgt.y[idx][f != j])) == 4 for j in np.unique(f)) and len(np.unique(f)) >= 2
                self.cvf[(r, K)] = f if ok else None


def jobs_for(ctx: Ctx, rs, used_pos, n_total, with_cv=True):
    """（r, K, 袋 j または None）と，仮面（長さ n_total．used_pos：ターゲットの行番号 → 行列の中の位置）。"""
    jobs, masks = [], []
    for r in rs:
        for K in ctx.ks:
            idx = ctx.idx[(r, K)]
            m = np.zeros(n_total, bool)
            m[[used_pos[int(i)] for i in idx]] = True
            jobs.append((r, K, None)); masks.append(m)
            f = ctx.cvf[(r, K)]
            if with_cv and f is not None:
                for j in np.unique(f):
                    mj = np.zeros(n_total, bool)
                    mj[[used_pos[int(i)] for i in idx[f != j]]] = True
                    jobs.append((r, K, int(j))); masks.append(mj)
    return jobs, masks


def collect(ctx, jobs, fits, cs, X_of, rowsel, emit, extra):
    """テストの UAR と，交差検証の予測をまとめた UAR を emit(r, K, c, uar, cv_uar, logits) に渡す。"""
    tgt = ctx.tgt
    cvpred = defaultdict(dict)
    for (r, K, j), f in zip(jobs, fits):
        if j is None:
            continue
        idx = ctx.idx[(r, K)]
        hold = idx[ctx.cvf[(r, K)] == j]
        Xh = X_of(r, hold)
        for c in cs:
            cvpred[(r, K, c)].update(dict(zip(hold.tolist(), f.predict(Xh, c))))
    for (r, K, j), f in zip(jobs, fits):
        if j is not None:
            continue
        Xte = X_of(r, np.flatnonzero(ctx.te[r]))
        yte = tgt.y[ctx.te[r]]
        for c in cs:
            L = f.logits(Xte, c).cpu().numpy()
            cv = np.nan
            if (r, K, c) in cvpred:
                idx = ctx.idx[(r, K)]
                pr = np.array([cvpred[(r, K, c)][int(i)] for i in idx])
                cv = uar(tgt.y[idx], pr)
            emit(r, K, c, uar(yte, L.argmax(1)), cv, L, **extra)


def ht_block(ctx, rs, src, C_s, cen: bool, add):
    """1 つの下書き（src．cen なら中心合わせ版）からの仮説転移．罰則 4 × 切片 3 の 12 設定（cen は固定）。"""
    tgt = ctx.tgt
    blocks, ys, jobs, masks = [], [], [], []
    if not cen:
        used = np.unique(np.concatenate([ctx.idx[(r, K)] for r in rs for K in ctx.ks]))
        blocks.append(M.rows(tgt.X, used)); ys.append(tgt.y[used])
        jobs, masks = jobs_for(ctx, rs, {int(i): k for k, i in enumerate(used)}, len(used))
    else:                                                     # 反復ごとに中心が違うので，反復ごとの行の塊を縦に並べる
        layout, off = [], 0
        for r in rs:
            used_r = np.unique(np.concatenate([ctx.idx[(r, K)] for K in ctx.ks]))
            blocks.append(M.rows(tgt.X, used_r) - ctx.ct[r]); ys.append(tgt.y[used_r])
            layout.append((r, used_r, off)); off += len(used_r)
        for r, used_r, o in layout:
            jj, mm = jobs_for(ctx, [r], {int(i): o + k for k, i in enumerate(used_r)}, off)
            jobs += jj; masks += mm
    X, yX = torch.cat(blocks), M.labels(np.concatenate(ys))
    X_of = (lambda r, ii: M.rows(tgt.X, ii) - ctx.ct[r]) if cen else (lambda r, ii: M.rows(tgt.X, ii))
    for cb in CBS:
        fits = logreg.fit_batch(X, yX, masks, CPS, anchor=src, anchor_c=C_s, stats=(src.mu, src.sd), Cb=cb)
        collect(ctx, jobs, fits, CPS, X_of, None, add, {"cen": int(cen), "cb": cbname(cb)})


def run_target(kind, tgt, data, ref_all, reps, ks, out):
    ctx = Ctx(tgt, reps, ks)
    rows, saved = [], {}
    for r in reps:
        saved[f"test_idx_r{r}"] = np.flatnonzero(ctx.te[r])
        saved[f"y_test_r{r}"] = tgt.y[ctx.te[r]]
    ref = map_rows(kind, tgt.name)
    nt = len(tgt.y)

    def adder(method, source):
        def add(r, K, c, u, cv, L, cen=0, cb="", alpha=np.nan, variant=""):
            rows.append({"target": tgt.name, "source": source, "method": method, "cen": cen, "cp": c, "cb": cb,
                         "alpha": alpha, "variant": variant, "rep": r, "K": K, "uar": u, "cv_uar": cv})
            key = None
            if method == "HT" and c == DEF_CP and ((cen == 1 and cb == f"{DEF_CB:g}") or cb == "free"):
                key = {(1, f"{DEF_CB:g}"): "p3", (0, "free"): "p1", (1, "free"): "p2"}.get((cen, cb))
            if method == "HT_ALL" and c == DEF_CP and cen == 1 and cb == f"{DEF_CB:g}":
                key = "p4"
            if method == "H3":
                key = "h3"
            if key:
                saved[f"{key}_{source}_K{K}_r{r}"] = L.astype(np.float32)
        return add

    srcs = [s for s in data if s != tgt.name]
    for sname in srcs:
        s = data[sname]
        t0 = time.perf_counter()
        groups = {}
        for r in reps:
            keep = ~np.isin(s.person, ctx.persons[r])
            C_s = ref[("src_only", sname, "", r)][0]
            groups.setdefault((keep.tobytes(), C_s), []).append(r)
        for (kb, C_s), rs in groups.items():
            keep = np.frombuffer(kb, dtype=bool)
            Xs, ys_np = M.rows(s.X, keep), s.y[keep]
            ys = M.labels(ys_np)
            cs = Xs.mean(0)
            src = logreg.fit(Xs, ys, (C_s,))
            src_c = logreg.fit(Xs - cs, ys, (C_s,))
            ht_block(ctx, rs, src, C_s, False, adder("HT", sname))
            ht_block(ctx, rs, src_c, C_s, True, adder("HT", sname))
            # ---- H3（帰納）：反復ごとに中心を引いたターゲットの K 本 ＋ 中心を引いたソース
            blocks, ylist, jobs, masks, off = [], [], [], [], 0
            ns = len(ys_np)
            for r in rs:
                used_r = np.unique(np.concatenate([ctx.idx[(r, K)] for K in ctx.ks]))
                blocks.append(M.rows(tgt.X, used_r) - ctx.ct[r]); ylist.append(tgt.y[used_r])
                p = {int(i): off + k for k, i in enumerate(used_r)}
                for K in ctx.ks:
                    jobs.append((r, K, None)); masks.append(np.array([p[int(i)] for i in ctx.idx[(r, K)]]))
                off += len(used_r)
            Xh = torch.cat(blocks + [Xs - cs])
            yh = M.labels(np.concatenate(ylist + [ys_np]))
            full = []
            for m in masks:
                f = np.zeros(len(yh), bool); f[off:] = True; f[m] = True; full.append(f)
            fits = logreg.fit_batch(Xh, yh, full, (C_s,))
            collect(ctx, jobs, fits, (C_s,), lambda r, ii: M.rows(tgt.X, ii) - ctx.ct[r], None, adder("H3", sname), {"cen": 1})
            # ---- O1：重みつき合併（中心合わせなし）
            used = np.unique(np.concatenate([ctx.idx[(r, K)] for r in rs for K in ctx.ks]))
            pos = {int(i): k for k, i in enumerate(used)}
            Xo = torch.cat([M.rows(tgt.X, used), Xs])
            yo = M.labels(np.r_[tgt.y[used], ys_np])
            nu = len(used)
            base_jobs, base_masks = jobs_for(ctx, rs, pos, nu)
            jobs, masks, weights = [], [], []
            ys_t = ys
            ws_src = torch.zeros(len(yo), dtype=logreg.DTYPE, device=DEV)
            cnt_s = torch.bincount(ys_t, minlength=4).to(logreg.DTYPE)
            ws_src[nu:] = 1.0 / (4 * cnt_s[ys_t])                          # ソースの中で感情を均等にし，和を 1 にする
            for alpha in ALPHAS:
                for job, m in zip(base_jobs, base_masks):
                    mt = torch.as_tensor(m, device=DEV)
                    yt_m = yo[:nu][mt]
                    cnt_t = torch.bincount(yt_m, minlength=4).to(logreg.DTYPE)
                    wt = torch.zeros(len(yo), dtype=logreg.DTYPE, device=DEV)
                    wt[:nu][mt] = 1.0 / (4 * cnt_t[yt_m].clamp_min(1))
                    S = float(m.sum() + ns)
                    w = (alpha * wt / wt.sum() + (1 - alpha) * ws_src) * S
                    fm = np.r_[m, np.ones(ns, bool)]
                    jobs.append(job + (alpha,)); masks.append(fm); weights.append(w)
            fits = logreg.fit_batch(Xo, yo, masks, (C_s,), weights=weights)
            for alpha in ALPHAS:
                sel = [k for k, jb in enumerate(jobs) if jb[3] == alpha]
                collect(ctx, [jobs[k][:3] for k in sel], [fits[k] for k in sel], (C_s,), lambda r, ii: M.rows(tgt.X, ii), None,
                        adder("O1", sname), {"alpha": alpha})
        print(f"  {kind} {tgt.name} ← {sname}：{time.perf_counter() - t0:.1f} 秒（束 {len(groups)}）", flush=True)

    # ---- P4：全ソースの下書き（同じコーパスの群を除く）
    t0 = time.perf_counter()
    pool = [s for s in srcs if GROUP_SPEC[s][0] != GROUP_SPEC[tgt.name][0]]
    ms = pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{tgt.name}.csv", dtype={"K": str})
    C_p = float(ms.C.iloc[0])
    Xp = torch.cat([data[s].X for s in pool]); yp_np = np.concatenate([data[s].y for s in pool]); yp = M.labels(yp_np)
    cp_mean = Xp.mean(0)
    src_p = logreg.fit(Xp, yp, (C_p,))
    src_pc = logreg.fit(Xp - cp_mean, yp, (C_p,))
    ht_block(ctx, reps, src_p, C_p, False, adder("HT_ALL", "ALL"))
    ht_block(ctx, reps, src_pc, C_p, True, adder("HT_ALL", "ALL"))
    print(f"  {kind} {tgt.name} P4：{time.perf_counter() - t0:.1f} 秒", flush=True)

    # ---- O6：最近傍セントロイド（K 本だけ．余弦）
    for r in reps:
        te = np.flatnonzero(ctx.te[r]); yte = tgt.y[te]
        for variant, m in (("none", torch.zeros_like(cp_mean)), ("src_mean", cp_mean), ("tgt_mean", ctx.ct[r])):
            Zte = M.rows(tgt.X, te) - m
            Zte = Zte / Zte.norm(dim=1, keepdim=True).clamp_min(1e-12)
            for K in ks:
                idx = ctx.idx[(r, K)]
                Zk = M.rows(tgt.X, idx) - m
                Zk = Zk / Zk.norm(dim=1, keepdim=True).clamp_min(1e-12)
                cen = torch.stack([Zk[torch.as_tensor(tgt.y[idx] == k, device=DEV)].mean(0) for k in range(4)])
                cen = cen / cen.norm(dim=1, keepdim=True)
                pred = (Zte @ cen.T).argmax(1).cpu().numpy()
                rows.append({"target": tgt.name, "source": "ALL", "method": "O6", "cen": int(variant == "tgt_mean"), "cp": np.nan,
                             "cb": "", "alpha": np.nan, "variant": variant, "rep": r, "K": K, "uar": uar(yte, pred), "cv_uar": np.nan})

    # ---- O7：メタ初期値（MAML．線形の頭．ソースを混ぜた統計で標準化．float32）
    t0 = time.perf_counter()
    mu_p, sd_p = Xp.mean(0).float(), Xp.std(0, correction=0).float()
    sd_p = torch.where(sd_p > 1e-12, sd_p, torch.ones_like(sd_p))
    corp = {s: ((data[s].X.float() - mu_p) / sd_p, torch.as_tensor(data[s].y, device=DEV)) for s in pool}
    Zt_all = (tgt.X.float() - mu_p) / sd_p
    d = Zt_all.shape[1]

    def inner(W, b, Z, y):
        for _ in range(O7_STEPS_IN):
            loss = torch.nn.functional.cross_entropy(Z @ W.T + b, y)
            gW, gb = torch.autograd.grad(loss, (W, b), create_graph=True)
            W, b = W - O7_LR_IN * gW, b - O7_LR_IN * gb
        return W, b

    for K in ks:
        kk = int(K)
        g = torch.Generator(device="cpu").manual_seed(zlib.crc32(f"O7|{kind}|{tgt.name}|{K}".encode()))
        W0 = torch.zeros(4, d, device=DEV, requires_grad=True)
        b0 = torch.zeros(4, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([W0, b0], lr=O7_LR_OUT)
        names = list(corp)
        for it in range(O7_ITERS):
            opt.zero_grad()
            tot = 0.0
            for _ in range(O7_BATCH):
                sname = names[int(torch.randint(len(names), (1,), generator=g))]
                Z, y = corp[sname]
                sup, qry = [], []
                for k in range(4):
                    ii = torch.nonzero(y == k).squeeze(1)
                    perm = ii[torch.randperm(len(ii), generator=g).to(DEV)]
                    sup.append(perm[:kk]); qry.append(perm[kk:kk + O7_QUERY // 4])
                sup, qry = torch.cat(sup), torch.cat(qry)
                W, b = inner(W0, b0, Z[sup], y[sup])
                tot = tot + torch.nn.functional.cross_entropy(Z[qry] @ W.T + b, y[qry])
            (tot / O7_BATCH).backward()
            opt.step()
        for r in reps:
            idx = ctx.idx[(r, K)]
            yk = torch.as_tensor(tgt.y[idx], device=DEV)
            W, b = inner(W0, b0, Zt_all[torch.as_tensor(idx, device=DEV)], yk)
            te = np.flatnonzero(ctx.te[r])
            pred = (Zt_all[torch.as_tensor(te, device=DEV)] @ W.T + b).argmax(1).detach().cpu().numpy()
            rows.append({"target": tgt.name, "source": "ALL", "method": "O7", "cen": 0, "cp": np.nan, "cb": "", "alpha": np.nan,
                         "variant": "maml", "rep": r, "K": K, "uar": uar(tgt.y[te], pred), "cv_uar": np.nan})
    print(f"  {kind} {tgt.name} O6・O7：{time.perf_counter() - t0:.1f} 秒", flush=True)
    np.savez_compressed(out / f"{tgt.name}.npz", **saved)
    d_ = pd.DataFrame(rows)
    d_.to_csv(out / f"{tgt.name}.csv", index=False)
    return d_


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(KS))
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / f"compare_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    (out / "provenance.json").write_text(json.dumps({
        "script": "exp/compare_b1.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "CPS": CPS, "CBS": [cbname(c) for c in CBS],
        "ALPHAS": ALPHAS, "O7": [O7_STEPS_IN, O7_LR_IN, O7_ITERS, O7_LR_OUT, O7_BATCH, O7_QUERY]}, ensure_ascii=False, indent=1))
    groups = groups_for(args.kind)
    data = {g: M.load(g, args.kind) for g in groups}
    for t in args.target or groups:
        if (out / f"{t}.npz").exists():
            print(f"{args.kind} {t}：済み（飛ばす）", flush=True)
            continue
        t0 = time.perf_counter()
        run_target(args.kind, data[t], data, None, args.reps, args.ks, out)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
