"""DAI-9 の下調べ：感情でないまとまりの成分を先に取り除くと，EM は感情の割り当てを保つか
（設計 DAI/docs/実験9_下調べ_設計.md）。DAI-8 の関数（dai8_probe.py）をそのまま使う。

単位の段：  python DAI/exp/dai9_probe.py --out results/dai9_probe [--units c_f_r,...]
集計の段：  python DAI/exp/dai9_probe.py --out results/dai9_probe --summarize [--allow-partial]
自己検査：  python DAI/exp/dai9_probe.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai_data as DD  # noqa: E402
import dai7_probe_v2 as P7  # noqa: E402
import dai8_probe as P8  # noqa: E402   lda_from_R，em_fit，delta，_top3，_top3_null，_top3_resid，dialogue_ids，_uar_pool

ROOT = P8.ROOT
UD, FROZEN = P7.UD, P7.FROZEN
DEV, DT = P8.DEV, P8.DT
ALPHA = 0.5
TAUS = ("raw", "gc", "nap10", "nap50", "pca64", "pca256")
KS_RULE = (1, 5, 10, 25)
N_EM, MAX_EM, TOL_EM = 10, 100, 1e-4
MIN_D = 0.5
R9_TEST_DROP, KEEP_RAND, KEEP_GAP = -2.0, 0.9, 0.2
HYP_GAIN = 2.0                # 予想の書き方：EM_truth^τ − LDA_O^τ が raw より 2 ポイント以上大きい
MIN_GROUP = 3                 # gc：これ未満の本数のグループは全体の平均を引く
N_UNITS = 40
SEED = 0


# ------------------------------------------------------------------ 取り除き方

def _restd(Xp, Xt):
    sd = Xp.std(0).clamp(min=1e-6)
    mu = Xp.mean(0)
    return (Xp - mu) / sd, (Xt - mu) / sd


def _center_groups(X, g):
    out = X - X.mean(0)
    _, gi, cnt = np.unique(g, return_inverse=True, return_counts=True)
    gi_t = torch.as_tensor(gi, device=X.device)
    M = torch.zeros((len(cnt), X.shape[1]), dtype=X.dtype, device=X.device).index_add_(0, gi_t, out)
    M = M / torch.as_tensor(cnt, dtype=X.dtype, device=X.device)[:, None]
    big = torch.as_tensor(cnt >= MIN_GROUP, device=X.device)
    M = torch.where(big[:, None], M, torch.zeros_like(M))                    # 3 本未満のグループは全体の平均だけを引く
    return out - M[gi_t]


def transform(tau: str, Xp, Xt, gp, gt):
    """候補で標準化した Xp（候補），Xt（テスト）に取り除き方 τ を当て，候補で標準化し直す（3.1 節）。"""
    if tau == "raw":
        return Xp, Xt
    if tau == "gc":
        return _restd(_center_groups(Xp, gp), _center_groups(Xt, gt))
    if tau.startswith("nap"):
        r = int(tau[3:])
        Xc = Xp - Xp.mean(0)
        _, gi, cnt = np.unique(gp, return_inverse=True, return_counts=True)
        gi_t = torch.as_tensor(gi, device=Xp.device)
        M = torch.zeros((len(cnt), Xp.shape[1]), dtype=Xp.dtype, device=Xp.device).index_add_(0, gi_t, Xc)
        w = torch.as_tensor(cnt / cnt.sum(), dtype=Xp.dtype, device=Xp.device)
        M = M / torch.as_tensor(cnt, dtype=Xp.dtype, device=Xp.device)[:, None]
        Sg = (M.T * w) @ M
        ev, V = torch.linalg.eigh((Sg + Sg.T) / 2)
        V = V[:, -r:]
        return _restd(Xp - (Xp @ V) @ V.T, Xt - (Xt @ V) @ V.T)
    if tau.startswith("pca"):
        p = int(tau[3:])
        mu = Xp.mean(0)
        C = (Xp - mu).T @ (Xp - mu) / Xp.shape[0]
        ev, V = torch.linalg.eigh((C + C.T) / 2)
        V = V[:, -p:]
        return _restd((Xp - mu) @ V, (Xt - mu) @ V)
    raise ValueError(tau)


# ------------------------------------------------------------------ 単位

def unit_list():
    out = []
    for c in DD.CORPORA:
        T = sorted(x for x in DD.CORPORA if x != c)[0]
        for f in range(5):
            for r in (0, 1):
                out.append((T, c, f, r))
    return out


def run_unit(T, c, f, r, out: Path, fz: dict, commit: str) -> dict:
    name = f"{c}_f{f}_r{r}"
    dst = out / "rows" / f"{name}.npz"
    if dst.exists():
        raise SystemExit(f"{dst} は既にある")
    t0 = time.time()
    F = fz[T]
    srcs = [x for x in DD.CORPORA if x not in (T, c)]
    uname = f"{T}_{c}_f{f}_r{r}"
    z1 = np.load(UD / f"{uname}_p1.npz")
    if str(z1["commit"]) != F["pseudo_commit"]:
        raise SystemExit(f"{uname}_p1 のコミットが固定のファイルの pseudo_commit と違う")
    u = dai1.Unit(ROOT / "results" / "dai1", T, c, srcs, f, r)
    sid_t = u.cp.df["sample_id"].to_numpy()[u.test].astype(str)
    sid_p = u.cp.df["sample_id"].to_numpy()[u.pool].astype(str)
    if not ((sid_t == z1["sample_test"].astype(str)).all() and (sid_p == z1["sample_pool"].astype(str)).all()):
        raise SystemExit(f"{name}：発話の並びが保存と違う")
    y_of, cl_of = ST.label_maps(c)
    yt, yp = u.ytest.astype(int), u.ypool.astype(int)
    if not ((y_of(sid_t) == yt).all() and (y_of(sid_p) == yp).all()):
        raise SystemExit(f"{name}：発話の並びとラベルの対応が読み込みと違う")
    gcol = "speaker_id" if c == "IEMOCAP" else "cluster"
    gp = u.cp.df[gcol].to_numpy()[u.pool].astype(str)
    gt = u.cp.df[gcol].to_numpy()[u.test].astype(str)
    dlg_p = P8.dialogue_ids(c, sid_p, u.cp.df["dialogue"].to_numpy()[u.pool])
    yp_t = torch.as_tensor(yp, device=DEV)
    onehot_all = torch.nn.functional.one_hot(yp_t, 4).to(DT)
    X0p, X0t = u.Xp["AVT"].to(DEV, DT), u.Xt["AVT"].to(DEV, DT)
    pred, diag, q1 = {}, {}, []
    empty = torch.zeros(0, dtype=torch.long, device=DEV)
    yrand = np.random.default_rng(SEED).permutation(yp)
    _, gi = np.unique(gp, return_inverse=True)
    _, di = np.unique(dlg_p, return_inverse=True)
    for tau in TAUS:
        Xp, Xt = transform(tau, X0p, X0t, gp, gt)
        if not (torch.isfinite(Xp).all() and torch.isfinite(Xt).all()):
            raise FloatingPointError(f"{name} {tau}：取り除いた特徴に有限でない値がある")
        XtX = Xp.T @ Xp
        fO = P8.lda_from_R(Xp, onehot_all, XtX, ALPHA)
        pred[f"{tau}_ldaO"] = P8.delta(Xt, fO).argmax(1)
        diag[f"{tau}_ldaO"] = dict(pool_uar=P8._uar_pool(yp, P8.delta(Xp, fO).argmax(1).cpu().numpy()))
        for tag, lab in (("truth", yp), ("rand", yrand)):
            q_ = torch.nn.functional.one_hot(torch.as_tensor(lab, device=DEV), 4).to(DT)
            res = P8.em_fit(Xp, XtX, empty, empty, q_, ALPHA, 0.0, n_iter=MAX_EM, tol=TOL_EM, yU=yp)
            pred[f"{tau}_em{tag}"] = P8.delta(Xt, res["fit"]).argmax(1)
            prc = res["R"].argmax(1).cpu().numpy()
            diag[f"{tau}_em{tag}"] = dict(pool_uar=P8._uar_pool(yp, prc), keep_init=float((prc == lab).mean()),
                                          n_iter_done=res["info"]["n_iter_done"])
        for K in KS_RULE:
            kpos = u.kpos[K]
            kpos_t = torch.as_tensor(kpos, dtype=torch.long, device=DEV)
            ylab = yp_t[kpos_t]
            XL = Xp[kpos_t]
            fk = P8.lda_from_R(XL, torch.nn.functional.one_hot(ylab, 4).to(DT), XL.T @ XL, ALPHA)
            pred[f"K{K}_{tau}_ldaK"] = P8.delta(Xt, fk).argmax(1)
            q0 = torch.softmax(P8.delta(Xp, fk), -1)
            res = P8.em_fit(Xp, XtX, kpos_t, ylab, q0, ALPHA, 0.0, n_iter=MAX_EM, tol=TOL_EM, snaps=(N_EM,), yU=None)
            pred[f"K{K}_{tau}_emK"] = P8.delta(Xt, res["snap"][N_EM]).argmax(1)
            pred[f"K{K}_{tau}_emKconv"] = P8.delta(Xt, res["fit"]).argmax(1)
            Um = np.ones(len(yp), bool)
            Um[kpos] = False
            prU = res["snapR"][N_EM].argmax(1).cpu().numpy()[Um]
            diag[f"K{K}_{tau}_emK"] = dict(pool_uar=P8._uar_pool(yp[Um], prU), n_iter_done=res["info"]["n_iter_done"],
                                           pool_uar_init=P8._uar_pool(yp[Um], q0.argmax(1).cpu().numpy()[Um]))
        if r == 0:                                                           # Q3 は (c, f) ごとに 1 回
            T_ = Xp.T @ Xp / Xp.shape[0]
            Ach = torch.linalg.cholesky((1 - ALPHA) * T_ + ALPHA * torch.eye(Xp.shape[1], dtype=DT, device=DEV))
            row = dict(tau=tau, top3_emo=P8._top3(Xp, yp, Ach))
            row["top3null_emo"], row["top3nullmax_emo"] = P8._top3_null(Xp, yp, Ach)
            for nm, lab in (("grp_xe", gi), ("dlg_xe", di)):
                v, nul, nmax = P8._top3_resid(Xp, lab, yp, Ach)
                row[f"top3_{nm}"], row[f"top3null_{nm}"], row[f"top3nullmax_{nm}"] = v, nul, nmax
            for nm in ("emo", "grp_xe", "dlg_xe"):
                row[f"top3ex_{nm}"] = row[f"top3_{nm}"] - row[f"top3null_{nm}"]
            q1.append(row)
        del Xp, Xt, XtX
        torch.cuda.empty_cache()
    (out / "rows").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, commit=commit, T=T, test_sample_id=sid_t, test_y=yt.astype(np.int8), test_cluster=cl_of(sid_t).astype(str),
                        diag=json.dumps(diag), **{k: v.cpu().numpy().astype(np.int8) for k, v in pred.items()})
    if q1:
        pd.DataFrame(q1).assign(c=c, fold=f, commit=commit).to_csv(out / "rows" / f"{name}_q1.csv", index=False)
    del u
    torch.cuda.empty_cache()
    sec = time.time() - t0
    print(f"{name}：{sec:.0f} 秒", flush=True)
    return dict(name=name, seconds=sec)


# ------------------------------------------------------------------ 集計

def summarize(out: Path, allow_partial: bool) -> None:
    files = sorted((out / "rows").glob("*.npz"))
    if len(files) != N_UNITS and not allow_partial:
        raise SystemExit(f"単位の出力が {len(files)} 個（{N_UNITS} のはず）")
    rc = {str(np.load(p, allow_pickle=False)["commit"]) for p in files}
    if len(rc) != 1:
        raise SystemExit(f"行のコミットが 1 つでない：{rc}")
    rng = np.random.default_rng(SEED)
    cells, cellrows, diagrows = {}, [], []
    for p in files:
        c, fs, rs = p.name[:-4].split("_")
        f, r = int(fs[1:]), int(rs[1:])
        z = np.load(p, allow_pickle=False)
        y, cl = z["test_y"].astype(int), z["test_cluster"].astype(str)
        for k in z.files:
            if z[k].dtype == np.int8 and z[k].shape == y.shape and k != "test_y":
                cnt = ST.counts(y, z[k].astype(int), cl)
                cells.setdefault((c, k), {})[(f, r)] = cnt
                cellrows.append(dict(c=c, fold=f, seed=r, key=k, uar=ST.uar_full(cnt)))
        for k, d in json.loads(str(z["diag"])).items():
            diagrows.append(dict(c=c, fold=f, seed=r, key=k, **d))
    U = pd.DataFrame(cellrows)
    U.groupby(["c", "key"]).uar.agg(["mean", "count"]).reset_index().to_csv(out / "cells.csv", index=False)
    DG = pd.DataFrame(diagrows)
    DG.to_csv(out / "diag_units.csv", index=False)
    q1 = pd.concat([pd.read_csv(p) for p in sorted((out / "rows").glob("*_q1.csv"))])
    q1.to_csv(out / "q1_units.csv", index=False)
    comps = []
    for tau in TAUS:
        comps.append((f"{tau}_ldaO_vs_raw", 0, f"{tau}_ldaO", "raw_ldaO"))
        comps.append((f"{tau}_emtruth_ldaO", 0, f"{tau}_emtruth", f"{tau}_ldaO"))
        for K in KS_RULE:
            comps.append((f"{tau}_emK_ldaK", K, f"K{K}_{tau}_emK", f"K{K}_{tau}_ldaK"))
            comps.append((f"{tau}_emKconv_ldaK", K, f"K{K}_{tau}_emKconv", f"K{K}_{tau}_ldaK"))
            comps.append((f"{tau}_emK_ldaKraw", K, f"K{K}_{tau}_emK", f"K{K}_raw_ldaK"))
            comps.append((f"{tau}_ldaK_ldaKraw", K, f"K{K}_{tau}_ldaK", f"K{K}_raw_ldaK"))
    rows = []
    for c in sorted({k[0] for k in cells}):
        g_eff = ST.g_eff_of(c)[0]
        for nm, K, a, b in comps:
            if (c, a) in cells and (c, b) in cells:
                rows.append(dict(c=c, K=K, compare=nm, **ST.effect_stats(cells[(c, a)], cells[(c, b)], g_eff, rng)))
        print(f"{c}：区間を作った", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(out / "diffs.csv", index=False)
    crit = []
    for c in sorted(D.c.unique()):
        r_ = dict(c=c)
        for tau in TAUS:
            d = D[(D.c == c) & (D["compare"] == f"{tau}_emK_ldaK") & D.K.isin(KS_RULE)]
            if not allow_partial and len(d) != 4:
                raise SystemExit(f"{c} {tau}：S1 の行が 4 でない")
            ok = (d.lo > 0) & (d.effect >= MIN_D)
            r_[f"S1_{tau}_n_K"], r_[f"S1_{tau}_n_hi_neg"] = int(ok.sum()), int((d.hi < 0).sum())
            r_[f"S1_{tau}"], r_[f"S1_{tau}_harm"] = bool(ok.sum() >= 3), bool((d.hi < 0).sum() >= 3)
            dt = D[(D.c == c) & (D["compare"] == f"{tau}_emtruth_ldaO")]
            r_[f"S2_{tau}_truth_minus_ldaO"] = float(dt.effect.iloc[0]) if len(dt) else np.nan
            tru = DG[(DG.c == c) & (DG.key == f"{tau}_emtruth")]
            rnd = DG[(DG.c == c) & (DG.key == f"{tau}_emrand")]
            kt, kr = float(tru.keep_init.mean()), float(rnd.keep_init.mean())
            r_[f"S2_{tau}_keep_truth"], r_[f"S2_{tau}_keep_rand"] = kt, kr
            if r_[f"S2_{tau}_truth_minus_ldaO"] <= R9_TEST_DROP:
                r_[f"S2_{tau}"] = "b_optimum_away_from_emotion"
            elif kr >= KEEP_RAND or (kt - kr) < KEEP_GAP:
                r_[f"S2_{tau}"] = "c_keeps_any_init"
            else:
                r_[f"S2_{tau}"] = "a_init_or_other"
            qq = q1[(q1.c == c) & (q1.tau == tau)]
            for nm in ("emo", "grp_xe", "dlg_xe"):
                r_[f"S4_{tau}_top3ex_{nm}"] = float(qq[f"top3ex_{nm}"].mean())
            tx = max(("grp_xe", "dlg_xe"), key=lambda x: r_[f"S4_{tau}_top3ex_{x}"])
            r_[f"S4_{tau}_nonemo_separates"] = bool(int((qq[f"top3_{tx}"] > qq[f"top3nullmax_{tx}"]).sum()) >= 3
                                                    and r_[f"S4_{tau}_top3ex_{tx}"] >= r_[f"S4_{tau}_top3ex_emo"])
        hyp = []
        for tau in ("gc", "nap10", "nap50"):
            if (r_["S2_raw"] == "b_optimum_away_from_emotion" and r_[f"S2_{tau}"] != "b_optimum_away_from_emotion"
                    and r_[f"S2_{tau}_truth_minus_ldaO"] >= r_["S2_raw_truth_minus_ldaO"] + HYP_GAIN):
                hyp.append(tau)
        r_["HYP_taus"] = ",".join(hyp)
        r_["HYP"] = bool(hyp)
        crit.append(r_)
    Cr = pd.DataFrame(crit)
    Cr["HYP_n_corpora"] = int(Cr.HYP.sum())
    Cr.to_csv(out / "criteria.csv", index=False)
    commit, dirty = dai1.git_commit()
    pp = out / "provenance.json"
    prov = json.loads(pp.read_text()) if pp.exists() else {}
    prov.update(summarize_commit=commit, summarize_dirty=dirty, rows_commits=sorted(rc), boot_seed=SEED, n_boot=ST.N_BOOT)
    pp.write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"書いた：cells.csv，diag_units.csv，q1_units.csv，diffs.csv（{len(D)} 行），criteria.csv", flush=True)


# ------------------------------------------------------------------ 自己検査

def selftest() -> None:
    g = torch.Generator().manual_seed(0)
    n, d = 3000, 20
    y = np.random.default_rng(0).integers(0, 4, n)
    grp = np.random.default_rng(1).integers(0, 30, n)
    mu = torch.randn(4, d, generator=g, dtype=DT) * 0.6
    gv = torch.randn(30, d, generator=g, dtype=DT) * 2.0                      # グループの成分が感情より大きい
    X = (mu[torch.as_tensor(y)] + gv[torch.as_tensor(grp)] + torch.randn(n, d, generator=g, dtype=DT)).to(DEV)
    X = (X - X.mean(0)) / X.std(0)
    Xt = X[:500]
    for tau in TAUS:
        if tau.startswith("pca") and int(tau[3:]) >= d:
            continue
        if tau.startswith("nap") and int(tau[3:]) >= d:
            continue
        a, b = transform(tau, X, Xt, grp, grp[:500])
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert float(a.std(0).sub(1).abs().max()) < 1e-6 or tau == "raw"
    Xg, _ = transform("gc", X, Xt, grp, grp[:500])
    _, gi = np.unique(grp, return_inverse=True)
    gm = torch.zeros((30, d), dtype=DT, device=DEV).index_add_(0, torch.as_tensor(gi, device=DEV), Xg)
    print(f"(1) gc の後のグループの平均の大きさの最大 {float(gm.abs().max() / 100):.2e}（0 に近い）")
    assert float(gm.abs().max()) / 100 < 1e-6
    # (2) nap：消した向きの成分が 0
    Xn, _ = transform("nap10", X, Xt, grp, grp[:500])
    print("(2) nap10 の後の値は有限")
    # (3) グループの成分が大きい人工データで，gc は全ラベルの LDA を良くする
    yt = torch.as_tensor(y, device=DEV)
    oh = torch.nn.functional.one_hot(yt, 4).to(DT)
    def acc(Z):
        f_ = P8.lda_from_R(Z, oh, Z.T @ Z, ALPHA)
        return float((P8.delta(Z, f_).argmax(1) == yt).double().mean())
    print(f"(3) 全ラベルの LDA の正解率：raw {acc(X):.3f}，gc {acc(Xg):.3f}")
    assert acc(Xg) > acc(X)
    print(f"(4) 単位の数 {len(unit_list())}")
    assert len(unit_list()) == N_UNITS
    print("自己検査：通った")


# ------------------------------------------------------------------ 入口

def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "dai9_probe")
    ap.add_argument("--units", default=None, help="c_f_r をコンマで（試走）")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    out = a.out if a.out.is_absolute() else ROOT / a.out
    if a.summarize:
        summarize(out, a.allow_partial)
        return
    commit, dirty = dai1.git_commit()
    fz = json.loads(FROZEN.read_text())
    units = unit_list()
    partial = bool(a.units)
    if partial:
        want = set(a.units.split(","))
        units = [x for x in units if f"{x[1]}_f{x[2]}_r{x[3]}" in want]
    else:
        if out.exists():
            raise SystemExit(f"{out} は既にある")
        if dirty:
            raise SystemExit("作業木に未コミットの変更がある")
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(commit=commit, dirty=dirty, partial=partial, started=time.strftime("%F %T"), python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                script_sha256=sha(Path(__file__)), dai8_sha256=sha(HERE / "dai8_probe.py"), frozen_sha256=sha(FROZEN),
                alpha=ALPHA, taus=TAUS, n_em=N_EM, max_em=MAX_EM, tol_em=TOL_EM, min_d=MIN_D, r9_test_drop=R9_TEST_DROP,
                keep_rand=KEEP_RAND, keep_gap=KEEP_GAP, hyp_gain=HYP_GAIN, min_group=MIN_GROUP,
                units=[f"{x[1]}_f{x[2]}_r{x[3]}" for x in units],
                inputs_sha256={f"{x[0]}_{x[1]}_f{x[2]}_r{x[3]}": sha(UD / f"{x[0]}_{x[1]}_f{x[2]}_r{x[3]}_p1.npz") for x in units})
    pname = f"provenance_partial_{time.strftime('%Y%m%d-%H%M%S')}.json" if partial else "provenance.json"
    (out / pname).write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"開始 {prov['started']} コミット {commit[:7]} 単位 {len(units)} 個", flush=True)
    runs = [run_unit(*x, out=out, fz=fz, commit=commit) for x in units]
    prov["runs"], prov["finished"] = runs, time.strftime("%F %T")
    (out / pname).write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"{len(runs)} 単位を終えた", flush=True)


if __name__ == "__main__":
    main()
