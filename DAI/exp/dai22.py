"""実験 22（DAI-22）：融合の分類器の当たる幅と再現できる分を評価の回で測る（設計 `docs/設計/実験22_設計.md` 第 1 版）．

本番の評価の回（実験 1．results/dai1/eval）の保存から，融合の生徒 AVT について，K 本だけの分類器の判定 J_none と
提案の主の版の疑似ラベル J_prop（prop_one）を候補の上で作り直し，J_prop を目標にした線形の識別器の交差の予測を 2 つの読み出しで作る．
  R0（実験 1 の作り．表 8.1 の P の列）：生徒の出発点と実験 1 の引き戻し（実験 14 の fit_r0 をそのまま使う）．
  R1（そろえた読み出し．副）：候補の上の主成分で 512 次元にそろえ，C′ を内側の交差の予測で選ぶ（実験 14 の fit_r1）．
量は実験 14 の conditions.csv と同じ名前：uar_none，uar_pseudo，delta，uar_imitation_R0/R1，imitable_R0/R1（K 本を除いた候補の上）．
K = 5（主）と 25（副）．

段：
  selftest  実験 14 の selftest と，要約の差の量が UAR の差と一致すること
  units     単位ごとの計算．results/dai22/units/<T>_f<f>_r<r>.npz と .csv（同じコミットの既存の単位は飛ばす）
  summary   区間．results/dai22/summary/*.csv
"""
from __future__ import annotations

import argparse
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
import dai3  # noqa: E402
import dai14 as D14  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT14 = ROOT / "results" / "dai14"
OUT_PROD = ROOT / "results" / "dai22"
KS = (5, 25)
STU = "AVT"
GRID_C = D14.GRID_C
EDGE_SHARE = D14.EDGE_SHARE
MEM_FRAC = 0.6
RNG_SEED = 20261115
SCRIPTS = ("dai22.py", "dai14.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py", "dai_data.py")
DESIGN = ROOT / "docs" / "設計" / "実験22_設計.md"
REPRO_SAVED = ROOT / "results" / "dai3" / "eval" / "repro_units.csv"
COND14 = OUT14 / "summary" / "conditions.csv"
TOL_REPRO = 1e-3
TOL_PSEUDO = 1e-9
QUANT = ("uar_none", "uar_pseudo", "delta", "uar_imitation_R0", "imitable_R0", "uar_imitation_R1", "imitable_R1")
RUN = {"commit": "", "dirty": True}


def script_shas() -> dict:
    return {n: D14.sha(HERE / n) for n in SCRIPTS}


# ------------------------------------------------------------------ 段 units

def run_unit(T: str, f: int, r: int, fz1T: dict, out: Path, repro: pd.DataFrame) -> pd.DataFrame | None:
    base = out / "units" / f"{T}_f{f}_r{r}"
    if Path(f"{base}.csv").exists() and Path(f"{base}.npz").exists():
        old = pd.read_csv(f"{base}.csv")
        if (old.commit == RUN["commit"]).all():
            return None                                               # 同じコミットで済んだ単位は飛ばす（再開）
        raise SystemExit(f"{base.name} が別のコミットで既にある（消してから流し直す）")
    t0 = time.time()
    sv = dai3.Saved(OUT1, "eval", T, T, f, r, fz1T)
    u = sv.u
    TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
    n = u.n
    y = np.asarray(u.ypool, int)
    cl = sv.cl_pool.astype(str)
    h = dai3.pool_halves(cl)
    halves = [np.flatnonzero(h == 0), np.flatnonzero(h == 1)]
    a14 = np.load(OUT14 / "units" / f"{T}_f{f}_r{r}.npz", allow_pickle=False)
    if not (a14["sid"].astype(str) == sv.sid_pool.astype(str)).all() or not (a14["half"].astype(int) == h).all():
        raise SystemExit(f"{base.name}：候補の並びか 2 組の分け方が実験 14 の保存と違う")
    arr = dict(y=y.astype(np.int8), cl=cl, sid=sv.sid_pool.astype(str), half=h.astype(np.int8))
    Z, pca = D14.aligned_features(u.Xp[STU])
    rows, tg_true = [], {}
    for K in KS:
        main = "prop_" + fz1T["main"][str(K)]
        nk = np.ones(n, bool); nk[u.kpos[K]] = False
        arr[f"nk__K{K}"] = nk
        arr[f"none__K{K}"] = S.logits(u.Xp[STU], *sv.none(STU, K)).argmax(1).cpu().numpy().astype(np.int8)
        Q = dai3.pool_teachers(sv, TK[K], STU, K, [main])
        qy = Q[main].argmax(1).cpu().numpy().astype(int)
        if not (a14[f"tgt__K{K}__{main}__true"].astype(int) == qy).all():
            raise SystemExit(f"{base.name}：K={K} の J_prop が実験 14 の保存と違う")
        arr[f"tgt__K{K}"] = qy.astype(np.int8)
        arr[f"maxp__K{K}"] = Q[main].max(1).values.cpu().numpy().astype(np.float16)
        tg_true[f"K{K}"] = qy
        res0 = D14.fit_r0(sv, STU, K, {f"K{K}": qy}, halves)
        full, ag, info = res0[f"K{K}"]
        arr[f"K{K}__R0"] = full
        arr[f"prob__K{K}__R0"] = info["prob"]
        sv_row = repro[(repro["T"] == T) & (repro.fold == f) & (repro.seed == r) & (repro.K == K) & (repro.teacher == main)]
        if len(sv_row) != 1:
            raise SystemExit(f"{base.name}：実験 3 の保存に K={K} の行が {len(sv_row)} 行")
        rows.append(dict(T=T, fold=f, seed=r, K=K, readout="R0", target=main, agree_halfmean=ag, agree_saved=float(sv_row.agree.iloc[0]),
                         abs_diff=abs(ag - float(sv_row.agree.iloc[0])), n_cand=n, n_nk=int(nk.sum()), n_half0=len(halves[0]),
                         n_half1=len(halves[1]), dim=info["dim"], Cw=info["Cw"], Cb=info["Cb"], grad=info["grad"], stalled=info["stalled"]))
    # R1：K = 5 と 25 の問を 1 回の束で解く（実験 14 と同じ）
    tg_all = {(k, "true"): v for k, v in tg_true.items()}
    res1, curves, _ = D14.fit_r1(Z, cl, halves, tg_true, tg_all)
    for (isp, hh, nm), cur in curves.items():
        arr.setdefault(f"inner__{nm}", np.zeros((2, 2, len(GRID_C))))[hh] = cur
    for K in KS:
        full, ag, info = res1[(f"K{K}", "true")]
        arr[f"K{K}__R1"] = full
        arr[f"prob__K{K}__R1"] = info["prob"]
        rows.append(dict(T=T, fold=f, seed=r, K=K, readout="R1", target="prop_" + fz1T["main"][str(K)], agree_halfmean=ag,
                         n_cand=n, n_nk=int(arr[f"nk__K{K}"].sum()), n_half0=len(halves[0]), n_half1=len(halves[1]),
                         dim=info["dim"], Cb=info["Cb"], C0=info["C0"], C1=info["C1"], grad=info["grad"], stalled=info["stalled"],
                         inner_grad=info["inner_grad"], inner_stalled=info["inner_stalled"], **pca))
    for K in KS:                                                      # 確率の argmax が予測と合うこと（実験 14 と同じ許し）
        for ro in ("R0", "R1"):
            pf = arr[f"prob__K{K}__{ro}"].astype(np.float32); pk = arr[f"K{K}__{ro}"].astype(int)
            if not ((pf.max(1) - pf[np.arange(n), pk]) <= 1e-3).all():
                raise SystemExit(f"{base.name}：K={K} {ro} の確率の argmax が予測と合わない")
    df = pd.DataFrame(rows)
    df["commit"], df["dirty"], df["sec"] = RUN["commit"], RUN["dirty"], time.time() - t0
    df["dai1_commits"] = ",".join(sorted(sv.commits))
    np.savez_compressed(f"{base}.npz", **arr)
    df.to_csv(f"{base}.csv", index=False)
    return df


# ------------------------------------------------------------------ 段 summary

def load_units(out: Path, T: str) -> dict:
    U = {}
    for p in sorted((out / "units").glob(f"{T}_f*_r*.npz")):
        f, r = (int(x[1:]) for x in p.stem.split("_")[1:3])
        U[(f, r)] = dict(np.load(p, allow_pickle=False))
    return U


def cells_of(U: dict, K: int) -> dict:
    """K 本を除いた候補の上の，本当の感情に対する塊ごとの数．J_none，J_prop，R0，R1 の 4 つ．"""
    out = {}
    for nm, key in (("none", f"none__K{K}"), ("pseudo", f"tgt__K{K}"), ("R0", f"K{K}__R0"), ("R1", f"K{K}__R1")):
        out[nm] = {k: D14.counts(a["y"][a[f"nk__K{K}"]], a[key][a[f"nk__K{K}"]], a["cl"][a[f"nk__K{K}"]]) for k, a in U.items()}
    return out


def quantities(c: dict) -> list:
    return [("uar_none", [c["none"]], [1]), ("uar_pseudo", [c["pseudo"]], [1]), ("delta", [c["pseudo"], c["none"]], [1, -1]),
            ("uar_imitation_R0", [c["R0"]], [1]), ("imitable_R0", [c["R0"], c["none"]], [1, -1]),
            ("uar_imitation_R1", [c["R1"]], [1]), ("imitable_R1", [c["R1"], c["none"]], [1, -1])]


def stage_summary(out: Path, targets: list) -> None:
    sdir = out / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    fz1, _ = dai3.load_fz1(OUT1)
    cond, conv, plateau = [], [], []
    c14 = pd.read_csv(COND14)
    for T in targets:
        U = load_units(out, T)
        need = {(f, r) for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"]))}
        if RUN.get("partial"):
            need = set(U)
        if set(U) != need:
            raise SystemExit(f"{T}：単位の格子がそろっていない（{len(U)} / {len(need)}）")
        csvs = pd.concat([pd.read_csv(out / "units" / f"{T}_f{f}_r{r}.csv") for f, r in sorted(U)])
        if not RUN.get("partial") and (csvs.commit.nunique() != 1 or csvs.dirty.astype(str).ne("False").any()):
            raise SystemExit(f"{T}：単位のコミットが 1 つでないか，汚れた作業木の単位がある")
        if len(csvs) != len(U) * len(KS) * 2:
            raise SystemExit(f"{T}：行の数 {len(csvs)}（見込み {len(U) * len(KS) * 2}）")
        conv.append(csvs)
        for (f, r), a in U.items():                                   # 下端の C′ が「罰を強めた極限に届いた」だけかの印（実験 14 と同じ）
            for K in KS:
                cur = a[f"inner__K{K}"].mean(1)
                for hh in (0, 1):
                    j = int(np.argmax(cur[hh] > cur[hh].max() - 1e-12))
                    plateau.append(dict(T=T, K=K, fold=f, seed=r, half=hh, chosen=GRID_C[j],
                                        low_plateau=bool(j == 0 and abs(cur[hh][0] - cur[hh][1]) < 1e-9)))
        g_eff, _ = ST.g_eff_of(T)
        rng = np.random.default_rng([RNG_SEED, DD.CORPUS_ID[T]])
        for K in KS:
            for qn, cells_, co in quantities(cells_of(U, K)):
                e = D14.combo_stats(cells_, co, g_eff, rng)
                cond.append(dict(T=T, K=K, student=STU, quantity=qn) | e | dict(verdict=D14.verdict(e["lo"], e["hi"])))
            ref = c14[(c14["T"] == T) & (c14.K == K) & (c14.quantity == "uar_pseudo")].effect.unique()
            got = [x["effect"] for x in cond if x["T"] == T and x["K"] == K and x["quantity"] == "uar_pseudo"][0]
            if not RUN.get("partial") and (len(ref) == 0 or np.max(np.abs(ref - got)) > TOL_PSEUDO):
                raise SystemExit(f"{T} K={K}：uar_pseudo が実験 14 の conditions.csv と合わない")
        dai1.Corpus.cache.pop(T, None)
    C = pd.DataFrame(cond); CV = pd.concat(conv); PL = pd.DataFrame(plateau)
    tag = dict(commit=RUN["commit"], dirty=RUN["dirty"])
    C.assign(**tag).to_csv(sdir / "conditions.csv", index=False)
    asm = []
    for (T, K), x in PL.groupby(["T", "K"]):                         # 端の印（実験 14 の 8 節と同じ基準）
        lo = float((x.chosen == GRID_C[0]).mean()); hi = float((x.chosen == GRID_C[-1]).mean())
        lo_np = float(((x.chosen == GRID_C[0]) & ~x.low_plateau).mean())
        asm.append(dict(kind="C_edge", T=T, K=K, share_low=lo, share_high=hi, share_low_nonplateau=lo_np, share_low_plateau=lo - lo_np,
                        flag=bool(max(hi, lo_np) > EDGE_SHARE)))
    r1 = CV[CV.readout == "R1"]
    for T, x in r1.groupby("T"):
        asm.append(dict(kind="pca", T=T, pca_kept_mean=float(x.pca_kept.mean()), pca_kept_min=float(x.pca_kept.min()),
                        eff_rank_mean=float(x.eff_rank.mean()), dim_in=int(x.dim_in.iloc[0])))
    for (T, ro), x in CV.groupby(["T", "readout"]):
        asm.append(dict(kind="n_per_dim", T=T, readout=ro, half_over_dim_min=float((x[["n_half0", "n_half1"]].min(1) / x.dim).min())))
    pd.DataFrame(asm).assign(**tag).to_csv(sdir / "assumptions.csv", index=False)
    r0 = CV[CV.readout == "R0"]
    conv_s = dict(n_rows=len(CV), n_units=int(CV[["T", "fold", "seed"]].drop_duplicates().shape[0]), grad_max=float(CV.grad.max()),
                  n_stalled=int(CV.stalled.astype(str).eq("True").sum()), inner_grad_max=float(r1.inner_grad.max()),
                  inner_stalled_max=int(r1.inner_stalled.max()), r0_saved_abs_diff_max=float(r0.abs_diff.max()),
                  r0_saved_n_over_tol=int((r0.abs_diff > TOL_REPRO).sum()),
                  C_chosen_counts={str(k): int(v) for k, v in pd.concat([r1.C0, r1.C1]).value_counts().items()},
                  sec_total=float(CV.drop_duplicates(["T", "fold", "seed"]).sec.sum()))
    json.dump(conv_s, open(sdir / "convergence.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: conv_s[k] for k in ("n_rows", "n_units", "grad_max", "n_stalled", "r0_saved_n_over_tol")}, ensure_ascii=False))


# ------------------------------------------------------------------ 段 selftest

def selftest() -> None:
    D14.selftest()
    rng = np.random.default_rng(0)
    U = {}
    for f in range(5):
        for r in range(2):
            n = 300
            y = rng.integers(0, 4, n)
            a = dict(y=y, cl=np.array([f"g{i % 11}" for i in range(n)]))
            for K in KS:
                nk = np.ones(n, bool); nk[rng.choice(n, 4 * K // 5, replace=False)] = False
                a[f"nk__K{K}"] = nk
                a[f"none__K{K}"] = np.where(rng.random(n) < 0.4, y, rng.integers(0, 4, n))
                a[f"tgt__K{K}"] = np.where(rng.random(n) < 0.6, y, rng.integers(0, 4, n))
                a[f"K{K}__R0"] = np.where(rng.random(n) < 0.8, a[f"tgt__K{K}"], rng.integers(0, 4, n))
                a[f"K{K}__R1"] = a[f"tgt__K{K}"].copy()
            U[(f, r)] = a
    for g in (8.0, 30.0):
        for K in KS:
            q = {qn: D14.combo_stats(c, co, g, np.random.default_rng(1)) for qn, c, co in quantities(cells_of(U, K))}
            assert abs(q["delta"]["effect"] - (q["uar_pseudo"]["effect"] - q["uar_none"]["effect"])) < 1e-9
            assert abs(q["imitable_R0"]["effect"] - (q["uar_imitation_R0"]["effect"] - q["uar_none"]["effect"])) < 1e-9
            assert abs(q["uar_imitation_R1"]["effect"] - q["uar_pseudo"]["effect"]) < 1e-9          # R1 の予測 = 目標なら UAR は同じ
            assert abs(q["imitable_R1"]["effect"] - q["delta"]["effect"]) < 1e-9
            assert q["delta"]["lo"] <= q["delta"]["effect"] <= q["delta"]["hi"]
    print("selftest（実験 22）：差の量 4 項目と区間の順序がすべて通った")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "units", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="試しの置き場だけ：格子のそろいと汚れと実験 14 との照合を省いて summary を作る")
    args = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
    if args.stage == "selftest":
        selftest()
        return
    if set(DD.CORPORA) != {"AFEW", "IEMOCAP", "MELD", "MOSEI"}:
        sys.exit(f"評価の回のコーパスが 4 つでない：{DD.CORPORA}")
    out = Path(args.out_dir).resolve()
    prod = out == OUT_PROD.resolve()
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    RUN["partial"] = bool(args.allow_partial)
    if dirty and (prod or not args.allow_dirty):
        sys.exit("exp・measure・features に未コミットの変更がある（試走は一時の置き場で --allow-dirty）")
    if prod and not dai1.committed_clean(DESIGN):
        sys.exit("設計の文書がコミットされていない")
    if prod and (args.folds is not None or args.seeds is not None or args.allow_partial):
        sys.exit("本番の置き場では分割と種を絞らない")
    dai1.ENC["enc"] = DD.ENC1
    fz1, fz1_sha = dai3.load_fz1(OUT1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "units").mkdir(exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                host=platform.node(), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""),
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                scripts=script_shas(), design_sha256=D14.sha(DESIGN), dai1_frozen_sha256=fz1_sha, repro_saved_sha256=D14.sha(REPRO_SAVED),
                cond14_sha256=D14.sha(COND14), tol_repro=TOL_REPRO, tol_pseudo=TOL_PSEUDO, mem_frac=MEM_FRAC, encoder=dai1.ENC["enc"],
                KS=KS, student=STU, grid_c=GRID_C, cb_r1=D14.CB_R1, dim_r1=D14.DIM_R1, edge_share=EDGE_SHARE, rng_seed=RNG_SEED,
                n_boot=ST.N_BOOT, min_n=ST.MIN_TEST)
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "units":
        rp = pd.read_csv(REPRO_SAVED, low_memory=False)
        rp = rp[(rp.student == STU) & (rp.subset == "all") & rp.K.isin(KS)]
        allrows = []
        for T in args.targets:
            fzT = fz1[T]
            seeds = list(range(int(fzT["n_seeds_eval"]))) if args.seeds is None else args.seeds
            folds = list(range(DD.N_FOLDS)) if args.folds is None else args.folds
            for f in folds:
                for r in seeds:
                    df = run_unit(T, f, r, fzT, out, rp)
                    if df is None:
                        print(f"{T}_f{f}_r{r}：済み（飛ばす）", flush=True)
                        continue
                    chk = df[df.readout == "R0"]
                    bad = chk[chk.abs_diff.isna() | (chk.abs_diff > TOL_REPRO)]
                    print(f"{T}_f{f}_r{r}：{df.sec.iloc[0]:.0f} 秒．実験 3 の保存との照合 {len(chk)} 行，最大差 {chk.abs_diff.max():.2e}，"
                          f"許しの外 {len(bad)}", flush=True)
                    if len(bad):
                        bad.to_csv(out / f"check_fail_{T}_f{f}_r{r}.csv", index=False)
                        sys.exit(f"{T}_f{f}_r{r}：実験 3 の保存の一致と {TOL_REPRO} を超えて違う行がある")
                    allrows.append(chk)
            dai1.Corpus.cache.pop(T, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if allrows:
            pd.concat(allrows).to_csv(out / f"check_saved_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
        if prod and set(args.targets) == set(DD.CORPORA):
            (out / "units.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    else:
        stage_summary(out, args.targets)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)
        if prod and set(args.targets) == set(DD.CORPORA):
            (out / "summary.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
