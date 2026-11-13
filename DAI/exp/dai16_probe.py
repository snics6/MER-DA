"""実験 16 の下調べ（DAI-16）：モダリティごと・感情ごとの信頼度の表（取り違えの表）で声・顔・言葉の予測を読み替えてから掛け合わせた
疑似ラベルが，読み替えない積（今の提案の疑似ラベル）より当たるかを，実験 1 の仮のターゲットの回の保存だけで測る．
設計：DAI/docs/実験16_下調べ_設計.md．評価の回の保存は開かない．生徒の学び直しはしない（新しい学習は K 本の交差の推定の作り直しだけ）．

段：
  selftest  人工の数での検算（表の形，読み替え，単位行列で積に戻ること，取り違えが戻ること）
  units     120 単位（T × 仮のターゲット c × 分割 × 種）．単位ごとに <out>/units/<名前>.npz と .csv
  summary   c × K ごとの UAR，差と区間，判定の数え

使い方：
  .venv/bin/python DAI/exp/dai16_probe.py --stage selftest
  .venv/bin/python DAI/exp/dai16_probe.py --stage units
  .venv/bin/python DAI/exp/dai16_probe.py --stage summary
試走：--out-dir <一時の置き場> --allow-dirty --allow-partial --targets AFEW --folds 0 --seeds 0
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
import dai12_probe as P12  # noqa: E402
import dai15  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT_PROD = ROOT / "results" / "dai16_probe"
DESIGN = ROOT / "DAI" / "docs" / "実験16_下調べ_設計.md"
KS = (5, 25)
K_MAIN = 5
NF = dai15.NF                           # K 本を感情ごとに分ける組の数（5）
D_MIN = 0.25                            # 表の対角の平均がこれ以下のモダリティは使わない（一様な表）
EPS = 1e-9                              # 読み替えの前に表の全成分に足す数
LAMS = (0.0, 0.5)                       # 交差の表の縮小（形 D・F を形 S へ）
CHECK_TEACHER = dai15.CHECK_TEACHER     # K 本を全部使って作り直した積と保存の積の差の許し
FLOOR = 1.0                             # 「上回る」「下回る」の点推定の下限（点）
N_UNITS = 120
SEED = 20261113
MODS = "AVT"
SCRIPTS = ("dai16_probe.py", "dai15.py", "dai12_probe.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py", "dai_data.py")
VERSIONS = ("prod", "orc_S", "orc_D", "orc_F", "orc_F_hard", "cf_S", "cf_D_l0", "cf_D_l05", "cf_F_l0", "cf_F_l05", "em1_S", "em1_D", "em1_F")
REFS = ("none_AVT", "joint_AVT", "ret_A", "ret_V", "ret_T")
MAIN = {"R1": ("orc_F", "orc_D", "orc_S"), "R2": ("cf_D_l05",), "R3": ("em1_D",)}
EXTRA_CONTRASTS = [("orc_D", "orc_S"), ("orc_F", "orc_D"), ("cf_D_l05", "cf_S"), ("joint_AVT", "none_AVT")]
PER_EMO = ("prod", "orc_D", "orc_F", "cf_D_l05", "em1_D", "none_AVT")
RUN = {"commit": "", "dirty": True}


# ------------------------------------------------------------------ 表の形と読み替え

def shape_S(C: np.ndarray) -> np.ndarray:
    """モダリティごとに 1 つの数：対角を対角の平均に，非対角を等しくした表．"""
    return dai15.shrink(C, 1.0)


def shape_D(C: np.ndarray) -> np.ndarray:
    """感情ごとの対角：対角はそのまま，行 i の非対角を (1 − C_ii)/3 にした表．"""
    d = np.diag(C)
    out = np.repeat(((1.0 - d) / 3.0)[:, None], 4, axis=1)
    np.fill_diagonal(out, d)
    return out


def shaped(C: np.ndarray, shape: str, lam: float = 0.0) -> tuple[np.ndarray, bool]:
    """表 C を形（S・D・F）にし，λ で形 S へ縮める．対角の平均が D_MIN 以下なら一様な表（そのモダリティは掛け合わせに寄与しない）．
    戻り：(表，使わない条件に当たったか)．"""
    if float(np.mean(np.diag(C))) <= D_MIN:
        return np.full((4, 4), 0.25), True
    base = {"S": shape_S(C), "D": shape_D(C), "F": C}[shape]
    return (1.0 - lam) * base + lam * shape_S(C), False


def posterior_matrix(C: np.ndarray) -> np.ndarray:
    """B[j, i] = C[i, j] / Σ_i′ C[i′, j]（感情の事前の割合は等しい．「j と答えたとき本当が i である確率」）．行 j の和は 1．"""
    Ce = C + EPS
    Ce = Ce / Ce.sum(1, keepdims=True)
    return (Ce / Ce.sum(0, keepdims=True)).T


def reread(P: np.ndarray, C: np.ndarray, hard: bool = False) -> np.ndarray:
    """読み替えた確率 r(i | x) = Σ_j p(j | x) B[j, i]．hard なら p の最大の感情だけを使う．"""
    B = posterior_matrix(C)
    if hard:
        return B[P.argmax(1)]
    return P @ B


def combine(Ps: dict, Cs: dict, hard: bool = False) -> np.ndarray:
    """q(i | x) ∝ Π_m r_m(i | x)．戻りは正規化した確率 (n, 4)．"""
    L = sum(np.log(np.clip(reread(Ps[m], Cs[m], hard), 1e-300, None)) for m in MODS)
    L = L - L.max(1, keepdims=True)
    q = np.exp(L)
    return q / q.sum(1, keepdims=True)


def hard_table(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """判定（最大の感情）の割合で数えた表．無い感情の行は単位行列の行．"""
    return dai15.table_from(np.eye(4)[P.argmax(1)], y)


def em1_table(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """仮のラベルの重み R (n, 4) で数えた表：C[i, j] = Σ_x R(x, i) p(j | x) / Σ_x R(x, i)．"""
    num = R.T @ P
    return num / np.clip(R.sum(0)[:, None], 1e-12, None)


# ------------------------------------------------------------------ 交差の推定（実験 1 の段 1・2 と同じ手順．dai15.teachers のモダリティごとの版）

def teachers_mod(u, tether: dict, K: int, kid_sets: list) -> tuple[dict, list]:
    """K 本の部分集合ごとに，融合の K 本だけの分類器 → その確率で声・顔・言葉を学び直す．戻り：({m: 候補の確率 (部分集合, n, 4)}, [勾配の最大, 止まり])．"""
    n = u.n
    G0 = torch.stack([S.balanced_onehot(u.ypool, ks, n) for ks in kid_sets])
    W0, b0 = u.start["AVT"]
    cw, cb = tether[("AVT", K)]
    r = S.fit(u.Xp["AVT"], G0, W0, b0, Cw=cw, Cb=cb)
    if not r["finite_p"].all():
        raise SystemExit("交差の推定の作り直し（融合）が有限でない")
    conv = [float(r["grad_max"]), int(r["stalled"])]
    F = torch.softmax(S.logits(u.Xp["AVT"], r["W"], r["b"]), -1)
    out = {}
    for m in MODS:
        G = G0 + torch.stack([S.soft_rows(F[j], float(n)) for j in range(len(kid_sets))])
        W0, b0 = u.start[m]
        cw, cb = tether[(m, K)]
        rm = S.fit(u.Xp[m], G, W0, b0, Cw=cw, Cb=cb)
        if not rm["finite_p"].all():
            raise SystemExit(f"交差の推定の作り直し（{m}）が有限でない")
        conv = [max(conv[0], float(rm["grad_max"])), conv[1] + int(rm["stalled"])]
        out[m] = torch.softmax(S.logits(u.Xp[m], rm["W"], rm["b"]), -1)
    return out, conv


# ------------------------------------------------------------------ 1 単位

def run_unit(sv) -> tuple[list, dict]:
    u = sv.u
    TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
    rows, arr = [], {}
    yt, yp = u.ytest, u.ypool
    arr["y_test"] = yt.astype(np.int8)
    arr["cl_test"] = sv.cl
    arr["sid_test"] = sv.sid
    for K in KS:
        kpos = u.kpos[K]
        yk = yp[kpos]
        nk = np.setdiff1d(np.arange(u.n), kpos)
        Pp = {m: TK[K][f"ret_{m}"][0].cpu().numpy().astype(np.float64) for m in MODS}
        Pt = {m: TK[K][f"ret_{m}"][1].cpu().numpy().astype(np.float64) for m in MODS}
        q0p = dai1.product(TK[K], (1, 1, 1), 0).cpu().numpy().astype(np.float64)
        q0t = dai1.product(TK[K], (1, 1, 1), 1).cpu().numpy().astype(np.float64)
        # 交差の推定
        fold = dai15.kfold_of(yk)
        sets = [kpos] + [kpos[fold != j] for j in range(NF)]
        Pm, conv = teachers_mod(u, sv.teth, K, sets)
        Lfull = sum(Pm[m][0].clamp(min=1e-12).log() for m in MODS)
        dev = float(np.abs(torch.softmax(Lfull, -1).cpu().numpy() - q0p).max())
        if dev > CHECK_TEACHER:
            raise SystemExit(f"{sv.base.name}：K 本を全部使って作り直した積が保存の積と {dev:.2e} 違う（K={K}）")
        pcf = {m: np.empty((len(kpos), 4)) for m in MODS}
        for j in range(NF):
            held = np.flatnonzero(fold == j)
            idx = torch.as_tensor(kpos[held], device=S.DEV)
            for m in MODS:
                pcf[m][held] = Pm[m][1 + j][idx].cpu().numpy()
        # 表
        R = q0p.copy()
        R[kpos] = np.eye(4)[yk]
        C = {"orc": {m: dai15.table_from(Pp[m][nk], yp[nk]) for m in MODS},
             "cf": {m: dai15.table_from(pcf[m], yk) for m in MODS},
             "em1": {m: em1_table(Pp[m], R) for m in MODS}}
        Chard = {m: hard_table(Pp[m][nk], yp[nk]) for m in MODS}
        for src in C:
            for m in MODS:
                arr[f"C_{src}_{m}_K{K}"] = C[src][m]
        for m in MODS:
            arr[f"C_orchard_{m}_K{K}"] = Chard[m]
            arr[f"pcf_{m}_K{K}"] = pcf[m].astype(np.float32)
        arr[f"kpos_K{K}"] = kpos.astype(np.int32)
        arr[f"yk_K{K}"] = yk.astype(np.int8)
        arr[f"y_pool_K{K}"] = yp[nk].astype(np.int8)
        arr[f"cl_pool_K{K}"] = sv.cl_pool[nk]
        # 版ごとの表
        specs = {"orc_S": ("orc", "S", 0.0), "orc_D": ("orc", "D", 0.0), "orc_F": ("orc", "F", 0.0),
                 "cf_S": ("cf", "S", 0.0), "cf_D_l0": ("cf", "D", 0.0), "cf_D_l05": ("cf", "D", 0.5), "cf_F_l0": ("cf", "F", 0.0), "cf_F_l05": ("cf", "F", 0.5),
                 "em1_S": ("em1", "S", 0.0), "em1_D": ("em1", "D", 0.0), "em1_F": ("em1", "F", 0.0)}
        preds_t, preds_p, unused = {"prod": q0t.argmax(1)}, {"prod": q0p[nk].argmax(1)}, {"prod": 0}
        I4 = {m: np.eye(4) for m in MODS}
        chk = float(np.abs(combine(Pt, I4) - q0t).max())                    # 単位行列の表で積に戻ること（EPS の分だけずれる）
        if chk > 1e-4:
            raise SystemExit(f"{sv.base.name}：単位行列の表の読み替えが積と {chk:.2e} 違う（K={K}）")
        for v, (src, shp, lam) in specs.items():
            Cs, nun = {}, 0
            for m in MODS:
                Cs[m], un = shaped(C[src][m], shp, lam)
                nun += int(un)
            preds_t[v] = combine(Pt, Cs).argmax(1)
            preds_p[v] = combine({m: Pp[m][nk] for m in MODS}, Cs).argmax(1)
            unused[v] = nun
        Cs, nun = {}, 0
        for m in MODS:
            Cs[m], un = shaped(Chard[m], "F")
            nun += int(un)
        preds_t["orc_F_hard"] = combine(Pt, Cs, hard=True).argmax(1)
        preds_p["orc_F_hard"] = combine({m: Pp[m][nk] for m in MODS}, Cs, hard=True).argmax(1)
        unused["orc_F_hard"] = nun
        # 参照
        Wj, bj = sv.prop("AVT", K)
        refs_t = {"none_AVT": TK[K]["none_AVT"][1].argmax(1).cpu().numpy(), "joint_AVT": S.logits(u.Xt["AVT"], Wj, bj).argmax(1).cpu().numpy()}
        refs_p = {"none_AVT": TK[K]["none_AVT"][0].argmax(1).cpu().numpy()[nk], "joint_AVT": S.logits(u.Xp["AVT"], Wj, bj).argmax(1).cpu().numpy()[nk]}
        for m in MODS:
            refs_t[f"ret_{m}"] = Pt[m].argmax(1)
            refs_p[f"ret_{m}"] = Pp[m][nk].argmax(1)
        for v in list(preds_t) + list(refs_t):
            pt = (preds_t | refs_t)[v]
            pp = (preds_p | refs_p)[v]
            arr[f"pred_test_K{K}_{v}"] = pt.astype(np.int8)
            arr[f"pred_pool_K{K}_{v}"] = pp.astype(np.int8)
            rows.append(dict(K=K, version=v, uar_test=100 * S.uar(yt, pt), uar_pool=100 * S.uar(yp[nk], pp), n_unused=unused.get(v, 0)))
        rows.append(dict(K=K, version="_check", teacher_dev=dev, identity_dev=chk, grad=conv[0], stalled=conv[1],
                         **{f"dbar_{src}_{m}": float(np.mean(np.diag(C[src][m]))) for src in C for m in MODS}))
    return rows, arr


# ------------------------------------------------------------------ 段

def stage_units(out: Path, targets: list, folds=None, seeds=None) -> None:
    fz1, _ = dai3.load_fz1(OUT1)
    udir = out / "units"
    udir.mkdir(parents=True, exist_ok=True)
    for T in targets:
        fzT = fz1[T]
        todo = dai3.pseudo_units(OUT1, T)
        if folds is not None:
            todo = [x for x in todo if x[1] in folds]
        if seeds is not None:
            todo = [x for x in todo if x[2] in seeds]
        for c, f, r in todo:
            name = f"{T}_{c}_f{f}_r{r}"
            if (udir / f"{name}.csv").exists():
                old = pd.read_csv(udir / f"{name}.csv")
                if (old.commit == RUN["commit"]).all():
                    print(f"{name}：済み（飛ばす）", flush=True)
                    continue
                raise SystemExit(f"{name} が別のコミットで既にある")
            t0 = time.time()
            sv = dai3.Saved(OUT1, "pseudo", T, c, f, r, fzT)
            rows, arr = run_unit(sv)
            df = pd.DataFrame(rows).assign(T=T, c=c, fold=f, seed=r, commit=RUN["commit"], dirty=RUN["dirty"], sec=time.time() - t0,
                                           dai1_commits=",".join(sorted(sv.commits)))
            np.savez_compressed(udir / f"{name}.npz", **arr)
            df.to_csv(udir / f"{name}.csv", index=False)
            ck = df[df.version == "_check"]
            print(f"{name}：{time.time() - t0:.0f} 秒．行 {len(df)}．作り直しの差 {ck.teacher_dev.max():.1e}．勾配の最大 {ck.grad.max():.1e}．止まり {int(ck.stalled.sum())}",
                  flush=True)
            del sv
        dai1.Corpus.cache.pop(T, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verdict(lo: float, hi: float, est: float) -> str:
    if lo > 0 and est >= FLOOR:
        return "上回る"
    if hi < 0 and est <= -FLOOR:
        return "下回る"
    return "言えない"


def stage_summary(out: Path, allow_partial: bool) -> None:
    files = sorted((out / "units").glob("*.npz"))
    if len(files) != N_UNITS and not allow_partial:
        raise SystemExit(f"単位の出力が {len(files)} 個（{N_UNITS} のはず）．試走なら --allow-partial")
    csvs = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in files], ignore_index=True)
    if csvs.commit.nunique() != 1 and not allow_partial:
        raise SystemExit(f"単位のコミットが 1 つでない：{csvs.commit.unique()}")
    rng = np.random.default_rng(SEED)
    names = VERSIONS + REFS
    cells = {"test": {}, "pool": {}}                                     # [where][(c, K, 版)][(f, r)] = 塊ごとの数（3 つの T の平均）
    emo = {}
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        f, r = int(f[1:]), int(r[1:])
        z = np.load(p, allow_pickle=False)
        for K in KS:
            for v in names:
                ct = ST.counts(z["y_test"].astype(int), z[f"pred_test_K{K}_{v}"].astype(int), z["cl_test"])
                cp = ST.counts(z[f"y_pool_K{K}"].astype(int), z[f"pred_pool_K{K}_{v}"].astype(int), z[f"cl_pool_K{K}"])
                P12._add(cells["test"].setdefault((c, K, v), {}), (f, r), ct, f"{p.stem} test {v}")
                P12._add(cells["pool"].setdefault((c, K, v), {}), (f, r), cp, f"{p.stem} pool {v}")
                if v in PER_EMO:
                    e = emo.setdefault((c, K, v), [np.zeros(4), np.zeros(4)])
                    e[0] += ct[1].sum(0); e[1] += ct[2].sum(0)
    for w in cells:
        P12._finish(cells[w])
    geff = {c: ST.g_eff_of(c)[0] for c in DD.CORPORA}
    sdir = out / "summary"
    sdir.mkdir(exist_ok=True)
    # 水準
    lev = []
    for (c, K, v), d in sorted(cells["test"].items()):
        ut = float(np.mean([np.mean([ST.uar_full(d[k]) for k in d if k[0] == f]) for f in sorted({k[0] for k in d})]))
        dp = cells["pool"][(c, K, v)]
        up = float(np.mean([np.mean([ST.uar_full(dp[k]) for k in dp if k[0] == f]) for f in sorted({k[0] for k in dp})]))
        lev.append(dict(c=c, K=K, version=v, uar_test=ut, uar_pool=up, n_units=len(d)))
    pd.DataFrame(lev).to_csv(sdir / "levels.csv", index=False)
    # 差
    contrasts = [(v, "prod") for v in VERSIONS if v != "prod"] + EXTRA_CONTRASTS
    contrasts += [(v, ref) for v in VERSIONS for ref in ("none_AVT", "joint_AVT")]
    eff = []
    for c in DD.CORPORA:
        for K in KS:
            for a, b in contrasts:
                for w in ("test", "pool"):
                    A, B = cells[w].get((c, K, a)), cells[w].get((c, K, b))
                    if A is None or B is None:
                        continue
                    if set(A) != set(B):
                        raise SystemExit(f"{c} K={K} {a} と {b} の単位がそろわない")
                    es = (P12.effect_stats_pool if w == "pool" else ST.effect_stats)(A, B, geff[c], rng)
                    eff.append(dict(c=c, K=K, a=a, b=b, where=w, est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], method=es["method"],
                                    n_units=es["n_units"], n_folds=es["n_folds"], verdict=verdict(es["lo"], es["hi"], es["effect"])))
    eff = pd.DataFrame(eff)
    eff.to_csv(sdir / "effects.csv", index=False)
    # 判定の数え（テスト．K ごと．K = 5 が主）
    crit = []
    for rule, vs in MAIN.items():
        for v in vs:
            for K in KS:
                d = eff[(eff.a == v) & (eff.b == "prod") & (eff["where"] == "test") & (eff.K == K)]
                up, dn = int((d.verdict == "上回る").sum()), int((d.verdict == "下回る").sum())
                crit.append(dict(rule=rule, version=v, K=K, main=(K == K_MAIN and v == vs[0]), n_corpora=len(d), n_up=up, n_down=dn,
                                 hit=bool(up >= 2), miss_down=bool(dn >= 2),
                                 up=",".join(d[d.verdict == "上回る"].c), down=",".join(d[d.verdict == "下回る"].c)))
    pd.DataFrame(crit).to_csv(sdir / "criteria.csv", index=False)
    # 感情ごとの再現率（テスト．単位をまとめた数から）
    pe = [dict(c=c, K=K, version=v, **{f"recall_{i}": 100 * e[0][i] / e[1][i] if e[1][i] > 0 else np.nan for i in range(4)},
               **{f"n_{i}": float(e[1][i]) for i in range(4)}) for (c, K, v), e in sorted(emo.items())]
    pd.DataFrame(pe).to_csv(sdir / "per_emotion.csv", index=False)
    # 表の成分の単位の平均
    tb = {}
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        z = np.load(p, allow_pickle=False)
        for K in KS:
            for src in ("orc", "cf", "em1", "orchard"):
                for m in MODS:
                    tb.setdefault((c, K, src, m), []).append(z[f"C_{src}_{m}_K{K}"])
    tr = []
    for (c, K, src, m), L in sorted(tb.items()):
        M = np.mean(L, 0)
        sd = np.std([np.diag(x) for x in L], 0)
        tr.append(dict(c=c, K=K, source=src, modality=m, n_units=len(L), **{f"C_{i}{j}": float(M[i, j]) for i in range(4) for j in range(4)},
                       **{f"sd_diag_{i}": float(sd[i]) for i in range(4)}))
    pd.DataFrame(tr).to_csv(sdir / "tables.csv", index=False)
    chk = csvs[csvs.version == "_check"]
    un = csvs[csvs.version != "_check"].groupby(["c", "K", "version"]).n_unused.sum().reset_index()
    un.to_csv(sdir / "unused.csv", index=False)
    json.dump(dict(n_units=len(files), teacher_dev_max=float(chk.teacher_dev.max()), identity_dev_max=float(chk.identity_dev.max()),
                   grad_max=float(chk.grad.max()), stalled=int(chk.stalled.sum()), sec_total=float(csvs.groupby(["T", "c", "fold", "seed"]).sec.first().sum())),
              open(sdir / "convergence.json", "w"), ensure_ascii=False, indent=1)
    print(f"summary：単位 {len(files)}，差の行 {len(eff)}，判定の行 {len(crit)}", flush=True)


# ------------------------------------------------------------------ 検算

def selftest() -> None:
    rng = np.random.default_rng(0)
    C = rng.dirichlet(np.ones(4) * 2, size=4) * 0.5 + np.eye(4) * 0.5
    # 1. 形：行の和 1．S は対角が等しい．D は対角を保ち非対角が行の中で等しい
    for shp in "SDF":
        for lam in (0.0, 0.5):
            M, un = shaped(C, shp, lam)
            assert np.allclose(M.sum(1), 1) and not un
    assert np.allclose(np.diag(shape_D(C)), np.diag(C))
    assert np.allclose(shape_D(C)[0, 1], shape_D(C)[0, 2]) and np.allclose(np.diag(shape_S(C)), np.mean(np.diag(C)))
    M, un = shaped(np.full((4, 4), 0.25), "F")
    assert un and np.allclose(M, 0.25)
    # 2. 読み替え：行の和 1．単位行列なら元の確率．一様な表なら一様
    P = rng.dirichlet(np.ones(4), size=50)
    assert np.allclose(reread(P, C).sum(1), 1)
    assert np.abs(reread(P, np.eye(4)) - P).max() < 1e-5
    assert np.allclose(reread(P, np.full((4, 4), 0.25)), 0.25)
    Ps = {m: rng.dirichlet(np.ones(4), size=50) for m in MODS}
    q = combine(Ps, {m: np.eye(4) for m in MODS})
    L = sum(np.log(Ps[m]) for m in MODS)
    q0 = np.exp(L - L.max(1, keepdims=True)); q0 /= q0.sum(1, keepdims=True)
    assert np.abs(q - q0).max() < 1e-4
    # 3. 一様な表のモダリティは判定を変えない
    q2 = combine(Ps, {"A": np.eye(4), "V": np.eye(4), "T": np.full((4, 4), 0.25)})
    L2 = np.log(Ps["A"]) + np.log(Ps["V"])
    assert (q2.argmax(1) == L2.argmax(1)).mean() > 0.98
    # 4. 取り違えが戻る：モダリティ A は感情 0 を 1 と答えやすい．正確な表で読み替えた組み合わせは積より当たる
    n = 6000
    y = rng.integers(0, 4, n)
    Ct = {"A": np.array([[0.30, 0.60, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]]),
          "V": np.full((4, 4), 0.15) + np.eye(4) * 0.40, "T": np.full((4, 4), 0.15) + np.eye(4) * 0.40}
    Pv = {}
    for m in MODS:
        vote = np.array([rng.choice(4, p=Ct[m][i]) for i in y])
        Pv[m] = np.full((n, 4), 0.1) + np.eye(4)[vote] * 0.6
    Ctab = {m: dai15.table_from(Pv[m], y) for m in MODS}
    u0 = S.uar(y, combine(Pv, {m: np.eye(4) for m in MODS}).argmax(1))
    u1 = S.uar(y, combine(Pv, Ctab).argmax(1))
    assert u1 > u0 + 0.01, (u0, u1)
    # 5. 仮のラベルの表：R が本当のラベルの one-hot なら全ラベルの表と同じ
    assert np.allclose(em1_table(Pv["A"], np.eye(4)[y]), dai15.table_from(Pv["A"], y))
    # 6. 判定の語
    assert verdict(0.1, 2.0, 1.2) == "上回る" and verdict(0.1, 1.0, 0.5) == "言えない" and verdict(-3, -0.1, -1.5) == "下回る"
    print("selftest：通った")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "units", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="試しの置き場だけ：単位のそろいの確かめを省く")
    args = ap.parse_args()
    if args.stage == "selftest":
        selftest()
        return
    out = Path(args.out_dir).resolve()
    prod = out == OUT_PROD.resolve()
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    if dirty and (prod or not args.allow_dirty):
        sys.exit("DAI/exp・measure・features に未コミットの変更がある（試走は一時の置き場で --allow-dirty）")
    if prod and not dai1.committed_clean(DESIGN):
        sys.exit("設計の文書がコミットされていない")
    if prod and (args.folds is not None or args.seeds is not None or set(args.targets) != set(DD.CORPORA) or args.allow_partial):
        sys.exit("本番の置き場では対象を絞らない")
    dai1.ENC["enc"] = DD.ENC1
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""), torch=torch.__version__, numpy=np.__version__,
                python=platform.python_version(), pandas=pd.__version__, scripts={n: dai15.sha(HERE / n) for n in SCRIPTS},
                design_sha256=dai15.sha(DESIGN), dai1_frozen_sha256=dai3.load_fz1(OUT1)[1], KS=KS, NF=NF, D_MIN=D_MIN, EPS=EPS, LAMS=LAMS,
                CHECK_TEACHER=CHECK_TEACHER, FLOOR=FLOOR, VERSIONS=VERSIONS, REFS=REFS, MAIN=MAIN, rng_seed=SEED)
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "units":
        stage_units(out, args.targets, args.folds, args.seeds)
        if prod:
            (out / "units.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    else:
        stage_summary(out, args.allow_partial)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
