"""実験 14（DAI-14）：どのモダリティの特徴が，ほかのモダリティの判断を再現できるか（設計 `DAI/docs/実験14_設計.md`）．

本番の評価の回（実験 1．results/dai1/eval）の保存から，K = 5 の各モダリティの K 本だけの分類器の判断 J_m と，提案の主の版の
疑似ラベル J_prop を候補の上で作り直し，生徒 s（A・V・T）の特徴から線形に再現できる度合い R(s ← m) と，本当の感情が同じ候補の
間で目標を入れ替えた対照 R_shuf を測る（実験 3 の 2 節の再現できる度合いと同じ作り）．記述として，当たる幅 Δ_s と評価の回の
真似できる分 P_s も出す．

段：
  selftest  合成の数で区間の関数と一致の関数を確かめる（GPU 不要）
  units     単位ごとの計算．results/dai14/units/<T>_f<f>_r<r>.npz と .csv
  summary   区間と予言の判定．results/dai14/summary/*.csv
使い方（本番）：
  python DAI/exp/dai14.py --stage units
  python DAI/exp/dai14.py --stage summary
試走：--out-dir に一時の置き場と --allow-dirty，--targets/--folds/--seeds で絞る．
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai3  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT_PROD = ROOT / "results" / "dai14"
K = 5
STU = ("A", "V", "T")
MOD_NAME = {"A": "声", "V": "顔", "T": "言葉"}
N_BOOT = ST.N_BOOT                     # 2,000
MIN_N = ST.MIN_TEST                    # 30 本未満は値を作らない（実験 1 と同じ）
RNG_SEED = 20261112
SCRIPTS = ("dai14.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py", "dai_data.py")
DESIGN = ROOT / "DAI" / "docs" / "実験14_設計.md"
REPRO_SAVED = ROOT / "results" / "dai3" / "eval" / "repro_units.csv"
TOL_REPRO = 1e-3                       # 保存の再現できる度合い（実験 3）との照合の許し（UAR のポイント）
RUN = {"commit": "", "dirty": True}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def script_shas() -> dict:
    return {n: sha(HERE / n) for n in SCRIPTS}


# ------------------------------------------------------------------ 一致の割合と区間（設計 4 節）

def counts(y: np.ndarray, pred: np.ndarray, cl: np.ndarray):
    """塊ごとの数：(塊の名前の並び, 感情ごとの一致の数 (塊, 4), 感情ごとの件数 (塊, 4))．y は「正解」とみなす目標
    （判断 J_m，入れ替えた目標，本当の感情のどれか）．dai1_stats.counts と同じ形．"""
    return ST.counts(np.asarray(y, int), np.asarray(pred, int), np.asarray(cl).astype(str))


def uar_present(cor: np.ndarray, tot: np.ndarray) -> np.ndarray:
    """(…, 4) の数 → 現れた感情（件数 > 0）だけの再現率の平均（%）．件数の合計が MIN_N 未満なら NaN．
    全体の値・塊を除いた値・引き直しの値のすべてでこの式を使う（目標の判断がある感情を出さない単位でも NaN にしない）．"""
    cor = np.asarray(cor, float); tot = np.asarray(tot, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(tot > 0, cor / np.where(tot > 0, tot, 1.0), np.nan)
        u = 100.0 * np.nanmean(rec, axis=-1)
    return np.where(tot.sum(-1) < MIN_N, np.nan, u)


def theta(d: dict) -> float:
    """{f: [単位の値]} → 分割ごとに種で平均してから分割で平均（空と NaN は除く）．"""
    vals = [np.nanmean(v) for v in d.values() if np.isfinite(v).any()]
    return float(np.mean(vals)) if vals else float("nan")


def combo_stats(cells: list, coefs: list, g_eff: float, rng) -> dict:
    """Σ coefs[i] · UAR(cells[i]) の区間．cells[i]：{(f, r): counts の戻り}．全部の cells は同じ (f, r) の鍵と，
    単位ごとに同じ塊の並びを持つ（同じ候補の同じ発話の上の数）．
      g_eff < 20：塊を全分割から同時に除く jackknife と種の項（dai1_stats.effect_stats と同じ式．自由度 max(1, ⌊g_eff⌋ − 1)）．
      g_eff ≥ 20：全分割で共通の塊の重みを 1 組引く引き直し N_BOOT 回と種の項（実験 12 の effect_stats_pool と同じ式）．"""
    keys = sorted(cells[0])
    for c in cells[1:]:
        if sorted(c) != keys:
            raise SystemExit("combo_stats：cells の (分割, 種) の鍵がそろっていない")
    for k in keys:
        n0 = cells[0][k][0]
        for c in cells[1:]:
            if len(c[k][0]) != len(n0) or not (c[k][0] == n0).all():
                raise SystemExit(f"combo_stats：単位 {k} で塊の並びが cells の間で違う")
    folds = sorted({f for f, _ in keys})
    rs_of = {f: [r for ff, r in keys if ff == f] for f in folds}

    def val_full(k):
        return float(sum(a * uar_present(c[k][1].sum(0), c[k][2].sum(0)) for c, a in zip(cells, coefs)))

    d = {f: [val_full((f, r)) for r in rs_of[f]] for f in folds}
    est = theta(d)
    R = {f: int(np.isfinite(d[f]).sum()) for f in folds}
    s2 = {f: (float(np.nanvar(d[f], ddof=1)) if R[f] > 1 else 0.0) for f in folds}
    v_seed = float(sum(s2[f] / max(R[f], 1) for f in folds) / len(folds) ** 2)
    base = dict(effect=est, n_folds=len(folds), n_units=int(sum(R.values())), g_eff=g_eff)
    if not np.isfinite(est):
        return base | dict(lo=np.nan, hi=np.nan, se=np.nan, df=np.nan, method="", v_cl=np.nan, v_seed=v_seed)
    if g_eff >= 20:
        all_cl = sorted({nm for k in keys for nm in cells[0][k][0]})
        pos = {nm: i for i, nm in enumerate(all_cl)}
        G = len(all_cl)
        Wall = rng.multinomial(G, np.full(G, 1.0 / G), size=N_BOOT).astype(float)        # (B, G)：全分割で共通の 1 組
        boots = np.zeros(N_BOOT); nval = np.zeros(N_BOOT)
        for f in folds:
            per_r = []
            for r in rs_of[f]:
                k = (f, r)
                Wf = Wall[:, [pos[nm] for nm in cells[0][k][0]]]
                per_r.append(sum(a * uar_present(Wf @ c[k][1], Wf @ c[k][2]) for c, a in zip(cells, coefs)))
            dm = np.nanmean(np.stack(per_r), axis=0)
            ok = np.isfinite(dm)
            boots[ok] += dm[ok]; nval[ok] += 1
        boots = np.where(nval > 0, boots / np.maximum(nval, 1), np.nan)
        boots = boots[np.isfinite(boots)]
        v_cl = float(np.var(boots, ddof=1))
        se = float(np.sqrt(v_cl + v_seed))
        return base | dict(lo=est - 1.96 * se, hi=est + 1.96 * se, se=se, df=np.nan, method="boot_cl_shared+seed",
                           v_cl=v_cl, v_seed=v_seed, n_clusters=G)
    all_cl = sorted({nm for k in keys for nm in cells[0][k][0]})
    jk = []
    for g in all_cl:
        dg = {}
        for f in folds:
            vals = []
            for r in rs_of[f]:
                k = (f, r)
                names = cells[0][k][0]
                hit = np.flatnonzero(names == g)
                if len(hit):
                    i = int(hit[0])
                    v = sum(a * uar_present(c[k][1].sum(0) - c[k][1][i], c[k][2].sum(0) - c[k][2][i]) for c, a in zip(cells, coefs))
                else:
                    v = val_full(k)
                vals.append(float(v))
            dg[f] = vals
        jk.append(theta(dg))
    jk = np.array(jk, float)
    jk = jk[np.isfinite(jk)]
    G = len(jk)
    v_cl = float((G - 1) / G * ((jk - jk.mean()) ** 2).sum()) if G > 1 else np.nan
    se = float(np.sqrt(v_cl + v_seed)) if np.isfinite(v_cl) else np.nan
    df = max(1, int(np.floor(g_eff)) - 1)
    q = float(st.t.ppf(0.975, df))
    return base | dict(lo=est - q * se, hi=est + q * se, se=se, df=df, method="jk+seed", v_cl=v_cl, v_seed=v_seed, n_clusters=G)


def verdict(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "判定できない"
    if lo > 0:
        return "上回る"
    if hi < 0:
        return "下回る"
    return "判定できない"


# ------------------------------------------------------------------ 段 units

def shuffle_within(qy: np.ndarray, y: np.ndarray, seed: list) -> np.ndarray:
    """本当の感情 y が同じ候補の間で目標 qy を入れ替える（感情ごとの qy の分布を保ち，発話ごとの対応だけを壊す）．
    乱数は (T, f, r, 目標の名前) だけで決める（生徒に依らない．同じ判断の入れ替えは生徒の間で同じ）．"""
    rng = np.random.default_rng(seed)
    out = qy.copy()
    for e in range(4):
        ii = np.flatnonzero(y == e)
        if len(ii) > 1:
            out[ii] = qy[rng.permutation(ii)]
    return out


def name_code(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def run_unit(T: str, f: int, r: int, fz1T: dict, out: Path) -> pd.DataFrame:
    base = out / "units" / f"{T}_f{f}_r{r}"
    if Path(f"{base}.npz").exists() or Path(f"{base}.csv").exists():
        raise SystemExit(f"{base.name} は既にある（重複を作らない．消してから流し直す）")
    t0 = time.time()
    sv = dai3.Saved(OUT1, "eval", T, T, f, r, fz1T)
    u = sv.u
    tk = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)[K]
    main = "prop_" + fz1T["main"][str(K)]
    n = u.n
    y = np.asarray(u.ypool, int)
    h = dai3.pool_halves(sv.cl_pool)
    halves = [np.flatnonzero(h == 0), np.flatnonzero(h == 1)]
    nk = np.ones(n, bool); nk[u.kpos[K]] = False
    shared = set(sv.dlg_pool[halves[0]]) & set(sv.dlg_pool[halves[1]]) if T in ("MELD",) else set()
    nonshared = ~np.isin(sv.dlg_pool, list(shared)) if shared else np.ones(n, bool)
    arr = dict(y=y.astype(np.int8), cl=sv.cl_pool.astype(str), sid=sv.sid_pool.astype(str), half=h.astype(np.int8),
               nk=nk, nonshared=nonshared, main=np.array(main))
    rows = []
    tgt_cache = {}
    for s in STU:
        names = [f"single_{m}" for m in "AVT" if m != s] + [main]
        Q = dai3.pool_teachers(sv, tk, s, K, names)
        tg = {}
        for nm in names:
            if nm not in tgt_cache:
                qy = Q[nm].argmax(1).cpu().numpy().astype(int)
                qs = shuffle_within(qy, y, [RNG_SEED, DD.CORPUS_ID[T], f, r, name_code(nm)])
                tgt_cache[nm] = (qy, qs)
            else:                                                     # 同じ判断は生徒に依らず同じであることを確かめる
                qy = Q[nm].argmax(1).cpu().numpy().astype(int)
                if not (qy == tgt_cache[nm][0]).all():
                    raise SystemExit(f"{base.name}：目標 {nm} が生徒 {s} で作り直すと違う")
            tg[nm], tg[nm + "#shuf"] = tgt_cache[nm]
        keys = list(tg)
        G = torch.stack([S.balanced_onehot(tg[k], halves[hh], n) for k in keys for hh in (0, 1)])
        W0, b0 = u.start[s]
        cw, cb = sv.teth[(s, K)]
        res = S.fit(u.Xp[s], G, W0, b0, Cw=cw, Cb=cb)
        pred = S.logits(u.Xp[s], res["W"], res["b"]).argmax(2).cpu().numpy()        # (2 · 目標の数, n)
        Wn, bn = sv.none(s, K)
        pnone = S.logits(u.Xp[s], Wn, bn).argmax(1).cpu().numpy()
        arr[f"none_{s}"] = pnone.astype(np.int8)
        for i, k in enumerate(keys):
            full = np.empty(n, np.int8)
            ag = []
            for hh in (0, 1):
                other = halves[1 - hh]
                full[other] = pred[2 * i + hh][other]
                ag.append(100 * S.uar(tg[k][other], pred[2 * i + hh][other]))
            arr[f"pred_{s}_{k}"] = full
            arr[f"tgt_{s}_{k}"] = tg[k].astype(np.int8)
            rows.append(dict(T=T, fold=f, seed=r, student=s, K=K, target=k,
                             agree_halfmean=float(np.mean(ag)),
                             agree_pooled=float(uar_present(*[x.sum(0) for x in counts(tg[k], full, sv.cl_pool)[1:]])),
                             grad=float(max(res["grad_p"][2 * i], res["grad_p"][2 * i + 1])),
                             stalled=bool(res["stalled_p"][2 * i] or res["stalled_p"][2 * i + 1]),
                             finite=bool(res["finite_p"][2 * i] and res["finite_p"][2 * i + 1]),
                             n_half0=len(halves[0]), n_half1=len(halves[1]), n_pool=n, n_nk=int(nk.sum()),
                             n_shared_dialogues=len(shared)))
    df = pd.DataFrame(rows)
    df["commit"], df["dirty"], df["sec"] = RUN["commit"], RUN["dirty"], time.time() - t0
    np.savez_compressed(f"{base}.npz", **arr)
    df.to_csv(f"{base}.csv", index=False)
    return df


def check_against_saved(df: pd.DataFrame) -> pd.DataFrame:
    """実験 3 の保存の再現できる度合い（2 つの組の平均．subset=all，K=5）と照合する．判断 single_m と主の版の疑似ラベル．"""
    sv_ = pd.read_csv(REPRO_SAVED, low_memory=False)
    sv_ = sv_[(sv_.K == K) & (sv_.subset == "all")]
    m = df[~df.target.str.endswith("#shuf")].merge(
        sv_[["T", "fold", "seed", "student", "teacher", "agree"]].rename(columns={"teacher": "target"}),
        on=["T", "fold", "seed", "student", "target"], how="left")
    m["abs_diff"] = (m.agree_halfmean - m.agree).abs()
    return m


# ------------------------------------------------------------------ 段 summary

def load_units(out: Path, T: str):
    U = {}
    for p in sorted((out / "units").glob(f"{T}_f*_r*.npz")):
        f, r = (int(x[1:]) for x in p.stem.split("_")[1:3])
        U[(f, r)] = dict(np.load(p, allow_pickle=False))
    return U


def cell_R(U: dict, s: str, tgt: str, subset: str = "all") -> dict:
    """R(s ← tgt)：目標 tgt を正解とした，生徒 s の特徴の交差の予測の一致（候補全体．subset=nonshared は共有の対話を除く）．"""
    out = {}
    for k, a in U.items():
        msk = a["nonshared"] if subset == "nonshared" else np.ones(len(a["y"]), bool)
        out[k] = counts(a[f"tgt_{s}_{tgt}"][msk], a[f"pred_{s}_{tgt}"][msk], a["cl"][msk])
    return out


def cell_true(U: dict, pred_key: str) -> dict:
    """本当の感情に対する判定 pred_key の数（候補のうち K 本を除く発話）．"""
    out = {}
    for k, a in U.items():
        m = a["nk"].astype(bool)
        out[k] = counts(a["y"][m], a[pred_key][m], a["cl"][m])
    return out


def stage_summary(out: Path, targets: list) -> None:
    sdir = out / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    desc, pred_rows, cond, conv = [], [], [], []
    for T in targets:
        U = load_units(out, T)
        if not U:
            raise SystemExit(f"{T}：単位の出力が無い")
        csvs = pd.concat([pd.read_csv(out / "units" / f"{T}_f{f}_r{r}.csv") for f, r in U])
        if csvs.commit.nunique() != 1:
            raise SystemExit(f"{T}：単位のコミットが 1 つでない")
        conv.append(csvs.assign(T=T))
        g_eff, _ = ST.g_eff_of(T)
        rng = np.random.default_rng([RNG_SEED, DD.CORPUS_ID[T]])
        main = str(next(iter(U.values()))["main"])
        # 記述：3×3 の R，R_shuf，D（主の版の疑似ラベルの行も）
        for s in STU:
            for m in [x for x in "AVT" if x != s] + ["prop"]:
                tgt = f"single_{m}" if m != "prop" else main
                for subset in ("all", "nonshared"):
                    if subset == "nonshared" and T not in ("MELD",):
                        continue
                    cR, cS = cell_R(U, s, tgt, subset), cell_R(U, s, tgt + "#shuf", subset)
                    for qn, cells, co in (("R", [cR], [1]), ("R_shuf", [cS], [1]), ("D", [cR, cS], [1, -1])):
                        e = combo_stats(cells, co, g_eff, rng)
                        desc.append(dict(T=T, student=s, judge=m, target=tgt, subset=subset, quantity=qn) | e |
                                    dict(verdict=verdict(e["lo"], e["hi"])))
        # 予言 P1〜P4（設計 5 節）
        R = {(s, m): cell_R(U, s, f"single_{m}") for s in STU for m in "AVT" if m != s}
        Rs = {(s, m): cell_R(U, s, f"single_{m}#shuf") for s in STU for m in "AVT" if m != s}
        P = {
            "P1": ([R[("A", "T")], R[("V", "T")]], [1, -1], "R(A←T) − R(V←T)"),
            "P2": ([R[("A", "V")], R[("T", "V")]], [1, -1], "R(A←V) − R(T←V)"),
            "P3": ([R[("A", "T")], Rs[("A", "T")]], [1, -1], "D(A←T)"),
            "P4a": ([R[("A", "T")], Rs[("A", "T")], R[("V", "T")], Rs[("V", "T")]], [1, -1, -1, 1], "D(A←T) − D(V←T)"),
            "P4b": ([R[("A", "V")], Rs[("A", "V")], R[("T", "V")], Rs[("T", "V")]], [1, -1, -1, 1], "D(A←V) − D(T←V)"),
        }
        for pn, (cells, co, lab) in P.items():
            e = combo_stats(cells, co, g_eff, rng)
            pred_rows.append(dict(T=T, prediction=pn, quantity=lab) | e | dict(verdict=verdict(e["lo"], e["hi"])))
        # 記述：3 条件の表の評価の回の版（当たる幅 Δ_s，評価の回の真似できる分 P_s）
        for s in STU:
            cN = cell_true(U, f"none_{s}")
            cJ = cell_true(U, f"tgt_{s}_{main}")
            cP = cell_true(U, f"pred_{s}_{main}")
            for qn, cells, co in (("uar_none", [cN], [1]), ("uar_pseudo", [cJ], [1]), ("uar_imitation", [cP], [1]),
                                  ("delta", [cJ, cN], [1, -1]), ("imitable", [cP, cN], [1, -1])):
                e = combo_stats(cells, co, g_eff, rng)
                cond.append(dict(T=T, student=s, quantity=qn) | e | dict(verdict=verdict(e["lo"], e["hi"])))
    D = pd.DataFrame(desc); Pp = pd.DataFrame(pred_rows); C = pd.DataFrame(cond); CV = pd.concat(conv)
    # 予言の数え上げ（4 コーパス，AFEW を除く 3 コーパス）
    crit = []
    for pn in ("P1", "P2", "P3", "P4a", "P4b"):
        x = Pp[Pp.prediction == pn]
        n4 = int((x.verdict == "上回る").sum()); n3 = int(((x.verdict == "上回る") & (x["T"] != "AFEW")).sum())
        crit.append(dict(prediction=pn, n_above_4=n4, n_targets_4=len(x), hit_4=n4 >= 3, n_above_3_noAFEW=n3,
                         n_targets_3=int((x["T"] != "AFEW").sum()), hit_3_noAFEW=n3 >= 2,
                         n_below_4=int((x.verdict == "下回る").sum())))
    p4 = pd.DataFrame(crit)
    p4.loc[len(p4)] = dict(prediction="P4", n_above_4=np.nan, n_targets_4=np.nan,
                           hit_4=bool(p4.set_index("prediction").loc["P4a", "hit_4"] and p4.set_index("prediction").loc["P4b", "hit_4"]),
                           n_above_3_noAFEW=np.nan, n_targets_3=np.nan,
                           hit_3_noAFEW=bool(p4.set_index("prediction").loc["P4a", "hit_3_noAFEW"] and p4.set_index("prediction").loc["P4b", "hit_3_noAFEW"]),
                           n_below_4=np.nan)
    tag = dict(commit=RUN["commit"], dirty=RUN["dirty"])
    for name, df in (("descriptive", D), ("predictions", Pp), ("criteria", p4), ("conditions", C)):
        df.assign(**tag).to_csv(sdir / f"{name}.csv", index=False)
    chk = CV[~CV.target.str.endswith("#shuf")]
    conv_s = dict(n_rows=len(CV), n_units=int(CV[["T", "fold", "seed"]].drop_duplicates().shape[0]),
                  grad_max=float(CV.grad.max()), n_stalled=int(CV.stalled.sum()), n_nonfinite=int((~CV.finite.astype(bool)).sum()),
                  n_targets_checked=len(chk))
    json.dump(conv_s, open(sdir / "convergence.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(conv_s, ensure_ascii=False))


# ------------------------------------------------------------------ 段 selftest

def selftest() -> None:
    rng = np.random.default_rng(0)
    # (1) 予測が目標と同じなら一致は 100，入れ替えと同じ目標なら差は 0
    y = rng.integers(0, 4, 400); cl = np.array([f"g{i % 13}" for i in range(400)])
    c = counts(y, y, cl)
    assert abs(uar_present(c[1].sum(0), c[2].sum(0)) - 100) < 1e-12
    # (2) uar_present は dai1_stats.uar_full と，すべての感情が現れるときに一致する
    pr = rng.integers(0, 4, 400)
    c2 = counts(y, pr, cl)
    assert abs(uar_present(c2[1].sum(0), c2[2].sum(0)) - ST.uar_full(c2)) < 1e-9
    # (3) ある感情が目標に現れないときも NaN にならない（現れた感情だけで平均）
    y3 = np.where(y == 3, 0, y)
    c3 = counts(y3, pr, cl)
    assert np.isfinite(uar_present(c3[1].sum(0), c3[2].sum(0)))
    # (4) 同じ cells の差は 0（推定も区間も），jackknife と引き直しの両方
    cellA = {(f, r): counts(rng.integers(0, 4, 300), rng.integers(0, 4, 300), np.array([f"g{i % 11}" for i in range(300)]))
             for f in range(5) for r in range(3)}
    for g in (8.0, 30.0):
        e = combo_stats([cellA, cellA], [1, -1], g, np.random.default_rng(1))
        assert abs(e["effect"]) < 1e-12 and abs(e["lo"]) < 1e-9 and abs(e["hi"]) < 1e-9, e
    # (5) 1 つの cell の jackknife の推定は分割と種の平均に一致する
    e = combo_stats([cellA], [1], 8.0, np.random.default_rng(1))
    ref = np.mean([np.mean([ST.uar_full(cellA[(f, r)]) for r in range(3)]) for f in range(5)])
    assert abs(e["effect"] - ref) < 1e-9
    # (6) dai1_stats.effect_stats と同じ差の推定（感情がすべて現れるとき）
    cellB = {k: counts(rng.integers(0, 4, 300), rng.integers(0, 4, 300), np.array([f"g{i % 11}" for i in range(300)])) for k in cellA}
    a = combo_stats([cellA, cellB], [1, -1], 8.0, np.random.default_rng(2))
    b = ST.effect_stats(cellA, cellB, 8.0, np.random.default_rng(2))
    assert abs(a["effect"] - b["effect"]) < 1e-9 and abs(a["se"] - b["se"]) < 1e-6, (a, b)
    # (7) 入れ替えは感情ごとの目標の分布を保つ
    qy = rng.integers(0, 4, 500); yy = rng.integers(0, 4, 500)
    qs = shuffle_within(qy, yy, [1, 2, 3])
    for e_ in range(4):
        assert (np.bincount(qy[yy == e_], minlength=4) == np.bincount(qs[yy == e_], minlength=4)).all()
    print("selftest：7 項目すべて通った")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "units", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true", help="試走だけ（本番の置き場では受け付けない）")
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
    if prod and (args.folds is not None or args.seeds is not None):
        sys.exit("本番の置き場では分割と種を絞らない")
    dai1.ENC["enc"] = DD.ENC1
    fz1, fz1_sha = dai3.load_fz1(OUT1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "units").mkdir(exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                host=platform.node(), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""),
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                scripts=script_shas(), design_sha256=sha(DESIGN), dai1_frozen_sha256=fz1_sha,
                repro_saved_sha256=sha(REPRO_SAVED), encoder=dai1.ENC["enc"], K=K, rng_seed=RNG_SEED, n_boot=N_BOOT)
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "units":
        allrows = []
        for T in args.targets:
            fzT = fz1[T]
            seeds = list(range(int(fzT["n_seeds_eval"]))) if args.seeds is None else args.seeds
            folds = list(range(DD.N_FOLDS)) if args.folds is None else args.folds
            for f in folds:
                for r in seeds:
                    df = run_unit(T, f, r, fzT, out)
                    chk = check_against_saved(df)
                    bad = chk[chk.abs_diff.isna() | (chk.abs_diff > TOL_REPRO)]
                    print(f"{T}_f{f}_r{r}：{df.sec.iloc[0]:.0f} 秒．保存との照合の最大差 {chk.abs_diff.max():.2e}，許しの外 {len(bad)}", flush=True)
                    if len(bad):
                        bad.to_csv(out / f"check_fail_{T}_f{f}_r{r}.csv", index=False)
                        sys.exit(f"{T}_f{f}_r{r}：保存の再現できる度合い（実験 3）と {TOL_REPRO} を超えて違う行がある")
                    allrows.append(chk)
            dai1.Corpus.cache.pop(T, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pd.concat(allrows).to_csv(out / f"check_saved_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
        (out / "units.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n") if prod else None
    else:
        stage_summary(out, args.targets)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)
        (out / "summary.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n") if prod else None


if __name__ == "__main__":
    main()
