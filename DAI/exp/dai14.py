"""実験 14（DAI-14）：どのモダリティの特徴が，ほかのモダリティの判断を再現できるか（設計 `DAI/docs/実験14_設計.md` 第 2 版）．

本番の評価の回（実験 1．results/dai1/eval）の保存から，各モダリティの K 本だけの分類器の判断 J_m と，提案の主の版の疑似ラベル
J_prop（prop_one）を候補の上で作り直し，生徒 s（A・V・T）の特徴から線形に再現できる度合い R(s ← m) を 2 つの読み出しで測る．
  R1（そろえた読み出し．主）：特徴を候補の上の主成分で 512 次元にそろえ，出発点 0，罰則の強さ C′ を目標の判断だけを使う内側の
      交差の予測で格子から選ぶ（本当の感情のラベルは使わない）．入れ替えの対照には同じ C′ を当てる．
  R0（実験 1 の作り．副）：生徒の出発点と実験 1 の引き戻し（実験 3 の 2 節と同じ．保存の repro_units.csv と照合する）．
対照の目標：本当の感情の中で入れ替えた #shuf，（感情，塊）の中で入れ替えた #shufcl（予言 P3・P4 の主），細かい層（IEMOCAP は
元のラベル × 覚醒度の 3 区分，MOSEI は感情の強さの最大の 3 区分）の中で入れ替えた #shuffine（記述）．
K = 5（主）と 25（副）．r = 0 の単位では，K = 5・R1 について候補の 2 組の別の分け方 4 通りでも測り，区間に分け方の揺れを足す．

段：
  selftest  合成の数で区間の関数と一致の関数と対照の作りを確かめる
  units     単位ごとの計算．results/dai14/units/<T>_f<f>_r<r>.npz と .csv（同じコミットの既存の単位は飛ばす）
  summary   区間と予言の判定．results/dai14/summary/*.csv
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
KS = (5, 25)
K_MAIN = 5
STU = ("A", "V", "T")
GRID_C = (1e-3, 1e-2, 1e-1, 1.0, 10.0)        # R1 の C′ の格子（小さいほど強い罰）
CB_R1 = 1.0                                    # R1 の切片の C_b（共通）
DIM_R1 = 512                                   # R1 でそろえる次元（顔の特徴の次元）
N_ALT = 4                                      # r = 0 で足す別の分け方の数
ALT_TRIES = 400
N_BOOT = ST.N_BOOT
MIN_N = ST.MIN_TEST
MEM_FRAC = 0.6
RNG_SEED = 20261112
VERS = ("true", "shuf", "shufcl", "shuffine")
SCRIPTS = ("dai14.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py", "dai_data.py")
DESIGN = ROOT / "DAI" / "docs" / "実験14_設計.md"
REPRO_SAVED = ROOT / "results" / "dai3" / "eval" / "repro_units.csv"
TOL_REPRO = 1e-3
RUN = {"commit": "", "dirty": True}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def script_shas() -> dict:
    return {n: sha(HERE / n) for n in SCRIPTS}


def code(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ 一致の割合と区間（設計 4 節）

def counts(y, pred, cl):
    return ST.counts(np.asarray(y, int), np.asarray(pred, int), np.asarray(cl).astype(str))


def uar_present(cor, tot):
    """(…, 4) の数 → 現れた感情（件数 > 0）だけの再現率の平均（%）．件数の合計が MIN_N 未満なら NaN．全体・塊を除いた値・引き直しで同じ式．"""
    cor = np.asarray(cor, float); tot = np.asarray(tot, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(tot > 0, cor / np.where(tot > 0, tot, 1.0), np.nan)
        u = 100.0 * np.nanmean(rec, axis=-1)
    return np.where(tot.sum(-1) < MIN_N, np.nan, u)


def theta(d: dict) -> float:
    vals = [np.nanmean(v) for v in d.values() if np.isfinite(v).any()]
    return float(np.mean(vals)) if vals else float("nan")


def combo_stats(cells: list, coefs: list, g_eff: float, rng, split_vals: dict | None = None) -> dict:
    """Σ coefs[i] · UAR(cells[i]) の区間．cells[i]：{(f, r): counts}．g_eff < 20 は塊の jackknife ＋ 種の項（自由度 max(1, ⌊g_eff⌋ − 1)），
    ≥ 20 は全分割で共通の塊の重みの引き直し ＋ 種の項．split_vals：{f: [分け方ごとの値]} を渡すと，分け方の揺れ
    (1/F²) Σ_f Var_f を SE² に足す（gate-stats の 1）．"""
    keys = sorted(cells[0])
    for c in cells[1:]:
        if sorted(c) != keys:
            raise SystemExit("combo_stats：cells の鍵がそろっていない")
    for k in keys:
        n0 = cells[0][k][0]
        for c in cells[1:]:
            if len(c[k][0]) != len(n0) or not (c[k][0] == n0).all():
                raise SystemExit(f"combo_stats：単位 {k} で塊の並びが違う")
    folds = sorted({f for f, _ in keys})
    rs_of = {f: [r for ff, r in keys if ff == f] for f in folds}
    F = len(folds)

    def val_full(k):
        return float(sum(a * uar_present(c[k][1].sum(0), c[k][2].sum(0)) for c, a in zip(cells, coefs)))

    d = {f: [val_full((f, r)) for r in rs_of[f]] for f in folds}
    est = theta(d)
    R = {f: int(np.isfinite(d[f]).sum()) for f in folds}
    s2 = {f: (float(np.nanvar(d[f], ddof=1)) if R[f] > 1 else 0.0) for f in folds}
    v_seed = float(sum(s2[f] / max(R[f], 1) for f in folds) / F ** 2)
    v_split = np.nan
    if split_vals is not None:
        vs = [float(np.nanvar(v, ddof=1)) for f, v in split_vals.items() if np.isfinite(v).sum() > 1]
        v_split = float(sum(vs) / F ** 2) if vs else np.nan
    extra = v_split if np.isfinite(v_split) else 0.0
    base = dict(effect=est, n_folds=F, n_units=int(sum(R.values())), g_eff=g_eff, v_seed=v_seed, v_split=v_split)
    if not np.isfinite(est):
        return base | dict(lo=np.nan, hi=np.nan, se=np.nan, df=np.nan, method="", v_cl=np.nan, n_clusters=0)
    if g_eff >= 20:
        all_cl = sorted({nm for k in keys for nm in cells[0][k][0]})
        pos = {nm: i for i, nm in enumerate(all_cl)}
        G = len(all_cl)
        Wall = rng.multinomial(G, np.full(G, 1.0 / G), size=N_BOOT).astype(float)
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
        se = float(np.sqrt(v_cl + v_seed + extra))
        return base | dict(lo=est - 1.96 * se, hi=est + 1.96 * se, se=se, df=np.nan, method="boot_cl_shared+seed", v_cl=v_cl, n_clusters=G)
    all_cl = sorted({nm for k in keys for nm in cells[0][k][0]})
    jk = []
    for g in all_cl:
        dg = {}
        for f in folds:
            vals = []
            for r in rs_of[f]:
                k = (f, r)
                hit = np.flatnonzero(cells[0][k][0] == g)
                if len(hit):
                    i = int(hit[0])
                    v = sum(a * uar_present(c[k][1].sum(0) - c[k][1][i], c[k][2].sum(0) - c[k][2][i]) for c, a in zip(cells, coefs))
                else:
                    v = val_full(k)
                vals.append(float(v))
            dg[f] = vals
        jk.append(theta(dg))
    jk = np.array(jk, float); jk = jk[np.isfinite(jk)]
    G = len(jk)
    v_cl = float((G - 1) / G * ((jk - jk.mean()) ** 2).sum()) if G > 1 else np.nan
    se = float(np.sqrt(v_cl + v_seed + extra)) if np.isfinite(v_cl) else np.nan
    df = max(1, int(np.floor(g_eff)) - 1)
    q = float(st.t.ppf(0.975, df))
    return base | dict(lo=est - q * se, hi=est + q * se, se=se, df=df, method="jk+seed", v_cl=v_cl, n_clusters=G)


def verdict(lo, hi) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "判定できない"
    return "上回る" if lo > 0 else ("下回る" if hi < 0 else "判定できない")


# ------------------------------------------------------------------ 層と部分集合（発話一覧から．ラベル以外の注釈）

_META: dict = {}


def corpus_meta(T: str) -> pd.DataFrame:
    """sample_id → 細かい層 fine（IEMOCAP・MOSEI だけ）と，書き起こしの記号の有無 marker．層の境はコーパス全体（DAI/splits の発話）の 3 分位．"""
    if T in _META:
        return _META[T]
    sp = pd.read_csv(ROOT / "DAI" / "splits" / f"{T}.csv", keep_default_na=False, usecols=["sample_id"])
    fn = {"IEMOCAP": "iemocap", "MOSEI": "mosei", "MELD": "meld", "AFEW": "afew"}[T]
    m = pd.read_csv(ROOT / "data" / "metadata" / f"{fn}_utterance_manifest.csv", keep_default_na=False)
    m = m[m.sample_id.isin(sp.sample_id)].set_index("sample_id")
    out = pd.DataFrame(index=sp.sample_id)
    if T == "IEMOCAP":
        a = pd.to_numeric(m["arousal"], errors="coerce")
        q = np.nanquantile(a, [1 / 3, 2 / 3])
        b = np.digitize(a, q)
        out["fine"] = (m["emotion_raw"].astype(str) + "|" + pd.Series(b, index=m.index).astype(str)).reindex(out.index)
    elif T == "MOSEI":
        ic = [c for c in m.columns if c.startswith("int_")]
        mx = m[ic].apply(pd.to_numeric, errors="coerce").max(axis=1)
        q = np.nanquantile(mx, [1 / 3, 2 / 3])
        out["fine"] = pd.Series(np.digitize(mx, q), index=m.index).astype(str).reindex(out.index)
    else:
        out["fine"] = ""
    if T == "AFEW":
        w = pd.read_csv(ROOT / "data" / "metadata" / "afew_transcript_whisper.csv", keep_default_na=False).set_index("sample_id")
        tr = w["transcript_asr"].reindex(out.index).fillna("")
    else:
        tr = m["transcript"].reindex(out.index).fillna("")
    out["marker"] = tr.astype(str).str.contains(r"\[|\]|!", regex=True)
    _META[T] = out
    return out


# ------------------------------------------------------------------ 段 units

def shuffle_within(qy: np.ndarray, strata: np.ndarray, seed: list) -> tuple[np.ndarray, float]:
    """層 strata が同じ候補の間で目標 qy を入れ替える．戻り：入れ替えた目標と，目標が実際に変わった発話の割合．"""
    rng = np.random.default_rng(seed)
    out = qy.copy()
    for g in np.unique(strata):
        ii = np.flatnonzero(strata == g)
        if len(ii) > 1:
            out[ii] = qy[rng.permutation(ii)]
    return out, float((out != qy).mean())


def alt_halves(cl: np.ndarray, seed: int) -> np.ndarray:
    """塊を乱数の順に並べ，そのときの本数の少ない組に配る（pool_halves と同じ規則で順だけを変える．ラベルを使わない）．"""
    rng = np.random.default_rng(seed)
    names, cnt = np.unique(cl, return_counts=True)
    order = rng.permutation(len(names))
    tot, h = [0, 0], {}
    for i in order:
        k = 0 if tot[0] <= tot[1] else 1
        h[names[i]] = k; tot[k] += cnt[i]
    return np.array([h[g] for g in cl], int)


def same_split(a: np.ndarray, b: np.ndarray) -> bool:
    return bool((a == b).all() or (a == 1 - b).all())


def aligned_features(X: torch.Tensor) -> torch.Tensor:
    """候補の特徴を中心化し，主成分の上位 DIM_R1 個の得点にする（ラベルを使わない）．次元あたりの分散の平均が 1 になるよう全体を一律に割る．"""
    X = S.t(X)
    Xc = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    k = min(DIM_R1, Vh.shape[0])
    Z = Xc @ Vh[:k].T
    return Z / Z.var(0).mean().sqrt()


def fit_r0(sv, s, K, tg: dict, halves) -> dict:
    """R0：生徒の出発点と実験 1 の引き戻し．tg：{名前: 目標}．戻り：{名前: (交差の予測, 2 組の平均の一致, 学びの記録)}．"""
    u = sv.u
    n = u.n
    keys = list(tg)
    G = torch.stack([S.balanced_onehot(tg[k], halves[hh], n) for k in keys for hh in (0, 1)])
    W0, b0 = u.start[s]
    cw, cb = sv.teth[(s, K)]
    res = S.fit(u.Xp[s], G, W0, b0, Cw=cw, Cb=cb)
    pred = S.logits(u.Xp[s], res["W"], res["b"]).argmax(2).cpu().numpy()
    return collect(keys, tg, pred, res, halves, n, dict(Cw=cw, Cb=cb, dim=int(u.Xp[s].shape[1])))


def collect(keys, tg, pred, res, halves, n, info) -> dict:
    out = {}
    for i, k in enumerate(keys):
        full = np.empty(n, np.int8)
        ag = []
        for hh in (0, 1):
            other = halves[1 - hh]
            full[other] = pred[2 * i + hh][other]
            ag.append(100 * S.uar(tg[k][other], pred[2 * i + hh][other]))
        fin = bool(res["finite_p"][2 * i] and res["finite_p"][2 * i + 1])
        if not fin:
            raise SystemExit(f"解が有限でない（{k}）")
        out[k] = (full, float(np.mean(ag)), info | dict(grad=float(max(res["grad_p"][2 * i], res["grad_p"][2 * i + 1])),
                                                         stalled=bool(res["stalled_p"][2 * i] or res["stalled_p"][2 * i + 1])))
    return out


def fit_r1(Z: torch.Tensor, cl: np.ndarray, halves, tg_true: dict, tg_all: dict) -> dict:
    """1 つの分け方の R1．fit_r1_multi の 1 本の版．"""
    return fit_r1_multi(Z, cl, [halves], tg_true, tg_all)[0]


def fit_r1_multi(Z: torch.Tensor, cl: np.ndarray, splits: list, tg_true: dict, tg_all: dict) -> list:
    """R1：出発点 0，C_b = CB_R1，C′ は学ぶ側の組を塊で 2 つに分けた内側の交差の予測で，目標の判断との一致が最大の格子の点
    （同点は小さい C′）．tg_true：{名前: 目標}（C′ を選ぶ目標），tg_all：{(名前, 版): 目標}（同じ名前の C′ を当てる）．
    splits：2 組の分け方の並び．すべての分け方の内側の問を 1 回，本体の問を 1 回の L-BFGS の束で解く（速さのため．解は問ごとに独立）．"""
    n, d = Z.shape
    zeroW = torch.zeros(4, d, device=S.DEV, dtype=S.DT); zerob = torch.zeros(4, device=S.DEV, dtype=S.DT)
    names = list(tg_true)
    probs, cws, index = [], [], []
    subs_of = {}
    for isp, halves in enumerate(splits):
        for hh in (0, 1):
            tr = halves[hh]
            sub = dai3.pool_halves(cl[tr])
            subs = [tr[sub == 0], tr[sub == 1]]
            subs_of[(isp, hh)] = subs
            for nm in names:
                for c in GRID_C:
                    for ss in (0, 1):
                        index.append((isp, hh, nm, c, ss))
                        probs.append(S.balanced_onehot(tg_true[nm], subs[ss], n)); cws.append(c)
    res = S.fit(Z, torch.stack(probs), zeroW, zerob, Cw=cws, Cb=CB_R1)
    if not all(res["finite_p"]):
        raise SystemExit("R1 の内側の解が有限でない")
    pr = S.logits(Z, res["W"], res["b"]).argmax(2).cpu().numpy()
    del res, probs
    ag = {}
    for j, (isp, hh, nm, c, ss) in enumerate(index):
        o = subs_of[(isp, hh)][1 - ss]
        ag.setdefault((isp, hh, nm, c), []).append(100 * S.uar(tg_true[nm][o], pr[j][o]))
    chosen = {}
    for isp in range(len(splits)):
        for hh in (0, 1):
            for nm in names:
                best, bc = -1.0, None
                for c in GRID_C:
                    v = float(np.mean(ag[(isp, hh, nm, c)]))
                    if v > best + 1e-12:
                        best, bc = v, c
                chosen[(isp, nm, hh)] = bc
    keys = list(tg_all)
    G, cws = [], []
    for isp, halves in enumerate(splits):
        for k in keys:
            for hh in (0, 1):
                G.append(S.balanced_onehot(tg_all[k], halves[hh], n)); cws.append(chosen[(isp, k[0], hh)])
    res = S.fit(Z, torch.stack(G), zeroW, zerob, Cw=cws, Cb=CB_R1)
    pred = S.logits(Z, res["W"], res["b"]).argmax(2).cpu().numpy()
    outs = []
    P1 = 2 * len(keys)
    for isp, halves in enumerate(splits):
        sl = slice(isp * P1, (isp + 1) * P1)
        sub_res = {kk: (v[sl] if hasattr(v, "__len__") and len(v) == len(G) else v) for kk, v in res.items()}
        out = collect(keys, dict(tg_all), pred[sl], sub_res, halves, n, dict(Cb=CB_R1, dim=int(d)))
        for k in keys:
            out[k][2]["C0"], out[k][2]["C1"] = chosen[(isp, k[0], 0)], chosen[(isp, k[0], 1)]
        outs.append(out)
    return outs


def run_unit(T: str, f: int, r: int, fz1T: dict, out: Path) -> pd.DataFrame | None:
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
    meta = corpus_meta(T).reindex(sv.sid_pool)
    fine = meta["fine"].to_numpy().astype(str)
    has_fine = T in ("IEMOCAP", "MOSEI")
    st_e = y.astype(str)
    st_cl = np.char.add(np.char.add(st_e, "|"), cl)
    st_f = np.char.add(np.char.add(st_e, "|"), fine)
    shared = set(sv.dlg_pool[halves[0]]) & set(sv.dlg_pool[halves[1]]) if T == "MELD" else set()
    nonshared = ~np.isin(sv.dlg_pool, list(shared)) if shared else np.ones(n, bool)
    arr = dict(y=y.astype(np.int8), cl=cl, sid=sv.sid_pool.astype(str), half=h.astype(np.int8), nonshared=nonshared,
               marker=meta["marker"].to_numpy().astype(bool))
    Z = {s: aligned_features(u.Xp[s]) for s in STU}
    for s in STU:
        arr[f"start__{s}"] = S.logits(u.Xp[s], *u.start[s]).argmax(1).cpu().numpy().astype(np.int8)
    rows = []

    def add_rows(K, ro, s, res, split, tgt_frac):
        for (nm, ver), (full, ag, info) in res.items():
            arr[f"K{K}__{split}__{ro}__{s}__{nm}__{ver}"] = full
            tg = TGT[(K, nm, ver)]
            cnt = np.bincount(tg, minlength=4)
            rows.append(dict(T=T, fold=f, seed=r, K=K, split=split, readout=ro, student=s, target=nm, version=ver,
                             agree_halfmean=ag, agree_pooled=float(uar_present(*[x.sum(0) for x in counts(tg, full, cl)[1:]])),
                             frac_changed=tgt_frac.get((K, nm, ver), 0.0), min_class=int(cnt.min()), **info))

    TGT, FRAC, R1_TRUE, R1_ALL = {}, {}, {}, {}
    for K in KS:
        tk = TK[K]
        main = "prop_" + fz1T["main"][str(K)]
        nk = np.ones(n, bool); nk[u.kpos[K]] = False
        arr[f"nk__K{K}"] = nk
        for s in STU:
            arr[f"none__K{K}__{s}"] = S.logits(u.Xp[s], *sv.none(s, K)).argmax(1).cpu().numpy().astype(np.int8)
            names = [f"single_{m}" for m in "AVT" if m != s] + [main]
            Q = dai3.pool_teachers(sv, tk, s, K, names)
            for nm in names:
                qy = Q[nm].argmax(1).cpu().numpy().astype(int)
                if (K, nm, "true") in TGT:
                    if not (TGT[(K, nm, "true")] == qy).all():
                        raise SystemExit(f"{base.name}：目標 {nm} が生徒 {s} で作り直すと違う")
                    continue
                TGT[(K, nm, "true")] = qy
                arr[f"maxp__K{K}__{nm}"] = Q[nm].max(1).values.cpu().numpy().astype(np.float16)
                for ver, strata in (("shuf", st_e), ("shufcl", st_cl), ("shuffine", st_f)):
                    if ver == "shuffine" and not has_fine:
                        continue
                    TGT[(K, nm, ver)], FRAC[(K, nm, ver)] = shuffle_within(qy, strata, [RNG_SEED, DD.CORPUS_ID[T], f, r, K, code(nm), code(ver)])
            vers = [v for v in VERS if v != "shuffine" or has_fine]
            tg0 = {(nm, v): TGT[(K, nm, v)] for nm in names for v in vers}
            add_rows(K, "R0", s, fit_r0_keys(sv, s, K, tg0, halves), "main", FRAC)
            R1_TRUE.update({(K, nm): TGT[(K, nm, "true")] for nm in names})
            R1_ALL.setdefault(s, {}).update({((K, nm), v): TGT[(K, nm, v)] for nm in names for v in vers})
    # R1：K = 5 と 25 の問を生徒ごとに 1 回の束で解く（速さのため．解は問ごとに独立）
    for s in STU:
        tg_all = R1_ALL[s]
        tg_true = {k[0]: R1_TRUE[k[0]] for k in tg_all}
        res = fit_r1(Z[s], cl, halves, tg_true, tg_all)
        for K in KS:
            add_rows(K, "R1", s, {(k[0][1], k[1]): v for k, v in res.items() if k[0][0] == K}, "main", FRAC)
    for (K, nm, ver), tg in TGT.items():
        arr[f"tgt__K{K}__{nm}__{ver}"] = tg.astype(np.int8)
    # r = 0：K = 5，R1，判断 J_m の真と #shuf・#shufcl を，別の分け方 N_ALT 通りでも測る（分け方の揺れ．gate-stats の 1）
    if r == 0:
        alts, j = [], 0
        while len(alts) < N_ALT and j < ALT_TRIES:
            a = alt_halves(cl, RNG_SEED + 1000 * DD.CORPUS_ID[T] + 100 * f + j)
            j += 1
            if same_split(a, h) or any(same_split(a, b) for b in alts):
                continue
            alts.append(a)
        arr["n_alt"] = np.array(len(alts))
        hvs = []
        for ia, a in enumerate(alts):
            hvs.append([np.flatnonzero(a == 0), np.flatnonzero(a == 1)])
            arr[f"althalf__{ia}"] = a.astype(np.int8)
        if hvs:
            for s in STU:
                names = [f"single_{m}" for m in "AVT" if m != s]
                tga = {(nm, v): TGT[(K_MAIN, nm, v)] for nm in names for v in ("true", "shuf", "shufcl")}
                outs = fit_r1_multi(Z[s], cl, hvs, {nm: TGT[(K_MAIN, nm, "true")] for nm in names}, tga)
                for ia, o in enumerate(outs):
                    add_rows(K_MAIN, "R1", s, o, f"alt{ia}", FRAC)
    df = pd.DataFrame(rows)
    df["commit"], df["dirty"], df["sec"] = RUN["commit"], RUN["dirty"], time.time() - t0
    df["dai1_commits"] = ",".join(sorted(sv.commits))
    df["n_shared_dialogues"] = len(shared)
    np.savez_compressed(f"{base}.npz", **arr)
    df.to_csv(f"{base}.csv", index=False)
    return df


def fit_r0_keys(sv, s, K, tg0: dict, halves) -> dict:
    """fit_r0 の鍵を (名前, 版) にしたもの．"""
    keys = list(tg0)
    flat = {f"{k[0]}#{k[1]}": tg0[k] for k in keys}
    res = fit_r0(sv, s, K, flat, halves)
    return {k: res[f"{k[0]}#{k[1]}"] for k in keys}


def check_against_saved(df: pd.DataFrame) -> pd.DataFrame:
    """R0 の真の目標の 2 組の平均の一致を，実験 3 の保存（subset=all）と照合する（K = 5・25．判断 single_m と主の版の疑似ラベル）．"""
    sv_ = pd.read_csv(REPRO_SAVED, low_memory=False)
    sv_ = sv_[sv_.K.isin(KS) & (sv_.subset == "all")]
    x = df[(df.readout == "R0") & (df.version == "true") & (df.split == "main")]
    m = x.merge(sv_[["T", "fold", "seed", "student", "K", "teacher", "agree"]].rename(columns={"teacher": "target"}),
                on=["T", "fold", "seed", "student", "K", "target"], how="left")
    m["abs_diff"] = (m.agree_halfmean - m.agree).abs()
    return m


# ------------------------------------------------------------------ 段 summary

def load_units(out: Path, T: str) -> dict:
    U = {}
    for p in sorted((out / "units").glob(f"{T}_f*_r*.npz")):
        f, r = (int(x[1:]) for x in p.stem.split("_")[1:3])
        U[(f, r)] = dict(np.load(p, allow_pickle=False))
    return U


def mask_of(a: dict, subset: str, K: int | None = None) -> np.ndarray:
    m = np.ones(len(a["y"]), bool)
    if subset == "nonshared":
        m &= a["nonshared"].astype(bool)
    elif subset == "nomarker":
        m &= ~a["marker"].astype(bool)
    if K is not None:
        m &= a[f"nk__K{K}"].astype(bool)
    return m


def cell(U: dict, pred_key, tgt_key, subset="all", K_nk=None) -> dict:
    out = {}
    for k, a in U.items():
        m = mask_of(a, subset, K_nk)
        out[k] = counts(a[tgt_key][m], a[pred_key][m], a["cl"][m])
    return out


def Rcell(U, K, ro, s, nm, ver, subset="all", split="main"):
    return cell(U, f"K{K}__{split}__{ro}__{s}__{nm}__{ver}", f"tgt__K{K}__{nm}__{ver}", subset)


def split_values(U: dict, K: int, terms: list) -> dict:
    """r = 0 の単位で，元の分け方と別の分け方ごとの値 Σ coef · UAR．terms：[(読み出し, 生徒, 名前, 版, 係数)]（R1 だけ）．"""
    sv = {}
    for (f, r), a in U.items():
        if r != 0 or "n_alt" not in a:
            continue
        vals = []
        for sp in ["main"] + [f"alt{i}" for i in range(int(a["n_alt"]))]:
            v = 0.0
            for ro, s, nm, ver, co in terms:
                pk = f"K{K}__{sp}__{ro}__{s}__{nm}__{ver}"
                if pk not in a:
                    v = np.nan; break
                c = counts(a[f"tgt__K{K}__{nm}__{ver}"], a[pk], a["cl"])
                v += co * float(uar_present(c[1].sum(0), c[2].sum(0)))
            vals.append(v)
        sv[f] = vals
    return sv


PRED_DEF = {   # 予言（設計 5 節）：[(生徒, 判断の生徒, 版, 係数)]．版 cl は #shufcl を対照にした差（主），e は #shuf（副）
    "P1": [("A", "T", "true", 1), ("V", "T", "true", -1)],
    "P2": [("A", "V", "true", 1), ("T", "V", "true", -1)],
    "P3": [("A", "T", "true", 1), ("A", "T", "shufcl", -1)],
    "P4a": [("A", "T", "true", 1), ("A", "T", "shufcl", -1), ("V", "T", "true", -1), ("V", "T", "shufcl", 1)],
    "P4b": [("A", "V", "true", 1), ("A", "V", "shufcl", -1), ("T", "V", "true", -1), ("T", "V", "shufcl", 1)],
}
PRIMARY_3 = ("P1", "P3", "P4a")         # 言葉の判断 J_T を使う予言：主の数えは AFEW を除く 3 コーパスで 2 以上（gate-stats の 2）


def pred_terms(pn: str, ro: str, ctrl: str) -> list:
    out = []
    for s, m, ver, co in PRED_DEF[pn]:
        v = ver if ver == "true" else ("shufcl" if ctrl == "cl" else "shuf")
        out.append((ro, s, f"single_{m}", v, co))
    return out


def stage_summary(out: Path, targets: list) -> None:
    sdir = out / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    fz1, _ = dai3.load_fz1(OUT1)
    desc, preds, cond, conv = [], [], [], []
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
        conv.append(csvs)
        g_eff, _ = ST.g_eff_of(T)
        rng = np.random.default_rng([RNG_SEED, DD.CORPUS_ID[T]])
        subsets = ["all", "nonshared"] if T == "MELD" else ["all"]
        for K in KS:
            main = "prop_" + fz1[T]["main"][str(K)]
            # 記述：R，対照，差（読み出し 2 つ）．学び直す前の一致（出発点の判定）
            for ro in ("R1", "R0"):
                for s in STU:
                    for nm in [f"single_{m}" for m in "AVT" if m != s] + [main]:
                        for subset in subsets:
                            cR = Rcell(U, K, ro, s, nm, "true", subset)
                            q = [("R", [cR], [1])]
                            for ver in ("shuf", "shufcl", "shuffine"):
                                if f"tgt__K{K}__{nm}__{ver}" not in next(iter(U.values())):
                                    continue
                                cS = Rcell(U, K, ro, s, nm, ver, subset)
                                q += [(f"R_{ver}", [cS], [1]), (f"D_{ver}", [cR, cS], [1, -1])]
                            for qn, cells_, co in q:
                                e = combo_stats(cells_, co, g_eff, rng)
                                desc.append(dict(T=T, K=K, readout=ro, student=s, target=nm, subset=subset, quantity=qn) | e |
                                            dict(verdict=verdict(e["lo"], e["hi"])))
            for s in STU:
                for nm in [f"single_{m}" for m in "AVT" if m != s] + [main]:
                    cs = cell(U, f"start__{s}", f"tgt__K{K}__{nm}__true")
                    e = combo_stats([cs], [1], g_eff, rng)
                    desc.append(dict(T=T, K=K, readout="start", student=s, target=nm, subset="all", quantity="R") | e |
                                dict(verdict=verdict(e["lo"], e["hi"])))
            # 予言（主：K = 5，R1，#shufcl，部分集合 all．副：R0，K = 25，#shuf，nonshared，nomarker）
            for ro in ("R1", "R0"):
                for ctrl in ("cl", "e"):
                    for subset in subsets + ["nomarker"]:
                        for pn in PRED_DEF:
                            if ctrl == "e" and pn in ("P1", "P2"):
                                continue
                            terms = pred_terms(pn, ro, ctrl)
                            cells_ = []
                            for ro_, s, nm, ver, co in terms:
                                cells_.append(Rcell(U, K, ro_, s, nm, ver, subset))
                            spv = split_values(U, K, terms) if (K == K_MAIN and ro == "R1" and ctrl == "cl" and subset == "all") else None
                            e = combo_stats(cells_, [t[-1] for t in terms], g_eff, rng, split_vals=spv)
                            preds.append(dict(T=T, K=K, readout=ro, control=ctrl, subset=subset, prediction=pn) | e |
                                         dict(verdict=verdict(e["lo"], e["hi"]),
                                              primary=bool(K == K_MAIN and ro == "R1" and ctrl == "cl" and subset == "all")))
            # 記述：3 条件の表の評価の回の版（R0 は生徒の作り，R1 はそろえた読み出し）と判断の当たり方
            for s in STU:
                cN = {k: counts(a["y"][a[f"nk__K{K}"]], a[f"none__K{K}__{s}"][a[f"nk__K{K}"]], a["cl"][a[f"nk__K{K}"]]) for k, a in U.items()}
                cJ = {k: counts(a["y"][a[f"nk__K{K}"]], a[f"tgt__K{K}__{main}__true"][a[f"nk__K{K}"]], a["cl"][a[f"nk__K{K}"]]) for k, a in U.items()}
                q = [("uar_none", [cN], [1]), ("uar_pseudo", [cJ], [1]), ("delta", [cJ, cN], [1, -1])]
                for ro in ("R0", "R1"):
                    cP = {k: counts(a["y"][a[f"nk__K{K}"]], a[f"K{K}__main__{ro}__{s}__{main}__true"][a[f"nk__K{K}"]], a["cl"][a[f"nk__K{K}"]])
                          for k, a in U.items()}
                    q += [(f"uar_imitation_{ro}", [cP], [1]), (f"imitable_{ro}", [cP, cN], [1, -1])]
                for qn, cells_, co in q:
                    e = combo_stats(cells_, co, g_eff, rng)
                    cond.append(dict(T=T, K=K, student=s, quantity=qn) | e | dict(verdict=verdict(e["lo"], e["hi"])))
            for m in "AVT":
                nm = f"single_{m}"
                cJm = {k: counts(a["y"][a[f"nk__K{K}"]], a[f"tgt__K{K}__{nm}__true"][a[f"nk__K{K}"]], a["cl"][a[f"nk__K{K}"]]) for k, a in U.items()}
                e = combo_stats([cJm], [1], g_eff, rng)
                cond.append(dict(T=T, K=K, student=m, quantity="uar_judge") | e | dict(verdict=verdict(e["lo"], e["hi"])))
        dai1.Corpus.cache.pop(T, None)
    D = pd.DataFrame(desc); P = pd.DataFrame(preds); C = pd.DataFrame(cond); CV = pd.concat(conv)
    crit = []
    for (K, ro, ctrl, subset), x in P.groupby(["K", "readout", "control", "subset"]):
        for pn, xx in x.groupby("prediction"):
            ab = set(xx[xx.verdict == "上回る"]["T"])
            if subset == "all" and "MELD" in ab:                        # MELD：共有の対話を除いた版で判定が違えば「上回る」に数えない（gate-leak の 4）
                alt = P[(P.K == K) & (P.readout == ro) & (P.control == ctrl) & (P.subset == "nonshared") & (P.prediction == pn) & (P["T"] == "MELD")]
                if len(alt) and alt.verdict.iloc[0] != "上回る":
                    ab.discard("MELD")
            if pn in PRIMARY_3:
                pool = [t for t in xx["T"] if t != "AFEW"]; need = 2
            else:
                pool = list(xx["T"]); need = 3
            n_ab = len([t for t in pool if t in ab])
            crit.append(dict(K=K, readout=ro, control=ctrl, subset=subset, prediction=pn, n_above=n_ab, n_pool=len(pool), need=need,
                             hit=n_ab >= need, n_above_all4=len(ab), n_below=int((xx.verdict == "下回る").sum()),
                             primary=bool(K == K_MAIN and ro == "R1" and ctrl == "cl" and subset == "all")))
    CR = pd.DataFrame(crit)
    tag = dict(commit=RUN["commit"], dirty=RUN["dirty"])
    for name, df in (("descriptive", D), ("predictions", P), ("criteria", CR), ("conditions", C)):
        df.assign(**tag).to_csv(sdir / f"{name}.csv", index=False)
    conv_s = dict(n_rows=len(CV), n_units=int(CV[["T", "fold", "seed"]].drop_duplicates().shape[0]), grad_max=float(CV.grad.max()),
                  n_stalled=int(CV.stalled.astype(str).eq("True").sum()), min_class_min=int(CV.min_class.min()),
                  frac_changed_by_version={v: [float(CV[CV.version == v].frac_changed.min()), float(CV[CV.version == v].frac_changed.max())]
                                           for v in ("shuf", "shufcl", "shuffine") if (CV.version == v).any()},
                  C_chosen_counts={str(k): int(v) for k, v in pd.concat([CV.get("C0", pd.Series(dtype=float)),
                                                                          CV.get("C1", pd.Series(dtype=float))]).value_counts().items()})
    json.dump(conv_s, open(sdir / "convergence.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: conv_s[k] for k in ("n_rows", "n_units", "grad_max", "n_stalled")}, ensure_ascii=False))


# ------------------------------------------------------------------ 段 selftest

def selftest() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400); cl = np.array([f"g{i % 13}" for i in range(400)])
    c = counts(y, y, cl)
    assert abs(uar_present(c[1].sum(0), c[2].sum(0)) - 100) < 1e-12
    pr = rng.integers(0, 4, 400)
    c2 = counts(y, pr, cl)
    assert abs(uar_present(c2[1].sum(0), c2[2].sum(0)) - ST.uar_full(c2)) < 1e-9
    y3 = np.where(y == 3, 0, y)
    c3 = counts(y3, pr, cl)
    assert np.isfinite(uar_present(c3[1].sum(0), c3[2].sum(0)))
    cellA = {(f, r): counts(rng.integers(0, 4, 300), rng.integers(0, 4, 300), np.array([f"g{i % 11}" for i in range(300)]))
             for f in range(5) for r in range(3)}
    for g in (8.0, 30.0):
        e = combo_stats([cellA, cellA], [1, -1], g, np.random.default_rng(1))
        assert abs(e["effect"]) < 1e-12 and abs(e["lo"]) < 1e-9 and abs(e["hi"]) < 1e-9, e
    e = combo_stats([cellA], [1], 8.0, np.random.default_rng(1))
    ref = np.mean([np.mean([ST.uar_full(cellA[(f, r)]) for r in range(3)]) for f in range(5)])
    assert abs(e["effect"] - ref) < 1e-9
    cellB = {k: counts(rng.integers(0, 4, 300), rng.integers(0, 4, 300), np.array([f"g{i % 11}" for i in range(300)])) for k in cellA}
    a_ = combo_stats([cellA, cellB], [1, -1], 8.0, np.random.default_rng(2))
    b_ = ST.effect_stats(cellA, cellB, 8.0, np.random.default_rng(2))
    assert abs(a_["effect"] - b_["effect"]) < 1e-9 and abs(a_["se"] - b_["se"]) < 1e-6, (a_, b_)
    # 分け方の揺れを足すと SE は広がる
    e1 = combo_stats([cellA], [1], 8.0, np.random.default_rng(1), split_vals={f: [50.0, 52.0, 48.0] for f in range(5)})
    assert e1["se"] > e["se"]
    qy = rng.integers(0, 4, 500); yy = rng.integers(0, 4, 500)
    qs, fr = shuffle_within(qy, yy.astype(str), [1, 2, 3])
    for e_ in range(4):
        assert (np.bincount(qy[yy == e_], minlength=4) == np.bincount(qs[yy == e_], minlength=4)).all()
    assert 0 < fr < 1
    # 別の分け方は元と違い，2 組とも空でない
    cl2 = np.array([f"g{i % 9}" for i in range(300)])
    a2 = alt_halves(cl2, 5)
    assert 0 < a2.sum() < len(a2)
    # 合成の対照（GPU の学び．R1 の作り）：特徴が感情だけを持つとき D ≈ 0，目標の発話ごとの違いを持つとき D > 0
    n, d = 1200, 24
    yv = rng.integers(0, 4, n); clv = np.array([f"c{i % 40}" for i in range(n)])
    noise = rng.integers(0, 4, n); flip = rng.random(n) < 0.4
    J = np.where(flip, noise, yv)                                      # 判断：感情 ＋ 発話ごとのずれ
    E = np.eye(4)
    hv = dai3.pool_halves(clv)
    halves = [np.flatnonzero(hv == 0), np.flatnonzero(hv == 1)]
    for case, X_ in (("neg", np.c_[3 * E[yv], rng.normal(size=(n, d - 4))]), ("pos", np.c_[3 * E[yv], 3 * E[J], rng.normal(size=(n, d - 8))])):
        Zt = aligned_features(torch.as_tensor(X_))
        Js, _ = shuffle_within(J, yv.astype(str), [9, 9])
        res = fit_r1(Zt, clv, halves, {"j": J}, {("j", "true"): J, ("j", "shuf"): Js})
        D = res[("j", "true")][1] - res[("j", "shuf")][1]
        if case == "neg":
            assert abs(D) < 5, D
        else:
            assert D > 15, D
    print("selftest：11 項目すべて通った")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "units", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="試しの置き場だけ：格子のそろいと汚れの確かめを省いて summary を作る")
    args = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
    if args.stage == "selftest":
        selftest()
        return
    out = Path(args.out_dir).resolve()
    prod = out == OUT_PROD.resolve()
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    RUN["partial"] = bool(args.allow_partial)
    if dirty and (prod or not args.allow_dirty):
        sys.exit("DAI/exp・measure・features に未コミットの変更がある（試走は一時の置き場で --allow-dirty）")
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
                scripts=script_shas(), design_sha256=sha(DESIGN), dai1_frozen_sha256=fz1_sha, repro_saved_sha256=sha(REPRO_SAVED),
                encoder=dai1.ENC["enc"], KS=KS, grid_c=GRID_C, cb_r1=CB_R1, dim_r1=DIM_R1, n_alt=N_ALT, rng_seed=RNG_SEED, n_boot=N_BOOT)
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
                    if df is None:
                        print(f"{T}_f{f}_r{r}：済み（飛ばす）", flush=True)
                        continue
                    chk = check_against_saved(df)
                    bad = chk[chk.abs_diff.isna() | (chk.abs_diff > TOL_REPRO)]
                    print(f"{T}_f{f}_r{r}：{df.sec.iloc[0]:.0f} 秒．保存との照合 {len(chk)} 行，最大差 {chk.abs_diff.max():.2e}，許しの外 {len(bad)}", flush=True)
                    if len(bad):
                        bad.to_csv(out / f"check_fail_{T}_f{f}_r{r}.csv", index=False)
                        sys.exit(f"{T}_f{f}_r{r}：保存の再現できる度合い（実験 3）と {TOL_REPRO} を超えて違う行がある")
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
