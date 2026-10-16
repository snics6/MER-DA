"""【探索的】E32：ソースの価値をラベルの本数で測る（n = q e²/(1 − e²)）。新しい学習なし。

規則は `docs/exploratory-analyses.md` E32 の 4 節（予言 V1・V2 は計算の前に書いた）。
測った n：方法の UAR がターゲットのみの学習曲線（地図の tgt_only，K ∈ {1, 5, 10, 25, 50}，K の対数で直線補間）で
何本のところに当たるかを読み，K を引いた値（感情ごとの本数）。予言した n：E13 の D_j・d_s,j と E28-9 の q_dir から，
同じ手順を模型の学習曲線（共通の雑音の橋．K 本で決めた基準の揺れ 1/(2K) を入れる）の上で踏んだ値。

    ./.venv/bin/python exp/explore_source_value.py

入力：`results/map_<kind>/`，`results/compare_<kind>/*.csv`（`explore_compare.pair_table`，K ≤ 25），
`results/fixc_analysis/explore_relation/`（e13_pairs.csv，e28t9_targets.csv，e28t7_targets.csv．HuBERT の E13 は
E28-9 と同じく `explore_pairs.measure` で測る）。
出力：`results/explore_relation/e32_pairs.csv`（組 × K），`e32_tests.csv`，`e32_kinds.csv`（種類 × K × 手法），`e32_allsrc.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import explore_compare as EC  # noqa: E402
from explore_pairs import measure  # noqa: E402
from explore_synergy import PAIRS  # noqa: E402
from explore_uar_link import shared  # noqa: E402

MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
NAMES = ["neutral", "happy", "sad", "angry"]
PAIR_NAMES = [f"{NAMES[a]}-{NAMES[b]}" for a, b in PAIRS]
KS_CURVE = [1, 5, 10, 25, 50]
KS = ["1", "5", "10", "25"]
METHODS = ["P1_def", "P3_def", "P4_def", "B4", "H3", "O1_meta"]
FIX = ROOT / "results" / "fixc_analysis" / "explore_relation"
OUT = ROOT / "results" / "explore_relation"
BIG = 1e4                                   # 打ち切りの組の順位用


# ---------------------------------------------------------------- 学習曲線から本数を読む

def curves(kind: str, targets) -> dict:
    mp = EC.load_map(kind, targets)
    b1 = mp[(mp.src == "cv") & (mp.method == "tgt_only")].groupby(["target", "K"]).uar.mean()
    return {t: np.maximum.accumulate(np.array([b1[(t, str(k))] for k in KS_CURVE])) for t in targets}


def k_equiv(us: np.ndarray, u: float) -> tuple[float, int]:
    """UAR u が曲線 us（KS_CURVE 上）に当たる本数。(K′, 0)，範囲の上を超えたら (nan, 1)，下なら (nan, −1)。"""
    if u > us[-1]:
        return np.nan, 1
    if u < us[0]:
        return np.nan, -1
    lk = np.log(KS_CURVE)
    for i in range(len(us) - 1):
        if us[i] <= u <= us[i + 1]:
            f = 0.0 if us[i + 1] == us[i] else (u - us[i]) / (us[i + 1] - us[i])
            return float(np.exp(lk[i] + f * (lk[i + 1] - lk[i]))), 0
    return np.nan, 1


def n_of(us, u, K):
    kp, flag = k_equiv(us, u)
    return (kp - K if flag == 0 else np.nan), flag


# ---------------------------------------------------------------- 模型

def bridge(c6) -> float:
    c = np.zeros((4, 4))
    for (k, l), v in zip(PAIRS, c6):
        c[k, l] = c[l, k] = v
    return shared(c)


def d_star(D, e, q, K):
    t = q / K
    r = e ** 2 / (1 - e ** 2)
    return D / np.sqrt(1 + t / (1 + r * t))


def jitter(K, pos: bool):
    return np.sqrt(1 + 1 / (2 * K)) if pos else 1.0


def n_model(D, e, q, K, pos: bool) -> tuple[float, int]:
    """模型の上で，仮説転移（K 本）の UAR がターゲットのみの模型の学習曲線で何本に当たるか。"""
    uh = bridge(d_star(D, e, q, K) / (2 * jitter(K, pos)))
    ut = lambda k: bridge(D / (2 * np.sqrt(1 + q / k) * jitter(k, pos)))  # noqa: E731
    lo, hi = np.log(1e-3), np.log(1e6)
    if uh >= ut(np.exp(hi)):
        return np.nan, 1
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ut(np.exp(mid)) < uh else (lo, mid)
    return float(np.exp((lo + hi) / 2)) - K, 0


# ---------------------------------------------------------------- 組ごと

def load_e13(kind: str) -> pd.DataFrame:
    """E13 の感情の対ごとの d・own_d。ファイルに無い種類（HuBERT）は E28-9 と同じく `explore_pairs.measure` で測る。"""
    e13 = pd.read_csv(FIX / "e13_pairs.csv")
    e13 = e13[e13.kind == kind]
    return e13 if len(e13) else measure(kind)


def pair_rows(kind: str, P: pd.DataFrame, cur: dict) -> list[dict]:
    e13 = load_e13(kind)
    qd = pd.read_csv(FIX / "e28t9_targets.csv").query("kind == @kind").set_index("target").q_dir
    rows = []
    for (t, s), g in P[~P.same].groupby(["target", "source"]):
        own = e13[e13.target == t].groupby("pair").own_d.mean().reindex(PAIR_NAMES).to_numpy()
        ds = e13[(e13.target == t) & (e13.source == s)].set_index("pair").d.reindex(PAIR_NAMES).to_numpy()
        if np.isnan(ds).any() or t not in qd:
            continue
        e = np.clip(ds / own, 0.0, 0.999)
        q = float(qd[t])
        eb = float(e.mean())
        for K in KS:
            u = float(g[g.K == K].P1_def.iloc[0])
            nm, fl = n_of(cur[t], u, int(K))
            npd, fp = n_model(own, e, q, int(K), pos=True)
            nn, fn = n_model(own, e, q, int(K), pos=False)
            rows.append({"kind": kind, "target": t, "source": s, "K": K, "uar_P1": u, "n_meas": nm, "flag_meas": fl,
                         "n_pred": npd, "flag_pred": fp, "n_pred_nopos": nn, "flag_pred_nopos": fn,
                         "n_simple": q * eb ** 2 / (1 - eb ** 2), "e_mean": eb, "q_dir": q,
                         "D": float(own.mean()), "d_s": float(ds.mean())})
    return rows


def rankable(v, flag):
    return np.where(flag == 1, BIG, np.where(flag == -1, -BIG, v))


def tests(Q: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (kind, K), g in Q.groupby(["kind", "K"]):
        meas = rankable(g.n_meas.to_numpy(), g.flag_meas.to_numpy())
        row = {"kind": kind, "K": K, "n": len(g), "censored_high": int((g.flag_meas == 1).sum()),
               "censored_low": int((g.flag_meas == -1).sum())}
        for col, fcol in (("n_pred", "flag_pred"), ("n_pred_nopos", "flag_pred_nopos"), ("n_simple", None)):
            pred = rankable(g[col].to_numpy(), g[fcol].to_numpy() if fcol else np.zeros(len(g)))
            row[f"rho_{col}"] = stats.spearmanr(pred, meas).statistic
            ok = (g.flag_meas == 0) & np.isfinite(g[col])
            row[f"ratio_{col}"] = float(np.median(g.n_meas[ok] / g[col][ok])) if ok.any() else np.nan
        w = []
        for _, h in g.groupby("target"):
            if len(h) >= 4:
                w.append(stats.spearmanr(h.n_pred, rankable(h.n_meas.to_numpy(), h.flag_meas.to_numpy())).statistic)
        row["rho_within_target"] = float(np.nanmean(w))
        row["median_n_meas"] = float(np.median(rankable(g.n_meas.to_numpy(), g.flag_meas.to_numpy())))
        row["median_n_pred"] = float(np.median(g.n_pred))
        out.append(row)
    T = pd.DataFrame(out)
    for kind in T.kind.unique():
        m = (T.kind == kind) & T.K.isin(["5", "10"])
        T.loc[T.kind == kind, "V1_kind"] = bool((T[m].rho_n_pred >= 0.5).all())
        T.loc[T.kind == kind, "V2_kind"] = bool(T[m].ratio_n_pred.between(0.5, 2).all())
    return T


def constancy(Q: pd.DataFrame) -> pd.DataFrame:
    W = Q[Q.flag_meas == 0].pivot_table(index=["kind", "target", "source"], columns="K", values="n_meas")
    out = []
    for kind, g in W.groupby(level="kind"):
        r = {"kind": kind}
        for a, b in (("10", "5"), ("25", "5"), ("25", "10")):
            ok = g[[a, b]].dropna()
            ok = ok[ok[b] > 0]
            r[f"n{a}/n{b}"] = float(np.median(ok[a] / ok[b])) if len(ok) else np.nan
            r[f"pairs_{a}_{b}"] = len(ok)
        out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 種類ごとの表と全ソース

def kind_rows(kind: str, P: pd.DataFrame, cur: dict) -> list[dict]:
    X = P[~P.same]
    us = np.mean([cur[t] for t in X.target], axis=0) if len(X) else None
    rows = []
    for K in KS:
        A = X[X.K == K]
        for m in METHODS:
            if m not in A or A[m].isna().all():
                continue
            u = float(A[m].mean())
            n, fl = n_of(us, u, int(K))
            rows.append({"kind": kind, "K": K, "method": m, "uar": u, "uar_B1": float(A.B1.mean()), "n": n, "flag": fl})
    return rows


def allsrc_rows(kind: str, P: pd.DataFrame, cur: dict) -> list[dict]:
    t7 = pd.read_csv(FIX / "e28t7_targets.csv").query("kind == @kind")
    t9 = pd.read_csv(FIX / "e28t9_targets.csv").query("kind == @kind").set_index("target")
    rows = []
    for (t, K), g in P.groupby(["target", "K"]):
        if t not in t9.index or K not in ("1", "5", "10"):
            continue
        x = t7[(t7.target == t) & (t7.K.astype(str) == K)]
        if x.empty:
            continue
        e = min(float(x.d_s_all.iloc[0]) / float(t9.own_d[t]), 0.999)
        q = float(t9.q_dir[t])
        n, fl = n_of(cur[t], float(g.P4_def.iloc[0]), int(K))
        rows.append({"kind": kind, "target": t, "K": K, "P4": float(g.P4_def.iloc[0]), "n_meas": n, "flag": fl,
                     "e_all": e, "q_dir": q, "n_simple_all": q * e ** 2 / (1 - e ** 2)})
    return rows


def main() -> int:
    EC.KS_USE = KS
    prs, krs, ars = [], [], []
    for kind in EC.KINDS:
        P = EC.pair_table(kind)
        cur = curves(kind, sorted(P.target.unique()))
        krs += kind_rows(kind, P, cur)
        if kind in MAIN:
            prs += pair_rows(kind, P, cur)
            ars += allsrc_rows(kind, P, cur)
        print(f"  {kind} 済み", flush=True)
    Q, Kt, A = pd.DataFrame(prs), pd.DataFrame(krs), pd.DataFrame(ars)
    T = tests(Q)
    C = constancy(Q)
    Q.to_csv(OUT / "e32_pairs.csv", index=False)
    T.to_csv(OUT / "e32_tests.csv", index=False)
    Kt.to_csv(OUT / "e32_kinds.csv", index=False)
    A.to_csv(OUT / "e32_allsrc.csv", index=False)
    C.to_csv(OUT / "e32_constancy.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(2).to_string(index=False))
    print(C.round(2).to_string(index=False))
    print(Kt.pivot_table(index=["kind", "method"], columns="K", values="n").reindex(columns=KS).round(1).to_string())
    ok = A[A.flag == 0]
    for kind, g in ok.groupby("kind"):
        print(kind, "全ソース：順位相関 %.2f，測った n ÷ n_simple の中央値 %.2f（%d 行）" % (
            stats.spearmanr(g.n_simple_all, g.n_meas).statistic, np.median(g.n_meas / g.n_simple_all), len(g)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
