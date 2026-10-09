"""【探索的】E16：少数のターゲットの例は，合併の分類器の向きと位置をどう変えるか（E15 の拡張）。

導出は `docs/exploratory-analyses.md` E16。E15 の λ の式は「向きはソースから受け取る」と置いたが，
実際の合併の分類器（E15 の段階 B）は K=1 で向きが悪くなり，K=10 で良くなる（映像は K=5 でも悪い）。
ここでは，合併の解をソースのみの解からのニュートン法の 1 歩として書く：

    ΔW = G H⁻¹,   G = Σ_{j∈抜いた K 本} s_j (e_{y_j} − p_j) x_jᵀ,   H = H_S + Σ_j s_j κ_j x_j x_jᵀ
    H_S = diag(A_s + I/C, N_s κ_s),   x = [z − z̄; 1]

（多クラスの曲がりの行列を κ(I − 𝟙𝟙ᵀ/k) で近似すると，感情ごとの行が同じ H を持つ．E15 の λ の式は，
残差 e − p を位置のずれの平均だけで近似したもの）。抜いた K 本ごとに ΔW を計算し，ターゲットのテストの
点数に足して，感情の対ごとの d・o を予言する。H⁻¹ はウッドベリーの公式で H_S⁻¹ から作る。

d の変化の分け方（ターゲット全発話で評価．30 通りのでたらめな抜き方の平均）：

    d_m / d_s ≈ (1 + I) / √(1 + S + N)
    I：平均の動きが 2 感情の点数の差を広げる分（情報）
    S：平均の動きが点数の散らばりを変える分（スイッチの雑音など）
    N：抜き方の揺れが散らばりに足す分（抜き方の雑音）

    ./.venv/bin/python exp/explore_direction.py [--kinds audio_wavlm ...]
    ./.venv/bin/python exp/explore_direction.py --stage diag    # 実測の d の変化を情報と雑音に分ける（E16 の 6 節）
    ./.venv/bin/python exp/explore_direction.py --stage drivers # 雑音と情報の大きさを決めるもの（E16 の 8 節）

入力：地図の C，特徴，ソースのみの分類器の点数（`results/shift_<kind>/logits/`），E15 の段階 B の点数
（`results/merged_scores_<kind>/`）。出力：`results/explore_relation/e16_*.csv`・`fig_e16.png`
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
sys.path.insert(0, str(ROOT / "features"))
import common  # noqa: E402
from data import groups_for, load_any  # noqa: E402
from explore_synergy import KINDS, KS, N_CLASSES, kappa, load_map, src_parts  # noqa: E402
from pair_table import OUT, OUT_ROOT, map_dir  # noqa: E402
from split import speaker_folds  # noqa: E402

PAIRS = [(k, j) for k in range(N_CLASSES) for j in range(k + 1, N_CLASSES)]
REPS = range(10)
N_MC = 30
O_MIN = 0.3


# ---------------------------------------------------------------- ニュートン法の 1 歩

def hs_inv(sp: dict, C: float):
    """H_S⁻¹ を掛ける関数（H_S = diag(A_s + I/C, N_s κ_s)）。M は (d+1, m)。"""
    V, inv = sp["V"], 1.0 / (sp["ev"] + 1.0 / C)
    a = sp["a"]
    def f(M: np.ndarray) -> np.ndarray:
        out = np.empty_like(M)
        out[:-1] = V @ ((V.T @ M[:-1]) * inv[:, None])
        out[-1] = M[-1] / a
        return out
    return f


KAPPA_FLOOR = 0.0          # 抜いた K 本の曲がりの下限（E20 の R3．既定は下限なし）


def newton_step(Hinv, X: np.ndarray, P: np.ndarray, y: np.ndarray, sw: np.ndarray) -> np.ndarray:
    """抜いた行 X（m, d+1），ソースのみの確率 P（m, k），ラベル y，重み sw から ΔW（k, d+1）を返す。"""
    k = P.shape[1]
    R = np.eye(k)[y] - P
    G = (R * sw[:, None]).T @ X                                   # (k, d+1)
    D = sw * np.maximum((1 - (P ** 2).sum(1)) / (k - 1), KAPPA_FLOOR)
    HiX = Hinv(X.T)                                               # (d+1, m)
    GHi = Hinv(G.T).T                                             # (k, d+1)
    inner = np.diag(1.0 / D) + X @ HiX                            # (m, m)
    return GHi - (GHi @ X.T) @ np.linalg.solve(inner, HiX.T)


def d_o(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """感情の対ごとの (d, o)。E13 と同じ式。"""
    out = []
    for k, j in PAIRS:
        zk, zj = L[y == k, k] - L[y == k, j], L[y == j, k] - L[y == j, j]
        sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
        out.append(((zk.mean() - zj.mean()) / sd, -(zk.mean() + zj.mean()) / 2 / sd))
    return np.array(out)


def sep_var_mid(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """感情の対ごとの (2 つの山の平均の差，散らばり（分散の平均），中点)。"""
    out = []
    for k, j in PAIRS:
        zk, zj = L[y == k, k] - L[y == k, j], L[y == j, k] - L[y == j, j]
        out.append((zk.mean() - zj.mean(), (zk.var(ddof=1) + zj.var(ddof=1)) / 2, (zk.mean() + zj.mean()) / 2))
    return np.array(out)


def decompose(L0: np.ndarray, y: np.ndarray, dLs: list[np.ndarray]) -> dict:
    """d_m/d_s ≈ (1 + I)/√(1 + S + N) の I・S・N（感情の対ごと）。dLs は抜き方ごとの点数の変化。"""
    base = sep_var_mid(L0, y)
    mean_dL = np.mean(dLs, axis=0)
    mu = sep_var_mid(L0 + mean_dL, y)
    var_each = np.mean([sep_var_mid(L0 + dL, y)[:, 1] for dL in dLs], axis=0)
    I = (mu[:, 0] - base[:, 0]) / base[:, 0]
    S = (mu[:, 1] - base[:, 1]) / base[:, 1]
    N = (var_each - mu[:, 1]) / base[:, 1]
    return {"I": I, "S": S, "N": N}


# ---------------------------------------------------------------- 実データ

def balanced_sw(n_src_c: np.ndarray, K: int) -> np.ndarray:
    """合併の学習データ（ソースのクラス c に n_c 本，ターゲットに K 本ずつ）のターゲットの 1 本の重み（クラスごと）。"""
    n_c = n_src_c + K
    return n_c.sum() / (N_CLASSES * n_c)


def map_draw(df: pd.DataFrame, name: str, r: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """地図と同じテストの袋と K 本（`exp/map.py` の test_mask・kshot_order・take_k を CPU で）。"""
    f = speaker_folds(df.speaker_group.astype(str).reset_index(drop=True), k=5, seed=r).to_numpy()
    te = f == r % 5
    pool = np.flatnonzero(~te)
    y = df.label_4.to_numpy()
    rng = common.rng_for(f"map-kshot:{name}", r)
    order = [rng.permutation(pool[y[pool] == c]) for c in range(N_CLASSES)]
    return te, np.sort(np.concatenate([o[:K] for o in order]))


def run_kind(kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mp = load_map(kind)
    groups = groups_for(kind)
    data = {g: load_any(g, kind) for g in groups}
    rows, dec = [], []
    rng = np.random.default_rng(0)
    for s in groups:
        dfs, Xs = data[s]
        ys = dfs.label_4.to_numpy()
        mu, sd = Xs.mean(0), Xs.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Zs = ((Xs - mu) / sd).astype(np.float64)
        z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
        ks_all = kappa(z["oof"])
        n_src_c = np.bincount(ys, minlength=N_CLASSES)
        sps = {K: src_parts(Zs, ys, ks_all, int(K)) for K in KS}
        for t in groups:
            if t == s:
                continue
            dft, Xt = data[t]
            yt = dft.label_4.to_numpy()
            Zt = ((Xt - mu) / sd).astype(np.float64)
            Lt = z[f"t_{t}"]
            Pt = softmax(Lt, axis=1)
            B = np.load(OUT_ROOT / f"merged_scores_{kind}" / f"{t}.npz")
            g = mp[(mp.target == t) & (mp.source == s)]
            for K in KS:
                sp = sps[K]
                Xa = np.c_[Zt - sp["zbar"], np.ones(len(Zt))]
                swc = balanced_sw(n_src_c, int(K))
                for r in REPS:
                    te, idx = map_draw(dft, t, r, int(K))
                    if not np.array_equal(np.flatnonzero(te), B[f"test_idx_r{r}"]):
                        raise SystemExit(f"{kind} {t} 反復 {r}：テストの袋が段階 B と合わない")
                    C = float(g[(g.K == K) & (g.rep == r)].C_m.iloc[0])
                    dW = newton_step(hs_inv(sp, C), Xa[idx], Pt[idx], yt[idx], swc[yt[idx]])
                    Ls_te, Lm_te = B[f"s_{s}_r{r}"].astype(np.float64), B[f"m_{s}_K{K}_r{r}"].astype(np.float64)
                    Lp_te = Ls_te + Xa[te] @ dW.T
                    y_te = yt[te]
                    ds, dm, dp = d_o(Ls_te, y_te), d_o(Lm_te, y_te), d_o(Lp_te, y_te)
                    for (k, j), a, b, c in zip(PAIRS, ds, dm, dp):
                        rows.append({"kind": kind, "target": t, "source": s, "K": K, "rep": r, "pair": f"{k}-{j}",
                                     "C": C, "d_s": a[0], "o_s": a[1], "d_m": b[0], "o_m": b[1],
                                     "d_pred": c[0], "o_pred": c[1]})
                # 分け方：ターゲット全発話で評価，でたらめな抜き方 N_MC 通り．C は反復 0〜9 で最も多く選ばれた値
                C = float(g[(g.K == K) & (g.rep < 10)].C_m.mode().iloc[0])
                Hinv = hs_inv(sp, C)
                dLs = []
                for _ in range(N_MC):
                    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(yt == c), int(K), replace=False)
                                                  for c in range(N_CLASSES)]))
                    dLs.append(Xa @ newton_step(Hinv, Xa[idx], Pt[idx], yt[idx], swc[yt[idx]]).T)
                dd = decompose(Lt, yt, dLs)
                for (k, j), I, S_, N_ in zip(PAIRS, dd["I"], dd["S"], dd["N"]):
                    dec.append({"kind": kind, "target": t, "source": s, "K": K, "pair": f"{k}-{j}", "C": C,
                                "I": I, "S": S_, "N": N_})
        print(f"  {kind} ソース {s} 済み", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(dec)


CW = (0.05, 0.01)


def split_obs(Ls: np.ndarray, Lm: np.ndarray, y: np.ndarray) -> list[dict]:
    """z_m = a·z_s + e と回帰し，d_m/d_s = (1 + I_obs)/√(1 + N_obs) の I_obs・N_obs を感情の対ごとに返す。"""
    out = []
    for k, j in PAIRS:
        m = (y == k) | (y == j)
        zs, zm, yy = Ls[m, k] - Ls[m, j], Lm[m, k] - Lm[m, j], y[m]
        cs = zs.copy(); cm = zm.copy()                         # 感情の中で平均を引く
        for c in (k, j):
            cs[yy == c] -= zs[yy == c].mean(); cm[yy == c] -= zm[yy == c].mean()
        a = float((cs * cm).sum() / (cs * cs).sum())
        e = zm - a * zs
        sep_s = zs[yy == k].mean() - zs[yy == j].mean()
        sep_e = e[yy == k].mean() - e[yy == j].mean()
        var_s = (zs[yy == k].var(ddof=1) + zs[yy == j].var(ddof=1)) / 2
        var_e = (e[yy == k].var(ddof=1) + e[yy == j].var(ddof=1)) / 2
        out.append({"pair": f"{k}-{j}", "a": a, "I_obs": sep_e / (a * sep_s), "N_obs": var_e / (a * a * var_s),
                    "d_s": sep_s / np.sqrt(var_s),
                    "d_m": (zm[yy == k].mean() - zm[yy == j].mean())
                           / np.sqrt((zm[yy == k].var(ddof=1) + zm[yy == j].var(ddof=1)) / 2)})
    return out


def run_diag(kind: str) -> pd.DataFrame:
    groups = groups_for(kind)
    data = {g: load_any(g, kind) for g in groups}
    rows = []
    for s in groups:
        z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
        for t in groups:
            if t == s:
                continue
            dft = data[t][0]
            yt = dft.label_4.to_numpy()
            Pt = softmax(z[f"t_{t}"], axis=1)
            p_true = Pt[np.arange(len(yt)), yt]
            B = np.load(OUT_ROOT / f"merged_scores_{kind}" / f"{t}.npz")
            for K in KS:
                for r in REPS:
                    te, idx = map_draw(dft, t, r, int(K))
                    if not np.array_equal(np.flatnonzero(te), B[f"test_idx_r{r}"]):
                        raise SystemExit(f"{kind} {t} 反復 {r}：テストの袋が段階 B と合わない")
                    Ls, Lm = B[f"s_{s}_r{r}"].astype(np.float64), B[f"m_{s}_K{K}_r{r}"].astype(np.float64)
                    base = {"kind": kind, "target": t, "source": s, "K": K, "rep": r,
                            **{f"n_cw{c:g}": int((p_true[idx] < c).sum()) for c in CW},
                            "p_true_min": float(p_true[idx].min())}
                    for o in split_obs(Ls, Lm, yt[te]):
                        rows.append({**base, **o})
        print(f"  {kind} ソース {s} 済み", flush=True)
    return pd.DataFrame(rows)


def summarize_diag(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = X.assign(log_dr=np.log(X.d_m / X.d_s))
    med = (X.groupby(["kind", "K"]).agg(I_obs=("I_obs", "median"), N_obs=("N_obs", "median"),
                                        log_dr=("log_dr", "median"), a=("a", "median"))
           .reset_index())
    keys = ["kind", "target", "source", "K", "rep"]
    R = X.groupby(keys).agg(log_dr=("log_dr", "mean"), **{f"n_cw{c:g}": (f"n_cw{c:g}", "first") for c in CW}).reset_index()
    tests = []
    for (kind, K), g in R.groupby(["kind", "K"], sort=False):
        row = {"setting": KINDS[kind], "K": K, "n_draws": len(g)}
        for c in CW:
            col = f"n_cw{c:g}"
            row[f"share_draws_cw{c:g}"] = float((g[col] >= 1).mean())
            dm = g.groupby(["target", "source"])
            u = g.log_dr - dm.log_dr.transform("mean")
            v = g[col] - dm[col].transform("mean")
            ok = v.abs() > 0
            if ok.sum() >= 10:
                r, p = stats.spearmanr(v, u, alternative="less")
                row[f"spearman_cw{c:g}"], row[f"p_cw{c:g}"] = r, p
            row[f"n_varying_cw{c:g}"] = int(ok.sum())
        tests.append(row)
    return med, pd.DataFrame(tests)


def switch_q(kind: str, C_mode: pd.DataFrame) -> pd.DataFrame:
    """Q = uᵀΣ_t u / (u·δ)²，u = (A_s + I/C)⁻¹ δ（組 × K．C は反復 0〜9 で最も多く選ばれた値）。"""
    groups = groups_for(kind)
    data = {g: load_any(g, kind) for g in groups}
    rows = []
    for s in groups:
        dfs, Xs = data[s]
        ys = dfs.label_4.to_numpy()
        mu, sd = Xs.mean(0), Xs.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Zs = ((Xs - mu) / sd).astype(np.float64)
        ks_all = kappa(np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")["oof"])
        sps = {K: src_parts(Zs, ys, ks_all, int(K)) for K in KS}
        for t in groups:
            if t == s:
                continue
            Zt = ((data[t][1] - mu) / sd).astype(np.float64)
            Zc = Zt - Zt.mean(0)
            for K in KS:
                sp = sps[K]
                C = float(C_mode[(C_mode.target == t) & (C_mode.source == s) & (C_mode.K == K)].C.iloc[0])
                delta = Zt.mean(0) - sp["zbar"]
                u = sp["V"] @ ((sp["V"].T @ delta) / (sp["ev"] + 1.0 / C))
                rows.append({"kind": kind, "target": t, "source": s, "K": K, "C_mode": C,
                             "Q": float(np.var(Zc @ u) / (u @ delta) ** 2)})
        print(f"  {kind} ソース {s} 済み", flush=True)
    return pd.DataFrame(rows)


def run_drivers(kinds: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from explore_pairs import measure
    X = pd.read_csv(OUT / "e16_diag_rows.csv", dtype={"K": str})
    X = X[X.kind.isin(kinds) & (X.d_s > 0) & (X.d_m > 0)]
    lam = pd.read_csv(OUT / "e15b_pairs.csv", dtype={"K": str})[["kind", "target", "source", "K", "lam_obs"]]
    o_s = pd.read_csv(OUT / "e16_rows.csv", dtype={"K": str}).groupby(
        ["kind", "target", "source", "K", "pair"]).agg(o_s=("o_s", "mean")).reset_index()
    curves = pd.concat([
        pd.read_csv(OUT / "snapshots" / "20261008-1310_e11_v30" / "e11_curves.csv")
          .rename(columns={"setting": "kind"})[["kind", "target", "q"]],
        pd.read_csv(ROOT / "results" / "check_mh" / "curves.csv").query("setting == 'audio_hubert'")
          .assign(kind="audio_hubert")[["kind", "target", "q"]]])
    names = [f"{a}-{b}" for a, b in PAIRS]
    own = []
    e13 = pd.read_csv(OUT / "snapshots" / "20261002-1339_e13" / "e13_pairs.csv")
    for kind in kinds:
        e = e13[e13.kind == kind] if kind in set(e13.kind) else measure(kind)
        name_map = dict(zip(sorted(e.pair.unique(), key=lambda p: PAIR_ORDER.index(p)), names))
        own.append(e.assign(pair=e.pair.map(name_map))[["kind", "target", "source", "pair", "own_d"]])
    own = pd.concat(own)
    mp = {k: load_map_full(k) for k in kinds}
    Cs = pd.concat([v[0] for v in mp.values()])
    C_mode = pd.concat([v[1] for v in mp.values()])
    Q = pd.concat([switch_q(k, C_mode[C_mode.kind == k]) for k in kinds])
    X = X.merge(Cs, on=["kind", "target", "source", "K", "rep"])
    P = (X.groupby(["kind", "target", "source", "K", "pair"])
         .agg(I_obs=("I_obs", "mean"), N_obs=("N_obs", "mean"), d_s=("d_s", "mean")).reset_index()
         .merge(lam, on=["kind", "target", "source", "K"]).merge(o_s, on=["kind", "target", "source", "K", "pair"])
         .merge(Q, on=["kind", "target", "source", "K"]).merge(curves, on=["kind", "target"])
         .merge(own, on=["kind", "target", "source", "pair"]))
    P["switch_pred"] = P.lam_obs ** 2 * P.o_s ** 2 * P.Q
    P["info_pred"] = (P.own_d / P.d_s - 1) * P.K.astype(int) / (P.K.astype(int) + P.q)
    T = []
    for kind, g in P.groupby("kind", sort=False):
        for K, h in g.groupby("K"):
            h2 = X[(X.kind == kind) & (X.K == K)]
            diff, same = h2[h2.C_m != h2.C_s].N_obs, h2[h2.C_m == h2.C_s].N_obs
            u = stats.mannwhitneyu(diff, same, alternative="greater") if len(diff) and len(same) else None
            r1, p1 = stats.spearmanr(h.switch_pred, h.N_obs, alternative="greater")
            T.append({"setting": KINDS[kind], "K": K, "n": len(h),
                      "N1_spearman": r1, "N1_p": p1,
                      "N2_share_C_differs": float((h2.C_m != h2.C_s).mean()),
                      "N2_median_N_differs": diff.median(), "N2_median_N_same": same.median(),
                      "N2_auc": u.statistic / (len(diff) * len(same)) if u else np.nan,
                      "N2_p": u.pvalue if u else np.nan})
        r, p = stats.spearmanr(g.info_pred, g.I_obs, alternative="greater")
        T.append({"setting": KINDS[kind], "K": "all", "n": len(g), "I1_spearman": r, "I1_p": p})
    return P, pd.DataFrame(T)


PAIR_ORDER = ["neutral-happy", "neutral-sad", "neutral-angry", "happy-sad", "happy-angry", "sad-angry"]


def load_map_full(kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """反復ごとの合併の C とソースのみの C，組 × K の C の最頻値（反復 0〜9）。"""
    fs = sorted(map_dir(kind).glob("*.csv"))
    d = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str},
                               usecols=["target", "source", "method", "K", "rep", "C"]) for f in fs])
    m = d[(d.method == "merged") & d.K.isin(KS) & (d.rep < 10)].rename(columns={"C": "C_m"})
    s_ = d[(d.method == "src_only") & (d.rep < 10)][["target", "source", "rep", "C"]].rename(columns={"C": "C_s"})
    Cs = m.merge(s_, on=["target", "source", "rep"])[["target", "source", "K", "rep", "C_m", "C_s"]].assign(kind=kind)
    mode = (m.groupby(["target", "source", "K"]).C_m.agg(lambda c: c.mode().iloc[0]).reset_index()
            .rename(columns={"C_m": "C"}).assign(kind=kind))
    return Cs, mode


def summarize(R: pd.DataFrame, D: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    R = R.assign(dc_obs=R.d_m - R.d_s, dc_pred=R.d_pred - R.d_s)
    P = R.groupby(["kind", "target", "source", "K"]).agg(
        dc_obs=("dc_obs", "mean"), dc_pred=("dc_pred", "mean"), d_s=("d_s", "mean")).reset_index()
    lam = []
    for key, g in R.groupby(["kind", "target", "source", "K"]):
        v = g[g.o_s.abs() >= O_MIN]
        f = lambda col: 1 - float((np.sign(v.o_s) * v[col]).sum() / v.o_s.abs().sum()) if len(v) else np.nan  # noqa: E731
        lam.append({**dict(zip(["kind", "target", "source", "K"], key)), "lam_obs": f("o_m"), "lam_newton": f("o_pred")})
    P = P.merge(pd.DataFrame(lam), on=["kind", "target", "source", "K"])
    Dp = D.groupby(["kind", "target", "source", "K"])[["I", "S", "N"]].mean().reset_index()
    P = P.merge(Dp, on=["kind", "target", "source", "K"])
    T = []
    for kind, g in P.groupby("kind", sort=False):
        row = {"setting": KINDS[kind], "n": len(g)}
        row["spearman_dc"], row["p_dc"] = stats.spearmanr(g.dc_pred, g.dc_obs, alternative="greater")
        row["spearman_lam"], row["p_lam"] = stats.spearmanr(g.lam_newton, g.lam_obs, alternative="greater",
                                                            nan_policy="omit")
        e = R[R.kind == kind]
        row["spearman_dc_emotionpair"] = stats.spearmanr(e.d_pred - e.d_s, e.d_m - e.d_s)[0]
        for K, h in g.groupby("K"):
            row[f"dc_obs_K{K}"], row[f"dc_pred_K{K}"] = h.dc_obs.mean(), h.dc_pred.mean()
            row[f"spearman_dc_K{K}"] = stats.spearmanr(h.dc_pred, h.dc_obs)[0]
            row[f"lam_obs_K{K}"], row[f"lam_newton_K{K}"] = h.lam_obs.mean(), h.lam_newton.mean()
            row[f"I_K{K}"], row[f"S_K{K}"], row[f"N_K{K}"] = h.I.mean(), h.S.mean(), h.N.mean()
        T.append(row)
    return P, pd.DataFrame(T), Dp


def fig(P: pd.DataFrame, path: Path) -> None:
    kinds = list(dict.fromkeys(P.kind))
    f, axes = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 4), squeeze=False)
    for ax, k in zip(axes[0], kinds):
        for K, h in P[P.kind == k].groupby("K"):
            ax.scatter(h.dc_pred, h.dc_obs, s=12, label=f"K={K}")
        lim = [P.dc_obs.min() - 0.02, P.dc_obs.max() + 0.02]
        ax.plot(lim, lim, "k--", lw=1); ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(KINDS[k]); ax.set_xlabel("predicted d_m - d_s"); ax.set_ylabel("observed d_m - d_s")
        ax.legend(fontsize=7)
    f.tight_layout(); f.savefig(path, dpi=120); plt.close(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    ap.add_argument("--kappa-floor", type=float, default=0.0,
                    help="抜いた K 本の曲がりの下限（E20 の R3）．0 より大きいと出力名に _kf<値> を付ける")
    ap.add_argument("--stage", choices=["predict", "diag", "drivers"], default="predict",
                    help="diag：段階 B の実測の d の変化を，新しく加わった部分の情報と雑音に分ける（6 節）")
    args = ap.parse_args()
    global KAPPA_FLOOR
    KAPPA_FLOOR = args.kappa_floor
    sfx = f"_kf{args.kappa_floor:g}" if args.kappa_floor > 0 else ""
    if args.stage == "drivers":
        P, T = run_drivers(args.kinds)
        P.to_csv(OUT / "e16_drivers_rows.csv", index=False)
        T.to_csv(OUT / "e16_drivers_tests.csv", index=False)
        pd.set_option("display.width", 250)
        print(T.round(3).to_string(index=False))
        return 0
    if args.stage == "diag":
        X = pd.concat([run_diag(k) for k in args.kinds])
        X.to_csv(OUT / "e16_diag_rows.csv", index=False)
        med, T = summarize_diag(X)
        med.to_csv(OUT / "e16_diag_median.csv", index=False)
        T.to_csv(OUT / "e16_diag_tests.csv", index=False)
        pd.set_option("display.width", 250)
        print(med.round(3).to_string(index=False))
        print(T.round(3).to_string(index=False))
        return 0
    Rs, Ds = [], []
    for kind in args.kinds:
        t0 = time.time()
        R, D = run_kind(kind)
        Rs.append(R); Ds.append(D)
        print(f"{kind}: {time.time() - t0:.0f} 秒", flush=True)
    R, D = pd.concat(Rs), pd.concat(Ds)
    R.to_csv(OUT / f"e16_rows{sfx}.csv", index=False)
    D.to_csv(OUT / f"e16_decomp_rows{sfx}.csv", index=False)
    P, T, _ = summarize(R, D)
    P.to_csv(OUT / f"e16_pairs{sfx}.csv", index=False)
    T.to_csv(OUT / f"e16_tests{sfx}.csv", index=False)
    fig(P, OUT / f"fig_e16{sfx}.png")
    pd.set_option("display.width", 250)
    print(T.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
