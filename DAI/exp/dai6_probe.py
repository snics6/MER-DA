"""DAI-6 の下調べ（`DAI/docs/実験6_下調べ_設計.md`）：モダリティをまたいで一緒に動く方向（正準相関の方向）に，感情の情報がどれだけあり，何が占めるか。

単位は c × 分割 f（c は 4 コーパス，f = 0〜4．評価のターゲット T に依らない）．手法の形の行と Q4 だけ (T, c, f)（アンカーが T と c で決まる）．
特徴は第 1 エンコーダ（`dai_data.load_corpus`），候補（`pool_mask`）の平均と標準偏差で候補とテストを標準化する（実験 1 の単位と同じ）．
K 本は使わない．候補の全ラベルは測るためだけに使い，部分空間はラベル無しで作る（設計 0・2 節）．

- 3.1 主成分：モダリティごとに p = min(200, ⌊0.022 n⌋) 次元．白色化した主成分の座標で正準相関を解く．
- 3.2 版：(i) 音声–言葉，顔–言葉（言葉の部分空間は 2 つの組の言葉の側の方向の和の空間），(ii) 残り 2 つをつないだものを相手にする，
  (iii) 音声–顔，(iv) (i) を塊の中で中心化した特徴で解く（IEMOCAP は `speaker_id`，ほかは `cluster`．部分空間は中心化した特徴で求め，
  分類器は中心化していない特徴で学ぶ）．固定のリッジ γ = 1e−3．
- 3.3 方向の数：k 番目は，両方の側からそれぞれ自分の 1〜k−1 番目の正準の変量を除いた残りの最大の正準相関を，発話の対応を入れ替えた帰無
  （N_PERM 回．(iv) は塊の中だけで入れ替える）の 95% 点と比べ，最初に越えなかった k で止める（r = k − 1，上限 K_MAX）．
  部分空間の基底は，除く前の問の上位 r 個の正準の方向．
- 3.4 Q1：白色化した座標で相互の共分散を感情の間の部分と感情の中の部分に分ける（1/n の正規化で厳密に和になる）．
- 3.5 Q2：帰無を越えた各変量の塊の η²（偶然の期待値 (k − 1)/(n − 1) と，塊の名前を入れ替えた帰無の 95% 点），音声の `duration_sec`・
  言葉の `n_token` との順位相関，IEMOCAP のマーカー（`DAI/frozen/subsets.csv`），AFEW の `compression_ratio`，参考に感情の η²．
- 3.6 Q3：部分空間 V（S，同じ次元の主成分，同じ次元のでたらめ 5 回，主成分の全部，元の全部の次元）で候補の全ラベルから学んだ分類器のテストの
  UAR．情報の量の行は重み 0 から（C は段 source の格子 12 点を候補の中の塊の 3 分割の交差検証で選ぶ）．手法の形の行は
  W = P_S W0 + ΔW（ΔW ∈ S．モダリティのブロックごとに Σ_mm の計量で射影）を実験 3 の `oracle_tether` の引き戻しで解く（版 (i)・(iv)）．
- 3.7 Q4：アンカーの誤差 W0 − W_full のうち Σ_mm の計量で部分空間の外にある割合と，K = 0 の射影 P_S W0 のテストの UAR（切片そのまま／
  候補の上で予測の感情ごとの総量をそろえた切片）．
- 3.8 安定：候補を塊で 2 つに分け，それぞれで主成分と正準相関を解いた上位 r 個（r は候補の全体の値）の方向の部分空間の主角の余弦の 2 乗の平均．
- 4 節の判定は `--summarize` が `criteria.csv` に書く．

使い方：
  ./.venv/bin/python DAI/exp/dai6_probe.py --selftest
  ./.venv/bin/python DAI/exp/dai6_probe.py --out <dir> [--corpora AFEW MOSEI] [--folds 0] [--n-perm 1000]
  ./.venv/bin/python DAI/exp/dai6_probe.py --out <dir> --summarize
出力は設計 5 節の表（`units.csv`，`seqtest.csv`，`decomp.csv`，`variates.csv`，`subspace_uar.csv`，`anchor_outside.csv`，`k0_proj.csv`，
`stability.csv`，`criteria.csv`，`fits/`，`provenance.json`）．単位ごとに追記し，同じ単位の行があれば止まる．
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
sys.path.insert(0, str(ROOT / "measure"))
sys.path.insert(0, str(ROOT / "features"))
import common  # noqa: E402
import data as D  # noqa: E402
from split import speaker_folds  # noqa: E402

SRC_OUT = ROOT / "results" / "dai1"                 # ソースの分類器（段 source）
MODS = ("A", "V", "T")
GAMMA = 1e-3
K_MAX = 30
N_PERM_ETA = 200
N_PERM_DECOMP = 200
N_RAND = 5
CV_FOLDS = 3
CHUNK = 25                                          # 並べ替えを一度に何回ぶん GPU に載せるか
VERSIONS = ("i", "ii", "iii", "iv")
PAIRS = {"i": [("A", "T"), ("V", "T")], "ii": [("A", "VT"), ("V", "AT"), ("T", "AV")], "iii": [("A", "V")], "iv": [("A", "T"), ("V", "T")]}
METHOD_VERSIONS = ("i", "iv")
DT = S.DT
DEV = S.DEV


def pdim(n: int) -> int:
    return min(200, int(math.floor(0.022 * n)))


def tt(x, dtype=DT):
    return S.t(x, dtype)


# ------------------------------------------------------------------ 線形代数の道具

def inv_sqrt(Sm: torch.Tensor) -> torch.Tensor:
    ev, V = torch.linalg.eigh(Sm)
    ev = ev.clamp(min=1e-12)
    return (V * ev.rsqrt()) @ V.T


def pca(X: torch.Tensor, p: int):
    """X は平均 0（候補の平均を引いた後）．戻り：E (d×p)，λ (p)，白色化した座標 Z (n×p)，残した分散の割合。"""
    n = X.shape[0]
    C = X.T @ X / (n - 1)
    ev, V = torch.linalg.eigh(C)
    order = torch.argsort(ev, descending=True)
    ev, V = ev[order], V[:, order]
    E, lam = V[:, :p], ev[:p].clamp(min=1e-12)
    Z = (X @ E) * lam.rsqrt()
    return E, lam, Z, float(lam.sum() / ev.clamp(min=0).sum())


def cca(Za: torch.Tensor, Zb: torch.Tensor, gamma: float = GAMMA):
    """正則化した正準相関．戻り：正準相関 s，a・b の方向（各座標）。"""
    n = Za.shape[0]
    Saa = Za.T @ Za / (n - 1) + gamma * torch.eye(Za.shape[1], device=DEV, dtype=Za.dtype)
    Sbb = Zb.T @ Zb / (n - 1) + gamma * torch.eye(Zb.shape[1], device=DEV, dtype=Zb.dtype)
    Ka, Kb = inv_sqrt(Saa), inv_sqrt(Sbb)
    M = Ka @ (Za.T @ Zb / (n - 1)) @ Kb
    U, s, Vh = torch.linalg.svd(M, full_matrices=False)
    return s, Ka @ U, Kb @ Vh.T


def residual_own(Z: torch.Tensor, Uv: torch.Tensor | None) -> torch.Tensor:
    """Z から，自分の側の変量（列）の張る空間への射影を除く。"""
    if Uv is None or Uv.shape[1] == 0:
        return Z
    Q, _ = torch.linalg.qr(Uv)
    return Z - Q @ (Q.T @ Z)


def make_perms(n: int, n_perm: int, groups: np.ndarray | None, seed: int) -> np.ndarray:
    """(n_perm, n) の並べ替え．groups を渡すと塊の中だけで入れ替える（Winkler ほか 2015 の交換できる塊）。"""
    rng = np.random.default_rng(seed)
    if groups is None:
        return np.stack([rng.permutation(n) for _ in range(n_perm)])
    codes = pd.factorize(groups)[0]
    base = np.argsort(codes, kind="stable")
    out = np.empty((n_perm, n), dtype=np.int64)
    for j in range(n_perm):
        order = np.lexsort((rng.random(n), codes))
        out[j, base] = order
    assert (codes[out] == codes[None, :]).all()
    return out


def perm_null_max(Ra: torch.Tensor, Rb: torch.Tensor, perms: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Rb の行を並べ替えたときの最大の正準相関（float32 で計算）。"""
    n = Ra.shape[0]
    Saa = Ra.T @ Ra / (n - 1) + gamma * torch.eye(Ra.shape[1], device=DEV, dtype=Ra.dtype)
    Sbb = Rb.T @ Rb / (n - 1) + gamma * torch.eye(Rb.shape[1], device=DEV, dtype=Rb.dtype)
    Ya = (Ra @ inv_sqrt(Saa)).float()
    Yb = (Rb @ inv_sqrt(Sbb)).float()
    out = []
    for s0 in range(0, len(perms), CHUNK):
        P = torch.as_tensor(perms[s0:s0 + CHUNK], device=DEV)
        M = torch.einsum("np,bnq->bpq", Ya, Yb[P]) / (n - 1)
        out.append(torch.linalg.matrix_norm(M, ord=2).double().cpu().numpy())
    return np.concatenate(out)


def seq_test(Za: torch.Tensor, Zb: torch.Tensor, n_perm: int, groups: np.ndarray | None, seed: int):
    """3.3 の順の検定．戻り：r，検定の記録の行，除く前の問の正準相関と方向。"""
    s_full, Da, Db = cca(Za, Zb)
    rows, Ua, Ub, r = [], None, None, 0
    for k in range(1, min(K_MAX, Za.shape[1], Zb.shape[1]) + 1):
        Ra, Rb = residual_own(Za, Ua), residual_own(Zb, Ub)
        s, da, db = cca(Ra, Rb)
        tp = time.time()
        perms = make_perms(Za.shape[0], n_perm, groups, seed + k)
        null = perm_null_max(Ra, Rb, perms)
        thr = float(np.quantile(null, 0.95))
        rows.append({"k": k, "perm_sec": time.time() - tp, "rho_obs": float(s[0]), "null_q95": thr, "null_mean": float(null.mean()),
                     "rho_full_k": float(s_full[k - 1]), "exceeds": bool(float(s[0]) > thr)})
        if not float(s[0]) > thr:
            break
        r = k
        ua, ub = (Ra @ da[:, :1]), (Rb @ db[:, :1])
        Ua = ua if Ua is None else torch.cat([Ua, ua], 1)
        Ub = ub if Ub is None else torch.cat([Ub, ub], 1)
    return r, rows, s_full, Da, Db


def sigma_orth(B: torch.Tensor, Sig: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
    """B の列の張る空間の，Σ の計量で正規直交な基底（BᵀΣB = I）。退化した列は落とす。"""
    if B.shape[1] == 0:
        return B
    G = B.T @ Sig @ B
    ev, V = torch.linalg.eigh((G + G.T) / 2)
    keep = ev > tol * ev.max().clamp(min=1e-30)
    return (B @ V[:, keep]) * ev[keep].rsqrt()


def proj_sigma(W: torch.Tensor, B: torch.Tensor, Sig: torch.Tensor) -> torch.Tensor:
    """W の行（4×d）を，Σ の計量で span(B) に射影する（B は BᵀΣB = I）。"""
    if B.shape[1] == 0:
        return torch.zeros_like(W)
    return ((W @ Sig @ B) @ B.T)


def sig_norm2(W: torch.Tensor, Sig: torch.Tensor) -> float:
    return float(torch.einsum("kd,de,ke->", W, Sig, W))


def principal_cos2(B1: torch.Tensor, B2: torch.Tensor, Sig: torch.Tensor) -> float:
    """Σ の計量で正規直交な基底どうしの主角の余弦の 2 乗の平均（小さい方の次元で数える）。"""
    m = min(B1.shape[1], B2.shape[1])
    if m == 0:
        return float("nan")
    sv = torch.linalg.svdvals(B1.T @ Sig @ B2)
    return float((sv[:m].clamp(max=1.0) ** 2).mean())


# ------------------------------------------------------------------ 分類器の学び直し

def uar_np(y: np.ndarray, Z: torch.Tensor) -> float:
    return 100.0 * S.uar(y, Z.argmax(-1).cpu().numpy())


def fit_info(F: torch.Tensor, Ft: torch.Tensor, y: np.ndarray, yt: np.ndarray, groups: np.ndarray) -> dict:
    """重み 0 から，感情の釣り合いの重みつきの交差エントロピー（C_b = ∞）．C は候補の中の塊の 3 分割の交差検証で選ぶ。"""
    if F.shape[1] == 0:
        return {"C": float("nan"), "cv_uar": float("nan"), "uar": 25.0, "grad": 0.0, "ok": True, "empty": True, "W": None, "b": None,
                "pred": np.zeros(len(yt), dtype=np.int8), "cv": None, "predC": None}
    Cs = np.array(dai1.C_GRID)
    folds = speaker_folds(pd.Series(groups), k=CV_FOLDS, seed=0).to_numpy()
    cv = np.zeros((CV_FOLDS, len(Cs)))
    for j in range(CV_FOLDS):
        tr, va = np.flatnonzero(folds != j), np.flatnonzero(folds == j)
        G = S.balanced_onehot(y[tr], np.arange(len(tr)), len(tr))[None].expand(len(Cs), -1, -1)
        r = S.fit(F[tr], G, torch.zeros(4, F.shape[1]), torch.zeros(4), Cw=Cs, Cb=math.inf)
        cv[j] = S.uar_batch(y[va], S.logits(F[va], r["W"], r["b"]))
    mean = cv.mean(0)
    best = int(np.flatnonzero(mean == mean.max())[0])                  # 同点なら小さい C（強い正則化）
    G = S.balanced_onehot(y, np.arange(len(y)), len(y))[None].expand(len(Cs), -1, -1)
    r = S.fit(F, G, torch.zeros(4, F.shape[1]), torch.zeros(4), Cw=Cs, Cb=math.inf)   # 12 個の C すべての最終の分類器（C の規則の感度のため）
    ZtC = S.logits(Ft, r["W"], r["b"])
    W, b = r["W"][best], r["b"][best]
    return {"C": float(Cs[best]), "cv_uar": float(mean[best]), "uar": uar_np(yt, ZtC[best]), "grad": float(r["grad_p"][best]), "ok": r["ok"],
            "empty": False, "W": W, "b": b, "pred": ZtC[best].argmax(1).cpu().numpy().astype(np.int8), "cv": cv,
            "predC": ZtC.argmax(2).cpu().numpy().astype(np.int8), "C_edge": bool(best in (0, len(Cs) - 1))}


def stash(store: dict, key: str, res: dict) -> None:
    """情報の量の行の保存（設計 5 節．gate-code の 1・2）：選んだ C の W・b・判定，交差検証の表，12 個の C の判定。"""
    store[f"pred_{key}"] = res["pred"]
    if res["W"] is not None:
        store[f"W_{key}"] = res["W"].float().cpu().numpy(); store[f"b_{key}"] = res["b"].float().cpu().numpy()
        store[f"cv_{key}"] = res["cv"].astype(np.float32); store[f"predC_{key}"] = res["predC"]


def balance_intercept(Zp: torch.Tensor, b: torch.Tensor, iters: int = 50) -> torch.Tensor:
    """候補の上で予測の感情ごとの確率の平均が 1/4 になるよう切片を直す（ラベルを使わない）。"""
    b = b.clone()
    for _ in range(iters):
        m = torch.softmax(Zp + b, 1).mean(0)
        b = b - torch.log(m * 4.0)
    return b


# ------------------------------------------------------------------ 単位

class ProbeUnit:
    def __init__(self, c: str, f: int):
        self.c, self.f = c, f
        cp = dai1.Corpus.get(c)
        self.cp = cp
        df = cp.df
        self.pool = np.flatnonzero(DD.pool_mask(df, f))
        self.test = np.flatnonzero(df["fold"].to_numpy() == f)
        self.y, self.yt = cp.y[self.pool], cp.y[self.test]
        self.n = len(self.pool)
        self.p = pdim(self.n)
        self.cluster = df["cluster"].to_numpy()[self.pool].astype(str)
        self.center_group = (df["speaker_id"].astype(str).to_numpy()[self.pool] if c == "IEMOCAP" else self.cluster)
        self.cluster_codes = pd.factorize(self.cluster)[0].astype(np.int64)
        self.center_codes = pd.factorize(self.center_group)[0].astype(np.int64)
        self.Xp, self.Xt, self.Sig = {}, {}, {}
        for m in MODS:
            X = cp.X[m]
            mu, sd = X[self.pool].mean(0), X[self.pool].std(0).clamp(min=1e-6)
            self.Xp[m], self.Xt[m] = (X[self.pool] - mu) / sd, (X[self.test] - mu) / sd
            self.Sig[m] = self.Xp[m].T @ self.Xp[m] / (self.n - 1)
        self.meta = load_meta(c, df["sample_id"].to_numpy()[self.pool])

    def centered(self, m: str) -> torch.Tensor:
        X = self.Xp[m].clone()
        codes = pd.factorize(self.center_group)[0]
        idx = torch.as_tensor(codes, device=DEV)
        k = codes.max() + 1
        sums = torch.zeros(k, X.shape[1], device=DEV, dtype=X.dtype).index_add_(0, idx, X)
        cnt = torch.zeros(k, device=DEV, dtype=X.dtype).index_add_(0, idx, torch.ones(len(codes), device=DEV, dtype=X.dtype))
        return X - (sums / cnt[:, None])[idx]


def load_meta(c: str, sids: np.ndarray) -> pd.DataFrame:
    """Q2 の量（ラベルを使わない）：音声の duration_sec，言葉の n_token，IEMOCAP のマーカー，AFEW の compression_ratio。"""
    corpus_key = D.GROUP_SPEC[DD.GROUP[c]][0]
    out = pd.DataFrame({"sample_id": sids})
    za = common.load_cache(DD.ENC1["A"], corpus_key)
    zt = common.load_cache(DD.ENC1["T"], corpus_key)
    out = out.merge(pd.DataFrame({"sample_id": za["sample_id"].astype(str), "duration_sec": za["duration_sec"]}), on="sample_id", how="left")
    out = out.merge(pd.DataFrame({"sample_id": zt["sample_id"].astype(str), "n_token": zt["n_token"]}), on="sample_id", how="left")
    sub = pd.read_csv(ROOT / "DAI" / "frozen" / "subsets.csv")
    if c == "IEMOCAP":
        mk = sub[(sub.corpus == "IEMOCAP") & (sub.subset == "marker")][["sample_id", "value"]]
        out = out.merge(mk.assign(marker=(mk["value"] == "marker").astype(float))[["sample_id", "marker"]], on="sample_id", how="left")
    if c == "AFEW":
        w = pd.read_csv(common.META_DIR / "afew_transcript_whisper.csv")[["sample_id", "compression_ratio"]]
        out = out.merge(w, on="sample_id", how="left")
    assert len(out) == len(sids) and (out["sample_id"].to_numpy() == sids).all()
    return out


def eta2(u: np.ndarray, g: np.ndarray) -> float:
    codes = g if np.issubdtype(np.asarray(g).dtype, np.integer) else pd.factorize(g)[0]
    tot = ((u - u.mean()) ** 2).sum()
    means = np.bincount(codes, weights=u) / np.bincount(codes)
    between = (np.bincount(codes) * (means - u.mean()) ** 2).sum()
    return float(between / tot) if tot > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float("nan")
    return float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))


def variate_rows(u: np.ndarray, side: str, k: int, U: ProbeUnit, base: dict, rng: np.random.Generator) -> dict:
    g = U.cluster_codes
    e = eta2(u, g)
    null = np.array([eta2(u, rng.permutation(g)) for _ in range(N_PERM_ETA)])
    row = {**base, "side": side, "k": k, "eta2_cluster": e, "eta2_cluster_chance": (g.max()) / (len(u) - 1),
           "eta2_cluster_null_q95": float(np.quantile(null, 0.95)), "eta2_emotion": eta2(u, U.y.astype(np.int64)),
           "rho_duration": spearman(u, U.meta["duration_sec"].to_numpy(float)), "rho_ntoken": spearman(u, U.meta["n_token"].to_numpy(float))}
    if U.c == "IEMOCAP":
        sc = U.center_codes
        row["eta2_speaker"] = eta2(u, sc)
        row["eta2_speaker_chance"] = sc.max() / (len(u) - 1)
        row["eta2_speaker_null_q95"] = float(np.quantile([eta2(u, rng.permutation(sc)) for _ in range(N_PERM_ETA)], 0.95))
        row["rho_marker"] = spearman(u, U.meta["marker"].to_numpy(float))
    if U.c == "AFEW":
        row["rho_compression"] = spearman(u, U.meta["compression_ratio"].to_numpy(float))
    return row


def _parts(Ya: torch.Tensor, Yb: torch.Tensor, y: np.ndarray):
    n = Ya.shape[0]
    T = Ya.T @ Yb / n
    Bt = torch.zeros_like(T)
    for k in range(4):
        ix = torch.as_tensor(np.flatnonzero(y == k), device=DEV)
        if len(ix):
            Bt += (len(ix) / n) * torch.outer(Ya[ix].mean(0), Yb[ix].mean(0))
    Wt = T - Bt
    out = {"fro_total": float(T.norm()), "fro_between": float(Bt.norm()), "fro_within": float(Wt.norm()),
           "fro2_total": float(T.norm() ** 2), "fro2_between": float(Bt.norm() ** 2), "fro2_within": float(Wt.norm() ** 2)}
    for nm, M in (("total", T), ("between", Bt), ("within", Wt)):
        for j, x in enumerate(torch.linalg.svdvals(M)[:5].cpu().numpy()):
            out[f"sv_{nm}_{j + 1}"] = float(x)
    return out


def decomp_row(Za: torch.Tensor, Zb: torch.Tensor, y: np.ndarray, base: dict, seed: int) -> dict:
    """Q1：両側を白色化した座標で，相互の共分散 = 感情の間 + 感情の中（1/n の正規化で厳密）．
    帰無（3.4・3.4.1）：発話の対応を全体で入れ替えた 200 回（null_）と，感情の中だけで入れ替えた 200 回（wnull_）の平均と 95% 点。"""
    n = Za.shape[0]
    Ya = (Za - Za.mean(0)) @ inv_sqrt((Za - Za.mean(0)).T @ (Za - Za.mean(0)) / n + GAMMA * torch.eye(Za.shape[1], device=DEV, dtype=DT))
    Yb = (Zb - Zb.mean(0)) @ inv_sqrt((Zb - Zb.mean(0)).T @ (Zb - Zb.mean(0)) / n + GAMMA * torch.eye(Zb.shape[1], device=DEV, dtype=DT))
    row = {**base, **_parts(Ya, Yb, y)}
    for tag, groups in (("null", None), ("wnull", y)):
        perms = make_perms(n, N_PERM_DECOMP, groups, seed + (0 if groups is None else 1))
        vals = pd.DataFrame([_parts(Ya, Yb[torch.as_tensor(P, device=DEV)], y) for P in perms])
        for col in vals.columns:
            row[f"{tag}_mean_{col}"] = float(vals[col].mean())
            row[f"{tag}_q95_{col}"] = float(vals[col].quantile(0.95))
    return row


def run_unit(U: ProbeUnit, n_perm: int, fz3: dict, out: Path, commit: str, dirty: bool) -> dict:
    t0 = time.time()
    tabs = {k: [] for k in ("units", "seqtest", "decomp", "variates", "subspace_uar", "anchor_outside", "k0_proj", "stability")}
    for k in tabs:                                 # 重複の確かめは fits/ を書く前に行う（gate-code の 11）
        pth = out / f"{k}.csv"
        if pth.exists():
            old = pd.read_csv(pth, low_memory=False, usecols=["c", "fold"])
            if ((old["c"] == U.c) & (old["fold"] == U.f)).any():
                sys.exit(f"{pth} に {U.c} 分割 {U.f} の行が既にある")
    print(f"{U.c} 分割 {U.f}：候補 {U.n}，p = {U.p} を始める", flush=True)
    ubase = {"c": U.c, "fold": U.f, "n_pool": U.n, "n_test": len(U.test), "p": U.p, "commit": commit, "dirty": dirty}
    preds = {}                                     # fits/ に保存する配列
    rng = np.random.default_rng(1000 * DD.CORPUS_ID[U.c] + U.f)
    # 主成分（中心化しない版と，塊の中で中心化した版）
    pc = {}
    for cen in (False, True):
        for m in MODS:
            X = U.centered(m) if cen else U.Xp[m]
            E, lam, Z, kept = pca(X, U.p)
            pc[(cen, m)] = (E, lam, Z)
            tabs["units"].append({**ubase, "centered": cen, "mod": m, "var_kept": kept})
    dirs, dirs3 = {}, {}                          # (版, モダリティ) → 方向の行列（元の標準化した座標，d × r）の並び．dirs3 は上位 3 個に固定
    for v in VERSIONS:
        cen = v == "iv"
        groups = U.center_group if cen else None
        for (a, b) in PAIRS[v]:
            Za = pc[(cen, a)][2]
            Zb = torch.cat([pc[(cen, m)][2] for m in b], 1) if len(b) > 1 else pc[(cen, b)][2]
            pbase = {**ubase, "version": v, "a": a, "b": b}
            ts = time.time()
            r, rows, s_full, Da, Db = seq_test(Za, Zb, n_perm, groups, seed=7919 * DD.CORPUS_ID[U.c] + 101 * U.f + VERSIONS.index(v))
            tabs["seqtest"] += [{**pbase, **x} for x in rows]
            tabs["decomp"].append(decomp_row(Za, Zb, U.y, {**pbase, "r": r}, seed=31 * DD.CORPUS_ID[U.c] + 7 * U.f + VERSIONS.index(v)))
            Ea, la = pc[(cen, a)][0], pc[(cen, a)][1]
            wa = (Ea * la.rsqrt()) @ Da[:, :r]                                 # a の側の方向（元の標準化した座標）
            dirs.setdefault((v, a), []).append(wa)
            dirs3.setdefault((v, a), []).append((Ea * la.rsqrt()) @ Da[:, :3])
            if len(b) == 1:
                Eb, lb = pc[(cen, b)][0], pc[(cen, b)][1]
                dirs.setdefault((v, b), []).append((Eb * lb.rsqrt()) @ Db[:, :r])
                dirs3.setdefault((v, b), []).append((Eb * lb.rsqrt()) @ Db[:, :3])
            ua = (Za @ Da[:, :r]).cpu().numpy()
            ub = ((Zb @ Db[:, :r]).cpu().numpy())
            for k in range(r):
                tabs["variates"].append(variate_rows(ua[:, k], "a", k + 1, U, pbase, rng))
                tabs["variates"].append(variate_rows(ub[:, k], "b", k + 1, U, pbase, rng))
            kc = min(K_MAX, Da.shape[1])
            preds[f"cca_rho_{v}_{a}_{b}"] = s_full[:kc].float().cpu().numpy()
            preds[f"cca_dir_{v}_{a}_{b}_a"] = ((Ea * la.rsqrt()) @ Da[:, :kc]).float().cpu().numpy()
            if len(b) == 1:
                preds[f"cca_dir_{v}_{a}_{b}_b"] = ((pc[(cen, b)][0] * pc[(cen, b)][1].rsqrt()) @ Db[:, :kc]).float().cpu().numpy()
            perm_sec = sum(x["perm_sec"] for x in rows)
            tabs["units"].append({**pbase, "r": r, "r_capped": bool(r >= K_MAX), "r_capped_p": bool(r >= min(K_MAX, Za.shape[1], Zb.shape[1])),
                                  "rho_1": float(s_full[0]), "sec": time.time() - ts, "perm_sec": perm_sec})
            print(f"  {U.c} f{U.f} 版 {v} {a}–{b}：段 {len(rows)}，並べ替え {perm_sec:.0f} 秒，計 {time.time() - ts:.0f} 秒", flush=True)
    # 部分空間（版 × モダリティ）．(i)・(iv) の言葉は 2 つの組の言葉の側の和，(ii) は各モダリティの a の側，(iii) は音声と顔だけ
    basis, basis3 = {}, {}
    for v in VERSIONS:
        for m in MODS:
            L = dirs.get((v, m), [])                  # (ii) は a の側だけが入る（b はつないだ 2 つなので方向を戻さない）
            if not L:
                continue
            B = sigma_orth(torch.cat(L, 1), U.Sig[m])
            basis[(v, m)] = B
            basis3[(v, m)] = sigma_orth(torch.cat(dirs3[(v, m)], 1), U.Sig[m])
            tabs["units"].append({**ubase, "version": v, "mod": m, "r_m": int(B.shape[1]), "r3_m": int(basis3[(v, m)].shape[1])})
    # Q3：情報の量の行
    students = ("A", "V", "T", "AVT")
    full_info = {}
    for s in students:
        mods = list(s) if s != "AVT" else list(MODS)
        groups = U.cluster
        F = torch.cat([U.Xp[m] for m in mods], 1); Ft = torch.cat([U.Xt[m] for m in mods], 1)
        res = fit_info(F, Ft, U.y, U.yt, groups)
        full_info[s] = res
        stash(preds, f"-_{s}_full", res)
        tabs["subspace_uar"].append({**ubase, "version": "-", "student": s, "space": "full", "row": "info", "dim": int(F.shape[1]),
                                     **{k: res[k] for k in ("C", "cv_uar", "uar", "grad", "ok", "empty")}, "C_edge": res.get("C_edge", False)})
        Fp = torch.cat([U.Xp[m] @ (pc[(False, m)][0] * pc[(False, m)][1].rsqrt()) for m in mods], 1)
        Fpt = torch.cat([U.Xt[m] @ (pc[(False, m)][0] * pc[(False, m)][1].rsqrt()) for m in mods], 1)
        res = fit_info(Fp, Fpt, U.y, U.yt, groups)
        stash(preds, f"-_{s}_pca_p", res)
        tabs["subspace_uar"].append({**ubase, "version": "-", "student": s, "space": "pca_p", "row": "info", "dim": int(Fp.shape[1]),
                                     **{k: res[k] for k in ("C", "cv_uar", "uar", "grad", "ok", "empty")}, "C_edge": res.get("C_edge", False)})
        for v in VERSIONS:
            if not all((v, m) in basis for m in mods):
                continue
            rm = {m: int(basis[(v, m)].shape[1]) for m in mods}
            r3 = {m: int(basis3[(v, m)].shape[1]) for m in mods}
            spaces = {"S": {m: basis[(v, m)] for m in mods},
                      "pca_r": {m: (pc[(False, m)][0][:, :rm[m]] * pc[(False, m)][1][:rm[m]].rsqrt()) for m in mods},
                      "S3": {m: basis3[(v, m)] for m in mods},
                      "pca_r3": {m: (pc[(False, m)][0][:, :r3[m]] * pc[(False, m)][1][:r3[m]].rsqrt()) for m in mods}}
            for j in range(N_RAND):
                g = torch.Generator(device="cpu").manual_seed(7 + 13 * j + 1009 * DD.CORPUS_ID[U.c] + 31 * U.f)
                Bs = {}
                for m in mods:
                    Q, _ = torch.linalg.qr(torch.randn(U.p, max(rm[m], 1), generator=g, dtype=torch.float64))
                    Bs[m] = (pc[(False, m)][0] * pc[(False, m)][1].rsqrt()) @ Q[:, :rm[m]].to(DEV)
                spaces[f"rand{j}"] = Bs
            for sp, Bm in spaces.items():
                F = torch.cat([U.Xp[m] @ Bm[m] for m in mods], 1); Ft = torch.cat([U.Xt[m] @ Bm[m] for m in mods], 1)
                res = fit_info(F, Ft, U.y, U.yt, groups)
                stash(preds, f"{v}_{s}_{sp}", res)
                if sp.startswith("rand"):
                    for m in mods:
                        preds[f"basis_{v}_{s}_{sp}_{m}"] = Bm[m].float().cpu().numpy()
                tabs["subspace_uar"].append({**ubase, "version": v, "student": s, "space": sp, "row": "info", "dim": int(F.shape[1]),
                                             "r_min": min(rm.values()), "r_lt3": any(x < 3 for x in rm.values()),
                                             **{k: res[k] for k in ("C", "cv_uar", "uar", "grad", "ok", "empty")}, "C_edge": res.get("C_edge", False)})
            # 3.8 安定（言葉を相手にした版と中心化した版だけ）
    halves_save = {}
    for v in ("i", "iv"):
        cen = v == "iv"
        halves = speaker_folds(pd.Series(U.cluster), k=2, seed=0).to_numpy()
        hb, hpc = {}, {}
        for h in (0, 1):
            ix = torch.as_tensor(np.flatnonzero(halves == h), device=DEV)
            hp = {}
            for m in MODS:
                X = (U.centered(m) if cen else U.Xp[m])[ix]
                X = X - X.mean(0)
                E, lam, Z, _ = pca(X, U.p)
                hp[m] = (E, lam, Z)
            dl = {}
            for (a, b) in PAIRS[v]:
                r = next(x["r"] for x in tabs["units"] if x.get("version") == v and x.get("a") == a and x.get("b") == b)
                s_, Da, Db = cca(hp[a][2], hp[b][2])
                dl.setdefault(a, []).append((hp[a][0] * hp[a][1].rsqrt()) @ Da[:, :r])
                dl.setdefault(b, []).append((hp[b][0] * hp[b][1].rsqrt()) @ Db[:, :r])
            hb[h] = {m: sigma_orth(torch.cat(L, 1), U.Sig[m]) for m, L in dl.items()}
            hpc[h] = hp
        for m in MODS:
            if m in hb[0] and m in hb[1] and (v, m) in basis:
                rm = int(basis[(v, m)].shape[1])
                P0 = sigma_orth(hpc[0][m][0][:, :rm], U.Sig[m]); P1 = sigma_orth(hpc[1][m][0][:, :rm], U.Sig[m])
                halves_save[f"half0_{v}_{m}"] = hb[0][m].float().cpu().numpy(); halves_save[f"half1_{v}_{m}"] = hb[1][m].float().cpu().numpy()
                tabs["stability"].append({**ubase, "version": v, "mod": m, "r_m": rm, "r_half0": int(hb[0][m].shape[1]), "r_half1": int(hb[1][m].shape[1]),
                                          "cos2_mean": principal_cos2(hb[0][m], hb[1][m], U.Sig[m]),
                                          "cos2_mean_pca": principal_cos2(P0, P1, U.Sig[m]), "cos2_random": rm / U.p if rm else float("nan")})
    # Q4・K = 0 の射影・手法の形の行（(T, c, f)）
    for T in DD.others(U.c):
        srcs = [s for s in DD.CORPORA if s not in (T, U.c)]
        for s in students:
            mods = list(s) if s != "AVT" else list(MODS)
            Ws, bs = zip(*[dai1.load_source(SRC_OUT, x, s) for x in srcs])
            W0, b0 = torch.stack(Ws).mean(0), torch.stack(bs).mean(0)
            dims = np.cumsum([0] + [U.Xp[m].shape[1] for m in mods])
            blk = {m: slice(int(dims[i]), int(dims[i + 1])) for i, m in enumerate(mods)}
            Wf = full_info[s]["W"]
            Xp = torch.cat([U.Xp[m] for m in mods], 1); Xt = torch.cat([U.Xt[m] for m in mods], 1)
            tb = {**ubase, "T": T, "student": s}
            for v in METHOD_VERSIONS:
                if not all((v, m) in basis for m in mods):
                    continue
                rm = {m: int(basis[(v, m)].shape[1]) for m in mods}
                spaces = {"S": {m: basis[(v, m)] for m in mods},
                          "pca_r": {m: (pc[(False, m)][0][:, :rm[m]] * pc[(False, m)][1][:rm[m]].rsqrt()) for m in mods}}
                for sp, Bm in spaces.items():
                    scale = math.sqrt(sum(sig_norm2(Wf[:, blk[m]], U.Sig[m]) for m in mods) / max(sum(sig_norm2(W0[:, blk[m]], U.Sig[m]) for m in mods), 1e-30))
                    num = den = numn = denn = 0.0
                    PW0 = torch.zeros_like(W0)
                    for m in mods:
                        D0 = W0[:, blk[m]] - Wf[:, blk[m]]
                        Pm = proj_sigma(D0, Bm[m], U.Sig[m])
                        num += sig_norm2(D0 - Pm, U.Sig[m]); den += sig_norm2(D0, U.Sig[m])
                        Dn = scale * W0[:, blk[m]] - Wf[:, blk[m]]                 # 大きさをそろえた版
                        numn += sig_norm2(Dn - proj_sigma(Dn, Bm[m], U.Sig[m]), U.Sig[m]); denn += sig_norm2(Dn, U.Sig[m])
                        PW0[:, blk[m]] = proj_sigma(W0[:, blk[m]], Bm[m], U.Sig[m])
                    tabs["anchor_outside"].append({**tb, "version": v, "space": sp, "share_outside": num / den if den > 0 else float("nan"),
                                                   "share_outside_normed": numn / denn if denn > 0 else float("nan"), "scale_W0": scale})
                    for nm, WW in (("W0", W0), ("P_W0", PW0)):
                        if nm == "W0" and (v, sp) != ("i", "S"):
                            continue
                        Zp, Zt = Xp @ WW.T, Xt @ WW.T
                        bb = balance_intercept(Zp, b0)
                        kk = f"{T}_{v if nm == 'P_W0' else '-'}_{s}_{sp if nm == 'P_W0' else 'full'}"
                        preds[f"k0W_{kk}"] = WW.float().cpu().numpy(); preds[f"k0b_{kk}"] = b0.float().cpu().numpy()
                        preds[f"k0bbal_{kk}"] = bb.float().cpu().numpy()
                        preds[f"k0pred_{kk}"] = (Zt + b0).argmax(1).cpu().numpy().astype(np.int8)
                        preds[f"k0predbal_{kk}"] = (Zt + bb).argmax(1).cpu().numpy().astype(np.int8)
                        tabs["k0_proj"].append({**tb, "version": v if nm == "P_W0" else "-", "space": sp if nm == "P_W0" else "full",
                                                "weights": nm, "uar_b0": uar_np(U.yt, Zt + b0), "uar_balanced": uar_np(U.yt, Zt + bb)})
                    # 手法の形の行：W = P W0 + ΔW，ΔW ∈ 部分空間（ブロックごとの Σ_mm の計量）．引き戻しは oracle_tether[T][s]
                    cw, cb = fz3["oracle_tether"][T][s]
                    Bo = torch.block_diag(*[torch.linalg.qr(Bm[m])[0] if Bm[m].shape[1] else Bm[m] for m in mods])
                    if Bo.shape[1] == 0:
                        continue
                    W0r = PW0 @ Bo
                    G = S.balanced_onehot(U.y, np.arange(U.n), U.n)[None]
                    r = S.fit(Xp @ Bo, G, W0r, b0, Cw=[cw], Cb=[cb])
                    Zt_m = (Xt @ Bo) @ r["W"][0].T + r["b"][0]
                    mk = f"{T}_{v}_{s}_{sp}"
                    preds[f"mW_{mk}"] = (r["W"][0] @ Bo.T).float().cpu().numpy(); preds[f"mb_{mk}"] = r["b"][0].float().cpu().numpy()
                    preds[f"mpred_{mk}"] = Zt_m.argmax(1).cpu().numpy().astype(np.int8)
                    tabs["subspace_uar"].append({**tb, "version": v, "space": sp, "row": "method", "dim": int(Bo.shape[1]), "C": cw, "Cb": cb,
                                                 "uar": uar_np(U.yt, Zt_m), "grad": r["grad_max"], "ok": r["ok"]})
            cw, cb = fz3["oracle_tether"][T][s]
            r = S.fit(Xp, S.balanced_onehot(U.y, np.arange(U.n), U.n)[None], W0, b0, Cw=[cw], Cb=[cb])
            Zt_m = Xt @ r["W"][0].T + r["b"][0]
            mk = f"{T}_-_{s}_full"
            preds[f"mW_{mk}"] = r["W"][0].float().cpu().numpy(); preds[f"mb_{mk}"] = r["b"][0].float().cpu().numpy()
            preds[f"mpred_{mk}"] = Zt_m.argmax(1).cpu().numpy().astype(np.int8)
            tabs["subspace_uar"].append({**tb, "version": "-", "space": "full", "row": "method", "dim": int(Xp.shape[1]), "C": cw, "Cb": cb,
                                         "uar": uar_np(U.yt, Zt_m), "grad": r["grad_max"], "ok": r["ok"]})
    # 保存
    fd = out / "fits"
    fd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fd / f"{U.c}_f{U.f}.npz", **{f"S_{v}_{m}": B.float().cpu().numpy() for (v, m), B in basis.items()},
                        **{f"pca_{'c' if cen else 'n'}_{m}": pc[(cen, m)][0].float().cpu().numpy() for (cen, m) in pc},
                        **{f"Wfull_{s}": full_info[s]["W"].float().cpu().numpy() for s in students if full_info[s]["W"] is not None},
                        **{f"S3_{v}_{m}": B.float().cpu().numpy() for (v, m), B in basis3.items()}, **halves_save, **preds,
                        pool_sample_id=U.cp.df["sample_id"].to_numpy()[U.pool].astype(str),
                        test_sample_id=U.cp.df["sample_id"].to_numpy()[U.test].astype(str),
                        test_cluster=U.cp.df["cluster"].to_numpy()[U.test].astype(str), test_y=U.yt.astype(np.int8))
    sec = time.time() - t0
    for k, rows in tabs.items():
        if not rows:
            continue
        p = out / f"{k}.csv"
        new = pd.DataFrame(rows)
        if p.exists():
            old = pd.read_csv(p, low_memory=False)
            if ((old["c"] == U.c) & (old["fold"] == U.f)).any():
                sys.exit(f"{p} に {U.c} 分割 {U.f} の行が既にある")
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(p, index=False)
    return {"c": U.c, "fold": U.f, "sec": sec}


# ------------------------------------------------------------------ 判定（4 節）

def _boot_fold_mean(per_fold: list, n_boot: int = 2000, seed: int = 0):
    """4 節のばらつき（gate-code の 5）：分割ごとにテストの塊を抜き直し（同じ抜き直しを全部の判定に使う），分割ごとの UAR の平均を作る．
    点推定（分割ごとの UAR の平均）と同じ量である．どこかの分割で感情が欠けた回は捨て，有効な回の数を返す。
    per_fold：[(y, cluster, {名前: 判定})]．戻り：({名前: (有効な回,) の UAR}, 有効な回の数)。"""
    rng = np.random.default_rng(seed)
    names = list(per_fold[0][2])
    acc = {nm: np.zeros(n_boot) for nm in names}
    valid = np.ones(n_boot, bool)
    for (y, cl, preds) in per_fold:
        codes = pd.factorize(cl)[0]
        k = codes.max() + 1
        n_kc = np.zeros((k, 4)); np.add.at(n_kc, (codes, y), 1)
        w = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(float)
        den = w @ n_kc
        valid &= (den > 0).all(1)
        for nm, pr in preds.items():
            cor = np.zeros((k, 4)); np.add.at(cor, (codes, y), (pr == y).astype(float))
            with np.errstate(invalid="ignore", divide="ignore"):
                acc[nm] += 100.0 * np.nanmean(np.where(den > 0, (w @ cor) / den, np.nan), 1) / len(per_fold)
    return {nm: v[valid] for nm, v in acc.items()}, int(valid.sum())


def summarize(out: Path, allow_partial: bool = False) -> None:
    """4 節の判定（criteria.csv），仮定の表（assumptions.csv），ラベル無しの診断（diagnostics.csv）。"""
    su = pd.read_csv(out / "subspace_uar.csv", low_memory=False)
    done = set(zip(su["c"], su["fold"]))
    want = {(c, f) for c in DD.CORPORA for f in range(DD.N_FOLDS)}
    commits = set(su["commit"].astype(str))
    if (done != want or len(commits) != 1) and not allow_partial:       # I12（gate-code の 10）
        sys.exit(f"単位がそろっていないかコミットが 1 つでない：欠け {sorted(want - done)}，コミット {sorted(commits)}")
    info = su[(su.row == "info") & (su.student == "AVT")]
    cm, dt = dai1.git_commit()
    rows = []
    for c in DD.CORPORA:
        x = info[info.c == c]
        if x.empty:
            continue
        full = x[x.space == "full"].set_index("fold")["uar"]
        Sx = x[(x.version == "i") & (x.space == "S")].set_index("fold")["uar"]
        pr = x[(x.version == "i") & (x.space == "pca_r")].set_index("fold")["uar"]
        folds = sorted(set(full.index) & set(Sx.index) & set(pr.index))
        if not folds:
            continue
        ratio = (Sx[folds].mean() - 25) / (full[folds].mean() - 25)
        gap = Sx[folds].mean() - pr[folds].mean()
        per_fold = []
        for f in folds:
            z = np.load(out / "fits" / f"{c}_f{f}.npz", allow_pickle=False)
            per_fold.append((z["test_y"].astype(int), z["test_cluster"],
                             {"full": z["pred_-_AVT_full"].astype(int), "S": z["pred_i_AVT_S"].astype(int), "pca_r": z["pred_i_AVT_pca_r"].astype(int)}))
        bt, n_valid = _boot_fold_mean(per_fold, seed=DD.CORPUS_ID[c])
        rb = (bt["S"] - 25) / (bt["full"] - 25); gb = bt["S"] - bt["pca_r"]
        r_lo, r_hi = np.nanpercentile(rb, [2.5, 97.5]); g_lo, g_hi = np.nanpercentile(gb, [2.5, 97.5])
        rows.append({"c": c, "n_folds": len(folds), "uar_full": full[folds].mean(), "uar_S": Sx[folds].mean(), "uar_pca_r": pr[folds].mean(),
                     "ratio": ratio, "ratio_lo": r_lo, "ratio_hi": r_hi, "gap_vs_pca": gap, "gap_lo": g_lo, "gap_hi": g_hi,
                     "cond1": bool(ratio >= 0.8), "cond2": bool(gap >= 3.0), "pass": bool(ratio >= 0.8 and gap >= 3.0),
                     "unstable": bool((r_lo < 0.8 < r_hi) or (g_lo < 3.0 < g_hi)), "n_boot_valid": n_valid,
                     **{f"ratio_f{f}": (Sx[f] - 25) / (full[f] - 25) for f in folds}, **{f"gap_f{f}": Sx[f] - pr[f] for f in folds}})
    cr = pd.DataFrame(rows)
    if len(cr):
        cr["n_pass"] = int(cr["pass"].sum())
        cr["proceed"] = bool(len(cr) == 4 and cr["pass"].sum() >= 2)
    cr.assign(commit=cm, dirty=dt).to_csv(out / "criteria.csv", index=False)
    # 仮定の表（3.4.1）
    de = pd.read_csv(out / "decomp.csv", low_memory=False)
    de = de[de.version.isin(["i", "ii", "iii", "iv"])].copy()
    de["A1"] = np.where(de["sv_within_1"] <= de["wnull_q95_sv_within_1"], "holds",
                        np.where(de["sv_within_1"] < de["sv_between_3"], "broken_top3_emotion", "broken_within_stronger"))
    de["A2"] = np.where(de["sv_between_3"] > de["null_q95_sv_between_3"], "holds", "not")
    for col in ("fro_total", "fro_between", "fro_within"):
        de[f"{col}_minus_null"] = de[col] - de[f"null_mean_{col}"]
    de[["c", "fold", "version", "a", "b", "r", "A1", "A2", "sv_within_1", "wnull_q95_sv_within_1", "sv_between_3", "null_q95_sv_between_3",
        "fro_total_minus_null", "fro_between_minus_null", "fro_within_minus_null"]].assign(commit=cm, dirty=dt).to_csv(out / "assumptions.csv", index=False)
    # ラベル無しの診断（3.5 の印の規則）
    va = pd.read_csv(out / "variates.csv", low_memory=False)
    va = va[va.version == "i"].copy()
    eta_col = np.where(va["c"] == "IEMOCAP", "speaker", "cluster")
    e = np.where(eta_col == "speaker", va.get("eta2_speaker", np.nan), va["eta2_cluster"])
    eq = np.where(eta_col == "speaker", va.get("eta2_speaker_null_q95", np.nan), va["eta2_cluster_null_q95"])
    ec = np.where(eta_col == "speaker", va.get("eta2_speaker_chance", np.nan), va["eta2_cluster_chance"])
    mark = (e > eq) & (e - ec >= 0.10)
    for col in ("rho_duration", "rho_ntoken", "rho_marker", "rho_compression"):
        if col in va:
            mark |= va[col].abs().fillna(0) >= 0.30
    va["mark"] = mark
    va["mod"] = np.where(va["side"] == "a", va["a"], va["b"])
    allu = pd.MultiIndex.from_tuples(sorted(set(zip(su["c"], su["fold"]))), names=["c", "fold"])
    rc = (va[~va["mark"]].groupby(["c", "fold", "mod"]).size().unstack("mod").reindex(columns=list(MODS))
          .reindex(allu).fillna(0))                                   # 印の無い変量が 0 の単位も残す（gate-code の 11）
    st = pd.read_csv(out / "stability.csv", low_memory=False)
    st = st[st.version == "i"].set_index(["c", "fold", "mod"])
    drows = []
    for (c, f), r in rc.iterrows():
        stab_ok = all((c, f, m) in st.index and st.loc[(c, f, m), "cos2_mean"] >= st.loc[(c, f, m), "cos2_mean_pca"] for m in MODS)
        drows.append({"c": c, "fold": f, **{f"r_clean_{m}": int(r[m]) for m in MODS}, "stab_ok": bool(stab_ok),
                      "fold_ok": bool(all(r[m] >= 3 for m in MODS) and stab_ok)})
    dg = pd.DataFrame(drows)
    if len(dg):
        agg = dg.groupby("c")["fold_ok"].sum().rename("n_folds_ok").reset_index()
        agg["predict_pass"] = agg["n_folds_ok"] >= 3
        if len(cr):
            agg = agg.merge(cr[["c", "pass"]], on="c", how="left")
            agg["agree"] = agg["predict_pass"] == agg["pass"]
        dg = dg.merge(agg, on="c", how="left")
    dg.assign(commit=cm, dirty=dt).to_csv(out / "diagnostics.csv", index=False)
    print("書いた:", out / "criteria.csv", out / "assumptions.csv", out / "diagnostics.csv")


# ------------------------------------------------------------------ 検算（人工データ）

def selftest() -> None:
    """ラベルを固定すると無相関な 3 つの見方の人工データで，(1) 上位 3 個の正準の方向の張る空間が Fisher の判別の部分空間と一致する，
    (2) 順の検定が r = 3 で止まる，(3) 共有の要因（話者）を足すと r が増え，塊の中で中心化すると r = 3 に戻る，(4) 塊の中の並べ替えが
    塊を保つ，(5) 射影 proj_sigma が冪等で Σ の計量で直交する，を確かめる。"""
    rng = np.random.default_rng(0)
    n, d = 4000, 40
    y = rng.integers(0, 4, n)
    mus = {m: rng.normal(size=(4, d)) * 0.6 for m in "ab"}
    covs = {}
    X = {}
    for m in "ab":
        L = rng.normal(size=(d, d)) / np.sqrt(d)
        covs[m] = L @ L.T + 0.5 * np.eye(d)
        X[m] = mus[m][y] + rng.multivariate_normal(np.zeros(d), covs[m], n)
    # (1)(2)
    Za_, Zb_ = (tt(X[m] - X[m].mean(0)) for m in "ab")
    Ea, la, Za, _ = pca(Za_, d)
    Eb, lb, Zb, _ = pca(Zb_, d)
    r, rows, s, Da, Db = seq_test(Za, Zb, 200, None, 1)
    assert r == 3, (r, rows)
    wa = (Ea * la.rsqrt()) @ Da[:, :3]
    Sw = tt(covs["a"])
    lda = torch.linalg.solve(Sw, tt((mus["a"] - mus["a"].mean(0)).T))     # Fisher の部分空間（母集団）
    Sig = Za_.T @ Za_ / (n - 1)
    c2 = principal_cos2(sigma_orth(wa, Sig), sigma_orth(lda, Sig), Sig)
    assert c2 > 0.95, c2
    # (3)
    spk = rng.integers(0, 40, n)
    common_spk = rng.normal(size=(40, 1)) * 2.0
    Xs = {m: X[m] + common_spk[spk] * rng.normal(size=(1, d)) for m in "ab"}     # 2 つの見方に共通の話者の要因
    Za2 = pca(tt(Xs["a"] - Xs["a"].mean(0)), d)[2]; Zb2 = pca(tt(Xs["b"] - Xs["b"].mean(0)), d)[2]
    r2 = seq_test(Za2, Zb2, 200, None, 2)[0]
    assert r2 > 3, r2
    cen = {}
    for m in "ab":
        Xm = Xs[m].copy()
        for g in np.unique(spk):
            Xm[spk == g] -= Xm[spk == g].mean(0)
        cen[m] = pca(tt(Xm - Xm.mean(0)), d)[2]
    r3 = seq_test(cen["a"], cen["b"], 200, spk, 3)[0]
    assert r3 == 3, r3
    # (4)
    P = make_perms(n, 5, spk, 0)
    assert (spk[P] == spk[None]).all()
    # (5)
    B = sigma_orth(wa, Sig)
    Wr = tt(rng.normal(size=(4, d)))
    P1 = proj_sigma(Wr, B, Sig)
    assert torch.allclose(proj_sigma(P1, B, Sig), P1, atol=1e-8)
    assert float(((Wr - P1) @ Sig @ B).abs().max()) < 1e-8
    print(f"selftest OK：r = {r}（無相関），主角の余弦の 2 乗の平均 {c2:.4f}，話者を足すと r = {r2}，塊の中で中心化すると r = {r3}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--corpora", nargs="+", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(DD.N_FOLDS)))
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="試走用：単位がそろわなくても集計する")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    a.out.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        summarize(a.out, a.allow_partial)
        return
    commit, dirty = dai1.git_commit()
    fz3 = json.loads((ROOT / "DAI" / "frozen" / "dai3_frozen.json").read_text(encoding="utf-8"))
    prov = {"commit": commit, "dirty": dirty, "n_perm": a.n_perm, "gamma": GAMMA, "k_max": K_MAX, "n_perm_eta": N_PERM_ETA,
            "n_rand": N_RAND, "cv_folds": CV_FOLDS, "dai3_frozen_sha256": dai1.sha(ROOT / "DAI" / "frozen" / "dai3_frozen.json"),
            "subsets_sha256": dai1.sha(ROOT / "DAI" / "frozen" / "subsets.csv"),
            "source_sha256": {p.name: dai1.sha(p) for p in sorted((SRC_OUT / "source").glob("*.npz"))},
            "splits_sha256": {p.name: dai1.sha(p) for p in sorted((ROOT / "DAI" / "splits").glob("*.csv"))},
            "features_sha256": {f"{k}_{c}": dai1.sha(common.out_path(k, D.GROUP_SPEC[DD.GROUP[c]][0])) for c in DD.CORPORA for k in DD.ENC1.values()},
            "src_out": str(SRC_OUT), "units": []}
    pp = a.out / "provenance.json"
    if pp.exists():
        old = json.loads(pp.read_text(encoding="utf-8"))
        prov["units"] = old.get("units", [])
    for c in a.corpora:
        for f in a.folds:
            U = ProbeUnit(c, f)
            res = run_unit(U, a.n_perm, fz3, a.out, commit, dirty)
            prov["units"].append({**res, "commit": commit, "dirty": dirty})
            pp.write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{c} 分割 {f}：候補 {U.n}，p = {U.p}，{res['sec']:.0f} 秒", flush=True)
            if S.DEV == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
