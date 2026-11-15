"""E85：配備する分類器は，疑似ラベルを「自分の特徴から予測できる分」だけ学ぶか．表せる分の上限と，手法が受け取った得を並べる（仮のターゲットだけ）．

設計：docs/exploratory-analyses.md の E85（規則は値を見る前に書いた．関門 1 の直しは checks/E85/fix_applied.md）．
下調べ（仮のターゲットの候補とテスト．評価のターゲットのラベルは使わない）．

入力（保存済み）
  - E80 の単位ファイル <e80>/<world>/units/<fused>/<T>/<c>_r<r>.npz：候補の行番号 cand，テストの行番号 te，ラベル y・yte，
    候補の上の各モデルの確率（P__<kind>：ソースだけの出発点，ref__K：学び直して掛けたモデル，poe0：出発点の確率の積，
    none__<f>__K：K 本で学び直した配備する分類器，none_te__<f>__K：同じくテストの上．log_softmax）
  - E77 の仮のターゲットの成績 <e77>_<fused>/<T>_pseudo.csv：行 (pseudo=c, rep, student=f, method, param, meff, K, uar)．
    method=none（出発点＋K 本），teacher（疑似ラベルのモデルそのもの．融合の student の下だけ），dist（手法．param="<g>:<重み>"，meff）
  - 特徴：measure/data.load_any(c, <fused>) の行を cand・te で引き，配備する分類器 f のブロックだけを使う（融合の f は全部）
新しい学習（proj）：候補の上で，疑似ラベル q を自分の特徴 z だけから当てる線形 softmax
    min (1/n) Σ_i −q_iᵀ log softmax(W z_i + b) + λ/2 ‖W‖²
  を exp/logreg.fit_batch（float64，問ごとに止める判定，勾配 1e−6）で解く．q は z を 4 回複製した 4n 行の重み（行 k·n+i に q_ik）として渡す
  （exp/distill_belief.py の weak_rows と同じ形）．fit_batch の目的関数 (1/S)Σ w·CE + ‖W‖²/(2CS) は S = Σw = n なので C = 1/(λ n) で一致する．
  λ ∈ {1e−3, 1e−2, 1e−1}．主の λ は，候補を話者で 2 分割し，片方で当てはめてもう片方で測った q への柔らかい交差エントロピー
  （入れ替えて平均）が最小のもの（問ごと．ラベルは使わない）．感度として固定の 1e−3 も集計する．

量（テストの上）：UAR_anchor（none_te），UAR_g（teacher の行），UAR_proj（当てはめた線形 softmax），UAR_method（dist の行）．
  J = UAR_g − UAR_anchor，表せる分の得 gain_proj = UAR_proj − UAR_anchor，手法の得 gain_method = UAR_method − UAR_anchor．
  割合 r_rep = gain_proj/J，r_m = gain_method/J は，単位（c のコーパス．T と反復）で得と J を平均してから割る（summarize）．
候補の 2 分割でも同じ量と，q への忠実さ（柔らかい交差エントロピー，argmax の一致率）を出す（teacher の無い g にも使える）．
対ごとの a（2 感情の正解率．切片を最良に直した値）も出す．proj のテストの確率と W・b は fits/ に保存する．

使い方：./.venv/bin/python exp/analyze_projection_e85.py [--out results/e85] [--worlds avt av] [--targets T ...] [--rep-list 0 1 2] [--smoke]
  本番（反復 0〜2，1 つの処理で avt → av の順．約 5.3 時間．同じ GPU で処理を分けて並行しても速くならない（再検査の実測）．設計 3 節）：
    tmux の mer-da で  ./.venv/bin/python exp/analyze_projection_e85.py --out results/e85
  分けて流したときは --summarize-from <置き場>... --out <まとめ先> で units・pairs・checks・fit_check を合わせて集計だけを書く
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import logreg  # noqa: E402
from data import GROUP_SPEC, fused_parts, load_any, load_group  # noqa: E402
from explore_pairs import PAIR_NAMES, PAIRS, ba  # noqa: E402
from split import speaker_folds  # noqa: E402

WORLDS = {"avt": "fuse-audio-video-text", "av": "fuse-audio-video"}
KS = (0, 5, 25)
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)   # λ の候補（再検査で 1・10 を足した：AFEW の主のセルで保留側の交差エントロピーが 1 で最小）．主は候補の 2 分割で選ぶ（lam_sel）．ここ 1 か所で決める
LAM_FIXED = 1e-3                 # 感度：固定の λ
REPS_MAIN = [0, 1, 2]            # 本番の反復．主の答えはこの 3 反復で読む（反復 3・4 は範囲の外．設計 3 節）
DELTA_FLOOR = 1.0                # ポイント．Q1 の 3 区分の δ の下限（δ ＝ max(これ, 単位の中の行ごとの gain_proj の標準偏差)）
DP_TOL = 1e-2                    # fit_check：束ねた解と 1 問ずつの解のテストの確率の最大差の基準（n < d で λ が小さいと損失が平らな向きがあり，勾配 1e−6 の止め方では未知の行で 1e−3 ほど違う）
ARGMAX_FRAC = 0.002              # fit_check：argmax の不一致の本数の基準 max(1, この割合 × テストの本数)．どちらかを超えれば止まる
# 疑似ラベルのモデル g．self は f 自身の出発点（none__<f>__K．完全に表せる疑似ラベルの参照），oracle は候補の正解ラベルの one-hot（f の特徴で届く上限の参照）
G_OF_K = {0: ("pF", "ref", "poe0", "self", "oracle"), 5: ("pF", "pFK", "ref", "poe0", "self", "oracle"), 25: ("pF", "pFK", "ref", "poe0", "self", "oracle")}
G_TEACHER = {"pF": "pF", "pFK": "pFK", "ref": "ref", "poe0": "poe0"}       # E77 の teacher の行の param（self は anchor 自身，oracle は無い）
G_DIST_SINGLE = ("pF", "pFK", "ref")     # 単モダリティの f に dist（手法）の行がある g（poe0 は無い → method の列は NaN）
G_DIST_FUSED = ("ref", "poe0")           # 融合の f に dist の行がある g（pF・pFK は無い）
WEIGHTS = ("raw", "bal", "hard0.5", "hard0.7", "hard0.9")
MEFFS = ("n", "500", "100", "2000")
MAIN_W, MAIN_M = "raw", "n"
J_FLOOR = 1.0            # ポイント．単位の平均の J がこれ未満なら割合を読まない
GAIN_FLOOR = 1.0         # ポイント．単位の平均の gain_proj がこれ未満なら Q1 を読まない
Q1_FRAC = 0.8
Q2_THR = 0.6
PAIR_FLOOR = 2.0         # ポイント．対ごとの a_g − a_anchor がこれ未満なら方向の割合を読まない
GRAD_OK = logreg.TOL     # 収束（勾配の最大絶対値）
GRAD_NEAR = 1e-4         # これを超えた問は判定から外し，台本は出力を書いた後に異常終了する
TOL_MATCH = 1e-3         # UAR の照合（ポイント）
DEV = logreg.DEVICE


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def corp(g):
    return GROUP_SPEC[g][0]


def uar(pred, y):
    return 100 * float(np.mean([(pred[y == c] == c).mean() for c in range(4) if (y == c).any()]))


def q_name(g, K, f, fused):
    """当てはめを共有する名前（K に依らない g は K を落とす）．self の K = 0 は P__<f> と同じ配列（Unit で確かめる）．"""
    if g == "self":
        return f"P__{f}" if K == 0 else f"none__{f}__{K}"
    return {"pF": f"P__{fused}", "pFK": f"none__{fused}__{K}", "ref": f"ref__{K}", "poe0": "poe0", "oracle": "oracle"}[g]


def soft_ce(P, Q):
    """柔らかい交差エントロピー −Σ q log p の平均（nat）．"""
    return float(-(Q * np.log(np.clip(P, 1e-12, 1))).sum(1).mean())


def fit_many(Z, Qs, lams, rows=None):
    """同じ特徴 Z（標準化済み，n×d）の上で，疑似ラベル Qs[j]（n×4，行の和 1）× λ の全問を logreg.fit_batch で束ねて解く．
    rows：学習に使う行（真偽，長さ n．None なら全行）．出発点は 0（温め直しは半分の当てはめで 17% しか縮めなかったので使わない）．
    戻り値 out[j][lam] = dict(W (4×d), b (4,), n_iter, grad, loss)．loss は設計の目的関数（(1/n_fit) Σ −qᵀlog softmax + λ/2‖W‖²）．"""
    n, d = Z.shape
    rows = np.ones(n, bool) if rows is None else np.asarray(rows, bool)
    n_fit = int(rows.sum())
    Zt = logreg.to_device(Z)
    X = Zt.repeat(4, 1)                                                     # 4n × d（行 k·n+i は発話 i，感情 k）
    y = torch.arange(4, device=DEV).repeat_interleave(n)
    mask = np.tile(rows, 4)
    weights = []
    for Q in Qs:
        w = torch.as_tensor(np.ascontiguousarray(Q.T.reshape(-1)), dtype=logreg.DTYPE, device=DEV)
        S = float(w[torch.as_tensor(mask, device=DEV)].sum())
        if abs(S - n_fit) > 1e-6 * n_fit:
            raise SystemExit(f"疑似ラベルの行の和が 1 でない（S={S}, n_fit={n_fit}）")
        weights.append(w)
    Cs = tuple(1.0 / (lam * n_fit) for lam in lams)                        # λ = 1/(C·S)，S = n_fit
    zero, one = torch.zeros(d, dtype=logreg.DTYPE, device=DEV), torch.ones(d, dtype=logreg.DTYPE, device=DEV)
    fits = logreg.fit_batch(X, y, [mask] * len(Qs), Cs, stats=(zero, one), weights=weights)
    out = []
    Zf = Zt[torch.as_tensor(rows, device=DEV)]
    for Q, ft in zip(Qs, fits):
        Qf = torch.as_tensor(Q, dtype=logreg.DTYPE, device=DEV)[torch.as_tensor(rows, device=DEV)]
        o = {}
        for lam, C in zip(lams, Cs):
            W, b = ft.W[C], ft.b[C]
            with torch.no_grad():
                loss = float(-(Qf * torch.log_softmax(Zf @ W.T + b, 1)).sum(1).mean() + lam / 2 * (W * W).sum())
            o[lam] = dict(W=W.cpu().numpy(), b=b.cpu().numpy(), n_iter=ft.n_iter[C], grad=ft.grad[C], loss=loss)
        out.append(o)
    del X, Zt, Zf
    return out


def fit_single_f64(Z, Q, lam, max_iter=3000, tol=1e-9):
    """検算用：1 問を torch の L-BFGS（float64）で解く．戻り値 (W (4×d), b, n_iter, grad, loss)．"""
    Zt = torch.tensor(Z, dtype=torch.float64, device=DEV)
    Qt = torch.tensor(Q, dtype=torch.float64, device=DEV)
    W = torch.zeros(Z.shape[1], 4, dtype=torch.float64, device=DEV, requires_grad=True)
    b = torch.zeros(4, dtype=torch.float64, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, tolerance_grad=tol, tolerance_change=1e-14, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -(Qt * torch.log_softmax(Zt @ W + b, 1)).sum(1).mean() + lam / 2 * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    loss = float(closure().detach())
    g = max(float(W.grad.abs().max()), float(b.grad.abs().max()))
    return W.detach().T.cpu().numpy(), b.detach().cpu().numpy(), int(opt.state[opt._params[0]]["n_iter"]), g, loss


def predict(W, b, Z):
    A = Z @ W.T + b
    A = A - A.max(1, keepdims=True)
    P = np.exp(A)
    return P / P.sum(1, keepdims=True)


def pair_a(P, y):
    """確率 P（n×4）とラベルから，対ごとの a = 100·Φ(d/2)（2 つの山の模型．切片を最良に直した 2 感情の正解率）．
    単位ファイルの none_te__* は log_softmax（負の値）で保存されているので，負なら対数のまま使う．"""
    P = np.asarray(P, dtype=np.float64)
    L = P if P.min() < 0 else np.log(np.clip(P, 1e-12, 1))
    out = []
    for p, (k, j) in enumerate(PAIRS):
        zk, zj = L[y == k, k] - L[y == k, j], L[y == j, k] - L[y == j, j]
        if len(zk) < 2 or len(zj) < 2:
            out.append(np.nan)
            continue
        sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
        out.append(100 * ba((zk.mean() - zj.mean()) / sd, 0) if sd > 0 else np.nan)
    return out


class Unit:
    def __init__(self, world, T, c, r, feats, e80, e77):
        fused = WORLDS[world]
        self.world, self.T, self.c, self.r, self.fused = world, T, c, r, fused
        self.z = np.load(e80 / world / "units" / fused / T / f"{c}_r{r}.npz")
        self.cand, self.te = self.z["cand"], self.z["te"]
        self.y, self.yte = self.z["y"], self.z["yte"]
        df, X, blocks = feats
        if len(self.cand) != len(self.y) or int(max(self.cand.max(), self.te.max())) >= len(df):
            raise SystemExit(f"{world} {T} {c} r{r}：候補・テストの行番号が発話一覧と合わない")
        if not np.array_equal(df.label_4.to_numpy()[self.cand], self.y) or not np.array_equal(df.label_4.to_numpy()[self.te], self.yte):
            raise SystemExit(f"{world} {T} {c} r{r}：ラベルが発話一覧と合わない")
        if len(set(self.cand) & set(self.te)):
            raise SystemExit(f"{world} {T} {c} r{r}：候補とテストの行が重なる")
        self.X, self.blocks, self.df = X, blocks, df
        p = pd.read_csv(Path(f"{e77}_{fused}") / f"{T}_pseudo.csv", comment="#")
        self.pseudo = p[(p.pseudo == c) & (p.rep == r)].copy()
        for col in ("K", "meff", "param"):
            self.pseudo[col] = self.pseudo[col].astype(str)
        for f in list(fused_parts(fused)[1]) + [fused]:                    # self の K = 0 は出発点そのもの
            if np.abs(self.z[f"none__{f}__0"] - self.z[f"P__{f}"]).max() > 1e-6:
                raise SystemExit(f"{world} {T} {c} r{r}：none__{f}__0 が P__{f} と違う")

    def feat(self, f, rows):
        lo, hi = self.blocks[f]
        return self.X[rows, lo:hi]

    def ps(self, f, K, method, param, meff=None, required=True):
        """E77 の成績の 1 行（ポイント）．required なら無いときに止まる．重複は常に止まる．"""
        q = self.pseudo[(self.pseudo.student == f) & (self.pseudo.K == str(K)) & (self.pseudo.method == method) & (self.pseudo.param == param)]
        if meff is not None:
            q = q[q.meff == str(meff)]
        if len(q) > 1:
            raise SystemExit(f"{self.world} {self.T} {self.c} r{self.r}：pseudo.csv の行が重複（{f} K={K} {method} {param} {meff}：{len(q)} 行）")
        if len(q) == 0:
            if required:
                raise SystemExit(f"{self.world} {self.T} {self.c} r{self.r}：pseudo.csv に行が無い（{f} K={K} {method} {param} {meff}）")
            return np.nan
        return 100 * float(q.uar.iloc[0])       # pseudo.csv の uar は割合（0〜1）．ポイントにそろえる

    def q_array(self, name, f):
        if name == "oracle":
            return np.eye(4)[self.y]
        Q = self.z[name].astype(np.float64)
        if Q.min() < 0:                          # log_softmax で保存された配列（none_te__*．候補の上の配列は確率）
            Q = np.exp(Q)
        return Q / Q.sum(1, keepdims=True)


def feats_for(c, fused):
    df, X = load_any(c, fused)
    mods, kinds = fused_parts(fused)
    dims = [load_group(c, kd)[1].shape[1] for kd in kinds]
    blocks, lo = {}, 0
    for kd, dm in zip(kinds, dims):              # 連結の順は融合の名前のモダリティの順（audio, video, text）
        blocks[kd] = (lo, lo + dm)
        lo += dm
    blocks[fused] = (0, lo)
    if lo != X.shape[1]:
        raise SystemExit(f"{c} {fused}：ブロックの次元の和 {lo} が特徴 {X.shape[1]} と合わない")
    return df, X, blocks


def run_unit(u, singles, e80, out_fits, rows_u, rows_p, rows_chk, do_fit_check):
    fused = u.fused
    fs = tuple(singles) + (fused,)
    sp = pd.read_csv(e80 / u.world / "units" / fused / f"samples_{u.c}.csv")
    if not np.array_equal(sp.sample_id.to_numpy(), u.df.sample_id.to_numpy()):
        raise SystemExit(f"{u.world} {u.c}：samples の sample_id が発話一覧と合わない")
    grp = sp.speaker_group.iloc[u.cand].reset_index(drop=True)
    gte = set(sp.speaker_group.iloc[u.te].dropna())
    if gte & set(grp.dropna()):
        raise SystemExit(f"{u.world} {u.T} {u.c} r{u.r}：候補とテストで話者が重なる")
    fold = speaker_folds(grp, k=2, seed=0).to_numpy()
    n_fold_neg = int((fold < 0).sum())
    sid_te = sp.sample_id.to_numpy()[u.te]
    # 照合（単位ファイルごと．合わなければその場で止まる）：none_te の UAR と E77 の none の行，none_te の融合と teacher の pF・pFK の行
    anc_uar = {}
    for f in fs:
        for K in KS:
            anc_uar[f, K] = uar(u.z[f"none_te__{f}__{K}"].argmax(1), u.yte)
            d = u.ps(f, K, "none", "-") - anc_uar[f, K]
            rows_chk.append(dict(world=u.world, T=u.T, c=u.c, r=u.r, f=f, K=K, what="none_te vs pseudo.csv none", diff=d))
            if abs(d) > TOL_MATCH:
                raise SystemExit(f"{u.world} {u.T} {u.c} r{u.r} {f} K={K}：none_te の UAR が pseudo.csv の none の行と合わない（差 {d:.4f}）")
    for K in KS:
        for g, Kk in (("pF", 0), ("pFK", K)):
            if g == "pFK" and K == 0:
                continue
            d = u.ps(fused, K, "teacher", g) - anc_uar[fused, Kk]
            rows_chk.append(dict(world=u.world, T=u.T, c=u.c, r=u.r, f=fused, K=K, what=f"teacher {g} vs none_te__{fused}__{Kk}", diff=d))
            if abs(d) > TOL_MATCH:
                raise SystemExit(f"{u.world} {u.T} {u.c} r{u.r} K={K}：teacher の {g} の行が none_te__{fused}__{Kk} の UAR と合わない（差 {d:.4f}）")
    fit_check = []
    for f in fs:
        Zc, Zte = u.feat(f, u.cand), u.feat(f, u.te)
        mu, sd = Zc.mean(0), Zc.std(0) + 1e-8
        Zc, Zte = (Zc - mu) / sd, (Zte - mu) / sd
        n, d = Zc.shape
        names, name_Ks = [], {}
        for K in KS:
            for g in G_OF_K[K]:
                nm = q_name(g, K, f, fused)
                if nm not in names:
                    names.append(nm)
                name_Ks.setdefault(nm, [])
                if K not in name_Ks[nm]:
                    name_Ks[nm].append(K)
        Qs = [u.q_array(nm, f) for nm in names]
        labmask = {}                                                        # K 本のラベルの行（候補の中の位置）．K ≥ 5 の保留側の測定から除く
        for K in KS:
            m = np.zeros(n, bool)
            lab = u.z[f"lab__{K}"].astype(int)
            if len(lab) and (lab.min() < 0 or lab.max() >= n):
                raise SystemExit(f"{u.world} {u.T} {u.c} r{u.r}：lab__{K} が候補の範囲の外")
            m[lab] = True
            labmask[K] = m
        t0 = time.time()
        full = fit_many(Zc, Qs, LAMS)                                       # テスト用：全候補で当てはめ
        # 候補の 2 分割：fold a で当てはめ，fold b で q への忠実さと UAR を測る（入れ替えて平均）
        half = {(nm, K): {lam: [] for lam in LAMS} for nm in names for K in name_Ks[nm]}
        Pc = {nm: {lam: np.full((n, 4), np.nan) for lam in LAMS} for nm in names}
        for a_, b_ in ((0, 1), (1, 0)):
            ra, rb = fold == a_, fold == b_
            fa = fit_many(Zc, Qs, LAMS, rows=ra)
            for nm, Q, o in zip(names, Qs, fa):
                for lam in LAMS:
                    Pb_all = predict(o[lam]["W"], o[lam]["b"], Zc[rb])
                    Pc[nm][lam][rb] = Pb_all
                    for K in name_Ks[nm]:
                        rbe = rb & ~labmask[K]                              # 保留側から，その K の学習に使った行を除く（K = 0 では該当なし）
                        Pb, Qb = Pb_all[rbe[rb]], Q[rbe]
                        half[nm, K][lam].append(dict(uar_proj=uar(Pb.argmax(1), u.y[rbe]), uar_g=uar(Qb.argmax(1), u.y[rbe]), ce=soft_ce(Pb, Qb), H=soft_ce(Qb, Qb),
                                                     agree=float((Pb.argmax(1) == Qb.argmax(1)).mean()), n_iter=o[lam]["n_iter"], grad=o[lam]["grad"],
                                                     n_fit=int(ra.sum()), n_eval=int(rbe.sum()), n_excl=int((rb & labmask[K]).sum()), rbe=rbe))
        t_fit = time.time() - t0
        # λ の選択：(q, K) ごとに，保留側（入れ替えて平均）の q への柔らかい交差エントロピーが最小の λ（同点なら小さい方）
        lam_sel = {key: min(LAMS, key=lambda l: (np.mean([h["ce"] for h in half[key][l]]), l)) for key in half}
        # 保存：テストの確率（float16）・W・b・標準化・2 分割の保留側の確率
        save = dict(sample_id_te=sid_te.astype(str), yte=u.yte, mu=mu.astype(np.float32), sd=sd.astype(np.float32), names=np.array(names, dtype=str), lams=np.array(LAMS),
                    fold=fold, sample_id_cand=sp.sample_id.to_numpy()[u.cand].astype(str))
        for (nm, K), lam in lam_sel.items():
            save[f"lam_sel|{nm}|{K}"] = np.float64(lam)
        Pte = {}
        for nm, o in zip(names, full):
            for lam in LAMS:
                Pte[nm, lam] = predict(o[lam]["W"], o[lam]["b"], Zte)
                save[f"P_te|{nm}|{lam:g}"] = Pte[nm, lam].astype(np.float32)   # テストの確率は float32（float16 では argmax が同点になり UAR を作り直せない）
                save[f"P_cand_heldout|{nm}|{lam:g}"] = Pc[nm][lam].astype(np.float16)
                save[f"W|{nm}|{lam:g}"] = o[lam]["W"].astype(np.float32)
                save[f"b|{nm}|{lam:g}"] = o[lam]["b"]
        fdir = out_fits / u.world / u.T
        fdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fdir / f"{u.c}_r{u.r}__{f}.npz", **save)
        # 検算（世界ごとに最初の単位ファイルだけ）：束ねた解と 1 問ずつ float64（勾配 1e−8）で解いた解の一致（K = 0 の ref と oracle，固定の λ）
        if do_fit_check:
            for nm, Q, o in zip(names, Qs, full):
                if nm not in (q_name("ref", 0, f, fused), "oracle"):
                    continue
                W1, b1, it1, g1, l1 = fit_single_f64(Zc, Q, LAM_FIXED, max_iter=2000, tol=1e-8)
                P1 = predict(W1, b1, Zte)
                P0 = Pte[nm, LAM_FIXED]
                fit_check.append(dict(world=u.world, T=u.T, c=u.c, r=u.r, f=f, q=nm, lam=LAM_FIXED, n_cand=n, d=d,
                                      n_iter_batch=o[LAM_FIXED]["n_iter"], grad_batch=o[LAM_FIXED]["grad"], loss_batch=o[LAM_FIXED]["loss"],
                                      n_iter_single=it1, grad_single=g1, loss_single=l1,
                                      d_uar_te=uar(P0.argmax(1), u.yte) - uar(P1.argmax(1), u.yte), maxabs_dP_te=float(np.abs(P0 - P1).max()),
                                      n_argmax_disagree=int((P0.argmax(1) != P1.argmax(1)).sum()), maxabs_dW=float(np.abs(o[LAM_FIXED]["W"] - W1).max()),
                                      n_te=len(u.te), n_argmax_tol=max(1, int(ARGMAX_FRAC * len(u.te))),
                                      agree_ok=bool(np.abs(P0 - P1).max() <= DP_TOL and int((P0.argmax(1) != P1.argmax(1)).sum()) <= max(1, int(ARGMAX_FRAC * len(u.te))))))
            bad = [r for r in fit_check if r["f"] == f and not r["agree_ok"]]
            if bad:
                raise SystemExit(f"{u.world} {u.T} {u.c} r{u.r} {f}：束ねた解と 1 問ずつの解が合わない（確率の最大差 ≤ {DP_TOL:g} かつ argmax の不一致 ≤ max(1, {ARGMAX_FRAC:g}·n_te）の基準）：{[(r['q'], r['maxabs_dP_te'], r['n_argmax_disagree']) for r in bad]}")
        # 行（K × g × λ）
        for K in KS:
            anc_te = u.z[f"none_te__{f}__{K}"]
            u_anchor_te = anc_uar[f, K]
            anc_c = u.q_array(f"none__{f}__{K}", f)
            a_anchor_te = pair_a(anc_te, u.yte)
            for g in G_OF_K[K]:
                nm = q_name(g, K, f, fused)
                j = names.index(nm)
                o = full[j]
                Q = Qs[j]
                if g in G_TEACHER:
                    u_g_te = u.ps(fused, K, "teacher", G_TEACHER[g])       # teacher の行は融合の student の下だけ（g の UAR は f に依らない）
                elif g == "self":
                    u_g_te = u_anchor_te
                else:
                    u_g_te = np.nan                                         # oracle：テストの上の teacher は無い
                has_dist = g in (G_DIST_FUSED if f == fused else G_DIST_SINGLE)
                ls = lam_sel[nm, K]
                for lam in LAMS:
                    P_te = Pte[nm, lam]
                    u_proj_te = uar(P_te.argmax(1), u.yte)
                    hs = half[nm, K][lam]
                    anc_cand_b = [uar(anc_c[h["rbe"]].argmax(1), u.y[h["rbe"]]) for h in hs]
                    row = dict(world=u.world, T=u.T, c=u.c, unit=corp(u.c), r=u.r, f=f, K=K, g=g, q=nm, lam=lam, lam_sel=ls, is_main=(lam == ls),
                               lam_edge=("lo" if ls == min(LAMS) else ("hi" if ls == max(LAMS) else "in")),
                               n_cand=n, n_te=len(u.te), d=d, n_over_d=n / d,
                               uar_anchor_te=u_anchor_te, uar_g_te=u_g_te, uar_proj_te=u_proj_te,
                               J_te=u_g_te - u_anchor_te, gain_proj_te=u_proj_te - u_anchor_te,
                               n_iter=o[lam]["n_iter"], grad_max=o[lam]["grad"], loss=o[lam]["loss"],
                               stop_reason=("grad" if o[lam]["grad"] <= GRAD_OK else ("max_iter" if o[lam]["n_iter"] >= logreg.MAX_ITER else "stalled")),
                               fit_ok=bool(o[lam]["grad"] <= GRAD_NEAR),
                               uar_anchor_cand=float(np.mean(anc_cand_b)), uar_g_cand=float(np.mean([h["uar_g"] for h in hs])), uar_proj_cand=float(np.mean([h["uar_proj"] for h in hs])),
                               ce_q_cand=float(np.mean([h["ce"] for h in hs])), H_q_cand=float(np.mean([h["H"] for h in hs])), agree_q_cand=float(np.mean([h["agree"] for h in hs])),
                               n_half_min=min(h["n_fit"] for h in hs), n_half_over_d=min(h["n_fit"] for h in hs) / d, n_fold_neg=n_fold_neg,
                               n_heldout_eval_min=min(h["n_eval"] for h in hs), n_heldout_excl=sum(h["n_excl"] for h in hs),
                               n_iter_cand_max=max(h["n_iter"] for h in hs), grad_max_cand=max(h["grad"] for h in hs), fit_ok_cand=bool(max(h["grad"] for h in hs) <= GRAD_NEAR),
                               uar_g_cand_all=uar(Q.argmax(1), u.y))
                    row["J_cand"] = row["uar_g_cand"] - row["uar_anchor_cand"]
                    row["gain_proj_cand"] = row["uar_proj_cand"] - row["uar_anchor_cand"]
                    for w in WEIGHTS:
                        for m in MEFFS:
                            um = u.ps(f, K, "dist", f"{g}:{w}", m, required=True) if has_dist else np.nan
                            row[f"uar_method_{w}_{m}"] = um
                            row[f"gain_method_{w}_{m}"] = um - u_anchor_te
                    rows_u.append(row)
                    # 対ごと（テスト）：g の確率はテストに無いので pF・pFK（融合の none_te）と self（anchor 自身）だけ．proj と anchor は全 g
                    a_proj = pair_a(P_te, u.yte)
                    if g == "pF":
                        a_g = pair_a(u.z[f"none_te__{fused}__0"], u.yte)   # pF は K に依らない
                    elif g == "pFK":
                        a_g = pair_a(u.z[f"none_te__{fused}__{K}"], u.yte)
                    elif g == "self":
                        a_g = a_anchor_te
                    else:
                        a_g = [np.nan] * len(PAIRS)
                    for pi, name in enumerate(PAIR_NAMES):
                        rows_p.append(dict(world=u.world, T=u.T, c=u.c, unit=corp(u.c), r=u.r, f=f, K=K, g=g, lam=lam, is_main=(lam == ls), pair=name,
                                           a_anchor=a_anchor_te[pi], a_g=a_g[pi], a_proj=a_proj[pi]))
        print(f"  {f}: {len(names)} q × {len(LAMS)} λ × 3 当てはめ {t_fit:.0f}s", flush=True)
    return fit_check


# ---------------------------------------------------------------- 集計（規則は E85 4 節）

def per_unit_table(U, mode):
    """単位（c のコーパス）ごとに，得と J を T と反復で平均してから割る．mode は sel（行ごとに選んだ λ）か fixed（λ = LAM_FIXED）．"""
    sel = U.is_main if mode == "sel" else np.isclose(U.lam, LAM_FIXED)
    rows = U[sel].copy()
    rows["J_le0"] = rows.J_te <= 0
    rows["absJ_lt1"] = rows.J_te.abs() < J_FLOOR
    rows["lam_lo"] = rows.lam_edge == "lo"
    rows["lam_hi"] = rows.lam_edge == "hi"
    rows["q1_diff"] = rows[f"gain_method_{MAIN_W}_{MAIN_M}"] - Q1_FRAC * rows.gain_proj_te   # 行ごとの判定の差（符号がそろうか）
    rows = rows[rows.fit_ok & rows.fit_ok_cand]                              # 収束しなかった問（勾配 > GRAD_NEAR．全候補か 2 分割）は判定から外す
    gcols = [c for c in U.columns if c.startswith("gain_method_")]
    agg = dict(n_rows=("r", "size"), n_rows_J_le0=("J_le0", "sum"), n_rows_absJ_lt1=("absJ_lt1", "sum"), n_rows_lam_lo=("lam_lo", "sum"), n_rows_lam_hi=("lam_hi", "sum"),
               J_te=("J_te", "mean"), J_te_min=("J_te", "min"), J_te_max=("J_te", "max"),
               gain_proj_te=("gain_proj_te", "mean"), gain_proj_te_min=("gain_proj_te", "min"), gain_proj_te_max=("gain_proj_te", "max"), gain_proj_te_sd=("gain_proj_te", "std"),
               gain_m_min=(f"gain_method_{MAIN_W}_{MAIN_M}", "min"), gain_m_max=(f"gain_method_{MAIN_W}_{MAIN_M}", "max"),
               q1_diff_min=("q1_diff", "min"), q1_diff_max=("q1_diff", "max"),
               J_cand=("J_cand", "mean"), gain_proj_cand=("gain_proj_cand", "mean"), ce_q_cand=("ce_q_cand", "mean"), H_q_cand=("H_q_cand", "mean"),
               agree_q_cand=("agree_q_cand", "mean"), n_over_d=("n_over_d", "mean"), n_half_over_d=("n_half_over_d", "mean"),
               n_iter_max=("n_iter", "max"), grad_max=("grad_max", "max"), lam_sel_mean=("lam", "mean"))
    agg.update({c: (c, "mean") for c in gcols})
    per = rows.groupby(["world", "f", "K", "g", "unit"]).agg(**agg).reset_index()
    per["mode"] = mode
    n_unfit = U[sel & ~(U.fit_ok & U.fit_ok_cand)].groupby(["world", "f", "K", "g", "unit"]).size().rename("n_rows_not_converged")
    per = per.merge(n_unfit.reset_index(), on=["world", "f", "K", "g", "unit"], how="left")
    per["n_rows_not_converged"] = per.n_rows_not_converged.fillna(0).astype(int)
    gm = per[f"gain_method_{MAIN_W}_{MAIN_M}"]
    okJ = per.J_te >= J_FLOOR
    per["r_rep_te"] = np.where(okJ, per.gain_proj_te / per.J_te, np.nan)
    per["r_m_te"] = np.where(okJ, gm / per.J_te, np.nan)
    per["gain_m_best"] = per[gcols].max(axis=1)                               # 変種ごとに平均してから最大（仮のターゲットのテストで選んだ上限）
    has = per[gcols].notna().any(axis=1)                                      # 手法の行が無い g（単モダリティの poe0，self，oracle など）は NaN
    per["best_variant"] = pd.Series(np.nan, index=per.index, dtype=object)
    if has.any():
        per.loc[has, "best_variant"] = per.loc[has, gcols].idxmax(axis=1)
    per["r_m_best_te"] = np.where(okJ, per.gain_m_best / per.J_te, np.nan)
    per["r_rep_cand"] = np.where(per.J_cand >= J_FLOOR, per.gain_proj_cand / per.J_cand, np.nan)
    # Q1 の 3 区分（E85 4 節）：δ ＝ max(DELTA_FLOOR, 単位の中の行ごとの gain_proj の標準偏差)
    per["delta"] = np.maximum(DELTA_FLOOR, per.gain_proj_te_sd.fillna(0.0))
    per["q1_readable"] = (per.gain_proj_te >= GAIN_FLOOR) & gm.notna()
    per["ratio_m_over_proj"] = np.where(per.q1_readable, gm / per.gain_proj_te, np.nan)
    cls = np.where(gm < Q1_FRAC * per.gain_proj_te, "loss", np.where(gm <= per.gain_proj_te + per.delta, "received", "above"))
    per["q1_class"] = np.where(per.q1_readable, cls, "unreadable")
    per["q1_true"] = per.q1_class == "received"                               # 「受け取った」（0.8 gain_proj 以上，gain_proj + δ 以下）
    per["q1_sign_consistent"] = np.where(per.q1_readable, (per.q1_diff_min >= 0) | (per.q1_diff_max < 0), np.nan)   # 行ごとの gain_method − 0.8 gain_proj の符号がそろう
    per["proj_ge_method"] = np.where(gm.notna(), per.gain_proj_te >= gm, np.nan)
    return per


def summary_table(per):
    out = []
    for key, g in per.groupby(["world", "f", "K", "g", "mode"]):
        pos = g[g.J_te >= J_FLOOR]
        q1 = g[g.q1_readable]
        n_q1 = len(q1)
        n_cls = {c: int((q1.q1_class == c).sum()) for c in ("loss", "received", "above")}
        n_known = int(g.proj_ge_method.notna().sum())
        n_ge = int(g.proj_ge_method.fillna(False).astype(bool).sum())
        n_cons = int(q1.q1_sign_consistent.fillna(False).astype(bool).sum())
        rec = dict(world=key[0], f=key[1], K=key[2], g=key[3], mode=key[4], n_units=len(g), n_units_J_ge1=len(pos), n_units_J_le0=int((g.J_te <= 0).sum()),
                   n_units_q1_readable=n_q1, n_units_q1_loss=n_cls["loss"], n_units_q1_received=n_cls["received"], n_units_q1_above=n_cls["above"],
                   n_units_q1_sign_consistent=n_cons,
                   J_te_median=g.J_te.median(), gain_proj_te_median=g.gain_proj_te.median(), gain_m_median=g[f"gain_method_{MAIN_W}_{MAIN_M}"].median(),
                   delta_median=g.delta.median(),
                   r_rep_te_median=pos.r_rep_te.median() if len(pos) else np.nan, r_rep_te_min=pos.r_rep_te.min() if len(pos) else np.nan, r_rep_te_max=pos.r_rep_te.max() if len(pos) else np.nan,
                   r_m_te_median=pos.r_m_te.median() if len(pos) else np.nan, r_m_best_te_median=pos.r_m_best_te.median() if len(pos) else np.nan,
                   ratio_m_over_proj_median=q1.ratio_m_over_proj.median() if n_q1 else np.nan,
                   n_units_proj_ge_method=n_ge, n_units_method_known=n_known,
                   J_cand_median=g.J_cand.median(), r_rep_cand_median=g[g.J_cand >= J_FLOOR].r_rep_cand.median() if (g.J_cand >= J_FLOOR).any() else np.nan,
                   ce_q_cand_median=g.ce_q_cand.median(), agree_q_cand_median=g.agree_q_cand.median(), n_over_d_min=g.n_over_d.min(), n_half_over_d_min=g.n_half_over_d.min(),
                   n_rows=int(g.n_rows.sum()), n_rows_lam_lo=int(g.n_rows_lam_lo.sum()), n_rows_lam_hi=int(g.n_rows_lam_hi.sum()),
                   n_rows_not_converged=int(g.n_rows_not_converged.sum()), grad_max=g.grad_max.max(),
                   is_main_cell=(key[2] == 0 and key[3] in ("pF", "ref")),
                   is_main_cell_p85_1=(key[2] == 0 and key[1].startswith("audio") and key[3] in ("pF", "ref")),
                   is_main_cell_p85_2=(key[2] == 0 and key[1].startswith("audio") and key[3] == "pF"))
        # 判定（値を見る前に書いた規則．E85 4 節）．読める単位が 2 未満なら判定不能（NaN）
        if n_q1 >= 2:
            maj = [c for c, v in n_cls.items() if v > n_q1 / 2]
            rec["Q1_class_majority"] = maj[0] if maj else "split"
        else:
            rec["Q1_class_majority"] = np.nan
        rec["Q1_no_loss"] = (rec["Q1_class_majority"] == "received") if n_q1 >= 2 else np.nan        # 「受け取った」が過半のときに限る
        rec["Q1_within_noise"] = ((n_q1 - n_cons) > n_q1 / 2) if n_q1 >= 2 else np.nan                 # 符号がそろわない単位が過半なら「ばらつきの中」
        rec["ceiling_ok"] = (n_ge > n_known / 2) if n_known >= 2 else np.nan                           # proj が手法の得の上限として働くか（過半の単位で gain_proj ≥ gain_method）
        rec["P85_1_true"] = rec["ceiling_ok"] if key[2] == 0 else np.nan                               # 予言としては K = 0 だけ
        if len(pos) >= 2 and rec["ceiling_ok"] is True:
            rec["Q2_unrepresentable"] = bool(rec["r_rep_te_median"] < Q2_THR)                          # 表せない分が主因（上限が働くセルに限る）
            rec["Q2_reason"] = "read"
        else:
            rec["Q2_unrepresentable"] = np.nan
            rec["Q2_reason"] = "J<1 units" if len(pos) < 2 else ("proj not ceiling" if rec["ceiling_ok"] is False else "ceiling undetermined")
        rec["P85_2_true"] = (rec["r_rep_te_median"] <= 0.8) if (key[1].startswith("audio") and key[3] in ("pF", "pFK") and len(pos) >= 2) else np.nan
        rec["cand_te_gap_ge02"] = (abs(rec["r_rep_cand_median"] - rec["r_rep_te_median"]) >= 0.2) if np.isfinite(rec["r_rep_cand_median"]) and np.isfinite(rec["r_rep_te_median"]) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def _verdict(v):
    """真偽の列（NaN を除いた後）を 1 つの当否にする：過半で yes，過半が偽で no，同数で split，2 未満で undetermined．"""
    v = v.dropna().astype(bool)
    if len(v) < 2:
        return len(v), int(v.sum()), "undetermined"
    return len(v), int(v.sum()), ("yes" if v.sum() > len(v) / 2 else ("no" if v.sum() < len(v) / 2 else "split"))


def answer_table(S):
    """主のセルの判定を数え上げて，世界ごと（と両方）の答えにする．Q1・Q2 は K = 0 の g ∈ {pF, ref}，P85-1 は K = 0 の音声の f × {pF, ref}，P85-2 は K = 0 の音声の f × pF．"""
    out = []
    for mode in ("sel", "fixed"):
        Sm = S[S["mode"] == mode]
        for world in list(S.world.unique()) + ["both"]:
            w = Sm if world == "both" else Sm[Sm.world == world]
            m = w[w.is_main_cell]
            rec = dict(world=world, mode=mode, n_main_cells=len(m))
            for q in ("Q1_no_loss", "Q2_unrepresentable"):
                rec[f"n_{q}_readable"], rec[f"n_{q}_true"], rec[f"answer_{q}"] = _verdict(m[q])
            rec["n_Q1_above_cells"] = int((m.Q1_class_majority == "above").sum())              # proj が上限として働かないセル
            rec["n_Q1_within_noise_cells"] = int(m.Q1_within_noise.fillna(False).astype(bool).sum())
            rec["n_main_rows_lam_hi"], rec["n_main_rows_lam_lo"], rec["n_main_rows"] = int(m.n_rows_lam_hi.sum()), int(m.n_rows_lam_lo.sum()), int(m.n_rows.sum())
            m1 = w[w.is_main_cell_p85_1]
            rec["n_P85_1_readable"], rec["n_P85_1_true"], rec["answer_P85_1"] = _verdict(m1.P85_1_true)
            m2 = w[w.is_main_cell_p85_2]
            rec["n_P85_2_readable"], rec["n_P85_2_true"], rec["answer_P85_2"] = _verdict(m2.P85_2_true)
            out.append(rec)
    A = pd.DataFrame(out)
    # λ の感度：選んだ λ と固定の λ で判定が変わったセルの数（主と副の全セル）
    k = ["world", "f", "K", "g"]
    j = S[S["mode"] == "sel"][k + ["Q1_no_loss", "Q2_unrepresentable"]].merge(S[S["mode"] == "fixed"][k + ["Q1_no_loss", "Q2_unrepresentable"]], on=k, suffixes=("_sel", "_fixed"))
    for q in ("Q1_no_loss", "Q2_unrepresentable"):
        ch = (j[f"{q}_sel"].astype(str) != j[f"{q}_fixed"].astype(str))
        A[f"n_cells_{q}_changed_by_lam"] = int(ch.sum())
        A[f"n_cells_{q}_compared"] = len(j)
    return A


def pairs_unit_table(Pp):
    """対ごとの a を単位で平均し，方向の割合 (a_proj − a_anchor)/(a_g − a_anchor) を a_g − a_anchor ≥ PAIR_FLOOR の対で出す（内訳．判定には使わない）．"""
    rows = Pp[Pp.is_main]
    per = rows.groupby(["world", "f", "K", "g", "unit", "pair"]).agg(n_rows=("r", "size"), a_anchor=("a_anchor", "mean"), a_g=("a_g", "mean"), a_proj=("a_proj", "mean")).reset_index()
    dg = per.a_g - per.a_anchor
    per["dir_ratio"] = np.where(dg >= PAIR_FLOOR, (per.a_proj - per.a_anchor) / dg, np.nan)
    return per


def summarize(U, Pp):
    per = pd.concat([per_unit_table(U, "sel"), per_unit_table(U, "fixed")], ignore_index=True)
    S = summary_table(per)
    A = answer_table(S)
    PU = pairs_unit_table(Pp)
    return per, S, A, PU


def write(df, path, header):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {header}\n")
        df.to_csv(fh, index=False)


def summarize_from(dirs, out, header, prov):
    """分けて流した出力（units.csv・pairs.csv）を合わせて集計だけを書く（例：世界ごと，反復ごとに並行して流した後）．"""
    Us, Ps, Cs_, Fs = [], [], [], []
    for d in dirs:
        d = (ROOT / d).resolve()
        Us.append(pd.read_csv(d / "units.csv", comment="#"))
        Ps.append(pd.read_csv(d / "pairs.csv", comment="#"))
        for lst, name in ((Cs_, "checks.csv"), (Fs, "fit_check.csv")):
            if (d / name).exists() and (d / name).stat().st_size > 1:
                lst.append(pd.read_csv(d / name).assign(source=str(d)))
        pj = d / "provenance.json"
        prov["sources"][str(d)] = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else None
    U, Pp = pd.concat(Us, ignore_index=True), pd.concat(Ps, ignore_index=True)
    if Cs_:
        pd.concat(Cs_, ignore_index=True).to_csv(out / "checks.csv", index=False)
    if Fs:
        pd.concat(Fs, ignore_index=True).to_csv(out / "fit_check.csv", index=False)
    key = ["world", "T", "c", "r", "f", "K", "g", "lam"]
    if U.duplicated(key).any():
        raise SystemExit("合わせた units の識別子が重複（同じ単位ファイルを 2 回流している）")
    per, S, A, PU = summarize(U, Pp)
    write(per, out / "per_unit.csv", header)
    write(S, out / "summary.csv", header)
    write(A, out / "answer.csv", header)
    write(PU, out / "pairs_unit.csv", header)
    prov["n_units_rows"] = len(U)
    prov["n_fits_grad_above_ok"] = int((U.grad_max > GRAD_OK).sum() + (U.grad_max_cand > GRAD_OK).sum())
    prov["n_fits_grad_above_near"] = int((~U.fit_ok).sum() + (~U.fit_ok_cand).sum())
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    print("summarized", out, len(U), "rows from", len(dirs), "dirs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e85")
    ap.add_argument("--e80", default="results/e80", help="E80 の単位ファイルの置き場")
    ap.add_argument("--e77", default="results/e77", help="E77 の成績の置き場の接頭辞（<e77>_<fused>/<T>_pseudo.csv）")
    ap.add_argument("--worlds", nargs="*", default=list(WORLDS))
    ap.add_argument("--rep-list", nargs="*", type=int, default=REPS_MAIN, help=f"流す反復（既定 {REPS_MAIN}．残りは別の --out に流し --summarize-from で合わせる）")
    ap.add_argument("--targets", nargs="*", default=None, help="流す評価のターゲット T（既定は世界の全部．世界と T で処理を分けて並行して流すためのもの）")
    ap.add_argument("--summarize-from", nargs="*", default=None, help="計算せず，これらの置き場の units.csv・pairs.csv を合わせて集計だけを --out に書く")
    ap.add_argument("--smoke", action="store_true", help="試走（avt の 1 ターゲット，反復 0．出力は --out か checks/E85/code_run/smoke）")
    args = ap.parse_args()
    if args.smoke:
        args.worlds, args.rep_list = ["avt"], [0]
        if args.out == "results/e85":
            args.out = "checks/E85/code_run/smoke"
    out = ROOT / args.out
    e80, e77 = ROOT / args.e80, str(ROOT / args.e77)
    if out.exists() and any(out.glob("*.csv")):
        raise SystemExit(f"{out} に古い出力がある．消してから流す")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), out / "script_copy.py")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "exp/analyze_projection_e85.py", "exp/logreg.py", "exp/explore_pairs.py", "measure/data.py", "measure/split.py", "data/metadata"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()          # E85 が読むファイルに限る（ほかの実験の未コミットの変更で真にしない）
    header = f"E85 commit={commit} dirty={bool(dirty)} lams={list(LAMS)} lam_fixed={LAM_FIXED} reps={args.rep_list} device={DEV} solver=logreg.fit_batch(float64,tol={logreg.TOL})"
    prov = {"commit": commit, "dirty": bool(dirty), "script_sha256": sha256(Path(__file__)), "args": vars(args) | {"out": str(out)}, "inputs": {}, "device": str(DEV),
            "lams": list(LAMS), "lam_fixed": LAM_FIXED, "weights": list(WEIGHTS), "meffs": list(MEFFS), "J_floor": J_FLOOR, "gain_floor": GAIN_FLOOR, "q1_frac": Q1_FRAC, "q2_thr": Q2_THR,
            "grad_ok": GRAD_OK, "grad_near": GRAD_NEAR, "time_per_unit_s": {}, "sources": {}}
    if args.summarize_from:
        summarize_from(args.summarize_from, out, header, prov)
        return 0
    rows_u, rows_p, rows_chk, fit_check = [], [], [], []
    t_start = time.time()
    for world in args.worlds:
        fused = WORLDS[world]
        singles = fused_parts(fused)[1]
        udir = e80 / world / "units" / fused
        Ts = sorted(p.name for p in udir.iterdir() if p.is_dir())
        if args.targets:
            missing = [T for T in args.targets if T not in Ts]
            if missing:
                raise SystemExit(f"{world}：--targets に無い T {missing}（ある T：{Ts}）")
            Ts = [T for T in Ts if T in args.targets]
        if args.smoke:
            Ts = Ts[:1]
        feats = {}
        first = True
        for T in Ts:
            files = sorted(udir.glob(f"{T}/*_r*.npz"))
            files = [f for f in files if not f.name.endswith("_agg.npz")]
            for fpath in files:
                c, r = fpath.stem.rsplit("_r", 1)
                r = int(r)
                if r not in args.rep_list:
                    continue
                if c not in feats:
                    feats[c] = feats_for(c, fused)
                prov["inputs"][str(fpath.relative_to(ROOT))] = sha256(fpath)
                t0 = time.time()
                u = Unit(world, T, c, r, feats[c], e80, e77)
                fit_check += run_unit(u, singles, e80, out / "fits", rows_u, rows_p, rows_chk, do_fit_check=first)
                first = False
                prov["time_per_unit_s"][f"{world}/{T}/{c}_r{r}"] = round(time.time() - t0, 1)
                print(world, T, c, r, f"done {time.time() - t0:.0f}s (total {(time.time() - t_start) / 60:.1f} min)", flush=True)
            prov["inputs"][f"e77_{fused}/{T}_pseudo.csv"] = sha256(Path(f"{e77}_{fused}") / f"{T}_pseudo.csv")
    U, Pp = pd.DataFrame(rows_u), pd.DataFrame(rows_p)
    key = ["world", "T", "c", "r", "f", "K", "g", "lam"]
    if U.duplicated(key).any():
        raise SystemExit("units の識別子が重複")
    write(U, out / "units.csv", header)
    write(Pp, out / "pairs.csv", header)
    pd.DataFrame(rows_chk).to_csv(out / "checks.csv", index=False)
    pd.DataFrame(fit_check).to_csv(out / "fit_check.csv", index=False)
    per, S, A, PU = summarize(U, Pp)
    write(per, out / "per_unit.csv", header)
    write(S, out / "summary.csv", header)
    write(A, out / "answer.csv", header)
    write(PU, out / "pairs_unit.csv", header)
    n_bad = int((~U.fit_ok).sum() + (~U.fit_ok_cand).sum())
    n_not_conv = int((U.grad_max > GRAD_OK).sum() + (U.grad_max_cand > GRAD_OK).sum())
    prov["n_fits_grad_above_ok"] = n_not_conv
    prov["n_fits_grad_above_near"] = n_bad
    prov["elapsed_s"] = round(time.time() - t_start, 1)
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out, len(U), "rows", f"({(time.time() - t_start) / 60:.1f} min)", f"fits grad>{GRAD_OK:g}: {n_not_conv}, grad>{GRAD_NEAR:g}: {n_bad}")
    if n_bad:
        raise SystemExit(f"収束しなかった問（勾配 > {GRAD_NEAR:g}）が {n_bad} 件ある（units.csv の fit_ok・fit_ok_cand）．出力は書いた")
    return 0


if __name__ == "__main__":
    sys.exit(main())
