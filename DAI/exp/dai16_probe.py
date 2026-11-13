"""実験 16 の下調べ（DAI-16．第 2 版）：声・顔・言葉の予測を「何をどの程度信頼するか」を決めてからまとめた疑似ラベルが，
読み替えない積（今の提案の主の版の疑似ラベル）より当たるか，その差を K 本から見積もれるかを，実験 1 の仮のターゲットの回の保存だけで測る．
設計：DAI/docs/実験16_下調べ_設計.md．評価の回の保存は開かない．生徒の学び直しはしない（新しい学習は，K 本の交差の推定の作り直しと，
小さな最適化（重み・切片・信頼の倍率の式）だけ）．

信頼を決める軸：感情ごと（取り違えの表），ソースのコーパス × モダリティ（重み），サンプルごと（確信度と一致から決める倍率）．
段：
  selftest  人工の数での検算
  units     120 単位（T × 仮のターゲット c × 分割 × 種）．単位ごとに <out>/units/<名前>.npz と .csv
  summary   c × K ごとの UAR，差と区間，判定の数え，見積もりの当たり方

使い方：
  .venv/bin/python DAI/exp/dai16_probe.py --stage selftest
  .venv/bin/python DAI/exp/dai16_probe.py --stage units
  .venv/bin/python DAI/exp/dai16_probe.py --stage summary
試走：--out-dir <一時の置き場> --allow-dirty --allow-partial --targets AFEW --folds 0 --seeds 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

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
D_MIN = 0.25                            # 表の対角の平均がこれ以下の分類器は使わない（一様な表）
EPS = 1e-9                              # 読み替えの前に表の全成分に足す数
CHECK_TEACHER = dai15.CHECK_TEACHER     # K 本を全部使って作り直した積と保存の積の差の許し
GRAD_TOL = 1e-5                         # 交差の推定の作り直しの勾配の最大の許し（実験 1 と同じ基準）
FLOOR = 1.0                             # 「上回る」「下回る」の点推定の下限（点）
TAU = 1.0                               # 重みと切片を K 本で決めるときの罰則の強さ（主）
TAUS_REC = (0.1, 10.0)                  # 記録の版の罰則の強さ
W_MAX = 5.0                             # 重みの上限（下限は 0）
GATE_C = 100.0                          # 信頼の倍率の式（ロジスティック回帰）の罰則の逆数
ASSUME_FLAG = 0.05                      # 仮定の表の旗の基準
N_UNITS = 120
SEED = 20261113
MODS = "AVT"
SCRIPTS = ("dai16_probe.py", "dai15.py", "dai12_probe.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py", "dai_data.py")
# 表の版：(表の作り方, 形, 縮小 λ, 3 つのモダリティで共通の表にするか)
TABLE_SPECS = {"orc_S": ("orc", "S", 0.0, False), "orc_D": ("orc", "D", 0.0, False), "orc_F": ("orc", "F", 0.0, False),
               "orc_Dc": ("orc", "D", 0.0, True), "orc_Fc": ("orc", "F", 0.0, True),
               "cf_S": ("cf", "S", 0.0, False), "cf_D_l0": ("cf", "D", 0.0, False), "cf_D_l05": ("cf", "D", 0.5, False),
               "cf_F_l0": ("cf", "F", 0.0, False), "cf_F_l05": ("cf", "F", 0.5, False), "cf_D_l05c": ("cf", "D", 0.5, True),
               "em1_S": ("em1", "S", 0.0, False), "em1_D": ("em1", "D", 0.0, False), "em1_F": ("em1", "F", 0.0, False), "em1_Dc": ("em1", "D", 0.0, True)}
ORC = ("orc_bias", "orc_S", "orc_D", "orc_F", "orc_Dc", "orc_Fc", "orc_F_hard", "orc_opt_M", "orc_opt_SM", "orc_gate", "orc_all")   # 候補の全ラベルを使う参照
PRACT = ("k_bias", "cf_S", "cf_D_l0", "cf_D_l05", "cf_F_l0", "cf_F_l05", "cf_D_l05c", "em1_S", "em1_D", "em1_F", "em1_Dc",
         "opt_M", "opt_SM", "opt_M_t01", "opt_M_t10", "opt_SM_t01", "opt_SM_t10", "gate", "all_K")                   # 実際に使える版
EST = ("k_bias", "cf_S", "cf_D_l05", "cf_F_l05", "em1_D", "opt_M", "opt_SM", "gate", "all_K")                         # K 本での見積もりを出す版
SEL = tuple(f"sel_{v}" for v in EST) + ("sel_best",)
VERSIONS = ("prod", "prod_sel") + ORC + PRACT + SEL
REFS = ("none_AVT", "joint_AVT", "ret_A", "ret_V", "ret_T")
MAIN_PRACT = {"感情": "cf_D_l05", "ソース": "opt_SM", "サンプル": "gate", "総合": "all_K"}
MAIN_ORC = {"感情（全部）": "orc_F", "感情（対角）": "orc_D", "モダリティ": "orc_S", "ソース": "orc_opt_SM", "サンプル": "orc_gate", "総合": "orc_all"}
COMMON_OF = {"orc_F": "orc_Fc", "orc_D": "orc_Dc", "cf_D_l05": "cf_D_l05c", "em1_D": "em1_Dc"}                        # モダリティで分けない表の対照
PER_EMO = ("prod", "orc_bias", "orc_D", "orc_F", "k_bias", "cf_D_l05", "em1_D", "opt_SM", "gate", "all_K", "none_AVT")
RUN = {"commit": "", "dirty": True}


# ------------------------------------------------------------------ 表の形と読み替え

def shape_S(C: np.ndarray) -> np.ndarray:
    """分類器ごとに 1 つの数：対角を対角の平均に，非対角を等しくした表．"""
    return dai15.shrink(C, 1.0)


def shape_D(C: np.ndarray) -> np.ndarray:
    """感情ごとの対角：対角はそのまま，行 i の非対角を (1 − C_ii)/3 にした表．"""
    d = np.diag(C)
    out = np.repeat(((1.0 - d) / 3.0)[:, None], 4, axis=1)
    np.fill_diagonal(out, d)
    return out


def shaped(C: np.ndarray, shape: str, lam: float = 0.0) -> tuple[np.ndarray, bool]:
    """表 C を形（S・D・F）にし，λ で形 S へ縮める．対角の平均が D_MIN 以下なら一様な表（その分類器は掛け合わせに寄与しない）．"""
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
    return B[P.argmax(1)] if hard else P @ B


def lg(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(P, 1e-300, None))


def norm_log(L: np.ndarray) -> np.ndarray:
    """対数の点数 (n, 4) → 正規化した対数確率．"""
    return L - logsumexp(L, axis=1, keepdims=True)


def hard_table(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return dai15.table_from(np.eye(4)[P.argmax(1)], y)


def em1_table(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """仮のラベルの重み R (n, 4) で数えた表：C[i, j] = Σ_x R(x, i) p(j | x) / Σ_x R(x, i)．"""
    return (R.T @ P) / np.clip(R.sum(0)[:, None], 1e-12, None)


def table_set(C: dict, shp: str, lam: float, common: bool = False) -> tuple[dict, int]:
    """3 つのモダリティの表を形にする．common なら 3 つの表の平均を全モダリティに使う（モダリティで分けない対照）．戻り：(表，使わない条件に当たった数)．"""
    if common:
        Cb = np.mean([C[m] for m in MODS], 0)
        C = {m: Cb for m in MODS}
    out, nun = {}, 0
    for m in MODS:
        out[m], un = shaped(C[m], shp, lam)
        nun += int(un)
    return out, nun


# ------------------------------------------------------------------ 重みと切片の最適化（凸）

def fit_wb(Ls: np.ndarray, y: np.ndarray, w0: np.ndarray, tau: float, fit_w: bool = True, fit_b: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """log q(i | x) ∝ Σ_k w_k Ls[k, x, i] + b_i．感情ごとに等しい重みの平均の負の対数尤度 ＋ (tau/2)(‖w − w0‖² ＋ ‖b‖²) を最小にする．
    0 ≤ w ≤ W_MAX．戻り：(w, b)．fit_w が偽なら w = w0，fit_b が偽なら b = 0．"""
    k, n, _ = Ls.shape
    cnt = np.bincount(y, minlength=4).astype(float)
    sw = 1.0 / (cnt[y] * (cnt > 0).sum())                                  # 感情ごとに等しい重み（和は 1）
    Y = np.eye(4)[y]
    nw = k if fit_w else 0

    def unpack(th):
        w = th[:nw] if fit_w else w0
        b = th[nw:] if fit_b else np.zeros(4)
        return w, b

    def fun(th):
        w, b = unpack(th)
        Z = np.tensordot(w, Ls, axes=1) + b
        lp = Z - logsumexp(Z, axis=1, keepdims=True)
        f = -float((sw * lp[np.arange(n), y]).sum()) + 0.5 * tau * (float(((w - w0) ** 2).sum()) * fit_w + float((b ** 2).sum()) * fit_b)
        D = (np.exp(lp) - Y) * sw[:, None]                                 # (n, 4)
        g = []
        if fit_w:
            g.append(np.einsum("kni,ni->k", Ls, D) + tau * (w - w0))
        if fit_b:
            g.append(D.sum(0) + tau * b)
        return f, np.concatenate(g)

    th0 = np.concatenate(([w0] if fit_w else []) + ([np.zeros(4)] if fit_b else []))
    bounds = [(0.0, W_MAX)] * nw + [(None, None)] * (4 if fit_b else 0)
    r = minimize(fun, th0, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=2000, ftol=1e-12, gtol=1e-8))
    return unpack(r.x)


# ------------------------------------------------------------------ サンプルごとの信頼の倍率（確信度と一致）

def feats(P: np.ndarray, others: list) -> np.ndarray:
    """ラベルを使わない特徴：確信度（最大の確率），ほかの 2 つのモダリティのうち判定が一致した数が 1，2 の印．"""
    a = P.argmax(1)
    agree = sum((Po.argmax(1) == a).astype(int) for Po in others)
    return np.column_stack([P.max(1), agree == 1, agree == 2]).astype(float)


def fit_gate(F: np.ndarray, correct: np.ndarray, sw: np.ndarray | None = None) -> np.ndarray:
    """当たるかどうかのロジスティック回帰．戻り：[切片, 係数 3]．どちらかのクラスしか無ければ割合の対数オッズだけ．"""
    if correct.min() == correct.max():
        p = float(np.clip(correct.mean(), 1e-3, 1 - 1e-3))
        return np.array([np.log(p / (1 - p)), 0.0, 0.0, 0.0])
    m = LogisticRegression(C=GATE_C, max_iter=2000).fit(F, correct.astype(int), sample_weight=sw)
    return np.concatenate([m.intercept_, m.coef_[0]])


def gate_prob(coef: np.ndarray, F: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(coef[0] + F @ coef[1:])))


GATE_CACHE: dict = {}


def source_gate(u) -> dict:
    """ソースのコーパスどうしで，一方で学んだ分類器をもう一方の全発話に当て，当たるかどうかを確信度と一致から当てる式をモダリティごとに学ぶ．
    特徴は当てる先のコーパスの全発話の平均と標準偏差で標準化する（ターゲットに当てるときと同じ扱い）．使うラベルはソースのラベルだけ．
    当てる先のコーパスごとの重みの和を等しくする．"""
    key = tuple(u.srcs)
    if key in GATE_CACHE:
        return GATE_CACHE[key]
    Fs, ys, ws = {m: [] for m in MODS}, {m: [] for m in MODS}, {m: [] for m in MODS}
    for tgt in u.srcs:
        cp = dai1.Corpus.get(tgt)
        for src in u.srcs:
            if src == tgt:
                continue
            P = {}
            for m in MODS:
                X = cp.X[m]
                Xs = (X - X.mean(0)) / X.std(0).clamp(min=1e-6)
                W, b = dai1.load_source(OUT1, src, m)
                P[m] = dai1.probs(Xs, W, b).cpu().numpy().astype(np.float64)
            for m in MODS:
                Fs[m].append(feats(P[m], [P[o] for o in MODS if o != m]))
                ys[m].append(P[m].argmax(1) == cp.y)
                ws[m].append(np.full(len(cp.y), 1.0 / len(cp.y)))
    out = {}
    for m in MODS:
        n_tot, n_dir = sum(len(a) for a in ys[m]), len(ys[m])
        out[m] = fit_gate(np.concatenate(Fs[m]), np.concatenate(ys[m]), np.concatenate(ws[m]) * n_tot / n_dir)
    GATE_CACHE[key] = out
    return out


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
        out[m] = torch.softmax(S.logits(u.Xp[m], rm["W"], rm["b"]), -1).cpu().numpy().astype(np.float64)
    if conv[0] > GRAD_TOL:
        raise SystemExit(f"交差の推定の作り直しの勾配の最大が {conv[0]:.1e}（K={K}）")
    return out, conv


# ------------------------------------------------------------------ 見方（発話の集まり）ごとの部品の確率と，版ごとの対数確率

class View:
    """発話の集まり 1 つの上の部品の確率．ret：{m: (n, 4)}（学び直した声・顔・言葉），src：{(s, m): (n, 4)}（ソースごとの分類器をそのまま当てたもの）．"""

    def __init__(self, ret: dict, src: dict, srcs: list):
        self.ret, self.src, self.srcs = ret, src, list(srcs)
        self.F_ret = {m: feats(ret[m], [ret[o] for o in MODS if o != m]) for m in MODS}
        self.F_src = {(s, m): feats(src[(s, m)], [src[(s, o)] for o in MODS if o != m]) for s in self.srcs for m in MODS}

    def take(self, idx: np.ndarray) -> "View":
        return View({m: self.ret[m][idx] for m in MODS}, {k: v[idx] for k, v in self.src.items()}, self.srcs)

    def comps(self) -> list:
        return [("ret", m) for m in MODS] + [("src", (s, m)) for s in self.srcs for m in MODS]


def gamma(view: View, gate: dict, gnorm: dict) -> dict:
    """部品ごとのサンプルの倍率 γ(x) = g(x) / (候補の上の g の平均)．"""
    out = {("ret", m): gate_prob(gate[m], view.F_ret[m]) / gnorm[("ret", m)] for m in MODS}
    for (s, m), F in view.F_src.items():
        out[("src", (s, m))] = gate_prob(gate[m], F) / gnorm[("src", (s, m))]
    return out


def gate_norm(pool: View, gate: dict) -> dict:
    out = {("ret", m): float(gate_prob(gate[m], pool.F_ret[m]).mean()) for m in MODS}
    for (s, m), F in pool.F_src.items():
        out[("src", (s, m))] = float(gate_prob(gate[m], F).mean())
    return out


def stack_all(view: View, tabs: dict, gate: dict | None, gnorm: dict | None) -> np.ndarray:
    """総合の版の部品の対数確率 (部品, n, 4)：学び直した分類器は表で読み替え，ソースごとの分類器はそのまま．gate があれば倍率 γ を掛ける．"""
    g = gamma(view, gate, gnorm) if gate is not None else None
    Ls = []
    for kind, key in view.comps():
        L = lg(reread(view.ret[key], tabs[key])) if kind == "ret" else lg(view.src[key])
        Ls.append(L * g[(kind, key)][:, None] if g is not None else L)
    return np.stack(Ls)


def w0_of(view: View, with_src: bool) -> np.ndarray:
    return np.concatenate([np.ones(3), np.zeros(3 * len(view.srcs) if with_src else 0)])


def stack_plain(view: View, with_src: bool) -> np.ndarray:
    Ls = [lg(view.ret[m]) for m in MODS]
    if with_src:
        Ls += [lg(view.src[(s, m)]) for s in view.srcs for m in MODS]
    return np.stack(Ls)


def fit_k(train: View, y: np.ndarray, gate: dict, gnorm: dict) -> dict:
    """K 本（の一部）の上の，抜いて作った確率 train とラベル y から，K 本で決める量を全部決める：交差の表，切片，重み，総合の版の重み．"""
    par = {"C_cf": {m: dai15.table_from(train.ret[m], y) for m in MODS}}
    prod = sum(lg(train.ret[m]) for m in MODS)[None]
    par["k_bias"] = fit_wb(prod, y, np.ones(1), TAU, fit_w=False, fit_b=True)[1]
    for name, tau in (("", TAU), ("_t01", TAUS_REC[0]), ("_t10", TAUS_REC[1])):
        par[f"opt_M{name}"] = fit_wb(stack_plain(train, False), y, w0_of(train, False), tau)[0]
        par[f"opt_SM{name}"] = fit_wb(stack_plain(train, True), y, w0_of(train, True), tau)[0]
    tabs = table_set(par["C_cf"], "D", 0.5)[0]
    par["all_K"] = fit_wb(stack_all(train, tabs, gate, gnorm), y, w0_of(train, True), TAU)[0]
    return par


def logq_pract(v: str, view: View, par: dict, C_em1: dict, gate: dict, gnorm: dict) -> np.ndarray:
    """実際に使える版 v の正規化した対数確率 (n, 4)．par は fit_k の戻り．"""
    if v == "prod":
        return norm_log(sum(lg(view.ret[m]) for m in MODS))
    if v == "k_bias":
        return norm_log(sum(lg(view.ret[m]) for m in MODS) + par["k_bias"])
    if v in TABLE_SPECS:
        src, shp, lam, common = TABLE_SPECS[v]
        tabs, _ = table_set(par["C_cf"] if src == "cf" else C_em1, shp, lam, common)
        return norm_log(sum(lg(reread(view.ret[m], tabs[m])) for m in MODS))
    if v.startswith("opt_M"):
        return norm_log(np.tensordot(par[v], stack_plain(view, False), axes=1))
    if v.startswith("opt_SM"):
        return norm_log(np.tensordot(par[v], stack_plain(view, True), axes=1))
    if v == "gate":
        g = gamma(view, gate, gnorm)
        return norm_log(sum(lg(view.ret[m]) * g[("ret", m)][:, None] for m in MODS))
    if v == "all_K":
        tabs = table_set(par["C_cf"], "D", 0.5)[0]
        return norm_log(np.tensordot(par["all_K"], stack_all(view, tabs, gate, gnorm), axes=1))
    raise KeyError(v)


def bal_ll(lq: np.ndarray, y: np.ndarray) -> float:
    """感情ごとに等しい重みの平均の対数尤度．"""
    v = lq[np.arange(len(y)), y]
    return float(np.mean([v[y == k].mean() for k in range(4) if (y == k).any()]))


# ------------------------------------------------------------------ 1 単位

def run_unit(sv) -> tuple[list, dict]:
    u = sv.u
    TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
    rows, arr = [], {}
    yt, yp = u.ytest, u.ypool
    arr["y_test"] = yt.astype(np.int8)
    arr["cl_test"] = sv.cl
    arr["sid_test"] = sv.sid
    arr["sid_pool"] = sv.sid_pool
    arr["cl_pool"] = sv.cl_pool
    arr["y_pool"] = yp.astype(np.int8)
    gate = source_gate(u)
    arr["gate_src"] = np.stack([gate[m] for m in MODS])
    src_p = {(s, m): dai1.probs(u.Xp[m], W, b).cpu().numpy().astype(np.float64) for m in MODS for s, W, b in u.src[m]}
    src_t = {(s, m): dai1.probs(u.Xt[m], W, b).cpu().numpy().astype(np.float64) for m in MODS for s, W, b in u.src[m]}
    for (s, m), P in src_p.items():                                       # 発話ごとの確率を保存する（計算の後の解析を学び直さずに出すため）
        arr[f"p_pool_src_{s}_{m}"] = P.astype(np.float32)
        arr[f"p_test_src_{s}_{m}"] = src_t[(s, m)].astype(np.float32)
    for K in KS:
        if sv.fz["main"][str(K)] != "one":
            raise SystemExit(f"{sv.base.name}：実験 1 の主の版が one でない（K={K}）")
        kpos = u.kpos[K]
        yk = yp[kpos]
        nk = np.setdiff1d(np.arange(u.n), kpos)
        pool = View({m: TK[K][f"ret_{m}"][0].cpu().numpy().astype(np.float64) for m in MODS}, src_p, u.srcs)
        test = View({m: TK[K][f"ret_{m}"][1].cpu().numpy().astype(np.float64) for m in MODS}, src_t, u.srcs)
        pnk = pool.take(nk)
        q0p = dai1.product(TK[K], (1, 1, 1), 0).cpu().numpy().astype(np.float64)
        # ---- 交差の推定：K 本の全部，1 組を抜いたもの 5 つ，2 組を抜いたもの 10
        fold = dai15.kfold_of(yk)
        assert all(np.bincount(fold[yk == i], minlength=NF).min() >= 1 for i in range(4)), "組に感情が欠ける"
        for m in MODS:
            arr[f"p_pool_ret_{m}_K{K}"] = pool.ret[m].astype(np.float32)
            arr[f"p_test_ret_{m}_K{K}"] = test.ret[m].astype(np.float32)
        pairs = list(itertools.combinations(range(NF), 2))
        sets = [kpos] + [kpos[fold != j] for j in range(NF)] + [kpos[(fold != i) & (fold != j)] for i, j in pairs]
        pidx = {p: 1 + NF + n for n, p in enumerate(pairs)}
        Pm, conv = teachers_mod(u, sv.teth, K, sets)
        dev = float(np.abs(np.exp(norm_log(sum(lg(Pm[m][0]) for m in MODS))) - q0p).max())
        dev_m = {f"teacher_dev_{m}": float(np.abs(Pm[m][0] - pool.ret[m]).max()) for m in MODS}
        if dev > CHECK_TEACHER:
            raise SystemExit(f"{sv.base.name}：K 本を全部使って作り直した積が保存の積と {dev:.2e} 違う（K={K}）")
        gnorm = gate_norm(pool, gate)
        held = View({m: np.stack([Pm[m][1 + fold[a]][kpos[a]] for a in range(len(kpos))]) for m in MODS},
                    {k: v[kpos] for k, v in src_p.items()}, u.srcs)                         # 抜いた K 本の発話の上の確率（K 本の並び）
        # ---- K 本で決める量（全部の K 本）と，候補の仮のラベルの表
        par = fit_k(held, yk, gate, gnorm)
        R = q0p.copy()
        R[kpos] = np.eye(4)[yk]
        C_em1 = {m: em1_table(pool.ret[m], R) for m in MODS}
        # ---- 候補の全ラベルを使う参照
        ynk = yp[nk]
        C_orc = {m: dai15.table_from(pnk.ret[m], ynk) for m in MODS}
        C_hard = {m: hard_table(pnk.ret[m], ynk) for m in MODS}
        orc = {"bias": fit_wb(sum(lg(pnk.ret[m]) for m in MODS)[None], ynk, np.ones(1), 0.0, fit_w=False, fit_b=True)[1],
               "opt_M": fit_wb(stack_plain(pnk, False), ynk, w0_of(pnk, False), 0.0)[0],
               "opt_SM": fit_wb(stack_plain(pnk, True), ynk, w0_of(pnk, True), 0.0)[0],
               "gate": {m: fit_gate(pnk.F_ret[m], pnk.ret[m].argmax(1) == ynk) for m in MODS}}
        orc["gnorm"] = gate_norm(pool, orc["gate"])
        orc["tabsD"] = table_set(C_orc, "D", 0.0)[0]
        orc["all"] = fit_wb(stack_all(pnk, orc["tabsD"], orc["gate"], orc["gnorm"]), ynk, w0_of(pnk, True), 0.0)[0]

        def logq(v: str, view: View) -> tuple[np.ndarray, int]:
            if v == "prod_sel":
                return norm_log(sum(w * lg(view.ret[m]) for w, m in zip(sv.wsel[K], MODS))), 0
            if v == "orc_bias":
                return norm_log(sum(lg(view.ret[m]) for m in MODS) + orc["bias"]), 0
            if v == "orc_F_hard":
                tabs, nun = table_set(C_hard, "F", 0.0)
                return norm_log(sum(lg(reread(view.ret[m], tabs[m], hard=True)) for m in MODS)), nun
            if v in TABLE_SPECS and TABLE_SPECS[v][0] == "orc":
                _, shp, lam, common = TABLE_SPECS[v]
                tabs, nun = table_set(C_orc, shp, lam, common)
                return norm_log(sum(lg(reread(view.ret[m], tabs[m])) for m in MODS)), nun
            if v == "orc_opt_M":
                return norm_log(np.tensordot(orc["opt_M"], stack_plain(view, False), axes=1)), 0
            if v == "orc_opt_SM":
                return norm_log(np.tensordot(orc["opt_SM"], stack_plain(view, True), axes=1)), 0
            if v == "orc_gate":
                g = gamma(view, orc["gate"], orc["gnorm"])
                return norm_log(sum(lg(view.ret[m]) * g[("ret", m)][:, None] for m in MODS)), 0
            if v == "orc_all":
                return norm_log(np.tensordot(orc["all"], stack_all(view, orc["tabsD"], orc["gate"], orc["gnorm"]), axes=1)), 0
            nun = 0
            if v in TABLE_SPECS:
                src, shp, lam, common = TABLE_SPECS[v]
                nun = table_set(par["C_cf"] if src == "cf" else C_em1, shp, lam, common)[1]
            return logq_pract(v, view, par, C_em1, gate, gnorm), nun

        chk = float(np.abs(np.exp(norm_log(sum(lg(reread(test.ret[m], np.eye(4))) for m in MODS))) - np.exp(logq("prod", test)[0])).max())
        if chk > 1e-4:
            raise SystemExit(f"{sv.base.name}：単位行列の表の読み替えが積と {chk:.2e} 違う（K={K}）")
        # ---- K 本での見積もり（入れ子）：組 j の発話は，組 j を抜いて作った確率と，組 j のラベルを使わずに決めた量で当てる
        est_lq = {v: np.empty((len(kpos), 4)) for v in ("prod",) + EST}
        for j in range(NF):
            te = np.flatnonzero(fold == j)
            tr = np.flatnonzero(fold != j)
            trv = View({m: np.stack([Pm[m][pidx[tuple(sorted((int(fold[a]), j)))]][kpos[a]] for a in tr]) for m in MODS},
                       {k: v[kpos[tr]] for k, v in src_p.items()}, u.srcs)
            par_j = fit_k(trv, yk[tr], gate, gnorm)
            Rj = np.exp(norm_log(sum(lg(Pm[m][1 + j]) for m in MODS)))
            Rj[kpos[tr]] = np.eye(4)[yk[tr]]
            Cem_j = {m: em1_table(Pm[m][1 + j], Rj) for m in MODS}
            hv = held.take(te)
            for v in est_lq:
                est_lq[v][te] = logq_pract(v, hv, par_j, Cem_j, gate, gnorm)
        est = {v: (bal_ll(L, yk), 100 * S.uar(yk, L.argmax(1))) for v, L in est_lq.items()}
        gain = {v: est[v][0] - est["prod"][0] for v in EST}
        # ---- 版ごとの判定
        target = torch.as_tensor(np.exp(logq("prod", pool)[0]).mean(0))
        lqt, lqp, unused = {}, {}, {}
        for v in ("prod", "prod_sel") + ORC + PRACT:
            lqt[v], unused[v] = logq(v, test)
            lqp[v] = logq(v, pool)[0]
        best = max(("prod",) + EST, key=lambda v: (est[v][0], v == "prod"))                # 同点は積
        for v in EST:
            src_v = v if gain[v] > 0 else "prod"
            lqt[f"sel_{v}"], lqp[f"sel_{v}"], unused[f"sel_{v}"] = lqt[src_v], lqp[src_v], unused[src_v]
        lqt["sel_best"], lqp["sel_best"], unused["sel_best"] = lqt[best], lqp[best], unused[best]
        Wj, bj = sv.prop("AVT", K)
        refs_t = {"none_AVT": TK[K]["none_AVT"][1].argmax(1).cpu().numpy(), "joint_AVT": S.logits(u.Xt["AVT"], Wj, bj).argmax(1).cpu().numpy()}
        refs_p = {"none_AVT": TK[K]["none_AVT"][0].argmax(1).cpu().numpy()[nk], "joint_AVT": S.logits(u.Xp["AVT"], Wj, bj).argmax(1).cpu().numpy()[nk]}
        for m in MODS:
            refs_t[f"ret_{m}"], refs_p[f"ret_{m}"] = test.ret[m].argmax(1), pnk.ret[m].argmax(1)
        for v in VERSIONS:
            dl, rs = dai15.match_prior(torch.as_tensor(lqp[v]), target)
            pt, pp, pa = lqt[v].argmax(1), lqp[v][nk].argmax(1), (lqt[v] + dl.numpy()).argmax(1)
            arr[f"pred_test_K{K}_{v}"] = pt.astype(np.int8)
            arr[f"pred_pool_K{K}_{v}"] = pp.astype(np.int8)
            arr[f"pred_testadj_K{K}_{v}"] = pa.astype(np.int8)
            arr[f"q_test_K{K}_{v}"] = np.exp(lqt[v]).astype(np.float16)
            base = v[4:] if v.startswith("sel_") and v != "sel_best" else v
            rows.append(dict(K=K, version=v, uar_test=100 * S.uar(yt, pt), uar_pool=100 * S.uar(ynk, pp), uar_test_adj=100 * S.uar(yt, pa),
                             ll_test=bal_ll(lqt[v], yt), n_unused=unused[v], adj_resid=rs,
                             est_ll=est[base][0] if base in est else np.nan, est_uar=est[base][1] if base in est else np.nan,
                             est_gain=gain.get(base, np.nan), used=(best if v == "sel_best" else (str(gain[base] > 0) if v.startswith("sel_") else ""))))
        for v in REFS:
            arr[f"pred_test_K{K}_{v}"] = refs_t[v].astype(np.int8)
            arr[f"pred_pool_K{K}_{v}"] = refs_p[v].astype(np.int8)
            rows.append(dict(K=K, version=v, uar_test=100 * S.uar(yt, refs_t[v]), uar_pool=100 * S.uar(ynk, refs_p[v])))
        # ---- 保存：表，重み，抜いた確率
        Cn = {m: dai15.table_from(TK[K][f"none_{m}"][0].cpu().numpy()[nk], ynk) for m in MODS}
        for m in MODS:
            for nm, C in (("orc", C_orc), ("cf", par["C_cf"]), ("em1", C_em1), ("orchard", C_hard), ("orcnone", Cn)):
                arr[f"C_{nm}_{m}_K{K}"] = C[m]
            arr[f"pcf_{m}_K{K}"] = held.ret[m].astype(np.float32)
            arr[f"gate_orc_{m}_K{K}"] = orc["gate"][m]
        for nm in ("k_bias", "opt_M", "opt_SM", "opt_M_t01", "opt_M_t10", "opt_SM_t01", "opt_SM_t10", "all_K"):
            arr[f"par_{nm}_K{K}"] = par[nm]
        for nm in ("bias", "opt_M", "opt_SM", "all"):
            arr[f"par_orc_{nm}_K{K}"] = orc[nm]
        arr[f"comps_K{K}"] = np.array([f"{kind}:{key if kind == 'ret' else key[0] + ':' + key[1]}" for kind, key in pool.comps()])
        arr[f"kpos_K{K}"] = kpos.astype(np.int32)
        arr[f"kfold_K{K}"] = fold.astype(np.int8)
        arr[f"yk_K{K}"] = yk.astype(np.int8)
        arr[f"y_pool_K{K}"] = ynk.astype(np.int8)
        arr[f"cl_pool_K{K}"] = sv.cl_pool[nk]
        # ---- 仮定の表：(a) 本当の感情を固定したときの判定の一致の余り，(b) 候補とテストの表のずれ，(c) 感情の割合，(d) 確率の平均の表と判定の表の対角の差
        ex = {}
        for a, b in (("A", "V"), ("A", "T"), ("V", "T")):
            ya, yb = pnk.ret[a].argmax(1), pnk.ret[b].argmax(1)
            ex[f"excess_{a}{b}"] = float(np.mean([(ya[ynk == i] == yb[ynk == i]).mean() - float(C_hard[a][i] @ C_hard[b][i]) for i in range(4) if (ynk == i).any()]))
        drift = {f"drift_{m}": float(np.abs(np.diag(dai15.table_from(test.ret[m], yt)) - np.diag(C_orc[m])).mean()) for m in MODS}
        calib = {f"calib_{m}": float((np.diag(C_hard[m]) - np.diag(C_orc[m])).mean()) for m in MODS}
        rows.append(dict(K=K, version="_check", teacher_dev=dev, **dev_m, identity_dev=chk, grad=conv[0], stalled=conv[1], **ex, **drift, **calib,
                         **{f"share_pool_{i}": float((ynk == i).mean()) for i in range(4)},
                         **{f"dbar_{nm}_{m}": float(np.mean(np.diag(C[m]))) for nm, C in (("orc", C_orc), ("cf", par["C_cf"]), ("em1", C_em1)) for m in MODS},
                         **{f"lowdiag_{nm}_{m}": int((np.diag(C[m]) < 0.25).sum()) for nm, C in (("orc", C_orc), ("cf", par["C_cf"]), ("em1", C_em1)) for m in MODS}))
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
            print(f"{name}：{time.time() - t0:.0f} 秒．行 {len(df)}．作り直しの差 {ck.teacher_dev.max():.1e}．勾配の最大 {ck.grad.max():.1e}．止まり {int(ck.stalled.sum())}．"
                  f"割合をそろえる残り {df.adj_resid.max():.1e}", flush=True)
            del sv
        GATE_CACHE.clear()
        for cname in list(dai1.Corpus.cache):
            dai1.Corpus.cache.pop(cname, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verdict(lo: float, hi: float, est: float) -> str:
    if lo > 0 and est >= FLOOR:
        return "上回る"
    if hi < 0 and est <= -FLOOR:
        return "下回る"
    return "言えない"


def row_rng(*key) -> np.random.Generator:
    """区間の引き直しの乱数を行ごとに作る（版を足しても既にある行の区間が変わらないようにする）．"""
    return np.random.default_rng([SEED, zlib.crc32("|".join(map(str, key)).encode())])


def stage_summary(out: Path, allow_partial: bool) -> None:
    files = sorted((out / "units").glob("*.npz"))
    if len(files) != N_UNITS and not allow_partial:
        raise SystemExit(f"単位の出力が {len(files)} 個（{N_UNITS} のはず）．試走なら --allow-partial")
    csvs = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in files], ignore_index=True)
    if csvs.commit.nunique() != 1 and not allow_partial:
        raise SystemExit(f"単位のコミットが 1 つでない：{csvs.commit.unique()}")
    cells = {"test": {}, "pool": {}, "adj": {}}                          # [where][(c, K, 版)][(f, r)] = 塊ごとの数（3 つの T の平均）
    emo = {}
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        f, r = int(f[1:]), int(r[1:])
        z = np.load(p, allow_pickle=False)
        yt = z["y_test"].astype(int)
        for K in KS:
            ypk = z[f"y_pool_K{K}"].astype(int)
            for v in VERSIONS + REFS:
                pt = z[f"pred_test_K{K}_{v}"].astype(int)
                ct = ST.counts(yt, pt, z["cl_test"])
                P12._add(cells["test"].setdefault((c, K, v), {}), (f, r), ct, f"{p.stem} test {v}")
                P12._add(cells["pool"].setdefault((c, K, v), {}), (f, r), ST.counts(ypk, z[f"pred_pool_K{K}_{v}"].astype(int), z[f"cl_pool_K{K}"]), f"{p.stem} pool {v}")
                if v in VERSIONS:
                    P12._add(cells["adj"].setdefault((c, K, v), {}), (f, r), ST.counts(yt, z[f"pred_testadj_K{K}_{v}"].astype(int), z["cl_test"]), f"{p.stem} adj {v}")
                if v in PER_EMO:
                    e = emo.setdefault((c, K, v), [np.zeros(4), np.zeros(4), np.zeros(4), 0])
                    e[0] += ct[1].sum(0); e[1] += ct[2].sum(0); e[2] += np.bincount(pt, minlength=4); e[3] += 1
    for w in cells:
        P12._finish(cells[w])
    geff = {c: ST.g_eff_of(c)[0] for c in DD.CORPORA}
    sdir = out / "summary"
    sdir.mkdir(exist_ok=True)

    def level(d: dict) -> float:
        return float(np.mean([np.mean([ST.uar_full(d[k]) for k in d if k[0] == f]) for f in sorted({k[0] for k in d})]))

    lev = [dict(c=c, K=K, version=v, uar_test=level(d), uar_pool=level(cells["pool"][(c, K, v)]),
                uar_test_adj=(level(cells["adj"][(c, K, v)]) if (c, K, v) in cells["adj"] else np.nan), n_fr=len(d))
           for (c, K, v), d in sorted(cells["test"].items())]
    pd.DataFrame(lev).to_csv(sdir / "levels.csv", index=False)
    # ---- 差
    con = [(v, "prod") for v in VERSIONS if v != "prod"]
    con += [(v, "orc_bias") for v in ORC if v != "orc_bias"] + [(v, "k_bias") for v in PRACT if v != "k_bias"]
    con += list(COMMON_OF.items())
    con += [(f"sel_{v}", v) for v in EST] + [("orc_D", "orc_S"), ("orc_F", "orc_D"), ("cf_D_l05", "cf_S"), ("cf_D_l05", "orc_D"), ("em1_D", "orc_D"),
                                             ("opt_SM", "opt_M"), ("orc_opt_SM", "orc_opt_M"), ("all_K", "orc_all"), ("joint_AVT", "none_AVT")]
    con += [(v, ref) for v in ("prod",) + tuple(MAIN_ORC.values()) + tuple(MAIN_PRACT.values()) for ref in ("none_AVT", "joint_AVT")]
    eff = []
    for c in DD.CORPORA:
        for K in KS:
            for a, b in con:
                for w in ("test", "pool", "adj"):
                    A = cells[w].get((c, K, a))
                    B = cells[w].get((c, K, b)) if (w != "adj" or b in VERSIONS) else cells["test"].get((c, K, b))
                    if A is None or B is None:
                        continue
                    if set(A) != set(B):
                        raise SystemExit(f"{c} K={K} {a} と {b} の単位がそろわない")
                    es = (P12.effect_stats_pool if w == "pool" else ST.effect_stats)(A, B, geff[c], row_rng(c, K, a, b, w))
                    eff.append(dict(c=c, K=K, a=a, b=b, where=w, est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], method=es["method"],
                                    n_fr=es["n_units"], n_folds=es["n_folds"], p=es.get("p", np.nan), df=es.get("df", np.nan), v_jk=es.get("v_jk", np.nan),
                                    v_seed=es.get("v_seed", np.nan), folds_dropped=es.get("folds_dropped", ""),
                                    verdict=verdict(es["lo"], es["hi"], es["effect"])))
    eff = pd.DataFrame(eff)
    eff.to_csv(sdir / "effects.csv", index=False)

    def count(a: str, b: str, K: int, where: str = "test") -> dict:
        d = eff[(eff.a == a) & (eff.b == b) & (eff["where"] == where) & (eff.K == K)]
        up, dn = d[d.verdict == "上回る"].c.tolist(), d[d.verdict == "下回る"].c.tolist()
        return dict(a=a, b=b, K=K, where=where, n_up=len(up), n_down=len(dn), n_up_noAFEW=len([x for x in up if x != "AFEW"]),
                    n_down_noAFEW=len([x for x in dn if x != "AFEW"]), up=",".join(up), down=",".join(dn))

    crit = []
    for K in KS:
        for axis, v in MAIN_ORC.items():
            for b in ("prod", "orc_bias") + ((COMMON_OF[v],) if v in COMMON_OF else ()):
                if v != b:
                    for w in ("test", "adj", "pool"):
                        crit.append(dict(rule="R1", axis=axis, main=(K == K_MAIN and w == "test"), **count(v, b, K, w)))
        for axis, v in MAIN_PRACT.items():
            for b in ("prod", "k_bias") + ((COMMON_OF[v],) if v in COMMON_OF else ()):
                for w in ("test", "adj", "pool"):
                    crit.append(dict(rule="R2", axis=axis, main=(K == K_MAIN and w == "test"), **count(v, b, K, w)))
            crit.append(dict(rule="R3", axis=axis, main=(K == K_MAIN), **count(f"sel_{v}", "prod", K)))
            crit.append(dict(rule="R3", axis=axis, main=(K == K_MAIN), **count(f"sel_{v}", v, K)))
        for v in ("em1_D",):
            for b in ("prod", "k_bias", COMMON_OF[v]):
                for w in ("test", "adj", "pool"):
                    crit.append(dict(rule="R2", axis="感情（候補の仮のラベル）", main=(K == K_MAIN and w == "test"), **count(v, b, K, w)))
        crit.append(dict(rule="R3", axis="最良を選ぶ", main=(K == K_MAIN), **count("sel_best", "prod", K)))
    crit = pd.DataFrame(crit)
    crit["hit"] = crit.n_up >= 2
    crit["down2"] = crit.n_down >= 2
    crit.to_csv(sdir / "criteria.csv", index=False)
    # ---- 見積もりの当たり方（単位ごとの見積もりの差と，単位ごとのテストの差）
    u = csvs[csvs.version.isin(("prod",) + EST)].copy()
    idx = ["T", "c", "fold", "seed", "K"]
    base = u[u.version == "prod"].set_index(idx)[["uar_test", "ll_test", "est_uar"]]
    er = []
    for v in EST:
        j = u[u.version == v].set_index(idx).join(base, rsuffix="_prod")
        j["act"] = j.uar_test - j.uar_test_prod
        j["act_ll"] = j.ll_test - j.ll_test_prod
        for (c, K), g in j.groupby(["c", "K"]):
            nz = g[g.act != 0]
            er.append(dict(c=c, K=K, version=v, n_units=len(g), est_gain_ll=float(g.est_gain.mean()), est_gain_uar=float((g.est_uar - g.est_uar_prod).mean()),
                           act_gain_uar=float(g.act.mean()), act_gain_ll=float(g.act_ll.mean()), share_used=float((g.est_gain > 0).mean()),
                           sign_agree_uar=(float(((nz.est_gain > 0) == (nz.act > 0)).mean()) if len(nz) else np.nan), n_nonzero=len(nz),
                           sign_agree_ll=float(((g.est_gain > 0) == (g.act_ll > 0)).mean()),
                           spearman_ll=float(g[["est_gain", "act_ll"]].corr(method="spearman").iloc[0, 1])))
    pd.DataFrame(er).to_csv(sdir / "estimate.csv", index=False)
    csvs[csvs.version == "sel_best"].groupby(["c", "K"]).used.value_counts().rename("n").reset_index().to_csv(sdir / "sel_best_used.csv", index=False)
    # ---- 感情ごとの再現率と判定の割合（テスト．単位をまとめた数．区間は無い．件数は単位あたりの平均）
    pe = [dict(c=c, K=K, version=v, **{f"recall_{i}": 100 * e[0][i] / e[1][i] if e[1][i] > 0 else np.nan for i in range(4)},
               **{f"pred_share_{i}": float(e[2][i] / e[2].sum()) for i in range(4)}, **{f"n_per_unit_{i}": float(e[1][i] / e[3]) for i in range(4)})
          for (c, K, v), e in sorted(emo.items())]
    pd.DataFrame(pe).to_csv(sdir / "per_emotion.csv", index=False)
    # ---- 表と重みの単位の平均
    tb, pr = {}, {}
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        z = np.load(p, allow_pickle=False)
        for K in KS:
            for src in ("orc", "cf", "em1", "orchard", "orcnone"):
                for m in MODS:
                    tb.setdefault((c, K, src, m), []).append(z[f"C_{src}_{m}_K{K}"])
            names = [str(x) for x in z[f"comps_K{K}"]]
            for nm in ("par_opt_M", "par_opt_SM", "par_all_K", "par_orc_opt_M", "par_orc_opt_SM", "par_orc_all"):
                for i, x in enumerate(z[f"{nm}_K{K}"]):
                    pr.setdefault((c, K, nm, names[i]), []).append(float(x))
            for nm in ("par_k_bias", "par_orc_bias"):
                for i, x in enumerate(z[f"{nm}_K{K}"]):
                    pr.setdefault((c, K, nm, f"b{i}"), []).append(float(x))
            for i, m in enumerate(MODS):
                for jx, lab in enumerate(("intercept", "conf", "agree1", "agree2")):
                    pr.setdefault((c, K, "gate_src", f"{m}:{lab}"), []).append(float(z["gate_src"][i][jx]))
                    pr.setdefault((c, K, "gate_orc", f"{m}:{lab}"), []).append(float(z[f"gate_orc_{m}_K{K}"][jx]))
    tr = []
    for (c, K, src, m), L in sorted(tb.items()):
        M = np.mean(L, 0)
        sd = np.std([np.diag(x) for x in L], 0)
        tr.append(dict(c=c, K=K, source=src, modality=m, n_units=len(L), **{f"C_{i}{j}": float(M[i, j]) for i in range(4) for j in range(4)},
                       **{f"sd_diag_{i}": float(sd[i]) for i in range(4)}))
    pd.DataFrame(tr).to_csv(sdir / "tables.csv", index=False)
    pd.DataFrame([dict(c=c, K=K, param=nm, item=it, mean=float(np.mean(v)), sd=float(np.std(v)), n=len(v)) for (c, K, nm, it), v in sorted(pr.items())]
                 ).to_csv(sdir / "params.csv", index=False)
    chk = csvs[csvs.version == "_check"]
    acols = [x for x in chk.columns if x.startswith(("excess_", "drift_", "calib_", "share_pool_", "dbar_", "lowdiag_"))]
    am = chk.groupby(["c", "K"])[acols].mean().reset_index()
    for x in [a for a in acols if a.startswith(("excess_", "drift_", "calib_"))]:
        am[f"flag_{x}"] = am[x].abs() > ASSUME_FLAG
    am.to_csv(sdir / "assumptions.csv", index=False)
    csvs[csvs.version != "_check"].groupby(["c", "K", "version"]).n_unused.sum().reset_index().to_csv(sdir / "unused.csv", index=False)
    json.dump(dict(n_units=len(files), teacher_dev_max=float(chk.teacher_dev.max()), identity_dev_max=float(chk.identity_dev.max()),
                   grad_max=float(chk.grad.max()), stalled=int(chk.stalled.sum()), adj_resid_max=float(csvs.adj_resid.max()),
                   sec_total=float(csvs.groupby(["T", "c", "fold", "seed"]).sec.first().sum())),
              open(sdir / "convergence.json", "w"), ensure_ascii=False, indent=1)
    print(f"summary：単位 {len(files)}，差の行 {len(eff)}，判定の行 {len(crit)}", flush=True)


# ------------------------------------------------------------------ 検算

def selftest() -> None:
    rng = np.random.default_rng(0)
    C = rng.dirichlet(np.ones(4) * 2, size=4) * 0.5 + np.eye(4) * 0.5
    # 1. 形
    for shp in "SDF":
        for lam in (0.0, 0.5):
            M, un = shaped(C, shp, lam)
            assert np.allclose(M.sum(1), 1) and not un
    assert np.allclose(np.diag(shape_D(C)), np.diag(C)) and np.allclose(shape_D(C)[0, 1], shape_D(C)[0, 2])
    assert np.allclose(np.diag(shape_S(C)), np.mean(np.diag(C)))
    M, un = shaped(np.full((4, 4), 0.25), "F")
    assert un and np.allclose(M, 0.25)
    # 2. 読み替え
    P = rng.dirichlet(np.ones(4), size=50)
    assert np.allclose(reread(P, C).sum(1), 1) and np.abs(reread(P, np.eye(4)) - P).max() < 1e-6
    assert np.allclose(reread(P, np.full((4, 4), 0.25)), 0.25)
    # 2b. 表の向き：非対称な表で引いた票から，「j と答えたとき本当が i」の割合を数え，posterior_matrix の成分と比べる（転置なら合わない）
    Ca = np.array([[0.70, 0.20, 0.05, 0.05], [0.05, 0.50, 0.40, 0.05], [0.10, 0.05, 0.80, 0.05], [0.30, 0.05, 0.05, 0.60]])
    yy0 = rng.integers(0, 4, 200000)
    cum = Ca.cumsum(1)
    vv0 = (rng.uniform(size=len(yy0))[:, None] > cum[yy0]).sum(1)
    emp = np.stack([np.bincount(yy0[vv0 == j], minlength=4) / max((vv0 == j).sum(), 1) for j in range(4)])
    assert np.abs(posterior_matrix(Ca) - emp).max() < 0.01 and np.abs(posterior_matrix(Ca.T) - emp).max() > 0.1
    n = 8000
    y = rng.integers(0, 4, n)

    def votes(Ct: dict) -> dict:
        return {m: np.full((n, 4), 0.1) + np.eye(4)[np.array([rng.choice(4, p=Ct[m][i]) for i in y])] * 0.6 for m in MODS}

    def uars(Pv: dict) -> tuple:
        tabs = {m: dai15.table_from(Pv[m], y) for m in MODS}
        prod = sum(lg(Pv[m]) for m in MODS)
        u_prod = S.uar(y, prod.argmax(1))
        u_tab = S.uar(y, sum(lg(reread(Pv[m], tabs[m])) for m in MODS).argmax(1))
        tc = table_set(tabs, "F", 0.0, common=True)[0]
        u_com = S.uar(y, sum(lg(reread(Pv[m], tc[m])) for m in MODS).argmax(1))
        b = fit_wb(prod[None], y, np.ones(1), 0.0, fit_w=False, fit_b=True)[1]
        return u_prod, u_tab, u_com, S.uar(y, (prod + b).argmax(1))

    # 3. 分類器ごとに答えにくい感情が違うとき：正確な表の読み替えは，積・共通の表・切片だけの対照のどれよりも当たる
    Ct = {}
    for a, m in enumerate(MODS):
        Cm = np.full((4, 4), 0.15) + np.eye(4) * 0.40
        Cm[a] = 0.25
        Cm[(a + 1) % 4] = np.eye(4)[(a + 1) % 4] * 0.76 + 0.06
        Ct[m] = Cm / Cm.sum(1, keepdims=True)
    u_prod, u_tab, u_com, u_b = uars(votes(Ct))
    assert u_tab > u_prod + 0.01 and u_tab > u_b + 0.005 and u_tab > u_com + 0.005, (u_prod, u_tab, u_com, u_b)
    # 3b. 3 つの分類器に同じ偏りだけがあるとき：モダリティごとの表と共通の表はほぼ同じで，どちらも積より当たる（対照で分けられる）
    Cb = np.full((4, 4), 0.15) + np.eye(4) * 0.40
    Cb[0] = np.array([0.20, 0.50, 0.15, 0.15])
    u_prod, u_tab, u_com, u_b = uars(votes({m: Cb for m in MODS}))
    assert u_tab > u_prod + 0.01 and abs(u_tab - u_com) < 0.01, (u_prod, u_tab, u_com, u_b)
    # 4. 重みの最適化：役に立たない部品の重みは 0 に近づき，罰則が強いと初期値に戻る．最適の点で勾配が 0（端を除く）
    Pa = votes(Ct)["A"]
    Ls = np.stack([lg(Pa), lg(rng.dirichlet(np.ones(4), size=n))])
    w = fit_wb(Ls, y, np.array([1.0, 0.0]), 0.0)[0]
    assert w[1] < 0.05 and w[0] > 0.5, w
    w = fit_wb(Ls, y, np.array([1.0, 0.7]), 1e6)[0]
    assert np.abs(w - np.array([1.0, 0.7])).max() < 1e-3
    yy, LL, w00 = y[:300], Ls[:, :300], np.array([1.0, 0.3])
    f0 = lambda t: -bal_ll(norm_log(np.tensordot(t, LL, axes=1)), yy) + 0.5 * 0.3 * ((t - w00) ** 2).sum()  # noqa: E731
    wopt = fit_wb(LL, yy, w00, 0.3)[0]
    num = np.array([(f0(wopt + e) - f0(wopt - e)) / 2e-5 for e in np.eye(2) * 1e-5])
    assert np.all((np.abs(num) < 1e-4) | (wopt < 1e-6) | (wopt > W_MAX - 1e-6)), (wopt, num)
    # 5. 信頼の倍率：確信度が高いほど当たる人工の数で係数が正．片方のクラスしか無ければ係数 0
    conf = rng.uniform(0.25, 1.0, 4000)
    F = np.column_stack([conf, np.zeros(4000), np.zeros(4000)])
    assert fit_gate(F, rng.uniform(size=4000) < conf)[1] > 0
    assert np.allclose(fit_gate(F, np.ones(4000, bool))[1:], 0)
    # 6. 仮のラベルの表，2 組を抜いた集合の番号，判定の語，版の名前の重複
    assert np.allclose(em1_table(Pa, np.eye(4)[y]), dai15.table_from(Pa, y))
    pairs = list(itertools.combinations(range(NF), 2))
    assert len(pairs) == 10 and all(tuple(sorted(p)) in pairs for p in itertools.permutations(range(NF), 2))
    assert verdict(0.1, 2.0, 1.2) == "上回る" and verdict(0.1, 1.0, 0.5) == "言えない" and verdict(-3, -0.1, -1.5) == "下回る"
    assert len(set(VERSIONS)) == len(VERSIONS) and all(v in VERSIONS for v in list(MAIN_PRACT.values()) + list(MAIN_ORC.values()) + list(COMMON_OF.values()))
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
                scipy=scipy.__version__, sklearn=sklearn.__version__, python=platform.python_version(), pandas=pd.__version__,
                scripts={n: dai15.sha(HERE / n) for n in SCRIPTS}, design_sha256=dai15.sha(DESIGN), dai1_frozen_sha256=dai3.load_fz1(OUT1)[1],
                KS=KS, NF=NF, D_MIN=D_MIN, EPS=EPS, TAU=TAU, TAUS_REC=TAUS_REC, W_MAX=W_MAX, GATE_C=GATE_C, CHECK_TEACHER=CHECK_TEACHER,
                GRAD_TOL=GRAD_TOL, FLOOR=FLOOR, ASSUME_FLAG=ASSUME_FLAG, VERSIONS=VERSIONS, REFS=REFS, EST=EST, MAIN_PRACT=MAIN_PRACT, MAIN_ORC=MAIN_ORC,
                COMMON_OF=COMMON_OF, TABLE_SPECS=TABLE_SPECS, rng_seed=SEED)
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
