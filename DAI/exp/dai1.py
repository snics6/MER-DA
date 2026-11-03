"""DAI 実験 1：コーパスごとの識別器と，中核の手法（設計は `DAI/docs/実験1_設計.md`．計画は `DAI/docs/計画_2026-11-04.tex`）。

段（--stage）：
  source  ：コーパスごと × 種類（A,V,T,AV,AT,VT,AVT）の識別器を作って保存する（C は塊の 5 分割の交差検証）。
  pseudo  ：評価のターゲット T ごとに，ソース 3 つの 1 つずつを仮のターゲット c にし，残り 2 つの識別器で回す。
            段 1：比較の基準（K 本だけ）を引き戻しの強さの格子で解き，(T, 種類, K) ごとに引き戻しの強さを選ぶ。
            段 2：予測を付ける側を作り，積のモデルのモダリティの指数を (T, K) ごとに選ぶ（候補のラベルでの UAR）。
            段 3：生徒（A,V,T,AVT）の全手法を，格子の全部の点で解く。
  select  ：仮のターゲットの成績で，(T, 生徒, K, 手法) ごとに設定を選び，固定のファイルを書く。
  eval    ：評価のターゲット T を，ソース 3 つで回す（段 1〜3 と同じ．引き戻しの強さとモダリティの指数は select の値）。

保存（保存の契約．計画 8.7 節）：単位（T, c, 分割, 種）ごとに
  `units/<T>_<c>_f<f>_r<r>.npz`：K 本・候補・テストの sample_id，標準化の平均と標準偏差，
    段 1 の全部の問の W・b，段 2 の予測を付ける側の確率（候補とテスト），学び直した各モダリティの識別器の W・b，
    段 3 の全部の問の W・b
  `rows.csv`：全部の問の UAR（テスト）．予測を付ける側の UAR（候補のラベルとテストのラベル）
評価の回の c は "ALL"。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
C_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0)                  # ソースの識別器の C
TETHER = [(cw, cb) for cw in (0.003, 0.01, 0.03, 0.1, 0.3) for cb in (1.0, 10.0)]   # 出発点への引き戻し (C′, C_b)
M_GRID = (100, 500, 2000, "n")                                       # 疑似ラベルの重みの合計（"n" は候補の本数）
A_GRID = (1, 4, 16)                                                  # K 本の倍率
KD_T = (1, 2, 4)                                                     # 温度つき蒸留の温度
AG_TAU = (0.5, 0.7, 0.9)                                             # 一致した発話だけの疑似ラベルの確信度のしきい値
AG_F = (0.25, 1, 4)                                                  # その疑似ラベル 1 本あたりの重み
W_EXP = (0.0, 0.5, 1.0, 2.0)                                         # 積のモデルのモダリティの指数
W_GRID = [w for w in itertools.product(W_EXP, repeat=3) if any(w)]
DEFAULTS = {"joint": {"m": "n", "a": 1}, "rolesep": {"m": "n", "a": 1}, "kd": {"T": 2, "m": "n"}, "agree": {"tau": 0.9, "f": 1}}
KS_ALL = (0,) + DD.KS


def mval(m, n):
    return float(n) if m == "n" else float(m)


def git_commit() -> tuple[str, bool]:
    c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "DAI/exp", "measure", "features"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return c, dirty


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Corpus:
    """1 コーパスの表と特徴（GPU に載せる）。"""
    cache: dict = {}

    def __init__(self, name: str):
        df, feats = DD.load_corpus(name)
        self.name, self.df = name, df
        self.y = df["label_4"].to_numpy().astype(int)
        self.X = {k: S.t(DD.kind_matrix(feats, k)) for k in DD.KINDS}

    @classmethod
    def get(cls, name):
        if name not in cls.cache:
            cls.cache[name] = Corpus(name)
        return cls.cache[name]


# ------------------------------------------------------------------ 段 source

def stage_source(out: Path, corpora=DD.CORPORA) -> None:
    (out / "source").mkdir(parents=True, exist_ok=True)
    rows = []
    for c in corpora:
        cp = Corpus.get(c)
        for kind in DD.KINDS:
            t0 = time.time()
            X = cp.X[kind]
            cv = np.zeros((DD.N_FOLDS, len(C_GRID)))
            for f in range(DD.N_FOLDS):
                tr = np.flatnonzero(DD.pool_mask(cp.df, f)); te = np.flatnonzero(cp.df["fold"].to_numpy() == f)
                mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
                Xs = (X - mu) / sd
                G = S.balanced_onehot(cp.y[tr], np.arange(len(tr)), len(tr))[None].expand(len(C_GRID), -1, -1)
                r = S.fit(Xs[tr], G, torch.zeros(4, X.shape[1]), torch.zeros(4), Cw=np.array(C_GRID), Cb=math.inf)
                cv[f] = S.uar_batch(cp.y[te], S.logits(Xs[te], r["W"], r["b"]))
            mean = cv.mean(0)
            best = int(np.flatnonzero(mean == mean.max())[0])               # 同点なら小さい C（強い正則化）
            mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
            Xs = (X - mu) / sd
            G = S.balanced_onehot(cp.y, np.arange(len(cp.y)), len(cp.y))[None]
            r = S.fit(Xs, G, torch.zeros(4, X.shape[1]), torch.zeros(4), Cw=C_GRID[best], Cb=math.inf)
            np.savez_compressed(out / "source" / f"{c}_{kind}.npz", W=r["W"][0].cpu().numpy(), b=r["b"][0].cpu().numpy(),
                                mu=mu.cpu().numpy(), sd=sd.cpu().numpy(), C=C_GRID[best], cv=cv, C_grid=np.array(C_GRID),
                                sample_id=cp.df["sample_id"].to_numpy().astype(str), grad_max=r["grad_max"])
            rows.append({"corpus": c, "kind": kind, "C": C_GRID[best], "cv_uar": float(mean[best]),
                         **{f"cv_{C}": float(v) for C, v in zip(C_GRID, mean)}, "grad_max": r["grad_max"], "sec": time.time() - t0})
            print(f"source {c} {kind}: C={C_GRID[best]} 交差検証の UAR {mean[best]:.2f} ({time.time()-t0:.0f} 秒)", flush=True)
    sm = out / "source" / "summary.csv"
    old = pd.read_csv(sm) if sm.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old["corpus"].isin(new["corpus"])]
    pd.concat([old, new]).to_csv(sm, index=False)


def load_source(out: Path, corpus: str, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    z = np.load(out / "source" / f"{corpus}_{kind}.npz")
    return S.t(z["W"]), S.t(z["b"])


# ------------------------------------------------------------------ 単位（T, c, 分割, 種）

class Unit:
    """1 つの単位の分割，K 本，標準化した特徴，出発点。"""

    def __init__(self, out: Path, T: str, c: str, srcs: list[str], f: int, r: int):
        self.T, self.c, self.srcs, self.f, self.r = T, c, srcs, f, r
        cp = Corpus.get(c)
        self.cp = cp
        self.pool = np.flatnonzero(DD.pool_mask(cp.df, f))
        self.test = np.flatnonzero(cp.df["fold"].to_numpy() == f)
        self.ypool, self.ytest = cp.y[self.pool], cp.y[self.test]
        korder = DD.k_orders(cp.y, self.pool, c, f, r)
        self.kpos = {0: np.array([], int), **{K: np.searchsorted(self.pool, korder[K]) for K in DD.KS}}
        assert all((self.pool[self.kpos[K]] == korder[K]).all() for K in DD.KS)
        self.Xp, self.Xt, self.stats, self.start = {}, {}, {}, {}
        for kind in DD.KINDS:
            X = cp.X[kind]
            mu, sd = X[self.pool].mean(0), X[self.pool].std(0).clamp(min=1e-6)   # 候補（K 本を含み，テストを含まない）の統計
            self.Xp[kind], self.Xt[kind] = (X[self.pool] - mu) / sd, (X[self.test] - mu) / sd
            self.stats[kind] = (mu.cpu().numpy(), sd.cpu().numpy())
            Ws, bs = zip(*[load_source(out, s, kind) for s in srcs])
            self.start[kind] = (torch.stack(Ws).mean(0), torch.stack(bs).mean(0))   # 点数の等しい重みの平均（8.4 節）
        self.n = len(self.pool)

    def name(self):
        return f"{self.T}_{self.c}_f{self.f}_r{self.r}"

    def gK(self, K, scale=1.0):
        return S.balanced_onehot(self.ypool, self.kpos[K], self.n, scale)


def probs(X, W, b):
    return torch.softmax(S.logits(X, W, b), -1)


# ------------------------------------------------------------------ 段 1：比較の基準と全ラベルの参照

def pass1(u: Unit, store: dict, rows: list, tether_only: dict | None = None) -> None:
    """種類ごとに，K 本だけ（K≥1）と候補の全ラベル（上限の参照）を，引き戻しの強さの格子で解く。
    tether_only を渡すと（評価の回），(種類, K) ごとにその値だけで解く。"""
    for kind in DD.KINDS:
        W0, b0 = u.start[kind]
        Z0 = S.logits(u.Xt[kind], W0, b0)
        rows.append(dict(stage=1, method="start", kind=kind, K=0, uar=S.uar(u.ytest, Z0.argmax(1).cpu().numpy()) * 100))
        items = []
        for K in DD.KS + ("all",):
            grid = TETHER if tether_only is None else [tether_only[(kind, 25 if K == "all" else K)]]
            for cw, cb in grid:
                items.append((K, cw, cb))
        G = torch.stack([S.balanced_onehot(u.ypool, np.arange(u.n), u.n) if K == "all" else u.gK(K) for K, _, _ in items])
        res = S.fit(u.Xp[kind], G, W0, b0, Cw=[x[1] for x in items], Cb=[x[2] for x in items])
        ua = S.uar_batch(u.ytest, S.logits(u.Xt[kind], res["W"], res["b"]))
        for (K, cw, cb), v in zip(items, ua):
            rows.append(dict(stage=1, method="oracle" if K == "all" else "none", kind=kind, K=-1 if K == "all" else K,
                             Cw=cw, Cb=cb, uar=v))
        store[f"p1_{kind}_W"], store[f"p1_{kind}_b"] = res["W"].cpu().numpy(), res["b"].cpu().numpy()
        store[f"p1_{kind}_items"] = np.array([f"{K}|{cw}|{cb}" for K, cw, cb in items])
        rows.append(dict(stage=1, method="_conv", kind=kind, grad_max=res["grad_max"], n_iter=res["n_iter"]))


# ------------------------------------------------------------------ 段 2：予測を付ける側

def none_wb(u: Unit, p1, kind: str, K: int, tether: dict):
    """比較の基準 none[K] の (W, b)．K=0 は出発点．段 1 の保存から，選んだ引き戻しの強さの解を読む。"""
    if K == 0:
        return u.start[kind]
    cw, cb = tether[(kind, K)]
    items = [str(x) for x in p1[f"p1_{kind}_items"]]
    j = items.index(f"{K}|{cw}|{cb}")
    return S.t(p1[f"p1_{kind}_W"][j]), S.t(p1[f"p1_{kind}_b"][j])


def none_probs(u: Unit, p1, K: int, tether: dict) -> dict:
    tk = {}
    for kind in DD.KINDS:
        W, b = none_wb(u, p1, kind, K, tether)
        tk[f"none_{kind}"] = (probs(u.Xp[kind], W, b), probs(u.Xt[kind], W, b))
    return tk


def pass2(u: Unit, p1, store: dict, rows: list, tether: dict) -> None:
    """K ごとに，積のモデルの各モダリティを学び直す（融合 AVT の none[K] の予測を疑似ラベルにし，重みの合計は候補の本数．
    K 本も入れる）．全部の指数の積と各種類の予測の当たり方を，候補のラベル（選ぶのに使う）とテストのラベル（記録だけ）で測る。"""
    for K in KS_ALL:
        tk = none_probs(u, p1, K, tether)
        F = tk["none_AVT"][0]
        for m in "AVT":
            W0, b0 = u.start[m]
            cw, cb = tether[(m, max(K, 1))]
            G = (u.gK(K) if K else torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)) + S.soft_rows(F, float(u.n))
            res = S.fit(u.Xp[m], G[None], W0, b0, Cw=cw, Cb=cb)
            store[f"p2_K{K}_ret_{m}_W"], store[f"p2_K{K}_ret_{m}_b"] = res["W"][0].cpu().numpy(), res["b"][0].cpu().numpy()
            tk[f"ret_{m}"] = (probs(u.Xp[m], res["W"][0], res["b"][0]), probs(u.Xt[m], res["W"][0], res["b"][0]))
            rows.append(dict(stage=2, method="_conv", kind=f"ret_{m}", K=K, grad_max=res["grad_max"], n_iter=res["n_iter"]))
        Lp = torch.stack([tk[f"ret_{m}"][0].clamp(min=1e-12).log() for m in "AVT"])
        Lt = torch.stack([tk[f"ret_{m}"][1].clamp(min=1e-12).log() for m in "AVT"])
        for w in W_GRID:
            wt = S.t(np.array(w))[:, None, None]
            up = S.uar(u.ypool, (wt * Lp).sum(0).argmax(1).cpu().numpy()) * 100
            ut = S.uar(u.ytest, (wt * Lt).sum(0).argmax(1).cpu().numpy()) * 100
            rows.append(dict(stage=2, method="product", teacher=str(w), K=K, uar_pool=up, uar=ut))
        tk["poe_none"] = (poe(tk, "AVT", 0), poe(tk, "AVT", 1))
        for name in [f"none_{k}" for k in DD.KINDS] + [f"ret_{m}" for m in "AVT"] + ["poe_none"]:
            P, Q = tk[name]
            rows.append(dict(stage=2, method="teacher", teacher=name, K=K,
                             uar_pool=S.uar(u.ypool, P.argmax(1).cpu().numpy()) * 100, uar=S.uar(u.ytest, Q.argmax(1).cpu().numpy()) * 100))
            store[f"p2_K{K}_{name}_pool"], store[f"p2_K{K}_{name}_test"] = P.half().cpu().numpy(), Q.half().cpu().numpy()


def build_tk(u: Unit, p1, p2, tether: dict) -> dict:
    """段 1・段 2 の保存（float32 の W・b）から，K ごとの予測を作り直す。"""
    TK = {}
    for K in KS_ALL:
        tk = none_probs(u, p1, K, tether)
        for m in "AVT":
            W, b = S.t(p2[f"p2_K{K}_ret_{m}_W"]), S.t(p2[f"p2_K{K}_ret_{m}_b"])
            tk[f"ret_{m}"] = (probs(u.Xp[m], W, b), probs(u.Xt[m], W, b))
        TK[K] = tk
    return TK


def product(tk: dict, w, which: int):
    L = sum(float(wi) * tk[f"ret_{m}"][which].clamp(min=1e-12).log() for wi, m in zip(w, "AVT"))
    return torch.softmax(L, -1)


def poe(tk: dict, mods: str, which: int, src="none"):
    L = sum(tk[f"{src}_{m}"][which].clamp(min=1e-12).log() for m in mods)
    return torch.softmax(L, -1)


# ------------------------------------------------------------------ 段 3：生徒

def pass3(u: Unit, store: dict, rows: list, tether: dict, TK: dict, wsel: dict) -> None:
    for s in DD.STUDENTS:
        X, Xt = u.Xp[s], u.Xt[s]
        W0, b0 = u.start[s]
        for K in KS_ALL:
            tk = TK[K]
            cw, cb = tether[(s, max(K, 1))]
            gk = u.gK(K) if K else torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
            teach = {"sel": product(tk, wsel[K], 0), "one": product(tk, (1, 1, 1), 0)}
            items, Gs = [], []
            # 提案（同時に解く版）
            for tv, Q in teach.items():
                for m in M_GRID:
                    for a in A_GRID:
                        items.append(dict(method="joint", teacher=tv, m=m, a=a))
                        Gs.append(a * gk + S.soft_rows(Q, mval(m, u.n)))
            # 温度つき蒸留（同じ予測．K 本の倍率 1）
            for Tt in KD_T:
                Qt = torch.softmax(teach["sel"].clamp(min=1e-12).log() / Tt, -1)
                for m in M_GRID:
                    items.append(dict(method="kd", teacher="sel", T=Tt, m=m))
                    Gs.append(gk + S.soft_rows(Qt, mval(m, u.n), balance=False))
            # 一致した発話だけの疑似ラベル（予測と，生徒自身の none[K] の判定が一致し，確信度が τ 以上）
            own = tk[f"none_{s}"][0].argmax(1)
            Q = teach["sel"]
            for tau in AG_TAU:
                keep = (Q.argmax(1) == own) & (Q.max(1).values >= tau)
                H = torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
                H[keep, Q.argmax(1)[keep]] = 1.0
                for fw in AG_F:
                    items.append(dict(method="agree", teacher="sel", tau=tau, f=fw, n_kept=int(keep.sum())))
                    Gs.append(gk + fw * H)
            # 予測を付ける側の種類（選択なしの値 m = n，a = 1．記述の比べ）
            tts = {f"single_{m}": tk[f"none_{m}"][0] for m in "AVT" if m != s or s == "AVT"}
            tts.update({f"pair_{k}": tk[f"none_{k}"][0] for k in ("AV", "AT", "VT")})
            tts["fused_AVT"] = tk["none_AVT"][0]
            tts["poe_none"] = poe(tk, "AVT", 0)
            tts.update({f"ret_{m}": tk[f"ret_{m}"][0] for m in "AVT"})
            for name, Qx in tts.items():
                items.append(dict(method="tt", teacher=name, m="n", a=1))
                Gs.append(gk + S.soft_rows(Qx, float(u.n)))
            G = torch.stack(Gs)
            res = S.fit(X, G, W0, b0, Cw=cw, Cb=cb)
            ua = S.uar_batch(u.ytest, S.logits(Xt, res["W"], res["b"]))
            for it, v in zip(items, ua):
                rows.append(dict(stage=3, student=s, K=K, uar=v, **it))
            store[f"p3_{s}_K{K}_W"], store[f"p3_{s}_K{K}_b"] = res["W"].cpu().numpy(), res["b"].cpu().numpy()
            store[f"p3_{s}_K{K}_items"] = np.array([json.dumps(it, sort_keys=True) for it in items])
            rows.append(dict(stage=3, method="_conv", student=s, K=K, grad_max=res["grad_max"], n_iter=res["n_iter"]))
            # 役割を分けた版：段 a は疑似ラベルだけ（K 本なし），段 b は W を固定して K 本で切片だけ
            itA, GA = [], []
            for tv, Qx in teach.items():
                for m in M_GRID:
                    itA.append(dict(teacher=tv, m=m)); GA.append(S.soft_rows(Qx, mval(m, u.n)))
            ra = S.fit(X, torch.stack(GA), W0, b0, Cw=cw, Cb=cb)
            store[f"p3r_{s}_K{K}_Wa"], store[f"p3r_{s}_K{K}_ba"] = ra["W"].cpu().numpy(), ra["b"].cpu().numpy()
            if K == 0:
                ua = S.uar_batch(u.ytest, S.logits(Xt, ra["W"], ra["b"]))
                for it, v in zip(itA, ua):
                    for a in A_GRID:
                        rows.append(dict(stage=3, student=s, K=K, method="rolesep", a=a, uar=v, **it))
                continue
            itB, WB, bB, GB = [], [], [], []
            for j, it in enumerate(itA):
                for a in A_GRID:
                    itB.append(dict(**it, a=a)); WB.append(ra["W"][j]); bB.append(ra["b"][j]); GB.append(a * gk)
            rb = S.fit(X, torch.stack(GB), torch.stack(WB), torch.stack(bB), Cw=math.inf, Cb=cb, fix_W=True)
            ua = S.uar_batch(u.ytest, S.logits(Xt, rb["W"], rb["b"]))
            for it, v in zip(itB, ua):
                rows.append(dict(stage=3, student=s, K=K, method="rolesep", uar=v, **it))
            store[f"p3r_{s}_K{K}_bb"] = rb["b"].cpu().numpy()
            rows.append(dict(stage=3, method="_conv_rolesep", student=s, K=K, grad_max=max(ra["grad_max"], rb["grad_max"])))


# ------------------------------------------------------------------ 進め方

def units_for(T: str, mode: str, folds, seeds):
    srcs = DD.others(T)
    if mode == "pseudo":
        for c in srcs:
            for f in folds:
                for r in seeds:
                    yield c, [s for s in srcs if s != c], f, r
    else:
        for f in folds:
            for r in seeds:
                yield T, srcs, f, r


def run_pass(out, tag, T, c, srcs, f, r, which, tether=None, wsel=None) -> pd.DataFrame:
    """1 単位の 1 つの段を回して保存する．保存があれば読むだけ（途中から再開できる）。"""
    cname = "ALL" if tag == "eval" else c
    base = out / tag / "units" / f"{T}_{cname}_f{f}_r{r}"
    path, rpath = Path(f"{base}_p{which}.npz"), Path(f"{base}_p{which}.csv")
    if path.exists() and rpath.exists():
        return pd.read_csv(rpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    u = Unit(out, T, c, srcs, f, r)
    store, rows = {}, []
    t0 = time.time()
    if which == 1:
        pass1(u, store, rows, tether_only=tether if tag == "eval" else None)
        store.update(sample_pool=u.cp.df["sample_id"].to_numpy()[u.pool].astype(str),
                     sample_test=u.cp.df["sample_id"].to_numpy()[u.test].astype(str),
                     **{f"sample_K{K}": u.cp.df["sample_id"].to_numpy()[u.pool[u.kpos[K]]].astype(str) for K in DD.KS},
                     **{f"stat_{k}_mu": v[0] for k, v in u.stats.items()}, **{f"stat_{k}_sd": v[1] for k, v in u.stats.items()})
    elif which == 2:
        p1 = np.load(f"{base}_p1.npz")
        pass2(u, p1, store, rows, tether)
        store["tether"] = np.array([f"{k[0]}|{k[1]}|{v[0]}|{v[1]}" for k, v in tether.items()])
    else:
        p1, p2 = np.load(f"{base}_p1.npz"), np.load(f"{base}_p2.npz")
        pass3(u, store, rows, tether, build_tk(u, p1, p2, tether), wsel)
        store["wsel"] = np.array([f"{K}|{w}" for K, w in wsel.items()])
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in store.items()})
    df = pd.DataFrame(rows).assign(T=T, c=cname, fold=f, seed=r, sec=time.time() - t0)
    df.to_csv(rpath, index=False)
    print(f"{tag} {base.name} 段 {which}: {time.time()-t0:.0f} 秒", flush=True)
    return df


def select_tether(rows: pd.DataFrame) -> dict:
    """(種類, K) ごとに，比較の基準の仮のターゲットの UAR の平均が最大の (C′, C_b)．同点は小さい C′・小さい C_b。"""
    out = {}
    d = rows[(rows.stage == 1) & (rows.method == "none")]
    for (kind, K), g in d.groupby(["kind", "K"]):
        m = g.groupby(["Cw", "Cb"])["uar"].mean().reset_index().sort_values(["uar", "Cw", "Cb"], ascending=[False, True, True])
        out[(kind, int(K))] = (float(m.iloc[0].Cw), float(m.iloc[0].Cb))
    return out


def select_w(rows: pd.DataFrame) -> dict:
    """K ごとに，積のモデルの候補のラベルでの UAR の平均が最大の指数．同点は (1,1,1) に近い方。"""
    d = rows[(rows.stage == 2) & (rows.method == "product")]
    out = {}
    for K, g in d.groupby("K"):
        m = g.groupby("teacher")["uar_pool"].mean()
        best = m.max()
        cands = [eval(w) for w in m.index[m >= best - 1e-9]]
        out[int(K)] = min(cands, key=lambda w: sum(abs(x - 1) for x in w))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("source", "pseudo", "select", "eval", "selftest"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dai1"))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(DD.N_FOLDS)))
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="既定は pseudo で 0 1，eval で 0〜9")
    ap.add_argument("--pass", dest="passes", default="all", choices=("1", "23", "all"))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_commit()
    if dirty and not args.allow_dirty and args.stage not in ("selftest",):
        sys.exit("DAI/exp・measure・features に未コミットの変更がある（試しは --allow-dirty）")
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, host=platform.node(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
                script_sha256=sha(Path(__file__)), solver_sha256=sha(HERE / "dai_solver.py"), data_sha256=sha(HERE / "dai_data.py"),
                time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")

    if args.stage == "selftest":
        S.selftest(); return
    if args.stage == "source":
        stage_source(out, args.targets); return

    tag = "pseudo" if args.stage == "pseudo" else ("eval" if args.stage == "eval" else "select")
    seeds = args.seeds if args.seeds is not None else ([0, 1] if args.stage == "pseudo" else list(range(10)))
    if args.stage == "pseudo":
        for T in args.targets:
            us = list(units_for(T, "pseudo", args.folds, seeds))
            if args.passes in ("1", "all"):
                for c, s, f, r in us:
                    run_pass(out, "pseudo", T, c, s, f, r, 1)
            if args.passes in ("23", "all"):
                r1 = pd.concat([pd.read_csv(out / "pseudo" / "units" / f"{T}_{c}_f{f}_r{r}_p1.csv") for c, _, f, r in us])
                teth = select_tether(r1)
                r2 = pd.concat([run_pass(out, "pseudo", T, c, s, f, r, 2, tether=teth) for c, s, f, r in us])
                w = select_w(r2)
                json.dump({"tether": {f"{k[0]}|{k[1]}": v for k, v in teth.items()}, "w": {str(k): v for k, v in w.items()},
                           "units": [f"{c}_f{f}_r{r}" for c, _, f, r in us]},
                          open(out / "pseudo" / f"{T}_tether_w.json", "w"), indent=1)
                for c, s, f, r in us:
                    run_pass(out, "pseudo", T, c, s, f, r, 3, tether=teth, wsel=w)
        return
    if args.stage == "select":
        import dai1_select
        dai1_select.main(out); return
    if args.stage == "eval":
        sel = json.load(open(out / "select" / "frozen.json"))
        for T in args.targets:
            teth = {tuple([k.split("|")[0], int(k.split("|")[1])]): tuple(v) for k, v in sel[T]["tether"].items()}
            w = {int(k): tuple(v) for k, v in sel[T]["w"].items()}
            for c, s, f, r in units_for(T, "eval", args.folds, seeds):
                for which in (1, 2, 3):
                    run_pass(out, "eval", T, c, s, f, r, which, tether=teth, wsel=w)

if __name__ == "__main__":
    main()
