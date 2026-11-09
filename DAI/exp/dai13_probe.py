"""DAI-13 の下調べ（音声の第 1 段）：凍結した WavLM-large の 25 個の隠れ状態の重みつき和の重みを，ほかの 2 つのモダリティの予測
（`pair`）を目標にラベル無しで学び，最終層だけより出所の予測が読めるようになるか（設計 `DAI/docs/実験13_下調べ_設計.md`）．

段：
  --stage units      単位ごとの計算（既定は種 r = 0 の 60 単位．`--units` で絞る）
  --stage summarize  集計（差と区間，規則の数え上げ）
  --stage selftest   人工データの自己検査

DAI-12 の台本 `dai12_probe.py` の部品（半分の分け方，束ねたネットの学習 `crossfit_nets`，候補の上の区間 `effect_stats_pool`，集計の道具）と，
DAI-11 の `Unit2`・`none_wb`・`representability` をそのまま使う．最終層だけの腕 `final` は DAI-12 の `lince` と同じ計算である．
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
import scipy
import sklearn
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai11_probe as P11  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = HERE.parents[1]
UD = P11.UD
FROZEN = P11.FROZEN
D11 = P12.D11
D12 = ROOT / "results" / "dai12_probe"
LAYER_KIND = "audio_wavlm_layers"
N_LAYERS = 25
FINAL_IDX = 24
M = "A"                                                                  # 学び直す分類器は音声だけ
KS_ALL, KS_RULE = P12.KS_ALL, P12.KS_RULE
TEACHERS = ("pair", "pair_shuf", "self")                                 # 主，入れ替えの対照，手順の対照
N_UNITS_R0 = 60
SEED = 0
WDS, CKPTS, STEPS, LRS = P12.WDS, P12.CKPTS, P12.STEPS, P12.LRS
FIXED_WD, FIXED_STEP, FIXED_LR = P12.FIXED_WD, P12.FIXED_STEP, P12.FIXED_LR
LAB_CKPTS, LAB_STEPS = P12.LAB_CKPTS, P12.LAB_STEPS
CONV_REL = P12.CONV_REL
READ_WDS, READ_STEPS = (1e-2, 1e-1, 1.0), 500                           # Q4：K 本の 4×4 の読み出し（固定の格子．選ばない）
KSHOT_WDS, KSHOT_STEPS = (1e-2, 1.0), 500                                # Q5：層の重みと線形の写像を K 本だけで学ぶ（固定の格子．選ばない．設計 3.5b 節）
ALPHA_LR = 1e-1                                                          # 層の重みの対数 α の Adam の学習率（W・c は FIXED_LR）．10⁻² だと 500 ステップで 1 つの層に寄り切れない（gate-probe 第 1 回の 1）
PSI_EPS = 1e-6                                                           # ψ の次元ごとの標準化の下駄（学習の半分の統計．α が one-hot のとき final と同じ尺度になる）
LAYER_DIFF_STOP = 1e-2                                                   # 添字 24 の生の特徴と既存の音声の特徴の最大の差の許し
FINAL_MISMATCH_STOP = 0.25                                               # final と DAI-12 の lince の判定の不一致の割合の許し（pair・self だけ．乱数の列が違うので 0 ではない．a2）
FINAL_UAR_STOP = 5.0                                                     # 同じく候補の上の UAR の差の許し（pair・self だけ）
MISMATCH_FRAC = P12.MISMATCH_FRAC
FLOOR_POOL = P12.FLOOR_POOL
SELF_TOL = P12.SELF_TOL
ALPHA_MAX_NOTE, ALPHA_FINAL_NOTE = 0.9, 0.5                              # 層の重みの集中の添え書きの目安（設計 3.2 節）
ALPHA_FAR = 0.2                                                          # 学んだ α が一様（1/25 = 0.04）から離れたとみなす最大の重みの下限（5 倍．見込み (f) を読む条件．review-ml 第 1 回の M2）
LOSS_NOTE_FRAC = 0.05                                                    # 入れ子の前の lw の損失が final を上回った割合がこれを超えたら R1 に「最適化の不足の疑い」を添える（m7）
N_BOOT = ST.N_BOOT

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ------------------------------------------------------------------ 全層の特徴

class LayerFeats:
    """コーパスの全層の特徴（CPU に置く）．`sample_id` で行を引く。"""
    cache: dict = {}

    def __init__(self, corpus: str):
        p = ROOT / "data" / "cache" / f"{LAYER_KIND}_{corpus}.npz"
        if not p.exists():
            raise SystemExit(f"{p} が無い（features/build_audio_layers.py で作る）")
        z = np.load(p, allow_pickle=False)
        self.X = z["X_layers"]
        if self.X.ndim != 3 or self.X.shape[1] != N_LAYERS:
            raise SystemExit(f"{p}：X_layers の形が {self.X.shape}（(n, {N_LAYERS}, d) のはず）")
        self.ok = z["ok"].astype(bool)
        self.pos = {s: i for i, s in enumerate(z["sample_id"].astype(str))}
        self.sha = P12.sha(p)

    @classmethod
    def get(cls, corpus: str):
        if corpus not in cls.cache:
            cls.cache[corpus] = LayerFeats(corpus)
        return cls.cache[corpus]

    def rows(self, sids: np.ndarray) -> np.ndarray:
        miss = [s for s in sids if s not in self.pos]
        if miss:
            raise SystemExit(f"全層の特徴に無い発話：{miss[:5]}（{len(miss)} 本）")
        ix = np.array([self.pos[s] for s in sids])
        if not self.ok[ix].all():
            raise SystemExit(f"全層の特徴の ok が偽の発話が {int((~self.ok[ix]).sum())} 本ある")
        X = self.X[ix]
        if not np.isfinite(X).all():
            raise SystemExit("全層の特徴に有限でない値がある")
        return X


# ------------------------------------------------------------------ 層の重みつき和のネット（束ねて学ぶ）

def _psi(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """X (n, L, d)，A (B, L)（softmax 後の層の重み）→ 重みつき和 ψ (B, n, d)。"""
    return torch.einsum("nld,bl->bnd", X, A)


def _psi_stats(Xstat: torch.Tensor, A: torch.Tensor) -> tuple:
    """統計を取る行 Xstat の上の ψ の次元ごとの平均と標準偏差 (B, 1, d)（設計 3.2 節．α が one-hot なら標準化した層そのものなので ≈ (0, 1)）。"""
    p = _psi(Xstat, A)
    return p.mean(1, keepdim=True), p.std(1, keepdim=True).clamp(min=PSI_EPS)


def _lw_forward(X: torch.Tensor, A: torch.Tensor, W: torch.Tensor, c: torch.Tensor, stats: tuple) -> torch.Tensor:
    """ψ を stats（学習の半分の統計）で次元ごとに標準化してから線形の写像に通す → 点数 (B, n, 4)。"""
    mu, sd = stats
    return torch.bmm((_psi(X, A) - mu) / sd, W) + c[:, None, :]


def train_lw_batched(Xtr, Qtr, Xte, Qte, Xtest, wds: tuple, seed: int, learn_alpha: bool, alpha_fixed, steps: int, ckpts: tuple, lr: float = FIXED_LR) -> dict:
    """目標 J 個 × 罰則 len(wds) 個 = B 個のネットを一括で学ぶ．α（層の重みの対数）は初期値 0（一様），学習率 ALPHA_LR，罰しない．W だけ罰する．
    ψ は学習の半分の統計で次元ごとに標準化する（α が one-hot のとき final と同じ尺度．D19）．
    alpha_fixed：(J, L) の softmax 後の重み（学ばないとき）。予測する半分 Xte の損失（Qte）が最小の保存点を残す。"""
    J, nw = Qtr.shape[0], len(wds)
    B = J * nw
    L, d = Xtr.shape[1], Xtr.shape[2]
    Qtr_b, Qte_b = Qtr.repeat_interleave(nw, 0), Qte.repeat_interleave(nw, 0)
    wd = torch.tensor(wds, dtype=torch.float32, device=S.DEV).repeat(J)
    g = torch.Generator(device=S.DEV).manual_seed(seed)
    b1 = 1 / np.sqrt(d)
    W = ((torch.rand(B, d, 4, generator=g, device=S.DEV) * 2 - 1) * b1).requires_grad_(True)
    c = ((torch.rand(B, 4, generator=g, device=S.DEV) * 2 - 1) * b1).requires_grad_(True)
    if learn_alpha:
        alpha = torch.zeros(B, L, device=S.DEV, requires_grad=True)
        opt = torch.optim.Adam([{"params": [alpha], "lr": ALPHA_LR}, {"params": [W, c], "lr": lr}])
        A_fixed = None
    else:
        A_fixed = alpha_fixed.to(S.DEV).float().repeat_interleave(nw, 0)
        opt = torch.optim.Adam([W, c], lr=lr)
    ce_best = torch.full((B,), float("inf"), device=S.DEV)
    step_best = torch.zeros(B, dtype=torch.long, device=S.DEV)
    te_best = torch.zeros(B, Xte.shape[0], 4, device=S.DEV)
    test_best = torch.zeros(B, Xtest.shape[0], 4, device=S.DEV)
    alpha_best = torch.zeros(B, L, device=S.DEV)
    te_at, test_at, alpha_at, ce_hist = {}, {}, {}, []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        A = torch.softmax(alpha, -1) if learn_alpha else A_fixed
        st = _psi_stats(Xtr, A)
        Ltr = _lw_forward(Xtr, A, W, c, st)
        loss_b = P12._ce(Qtr_b, Ltr) + wd / 2 * (W ** 2).sum((1, 2))
        loss_b.sum().backward()
        opt.step()
        if step in ckpts:
            with torch.no_grad():
                A = torch.softmax(alpha, -1) if learn_alpha else A_fixed
                st = _psi_stats(Xtr, A)
                Lte, Ltest = _lw_forward(Xte, A, W, c, st), _lw_forward(Xtest, A, W, c, st)
                ce_te = P12._ce(Qte_b, Lte)
                better = ce_te < ce_best
                ce_best = torch.where(better, ce_te, ce_best)
                step_best = torch.where(better, torch.full_like(step_best, step), step_best)
                te_best[better] = Lte[better]; test_best[better] = Ltest[better]; alpha_best[better] = A[better]
                te_at[step] = Lte.clone(); test_at[step] = Ltest.clone(); alpha_at[step] = A.clone()
                ce_hist.append(ce_te.clone())
    ce_hist = torch.stack(ce_hist, 1)
    if not bool(torch.isfinite(ce_hist).all()):
        raise SystemExit("層の重みつき和のネットの損失に有限でない値がある")
    return dict(te_best=te_best, test_best=test_best, ce_best=ce_best, step_best=step_best, te_at=te_at, test_at=test_at, alpha_at=alpha_at,
                alpha_best=alpha_best, ce_hist=ce_hist, train_loss_final=loss_b.detach())


def crossfit_lw(X, Q, half: np.ndarray, Xtest, seed: int, learn_alpha: bool = True, alpha_fixed=None, wds: tuple = WDS, ckpts: tuple = CKPTS, steps: int = STEPS) -> dict:
    """各半分を，もう一方の半分で学んだネットで予測する（交差の予測）．半分ごと・目標ごとに，予測する半分の損失が最小の (wd, 保存点) を選ぶ．
    X (n, L, d)．alpha_fixed：(2, J, L)（半分ごと）。戻りは dai12_probe.crossfit_nets と同じ鍵に，alpha_sel (2, J, L)，alpha_fix (2, J, L) を足したもの。"""
    J, n = Q.shape[0], Q.shape[1]
    nw, nck = len(wds), len(ckpts)
    iw_f, ick_f = wds.index(FIXED_WD), list(ckpts).index(FIXED_STEP)
    L = X.shape[1]
    sel = torch.zeros(J, n, 4, device=S.DEV); fix = torch.zeros(J, n, 4, device=S.DEV)
    sel_test = torch.zeros(2, J, Xtest.shape[0], 4, device=S.DEV); fix_test = torch.zeros_like(sel_test)
    grid_pred = np.zeros((1, nw, nck, J, n), np.int8)
    cfg, ce_sel, ce_fix = {}, np.zeros((2, J)), np.zeros((2, J))
    ce_hist, tl = np.zeros((2, J, 1, nw, nck)), np.zeros((2, J, 1, nw))
    alpha_sel, alpha_fix = np.zeros((2, J, L), np.float32), np.zeros((2, J, L), np.float32)
    for h in (0, 1):
        tr = torch.as_tensor(np.flatnonzero(half != h), device=S.DEV); te = torch.as_tensor(np.flatnonzero(half == h), device=S.DEV)
        te_np = te.cpu().numpy()
        res = train_lw_batched(X[tr], Q[:, tr], X[te], Q[:, te], Xtest, wds, seed * 4 + h * 2, learn_alpha,
                               None if alpha_fixed is None else alpha_fixed[h], steps, ckpts)
        ce_all = res["ce_best"].view(J, nw)
        iw_best = ce_all.argmin(1).cpu().numpy()
        for ic, step in enumerate(ckpts):
            grid_pred[0, :, ic][:, :, te_np] = res["te_at"][step].argmax(-1).to(torch.int8).view(J, nw, -1).permute(1, 0, 2).cpu().numpy()
        ce_hist[h, :, 0] = res["ce_hist"].view(J, nw, nck).cpu().numpy()
        tl[h, :, 0] = res["train_loss_final"].view(J, nw).cpu().numpy()
        rows_fix = torch.arange(J, device=S.DEV) * nw + iw_f
        fix[:, te] = res["te_at"][FIXED_STEP][rows_fix]; fix_test[h] = res["test_at"][FIXED_STEP][rows_fix]
        ce_fix[h] = res["ce_hist"].view(J, nw, nck)[:, iw_f, ick_f].cpu().numpy()
        alpha_fix[h] = res["alpha_at"][FIXED_STEP][rows_fix].cpu().numpy()
        for j in range(J):
            b = j * nw + int(iw_best[j])
            sel[j, te] = res["te_best"][b]; sel_test[h, j] = res["test_best"][b]
            ce_sel[h, j] = float(res["ce_best"][b])
            cfg[(h, j)] = (float(FIXED_LR), float(wds[int(iw_best[j])]), int(res["step_best"][b]))
            alpha_sel[h, j] = res["alpha_best"][b].cpu().numpy()
        del res
    return dict(sel=sel, fix=fix, sel_test=sel_test, fix_test=fix_test, grid_pred=grid_pred, cfg=cfg, ce_sel=ce_sel, ce_fix=ce_fix,
                ce_hist=ce_hist, train_loss=tl, alpha_sel=alpha_sel, alpha_fix=alpha_fix)


def readout_kshot(logp: torch.Tensor, y: np.ndarray, kpos: np.ndarray, wds: tuple = READ_WDS, steps: int = READ_STEPS, lr: float = 1e-2) -> np.ndarray:
    """Q4：K 本の行だけで 4×4 の読み出し M（初期値 I）と切片 b を学ぶ（釣り合いの重みつき交差エントロピー ＋ (wd/2)‖M − I‖²）．
    wd ごとに全候補の判定 (len(wds), n) int8 を返す（選ばない）。"""
    B = len(wds)
    I4 = torch.eye(4, device=S.DEV)
    Mx = I4.repeat(B, 1, 1).clone().requires_grad_(True)
    bb = torch.zeros(B, 4, device=S.DEV, requires_grad=True)
    wd = torch.tensor(wds, dtype=torch.float32, device=S.DEV)
    rows = torch.as_tensor(kpos, device=S.DEV)
    yk = y[kpos]
    cnt = np.bincount(yk, minlength=4).astype(float)
    w = torch.as_tensor(len(kpos) / (4 * np.maximum(cnt[yk], 1)), device=S.DEV, dtype=torch.float32)
    yk_t = torch.as_tensor(yk, device=S.DEV)
    Xk = logp[rows].float()
    opt = torch.optim.Adam([Mx, bb], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        Z = torch.einsum("nk,bjk->bnj", Xk, Mx) + bb[:, None, :]
        nll = -(torch.log_softmax(Z, -1)[:, torch.arange(len(yk)), yk_t] * w).sum(1) / w.sum()
        loss = nll + wd / 2 * ((Mx - I4) ** 2).sum((1, 2))
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        Z = torch.einsum("nk,bjk->bnj", logp.float(), Mx) + bb[:, None, :]
    return Z.argmax(-1).to(torch.int8).cpu().numpy()


def kshot_lw(Xp: torch.Tensor, y: np.ndarray, kpos: np.ndarray, learn_alpha: bool, seed: int, wds: tuple = KSHOT_WDS, steps: int = KSHOT_STEPS, lr: float = FIXED_LR):
    """Q5：K 本の行だけで，層の重み α（learn_alpha が偽なら添字 24 の one-hot に固定）と線形の写像 W，c を学ぶ（釣り合いの重みつき交差エントロピー ＋ (wd/2)‖W‖²）．
    wd ごとに全候補の判定 (len(wds), n) int8 と，softmax 後の層の重み (len(wds), L) を返す（選ばない）。"""
    B = len(wds)
    L, d = Xp.shape[1], Xp.shape[2]
    g = torch.Generator(device=S.DEV).manual_seed(seed)
    b1 = 1 / np.sqrt(d)
    W = ((torch.rand(B, d, 4, generator=g, device=S.DEV) * 2 - 1) * b1).requires_grad_(True)
    c = ((torch.rand(B, 4, generator=g, device=S.DEV) * 2 - 1) * b1).requires_grad_(True)
    wd = torch.tensor(wds, dtype=torch.float32, device=S.DEV)
    if learn_alpha:
        alpha = torch.zeros(B, L, device=S.DEV, requires_grad=True)
        params = [alpha, W, c]
    else:
        A_fixed = torch.zeros(B, L, device=S.DEV); A_fixed[:, FINAL_IDX] = 1.0
        params = [W, c]
    rows = torch.as_tensor(kpos, device=S.DEV)
    Xk = Xp[rows]
    Q = P12.balanced_weights(y, kpos, Xp.shape[0])[rows][None].expand(B, -1, -1).contiguous()     # (B, nk, 4)
    opt = torch.optim.Adam([{"params": [alpha], "lr": ALPHA_LR}, {"params": [W, c], "lr": lr}]) if learn_alpha else torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        A = torch.softmax(alpha, -1) if learn_alpha else A_fixed
        st = _psi_stats(Xp, A)                                                                      # ψ の統計は候補の全発話（ラベルを使わない．K 本では標準偏差が立たない）
        loss = P12._ce(Q, _lw_forward(Xk, A, W, c, st)) + wd / 2 * (W ** 2).sum((1, 2))
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        A = torch.softmax(alpha, -1) if learn_alpha else A_fixed
        pred = _lw_forward(Xp, A, W, c, _psi_stats(Xp, A)).argmax(-1).to(torch.int8).cpu().numpy()
    return pred, A.cpu().numpy()


def nest_final(lw: dict, fin: dict, half: np.ndarray) -> np.ndarray:
    """入れ子の選択（gate-probe 第 1 回の 1）：半分 h・目標 j ごとに，final の選んだ版の損失が lw の選んだ版の損失以下なら，
    lw の選んだ版（候補の点数，テストの点数，損失，選んだ点，層の重み）を final のもの（層の重みは添字 24 の one-hot）で置き換える．
    戻り：used_layers (2, J) 真偽（層の重みを学んだ版が選ばれたか）。固定の版と格子の全点は変えない。"""
    J, L = lw["sel"].shape[0], lw["alpha_sel"].shape[-1]
    used = np.ones((2, J), bool)
    onehot = np.zeros(L, np.float32); onehot[FINAL_IDX] = 1.0
    for h in (0, 1):
        te = torch.as_tensor(np.flatnonzero(half == h), device=S.DEV)
        for j in range(J):
            if fin["ce_sel"][h, j] <= lw["ce_sel"][h, j]:
                used[h, j] = False
                lw["sel"][j, te] = fin["sel"][j, te]
                lw["sel_test"][h, j] = fin["sel_test"][h, j]
                lw["ce_sel"][h, j] = fin["ce_sel"][h, j]
                lw["cfg"][(h, j)] = fin["cfg"][(h, j)]
                lw["alpha_sel"][h, j] = onehot
    lw["used_layers"] = used
    return used


def psi_var(Xp: torch.Tensor, a: np.ndarray) -> float:
    """層の重み a（softmax 後）での ψ の，標準化する前の次元ごとの分散の平均（候補の全発話．D19 の記録）。"""
    with torch.no_grad():
        p = _psi(Xp, torch.as_tensor(a, device=S.DEV, dtype=torch.float32)[None])[0]
        return float(p.var(0).mean())


def alpha_stats(a: np.ndarray) -> dict:
    """softmax 後の層の重み (L,) → 最大の重み，その層，添字 24 の重み，エントロピー（nats）。"""
    a = np.asarray(a, float)
    return dict(alpha_max=float(a.max()), alpha_argmax=int(a.argmax()), alpha_final=float(a[FINAL_IDX]),
                alpha_entropy=float(-(a * np.log(np.maximum(a, 1e-12))).sum()), alpha_mean_layer=float((a * np.arange(len(a))).sum()))


# ------------------------------------------------------------------ 単位

def run_unit(p3: Path, out: Path, fz: dict, commit: str, commit11: str, commit12: str) -> dict:
    name, T, c, f, r = P11.unit_parts(p3)
    rows_dir = out / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    for p in (rows_dir / f"{name}.npz", rows_dir / f"{name}_lw.csv", rows_dir / f"{name}_lab.csv", rows_dir / f"{name}_layers.csv"):
        if p.exists():
            raise SystemExit(f"{p} は既にある（重複を作らない）")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    srcs = [x for x in DD.CORPORA if x not in (T, c)]
    z1, z2, z3 = P11.load_unit_npz(name, fz, T)
    p12 = D12 / "rows" / f"{name}.npz"
    if not p12.exists():
        raise SystemExit(f"DAI-12 の保存 {p12} が無い")
    z12 = np.load(p12, allow_pickle=False)
    if str(z12["commit"]) != commit12:
        raise SystemExit(f"{name}：DAI-12 の保存のコミット {z12['commit']} が provenance.json の {commit12} と違う")
    if str(z12["commit11"]) != commit11:
        raise SystemExit(f"{name}：DAI-12 の保存の DAI-11 のコミットが違う")
    tether = P11.parse_tether(z3["tether"])
    u = P11.Unit2(D11 / "enc2", T, c, srcs, f, r)
    n, nt = u.n, len(u.test)
    if int(z12["n_pool"]) != n or len(z12["test_y"]) != nt:
        raise SystemExit(f"{name}：候補・テストの本数が DAI-12 の保存と違う")
    tol_n = max(1, int(MISMATCH_FRAC * n))
    store, mism, conv = {}, {}, {}

    def check(key: str, k: int, tol: int, what: str):
        mism[key] = k
        if k > tol:
            raise SystemExit(f"{name} {key}：{what}が保存と {k} 本違う")

    # --- 全層の特徴（候補とテストの行を sample_id で引く）．添字 24 の生の特徴を既存の音声の特徴と照合
    sid_pool = u.cp.df["sample_id"].to_numpy()[u.pool].astype(str)
    sid_test = u.cp.df["sample_id"].to_numpy()[u.test].astype(str)
    if not (sid_pool == z12["pool_sample_id"].astype(str)).all() or not (sid_test == z12["test_sample_id"].astype(str)).all():
        raise SystemExit(f"{name}：候補・テストの sample_id の並びが DAI-12 の保存と違う")
    LF = LayerFeats.get(c)
    Xl_pool = torch.as_tensor(LF.rows(sid_pool), device=S.DEV)                               # (n, L, d) float32
    Xl_test = torch.as_tensor(LF.rows(sid_test), device=S.DEV)
    raw_A = u.cp.X[M]
    diff24 = float((Xl_pool[:, FINAL_IDX] - raw_A[torch.as_tensor(u.pool, device=S.DEV)]).abs().max())
    mism["layer24_raw_max_absdiff"] = diff24
    if diff24 > LAYER_DIFF_STOP:
        raise SystemExit(f"{name}：添字 24 の生の特徴と既存の音声の特徴の最大の差が {diff24:.3e}（{LAYER_DIFF_STOP} を超える）")
    mu, sd = Xl_pool.mean(0), Xl_pool.std(0).clamp(min=1e-6)                                  # 層ごと・次元ごと（候補の統計）
    Xp, Xt = (Xl_pool - mu) / sd, (Xl_test - mu) / sd
    del Xl_pool, Xl_test
    X24, X24t = Xp[:, FINAL_IDX].contiguous(), Xt[:, FINAL_IDX].contiguous()
    mism["layer24_std_max_absdiff"] = float((X24 - u.Xp[M]).abs().max())                     # 標準化した添字 24 と Unit2 の音声の特徴の差（記録）

    # --- 目標（DAI-12 の 3.1 節と同じ）
    none_p = {}
    for m in P12.MODS:
        for K in KS_ALL:
            W, b = P11.none_wb(u, z1, m, K, tether)
            none_p[(m, K)] = P11.probs(u.Xp[m], W, b)
    for K in KS_ALL:
        pool_none = P12.argmax8(none_p[(M, K)])
        check(f"K{K}_pool_none", int((pool_none != z12[f"pool_none_K{K}_{M}"].astype(np.int8)).sum()), tol_n, "none の候補の判定")
        store[f"pool_none_K{K}_{M}"] = pool_none
        W, b = P11.none_wb(u, z1, M, K, tether)
        store[f"K{K}_{M}_none"] = P12.argmax8(P11.probs(u.Xt[M], W, b))                       # none のテストの判定（DAI-12 の保存と同じもの）
        check(f"K{K}_test_none", int((store[f"K{K}_{M}_none"] != z12[f"K{K}_{M}_none"].astype(np.int8)).sum()), max(1, int(MISMATCH_FRAC * nt)), "none のテストの判定")
    pair_p = {}
    for K in KS_ALL:
        Lg = 0
        for o in P12.MODS:
            if o != M:
                Lg = Lg + none_p[(o, K)].clamp(min=1e-12).log()
        pair_p[K] = torch.softmax(Lg, -1)
    cl_pool = u.cp.df["cluster"].to_numpy()[u.pool].astype(str)
    half = P12.balanced_halves(cl_pool)
    check("pool_half", int((half.astype(np.int8) != z12["pool_half"].astype(np.int8)).sum()), 0, "半分の分け方")
    store["pool_half"] = half.astype(np.int8)
    seed_base = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % (2 ** 31)
    rng_s = np.random.default_rng(seed_base + 100 + 0)                                        # DAI-12 の m = A の入れ替えと同じ乱数
    src = np.arange(n)
    for h in (0, 1):
        for k in range(4):
            rows_h = np.flatnonzero((half == h) & (u.ypool == k)); rows_o = np.flatnonzero((half != h) & (u.ypool == k))
            if len(rows_h) and len(rows_o):
                src[rows_h] = rng_s.choice(rows_o, size=len(rows_h), replace=True)
    check("pool_shuf_src", int((src != z12[f"pool_shuf_src_{M}"].astype(int)).sum()), 0, "入れ替えの元の行")
    store[f"pool_shuf_src_{M}"] = src.astype(np.int32)
    for k in z12.files:                                                                        # 候補の部分集合（DAI-12 の保存を写す）
        if k.startswith("pool_sub_"):
            store[k] = z12[k].astype(np.int8)

    def teacher(t_: str, K: int) -> torch.Tensor:
        if t_ == "pair":
            return pair_p[K]
        if t_ == "pair_shuf":
            return pair_p[K][torch.as_tensor(src, device=S.DEV)]
        return none_p[(M, K)]

    cols = [(t_, K) for t_ in TEACHERS for K in KS_ALL]
    J = len(cols)
    P_list = [teacher(t_, K) for t_, K in cols]
    for j, (t_, K) in enumerate(cols):
        pt = P12.argmax8(P_list[j])
        store[f"pool_teacher_K{K}_{M}_{t_}"] = pt
        check(f"K{K}_pool_teacher_{t_}", int((pt != z12[f"pool_teacher_K{K}_{M}_{t_}"].astype(np.int8)).sum()), tol_n, f"出所 {t_} の候補の判定")
    Q = torch.stack([P.float() for P in P_list])

    # --- lw（層の重みを学ぶ），final（添字 24 だけ．DAI-12 の lince と同じ計算），unif（α を一様 1/25 に固定．層の重みを「学ぶ」ことの効果を分ける腕．review-ml 第 1 回の M2）
    #     3 つの腕の W・c の初期値は同じ種で引くので同じ（review-ml 第 1 回の m1）．違いは α（学ぶ／添字 24／一様）だけ
    Xu_raw, Xut_raw = Xp.mean(1), Xt.mean(1)
    mu_u, sd_u = Xu_raw.mean(0), Xu_raw.std(0).clamp(min=PSI_EPS)                               # 一様の ψ も候補の統計で標準化（lw の ψ の標準化と同じ尺度）
    Xu, Xut = ((Xu_raw - mu_u) / sd_u).contiguous(), ((Xut_raw - mu_u) / sd_u).contiguous()
    del Xu_raw, Xut_raw
    lw = crossfit_lw(Xp, Q, half, Xt, seed_base + 10, learn_alpha=True)
    final = P12.crossfit_nets(X24, Q, half, X24t, seed_base + 10, 0)
    unif = P12.crossfit_nets(Xu, Q, half, Xut, seed_base + 10, 0)
    j_pair = {K: cols.index(("pair", K)) for K in KS_ALL}
    pre_lw = dict(sel=lw["sel"].clone(), sel_test=lw["sel_test"].clone(), cfg=dict(lw["cfg"]), ce_sel=lw["ce_sel"].copy(), alpha_sel=lw["alpha_sel"].copy())   # 入れ子の前（層の重みを学んだ版）
    used_layers = nest_final(lw, final, half)                                                      # 入れ子の選択（設計 3.2 節．gate-probe 第 1 回の 1）

    # --- Q2：全ラベルの分類器（lw_lab，final_lab，unif_lab），Q3：α を pair の解（入れ子の前）に固定した全ラベルの分類器
    Qlab = torch.zeros(1, n, 4, device=S.DEV)
    for h in (0, 1):
        te = np.flatnonzero(half == h)
        Qlab[0, torch.as_tensor(te, device=S.DEV)] = P12.balanced_weights(u.ypool, te, n)[torch.as_tensor(te, device=S.DEV)]
    lw_lab = crossfit_lw(Xp, Qlab, half, Xt, seed_base + 30, learn_alpha=True, ckpts=LAB_CKPTS, steps=LAB_STEPS)
    final_lab = P12.crossfit_nets(X24, Qlab, half, X24t, seed_base + 30, 0, ckpts=LAB_CKPTS, steps=LAB_STEPS)
    unif_lab = P12.crossfit_nets(Xu, Qlab, half, Xut, seed_base + 30, 0, ckpts=LAB_CKPTS, steps=LAB_STEPS)
    pre_lw_lab = dict(sel=lw_lab["sel"].clone(), sel_test=lw_lab["sel_test"].clone(), cfg=dict(lw_lab["cfg"]), ce_sel=lw_lab["ce_sel"].copy(), alpha_sel=lw_lab["alpha_sel"].copy())
    lab_used_layers = nest_final(lw_lab, final_lab, half)
    alpha_fixed = torch.as_tensor(np.stack([pre_lw["alpha_sel"][:, j_pair[K]] for K in KS_ALL], 1))      # (2, 5, L)．入れ子の前の α（Q3 は「学んだ α」を問う）
    lw_lab_fix = crossfit_lw(Xp, Qlab.expand(len(KS_ALL), -1, -1).contiguous(), half, Xt, seed_base + 62, learn_alpha=False, alpha_fixed=alpha_fixed,
                             ckpts=LAB_CKPTS, steps=LAB_STEPS)

    # --- 層ごとの線形に読める分（リッジ．pair の 5 本の K）と，全ラベルのリッジ（記述）
    Ypair = torch.cat([pair_p[K].clamp(min=1e-12).log() for K in KS_ALL], 1)
    Ypair = Ypair - Ypair.reshape(n, len(KS_ALL), 4).mean(-1, keepdim=True).expand(-1, -1, 4).reshape(n, -1)
    Ylab = torch.zeros(n, 4, device=S.DEV); Ylab[torch.arange(n, device=S.DEV), torch.as_tensor(u.ypool, device=S.DEV)] = 1.0
    layer_rows = []
    for l in range(N_LAYERS):
        Yh, lam_l, r2_l = P11.representability(Xp[:, l].contiguous(), Ypair, half)
        Yl, lam_b, r2_b = P11.representability(Xp[:, l].contiguous(), Ylab, half)
        pl = P12.argmax8(Yl)
        store[f"pool_layer{l}_ridgelab_{M}"] = pl
        for ki, K in enumerate(KS_ALL):
            pr = P12.argmax8(Yh[:, 4 * ki: 4 * ki + 4])
            store[f"pool_layer{l}_ridge_K{K}_{M}_pair"] = pr
            nk = u.nk(K)
            layer_rows.append(dict(unit=name, T=T, c=c, fold=f, seed=r, layer=l, K=K, n_nk=len(nk),
                                   uar_layer_ridge_pair=P11.metrics(u.ypool[nk], pr.astype(int)[nk])["uar"], r2_layer_ridge_pair=float(r2_l[ki]), lam_layer_ridge_pair=float(lam_l[ki]),
                                   uar_layer_ridgelab=P11.metrics(u.ypool[nk], pl.astype(int)[nk])["uar"], r2_layer_ridgelab=float(r2_b[0])))

    # --- 保存（判定・点数・層の重み・格子）
    for kind, res in (("lw", lw), ("final", final), ("unif", unif)):
        store[f"grid_pred_{kind}_{M}"] = res["grid_pred"]
        conv[f"{kind}"] = dict(items=[f"{t_}_K{K}" for t_, K in cols], wds=list(WDS), ckpts=list(CKPTS), ce_hist=res["ce_hist"].tolist(), train_loss_final=res["train_loss"].tolist(),
                               cfg={f"h{h}_{t_}_K{K}": list(res["cfg"][(h, j)]) for h in (0, 1) for j, (t_, K) in enumerate(cols)})
    conv["lw_pre_cfg"] = {f"h{h}_{t_}_K{K}": list(pre_lw["cfg"][(h, j)]) for h in (0, 1) for j, (t_, K) in enumerate(cols)}
    for kind, res in (("lw_lab", lw_lab), ("final_lab", final_lab), ("unif_lab", unif_lab), ("lw_lab_fixalpha", lw_lab_fix)):
        store[f"grid_pred_{kind}_{M}"] = res["grid_pred"]
        conv[kind] = dict(wds=list(WDS), ckpts=list(LAB_CKPTS), ce_hist=res["ce_hist"].tolist(), cfg={f"h{h}_j{j}": list(v) for (h, j), v in res["cfg"].items()})
    for j, (t_, K) in enumerate(cols):
        for h in (0, 1):
            store[f"alpha_lw_K{K}_{M}_{t_}_h{h}"] = lw["alpha_sel"][h, j]                       # 入れ子の後（置き換わった半分は添字 24 の one-hot）
            store[f"alphapre_lw_K{K}_{M}_{t_}_h{h}"] = pre_lw["alpha_sel"][h, j]                # 入れ子の前（学んだ α）
            store[f"alphafix_lw_K{K}_{M}_{t_}_h{h}"] = lw["alpha_fix"][h, j]
    for h in (0, 1):
        store[f"alpha_lw_lab_{M}_h{h}"] = lw_lab["alpha_sel"][h, 0]
        store[f"alphapre_lw_lab_{M}_h{h}"] = pre_lw_lab["alpha_sel"][h, 0]

    # final と DAI-12 の lince の判定の不一致と UAR の差（乱数の列が違うので 0 ではない．記録．pair と self で大きければ止める．設計 3.2 節，a2）
    nl12 = pd.read_csv(D12 / "rows" / f"{name}_nl.csv")
    nl12 = nl12[nl12["student"] == M].set_index(["teacher", "K"])["uar_lince"]
    fr, du = {}, {}
    for j, (t_, K) in enumerate(cols):
        pf = P12.argmax8(final["sel"][j])
        fr[(t_, K)] = float((pf != z12[f"pool_lince_K{K}_{M}_{t_}"].astype(np.int8)).mean())
        nk = u.nk(K)
        du[(t_, K)] = abs(P11.metrics(u.ypool[nk], pf.astype(int)[nk])["uar"] - float(nl12.loc[(t_, K)]))
        mism[f"final_vs_dai12_lince_frac_{t_}_K{K}"] = fr[(t_, K)]
        mism[f"final_vs_dai12_lince_uar_absdiff_{t_}_K{K}"] = du[(t_, K)]
    mism["final_vs_dai12_lince_frac_max"] = max(fr.values()); mism["final_vs_dai12_lince_uar_absdiff_max"] = max(du.values())
    stop_fr = max(v for (t_, K), v in fr.items() if t_ in ("pair", "self")); stop_du = max(v for (t_, K), v in du.items() if t_ in ("pair", "self"))
    if stop_fr > FINAL_MISMATCH_STOP or stop_du > FINAL_UAR_STOP:
        raise SystemExit(f"{name}：final と DAI-12 の lince の不一致が大きい（pair・self の判定の不一致の割合の最大 {stop_fr:.3f}，UAR の差の最大 {stop_du:.2f}）")
    p_flab = P12.argmax8(final_lab["sel"][0])
    mism["final_lab_vs_dai12_lab_lince_frac"] = float((p_flab != z12[f"pool_lab_lince_{M}"].astype(np.int8)).mean())

    # --- Q2・Q3 の行
    p_lwlab, p_lwlab_test = P12.argmax8(lw_lab["sel"][0]), P12.argmax8(torch.log_softmax(lw_lab["sel_test"][:, 0], -1).mean(0))
    p_lwlab_pre = P12.argmax8(pre_lw_lab["sel"][0])
    p_flab_test = P12.argmax8(torch.log_softmax(final_lab["sel_test"][:, 0], -1).mean(0))
    p_ulab, p_ulab_test = P12.argmax8(unif_lab["sel"][0]), P12.argmax8(torch.log_softmax(unif_lab["sel_test"][:, 0], -1).mean(0))
    store[f"pool_lw_lab_{M}"], store[f"pool_final_lab_{M}"], store[f"pool_unif_lab_{M}"], store[f"pool_lwpre_lab_{M}"] = p_lwlab, p_flab, p_ulab, p_lwlab_pre
    store[f"labtest_lw_{M}"], store[f"labtest_final_{M}"], store[f"labtest_unif_{M}"] = p_lwlab_test, p_flab_test, p_ulab_test
    store[f"pool_lw_labfix_{M}"], store[f"pool_final_labfix_{M}"] = P12.argmax8(lw_lab["fix"][0]), P12.argmax8(final_lab["fix"][0])
    mism["unif_lab_vs_final_lab_agree"] = float((p_ulab == p_flab).mean())
    lab_rows = []
    for ki, K in enumerate(KS_ALL):
        pfa = P12.argmax8(lw_lab_fix["sel"][ki])
        store[f"pool_lw_lab_fixalpha_K{K}_{M}"] = pfa
        nk = u.nk(K); yk = u.ypool[nk]
        u_lw, u_f, u_fa, u_u, u_lwpre = (P11.metrics(yk, z_.astype(int)[nk])["uar"] for z_ in (p_lwlab, p_flab, pfa, p_ulab, p_lwlab_pre))
        u_lwf, u_ff = (P11.metrics(yk, z_.astype(int)[nk])["uar"] for z_ in (store[f"pool_lw_labfix_{M}"], store[f"pool_final_labfix_{M}"]))
        flags = {}
        for kind, res in (("lw_lab", lw_lab), ("final_lab", final_lab), ("unif_lab", unif_lab)):
            for h in (0, 1):
                _, wd_, st_ = res["cfg"][(h, 0)]
                src = final_lab if (kind == "lw_lab" and not lab_used_layers[h, 0]) else res
                hist = src["ce_hist"][h, 0, 0, WDS.index(wd_)]
                flags[f"{kind}_edge_wd_low_h{h}"] = wd_ == min(WDS); flags[f"{kind}_edge_wd_high_h{h}"] = wd_ == max(WDS)
                flags[f"{kind}_edge_step_low_h{h}"] = st_ == LAB_CKPTS[0]; flags[f"{kind}_edge_step_high_h{h}"] = st_ == LAB_CKPTS[-1]
                flags[f"{kind}_conv_h{h}"] = not (st_ == LAB_CKPTS[-1] and (hist[-2] - hist[-1]) > CONV_REL * hist[-2])
                flags[f"{kind}_wd_h{h}"], flags[f"{kind}_step_h{h}"] = wd_, st_
        for h in (0, 1):
            flags.update({f"lab_{k_}_h{h}": v_ for k_, v_ in alpha_stats(pre_lw_lab["alpha_sel"][h, 0]).items()})       # 学んだ α（入れ子の前）
            flags[f"lw_lab_loss_above_final_h{h}"] = bool(pre_lw_lab["ce_sel"][h, 0] > final_lab["ce_sel"][h, 0])        # 入れ子の前の損失で測る
            flags[f"lw_lab_used_layers_h{h}"] = bool(lab_used_layers[h, 0])
            _, wd_p, st_p = pre_lw_lab["cfg"][(h, 0)]
            flags[f"lw_lab_pre_edge_step_low_h{h}"] = st_p == LAB_CKPTS[0]; flags[f"lw_lab_pre_edge_step_high_h{h}"] = st_p == LAB_CKPTS[-1]
            flags[f"lw_lab_pre_edge_wd_low_h{h}"] = wd_p == min(WDS); flags[f"lw_lab_pre_edge_wd_high_h{h}"] = wd_p == max(WDS)
        lab_rows.append(dict(unit=name, T=T, c=c, fold=f, seed=r, student=M, K=K, n_pool=n, n_nk=len(nk),
                             uar_lab_lw=u_lw, uar_lab_final=u_f, lab_layers_minus_final=u_lw - u_f, uar_lab_lw_pre=u_lwpre, lab_layers_minus_final_pre=u_lwpre - u_f,
                             uar_lab_unif=u_u, lab_unif_minus_final=u_u - u_f, lab_lw_minus_unif=u_lw - u_u,
                             uar_lab_fixalpha=u_fa, lab_fixalpha_minus_final=u_fa - u_f,
                             uar_lab_lw_fixed=u_lwf, uar_lab_final_fixed=u_ff, lab_layers_minus_final_fixed=u_lwf - u_ff,
                             test_uar_lab_lw=P11.metrics(u.ytest, p_lwlab_test.astype(int))["uar"], test_uar_lab_final=P11.metrics(u.ytest, p_flab_test.astype(int))["uar"],
                             test_uar_lab_unif=P11.metrics(u.ytest, p_ulab_test.astype(int))["uar"],
                             ce_lab_lw_h0=float(lw_lab["ce_sel"][0, 0]), ce_lab_lw_h1=float(lw_lab["ce_sel"][1, 0]), ce_lab_final_h0=float(final_lab["ce_sel"][0, 0]), ce_lab_final_h1=float(final_lab["ce_sel"][1, 0]),
                             ce_lab_lw_pre_h0=float(pre_lw_lab["ce_sel"][0, 0]), ce_lab_lw_pre_h1=float(pre_lw_lab["ce_sel"][1, 0]), ce_lab_unif_h0=float(unif_lab["ce_sel"][0, 0]), ce_lab_unif_h1=float(unif_lab["ce_sel"][1, 0]),
                             psi_var_lab_h0=psi_var(Xp, pre_lw_lab["alpha_sel"][0, 0]), psi_var_lab_h1=psi_var(Xp, pre_lw_lab["alpha_sel"][1, 0]),
                             **flags))

    # --- Q4：K 本の 4×4 の読み出し（pair の lw と final の交差の予測の対数確率の上）
    read_pred = {}
    for K in KS_RULE:
        j = j_pair[K]
        for kind, res in (("lw", lw), ("final", final)):
            pr = readout_kshot(torch.log_softmax(res["sel"][j], -1), u.ypool, u.kpos[K])
            for wi in range(len(READ_WDS)):
                store[f"pool_read_{kind}_K{K}_{M}_wd{wi}"] = pr[wi]
            read_pred[(kind, K)] = pr

    # --- Q5：層の重みと線形の写像を K 本だけで学ぶ（lw の腕と final の腕）
    kshot_pred = {}
    for K in KS_RULE:
        for kind, learn in (("lw", True), ("final", False)):
            pr, al = kshot_lw(Xp, u.ypool, u.kpos[K], learn, seed_base + 70 + K)
            for wi in range(len(KSHOT_WDS)):
                store[f"pool_kshot_{kind}_K{K}_{M}_wd{wi}"] = pr[wi]
                if learn:
                    store[f"alpha_kshot_K{K}_{M}_wd{wi}"] = al[wi]
            kshot_pred[(kind, K)] = pr

    # --- 行（出所, K）
    lw_rows = []
    for j, (t_, K) in enumerate(cols):
        nk = u.nk(K); yk = u.ypool[nk]
        P = P_list[j]
        Lw_sel, Lw_fix, Lf_sel, Lf_fix, Lu_sel, Lu_fix, Lw_pre = lw["sel"][j], lw["fix"][j], final["sel"][j], final["fix"][j], unif["sel"][j], unif["fix"][j], pre_lw["sel"][j]
        p_lw, p_lwf, p_f, p_ff, p_u, p_uf, p_lwpre = (P12.argmax8(Z) for Z in (Lw_sel, Lw_fix, Lf_sel, Lf_fix, Lu_sel, Lu_fix, Lw_pre))
        store[f"pool_lw_K{K}_{M}_{t_}"], store[f"pool_lwfix_K{K}_{M}_{t_}"], store[f"pool_lwpre_K{K}_{M}_{t_}"] = p_lw, p_lwf, p_lwpre
        store[f"pool_final_K{K}_{M}_{t_}"], store[f"pool_finalfix_K{K}_{M}_{t_}"] = p_f, p_ff
        store[f"pool_unif_K{K}_{M}_{t_}"], store[f"pool_uniffix_K{K}_{M}_{t_}"] = p_u, p_uf
        store[f"score_lw_K{K}_{M}_{t_}"], store[f"score_final_K{K}_{M}_{t_}"], store[f"score_unif_K{K}_{M}_{t_}"] = P12.f16(Lw_sel), P12.f16(Lf_sel), P12.f16(Lu_sel)
        for kind, res in (("lw", lw), ("final", final), ("unif", unif)):
            store[f"K{K}_{M}_{kind}_{t_}"] = P12.argmax8(torch.log_softmax(res["sel_test"][:, j], -1).mean(0))
            store[f"K{K}_{M}_{kind}fix_{t_}"] = P12.argmax8(torch.log_softmax(res["fix_test"][:, j], -1).mean(0))
        store[f"K{K}_{M}_lwpre_{t_}"] = P12.argmax8(torch.log_softmax(pre_lw["sel_test"][:, j], -1).mean(0))
        pt, ps = store[f"pool_teacher_K{K}_{M}_{t_}"], store[f"pool_none_K{K}_{M}"]
        mt, ms = P11.metrics(yk, pt.astype(int)[nk]), P11.metrics(yk, ps.astype(int)[nk])
        m_lw, m_lwf, m_f, m_ff, m_u, m_uf, m_lwpre = (P11.metrics(yk, Z.astype(int)[nk]) for Z in (p_lw, p_lwf, p_f, p_ff, p_u, p_uf, p_lwpre))
        flags = {}
        for kind, res in (("lw", lw), ("final", final), ("unif", unif)):
            for h in (0, 1):
                lr_, wd_, st_ = res["cfg"][(h, j)]
                src = final if (kind == "lw" and not used_layers[h, j]) else res                 # 置き換わった半分は final の履歴
                hist = src["ce_hist"][h, j, 0, WDS.index(wd_)]
                flags[f"{kind}_conv_h{h}"] = not (st_ == CKPTS[-1] and (hist[-2] - hist[-1]) > CONV_REL * hist[-2])
                flags[f"{kind}_edge_wd_low_h{h}"] = wd_ == min(WDS); flags[f"{kind}_edge_wd_high_h{h}"] = wd_ == max(WDS)
                flags[f"{kind}_edge_step_low_h{h}"] = st_ == CKPTS[0]; flags[f"{kind}_edge_step_high_h{h}"] = st_ == CKPTS[-1]
                flags[f"{kind}_wd_h{h}"], flags[f"{kind}_step_h{h}"] = wd_, st_
                flags[f"ce_{kind}_h{h}"] = float(res["ce_sel"][h, j])
            flags[f"{kind}_conv"] = flags[f"{kind}_conv_h0"] and flags[f"{kind}_conv_h1"]
        for h in (0, 1):                                                                            # 入れ子の前（学んだ α の版）の量
            _, wd_p, st_p = pre_lw["cfg"][(h, j)]
            flags[f"lw_pre_edge_wd_low_h{h}"] = wd_p == min(WDS); flags[f"lw_pre_edge_wd_high_h{h}"] = wd_p == max(WDS)
            flags[f"lw_pre_edge_step_low_h{h}"] = st_p == CKPTS[0]; flags[f"lw_pre_edge_step_high_h{h}"] = st_p == CKPTS[-1]
            flags[f"ce_lw_pre_h{h}"] = float(pre_lw["ce_sel"][h, j])
            flags[f"lw_loss_above_final_h{h}"] = bool(pre_lw["ce_sel"][h, j] > final["ce_sel"][h, j])      # 入れ子の前の損失で測る（最適化の不足の目安）
            flags[f"lw_used_layers_h{h}"] = bool(used_layers[h, j])
            flags.update({f"{k_}_h{h}": v_ for k_, v_ in alpha_stats(pre_lw["alpha_sel"][h, j]).items()})     # 学んだ α の統計（入れ子の前）
            flags[f"psi_var_h{h}"] = psi_var(Xp, pre_lw["alpha_sel"][h, j])                                   # 標準化する前の ψ の分散（D19 の記録）
        row = dict(unit=name, T=T, c=c, fold=f, seed=r, student=M, teacher=t_, K=K, n_pool=n, n_nk=len(nk), n_half0=int((half == 0).sum()), n_half1=int((half == 1).sum()),
                   uar_s=ms["uar"], uar_t=mt["uar"], delta_uar=mt["uar"] - ms["uar"],
                   uar_final=m_f["uar"], uar_lw=m_lw["uar"], layers_minus_final=m_lw["uar"] - m_f["uar"],
                   uar_lw_pre=m_lwpre["uar"], layers_minus_final_pre=m_lwpre["uar"] - m_f["uar"],
                   uar_unif=m_u["uar"], unif_minus_final=m_u["uar"] - m_f["uar"], lw_minus_unif=m_lw["uar"] - m_u["uar"],
                   uar_final_fixed=m_ff["uar"], uar_lw_fixed=m_lwf["uar"], uar_unif_fixed=m_uf["uar"], layers_minus_final_fixed=m_lwf["uar"] - m_ff["uar"],
                   lw_gain=m_lw["uar"] - ms["uar"], final_gain=m_f["uar"] - ms["uar"], info_gap=mt["uar"] - m_f["uar"],
                   ce_lw=P12.soft_ce(P, Lw_sel), ce_final=P12.soft_ce(P, Lf_sel), ce_unif=P12.soft_ce(P, Lu_sel), ce_lw_pre=P12.soft_ce(P, Lw_pre),
                   ce_lw_fixed=P12.soft_ce(P, Lw_fix), ce_final_fixed=P12.soft_ce(P, Lf_fix),
                   agree_lw_final=float((p_lw[nk] == p_f[nk]).mean()), agree_lw_t=float((p_lw[nk] == pt[nk]).mean()), agree_final_t=float((p_f[nk] == pt[nk]).mean()),
                   final_vs_dai12_lince_frac=mism[f"final_vs_dai12_lince_frac_{t_}_K{K}"], final_vs_dai12_lince_uar_absdiff=mism[f"final_vs_dai12_lince_uar_absdiff_{t_}_K{K}"], **flags)
        if t_ == "pair" and K in KS_RULE:
            for kind in ("lw", "final"):
                for wi in range(len(READ_WDS)):
                    row[f"uar_read_{kind}_wd{wi}"] = P11.metrics(yk, read_pred[(kind, K)][wi].astype(int)[nk])["uar"]
                for wi in range(len(KSHOT_WDS)):
                    row[f"uar_kshot_{kind}_wd{wi}"] = P11.metrics(yk, kshot_pred[(kind, K)][wi].astype(int)[nk])["uar"]
            for wi in range(len(KSHOT_WDS)):
                row[f"kshot_layers_minus_final_wd{wi}"] = row[f"uar_kshot_lw_wd{wi}"] - row[f"uar_kshot_final_wd{wi}"]
                row[f"lw_minus_kshot_lw_wd{wi}"] = m_lw["uar"] - row[f"uar_kshot_lw_wd{wi}"]
                row.update({f"kshot_{k_}_wd{wi}": v_ for k_, v_ in alpha_stats(store[f"alpha_kshot_K{K}_{M}_wd{wi}"]).items()})
        lw_rows.append(row)

    np.savez_compressed(rows_dir / f"{name}.npz", commit=commit, commit11=commit11, commit12=commit12, unit=name, T=T, c=c, fold=f, seed=r, n_pool=n,
                        test_sample_id=sid_test, test_y=u.ytest.astype(np.int8), test_cluster=u.cp.df["cluster"].to_numpy()[u.test].astype(str),
                        pool_sample_id=sid_pool, pool_y=u.ypool.astype(np.int8), pool_cluster=cl_pool, **{f"kpos_K{K}": u.kpos[K] for K in DD.KS},
                        half_counts=np.stack([np.bincount(u.ypool[half == h], minlength=4) for h in (0, 1)]),
                        mismatch=json.dumps(mism), conv=json.dumps(conv), **store)
    pd.DataFrame(lw_rows).to_csv(rows_dir / f"{name}_lw.csv", index=False)
    pd.DataFrame(lab_rows).to_csv(rows_dir / f"{name}_lab.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(rows_dir / f"{name}_layers.csv", index=False)
    sec = time.time() - t0
    ints = [v for k, v in mism.items() if "frac" not in k and "absdiff" not in k]
    gpu_gb = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else float("nan")
    gpu_res = torch.cuda.max_memory_reserved() / 2 ** 30 if torch.cuda.is_available() else float("nan")
    print(f"単位 {name}: 候補 {n}（半分 {int((half == 0).sum())}/{int((half == 1).sum())}）テスト {nt} 保存 {len(store)} 行 {len(lw_rows)}+{len(lab_rows)}+{len(layer_rows)} "
          f"判定の不一致の最大 {max(ints)} 本 final と DAI-12 の不一致の割合の最大 {mism['final_vs_dai12_lince_frac_max']:.3f} 添字 24 の差 {diff24:.1e} "
          f"GPU 最大 {gpu_gb:.1f} GB（予約 {gpu_res:.1f} GB）({sec:.0f} 秒)", flush=True)
    return dict(unit=name, sec=sec, gpu_max_gb=gpu_gb, gpu_reserved_gb=gpu_res)


# ------------------------------------------------------------------ 集計

def summarize(root: Path, allow_partial: bool, n_expected: int) -> Path:
    """集計．出力は root/summary_seeds{種の数}/ に書く（r = 0 だけの版と r = 0, 1 の版を上書きしない．review-ml 第 2 回の M2）．戻りは出力先。"""
    files = sorted((root / "rows").glob("*.npz"))
    if len(files) != n_expected and not allow_partial:
        raise SystemExit(f"単位の出力が {len(files)} 個（{n_expected} のはず）．試走なら --allow-partial")
    rng = np.random.default_rng(SEED)
    commits = {str(np.load(p, allow_pickle=False)["commit"]) for p in files}
    if len(commits) != 1:
        raise SystemExit(f"行のコミットが 1 つでない：{commits}")
    for p in files:
        for suf in ("_lw.csv", "_lab.csv", "_layers.csv"):
            if not (root / "rows" / f"{p.stem}{suf}").exists():
                raise SystemExit(f"{p.stem}{suf} が無い")
    seeds = sorted({int(np.load(p, allow_pickle=False)["seed"]) for p in files})
    out = root / f"summary_seeds{len(seeds)}"
    if out.exists() and any(out.iterdir()) and not allow_partial:
        raise SystemExit(f"{out} は既にある（集計の版を上書きしない）")
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows").symlink_to(root / "rows") if not (out / "rows").exists() else None
    pool_kinds = ("none", "teacher", "lw", "lwfix", "lwpre", "final", "finalfix", "unif", "uniffix", "read_lw", "read_final", "kshot_lw", "kshot_final")
    cells, unit_rows, mism_rows, half_rows = {}, [], [], []
    for p in files:
        z = np.load(p, allow_pickle=False)
        T, c, f, r = str(z["T"]), str(z["c"]), int(z["fold"]), int(z["seed"])
        y, cl = z["test_y"].astype(int), z["test_cluster"].astype(str)
        yp, clp = z["pool_y"].astype(int), z["pool_cluster"].astype(str)
        kpos = {0: np.array([], int), **{K: z[f"kpos_K{K}"] for K in DD.KS}}
        nks = {K: np.setdiff1d(np.arange(len(yp)), kpos[K]) for K in KS_ALL}
        psubs = {k[len("pool_sub_"):]: z[k].astype(bool) for k in z.files if k.startswith("pool_sub_")}
        hc = z["half_counts"]
        half_rows.append(dict(c=c, T=T, fold=f, seed=r, n_half0=int(hc[0].sum()), n_half1=int(hc[1].sum()), **{f"h{h}_e{e}": int(hc[h, e]) for h in (0, 1) for e in range(4)}))

        def add_pool(key, prd, K, label):
            nk = nks[K]
            P12._add(cells.setdefault((c, f"{key}|nk"), {}), (f, r), ST.counts(yp[nk], prd[nk], clp[nk]), label)
            for nm, mk in psubs.items():
                mk2 = nk[mk[nk]]
                if len(mk2) >= ST.MIN_TEST:
                    P12._add(cells.setdefault((c, f"{key}|nk|sub_{nm}"), {}), (f, r), ST.counts(yp[mk2], prd[mk2], clp[mk2]), f"{label} {nm}")

        for k in z.files:
            if z[k].dtype != np.int8:
                continue
            if z[k].shape == y.shape and (k.startswith("K") or k.startswith("labtest_")):
                prd = z[k].astype(int)
                P12._add(cells.setdefault((c, k), {}), (f, r), ST.counts(y, prd, cl), f"{p.name} {k}")
                unit_rows.append(dict(c=c, T=T, fold=f, seed=r, key=k, **P11.metrics(y, prd)))
            elif z[k].shape == yp.shape and k.startswith("pool_") and any(k.startswith(f"pool_{kd}_K") for kd in pool_kinds):
                K = int(k.split("_K")[1].split("_")[0])
                add_pool(k, z[k].astype(int), K, f"{p.name} {k}")
            elif z[k].shape == yp.shape and k.startswith("pool_lw_lab_fixalpha_K"):
                K = int(k.split("_K")[1].split("_")[0])
                add_pool(k, z[k].astype(int), K, f"{p.name} {k}")
            elif z[k].shape == yp.shape and (k.startswith("pool_lw_lab") or k.startswith("pool_final_lab") or k.startswith("pool_unif_lab") or k.startswith("pool_lwpre_lab")):
                for K in KS_ALL:
                    add_pool(f"{k}_K{K}", z[k].astype(int), K, f"{p.name} {k} K{K}")
        for k, v in json.loads(str(z["mismatch"])).items():
            mism_rows.append(dict(unit=str(z["unit"]), key=k, value=v))
    P12._finish(cells)
    U = pd.DataFrame(unit_rows); U.to_csv(out / "uar_units.csv", index=False)
    pd.DataFrame(mism_rows).to_csv(out / "checks_units.csv", index=False)
    pd.DataFrame(half_rows).to_csv(out / "halves_units.csv", index=False)
    R = pd.concat([pd.read_csv(p) for p in sorted((out / "rows").glob("*_lw.csv"))]); R.to_csv(out / "lw_units.csv", index=False)
    Lb = pd.concat([pd.read_csv(p) for p in sorted((out / "rows").glob("*_lab.csv"))]); Lb.to_csv(out / "lab_units.csv", index=False)
    Ly = pd.concat([pd.read_csv(p) for p in sorted((out / "rows").glob("*_layers.csv"))]); Ly.to_csv(out / "layers_units.csv", index=False)
    num = [c_ for c_ in R.columns if c_ not in ("unit", "T", "c", "fold", "seed", "student", "teacher", "K")]
    Rc = R.groupby(["c", "student", "teacher", "K"])[num].mean().reset_index()
    Rc["n_units"] = R.groupby(["c", "student", "teacher", "K"]).size().to_numpy()
    Rc.to_csv(out / "cells_lw.csv", index=False)
    lnum = [c_ for c_ in Lb.columns if c_ not in ("unit", "T", "c", "fold", "seed", "student", "K")]
    Lc = Lb.groupby(["c", "student", "K"])[lnum].mean().reset_index(); Lc["n_units"] = Lb.groupby(["c", "student", "K"]).size().to_numpy(); Lc.to_csv(out / "cells_lab.csv", index=False)
    Lyc = Ly.groupby(["c", "layer", "K"])[["uar_layer_ridge_pair", "r2_layer_ridge_pair", "uar_layer_ridgelab", "r2_layer_ridgelab"]].mean().reset_index()
    Lyc.to_csv(out / "cells_layers.csv", index=False)

    # 層の重みの平均（c × 出所 × K × 層）
    arows = []
    for p in files:
        z = np.load(p, allow_pickle=False)
        c = str(z["c"])
        for k in z.files:
            for prefix, kind in (("alpha_lw_", "lw"), ("alphapre_lw_", "lw_pre")):
                if k.startswith(prefix) and not k.startswith(prefix + "lab"):
                    parts = k.split("_")
                    Kp, t_, h_ = parts[2], "_".join(parts[4:-1]), parts[-1]
                    arows.append(dict(c=c, unit=str(z["unit"]), kind=kind, teacher=t_, K=int(Kp[1:]), half=int(h_[1:]), **{f"w{l}": float(v) for l, v in enumerate(z[k])}))
                elif k.startswith(prefix + "lab_"):
                    arows.append(dict(c=c, unit=str(z["unit"]), kind=kind + "_lab", teacher="labels", K=-1, half=int(k.split("_")[-1][1:]), **{f"w{l}": float(v) for l, v in enumerate(z[k])}))
    A = pd.DataFrame(arows)
    wcols = [f"w{l}" for l in range(N_LAYERS)]
    Ac = A.groupby(["c", "kind", "teacher", "K"])[wcols].mean().reset_index()
    Ac["argmax_layer"] = Ac[wcols].to_numpy().argmax(1); Ac["w_final"] = Ac[f"w{FINAL_IDX}"]; Ac["w_max"] = Ac[wcols].max(1)
    Ac.to_csv(out / "alpha_summary.csv", index=False)

    # 差と区間（候補の上）
    geff = {c: ST.g_eff_of(c)[0] for c in DD.CORPORA}
    have_c = sorted(R["c"].unique())
    psubs_all = sorted({k.split("|sub_")[1] for (c_, k) in cells if "|nk|sub_" in k})
    rows = []
    for c in have_c:
        for K in KS_ALL:
            pre = f"K{K}_{M}"
            for t_ in TEACHERS:
                pairs = [(f"pool_lw_{pre}_{t_}", f"pool_final_{pre}_{t_}", "layers_minus_final"), (f"pool_lwfix_{pre}_{t_}", f"pool_finalfix_{pre}_{t_}", "layers_minus_final_fixed"),
                         (f"pool_lwpre_{pre}_{t_}", f"pool_final_{pre}_{t_}", "layers_minus_final_pre"),
                         (f"pool_unif_{pre}_{t_}", f"pool_final_{pre}_{t_}", "unif_minus_final"), (f"pool_lw_{pre}_{t_}", f"pool_unif_{pre}_{t_}", "lw_minus_unif"),
                         (f"pool_lw_{pre}_{t_}", f"pool_none_{pre}", "lw_gain"), (f"pool_final_{pre}_{t_}", f"pool_none_{pre}", "final_gain"),
                         (f"pool_teacher_{pre}_{t_}", f"pool_none_{pre}", "delta_uar"), (f"pool_teacher_{pre}_{t_}", f"pool_final_{pre}_{t_}", "info_gap")]
                if t_ == "pair":
                    pairs += [(f"pool_lw_{pre}_pair", f"pool_lw_{pre}_pair_shuf", "shuf_lw"), (f"pool_final_{pre}_pair", f"pool_final_{pre}_pair_shuf", "shuf_final")]
                for ka, kb, lab in pairs:
                    d = P12._diff(cells, c, f"{ka}|nk", f"{kb}|nk", geff[c], rng, dict(student=M, teacher=t_, K=K, where="pool", quantity=lab, subset="all"), True)
                    if d:
                        rows.append(d)
                        if lab in ("layers_minus_final", "layers_minus_final_fixed", "layers_minus_final_pre", "lw_minus_unif"):
                            for nm in psubs_all:
                                d2 = P12._diff(cells, c, f"{ka}|nk|sub_{nm}", f"{kb}|nk|sub_{nm}", geff[c], rng, dict(student=M, teacher=t_, K=K, where="pool", quantity=lab, subset=nm), True)
                                if d2:
                                    rows.append(d2)
                test_keys = {k for (_, k) in cells}
                for ka, kb, lab in ((f"{pre}_lw_{t_}", f"{pre}_final_{t_}", "lw_vs_final_test"), (f"{pre}_unif_{t_}", f"{pre}_final_{t_}", "unif_vs_final_test"),
                                    (f"{pre}_lw_{t_}", f"{pre}_none", "lw_vs_none_test"), (f"{pre}_final_{t_}", f"{pre}_none", "final_vs_none_test")):
                    if ka not in test_keys or kb not in test_keys:
                        continue
                    d = P12._diff(cells, c, ka, kb, geff[c], rng, dict(student=M, teacher=t_, K=K, where="test", quantity=lab, subset="all"), False)
                    if d:
                        rows.append(d)
            lab_pairs = [(f"pool_lw_lab_{M}_K{K}", f"pool_final_lab_{M}_K{K}", "lab_layers_minus_final"), (f"pool_lw_lab_fixalpha_K{K}_{M}", f"pool_final_lab_{M}_K{K}", "lab_fixalpha_minus_final"),
                         (f"pool_lwpre_lab_{M}_K{K}", f"pool_final_lab_{M}_K{K}", "lab_layers_minus_final_pre"),
                         (f"pool_unif_lab_{M}_K{K}", f"pool_final_lab_{M}_K{K}", "lab_unif_minus_final"), (f"pool_lw_lab_{M}_K{K}", f"pool_unif_lab_{M}_K{K}", "lab_lw_minus_unif"),
                         (f"pool_lw_labfix_{M}_K{K}", f"pool_final_labfix_{M}_K{K}", "lab_layers_minus_final_fixed")]
            for ka, kb, lab in lab_pairs:
                d = P12._diff(cells, c, f"{ka}|nk", f"{kb}|nk", geff[c], rng, dict(student=M, teacher="labels", K=K, where="pool", quantity=lab, subset="all"), True)
                if d:
                    rows.append(d)
            if K in KS_RULE:
                for kind in ("lw", "final"):
                    for wi in range(len(READ_WDS)):
                        for kb, lab in ((f"pool_none_{pre}", f"read_{kind}_wd{wi}_minus_none"), (f"pool_{kind}_{pre}_pair", f"read_{kind}_wd{wi}_minus_{kind}")):
                            d = P12._diff(cells, c, f"pool_read_{kind}_{pre}_wd{wi}|nk", f"{kb}|nk", geff[c], rng, dict(student=M, teacher="pair", K=K, where="pool", quantity=lab, subset="all"), True)
                            if d:
                                rows.append(d)
                for wi in range(len(KSHOT_WDS)):                                                    # Q5
                    for ka, kb, lab in ((f"pool_kshot_lw_{pre}_wd{wi}", f"pool_kshot_final_{pre}_wd{wi}", f"kshot_layers_minus_final_wd{wi}"),
                                        (f"pool_lw_{pre}_pair", f"pool_kshot_lw_{pre}_wd{wi}", f"lw_minus_kshot_lw_wd{wi}"),
                                        (f"pool_kshot_lw_{pre}_wd{wi}", f"pool_none_{pre}", f"kshot_lw_wd{wi}_minus_none")):
                        d = P12._diff(cells, c, f"{ka}|nk", f"{kb}|nk", geff[c], rng, dict(student=M, teacher="pair", K=K, where="pool", quantity=lab, subset="all"), True)
                        if d:
                            rows.append(d)
    Dd = pd.DataFrame(rows)
    # 初期値の揺れの代わりの見積もり v_init（設計 4.1 節．gate-probe 第 1 回の 2）：final は DAI-12 の lince を初期値だけ変えて流した複製なので，
    # 分割ごとの UAR(final) − UAR(lince₁₂)（3 つの T の平均）の分散 ÷ 分割の数を，層の重みの量の差の se² に足す（K 本の抜き方の揺れは含まない）
    d12 = []
    for p in files:
        z = np.load(p, allow_pickle=False)
        nl12 = pd.read_csv(D12 / "rows" / f"{p.stem}_nl.csv")
        nl12 = nl12[nl12["student"] == M][["teacher", "K", "uar_lince"]]
        nl12["unit"] = p.stem
        d12.append(nl12)
    D12nl = pd.concat(d12)
    Rm = R.merge(D12nl, on=["unit", "teacher", "K"], how="left")
    if Rm["uar_lince"].isna().any():
        raise SystemExit("DAI-12 の lince の UAR と合わない行がある")
    Rm["d_init"] = Rm["uar_final"] - Rm["uar_lince"]
    vi_rows = []
    for (c, t_, K), g in Rm.groupby(["c", "teacher", "K"]):
        per_fold = g.groupby("fold")["d_init"].mean()
        v_init = float(per_fold.var(ddof=1) / len(per_fold)) if len(per_fold) > 1 else float("nan")
        vi_rows.append(dict(c=c, teacher=t_, K=K, v_init=v_init, n_folds=len(per_fold), d_init_mean=float(per_fold.mean())))
    VI = pd.DataFrame(vi_rows); VI.to_csv(out / "v_init.csv", index=False)
    adj_q = ("layers_minus_final", "layers_minus_final_pre", "layers_minus_final_fixed", "lw_minus_unif", "unif_minus_final")
    Dd = Dd.merge(VI[["c", "teacher", "K", "v_init"]], on=["c", "teacher", "K"], how="left")
    Dd["lo_raw"], Dd["hi_raw"], Dd["se_raw"], Dd["p_raw"] = Dd["lo"], Dd["hi"], Dd["se"], Dd["p"]
    # 種が (c, 出所, K, 分割) ごとに 2 つ以上そろっていれば v_seed が立つので v_init は足さない（設計 4.1 節．review-ml 第 2 回の M2）
    seeds_per = R.groupby(["c", "teacher", "K", "fold"])["seed"].nunique()
    v_init_applied = bool((seeds_per < 2).any())
    adj = Dd["quantity"].isin(adj_q) & (Dd["where"] == "pool") & Dd["v_init"].notna() & v_init_applied
    # 元の区間の分位点 q（正規なら 1.96，jackknife の枝なら t の分位点）を行ごとに復元して使う（review-ml 第 2 回の M1）
    q = ((Dd["est"] - Dd["lo_raw"]) / Dd["se_raw"]).where(Dd["se_raw"] > 0, 1.96)
    Dd.loc[adj, "se"] = np.sqrt(Dd.loc[adj, "se_raw"] ** 2 + Dd.loc[adj, "v_init"])
    Dd.loc[adj, "lo"] = Dd.loc[adj, "est"] - q[adj] * Dd.loc[adj, "se"]; Dd.loc[adj, "hi"] = Dd.loc[adj, "est"] + q[adj] * Dd.loc[adj, "se"]
    from scipy import stats as _st
    from scipy.optimize import brentq as _brentq

    def _p_adj(est, se, qq):
        if not np.isfinite(se) or se <= 0:
            return np.nan
        if abs(qq - 1.959964) < 1e-3:
            return float(2 * _st.norm.sf(abs(est) / se))
        try:
            df_ = _brentq(lambda d_: _st.t.ppf(0.975, d_) - qq, 1.0, 500.0)
        except ValueError:
            return np.nan
        return float(2 * _st.t.sf(abs(est) / se, df_))
    Dd.loc[adj, "p"] = [_p_adj(e, s, qq) for e, s, qq in zip(Dd.loc[adj, "est"], Dd.loc[adj, "se"], q[adj])]
    Dd["quantile"] = q
    Dd["se_adjusted"] = adj
    Dd["v_init_applied"] = v_init_applied
    Dd.to_csv(out / "diffs.csv", index=False)
    se_rows = [dict(c=c, quantity=q_, se_median=g["se"].median(), se_min=g["se"].min(), se_max=g["se"].max(), se_raw_median=g["se_raw"].median(), n=len(g))
               for (c, q_), g in Dd[Dd["subset"] == "all"].groupby(["c", "quantity"])]
    pd.DataFrame(se_rows).to_csv(out / "se_summary.csv", index=False)
    json.dump(dict(v_init_applied=v_init_applied, n_seeds=int(R["seed"].nunique()), n_units=len(files)), open(out / "summary_meta.json", "w"), ensure_ascii=False, indent=1)

    # 規則の数え上げ（R1・R1c・R1shuf，Q2・Q3，添え書き）
    crit = []
    main_sub = {"MOSEI": "channel", "MELD": "dlgfree"}                                       # 部分集合の頑健さを判定に使うコーパス（設計 4.2 節）
    for c in have_c:
        rows_c = {}
        for t_ in TEACHERS:
            d = Dd[(Dd["c"] == c) & (Dd["teacher"] == t_) & (Dd["quantity"] == "layers_minus_final") & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
            verdict, up, down = P12._verdict(d, FLOOR_POOL)
            verdict = {"上回った": "取り戻した"}.get(verdict, verdict)
            row = dict(c=c, student=M, rule={"pair": "R1", "pair_shuf": "R1shuf", "self": "R1c"}[t_], teacher=t_, quantity="layers_minus_final", n_K=len(d), n_K_up=len(up), n_K_down=len(down),
                       verdict=verdict if t_ != "self" else ("手順の雑音は小さい" if (d["est"].abs() <= SELF_TOL).all() else "手順の雑音が大きい"),
                       est_min=d["est"].min(), est_max=d["est"].max(), lo_min=d["lo"].min(), hi_max=d["hi"].max(), lo_raw_min=d["lo_raw"].min(), v_init_mean=d["v_init"].mean())
            for nm in psubs_all:
                row[f"sub_{nm}_n_K_robust"] = P12._robust(Dd, c, M, t_, "layers_minus_final", nm, up) if len(up) else 0
            if t_ == "pair":
                ds = Dd[(Dd["c"] == c) & (Dd["teacher"] == "pair_shuf") & (Dd["quantity"] == "layers_minus_final") & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
                row["pair_minus_shuf_gap"] = float(d["est"].mean() - ds["est"].mean()) if len(ds) else np.nan
                dg = Dd[(Dd["c"] == c) & (Dd["teacher"] == "pair") & (Dd["quantity"] == "info_gap") & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
                row["info_gap_mean"] = float(dg["est"].mean()) if len(dg) else np.nan
                dp = Dd[(Dd["c"] == c) & (Dd["teacher"] == "pair") & (Dd["quantity"] == "layers_minus_final_pre") & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
                row["pre_est_mean"] = float(dp["est"].mean()) if len(dp) else np.nan
                du_ = Dd[(Dd["c"] == c) & (Dd["teacher"] == "pair") & (Dd["quantity"] == "lw_minus_unif") & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
                row["lw_minus_unif_mean"] = float(du_["est"].mean()) if len(du_) else np.nan
            rows_c[t_] = row
            crit.append(row)
        # 入れ替えの対照の 3 つの場合（設計 4.2 節の (A)(B)(C)．pair が「取り戻した」のときだけ）
        rp, rs = rows_c["pair"], rows_c["pair_shuf"]
        if rp["verdict"] != "取り戻した":
            case = "—"
        else:
            shuf_up = rs["n_K_up"] >= 3
            robust = True if c not in main_sub else (rs.get(f"sub_{main_sub[c]}_n_K_robust", 0) >= 3)
            case = "A" if (shuf_up and robust) else ("B" if shuf_up else "C")
        rp["shuf_case"] = case
        rp["pair_sub_robust_ok"] = True if c not in main_sub else (rp.get(f"sub_{main_sub[c]}_n_K_robust", 0) >= 3)
        for q in ("lab_layers_minus_final", "lab_fixalpha_minus_final"):
            d = Dd[(Dd["c"] == c) & (Dd["quantity"] == q) & (Dd["subset"] == "all")]
            crit.append(dict(c=c, student=M, rule="Q2" if q == "lab_layers_minus_final" else "Q3", teacher="labels", quantity=q, n_K=len(d), n_K_up=int(((d["lo"] > 0) & (d["est"] >= FLOOR_POOL)).sum()),
                             n_K_down=int(((d["hi"] < 0) & (d["est"] <= -FLOOR_POOL)).sum()), verdict="記述（規則の語なし）", est_min=d["est"].min(), est_max=d["est"].max(), lo_min=d["lo"].min(), hi_max=d["hi"].max()))
        d2 = Dd[(Dd["c"] == c) & (Dd["quantity"] == "lab_layers_minus_final") & (Dd["subset"] == "all")]; d3 = Dd[(Dd["c"] == c) & (Dd["quantity"] == "lab_fixalpha_minus_final") & (Dd["subset"] == "all")]
        if len(d2) and len(d3) and d2["est"].mean() >= FLOOR_POOL:
            crit.append(dict(c=c, student=M, rule="Q3/Q2", teacher="labels", quantity="ratio", verdict="記述", est_min=d3["est"].mean() / d2["est"].mean(), est_max=np.nan))
        Rc_ = R[R["c"] == c]
        edge = {}
        for kind in ("lw", "final", "unif"):
            for ax in ("wd", "step"):
                for side in ("low", "high"):
                    edge[f"{kind}_edge_{ax}_{side}_frac"] = float(np.mean([Rc_[f"{kind}_edge_{ax}_{side}_h{h}"].mean() for h in (0, 1)]))
            edge[f"{kind}_conv_frac"] = float(Rc_[f"{kind}_conv"].mean())
        for ax in ("wd", "step"):
            for side in ("low", "high"):
                edge[f"lw_pre_edge_{ax}_{side}_frac"] = float(np.mean([Rc_[f"lw_pre_edge_{ax}_{side}_h{h}"].mean() for h in (0, 1)]))
        edge["lw_loss_above_final_frac"] = float(np.mean([Rc_[f"lw_loss_above_final_h{h}"].mean() for h in (0, 1)]))                  # 入れ子の前（全出所・全 K．記述）
        for t_ in TEACHERS:                                                                                                               # 出所ごと（review-ml 第 2 回の m2）
            Rt_ = Rc_[(Rc_["teacher"] == t_) & (Rc_["K"].isin(KS_RULE))]
            edge[f"lw_loss_above_final_frac_{t_}"] = float(np.mean([Rt_[f"lw_loss_above_final_h{h}"].mean() for h in (0, 1)]))
            edge[f"lw_used_layers_frac_{t_}"] = float(np.mean([Rt_[f"lw_used_layers_h{h}"].mean() for h in (0, 1)]))
            edge[f"alpha_far_from_uniform_frac_{t_}"] = float(np.mean([(Rt_[f"alpha_max_h{h}"] >= ALPHA_FAR).mean() for h in (0, 1)]))
        edge["lw_loss_note"] = edge["lw_loss_above_final_frac_pair"] > LOSS_NOTE_FRAC                                                 # m7：R1（pair）に「最適化の不足の疑い」を添える
        edge["lw_used_layers_frac"] = float(np.mean([Rc_[f"lw_used_layers_h{h}"].mean() for h in (0, 1)]))                           # NEST：層の重みを学んだ版が選ばれた割合（全出所）
        edge["alpha_max_over_0.9_frac"] = float(np.mean([(Rc_[f"alpha_max_h{h}"] > ALPHA_MAX_NOTE).mean() for h in (0, 1)]))
        edge["alpha_final_over_0.5_frac"] = float(np.mean([(Rc_[f"alpha_final_h{h}"] > ALPHA_FINAL_NOTE).mean() for h in (0, 1)]))
        edge["alpha_far_from_uniform_frac"] = float(np.mean([(Rc_[f"alpha_max_h{h}"] >= ALPHA_FAR).mean() for h in (0, 1)]))        # 全出所・全 K（記述）
        edge["alpha_argmax_median"] = float(np.median(np.concatenate([Rc_[f"alpha_argmax_h{h}"].to_numpy() for h in (0, 1)])))
        edge["alpha_final_mean"] = float(np.mean([Rc_[f"alpha_final_h{h}"].mean() for h in (0, 1)]))
        Rp_ = Rc_[(Rc_["teacher"] == "pair") & (Rc_["K"].isin(KS_RULE))]                                                               # 見込み (f) を読む条件は pair・K ∈ KS_RULE に絞る（b1）
        edge["alpha_far_from_uniform_frac_pair"] = float(np.mean([(Rp_[f"alpha_max_h{h}"] >= ALPHA_FAR).mean() for h in (0, 1)]))
        edge["alpha_argmax_median_pair"] = float(np.median(np.concatenate([Rp_[f"alpha_argmax_h{h}"].to_numpy() for h in (0, 1)])))
        edge["alpha_final_mean_pair"] = float(np.mean([Rp_[f"alpha_final_h{h}"].mean() for h in (0, 1)]))
        edge["self_alpha_final_mean"] = float(np.mean([Rc_[Rc_["teacher"] == "self"][f"alpha_final_h{h}"].mean() for h in (0, 1)]))   # 仮定 (i)：self の添字 24 の重み
        edge["psi_var_mean"] = float(np.mean([Rc_[f"psi_var_h{h}"].mean() for h in (0, 1)]))
        edge["final_vs_dai12_lince_frac_mean"] = float(Rc_["final_vs_dai12_lince_frac"].mean()); edge["final_vs_dai12_lince_frac_max"] = float(Rc_["final_vs_dai12_lince_frac"].max())
        edge["final_vs_dai12_lince_uar_absdiff_max"] = float(Rc_["final_vs_dai12_lince_uar_absdiff"].max())
        crit.append(dict(c=c, student=M, rule="EDGE/OPT/LOSS/NEST/ALPHA/MISMATCH", teacher="all", quantity="fractions", verdict="記述", **edge))
        # 量の大きさと得の順位（設計 1 節．記述）
        crit.append(dict(c=c, student=M, rule="GAP_vs_GAIN", teacher="pair", quantity="info_gap_mean / layers_minus_final_mean", verdict="記述",
                         est_min=rows_c["pair"]["info_gap_mean"], est_max=float(Dd[(Dd["c"] == c) & (Dd["teacher"] == "pair") & (Dd["quantity"] == "layers_minus_final") & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]["est"].mean())))
    pd.DataFrame(crit).to_csv(out / "criteria.csv", index=False)
    print(f"集計：単位 {len(files)}（種 {seeds}），差の行 {len(Dd)}，規則の行 {len(crit)}，v_init を足した {v_init_applied} → {out}", flush=True)
    return out


# ------------------------------------------------------------------ 自己検査（人工データ）

def selftest() -> None:
    """(1) 層 10 だけに線形の信号があり最終層に無い → lw は読め，final は読めない．(2) 最終層だけに信号 → lw ≈ final（層の重みが最終層に寄る）．
    (3) 目標が最終層の線形関数（self の形）→ layers_minus_final ≈ 0。"""
    g = np.random.default_rng(0)
    n, L, d = 1200, N_LAYERS, 64
    Xn = g.standard_normal((n, L, d)).astype(np.float32)
    y = g.integers(0, 4, n)
    Wsig = g.standard_normal((d, 4)).astype(np.float32)
    onehot = np.eye(4, dtype=np.float32)[y]
    half = (np.arange(n) % 2)
    cl = np.array([f"c{i % 40}" for i in range(n)])
    half = P12.balanced_halves(cl)
    out = {}
    for case in ("layer10", "final", "self"):
        X = Xn.copy()
        sig = onehot @ Wsig.T * 1.5
        if case == "layer10":
            X[:, 10] += sig
        else:
            X[:, FINAL_IDX] += sig
        Xt = torch.as_tensor(X, device=S.DEV)
        mu, sd = Xt.mean(0), Xt.std(0).clamp(min=1e-6)
        Xt = (Xt - mu) / sd
        if case == "self":
            Wl = torch.as_tensor(g.standard_normal((d, 4)).astype(np.float32), device=S.DEV)
            Q = torch.softmax(Xt[:, FINAL_IDX] @ Wl * 0.7, -1)[None]
        else:
            Q = torch.as_tensor(onehot * 0.85 + 0.05, device=S.DEV)[None]
        res_lw = crossfit_lw(Xt, Q, half, Xt[:5], 1, learn_alpha=True)
        res_f = P12.crossfit_nets(Xt[:, FINAL_IDX].contiguous(), Q, half, Xt[:5, FINAL_IDX].contiguous(), 2, 0)
        tgt = Q[0].argmax(-1).cpu().numpy()
        u_lw = P11.metrics(tgt, res_lw["sel"][0].argmax(-1).cpu().numpy())["uar"]; u_f = P11.metrics(tgt, res_f["sel"][0].argmax(-1).cpu().numpy())["uar"]
        a = res_lw["alpha_sel"].mean(0)[0]
        pv = psi_var(Xt, a)
        used = nest_final(res_lw, res_f, half)                                                   # 入れ子の選択も通す（置き換えの割合を印字）
        u_nest = P11.metrics(tgt, res_lw["sel"][0].argmax(-1).cpu().numpy())["uar"]
        out[case] = (u_lw, u_f, int(a.argmax()), float(a.max()))
        print(f"自己検査 {case}: 目標の当たり lw {u_lw:.1f} / final {u_f:.1f}（入れ子の後 {u_nest:.1f}，層の重みを学んだ版が残った半分 {int(used.sum())}/2），"
              f"層の重みの最大は層 {int(a.argmax())}（{a.max():.2f}），標準化する前の ψ の分散 {pv:.2f}", flush=True)
        if case == "layer10":
            assert used.all(), "層 10 の信号で入れ子が final に置き換えた"
            assert u_nest == u_lw
    assert out["layer10"][0] > out["layer10"][1] + 20 and out["layer10"][2] == 10, "層 10 の信号を lw が拾えていない"
    assert abs(out["final"][0] - out["final"][1]) < 3.0, "最終層の信号で lw と final が離れた"
    assert abs(out["self"][0] - out["self"][1]) < 3.0, "self の形で lw と final が離れた"
    # 読み出しの自己検査：4 クラスを入れ替えた対数確率を K 本で直す
    perm = np.array([1, 2, 3, 0])
    logp = torch.log(torch.as_tensor(onehot[:, perm] * 0.85 + 0.05, device=S.DEV))
    kpos = np.concatenate([np.flatnonzero(y == k)[:10] for k in range(4)])
    pr = readout_kshot(logp, y, kpos)
    u0 = P11.metrics(y, logp.argmax(-1).cpu().numpy())["uar"]; u1 = [P11.metrics(y, pr[i].astype(int))["uar"] for i in range(len(READ_WDS))]
    print(f"自己検査 読み出し: 直す前 {u0:.1f} → 直した後 {[round(v, 1) for v in u1]}（wd {READ_WDS}）", flush=True)
    assert u1[0] > 90, "4×4 の読み出しが入れ替えを直せていない"
    # K 本だけで層の重みを学ぶ自己検査（層 10 の信号．感情ごとに 10 本）：走ること，形，final の腕が当てずっぽうに近いこと
    X = Xn.copy(); X[:, 10] += onehot @ Wsig.T * 1.5
    Xt = torch.as_tensor(X, device=S.DEV); Xt = (Xt - Xt.mean(0)) / Xt.std(0).clamp(min=1e-6)
    pk, ak = kshot_lw(Xt, y, kpos, True, 3); pf, _ = kshot_lw(Xt, y, kpos, False, 4)
    uk = [P11.metrics(y, pk[i].astype(int))["uar"] for i in range(len(KSHOT_WDS))]; uf = [P11.metrics(y, pf[i].astype(int))["uar"] for i in range(len(KSHOT_WDS))]
    print(f"自己検査 K 本の層の重み: lw {[round(v, 1) for v in uk]} / final {[round(v, 1) for v in uf]}，層の重みの最大は層 {int(ak[0].argmax())}（{ak[0].max():.2f}）", flush=True)
    assert pk.shape == (len(KSHOT_WDS), n) and ak.shape == (len(KSHOT_WDS), L) and max(uf) < 40, "K 本の層の重みの自己検査の形か final の腕がおかしい"
    # 候補の上の区間の自己検査は dai12_probe.selftest に同じものがある
    print("自己検査 OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("units", "summarize", "selftest"))
    ap.add_argument("--out", default=str(ROOT / "results" / "dai13_probe"))
    ap.add_argument("--units", default="*_r0", help="単位の名前の glob（既定は種 0 の 60 単位）")
    ap.add_argument("--n-expected", type=int, default=N_UNITS_R0)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.stage == "selftest":
        selftest(); return
    out = Path(args.out)
    commit, dirty = dai1.git_commit()
    if dirty and not args.allow_dirty:
        sys.exit("DAI/exp・measure・features に未コミットの変更がある（試走は --allow-dirty）")
    prov11 = json.load(open(D11 / "provenance.json")); commit11 = str(prov11["commit"])
    prov12 = json.load(open(D12 / "provenance.json")); commit12 = str(prov12["commit"])
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, host=platform.node(), python=platform.python_version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                script_sha256=P12.sha(Path(__file__)), dai12_probe_sha256=P12.sha(HERE / "dai12_probe.py"), dai11_probe_sha256=P12.sha(HERE / "dai11_probe.py"),
                dai1_sha256=P12.sha(HERE / "dai1.py"), solver_sha256=P12.sha(HERE / "dai_solver.py"), data_sha256=P12.sha(HERE / "dai_data.py"), stats_sha256=P12.sha(HERE / "dai1_stats.py"),
                frozen_sha256=P12.sha(FROZEN), build_layers_sha256=P12.sha(ROOT / "features" / "build_audio_layers.py"),
                design_sha256=P12.sha(ROOT / "DAI" / "docs" / "実験13_下調べ_設計.md") if (ROOT / "DAI" / "docs" / "実験13_下調べ_設計.md").exists() else None,
                dai11_commit=commit11, dai12_commit=commit12, dai12_script_sha256=prov12.get("script_sha256"),
                pseudo_commit={T: json.load(open(FROZEN))[T]["pseudo_commit"] for T in DD.CORPORA},
                layer_features_sha256={c: P12.sha(ROOT / "data" / "cache" / f"{LAYER_KIND}_{c}.npz") for c in DD.CORPORA if (ROOT / "data" / "cache" / f"{LAYER_KIND}_{c}.npz").exists()},
                constants=dict(KS_ALL=KS_ALL, KS_RULE=KS_RULE, TEACHERS=TEACHERS, N_LAYERS=N_LAYERS, FINAL_IDX=FINAL_IDX, WDS=WDS, CKPTS=CKPTS, STEPS=STEPS, LRS=LRS, FIXED_WD=FIXED_WD, FIXED_STEP=FIXED_STEP,
                               LAB_CKPTS=LAB_CKPTS, LAB_STEPS=LAB_STEPS, READ_WDS=READ_WDS, READ_STEPS=READ_STEPS, KSHOT_WDS=KSHOT_WDS, KSHOT_STEPS=KSHOT_STEPS,
                               CONV_REL=CONV_REL, LAYER_DIFF_STOP=LAYER_DIFF_STOP, FINAL_MISMATCH_STOP=FINAL_MISMATCH_STOP,
                               FINAL_UAR_STOP=FINAL_UAR_STOP, ALPHA_LR=ALPHA_LR, PSI_EPS=PSI_EPS, ALPHA_FAR=ALPHA_FAR,
                               FLOOR_POOL=FLOOR_POOL, SELF_TOL=SELF_TOL, ALPHA_MAX_NOTE=ALPHA_MAX_NOTE, ALPHA_FINAL_NOTE=ALPHA_FINAL_NOTE, SEED=SEED, N_BOOT=N_BOOT, MISMATCH_FRAC=MISMATCH_FRAC))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    fz = json.load(open(FROZEN))
    if args.stage == "units":
        files = sorted(UD.glob(f"{args.units}_p3.npz"))
        if not files:
            sys.exit("入力の単位が無い")
        if args.resume:
            files = [p for p in files if not (out / "rows" / f"{p.name[:-len('_p3.npz')]}.npz").exists()]
            print(f"続き：残り {len(files)} 単位", flush=True)
        rows = [run_unit(p, out, fz, commit, commit11, commit12) for p in files]
        pd.DataFrame(rows).to_csv(out / f"units_time_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
        return
    if args.stage == "summarize":
        sdir = summarize(out, args.allow_partial, args.n_expected)
        prov["summary_dir"] = str(sdir)
        prov.update(json.load(open(sdir / "summary_meta.json")))
        json.dump(prov, open(sdir / "provenance.json", "w"), ensure_ascii=False, indent=1)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)                 # 最新の集計の写し（DAI/CLAUDE.md の取り決めの置き場）


if __name__ == "__main__":
    main()
