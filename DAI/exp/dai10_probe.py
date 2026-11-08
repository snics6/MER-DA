"""DAI-10 の下調べ：ラベル無しの発話を計量にだけ使うと，K 本のガウスの判別は良くなるか
（設計 DAI/docs/実験10_下調べ_設計.md）。DAI-8 の関数（dai8_probe.py）と DAI-8 の保存した判定を使う。

単位の段：  python DAI/exp/dai10_probe.py --out results/dai10_probe [--units 名前,...]
集計の段：  python DAI/exp/dai10_probe.py --out results/dai10_probe --summarize [--allow-partial]
自己検査：  python DAI/exp/dai10_probe.py --selftest
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai_data as DD  # noqa: E402
import dai7_probe_v2 as P7  # noqa: E402
import dai8_probe as P8  # noqa: E402   lda_from_R，delta，_add，_finish
import dai3  # noqa: E402   部分集合の印（MOSEI の channel，AFEW の asr_ok，MELD の face_single）

ROOT = P8.ROOT
UD, FROZEN = P7.UD, P7.FROZEN
DEV, DT = P8.DEV, P8.DT
SPACES = P8.SPACES
BLOCKS = P8.BLOCKS
ALPHAS = (0.01, 0.1, 0.5, 0.9)   # 0.01 は gate-code の第 1 回（縮小をほぼ使わない先行研究の推定量に近い点．記述）
A_RULE = 0.5                  # 規則で数える点（4 節．計算の前に固定）
KS_RULE = (1, 5, 10, 25)
MIN_D = 0.5
N_UNITS = 120
SEED = 0
D8 = ROOT / "results" / "dai8_probe"
D8_COMMIT = "a90af78a2bef71a8372fcf52c9ca093853c29661"
LDAK_TOL = 0.001              # K 本だけの LDA を解き直した判定と DAI-8 の保存の不一致の許し（数値の丸めでの同点の入れ替わり）
FX8 = "a0.5_b1.0"
N_EM = 10                     # EM_Kmet の回の数（DAI-8 の EM_fromK と同じ）
# DAI-8 の保存から写す判定（{s} は空間，{K} は K）
COPY_K = ("K{K}_{s}_emfromK_a0.5_b0.0", f"K{{K}}_{{s}}_em_{FX8}", "K{K}_{s}_none", "K{K}_{s}_joint", "K{K}_{s}_noneT")
D8_ALPHAS = (0.1, 0.5, 0.9)   # DAI-8 が LDA_K を保存した α（台本で解き直した LDA_K と照合する）
N_PERM = 5                    # 縮みの係数の並べ替えの帰無の回数（gate-probe の第 1 回）
COPY_0 = ("{s}_ldaO_a0.5", "{s}_oracle")


def lda_metric(mu, chol):
    """平均 mu (4, d) と計量 Σ = L Lᵀ（chol）から判別の係数を作る（等しい事前）。"""
    A = torch.cholesky_solve(mu.T.contiguous(), chol)
    c = -0.5 * (mu * A.T).sum(1)
    if not (torch.isfinite(A).all() and torch.isfinite(c).all()):
        raise FloatingPointError("判別の係数に有限でない値がある")
    return dict(A=A, c=c, mu=mu)


def class_means(XL, ylab):
    M = torch.zeros((4, XL.shape[1]), dtype=XL.dtype, device=XL.device).index_add_(0, ylab, XL)
    n = torch.bincount(ylab, minlength=4).to(XL.dtype)
    if (n == 0).any():
        raise SystemExit("K 本に欠けた感情がある")
    return M / n[:, None]


def metric_chol(T_, alpha, block=None):
    S = T_
    if block is not None:
        mask = torch.zeros_like(S, dtype=torch.bool)
        o = 0
        for _, w in block:
            mask[o:o + w, o:o + w] = True
            o += w
        S = torch.where(mask, S, torch.zeros_like(S))
    A = (1.0 - alpha) * S + alpha * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
    return torch.linalg.cholesky(A)


def em_kmetric(Xp, XtX, kpos_t, ylab, q0, chol, n_iter=N_EM, yU=None):
    """4 つ目の角：K 本の計量 Σ_K（chol）を固定し，K 本を one-hot に固定して，平均と割合だけを EM で回す（β = 0）。
    回 t の分類器 = t 回の E 段の後の M 段の解（DAI-8 の em_fit と同じ数え方）。"""
    R = q0.clone()
    R[kpos_t] = 0.0
    R[kpos_t, ylab] = 1.0
    U = torch.ones(Xp.shape[0], dtype=torch.bool, device=Xp.device)
    U[kpos_t] = False
    hist, pool_hist = [], []
    for t in range(n_iter + 1):
        f = P8.lda_from_R(Xp, R, XtX, A_RULE, chol=chol)
        if yU is not None:
            pool_hist.append(float(ST.uar_full(ST.counts(yU, R[U].argmax(1).cpu().numpy(), np.zeros(len(yU), int)))))
        if t == n_iter:
            break
        newU = torch.softmax(torch.log(f["pi"].clamp(min=1e-12)) + P8.delta(Xp[U], f), -1)
        hist.append(float((newU - R[U]).abs().sum(1).mean()))
        R[U] = newU
    moved = float((R[U].argmax(1) != q0[U].argmax(1)).double().mean())
    return f, dict(hist=hist, pool_hist=pool_hist, moved=moved, R=R)


def emo_eigs(Xp, yp_t, chol):
    """縮みの係数の記述：候補の全ラベルの感情の平均の間の散らばり B（割合 π の重み）について，A⁻¹B の固有値 λ（大きい順に 3 つ）。
    全ラベルでは A = Σ_O + (1 − α)B が厳密に成り立つので，信号の縮みは 1 − (1 − α)λ，向きの誤差の拡大は 1/(1 − (1 − α)λ)（review_ml の第 1 回の M2）。"""
    R = torch.nn.functional.one_hot(yp_t, 4).to(Xp.dtype)
    n = R.sum(0)
    M = (R.T @ Xp) / n[:, None]
    M = M - (n[:, None] * M).sum(0, keepdim=True) / n.sum()
    w = (n / n.sum()).sqrt()
    G = (M @ torch.cholesky_solve(M.T.contiguous(), chol)) * w[:, None] * w[None, :]
    return torch.linalg.eigvalsh((G + G.T) / 2).sort(descending=True).values[:3].cpu().tolist()


def emo_eigs_null(Xp, yp, chol, n_perm=N_PERM):
    """同じ割合でラベルを並べ替えた帰無（n_perm 回）の固有値（大きい順に 3 つ）の平均と，和の最大。"""
    v = [emo_eigs(Xp, torch.as_tensor(np.random.default_rng(SEED + 1 + i).permutation(yp), device=Xp.device), chol) for i in range(n_perm)]
    return np.mean(v, 0).tolist(), float(max(sum(x) for x in v))


def lam_max_sub(muc, chol, alpha):
    """引き算の制約の版 Σ = A − (1 − α)M̂Π̂M̂ᵀ（Π̂ = I/4）が正定値 ⇔ この値 < 1（調べの 6.2 節．記述）。"""
    G = (muc @ torch.cholesky_solve(muc.T.contiguous(), chol)) * (1.0 - alpha) / 4.0
    return float(torch.linalg.eigvalsh((G + G.T) / 2).max())


def _cos_dirs(Wa, Wb):
    """2 つの係数行列 (d, 4) の向きの近さ：感情の平均を引いた各列のコサインの平均（記述）。"""
    a = Wa - Wa.mean(1, keepdims=True)
    b = Wb - Wb.mean(1, keepdims=True)
    num = (a * b).sum(0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return float(np.mean(num / np.maximum(den, 1e-30)))


def run_unit(p3: Path, out: Path, fz: dict, commit: str) -> dict:
    name = P7.unit_name(p3)
    T, c, fs, rs = name.split("_")
    f, r = int(fs[1:]), int(rs[1:])
    dst = out / "rows" / f"{name}.npz"
    if dst.exists():
        raise SystemExit(f"{dst} は既にある（重複を作らない）")
    t0 = time.time()
    F = fz[T]
    srcs = [x for x in DD.CORPORA if x not in (T, c)]
    z1 = np.load(UD / f"{name}_p1.npz")
    if str(z1["commit"]) != F["pseudo_commit"]:
        raise SystemExit(f"{name}_p1 のコミットが固定のファイルの pseudo_commit と違う")
    z8 = np.load(D8 / "rows" / f"{name}.npz", allow_pickle=False)
    if str(z8["commit"]) != D8_COMMIT:
        raise SystemExit(f"{name}：DAI-8 の行のコミットが {D8_COMMIT[:7]} でない")
    u = dai1.Unit(ROOT / "results" / "dai1", T, c, srcs, f, r)
    sid_t = u.cp.df["sample_id"].to_numpy()[u.test].astype(str)
    sid_p = u.cp.df["sample_id"].to_numpy()[u.pool].astype(str)
    if not ((sid_t == z1["sample_test"].astype(str)).all() and (sid_p == z1["sample_pool"].astype(str)).all()):
        raise SystemExit(f"{name}：発話の並びが保存と違う")
    if not (sid_t == z8["test_sample_id"].astype(str)).all():
        raise SystemExit(f"{name}：テストの並びが DAI-8 の保存と違う")
    y_of, cl_of = ST.label_maps(c)
    yt, yp = u.ytest.astype(int), u.ypool.astype(int)
    if not ((y_of(sid_t) == yt).all() and (y_of(sid_p) == yp).all()):
        raise SystemExit(f"{name}：発話の並びとラベルの対応が読み込みと違う")
    if not ((z8["test_y"].astype(int) == yt).all() and (z8["pool_y"].astype(int) == yp).all()):
        raise SystemExit(f"{name}：ラベルが DAI-8 の保存と違う")
    yp_t = torch.as_tensor(yp, device=DEV)
    pred, coef, diag = {}, {}, {}
    checks = dict(ldaK_mismatch=0, ldaK_compared=0, ldaO_mismatch=0, ldaO_compared=0)
    poolpred = {}
    for s in SPACES:
        Xp, Xt = u.Xp[s].to(DEV, DT), u.Xt[s].to(DEV, DT)
        N = Xp.shape[0]
        T_ = Xp.T @ Xp / N                                                   # 候補の全部の散らばり（標準化で平均 0．ラベルに依らない）
        chols = {a: metric_chol(T_, a) for a in ALPHAS}
        cholb = {a: metric_chol(T_, a, BLOCKS[s]) for a in ALPHAS} if s in BLOCKS else {}
        WO = z8[f"C_A_{s}_ldaO_a0.5"].astype(np.float64)
        nm_, nmax_ = emo_eigs_null(Xp, yp, chols[A_RULE])
        diag[f"{s}_emo_eigs_a{A_RULE}"] = dict(eigs=emo_eigs(Xp, yp_t, chols[A_RULE]), eigs_null_mean=nm_, eigs_null_summax=nmax_,
                                              n_over_d=float(N / Xp.shape[1]), n_pool=int(N), dim=int(Xp.shape[1]))
        onehot_all = torch.nn.functional.one_hot(yp_t, 4).to(DT)
        fO = P8.lda_from_R(Xp, onehot_all, Xp.T @ Xp, A_RULE)                  # 候補の全ラベルの LDA（DAI-8 の LDA_O と同じ）
        pO = P8.delta(Xt, fO).argmax(1).cpu().numpy()
        mO = int((pO != z8[f"{s}_ldaO_a0.5"].astype(int)).sum())
        checks["ldaO_mismatch"] += mO
        checks["ldaO_compared"] += len(pO)
        if mO > max(1, int(LDAK_TOL * len(pO))):                             # LDA_SO の Σ_O が DAI-8 の LDA_O の Σ_O と同じことの確かめ（gate-probe の第 2 回の 4）
            raise SystemExit(f"{name} {s}：全ラベルの LDA の判定が DAI-8 の保存と {mO} 本違う")
        pred[f"{s}_ldaOT_a{A_RULE}"] = P8.delta(Xt, lda_metric(fO["mu"], chols[A_RULE])).argmax(1)   # オラクル：全ラベルの平均 × 計量 A
        piC = torch.bincount(yp_t, minlength=4).to(DT) / len(yp)                # 候補の感情の割合（ラベルを使う．LDA_Tcpi は記述のオラクル）
        for K in KS_RULE:
            kpos_t = torch.as_tensor(u.kpos[K], dtype=torch.long, device=DEV)
            ylab = yp_t[kpos_t]
            XL = Xp[kpos_t]
            mu = class_means(XL, ylab)
            muc = mu - mu.mean(0, keepdim=True)                              # 全体の平均（0）に合わせる（LDA_Tc．等しい割合）
            mucp = mu - (piC[:, None] * mu).sum(0, keepdim=True)            # 候補の感情の割合で合わせる（LDA_Tcpi．記述のオラクル）
            RL = torch.nn.functional.one_hot(ylab, 4).to(DT)
            for a in ALPHAS:                                                 # K 本だけの LDA を台本で解く．DAI-8 の保存がある α では照合する
                fka = P8.lda_from_R(XL, RL, XL.T @ XL, a)
                pka = P8.delta(Xt, fka).argmax(1)
                pred[f"K{K}_{s}_ldaK_a{a}"] = pka
                if a in D8_ALPHAS:
                    mism = int((pka.cpu().numpy() != z8[f"K{K}_{s}_ldaK_a{a}"].astype(int)).sum())
                    checks["ldaK_mismatch"] += mism
                    checks["ldaK_compared"] += len(pka)
                    if mism > max(1, int(LDAK_TOL * len(pka))):
                        raise SystemExit(f"{name} {s} K={K} α={a}：K 本だけの LDA の判定が DAI-8 の保存と {mism} 本違う")
                if a == A_RULE:
                    fk = fka
            q0 = torch.softmax(P8.delta(Xp, fk), -1)                         # EM_fromK と同じ初期値
            Um = np.ones(len(yp), bool)
            Um[u.kpos[K]] = False
            fkm, dkm = em_kmetric(Xp, Xp.T @ Xp, kpos_t, ylab, q0, fk["chol"], yU=yp[Um])
            pred[f"K{K}_{s}_emKmet_a{A_RULE}"] = P8.delta(Xt, fkm).argmax(1)
            poolpred[f"K{K}_{s}_emKmet_a{A_RULE}"] = dkm["R"].argmax(1).cpu().numpy()   # 候補の全部の最後の割り当て（gate-code の第 1 回の R9）
            pred[f"K{K}_{s}_ldaSO_a{A_RULE}"] = P8.delta(Xt, lda_metric(mu, fO["chol"])).argmax(1)   # オラクル：K 本の平均 × 候補の全ラベルの群内の Σ
            eye = torch.eye(Xp.shape[1], dtype=DT, device=DEV)
            pred[f"K{K}_{s}_ncm"] = P8.delta(Xt, lda_metric(mu, eye)).argmax(1)   # NCM：Σ = I（標準化した特徴の普通の距離．review_ml の第 1 回の M1）
            fTr = lda_metric(mu, chols[A_RULE])
            qT0 = torch.softmax(P8.delta(Xp, fTr), -1)                         # EM_Amet：計量 A を固定し，LDA_T の確率から割り当てだけを回す（review_ml の m1 の (b)）
            fam, dam = em_kmetric(Xp, Xp.T @ Xp, kpos_t, ylab, qT0, chols[A_RULE], yU=yp[Um])
            pred[f"K{K}_{s}_emAmet_a{A_RULE}"] = P8.delta(Xt, fam).argmax(1)
            poolpred[f"K{K}_{s}_emAmet_a{A_RULE}"] = dam["R"].argmax(1).cpu().numpy()
            for a in ALPHAS:
                ft = lda_metric(mu, chols[a])
                pred[f"K{K}_{s}_ldaT_a{a}"] = P8.delta(Xt, ft).argmax(1)
                pred[f"K{K}_{s}_ldaTc_a{a}"] = P8.delta(Xt, lda_metric(muc, chols[a])).argmax(1)
                pred[f"K{K}_{s}_ldaTcpi_a{a}"] = P8.delta(Xt, lda_metric(mucp, chols[a])).argmax(1)
                if cholb:
                    pred[f"K{K}_{s}_ldaTblk_a{a}"] = P8.delta(Xt, lda_metric(mu, cholb[a])).argmax(1)
                if a == A_RULE:
                    coef[f"K{K}_{s}_ldaT_a{a}"] = ft
                    WT = ft["A"].cpu().numpy()
                    WK = z8[f"C_A_K{K}_{s}_ldaK_a0.5"].astype(np.float64)
                    WE = z8[f"C_A_K{K}_{s}_emfromK_a0.5_b0.0"].astype(np.float64)
                    WM = fkm["A"].cpu().numpy()
                    diag[f"K{K}_{s}"] = dict(cos_ldaT_ldaO=_cos_dirs(WT, WO), cos_ldaK_ldaO=_cos_dirs(WK, WO),
                                             cos_emfromK_ldaO=_cos_dirs(WE, WO), cos_ldaT_emfromK=_cos_dirs(WT, WE),
                                             cos_emKmet_ldaO=_cos_dirs(WM, WO), emKmet_moved=dkm["moved"],
                                             emKmet_hist_last=(dkm["hist"][-1] if dkm["hist"] else 0.0),
                                             emKmet_hist=dkm["hist"], emKmet_pool_hist=dkm["pool_hist"],
                                             emAmet_moved=dam["moved"], emAmet_hist=dam["hist"], emAmet_pool_hist=dam["pool_hist"],
                                             cos_emAmet_ldaO=_cos_dirs(fam["A"].cpu().numpy(), WO),
                                             lam_max_sub=lam_max_sub(muc, chols[a], a))
        for K in KS_RULE:
            for tpl in COPY_K:
                k = tpl.format(K=K, s=s)
                pred[k] = torch.as_tensor(z8[k].astype(np.int64))
        for tpl in COPY_0:
            k = tpl.format(s=s)
            pred[k] = torch.as_tensor(z8[k].astype(np.int64))
        del Xp, Xt, T_, chols, cholb, fO
        torch.cuda.empty_cache()
    (out / "rows").mkdir(parents=True, exist_ok=True)
    cf = {}
    for k, ft in coef.items():
        cf[f"C_A_{k}"] = ft["A"].cpu().numpy().astype(np.float16)
        cf[f"C_mu_{k}"] = ft["mu"].cpu().numpy().astype(np.float16)
        cf[f"C_c_{k}"] = ft["c"].cpu().numpy().astype(np.float32)
    np.savez_compressed(dst, commit=commit, test_sample_id=sid_t, test_y=yt.astype(np.int8), test_cluster=cl_of(sid_t).astype(str),
                        pool_y=yp.astype(np.int8), diag=json.dumps(diag), checks=json.dumps(checks),
                        **{k: v.cpu().numpy().astype(np.int8) for k, v in pred.items()},
                        **{f"P_{k}": np.asarray(v).astype(np.int8) for k, v in poolpred.items()}, **cf)
    del u
    torch.cuda.empty_cache()
    sec = time.time() - t0
    print(f"{name}：{sec:.0f} 秒（照合の不一致：K 本だけの LDA {checks['ldaK_mismatch']} / {checks['ldaK_compared']}，"
          f"全ラベルの LDA {checks['ldaO_mismatch']} / {checks['ldaO_compared']}）", flush=True)
    return dict(name=name, seconds=sec, **checks)


# ------------------------------------------------------------------ 集計

COMPARE = {   # 名前 → (A, B)
    "ldaT_ldaK": ("K{K}_{s}_ldaT_a0.5", "K{K}_{s}_ldaK_a0.5"),
    "emfromK_ldaT": ("K{K}_{s}_emfromK_a0.5_b0.0", "K{K}_{s}_ldaT_a0.5"),
    "emfromK_ldaK": ("K{K}_{s}_emfromK_a0.5_b0.0", "K{K}_{s}_ldaK_a0.5"),
    "ldaT_none": ("K{K}_{s}_ldaT_a0.5", "K{K}_{s}_none"),
    "ldaT_joint": ("K{K}_{s}_ldaT_a0.5", "K{K}_{s}_joint"),
    "em_ldaT": (f"K{{K}}_{{s}}_em_{FX8}", "K{K}_{s}_ldaT_a0.5"),
    "ldaTc_ldaT": ("K{K}_{s}_ldaTc_a0.5", "K{K}_{s}_ldaT_a0.5"),
    "ldaTcpi_ldaT": ("K{K}_{s}_ldaTcpi_a0.5", "K{K}_{s}_ldaT_a0.5"),
    "emKmet_ldaK": ("K{K}_{s}_emKmet_a0.5", "K{K}_{s}_ldaK_a0.5"),          # 割り当ての分（K 本の計量のまま）
    "emfromK_emKmet": ("K{K}_{s}_emfromK_a0.5_b0.0", "K{K}_{s}_emKmet_a0.5"),  # 計量の分（割り当てを回した後）
    "ldaTblk_ldaT": ("K{K}_{s}_ldaTblk_a0.5", "K{K}_{s}_ldaT_a0.5"),
    "ldaSO_ldaK": ("K{K}_{s}_ldaSO_a0.5", "K{K}_{s}_ldaK_a0.5"),             # 群内の計量を良くした分（オラクル）
    "ldaT_ldaSO": ("K{K}_{s}_ldaT_a0.5", "K{K}_{s}_ldaSO_a0.5"),             # 計量に群間の散らばりを含めた分（縮みの損）
    "ldaO_ldaSO": ("{s}_ldaO_a0.5", "K{K}_{s}_ldaSO_a0.5"),                  # 平均を良くした分（群内の計量のもとで）
    "ldaOT_ldaT": ("{s}_ldaOT_a0.5", "K{K}_{s}_ldaT_a0.5"),                  # 平均を良くした分（計量 A のもとで）
    "ldaT_noneT": ("K{K}_{s}_ldaT_a0.5", "K{K}_{s}_noneT"),
    "ldaT_ncm": ("K{K}_{s}_ldaT_a0.5", "K{K}_{s}_ncm"),                      # 物差しを曲げない基準との比べ（review_ml の第 1 回の M1）
    "ldaK_ncm": ("K{K}_{s}_ldaK_a0.5", "K{K}_{s}_ncm"),
    "ldaSO_ncm": ("K{K}_{s}_ldaSO_a0.5", "K{K}_{s}_ncm"),
    "emAmet_ldaT": ("K{K}_{s}_emAmet_a0.5", "K{K}_{s}_ldaT_a0.5"),          # 割り当ての分（計量 A のまま．m1 の (b)）
    "emfromK_emAmet": ("K{K}_{s}_emfromK_a0.5_b0.0", "K{K}_{s}_emAmet_a0.5"),
    **{f"ldaT_a{a}_ldaK_a{a}": (f"K{{K}}_{{s}}_ldaT_a{a}", f"K{{K}}_{{s}}_ldaK_a{a}") for a in ALPHAS if a != A_RULE},
}
COMPARE_0 = {   # K に依らない比べ（K = 0 の行に書く）
    "ldaO_ldaOT": ("{s}_ldaO_a0.5", "{s}_ldaOT_a0.5"),                       # 全ラベルの平均のもとで，群内の Σ と計量 A の差
    "ldaO_oracle": ("{s}_ldaO_a0.5", "{s}_oracle"),
}
RULES = (("G1", "ldaT_ldaK"), ("G2", "emfromK_ldaT"), ("G3n", "ldaT_none"), ("G3j", "ldaT_joint"),
         ("Aa", "ldaSO_ldaK"), ("Aa2", "ldaSO_ncm"), ("Ab", "ldaT_ldaSO"), ("Gn", "ldaT_ncm"))                        # Aa・Ab は仮定の表の (a)・(b)（記述．gate-code の第 2 回の記録 5）
SUB_COMPARE = ("ldaT_ldaK", "emfromK_ldaT", "ldaT_none", "ldaT_joint", "ldaT_ncm")         # 部分集合で並べる比べ（AVT．review_affect の A1 の 3，b3）
CHAN_RULES = (("G1", "ldaT_ldaK"), ("G3n", "ldaT_none"), ("G3j", "ldaT_joint"))  # MOSEI の channel の確かめ（DAI-8 の R1 の添え書きと同じ形）
REC_KEYS = ("ncm", "emAmet_a0.5", "ldaK_a0.5", "ldaT_a0.5", "ldaTc_a0.5", "ldaTcpi_a0.5", "ldaSO_a0.5", "emKmet_a0.5", "emfromK_a0.5_b0.0", "none", "joint")


def summarize(out: Path, allow_partial: bool) -> None:
    files = sorted((out / "rows").glob("*.npz"))
    if len(files) != N_UNITS and not allow_partial:
        raise SystemExit(f"単位の出力が {len(files)} 個（{N_UNITS} のはず）．試走なら --allow-partial")
    rc = {str(np.load(p, allow_pickle=False)["commit"]) for p in files}
    if len(rc) != 1:
        raise SystemExit(f"行のコミットが 1 つでない：{rc}")
    pvf = out / "provenance.json"
    if pvf.exists() and json.loads(pvf.read_text())["commit"] not in rc:
        raise SystemExit(f"行のコミット {rc} が provenance のコミットと違う")
    rng = np.random.default_rng(SEED)
    cells, cellrows, diagrows, chk, recrows = {}, [], [], [], []
    for p in files:
        T, c, fs, rs = p.name[:-4].split("_")
        f, r = int(fs[1:]), int(rs[1:])
        z = np.load(p, allow_pickle=False)
        y, cl, sid = z["test_y"].astype(int), z["test_cluster"].astype(str), z["test_sample_id"].astype(str)
        subs = {nm: np.asarray(mk, bool) for nm, mk in dai3.subset_masks(c, sid, cl).items() if nm in ("channel", "asr_ok", "face_single")}
        for k in z.files:
            if (k.startswith("K") or k.split("_")[0] in SPACES) and z[k].dtype == np.int8 and z[k].shape == y.shape:
                prd = z[k].astype(int)
                cnt = ST.counts(y, prd, cl)
                P8._add(cells.setdefault((c, k), {}), (f, r), cnt, f"{p.name} {k}")
                cellrows.append(dict(c=c, T=T, fold=f, seed=r, key=k, uar=ST.uar_full(cnt)))
                if k.startswith("K") and "_AVT_" in k:
                    for nm, mk in subs.items():
                        if mk.any():
                            P8._add(cells.setdefault((c, f"{k}|sub_{nm}"), {}), (f, r), ST.counts(y[mk], prd[mk], cl[mk]), f"{p.name} {k} {nm}")
                    if any(k.endswith(f"_AVT_{x}") for x in REC_KEYS):        # 感情ごとの再現率（記述．review_affect の b2）
                        rec = cnt[1].sum(0) / np.maximum(cnt[2].sum(0), 1)
                        recrows.append(dict(c=c, T=T, fold=f, seed=r, key=k, **{f"rec_e{e}": float(rec[e]) for e in range(4)},
                                            **{f"pred_share_e{e}": float((prd == e).mean()) for e in range(4)}))
        for k, d in json.loads(str(z["diag"])).items():
            row = dict(c=c, T=T, fold=f, seed=r, key=k)
            for kk, vv in d.items():
                if isinstance(vv, list) and len(vv) <= 3:                     # 固有値などは列に展開する
                    for i, x in enumerate(vv):
                        row[f"{kk}_{i + 1}"] = x
                elif isinstance(vv, list):
                    row[kk] = json.dumps(vv)
                else:
                    row[kk] = vv
            diagrows.append(row)
        chk.append(dict(name=p.name[:-4], **json.loads(str(z["checks"]))))
    P8._finish(cells)
    U = pd.DataFrame(cellrows)
    U.groupby(["c", "key"]).uar.agg(["mean", "count"]).reset_index().to_csv(out / "cells.csv", index=False)
    DG = pd.DataFrame(diagrows)
    DG.to_csv(out / "diag_units.csv", index=False)
    DG.drop(columns=["fold", "seed"]).groupby(["c", "key"]).mean(numeric_only=True).reset_index().to_csv(out / "diag.csv", index=False)
    pd.DataFrame(recrows).to_csv(out / "recall_units.csv", index=False)
    pd.DataFrame(chk).to_csv(out / "checks_units.csv", index=False)
    rows = []
    for c in sorted({k[0] for k in cells}):
        g_eff = ST.g_eff_of(c)[0]
        for s in SPACES:
            for nm, (a, b) in COMPARE_0.items():
                ka, kb = a.format(s=s), b.format(s=s)
                if (c, ka) in cells and (c, kb) in cells:
                    rows.append(dict(c=c, K=0, space=s, compare=nm, subset="all", **ST.effect_stats(cells[(c, ka)], cells[(c, kb)], g_eff, rng)))
            for K in KS_RULE:
                for nm, (a, b) in COMPARE.items():
                    ka, kb = a.format(K=K, s=s), b.format(K=K, s=s)
                    if (c, ka) in cells and (c, kb) in cells:
                        rows.append(dict(c=c, K=K, space=s, compare=nm, subset="all", **ST.effect_stats(cells[(c, ka)], cells[(c, kb)], g_eff, rng)))
                    if s == "AVT" and nm in SUB_COMPARE:
                        for sk in [k for k in cells if k[0] == c and k[1].startswith(f"{ka}|sub_")]:
                            sub = sk[1].split("|sub_")[1]
                            kb2 = (c, f"{kb}|sub_{sub}")
                            if kb2 in cells:
                                rows.append(dict(c=c, K=K, space=s, compare=nm, subset=sub, **ST.effect_stats(cells[sk], cells[kb2], g_eff, rng)))
        print(f"{c}：区間を作った", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(out / "diffs.csv", index=False)
    crit = []
    for c in sorted(D.c.unique()):
        r_ = dict(c=c)
        for tag, nm in RULES:
            d = D[(D.c == c) & (D.space == "AVT") & (D["compare"] == nm) & (D.subset == "all") & D.K.isin(KS_RULE)]
            if not allow_partial and len(d) != len(KS_RULE):
                raise SystemExit(f"{c} {nm}：規則の行が {len(d)} 行（{len(KS_RULE)} のはず）")
            ok = (d.lo > 0) & (d.effect >= MIN_D)
            r_[f"{tag}_n_K"], r_[f"{tag}_n_hi_neg"] = int(ok.sum()), int((d.hi < 0).sum())
            r_[f"{tag}_pos"], r_[f"{tag}_neg"] = bool(ok.sum() >= 3), bool((d.hi < 0).sum() >= 3)
        for K in KS_RULE:                                                    # 2 つの経路の分け方（設計 3 節）
            nms = ("ldaT_ldaK", "emfromK_ldaT", "emfromK_ldaK", "emKmet_ldaK", "emfromK_emKmet")
            g = {nm: D[(D.c == c) & (D.space == "AVT") & (D["compare"] == nm) & (D.subset == "all") & (D.K == K)].effect for nm in nms}
            if all(len(v) == 1 for v in g.values()):
                v = {nm: float(g[nm].iloc[0]) for nm in nms}
                r_[f"dec_K{K}_total"] = v["emfromK_ldaK"]
                r_[f"dec_K{K}_metric_first"], r_[f"dec_K{K}_assign_after"] = v["ldaT_ldaK"], v["emfromK_ldaT"]       # 経路 1：計量 → 割り当て
                r_[f"dec_K{K}_assign_first"], r_[f"dec_K{K}_metric_after"] = v["emKmet_ldaK"], v["emfromK_emKmet"]   # 経路 2：割り当て → 計量
                r_[f"dec_K{K}_interaction"] = v["emfromK_emKmet"] - v["ldaT_ldaK"]
        if c == "MOSEI":                                                     # channel が分かる発話だけの確かめ（review_affect の A1 の 3）
            for tag, nm in CHAN_RULES:
                n_rob = 0
                for K in KS_RULE:
                    da = D[(D.c == c) & (D.space == "AVT") & (D["compare"] == nm) & (D.subset == "all") & (D.K == K)]
                    ds = D[(D.c == c) & (D.space == "AVT") & (D["compare"] == nm) & (D.subset == "channel") & (D.K == K)]
                    if len(da) == 1 and len(ds) == 1 and float(ds.lo.iloc[0]) > 0 and float(ds.effect.iloc[0]) >= 0.5 * float(da.effect.iloc[0]):
                        n_rob += 1
                r_[f"{tag}_chan_n_K_robust"] = n_rob
        crit.append(r_)
    Cr = pd.DataFrame(crit)
    for tag, _ in RULES:
        for x in ("pos", "neg"):
            Cr[f"{tag}_{x}_n_corpora"] = int(Cr[f"{tag}_{x}"].sum())
            Cr[f"{tag}_{x}_n_corpora_excl_IEMOCAP"] = int(Cr[Cr.c != "IEMOCAP"][f"{tag}_{x}"].sum())
    Cr.to_csv(out / "criteria.csv", index=False)
    prov = out / "provenance.json"
    if prov.exists():
        pv = json.loads(prov.read_text())
        cm, dirty = dai1.git_commit()
        pv.update(summarize_commit=cm, summarize_dirty=dirty, rows_commits=sorted(rc), n_units_found=len(files),
                  ldaK_mismatch_total=int(pd.DataFrame(chk).ldaK_mismatch.sum()), boot_seed=SEED, n_boot=ST.N_BOOT)
        prov.write_text(json.dumps(pv, indent=1, ensure_ascii=False))
    print(f"書いた：cells.csv，diffs.csv（{len(D)} 行），criteria.csv，diag.csv，diag_units.csv，checks_units.csv", flush=True)


# ------------------------------------------------------------------ 自己検査

def selftest() -> None:
    g = torch.Generator().manual_seed(0)
    d, n = 30, 400
    X = torch.randn(n, d, generator=g, dtype=DT)
    X = (X - X.mean(0)) / X.std(0)
    y = torch.randint(0, 4, (n,), generator=g)
    T_ = X.T @ X / n
    for a in ALPHAS:
        L = metric_chol(T_, a)
        Aref = (1 - a) * T_ + a * torch.eye(d, dtype=DT)
        assert torch.allclose(L @ L.T, Aref, atol=1e-10)
        mu = class_means(X[:40], y[:40])
        ft = lda_metric(mu, L)
        Wref = torch.linalg.solve(Aref, mu.T)
        assert torch.allclose(ft["A"], Wref, atol=1e-8)
        # 全体の平均に合わせる移動は判定の切片だけを変える：δ_k − δ'_k は x に依らない
        muc = mu - mu.mean(0, keepdim=True)
        d1, d2 = P8.delta(X, ft), P8.delta(X, lda_metric(muc, L))
        diff = (d1 - d1[:, :1]) - (d2 - d2[:, :1])
        assert torch.allclose(diff, diff[:1].expand_as(diff), atol=1e-8)
        # Woodbury：Σ = A − (1 − α)MΠMᵀ で Σ⁻¹M = A⁻¹MC
        R = torch.nn.functional.one_hot(y, 4).to(DT)
        fo = P8.lda_from_R(X, R, X.T @ X, a)
        M, Pi = fo["mu"].T, torch.diag(fo["pi"])
        Sig = Aref - (1 - a) * M @ Pi @ M.T
        C = torch.linalg.inv(torch.eye(4, dtype=DT) - (1 - a) * Pi @ M.T @ torch.linalg.solve(Aref, M))
        assert torch.allclose(torch.linalg.solve(Sig, M), torch.linalg.solve(Aref, M) @ C, atol=1e-8)
        assert torch.allclose(fo["A"], torch.linalg.solve(Sig, M), atol=1e-8)
    # 4 つ目の角：最初の M 段の Σ を渡せば，DAI-8 の em_fit（fix_sigma=True，β = 0）と同じ解になる
    kpos = torch.arange(8)
    ylab = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    yk = y.clone()
    yk[kpos] = ylab
    q0 = torch.softmax(torch.randn(n, 4, generator=g, dtype=DT), -1)
    ref = P8.em_fit(X, X.T @ X, kpos, ylab, q0, A_RULE, 0.0, fix_sigma=True)
    R0 = q0.clone()
    R0[kpos] = 0.0
    R0[kpos, ylab] = 1.0
    ch0 = P8.lda_from_R(X, R0, X.T @ X, A_RULE)["chol"]
    fk_, _ = em_kmetric(X, X.T @ X, kpos, ylab, q0, ch0)
    assert torch.allclose(fk_["A"], ref["fit"]["A"], atol=1e-8) and torch.allclose(fk_["c"], ref["fit"]["c"], atol=1e-8)
    # 固有値の和は DAI-8 の _top3 と同じ
    Lr = metric_chol(T_, A_RULE)
    assert abs(sum(emo_eigs(X, y, Lr)) - P8._top3(X, y.numpy(), Lr)) < 1e-8
    # 引き算の制約の版：Σ = A − (1 − α)M̂(I/4)M̂ᵀ が正定値 ⇔ λ_max < 1
    for scale in (0.3, 3.0):
        mu_ = class_means(X[:8], y[:8].clone().fill_(0) + torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])) * scale
        muc_ = mu_ - mu_.mean(0, keepdim=True)
        Lr = metric_chol(T_, A_RULE)
        lm = lam_max_sub(muc_, Lr, A_RULE)
        Sig_ = Lr @ Lr.T - (1 - A_RULE) * muc_.T @ muc_ / 4.0
        assert (lm < 1) == bool(torch.linalg.eigvalsh(Sig_).min() > 0)
    # α = 1 の LDA_T，α = 1 の LDA_K，NCM の判定は一致する（review_ml の第 1 回の M1）．K = 1 の LDA_K も NCM と同じ
    Xk, yk_ = X[:20], torch.arange(20) % 4
    mu1 = class_means(Xk, yk_)
    eye = torch.eye(d, dtype=DT)
    p_ncm = P8.delta(X, lda_metric(mu1, eye)).argmax(1)
    p_T1 = P8.delta(X, lda_metric(mu1, torch.linalg.cholesky(1e-12 * T_ + 1.0 * eye))).argmax(1)
    p_K1 = P8.delta(X, P8.lda_from_R(Xk, torch.nn.functional.one_hot(yk_, 4).to(DT), Xk.T @ Xk, 1.0)).argmax(1)
    assert (p_ncm == p_T1).all() and (p_ncm == p_K1).all()
    X1, y1 = X[:4], torch.arange(4)
    p_K1b = P8.delta(X, P8.lda_from_R(X1, torch.nn.functional.one_hot(y1, 4).to(DT), X1.T @ X1, 0.5)).argmax(1)
    assert (p_K1b == P8.delta(X, lda_metric(class_means(X1, y1), eye)).argmax(1)).all()
    blk = (("a", 10), ("b", 20))
    Lb = metric_chol(T_, 0.5, blk)
    Ab = Lb @ Lb.T
    assert float(Ab[:10, 10:].abs().max()) < 1e-12 and torch.allclose(Ab[:10, :10], 0.5 * T_[:10, :10] + 0.5 * torch.eye(10, dtype=DT))
    print("自己検査：通った（計量の Cholesky，係数の閉じた形，切片だけの移動，Woodbury，4 つ目の角と em_fit の一致，固有値と _top3 の一致，引き算の版の正定値と λ_max，α = 1 と K = 1 で NCM と一致，ブロック対角）")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "dai10_probe")
    ap.add_argument("--units", default=None, help="単位の名前をコンマで（試走）")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    out = a.out if a.out.is_absolute() else ROOT / a.out
    if a.summarize:
        summarize(out, a.allow_partial)
        return
    commit, dirty = dai1.git_commit()
    fz = json.loads(FROZEN.read_text())
    p3s = sorted(UD.glob("*_p3.npz"))
    partial = bool(a.units)
    if partial:
        want = set(a.units.split(","))
        p3s = [p for p in p3s if P7.unit_name(p) in want]
    else:
        if dirty:
            raise SystemExit("作業木に未コミットの変更がある（本計算はコミットしてから）")
        if len(p3s) != N_UNITS:
            raise SystemExit(f"入力の単位が {len(p3s)} 個（{N_UNITS} のはず）")
        if out.exists():
            raise SystemExit(f"{out} は既にある（本計算は空の出力先に書く）")
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(commit=commit, dirty=dirty, partial=partial, started=time.strftime("%F %T"), python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                script_sha256=sha(Path(__file__)), dai8_sha256=sha(HERE / "dai8_probe.py"), frozen_sha256=sha(FROZEN),
                alphas=ALPHAS, a_rule=A_RULE, min_d=MIN_D, seed=SEED, d8_commit=D8_COMMIT, ldak_tol=LDAK_TOL,
                units=[P7.unit_name(p) for p in p3s],
                inputs_sha256={P7.unit_name(p): dict(p1=sha(UD / f"{P7.unit_name(p)}_p1.npz"),
                                                     d8=sha(D8 / "rows" / f"{P7.unit_name(p)}.npz")) for p in p3s})
    pname = f"provenance_partial_{time.strftime('%Y%m%d-%H%M%S')}.json" if partial else "provenance.json"
    (out / pname).write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"開始 {prov['started']} コミット {commit[:7]} 単位 {len(p3s)} 個", flush=True)
    runs = [run_unit(p, out, fz, commit) for p in p3s]
    prov["runs"] = runs
    prov["finished"] = time.strftime("%F %T")
    (out / pname).write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"{len(runs)} 単位を終えた", flush=True)


if __name__ == "__main__":
    main()
