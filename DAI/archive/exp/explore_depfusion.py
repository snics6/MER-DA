"""探索の解析（事前登録の外．2026-11-07．研究者の承認「よい．先行研究の調べを進めて．同時に，軽くその方法がうまくいきそうか試してみて」）：
単独の分類器（声・顔・言葉）の点数を，ターゲットのモダリティの依存に合わせた行列で組み直す融合が，確率の積と融合の生徒を上回りそうかを，
**仮のターゲットの回**（`results/dai1/pseudo/units/`．評価のターゲット T は単位に入らない）の保存した重みから測る（新しい学習は 9 次元の判別分析だけ）．

単位（T, c, f, r）と K ∈ {0, 1, 5, 10, 25} ごとに：
- 入力：単独の分類器 m の点数 Z_m = X_m W_mᵀ + b_m（4 個）を，感情の平均を引く向きの正規直交の 4×3 の基底 H で 3 個にした z_m．3 つをつないで 9 個．
  点数は 2 組：joint（提案で直した単独の生徒．固定のファイルの主の版・選んだ設定．explore_fusion.py と同じ選び方）と ret（疑似ラベルのモデルの
  因子．段 2 の学び直した単独の分類器）．
- 比べる相手：prod（Σ_m Z_m の argmax ＝ 確率の積），avt_joint（融合の生徒の提案），avt_none（融合の K 本だけ）．
- 組み直し（9 次元の判別分析．判定は感情の事前を等しくした δ_c(z) = μ_cᵀ P z − ½ μ_cᵀ P μ_c，P = (Σ + 1e-3·tr(Σ)/9·I)⁻¹）：
  - lda_K_SWK（K ≥ 5）：μ も Σ も K 本だけ（ラベルだけの対照）．
  - lda_K_ST（K ≥ 1）：μ は K 本，Σ はラベル無しの候補の全体の共分散．
  - lda_Q（全 K）：候補の各発話に積の事後確率 R（K 本の発話は正解の one-hot に置き換える）を付け，μ と同じ感情の中の Σ を R の重みで推す（EM の 1 段）．
  - em（全 K）：lda_Q から始め，共有の共分散の正規分布の混合で EM を 10 回（K 本は正解に固定，感情の割合も推す）．
  - *_blk：同じものの Σ のモダリティをまたぐ 3×3 のブロックを 0 にした版（モダリティごとの直しだけで，依存の直しが無い対照）．
- 診断（仮のターゲットの候補の正解を使う）：感情ごとの，モダリティの組の z の最大の正準相関の平均（rho_AV，rho_AT，rho_VT）．
UAR は 4 感情の再現率の平均（30 本の規則は当てない．探索の記述）．出力：<out>/dep_units.csv，dep_cells.csv．
使い方：python DAI/exp/explore_depfusion.py <out> [<単位の名前の glob>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = HERE.parents[1]
KS = (0, 1, 5, 10, 25)
H = np.linalg.svd(np.eye(4) - 0.25)[0][:, :3]          # 4×3，列は 1 のベクトルに直交
BLK = [slice(0, 3), slice(3, 6), slice(6, 9)]


def uar(y, p):
    return 100.0 * np.mean([np.mean(p[y == k] == k) for k in range(4) if (y == k).any()])


def pick_idx(items, want):
    hits = [j for j, it in enumerate(items) if all(ST.norm(it.get(k)) == ST.norm(v) for k, v in want.items())]
    assert len(hits) == 1, (want, len(hits))
    return hits[0]


def block(Sig):
    B = np.zeros_like(Sig)
    for s in BLK:
        B[s, s] = Sig[s, s]
    return B


def lda_pred(z, mu, Sig):
    P = np.linalg.inv(Sig + 1e-3 * np.trace(Sig) / Sig.shape[0] * np.eye(Sig.shape[0]))
    A = mu @ P                                          # 4×9
    return (z @ A.T - 0.5 * np.sum(A * mu, 1)).argmax(1)


def soft_stats(z, R):
    w = R.sum(0)
    mu = (R.T @ z) / w[:, None]
    Sig = sum((R[:, c:c + 1] * (z - mu[c])).T @ (z - mu[c]) for c in range(4)) / len(z)
    return mu, Sig, w / w.sum()


def em(z, R0, kp, yk, blk, n_iter=10):
    R = R0.copy()
    for _ in range(n_iter):
        mu, Sig, pi = soft_stats(z, R)
        Sig = block(Sig) if blk else Sig
        Sr = Sig + 1e-3 * np.trace(Sig) / 9 * np.eye(9)
        P = np.linalg.inv(Sr)
        L = np.stack([-0.5 * np.einsum("ij,jk,ik->i", z - mu[c], P, z - mu[c]) for c in range(4)], 1) + np.log(np.clip(pi, 1e-9, None))
        L -= L.max(1, keepdims=True)
        R = np.exp(L); R /= R.sum(1, keepdims=True)
        if len(kp):
            R[kp] = np.eye(4)[yk]
    mu, Sig, _ = soft_stats(z, R)
    return mu, block(Sig) if blk else Sig


def max_cc(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    Sa, Sb, Sab = a.T @ a, b.T @ b, a.T @ b
    ia = np.linalg.inv(np.linalg.cholesky(Sa + 1e-9 * np.eye(3)))
    ib = np.linalg.inv(np.linalg.cholesky(Sb + 1e-9 * np.eye(3)))
    return float(np.linalg.svd(ia @ Sab @ ib.T, compute_uv=False)[0])


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(ROOT / "DAI" / "frozen" / "dai1_frozen.json"))
    UD = ROOT / "results" / "dai1" / "pseudo" / "units"
    rows = []
    pat = sys.argv[2] if len(sys.argv) > 2 else "*"            # 試走用の絞り込み（例：AFEW_IEMOCAP_f0_r0）
    for p3 in sorted(UD.glob(f"{pat}_p3.npz")):
        T, c, fs, rs = p3.name[:-len("_p3.npz")].split("_")
        f, r = int(fs[1:]), int(rs[1:])
        F = fz[T]
        srcs = [x for x in DD.CORPORA if x not in (T, c)]
        u = dai1.Unit(ROOT / "results" / "dai1", T, c, srcs, f, r)
        z3, z2 = np.load(p3), np.load(str(p3).replace("_p3.npz", "_p2.npz"))
        for K in KS:
            ver = F["main"][str(K)]
            kp = u.kpos[K]; yk = u.ypool[kp]
            row = dict(T=T, c=c, fold=f, seed=r, K=K)
            j = pick_idx([json.loads(x) for x in z3[f"p3_AVT_K{K}_items"]],
                         dict(method="joint", teacher=ver, **{k: v for k, v in F["select"][f"AVT|{K}|joint|{ver}"].items() if k != "pseudo_uar"}))
            row["avt_joint"] = uar(u.ytest, z3[f"p3_AVT_K{K}_pred"][j])
            row["avt_none"] = uar(u.ytest, z2[f"p2_K{K}_none_AVT_test"].astype(np.float32).argmax(1))
            for src in ("joint", "ret"):
                Zp, Zt = {}, {}
                for m in "AVT":
                    if src == "joint":
                        items = [json.loads(x) for x in z3[f"p3_{m}_K{K}_items"]]
                        jm = pick_idx(items, dict(method="joint", teacher=ver, **{k: v for k, v in F["select"][f"{m}|{K}|joint|{ver}"].items() if k != "pseudo_uar"}))
                        W, b = z3[f"p3_{m}_K{K}_W"][jm], z3[f"p3_{m}_K{K}_b"][jm]
                    else:
                        W, b = z2[f"p2_K{K}_ret_{m}_W"], z2[f"p2_K{K}_ret_{m}_b"]
                    W = torch.tensor(W, dtype=u.Xt[m].dtype, device=u.Xt[m].device)
                    b = torch.tensor(b, dtype=u.Xt[m].dtype, device=u.Xt[m].device)
                    Zp[m] = S.logits(u.Xp[m], W, b).double().cpu().numpy()
                    Zt[m] = S.logits(u.Xt[m], W, b).double().cpu().numpy()
                    row[f"{src}_{m}"] = uar(u.ytest, Zt[m].argmax(1))
                row[f"{src}_prod"] = uar(u.ytest, sum(Zt.values()).argmax(1))
                zp = np.concatenate([Zp[m] @ H for m in "AVT"], 1)
                zt = np.concatenate([Zt[m] @ H for m in "AVT"], 1)
                Lp = sum(Zp.values()); Lp = Lp - Lp.max(1, keepdims=True)
                R = np.exp(Lp); R /= R.sum(1, keepdims=True)
                if len(kp):
                    R[kp] = np.eye(4)[yk]
                muQ, SQ, _ = soft_stats(zp, R)
                ST_ = np.cov(zp.T, bias=True)
                for blk in (False, True):
                    sfx = "_blk" if blk else ""
                    g = block if blk else (lambda x: x)
                    row[f"{src}_lda_Q{sfx}"] = uar(u.ytest, lda_pred(zt, muQ, g(SQ)))
                    mu_e, S_e = em(zp, R, kp, yk, blk)
                    row[f"{src}_em{sfx}"] = uar(u.ytest, lda_pred(zt, mu_e, S_e))
                    if K >= 1:
                        muK = np.stack([zp[kp][yk == k].mean(0) for k in range(4)])
                        row[f"{src}_lda_K_ST{sfx}"] = uar(u.ytest, lda_pred(zt, muK, g(ST_)))
                    if K >= 5:
                        SWK = sum(((zp[kp][yk == k] - muK[k]).T @ (zp[kp][yk == k] - muK[k])) for k in range(4)) / len(kp)
                        row[f"{src}_lda_K_SWK{sfx}"] = uar(u.ytest, lda_pred(zt, muK, g(SWK)))
                if src == "joint" and K == 5:
                    yp = u.ypool
                    for a_, b_ in (("A", "V"), ("A", "T"), ("V", "T")):
                        ia, ib = "AVT".index(a_), "AVT".index(b_)
                        row[f"rho_{a_}{b_}"] = float(np.mean([max_cc(zp[yp == k][:, BLK[ia]], zp[yp == k][:, BLK[ib]]) for k in range(4)]))
            rows.append(row)
        print(p3.name, flush=True)
    D = pd.DataFrame(rows); D.to_csv(out / "dep_units.csv", index=False)
    cols = [x for x in D.columns if x not in ("T", "c", "fold", "seed", "K")]
    C = D.groupby(["c", "K"])[cols].mean().reset_index(); C.insert(2, "n_units", D.groupby(["c", "K"]).size().values)
    C.to_csv(out / "dep_cells.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    print(C.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
