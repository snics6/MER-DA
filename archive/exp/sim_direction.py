"""【理論の検算】E16：ニュートン法の 1 歩で，合併の分類器の向き d と位置 o の変化を予言できるか（人工データ）。

導出は `docs/exploratory-analyses.md` E16，計算は `exp/explore_direction.py` の `newton_step`。
人工データの作り方は E15（`exp/sim_position.py` の 4 クラス）と同じ。ターゲットのクラスの中心を角度 θ だけ
回す（θ が大きいほど，ソースの向きとターゲットの最良の向きの差が大きい）。実データではターゲットが見分けやすさの
向きにソースの 20〜70 倍散らばっている（E15 の τ）ので，ターゲットにだけその向きの散らばりを足す設定も置く。

各設定で，ソースのみと合併を sklearn で学び，テストでの d・o を測る。ニュートン法の 1 歩は，ソースのみの
分類器（ソースの統計で標準化した空間），その学習データでの確率から作った曲がり，抜いた K 本だけから計算する。

    ./.venv/bin/python exp/sim_direction.py

出力：`results/sim_position/direction.csv`・`summary_direction.csv`
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
import sim_position as SP  # noqa: E402
from explore_direction import d_o, hs_inv, newton_step  # noqa: E402
from explore_synergy import src_parts  # noqa: E402

THETAS = [0, 45, 70]
GAMMA_O = 1.0
GAMMA_X = [0, 10]
TAU_EXTRA = [0, 30]      # ターゲットだけが見分けやすさの向き u_x に余分に散らばる分（白色化した空間の分散．実データの τ は 20〜70）
KS = [1, 5, 10, 25]
CS = [0.01, 1.0]
REPS = 5
K_CLS = 4


def fit_std(X, y, C, mu, sd):
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, tol=1e-6)
    clf.fit((X - mu) / sd, y)
    return clf


def run_job(rep: int, C: float) -> list[dict]:
    w = SP.world(1000 + rep, K_CLS)
    rng = np.random.default_rng(20_000 * rep + int(C * 1000))
    Xs, ys = SP.draw(rng, w, SP.N_SRC_PER_CLASS * 2 // K_CLS, SP.means(w, 0, rotate=False))
    mu, sd = Xs.mean(0), Xs.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    src = fit_std(Xs, ys, C, mu, sd)
    Zs = (Xs - mu) / sd
    ks = (1 - (src.predict_proba(Zs) ** 2).sum(1)) / (K_CLS - 1)        # 人工データでは学習データでの確率を使う
    xi_tr = rng.standard_normal((K_CLS * SP.K_MAX, SP.P))
    y_tr = np.tile(np.arange(K_CLS), SP.K_MAX)
    xi_te = rng.standard_normal((K_CLS * SP.N_TEST_PER_CLASS, SP.P))
    y_te = np.repeat(np.arange(K_CLS), SP.N_TEST_PER_CLASS)
    n_src_c = np.bincount(ys, minlength=K_CLS)
    rows = []
    eta_tr = rng.standard_normal(len(y_tr))
    eta_te = rng.standard_normal(len(y_te))
    lx = w["L"] @ w["ux"]["high"]
    for th in THETAS:
        mu_t = SP.means(w, th, rotate=True)
        for gx, tau in [(g, t) for g in GAMMA_X for t in TAU_EXTRA]:
            shift = w["L"] @ (GAMMA_O * w["uo"] + (gx * w["ux"]["high"] if gx > 0 else 0))
            Xtr = xi_tr @ w["L"].T + mu_t[y_tr] + shift + np.sqrt(tau) * eta_tr[:, None] * lx
            Xte = xi_te @ w["L"].T + mu_t[y_te] + shift + np.sqrt(tau) * eta_te[:, None] * lx
            Zte, Ztr = (Xte - mu) / sd, (Xtr - mu) / sd
            Ls = src.decision_function(Zte)
            ds = d_o(Ls, y_te)
            for K in KS:
                sel = np.arange(K_CLS * K)
                sp = src_parts(Zs, ys, ks, K)
                Xa_tr = np.c_[Ztr[sel] - sp["zbar"], np.ones(len(sel))]
                Xa_te = np.c_[Zte - sp["zbar"], np.ones(len(Zte))]
                n_c = n_src_c + K
                swc = n_c.sum() / (K_CLS * n_c)
                P = softmax(src.decision_function(Ztr[sel]), axis=1)
                dW = newton_step(hs_inv(sp, C), Xa_tr, P, y_tr[sel], swc[y_tr[sel]])
                dp = d_o(Ls + Xa_te @ dW.T, y_te)
                Xm, ym = np.vstack([Xs, Xtr[sel]]), np.r_[ys, y_tr[sel]]
                mm, sm = Xm.mean(0), np.where(Xm.std(0) > 1e-12, Xm.std(0), 1.0)
                merged = fit_std(Xm, ym, C, mm, sm)
                dm = d_o(merged.decision_function((Xte - mm) / sm), y_te)
                for (k, j), a, b, c in zip([(a, b) for a in range(4) for b in range(a + 1, 4)], ds, dm, dp):
                    rows.append({"rep": rep, "C": C, "theta": th, "gamma_x": gx, "tau_extra": tau, "K": K, "pair": f"{k}-{j}",
                                 "d_s": a[0], "o_s": a[1], "d_m": b[0], "o_m": b[1], "d_pred": c[0], "o_pred": c[1]})
    return rows


def main() -> int:
    t0 = time.time()
    res = Parallel(n_jobs=8)(delayed(run_job)(r, C) for r in range(REPS) for C in CS)
    d = pd.DataFrame([x for rows in res for x in rows])
    d["dc_obs"], d["dc_pred"] = d.d_m - d.d_s, d.d_pred - d.d_s
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    out = SP.OUT
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "direction.csv", "w") as f:
        f.write("# " + json.dumps({"script": "exp/sim_direction.py", "commit": commit + ("+dirty" if dirty else ""),
                                   "seconds": round(time.time() - t0, 1)}) + "\n")
        d.to_csv(f, index=False)
    v = d[d.o_s.abs() >= SP.O_MIN]
    lam = lambda g, col: 1 - float((np.sign(g.o_s) * g[col]).sum() / g.o_s.abs().sum())  # noqa: E731
    S = (d.groupby(["C", "theta", "gamma_x", "tau_extra", "K"])[["dc_obs", "dc_pred", "d_s"]].mean()
         .join(v.groupby(["C", "theta", "gamma_x", "tau_extra", "K"]).apply(lambda g: pd.Series(
             {"lam_obs": lam(g, "o_m"), "lam_newton": lam(g, "o_pred")}))).reset_index())
    S.to_csv(out / "summary_direction.csv", index=False)
    r_dc = stats.pearsonr(d.dc_pred, d.dc_obs)[0]
    r_cell = stats.pearsonr(S.dc_pred, S.dc_obs)[0]
    print(f"{len(d)} 行，{time.time() - t0:.0f} 秒．d の変化の相関：感情の対ごと {r_dc:.3f}，設定の平均ごと {r_cell:.3f}")
    print(S.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
