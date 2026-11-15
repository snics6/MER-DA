"""【探索的】E28 の 5 節：平均の差 δ を「感情の割合の差」δ_mix と残り δ_res に分け，割合の差を除いた中心合わせを試す（T5-1〜T5-3）。

ソースのみの分類器（全ソース，C=0.01）だけ GPU で学び，ターゲットの特徴を v だけずらして当てる：
v = 0（ソースのみ），δ（ふつうの中心合わせ），δ − δ_mix（割合の差を除いた中心合わせ．π_t は EM とオラクルの 2 通り）。
δ_mix = Σ_k (π_t,k − π_s,k) m_{s,k}（m_{s,k} はソースの感情 k の中心）。大きさは E15 の見分けやすさと同じ重み G で測る。
中心合わせの得（E25 の手 2 − ソースのみ）は `OUT/e27_pairs.csv` から取る。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_label_shift.py

出力：`OUT/e28t5_pairs.csv`，`OUT/e28t5_tests.csv`
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import logreg  # noqa: E402
import map as M  # noqa: E402
import shift  # noqa: E402
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_direction import map_draw  # noqa: E402
from explore_synergy import N_CLASSES, kappa, src_parts  # noqa: E402
from pair_table import OUT  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
C_SRC = 0.01
REPS = range(10)


def em_prior(P: np.ndarray, iters: int = 200) -> np.ndarray:
    """Saerens ら 2002：ソースのみの（感情を均等にした）事後確率から，ターゲットの感情の割合を推定する。"""
    pi = np.full(N_CLASSES, 1 / N_CLASSES)
    for _ in range(iters):
        q = P * (pi / (1 / N_CLASSES))
        q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - pi).max() < 1e-8:
            pi = new
            break
        pi = new
    return pi


def g_metric(sp: dict, sd: np.ndarray, C: float):
    V, ev, a = sp["V"], sp["ev"], sp["a"]
    def g(v_raw: np.ndarray) -> float:
        v = v_raw / sd
        return float(a * (((V.T @ v) ** 2) / (ev + 1 / C)).sum())
    return g


def main() -> int:
    t0 = time.time()
    e27 = pd.read_csv(OUT / "e27_pairs.csv")[["kind", "target", "source", "gain", "tv_free"]]
    rows = []
    for kind in KINDS:
        groups = groups_for(kind)
        data = {g: load_any(g, kind) for g in groups}
        for s in groups:
            dfs, Xs = data[s]
            ys = dfs.label_4.to_numpy()
            pi_s = np.bincount(ys, minlength=N_CLASSES) / len(ys)
            m_s = np.stack([Xs[ys == k].mean(0) for k in range(N_CLASSES)])
            mu_s = Xs.mean(0)
            sd = Xs.std(0); sd = np.where(sd > 1e-12, sd, 1.0)
            ks_all = kappa(np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")["oof"])
            G = g_metric(src_parts(((Xs - mu_s) / sd).astype(np.float64), ys, ks_all, 5), sd, C_SRC)
            Xs_t = logreg.to_device(Xs)
            clf = logreg.fit(Xs_t, M.labels(ys), (C_SRC,))
            for t in groups:
                if t == s:
                    continue
                dft, Xt = data[t]
                yt = dft.label_4.to_numpy()
                pi_t = np.bincount(yt, minlength=N_CLASSES) / len(yt)
                Xt_t = logreg.to_device(Xt)
                P = softmax(clf.logits(Xt_t, C_SRC).cpu().numpy(), axis=1)
                pi_em = em_prior(P)
                delta = Xt.mean(0) - mu_s
                mix_or = ((pi_t - pi_s)[:, None] * m_s).sum(0)
                mix_em = ((pi_em - pi_s)[:, None] * m_s).sum(0)
                shifts = {"src": np.zeros_like(delta), "center": delta, "bal_em": delta - mix_em, "bal_or": delta - mix_or}
                u = {k: [] for k in shifts}
                for r in REPS:
                    te, _ = map_draw(dft, t, r, 1)
                    y_te = yt[te]
                    for name, v in shifts.items():
                        L = clf.logits(Xt_t[torch_idx(te)] - logreg.to_device(v)[None, :], C_SRC).cpu().numpy()
                        u[name].append(shift.uar(y_te, L.argmax(1)))
                rows.append({"kind": kind, "target": t, "source": s, "same_corpus": GROUP_SPEC[t][0] == GROUP_SPEC[s][0],
                             "G_delta": G(delta), "G_mix": G(mix_or), "G_res": G(delta - mix_or), "G_mix_em": G(mix_em),
                             "norm2_delta": float(((delta / sd) ** 2).sum()), "norm2_mix": float(((mix_or / sd) ** 2).sum()),
                             "norm2_res": float((((delta - mix_or) / sd) ** 2).sum()),
                             "tv_pi": 0.5 * float(np.abs(pi_t - pi_s).sum()), "tv_pi_em": 0.5 * float(np.abs(pi_em - pi_s).sum()),
                             "tv_pi_em_err": 0.5 * float(np.abs(pi_em - pi_t).sum()),
                             **{f"uar_{k}": float(np.mean(v)) for k, v in u.items()}})
            print(f"  {kind} ソース {s} 済み", flush=True)
    A = pd.DataFrame(rows).merge(e27, on=["kind", "target", "source"], how="left")
    A["bal_em_minus_center"] = A.uar_bal_em - A.uar_center
    A["bal_or_minus_center"] = A.uar_bal_or - A.uar_center
    A["center_minus_src"] = A.uar_center - A.uar_src
    A.to_csv(OUT / "e28t5_pairs.csv", index=False)
    tests = []
    for kind in KINDS:
        a = A[(A.kind == kind) & ~A.same_corpus]
        r_res, p_res = stats.spearmanr(a.gain, a.G_res)
        r_mix, _ = stats.spearmanr(a.gain, a.G_mix)
        r_del, _ = stats.spearmanr(a.gain, a.G_delta)
        tests.append({"kind": kind, "test": "T5-1", "rho_res": r_res, "rho_mix": r_mix, "rho_delta": r_del, "p_res": p_res,
                      "holds": r_res >= 0.5 and r_res > r_mix, "n": len(a)})
        m = a.bal_em_minus_center.mean()
        r_tv, p_tv = stats.spearmanr(a.bal_em_minus_center, a.tv_pi)
        tests.append({"kind": kind, "test": "T5-2", "mean_bal_em_minus_center": m, "n_pos": int((a.bal_em_minus_center > 0).sum()),
                      "rho_tv": r_tv, "p_tv": p_tv, "holds": m > 0 and r_tv > 0.3, "n": len(a),
                      "mean_center_minus_src": a.center_minus_src.mean(), "mean_center_minus_src_e25": a.gain.mean()})
        harmed = a[a.gain < -0.005]
        tests.append({"kind": kind, "test": "T5-3", "mean_bal_or_minus_center": a.bal_or_minus_center.mean(),
                      "n_pos_or": int((a.bal_or_minus_center > 0).sum()), "em_recovers": a.bal_em_minus_center.mean() / a.bal_or_minus_center.mean()
                      if a.bal_or_minus_center.mean() != 0 else np.nan, "tv_pi_em_err_mean": a.tv_pi_em_err.mean(),
                      "share_mix_in_delta": (a.G_mix / a.G_delta).median(), "harmed_n": len(harmed),
                      "harmed_share_mix": (harmed.G_mix / harmed.G_delta).median() if len(harmed) else np.nan,
                      "harmed_bal_em_minus_center": harmed.bal_em_minus_center.mean() if len(harmed) else np.nan})
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "e28t5_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"{len(A)} 組，{time.time() - t0:.0f} 秒")
    print(T.round(4).to_string(index=False))
    return 0


def torch_idx(mask: np.ndarray):
    import torch
    return torch.as_tensor(np.flatnonzero(mask), device=logreg.DEVICE)


if __name__ == "__main__":
    raise SystemExit(main())
