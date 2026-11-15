"""【探索的】E19：位置の直り λ は感情の対によるか。

予言は `docs/exploratory-analyses.md` E19 の 1 節（計算の前に書いた）。新しい学習はしない。
段階 B の行（`OUT/e15b_rows.csv`）から，種類 × 組 × K × 感情の対ごとに測った λ を作り，
「組の効果 ＋ 感情の対の効果 ＋ 残り」に分ける（L1）。感情の対ごとの λ と組ごとの式の λ（`OUT/e15_lambda.csv`
の `lam_pred_rep09`）の順位相関を見る（L2）。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_lambda_pairs.py

出力：`OUT/e19_rows.csv`，`OUT/e19_tests.csv`，`OUT/e19_pair_effects.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
from pair_table import OUT  # noqa: E402

KINDS = {"audio_wavlm": "W7", "audio_hubert": "Hu7", "video": "V", "fuse-audio-video": "AV"}
EMO = ["平静", "喜び", "悲しみ", "怒り"]
O_MIN, MIN_REPS = 0.3, 3


def shares(d: pd.DataFrame) -> dict:
    """λ = 平均 ＋ 組の効果 ＋ 感情の対の効果 ＋ 残り（足し算．欠けがあるので最小二乗）。各部分の二乗和の割合。"""
    cp = pd.get_dummies(d.cp, drop_first=True, dtype=float)
    ep = pd.get_dummies(d.pair, drop_first=True, dtype=float)
    y = d.lam_obs.to_numpy()
    def fit(*blocks):
        X = np.column_stack([np.ones(len(y))] + [b.to_numpy() for b in blocks])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return X @ beta, beta
    tot = ((y - y.mean()) ** 2).sum()
    f_cp, _ = fit(cp)
    f_ep, _ = fit(ep)
    f_both, beta = fit(cp, ep)
    r2 = lambda f: 1 - ((y - f) ** 2).sum() / tot  # noqa: E731
    # 種類 III に近い分け方：もう一方を入れた後に足したときの増え分
    eff = dict(zip(["0-1"] + list(ep.columns), [0.0] + list(beta[-ep.shape[1]:])))
    return {"r2_both": r2(f_both), "share_cp": r2(f_both) - r2(f_ep), "share_ep": r2(f_both) - r2(f_cp),
            "r2_cp_alone": r2(f_cp), "r2_ep_alone": r2(f_ep), "effects": eff}


def main() -> int:
    R = pd.read_csv(OUT / "e15b_rows.csv", dtype={"K": str})
    lam = pd.read_csv(OUT / "e15_lambda.csv", dtype={"K": str})[["kind", "target", "source", "K", "lam_pred_rep09"]]
    rows = []
    for key, g in R.groupby(["kind", "target", "source", "K", "pair"]):
        v = g[g.o_s.abs() >= O_MIN]
        if len(v) < MIN_REPS:
            continue
        rows.append({**dict(zip(["kind", "target", "source", "K", "pair"], key)), "n_reps": len(v),
                     "lam_obs": 1 - float((np.sign(v.o_s) * v.o_m).sum() / v.o_s.abs().sum()),
                     "abs_o_s": float(v.o_s.abs().mean())})
    D = pd.DataFrame(rows).merge(lam, on=["kind", "target", "source", "K"])
    D["cp"] = D.target + "←" + D.source
    D.to_csv(OUT / "e19_rows.csv", index=False)
    T, E = [], []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        sh = shares(g)
        r, p = stats.spearmanr(g.lam_pred_rep09, g.lam_obs, alternative="greater")
        # 組の中での感情の対ごとの λ の散らばり（標準偏差の平均）と，組の平均の散らばり
        within = g.groupby("cp").lam_obs.std(ddof=1).mean()
        between = g.groupby("cp").lam_obs.mean().std(ddof=1)
        T.append({"setting": KINDS[kind], "K": K, "n": len(g), "n_cp": g.cp.nunique(),
                  **{k: v for k, v in sh.items() if k != "effects"},
                  "spearman_pred": r, "p_pred": p, "sd_within_cp": within, "sd_between_cp": between})
        for pr, eff in sh["effects"].items():
            a, b = map(int, pr.split("-"))
            E.append({"setting": KINDS[kind], "K": K, "pair": pr, "name": f"{EMO[a]}-{EMO[b]}",
                      "effect_vs_0-1": eff, "n": int((g.pair == pr).sum()),
                      "mean_lam_obs": g[g.pair == pr].lam_obs.mean()})
    T, E = pd.DataFrame(T), pd.DataFrame(E)
    T.to_csv(OUT / "e19_tests.csv", index=False)
    E.to_csv(OUT / "e19_pair_effects.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
    print(E[E.K == "5"].pivot_table(index="name", columns="setting", values="mean_lam_obs").round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
