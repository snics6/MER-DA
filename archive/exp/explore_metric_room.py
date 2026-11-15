"""【探索的】E38 の下見（CPU．新しい学習なし）：ラベル無しのターゲットの散らばりで計量を変えると，K 本からの向きの推定は良くなるか。

ターゲットのテスト以外の発話（ラベル無し）の共分散 Σ（ソースの統計で標準化した中心合わせ後の座標）で M_ρ = (1 − ρ)Σ/τ + ρI
（τ は Σ の固有値の平均）を作り，K 本（感情ごとに K．`compare_b1.Ctx` と同じ K 本）の感情の平均の差 m̂ から向き u = M_ρ⁻¹ m̂ を作る。
ρ = 1 は中心の差そのもの（等方の計量），ρ → 0 はラベル無しの散らばりで白色化した向き。テストの発話で感情の対ごとの d を測り，
反復 0〜9 で平均する（ターゲットのみの学習器の向きの部分だけを見る．下書きとの組み合わせは見ない）。

    ./.venv/bin/python exp/explore_metric_room.py

出力：`results/explore_relation/e38pre_curve.csv`（種類 × ターゲット × K × ρ），`e38pre_summary.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import explore_compare as EC  # noqa: E402
import explore_source_value as SV  # noqa: E402
from data import load_any  # noqa: E402
from explore_synergy import PAIRS  # noqa: E402

RHOS = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.03]
KS = (5, 10, 25)


def sep(z, y, k, l):
    a, b = z[y == k], z[y == l]
    return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))


def run(kind: str) -> list[dict]:
    rows = []
    for f in sorted((ROOT / "results" / f"multisource_scores_{kind}").glob("*.npz")):
        t = f.stem
        zm = np.load(f)
        srcs = [str(s) for s in zm["src_names"]]
        Xs = np.concatenate([load_any(s, kind)[1] for s in srcs])
        mu, sd = Xs.mean(0), Xs.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        del Xs
        df, X = load_any(t, kind)
        y_all = df["label_4"].to_numpy()
        acc = {(K, rho): [] for K in KS for rho in RHOS}
        for r in EC.REPS:
            if f"test_idx_r{r}" not in zm.files:
                continue
            te = np.zeros(len(y_all), bool)
            te[zm[f"test_idx_r{r}"]] = True
            pool = np.flatnonzero(~te)
            Z = (X - X[pool].mean(0) - mu * 0) / sd                        # 中心合わせ（テスト以外の平均）後，ソースの標準偏差で割る
            Zp, Zt, yp, yt = Z[pool], Z[te], y_all[pool], y_all[te]
            S = np.cov(Zp, rowvar=False)
            lam, V = np.linalg.eigh(S)
            lam = np.clip(lam, 0, None)
            tau = lam.mean()
            rng = np.random.default_rng(1000 + r)
            for K in KS:
                pick = np.concatenate([rng.choice(np.flatnonzero(yp == c), K, replace=False) for c in range(4)])
                m = np.stack([Zp[pick][yp[pick] == c].mean(0) for c in range(4)])
                for rho in RHOS:
                    inv = 1.0 / ((1 - rho) * lam / tau + rho)
                    ds = []
                    for k, l in PAIRS:
                        u = V @ (inv * (V.T @ (m[k] - m[l])))
                        ds.append(sep(Zt @ u, yt, k, l))
                    acc[(K, rho)].append(np.mean(ds))
        for (K, rho), v in acc.items():
            rows.append({"kind": kind, "target": t, "K": K, "rho": rho, "d": float(np.mean(v))})
        print(f"  {kind} {t}", flush=True)
    return rows


def main() -> int:
    C = pd.DataFrame([r for k in SV.MAIN for r in run(k)])
    S = C.groupby(["kind", "K", "rho"]).d.median().unstack("rho")
    C.to_csv(SV.OUT / "e38pre_curve.csv", index=False)
    S.to_csv(SV.OUT / "e38pre_summary.csv")
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
