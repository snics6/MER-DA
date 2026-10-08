"""【探索的】E3〜E5（ソースの選び方）を，音声から RAVDESS を除いた 6 群でやり直す（E7）。

E6 で，音声の「組に固有の部分」の 60% が RAVDESS を含む 12 組から来ていた。音声で距離を足すと
ソースの選び方がよくなった結果（E3・E4・E5）も RAVDESS の組から来ていたのかを，映像と同じ 6 群
（IEMOCAP-impro，IEMOCAP-script，MOSEI，MELD，AFEW，CREMA-D）だけで計算し直して確かめる。
方法は E3（`explore_select.py`），E4（`explore_metrics.py`），E5（`explore_strategies.py`）と同じ。

    ./.venv/bin/python exp/side/explore_subset.py

出力：`results/explore_relation/e7_e34_single.csv`・`e7_e34_nested.csv`・`e7_e5_summary.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # exp/side/ から見たリポジトリの根
sys.path.insert(0, str(ROOT / "exp"))
import explore_strategies as ES  # noqa: E402
from explore_metrics import K_ORDER, loto, score, table  # noqa: E402
from explore_select import fit_predict  # noqa: E402
from pair_table import OUT  # noqa: E402

SIX = ["IEMOCAP-impro", "IEMOCAP-script", "MOSEI", "MELD", "AFEW", "CREMA-D"]
KIND = "audio_wavlm"


def e34(kind: str, groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d, cands = table(kind)
    d = d[d.target.isin(groups) & d.source.isin(groups)]
    options = [None] + cands
    single, nested = [], []
    for K in K_ORDER:
        dk = d.rename(columns={K: "gain"}).copy()
        dk["rsrc_K"], dk["dce_K"] = 1 - dk[f"uar_t_K{K}"], dk[f"dce_K{K}"]
        targets = sorted(dk.target.unique())
        for m in options:
            for t, rho, top1, reg, tb, tw in loto(dk, targets, [] if m is None else [m]):
                single.append({"K": K, "metric": m or "none", "target": t, "rho": rho,
                               "top1": top1, "regret": reg, "tau_b": tb, "tau_w": tw})
        for t in targets:
            inner = dk[(dk.target != t) & (dk.source != t)]
            inner_targets = [u for u in targets if u != t]
            best, best_rho = None, -np.inf
            for m in options:
                r = np.mean([x[1] for x in loto(inner, inner_targets, [] if m is None else [m])])
                if r > best_rho:
                    best, best_rho = m, r
            test = dk[dk.target == t]
            rho, top1, reg, tb, tw = score(fit_predict(inner, test, "gain",
                                                       [] if best is None else [best]),
                                           test.gain.to_numpy())
            nested.append({"K": K, "target": t, "chosen": best or "none", "rho": rho,
                           "top1": top1, "regret": reg, "tau_b": tb, "tau_w": tw})
    return pd.DataFrame(single), pd.DataFrame(nested)


def main() -> int:
    single, nested = e34(KIND, SIX)
    single.groupby(["K", "metric"])[["rho", "tau_b", "tau_w", "top1", "regret"]].mean().reset_index().to_csv(
        OUT / "e7_e34_single.csv", index=False)
    nested.to_csv(OUT / "e7_e34_nested.csv", index=False)
    e5 = pd.DataFrame(ES.run_kind(KIND, SIX))
    (e5.groupby(["K", "strategy"]).agg(rho=("rho", "mean"), tau_b=("tau_b", "mean"),
                                       tau_w=("tau_w", "mean"), top1=("top1", "mean"),
                                       regret=("regret", "mean"), n_targets=("target", "nunique"))
       .reset_index().to_csv(OUT / "e7_e5_summary.csv", index=False))
    print(f"書いた: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
