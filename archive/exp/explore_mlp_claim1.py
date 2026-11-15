"""【探索的】E30：非線形の分類器での主張 1 の判定（N1〜N4）。新しい学習なし。

入力：`results/mlp_check_<kind>/<ターゲット>.csv`（UAR）と `.npz`（テストの点数）。同じコーパスの組は除く。

    ./.venv/bin/python exp/explore_mlp_claim1.py

出力：`results/explore_relation/e30_pairs.csv`，`e30_tests.csv`，`e30_summary.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, groups_for  # noqa: E402
from explore_pairs import ba  # noqa: E402
from explore_synergy import PAIRS, d_o  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
KS = ["1", "5", "10"]
WDS = ["0.0001", "0.01"]
OUT = ROOT / "results" / "explore_relation"


def main() -> int:
    rows = []
    for kind in KINDS:
        for t in groups_for(kind):
            d = pd.read_csv(ROOT / "results" / f"mlp_check_{kind}" / f"{t}.csv", dtype={"K": str})
            z = np.load(ROOT / "results" / f"mlp_check_{kind}" / f"{t}.npz")
            d["wd"] = d.wd.map(lambda w: f"{w:g}")
            for s in d.source.unique():
                if GROUP_SPEC[t][0] == GROUP_SPEC[s][0]:
                    continue
                for wd in WDS:
                    w = f"{float(wd):g}"
                    reps = sorted(d[(d.source == s) & (d.method == "src")].rep.unique())
                    U = d[(d.source == s) & (d.wd == w)]
                    src = U[U.method == "src"].set_index("rep").uar
                    srcc = U[U.method == "srcc"].set_index("rep").uar
                    for K in KS:
                        get = lambda m: U[(U.method == m) & (U.K == K)].set_index("rep").uar  # noqa: E731
                        m, mc, d1, d1c = get("m"), get("mc"), get("d1"), get("d1c")
                        pos, dirc, lcrit, ldisc_proxy = [], [], [], []
                        for r in reps:
                            y = z[f"y_test_r{r}"]
                            Ls = z[f"src_wd{w}_{s}_r{r}"].astype(np.float64)
                            Lm = z[f"m_wd{w}_{s}_K{K}_r{r}"].astype(np.float64)
                            for (ds, os_), (dm, om) in zip(d_o(Ls, y), d_o(Lm, y)):
                                pos.append(ba(ds, om) - ba(ds, os_))
                                dirc.append(ba(dm, om) - ba(ds, om))
                                lcrit.append(ba(ds, 0.0) - ba(ds, os_))
                        p_, d_ = 100 * np.mean(pos), 100 * np.mean(dirc)
                        rows.append({"kind": kind, "target": t, "source": s, "wd": wd, "K": K,
                                     "G_merge": 100 * (m - src).mean(), "G_crit": 100 * (d1 - src).mean(),
                                     "G_merge_c": 100 * (mc - srcc).mean(), "G_crit_c": 100 * (d1c - srcc).mean(),
                                     "G_center": 100 * (srcc - src).mean(), "src": 100 * src.mean(), "merged": 100 * m.mean(),
                                     "L_crit_nl": 100 * np.mean(lcrit), "pos": p_, "dir": d_,
                                     "share_pos": p_ / (p_ + d_) if (p_ + d_) != 0 else np.nan})
        print(f"  {kind} 済み", flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "e30_pairs.csv", index=False)
    tests, summ = [], []
    for kind in KINDS:
        for wd in WDS:
            g = P[(P.kind == kind) & (P.wd == wd) & (P.K == "5")]
            main_k = kind in ("audio_wavlm", "audio_hubert")
            r3, p3 = stats.spearmanr(g.G_crit, g.L_crit_nl, alternative="greater")
            tests += [
                {"kind": kind, "wd": wd, "test": "N1", "G_crit": g.G_crit.mean(), "G_merge": g.G_merge.mean(),
                 "ratio": g.G_crit.mean() / g.G_merge.mean(), "holds": (g.G_crit.mean() >= 0.7 * g.G_merge.mean()) if main_k else np.nan},
                {"kind": kind, "wd": wd, "test": "N2", "G_crit": g.G_crit.mean(), "G_crit_c": g.G_crit_c.mean(),
                 "G_merge": g.G_merge.mean(), "G_merge_c": g.G_merge_c.mean(),
                 "holds": (g.G_crit_c.mean() <= g.G_crit.mean() / 2 and g.G_merge_c.mean() <= g.G_merge.mean() / 2) if main_k else np.nan},
                {"kind": kind, "wd": wd, "test": "N3", "rho": float(r3), "p": float(p3), "holds": (r3 > 0.5) if main_k else np.nan},
                {"kind": kind, "wd": wd, "test": "N4", "share_pos_median": float(g.share_pos.median()),
                 "holds": (g.share_pos.median() >= 0.5) if main_k else np.nan},
            ]
            for K in KS:
                h = P[(P.kind == kind) & (P.wd == wd) & (P.K == K)]
                summ.append({"kind": kind, "wd": wd, "K": K, "n": len(h), "src": h.src.mean(), "merged": h.merged.mean(),
                             "G_merge": h.G_merge.mean(), "G_crit": h.G_crit.mean(), "G_center": h.G_center.mean(),
                             "G_merge_c": h.G_merge_c.mean(), "G_crit_c": h.G_crit_c.mean(), "L_crit_nl": h.L_crit_nl.mean(),
                             "pos": h.pos.mean(), "dir": h.dir.mean(), "share_pos_median": h.share_pos.median()})
    T, S = pd.DataFrame(tests), pd.DataFrame(summ)
    T.to_csv(OUT / "e30_tests.csv", index=False)
    S.to_csv(OUT / "e30_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
    print(S.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
