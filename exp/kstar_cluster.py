"""A6：IEMOCAP がターゲットの組の K* の区間を，テスト話者の組を単位にした t 区間で作る（`docs/researchvision.md` 3.9 節）。

IEMOCAP-impro はテストの袋が 5 通り，IEMOCAP-script は 7 通りしか無く，30 回の差は独立でない。
そこで，同じテスト話者の組（クラスタ）の反復の差（合併 − ターゲットのみ）を平均してから，クラスタの平均の t 区間
（自由度 ＝ クラスタの数 − 1）を作り，下端が 0 を上回る最大の K を K* とする。比べるため，30 個の差をそのまま使う
今の区間（`exp/summarize_map.py` と同じ）も並べる。ほかのターゲットにも同じ計算をかけ，ずれを確かめる。

    ./.venv/bin/python exp/kstar_cluster.py

出力：`results/explore_relation/kstar_cluster.csv`（地図 × 種類 × 組），`kstar_cluster_summary.csv`
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
import map as M  # noqa: E402
from data import groups_for  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
KS = ["1", "5", "10", "25", "50", "all"]
OUT = ROOT / "results" / "explore_relation"


def lo_t(x: np.ndarray) -> float:
    n = len(x)
    return float(x.mean() - stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)) if n >= 2 else np.nan


def kstar(los: dict) -> str:
    sig = [K for K in KS if K in los and los[K] > 0]
    return sig[-1] if sig else "none"


def main() -> int:
    rows = []
    for kind in KINDS:
        groups = groups_for(kind)
        for t in groups:
            g = M.load(t, kind)
            for mp_name, mp_dir in (("cv", ROOT / "results" / f"map_{kind}"), ("fixc", ROOT / "results" / "fixc_map" / kind)):
                d = pd.read_csv(mp_dir / f"{t}.csv", comment="#", dtype={"K": str, "source": str})
                reps = sorted(d.rep.unique())
                clus = {r: frozenset(np.unique(g.person[M.test_mask(g, r)])) for r in reps}
                cid = {c: i for i, c in enumerate(sorted(set(clus.values()), key=lambda c: sorted(c)))}
                tt = d[d.method == "tgt_only"].set_index(["K", "rep"]).uar
                m = d[d.method == "merged"]
                for s, h in m.groupby("source"):
                    lo_n, lo_c = {}, {}
                    for K, hk in h.groupby("K"):
                        diff = np.array([u - tt[(K, r)] for u, r in zip(hk.uar, hk.rep)])
                        cl = np.array([cid[clus[r]] for r in hk.rep])
                        lo_n[K] = lo_t(diff)
                        cm = np.array([diff[cl == c].mean() for c in np.unique(cl)])
                        lo_c[K] = lo_t(cm)
                    rows.append({"map": mp_name, "kind": kind, "target": t, "source": s, "n_clusters": len(cid),
                                 "K_star_naive": kstar(lo_n), "K_star_cluster": kstar(lo_c),
                                 **{f"lo_naive_K{K}": lo_n.get(K, np.nan) for K in KS},
                                 **{f"lo_cluster_K{K}": lo_c.get(K, np.nan) for K in KS}})
            print(f"  {kind} {t}：済み（テストの組 {len(cid)} 通り）", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "kstar_cluster.csv", index=False)
    R["iemocap"] = R.target.str.startswith("IEMOCAP")
    R["same"] = R.K_star_naive == R.K_star_cluster
    S = R.groupby(["map", "kind", "iemocap"]).agg(n_pairs=("same", "size"), frac_same=("same", "mean"),
                                                   n_clusters_min=("n_clusters", "min")).reset_index()
    S.to_csv(OUT / "kstar_cluster_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))
    print(R[R.iemocap & (R.map == "fixc")][["kind", "target", "source", "n_clusters", "K_star_naive", "K_star_cluster"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
