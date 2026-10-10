"""【探索的】E28 の 6 節：向きの改善の余地 r = own_d/d_s − 1 を，K 本の量で見積もれるか（T6-1〜T6-3）。新しい学習はしない。

r は E13（`OUT/e13_pairs.csv`，6 対の平均）。K 本の量は，ソースのみの保存した点数（`results/shift_<kind>/logits/`）と
地図と同じ K 本（`explore_direction.map_draw`）から作る。ラベル無しの量は TV（`OUT/e27_pairs.csv`）と確信（最大確率の平均）。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_room.py

出力：`OUT/e28t6_pairs.csv`（組ごとの r と K 本の量），`OUT/e28t6_tests.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_direction import map_draw  # noqa: E402
from explore_synergy import KS, PAIRS  # noqa: E402
from pair_table import OUT  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
REPS = range(10)
NAMES = ["neutral", "happy", "sad", "angry"]


def d_hat(L: np.ndarray, y: np.ndarray) -> float:
    """K 本の点数から測った d（6 対の平均．K ≥ 5）。"""
    out = []
    for k, j in PAIRS:
        zk, zj = L[y == k, k] - L[y == k, j], L[y == j, k] - L[y == j, j]
        sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
        out.append((zk.mean() - zj.mean()) / sd if sd > 0 else np.nan)
    return float(np.nanmean(out))


def main() -> int:
    e13 = pd.read_csv(OUT / "e13_pairs.csv")
    e13 = e13[e13.kind.isin(KINDS)]
    e13["room"] = e13.own_d / e13.d - 1
    room = e13.groupby(["kind", "target", "source"]).agg(room=("room", "mean"), d_s=("d", "mean"), own_d=("own_d", "mean")).reset_index()
    e27 = pd.read_csv(OUT / "e27_pairs.csv")[["kind", "target", "source", "tv_free"]]
    rows = []
    for kind in KINDS:
        groups = groups_for(kind)
        dfs = {g: load_any(g, kind)[0] for g in groups}
        for s in groups:
            z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
            for t in groups:
                if t == s:
                    continue
                df = dfs[t]
                y = df.label_4.to_numpy()
                L = z[f"t_{t}"].astype(np.float64)
                P = softmax(L, axis=1)
                rec = {"kind": kind, "target": t, "source": s, "same_corpus": GROUP_SPEC[t][0] == GROUP_SPEC[s][0],
                       "conf": float(P.max(1).mean())}
                for K in KS:
                    rh, dh = [], []
                    for r in REPS:
                        _, idx = map_draw(df, t, r, int(K))
                        pred = L[idx].argmax(1)
                        rh.append(float(np.mean([np.mean(pred[y[idx] == k] == k) for k in range(4)])))
                        if int(K) >= 5:
                            dh.append(d_hat(L[idx], y[idx]))
                    rec[f"R_K{K}"] = float(np.mean(rh))
                    rec[f"R_K{K}_rep0"] = rh[0]
                    rec[f"R_K{K}_sd"] = float(np.std(rh, ddof=1))
                    if dh:
                        rec[f"dhat_K{K}"] = float(np.nanmean(dh))
                rows.append(rec)
        print(f"  {kind} 済み", flush=True)
    A = pd.DataFrame(rows).merge(room, on=["kind", "target", "source"]).merge(e27, on=["kind", "target", "source"], how="left")
    A.to_csv(OUT / "e28t6_pairs.csv", index=False)
    tests = []
    for kind in KINDS:
        a = A[(A.kind == kind) & ~A.same_corpus].copy()
        dm = lambda col: a[col] - a.groupby("target")[col].transform("mean")  # noqa: E731
        r1, p1 = stats.spearmanr(dm("R_K5"), dm("room"))
        tests.append({"kind": kind, "test": "T6-1", "quantity": "R̂_5（ターゲットの効果を除く）", "rho": float(r1), "p": float(p1),
                      "holds": r1 <= -0.5, "n": len(a)})
        r2, p2 = stats.spearmanr(a.R_K5, a.room)
        tests.append({"kind": kind, "test": "T6-2", "quantity": "R̂_5", "rho": float(r2), "p": float(p2), "holds": r2 <= -0.3, "n": len(a)})
        for q in ["tv_free", "conf", "R_K1", "R_K10", "R_K5_rep0", "R_K1_rep0", "dhat_K5", "dhat_K10"]:
            r3, p3 = stats.spearmanr(a[q], a.room, nan_policy="omit")
            r4, _ = stats.spearmanr(dm(q), dm("room"), nan_policy="omit")
            tests.append({"kind": kind, "test": "T6-3", "quantity": q, "rho": float(r3), "p": float(p3), "rho_within_target": float(r4), "n": len(a)})
        for q in ["dhat_K5", "dhat_K10", "R_K5"]:
            tests.append({"kind": kind, "test": "T6-3 d_s との相関", "quantity": q, "rho": float(stats.spearmanr(a[q], a.d_s, nan_policy="omit")[0]), "n": len(a)})
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "e28t6_tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
