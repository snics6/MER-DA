"""【探索的】E22・E23：どの K 本を引くと得か。話者の数（E22）と，ソースのみが自信を持って外している例（E23）。

予言は `docs/exploratory-analyses.md` E22〜E24 の 1 節（計算の前に書いた）。新しい学習はしない。
C を固定した地図（MERDA_MAP_ROOT=results/fixc_map）の反復ごとの UAR を使う。K 本の抜き方は地図と同じ手順で作り直す。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_draws.py

出力：`OUT/e22_rows.csv`・`OUT/e22_tests.csv`，`OUT/e23_rows.csv`・`OUT/e23_tests.csv`
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
from data import groups_for, load_any  # noqa: E402
from explore_direction import map_draw  # noqa: E402
from explore_synergy import KS, load_map  # noqa: E402
from pair_table import OUT  # noqa: E402

KINDS = {"audio_wavlm": "W7", "audio_hubert": "Hu7", "video": "V", "fuse-audio-video": "AV"}
CW = 0.05


def demeaned_spearman(g: pd.DataFrame, by: list[str], x: str, y: str) -> tuple[float, float, int]:
    u = g[y] - g.groupby(by)[y].transform("mean")
    v = g[x] - g.groupby(by)[x].transform("mean")
    ok = v.abs() > 1e-12
    if ok.sum() < 10:
        return np.nan, np.nan, int(ok.sum())
    r, p = stats.spearmanr(v[ok], u[ok], alternative="greater")
    return float(r), float(p), int(ok.sum())


def main() -> int:
    r22, r23 = [], []
    for kind in KINDS:
        mp = load_map(kind)
        groups = groups_for(kind)
        dfs = {g: load_any(g, kind)[0] for g in groups}
        for t in groups:
            df = dfs[t]
            spk = df.speaker_group.astype(str).to_numpy()
            y = df.label_4.to_numpy()
            draws = {(K, r): map_draw(df, t, r, int(K))[1] for K in KS for r in sorted(mp.rep.unique())}
            g = mp[mp.target == t]
            per = g.groupby(["K", "rep"]).agg(t=("t", "first"), m_minus_s=("m", "mean"), s=("s", "mean")).reset_index()
            per["m_minus_s"] = per.m_minus_s - per.s
            per["n_speakers"] = [len(np.unique(spk[draws[(K, r)]])) for K, r in zip(per.K, per.rep)]
            r22.append(per.assign(kind=kind, target=t))
            for s in groups:
                if s == t:
                    continue
                P = softmax(np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")[f"t_{t}"], axis=1)
                p_true = P[np.arange(len(y)), y]
                h = g[g.source == s][["K", "rep", "m", "s", "t"]].copy()
                h["n_cw"] = [int((p_true[draws[(K, r)]] < CW).sum()) for K, r in zip(h.K, h.rep)]
                r23.append(h.assign(kind=kind, target=t, source=s, m_minus_s=h.m - h.s))
        print(f"  {kind} 済み", flush=True)
    A, B = pd.concat(r22), pd.concat(r23)
    A.to_csv(OUT / "e22_rows.csv", index=False)
    B.to_csv(OUT / "e23_rows.csv", index=False)
    T22, T23 = [], []
    for kind, g in A.groupby("kind", sort=False):
        for K, h in g.groupby("K"):
            for name, col in (("S1 ターゲットのみ", "t"), ("S2 合併 − ソースのみ", "m_minus_s")):
                r, p, n = demeaned_spearman(h, ["target"], "n_speakers", col)
                T22.append({"setting": KINDS[kind], "K": K, "test": name, "spearman": r, "p_one_sided": p,
                            "n_varying": n, "mean_speakers": h.n_speakers.mean(),
                            "min_speakers": h.n_speakers.min(), "max_speakers": h.n_speakers.max()})
    for kind, g in B.groupby("kind", sort=False):
        for K, h in g.groupby("K"):
            r, p, n = demeaned_spearman(h, ["target", "source"], "n_cw", "m_minus_s")
            T23.append({"setting": KINDS[kind], "K": K, "test": "W1 合併 − ソースのみ", "spearman": r, "p_one_sided": p,
                        "n_varying": n, "share_draws_with_cw": float((h.n_cw >= 1).mean()), "mean_n_cw": h.n_cw.mean()})
    T22, T23 = pd.DataFrame(T22), pd.DataFrame(T23)
    T22.to_csv(OUT / "e22_tests.csv", index=False)
    T23.to_csv(OUT / "e23_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(T22.round(3).to_string(index=False)); print(T23.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
