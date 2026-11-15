"""【探索的】E18：距離の 2 つの働き。コーパスの違いは位置のずれを作り（ソースのみに損），それを直す道も作る（合併に得）。

予言と判定は `docs/exploratory-analyses.md` E18 の 1 節（計算の前に書いた）。新しい学習はしない。

    得 m − t = (s − t) + (m − s)
    D1  距離 ↔ 位置による損（E13）       正
    D2  距離 ↔ 見分けやすさ G（E15）      正
    D3  距離 ↔ s − t（K=5）              負
    D4  距離 ↔ m − s（K=5）              正
    D5  順位の重回帰 rank(m − t) ~ rank(距離) + rank(式の位置の直り)：距離の係数は負，位置の直りの係数は正

距離は `results/shift_<kind>/input.csv`（登録した規則で選んだ空間，`balanced` が偽）の PAD（主），MMD²・フレシェ距離（副）。
組ごとの地図と E15 の量は `OUT/e15_pairs.csv`（C を固定した地図なら MERDA_OUT_ROOT=results/fixc_analysis）。

    ./.venv/bin/python exp/explore_dual.py
    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_dual.py
    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/text30 ./.venv/bin/python exp/explore_dual.py \
        --kinds text_roberta fuse-audio-text fuse-video-text fuse-audio-video-text
                                   # 言葉を含む 4 種類（4 群・12 組．`e15_pairs.csv` は E21 の results/fixc_text/ から写す）

出力：`OUT/e18_rows.csv`（種類 × 組 × K），`OUT/e18_tests.csv`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
from pair_table import OUT, SIX, holm, partial_rank  # noqa: E402

KINDS = {"audio_wavlm": "W7", "audio_hubert": "Hu7", "video": "V", "fuse-audio-video": "AV"}
TEXT_KINDS = {"text_roberta": "T", "fuse-audio-text": "AT", "fuse-video-text": "VT", "fuse-audio-video-text": "AVT"}
NAMES = {**KINDS, **TEXT_KINDS}      # 既定で流すのは KINDS だけ（言葉を含む 4 種類は --kinds で指定）
SIX_NAME = {"audio_wavlm": "W6", "audio_hubert": "Hu6"}
DISTS = ("pad", "mmd2", "frechet")


def distances(kind: str) -> pd.DataFrame:
    s = ROOT / "results" / f"shift_{kind}"
    space = json.loads((s / "space_choice.json").read_text())["choice"]
    d = pd.read_csv(s / "input.csv", comment="#")
    d = d[(d.space == space) & (~d.balanced)]
    return d[["target", "source", *DISTS]]


def one_sided(r: float, p_two: float, sign: int) -> float:
    return p_two / 2 if np.sign(r) == sign else 1 - p_two / 2


def rank_ols(y, X):
    """順位に直した重回帰。係数と片側でない t を返す（列は標準化しない順位）。"""
    Y = stats.rankdata(y)
    Z = np.column_stack([np.ones(len(Y))] + [stats.rankdata(x) for x in X])
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    e = Y - Z @ beta
    df = len(Y) - Z.shape[1]
    cov = (e @ e) / df * np.linalg.inv(Z.T @ Z)
    return beta[1:], beta[1:] / np.sqrt(np.diag(cov)[1:]), df


def tests(d: pd.DataFrame, setting: str, dist: str) -> list[dict]:
    out = []
    x = d[dist]
    for name, col, sign in (("D1", "loss_offset", 1), ("D2", "G", 1), ("D3", "s_minus_t", -1), ("D4", "m_minus_s", 1)):
        r, p = stats.spearmanr(x, d[col])
        pr = partial_rank(d.rename(columns={dist: "dist_"}), col, "dist_")
        out.append({"test": name, "quantity": col, "stat": r, "p": one_sided(r, p, sign),
                    "partial": pr["r_partial_rank"] if pr else np.nan})
    b, t, df = rank_ols(d.m_minus_t, [x, d.pos_fix_pred])
    b0, t0, _ = rank_ols(d.m_minus_t, [x])
    r_raw, p_raw = stats.spearmanr(x, d.m_minus_t)
    out.append({"test": "D5 距離", "quantity": "m_minus_t ~ dist + pos_fix", "stat": b[0], "t": t[0],
                "p": stats.t.cdf(t[0], df), "coef_dist_alone": b0[0], "spearman_raw": r_raw, "p_raw_two": p_raw})
    out.append({"test": "D5 位置の直り", "quantity": "m_minus_t ~ dist + pos_fix", "stat": b[1], "t": t[1],
                "p": stats.t.sf(t[1], df)})
    adj = holm(np.array([o["p"] for o in out]))
    for o, h in zip(out, adj):
        o.update({"setting": setting, "dist": dist, "n_pairs": len(d), "p_holm": h})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="E18")
    ap.add_argument("--kinds", nargs="+", choices=list(NAMES), default=list(KINDS))
    a = ap.parse_args()
    P = pd.read_csv(OUT / "e15_pairs.csv", dtype={"K": str})
    rows, T = [], []
    for kind, name in ((k, NAMES[k]) for k in a.kinds):
        dist = distances(kind)
        for K in ("1", "5", "10"):
            d = P[(P.kind == kind) & (P.K == K)].merge(dist, on=["target", "source"])
            d = d.assign(s_minus_t=d.s - d.t)
            rows.append(d)
            for dn in DISTS:
                for setting, groups in [(name, None)] + ([(SIX_NAME[kind], SIX)] if kind in SIX_NAME else []):
                    h = d if groups is None else d[d.target.isin(groups) & d.source.isin(groups)]
                    for o in tests(h, setting, dn):
                        T.append({"K": K, **o})
    R, T = pd.concat(rows), pd.DataFrame(T)
    R.to_csv(OUT / "e18_rows.csv", index=False)
    T.to_csv(OUT / "e18_tests.csv", index=False)
    pd.set_option("display.width", 250)
    show = T[(T.K == "5") & (T.dist == "pad")][["setting", "test", "stat", "p", "p_holm", "partial", "coef_dist_alone", "spearman_raw"]]
    print(show.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
