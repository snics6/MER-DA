"""【探索的・途中の見立て】②のずれの指標と，①の地図の結果の関係を，組をまたぐ順位相関で見る。

**事前登録（`docs/researchvision.md` 8 節）の判定ではない。**登録した判定は，30 回そろった結果で，
登録した手順（主判定・Holm）で行う。ここでは，いまある回数の結果で傾向を見るだけにする。

    ./.venv/bin/python exp/side/explore_relation.py

入力：`results/map_<kind>/summary/{pairs,kstar}.csv`（先に `summarize_map.py` を流す），
`results/shift_<kind>/{input,label,labeling}.csv` と `space_choice.json`（選ばれた空間の行だけ使う）。
出力：`results/explore_relation/`

| ファイル | 中身 |
|---|---|
| `corr.csv` | 種類 × 結果の量 × 指標ごとの Spearman の ρ，p 値（目安），組の数，地図の回数 |
| `fig_rho.png` | 上の ρ を種類ごとに並べた表 |
| `fig_rank_audio.png` | 音声の主な組み合わせの，順位で描いた散布図（順位に引いた直線の傾きが ρ） |

結果の量（組ごと）：`gain_K1`・`gain_K5`＝合併 − ターゲットのみ の UAR の差の平均（K=1・5），
`worst`＝K をまたいだ差の平均の最小値（予言 P2-1 の m），`kstar`＝暫定の K*（順位：none < 1 < 5 < … < all．
反復 2 回以上の種類だけ．IEMOCAP がターゲットの組は除く．`docs/researchvision.md` 7 節）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # exp/side/ から見たリポジトリの根
sys.path.insert(0, str(ROOT / "exp"))
from pair_table import KINDS, METRICS, OUT, per_pair  # noqa: E402
from summarize_map import (C_MERGED, GRID, INK, INK2, MUTED, SURFACE, DIVERGING,  # noqa: E402
                           IEMOCAP_TARGETS, _style)

RESPONSES = {"gain_K1": "gain at K=1", "gain_K5": "gain at K=5", "worst": "worst gain over K",
             "kstar": "K* (provisional)"}


def correlations(d: pd.DataFrame) -> list[dict]:
    rows = []
    for resp in RESPONSES:
        for met in METRICS:
            x = d[[resp, met]].dropna()
            if len(x) < 5 or x[resp].nunique() < 2:
                continue
            rho, p = stats.spearmanr(x[resp], x[met])
            rows.append({"kind": d.kind.iloc[0], "response": resp, "metric": met,
                         "rho": rho, "p": p, "n_pairs": len(x),
                         "n_rep_min": int(d.n_rep.min()), "n_rep_max": int(d.n_rep.max())})
    return rows


def reps_note(corr: pd.DataFrame) -> str:
    """種類ごとの地図の回数（最小〜最大）を図の題に書く文字列にする。"""
    g = corr.groupby("kind")[["n_rep_min", "n_rep_max"]].agg({"n_rep_min": "min", "n_rep_max": "max"})
    short = lambda k: k.replace("fuse-", "f:").replace("_wavlm", "").replace("_roberta", "")  # noqa: E731
    return ", ".join(f"{short(k)} {int(r.n_rep_min)}" + (f"–{int(r.n_rep_max)}" if r.n_rep_max != r.n_rep_min else "")
                     for k, r in g.reindex([k for k in KINDS if k in g.index]).iterrows())


def fig_rho(corr: pd.DataFrame, path: Path) -> None:
    kinds = [k for k in KINDS if k in set(corr.kind)]
    fig, axes = plt.subplots(1, len(RESPONSES), figsize=(4.2 * len(RESPONSES), 6.4),
                             facecolor=SURFACE, sharey=True)
    for ax, resp in zip(axes, RESPONSES):
        h = corr[corr.response == resp]
        M = np.full((len(METRICS), len(kinds)), np.nan)
        N = np.zeros_like(M, dtype=int)
        for r in h.itertuples():
            i, j = list(METRICS).index(r.metric), kinds.index(r.kind)
            M[i, j], N[i, j] = r.rho, r.n_pairs
        ax.imshow(M, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.2f}\n(n={N[i, j]})", ha="center", va="center",
                            fontsize=6.5, color="#ffffff" if abs(M[i, j]) > 0.55 else INK)
        ax.set_xticks(range(len(kinds)), [k.replace("fuse-", "f:").replace("_wavlm", "")
                                          .replace("_roberta", "") for k in kinds],
                      rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(METRICS)), list(METRICS.values()), fontsize=7)
        ax.set_title(RESPONSES[resp], fontsize=9, color=INK, loc="left")
    fig.suptitle("EXPLORATORY (not the pre-registered test): Spearman ρ across pairs between "
                 f"shift metrics and map results.  reps: {reps_note(corr)}",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_rank(d: pd.DataFrame, path: Path) -> None:
    lo, hi = int(d.n_rep.min()), int(d.n_rep.max())
    panels = [("kstar", "pad", "P1-2 preview: K* vs PAD"),
              ("worst", "delta_ce", "P2-1 preview: worst gain vs Δ_CE"),
              ("worst", "coverage_t_by_s", "P2-1 preview: worst gain vs coverage"),
              ("gain_K1", "r_src", "gain at K=1 vs R_src")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.8), facecolor=SURFACE)
    for ax, (resp, met, title) in zip(axes, panels):
        x = d[[resp, met]].dropna()
        _style(ax)
        rx, ry = stats.rankdata(x[met]), stats.rankdata(x[resp])
        ax.scatter(rx, ry, s=28, color=C_MERGED, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if len(x) >= 3:
            b, a = np.polyfit(rx, ry, 1)
            xx = np.array([rx.min(), rx.max()])
            ax.plot(xx, a + b * xx, color=MUTED, linewidth=1.5)
            rho = stats.spearmanr(x[resp], x[met])[0]
            ax.set_title(f"{title}\nρ = {rho:+.2f}  (n = {len(x)} pairs)", fontsize=8.5,
                         color=INK, loc="left")
        ax.set_xlabel(f"rank of {METRICS[met]}", fontsize=8, color=INK2)
        ax.set_ylabel(f"rank of {RESPONSES[resp]}", fontsize=8, color=INK2)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    fig.suptitle(f"EXPLORATORY, audio (WavLM), {lo}–{hi} of 30 reps.  K* excludes IEMOCAP targets "
                 "(interval method undecided)", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    data = [d for k in KINDS if (d := per_pair(k)) is not None]
    per = pd.concat(data, ignore_index=True)
    per.to_csv(out / "per_pair.csv", index=False)
    corr = pd.DataFrame([r for d in data for r in correlations(d)])
    corr.to_csv(out / "corr.csv", index=False)
    fig_rho(corr, out / "fig_rho.png")
    fig_rank(per[per.kind == "audio_wavlm"], out / "fig_rank_audio.png")
    print(f"種類: {[d.kind.iloc[0] for d in data]}")
    print(f"書いた: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
