"""【探索的】新しいターゲットに対して，どのソースを混ぜると得が大きいかの順位を当てられるか（E3）。

ターゲットを 1 つ（t）抜き，t を含む組（t がターゲットの組も，t がソースの組も）を使わずに，
残りの組で 得 = μ + a_{t'} + b_s (+ β × 指標) を最小二乗法で当てはめる。t の候補ソース s の
予測は b_s (+ β × 指標(t, s))。a_t はどのソースにも同じだけかかるので，順位には影響しない。

予測の仕方（`MODELS`）：
- `source`：ソースの効果だけ
- `+pad`・`+coverage`・`+pad+coverage`：距離（ターゲットのラベル不要．登録した空間の値）を足す
- `+rsrc_K`：K 本で測った R_src（ソースのみの分類器の，ターゲットの K 本での誤り．反復 0 の抽出）を足す
- `+pad+coverage+rsrc_K`：全部

当たりの測り方（ターゲットごとに求めて平均する）：予測と実際の得の Spearman の ρ，1 位を当てたか，
予測 1 位のソースを選んだときの損（実際の 1 位の得 − 予測 1 位の得．UAR の点）。

    ./.venv/bin/python exp/explore_decomp.py      # 先に（per_pair.csv）
    ./.venv/bin/python exp/side/explore_select.py

出力：`results/explore_relation/select_by_target.csv`・`select_summary.csv`・`fig_select.png`
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
from pair_table import OUT, design, gains_all_k  # noqa: E402
from summarize_map import C_MERGED, C_SRC, C_TGT, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

KINDS = ("audio_wavlm", "video", "fuse-audio-video")          # 候補ソースが 5 つ以上ある種類
K_ORDER = ["1", "5", "10", "25", "50", "all"]
MODELS = {"source": [], "+pad": ["pad"], "+coverage": ["coverage_t_by_s"],
          "+pad+coverage": ["pad", "coverage_t_by_s"], "+rsrc_K": ["rsrc_K"],
          "+pad+coverage+rsrc_K": ["pad", "coverage_t_by_s", "rsrc_K"]}


def taus(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Kendall の τ-b（同点を扱う）と重みつき τ（Vigna 2015．上位ほど重い．scipy の既定で両向きの平均）。"""
    return {"tau_b": float(stats.kendalltau(pred, actual, variant="b")[0]),
            "tau_w": float(stats.weightedtau(pred, actual)[0])}


def table(kind: str) -> pd.DataFrame:
    per = pd.read_csv(OUT / "per_pair.csv")
    d = per[per.kind == kind][["target", "source", "pad", "coverage_t_by_s"]]
    g = gains_all_k(kind)
    lab = pd.read_csv(ROOT / "results" / f"shift_{kind}" / "labeling.csv", comment="#")
    lab = lab[["target", "source"] + [f"uar_t_K{K}" for K in K_ORDER]]
    return d.merge(g, on=["target", "source"]).merge(lab, on=["target", "source"])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, y: str, xs: list[str]) -> np.ndarray:
    Xtr = design(train, True, True)
    mu = {c: (train[c].mean(), train[c].std() or 1.0) for c in xs}
    Z = [(train[c] - mu[c][0]) / mu[c][1] for c in xs]
    if xs:
        Xtr = np.column_stack([Xtr] + [z.to_numpy() for z in Z])
    beta, *_ = np.linalg.lstsq(Xtr, train[y].to_numpy(float), rcond=None)
    sources = sorted(train.source.unique())
    n_t = train.target.nunique()
    b = dict(zip(sources, np.r_[0.0, beta[n_t:n_t + len(sources) - 1]]))   # 1 つ目のソースが基準
    pred = np.array([b[s] for s in test.source])
    for j, c in enumerate(xs):
        pred = pred + beta[-len(xs) + j] * ((test[c] - mu[c][0]) / mu[c][1]).to_numpy()
    return pred


def main() -> int:
    rows = []
    for kind in KINDS:
        d = table(kind)
        for K in K_ORDER:
            dk = d.rename(columns={K: "gain"}).copy()
            dk["rsrc_K"] = 1 - dk[f"uar_t_K{K}"]
            for t in sorted(dk.target.unique()):
                train = dk[(dk.target != t) & (dk.source != t)]
                test = dk[dk.target == t]
                actual = test.gain.to_numpy()
                for name, xs in MODELS.items():
                    pred = fit_predict(train, test, "gain", xs)
                    best, pick = actual.max(), actual[int(np.argmax(pred))]
                    rows.append({"kind": kind, "K": K, "target": t, "model": name,
                                 "rho": stats.spearmanr(pred, actual)[0],
                                 "top1": bool(np.argmax(pred) == np.argmax(actual)),
                                 "regret": (best - pick) * 100, "n_sources": len(test),
                                 "n_train_pairs": len(train), **taus(pred, actual)})
    by = pd.DataFrame(rows)
    by.to_csv(OUT / "select_by_target.csv", index=False)
    sm = (by.groupby(["kind", "K", "model"])
            .agg(rho=("rho", "mean"), tau_b=("tau_b", "mean"), tau_w=("tau_w", "mean"),
                 top1=("top1", "mean"), regret=("regret", "mean"),
                 n_targets=("target", "nunique")).reset_index())
    sm.to_csv(OUT / "select_summary.csv", index=False)
    fig(sm, OUT / "fig_select.png")
    print(f"書いた: {OUT}")
    return 0


def fig(sm: pd.DataFrame, path: Path) -> None:
    show = {"source": ("source effect only", MUTED, "-"),
            "+pad+coverage": ("+ PAD + coverage (label-free)", C_MERGED, "-"),
            "+rsrc_K": ("+ R_src from the K labels", C_TGT, "-"),
            "+pad+coverage+rsrc_K": ("+ all three", C_SRC, "--")}
    kinds = [k for k in KINDS if k in set(sm.kind)]
    fig, axes = plt.subplots(2, len(kinds), figsize=(4.3 * len(kinds), 6.6), facecolor=SURFACE,
                             squeeze=False)
    x = np.arange(len(K_ORDER))
    for j, kind in enumerate(kinds):
        for i, (metric, ylab) in enumerate((("rho", "mean Spearman ρ (pred vs actual)"),
                                            ("regret", "mean regret of top-1 pick (UAR pts)"))):
            ax = axes[i, j]
            _style(ax)
            for name, (lab, col, ls) in show.items():
                h = sm[(sm.kind == kind) & (sm.model == name)].set_index("K").reindex(K_ORDER)
                ax.plot(x, h[metric], color=col, linewidth=2, linestyle=ls, marker="o",
                        markersize=5, label=lab)
            ax.set_xticks(x, K_ORDER)
            if i == 0:
                ax.set_title(f"{kind}  ({int(h.n_targets.max())} held-out targets)", fontsize=9,
                             color=INK, loc="left")
                ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
            else:
                ax.set_xlabel("K", fontsize=9, color=INK2)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8.5, color=INK2)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("EXPLORATORY: ranking sources for a held-out target (leave-one-target-out)",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
