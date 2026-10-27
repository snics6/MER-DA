"""Fig. finetune：E63・E63c．凍結した特徴（E39 の学習器）と WavLM の LoRA 微調整の音声の生徒の UAR（K=0・5）．

出所と集計は台帳 comparison.py の E63 の部分と同じ（comparison._ft_runs と同じ読み方）：
  results/finetune_check/<ターゲット>.csv（中心化なし），<ターゲット>_c.csv（中心化あり），teacher_<ターゲット>.csv（凍結版）
4 ターゲット（IEMOCAP 即興・台本，MELD，MOSEI），反復 0〜2．値は反復の平均をターゲットごとに取り，ターゲットで平均した UAR．
罰則の強さは固定版の λ=0.01（L2-SP の既定）．台帳の cmp-frozen-uar-*，cmp-ft-uar-*，cmp-ftc-uar-* と同じ値．
棒は偶然の水準（25%）から立てる．
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import style  # noqa: E402
from ledger import comparison as C  # noqa: E402

LAM = C.FT_LAM


def tables():
    ft = pd.concat([C._ft_runs(C.FT_DIR / f"{t}.csv") for t in C.FT_TARGETS])
    fc = pd.concat([C._ft_runs(C.FT_DIR / f"{t}_c.csv") for t in C.FT_TARGETS])
    fr = pd.concat([pd.read_csv(C.FT_DIR / f"teacher_{t}.csv", comment="#", dtype={"K": str}) for t in C.FT_TARGETS])
    fr = fr.assign(uar=fr.uar * 100, lam=0.0)
    return ft, fc, fr


def tmean(df, method, K, lam=None) -> float:
    x = df[(df.method == method) & (df.K == K)]
    if lam is not None:
        x = x[np.isclose(x.lam, lam)]
    if len(x) != len(C.FT_TARGETS) * len(C.FT_REPS):
        raise ValueError(f"fig_finetune: {method} K={K} λ={lam} が {len(x)} 行（12 行のはず）")
    return float(x.groupby("target").uar.mean().mean())


def main() -> None:
    style.setup()
    ft, fc, fr = tables()
    DARK, LIGHT, WEAK = style.INK2, "#bdbcb8", style.METHOD["ours"]["color"]
    # (K, 区切り, 棒の並び [(値, 色)])
    spec = {
        "0": [("Frozen", [(tmean(fr, "frozen_e39", "0"), DARK), (tmean(fr, "frozen_ours", "0"), WEAK)]),
              ("LoRA", [(tmean(ft, "start", "0"), DARK), (tmean(ft, "ours", "0", LAM), WEAK)]),
              ("LoRA +\ncentering", [(tmean(fc, "start", "0"), DARK), (tmean(fc, "ours", "0", LAM), WEAK)])],
        "5": [("Frozen", [(tmean(fr, "frozen_e39", "5"), LIGHT), (tmean(fr, "frozen_ours", "5"), WEAK)]),
              ("LoRA", [(tmean(ft, "k_only", "5"), DARK), (tmean(ft, "anchor", "5", LAM), LIGHT),
                        (tmean(ft, "ours", "5", LAM), WEAK)]),
              ("LoRA +\ncentering", [(tmean(fc, "k_only", "5"), DARK), (tmean(fc, "anchor", "5", LAM), LIGHT),
                                     (tmean(fc, "ours", "5", LAM), WEAK)])],
    }
    fig, axes = style.plt.subplots(1, 2, figsize=(style.COL, 1.95), sharey=True, layout="constrained",
                                   gridspec_kw=dict(width_ratios=[6, 8]))
    w = 0.8
    for ax, K, tag in zip(axes, ("0", "5"), ("(a)", "(b)")):
        x, centers, names = 0.0, [], []
        for name, bars in spec[K]:
            xs = [x + i for i in range(len(bars))]
            for xi, (v, c) in zip(xs, bars):
                ax.bar(xi, v - 25, bottom=25, width=w, color=c, lw=0, zorder=2)
            centers.append(np.mean(xs))
            names.append(name)
            x = xs[-1] + 1.6
        ax.set_xticks(centers)
        ax.set_xticklabels(names)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.7, x - 1.6 + 0.7)
        ax.text(0.02, 0.98, f"{tag} K = {K}", transform=ax.transAxes, fontsize=7, va="top", ha="left")
    axes[0].set_ylim(25, 56)
    axes[0].set_yticks([25, 35, 45, 55])
    axes[0].set_ylabel("UAR (%)")
    hs = [Patch(color=DARK, label="Start or K labels only"), Patch(color=LIGHT, label="K labels + L2 toward start"),
          Patch(color=WEAK, label="+ weak labels (fused teacher)")]
    fig.legend(handles=hs, loc="outside upper center", ncol=2, columnspacing=1.0,
               handlelength=1.2, borderaxespad=0.0)
    fig.get_layout_engine().set(wspace=0.02)
    style.save(fig, "finetune")


if __name__ == "__main__":
    main()
