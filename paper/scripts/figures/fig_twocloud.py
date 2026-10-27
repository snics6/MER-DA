"""Fig. twocloud：2 つの山の模型と，ソースのみの分類器の損の内訳（E13）．

(a) 模式図．感情 k と l の点数の差 z（σ 単位）の 2 つの正規の山，境目 z = 0，中点のずれ o，離れ d．
    各山は 1/2 倍して描くので，薄く塗った面積の和が 1 − a(d, o)（2 つの再現率の平均の誤り），
    0 と中点の間で 2 つの山に挟まれた面積が判断基準の損 a(d, 0) − a(d, o) にちょうど等しい．
    d と o の値は説明用で，データではない（図に数値を書かない）
(b) ソースのみの分類器の判断基準の損（loss_offset）と向きの損（loss_direction）を 8 種類で．
    台帳 p1-e13-loss-offset-<kind>・p1-e13-loss-dir-<kind> と同じ出所と集計
    （組ごとに 6 つの感情の対で平均し，組で平均．×100 で 2 つの感情の正解率の点）

入力：results/rerun_fixc/explore_relation/snapshots/20261018-0213_fixc/e13_pairs.csv（7 種類），
      同 check_mh/e13.csv（A (HuBERT)．組 × 対の表は無く，要約だけ）

    ../.venv/bin/python scripts/figures/fig_twocloud.py      # paper/ で
"""
from __future__ import annotations

import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

import style  # noqa: E402
from ledger import part1 as P1  # noqa: E402  （結果ファイルの場所を台帳と共有する）

# 模式図の値（説明用）
D_ILL, O_ILL = 2.4, 0.9
ERR = "#d9d8d4"


def e13_losses() -> dict[str, tuple[float, float]]:
    p = pd.read_csv(P1.FIXC_RERUN / "e13_pairs.csv")
    hub = pd.read_csv(P1.FIXC_RERUN / "check_mh" / "e13.csv").set_index("setting").loc["audio_hubert"]
    out = {}
    for k in style.CONFIG_ORDER:
        kind = style.CONFIG[k]["kind"]
        if k == "ah":
            out[k] = (hub.mean_loss_offset * 100, hub.mean_loss_direction * 100)
        else:
            # 同じコーパスの組（IEMOCAP の即興↔台本）を除く（本文 4.3 節・表 2・台帳 p1-e13-loss-*-xsame と同じ組．2026-10-27）
            g = p[(p.kind == kind) & ~(p.target.str.startswith("IEMOCAP") & p.source.str.startswith("IEMOCAP"))]
            pp = g.groupby(["target", "source"])[["loss_offset", "loss_direction"]].mean() * 100
            out[k] = (pp.loss_offset.mean(), pp.loss_direction.mean())
    return out


def schematic(ax) -> None:
    z = np.linspace(-3.6, 4.6, 800)
    mu_k, mu_l = O_ILL + D_ILL / 2, O_ILL - D_ILL / 2
    fk = 0.5 * stats.norm.pdf(z, mu_k)
    fl = 0.5 * stats.norm.pdf(z, mu_l)
    # 誤り：k は z < 0，l は z > 0（境目 z = 0 のとき）
    ax.fill_between(z, 0, fk, where=z <= 0, color=ERR, lw=0, zorder=1)
    ax.fill_between(z, 0, fl, where=z >= 0, color=ERR, lw=0, zorder=1)
    # 判断基準の損：0 と中点の間で 2 つの山に挟まれた部分
    m = (z >= 0) & (z <= O_ILL)
    ax.fill_between(z, fk, fl, where=m, facecolor="white", edgecolor=style.INK, hatch="//////", lw=0, zorder=2)
    ax.plot(z, fl, color=style.INK, lw=1.1, zorder=3)
    ax.plot(z, fk, color=style.INK, lw=1.1, ls=(0, (4, 1.5)), zorder=3)
    top = 0.5 * stats.norm.pdf(0)
    ax.axvline(0, color=style.INK, lw=1.0, zorder=4)
    ax.axvline(O_ILL, color=style.INK2, lw=0.8, ls=(0, (2, 1.5)), zorder=4)
    # 名前
    ax.text(mu_l - 0.3, top * 1.0, "emotion $l$", ha="right", va="bottom")
    ax.text(mu_k + 0.3, top * 1.0, "emotion $k$", ha="left", va="bottom")
    ax.text(-0.08, top * 1.62, "boundary $z=0$", ha="right", va="center")
    ax.text(O_ILL + 0.08, top * 1.62, "midpoint", ha="left", va="center", color=style.INK2)
    # o の矢印（境目から中点まで）
    yo = top * 1.40
    ax.annotate("", xy=(O_ILL, yo), xytext=(0, yo),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=style.INK, shrinkA=0, shrinkB=0))
    ax.text(O_ILL + 0.1, yo, "$o$", ha="left", va="center")
    # d の矢印（2 つの平均の間）
    yd = top * 1.17
    ax.annotate("", xy=(mu_k, yd), xytext=(mu_l, yd),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=style.INK, shrinkA=0, shrinkB=0))
    for mu in (mu_l, mu_k):
        ax.plot([mu, mu], [top, yd], color=style.INK2, lw=0.5, ls=":")
    ax.text(mu_k - 0.45, yd + top * 0.03, "$d$", ha="center", va="bottom")
    ax.set_xlim(z[0], z[-1])
    ax.set_ylim(0, top * 1.75)
    ax.set_yticks([])
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3, 4])
    ax.set_xlabel(r"Score difference $z$ (units of $\sigma$)")
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    handles = [Patch(facecolor=ERR, lw=0, label="Errors, $1-a(d,o)$"),
               Patch(facecolor="white", edgecolor=style.INK, hatch="//////", lw=0,
                     label="Threshold loss, $a(d,0)-a(d,o)$")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.4,
              columnspacing=1.0, borderaxespad=0.1)


def bars(ax, loss) -> None:
    ks = style.CONFIG_ORDER
    x = np.arange(len(ks))
    w = 0.38
    for i, k in enumerate(ks):
        c = style.CONFIG[k]["color"]
        th, di = loss[k]
        ax.bar(x[i] - w / 2, th, w, color=c, lw=0)
        ax.bar(x[i] + w / 2, di, w, facecolor="white", edgecolor=c, hatch="//////", lw=0.8)
    ax.set_xticks(x, [style.CONFIG[k]["label"].replace(" (", "\n(") for k in ks])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Loss (points of\npairwise accuracy)")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(ks) - 0.4)
    handles = [Patch(facecolor=style.INK2, lw=0, label="Threshold loss"),
               Patch(facecolor="white", edgecolor=style.INK2, hatch="//////", lw=0.8, label="Direction loss")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handlelength=1.4,
              borderaxespad=0.1)


def main() -> None:
    style.setup()
    loss = e13_losses()
    fig = plt.figure(figsize=(style.COL, 3.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.7, left=0.16, right=0.99, bottom=0.1, top=0.93)
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[1])
    schematic(ax)
    bars(bx, loss)
    ax.text(-0.15, 1.02, "(a)", transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")
    bx.text(-0.15, 1.02, "(b)", transform=bx.transAxes, fontweight="bold", va="bottom", ha="left")
    style.save(fig, "twocloud")


if __name__ == "__main__":
    main()
