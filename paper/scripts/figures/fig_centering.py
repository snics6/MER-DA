"""Fig. centering：ラベル無しの中心化と K=5 のラベルが，ソースのみの分類器をどれだけ上げるか（E12）．

棒は「ソースのみからの UAR の差（点）」の組の平均．A，A(HuBERT)，V の 3 種類．
  Centering            中心をそろえたソースのみ − ソースのみ          （台帳 p1-center-srconly-<kind>）
  Threshold fit D1     ソースのみ＋D1（K=5）− ソースのみ              （p1-d1-gain-<kind>-k5）
  Pooled               合併（K=5，C を固定した地図）− ソースのみ       （p1-merged-gain-<kind>-k5）
  Centering + D1       中心化の後の D1 − ソースのみ   ＝ 中心化の分 ＋ p1-center-d1-gain-<kind>-k5
  Centering + pooled   中心化の後の合併 − ソースのみ ＝ 中心化の分 ＋ p1-center-merged-gain-<kind>-k5
後ろの 2 つには中心化だけの高さを黒い横線で重ね，棒との差が中心化の後のラベルの上乗せになる．
集計は台帳 part1._e12 と同じ：(target, source, rep) で対にした差を反復で平均し，組で平均．
中心化の後の合併は学び直し（results/mech_<kind>/）の C（交差検証）で，地図の合併（C 固定）とは C の決め方が違う．

入力：results/mech_<kind>/*.csv（op = center・none，method = src_only・src_only+D1・merged），
      results/fixc_map/<kind>/*.csv（src_only，merged K=5）．反復は学び直しの全反復

    ../.venv/bin/python scripts/figures/fig_centering.py      # paper/ で
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import style  # noqa: E402
from ledger import part1 as P1  # noqa: E402

KINDS = ("a", "ah", "v")
COLS = ["center", "d1", "pooled", "c_d1", "c_pooled"]
NAMES = {"center": "Centering\n(no labels)", "d1": "Threshold\nfit D1", "pooled": "Pooled",
         "c_d1": "Centering\n+ D1", "c_pooled": "Centering\n+ pooled"}


def per_pair(k: str) -> pd.DataFrame:
    """組ごとの，ソースのみからの差（×100，反復で平均）．列 COLS と，中心化の後の上乗せ c_d1_x・c_pooled_x．"""
    kind = style.CONFIG[k]["kind"]
    mech = P1._cat(style.RESULTS / f"mech_{kind}", ["op", "target", "source", "method", "K", "rep", "uar"])
    tr = mech[["target", "rep"]].drop_duplicates()
    mp = P1._cat(style.RESULTS / "fixc_map" / kind, ["target", "source", "method", "K", "rep", "uar"]).merge(tr)
    s = mp[mp.method == "src_only"][["target", "source", "rep", "uar"]].rename(columns={"uar": "s"})
    m = mp[(mp.method == "merged") & (mp.K == "5")][["target", "source", "rep", "uar"]].rename(columns={"uar": "m"})

    def mm(op, method, name, K=None):
        x = mech[(mech.op == op) & (mech.method == method)]
        if K is not None:
            x = x[x.K == K]
        return x[["target", "source", "rep", "uar"]].rename(columns={"uar": name})

    key = ["target", "source"]
    c = mm("center", "src_only", "sc").merge(s)
    out = ((c.sc - c.s).groupby([c.target, c.source]).mean() * 100).rename("center").to_frame()
    y = mm("none", "src_only+D1", "d1", "5").merge(s).merge(m)
    h = y.groupby(key)[["d1", "s", "m"]].mean() * 100
    out["d1"] = h.d1 - h.s
    out["pooled"] = h.m - h.s
    z = mm("center", "src_only+D1", "d1c", "5").merge(mm("center", "src_only", "sc")).merge(mm("center", "merged", "mc", "5"))
    h = z.groupby(key)[["d1c", "sc", "mc"]].mean() * 100
    out["c_d1_x"] = h.d1c - h.sc
    out["c_pooled_x"] = h.mc - h.sc
    out["c_d1"] = out.center + out.c_d1_x
    out["c_pooled"] = out.center + out.c_pooled_x
    return out


def main() -> None:
    style.setup()
    tabs = {k: per_pair(k) for k in KINDS}

    fig, ax = plt.subplots(figsize=(style.COL, 1.95))
    fig.subplots_adjust(left=0.14, right=0.975, bottom=0.2, top=0.84)
    ng = len(COLS)
    w = 0.26
    for j, k in enumerate(KINDS):
        c = style.CONFIG[k]
        t = tabs[k]
        for i, col in enumerate(COLS):
            xc = i + (j - 1) * w
            ax.bar(xc, t[col].mean(), w * 0.9, color=c["color"], lw=0, zorder=2)
            if col in ("c_d1", "c_pooled"):
                # 中心化だけの高さ（この差がラベルの上乗せ）
                ax.plot([xc - w * 0.45, xc + w * 0.45], [t["center"].mean()] * 2, color=style.INK, lw=1.0,
                        zorder=3, solid_capstyle="butt")
    ax.axhline(0, color=style.INK, lw=0.8, zorder=2.5)
    ax.axvline(2.5, color=style.INK2, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_xticks(range(ng), [NAMES[c] for c in COLS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, ng - 0.5)
    ax.set_ylabel("UAR change from\nsource-only (points)")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.text(1.0, 1.0, "$K=5$ for D1 and pooled", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=6.5, color=style.INK2)
    handles = [Patch(color=style.CONFIG[k]["color"], label=style.CONFIG[k]["label"]) for k in KINDS]
    handles += [Line2D([], [], color=style.INK, lw=1.0, label="Centering alone")]
    ax.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.07), handlelength=1.1,
              columnspacing=0.9, borderaxespad=0.0)
    style.save(fig, "centering")

    # 台帳の値との照合（標準出力）
    for k in KINDS:
        t = tabs[k]
        print(k, {c: round(float(t[c].mean()), 3) for c in COLS + ["c_d1_x", "c_pooled_x"]}, "pairs", len(t))


if __name__ == "__main__":
    main()
