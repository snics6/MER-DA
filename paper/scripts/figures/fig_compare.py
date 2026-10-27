"""Fig. compare：K=0 と K=5 の，E39（出発点の 5 つの値の仮説転移）との UAR の差．

生徒ごとに，最も強い相手の調整版・ours（E60 の選んだ形）・oursx（拡張版）の 3 本．
出所と集計は台帳 comparison.py と同じ（comparison._world をそのまま呼ぶ）：
  results/baselines_<世界>/<ターゲット>.csv（調整版の選択は <ターゲット>_pseudo.csv から作り直す），
  results/multimodal_distill_<世界>/<ターゲット>.csv（ours の選択は同じ場所の _pseudo.csv から）．
- 棒：反復 0〜9 の平均をターゲットごとに取り，ターゲットで平均した差（台帳の cmp-*-vs-e39-* と同じ値）
- ひげ：ターゲットごとの差の最小から最大
- 最も強い相手：oursx 以外の相手の調整版のうち，ターゲットの平均が最大のもの（生徒・K ごと．同点は名前の順で最初）
他の図の台本（fig_kcurves.py，fig_entmin.py）は world_table() と best_of() を使う．
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

# 相手の略称（キャプションで定義する）
ABBR = {"self": "ST", "entmin": "EntM", "kd": "KD", "gd": "GD", "hard": "HPL", "cotrain": "CoT",
        "cotrain_prod": "CoT-P", "cotrain_fused": "CoT-F", "coreg": "CoR", "coreg_sum": "CoR-S",
        "saerens": "SEM", "massbal": "MB"}

# 描く生徒（世界の key, 生徒の名前, 目盛りの名前, 区切りの名前）
GROUPS = [
    ("avt", "audio_wavlm", "A", "First encoders, AVT world"),
    ("avt", "video", "V", "First encoders, AVT world"),
    ("avt", "text_roberta", "T", "First encoders, AVT world"),
    ("avt", "fuse-audio-video-text", "AVT", "First encoders, AVT world"),
    ("av", "fuse-audio-video", "AV", "First encoders, fused students"),
    ("vt", "fuse-video-text", "VT", "First encoders, fused students"),
    ("at", "fuse-audio-text", "AT", "First encoders, fused students"),
    ("w2", "audio_hubert", "A", "Second encoders, AVT world"),
    ("w2", "video_clip", "V", "Second encoders, AVT world"),
    ("w2", "text_deberta", "T", "Second encoders, AVT world"),
    ("w2", "fuse2-audio-video-text", "AVT", "Second encoders, AVT world"),
]

_CACHE: dict[str, pd.DataFrame] = {}


def world_table(wk: str) -> pd.DataFrame:
    """世界 wk の（ターゲット, 生徒, K）ごとの値（ターゲットごとの反復平均）．台帳の _world と同じ．"""
    if wk not in _CACHE:
        rows, chk = C._world(C.WORLDS[wk])
        if chk["meta_mismatch"] or chk["e60_mismatch"]:
            print(f"fig_compare: {wk} で作り直した選択が保存の chosen と食い違う: {chk}")
        _CACHE[wk] = pd.DataFrame(rows)
    return _CACHE[wk]


def best_of(wk: str, s: str, K: str) -> str:
    """oursx 以外の相手の調整版のうち，ターゲットの平均が最大のもの（台帳の best_of と同じ規則）．"""
    d = world_table(wk)
    g = d[(d.student == s) & (d.K == K)]
    comps = {c.split(":")[0]: g[c].mean() for c in g.columns
             if c.endswith(":tuned") and not c.startswith("oursx") and g[c].notna().all()}
    return max(sorted(comps), key=lambda m: comps[m])


def diffs(wk: str, s: str, K: str, col: str) -> pd.Series:
    """ターゲットごとの col − e39．"""
    d = world_table(wk)
    g = d[(d.student == s) & (d.K == K)].set_index("target")
    return g[col] - g["e39"]


def best_label(wk: str, s: str, K: str) -> str:
    """最も強い相手の略称．調整版が全ターゲットで「使わない」を選んだときは none．"""
    b = best_of(wk, s, K)
    d = world_table(wk)
    picks = d[(d.student == s) & (d.K == K)][f"{b}:pick"]
    return "none" if picks.str.startswith("none|").all() else ABBR[b]


def main() -> None:
    style.setup()
    fig, axes = style.plt.subplots(2, 1, figsize=(style.PAGE, 3.3), layout="constrained")
    keys = [("best", None), ("ours", "ours"), ("oursx", "oursx:tuned")]
    w = 0.26
    xs = np.arange(len(GROUPS), dtype=float)
    xs[4:] += 0.5
    xs[7:] += 0.5
    lo_all, hi_all = 0.0, 0.0
    for ax, K, tag in zip(axes, ("0", "5"), ("(a)", "(b)")):
        for i, (wk, s, lab, _) in enumerate(GROUPS):
            b = best_of(wk, s, K)
            for j, (mk, col) in enumerate(keys):
                c = f"{b}:tuned" if mk == "best" else col
                v = diffs(wk, s, K, c)
                x = xs[i] + (j - 1) * w
                ax.bar(x, v.mean(), width=w * 0.92, color=style.METHOD[mk]["color"], zorder=2, lw=0)
                ax.plot([x, x], [v.min(), v.max()], color=style.INK, lw=0.6, zorder=3)
                lo_all, hi_all = min(lo_all, v.min()), max(hi_all, v.max())
            ax.text(xs[i] - w, -0.15, best_label(wk, s, K), transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=6.5, color=style.METHOD["best"]["color"])
        ax.axhline(0, color=style.INK2, lw=0.6, zorder=2)
        ax.set_ylabel("UAR − start (points)")
        ax.text(0.003, 0.98, f"{tag}  K = {K}", transform=ax.transAxes, fontsize=7, va="top", ha="left")
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_locator(style.matplotlib.ticker.MaxNLocator(integer=True, nbins=5))
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([g[2] for g in GROUPS])
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
        for x in ((xs[3] + xs[4]) / 2, (xs[6] + xs[7]) / 2):
            ax.axvline(x, color="#c9c8c4", lw=0.6, zorder=1)
    for ax in axes:                                   # 2 枚の縦軸をそろえる
        ax.set_ylim(np.floor(lo_all) - 0.2, np.ceil(hi_all) + 0.8)
    # 区切りの名前
    for name in dict.fromkeys(g[3] for g in GROUPS):
        idx = [i for i, g in enumerate(GROUPS) if g[3] == name]
        axes[-1].text((xs[idx[0]] + xs[idx[-1]]) / 2, -0.30, name, transform=axes[-1].get_xaxis_transform(),
                      ha="center", va="top", fontsize=7, color=style.INK)
    hs = [Patch(color=style.METHOD[m]["color"], label=style.METHOD[m]["label"]) for m, _ in keys]
    axes[0].legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.1,
                   handlelength=1.2, columnspacing=1.5)
    fig.get_layout_engine().set(h_pad=0.06, hspace=0.06)
    style.save(fig, "compare")


if __name__ == "__main__":
    main()
