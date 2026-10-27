"""論文の図の共通の約束（大きさ・字・色・印・保存）．

すべての図の台本はこのモジュールを使う．図の数値は MER-DA の結果ファイルから読み（台帳と同じ出所），
図の中に本文の主張の数値を手で書かない．

- 大きさ：段 1 つ COL（3.45 in），2 段ぶち抜き PAGE（7.1 in）．IEEEtran の journal・compsoc に合わせた
- 字：サンセリフ 7 pt（目盛り 6.5 pt）．本文（10 pt）を縮めたときに読める大きさ
- 色：検証済みの 8 色（dataviz の既定の配色）を固定の順に割り当てる．種類（特徴の世界）の色は図をまたいで同じにする．
  黄・桃・水色は白地で対比が弱いので，線と点には必ず印（marker）を併用し，凡例か直接の名前を付ける
- 手法の色は別の表（METHOD）．E39（基準）は灰色，私たちの手法は青と紫，相手は橙・赤
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = pathlib.Path(__file__).resolve().parents[2]
MERDA = PAPER.parent
RESULTS = MERDA / "results"
OUT = PAPER / "figures"

COL = 3.45
PAGE = 7.1

# dataviz の既定の 8 色（明るい地）．順は固定
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"

# 種類（特徴の世界）．並び順は A, V, T, AV, VT, AT, AVT（研究者の約束），第 2 エンコーダの音声 AH は 8 番目
CONFIG_ORDER = ["a", "v", "t", "av", "vt", "at", "avt", "ah"]
CONFIG = {
    "a":   dict(label="A",           color=SLOT[0], marker="o", kind="audio_wavlm"),
    "v":   dict(label="V",           color=SLOT[1], marker="s", kind="video"),
    "t":   dict(label="T",           color=SLOT[2], marker="^", kind="text_roberta"),
    "av":  dict(label="AV",          color=SLOT[3], marker="D", kind="fuse-audio-video"),
    "vt":  dict(label="VT",          color=SLOT[4], marker="v", kind="fuse-video-text"),
    "at":  dict(label="AT",          color=SLOT[5], marker="P", kind="fuse-audio-text"),
    "avt": dict(label="AVT",         color=SLOT[6], marker="X", kind="fuse-audio-video-text"),
    "ah":  dict(label="A (HuBERT)",  color=SLOT[7], marker="h", kind="audio_hubert"),
}

# 手法（Part II と比較の図）
METHOD = {
    "e39":    dict(label="Start (five-value HT)", color=GRAY),
    "best":   dict(label="Strongest tuned competitor", color=SLOT[1]),
    "ours":   dict(label="Ours", color=SLOT[0]),
    "oursx":  dict(label="Ours-ext", color=SLOT[6]),
    "entmin": dict(label="Entropy min. (fixed)", color=SLOT[7]),
}


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, name: str) -> None:
    """figures/<name>.pdf（本文用）と figures/<name>.png（確認用，200 dpi）に保存する．"""
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
