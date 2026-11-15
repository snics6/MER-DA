"""研究計画（`docs/researchvision.md` 8.3）に登録した仮説 H7（組に固有のずれ）を，第 2 エンコーダ（HuBERT）の地図で判定する。

手順は登録した文のとおりにする。

- 結果の量：`g_small`＝K∈{1,5} での「合併 − ターゲットのみ」の UAR の差の平均（30 回の平均），
  `kstar`＝K*（`summarize_map.py` の定義．順位 none < 1 < 5 < … < all．IEMOCAP がターゲットの組は除く）
- 指標：PAD と coverage（ターゲットがソースに覆われる向き）。空間は `space_choice.json` で選んだもの
- 解析：`explore_decomp.partial_rank`（順位に直し，結果の量 = μ + a_t + b_s + β × 指標 を最小二乗法で
  当てはめ，β を残りの自由度で t 検定する）
- 主判定 4 つ（H7a〜d）：片側，Holm，全体の有意水準 5%。「支持」＝予言の向きの片側 p の Holm 補正後が
  5% 未満，「反証」＝逆の向きの片側 p（4 つにまとめて Holm）が 5% 未満，どちらでもなければ「判定できない」
- 副判定（記述のみ）：MMD²・フレシェ距離・density・R_src・Δ_CE の同じ解析，ばらつきの内訳，
  組をそのまま並べた順位相関（E1 の方法）

比べるために，WavLM（`audio_wavlm`．こちらは結果を見た後の探索的な値）も同じ手順で計算して並べる。

    ./.venv/bin/python exp/summarize_map.py --dir results/map_audio_hubert
    ./.venv/bin/python exp/side/check_h7.py

出力：`results/check_h7/primary.csv`・`secondary.csv`・`share.csv`・`raw_corr.csv`・`fig_h7.png`・`provenance.json`
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # exp/side/ から見たリポジトリの根
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "exp" / "side"))   # exp/ の同じ名前の台本（explore_select.py は E33 の解析）より先に読む
from pair_table import OUT_ROOT, decompose, design, holm, partial_rank, per_pair  # noqa: E402
from summarize_map import C_MERGED, GRID, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

OUT = OUT_ROOT / "check_h7"
KINDS = ("audio_hubert", "audio_wavlm")           # 先頭が判定に使う種類
PRIMARY = [("H7a", "pad", "g_small", -1), ("H7b", "pad", "kstar", -1),
           ("H7c", "coverage_t_by_s", "g_small", +1), ("H7d", "coverage_t_by_s", "kstar", +1)]
SECONDARY = ["mmd2", "frechet", "density_t_by_s", "r_src", "delta_ce"]
ALPHA = 0.05


def table(kind: str) -> pd.DataFrame:
    d = per_pair(kind)
    if d is None:
        raise SystemExit(f"{kind}: summary か shift の結果が無い")
    d["g_small"] = (d.gain_K1 + d.gain_K5) / 2
    return d


def verdict(p_pred: float, p_opp: float) -> str:
    if p_pred < ALPHA:
        return "支持"
    if p_opp < ALPHA:
        return "反証"
    return "判定できない"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prim, sec, share, raw = [], [], [], []
    for kind in KINDS:
        d = table(kind)
        rows = []
        for name, met, resp, sign in PRIMARY:
            r = partial_rank(d, resp, met)
            p_pred = float(stats.t.sf(sign * r["t"], r["df"]))        # 予言の向きの片側 p
            rows.append({"kind": kind, "hypothesis": name, "metric": met, "response": resp,
                         "predicted_sign": sign, "r_partial_rank": r["r_partial_rank"], "t": r["t"],
                         "df": r["df"], "n_pairs": r["n"], "p_one_sided": p_pred,
                         "p_one_sided_opposite": 1 - p_pred})
        h = pd.DataFrame(rows)
        h["p_holm"] = holm(h.p_one_sided.to_numpy())
        h["p_holm_opposite"] = holm(h.p_one_sided_opposite.to_numpy())
        h["verdict"] = [verdict(a, b) for a, b in zip(h.p_holm, h.p_holm_opposite)]
        prim.append(h)
        for resp in ("g_small", "kstar"):
            for met in SECONDARY:
                r = partial_rank(d, resp, met)
                sec.append({"kind": kind, "metric": met, "response": resp, **r})
        for v in ["g_small", "kstar", "pad", "coverage_t_by_s"] + SECONDARY:
            r = decompose(d, v)
            share.append({"kind": kind, "variable": v,
                          **{k: r[k] for k in ("n_pairs", "df_resid", "r2_target", "r2_source",
                                               "r2_both", "pair_specific")}})
        for resp in ("g_small", "kstar"):
            for met in ["pad", "coverage_t_by_s"] + SECONDARY:
                x = d[[resp, met]].dropna()
                rho, p = stats.spearmanr(x[resp], x[met])
                raw.append({"kind": kind, "metric": met, "response": resp, "rho": rho, "p": p,
                            "n_pairs": len(x)})
    prim = pd.concat(prim, ignore_index=True)
    prim.to_csv(OUT / "primary.csv", index=False)
    pd.DataFrame(sec).to_csv(OUT / "secondary.csv", index=False)
    pd.DataFrame(share).to_csv(OUT / "share.csv", index=False)
    pd.DataFrame(raw).to_csv(OUT / "raw_corr.csv", index=False)
    fig(table(KINDS[0]), prim[prim.kind == KINDS[0]], OUT / "fig_h7.png")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip())
    n_rep = {k: int(table(k).n_rep.min()) for k in KINDS}
    (OUT / "provenance.json").write_text(json.dumps(
        {"script": "exp/check_h7.py", "commit": commit, "dirty": dirty,
         "run": datetime.now().astimezone().isoformat(timespec="seconds"), "n_rep_min": n_rep,
         "space": {k: json.loads((ROOT / "results" / f"shift_{k}" / "space_choice.json")
                                 .read_text())["choice"] for k in KINDS}},
        ensure_ascii=False, indent=1))
    print(prim[["kind", "hypothesis", "r_partial_rank", "df", "p_one_sided", "p_holm",
                "p_holm_opposite", "verdict"]].to_string(index=False))
    print(f"書いた: {OUT}")
    return 0


def fig(d: pd.DataFrame, h: pd.DataFrame, path: Path) -> None:
    """主判定 4 つ：ターゲットとソースの効果を除いた残りどうしの順位の散布図。"""
    labels = {"pad": "PAD", "coverage_t_by_s": "coverage", "g_small": "gain at K∈{1,5}",
              "kstar": "K*"}
    fig, axes = plt.subplots(1, len(PRIMARY), figsize=(4.0 * len(PRIMARY), 3.9), facecolor=SURFACE)
    for ax, (name, met, resp, sign) in zip(axes, PRIMARY):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        x = d[["target", "source", resp, met]].dropna()
        y, m = stats.rankdata(x[resp]), stats.rankdata(x[met])
        X0 = design(x, True, True)
        ry = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
        rm = m - X0 @ np.linalg.lstsq(X0, m, rcond=None)[0]
        ax.scatter(rm, ry, s=28, color=C_MERGED, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        b, a = np.polyfit(rm, ry, 1)
        xx = np.array([rm.min(), rm.max()])
        ax.plot(xx, a + b * xx, color=MUTED, linewidth=1.5)
        r = h[h.hypothesis == name].iloc[0]
        ax.set_title(f"{name}: {labels[resp]} vs {labels[met]} (predicted {'−' if sign < 0 else '+'})\n"
                     f"r = {r.r_partial_rank:+.2f}, Holm p = {r.p_holm:.3g}, n = {int(r.n_pairs)}",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel(f"rank of {labels[met]} (residual)", fontsize=8, color=INK2)
        ax.set_ylabel(f"rank of {labels[resp]} (residual)", fontsize=8, color=INK2)
    fig.suptitle("Pre-registered H7 (docs/researchvision.md 8.3), audio_hubert, 30 reps: "
                 "after removing target and source effects", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
