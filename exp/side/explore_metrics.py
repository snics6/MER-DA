"""【探索的】どの指標がソースの順位の予想に役立つか（E4）。E3（`explore_select.py`）を全指標に広げる。

1. **1 つずつ足す**：ターゲットを 1 つ抜き（その組はソースとしても使わない），残りの組で
   得 = μ + a_{t'} + b_s + β × 指標 を当てはめ，抜いたターゲットの候補ソースの順位を予想する。
   指標なし（ソースの効果だけ）との差で，その指標が役立つかを見る。
2. **選び方も含める（入れ子）**：抜いたターゲットを見ずに，残りのターゲットだけの「抜いて当てる」
   で最も当たった指標（指標なしも候補）を選び，その指標で抜いたターゲットを予想する。
   よく見えた指標を後から選ぶことによる甘さを除いた当たりになる。
3. **大事さの一覧**：指標ごとに，1 の当たりの伸び（K ≤ 25 の平均）と，E2 の「ターゲットと
   ソースの効果を除いた順位の偏相関」（`decomp_partial.csv`）を並べる。

候補（`candidates`）：空間 logit（登録した規則で選んだ空間）と pca64 の距離（PAD，割合をそろえた PAD，
MMD²，フレシェ距離，coverage と density の両向き，共分散の支配），推定した割合の差（MLLS），
K 本だけで測れる量（R_src と Δ_CE．反復 0 の抽出）。ターゲットの全ラベルを使う量は使わない。

    ./.venv/bin/python exp/explore_decomp.py      # 先に（per_pair.csv・decomp_partial.csv）
    ./.venv/bin/python exp/side/explore_metrics.py

出力：`results/explore_relation/e4_single.csv`・`e4_nested.csv`・`e4_importance.csv`・`fig_e4_importance.png`
"""

from __future__ import annotations

import json
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
from pair_table import OUT, gains_all_k  # noqa: E402
from explore_select import fit_predict, taus  # noqa: E402
from summarize_map import C_MERGED, C_TGT, GRID, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

KINDS = ("audio_wavlm", "video", "fuse-audio-video")
K_ORDER = ["1", "5", "10", "25", "50", "all"]
K_SMALL = ["1", "5", "10", "25"]
DIST = ["pad", "mmd2", "frechet", "coverage_t_by_s", "density_t_by_s", "coverage_s_by_t",
        "density_s_by_t", "cov_dominance"]
LABELS = {"pad": "PAD", "pad_bal": "PAD (class-balanced)", "mmd2": "MMD²", "frechet": "Fréchet",
          "coverage_t_by_s": "coverage (target covered)", "density_t_by_s": "density (target)",
          "coverage_s_by_t": "coverage (source covered)", "density_s_by_t": "density (source)",
          "cov_dominance": "cov. dominance c", "l1_est": "label-prior L1 (MLLS est.)",
          "rsrc_K": "R_src from K labels", "dce_K": "Δ_CE from K labels"}


def table(kind: str) -> tuple[pd.DataFrame, list[str]]:
    s = ROOT / "results" / f"shift_{kind}"
    space = json.loads((s / "space_choice.json").read_text())["choice"]
    inp = pd.read_csv(s / "input.csv", comment="#")
    base = inp[(inp.space == space) & (~inp.balanced)].set_index(["target", "source"])[DIST]
    pca = inp[(inp.space == "pca64") & (~inp.balanced)].set_index(["target", "source"])[DIST]
    bal = inp[(inp.space == space) & (inp.balanced)].set_index(["target", "source"])["pad"]
    d = base.join(pca.add_suffix("@pca64")).join(bal.rename("pad_bal"))
    lb = pd.read_csv(s / "label.csv", comment="#").set_index(["target", "source"])
    d["l1_est"] = lb["l1_est"]
    lab = pd.read_csv(s / "labeling.csv", comment="#").set_index(["target", "source"])
    for K in K_ORDER:
        d[f"uar_t_K{K}"], d[f"dce_K{K}"] = lab[f"uar_t_K{K}"], lab[f"delta_ce_K{K}"]
    d = d.reset_index().merge(gains_all_k(kind), on=["target", "source"])
    cands = (["pad", "pad_bal"] + DIST[1:] + [c + "@pca64" for c in DIST] + ["l1_est", "rsrc_K", "dce_K"])
    return d, cands


def score(pred: np.ndarray, actual: np.ndarray) -> tuple[float, bool, float, float, float]:
    """(ρ, 1 位を当てたか, 損, τ-b, 重みつき τ)。入れ子の選択の基準には ρ（先頭）を使う。"""
    t = taus(pred, actual)
    return (stats.spearmanr(pred, actual)[0], bool(np.argmax(pred) == np.argmax(actual)),
            (actual.max() - actual[int(np.argmax(pred))]) * 100, t["tau_b"], t["tau_w"])


def loto(dk: pd.DataFrame, targets: list[str], xs: list[str]) -> list[tuple[str, float, bool, float]]:
    """targets を 1 つずつ抜いて当てる。dk は使ってよい組だけを含むこと。"""
    out = []
    for t in targets:
        train = dk[(dk.target != t) & (dk.source != t)]
        test = dk[dk.target == t]
        out.append((t, *score(fit_predict(train, test, "gain", xs), test.gain.to_numpy())))
    return out


def main() -> int:
    single, nested = [], []
    for kind in KINDS:
        d, cands = table(kind)
        options = [None] + cands
        for K in K_ORDER:
            dk = d.rename(columns={K: "gain"}).copy()
            dk["rsrc_K"], dk["dce_K"] = 1 - dk[f"uar_t_K{K}"], dk[f"dce_K{K}"]
            targets = sorted(dk.target.unique())
            for m in options:
                for t, rho, top1, reg, tb, tw in loto(dk, targets, [] if m is None else [m]):
                    single.append({"kind": kind, "K": K, "metric": m or "none", "target": t,
                                   "rho": rho, "top1": top1, "regret": reg, "tau_b": tb, "tau_w": tw})
            for t in targets:                                # 入れ子：t を見ずに指標を選ぶ
                inner = dk[(dk.target != t) & (dk.source != t)]
                inner_targets = [u for u in targets if u != t]
                best, best_rho = None, -np.inf
                for m in options:
                    r = np.mean([x[1] for x in loto(inner, inner_targets, [] if m is None else [m])])
                    if r > best_rho:
                        best, best_rho = m, r
                train = dk[(dk.target != t) & (dk.source != t)]
                test = dk[dk.target == t]
                rho, top1, reg, tb, tw = score(fit_predict(train, test, "gain",
                                                           [] if best is None else [best]),
                                               test.gain.to_numpy())
                nested.append({"kind": kind, "K": K, "target": t, "chosen": best or "none",
                               "rho": rho, "top1": top1, "regret": reg, "tau_b": tb, "tau_w": tw})
    s1, sn = pd.DataFrame(single), pd.DataFrame(nested)
    s1.to_csv(OUT / "e4_single_by_target.csv", index=False)
    sn.to_csv(OUT / "e4_nested.csv", index=False)
    agg = (s1.groupby(["kind", "K", "metric"]).agg(rho=("rho", "mean"), tau_b=("tau_b", "mean"),
                                                   tau_w=("tau_w", "mean"), top1=("top1", "mean"),
                                                   regret=("regret", "mean")).reset_index())
    none = agg[agg.metric == "none"].set_index(["kind", "K"])
    agg["d_rho"] = agg.apply(lambda r: r.rho - none.loc[(r.kind, r.K), "rho"], axis=1)
    agg["d_regret"] = agg.apply(lambda r: r.regret - none.loc[(r.kind, r.K), "regret"], axis=1)
    agg["d_tau_b"] = agg.apply(lambda r: r.tau_b - none.loc[(r.kind, r.K), "tau_b"], axis=1)
    agg.to_csv(OUT / "e4_single.csv", index=False)

    small = agg[agg.K.isin(K_SMALL)].groupby(["kind", "metric"])[["d_rho", "d_tau_b", "d_regret"]].mean()
    part = pd.read_csv(OUT / "decomp_partial.csv")
    part = part[part.response.isin(["gain_K1", "gain_K5", "kstar"])]
    part = part.pivot_table(index=["kind", "metric"], columns="response", values="r_partial_rank")
    imp = small.join(part, how="left").reset_index()
    imp.to_csv(OUT / "e4_importance.csv", index=False)
    fig_importance(imp[imp.kind == "audio_wavlm"], OUT / "fig_e4_importance.png")
    print(f"書いた: {OUT}")
    return 0


def fig_importance(imp: pd.DataFrame, path: Path) -> None:
    h = imp[imp.metric != "none"].sort_values("d_rho", ascending=True)
    lab = [LABELS.get(m.split("@")[0], m) + ("  [pca64]" if "@pca64" in m else "") for m in h.metric]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(h) + 1.6), facecolor=SURFACE, sharey=True)
    y = np.arange(len(h))
    for ax, col, xlab, color in ((axes[0], "d_rho", "gain in mean rank correlation vs source-only", C_MERGED),
                                 (axes[1], "d_regret", "change in regret of top-1 pick (UAR pts; < 0 is better)", C_TGT)):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.barh(y, h[col], color=color, height=0.6, edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_xlabel(xlab, fontsize=8, color=INK2)
    axes[0].set_yticks(y, lab, fontsize=7.5)
    fig.suptitle("EXPLORATORY, audio (7 held-out targets, K ≤ 25 averaged): "
                 "which metric helps rank sources for a new target?",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0.24)                       # 長い指標名が切れないように
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
