"""【探索的】ソースの選び方の比較（E5）。新しいターゲットに混ぜるソースを，どの方法で選ぶと得が大きいか。

比べる方法（`STRATEGIES`）：

| 名前 | 使う情報 | 何を点数にするか |
|---|---|---|
| `random` | − | でたらめ（期待値を解析的に出す） |
| `source_quality` | ほかのターゲットでの得 | ターゲットを抜いて当てはめた得 = μ + a_t + b_s の b_s（E3） |
| `kshot_validation` | ターゲットの K 本 | 合併の分類器の，K 本での交差検証の UAR（地図の `cv_uar_C*` の，選ばれた C の値．Mansour 2021 の「少数のラベルで検証して選ぶ」） |
| `leep` | ターゲットの K 本 | LEEP（Nguyen 2020）．ソースの分類器の予測確率と K 本のラベルから |
| `logme` | ターゲットの K 本 | LogME（You 2021）．ソースの分類器の出力（和 0 の 3 次元の座標）を特徴とみなして K 本で計算 |
| `quality+coverage` | ほかのターゲット＋ラベル無しのターゲット | b_s + β × coverage（E3 で取り上げた組み合わせ） |
| `quality+distance_nested` | 同上 | b_s + β × 距離．距離の種類は，抜いたターゲットを見ずに，ラベル無しの候補から選ぶ（E4 の入れ子） |

評価：ターゲットごとに候補ソースの順位を付け，**正解は各ソースの反復平均の得**（`summary/pairs.csv` の `diff_mean`）。
K 本で決まる方法（`kshot_validation`・`leep`・`logme`）は反復ごとに選び，反復で平均する。
当たりの測り方は E3 と同じ（順位の相関，1 位を当てたか，予想 1 位の損［UAR の点］）。

LogME は本来，事前学習モデルを選ぶ指標である。ここでは特徴を作るモデルがどのソースでも同じなので，
ソースで学んだ分類器の出力を特徴とみなす。ソースの分類器の出力は `results/shift_<kind>/logits/`（②の段で保存．
同じコーパスの組は交差当てはめ），K 本の抜き方は地図と同じ乱数（`map.kshot_order`）から作り直す。

    ./.venv/bin/python exp/explore_decomp.py      # 先に（per_pair.csv）
    ./.venv/bin/python exp/side/explore_strategies.py

出力：`results/explore_relation/e5_by_target.csv`・`e5_summary.csv`・`fig_e5.png`
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # exp/side/ から見たリポジトリの根
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "exp" / "side"))   # exp/ の同じ名前の台本（explore_select.py は E33 の解析）より先に読む
import map as M  # noqa: E402
from data import GROUP_SPEC, load_any  # noqa: E402
from explore_metrics import loto, table as metric_table  # noqa: E402
from explore_select import fit_predict, taus  # noqa: E402
from pair_table import OUT, map_dir  # noqa: E402
from shift_map import sum_zero_coords  # noqa: E402
from summarize_map import C_MERGED, C_SRC, C_TGT, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

KINDS = ("audio_wavlm", "video", "fuse-audio-video")
K_ORDER = ["1", "5", "10", "25", "50", "all"]
LABEL_FREE = ["pad", "pad_bal", "mmd2", "frechet", "coverage_t_by_s", "density_t_by_s",
              "coverage_s_by_t", "density_s_by_t", "cov_dominance",
              "mmd2@pca64", "frechet@pca64", "l1_est"]
STRATEGIES = ["random", "source_quality", "kshot_validation", "leep", "logme",
              "quality+coverage", "quality+distance_nested"]


# ---------------------------------------------------------------- 既存の転移しやすさの指標

def leep(prob: np.ndarray, y: np.ndarray, k: int = 4) -> float:
    """LEEP（Nguyen 2020 式 (2)）。prob はソースの分類器の予測確率（n × ソースのクラス）。"""
    joint = np.zeros((k, prob.shape[1]))
    for c in range(k):
        joint[c] = prob[y == c].sum(0)
    joint /= len(y)
    cond = joint / np.maximum(joint.sum(0, keepdims=True), 1e-12)     # P(y | z)
    return float(np.mean(np.log(np.maximum((prob @ cond.T)[np.arange(len(y)), y], 1e-12))))


def logme(F: np.ndarray, y: np.ndarray, k: int = 4, iters: int = 100) -> float:
    """LogME（You 2021 算法 1）。F は特徴（n × D），y はラベル。クラスごとの対数証拠の平均 ÷ n。"""
    n, D = F.shape
    u, s, vt = np.linalg.svd(F, full_matrices=False)
    s2 = s ** 2
    out = []
    for c in range(k):
        t = (y == c).astype(float)
        a, b = 1.0, 1.0
        ut = u.T @ t
        for _ in range(iters):
            g = np.sum(b * s2 / (a + b * s2))
            m = vt.T @ (b * s / (a + b * s2) * ut)               # 事後平均
            res = np.sum((F @ m - t) ** 2)
            a_new, b_new = g / max(m @ m, 1e-12), (n - g) / max(res, 1e-12)
            done = abs(a_new - a) / a < 1e-5 and abs(b_new - b) / b < 1e-5
            a, b = a_new, b_new
            if done:
                break
        m = vt.T @ (b * s / (a + b * s2) * ut)
        res = np.sum((F @ m - t) ** 2)
        ev = (n / 2 * np.log(b) + D / 2 * np.log(a) - n / 2 * np.log(2 * np.pi)
              - b / 2 * res - a / 2 * (m @ m) - 0.5 * np.sum(np.log(a + b * s2)))
        out.append(ev / n)
    return float(np.mean(out))


# ---------------------------------------------------------------- 評価

def evaluate(scores: np.ndarray, gains: np.ndarray) -> dict:
    ok = np.isfinite(scores)
    if not ok.any():
        return {"rho": np.nan, "top1": np.nan, "regret": np.nan, "tau_b": np.nan, "tau_w": np.nan}
    s = np.where(ok, scores, -np.inf)
    pick = int(np.argmax(s))
    t = taus(s, gains) if ok.all() else {"tau_b": np.nan, "tau_w": np.nan}
    return {"rho": stats.spearmanr(s, gains)[0] if ok.all() else np.nan,
            "top1": float(pick == int(np.argmax(gains))),
            "regret": (gains.max() - gains[pick]) * 100, **t}


def groups_light(kind: str, names: list[str]) -> dict:
    """K 本の抜き方を作り直すための軽い群（特徴は持たない）。"""
    out = {}
    for g in names:
        df, _ = load_any(g, kind)
        sp = df["speaker_group"].astype(str).reset_index(drop=True)
        out[g] = M.Group(g, None, df["label_4"].to_numpy(), sp,
                         (GROUP_SPEC[g][0] + ":" + sp).to_numpy())
    return out


def run_kind(kind: str, groups: list[str] | None = None) -> list[dict]:
    """groups を渡すと，その群どうしの組だけで比べる（E7：RAVDESS を除いた音声など）。"""
    pairs = pd.read_csv(map_dir(kind) / "summary" / "pairs.csv", dtype={"K": str})   # MERDA_MAP_ROOT に従う
    if groups is not None:
        pairs = pairs[pairs.target.isin(groups) & pairs.source.isin(groups)]
    gains = pairs.pivot_table(index=["target", "source"], columns="K", values="diff_mean")
    rows_map = pd.concat([pd.read_csv(p, comment="#", dtype={"K": str, "source": str})
                          for p in sorted(map_dir(kind).glob("*.csv"))])
    merged = rows_map[rows_map.method == "merged"].copy()
    merged["cv_at_C"] = [r[f"cv_uar_C{r['C']:g}"] for _, r in merged.iterrows()]
    cv = merged.set_index(["target", "source", "K", "rep"])["cv_at_C"]
    targets = sorted(pairs.target.unique())
    light = groups_light(kind, targets)
    logits = {s: np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz") for s in targets}
    mt, cands = metric_table(kind)
    if groups is not None:
        mt = mt[mt.target.isin(groups) & mt.source.isin(groups)]
    label_free = [c for c in cands if c in LABEL_FREE]
    reps = sorted(int(r) for r in merged.rep.unique())

    out = []
    for K in K_ORDER:
        dk = mt.rename(columns={K: "gain"}).copy()
        for t in targets:
            srcs = sorted(s for s in targets if s != t)
            g_true = np.array([gains.loc[(t, s), K] for s in srcs])
            base = {"kind": kind, "K": K, "target": t, "n_sources": len(srcs)}
            # でたらめ：期待値
            out.append({**base, "strategy": "random", "rep": -1, "rho": 0.0, "tau_b": 0.0, "tau_w": 0.0,
                        "top1": 1 / len(srcs), "regret": float(np.mean(g_true.max() - g_true)) * 100})
            # ほかのターゲットから作る方法（反復に依らない）
            train = dk[(dk.target != t) & (dk.source != t)]
            test = dk[dk.target == t].set_index("source").loc[srcs].reset_index()
            for name, xs in (("source_quality", []), ("quality+coverage", ["coverage_t_by_s"])):
                out.append({**base, "strategy": name, "rep": -1,
                            **evaluate(fit_predict(train, test, "gain", xs), g_true)})
            inner_targets = [u for u in targets if u != t]
            best, best_rho = None, -np.inf
            for m in [None] + label_free:
                r = np.mean([x[1] for x in loto(train, inner_targets, [] if m is None else [m])])
                if r > best_rho:
                    best, best_rho = m, r
            out.append({**base, "strategy": "quality+distance_nested", "rep": -1, "chosen": best or "none",
                        **evaluate(fit_predict(train, test, "gain", [] if best is None else [best]), g_true)})
            # K 本で決まる方法（反復ごと）
            g = light[t]
            for r in reps:
                if not all((t, s, K, r) in cv.index for s in srcs):
                    continue
                pool = np.flatnonzero(~M.test_mask(g, r))
                idx = M.take_k(M.kshot_order(g, pool, r), K)
                yk = g.y[idx]
                sc_cv = np.array([cv.loc[(t, s, K, r)] for s in srcs], float)
                sc_leep, sc_logme = [], []
                for s in srcs:
                    L = logits[s][f"t_{t}"][idx]
                    sc_leep.append(leep(softmax(L, axis=1), yk))
                    sc_logme.append(logme(sum_zero_coords(L), yk))
                for name, sc in (("kshot_validation", sc_cv), ("leep", np.array(sc_leep)),
                                 ("logme", np.array(sc_logme))):
                    out.append({**base, "strategy": name, "rep": r, **evaluate(sc, g_true)})
    return out


def fig(sm: pd.DataFrame, path: Path) -> None:
    style = {"random": ("random (expected)", "#c3c2b7", ":"),
             "source_quality": ("source quality only", MUTED, "-"),
             "kshot_validation": ("validate on K labels", C_TGT, "-"),
             "leep": ("LEEP (K labels)", "#e87ba4", "-"),
             "logme": ("LogME (K labels)", "#4a3aa7", "-"),
             "quality+coverage": ("quality + coverage", C_MERGED, "--"),
             "quality+distance_nested": ("quality + distance (nested)", C_MERGED, "-")}
    kinds = [k for k in KINDS if k in set(sm.kind)]
    fig, axes = plt.subplots(2, len(kinds), figsize=(4.4 * len(kinds), 6.8), facecolor=SURFACE,
                             squeeze=False)
    x = np.arange(len(K_ORDER))
    for j, kind in enumerate(kinds):
        for i, (metric, ylab) in enumerate((("rho", "mean Spearman ρ (score vs mean gain)"),
                                            ("regret", "mean regret of top-1 pick (UAR pts)"))):
            ax = axes[i, j]
            _style(ax)
            for name, (lab, col, ls) in style.items():
                h = sm[(sm.kind == kind) & (sm.strategy == name)].set_index("K").reindex(K_ORDER)
                ax.plot(x, h[metric], color=col, linewidth=2, linestyle=ls, marker="o",
                        markersize=4.5, label=lab)
            ax.set_xticks(x, K_ORDER)
            if i == 0:
                reps = sm[(sm.kind == kind)]["n_reps"].max()
                ax.set_title(f"{kind}  (reps used for K-label methods: {int(reps)})", fontsize=9,
                             color=INK, loc="left")
            else:
                ax.set_xlabel("K", fontsize=9, color=INK2)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8.5, color=INK2)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("EXPLORATORY: strategies for choosing a source for a new target (E5)",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    rows = [r for kind in KINDS for r in run_kind(kind)]
    by = pd.DataFrame(rows)
    by.to_csv(OUT / "e5_by_target.csv", index=False)
    per_t = by.groupby(["kind", "K", "strategy", "target"])[["rho", "tau_b", "tau_w", "top1", "regret"]].mean()
    n_reps = by[by.rep >= 0].groupby(["kind"]).rep.nunique()
    sm = per_t.groupby(["kind", "K", "strategy"]).mean().reset_index()
    sm["n_targets"] = per_t.reset_index().groupby(["kind", "K", "strategy"]).target.nunique().values
    sm["n_reps"] = sm.kind.map(n_reps).fillna(0)
    sm.to_csv(OUT / "e5_summary.csv", index=False)
    fig(sm, OUT / "fig_e5.png")
    print(f"書いた: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
