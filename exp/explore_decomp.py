"""【探索的・途中の見立て】得と距離の指標を「ターゲットの効果＋ソースの効果＋組に固有の部分」に分ける。

**事前登録（`researchvision.md` 6.5.1）の判定ではない。**

組 (t, s) ごとの量 v を，最小二乗法で v = μ + a_t + b_s + e に当てはめる（対角 t = s は無いので，
行・列の平均ではなく，ターゲットとソースの指示変数の回帰で解く）。

- **ばらつきの内訳**：ターゲットだけの回帰の決定係数 R²_t，ソースだけの R²_s，両方の R²_ts，
  組に固有の部分 1 − R²_ts
- **組に固有の部分どうしの関係**：得の残り e と，距離の指標の残り e の Spearman の ρ。
  組どうしが同じコーパスを共有することによる似通いは a_t・b_s として除かれる。
  残りの自由度は 組の数 − (ターゲットの数 + ソースの数 − 1) なので，p 値はさらに目安にとどまる

    ./.venv/bin/python exp/explore_relation.py     # 先に（per_pair.csv を作る）
    ./.venv/bin/python exp/explore_decomp.py

- **自由度を正しく数えた検定**（`decomp_partial.csv`）：順位に直した v を v = μ + a_t + b_s + β·指標 に
  当てはめ，β の t 検定を残りの自由度で行う（偏相関）。音声は 36 検定に Holm をかけた `p_holm` も出す

出力：`results/explore_relation/decomp_share.csv`・`decomp_corr.csv`・`decomp_partial.csv`・
`fig_decomp_share.png`・`fig_decomp_resid_audio.png`
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
from explore_relation import KINDS, METRICS, per_pair  # noqa: E402
from summarize_map import C_MERGED, C_SRC, C_TGT, GRID, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

OUT = ROOT / "results" / "explore_relation"
K_ORDER = ["1", "5", "10", "25", "50", "all"]
RESP = ["gain_K1", "gain_K5", "worst", "kstar"]


def design(d: pd.DataFrame, use_t: bool, use_s: bool) -> np.ndarray:
    cols = [np.ones(len(d))]
    for use, col in ((use_t, "target"), (use_s, "source")):
        if use:
            levels = sorted(d[col].unique())
            cols += [(d[col] == lv).to_numpy(float) for lv in levels[1:]]
    return np.column_stack(cols)


def r2(y: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    return (1 - float((e ** 2).sum()) / sst if sst > 0 else np.nan), e


def decompose(d: pd.DataFrame, col: str) -> dict | None:
    x = d[["target", "source", col]].dropna()
    if len(x) < 8 or x[col].nunique() < 2:
        return None
    y = x[col].to_numpy(float)
    rt, _ = r2(y, design(x, True, False))
    rs, _ = r2(y, design(x, False, True))
    rts, e = r2(y, design(x, True, True))
    df_res = len(x) - (x.target.nunique() + x.source.nunique() - 1)
    return {"n_pairs": len(x), "df_resid": df_res, "r2_target": rt, "r2_source": rs,
            "r2_both": rts, "pair_specific": 1 - rts,
            "resid": pd.Series(e, index=pd.MultiIndex.from_frame(x[["target", "source"]]))}


def partial_rank(d: pd.DataFrame, resp: str, met: str) -> dict | None:
    """順位に直し，ターゲットとソースの指示変数を入れた回帰で，指標の係数を t 検定する。"""
    x = d[["target", "source", resp, met]].dropna()
    if len(x) < 8:
        return None
    y, m = stats.rankdata(x[resp]), stats.rankdata(x[met])
    X0 = design(x, True, True)
    X1 = np.column_stack([X0, m])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    e = y - X1 @ beta
    df = len(y) - np.linalg.matrix_rank(X1)
    se = np.sqrt((e @ e) / df * np.linalg.inv(X1.T @ X1)[-1, -1])
    ry = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    rm = m - X0 @ np.linalg.lstsq(X0, m, rcond=None)[0]
    return {"r_partial_rank": float(np.corrcoef(ry, rm)[0, 1]),
            "p": float(2 * stats.t.sf(abs(beta[-1] / se), df)), "df": int(df), "n": len(y)}


def holm(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p)
    adj = np.minimum(np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]), 1)
    out = np.empty_like(adj)
    out[o] = adj
    return out


def gains_all_k(kind: str) -> pd.DataFrame:
    p = pd.read_csv(ROOT / "results" / f"map_{kind}" / "summary" / "pairs.csv", dtype={"K": str})
    return p.pivot_table(index=["target", "source"], columns="K", values="diff_mean").reset_index()


def main() -> int:
    per = pd.read_csv(OUT / "per_pair.csv")
    shares, corrs, partials, resid_audio = [], [], [], {}
    for kind in [k for k in KINDS if k in set(per.kind)]:
        d = per[per.kind == kind].copy()
        g = gains_all_k(kind)
        for K in K_ORDER:                                  # 得の内訳を K ごとにも
            d = d.merge(g[["target", "source", K]].rename(columns={K: f"gain_all_K{K}"}),
                        on=["target", "source"], how="left")
        variables = [f"gain_all_K{K}" for K in K_ORDER] + RESP + list(METRICS)
        dec = {v: decompose(d, v) for v in variables}
        for v, r in dec.items():
            if r is not None:
                shares.append({"kind": kind, "variable": v,
                               **{k: r[k] for k in ("n_pairs", "df_resid", "r2_target",
                                                    "r2_source", "r2_both", "pair_specific")}})
        for resp in RESP:
            for met in METRICS:
                pr = partial_rank(d, resp, met)
                if pr is not None and d.target.nunique() >= 5:        # 4 群の種類は自由度が 5 しかない
                    partials.append({"kind": kind, "response": resp, "metric": met, **pr})
                a, b = dec.get(resp), dec.get(met)
                if a is None or b is None:
                    continue
                j = pd.concat([a["resid"].rename("r"), b["resid"].rename("m")], axis=1).dropna()
                if len(j) < 8:
                    continue
                rho, pv = stats.spearmanr(j.r, j.m)
                corrs.append({"kind": kind, "response": resp, "metric": met, "rho_resid": rho,
                              "p": pv, "n_pairs": len(j),
                              "df_resid": min(a["df_resid"], b["df_resid"])})
                if kind == "audio_wavlm":
                    resid_audio[(resp, met)] = j
    sh, co = pd.DataFrame(shares), pd.DataFrame(corrs)
    pa = pd.DataFrame(partials)
    pa["p_holm"] = np.nan
    au = pa.kind == "audio_wavlm"
    pa.loc[au, "p_holm"] = holm(pa.loc[au, "p"].to_numpy())
    pa.to_csv(OUT / "decomp_partial.csv", index=False)
    sh.to_csv(OUT / "decomp_share.csv", index=False)
    co.to_csv(OUT / "decomp_corr.csv", index=False)
    fig_share(sh[sh.kind == "audio_wavlm"], OUT / "fig_decomp_share.png")
    fig_resid(resid_audio, OUT / "fig_decomp_resid_audio.png")
    print(f"書いた: {OUT}")
    return 0


def fig_share(sh: pd.DataFrame, path: Path) -> None:
    order = ([f"gain_all_K{K}" for K in K_ORDER] + ["kstar"]
             + ["pad", "mmd2", "frechet", "coverage_t_by_s", "density_t_by_s", "cov_dominance",
                "r_src", "delta_ce", "l1_true"])
    labels = {**{f"gain_all_K{K}": f"gain K={K}" for K in K_ORDER}, "kstar": "K* (prov.)",
              "pad": "PAD", "mmd2": "MMD²", "frechet": "Fréchet", "coverage_t_by_s": "coverage",
              "density_t_by_s": "density", "cov_dominance": "cov. dom. c", "r_src": "R_src",
              "delta_ce": "Δ_CE", "l1_true": "label L1"}
    h = sh.set_index("variable").reindex([o for o in order if o in set(sh.variable)])
    # 内訳：ターゲットだけで説明できる分，ソースを足して増える分，組に固有の分
    t_part = h.r2_target.clip(lower=0)
    s_part = (h.r2_both - h.r2_target).clip(lower=0)
    rest = h.pair_specific.clip(lower=0)
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(h))
    ax.bar(x, t_part, color=C_TGT, width=0.7, label="target effect", edgecolor=SURFACE, linewidth=2)
    ax.bar(x, s_part, bottom=t_part, color=C_SRC, width=0.7, label="+ source effect",
           edgecolor=SURFACE, linewidth=2)
    ax.bar(x, rest, bottom=t_part + s_part, color="#c3c2b7", width=0.7,
           label="pair-specific (residual)", edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(x, [labels[v] for v in h.index], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("share of variance across pairs", fontsize=9, color=INK2)
    ax.set_ylim(0, 1.02)
    ax.axvline(6.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.13))
    ax.set_title("EXPLORATORY, audio (42 pairs; K* 30 pairs): where does the variation come from?",
                 fontsize=9, color=INK, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_resid(resid: dict, path: Path) -> None:
    panels = [(("kstar", "pad"), "K* vs PAD"), (("gain_K1", "pad"), "gain K=1 vs PAD"),
              (("gain_K5", "mmd2"), "gain K=5 vs MMD²"), (("gain_K1", "r_src"), "gain K=1 vs R_src")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.8), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, panels):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        if key not in resid:
            ax.set_visible(False)
            continue
        j = resid[key]
        rx, ry = stats.rankdata(j.m), stats.rankdata(j.r)
        ax.scatter(rx, ry, s=28, color=C_MERGED, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        b, a = np.polyfit(rx, ry, 1)
        xx = np.array([rx.min(), rx.max()])
        ax.plot(xx, a + b * xx, color=MUTED, linewidth=1.5)
        ax.set_title(f"residuals: {title}\nρ = {stats.spearmanr(j.r, j.m)[0]:+.2f} (n = {len(j)})",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("rank of metric residual", fontsize=8, color=INK2)
        ax.set_ylabel("rank of response residual", fontsize=8, color=INK2)
    fig.suptitle("EXPLORATORY, audio: after removing target and source effects",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
