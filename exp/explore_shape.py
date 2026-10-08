"""【探索的】ソースで引いた感情の境目の向きが，ターゲットの感情をどれだけ分けるか（E10）。

E9 は感情の重心の差（矢印）だけで向きのそろい方を測り，感情ごとの雲の散らばり方を使わなかった。
直線の学習器は散らばり方も使って境目の向きを決める（散らばりの大きい向きを軽く見る）。
ここでは，ソースで決めた向きにターゲットの発話を射影し，2 つの感情の雲がどれだけ離れて見えるかを測る。

組 (t, s) と感情の対 (k, l)（6 対）ごとに，ソースで向き u を決める。
- `mean`：重心の差 m_k − m_l（散らばり方を使わない．E9 と同じ考え方）
- `lda`：Σ⁻¹(m_k − m_l)．Σ はソースの，感情ごとに中心をそろえた共分散を Ledoit–Wolf で縮めたもの
  （線形判別分析の向き．散らばり方を使う）

ターゲットの発話を u に射影し，`tdp` = (k の平均 − l の平均) / √((k の分散 + l の分散)/2) を求める
（符号つき．負ならソースの向きがターゲットでは逆を向く）。6 対で平均する。
ターゲット自身で決めた向きの値（`own_*`．話者で 2 つに分け，片方で向き，もう片方で測る．入れ替えて平均）も求め，
`eff_*` = `tdp_*` / `own_*`（ターゲット自身の向きに比べて，ソースの向きでどれだけ分けられるか）とする。

空間は E9 と同じ（等しい重みで合わせた統計で標準化）。解析も E9 と同じく，ソースのみの UAR と得の
組に固有の部分との順位の偏相関（`partial_rank`）をとる。Holm は種類ごとに主な 4 量 × 7 応答にかける。
量の組を一度に入れたとき，組に固有の部分（順位）の何割を説明するかも出す（`e10_joint.csv`．
量 2 つでも，でたらめでおよそ 2 / 残りの自由度 は説明する）。
`eff_lda` の組に固有の部分が，`align` やこれまでの距離の組に固有の部分とどれだけ重なるかも出す（`e10_overlap.csv`）。

    ./.venv/bin/python exp/explore_shape.py

出力：`results/explore_relation/e10_pair.csv`・`e10_corpus.csv`・`e10_partial.csv`・`e10_joint.csv`・`e10_overlap.csv`・
`e10_modality.csv`・`fig_e10.png`
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import groups_for, load_any  # noqa: E402
from explore_geometry import KINDS, K_ORDER, common_reps, standardize  # noqa: E402
from pair_table import OUT, SIX, design, fit, gains, holm, partial_rank  # noqa: E402
from split import speaker_folds  # noqa: E402
from summarize_map import C_MERGED, C_SRC, C_TGT, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

PAIRS = [(k, j) for k in range(4) for j in range(k + 1, 4)]
MAIN_Q = ["tdp_mean", "tdp_lda", "eff_mean", "eff_lda"]
JOINT = {"eff_lda+shift_emo": ["eff_lda", "shift_emo"], "align+shift_emo": ["align", "shift_emo"],
         "eff_lda+shift_emo+align": ["eff_lda", "shift_emo", "align"], "pad+coverage": ["pad", "coverage_t_by_s"]}


def directions(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """6 対ぶんの向き（6 × 次元）。"""
    m = np.stack([X[y == k].mean(0) for k in range(4)])
    diff = np.stack([m[k] - m[j] for k, j in PAIRS])
    W = X - m[y]
    S = LedoitWolf(assume_centered=True).fit(W).covariance_
    return {"mean": diff, "lda": np.linalg.solve(S, diff.T).T}


def tdp(U: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """向き U（6 × 次元）にターゲットを射影したときの，6 対それぞれの離れ方。"""
    out = []
    for u, (k, j) in zip(U, PAIRS):
        pk, pj = X[y == k] @ u, X[y == j] @ u
        out.append((pk.mean() - pj.mean()) / np.sqrt((pk.var(ddof=1) + pj.var(ddof=1)) / 2))
    return np.array(out)


def shape(kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = groups_for(kind)
    data = standardize({g: load_any(g, kind) for g in groups})
    full, own = {}, {}
    for g, (df, X) in data.items():
        y = df.label_4.to_numpy()
        full[g] = directions(X, y)
        half = speaker_folds(df.speaker_group, k=2, seed=0).to_numpy()
        A, B = half == 0, half == 1
        dA, dB = directions(X[A], y[A]), directions(X[B], y[B])
        own[g] = {v: (tdp(dA[v], X[B], y[B]) + tdp(dB[v], X[A], y[A])) / 2 for v in ("mean", "lda")}
    lab = pd.read_csv(ROOT / "results" / f"shift_{kind}" / "labeling.csv", comment="#")
    src_uar = lab.set_index(["target", "source"]).uar_t
    rows = []
    for t in groups:
        df, X = data[t]
        y = df.label_4.to_numpy()
        for s in groups:
            if t == s:
                continue
            r = {"kind": kind, "target": t, "source": s, "src_uar": float(src_uar.get((t, s), np.nan))}
            for v in ("mean", "lda"):
                d = tdp(full[s][v], X, y)
                r[f"tdp_{v}"] = float(d.mean())
                r[f"eff_{v}"] = float(d.mean() / own[t][v].mean())
                r[f"n_reversed_{v}"] = int((d < 0).sum())
            rows.append(r)
    corp = pd.DataFrame([{"kind": kind, "group": g, "own_mean": own[g]["mean"].mean(),
                          "own_lda": own[g]["lda"].mean()} for g in groups])
    return pd.DataFrame(rows), corp


def joint_share(d: pd.DataFrame, xs: list[str]) -> float:
    """順位に直し，ターゲットとソースの指示変数を入れた後，xs が残りの二乗和の何割を減らすか。"""
    d = d.dropna(subset=["y"] + xs)
    y = stats.rankdata(d.y)
    X0 = design(d, True, True)
    X1 = np.column_stack([X0] + [stats.rankdata(d[x]) for x in xs])
    sse = lambda X: float(((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum())  # noqa: E731
    return 1 - sse(X1) / sse(X0)


def analyse(pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    e9 = pd.read_csv(OUT / "e9_pair.csv")[["kind", "target", "source", "align", "shift_emo"]]
    per = pd.read_csv(OUT / "per_pair.csv")[["kind", "target", "source", "pad", "coverage_t_by_s"]]
    pair = pair.merge(e9, on=["kind", "target", "source"]).merge(per, on=["kind", "target", "source"])
    G = {k: gains(k) for k in KINDS}
    seven = sorted(G["audio_wavlm"].target.unique())
    settings = [("audio 7 groups", "audio_wavlm", seven), ("audio 6 groups", "audio_wavlm", SIX),
                ("video", "video", SIX), ("audio+video", "fuse-audio-video", SIX)]
    out, joint, overlap = [], [], []
    for name, kind, groups in settings:
        reps = common_reps(G[kind])
        pq = pair[(pair.kind == kind) & pair.target.isin(groups) & pair.source.isin(groups)]
        e = fit(pq, "eff_lda")["resid"]
        for q in ("align", "pad", "coverage_t_by_s"):
            overlap.append({"setting": name, "other": q,
                            "rho_resid": stats.spearmanr(e, fit(pq, q)["resid"].reindex(e.index))[0]})
        rows = []
        for K in K_ORDER + ["src"]:
            if K == "src":
                d = pq.rename(columns={"src_uar": "y"})
            else:
                h = G[kind]
                h = h[(h.K == K) & h.target.isin(groups) & h.source.isin(groups) & h.rep.isin(reps)]
                d = (h.groupby(["target", "source"], as_index=False).gain.mean()
                     .rename(columns={"gain": "y"}).merge(pq, on=["target", "source"]))
            for q in MAIN_Q + ["align", "shift_emo"]:
                rows.append({"setting": name, "reps": len(reps), "K": K, "quantity": q,
                             "rho_raw": stats.spearmanr(d.y, d[q])[0], **partial_rank(d, "y", q)})
            df_res = len(d) - (d.target.nunique() + d.source.nunique() - 1)
            for jn, xs in JOINT.items():
                joint.append({"setting": name, "K": K, "quantities": jn, "share": joint_share(d, xs),
                              "df_resid": df_res, "chance_approx": len(xs) / df_res})
        r = pd.DataFrame(rows)
        m = r.quantity.isin(MAIN_Q)
        r.loc[m, "p_holm"] = holm(r.loc[m, "p"].to_numpy())
        out.append(r)
    return pd.concat(out, ignore_index=True), pd.DataFrame(joint), pd.DataFrame(overlap)


def modality(pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in KINDS:
        h = pair[(pair.kind == k) & pair.target.isin(SIX) & pair.source.isin(SIX)]
        rows.append({"kind": k, "n_pairs": len(h), **{f"{q}_{f}": getattr(h[q], f)()
                                                      for q in MAIN_Q for f in ("mean", "min", "max")},
                     "n_reversed_lda": int(h.n_reversed_lda.sum())})
    return pd.DataFrame(rows)


def main() -> int:
    pairs, corps = zip(*[shape(k) for k in KINDS])
    pair, corp = pd.concat(pairs, ignore_index=True), pd.concat(corps, ignore_index=True)
    pair.to_csv(OUT / "e10_pair.csv", index=False)
    corp.to_csv(OUT / "e10_corpus.csv", index=False)
    part, joint, overlap = analyse(pair)
    overlap.to_csv(OUT / "e10_overlap.csv", index=False)
    part.to_csv(OUT / "e10_partial.csv", index=False)
    joint.to_csv(OUT / "e10_joint.csv", index=False)
    modality(pair).to_csv(OUT / "e10_modality.csv", index=False)
    fig(pair, part, OUT / "fig_e10.png")
    print(f"書いた: {OUT}")
    return 0


def fig(pair: pd.DataFrame, part: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor=SURFACE)
    col = {"audio_wavlm": C_MERGED, "video": C_TGT, "fuse-audio-video": C_SRC}
    for ax, v in zip(axes[:2], ("mean", "lda")):
        _style(ax)
        for k in KINDS:
            h = pair[pair.kind == k]
            ax.scatter(h[f"eff_{v}"], h.src_uar, s=18, color=col[k], label=k, zorder=3)
        ax.axhline(0.25, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_xlabel(f"eff_{v}: separation on target along source direction / target's own",
                      fontsize=8, color=INK2)
        ax.set_ylabel("source-only UAR on target", fontsize=8, color=INK2)
        ax.set_title({"mean": "direction = difference of class means (no spread)",
                      "lda": "direction = LDA (uses within-class spread)"}[v],
                     fontsize=8.5, color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=7)
    ax = axes[2]
    _style(ax)
    h = part[part.K == "src"].pivot(index="setting", columns="quantity", values="r_partial_rank")
    qs = ["align", "eff_mean", "eff_lda", "tdp_lda"]
    for j, (q, c) in enumerate(zip(qs, (MUTED, C_MERGED, C_TGT, C_SRC))):
        ax.barh(np.arange(len(h)) + (j - 1.5) * 0.2, h[q], height=0.2, color=c, label=q)
    ax.set_yticks(range(len(h)), h.index, fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("source-only UAR: rank partial r (after target and source effects)", fontsize=8.5,
                 color=INK, loc="left")
    fig.suptitle("EXPLORATORY E10: does the source's decision direction separate the target's emotions?",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
