"""【探索的】E32 の 5 節（指針 D の段 1）：ソースのみの分類器を重みを付けて束ねたときの e と価値の予言。新しい学習なし。

規則は `docs/exploratory-analyses.md` E32 の 5 節（段 1 の判定の決まりは計算の前に書いた）。分類器は線形なので，
ソースごとのテストの点数（`results/merged_scores_<kind>/` の `s_<ソース>_r<r>`）をラベル無しで尺度をそろえて重み付きで足すと，
束ねた分類器の点数になる。束ね方は 本数（∝ N_s），等しい，価値（∝ ē/(1 − ē²)．e はテストのラベルで測る仕組みの確かめ）。
比べる相手は実際の全ソースの下書き（`results/multisource_scores_<kind>/` の `s_all_r<r>`）と，向きのずれが独立なら届く上限 e_ind。
価値の予言は `explore_source_value.n_model`（4 節と同じ模型．基準の揺れを入れる）。

    ./.venv/bin/python exp/explore_source_combine.py

出力：`results/explore_relation/e32d_targets.csv`（種類 × ターゲット × 束ね方），`e32d_summary.csv`，`e32d_weights.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import explore_compare as EC  # noqa: E402
import explore_source_value as SV  # noqa: E402
from data import load_any  # noqa: E402
from explore_synergy import PAIRS, d_o  # noqa: E402

DRAFTS = ["count", "equal", "value", "pooled", "ind"]


def scale(L: np.ndarray) -> float:
    """ラベルを使わない尺度：感情の対の差 z_kl の分散の 6 対平均の平方根。"""
    Lc = L - L.mean(0)
    return float(np.sqrt(np.mean([np.var(Lc[:, k] - Lc[:, l]) for k, l in PAIRS])))


def d_pairs(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([d for d, _ in d_o(L, y)])


def curve_uar(us: np.ndarray, kp: float) -> float:
    if not np.isfinite(kp) or kp > SV.KS_CURVE[-1]:
        return np.nan
    return float(np.interp(np.log(max(kp, SV.KS_CURVE[0])), np.log(SV.KS_CURVE), us))


def run_kind(kind: str) -> tuple[list[dict], list[dict]]:
    e13 = SV.load_e13(kind)
    qd = pd.read_csv(SV.FIX / "e28t9_targets.csv").query("kind == @kind").set_index("target").q_dir
    targets = sorted(qd.index)
    cur = SV.curves(kind, targets)
    n_rows = {}
    rows, wrows = [], []
    for t in targets:
        own = e13[e13.target == t].groupby("pair").own_d.mean().reindex(SV.PAIR_NAMES).to_numpy()
        zm = np.load(ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz")
        zs = np.load(ROOT / "results" / f"merged_scores_{kind}" / f"{t}.npz")
        srcs = [str(s) for s in zm["src_names"]]
        for s in srcs:
            if s not in n_rows:
                n_rows[s] = len(load_any(s, kind)[0])
        reps = [r for r in EC.REPS if f"s_all_r{r}" in zm.files]
        single = {s: [] for s in srcs}
        Ls_all, ys, pooled = [], [], []
        for r in reps:
            if not np.array_equal(zm[f"test_idx_r{r}"], zs[f"test_idx_r{r}"]):
                raise ValueError(f"テストの行が合わない: {kind} {t} r{r}")
            y = zm[f"y_test_r{r}"]
            Ls = {s: zs[f"s_{s}_r{r}"].astype(np.float64) for s in srcs}
            Ls = {s: L / scale(L) for s, L in Ls.items()}
            for s in srcs:
                single[s].append(d_pairs(Ls[s], y))
            Ls_all.append(Ls)
            ys.append(y)
            pooled.append(d_pairs(zm[f"s_all_r{r}"].astype(np.float64), y))
        e_s = {s: np.clip(np.mean(single[s], axis=0) / own, 0.0, 0.999) for s in srcs}
        eb = {s: float(e_s[s].mean()) for s in srcs}
        wts = {"count": {s: float(n_rows[s]) for s in srcs},
               "equal": {s: 1.0 for s in srcs},
               "value": {s: eb[s] / (1 - eb[s] ** 2) for s in srcs}}
        for name, w in wts.items():
            tot = sum(w.values())
            wts[name] = {s: v / tot for s, v in w.items()}
            wrows += [{"kind": kind, "target": t, "draft": name, "source": s, "weight": wts[name][s], "e_mean": eb[s]}
                      for s in srcs]
        e_d = {}
        for name, w in wts.items():
            dj = [d_pairs(sum(w[s] * Ls[s] for s in srcs), y) for Ls, y in zip(Ls_all, ys)]
            e_d[name] = np.clip(np.mean(dj, axis=0) / own, 0.0, 0.999)
        e_d["pooled"] = np.clip(np.mean(pooled, axis=0) / own, 0.0, 0.999)
        r_ind = sum(e ** 2 / (1 - e ** 2) for e in e_s.values())
        e_d["ind"] = np.clip(np.sqrt(r_ind / (1 + r_ind)), 0.0, 0.999)
        q = float(qd[t])
        for name in DRAFTS:
            row = {"kind": kind, "target": t, "draft": name, "e_mean": float(e_d[name].mean()),
                   "clip_share": float((e_d[name] >= 0.999).mean()),
                   "e_single_mean": float(np.mean(list(eb.values()))), "e_single_max": float(max(eb.values())),
                   "n_sources": len(srcs), "q_dir": q}
            for K in SV.KS:
                n, fl = SV.n_model(own, e_d[name], q, int(K), pos=True)
                row[f"n_K{K}"] = n if fl == 0 else np.inf
                row[f"uar_K{K}"] = curve_uar(cur[t], int(K) + n) if fl == 0 else np.nan
            rows.append(row)
        print(f"  {kind} {t} 済み", flush=True)
    return rows, wrows


def summarize(D: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kind, g in D.groupby("kind"):
        W = g.pivot(index="target", columns="draft", values="e_mean")
        gap = (W["value"] - W["count"]) / (W["ind"] - W["count"])
        row = {"kind": kind, "targets": len(W), "gap_closed_median": float(gap.median()),
               "gap_closed_by_equal_median": float(((W["equal"] - W["count"]) / (W["ind"] - W["count"])).median())}
        for name in DRAFTS:
            h = g[g.draft == name]
            row[f"e_{name}"] = float(h.e_mean.median())
            row[f"clip_{name}"] = float(h.clip_share.mean())
            for K in ("5", "10"):
                row[f"n{K}_{name}"] = float(h[f"n_K{K}"].median())
                if name != "pooled":         # 予言の UAR はターゲットのみの曲線の K + n．曲線（K ≤ 50）の中に入るターゲットだけで実際の下書きと比べる
                    U = g.pivot(index="target", columns="draft", values=f"uar_K{K}")[[name, "pooled"]].dropna()
                    row[f"duar{K}_{name}_vs_pooled"] = float((U[name] - U["pooled"]).median()) if len(U) else np.nan
                    row[f"duar{K}_{name}_targets"] = len(U)
        row["e_single_mean"] = float(g[g.draft == "count"].e_single_mean.median())
        row["e_single_max"] = float(g[g.draft == "count"].e_single_max.median())
        g_ = row["gap_closed_median"]
        row["reading"] = "(b) が主" if g_ >= 0.5 else ("(a) が主" if g_ < 0.2 else "両方")
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    rows, wrows = [], []
    for kind in SV.MAIN:
        r, w = run_kind(kind)
        rows += r
        wrows += w
    D, Wt = pd.DataFrame(rows), pd.DataFrame(wrows)
    S = summarize(D)
    D.to_csv(SV.OUT / "e32d_targets.csv", index=False)
    S.to_csv(SV.OUT / "e32d_summary.csv", index=False)
    Wt.to_csv(SV.OUT / "e32d_weights.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
