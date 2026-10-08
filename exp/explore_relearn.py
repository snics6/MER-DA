"""【探索的】得の仕組みの仮説 MH3・MH4 を，学び直した結果で確かめる（E12．学び直しは `exp/mech.py`）。

比べる相手は地図（`results/map_<kind>/`）の同じ (target, source, K, rep) の行。`mech.py` が終えた反復だけを使う。

MH3（中心のずれ）
- ソースのみ：中心をそろえる（center）・標準化もそろえる（I1）と UAR がどれだけ上がるか。上がり幅が，
  ソースのみの答えの偏り TV（E11）・`shift_emo`・`shift_total` の大きい組ほど大きいか
- D1（ソースのみの出力に，K 本で学んだ感情ごとのずらしを足す）：上がり幅が，合併 − ソースのみ（E11 の MH3c）と
  一致するか。少数の例の働きが位置のずれの修正なら，D1 だけで合併の上がり幅の大部分が出るはず
- 中心をそろえた合併：地図の合併よりどれだけ上がるか。そろえた後は，合併 − ソースのみが小さくなるか

MH4（ターゲットが埋もれる）
- α を交差検証で選んだ合併（wt）の得が，地図の合併（nat）の得より大きいか。特に K ≥ 25 で負の得が消えるか
- wt − nat の上がり幅が，同じターゲット・同じ K の中で，大きいソースほど大きいか
- 選ばれた α の分布。テストで最良の α（オラクル）との差

    ./.venv/bin/python exp/explore_relearn.py                      # 3 種類すべて（結果がある分だけ）
    ./.venv/bin/python exp/explore_relearn.py --kinds audio_hubert # 種類を選ぶ

種類は音声（WavLM），映像，音声（HuBERT．最後の確認に使う別のエンコーダ）。音声の種類では RAVDESS を除く
6 群の設定も出す。`shift_emo`・`shift_total` は E9 の `e9_pair.csv` から読み，そこに無い種類（HuBERT）は
`explore_geometry.geometry` でその場で計算する。

出力：`results/explore_relation/e12_*.csv`・`fig_e12.png`
"""

from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT / "measure"))
from data import groups_for  # noqa: E402
from explore_geometry import geometry  # noqa: E402
from explore_mechanism import bias_table, partial_rank  # noqa: E402
from pair_table import OUT, SIX  # noqa: E402
from summarize_map import C_MERGED, C_SRC, C_TGT, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

KINDS = ("audio_wavlm", "video", "audio_hubert")
K_ORDER = ["1", "5", "10", "25", "50", "all"]
ALPHAS = ("nat", "0.25", "0.5", "0.75")
C_GRID = ("0.01", "0.1", "1", "10")


def read(dirname: str) -> pd.DataFrame:
    fs = sorted((ROOT / "results" / dirname).glob("*.csv"))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str, "alpha": str})
                      for f in fs], ignore_index=True)


def done_reps(mech: pd.DataFrame) -> pd.DataFrame:
    """ターゲット × 反復のうち，mech.py が書き終えたもの（1 回の書き込みで足すので行があれば完了）。"""
    return mech[["target", "rep"]].drop_duplicates()


def rho(a, b) -> float:
    """Spearman の ρ。3 組未満・値が 1 種類なら NaN。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or len(np.unique(a[ok])) < 2 or len(np.unique(b[ok])) < 2:
        return np.nan
    return float(stats.spearmanr(a[ok], b[ok])[0])


def wilcox(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    return float(stats.wilcoxon(x).pvalue) if len(x) >= 6 else np.nan


def e9_pairs(kind: str) -> pd.DataFrame:
    """組ごとの `shift_emo`・`shift_total`（E9）。`e9_pair.csv` に無い種類はその場で計算する。"""
    e9 = pd.read_csv(OUT / "e9_pair.csv")
    e9 = e9[e9.kind == kind]
    if e9.empty:
        e9 = geometry(kind)[1]
    return e9[["target", "source", "shift_emo", "shift_total"]]


def analyse(kind: str) -> dict[str, pd.DataFrame] | None:
    mech = read(f"mech_{kind}")
    if mech.empty:
        return None
    mp = read(f"map_{kind}")
    tr = done_reps(mech)
    mp = mp.merge(tr)
    t = mp[mp.method == "tgt_only"][["target", "K", "rep", "uar"]].rename(columns={"uar": "t"})
    tcv = mp[mp.method == "tgt_only"].set_index(["target", "K", "rep"])[[f"cv_uar_C{c}" for c in C_GRID]]
    s = mp[mp.method == "src_only"][["target", "source", "rep", "uar"]].rename(columns={"uar": "s"})
    m = mp[mp.method == "merged"][["target", "source", "K", "rep", "uar"]].rename(columns={"uar": "m"})
    B = bias_table(kind)[["target", "source", "tv"]]
    e9 = e9_pairs(kind)
    groups = groups_for(kind)
    settings = [(kind, groups)] + ([(f"{kind} 6 groups", SIX)] if kind.startswith("audio_") else [])
    out = {k: [] for k in ("src", "d1", "center", "wt", "wt_alpha")}

    for name, gs in settings:
        sel = lambda d: d[d.target.isin(gs) & d.source.isin(gs)]  # noqa: E731

        # ---- MH3：ソースのみの中心・標準化
        for op in ("center", "I1"):
            x = sel(mech[(mech.op == op) & (mech.method == "src_only")])[["target", "source", "rep", "uar"]]
            x = x.merge(s).assign(d=lambda h: h.uar - h.s)
            pp = x.groupby(["target", "source"], as_index=False).agg(d=("d", "mean"), s=("s", "mean"),
                                                                       u=("uar", "mean"))
            pp = pp.merge(B).merge(e9)
            row = {"setting": name, "op": op, "n_pairs": len(pp), "n_rows": len(x),
                   "mean_s": float(pp.s.mean()), "mean_op": float(pp.u.mean()),
                   "mean_diff": float(pp.d.mean()), "median_diff": float(pp.d.median()),
                   "n_up": int((pp.d > 0).sum()), "p_wilcoxon": wilcox(pp.d)}
            for q in ("tv", "shift_emo", "shift_total"):
                row[f"rho_{q}"] = rho(pp.d, pp[q])
                pr = partial_rank(pp, "d", q)
                row[f"r_partial_{q}"] = pr["r_partial_rank"] if pr else np.nan
                row[f"p_partial_{q}"] = pr["p"] if pr else np.nan
            out["src"].append(row)

        # ---- MH3c：D1 の上がり幅と，合併の上がり幅
        for op in ("none", "center"):
            x = sel(mech[(mech.op == op) & (mech.method == "src_only+D1")])[["target", "source", "K", "rep", "uar"]]
            base = s if op == "none" else sel(mech[(mech.op == "center") & (mech.method == "src_only")])[
                ["target", "source", "rep", "uar"]].rename(columns={"uar": "s"})
            x = x.rename(columns={"uar": "d1"}).merge(base).merge(m).merge(t)
            for K in K_ORDER:
                h = x[x.K == K]
                if h.empty:
                    continue
                pp = h.groupby(["target", "source"], as_index=False)[["d1", "s", "m", "t"]].mean().merge(B)
                pp["d1_s"], pp["m_s"] = pp.d1 - pp.s, pp.m - pp.s
                pr = partial_rank(pp, "d1_s", "tv")
                out["d1"].append({"setting": name, "op": op, "K": K, "n_pairs": len(pp),
                                  "mean_d1_minus_s": float(pp.d1_s.mean()),
                                  "mean_m_minus_s": float(pp.m_s.mean()),
                                  "mean_d1_minus_t": float((pp.d1 - pp.t).mean()),
                                  "share_of_merged": float(pp.d1_s.sum() / pp.m_s.sum()) if pp.m_s.sum() else np.nan,
                                  "rho_d1s_ms": rho(pp.d1_s, pp.m_s),
                                  "r_partial_d1s_tv": pr["r_partial_rank"] if pr else np.nan,
                                  "p_partial_d1s_tv": pr["p"] if pr else np.nan})

        # ---- MH3：中心をそろえた合併
        x = sel(mech[(mech.op == "center") & (mech.method == "merged")])[["target", "source", "K", "rep", "uar"]]
        sc = sel(mech[(mech.op == "center") & (mech.method == "src_only")])[["target", "source", "rep", "uar"]]
        x = x.rename(columns={"uar": "mc"}).merge(m).merge(t).merge(s).merge(sc.rename(columns={"uar": "sc"}))
        for K in K_ORDER:
            h = x[x.K == K]
            if h.empty:
                continue
            pp = h.groupby(["target", "source"], as_index=False)[["mc", "m", "t", "s", "sc"]].mean().merge(B)
            pp["d"] = pp.mc - pp.m
            pr = partial_rank(pp, "d", "tv")
            out["center"].append({"setting": name, "K": K, "n_pairs": len(pp),
                                  "gain_map": float((pp.m - pp.t).mean()), "gain_center": float((pp.mc - pp.t).mean()),
                                  "mean_diff": float(pp.d.mean()), "n_up": int((pp.d > 0).sum()),
                                  "p_wilcoxon": wilcox(pp.d),
                                  "m_minus_s_map": float((pp.m - pp.s).mean()),
                                  "m_minus_s_center": float((pp.mc - pp.sc).mean()),
                                  "r_partial_diff_tv": pr["r_partial_rank"] if pr else np.nan,
                                  "p_partial_diff_tv": pr["p"] if pr else np.nan})

        # ---- MH4：α を選んだ合併
        w = sel(mech[(mech.op == "wt") & (mech.method == "merged")]).copy()
        w = w.merge(m).merge(t)
        test_cols = {(a, c): f"test_a{a}_C{c}" for a in ALPHAS for c in C_GRID}
        w["oracle"] = w[list(test_cols.values())].max(axis=1)
        # ターゲットのみも候補に入れて交差検証で選ぶ（ターゲットのみの袋は合併のターゲットの袋と同じ）
        cvw = w[[f"cv_a{a}_C{c}" for a in ALPHAS for c in C_GRID]].max(axis=1)
        cvt = tcv.reindex(list(zip(w.target, w.K, w.rep))).max(axis=1).to_numpy()
        w["safe"] = np.where(cvt > cvw.to_numpy() + 1e-12, w.t, w.uar)
        for K in K_ORDER:
            h = w[w.K == K]
            if h.empty:
                continue
            pp = h.groupby(["target", "source"], as_index=False).agg(
                wt=("uar", "mean"), m=("m", "mean"), t=("t", "mean"), oracle=("oracle", "mean"),
                safe=("safe", "mean"), n_src=("n_src", "mean"))
            pp["d"] = pp.wt - pp.m
            rhos = [r for r in (rho(g.n_src, g.d) for _, g in pp.groupby("target") if len(g) >= 4) if np.isfinite(r)]
            out["wt"].append({"setting": name, "K": K, "n_pairs": len(pp), "n_rows": len(h),
                              "gain_nat": float((pp.m - pp.t).mean()), "gain_wt": float((pp.wt - pp.t).mean()),
                              "gain_safe": float((pp.safe - pp.t).mean()),
                              "gain_oracle": float((pp.oracle - pp.t).mean()),
                              "neg_nat": int((pp.m - pp.t < 0).sum()), "neg_wt": int((pp.wt - pp.t < 0).sum()),
                              "neg_safe": int((pp.safe - pp.t < 0).sum()),
                              "mean_wt_minus_nat": float(pp.d.mean()), "p_wilcoxon": wilcox(pp.d),
                              "rho_d_nsrc_within_target": float(np.nanmean(rhos)) if rhos else np.nan,
                              "n_targets_rho": len(rhos)})
            for a in ALPHAS:
                out["wt_alpha"].append({"setting": name, "K": K, "alpha": a,
                                        "share_chosen": float((h.alpha == a).mean())})
    return {k: pd.DataFrame(v) for k, v in out.items()} | {"_reps": tr}


def analyse_cwt(kind: str) -> dict[str, pd.DataFrame] | None:
    """MH3 と MH4 を同時に取り除いた合併（`mech.py --mode cwt` → `results/mech2_<kind>/`）を，
    地図の合併・中心をそろえた合併・α を選んだ合併・ターゲットのみと比べる。"""
    c2 = read(f"mech2_{kind}")
    if c2.empty:
        return None
    tr = done_reps(c2)
    mp = read(f"map_{kind}").merge(tr)
    mech = read(f"mech_{kind}").merge(tr)
    t = mp[mp.method == "tgt_only"][["target", "K", "rep", "uar"]].rename(columns={"uar": "t"})
    m = mp[mp.method == "merged"][["target", "source", "K", "rep", "uar"]].rename(columns={"uar": "m"})
    cw = c2[(c2.op == "cwt") & (c2.method == "merged")]
    x = cw[["target", "source", "K", "rep", "uar", "alpha"]].rename(columns={"uar": "cwt"}).merge(m).merge(t)
    for op, col in (("center", "center"), ("wt", "wt")):
        y = mech[(mech.op == op) & (mech.method == "merged")][["target", "source", "K", "rep", "uar"]]
        x = x.merge(y.rename(columns={"uar": col}), how="left")
    e10 = pd.read_csv(OUT / "e10_pair.csv")
    e10 = e10[e10.kind == kind].set_index(["target", "source"]).eff_lda
    groups = groups_for(kind)
    settings = [(kind, groups)] + ([(f"{kind} 6 groups", SIX)] if kind.startswith("audio_") else [])
    rows, alphas, pairs = [], [], []
    for name, gs in settings:
        h0 = x[x.target.isin(gs) & x.source.isin(gs)]
        for K in K_ORDER:
            h = h0[h0.K == K]
            if h.empty:
                continue
            pp = h.groupby(["target", "source"], as_index=False)[["cwt", "m", "t", "center", "wt"]].mean()
            pp["eff_lda"] = [e10.get((a, b), np.nan) for a, b in zip(pp.target, pp.source)]
            pp["loss_cwt"] = pp.cwt - pp.t
            row = {"setting": name, "K": K, "n_pairs": len(pp), "n_rows": len(h)}
            for col in ("m", "center", "wt", "cwt"):
                if pp[col].notna().all():
                    g = pp[col] - pp.t
                    row[f"gain_{col}"] = float(g.mean())
                    row[f"neg1_{col}"] = int((g < -0.01).sum())
                    row[f"worst_{col}"] = float(g.min())
            row["rho_losscwt_efflda"] = rho(pp.loss_cwt, pp.eff_lda)
            pr = partial_rank(pp.dropna(subset=["eff_lda"]), "loss_cwt", "eff_lda")
            row["r_partial_losscwt_efflda"] = pr["r_partial_rank"] if pr else np.nan
            row["p_partial_losscwt_efflda"] = pr["p"] if pr else np.nan
            neg = pp[pp.loss_cwt < -0.01]
            row["eff_lda_neg_mean"] = float(neg.eff_lda.mean()) if len(neg) else np.nan
            row["eff_lda_rest_mean"] = float(pp[pp.loss_cwt >= -0.01].eff_lda.mean())
            rows.append(row)
            for a in sorted(h.alpha.astype(str).unique()):
                alphas.append({"setting": name, "K": K, "alpha": a,
                               "share_chosen": float((h.alpha.astype(str) == a).mean())})
            pairs.append(pp.assign(setting=name, K=K))
    return {"cwt": pd.DataFrame(rows), "cwt_alpha": pd.DataFrame(alphas),
            "cwt_pairs": pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()}


def main() -> int:
    ap = argparse.ArgumentParser(description="E12：学び直しの結果の解析")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    kinds = ap.parse_args().kinds
    cw = {k: analyse_cwt(k) for k in kinds}
    cw = {k: v for k, v in cw.items() if v is not None}
    for name in ("cwt", "cwt_alpha", "cwt_pairs"):
        if cw:
            pd.concat([r[name] for r in cw.values()], ignore_index=True).to_csv(OUT / f"e12_{name}.csv", index=False)
    res = {k: analyse(k) for k in kinds}
    res = {k: v for k, v in res.items() if v is not None}
    if not res:
        print("mech の結果がまだ無い")
        return 1
    state = {}
    for name in ("src", "d1", "center", "wt", "wt_alpha"):
        pd.concat([r[name] for r in res.values()], ignore_index=True).to_csv(OUT / f"e12_{name}.csv", index=False)
    for k, r in res.items():
        state[k] = {t: sorted(int(x) for x in g.rep) for t, g in r["_reps"].groupby("target")}
    (OUT / "e12_state.json").write_text(json.dumps(state, ensure_ascii=False))
    fig(OUT / "fig_e12.png")
    print(f"書いた: {OUT}（{json.dumps(state, ensure_ascii=False)}）")
    return 0


def fig(path: Path) -> None:
    src = pd.read_csv(OUT / "e12_src.csv")
    d1 = pd.read_csv(OUT / "e12_d1.csv", dtype={"K": str})
    wt = pd.read_csv(OUT / "e12_wt.csv", dtype={"K": str})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    h = src.set_index(["setting", "op"]).mean_diff.unstack()
    h.plot.bar(ax=ax, color=[C_MERGED, C_SRC], rot=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("source-only UAR change", fontsize=8, color=INK2)
    ax.set_title("MH3: centring / per-corpus standardisation", fontsize=8.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    _style(ax)
    for (s, op), g in d1[d1.op == "none"].groupby(["setting", "op"]):
        g = g.set_index("K").reindex(K_ORDER)
        ax.plot(range(6), g.mean_d1_minus_s, marker="o", label=f"{s} D1 − src")
        ax.plot(range(6), g.mean_m_minus_s, marker="s", linestyle="--", label=f"{s} merged − src")
    ax.set_xticks(range(6), K_ORDER)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("MH3c: offset-only fix (D1) vs merged", fontsize=8.5, color=INK, loc="left")
    ax = axes[2]
    _style(ax)
    for s, g in wt.groupby("setting"):
        g = g.set_index("K").reindex([k for k in K_ORDER if k in set(g.K)])
        x = [K_ORDER.index(k) for k in g.index]
        ax.plot(x, g.gain_nat, marker="o", color=MUTED, label=f"{s} nat")
        ax.plot(x, g.gain_wt, marker="s", label=f"{s} α by CV")
    ax.set_xticks(range(6), K_ORDER)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("gain over target-only", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("MH4: choosing the target weight α", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
