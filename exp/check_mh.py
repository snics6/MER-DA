"""研究計画（`researchvision.md` 8.4）に登録した，得の仕組みの仮説 MH1〜MH5 の主な検定を，第 2 エンコーダ（HuBERT）の地図で確かめる。

主判定は `docs/exploratory-analyses.md` E11 の主な検定 6 つと同じ量・同じ向き・同じ検定である。計算は
`explore_mechanism.py`（E11）・`explore_geometry.py`（E9）・`explore_shape.py`（E10）・`explore_pairs.py`（E13）の
関数をそのまま使う。比べるために WavLM（`audio_wavlm`．結果を見た後に決めた探索的な値）も同じ台本で計算し，
保存済みの E11 の値（30 回）と一致することを確かめる。一致しなければ何も書かずに止まる。

- 主判定：音声 7 群（42 組）。片側，6 つに Holm，全体の有意水準 5%。「支持」＝予言の向きの片側 p の Holm 後が
  5% 未満，「反証」＝逆の向きの片側 p（6 つにまとめて Holm）が 5% 未満，どちらでもなければ「判定できない」
- 副判定（記述のみ）：6 群（RAVDESS を除く）での同じ 6 つ，MH1 の傾き，学習曲線，追い越しの K の予言と観測の
  順位相関，まとめのモデル，E13 の位置による損・向きによる損，2 つのエンコーダでの組ごとの量の順位相関

    ./.venv/bin/python exp/check_mh.py                      # HuBERT と WavLM（結果を書く）
    ./.venv/bin/python exp/check_mh.py --only audio_wavlm   # 台本の検算（E11 の値の再現を確かめるだけ．何も書かない）

出力：`results/check_mh/primary.csv`・`secondary.csv`・`curves.csv`・`model.csv`・`e13.csv`・`pairs.csv`・
`encoders.csv`・`fig_mh.png`・`provenance.json`
"""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from explore_geometry import geometry  # noqa: E402
from explore_mechanism import (K_ORDER, MIN_GAP, beta_table, bias_table, crossing, curve, d_eff,  # noqa: E402
                               fit_curves, kstar, kx_pred, load_map, one_sided, partial_rank, sub, uar4)
from explore_pairs import measure  # noqa: E402
from pair_table import OUT_ROOT, RESULTS, SIX, fit, holm  # noqa: E402
from explore_shape import shape  # noqa: E402
from summarize_map import C_MERGED, C_SRC, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

OUT = OUT_ROOT / "check_mh"
E11 = RESULTS / "explore_relation" / "snapshots" / "20261002-2100_e11_30rep" / "e11_primary.csv"
KINDS = ("audio_hubert", "audio_wavlm")           # 先頭が判定に使う種類
ALPHA = 0.05
TESTS = ["MH2", "MH3a", "MH3b", "MH3c", "MH4", "MH5"]
LABEL = {"MH2": "K_x ~ e_geom (partial)", "MH3a": "top predicted = top beta (rate)",
         "MH3b": "TV ~ shift_emo (partial)", "MH3c": "merged - src at K=5 ~ TV (partial)",
         "MH4": "phi ~ alpha within target (mean rho)", "MH5": "gain K=5 ~ -|d' diff| (partial)"}
E11_NAME = {"MH2": "H2 K_x ~ e_geom (partial)", "MH3a": "H3 top predicted = top beta",
            "MH3b": "H3 TV ~ shift_emo (partial)", "MH3c": "H3c merged - src at K=5 ~ TV (partial)",
            "MH4": "H4 phi ~ alpha within target", "MH5": "H5 gain K=5 ~ -|d' diff| (partial)"}


def inputs(kind: str) -> dict:
    """地図（反復平均）と，E9・E10・E11 で使う特徴空間の量。"""
    g, reps = load_map(kind)
    e9c, e9 = geometry(kind)
    e10, e10c = shape(kind)
    return {"map": g, "reps": reps, "e9": e9, "e9c": e9c, "e10": e10, "e10c": e10c,
            "bias": bias_table(kind), "beta": beta_table(kind)}


def partial_test(test: str, pr: dict | None) -> dict:
    r, p = one_sided(pr)                     # 予言はどれも正の向き
    return {"test": test, "stat": r, "n": pr["n"] if pr else np.nan, "df": pr["df"] if pr else np.nan,
            "p_one_sided": p, "p_one_sided_opposite": 1 - p}


def analyse(name: str, kind: str, groups: list[str], I: dict) -> tuple[list[dict], dict]:
    """E11 の主な検定 6 つ（`explore_mechanism.main` と同じ計算）と，記述用の量。"""
    g = sub(I["map"], groups)
    prim, sec = [], {}

    # ---- MH2：追い越しの K と向きの効率
    cv = fit_curves(g)
    P = g[g.K == "1"][["target", "source", "s"]].copy()
    P["delta"] = P.target.map(cv.delta)
    P["q"] = P.target.map(cv.q)
    P["e_emp"] = d_eff(P.s.to_numpy()) / P.delta
    e = I["e10"].set_index(["target", "source"]).eff_lda
    P["e_geom"] = [e.get((t, s), np.nan) for t, s in zip(P.target, P.source)]
    P["kx_obs"] = [crossing(g[(g.target == t) & (g.source == s)], cv.K_all[t])
                   for t, s in zip(P.target, P.source)]
    ks = kstar(g)
    P["kstar_idx"] = [ks[(t, s)] for t, s in zip(P.target, P.source)]
    for ecol in ("e_emp", "e_geom"):
        P[f"kx_{ecol}"] = [kx_pred(q, v, cv.K_all[t]) for q, v, t in zip(P.q, P[ecol], P.target)]
        P[f"log_e2_{ecol}"] = np.log(np.clip(P[ecol], 1e-3, 0.999) ** 2 /
                                     (1 - np.clip(P[ecol], 1e-3, 0.999) ** 2))
        sec[f"rho_kxpred_kxobs_{ecol}"] = stats.spearmanr(P[f"kx_{ecol}"], P.kx_obs, nan_policy="omit")[0]
        pr = partial_rank(P, "kx_obs", f"log_e2_{ecol}")
        sec[f"partial_kxobs_{ecol}"] = pr["r_partial_rank"] if pr else np.nan
    prim.append(partial_test("MH2", partial_rank(P, "kx_obs", "log_e2_e_geom")))
    tf = fit(P.assign(v=P.kx_obs.apply(np.log)), "v")
    sec["rho_target_effect_logkx_logq"] = stats.spearmanr(tf["a"].reindex(cv.index), np.log(cv.q))[0]

    # ---- MH3：答えの偏りとずれの向き
    B = I["bias"].merge(I["beta"], on=["target", "source"])
    B = B.merge(I["e9"][["target", "source", "shift_emo", "shift_total"]], on=["target", "source"])
    B = sub(B, groups)
    beta = B[[f"beta_{k}" for k in range(4)]].to_numpy()
    pred = B[[f"pred_{k}" for k in range(4)]].to_numpy()
    match = beta.argmax(1) == pred.argmax(1)
    n_hit, n = int(match.sum()), len(match)
    prim.append({"test": "MH3a", "stat": n_hit / n, "n": n, "df": np.nan,
                 "p_one_sided": stats.binomtest(n_hit, n, 0.25, alternative="greater").pvalue,
                 "p_one_sided_opposite": stats.binomtest(n_hit, n, 0.25, alternative="less").pvalue})
    prim.append(partial_test("MH3b", partial_rank(B, "tv", "shift_emo")))
    sec["tv_mean"], sec["tv_max"] = float(B.tv.mean()), float(B.tv.max())
    pr = partial_rank(B, "src_uar", "tv")
    sec["partial_srcuar_tv"] = pr["r_partial_rank"] if pr else np.nan

    # ---- MH3c：K=5 の「合併 − ソースのみ」と答えの偏り
    h5 = g[g.K == "5"].assign(m_s=lambda x: x.m - x.s).merge(B, on=["target", "source"])
    prim.append(partial_test("MH3c", partial_rank(h5, "m_s", "tv")))
    sec["mean_m_minus_s_K5"] = float(h5.m_s.mean())

    # ---- MH4：同じターゲット・同じ K の中で，合併の位置 φ とターゲットの割合 α
    signs = []
    for K in K_ORDER[1:]:
        for _, h in g[g.K == K].groupby("target"):
            h = h[(h.t - h.s).abs() >= MIN_GAP]
            if len(h) >= 4 and h.alpha.nunique() >= 4:
                signs.append(stats.spearmanr(h.alpha, (h.m - h.s) / (h.t - h.s))[0])
    prim.append({"test": "MH4", "stat": float(np.mean(signs)), "n": len(signs), "df": np.nan,
                 "p_one_sided": stats.wilcoxon(signs, alternative="greater").pvalue,
                 "p_one_sided_opposite": stats.wilcoxon(signs, alternative="less").pvalue})
    sec["mh4_n_positive"] = int((np.array(signs) > 0).sum())

    # ---- MH5：K=5 の得と，感情の分かれ方 dprime の差
    x = I["e9c"].set_index("group").dprime.reindex(groups)
    D = g[g.K == "5"][["target", "source", "gain"]].copy()
    D["neg_absdiff"] = [-abs(x[t] - x[s]) for t, s in zip(D.target, D.source)]
    prim.append(partial_test("MH5", partial_rank(D, "gain", "neg_absdiff")))

    # ---- 記述：MH1 の傾き，学習曲線，まとめのモデル
    h1 = g[g.K == "1"]
    lr = stats.linregress((h1.s - h1.t).to_numpy(), h1.gain.to_numpy())
    sec["mh1_slope_K1"], sec["mh1_r2_K1"] = lr.slope, lr.rvalue ** 2
    sec["mean_alpha_K5"] = float(g[g.K == "5"].alpha.mean())
    for ecol in ("e_emp", "e_geom"):
        E = P.set_index(["target", "source"])[ecol]
        gg = g.copy()
        gg["e"] = [E.get((t, s), np.nan) for t, s in zip(gg.target, gg.source)]
        gg["delta"] = gg.target.map(cv.delta)
        gg["Kn"] = [int(k) if k != "all" else cv.K_all[t] for k, t in zip(gg.K, gg.target)]
        gg["t_hat"] = uar4(curve(gg.Kn, gg.delta, gg.target.map(cv.q)))
        gg = gg.dropna(subset=["e"])

        def predict(gam, gg=gg):
            return uar4(gg.delta * (gg.e + (1 - gg.e) * gg.alpha ** gam)) - gg.t_hat

        gams = np.exp(np.linspace(np.log(0.05), np.log(5), 200))
        sse = [float(((gg.gain - predict(c)) ** 2).sum()) for c in gams]
        sec[f"model_gamma_{ecol}"] = float(gams[int(np.argmin(sse))])
        sec[f"model_r2_{ecol}"] = 1 - min(sse) / float(((gg.gain - gg.gain.mean()) ** 2).sum())

    pairs = P[["target", "source", "e_emp", "e_geom", "kx_obs", "kx_e_geom", "kx_e_emp", "kstar_idx"]].merge(
        B[["target", "source", "tv", "shift_emo", "shift_total", "src_uar"]], on=["target", "source"]).merge(
        h5[["target", "source", "m_s", "gain"]].rename(columns={"gain": "gain_K5"}), on=["target", "source"])
    pairs["top_pred"], pairs["top_beta"] = pred.argmax(1), beta.argmax(1)
    pairs["neg_absdiff_dprime"] = [-abs(x[t] - x[s]) for t, s in zip(pairs.target, pairs.source)]
    curves = cv.reset_index()
    for r in prim:
        r["setting"], r["kind"] = name, kind
    return prim, {"sec": {"setting": name, "kind": kind, "n_reps": len(I["reps"]), **sec},
                  "pairs": pairs.assign(setting=name, kind=kind), "curves": curves.assign(setting=name, kind=kind)}


def e13(kind: str, groups_by_setting: dict[str, list[str]]) -> pd.DataFrame:
    """ソースのみの分類器の d と o（`explore_pairs.measure`）を組ごとに 6 対で平均し，ソースのみの UAR と比べる。"""
    M = measure(kind)
    rows = []
    for name, gs in groups_by_setting.items():
        x = M[M.target.isin(gs) & M.source.isin(gs)]
        pp = x.groupby(["target", "source"], as_index=False).agg(
            loss_offset=("loss_offset", "mean"), loss_direction=("loss_direction", "mean"),
            src_uar=("src_uar", "first"))
        row = {"setting": name, "kind": kind, "n_pairs": len(pp),
               "r_model_actual": float(np.corrcoef(x.acc_model, x.acc_actual)[0, 1]),
               "mean_abs_err": float((x.acc_model - x.acc_actual).abs().mean()),
               "mean_loss_offset": float(pp.loss_offset.mean()),
               "mean_loss_direction": float(pp.loss_direction.mean())}
        for q in ("loss_offset", "loss_direction"):
            pr = partial_rank(pp, "src_uar", q)
            row[f"r_partial_srcuar_{q}"] = pr["r_partial_rank"] if pr else np.nan
            row[f"p_partial_srcuar_{q}"] = pr["p"] if pr else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(p_pred: float, p_opp: float) -> str:
    if p_pred < ALPHA:
        return "支持"
    if p_opp < ALPHA:
        return "反証"
    return "判定できない"


def run(kinds: tuple[str, ...]) -> tuple[pd.DataFrame, list[dict]]:
    prim, extra = [], []
    for kind in kinds:
        I = inputs(kind)
        groups = sorted(I["map"].target.unique())
        settings = {kind: groups, f"{kind} 6 groups": SIX}
        for name, gs in settings.items():
            p, x = analyse(name, kind, gs, I)
            h = pd.DataFrame(p)
            h["p_holm"] = holm(h.p_one_sided.to_numpy())
            h["p_holm_opposite"] = holm(h.p_one_sided_opposite.to_numpy())
            h["verdict"] = [verdict(a, b) for a, b in zip(h.p_holm, h.p_holm_opposite)]
            prim.append(h)
            extra.append(x)
        extra.append({"e13": e13(kind, settings)})
    return pd.concat(prim, ignore_index=True), extra


def check_e11(prim: pd.DataFrame) -> float:
    """WavLM の主判定の値が，保存済みの E11（30 回）と一致するか。差の最大を返す。"""
    ref = pd.read_csv(E11)
    worst = 0.0
    for setting in ("audio_wavlm", "audio_wavlm 6 groups"):
        r = ref[ref.setting == setting].set_index("test")
        h = prim[prim.setting == setting].set_index("test")
        for t in TESTS:
            a, b = r.loc[E11_NAME[t]], h.loc[t]
            worst = max(worst, abs(a.stat - b.stat), abs(a.p - b.p_one_sided), abs(a.p_holm - b.p_holm))
    return worst


def encoders(pairs: pd.DataFrame) -> pd.DataFrame:
    """同じ組の量が，2 つのエンコーダでどれだけ似ているか（そのままと，ターゲット・ソースの効果を除いた残り）。"""
    a = pairs[pairs.setting == KINDS[0]].set_index(["target", "source"])
    b = pairs[pairs.setting == KINDS[1]].set_index(["target", "source"]).reindex(a.index)
    rows = []
    for q in ("e_geom", "e_emp", "kx_obs", "tv", "shift_emo", "m_s", "gain_K5", "src_uar", "neg_absdiff_dprime"):
        d = pd.DataFrame({"target": a.index.get_level_values(0), "source": a.index.get_level_values(1),
                          "x": a[q].to_numpy(), "y": b[q].to_numpy()}).dropna()
        ra = fit(d, "x")["resid"]
        rb = fit(d, "y")["resid"].reindex(ra.index)
        rows.append({"quantity": q, "n_pairs": len(d), "rho_raw": stats.spearmanr(d.x, d.y)[0],
                     "rho_resid": stats.spearmanr(ra, rb)[0]})
    return pd.DataFrame(rows)


def fig(prim: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(TESTS), figsize=(3.0 * len(TESTS), 3.6), facecolor=SURFACE)
    col = {KINDS[0]: C_MERGED, KINDS[1]: C_SRC}
    for ax, t in zip(axes, TESTS):
        _style(ax)
        for i, k in enumerate(KINDS):
            r = prim[(prim.setting == k) & (prim.test == t)].iloc[0]
            ax.bar(i, r.stat, color=col[k], width=0.6)
            ax.text(i, r.stat, f"{r.stat:+.2f}\nHolm p={r.p_holm:.3f}", ha="center",
                    va="bottom" if r.stat >= 0 else "top", fontsize=7, color=INK)
        base = 0.25 if t == "MH3a" else 0.0
        ax.axhline(base, color=MUTED, linewidth=0.8, linestyle=":" if base else "-")
        ax.set_xticks(range(len(KINDS)), [k.replace("audio_", "") for k in KINDS], fontsize=8)
        ax.set_ylim(min(-0.3, ax.get_ylim()[0]), max(0.95, ax.get_ylim()[1]))
        ax.set_title(f"{t}: {LABEL[t]}", fontsize=7.5, color=INK, loc="left")
    axes[0].set_ylabel("statistic (one-sided, predicted > 0; MH3a vs 0.25)", fontsize=7.5, color=INK2)
    fig.suptitle("Mechanism hypotheses MH2–MH5 (researchvision.md 8.4): HuBERT (judged) vs WavLM "
                 "(exploratory), audio 7 groups, 30 reps", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["audio_wavlm"],
                    help="WavLM だけを計算し，E11 の値の再現を確かめる。何も書かない（検算用）")
    a = ap.parse_args()
    if a.only:
        prim, _ = run((a.only,))
        if a.only == "audio_wavlm":
            print(f"E11 との差の最大：{check_e11(prim):.2e}")
        print(prim[["setting", "test", "stat", "p_one_sided", "p_holm", "verdict"]].to_string(index=False))
        return 0

    prim, extra = run(KINDS)
    diff = check_e11(prim)
    if diff > 1e-9:
        raise SystemExit(f"WavLM の値が E11 と一致しない（差の最大 {diff:.2e}）。何も書かずに止まる")
    OUT.mkdir(parents=True, exist_ok=True)
    prim.to_csv(OUT / "primary.csv", index=False)
    pd.DataFrame([x["sec"] for x in extra if "sec" in x]).to_csv(OUT / "secondary.csv", index=False)
    pairs = pd.concat([x["pairs"] for x in extra if "pairs" in x], ignore_index=True)
    pairs.to_csv(OUT / "pairs.csv", index=False)
    pd.concat([x["curves"] for x in extra if "curves" in x], ignore_index=True).to_csv(OUT / "curves.csv", index=False)
    pd.concat([x["e13"] for x in extra if "e13" in x], ignore_index=True).to_csv(OUT / "e13.csv", index=False)
    encoders(pairs).to_csv(OUT / "encoders.csv", index=False)
    fig(prim, OUT / "fig_mh.png")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip())
    reps = {x["sec"]["kind"]: x["sec"]["n_reps"] for x in extra if "sec" in x}
    (OUT / "provenance.json").write_text(json.dumps(
        {"script": "exp/check_mh.py", "commit": commit, "dirty": dirty,
         "run": datetime.now().astimezone().isoformat(timespec="seconds"), "n_reps": reps,
         "e11_reproduction_max_abs_diff": diff}, ensure_ascii=False, indent=1))
    print(prim[prim.setting == KINDS[0]][["test", "stat", "n", "p_one_sided", "p_holm", "p_holm_opposite",
                                          "verdict"]].to_string(index=False))
    print(f"書いた: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
