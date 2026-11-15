"""実験 16 の下調べ（DAI-16）の，計算の後に出すと設計に書いた量（設計 4 節の R0 と 10 節の第 6 版への変更）．新しい学習はしない．保存した単位の出力だけを読む．

1. 足し算だけを引いた残りの相関：信頼度の表（分類器 × 感情）から，全体の平均・分類器の差・感情の差だけを引いた残り（第 4 版の量）について，
   ソースで測った表（home・ext・away）・K 本の表と，全ラベルの表の相関を単位ごとに出す．掛け算の項まで引いた残り（R0 の主の量）と並べる．
2. 足し算だけの対照：重みを「全体の平均 ＋ 分類器の差 ＋ 感情の差」の当てはめ（負は 0）に置き換えた版（M_F6add）の UAR（adj の層）を，
   保存した分類器の確率と信頼度の表から作り直す．元の版（M_F6）と，掛け算の項まで入れた対照（M_F6n）の保存の判定と並べる．
   作り直しの検算として，同じ手順で M_F6 と M_F6n を作り直し，保存の判定との一致の割合を出す．
3. 対照の重みで負の値を 0 に切った単位の数（掛け算の項まで入れた対照と，足し算だけの対照）．

出力：<out>/posthoc/additive_units.csv（単位 × K × 出どころ），additive_corr.csv（c × K × 出どころ），additive_levels.csv（c × K × 版の UAR），
additive_effects.csv（差と区間），additive_clip.csv（負を切った単位の数），posthoc_provenance.json

使い方：.venv/bin/python exp/dai16_posthoc.py [--out-dir results/dai16_probe] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai15  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

MODS = "AVT"
KS_RUN: tuple = P16.KS
SRC = ("home", "ext", "away", "K", "orc")


def add_fit(X: np.ndarray) -> np.ndarray:
    """全体の平均 ＋ 行の差 ＋ 列の差（足し算だけの当てはめ）．"""
    return X.mean(1, keepdims=True) + X.mean(0, keepdims=True) - X.mean()


def tables(z, K: int) -> dict:
    return {"home": z["SR_M_home"], "ext": z["SR_M_ext"], "away": z["SR_M_away"], "K": z[f"D_M_K_K{K}"], "orc": z[f"D_M_orc_K{K}"]}


def run(out: Path, limit: int | None) -> None:
    global KS_RUN
    KS_RUN = tuple(json.load(open(out / "provenance.json"))["KS"])            # 単位の段で流した K（追補の K = 10 の回では (10,)）
    files = sorted((out / "units").glob("*.npz"))
    if limit:
        files = files[:limit]
    mix = pd.read_csv(out / "summary" / "mixsel.csv") if (out / "summary" / "mixsel.csv").exists() else pd.DataFrame(columns=["c", "K", "version", "chosen"])
    choice = {(r.c, int(r.K)): r.chosen for r in mix[mix.version == "M_F6_mixsel"].itertuples()}
    geff = {c: ST.g_eff_of(c)[0] for c in DD.CORPORA}
    rows, cells, clip = [], {}, []
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        f, r = int(f[1:]), int(r[1:])
        z = np.load(p, allow_pickle=False)
        yt = z["y_test"].astype(int)
        grid = [g.split("|") for g in z["grid"]]
        for K in KS_RUN:
            D = tables(z, K)
            if (c, K) in choice:
                gi = int(choice[(c, K)].split("mix")[1])
                n0, beta, prior = float(grid[gi][0]), float(grid[gi][1]), grid[gi][2]
                D["mixsel"] = P16.mix_rel(D["K"], D["ext"], 4 * K, n0, beta, prior)
            for s in ("home", "ext", "away", "K"):
                rows.append(dict(T=T, c=c, fold=f, seed=r, K=K, source=s,
                                 corr_add=P16.corr(D[s] - add_fit(D[s]), D["orc"] - add_fit(D["orc"])),
                                 corr_tukey=P16.corr(P16.dcenter(D[s]), P16.dcenter(D["orc"])),
                                 size_add=float(np.abs(D[s] - add_fit(D[s])).mean()), size_tukey=float(np.abs(P16.dcenter(D[s])).mean()),
                                 size_add_orc=float(np.abs(D["orc"] - add_fit(D["orc"])).mean()), size_tukey_orc=float(np.abs(P16.dcenter(D["orc"])).mean())))
            LMp = np.stack([P16.centered(z[f"p_pool_ret_{m}_K{K}"].astype(np.float64)) for m in MODS])
            LMt = np.stack([P16.centered(z[f"p_test_ret_{m}_K{K}"].astype(np.float64)) for m in MODS])
            target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)
            for s in ("orc", "ext", "K", "mixsel"):
                if s not in D:
                    continue
                W = P16.to_w(D[s])
                tk, ad = P16.tukey_fit(W), add_fit(W)
                clip.append(dict(T=T, c=c, fold=f, seed=r, K=K, source=s, neg_tukey=bool((tk < 0).any()), neg_add=bool((ad < 0).any()),
                                 n_neg_tukey=int((tk < 0).sum()), n_neg_add=int((ad < 0).sum())))
                for v, Wv in ((f"M_F6_{s}", W), (f"M_F6n_{s}", P16.noint(W)), (f"M_F6add_{s}", np.maximum(0.0, ad))):
                    Sp, St = P16.combine("F6", LMp, Wv), P16.combine("F6", LMt, Wv)
                    dl, _ = P16.match_shift(Sp, target)
                    pred = (St + dl).argmax(1)
                    name = v if s != "mixsel" else v
                    saved_key = f"pred_adj_K{K}_{v}" if s != "mixsel" else f"pred_adj_K{K}_{v.replace('mixsel', choice[(c, K)].split('_')[-1])}"
                    agree = float((z[saved_key].astype(int) == pred).mean()) if saved_key in z.files else np.nan
                    rows.append(dict(T=T, c=c, fold=f, seed=r, K=K, source=s, version=name, uar_adj=100 * S.uar(yt, pred), agree_saved=agree))
                    P12._add(cells.setdefault((c, K, name), {}), (f, r), ST.counts(yt, pred, z["cl_test"]), f"{p.stem} {name}")
            P12._add(cells.setdefault((c, K, "prod"), {}), (f, r), ST.counts(yt, z[f"pred_adj_K{K}_prod"].astype(int), z["cl_test"]), f"{p.stem} prod")
    P12._finish(cells)
    pdir = out / "posthoc"
    pdir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(pdir / "additive_units.csv", index=False)
    cr = df[df.version.isna()] if "version" in df else df
    cr.groupby(["c", "K", "source"])[["corr_add", "corr_tukey", "size_add", "size_tukey", "size_add_orc", "size_tukey_orc"]].agg(["mean", "std", "count"]).pipe(
        lambda q: q.set_axis([f"{a}_{b}" for a, b in q.columns], axis=1)).reset_index().to_csv(pdir / "additive_corr.csv", index=False)

    def level(d: dict) -> float:
        return float(np.mean([np.mean([ST.uar_full(d[k]) for k in d if k[0] == f]) for f in sorted({k[0] for k in d})]))

    pd.DataFrame([dict(c=c, K=K, version=v, uar_adj=level(d), n_fr=len(d)) for (c, K, v), d in sorted(cells.items())]).to_csv(pdir / "additive_levels.csv", index=False)
    eff = []
    for c in DD.CORPORA:
        for K in KS_RUN:
            for s in ("orc", "ext", "K", "mixsel"):
                for a, b in ((f"M_F6_{s}", f"M_F6add_{s}"), (f"M_F6_{s}", f"M_F6n_{s}"), (f"M_F6n_{s}", f"M_F6add_{s}"), (f"M_F6add_{s}", "prod")):
                    A, B = cells.get((c, K, a)), cells.get((c, K, b))
                    if A is None or B is None or set(A) != set(B):
                        continue
                    es = ST.effect_stats(A, B, geff[c], P16.row_rng("posthoc", c, K, a, b))
                    eff.append(dict(c=c, K=K, a=a, b=b, where="adj", est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], method=es["method"], n_fr=es["n_units"],
                                    verdict=P16.verdict(es["lo"], es["hi"], es["effect"])))
    pd.DataFrame(eff).to_csv(pdir / "additive_effects.csv", index=False)
    cl = pd.DataFrame(clip)
    cl.groupby(["c", "K", "source"]).agg(n_units=("neg_tukey", "size"), units_neg_tukey=("neg_tukey", "sum"), units_neg_add=("neg_add", "sum"),
                                         cells_neg_tukey=("n_neg_tukey", "sum"), cells_neg_add=("n_neg_add", "sum")).reset_index().to_csv(pdir / "additive_clip.csv", index=False)
    ag = df[df.version.notna()].groupby("version").agree_saved.agg(["min", "mean", "count"]) if "version" in df else pd.DataFrame()
    commit, dirty = dai1.git_commit()
    json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), commit=commit, dirty=dirty, n_units=len(files), script_sha256=dai15.sha(Path(__file__).resolve()),
                   units_commit=sorted({str(x) for x in pd.read_csv(files[0].with_suffix(".csv")).commit.unique()}) if files else [],
                   agree_saved_min={k: float(v) for k, v in ag["min"].dropna().items()} if len(ag) else {}),
              open(pdir / "posthoc_provenance.json", "w"), ensure_ascii=False, indent=1)
    print(f"posthoc：単位 {len(files)}，行 {len(df)}，差の行 {len(eff)}．保存の判定との一致の最小 {float(ag['min'].min()) if len(ag) else float('nan'):.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(P16.OUT_PROD))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(Path(a.out_dir).resolve(), a.limit)
