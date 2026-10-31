"""【探索的】E35 の段 1：適応した後の成績の見込みを目的にして，全ソースの下書きを学ぶ（CPU．ロジスティック回帰の学習なし）。

規則は `docs/exploratory-analyses.md` E35（設計と予言 Q35-1・Q35-2 は計算の前に書いた）。
各ソースコーパス c を仮のターゲットとみなし，K 本で適応した後の UAR の見込み（E28-9 の d*(K) と E28-4 の共通の雑音の橋．
E32 の 4 節と同じ模型）を K ∈ {1, 5, 10, 25} とコーパスで平均した量を最大にする下書き W（4 × 次元，切片なし）を L-BFGS で学ぶ。
c での向きの良さは e_cj = d_cj(W)/D_cj，d_cj(W) = u_jᵀΔμ_cj/√(u_jᵀ Σ_c u_j)，D_cj = √(Δμ_cjᵀ Σ_c⁻¹ Δμ_cj)
（Σ_c は感情の平均を引いた残りの Ledoit–Wolf の共分散，u_j = w_k − w_l）。ターゲットの情報は学習に使わない。

評価はターゲットのテストの発話での感情の対ごとの d（E32 の 5 節と同じ測り方．テストのラベルを使う仕組みの確かめ）。
比べる相手（pooled，count，equal，value，ind，1 ソース）は E32 の 5 節と同じ点数と式で計算し直し，`e32d_targets.csv` と照合する。

    ./.venv/bin/python exp/explore_draft_value.py [--kinds audio_wavlm …] [--check]

出力：`results/explore_relation/e35_targets.csv`（種類 × ターゲット × 下書き），`e35_summary.csv`，`e35_tests.csv`，
`e35_provenance.json`，学んだ下書き `results/draft_value_<kind>/<ターゲット>.npz`（W と標準化の値．段 2 用）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import explore_compare as EC  # noqa: E402
import explore_source_combine as SC  # noqa: E402
import explore_source_value as SV  # noqa: E402
from data import load_any  # noqa: E402
from explore_synergy import PAIRS  # noqa: E402

torch.set_num_threads(8)                      # GPU の仕事と並行するので CPU の並列数を絞る（researchvision 6.3）
DT = torch.float64
KS_LOSS = (1, 5, 10, 25)
OURS = ["adapt", "mean_e", "adapt_count", "init"]
OTHERS = ["pooled", "count", "equal", "value", "single_mean", "single_max"]
_GU, _GW = np.polynomial.hermite_e.hermegauss(40)    # explore_uar_link.shared と同じ節点
_GW = _GW / _GW.sum()
GU, GW = torch.tensor(_GU, dtype=DT), torch.tensor(_GW, dtype=DT)
PA = torch.tensor([a for a, _ in PAIRS])
PB = torch.tensor([b for _, b in PAIRS])
CLASS_PAIRS = [[j for j, p in enumerate(PAIRS) if k in p] for k in range(4)]


# ---------------------------------------------------------------- 模型（torch．微分できる形）

def bridge_t(c6: torch.Tensor) -> torch.Tensor:
    """共通の雑音の橋（`explore_uar_link.shared`）を，対称な c（o = 0）について (..., 6) → (...) で計算する。"""
    acc = 0.0
    for k in range(4):
        p = torch.special.ndtr(np.sqrt(2) * c6[..., CLASS_PAIRS[k], None] + GU)    # (..., 3, 40)
        acc = acc + (p.prod(-2) * GW).sum(-1)
    return acc / 4


def e_of(W: torch.Tensor, P: dict) -> torch.Tensor:
    """(M, 6) の e_cj。負（向きが逆）は 0 に，数値の誤差で 1 を超えた分は 1 に切る。"""
    U = W[PA] - W[PB]                                                  # (6, d)
    num = torch.einsum("jd,mjd->mj", U, P["dm"])
    den = torch.sqrt(torch.einsum("jd,mde,je->mj", U, P["S"], U))
    return (num / den / P["D"]).clamp(0.0, 1.0)


def adapt_uar(e: torch.Tensor, P: dict) -> torch.Tensor:
    """(M,) の適応後の UAR の見込み（K ∈ KS_LOSS の平均）。d* は r を使わない形（e = 1 でも計算できる）。"""
    tot = 0.0
    for K in KS_LOSS:
        t = P["q"][:, None] / K
        one = 1 - e ** 2
        dstar = P["D"] / torch.sqrt(1 + t * one / (one + e ** 2 * t))
        tot = tot + bridge_t(dstar / (2 * np.sqrt(1 + 1 / (2 * K))))
    return tot / len(KS_LOSS)


def objective(W: torch.Tensor, P: dict, kind: str, w: torch.Tensor) -> torch.Tensor:
    e = e_of(W, P)
    if kind == "mean_e":
        return -(w[:, None] * e).sum() / 6
    return -(w * adapt_uar(e, P)).sum()


# ---------------------------------------------------------------- ソースの統計と学習

def source_stats(kind: str, srcs: list[str], qd: pd.Series) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    data = {s: load_any(s, kind) for s in srcs}
    Xall = np.concatenate([X for _, X in data.values()])
    mu, sd = Xall.mean(0), Xall.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    dm, S, D, W_lda, n_rows = [], [], [], [], {}
    for s in srcs:
        df, X = data[s]
        Z, y = (X - mu) / sd, df["label_4"].to_numpy()
        m = np.stack([Z[y == k].mean(0) for k in range(4)])
        Sc = LedoitWolf(assume_centered=True).fit(Z - m[y]).covariance_
        dmc = np.stack([m[a] - m[b] for a, b in PAIRS])
        Si = np.linalg.solve(Sc, dmc.T).T                              # Σ⁻¹Δμ（対ごとの最良の向き）
        D.append(np.sqrt(np.einsum("jd,jd->j", dmc, Si)))
        wl = np.linalg.solve(Sc, m.T).T                                 # LDA の重み Σ⁻¹μ_k
        ul = np.stack([wl[a] - wl[b] for a, b in PAIRS])
        W_lda.append(wl / np.mean(np.sqrt(np.einsum("jd,de,je->j", ul, Sc, ul))))
        dm.append(dmc)
        S.append(Sc)
        n_rows[s] = len(y)
    P = {"dm": torch.tensor(np.stack(dm), dtype=DT), "S": torch.tensor(np.stack(S), dtype=DT),
         "D": torch.tensor(np.stack(D), dtype=DT), "q": torch.tensor([float(qd[s]) for s in srcs], dtype=DT)}
    return P, mu, sd, {"W_init": np.mean(W_lda, axis=0), "n_rows": n_rows}


def learn(P: dict, W0: np.ndarray, kind: str, w: torch.Tensor) -> tuple[np.ndarray, dict]:
    W = torch.tensor(W0, dtype=DT, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=1, max_iter=500, tolerance_grad=1e-8, tolerance_change=1e-14,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        f = objective(W, P, kind, w)
        f.backward()
        return f

    f0 = float(objective(W, P, kind, w))
    opt.step(closure)
    n_iter = int(opt.state[opt._params[0]].get("n_iter", -1))
    Wd = W.detach().clone().requires_grad_(True)
    f1 = objective(Wd, P, kind, w)
    (g,) = torch.autograd.grad(f1, [Wd])
    return W.detach().numpy(), {"f0": f0, "f1": float(f1), "n_iter": n_iter, "gmax": float(g.abs().max())}


# ---------------------------------------------------------------- 比べる相手（E32 の 5 節と同じ）

def others(kind: str, t: str, zm, srcs: list[str], reps: list[int]) -> tuple[dict, dict, dict]:
    """1 ソースごとの 6 対の d（反復の平均），束ねるための尺度をそろえた点数，いまの下書きの d（E32 の 5 節と同じ）。"""
    zs = np.load(ROOT / "results" / f"merged_scores_{kind}" / f"{t}.npz")
    single = {s: [] for s in srcs}
    Ls_all, ys, pooled = [], [], []
    for r in reps:
        if not np.array_equal(zm[f"test_idx_r{r}"], zs[f"test_idx_r{r}"]):
            raise ValueError(f"テストの行が合わない: {kind} {t} r{r}")
        y = zm[f"y_test_r{r}"]
        Ls = {s: zs[f"s_{s}_r{r}"].astype(np.float64) for s in srcs}
        Ls = {s: L / SC.scale(L) for s, L in Ls.items()}
        for s in srcs:
            single[s].append(SC.d_pairs(Ls[s], y))
        Ls_all.append(Ls)
        ys.append(y)
        pooled.append(SC.d_pairs(zm[f"s_all_r{r}"].astype(np.float64), y))
    d_single = {s: np.mean(v, axis=0) for s, v in single.items()}
    return d_single, {"Ls_all": Ls_all, "ys": ys}, {"pooled": np.mean(pooled, axis=0)}


def combos(d_single: dict, aux: dict, own: np.ndarray, srcs: list[str], n_rows: dict) -> tuple[dict, np.ndarray]:
    e_s = {s: np.clip(d_single[s] / own, 0.0, 0.999) for s in srcs}
    eb = {s: float(e_s[s].mean()) for s in srcs}
    wts = {"count": {s: float(n_rows[s]) for s in srcs}, "equal": {s: 1.0 for s in srcs},
           "value": {s: eb[s] / (1 - eb[s] ** 2) for s in srcs}}
    out = {}
    for name, w in wts.items():
        tot = sum(w.values())
        dj = [SC.d_pairs(sum(w[s] / tot * Ls[s] for s in srcs), y) for Ls, y in zip(aux["Ls_all"], aux["ys"])]
        out[name] = np.mean(dj, axis=0)
    r_ind = sum(e ** 2 / (1 - e ** 2) for e in e_s.values())
    return out, np.clip(np.sqrt(r_ind / (1 + r_ind)), 0.0, 0.999)


# ---------------------------------------------------------------- 1 ターゲット

def run_target(kind: str, t: str, qd: pd.Series, e13: pd.DataFrame, cur: dict, out_dir: Path) -> list[dict]:
    t0 = time.perf_counter()
    zm = np.load(ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz")
    srcs = [str(s) for s in zm["src_names"]]
    reps = [r for r in EC.REPS if f"s_all_r{r}" in zm.files]
    own = e13[e13.target == t].groupby("pair").own_d.mean().reindex(SV.PAIR_NAMES).to_numpy()
    P, mu, sd, aux0 = source_stats(kind, srcs, qd)
    n_rows = aux0["n_rows"]
    w_eq = torch.full((len(srcs),), 1 / len(srcs), dtype=DT)
    w_ct = torch.tensor([n_rows[s] for s in srcs], dtype=DT)
    w_ct = w_ct / w_ct.sum()
    Ws, info = {"init": aux0["W_init"]}, {"init": {"f1": float(objective(torch.tensor(aux0["W_init"]), P, "adapt", w_eq)),
                                                    "n_iter": 0}}
    for name, kind_obj, w in (("adapt", "adapt", w_eq), ("mean_e", "mean_e", w_eq), ("adapt_count", "adapt", w_ct)):
        Ws[name], info[name] = learn(P, aux0["W_init"], kind_obj, w)
    # 学習の側の値：等しい重みの適応の見込み（4 つの K の平均の UAR）と e の平均
    train = {}
    for name, W in Ws.items():
        e = e_of(torch.tensor(W), P)
        train[name] = (float(adapt_uar(e, P).mean()), float(e.mean()))
    # ターゲットでの評価
    dfT, XT = load_any(t, kind)
    yT = dfT["label_4"].to_numpy()
    d_ours = {name: [] for name in Ws}
    for r in reps:
        idx, y = zm[f"test_idx_r{r}"], zm[f"y_test_r{r}"]
        if not np.array_equal(yT[idx], y):
            raise ValueError(f"テストのラベルが合わない: {kind} {t} r{r}")
        Z = (XT[idx] - mu) / sd
        for name, W in Ws.items():
            d_ours[name].append(SC.d_pairs(Z @ W.T, y))
    d_all = {name: np.mean(v, axis=0) for name, v in d_ours.items()}
    d_single, aux, pooled = others(kind, t, zm, srcs, reps)
    cb, e_ind = combos(d_single, aux, own, srcs, n_rows)
    d_all.update(pooled)
    d_all.update(cb)
    dbar_single = {s: float(v.mean()) for s, v in d_single.items()}
    best = max(dbar_single, key=dbar_single.get)
    d_all["single_max"] = d_single[best]
    d_all["single_mean"] = np.mean(list(d_single.values()), axis=0)
    q = float(qd[t])
    rows = []
    for name, d in list(d_all.items()) + [("ind", None)]:
        e = e_ind if d is None else np.clip(d / own, 0.0, 0.999)
        row = {"kind": kind, "target": t, "draft": name, "d_mean": np.nan if d is None else float(d.mean()),
               "e_mean": float(e.mean()), "clip_share": float((e >= 0.999).mean()), "n_sources": len(srcs), "q_dir": q,
               "train_adapt_uar": train.get(name, (np.nan, np.nan))[0], "train_e": train.get(name, (np.nan, np.nan))[1],
               "obj": info.get(name, {}).get("f1", np.nan), "n_iter": info.get(name, {}).get("n_iter", np.nan),
               "gmax": info.get(name, {}).get("gmax", np.nan),
               "best_single": best if name == "single_max" else ""}
        if d is not None:
            row.update({f"d_{p}": float(v) for p, v in zip(SV.PAIR_NAMES, d)})
        for K in SV.KS:
            n, fl = SV.n_model(own, e, q, int(K), pos=True)
            row[f"n_K{K}"] = n if fl == 0 else np.inf
            row[f"uar_K{K}"] = SC.curve_uar(cur[t], int(K) + n) if fl == 0 else np.nan
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{t}.npz", mu=mu, sd=sd, src_names=np.array(srcs), **{f"W_{k}": v for k, v in Ws.items()})
    print(f"  {kind} {t}：{time.perf_counter() - t0:.0f} 秒（反復 "
          + "，".join(f"{k} {info[k]['n_iter']}" for k in ("adapt", "mean_e", "adapt_count")) + "）", flush=True)
    return rows


# ---------------------------------------------------------------- まとめと判定

def check_e32d(D: pd.DataFrame) -> float:
    ref = pd.read_csv(SV.OUT / "e32d_targets.csv")
    m = D.merge(ref[["kind", "target", "draft", "e_mean"]], on=["kind", "target", "draft"], suffixes=("", "_ref"))
    if not len(m):
        raise SystemExit("e32d_targets.csv と照合できる行が無い")
    return float((m.e_mean - m.e_mean_ref).abs().max())


def summarize(D: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    srows, trows = [], []
    for kind, g in D.groupby("kind"):
        Wd = g.pivot(index="target", columns="draft", values="d_mean")
        We = g.pivot(index="target", columns="draft", values="e_mean")
        row = {"kind": kind, "targets": len(Wd)}
        for name in OURS + OTHERS:
            row[f"d_{name}"] = float(Wd[name].median())
            row[f"e_{name}"] = float(We[name].median())
            h = g[g.draft == name]
            row[f"clip_{name}"] = float(h.clip_share.mean())
            row[f"train_uar_{name}"] = float(h.train_adapt_uar.median())
        row["e_ind"] = float(We["ind"].median())
        for a, b in (("adapt", "equal"), ("adapt", "pooled"), ("adapt", "value"), ("mean_e", "equal"),
                     ("adapt_count", "adapt"), ("init", "equal"), ("adapt", "init")):
            diff = Wd[a] - Wd[b]
            row[f"dd_{a}_vs_{b}"] = float(diff.median())
            row[f"win_{a}_vs_{b}"] = int((diff > 0).sum())
        for K in ("5", "10"):
            for name in ("adapt", "equal", "init"):
                U = g.pivot(index="target", columns="draft", values=f"uar_K{K}")[[name, "pooled"]].dropna()
                row[f"duar{K}_{name}_vs_pooled"] = float((U[name] - U["pooled"]).median()) if len(U) else np.nan
                row[f"duar{K}_{name}_targets"] = len(U)
        srows.append(row)
        for test, a, b in (("Q35-1", "adapt", "equal"), ("Q35-2", "adapt", "pooled")):
            diff = Wd[a] - Wd[b]
            trows.append({"kind": kind, "test": test, "median_diff": float(diff.median()),
                          "wins": int((diff > 0).sum()), "targets": len(diff), "pass": bool(diff.median() > 0)})
    T = pd.DataFrame(trows)
    for test in ("Q35-1", "Q35-2"):
        h = T[(T.test == test) & T.kind.isin(["audio_wavlm", "audio_hubert"])]
        T.loc[T.test == test, "verdict"] = ("支持" if len(h) == 2 and h["pass"].all() else "支持されない")
    return pd.DataFrame(srows), T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=SV.MAIN)
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--check", action="store_true", help="橋の検算だけ行う")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for _ in range(20):                                       # 橋の検算：numpy の shared と一致すること
        c6 = rng.uniform(-0.5, 2.0, 6)
        a, b = SV.bridge(c6), float(bridge_t(torch.tensor(c6, dtype=DT)))
        if abs(a - b) > 1e-12:
            raise SystemExit(f"橋が合わない：{a} 対 {b}")
    print("橋の検算：20 通りで numpy の shared と一致", flush=True)
    if args.check:
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/explore_draft_value.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "exp", "measure")),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "KS_LOSS": list(KS_LOSS), "reps": list(EC.REPS)}
    rows = []
    for kind in args.kinds:
        e13 = SV.load_e13(kind)
        qd = pd.read_csv(SV.FIX / "e28t9_targets.csv").query("kind == @kind").set_index("target").q_dir
        targets = [t for t in sorted(qd.index) if (ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz").exists()]
        if args.targets:
            targets = [t for t in targets if t in args.targets]
        cur = SV.curves(kind, targets)
        for t in targets:
            rows += run_target(kind, t, qd, e13, cur, ROOT / "results" / f"draft_value_{kind}")
    D = pd.DataFrame(rows)
    gap = check_e32d(D)
    print(f"E32 の 5 節の e との最大の差：{gap:.2e}", flush=True)
    S, T = summarize(D)
    prov.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "e32d_max_abs_diff": gap})
    sfx = "" if not args.targets and args.kinds == SV.MAIN else "_partial"
    D.to_csv(SV.OUT / f"e35_targets{sfx}.csv", index=False)
    S.to_csv(SV.OUT / f"e35_summary{sfx}.csv", index=False)
    T.to_csv(SV.OUT / f"e35_tests{sfx}.csv", index=False)
    (SV.OUT / f"e35_provenance{sfx}.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 250)
    print(S.round(3).T.to_string())
    print(T.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
