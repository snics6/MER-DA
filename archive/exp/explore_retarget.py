"""【探索的】E39 の下見（CPU．新しい学習なし）：ソースの向きを作るときの散らばりを，ターゲットのラベル無しの散らばりに置き換える。

E35 の追加（`explore_draft_shrink.py`）の下書きは，ソース s ごとに W_s = ((1 − ρ)Σ_s + ρτ_s I)⁻¹ m̃_s（Σ_s はソースの感情の中で
の散らばり，m̃_s はソースの全体の平均を引いた感情の平均）を等しい重みで足した。ここでは散らばりをターゲットのラベル無しの散らばり
Σ_T（テスト以外の発話の共分散．感情を分けない全体の散らばり）に置き換え，W_s = ((1 − γ)Σ_T + γτ_T I)⁻¹ m̃_s とする。
「感情の中心の差はコーパスをまたいで似ていて，違いは主に散らばり（録音条件などの方向）にある」なら，ターゲットに合った向きになる。

評価は E35 の追加と同じ（反復 0〜9 のテストの発話での 6 対の平均 d̄．テストのラベルを使う仕組みの確かめ）。γ の選び方も同じく，
ソースコーパス c を仮のターゲットにして（c の散らばりは c の全発話からラベル無しで作る）適応後の UAR の見込みが最大の γ を選ぶ。

    ./.venv/bin/python exp/explore_retarget.py

出力：`results/explore_relation/e39pre_curve.csv`，`e39pre_targets.csv`，`e39pre_summary.csv`
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import explore_compare as EC  # noqa: E402
import explore_source_combine as SC  # noqa: E402
import explore_source_value as SV  # noqa: E402
from data import GROUP_SPEC, load_any  # noqa: E402
from explore_draft_shrink import adapt_uar, draft_scores, shrunk_dirs  # noqa: E402

GAMMAS = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]


def centered_means(Z, y):
    m = np.stack([Z[y == k].mean(0) for k in range(4)])
    return m - Z.mean(0)


def eig(Z):
    Zc = Z - Z.mean(0)
    lam, V = np.linalg.eigh(Zc.T @ Zc / len(Z))
    return np.clip(lam, 0, None), V


def dirs_with(lamV, mt, g):
    lam, V = lamV
    return (V @ ((V.T @ mt.T) / ((1 - g) * lam + g * lam.mean())[:, None])).T


def run_target(kind, t, qd, own_of, rho_star):
    t0 = time.perf_counter()
    zm = np.load(ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz")
    srcs = [str(s) for s in zm["src_names"]]
    reps = [r for r in EC.REPS if f"s_all_r{r}" in zm.files]
    data = {s: load_any(s, kind) for s in srcs}
    Xall = np.concatenate([X for _, X in data.values()])
    mu, sd = Xall.mean(0), Xall.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    del Xall
    Zs = {s: (X - mu) / sd for s, (_, X) in data.items()}
    ys = {s: df["label_4"].to_numpy() for s, (df, _) in data.items()}
    mts = {s: centered_means(Zs[s], ys[s]) for s in srcs}
    own_dirs = {s: shrunk_dirs(Zs[s], ys[s])[rho_star] for s in srcs}       # E35 の追加の ρ* の下書き（照合用）
    dfT, XT = load_any(t, kind)
    yT = dfT["label_4"].to_numpy()
    ZT = (XT - mu) / sd
    d_t = {g: [] for g in GAMMAS}
    d_own = []
    for r in reps:
        idx, y = zm[f"test_idx_r{r}"], zm[f"y_test_r{r}"]
        te = np.zeros(len(yT), bool)
        te[idx] = True
        lv = eig(ZT[~te])                                                     # ターゲットのラベル無しの散らばり
        Zt = ZT[idx]
        d_own.append(SC.d_pairs(draft_scores(Zt, [own_dirs[s] for s in srcs]), y))
        for g in GAMMAS:
            d_t[g].append(SC.d_pairs(draft_scores(Zt, [dirs_with(lv, mts[s], g) for s in srcs]), y))
    loco = {g: [] for g in GAMMAS}
    for c in srcs:
        rest = [s for s in srcs if s != c and GROUP_SPEC[s][0] != GROUP_SPEC[c][0]]
        lv = eig(Zs[c])
        own = own_of[c]
        for g in GAMMAS:
            d = SC.d_pairs(draft_scores(Zs[c], [dirs_with(lv, mts[s], g) for s in rest]), ys[c])
            loco[g].append(adapt_uar(own, np.clip(d / own, 0.0, 0.999), float(qd[c])))
    rows = [{"kind": kind, "target": t, "gamma": g, "d_target": float(np.mean(d_t[g], axis=0).mean()),
             "loco_uar": float(np.mean(loco[g])), "d_own_rho_star": float(np.mean(d_own, axis=0).mean())} for g in GAMMAS]
    print(f"  {kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return rows


def main() -> int:
    e35 = pd.read_csv(SV.OUT / "e35_targets.csv")
    e35s = pd.read_csv(SV.OUT / "e35s_targets.csv").set_index(["kind", "target"])
    rows = []
    for kind in SV.MAIN:
        e13 = SV.load_e13(kind)
        own_of = {g: h.groupby("pair").own_d.mean().reindex(SV.PAIR_NAMES).to_numpy() for g, h in e13.groupby("target")}
        qd = pd.read_csv(SV.FIX / "e28t9_targets.csv").query("kind == @kind").set_index("target").q_dir
        for t in [t for t in sorted(qd.index) if (ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz").exists()]:
            rows += run_target(kind, t, qd, own_of, float(e35s.loc[(kind, t), "rho_star"]))
    C = pd.DataFrame(rows)
    T = []
    for (kind, t), g in C.groupby(["kind", "target"]):
        g = g.set_index("gamma")
        gs = g.loco_uar.idxmax()
        ref = e35[(e35.kind == kind) & (e35.target == t)].set_index("draft").d_mean
        T.append({"kind": kind, "target": t, "gamma_star": gs, "gamma_best": g.d_target.idxmax(),
                  "d_star": g.d_target[gs], "d_best": g.d_target.max(), "d_g1": g.d_target[1.0],
                  "d_own_rho_star": g.d_own_rho_star.iloc[0], "d_shrink_e35s": float(e35s.loc[(kind, t), "d_star"]),
                  "d_equal": float(ref["equal"]), "d_pooled": float(ref["pooled"])})
    T = pd.DataFrame(T)
    S = T.drop(columns="target").groupby("kind").median()
    for a in ("d_star", "d_best"):
        S[f"{a}-shrink"] = T.assign(x=T[a] - T.d_shrink_e35s).groupby("kind").x.median()
        S[f"{a}>shrink"] = T.assign(x=T[a] > T.d_shrink_e35s).groupby("kind").x.sum()
    C.to_csv(SV.OUT / "e39pre_curve.csv", index=False)
    T.to_csv(SV.OUT / "e39pre_targets.csv", index=False)
    S.to_csv(SV.OUT / "e39pre_summary.csv")
    pd.set_option("display.width", 250)
    print(C.groupby(["kind", "gamma"]).d_target.median().unstack(0).round(3).to_string())
    print(S.round(3).T.to_string())
    print("照合：E35 の追加の ρ* の下書きの d̄ との最大の差", float((T.d_own_rho_star - T.d_shrink_e35s).abs().max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
