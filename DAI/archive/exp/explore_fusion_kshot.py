"""探索の解析（事前登録の外．2026-11-05．研究者の承認「初めて良い」）：積の融合の指数を K 本のラベル付き発話で選ぶと，
テストの UAR が上限（テストで選んだ指数）にどこまで近づくかを，評価の回の保存した重みから測る（新しい学習はしない）．

単位（T, f, r），K ∈ {5, 25} ごとに，提案で直した A・V・T の単独の識別器の確率を，K 本の発話（候補の中の K 本．学習に使った発話なので
選び方に甘さが入る）とテストに当てる．指数の格子 e ∈ {0, 0.5, 1, 2}^3（全部 0 を除く）の各点で K 本の UAR を出し，最大の点（同点は
(1,1,1) に近い点，次に格子の並びで先）を選んで，その点のテストの UAR を出す（kshot_sel）．比べる：AVT の生徒の提案（avt_joint），
(1,1,1)（prod111），固定のファイルの w[K]（prod_w），上限（oracle：コーパス・K ごとにテストの平均が最大の点．explore_fusion_grid.py と同じ）．
あわせて，K 本で選んだ指数が上限の指数と同じ「最も重いモダリティ」を持つ割合を出す．
出力：<out>/kshot_units.csv，kshot_cells.csv．使い方：python DAI/exp/explore_fusion_kshot.py <out> <T> [<T> ...]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai_data as DD  # noqa: E402

ROOT = HERE.parents[1]
KS = (5, 25)
GRID = [e for e in itertools.product((0.0, 0.5, 1.0, 2.0), repeat=3) if sum(e) > 0]
GRID.sort(key=lambda e: (sum(abs(x - 1.0) for x in e), GRID.index(e) if False else 0))   # (1,1,1) に近い順


def uar(y, p):
    return 100.0 * np.mean([np.mean(p[y == k] == k) for k in range(4) if (y == k).any()])


def pick_idx(items, want):
    hits = [j for j, it in enumerate(items) if all(ST.norm(it.get(k)) == ST.norm(v) for k, v in want.items())]
    assert len(hits) == 1, (want, len(hits))
    return hits[0]


def prod(logP, e):
    return sum(float(ei) * logP[m] for ei, m in zip(e, "AVT")).argmax(1)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(ROOT / "DAI" / "frozen" / "dai1_frozen.json"))
    U = ROOT / "results" / "dai1" / "eval" / "units"
    G = pd.read_csv(out / "grid_units.csv")
    oracle = {}
    for (T, K), g in G.groupby(["T", "K"]):
        m = g.groupby(["eA", "eV", "eT"]).uar.mean(); oracle[(T, K)] = (m.idxmax(), float(m.max()))
    rows = []
    for T in sys.argv[2:]:
        F = fz[T]; srcs = [c for c in DD.CORPORA if c != T]
        for f in range(5):
            for r in range(int(F["n_seeds_eval"])):
                base = U / f"{T}_ALL_f{f}_r{r}"
                if not Path(f"{base}_p3.npz").exists():
                    continue
                u = dai1.Unit(ROOT / "results" / "dai1", T, T, srcs, f, r)
                z3 = np.load(f"{base}_p3.npz")
                for K in KS:
                    ver = F["main"][str(K)]
                    kp = u.kpos[K]; yk = u.ypool[kp]
                    logT, logK, pred_avt = {}, {}, None
                    for s in ("A", "V", "T", "AVT"):
                        items = [json.loads(x) for x in z3[f"p3_{s}_K{K}_items"]]
                        j = pick_idx(items, dict(method="joint", teacher=ver, **{k: v for k, v in F["select"][f"{s}|{K}|joint|{ver}"].items() if k != "pseudo_uar"}))
                        if s == "AVT":
                            pred_avt = z3[f"p3_{s}_K{K}_pred"][j]; continue
                        W = torch.tensor(z3[f"p3_{s}_K{K}_W"][j], dtype=u.Xt[s].dtype, device=u.Xt[s].device)
                        b = torch.tensor(z3[f"p3_{s}_K{K}_b"][j], dtype=u.Xt[s].dtype, device=u.Xt[s].device)
                        logT[s] = np.log(np.clip(dai1.probs(u.Xt[s], W, b).double().cpu().numpy(), 1e-12, None))
                        logK[s] = np.log(np.clip(dai1.probs(u.Xp[s][kp], W, b).double().cpu().numpy(), 1e-12, None))
                    best, best_u = None, -1.0
                    for e in GRID:                                     # 同点は (1,1,1) に近い点が先（GRID の並び）
                        v = uar(yk, prod(logK, e))
                        if v > best_u + 1e-12:
                            best, best_u = e, v
                    oe, ou = oracle[(T, K)]
                    w = tuple(float(x) for x in F["w"][str(K)])
                    heavy = lambda e: "AVT"[int(np.argmax(e))] if max(e) > sorted(e)[-2] else "tie"
                    rows.append(dict(T=T, fold=f, seed=r, K=K, avt_joint=uar(u.ytest, pred_avt), prod111=uar(u.ytest, prod(logT, (1, 1, 1))),
                                     prod_w=uar(u.ytest, prod(logT, w)), kshot_sel=uar(u.ytest, prod(logT, best)), kshot_uar_on_K=best_u,
                                     eA=best[0], eV=best[1], eT=best[2], heavy_sel=heavy(best), heavy_oracle=heavy(oe), oracle=ou))
        print(T, "単位", sum(1 for x in rows if x["T"] == T) // len(KS), flush=True)
    D = pd.DataFrame(rows); D.to_csv(out / "kshot_units.csv", index=False)
    C = D.groupby(["T", "K"]).agg(n_units=("seed", "size"), avt_joint=("avt_joint", "mean"), prod111=("prod111", "mean"), prod_w=("prod_w", "mean"),
                                  kshot_sel=("kshot_sel", "mean"), oracle=("oracle", "first"),
                                  heavy_match=("heavy_sel", lambda s: float(np.mean(s.values == D.loc[s.index, "heavy_oracle"].values))),
                                  heavy_oracle=("heavy_oracle", "first")).reset_index()
    C.to_csv(out / "kshot_cells.csv", index=False)
    pd.set_option("display.width", 220)
    print(C.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
