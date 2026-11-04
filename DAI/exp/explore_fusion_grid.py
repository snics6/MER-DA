"""探索の解析（事前登録の外．2026-11-05．研究者の承認「良い」）：直した 3 つの単独の識別器の積による融合で，モダリティの指数を
ターゲットに合わせて決めると，AVT の生徒をどれだけ上回れるかを，評価の回の保存した重みから測る（新しい学習はしない）．

単位（T, f, r），K ∈ {0, 5, 25} ごとに，提案（主の版・選んだ設定）で直した A・V・T の単独の識別器の確率をテストと候補に当て，
- 指数の格子 e ∈ {0, 0.5, 1, 2}^3（全部 0 を除く 63 点）の各点で，積（Σ_m e_m log P_m の softmax）のテストの UAR．
- 上限（oracle_grid）：コーパス・K ごとに，単位の平均の UAR が最大の格子の点（テストのラベルで選ぶので上限）．
- ラベル無しの規則（rule_agree）：候補（K 本を除く）で，モダリティ m の判定が残り 2 つの判定と一致する割合の平均 a_m を取り，
  e_m = a_m / max_m a_m（連続の指数．最も一致するモダリティを 1）．印の診断と同じ量で，追加の情報は要らない．
- 参考：(1,1,1)，実験 1 の固定のファイルの w[K]，AVT の生徒の提案．
UAR は 4 感情の再現率の平均（30 本の規則は当てない．探索の記述）．出力：<out>/grid_units.csv（単位 × 格子），grid_cells.csv．
使い方：python DAI/exp/explore_fusion_grid.py <out> <T> [<T> ...]
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
KS = (0, 5, 25)
GRID = [e for e in itertools.product((0.0, 0.5, 1.0, 2.0), repeat=3) if sum(e) > 0]


def uar(y, p):
    return 100.0 * np.mean([np.mean(p[y == k] == k) for k in range(4) if (y == k).any()])


def pick_idx(items, want):
    hits = [j for j, it in enumerate(items) if all(ST.norm(it.get(k)) == ST.norm(v) for k, v in want.items())]
    assert len(hits) == 1, (want, len(hits))
    return hits[0]


def prod(logP, e):
    L = sum(float(ei) * logP[m] for ei, m in zip(e, "AVT"))
    return L.argmax(1)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(ROOT / "DAI" / "frozen" / "dai1_frozen.json"))
    U = ROOT / "results" / "dai1" / "eval" / "units"
    grid_rows, unit_rows = [], []
    for T in sys.argv[2:]:
        F = fz[T]; y_of, _ = ST.label_maps(T)
        srcs = [c for c in DD.CORPORA if c != T]
        for f in range(5):
            for r in range(int(F["n_seeds_eval"])):
                base = U / f"{T}_ALL_f{f}_r{r}"
                if not Path(f"{base}_p3.npz").exists():
                    continue
                u = dai1.Unit(ROOT / "results" / "dai1", T, T, srcs, f, r)
                z1, z3 = np.load(f"{base}_p1.npz"), np.load(f"{base}_p3.npz")
                ytest = u.ytest
                pool_ids = z1["sample_pool"].astype(str)
                for K in KS:
                    ver = F["main"][str(K)]
                    nk = ~np.isin(pool_ids, z1[f"sample_K{K}"].astype(str)) if K else np.ones(len(pool_ids), bool)
                    logP, am, pred_avt = {}, {}, None
                    for s in ("A", "V", "T", "AVT"):
                        items = [json.loads(x) for x in z3[f"p3_{s}_K{K}_items"]]
                        j = pick_idx(items, dict(method="joint", teacher=ver, **{k: v for k, v in F["select"][f"{s}|{K}|joint|{ver}"].items() if k != "pseudo_uar"}))
                        if s == "AVT":
                            pred_avt = z3[f"p3_{s}_K{K}_pred"][j]; continue
                        W = torch.tensor(z3[f"p3_{s}_K{K}_W"][j], dtype=u.Xt[s].dtype, device=u.Xt[s].device)
                        b = torch.tensor(z3[f"p3_{s}_K{K}_b"][j], dtype=u.Xt[s].dtype, device=u.Xt[s].device)
                        logP[s] = np.log(np.clip(dai1.probs(u.Xt[s], W, b).double().cpu().numpy(), 1e-12, None))
                        am[s] = dai1.probs(u.Xp[s], W, b).double().cpu().numpy().argmax(1)[nk]
                    a = {m: np.mean([(am[m] == am[o]).mean() for o in "AVT" if o != m]) for m in "AVT"}
                    amax = max(a.values())
                    e_rule = tuple(a[m] / amax for m in "AVT")
                    w = tuple(float(x) for x in F["w"][str(K)])
                    for e in GRID:
                        grid_rows.append(dict(T=T, fold=f, seed=r, K=K, eA=e[0], eV=e[1], eT=e[2], uar=uar(ytest, prod(logP, e))))
                    unit_rows.append(dict(T=T, fold=f, seed=r, K=K, avt_joint=uar(ytest, pred_avt), prod111=uar(ytest, prod(logP, (1, 1, 1))),
                                          prod_w=uar(ytest, prod(logP, w)), prod_rule_agree=uar(ytest, prod(logP, e_rule)),
                                          aA=a["A"], aV=a["V"], aT=a["T"], eA_rule=e_rule[0], eV_rule=e_rule[1], eT_rule=e_rule[2]))
        print(T, "単位", sum(1 for x in unit_rows if x["T"] == T) // len(KS), flush=True)
    G = pd.DataFrame(grid_rows); Un = pd.DataFrame(unit_rows)
    G.to_csv(out / "grid_units.csv", index=False); Un.to_csv(out / "rule_units.csv", index=False)
    cells = []
    for (T, K), g in G.groupby(["T", "K"]):
        m = g.groupby(["eA", "eV", "eT"]).uar.mean()
        best = m.idxmax()
        uu = Un[(Un["T"] == T) & (Un.K == K)]
        cells.append(dict(T=T, K=K, n_units=int(uu.shape[0]), avt_joint=uu.avt_joint.mean(), prod111=uu.prod111.mean(), prod_w=uu.prod_w.mean(),
                          prod_rule_agree=uu.prod_rule_agree.mean(), oracle_grid=m.max(), oracle_e=json.dumps(best),
                          rule_e_mean=json.dumps([round(uu.eA_rule.mean(), 2), round(uu.eV_rule.mean(), 2), round(uu.eT_rule.mean(), 2)]),
                          agree_mean=json.dumps([round(uu.aA.mean(), 3), round(uu.aV.mean(), 3), round(uu.aT.mean(), 3)])))
    C = pd.DataFrame(cells); C.to_csv(out / "grid_cells.csv", index=False)
    pd.set_option("display.width", 250)
    print(C.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
