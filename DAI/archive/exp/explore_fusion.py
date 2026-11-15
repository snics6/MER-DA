"""探索の解析（事前登録の外．2026-11-05．研究者の承認「良い」）：最後の多モダリティの予測器を，つないだ識別器（AVT の生徒）から
「直した 3 つの単独の識別器の積」に替えると，テストの UAR が上がるかを，評価の回の保存した重みから測る（新しい学習はしない）．

単位（T, f, r），K ∈ {0, 5, 25} ごとに，テストで次の UAR を出す．
- avt_none：AVT の生徒の K 本だけ（p2 の none_AVT_test）．avt_joint：AVT の生徒の提案（主の版・選んだ設定．p3 の判定）．
- ret_prod：段 2 の学び直した A・V・T（K 本だけで学び直した単独の識別器）の確率の積（指数 (1,1,1)）．教師をテストに当てたもの．
- joint_prod：提案（主の版・選んだ設定）で直した A・V・T の単独の識別器の確率の積（指数 (1,1,1)）．保存した W・b をテストの標準化した特徴に当てる．
- joint_prod_w：同じ積に，実験 1 の固定のファイルの指数 w[K] を使ったもの．
- joint_prod_x_avt：joint_prod の確率と AVT の生徒の提案の確率の積（4 つの積）．
- 参考：各単独の生徒の提案の UAR（p3 の判定）．
UAR は 4 感情の再現率の平均（30 本の規則は当てない．探索の記述）．ラベルは dai1_stats.label_maps（manifest）から引く．
出力：<out>/fusion_units.csv，fusion_cells.csv．使い方：python DAI/exp/explore_fusion.py <out> <T> [<T> ...]
"""
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


def uar(y, p):
    return 100.0 * np.mean([np.mean(p[y == k] == k) for k in range(4) if (y == k).any()])


def pick_idx(items, want):
    hits = [j for j, it in enumerate(items) if all(ST.norm(it.get(k)) == ST.norm(v) for k, v in want.items())]
    assert len(hits) == 1, (want, len(hits))
    return hits[0]


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(ROOT / "DAI" / "frozen" / "dai1_frozen.json"))
    U = ROOT / "results" / "dai1" / "eval" / "units"
    rows = []
    for T in sys.argv[2:]:
        F = fz[T]; y_of, _ = ST.label_maps(T)
        srcs = [c for c in DD.CORPORA if c != T]
        for f in range(5):
            for r in range(int(F["n_seeds_eval"])):
                base = U / f"{T}_ALL_f{f}_r{r}"
                if not Path(f"{base}_p3.npz").exists():
                    continue
                u = dai1.Unit(ROOT / "results" / "dai1", T, T, srcs, f, r)
                z2, z3 = np.load(f"{base}_p2.npz"), np.load(f"{base}_p3.npz")
                ytest = y_of(np.load(f"{base}_p1.npz")["sample_test"].astype(str))
                assert (ytest == u.ytest).all()
                for K in KS:
                    ver = F["main"][str(K)]
                    P, pred = {}, {}
                    for s in ("A", "V", "T", "AVT"):
                        items = [json.loads(x) for x in z3[f"p3_{s}_K{K}_items"]]
                        j = pick_idx(items, dict(method="joint", teacher=ver, **{k: v for k, v in F["select"][f"{s}|{K}|joint|{ver}"].items() if k != "pseudo_uar"}))
                        W = torch.tensor(z3[f"p3_{s}_K{K}_W"][j], dtype=u.Xt[s].dtype, device=u.Xt[s].device)
                        b = torch.tensor(z3[f"p3_{s}_K{K}_b"][j], dtype=u.Xt[s].dtype, device=u.Xt[s].device)
                        P[s] = dai1.probs(u.Xt[s], W, b).double().cpu().numpy()
                        pred[s] = z3[f"p3_{s}_K{K}_pred"][j]
                        assert (P[s].argmax(1) == pred[s]).mean() > 0.99, (T, f, r, K, s, (P[s].argmax(1) == pred[s]).mean())
                    L1 = sum(np.log(np.clip(P[m], 1e-12, None)) for m in "AVT")
                    w = F["w"][str(K)]
                    Lw = sum(float(wi) * np.log(np.clip(P[m], 1e-12, None)) for wi, m in zip(w, "AVT"))
                    Lr = sum(np.log(np.clip(z2[f"p2_K{K}_ret_{m}_test"].astype(np.float64), 1e-12, None)) for m in "AVT")
                    Lx = L1 + np.log(np.clip(P["AVT"], 1e-12, None))
                    rows.append(dict(T=T, fold=f, seed=r, K=K,
                                     avt_none=uar(ytest, z2[f"p2_K{K}_none_AVT_test"].astype(np.float32).argmax(1)),
                                     avt_joint=uar(ytest, pred["AVT"]), ret_prod=uar(ytest, Lr.argmax(1)),
                                     joint_prod=uar(ytest, L1.argmax(1)), joint_prod_w=uar(ytest, Lw.argmax(1)),
                                     joint_prod_x_avt=uar(ytest, Lx.argmax(1)),
                                     A_joint=uar(ytest, pred["A"]), V_joint=uar(ytest, pred["V"]), T_joint=uar(ytest, pred["T"]),
                                     w=json.dumps(w)))
                dai1.Corpus.cache.pop(T, None) if False else None
        print(T, "単位", sum(1 for x in rows if x["T"] == T) // len(KS), flush=True)
    D = pd.DataFrame(rows); D.to_csv(out / "fusion_units.csv", index=False)
    C = D.groupby(["T", "K"]).agg(n_units=("seed", "size"), **{c: (c, "mean") for c in
                                   ("avt_none", "avt_joint", "ret_prod", "joint_prod", "joint_prod_w", "joint_prod_x_avt", "A_joint", "V_joint", "T_joint")}).reset_index()
    C.to_csv(out / "fusion_cells.csv", index=False)
    pd.set_option("display.width", 220)
    print(C.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
