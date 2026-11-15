"""探索の解析（事前登録の外．2026-11-05．研究者の承認「良い」）：モダリティの信頼度を K 本のラベルの leave-one-out（LOO）で見積もると，
そのターゲットで強いモダリティの順位を当てられるか．当てられるなら，その信頼度で決めた指数の積の融合が AVT の生徒と上限にどれだけ近いか．

単位（T, f, r），K ∈ {5, 25}，モダリティ m ∈ {A, V, T} ごとに：
- LOO：K 本（4K 本）のラベル付き発話のうち 1 本 i を抜き，残り 4K − 1 本で K 本だけの識別器（none と同じ形．出発点 ＝ ソースの平均，
  引き戻し ＝ 実験 1 の固定の値，感情の釣り合いの重み）を学び，抜いた i を当てるかを見る．4K 個の問を `dai_solver.fit` で 1 束にして解く．
  loo_acc_m：当てた割合．loo_uar_m：抜いた発話の判定から作った 4 感情の再現率の平均．
- 本当の強さ：none[K]（K 本だけ）のテストの UAR（p2 の none_m_test）と，提案で直した単独の識別器のテストの UAR（p3 の判定）．
- 順位の一致：LOO の順位（loo_uar）と，直した単独の識別器のテストの UAR の順位（最も強いモダリティが一致するか．Spearman）．
- 積の融合：提案で直した A・V・T の確率（explore_fusion_grid.py と同じ作り方）に，次の指数で積を取ったテストの UAR．
  (a) rule_loo：e_m = max(loo_uar_m − 25, 0) / max_m(…)（でたらめの 25% を引いて最強を 1）．
  (b) rule_loo_prior：ソースの事前分布で縮めた版．事前は，3 つのソースの段 source の交差検証の UAR（`results/dai1/source/summary.csv` の
      その種類の値）の平均 prior_m．post_m = (n_loo · loo_uar_m + n0 · prior_m) / (n_loo + n0)，n0 = 20，n_loo = 4K．指数は (a) と同じ変換．
  (c) 参考：(1,1,1)，上限（explore_fusion_grid.py の oracle_e），AVT の生徒の提案，事前分布だけで決めた指数の積（rule_prior．n_loo = 0 の対照）．
- 塊ごとの LOO（gate-code の P1）：発話単位の LOO は同じ話者（塊）の別の発話が学習側に残るので，話者の情報が強い音声が水増しされうる．
  K 本の塊（話者．MOSEI はチャンネル）を 1 つずつ抜いて学び直し，抜いた塊の発話を当てる版を cl_uar_m として並べ，同じ規則（rule_cl）と順位の一致も出す．
- 同点（gate-code の P2）：K = 5 の LOO の UAR は 1/20 刻みで同点が出る．抜いた発話の対数損失 loo_nll_m（連続）も残し，同点のセルの数を数え，
  heavy_*_ok は同点を 1/(同点の数) の得点にする．
教師の積の指数は変えない（この探索は最後の融合だけ）．UAR は 4 感情の再現率の平均（30 本の規則は当てない．探索の記述）．
出力：<out>/loo_units.csv，loo_cells.csv．使い方：python DAI/exp/explore_reliability_loo.py <out> <T> [<T> ...]
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
import dai_solver as S  # noqa: E402

ROOT = HERE.parents[1]
KS = (5, 25)
N0 = 20.0


def uar(y, p):
    return 100.0 * np.mean([np.mean(p[y == k] == k) for k in range(4) if (y == k).any()])


def pick_idx(items, want):
    hits = [j for j, it in enumerate(items) if all(ST.norm(it.get(k)) == ST.norm(v) for k, v in want.items())]
    assert len(hits) == 1, (want, len(hits))
    return hits[0]


def prod(logP, e):
    return sum(float(ei) * logP[m] for ei, m in zip(e, "AVT")).argmax(1)


def heavy(scores):
    v = np.asarray(scores, float); top = np.flatnonzero(v >= v.max() - 1e-9)
    return "".join("AVT"[i] for i in top)                              # 同点なら複数のモダリティを並べる


def heavy_score(h, true):
    return (1.0 / len(h)) if true in h else 0.0


def exponents(scores):
    v = np.maximum(np.asarray(scores, float) - 25.0, 0.0)
    return tuple(v / v.max()) if v.max() > 0 else (1.0, 1.0, 1.0)


def source_prior(T):
    """3 つのソースの段 source の交差検証の UAR の平均（モダリティごと）．summary.csv の列名は台本が読む．"""
    sm = pd.read_csv(ROOT / "results" / "dai1" / "source" / "summary.csv")
    col = [c for c in sm.columns if "uar" in c.lower()][0]
    srcs = [c for c in DD.CORPORA if c != T]
    out = {}
    for m in "AVT":
        rows = sm[(sm["corpus"].isin(srcs)) & (sm["kind"] == m)]
        out[m] = float(rows[col].mean()) if len(rows) else np.nan
    return out


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(ROOT / "DAI" / "frozen" / "dai1_frozen.json"))
    U = ROOT / "results" / "dai1" / "eval" / "units"
    G = pd.read_csv(out / "grid_units.csv")                           # explore_fusion_grid.py の出力（4 コーパス × 50 単位）．上限は 50 単位の平均
    oracle = {}
    for (T, K), g in G.groupby(["T", "K"]):
        m = g.groupby(["eA", "eV", "eT"]).uar.mean(); oracle[(T, K)] = (m.idxmax(), float(m.max()))
    rows = []
    for T in sys.argv[2:]:
        F = fz[T]; srcs = [c for c in DD.CORPORA if c != T]; prior = source_prior(T)
        for f in range(5):
            for r in range(int(F["n_seeds_eval"])):
                base = U / f"{T}_ALL_f{f}_r{r}"
                if not Path(f"{base}_p3.npz").exists():
                    continue
                u = dai1.Unit(ROOT / "results" / "dai1", T, T, srcs, f, r)
                z2, z3 = np.load(f"{base}_p2.npz"), np.load(f"{base}_p3.npz")
                for K in KS:
                    ver = F["main"][str(K)]
                    kp = u.kpos[K]; yk = u.ypool[kp]; nK = len(kp)
                    cl = u.cp.df["cluster"].to_numpy()[u.pool][kp]; uc = pd.unique(cl)
                    rec = dict(T=T, fold=f, seed=r, K=K, n_clusters=len(uc))
                    logT, test_joint = {}, {}
                    for m in "AVT":
                        # LOO：4K 個の問．問 i は K 本から i を抜いた釣り合いの one-hot
                        Gs = torch.stack([S.balanced_onehot(u.ypool, np.delete(kp, i), u.n) for i in range(nK)])
                        cw, cb = F["tether"][f"{m}|{K}"]
                        W0, b0 = u.start[m]
                        res = S.fit(u.Xp[m], Gs, W0, b0, Cw=cw, Cb=cb)
                        Z = S.logits(u.Xp[m][kp], res["W"], res["b"])          # (4K 問, 4K 本, 4)
                        held = Z[torch.arange(nK), torch.arange(nK)].argmax(1).cpu().numpy()   # 問 i で発話 i の判定
                        rec[f"loo_acc_{m}"] = float((held == yk).mean()); rec[f"loo_uar_{m}"] = uar(yk, held)
                        rec[f"loo_stalled_{m}"] = int(res["stalled"]); rec[f"loo_gradmax_{m}"] = float(res["grad_max"])
                        Zh = Z[torch.arange(nK), torch.arange(nK)]
                        rec[f"loo_nll_{m}"] = float(-torch.log_softmax(Zh, 1)[torch.arange(nK), torch.as_tensor(yk, device=Zh.device)].mean())
                        if len(uc) >= 2:                                       # 塊ごとの LOO（P1）
                            Gc = torch.stack([S.balanced_onehot(u.ypool, kp[cl != c], u.n) for c in uc])
                            rc = S.fit(u.Xp[m], Gc, W0, b0, Cw=cw, Cb=cb)
                            Zc = S.logits(u.Xp[m][kp], rc["W"], rc["b"])
                            heldc = np.zeros(nK, int)
                            for pidx, c in enumerate(uc):
                                h = np.flatnonzero(cl == c)
                                heldc[h] = Zc[pidx][torch.as_tensor(h, device=Zc.device)].argmax(1).cpu().numpy()
                            rec[f"cl_uar_{m}"] = uar(yk, heldc); rec[f"cl_stalled_{m}"] = int(rc["stalled"])
                        else:
                            rec[f"cl_uar_{m}"] = np.nan; rec[f"cl_stalled_{m}"] = 0
                        rec[f"none_test_{m}"] = uar(u.ytest, z2[f"p2_K{K}_none_{m}_test"].astype(np.float32).argmax(1))
                        items = [json.loads(x) for x in z3[f"p3_{m}_K{K}_items"]]
                        j = pick_idx(items, dict(method="joint", teacher=ver, **{k: v for k, v in F["select"][f"{m}|{K}|joint|{ver}"].items() if k != "pseudo_uar"}))
                        W = torch.tensor(z3[f"p3_{m}_K{K}_W"][j], dtype=u.Xt[m].dtype, device=u.Xt[m].device)
                        b = torch.tensor(z3[f"p3_{m}_K{K}_b"][j], dtype=u.Xt[m].dtype, device=u.Xt[m].device)
                        logT[m] = np.log(np.clip(dai1.probs(u.Xt[m], W, b).double().cpu().numpy(), 1e-12, None))
                        test_joint[m] = uar(u.ytest, z3[f"p3_{m}_K{K}_pred"][j]); rec[f"joint_test_{m}"] = test_joint[m]
                        rec[f"prior_{m}"] = prior[m]
                    items = [json.loads(x) for x in z3[f"p3_AVT_K{K}_items"]]
                    j = pick_idx(items, dict(method="joint", teacher=ver, **{k: v for k, v in F["select"][f"AVT|{K}|joint|{ver}"].items() if k != "pseudo_uar"}))
                    rec["avt_joint"] = uar(u.ytest, z3[f"p3_AVT_K{K}_pred"][j])
                    loo = [rec[f"loo_uar_{m}"] for m in "AVT"]; clu = [rec[f"cl_uar_{m}"] for m in "AVT"]
                    nll = [-rec[f"loo_nll_{m}"] * 100 + 100 for m in "AVT"]       # 対数損失の小さいほど高い（同点を減らす連続の量）
                    post = [(nK * rec[f"loo_uar_{m}"] + N0 * prior[m]) / (nK + N0) for m in "AVT"]
                    pri = [prior[m] for m in "AVT"]; tj = [test_joint[m] for m in "AVT"]
                    e_loo, e_post, e_prior = exponents(loo), exponents(post), exponents(pri)
                    e_cl = exponents(clu) if not any(np.isnan(clu)) else (1.0, 1.0, 1.0)
                    htrue = heavy(tj)
                    rec.update(prod111=uar(u.ytest, prod(logT, (1, 1, 1))), rule_loo=uar(u.ytest, prod(logT, e_loo)),
                               rule_loo_prior=uar(u.ytest, prod(logT, e_post)), rule_prior=uar(u.ytest, prod(logT, e_prior)),
                               rule_cl=uar(u.ytest, prod(logT, e_cl)), oracle=oracle[(T, K)][1],
                               e_loo=json.dumps([round(x, 2) for x in e_loo]), e_cl=json.dumps([round(x, 2) for x in e_cl]),
                               heavy_loo=heavy(loo), heavy_cl=heavy(clu) if not any(np.isnan(clu)) else "", heavy_nll=heavy(nll),
                               heavy_post=heavy(post), heavy_prior=heavy(pri), heavy_true=htrue,
                               heavy_oracle=heavy(oracle[(T, K)][0]),
                               tie_loo=int(len(heavy(loo)) > 1), tie_cl=int(len(heavy(clu)) > 1) if not any(np.isnan(clu)) else 0,
                               loo_ok=heavy_score(heavy(loo), htrue), cl_ok=heavy_score(heavy(clu), htrue) if not any(np.isnan(clu)) else np.nan,
                               nll_ok=heavy_score(heavy(nll), htrue), post_ok=heavy_score(heavy(post), htrue), prior_ok=heavy_score(heavy(pri), htrue),
                               spearman=float(pd.Series(loo).corr(pd.Series(tj), method="spearman")),
                               spearman_cl=float(pd.Series(clu).corr(pd.Series(tj), method="spearman")))
                    rows.append(rec)
        print(T, "単位", sum(1 for x in rows if x["T"] == T) // len(KS), flush=True)
        dai1.Corpus.cache.clear(); torch.cuda.empty_cache()
    D = pd.DataFrame(rows); D.to_csv(out / "loo_units.csv", index=False)
    D["stalled_all"] = D[[c for c in D.columns if "stalled" in c]].sum(1)
    C = D.groupby(["T", "K"]).agg(n_units=("seed", "size"), avt_joint=("avt_joint", "mean"), prod111=("prod111", "mean"),
                                  rule_loo=("rule_loo", "mean"), rule_cl=("rule_cl", "mean"), rule_loo_prior=("rule_loo_prior", "mean"),
                                  rule_prior=("rule_prior", "mean"), oracle=("oracle", "first"),
                                  loo_ok=("loo_ok", "mean"), cl_ok=("cl_ok", "mean"), nll_ok=("nll_ok", "mean"), post_ok=("post_ok", "mean"),
                                  prior_ok=("prior_ok", "mean"), tie_loo=("tie_loo", "sum"), tie_cl=("tie_cl", "sum"),
                                  spearman=("spearman", "mean"), spearman_cl=("spearman_cl", "mean"), stalled=("stalled_all", "sum"),
                                  n_clusters=("n_clusters", "mean"),
                                  loo_uar_A=("loo_uar_A", "mean"), loo_uar_V=("loo_uar_V", "mean"), loo_uar_T=("loo_uar_T", "mean"),
                                  joint_test_A=("joint_test_A", "mean"), joint_test_V=("joint_test_V", "mean"), joint_test_T=("joint_test_T", "mean")).reset_index()
    C.to_csv(out / "loo_cells.csv", index=False)
    pd.set_option("display.width", 250)
    print(C.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
