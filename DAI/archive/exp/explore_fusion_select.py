"""探索の解析（事前登録の外．2026-11-05．研究者の承認「1」）：K 本のラベルの leave-one-out（LOO）で，「つないだ識別器（AVT の生徒）」と
「信頼度つきの積（直した A・V・T の積．指数は LOO の信頼度）」のどちらを使うかを単位ごとに選び，選んだ方のテストの UAR を測る（新しい学習は
LOO の K 本だけの識別器．テストのラベルは選択に使わない）．

単位（T, f, r），K ∈ {5, 25} ごとに：
1. explore_reliability_loo.py と同じ LOO（発話ごと loo，塊ごと cl）を A・V・T・AVT の 4 つの「K 本だけの識別器」で行い，抜いた発話の点数を残す．
   - 信頼度 r_m：塊ごとの LOO の UAR（cl_uar_m）．指数 e_m = max(r_m − 25, 0) / max_m(…)．
   - 積の LOO の見込み（cl_est_prod）：抜いた発話 i について，各モダリティの「i を含む塊を抜いた問」の対数確率を e で重みづけて足し，argmax を
     取った判定の UAR（4K 本）．AVT の LOO の見込み（cl_est_avt）：同じ形で AVT の識別器．指数 e と見込みが同じ塊ごとの LOO から作られるので，積の見込みは AVT の見込みより楽観的になり，選択は積に傾く（gate-code の記録）．
2. 選択（select_cl）：cl_prod_uar > cl_uar_AVT なら積，そうでなければ AVT．テストでは，積は提案で直した A・V・T の確率の積（指数 e），
   AVT は提案で直した AVT の生徒（explore_fusion_grid.py と同じ）を使う．選んだ方のテストの UAR を記録する．
   - 発話ごとの LOO で同じ選択をした版（select_loo）も出す．
3. 参考：avt_joint（AVT の生徒），rule_cl（常に積），oracle_select（単位ごとにテストで良い方を選んだ上限．cl の版の上限），上限の格子（oracle_grid．explore_fusion_grid.py）．
4. 保存：抜いた発話の対数確率（cl・loo × A・V・T・AVT，float32）と sample_id・塊・ラベルを <out>/held/<T>_f<f>_r<r>_K<K>.npz に残す（選び方の感度を学び直さずに出すため）．
見込みは「K 本だけの識別器」で作り，テストで使うのは「直した識別器」である（代理）．
UAR は 4 感情の再現率の平均（30 本の規則は当てない．探索の記述）．出力：<out>/select_units.csv，select_cells.csv．
使い方：python DAI/exp/explore_fusion_select.py <out> <T> [<T> ...]
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
import dai_solver as S  # noqa: E402

ROOT = HERE.parents[1]
KS = (5, 25)
MODS = ("A", "V", "T", "AVT")


def uar(y, p):
    return 100.0 * np.mean([np.mean(p[y == k] == k) for k in range(4) if (y == k).any()])


def pick_idx(items, want):
    hits = [j for j, it in enumerate(items) if all(ST.norm(it.get(k)) == ST.norm(v) for k, v in want.items())]
    assert len(hits) == 1, (want, len(hits))
    return hits[0]


def exponents(scores):
    v = np.maximum(np.asarray(scores, float) - 25.0, 0.0)
    return tuple(v / v.max()) if v.max() > 0 else (1.0, 1.0, 1.0)


def loo_logits(u, m, kp, W0, b0, cw, cb, groups):
    """groups：抜く集合の一覧（発話ごとなら [[i]]，塊ごとなら塊の発話の添字の一覧）．戻り：(4K, 4) の抜いた発話の対数確率と stalled の数．"""
    Gs = torch.stack([S.balanced_onehot(u.ypool, np.delete(kp, g), u.n) for g in groups])
    res = S.fit(u.Xp[m], Gs, W0, b0, Cw=cw, Cb=cb)
    Z = S.logits(u.Xp[m][kp], res["W"], res["b"])                       # (問, 4K, 4)
    out = torch.zeros(len(kp), 4, dtype=Z.dtype, device=Z.device)
    for p, g in enumerate(groups):
        gi = torch.as_tensor(g, device=Z.device)
        out[gi] = Z[p][gi]
    return torch.log_softmax(out, 1).double().cpu().numpy(), int(res["stalled"]), float(res["grad_max"])


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(ROOT / "DAI" / "frozen" / "dai1_frozen.json"))
    U = ROOT / "results" / "dai1" / "eval" / "units"
    (out / "held").mkdir(exist_ok=True)
    G = pd.read_csv(out / "grid_units.csv")
    oracle = {k: float(g.groupby(["eA", "eV", "eT"]).uar.mean().max()) for k, g in G.groupby(["T", "K"])}
    rows = []
    for T in sys.argv[2:]:
        F = fz[T]; srcs = [c for c in DD.CORPORA if c != T]
        for f in range(5):
            for r in range(int(F["n_seeds_eval"])):
                base = U / f"{T}_ALL_f{f}_r{r}"
                if not Path(f"{base}_p3.npz").exists():
                    continue
                u = None; torch.cuda.empty_cache()                        # 前の単位の特徴を先に解放する（gate-code の指摘）
                u = dai1.Unit(ROOT / "results" / "dai1", T, T, srcs, f, r)
                z3 = np.load(f"{base}_p3.npz")
                for K in KS:
                    ver = F["main"][str(K)]
                    kp = u.kpos[K]; yk = u.ypool[kp]; nK = len(kp)
                    cl = u.cp.df["cluster"].to_numpy()[u.pool][kp]; uc = pd.unique(cl)
                    g_loo = [[i] for i in range(nK)]
                    g_cl = [np.flatnonzero(cl == c) for c in uc] if len(uc) >= 2 else None
                    rec = dict(T=T, fold=f, seed=r, K=K, n_clusters=len(uc))
                    LP_loo, LP_cl, logT, pred_avt = {}, {}, {}, None
                    for m in MODS:
                        cw, cb = F["tether"][f"{m}|{K}"]; W0, b0 = u.start[m]
                        LP_loo[m], st1, gm1 = loo_logits(u, m, kp, W0, b0, cw, cb, g_loo)
                        rec[f"loo_uar_{m}"] = uar(yk, LP_loo[m].argmax(1)); rec[f"stalled_{m}"] = st1; rec[f"gradmax_{m}"] = gm1
                        if g_cl is not None:
                            LP_cl[m], st2, _ = loo_logits(u, m, kp, W0, b0, cw, cb, g_cl)
                            rec[f"cl_uar_{m}"] = uar(yk, LP_cl[m].argmax(1)); rec[f"stalled_{m}"] += st2
                        else:
                            LP_cl[m] = LP_loo[m]; rec[f"cl_uar_{m}"] = np.nan
                        items = [json.loads(x) for x in z3[f"p3_{m}_K{K}_items"]]
                        j = pick_idx(items, dict(method="joint", teacher=ver, **{k: v for k, v in F["select"][f"{m}|{K}|joint|{ver}"].items() if k != "pseudo_uar"}))
                        if m == "AVT":
                            pred_avt = z3[f"p3_{m}_K{K}_pred"][j]
                        else:
                            W = torch.tensor(z3[f"p3_{m}_K{K}_W"][j], dtype=u.Xt[m].dtype, device=u.Xt[m].device)
                            b = torch.tensor(z3[f"p3_{m}_K{K}_b"][j], dtype=u.Xt[m].dtype, device=u.Xt[m].device)
                            logT[m] = np.log(np.clip(dai1.probs(u.Xt[m], W, b).double().cpu().numpy(), 1e-12, None))
                    for tag, LP in (("cl", LP_cl), ("loo", LP_loo)):
                        rel = [rec[f"{tag}_uar_{m}"] for m in "AVT"]
                        if any(np.isnan(rel)):
                            rel = [rec[f"loo_uar_{m}"] for m in "AVT"]
                        e = exponents(rel)
                        prod_k = sum(float(ei) * LP[m] for ei, m in zip(e, "AVT")).argmax(1)
                        est_prod, est_avt = uar(yk, prod_k), uar(yk, LP["AVT"].argmax(1))
                        prod_t = uar(u.ytest, sum(float(ei) * logT[m] for ei, m in zip(e, "AVT")).argmax(1))
                        avt_t = uar(u.ytest, pred_avt)
                        choose_prod = est_prod > est_avt                      # 同点は AVT（既定の形）
                        rec.update({f"{tag}_est_prod": est_prod, f"{tag}_est_avt": est_avt, f"{tag}_prod_test": prod_t,
                                    f"{tag}_choose_prod": int(choose_prod), f"select_{tag}": prod_t if choose_prod else avt_t,
                                    f"e_{tag}": json.dumps([round(x, 2) for x in e])})
                    np.savez_compressed(out / "held" / f"{T}_f{f}_r{r}_K{K}.npz",
                                        sample_id=u.cp.df["sample_id"].to_numpy()[u.pool[kp]].astype(str), cluster=cl.astype(str), y=yk,
                                        **{f"cl_{m}": LP_cl[m].astype(np.float32) for m in MODS}, **{f"loo_{m}": LP_loo[m].astype(np.float32) for m in MODS})
                    rec.update(avt_joint=avt_t, oracle_select=max(avt_t, rec["cl_prod_test"]), oracle_grid=oracle[(T, K)])
                    rows.append(rec)
        print(T, "単位", sum(1 for x in rows if x["T"] == T) // len(KS), flush=True)
        dai1.Corpus.cache.clear(); torch.cuda.empty_cache()
    D = pd.DataFrame(rows)
    D["stalled_all"] = D[[c for c in D.columns if c.startswith("stalled_")]].sum(axis=1)
    D.to_csv(out / "select_units.csv", index=False)
    C = D.groupby(["T", "K"]).agg(n_units=("seed", "size"), avt_joint=("avt_joint", "mean"), rule_cl=("cl_prod_test", "mean"),
                                  select_cl=("select_cl", "mean"), select_loo=("select_loo", "mean"), oracle_select=("oracle_select", "mean"),
                                  oracle_grid=("oracle_grid", "first"), share_prod_cl=("cl_choose_prod", "mean"), share_prod_loo=("loo_choose_prod", "mean"),
                                  cl_est_prod=("cl_est_prod", "mean"), cl_est_avt=("cl_est_avt", "mean"), stalled=("stalled_all", "sum"),
                                  n_clusters=("n_clusters", "mean")).reset_index()
    C.to_csv(out / "select_cells.csv", index=False)
    pd.set_option("display.width", 250)
    print(C.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
