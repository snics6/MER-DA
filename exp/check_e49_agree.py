"""【探索的・結果を見た後】E49 の確かめ：E44 の見込み q は，E39 の下書きの予測と P4 の出発点の予測にどれだけ近く，どれだけ当たるか。

E49 で，生徒を E39 の学習器に替えると見込みで動かす教える選び方の上乗せが消えたので，q と下書きの予測の食い違いを測る（`docs/exploratory-analyses.md` E49 の結果）。
A と A(HuBERT) の 14 ターゲット，反復 0〜9，候補（テスト以外）の上の UAR と食い違う割合。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/check_e49_agree.py

出力：`results/explore_relation/e49_agree.csv`
"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "exp"); sys.path.insert(0, "measure")
import map as M, compare_b1 as CB, meta5 as M5, committee as CM
from data import groups_for
from teach_student39 import Student
rows = []
for kind in ["audio_wavlm", "audio_hubert"]:
    data = {g: M.load(g, kind) for g in groups_for(kind)}
    f, Cs, P, Y = CM.prepare(kind, data)
    sel = pd.read_csv("results/explore_relation/e40_targets.csv", dtype={"K": str}).drop_duplicates(["kind","target"]).set_index(["kind","target"])
    cache = {}
    for T in groups_for(kind):
        S = [s for s in data if CM.corp(s) != CM.corp(T)]
        e44 = pd.read_csv(f"results/committee_{kind}/{T}.csv")
        pool_C = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), float(e44[e44.variant == "C"].lam.iloc[0]))
        C_T = float(pd.read_csv(f"results/multisource_scores_{kind}/{T}.csv", dtype={"K": str}).C.iloc[0])
        src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
        s_ = sel.loc[(kind, T)]
        ev = data[T]; ctx = CB.Ctx(ev, list(range(10)), ["5"], center="ind")
        for r in range(10):
            pool = np.flatnonzero(~ctx.te[r]); yq = ev.y[pool]
            Xc = M.rows(ev.X, pool) - ctx.ct[r]
            st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, float(s_.gamma), float(s_.rho_s), [1.0])
            Ps = [torch.softmax(f[s].logits(Xc, Cs[s]), 1).cpu().numpy() for s in S]
            F = torch.as_tensor(np.stack([np.log(np.clip(p, 1e-12, None)) for p in Ps], 1), dtype=torch.float64, device=Xc.device)
            q = pool_C.predict(F, torch.ones(len(yq), len(S), dtype=torch.float64, device=Xc.device)).argmax(1)
            p39 = st.L0.argmax(1).cpu().numpy(); p4 = src_pc.logits(Xc, C_T).argmax(1).cpu().numpy()
            rows.append({"kind": kind, "target": T, "rep": r, "uar_q": CB.uar(yq, q), "uar_p39": CB.uar(yq, p39), "uar_p4": CB.uar(yq, p4),
                         "dis_q_p39": float((q != p39).mean()), "dis_q_p4": float((q != p4).mean())})
d = pd.DataFrame(rows)
d.to_csv("results/explore_relation/e49_agree.csv", index=False)
g = d.groupby(["kind", "target"]).mean(numeric_only=True).groupby("kind").mean()
print((g.drop(columns="rep") * 100).round(2).to_string())
