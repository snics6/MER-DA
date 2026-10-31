"""E60 の下調べ 2：別モダリティの見込みを E59 の蒸留に入れたときの UAR（AV の 6 ターゲット，反復 0・1，K ∈ {0, 5}）。
  生徒 A（音声だけ．融合の特徴の音声ブロック）に，見込み q_A（同じモダリティ．E44 相当は省き，出発点自身 p_A を対照に），p_V（映像の出発点），
  p_AV（融合の出発点），poe（A・V の出発点の積），agree（A と V が一致した発話だけ p_AV の確率，不一致は重み 0.2）を蒸留する．
  生徒 AV（融合）に poe と p_A・p_V を蒸留する（対照：p_AV 自身）．
"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "exp"); sys.path.insert(0, "measure")
import committee as CM, compare_b1 as CB, logreg, map as M, meta5 as M5
from data import groups_for
from geom_e0 import settings
from teach_student39 import Student
from distill_belief import solve_all, uar_of
dev = logreg.DEVICE
fkind = "fuse-audio-video"; blocks = [("audio_wavlm", 0, 1024), ("video", 1024, 1536)]
groups = groups_for(fkind); data_f = {g: M.load(g, fkind) for g in groups}
def sub_group(g, a, b): return M.Group(g.name, g.X[:, a:b], g.y, g.speaker, g.person)
data_m = {k: {g: sub_group(data_f[g], a, b) for g in groups} for k, a, b in blocks}
cache = {k: {} for k in ["audio_wavlm", "video", fkind]}
MEFFS = (100, 500, 2000)
rows = []
for T in groups:
    S = [s for s in groups if CM.corp(s) != CM.corp(T)]
    st_set = {k: settings(k, T) for k in ["audio_wavlm", "video", fkind]}
    ev = data_f[T]; ctx = CB.Ctx(ev, [0, 1], ["1", "5"], center="ind")
    info = {k: M5.pool_info(data_m[k] if k != fkind else data_f, S, 0.01, cache[k]) for k in st_set}
    for r in [0, 1]:
        pool = np.flatnonzero(~ctx.te[r]); yq = ev.y[pool]; te = np.flatnonzero(ctx.te[r]); yte = ev.y[te]
        pos_of = {int(i): j for j, i in enumerate(pool)}
        label_sets = {"0": np.array([], int), "5": np.array([pos_of[int(i)] for i in ctx.idx[(r, "5")]])}
        stu, Z, Zte, P = {}, {}, {}, {}
        for k in st_set:
            dat = data_m[k][T] if k != fkind else ev
            Xc = M.rows(dat.X, pool) - M.rows(dat.X, pool).mean(0)
            gamma, rho_s, rho_m, cb, cp, _ = st_set[k]
            src, lin, n_of = info[k]
            st = Student(dat, Xc, src, lin, n_of, S, 0.01, gamma, rho_s, [rho_m]); Am = st.mats[rho_m][0]
            stu[k] = (st, rho_m, cb, cp); Z[k] = st.Xstd @ Am
            Zte[k] = ((M.rows(dat.X, te) - M.rows(dat.X, pool).mean(0) - st.src.mu) / st.src.sd) @ Am
            P[k] = torch.softmax(st.L0, 1).cpu().numpy()
        pA, pV, pF = P["audio_wavlm"], P["video"], P[fkind]
        poe = CM.softmax_np(np.log(np.clip(pA, 1e-12, None)) + np.log(np.clip(pV, 1e-12, None)))
        ag = (pA.argmax(1) == pV.argmax(1)); agree = pF * np.where(ag, 1.0, 0.2)[:, None]
        for sk, beliefs in (("audio_wavlm", {"p": pA, "pV": pV, "pAV": pF, "poe": poe, "agree": agree}),
                            (fkind, {"p": pF, "pA": pA, "pV": pV, "poe": poe})):
            st, rho_m, cb, cp = stu[sk]
            combos = [("p", 0)] + [(b, m) for b in beliefs for m in MEFFS]
            sol = solve_all(Z[sk], yq, beliefs, label_sets, combos, st, rho_m, 0.01, cp, cb)
            row = {"target": T, "rep": r, "student": sk, "uar_draft": uar_of(st.W_std @ st.mats[rho_m][1], st.b_std, Zte[sk], yte)}
            for (b, m, K), (W, bb) in sol.items():
                row[f"{'none' if m == 0 else b}{m}_K{K}"] = uar_of(W, bb, Zte[sk], yte)
            rows.append(row)
    print(T, "済み", flush=True)
d = pd.DataFrame(rows); pd.set_option("display.width", 250)
d.to_csv("results/explore_relation/e60_mm_distill.csv", index=False)
for sk, g in d.groupby("student"):
    m = g.drop(columns=["target", "rep", "student"]).mean() * 100
    base = {"0": m["uar_draft"], "5": m["none0_K5"]}                           # K=0 で蒸留しない問題は出発点そのもの
    print("== 生徒", sk, " 出発点", round(m["uar_draft"], 2), " E39 K=0/5:", round(base["0"], 2), round(base["5"], 2))
    out = {}
    for c in m.index:
        if c.startswith("none") or c == "uar_draft": continue
        b, K = c.split("_K"); out.setdefault(b, {})[K] = round(m[c] - base[K], 2)
    print(pd.DataFrame(out).T.to_string())
