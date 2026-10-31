"""E60 の下調べ：モダリティごとの出発点の票をまとめた見込みは，融合の出発点（と単一モダリティの出発点）をどれだけ上回るか．
AV の 6 ターゲットと AVT の 4 ターゲット，反復 0・1．候補のラベルは測定にだけ使う．
  上限 any：発話ごとに当たるモダリティを選べたときの正解率
  poe    ：モダリティの出発点の確率の積（等しい重み）
  poolC  ：E44 と同じ log-linear pool（仲間＝モダリティの出発点．表は仮のターゲットで学び，λ は仮のターゲットを 1 つずつ外して選ぶ）
  mlp    ：小さな MLP（12→16→4）のまとめ方（仮のターゲットで学ぶ）
  一致   ：2 つ（3 つ）のモダリティの票が一致した発話の割合と，そのときの正解率
純相補性 = P(基準が外れ ∧ 見込みが正解) − P(基準が当たり ∧ 見込みが外れ)．J = 平均 ⟨q − p_base, e_y − p_base⟩．
"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "exp"); sys.path.insert(0, "measure")
import committee as CM, compare_b1 as CB, logreg, map as M, meta5 as M5
from data import groups_for
from geom_e0 import settings
from teach_student39 import Student

BLOCKS = {"fuse-audio-video": [("audio_wavlm", 0, 1024), ("video", 1024, 1536)],
          "fuse-audio-video-text": [("audio_wavlm", 0, 1024), ("video", 1024, 1536), ("text_roberta", 1536, 2304)]}
dev = logreg.DEVICE
def sub_group(g, a, b): return M.Group(g.name, g.X[:, a:b], g.y, g.speaker, g.person)
def draft_p(data_m, S, g_name, Xc, gamma, rho_s, cache, C=0.01):
    src, lin, n_of = M5.pool_info(data_m, S, C, cache)
    st = Student(data_m[g_name], Xc, src, lin, n_of, S, C, gamma, rho_s, [1.0])
    return torch.softmax(st.L0, 1).cpu().numpy()
def stats(name, q, yq, pbase, out):
    a = q.argmax(1); b = pbase.argmax(1); r_q = a == yq; r_b = b == yq
    e = np.eye(4)[yq]
    out[f"uar_{name}"] = CB.uar(yq, a); out[f"net_{name}"] = float((r_q & ~r_b).mean() - (~r_q & r_b).mean())
    out[f"J_{name}"] = float(((q - pbase) * (e - pbase)).sum(1).mean()); out[f"dis_{name}"] = float((a != b).mean())
class MLP(torch.nn.Module):
    def __init__(s, d): super().__init__(); s.net = torch.nn.Sequential(torch.nn.Linear(d, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    def forward(s, x): return s.net(x)
def fit_mlp(F, y, w, wd=1e-3, iters=300):
    m = MLP(F.shape[1]).to(dev).double(); opt = torch.optim.LBFGS(m.parameters(), lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = (w * torch.nn.functional.cross_entropy(m(F), y, reduction="none")).sum() + wd * sum((p ** 2).sum() for p in m.parameters()); loss.backward(); return loss
    opt.step(closure); return m
rows = []
for fkind, blocks in BLOCKS.items():
    groups = groups_for(fkind); data_f = {g: M.load(g, fkind) for g in groups}
    data_m = {k: {g: sub_group(data_f[g], a, b) for g in groups} for k, a, b in blocks}
    mods = [k for k, _, _ in blocks]
    cache = {k: {} for k in mods + [fkind]}
    for T in groups:
        S = [s for s in groups if CM.corp(s) != CM.corp(T)]
        gs = {k: settings(k, T)[:2] for k in mods}; gf = settings(fkind, T)[:2]
        # 仮のターゲット c の上の，モダリティごとの出発点の log 確率（c のコーパスを除くソースで作る）
        pseudo = {}
        for c in S:
            pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
            if not pool_c: continue
            Xc_all = {k: data_m[k][c].X - data_m[k][c].X.mean(0) for k in mods}
            pseudo[c] = (np.stack([np.log(np.clip(draft_p(data_m[k], pool_c, c, Xc_all[k], *gs[k], cache[k]), 1e-12, None)) for k in mods], 1), np.asarray(data_f[c].y))
        def pool_fit(members, lam):
            Fp = torch.as_tensor(np.concatenate([pseudo[c][0] for c in members]), dtype=logreg.DTYPE, device=dev)
            yp = torch.as_tensor(np.concatenate([pseudo[c][1] for c in members]), dtype=torch.long, device=dev)
            w = torch.cat([torch.as_tensor(1.0 / (len(members) * 4 * np.bincount(pseudo[c][1], minlength=4)[pseudo[c][1]]), dtype=logreg.DTYPE, device=dev) for c in members])
            A = torch.ones(len(yp), len(mods), dtype=logreg.DTYPE, device=dev)
            return CM.Pool(mods).fit(Fp, A, yp, w, lam), (Fp, yp, w)
        sc = {}
        for lam in CM.LAMS:
            for c in pseudo:
                others = [o for o in pseudo if o != c]
                if not others: continue
                pool, _ = pool_fit(others, lam)
                Fc = torch.as_tensor(pseudo[c][0], dtype=logreg.DTYPE, device=dev)
                sc.setdefault(lam, []).append(CM.uar(pseudo[c][1], pool.predict(Fc, torch.ones(len(Fc), len(mods), dtype=logreg.DTYPE, device=dev))))
        lam = max(sc, key=lambda k: float(np.mean(sc[k])))
        poolC, (Fp, yp, w) = pool_fit(list(pseudo), lam)
        mlp = fit_mlp(Fp.reshape(len(Fp), -1), yp, w)
        ev = data_f[T]; ctx = CB.Ctx(ev, [0, 1], ["5"], center="ind")
        for r in [0, 1]:
            pool = np.flatnonzero(~ctx.te[r]); yq = ev.y[pool]
            P = {}
            for k in mods:
                Xc = M.rows(data_m[k][T].X, pool) - M.rows(data_m[k][T].X, pool).mean(0)
                P[k] = draft_p(data_m[k], S, T, Xc, *gs[k], cache[k])
            Xcf = M.rows(ev.X, pool) - ctx.ct[r]
            pF = draft_p(data_f, S, T, Xcf, *gf, cache[fkind])
            L = np.stack([np.log(np.clip(P[k], 1e-12, None)) for k in mods], 1)
            Lt = torch.as_tensor(L, dtype=logreg.DTYPE, device=dev)
            q_poe = CM.softmax_np(L.sum(1)); q_C = poolC.predict(Lt, torch.ones(len(Lt), len(mods), dtype=logreg.DTYPE, device=dev))
            with torch.no_grad(): q_mlp = torch.softmax(mlp(Lt.reshape(len(Lt), -1)), 1).cpu().numpy()
            row = {"world": fkind, "target": T, "rep": r, "lam": lam, "uar_fused": CB.uar(yq, pF.argmax(1)), "acc_fused": float((pF.argmax(1) == yq).mean())}
            right = {k: P[k].argmax(1) == yq for k in mods}
            for k in mods: row[f"uar_{k}"] = CB.uar(yq, P[k].argmax(1)); row[f"acc_{k}"] = float(right[k].mean())
            row["acc_any"] = float(np.any(np.stack(list(right.values())), 0).mean()); row["acc_any_or_fused"] = float((np.any(np.stack(list(right.values())), 0) | (pF.argmax(1) == yq)).mean())
            for name, q in (("poe", q_poe), ("poolC", q_C), ("mlp", q_mlp)):
                stats(name + "|fused", q, yq, pF, row); stats(name + "|A", q, yq, P["audio_wavlm"], row)
            for k in mods:
                if k != "audio_wavlm": stats(f"{k}|A", P[k], yq, P["audio_wavlm"], row)
            stats("fused|A", pF, yq, P["audio_wavlm"], row)
            am = {k: P[k].argmax(1) for k in mods}
            pairs = [("audio_wavlm", "video")] + ([("audio_wavlm", "text_roberta"), ("video", "text_roberta")] if "text_roberta" in mods else [])
            for a_, b_ in pairs:
                ag = am[a_] == am[b_]; row[f"agree_{a_[:1]}{b_[:1]}"] = float(ag.mean()); row[f"prec_{a_[:1]}{b_[:1]}"] = float((am[a_][ag] == yq[ag]).mean())
            if len(mods) == 3:
                ag3 = (am[mods[0]] == am[mods[1]]) & (am[mods[1]] == am[mods[2]]); row["agree_avt"] = float(ag3.mean()); row["prec_avt"] = float((am[mods[0]][ag3] == yq[ag3]).mean())
            rows.append(row)
        print(fkind, T, "済み", flush=True)
d = pd.DataFrame(rows); pd.set_option("display.width", 250)
d.to_csv("results/explore_relation/e60_mm_check.csv", index=False)
for w, g in d.groupby("world"):
    print("==", w); m = g.drop(columns=["target", "rep", "world"]).mean(numeric_only=True)
    print((m * 100).round(2).to_string())
    print(g.groupby("target")[["uar_fused", "uar_audio_wavlm", "uar_video", "acc_any", "uar_poe|fused", "uar_poolC|fused", "uar_mlp|fused", "net_poolC|fused", "net_mlp|fused"]].mean().mul(100).round(1).to_string())
