import sys, json
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd
import dai1, dai4 as D4, dai4_stats as D, dai_data as DD, dai1_stats as ST
CR = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run_v2")
# --- pseudo
d = pd.read_csv(CR / "out_p/pseudo/units/AFEW_MELD_f1_r0_p4.csv", low_memory=False)
print("pseudo 行:", d.method.value_counts().to_dict())
g = d[d.method == "_g1check"]; print("g1check n", len(g), "pred_agree min", g.pred_agree.min(), "W max", g.W_maxdiff.max(), "b max", g.b_maxdiff.max(), "commit", d.commit.iloc[0][:7], "dirty", d.dirty.iloc[0])
num = d.select_dtypes("number"); print("pseudo NaN(uar,grad):", int(d[d.method.isin(['cons','cons_self','cons_own','cons_conf','cons_rand'])][['uar','grad']].isna().sum().sum()))
# --- eval
e = pd.read_csv(CR / "out_e/eval/units/AFEW_ALL_f0_r0_p4.csv", low_memory=False)
print("eval 行:", e.method.value_counts().to_dict())
t = e[e.method == "cons_tether"]; print("cons_tether 行", len(t), " 収束", int(((t.grad <= 1e-5) & ~t.stalled.astype(bool)).sum()), "grad max", t.grad.max(), "NaN uar", int(t.uar.isna().sum()), "finite False", int((t.finite.astype(str)=="False").sum()))
print(t.groupby(["student", "K"]).size().unique(), "(s,K) 数", t.groupby(["student", "K"]).ngroups)
w = e[(e.method == "_weights") & (e.variant == "cons_tether_base")]; print("_weights cons_tether_base 行", len(w))
z4 = np.load(CR / "out_e/eval/units/AFEW_ALL_f0_r0_p4.npz")
print("p4t キー数", len([k for k in z4.files if k.startswith("p4t_")]), " grid 末尾", list(z4["grid"])[-2:], " 全 grid", len(z4["grid"]))
sel = json.load(open(CR / "fake_sel.json"))
fz1 = json.load(open(D4.FROZEN1)); T = "AFEW"
up = ST.UnitPreds(D4.IN_DIR / "eval/units/AFEW_ALL_f0_r0"); p4 = D4.P4(CR / "out_e/eval/units/AFEW_ALL_f0_r0")
y_of, cl_of = ST.label_maps("AFEW") if False else (None, None)
sp = pd.read_csv("/home/snics/MER-DA/DAI/splits/AFEW.csv", keep_default_na=False); ymap = dict(zip(sp.sample_id.astype(str), sp.label_4.astype(int)))
y = np.array([ymap[x] for x in up.sid])
bad_pair = 0; tcmatch = []; tjmatch = []; same = []
for s in DD.STUDENTS:
    for K in dai1.KS_ALL:
        it4 = p4.items(s, K, "p4t"); it3 = [json.loads(x) for x in up.p3[f"p3t_{s}_K{K}_items"]]
        assert len(it4) == 14, len(it4)
        rho, gam = sel[f"{s}|{K}"]["rho"], sel[f"{s}|{K}"]["gamma"]
        assert all(abs(x["rho"] - rho) < 1e-9 and abs(x["gamma"] - gam) < 1e-9 for x in it4), "p4t の点が選んだ点でない"
        # 順序：14 点が TETHER の順
        o4 = [(x["Cw"], x["Cb"]) for x in it4]; o3 = [(x["Cw"], x["Cb"]) for x in it3 if x.get("tag") == "prop_sel"]
        bad_pair += int(o4 != list(dai1.TETHER) or o3 != list(dai1.TETHER))
        # 既定の引き戻しの点が 14 点のどれか → cons の本体 (p4) と p4t のその点が一致するか，joint と p3t のその点が一致するか
        cw, cb = fz1[T]["tether"][f"{s}|{max(K,1)}"]
        j4 = o4.index((cw, cb)); j3 = o3.index((cw, cb))
        full3 = [j for j, x in enumerate(it3) if x.get("tag") == "prop_sel"]
        pc = p4.z[f"p4t_{s}_K{K}_pred"][j4]; pcons = p4.pred(s, K, rho, gam)
        jj = D.joint_sel(fz1[T], s, K)
        pj = up.p3item(s, K, method="joint", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"]); ptj = up.p3[f"p3t_{s}_K{K}_pred"][full3[j3]]
        tcmatch.append((pc == pcons).mean()); tjmatch.append((pj == ptj).mean())
        same.append(abs(ST.counts.__call__ if False else 0))
print("順序の不一致 (s,K):", bad_pair, "/ 20")
print("p4t の既定の点 vs p4 の cons: 判定の一致率 min", min(tcmatch), "平均", np.mean(tcmatch))
print("p3t の既定の点 vs p3 の joint: 判定の一致率 min", min(tjmatch), "平均", np.mean(tjmatch))
# collect でセルが作れるか（eval の経路）
from dai4_stats import collect
selc = {(s, K): (sel[f"{s}|{K}"]["rho"], sel[f"{s}|{K}"]["gamma"]) for s in DD.STUDENTS for K in dai1.KS_ALL}
yo, co = ST.label_maps("AFEW")
sc = {m: selc for m in D4.CTRL_PREFIX}
cells, mism, tot = collect(CR / "out_e", D4.IN_DIR, "eval", "AFEW", "AFEW", [(0, 0)], yo, co, selc, fz1[T], extras=True, check_rows=True, sel_ctrl=sc, sel_rho25=selc)
tc = [k for k in cells if k[2].startswith("tc|")]; tj = [k for k in cells if k[2].startswith("tj|")]
print("collect: mismatch", mism, "/", tot, " tc セル", len(tc), " tj セル", len(tj), "(期待 14×20=280 ずつ)")
bad = 0
for (s, K, n), v in cells.items():
    a = cells[(s, K, "cons")][(0, 0)]; b = v[(0, 0)]; bad += int(not ((a[0] == b[0]).all() and (a[2] == b[2]).all()))
print("塊と件数の不一致", bad)
# effect_stats が tc − tj で動くか
rng = np.random.default_rng(0)
k = ("AVT", 5)
for tcw, tcb in [(0.001, 1.0), (1.0, 10.0)]:
    r = ST.effect_stats(cells[(*k, f"tc|{tcw}|{tcb}")], cells[(*k, f"tj|{tcw}|{tcb}")], 1.0, rng)
    print("effect_stats ok", tcw, tcb, sorted(r)[:6])
