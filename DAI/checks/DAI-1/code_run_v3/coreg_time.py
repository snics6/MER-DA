"""共正則化だけの所要（本番の格子．引き戻しは全部 (0.01, 10) の仮の値．UAR は出さない）。引数：out T c Ks rhos"""
import sys, time
import dai1, dai_data as DD
from pathlib import Path
out, T, c = sys.argv[1], sys.argv[2], sys.argv[3]
Ks = [int(x) for x in sys.argv[4].split(",")]; rhos = [float(x) for x in sys.argv[5].split(",")]
srcs = [s for s in DD.others(T) if s != c]
u = dai1.Unit(Path(out), T, c, srcs, 0, 0)
teth = {(k, K): (0.01, 10.0) for k in DD.KINDS for K in DD.KS}
print("n =", u.n, "装置", dai1.S.DEV, flush=True)
for K in Ks:
    for rho in rhos:
        dai1.RHO_GRID = (rho,)
        t1 = time.time(); st, rows = {}, []
        dai1.coreg(u, K, teth, st, rows)
        cv = [x for x in rows if x["method"] == "_conv_coreg"][0]
        print(f"K={K} ρ={rho}: {time.time()-t1:.1f} 秒 反復 {cv['n_iter']} 勾配の最大 {cv['grad_max']:.1e}", flush=True)
