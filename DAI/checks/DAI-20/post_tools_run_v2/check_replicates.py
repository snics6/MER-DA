"""乱数の鍵を共有したとき，塊の重みを引き直すコーパス（MOSEI）で，引き直しの 1 回ごとに「全体 ＝ 切片の分 ＋ 向きの分」が成り立つかを確かめる．
作り物の評価の回の単位（fake_eval/units）の MOSEI を使い，dai1_stats.effect_stats の引き直しの部分（173〜184 行）と同じ計算で，引き直しの値の並びを返す．"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1_stats as ST  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai20_eval as E  # noqa: E402

U = Path("/home/snics/MER-DA/DAI/checks/DAI-20/post_tools_run_v2/fake_eval/units")
fz = E.load_frozen()
c = "MOSEI"
out = []
for K, sn in ((5, "all"), (25, "2-9")):
    i = fz[c][K][0]
    keys = {"v@adj": f"pred_adj_K{K}_M_F6_mix{i}", "v@ob": f"pred_ob_K{K}_M_F6_mix{i}", "p@adj": f"pred_adj_K{K}_prod", "p@ob": f"pred_ob_K{K}_prod"}
    cells = {}
    for p in sorted(U.glob(f"{c}_ALL_*.npz")):
        _, _, f, r = p.stem.split("_")
        f, r = int(f[1:]), int(r[1:])
        if E.SEED_SETS[sn] is not None and r not in E.SEED_SETS[sn]:
            continue
        z = np.load(p, allow_pickle=False)
        for nm, kk in keys.items():
            P12._add(cells.setdefault(nm, {}), (f, r), ST.counts(z["y_test"].astype(int), z[kk].astype(int), z["cl_test"]), p.stem)
    P12._finish(cells)

    def comb(names):
        cs = [cells[n] for n in names]
        return {k: (cs[0][k][0], sum(x[k][1] for x in cs), cs[0][k][2]) for k in cs[0]}

    def boots(A, B, rng):
        ks = sorted(A)
        folds = sorted({f for f, _ in ks})
        bs = np.zeros(ST.N_BOOT); nv = np.zeros(ST.N_BOOT)
        for f in folds:
            rs = [r for ff, r in ks if ff == f]
            ncl = len(A[(f, rs[0])][0])
            Wb = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl), size=ST.N_BOOT).astype(float)
            ua = np.stack([ST.uar_rows(Wb @ A[(f, r)][1], Wb @ A[(f, r)][2]) for r in rs])
            ub = np.stack([ST.uar_rows(Wb @ B[(f, r)][1], Wb @ B[(f, r)][2]) for r in rs])
            dm = np.nanmean(ua - ub, axis=0)
            ok = np.isfinite(dm); bs[ok] += dm[ok]; nv[ok] += 1
        return np.where(nv > 0, bs / np.maximum(nv, 1), np.nan)

    key = ("dai20", "A", "prop", "prod", c, K, sn)                           # 台本の eff の鍵
    bT = boots(comb(["v@adj"]), comb(["p@adj"]), P16.row_rng(*key))
    bI = boots(comb(["v@adj", "p@ob"]), comb(["v@ob", "p@adj"]), P16.row_rng(*key))
    bD = boots(comb(["v@ob"]), comb(["p@ob"]), P16.row_rng(*key))
    bD_other = boots(comb(["v@ob"]), comb(["p@ob"]), P16.row_rng("dai20post", "D", c, K, "prop"))   # 前の版の鍵（共有しない）
    out.append((K, sn, float(np.nanmax(np.abs(bT - bI - bD))), float(np.nanmax(np.abs(bT - bI - bD_other))), int(np.isnan(bT).sum())))
for K, sn, d_shared, d_other, nnan in out:
    print(f"MOSEI K={K} 種 {sn}：引き直し 2000 回の「全体 − 切片 − 向き」の最大のずれ：鍵を共有 {d_shared:.2e}，鍵が別 {d_other:.2e}（NaN の回 {nnan}）")
