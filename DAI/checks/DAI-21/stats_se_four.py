"""関門 1 の統計の確かめ（DAI-21）：4 コーパスの学び直しの対の差（提案の疑似ラベル − 積の疑似ラベル，ρ = 1，a = 1）の区間の半幅の見込み．
実験 18 の T = RAVDESS の単位（ソースは CREMA-D を含む 4 つで，評価の回とは違う）の保存から標準誤差の項だけを出す．差の推定値は出さない．"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np
import dai1_stats as ST
import dai1

R18 = "/home/snics/MER-DA/results/dai18_probe"
SGRID = [(r, a) for r in dai1.RHO_GRID for a in dai1.A_GRID]
gi = SGRID.index((1.0, 1))
rng = np.random.default_rng(0)
for c in ("AFEW", "IEMOCAP", "MOSEI", "MELD"):
    ge = ST.g_eff_of(c)[0]
    for K in (5, 10, 25):
        A, B, dis = {}, {}, []
        for f in range(5):
            for r in (0, 1):
                z = np.load(f"{R18}/students/RAVDESS_{c}_f{f}_r{r}.npz")
                y, cl = z["y_test"].astype(int), z["cl_test"]
                pa, pb = z[f"pred_K{K}_AVT_M_F6_mixsel_g{gi}"].astype(int), z[f"pred_K{K}_AVT_prod_g{gi}"].astype(int)
                dis.append(float((pa != pb).mean()))
                A[(f, r)], B[(f, r)] = ST.counts(y, pa, cl), ST.counts(y, pb, cl)
        es = ST.effect_stats(A, B, ge, rng)
        q = 1.96 if es["method"].startswith("boot") else float(__import__("scipy.stats").stats.t.ppf(0.975, es["df"]))
        se10 = np.sqrt(es["v_jk"] + es["v_seed"] * 2 / 10)
        print(f"{c:8s} K={K:2d}：g_eff {ge:6.1f}，判定が違う割合 {np.mean(dis):.3f}，塊の項の SE {np.sqrt(es['v_jk']):.3f}，種の項の SE（2 種）{np.sqrt(es['v_seed']):.3f}，"
              f"2 種の半幅 {q * es['se']:.2f}，10 種の見込みの半幅 {q * se10:.2f}（{es['method']}，df {es['df']}）")
