"""関門 1 の統計の確かめ（DAI-21）：CREMA-D の区間の半幅の見込み．
実験 18 の「RAVDESS を除いた CREMA-D の単位」（分割 5 × 種 2）の保存の判定から，対の差の標準誤差の項だけを出す．
差の推定値（効果の大きさ）は出さない．GPU を使わない．"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np
import dai1_stats as ST

R18 = "/home/snics/MER-DA/results/dai18_probe"
SG = {5: (1.0, 16), 10: (4.0, 1), 25: (4.0, 1)}   # 設計 3.1 節の CREMA-D の主の点
rhos, As = (0.0625, 0.25, 1.0, 4.0), (1, 4, 16)
SGRID = [(r, a) for r in rhos for a in As]
rng = np.random.default_rng(0)
for K in (5, 10, 25):
    gi = SGRID.index(SG[K])
    pairs = {"学び直し 提案−積": ("stud", f"pred_K{K}_AV_M_F6_mixsel_g{gi}", f"pred_K{K}_AV_prod_g{gi}"),
             "推定時 提案(mix2,adj)−積": ("unit", f"pred_adj_K{K}_M_F6_mix2", f"pred_adj_K{K}_prod")}
    for nm, (kind, ka, kb) in pairs.items():
        A, B, dis = {}, {}, []
        for f in range(5):
            for r in (0, 1):
                zs = np.load(f"{R18}/students/RAVDESS_CREMA-D_f{f}_r{r}.npz")
                z = zs if kind == "stud" else np.load(f"{R18}/units/RAVDESS_CREMA-D_f{f}_r{r}.npz")
                y, cl = zs["y_test"].astype(int), zs["cl_test"]
                pa, pb = z[ka].astype(int), z[kb].astype(int)
                dis.append(float((pa != pb).mean()))
                A[(f, r)], B[(f, r)] = ST.counts(y, pa, cl), ST.counts(y, pb, cl)
        es = ST.effect_stats(A, B, 91.0, rng)
        se10 = np.sqrt(es["v_jk"] + es["v_seed"] * 2 / 10)
        print(f"K={K:2d} {nm}：判定が違う割合 {np.mean(dis):.3f}，塊の項の SE {np.sqrt(es['v_jk']):.3f}，種の項の SE（2 種）{np.sqrt(es['v_seed']):.3f}，"
              f"2 種の半幅 {1.96 * es['se']:.2f}，10 種の見込みの半幅 {1.96 * se10:.2f}（{es['method']}）")
