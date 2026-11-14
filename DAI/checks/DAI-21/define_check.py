"""関門 1（定義）の確かめ：読むだけ．GPU を使わない．
1. 実験 20 の add_extra と，実験 21 の combine_all が，同じ 3 つの分類器の入力（実験 16 の下調べの保存 1 単位）で同じ判定を出すか
   （check3 が照らさない版：avg，conf，tcal，temp，tempb，wf10/100/1000，M_F6h_mix*）．
2. 分類器が 2 つのときの式の性質（Tukey の当てはめの自由度，n0 = 0 で戻し先が効かないこと，相性を外した版と水準だけの重みの一致）．
使い方：PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python DAI/checks/DAI-21/define_check.py
"""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai16_probe as P16  # noqa: E402
import dai20_eval as P20  # noqa: E402

spec = importlib.util.spec_from_file_location("d21", "/home/snics/MER-DA/DAI/docs/tools_pending/dai21_eval5.py")
D21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D21)

ROOT = P20.ROOT
GRID = P16.GRID

# ---- 1. add_extra と combine_all の照合
z = np.load(ROOT / "results" / "dai16_probe" / "units" / "AFEW_IEMOCAP_f0_r0.npz", allow_pickle=False)
K = 5
P20.KS = (K,)
arr = {k: z[k] for k in z.files}
add_rows = []
P20.add_extra(arr, add_rows, {K: {}}, {K: (1, 1, 1)})
g = lambda pre: [z[f"{pre}_{m}_K{K}"].astype(np.float64) for m in "AVT"]  # noqa: E731
arr2, rows2 = {}, []
kpos, fold = z[f"kpos_K{K}"], z[f"kfold_K{K}"].astype(int)
yp, yt = z["y_pool"].astype(int), z["y_test"].astype(int)
D21.combine_all(arr2, rows2, K, g("p_test_ret"), g("p_pool_ret"), g("pcf"), g("pcf2"), g("ppj"), kpos, fold, yp[kpos], yt, yp,
                z[f"D_M_K_K{K}"], z[f"D_M_orc_K{K}"], z["SR_M_ext"], z["SR_M_home"], D21.tables_of(3))
vs = ["avg", "conf", "tcal", "temp", "tempb", "wf10", "wf100", "wf1000"] + [f"M_F6h_mix{i}" for i in range(len(GRID)) if GRID[i][1] == 1.0]
worst = 1.0
for v in vs:
    for w in ("raw", "adj", "pool"):
        a = float((arr[f"pred_{w}_K{K}_{v}"] == arr2[f"pred_{w}_K{K}_{v}"]).mean())
        worst = min(worst, a)
        if a < 1.0:
            print(f"  不一致 {v} {w}：{a:.4f}")
print(f"1. add_extra と combine_all（{len(vs)} 版 × 3 層）：一致の最小 {worst:.6f}")
for v in ("tcal", "temp", "tempb"):
    print(f"   重み {v}：add_extra {np.round(arr[f'W_{v}_K{K}'], 6)}  combine_all {np.round(arr2[f'W_{v}_K{K}'], 6)}")

# ---- 2. 分類器が 2 つのときの式の性質
rng = np.random.default_rng(0)
res = dict(n=0, tukey_resid_rank=[], noprof_eq_F2=0, prior_irrelevant=0, noint_eq_W=0)
for _ in range(2000):
    DK = rng.uniform(0, 2.5, size=(2, 4))
    Dext = rng.uniform(0.1, 2.0, size=(2, 4))
    # Tukey の当てはめの残りの自由度：加法の残り（行と列の平均を引いた表）の階数は 2 × 4 で 1（3 成分が 1 つの向き），λ の項がその 1 つの向きの中の b 方向を取る
    X = rng.normal(size=(2, 4))
    R = X - P16.tukey_fit(X)
    res["tukey_resid_rank"].append(int(np.linalg.matrix_rank(R, tol=1e-9)))
    # n0 = 0 では戻し先（eq・ext）が効かない
    a = P16.mix_rel(DK, Dext, 20, 0.0, 1.0, "eq")
    b = P16.mix_rel(DK, Dext, 20, 0.0, 1.0, "ext")
    res["prior_irrelevant"] += int(np.array_equal(a, b))
    # 相性を外した版（n0 = 0，β = 0）の重みと，水準だけの重み（M_F2_K）の重み
    W0 = P16.to_w(P16.mix_rel(DK, Dext, 20, 0.0, 0.0, "eq"))
    w2 = P16.to_w(DK).mean(1)
    res["noprof_eq_F2"] += int(np.allclose(W0, w2[:, None], rtol=0, atol=1e-12))
    # 2 × 4 の重みでは，Tukey の当てはめが元の重みと一致する（分類器 × 感情の成分が全部 λ の項に入る）割合
    W = P16.to_w(P16.mix_rel(DK, Dext, 20, 0.0, 1.0, "eq"))
    res["noint_eq_W"] += int(np.allclose(P16.noint(W), W, atol=1e-9))
    res["n"] += 1
rk = np.bincount(res["tukey_resid_rank"], minlength=3)
print(f"2. 2 × 4 の表 {res['n']} 組：Tukey の当てはめの残りの階数の分布 {rk.tolist()}（0,1,2）")
print(f"   n0 = 0 で戻し先が効かない：{res['prior_irrelevant']}/{res['n']}．相性を外した版の重み = 水準だけの重み：{res['noprof_eq_F2']}/{res['n']}．"
      f"noint が重みを変えない：{res['noint_eq_W']}/{res['n']}")

# 残りの大きさ：noint で消える分の割合（2 × 4 と 3 × 4）
for nm in (2, 3):
    fr = []
    for _ in range(2000):
        DK = rng.uniform(0, 2.5, size=(nm, 4))
        Dext = rng.uniform(0.1, 2.0, size=(nm, 4))
        W = P16.to_w(P16.mix_rel(DK, Dext, 20, 0.0, 1.0, "eq"))
        inter = W - (W.mean() + (W.mean(1, keepdims=True) - W.mean()) + (W.mean(0, keepdims=True) - W.mean()))
        lost = W - P16.noint(W)
        fr.append(float(np.abs(lost).sum() / max(np.abs(inter).sum(), 1e-12)))
    print(f"   分類器 {nm}：noint が消す分 ÷ 加法の残り（絶対値の和）の中央 {np.median(fr):.3f}，90% 点 {np.quantile(fr, 0.9):.3f}")

# ---- 3. 重みの表を探す版：2 つのときの表の値
W2 = D21.tables_of(2)
print(f"3. tables_of(2)：形 {W2.shape}，感情ごとの和の範囲 {W2.sum(1).min():.6f}〜{W2.sum(1).max():.6f}，"
      f"表 0 が全部 1：{bool(np.allclose(W2[0], 1))}，声の重みの範囲 {W2[1:, 0].min():.3f}〜{W2[1:, 0].max():.3f}")
