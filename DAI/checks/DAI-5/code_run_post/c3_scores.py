"""計算の後の点検 3：評価の回の採点 U_sm が K 本のラベルだけから出ているか（2 単位．特徴から識別器を当て直す．dai5 の関数は使わない）。
dai1.Unit（分割・K 本・標準化・段 source の識別器を読む共通の部品）だけを使う。"""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/snics/MER-DA"); sys.path.insert(0, str(ROOT / "DAI/exp"))
import dai1, dai_data as DD
def uar(y, p): return 100 * np.mean([(p[y == c] == c).mean() for c in range(4) if (y == c).any()])
for T, f, r in (("IEMOCAP", 2, 3), ("AFEW", 4, 7)):
    u = dai1.Unit(ROOT / "results/dai1", T, T, DD.others(T), f, r)
    z = np.load(ROOT / f"results/dai5/eval/units/{T}_ALL_f{f}_r{r}_p5.npz")
    names = [f"{c}_{m}" for m in "AVT" for c, _, _ in u.src[m]]
    assert names == [str(x) for x in z["p5_src_names"]]
    pp = [(u.Xp[m] @ W.T + b).argmax(1).cpu().numpy() for m in "AVT" for _, W, b in u.src[m]]
    pt = np.stack([(u.Xt[m] @ W.T + b).argmax(1).cpu().numpy() for m in "AVT" for _, W, b in u.src[m]])
    print(T, f, r, "各識別器のテストの判定と保存の p5_src_test_pred の一致", bool((pt == z["p5_src_test_pred"]).all()))
    for K in (1, 5, 10, 25):
        kp = u.kpos[K]; nk = np.setdiff1d(np.arange(u.n), kp)
        mine = np.array([uar(u.ypool[kp], p[kp]) for p in pp])
        leak_pool = np.array([uar(u.ypool[nk], p[nk]) for p in pp])
        leak_test = np.array([uar(u.ytest, p) for p in pt])
        sv = z[f"p5_K{K}_scores"]
        print(f"  K={K}: K 本だけの UAR と保存の採点の差の最大 {np.abs(mine - sv).max():.1e}；K 本以外の候補の UAR との差の最大 {np.abs(leak_pool - sv).max():.1f}；テストの UAR との差の最大 {np.abs(leak_test - sv).max():.1f}")
