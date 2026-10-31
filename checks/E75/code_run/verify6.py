import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import optimize
ROOT = Path("/home/snics/MER-DA")
P = pd.read_csv("runB/e75_6_pairs.csv")
v = P.loc_loss_after_ce_fit - P.loc_loss_after_joint_model
print("CE で学んだ切片の後の位置の損 − 模型の最小（0 以上のはず）：最小", v.min(), "負の組", int((v < -1e-6).sum()), "／", len(P))
f = P.frac_recovered_joint_model
print("取り戻せた割合の範囲", f.min(), f.max(), "1 を超える組", int((f > 1 + 1e-9).sum()), "負の組", int((f < -1e-9).sum()))
print("模型の最小が負の組", int((P.loc_loss_after_joint_model < -1e-9).sum()))
# 収束の旗：台本の関数を同じ入力で呼び直し，L-BFGS-B の success と勾配の大きさを見る
sys.path[:0] = [str(ROOT / "exp"), str(ROOT / "measure"), str(ROOT / "features")]
import reanalysis_e75 as E
from explore_synergy import d_o
from data import load_any
bad, n = 0, 0
for kind in ("audio_wavlm", "video"):
    for s in ("MELD", "AFEW"):
        z = np.load(ROOT / f"results/shift_{kind}/logits/{s}.npz")
        for t in ("CREMA-D", "MOSEI", "IEMOCAP-impro"):
            L = z[f"t_{t}"].astype(np.float64); y = load_any(t, kind)[0].label_4.to_numpy()
            base = d_o(L, y); d = np.array([b[0] for b in base]); o = np.array([b[1] for b in base])
            sd = E.pair_sd(L, y, base)
            A = np.zeros((6, 3))
            for i, (a, b) in enumerate(E.PAIRS):
                if a > 0: A[i, a - 1] += 1 / sd[i]
                if b > 0: A[i, b - 1] -= 1 / sd[i]
            fobj = lambda c: float(np.mean(E.ba(d, 0) - E.ba(d, o - A @ c)))
            for s0 in (np.zeros(3), np.linalg.lstsq(A, o, rcond=None)[0]):
                r = optimize.minimize(fobj, s0, method="L-BFGS-B"); n += 1
                bad += int(not r.success)
            # 別の出発点（乱数 20 個）でもっと低い値が出るか
            resid, _ = E.joint_offset(d, o, sd)
            rng = np.random.default_rng(0)
            best = min(optimize.minimize(fobj, rng.normal(0, 2, 3), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000}).fun for _ in range(20))
            print(f"  {kind} {t}←{s}: 台本の最小 {resid*100:.5f} 点，乱数 20 出発の最小 {best*100:.5f} 点")
print("L-BFGS-B が success でない回", bad, "／", n)
