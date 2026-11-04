"""dai-h1 の単位（T = IEMOCAP，MELD）と dai-h2 の単位（T = AFEW，MOSEI）の出発点の判定を，results/dai1/source と特徴から作り直す（読むだけ）。"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1, dai_data as DD, dai_solver as S
out = Path("/home/snics/MER-DA/results/dai1"); U = out / "pseudo/units"
tot = bad = 0
for T, c in [("IEMOCAP", "AFEW"), ("IEMOCAP", "MELD"), ("MELD", "MOSEI"), ("MELD", "IEMOCAP"), ("AFEW", "MELD"), ("MOSEI", "IEMOCAP")]:
    for f in (0, 3):
        srcs = [s for s in DD.others(T) if s != c]
        u = dai1.Unit(out, T, c, srcs, f, 0)
        z = np.load(U / f"{T}_{c}_f{f}_r0_p1.npz")
        for kind in DD.KINDS:
            W0, b0 = u.start[kind]
            p = S.logits(u.Xt[kind], W0, b0).argmax(1).cpu().numpy()
            sv = z[f"p1_{kind}_start_pred"]
            Xraw = u.cp.X[kind][u.test]
            Zs = sum(S.logits((Xraw - mu) / sd, Wc, bc) for (_, Wc, bc), (mu, sd) in zip(u.src[kind], u.src_stats[kind])) / len(u.src[kind])
            q = Zs.argmax(1).cpu().numpy()
            mu_ok = np.allclose(z[f"stat_{kind}_mu"], u.stats[kind][0], atol=1e-6)
            tot += 1; agree = (p == sv).mean(); agree2 = (q == z[f"p1_{kind}_start_srcstd_pred"]).mean()
            if agree < 1 or agree2 < 1 or not mu_ok:
                bad += 1; print("diff", T, c, f, kind, agree, agree2, mu_ok)
print("checked", tot, "with differences", bad)
