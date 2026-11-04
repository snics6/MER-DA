"""gate-code（DAI-1 code_v5）：MELD をテストにした区間を少数のセルで作り直す（verify_indep2 の関数を使う．全セルは遅いため）。
話者の jackknife（水準と族 1 型の差），顔が競らない発話，対話を塊にした jackknife（自由度 7 と対話の自由度の両方）。"""
import sys
sys.argv = [sys.argv[0], sys.argv[1], "MOSEI", "MELD"]
sys.path.insert(0, "/home/snics/MER-DA/DAI/checks/DAI-1/code_run_desc")
import numpy as np, pandas as pd
from scipy import stats as st
import verify_indep2 as V

c = "MELD"
lab = pd.read_csv(V.ROOT / f"DAI/splits/{c}.csv", keep_default_na=False, dtype={"cluster": str, "dialogue": str})
units = [V.U(c, f, r, lab) for f in range(5) for r in (0, 1)]
ge = V.g_eff(c)
L = pd.read_csv(V.DEST / "levels.csv"); L = L[L.c == c]
S = pd.read_csv(V.DEST / "sensitivity.csv"); S = S[S.c == c]
sub = pd.read_csv(V.ROOT / "DAI/frozen/subsets.csv", dtype=str)
fc = dict(zip(sub[sub.subset == "face_compete"].sample_id, sub[sub.subset == "face_compete"].value))
out = []
for s_, K in (("V", 5), ("AVT", 25)):
    v = V.FZ["main"][str(K)]
    a = V.spec_of(s_, K, "joint", v, V.sel(s_, K, "joint", v)); b = V.none_spec(s_, K)
    m = V.jk(units, lambda u: u.pred(b), lambda u: np.full(len(u.y), -1), ge)
    row = L[(L.student == s_) & (L.K == K) & (L.row == "none")].iloc[0]
    out.append((f"水準 none {s_} K={K}", m["effect"] - row.effect, m["lo"] - row.lo, m["hi"] - row.hi))
    for vname, mf, ck in (("face_single", lambda u: np.array([fc.get(x) == "single" for x in u.sid]), "cl"), ("dialogue", None, "dl")):
        m = V.jk(units, lambda u: u.pred(a), lambda u: u.pred(b), ge, mask=mf, clkey=ck)
        row = S[(S.subset == vname) & (S.family == 1) & (S.student == s_) & (S.K == K)].iloc[0]
        extra = ()
        if vname == "dialogue":
            df_d = int(row.df_dialogue); q = st.t.ppf(0.975, df_d)
            dl = pd.Series(np.concatenate([u.dl for u in units if u.r == 0])).value_counts().to_numpy().astype(float)
            ged = dl.sum() ** 2 / (dl ** 2).sum()
            extra = (m["effect"] - q * m["se"] - row.lo_dlgdf, m["effect"] + q * m["se"] - row.hi_dlgdf, max(1, int(np.floor(ged)) - 1) - df_d)
        out.append((f"{vname} 族 1 {s_} K={K}", m["effect"] - row.effect, m["lo"] - row.lo, m["hi"] - row.hi) + extra)
for x in out:
    print(x[0], "差の最大", float(np.nanmax(np.abs(x[1:]))), "NaN", int(np.isnan(np.array(x[1:], float)).sum()), flush=True)
