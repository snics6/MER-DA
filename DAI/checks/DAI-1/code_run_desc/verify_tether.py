"""gate-code（DAI-1 code_v5）：偽の評価の置き場（AFEW の名前，中身は仮のターゲット MOSEI・c = AFEW）の引き戻しの曲線の行を，
p3t の判定と DAI/splits のラベルから作り直す（verify_indep2 の jackknife を使う）。値は画面に出さず差の最大だけを出す。"""
import sys, json
place, dest = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], dest, "MOSEI", "AFEW"]
sys.path.insert(0, "/home/snics/MER-DA/DAI/checks/DAI-1/code_run_desc")
import numpy as np, pandas as pd
from pathlib import Path
import verify_indep2 as V

V.UNITS = Path(place) / "eval" / "units"
V.T = "AFEW"                                                          # 置き場の名前（AFEW_ALL_f…）
V.FZ = json.load(open(Path(place) / "select" / "frozen.json"))["AFEW"]   # 中身は MOSEI の設定
lab = pd.read_csv(V.ROOT / "DAI/splits/AFEW.csv", keep_default_na=False, dtype={"cluster": str, "dialogue": str})
units = [V.U("ALL", f, r, lab) for f in range(5) for r in (0, 1)]
ge = V.g_eff("AFEW")
TC = pd.read_csv(Path(dest) / "tether_curve.csv")
TS = pd.read_csv(Path(dest) / "tether_sign.csv")
mx, n, nn = 0.0, 0, 0
for s_ in ("A", "AVT"):
    for K in (0, 5, 25):
        items = [json.loads(x) for x in units[0].p3[f"p3t_{s_}_K{K}_items"]]
        for cw, cb in ((0.001, 1.0), (0.1, 10.0), (1.0, 1.0)):
            j = [i for i, it in enumerate(items) if it["tag"] == "prop_sel" and float(it["Cw"]) == cw and float(it["Cb"]) == cb]
            assert len(j) == 1
            fa = lambda u, j=j[0], s_=s_, K=K: u.p3[f"p3t_{s_}_K{K}_pred"][j].astype(int)
            fb = (lambda u, s_=s_: u.p1[f"p1_{s_}_start_pred"].astype(int)) if K == 0 else \
                 (lambda u, s_=s_, K=K, cw=cw, cb=cb: u.pred(("none", s_, K, cw, cb)))
            m1 = V.jk(units, fa, lambda u: np.full(len(u.y), -1), ge)
            m2 = V.jk(units, fa, fb, ge)
            r1 = TC[(TC.student == s_) & (TC.K == K) & np.isclose(TC.Cw, cw) & np.isclose(TC.Cb, cb) & (TC.kind == "level") & (TC.row == "prop")].iloc[0]
            r2 = TC[(TC.student == s_) & (TC.K == K) & np.isclose(TC.Cw, cw) & np.isclose(TC.Cb, cb) & (TC.kind == "family1")].iloc[0]
            for a, b in ((m1["effect"], r1.effect), (m1["lo"], r1.lo), (m1["hi"], r1.hi), (m2["effect"], r2.effect), (m2["lo"], r2.lo), (m2["hi"], r2.hi)):
                if not np.isfinite(a - b):
                    nn += 1
                else:
                    mx = max(mx, abs(a - b))
            n += 1
print("引き戻しの曲線の点", n, "（水準と族 1 型の差）差の最大", mx, "NaN", nn)
print("選んだ引き戻しの点の判定が主の行と同じ：", TS.selected_point_matches_main.value_counts(dropna=False).to_dict(), "曲線の点の数", sorted(TS.n_points.unique()))
