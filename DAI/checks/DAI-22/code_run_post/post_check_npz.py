"""DAI-22 の計算の後の点検（2）：npz の鍵と中身，設計 13 節の「計算の後に保存から出すもの」が npz から出せるか．
読むだけ．画面には鍵の数，形，NaN の数，照合の差の最大だけを出す（一致の割合や UAR は出さない）．"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA/DAI")
R = ROOT / "results" / "dai22"; U = R / "units"; U14 = ROOT / "results" / "dai14" / "units"
KS = (5, 25)
need = {"y", "cl", "sid", "half", "nonshared", "marker", "sex", "pca", "start"}
for K in KS:
    need |= {f"nk__K{K}", f"none__K{K}", f"tgt__K{K}", f"maxp__K{K}", f"clamped__K{K}", f"K{K}__R0", f"prob__K{K}__R0",
             f"K{K}__R1", f"prob__K{K}__R1", f"inner__K{K}"}
bad_keys, n_nan, n_inf, bad_shape, bad_sum, bad_range, bad_nonshared = 0, 0, 0, 0, 0, 0, 0
csvs = {p.stem: pd.read_csv(p) for p in sorted(U.glob("*.csv"))}
dis_shapes, start_agree_ok, gfin = set(), 0, []
for p in sorted(U.glob("*.npz")):
    a = np.load(p, allow_pickle=False); ks = set(a.files) - {k for k in a.files if k.endswith("R0_tight")}
    if ks != need:
        bad_keys += 1; print(p.name, "鍵の違い", sorted(ks ^ need))
    n = len(a["y"])
    for k in a.files:
        v = a[k]
        if v.dtype.kind == "f":
            n_nan += int(np.isnan(v).sum()); n_inf += int(np.isinf(v).sum())
        if k not in ("pca",) and not k.startswith("inner") and v.shape[0] != n:
            bad_shape += 1; print(p.name, k, v.shape, n)
    if not set(np.unique(a["y"])) <= {0, 1, 2, 3}:
        bad_range += 1
    for K in KS:
        for ro in ("R0", "R1"):
            s = a[f"prob__K{K}__{ro}"].astype(np.float64).sum(1)
            bad_sum += int((np.abs(s - 1) > 1e-4).sum())
        for k in (f"none__K{K}", f"tgt__K{K}", f"K{K}__R0", f"K{K}__R1", "start"):
            if not set(np.unique(a[k])) <= {0, 1, 2, 3}:
                bad_range += 1
    c = csvs[p.stem]
    if int(a["nonshared"].astype(bool).sum()) != int(c[c.readout == "R0"].n_nonshared.iloc[0]):
        bad_nonshared += 1
    # 13 節の 1：出発点と J_prop の一致（融合の行は dai22 の npz，声・顔・言葉の行は実験 14 の npz）
    b = np.load(U14 / p.name, allow_pickle=False)
    ok = True
    for K in KS:
        nk = a[f"nk__K{K}"].astype(bool)
        vals = [float((a["start"][nk] == a[f"tgt__K{K}"][nk]).mean())]
        vals += [float((b[f"start__{s}"][nk] == b[f"tgt__K{K}__prop_one__true"][nk]).mean()) for s in "AVT"]
        ok &= all(np.isfinite(vals))
        # 13 節の 2：本当の感情ごとの読み出しと J_prop の不一致の割合（R0 と R1）
        y = a["y"].astype(int)
        dis = np.array([[float((a[f"K{K}__{ro}"][nk & (y == e)] != a[f"tgt__K{K}"][nk & (y == e)]).mean()) for e in range(4)] for ro in ("R0", "R1")])
        dis_shapes.add((dis.shape, bool(np.isfinite(dis).all())))
    start_agree_ok += int(ok)
    # 13 節の 3：MELD の共有を除いた候補（K 本を除く）の上の塊の実効の数
    if p.stem.startswith("MELD"):
        for K in KS:
            m = a[f"nk__K{K}"].astype(bool) & a["nonshared"].astype(bool)
            _, cnt = np.unique(a["cl"][m], return_counts=True)
            gfin.append(float(cnt.sum() ** 2 / (cnt ** 2).sum()))
print("npz", len(csvs), "| 鍵の違い", bad_keys, "| 浮動の NaN", n_nan, "Inf", n_inf, "| 行の数の違い", bad_shape,
      "| 確率の和が 1 でない発話", bad_sum, "| ラベルか判定が 0〜3 の外", bad_range, "| nonshared の数が csv と違う", bad_nonshared)
print("13 節の 1（出発点と J_prop の一致，4 つの行）：有限に出せた単位", start_agree_ok, "/ 200")
print("13 節の 2（感情ごとの不一致．単位×K ごと）：形と有限", dis_shapes)
print("13 節の 3（MELD の共有を除いた候補の塊の実効の数）：単位×K の数", len(gfin), "有限", bool(np.isfinite(gfin).all()),
      "範囲", f"{min(gfin):.1f}〜{max(gfin):.1f}")

# 13 節の 4：start_minus_none_s + R0_minus_start_s を実験 14 の imitable_R0 と照合（点推定）
C = pd.read_csv(R / "summary" / "conditions.csv"); C = C[C.subset == "all"]
c14 = pd.read_csv(ROOT / "results" / "dai14" / "summary" / "conditions.csv")
mx, nn = 0.0, 0
for T in C["T"].unique():
    for K in KS:
        for s in "AVT":
            g = lambda q: float(C[(C["T"] == T) & (C.K == K) & (C.quantity == q)].effect.iloc[0])
            ref = c14[(c14["T"] == T) & (c14.K == K) & (c14.student == s) & (c14.quantity == "imitable_R0")].effect
            if len(ref) == 1:
                nn += 1; mx = max(mx, abs(g(f"start_minus_none_{s}") + g(f"R0_minus_start_{s}") - float(ref.iloc[0])))
print("13 節の 4（start_minus_none_s ＋ R0_minus_start_s と実験 14 の imitable_R0）：照合した組", nn, "最大差", f"{mx:.1e}")
# あわせて，delta_minus_delta_s と実験 14 の delta の差の恒等式（Δ_AVT − Δ_s）
mx2, n2 = 0.0, 0
for T in C["T"].unique():
    for K in KS:
        d = float(C[(C["T"] == T) & (C.K == K) & (C.quantity == "delta")].effect.iloc[0])
        for s in "AVT":
            ref = c14[(c14["T"] == T) & (c14.K == K) & (c14.student == s) & (c14.quantity == "delta")].effect
            got = float(C[(C["T"] == T) & (C.K == K) & (C.quantity == f"delta_minus_delta_{s}")].effect.iloc[0])
            if len(ref) == 1:
                n2 += 1; mx2 = max(mx2, abs(d - float(ref.iloc[0]) - got))
print("delta_minus_delta_s と（Δ_AVT − 実験 14 の Δ_s）の照合：組", n2, "最大差", f"{mx2:.1e}")
# 13 節の 5：student 列
print("13 節の 5：conditions.csv の student の値", C.student.unique().tolist(), "（_A/_V/_T の量は声・顔・言葉の行として読み替える）")
