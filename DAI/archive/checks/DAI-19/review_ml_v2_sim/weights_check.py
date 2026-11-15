"""ML の査読者の検算（DAI-19 第 2 版）：ソースのラベルだけの表から，設計 3・4 節の重み w・wn と残り方 r_s・r'_s・r_m を計算する．

入力は results/dai16_probe/summary/reliability_src.csv（ソースの中の交差検証の d′（home）と，別のソースに当てた d′（away）．声・顔・言葉）だけ．
仮のターゲット・評価のターゲットのラベルやテストの判定は使わない．融合 AVT の値はこの表に無いので計算しない．
使い方：.venv/bin/python DAI/checks/DAI-19/review_ml_v2_sim/weights_check.py results/dai16_probe/summary/reliability_src.csv
"""
import itertools
import sys

import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1])
d["sum"] = d[["d_0", "d_1", "d_2", "d_3"]].sum(axis=1)
corp = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
H = {(r.source, r.modality): r["sum"] for _, r in d[d["where"] == "home"].iterrows()}
A = {(r.source, r["where"].split("→")[1], r.modality): r["sum"] for _, r in d[d["where"].str.startswith("away")].iterrows()}
rows = []
for m in "AVT":
    for s, t in itertools.combinations(corp, 2):
        As, At, Hs, Ht = A[(s, t, m)], A[(t, s, m)], H[(s, m)], H[(t, m)]
        w = As / (As + At)
        wn = (As / Ht) / (As / Ht + At / Hs)
        assert abs(wn - As * Hs / (As * Hs + At * Ht)) < 1e-12          # wn は「別のソースで測った値 × 自分の中の値」に比例する形と同じ
        rows.append(dict(m=m, s=s, t=t, Hs=Hs, Ht=Ht, As=As, At=At, w_s=w, wn_s=wn, r_s=As / Hs, r_t=At / Ht, rn_s=As / Ht, rn_t=At / Hs,
                         r_m=(As + At) / (Hs + Ht)))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("== ソースの対 × モダリティ（18 組）の中 H，外 A，重み，残り方 ==")
print(R.to_string(index=False))
for name in ("w_s", "wn_s"):
    x = R[name].to_numpy()
    big = np.maximum(x, 1 - x)
    heavy_is_s = x > 0.5
    partner_home_higher = np.where(heavy_is_s, R.Ht > R.Hs, R.Hs > R.Ht)
    print(f"{name}：重い側の重み 最小 {big.min():.3f} 中央 {np.median(big):.3f} 最大 {big.max():.3f}．"
          f"重い側が『相手の中の値が高いソース』である組 {int(partner_home_higher.sum())} / 18，『自分の中の値が高いソース』である組 {int((~partner_home_higher).sum())} / 18．"
          f"0.5 との差が 0.05 未満の組 {int((abs(x - 0.5) < 0.05).sum())} / 18")
print("w と wn で重い側が同じ組：", int(((R.w_s > 0.5) == (R.wn_s > 0.5)).sum()), "/ 18")
rr, rn = np.concatenate([R.r_s, R.r_t]), np.concatenate([R.rn_s, R.rn_t])
print(f"r_s > 1.5 の向き {int((rr > 1.5).sum())} / 36，r'_s > 1.5 の向き {int((rn > 1.5).sum())} / 36，r'_s の範囲 {rn.min():.3f}〜{rn.max():.3f}，r_m の範囲 {R.r_m.min():.3f}〜{R.r_m.max():.3f}")
c = lambda v: np.clip(v, 0, 1.5)
R["lvl_ext"] = 0.125 * (R.As + R.At)                                    # eq の腕の表の水準（4 感情の平均）
R["lvl_extpc"] = 0.125 * (R.Hs * c(R.r_s) + R.Ht * c(R.r_t))
R["lvl_extpcn"] = 0.125 * (R.Hs * c(R.rn_s) + R.Ht * c(R.rn_t))
R["extpc/ext"] = R.lvl_extpc / R.lvl_ext
R["extpcn/ext"] = R.lvl_extpcn / R.lvl_ext
R["share_s_ext"] = R.Hs / (R.Hs + R.Ht)                                  # 表の水準のうちソース s の分（相性の混ざり方の目安）
R["share_s_extpc"] = R.Hs * c(R.r_s) / (R.Hs * c(R.r_s) + R.Ht * c(R.r_t))
R["share_s_extpcn"] = R.Hs * c(R.rn_s) / (R.Hs * c(R.rn_s) + R.Ht * c(R.rn_t))
print("== eq の腕の表の水準と，水準のうちソース s の分 ==")
print(R[["m", "s", "t", "lvl_ext", "lvl_extpc", "lvl_extpcn", "extpc/ext", "extpcn/ext", "share_s_ext", "share_s_extpc", "share_s_extpcn"]].to_string(index=False))
print("== ソースの対ごとの 3 モダリティの水準（ext と extpc）==")
for (s, t), g in R.groupby(["s", "t"], sort=False):
    print(f"{s}–{t}：ext " + " ".join(f"{r.m}={r.lvl_ext:.2f}" for r in g.itertuples()) + "｜extpc " + " ".join(f"{r.m}={r.lvl_extpc:.2f}" for r in g.itertuples()))
