"""gate-code の再検査 3（DAI-6，コミット b0f858a）：足した量の検算。results/ には書かない。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
B = Path(__file__).resolve().parents[1]
ROOT = B.parents[2]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai6_probe as P
o_old, o_new = B / "code_run_v2/afew_f0", B / "code_run_v4/out"
# (1) 第 3 版（b47568f）と変わっていないはずの表
for t in ["seqtest", "subspace_uar", "anchor_outside", "k0_proj", "stability"]:
    a = pd.read_csv(o_old / f"{t}.csv"); b = pd.read_csv(o_new / f"{t}.csv"); b = b[b.c == "AFEW"].reset_index(drop=True)
    num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and c not in ("sec", "perm_sec")]
    print(f"(1) {t}: 行 {len(a)}→{len(b)}，共通の数値の列の差の最大 {np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float))):.2e}")
va_o = pd.read_csv(o_old / "variates.csv"); va_n = pd.read_csv(o_new / "variates.csv"); va_n = va_n[va_n.c == "AFEW"].reset_index(drop=True)
for col in ["eta2_cluster", "eta2_emotion", "rho_duration", "rho_ntoken", "rho_compression", "eta2_cluster_null_q95"]:
    print(f"(1) variates.{col} の差の最大 {np.nanmax(np.abs(va_o[col] - va_n[col])):.2e}")
de_o = pd.read_csv(o_old / "decomp.csv"); de_n = pd.read_csv(o_new / "decomp.csv"); de_n = de_n[de_n.c == "AFEW"].reset_index(drop=True)
for col in ["fro2_total", "fro2_between", "sv_within_1", "null_q95_sv_between_3", "wnull_q95_sv_within_1"]:
    print(f"(1) decomp.{col} の差の最大 {np.nanmax(np.abs(de_o[col] - de_n[col])):.2e}")
# (2) 感情だけで決まる変量の η² の範囲：でたらめの組み合わせ 2,000 個と，各変量の emo_eta2_cluster が [min, max] に入るか
un = pd.read_csv(o_new / "units.csv"); va = pd.read_csv(o_new / "variates.csv")
for c in ["AFEW", "IEMOCAP"]:
    if c not in set(un.c):
        continue
    U = P.ProbeUnit(c, 0)
    er = un[(un.c == c) & (un.get("kind") == "emotion_only_range")].iloc[0]
    for nm, codes in (("cluster", U.cluster_codes), ("speaker", U.center_codes)):
        if nm == "speaker" and c != "IEMOCAP":
            continue
        rng = np.random.default_rng(0)
        vals = [P.eta2(rng.normal(size=4)[U.y], codes) for _ in range(2000)]
        lo, hi = er[f"emo_eta2_{nm}_min"], er[f"emo_eta2_{nm}_max"]
        v = va[va.c == c][f"emo_eta2_{nm}"]
        print(f"(2) {c} {nm}：式の範囲 [{lo:.4f}, {hi:.4f}]，でたらめの組み合わせ [{min(vals):.4f}, {max(vals):.4f}]，"
              f"変量の emo_ の値 [{v.min():.4f}, {v.max():.4f}]，範囲の外の数 {int(((v < lo - 1e-9) | (v > hi + 1e-9)).sum())}")
    for nm, col in (("duration", "duration_sec"), ("ntoken", "n_token"), ("marker", "marker"), ("compression", "compression_ratio")):
        if f"emo_R_{nm}" not in er or pd.isna(er[f"emo_R_{nm}"]):
            continue
        v = va[va.c == c][f"emo_rho_{nm}"].abs()
        print(f"(2) {c} {nm}：重相関の上限 {er[f'emo_R_{nm}']:.4f}，変量の |emo_ の順位相関| の最大 {v.max():.4f}")
    # (3) 残り = 変量 − 感情の平均：感情の η² が 0
    print(f"(3) {c}：res_ の変量の数 {int(va[va.c == c]['res_eta2_cluster'].notna().sum())}，emo_ と res_ の列が全行にある {bool(va[va.c == c][['emo_eta2_cluster', 'res_eta2_cluster']].notna().all().all())}")
    if c == "IEMOCAP":
        print("(3) IEMOCAP の筋書きの η² の列", [x for x in va.columns if "scenario" in x], "筋書きの数", U.meta["scenario"].nunique(), sorted(U.meta["scenario"].unique()))
# (4) between_eq：感情の割合が等しいときは between と一致するはず → 人工の判定で
import torch
rng = np.random.default_rng(1)
y = np.repeat(np.arange(4), 500); Ya = P.tt(rng.normal(size=(2000, 5)) + np.eye(4, 5)[y]); Yb = P.tt(rng.normal(size=(2000, 5)) + np.eye(4, 5)[y])
r = P._parts(Ya - Ya.mean(0), Yb - Yb.mean(0), y)
print(f"(4) 感情の割合が等しい人工データで fro_between {r['fro_between']:.6f}，fro_between_eq {r['fro_between_eq']:.6f}")
# (5) _hypergeom_p：2 対 2 で 4 つとも一致 = 1/6
print("(5) _hypergeom_p(4, 2, 2) =", P._hypergeom_p(4, 2, 2), "，(4, 1, 1) =", P._hypergeom_p(4, 1, 1), "，(2, 2, 2) =", P._hypergeom_p(2, 2, 2))
# (6) ブートストラップの重みつき平均：重みがすべて 1 なら点推定
class One:
    def multinomial(self, K, p, size): return np.ones((size, K), dtype=np.int64)
orig = np.random.default_rng
pf = []
r2 = orig(3)
for j in range(5):
    yy = r2.integers(0, 4, 300); cl = np.array([f"f{j}c{k}" for k in r2.integers(0, 7 if j else 2, 300)])
    pf.append((yy, cl, {"S": np.where(r2.random(300) < .5, yy, r2.integers(0, 4, 300))}))
pt = np.mean([100 * np.mean([np.mean(p["S"][y_ == k] == k) for k in range(4)]) for y_, _, p in pf])
np.random.default_rng = lambda seed=None: One()
bt, nv = P._boot_fold_mean(pf, n_boot=2)
np.random.default_rng = orig
bt2, nv2 = P._boot_fold_mean(pf)
print(f"(6) 重み 1：{bt['S'][0]:.10f}，点推定 {pt:.10f}；本当の抜き直し：有効 {nv2}，平均 {bt2['S'].mean():.3f}，区間 [{np.percentile(bt2['S'], 2.5):.3f}, {np.percentile(bt2['S'], 97.5):.3f}]")
