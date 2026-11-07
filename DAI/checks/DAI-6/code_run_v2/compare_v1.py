"""gate-code の再検査（DAI-6）：第 3 版の AFEW 分割 0 の出力を第 2 版の出力と照らし，足した保存を確かめる。CPU だけ。"""
import numpy as np, pandas as pd
from pathlib import Path
B = Path(__file__).resolve().parents[1]
o1, o2 = B / "code_run/afew_f0", B / "code_run_v2/afew_f0"
for t in ["units", "seqtest", "decomp", "variates", "subspace_uar", "anchor_outside", "k0_proj", "stability"]:
    a, b = pd.read_csv(o1 / f"{t}.csv"), pd.read_csv(o2 / f"{t}.csv")
    num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and c not in ("sec", "perm_sec")]
    d = np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float))) if len(a) == len(b) else None
    print(f"{t}: 行 {len(a)}→{len(b)}，新しい列 {sorted(set(b.columns) - set(a.columns))}，共通の数値の列の差の最大 {d}")
su = pd.read_csv(o2 / "subspace_uar.csv")
print("commit", su.commit.unique(), "dirty", su.dirty.unique(), "ok", su.ok.unique(), "C_edge", su.C_edge.value_counts(dropna=False).to_dict())
u = pd.read_csv(o2 / "units.csv"); print("r_capped_p", u.r_capped_p.value_counts(dropna=True).to_dict())
de = pd.read_csv(o2 / "decomp.csv"); print("fro2 = fro² の差の最大", float((de.fro2_total - de.fro_total ** 2).abs().max()), "帰無の列", [c for c in de.columns if c.startswith("null_mean_fro2")])
z = np.load(o2 / "fits/AFEW_f0.npz"); f = z.files
pre = pd.Series([k.split("_")[0] for k in f]).value_counts().to_dict(); print("配列の種類", pre, "計", len(f))
info = su[su.row == "info"]
miss = [k for k in (f"{r.version}_{r.student}_{r.space}" for r in info.itertuples()) if f"pred_{k}" not in f]
print("情報の量の行で判定の無い行", len(miss), "，W の無い行", sum(f"W_{r.version}_{r.student}_{r.space}" not in f for r in info.itertuples()))
yt = z["test_y"].astype(int)
def uar(p): return 100 * np.mean([np.mean(p[yt == k] == k) for k in range(4)])
d1 = max(abs(uar(z[f"pred_{r.version}_{r.student}_{r.space}"]) - r.uar) for r in info.itertuples())
Cg = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]
d2 = max(abs(uar(z[f"predC_{r.version}_{r.student}_{r.space}"][Cg.index(r.C)]) - r.uar) for r in info.itertuples() if f"predC_{r.version}_{r.student}_{r.space}" in f)
d3 = max(abs(z[f"cv_{r.version}_{r.student}_{r.space}"].mean(0).max() - r.cv_uar) for r in info.itertuples() if f"cv_{r.version}_{r.student}_{r.space}" in f)
print(f"保存した判定の UAR と表の UAR の差の最大 {d1:.2e}，12 個の C の判定の選んだ C の UAR の差 {d2:.2e}，交差検証の表の最大と cv_uar の差 {d3:.2e}")
# 保存した W・b からテストの判定を作り直す（AFEW の特徴を読み直す）
import sys, os
sys.path.insert(0, str(B.parents[2] / "DAI" / "exp"))
import dai6_probe as P
U = P.ProbeUnit("AFEW", 0)
Xt = {m: U.Xt[m].float().cpu().numpy() for m in P.MODS}
def X(s): return np.hstack([Xt[m] for m in (list(s) if s != "AVT" else list(P.MODS))])
r = info[(info.version == "i") & (info.student == "AVT") & (info.space == "S")].iloc[0]
B_ = {m: z[f"S_i_{m}"] for m in P.MODS}
F = np.hstack([Xt[m] @ B_[m] for m in P.MODS])
p = (F @ z["W_i_AVT_S"].T + z["b_i_AVT_S"]).argmax(1)
print("W・b から作り直した判定と保存した判定の一致", float((p == z["pred_i_AVT_S"]).mean()))
m = su[(su.row == "method") & (su["T"] == "MOSEI") & (su.student == "AVT") & (su.space == "S") & (su.version == "i")].iloc[0]
pm = (X("AVT") @ z["mW_MOSEI_i_AVT_S"].T + z["mb_MOSEI_i_AVT_S"]).argmax(1)
print("手法の形の行の W・b から作り直した UAR", uar(pm), "表", m.uar, "保存の判定との一致", float((pm == z["mpred_MOSEI_i_AVT_S"]).mean()))
k0 = pd.read_csv(o2 / "k0_proj.csv").query("T=='MOSEI' and student=='AVT' and weights=='P_W0' and version=='i' and space=='S'").iloc[0]
pk = (X("AVT") @ z["k0W_MOSEI_i_AVT_S"].T + z["k0bbal_MOSEI_i_AVT_S"]).argmax(1)
print("K=0 の射影の切片を直した UAR", uar(pk), "表", k0.uar_balanced)
print("正準の方向の形", z["cca_dir_i_A_T_a"].shape, z["cca_dir_i_A_T_b"].shape, z["cca_rho_i_A_T"][:3], "（ii）", z["cca_dir_ii_A_VT_a"].shape, "b の側", "cca_dir_ii_A_VT_b" in f)
sc = Xt["A"]  # 方向が変量を作るか：候補の上の分散 ≈ 1/(1+γ)
Xp = U.Xp["A"].float().cpu().numpy(); v = Xp @ z["cca_dir_i_A_T_a"]; print("候補の上の変量の分散（先頭 3）", v.var(0, ddof=1)[:3])
print("ファイルの大きさ MB", round(os.path.getsize(o2 / "fits/AFEW_f0.npz") / 2**20, 2))
