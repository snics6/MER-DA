"""gate-code の再検査（第 3 版）：回の数え方，(f, r) への足し合わせ，係数の保存の精度，対話の列，保存の大きさ。UAR は出さない。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai_data as DD  # noqa: E402
import dai7_probe_v2 as P7  # noqa: E402
import dai8_probe as P8  # noqa: E402

ROWS = ROOT / "DAI" / "checks" / "DAI-8" / "code_run_v2" / "main" / "rows"
DEV, DT = P8.DEV, P8.DT

print("== (1) (f, r) への足し合わせ：足した数の UAR と，T ごとの UAR の平均の差（同じ (c, f, r) の 2 つの T）")
za, zb = np.load(ROWS / "AFEW_IEMOCAP_f1_r0.npz"), np.load(ROWS / "MOSEI_IEMOCAP_f1_r0.npz")
keys = [k for k in za.files if k.startswith("K") and za[k].dtype == np.int8 and za[k].shape == za["test_y"].shape]
mx, nm_bad = 0.0, 0
missing = [k for k in keys if k not in zb.files]
print("片方の T にだけある鍵", len(missing), missing[:6])
for k in [k for k in keys if k in zb.files]:
    y, cl = za["test_y"].astype(int), za["test_cluster"].astype(str)
    ca, cb = ST.counts(y, za[k].astype(int), cl), ST.counts(zb["test_y"].astype(int), zb[k].astype(int), zb["test_cluster"].astype(str))
    nm_bad += int(not (ca[0] == cb[0]).all())
    s = (ca[0], ca[1] + cb[1], ca[2] + cb[2])
    mx = max(mx, abs(ST.uar_full(s) - 0.5 * (ST.uar_full(ca) + ST.uar_full(cb))))
print(f"鍵 {len(keys)}：最大の差 {mx:.2e}（UAR のポイント），塊の並びの不一致 {nm_bad}")

print("== (2) 回の数え方（実データ，AVT，K = 5，固定の点）")
nm = "AFEW_IEMOCAP_f1_r0"
T, c, fs, rs = nm.split("_")
f, r = int(fs[1:]), int(rs[1:])
u = dai1.Unit(ROOT / "results" / "dai1", T, c, [x for x in DD.CORPORA if x not in (T, c)], f, r)
z1, z3 = np.load(P8.UD / f"{nm}_p1.npz"), np.load(P8.UD / f"{nm}_p3.npz")
tether = P7.parse_tether(z3["tether"])
Xp, Xt = u.Xp["AVT"].to(DEV, DT), u.Xt["AVT"].to(DEV, DT)
XtX = Xp.T @ Xp
yp = torch.as_tensor(u.ypool.astype(int), device=DEV)
cw, cb = tether[("AVT", 5)]
Wn, bn = P7.p1_wb(z1, "AVT", f"5|{cw}|{cb}")
q = torch.softmax(P8.logits_np(Xp, Wn, bn), -1)
kp = torch.as_tensor(u.kpos[5], dtype=torch.long, device=DEV)
r10 = P8.em_fit(Xp, XtX, kp, yp[kp], q, 0.5, 1.0)
rc = P8.em_fit(Xp, XtX, kp, yp[kp], q, 0.5, 1.0, n_iter=P8.MAX_EM, tol=P8.TOL_EM, snaps=P8.SNAPS)
# 第 1 版の回の数え方（M，E を 10 回，最後に M）
R = q.clone(); R[kp] = 0.0; R[kp, yp[kp]] = 1.0
U = torch.ones(len(yp), dtype=torch.bool, device=DEV); U[kp] = False
lq = torch.log(q[U].clamp(min=1e-12))
for _ in range(10):
    fv = P8.lda_from_R(Xp, R, XtX, 0.5)
    R[U] = torch.softmax(lq + torch.log(fv["pi"].clamp(min=1e-12)) + P8.delta(Xp[U], fv), -1)
fv = P8.lda_from_R(Xp, R, XtX, 0.5)
f0 = P8.lda_from_R(Xp, torch.where(U[:, None], q, torch.nn.functional.one_hot(yp, 4).to(DT)), XtX, 0.5)
print(f"n_iter=10 と収束版の回 10：係数の最大の差 {float((r10['fit']['A'] - rc['snap'][10]['A']).abs().max()):.2e}")
print(f"第 1 版の数え方の 10 回と：係数の最大の差 {float((r10['fit']['A'] - fv['A']).abs().max()):.2e}")
print(f"回 0 と「初期の重みで 1 回解いた LDA」：係数の最大の差 {float((rc['snap'][0]['A'] - f0['A']).abs().max()):.2e}")
print(f"収束版：回の数 {rc['info']['n_iter_done']}，hist の長さ {len(rc['hist'])}，pool_hist の長さ {len(rc['pool_hist'])}，保存した回 {sorted(rc['snap'])}，"
      f"最後の変化 < tol {rc['hist'][-1] < P8.TOL_EM if rc['hist'] else None}")
# EM_truth の回 0 は LDA_oracle と同じ
oh = torch.nn.functional.one_hot(yp, 4).to(DT)
fo = P8.lda_from_R(Xp, oh, XtX, 0.5)
emp = torch.zeros(0, dtype=torch.long, device=DEV)
rt = P8.em_fit(Xp, XtX, emp, emp, oh, 0.5, 0.0, n_iter=0)
print(f"EM_truth の回 0 と LDA_oracle：係数の最大の差 {float((rt['fit']['A'] - fo['A']).abs().max()):.2e}")

print("== (3) float16 で保存した係数から判定を作り直したときの不一致（テスト）")
z = np.load(ROWS / f"{nm}.npz")
Xtn = {s: u.Xt[s].double().cpu().numpy() for s in P8.SPACES}
tot = bad = 0
for k in [k[4:] for k in z.files if k.startswith("C_A_")]:
    s = k.split("_")[1] if k.startswith("K") else k.split("_")[0]
    A16, c32 = z[f"C_A_{k}"].astype(np.float64), z[f"C_c_{k}"].astype(np.float64)
    pr = (Xtn[s] @ A16 + c32).argmax(1)
    bad += int((pr != z[k]).sum()); tot += len(pr)
    if not np.isfinite(A16).all():
        print("有限でない float16 の係数", k)
print(f"係数の組 {len([k for k in z.files if k.startswith('C_A_')])}：判定の不一致 {bad}/{tot}")
print("float16 の A の絶対値の最大と 65504 の比：", max(float(np.abs(z[k].astype(np.float64)).max()) for k in z.files if k.startswith("C_A_")) / 65504)
# μ を float16 から作り直した c と保存の c
k0 = [k[4:] for k in z.files if k.startswith("C_A_") and "AVT_em_" in k][0]
c_rec = -0.5 * (z[f"C_mu_{k0}"].astype(np.float64) * z[f"C_A_{k0}"].astype(np.float64).T).sum(1)
print(f"{k0}：float16 の μ と A から作った c と保存の c の最大の相対差 {float(np.abs(c_rec - z[f'C_c_{k0}']).max() / np.abs(z[f'C_c_{k0}']).max()):.2e}")
D16 = z[f"D_{k0}"].argmax(1)
print(f"δ（float32）の argmax と保存の判定の不一致 {int((D16 != z[k0]).sum())}/{len(D16)}")
del u, Xp, Xt, XtX
torch.cuda.empty_cache()

print("== (4) 対話の列（Q1 と R5）")
for p in sorted(ROWS.glob("*_knn.csv")):
    kn = pd.read_csv(p)
    print(p.name, "対話の数", kn.n_dialogues.tolist(), "dlg0 = 1 の空間", int((kn.dlg0 == 1).sum()), "emo0_xdlg = 0 の空間", int((kn.emo0_xdlg == 0).sum()),
          "グループの数", kn.n_groups.tolist())

print("== (5) 行のファイル：大きさ，鍵の種類，NaN，回の数")
for p in sorted(ROWS.glob("*.npz")):
    z = np.load(p)
    fl = z.files
    kinds = {"判定": sum(1 for k in fl if z[k].dtype == np.int8 and k != "test_y" and not k.startswith("P_")),
             "D_": sum(k.startswith("D_") for k in fl), "C_": sum(k.startswith("C_") for k in fl), "P_": sum(k.startswith("P_") for k in fl)}
    nan = sum(int((~np.isfinite(z[k])).sum()) for k in fl if z[k].dtype in (np.float16, np.float32, np.float64) and z[k].ndim > 0)
    dg = json.loads(str(z["diag"]))
    fix = {k: v["n_iter_done"] for k, v in dg.items() if k.endswith("_em_a0.5_b1.0")}
    lt10 = sum(v < 10 for v in fix.values())
    hit = sum(v >= P8.MAX_EM for v in fix.values())
    emit = {t: sum(1 for k in fl if f"_emit{t}_" in k) for t in P8.SNAPS}
    nP = {k: len(z[k]) for k in fl if k.startswith("P_")}
    print(f"{p.name}: {p.stat().st_size / 2**20:.2f} MiB，{kinds}，NaN {nan}，固定の点の回 < 10 の数 {lt10}，100 回で止まった数 {hit}/{len(fix)}，"
          f"emit の数 {emit}，P_ の長さの種類 {sorted(set(nP.values()))}，n_pool {int(z['n_pool'])}，commit {str(z['commit'])}")
