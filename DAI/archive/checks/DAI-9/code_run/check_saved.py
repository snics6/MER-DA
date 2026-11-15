"""gate-code（DAI-9 第 2 回）：保存した基底と係数から判定を作り直せるか，nap の消した向きの成分，鍵の数（CPU）。値は出さない。"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1, dai_data as DD, dai9_probe as P9  # noqa: E402
T, c, f, r = "AFEW", "IEMOCAP", 1, 0
z = np.load(ROOT / "DAI/checks/DAI-9/code_run/run_v2/rows" / f"{c}_f{f}_r{r}.npz")
dg = json.loads(str(z["diag"]))
print("nap の消した向きの数", {t: dg[f"{t}_info"]["nap_r_used"] for t in ("nap10", "nap50")}, "基底の形", {k: z[k].shape for k in z.files if k.startswith("B_")})
print("P_ の数", sum(k.startswith("P_") for k in z.files), "長さが候補と違う", sum(len(z[k]) != len(z["pool_y"]) for k in z.files if k.startswith("P_")),
      "係数の組", sum(k.startswith("C_A_") for k in z.files), "大きさ", f"{(ROOT/'DAI/checks/DAI-9/code_run/run_v2/rows'/f'{c}_f{f}_r{r}.npz').stat().st_size/2**20:.2f} MiB")
u = dai1.Unit(ROOT / "results" / "dai1", T, c, [x for x in DD.CORPORA if x not in (T, c)], f, r)
Xp, Xt = u.Xp["AVT"].double(), u.Xt["AVT"].double()
gp = u.cp.df["speaker_id"].to_numpy()[u.pool].astype(str); gt = u.cp.df["speaker_id"].to_numpy()[u.test].astype(str)
for tau in P9.TAUS:
    if tau.startswith("pca"):
        B = torch.as_tensor(z[f"B_{tau}"].astype(np.float64)); mu = Xp.mean(0)
        a, b = P9._restd((Xp - mu) @ B, (Xt - mu) @ B)
    else:
        a, b = P9.transform(tau, Xp, Xt, gp, gt)
        if tau.startswith("nap"):
            B = torch.as_tensor(z[f"B_{tau}"].astype(np.float64))
            print(f"{tau}：作り直した特徴の保存した基底の向きの成分の最大 候補 {float((a @ B).abs().max()):.1e}，テスト {float((b @ B).abs().max()):.1e}")
    bad = tot = 0
    for k in [k[4:] for k in z.files if k.startswith("C_A_") and (k[4:].startswith(f"{tau}_") or f"_{tau}_" in k[4:])]:
        A, cc = z[f"C_A_{k}"].astype(np.float64), z[f"C_c_{k}"].astype(np.float64)
        pr = (b.numpy() @ A + cc).argmax(1)
        bad += int((pr != z[k]).sum()); tot += len(pr)
    print(f"{tau}：保存した係数から作り直した判定の不一致 {bad}/{tot}")
