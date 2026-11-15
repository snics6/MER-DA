# 診断の追加（感情認識の査読者の A1・A2）：仮のターゲットの回で，一致の群と不一致の群の感情の内訳（真のラベルの分布，教師の判定の分布，
# 感情ごとの教師の再現率），γ = 0 のときに各感情の疑似ラベルの重みが何本の発話に集まるか（有効な本数）．評価のテストのラベルは使わない．
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1_stats as ST
from pathlib import Path
ROOT = Path("/home/snics/MER-DA"); U = ROOT/"results/dai1/pseudo/units"
C4 = ("AFEW", "IEMOCAP", "MOSEI", "MELD"); E = ("Neu", "Hap", "Sad", "Ang")
rows, neff_rows = [], []
for T in C4:
    for c in C4:
        if c == T: continue
        y_of, _ = ST.label_maps(c)
        for f in range(5):
            for r in range(2):
                b = U/f"{T}_{c}_f{f}_r{r}"
                z1, z2 = np.load(f"{b}_p1.npz"), np.load(f"{b}_p2.npz")
                ids = z1["sample_pool"].astype(str); y = y_of(ids)
                for K in (5, 25):
                    nk = ~np.isin(ids, z1[f"sample_K{K}"].astype(str))
                    P = {m: z2[f"p2_K{K}_ret_{m}_pool"].astype(np.float64) for m in "AVT"}
                    L = sum(np.log(np.clip(P[m], 1e-12, None)) for m in "AVT"); L -= L.max(1, keepdims=True)
                    Q = np.exp(L); Q /= Q.sum(1, keepdims=True); tq = Q.argmax(1)
                    am = {m: P[m].argmax(1) for m in "AVT"}
                    for s in ("A", "V", "T", "AVT"):
                        others = [m for m in "AVT" if m != s] if s != "AVT" else list("AVT")
                        cons = np.all([am[others[0]] == am[o] for o in others[1:]], axis=0)
                        for name, g in (("一致", cons & nk), ("不一致", ~cons & nk)):
                            for k in range(4):
                                yk = g & (y == k)
                                rows.append(dict(T=T, c=c, K=K, student=s, group=name, emotion=E[k], share_true=yk.sum()/g.sum(),
                                                 share_pred=((tq == k) & g).sum()/g.sum(), recall=(tq[yk] == k).mean() if yk.any() else np.nan, n=int(yk.sum())))
                        # γ = 0：一致の群だけに重み．感情ごとの総量をそろえた後の有効な本数（実験 1 の D11 と同じ式）
                        for gamma in (0.0, 0.25, 1.0):
                            m = np.where(cons, 1.0, gamma)[:, None] * nk[:, None]
                            Wt = Q * m; Wt = Wt / np.clip(Wt.sum(0, keepdims=True), 1e-12, None)
                            neff = (Wt.sum(0) ** 2) / np.clip((Wt ** 2).sum(0), 1e-30, None)
                            for k in range(4):
                                neff_rows.append(dict(T=T, c=c, K=K, student=s, gamma=gamma, emotion=E[k], neff=neff[k]))
D = pd.DataFrame(rows); N = pd.DataFrame(neff_rows)
out = ROOT/"results/explore_consensus"; D.to_csv(out/"consensus_emotion_units.csv", index=False); N.to_csv(out/"consensus_neff_units.csv", index=False)
pd.set_option("display.width", 250)
Dc = D.groupby(["c", "K", "student", "group", "emotion"]).mean(numeric_only=True).reset_index(); Dc.to_csv(out/"consensus_emotion_cells.csv", index=False)
x = Dc[(Dc.K == 5) & (Dc.student.isin(["A", "T", "AVT"]))].copy()
for col in ("share_true", "share_pred", "recall"): x[col] = (x[col]*100).round(0)
print("真のラベルの割合（%）"); print(x.pivot_table(index=["c", "student"], columns=["group", "emotion"], values="share_true").to_string())
print("教師の再現率（%）"); print(x.pivot_table(index=["c", "student"], columns=["group", "emotion"], values="recall").to_string())
Nc = N.groupby(["c", "K", "student", "gamma", "emotion"]).neff.mean().reset_index(); Nc.to_csv(out/"consensus_neff_cells.csv", index=False)
print("γ ごとの有効な本数（K = 5）"); print(Nc[Nc.K == 5].pivot_table(index=["c", "student"], columns=["gamma", "emotion"], values="neff").round(0).to_string())
