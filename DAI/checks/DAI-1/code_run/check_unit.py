"""gate-code: 1 単位の出力の検査。UAR の値は画面に出さず，差と数だけを出す。"""
import sys, json
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
from pathlib import Path
import torch
torch.cuda.set_per_process_memory_fraction(0.5)
import numpy as np, pandas as pd
import dai1, dai_data as DD, dai_solver as S
out = Path(sys.argv[1]); base = out / "pseudo" / "units" / "MOSEI_MELD_f0_r0"
p1, p2, p3 = (np.load(f"{base}_p{i}.npz") for i in (1, 2, 3))
r1, r2, r3 = (pd.read_csv(f"{base}_p{i}.csv", low_memory=False) for i in (1, 2, 3))
print("dtype：p1 W", p1["p1_A_W"].dtype, " p2 ret W", p2["p2_K5_ret_A_W"].dtype, " p2 確率", p2["p2_K5_none_A_test"].dtype, " p3 W", p3["p3_A_K5_W"].dtype, " stat", p1["stat_A_mu"].dtype)
print("p3 の鍵の数", len(p3.files), " 予測の確率の保存（p3 に test の確率があるか）", any("prob" in k or "test" in k for k in p3.files))
teth = {(k, K): (0.01, 10.0) for k in DD.KINDS for K in DD.KS}
w = {K: (2.0, 0.5, 1.0) for K in dai1.KS_ALL}
u = dai1.Unit(out, "MOSEI", "MELD", ["AFEW", "IEMOCAP"], 0, 0)
# 1) sample_id の照合
print("sample_id 照合：pool", bool((p1["sample_pool"] == u.cp.df.sample_id.values[u.pool]).all()), " test", bool((p1["sample_test"] == u.cp.df.sample_id.values[u.test]).all()))
# 2) 保存の W・b（と保存の平均・標準偏差）と生の特徴から，テストの判定を作り直して csv の UAR と照合（段 3）
raw = {k: DD.kind_matrix(DD.load_corpus("MELD")[1], k) for k in ("A", "V", "T", "AVT")}
df = u.cp.df; tpos = pd.Series(np.arange(len(df)), index=df.sample_id.values).loc[p1["sample_test"]].values
nbad, ntot, maxd = 0, 0, 0.0
for s in DD.STUDENTS:
    mu, sd = p1[f"stat_{s}_mu"].astype(np.float64), p1[f"stat_{s}_sd"].astype(np.float64)
    Xt = (raw[s][tpos].astype(np.float64) - mu) / sd
    for K in dai1.KS_ALL:
        W, b = p3[f"p3_{s}_K{K}_W"], p3[f"p3_{s}_K{K}_b"]
        items = [json.loads(x) for x in p3[f"p3_{s}_K{K}_items"]]
        for dt in (np.float64, np.float32):
            Z = np.einsum("nd,pkd->pnk", Xt.astype(dt), W.astype(dt)) + b.astype(dt)[:, None, :]
            ua = np.array([100 * S.uar(u.ytest, z.argmax(1)) for z in Z])
            sub = r3[(r3.stage == 3) & (r3.student == s) & (r3.K == K) & (r3.method.isin(["joint", "kd", "agree", "tt"]))]
            assert len(sub) == len(items)
            d = np.abs(ua - sub.uar.values)
            if dt is np.float32:
                nbad += int((d > 1e-9).sum()); ntot += len(d); maxd = max(maxd, d.max())
            else:
                assert d.max() < 1e-9, (s, K, d.max())
print(f"段 3 の作り直し：float64 は全問で一致．float32 に丸めた W・特徴で UAR がずれた問 {nbad}/{ntot}，最大のずれ {maxd:.3g} ポイント")
# 3) 退化した場合の検算：疑似ラベルの重みを 0 にした問（G = K 本だけ）を pass3 と同じ形で解き，段 1 の none[K] と比べる
TK = dai1.build_tk(u, p1, p2, teth)
for s in DD.STUDENTS:
    for K in DD.KS:
        cw, cb = teth[(s, K)]
        gk = u.gK(K)
        Q = dai1.product(TK[K], (1, 1, 1), 0)
        G = torch.stack([gk + S.soft_rows(Q, 0.0), gk + 0.0 * S.soft_rows(Q, float(u.n), balance=False)])
        res = S.fit(u.Xp[s], G, *u.start[s], Cw=cw, Cb=cb)
        Wn, bn = dai1.none_wb(u, p1, s, K, teth)
        dW = float((res["W"] - Wn).abs().max()); 
        pr_a = S.logits(u.Xt[s], res["W"], res["b"]).argmax(2); pr_n = S.logits(u.Xt[s], Wn, bn).argmax(0 if False else 1)
        flips = int((pr_a != pr_n[None]).sum())
        print(f"  退化 {s} K={K}: W の差の最大 {dW:.2e}，テストの判定が食い違った発話 {flips}")
# 4) 段 2 の float16 の確率と，float32 の W から作り直した確率の差
d16 = max(float(np.abs(p2[f"p2_K{K}_{n}_test"].astype(np.float64) - TK[K][n][1].cpu().numpy()).max()) for K in dai1.KS_ALL for n in ["none_A", "none_AVT", "ret_V"])
print(f"段 2 の float16 の確率と作り直した確率の差の最大 {d16:.2e}")
# 5) 行の数と格子
d3 = r3[r3.stage == 3]
print("段 3 の手法ごとの行（生徒 × K）：", d3.groupby("method").size().to_dict())
print("tt の予測（生徒 A）：", sorted(d3[(d3.student == "A") & (d3.method == "tt") & (d3.K == 5)].teacher.unique()))
print("rolesep の行に S・grad があるか：", d3[d3.method == "rolesep"][["S", "grad"]].notna().any().to_dict())
print("段 3 の grad が tol の 10 倍を超えた問：", int((d3.grad > 1e-5).sum()), " grad が NaN の問（rolesep 以外）：", int(d3[d3.method != "rolesep"].grad.isna().sum()))
print("agree の残った本数（K=0, sel, 生徒 A）：", d3[(d3.method == "agree") & (d3.K == 0) & (d3.student == "A")][["teacher", "tau", "n_kept"]].drop_duplicates(["teacher","tau"]).to_dict("records"))
print("csv の列：", list(r3.columns))
print("同じ識別子の重複行（段 3）：", int(d3.drop(columns=["uar", "S", "grad", "sec"], errors="ignore").astype(str).duplicated().sum()))
