"""gate-code 再検査（DAI-22 第 4 版）：試走 t1 の出力の形と，第 4 版で足した量を独立に作り直した値との差だけを出す．量の値は出さない．"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai14 as D14  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai_solver as S  # noqa: E402

C = Path("/home/snics/MER-DA/DAI/checks/DAI-22/code_run_v4/t1")
R1 = Path("/home/snics/MER-DA/DAI/results/dai1/eval/units")
R14 = Path("/home/snics/MER-DA/DAI/results/dai14/units")
csvs = pd.concat([pd.read_csv(p) for p in sorted((C / "units").glob("*.csv"))])
print("単位の csv：行", len(csvs), "列", csvs.shape[1], "識別子の重複", int(csvs.duplicated(["T", "fold", "seed", "K", "readout"]).sum()))
r0 = csvs[csvs.readout == "R0"]
new = ["n_nonshared", "n_clusters_nonshared", "dai1_none_abs_diff", "dai1_prop_p2_abs_diff", "dai1_prop_p3_abs_diff"]
print("足した列が R0 の行にあるか", {c: (c in r0.columns) for c in new}, "R0 の行の NaN", {c: int(r0[c].isna().sum()) for c in new if c in r0})
print("照合の差の最大（J_none, J_prop p2, J_prop p3）", float(r0.dai1_none_abs_diff.max()), float(r0.dai1_prop_p2_abs_diff.max()), float(r0.dai1_prop_p3_abs_diff.max()))

# sha256：単位の csv の dai1_unit_sha256 が p1/p2/p3 の npz と csv の 6 本を指すか
shad = 0
for (T, f, r), x in csvs.groupby(["T", "fold", "seed"]):
    files = sorted(R1.glob(f"{T}_ALL_f{f}_r{r}_p*.*"))
    want = ",".join(hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in files)
    shad += int(x.dai1_unit_sha256.nunique() != 1 or x.dai1_unit_sha256.iloc[0] != want or len(files) != 6)
print("sha256 の列が 6 本のファイルと合わない単位", shad, "ファイルの名前（AFEW f0 r0）", [p.name for p in sorted(R1.glob("AFEW_ALL_f0_r0_p*.*"))])

# MELD の nonshared の大きさを npz から作り直す
nsd = 0
for _, row in r0[r0["T"] == "MELD"].iterrows():
    a = np.load(C / "units" / f"MELD_f{row.fold}_r{row.seed}.npz")
    ns = a["nonshared"].astype(bool)
    nsd += int(row.n_nonshared != ns.sum()) + int(row.n_clusters_nonshared != len(np.unique(a["cl"][ns])))
    nkns = int((ns & a[f"nk__K{int(row.K)}"].astype(bool)).sum())
    print(f"MELD f{row.fold} r{row.seed} K{int(row.K)}：n_nonshared − （K 本を除いた nonshared）", int(row.n_nonshared) - nkns)
print("nonshared の数の作り直しとの違い", nsd)

sdir = C / "summary"
asm = pd.read_csv(sdir / "assumptions.csv")
cond = pd.read_csv(sdir / "conditions.csv")
print("assumptions の kind ごとの行：", asm.kind.value_counts().sort_index().to_dict())
print("conditions：行", len(cond), "重複", int(cond.duplicated(["T", "K", "subset", "quantity"]).sum()), "Inf", int(np.isinf(cond.select_dtypes(float)).sum().sum()),
      "quantity の種類", cond.quantity.nunique(), "subset ごとの行", cond.subset.value_counts().to_dict())
newq = ["R0_minus_pseudo", "R1_minus_pseudo"] + [f"{p}_{s}" for p in ("start_minus_none", "R0_minus_start") for s in "AVT"]
print("新しい量の行（4 コーパス × 2 K × all ＋ MELD nonshared 2）", {q: int((cond.quantity == q).sum()) for q in newq})
x = cond[cond.quantity.isin(newq)]
print("新しい量の NaN（effect, lo, hi）", int(x[["effect", "lo", "hi"]].isna().sum().sum()), "区間の順序が崩れた行", int(((x.lo > x.effect + 1e-12) | (x.effect > x.hi + 1e-12)).sum()))

# 恒等式：R0_minus_pseudo = imitable_R0 − delta，R1_minus_pseudo = imitable_R1 − delta
idd = 0.0
for (T, K, sub), g in cond.groupby(["T", "K", "subset"]):
    e = g.set_index("quantity").effect
    idd = max(idd, abs(e["R0_minus_pseudo"] - (e["imitable_R0"] - e["delta"])), abs(e["R1_minus_pseudo"] - (e["imitable_R1"] - e["delta"])))
print("R0/R1_minus_pseudo の恒等式の差の最大", idd)

# 声・顔・言葉の行：実験 14 の npz から独立に作り直す（効果は combo_stats の effect と同じ作りで比べる）
d_eff, d_ag, d_k = 0.0, 0.0, 0.0
for T in ("AFEW", "IEMOCAP", "MELD", "MOSEI"):
    keys = sorted((int(p.stem.split("_")[1][1:]), int(p.stem.split("_")[2][1:])) for p in (C / "units").glob(f"{T}_f*_r*.npz"))
    U = {k: np.load(C / "units" / f"{T}_f{k[0]}_r{k[1]}.npz") for k in keys}
    U14 = {k: np.load(R14 / f"{T}_f{k[0]}_r{k[1]}.npz") for k in keys}
    g_eff, _ = ST.g_eff_of(T)
    for K in (5, 25):
        for sub in (["all", "nonshared"] if T == "MELD" else ["all"]):
            def msk(k):
                m = U[k][f"nk__K{K}"].astype(bool)
                return m & U[k]["nonshared"].astype(bool) if sub == "nonshared" else m
            def cnt(arr_key):
                return {k: D14.counts(U[k]["y"][msk(k)], U14[k][arr_key][msk(k)], U[k]["cl"][msk(k)]) for k in keys}
            for s in "AVT":
                st_, no_, r0_ = cnt(f"start__{s}"), cnt(f"none__K{K}__{s}"), cnt(f"K{K}__main__R0__{s}__prop_one__true")
                for qn, cs, co in ((f"start_minus_none_{s}", [st_, no_], [1, -1]), (f"R0_minus_start_{s}", [r0_, st_], [1, -1])):
                    e = D14.combo_stats(cs, co, g_eff, np.random.default_rng(0))["effect"]
                    got = cond[(cond["T"] == T) & (cond.K == K) & (cond.subset == sub) & (cond.quantity == qn)].effect
                    assert len(got) == 1
                    d_eff = max(d_eff, abs(float(got.iloc[0]) - e))
                # 単純な UAR の平均での別の作り直し（K 本を除いた候補，単位ごとの UAR の平均ではなく combo_stats の effect と同じかは上で見た）
        for s in "AVT":
            for nm, (xk, zk) in ((f"agree_R0_tgt_{s}", (f"K{K}__main__R0__{s}__prop_one__true", f"tgt__K{K}__prop_one__true")),
                                 (f"agree_R0_start_{s}", (f"K{K}__main__R0__{s}__prop_one__true", f"start__{s}"))):
                v = [float((U14[k][xk][U[k][f"nk__K{K}"].astype(bool)] == U14[k][zk][U[k][f"nk__K{K}"].astype(bool)]).mean()) for k in keys]
                row = asm[(asm.kind == nm) & (asm["T"] == T) & (asm.K == K)]
                assert len(row) == 1
                d_ag = max(d_ag, abs(row.agree_mean.iloc[0] - np.mean(v)), abs(row.agree_min.iloc[0] - min(v)), abs(row.agree_max.iloc[0] - max(v)))
            # κ を sklearn で作り直す
            from sklearn.metrics import cohen_kappa_score
            kv = [cohen_kappa_score(U14[k][f"tgt__K{K}__prop_one__true"][U[k][f"nk__K{K}"].astype(bool)],
                                    U14[k][f"none__K{K}__{s}"][U[k][f"nk__K{K}"].astype(bool)]) for k in keys]
            row = asm[(asm.kind == f"kappa_pseudo_none_{s}") & (asm["T"] == T) & (asm.K == K)]
            assert len(row) == 1
            d_k = max(d_k, abs(row.kappa_mean.iloc[0] - np.mean(kv)), abs(row.kappa_min.iloc[0] - min(kv)), abs(row.kappa_max.iloc[0] - max(kv)))
        # 融合の行の tgt と実験 14 の tgt が同じか（κ_s の q が融合の行の q と同じであることの確認）
        for k in keys:
            assert (U[k][f"tgt__K{K}"].astype(int) == U14[k][f"tgt__K{K}__prop_one__true"].astype(int)).all()
print("声・顔・言葉の行の差の量の作り直しとの差の最大", d_eff)
print("声・顔・言葉の行の一致の作り直しとの差の最大", d_ag)
print("声・顔・言葉の行の κ を sklearn で作り直した差の最大", d_k)
# 融合の行の κ も sklearn で
from sklearn.metrics import cohen_kappa_score  # noqa: E402
dk2 = 0.0
for _, row in r0.iterrows():
    a = np.load(C / "units" / f"{row['T']}_f{row.fold}_r{row.seed}.npz")
    nk = a[f"nk__K{int(row.K)}"].astype(bool)
    dk2 = max(dk2, abs(row.kappa_pseudo_none - cohen_kappa_score(a[f"tgt__K{int(row.K)}"][nk], a[f"none__K{int(row.K)}"][nk])))
print("融合の行の κ を sklearn で作り直した差の最大", dk2)
ns = asm[asm.kind == "nonshared_size"]
print("nonshared_size の行", len(ns), "列", [c for c in ns.columns if ns[c].notna().any()])
