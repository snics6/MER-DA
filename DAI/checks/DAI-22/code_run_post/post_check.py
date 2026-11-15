"""DAI-22 の計算の後の点検（gate-code）．results/dai22 を読むだけで書き換えない．
画面に出すのは行の数，照合の差，予測の違いの本数，収束，NaN の数，照合の最大差だけ（量の値は出さない）．"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA/DAI")
sys.path.insert(0, str(ROOT / "exp"))
import dai1_stats as ST  # noqa: E402

R = ROOT / "results" / "dai22"
U = R / "units"
CORP = ("AFEW", "IEMOCAP", "MELD", "MOSEI")
KS = (5, 25)

print("== 1. 格子と重複")
npz = sorted(U.glob("*.npz")); csv = sorted(U.glob("*.csv"))
print("npz", len(npz), "csv", len(csv), "その他", len([p for p in U.iterdir() if p.suffix not in (".npz", ".csv")]))
need = {(T, f, r) for T in CORP for f in range(5) for r in range(10)}
got_n = {(p.stem.split("_")[0], int(p.stem.split("_")[1][1:]), int(p.stem.split("_")[2][1:])) for p in npz}
got_c = {(p.stem.split("_")[0], int(p.stem.split("_")[1][1:]), int(p.stem.split("_")[2][1:])) for p in csv}
print("npz 欠け", len(need - got_n), "余り", len(got_n - need), "| csv 欠け", len(need - got_c), "余り", len(got_c - need))
df = pd.concat([pd.read_csv(p) for p in csv], ignore_index=True)
print("csv 行", len(df), "見込み", 200 * 4)
key = ["T", "fold", "seed", "K", "readout"]
print("識別子の重複", int(df.duplicated(key).sum()), "| 単位ごとの行の数", df.groupby(["T", "fold", "seed"]).size().value_counts().to_dict())
print("(T,K,readout) ごとの行", df.groupby(["T", "K", "readout"]).size().unique().tolist())
for p in csv:  # ファイル名と中身の識別子が合うか
    x = pd.read_csv(p); T, f, r = p.stem.split("_")[0], int(p.stem.split("_")[1][1:]), int(p.stem.split("_")[2][1:])
    if not ((x["T"] == T) & (x.fold == f) & (x.seed == r)).all():
        print("ファイル名と中身が違う", p.name)

print("== 2. コミットと汚れ")
print("commit", df.commit.unique().tolist(), "dirty", df.dirty.astype(str).unique().tolist())
print("dai1_commits", df.dai1_commits.unique().tolist())
print("target", df.target.unique().tolist())
print("sec 合計（単位ごと）", round(df.drop_duplicates(["T", "fold", "seed"]).sec.sum() / 60, 1), "分")

print("== 3. 収束")
print("grad 最大", df.grad.max(), "stalled 真", int(df.stalled.astype(str).eq("True").sum()))
r1 = df[df.readout == "R1"]
print("R1 inner_grad 最大", r1.inner_grad.max(), "inner_stalled 最大", r1.inner_stalled.max())
for T, x in df.groupby("T"):
    print(" ", T, "grad 最大", f"{x.grad.max():.3e}")
num = df.select_dtypes("number")
print("数の列の Inf", int(np.isinf(num.to_numpy()).sum()))
r0 = df[df.readout == "R0"]
nan_cols = {c: int(r0[c].isna().sum()) for c in r0.columns if r0[c].isna().any()}
print("R0 の行の NaN の列（解き直しの列は解き直した行にだけ値）", nan_cols)

print("== 4. 実験 3 の保存との照合が外れた行")
over = r0[r0.abs_diff > 1e-3]
print("abs_diff の NaN", int(r0.abs_diff.isna().sum()), "| 0.001 を超えた行", len(over), "| 超えない行の最大", f"{r0[r0.abs_diff <= 1e-3].abs_diff.max():.2e}")
cols = ["T", "fold", "seed", "K", "n_cand", "abs_diff", "resolve_kind", "tol_resolved", "n_pred_diff_tol1e-08", "n_pred_diff_tol1e-10",
        "n_pred_diff_tight", "uar_diff_tight_vs_unit", "grad_tol1e-08", "grad_tol1e-10", "stalled_tol1e-08", "stalled_tol1e-10",
        "n_iter_tol1e-08", "n_iter_tol1e-10"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(over[cols].to_string(index=False))
print("解き直していない行で resolve_kind が埋まった数", int(r0[r0.abs_diff <= 1e-3].resolve_kind.notna().sum()))
for _, x in over.iterrows():  # near_tie の 3 条件と same の条件を csv から作り直す
    n = x.n_cand; d8, d10 = x["n_pred_diff_tol1e-08"], x["n_pred_diff_tol1e-10"]
    same = d8 == 0 and d10 == 0
    near = x.n_pred_diff_tight == 0 and max(d8, d10) <= 1e-3 * n and x.abs_diff * n / 100 <= max(2, 1e-3 * n)
    kind = "same" if same else ("near_tie" if near else "differ")
    print(f"  {x['T']}_f{x.fold}_r{x.seed} K={x.K}：作り直した種類 {kind}（csv {x.resolve_kind}），"
          f"許す本数 {1e-3*n:.1f}，保存との差の本数換算 {x.abs_diff*n/100:.2f} ≤ {max(2,1e-3*n):.1f}")
    a = np.load(U / f"{x['T']}_f{x.fold}_r{x.seed}.npz")
    k = f"K{x.K}__R0_tight"
    nd = int((a[k] != a[f"K{x.K}__R0"]).sum())
    nk = a[f"nk__K{x.K}"].astype(bool); y = a["y"].astype(int)

    def uar(p):
        return 100 * np.mean([(p[nk][y[nk] == c] == c).mean() for c in range(4) if (y[nk] == c).any()])
    du = uar(a[k].astype(int)) - uar(a[f"K{x.K}__R0"].astype(int))
    print(f"    npz の tight と単位の解の違い {nd} 本（csv の 1e-08 列 {d8}），UAR の差の作り直しと csv の差 {abs(du - x.uar_diff_tight_vs_unit):.1e}")
tight_keys = sum(1 for p in npz for kk in np.load(p).files if kk.endswith("R0_tight"))
print("npz の R0_tight の鍵の数", tight_keys, "（外れた行の数", len(over), "）")
print("コーパスごとの通した行（上限 max(4, 0.25×100) = 25）", r0.groupby("T").tol_resolved.apply(lambda s: int(s.astype(str).eq("True").sum())).to_dict())
print("resolve_kind の数", over.resolve_kind.value_counts().to_dict())

print("== 5. 実験 1 の照合と線形の恒等式の記録")
for c in ("dai1_none_abs_diff", "dai1_prop_p2_abs_diff", "dai1_prop_p3_abs_diff", "cat_max_abs_diff"):
    print(f"  {c} 最大 {r0[c].max():.2e}")
print("  min(p2,p3) の最大", f"{np.minimum(r0.dai1_prop_p2_abs_diff, r0.dai1_prop_p3_abs_diff).max():.2e}")
print("  n_linear_diff_nontie_unclamped 最大", int(r0.n_linear_diff_nontie_unclamped.max()), "n_tie 最大", int(r0.n_tie.max()), "n_clamped 最大", int(r0.n_clamped.max()))

print("== 6. check_saved")
cs = sorted(R.glob("check_saved_*.csv")); print([p.name for p in cs]); csd = pd.read_csv(cs[0])
print("行", len(csd), "重複", int(csd.duplicated(["T", "fold", "seed", "K"]).sum()), "単位の R0 の行と一致",
      bool(csd.sort_values(["T", "fold", "seed", "K"])[["abs_diff"]].reset_index(drop=True).equals(r0.sort_values(["T", "fold", "seed", "K"])[["abs_diff"]].reset_index(drop=True))))
print("check_fail のファイル", [p.name for p in R.glob("check_fail_*")])

print("== 7. 要約の表")
C = pd.read_csv(R / "summary" / "conditions.csv")
print("conditions 行", len(C), "| (T,K,subset) ごとの量の数", C.groupby(["T", "K", "subset"]).size().to_dict())
print("  識別子の重複", int(C.duplicated(["T", "K", "subset", "quantity"]).sum()), "| 列", C.columns.tolist())
num = C.select_dtypes("number")
print("  NaN（列ごと）", {c: int(num[c].isna().sum()) for c in num.columns if num[c].isna().any()}, "Inf", int(np.isinf(num.to_numpy()).sum()))
print("  lo ≤ effect ≤ hi を破る行", int(((C.lo > C.effect + 1e-9) | (C.effect > C.hi + 1e-9)).sum()), "| commit", C.commit.unique().tolist(), C.dirty.unique().tolist())
print("  量の名前", sorted(C.quantity.unique()))
A = pd.read_csv(R / "summary" / "assumptions.csv")
print("assumptions 行", len(A), A.groupby("kind").size().to_dict())
an = A.drop(columns=["commit", "dirty"]).select_dtypes("number")
print("  数の列の Inf", int(np.isinf(an.to_numpy()).sum()), "| commit", A.commit.unique().tolist())
# 種類ごとに，その種類の行で使う列に NaN が無いか
for kd, x in A.groupby("kind"):
    used = [c for c in x.columns if c not in ("commit", "dirty") and x[c].notna().any()]
    nn = int(x[used].isna().sum().sum())
    if nn:
        print("  種類", kd, "の使う列の NaN", nn)
cv = json.load(open(R / "summary" / "convergence.json"))
print("convergence", {k: v for k, v in cv.items() if k not in ("C_chosen_counts",)})
print("  C_chosen 合計", sum(cv["C_chosen_counts"].values()), "（見込み 200 単位 × 2 K × 2 組 = 800）")

print("== 8. uar_pseudo と実験 14 の照合（要約の段の照合の作り直し）")
c14 = pd.read_csv(ROOT / "results" / "dai14" / "summary" / "conditions.csv")
mx = 0.0; nref = 0
for (T, K), x in C[(C.subset == "all") & (C.quantity == "uar_pseudo")].groupby(["T", "K"]):
    ref = c14[(c14["T"] == T) & (c14.K == K) & (c14.quantity == "uar_pseudo")].effect.to_numpy()
    nref += len(ref); mx = max(mx, float(np.max(np.abs(ref - x.effect.iloc[0]))))
print("実験 14 の行", nref, "最大差", f"{mx:.1e}")
