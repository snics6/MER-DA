"""E78 関門 1 の再検査：出力の点検（W・b から作り直した対数の確率，coral_lin_check，cotzero と none の差，種の行）。引数：出力の場所，名前，ターゲット"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "measure")); sys.path.insert(0, str(ROOT / "exp"))
from data import load_any  # noqa: E402
from multimodal_distill import blocks, WORLDS  # noqa: E402
d, name, T = sys.argv[1:4]
a = pd.read_csv(f"{d}/{T}.csv", dtype={"K": str, "meff": str, "param": str})
print("ファイル", sorted(x.split("/")[-1] for x in glob.glob(f"{d}/*")))
print("行", len(a), "鍵の重複", int(a.duplicated(["rep", "student", "method", "param", "meff", "K"]).sum()), "UAR の NaN", int(a.uar.isna().sum()),
      "コミット", sorted(set(a.commit.astype(str))))
print(a.groupby(["method", "student"]).size().unstack().fillna(0).astype(int).to_string())
L = np.load(f"{d}/{T}_r0_logp.npy"); pr = np.load(f"{d}/{T}_r0_pred.npy"); te = np.load(f"{d}/{T}_r0_te.npy")
df, X = load_any(T, name); Xte = X[te].astype(np.float64)
print("logp", L.shape, L.dtype, "非有限", int((~np.isfinite(L)).sum()), "pred と argmax の食い違い", int((L.argmax(2) != pr).sum()))
blk = {m: (s, e) for m, s, e in blocks(name)} if name in WORLDS else {}
withw = []
for f in sorted(glob.glob(f"{d}/{T}_r0_wrows_*.npy")):
    st = f.split("_wrows_")[1][:-4]; w = np.load(f); W = np.load(f.replace("_wrows_", "_W_")); b = np.load(f.replace("_wrows_", "_b_"))
    withw.append(w)
    base = st.split("@")[0]
    stud = a.set_index("row_id").loc[w, "student"]
    Xs = Xte[:, blk[base][0]:blk[base][1]] if base in blk else Xte
    lg = Xs @ W.astype(np.float64).transpose(0, 2, 1) + b.astype(np.float64)[:, None, :]
    mx = lg.max(2, keepdims=True)
    lp = lg - mx - np.log(np.exp(lg - mx).sum(2, keepdims=True))
    print(f"W {st}：{W.shape} {W.dtype}，行の生徒が一致 {bool((stud == base).all())}，作り直した対数の確率との差の最大 {float(np.abs(lp - L[w]).max()):.1e}，"
          f"argmax の食い違い {int((lp.argmax(2) != pr[w]).sum())}，手法 {a.set_index('row_id').loc[w].method.value_counts().to_dict()}")
ww = np.concatenate(withw) if withw else np.array([], int)
print("W を持つ行", len(ww), "/", len(a), "／W を持たない手法", sorted(set(a.method) - set(a.method[a.row_id.isin(ww)])))
# cotzero と none の差
if (a.method == "cotzero").any():
    for st in sorted(a[a.method == "cotzero"].student.unique()):
        for K in ["0", "1", "5", "10", "25"]:
            rn = a[(a.student == st) & (a.method == "none") & (a.K == K)].row_id.iloc[0]
            rz = a[(a.student == st) & (a.method == "cotzero") & (a.K == K)]
            dl = [float(np.abs(L[i] - L[rn]).max()) for i in rz.row_id]
            dp = [int((pr[i] != pr[rn]).sum()) for i in rz.row_id]
            byl = {lr: (max(float(np.abs(L[i] - L[rn]).max()) for i in g.row_id), max(int((pr[i] != pr[rn]).sum()) for i in g.row_id))
                   for lr, g in rz.groupby(rz.param.str.extract(r"lr([\d.e-]+)s")[0])}
            print(f"cotzero {st} K={K}：none との対数の確率の差の最大と予測の食い違いの最大（学習率ごと）{ {k: (round(v[0], 4), v[1]) for k, v in byl.items()} } / {len(te)}")
# coralctl の K=0 と none の K=0
if (a.method == "coralctl").any():
    for st in sorted(a[a.method == "coralctl"].student.unique()):
        rn = a[(a.student == st) & (a.method == "none") & (a.K == "0")].row_id.iloc[0]
        rc = a[(a.student == st) & (a.method == "coralctl") & (a.K == "0")].row_id
        print(f"coralctl {st} K=0：none の K=0 との対数の確率の差の最大 {max(float(np.abs(L[i] - L[rn]).max()) for i in rc):.2e}")
    g = a[a.method.isin(["coral", "coralctl"])]
    print("CORAL の候補の数（生徒 × r）:", g.assign(r=g.param.str.extract(r"^(r\d+)")[0].fillna("ctl")).groupby(["student", "r"]).param.nunique().to_dict())
for f in ("conv", "diag"):
    p = Path(f"{d}/{T}_{f}.csv")
    if p.exists() and p.stat().st_size > 1:
        x = pd.read_csv(p)
        print(f, len(x), "列", list(x.columns))
        if f == "conv":
            print(x.groupby(["student", "what"]).agg(n=("sec", "size"), sec=("sec", "sum")).round(1).to_string() if "sec" in x else "")
            if "rel_diff" in x:
                print("coral_lin_check の rel_diff:", x.rel_diff.dropna().round(12).tolist())
        if f == "diag":
            num = x.select_dtypes("number")
            print("diag の方法ごとの行", x.method.value_counts().to_dict())
            for c in ("anorm", "dom_first", "dom_last", "ent_first", "ent_last", "kl_first", "kl_last"):
                if c in x:
                    print(f"  {c}: 値のある行 {int(x[c].notna().sum())}，非有限 {int((~np.isfinite(x[c].dropna())).sum())}")
s = Path(f"{d}/{T}_settings.json")
print("settings.json", s.exists())
t = Path(f"{d}/{T}_r0_teachers.npz")
if t.exists():
    z = np.load(t); print("teachers.npz の鍵", sorted(z.files))
