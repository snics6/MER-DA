"""E77 関門 1 の再検査（dc600b8）：書き出されたファイル，W・b の行の生徒，生の特徴の上の W・b から作り直した対数の確率，UAR の再現，
参照の行，diag・conv の中身，86b7418 の試走との一致を点検する。引数：出力の場所，名前（種類か世界），ターゲット，比べる前の版の csv（任意）。"""
import sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, "measure"); sys.path.insert(0, "exp")
from data import load_any
d, name, T = sys.argv[1:4]
old = sys.argv[4] if len(sys.argv) > 4 else None
a = pd.read_csv(f"{d}/{T}.csv", dtype={"K": str, "meff": str, "param": str})
print("ファイル", sorted(x.split("/")[-1] for x in glob.glob(f"{d}/*")))
print("行", len(a), "鍵の重複", int(a.duplicated(["rep", "student", "method", "param", "meff", "K"]).sum()))
print(a.groupby(["method", "student"]).size().unstack().fillna(0).astype(int).to_string())
L = np.load(f"{d}/{T}_r0_logp.npy"); pr = np.load(f"{d}/{T}_r0_pred.npy"); te = np.load(f"{d}/{T}_r0_te.npy")
df, X = load_any(T, name); y = df.label_4.to_numpy()[te]; Xte = X[te].astype(np.float64)
uar = lambda y, p: np.mean([np.mean(p[y == c] == c) for c in range(4) if (y == c).any()])
u = np.array([uar(y, pr[i]) for i in a.row_id])
print("logp", L.shape, L.dtype, "非有限", int((~np.isfinite(L)).sum()), "pred と logp の argmax の食い違い", int((L.argmax(2) != pr).sum()),
      "pred の UAR と csv の差の最大", float(np.abs(u - a.uar).max()))
from multimodal_distill import blocks, WORLDS
blk = {m: (s, e) for m, s, e in blocks(name)} if name in WORLDS else {}
tot = 0
for f in sorted(glob.glob(f"{d}/{T}_r0_wrows_*.npy")):
    st = f.split("_wrows_")[1][:-4]; w = np.load(f); W = np.load(f.replace("_wrows_", "_W_")); b = np.load(f.replace("_wrows_", "_b_"))
    tot += len(w)
    stud = a.set_index("row_id").loc[w, "student"]
    Xs = Xte[:, blk[st][0]:blk[st][1]] if st in blk else Xte
    lg = Xs @ W.astype(np.float64).transpose(0, 2, 1) + b.astype(np.float64)[:, None, :]
    lp = lg - np.log(np.exp(lg - lg.max(2, keepdims=True)).sum(2, keepdims=True)) - lg.max(2, keepdims=True)
    print(f"W {st}：{W.shape} {W.dtype}，行の生徒が一致 {bool((stud == st).all())}，作り直した対数の確率との差の最大 {float(np.abs(lp - L[w]).max()):.1e}，"
          f"argmax の食い違い {int((lp.argmax(2) != pr[w]).sum())}，手法 {a.set_index('row_id').loc[w].method.value_counts().to_dict()}")
withW = set(a.method[a.row_id.isin(np.concatenate([np.load(f) for f in glob.glob(f'{d}/{T}_r0_wrows_*.npy')]))])
print("W を持つ行", tot, "／ W を持たない手法", sorted(set(a.method) - withW))
dg = pd.read_csv(f"{d}/{T}_diag.csv"); cv = pd.read_csv(f"{d}/{T}_conv.csv")
print("diag", len(dg), dg.groupby(["method"]).size().to_dict(), "列", list(dg.columns))
print(dg.groupby("method")[["n_pseudo", "acc_pseudo"]].describe().loc[:, (slice(None), ["min", "max"])].round(3).to_string())
print("conv", len(cv), "列", list(cv.columns)); print(cv.groupby("what").agg(n=("sec", "size"), sec=("sec", "sum"), it=("n_iter_max", "max"), g=("grad_max", "max")).to_string())
if old:
    o = pd.read_csv(old, dtype={"K": str, "meff": str, "param": str})
    m = a.merge(o, on=["student", "method", "param", "meff", "K"], suffixes=("", "_o"))
    print("前の版と同じ鍵の行", len(m), "UAR の差の最大（t3a を除く）", float((m.uar - m.uar_o).abs()[~m.method.str.startswith("t3a")].max()))
