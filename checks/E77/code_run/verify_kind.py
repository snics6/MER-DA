"""E77 関門 1 の検算：種類の試走（audio_wavlm RAVDESS 反復 0）の出力を点検し，E59 の保存済みの値と突き合わせる。"""
import sys
import numpy as np
import pandas as pd
d = sys.argv[1]; T = sys.argv[2]
a = pd.read_csv(f"{d}/{T}.csv", dtype={"K": str, "meff": str, "param": str})
print("行数", len(a), "列", list(a.columns))
print(a.groupby(["method"]).size().to_string())
print("K ごとの行数", a.groupby("K").size().to_dict())
P = np.load(f"{d}/{T}_r0_probs.npy"); te = np.load(f"{d}/{T}_r0_te.npy"); rid = np.load(f"{d}/{T}_r0_rowids.npy")
print("probs", P.shape, P.dtype, "te", te.shape, "非有限", int((~np.isfinite(P.astype(np.float32))).sum()))
s = P.astype(np.float64).sum(2); print("確率の和の幅", s.min(), s.max())
print("確率 0 の要素の数（float16 で下に溢れた）", int((P == 0).sum()), "/", P.size)
# probs から UAR を出し直して csv と比べる
sm = pd.read_csv(f"{d}/{T}_samples.csv")
meta = pd.read_csv("data/metadata/" + "manifest_" + "x.csv") if False else None
y = None
import sys as _s; _s.path.insert(0, "measure"); _s.path.insert(0, "exp")
from data import load_any
df, _ = load_any(T, "audio_wavlm")
y = df["label_4"].to_numpy()[te]
assert (df["sample_id"].reset_index(drop=True) == sm["sample_id"]).all()
def uar(y, p):
    return np.mean([np.mean(p[y == c] == c) for c in range(4) if (y == c).any()])
r0 = a[a.rep == 0].sort_values("row_id")
u2 = np.array([uar(y, P[i].astype(np.float32).argmax(1)) for i in r0.row_id])
print("probs から出し直した UAR と csv の差の最大", np.abs(u2 - r0.uar.to_numpy()).max())
# 同値の列
print("UAR が全行で同じ手法：", [m for m, g in a.groupby("method") if g.uar.nunique() == 1])
# 退化：t3a・massbal・saerens が none と同じか
r0 = r0.assign(u2=u2, diff=np.abs(u2 - r0.uar.to_numpy()))
bad = r0[r0["diff"] > 1e-12]
print("不一致の行", len(bad), "/", len(r0)); print(bad.groupby("method").size().to_dict())
print(bad.sort_values("diff", ascending=False).head(8)[["method", "param", "meff", "K", "uar", "u2"]].to_string())
# 同点の数：float16 の確率で最大が 2 つ以上
Pf = P.astype(np.float32)
mx = Pf.max(2, keepdims=True); ties = ((Pf == mx).sum(2) > 1)
print("float16 で最大が同点になった発話の数（行ごとの合計）", int(ties.sum()))
print("同点のある行の手法", r0.assign(t=ties[r0.row_id].sum(1)).query("t > 0").groupby("method").t.sum().to_dict())
# T3A の確率の幅
t3 = r0[r0.method == "t3a"].row_id.to_numpy()
print("t3a の確率の最大−最小（発話ごと）の中央値", float(np.median(Pf[t3].max(2) - Pf[t3].min(2))))
