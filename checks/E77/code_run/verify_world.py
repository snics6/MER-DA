"""E77 関門 1 の検算：世界の試走（反復 0）の出力を E61 の保存済みの値と突き合わせ，出力の形・NaN・W の行の対応・float16 の argmax を点検する。"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "measure"); sys.path.insert(0, "exp")
from data import load_any
d, world, T = sys.argv[1:4]
a = pd.read_csv(f"{d}/{T}.csv", dtype={"K": str, "meff": str, "param": str})
print("行数", len(a)); print(a.groupby(["student", "method"]).size().unstack(0).fillna(0).astype(int).to_string())
e = pd.read_csv(f"results/baselines_{world}/{T}.csv", dtype={"K": str, "meff": str, "param": str}).query("rep == 0")
e = e.assign(method=e.method.replace({"oursx": "dist"}))
m = a.merge(e, on=["student", "method", "param", "meff", "K"], suffixes=("", "_e61"))
print("E61 の反復 0 の行", len(e), "突き合わせた行", len(m), "差の最大", float((m.uar - m.uar_e61).abs().max()))
print("突き合わせた手法", sorted(set(m.method)), "突き合わない E61 の手法", sorted(set(e.method) - set(m.method)))
P = np.load(f"{d}/{T}_r0_probs.npy"); te = np.load(f"{d}/{T}_r0_te.npy")
print("probs", P.shape, "非有限", int((~np.isfinite(P.astype(np.float32))).sum()), "0 に溢れた要素", int((P == 0).sum()), "/", P.size)
df, _ = load_any(T, world); y = df["label_4"].to_numpy()[te]
def uar(y, p): return np.mean([np.mean(p[y == c] == c) for c in range(4) if (y == c).any()])
r0 = a.sort_values("row_id"); u2 = np.array([uar(y, P[i].astype(np.float32).argmax(1)) for i in r0.row_id])
bad = r0[np.abs(u2 - r0.uar.to_numpy()) > 1e-12]
print("float16 の確率から出し直した UAR が csv と食い違う行", len(bad), "/", len(r0), "最大の差", float(np.abs(u2 - r0.uar.to_numpy()).max()))
print(bad.groupby(["student", "method"]).size().to_dict())
import glob
for f in sorted(glob.glob(f"{d}/{T}_r0_wrows_*.npy")):
    st = f.split("_wrows_")[1][:-4]; w = np.load(f); W = np.load(f.replace("_wrows_", "_W_"))
    ok = (a.set_index("row_id").loc[w, "student"] == st).all()
    print("W の行", st, W.shape, W.dtype, "行の生徒が一致", bool(ok), "W の行の手法", a.set_index("row_id").loc[w].method.value_counts().to_dict())
print("UAR が全行で同じ (生徒, 手法)", [k for k, g in a.groupby(["student", "method"]) if g.uar.nunique() == 1 and len(g) > 1])
print("UAR の範囲", a.uar.min(), a.uar.max())
c = pd.read_csv(f"{d}/{T}_conv.csv"); print(c.to_string())
