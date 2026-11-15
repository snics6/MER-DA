"""gate-code（DAI-5）：B の表で (a) に分けた解析を，試走の 1 単位の保存から学び直さずに計算できるかを確かめる。値は出さない（形と有限かだけ）。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1_stats as ST, dai5 as D5, dai_data as DD
out, T, c, f, r = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
name = f"{T}_{c}_f{f}_r{r}"
z = np.load(out / "pseudo/units" / f"{name}_p5.npz")
y_of, cl_of = ST.label_maps(c)
sid = z["sample_test"].astype(str); y = y_of(sid); cl = z["cluster_test"].astype(str)
p5 = D5.P5(out / "pseudo/units" / name)
pr = p5.pred("A", 25, 5.0).astype(int)
rec = [float((pr[y == k] == k).mean()) for k in range(4)]                               # 感情ごとの再現率
cm = pd.crosstab(y, pr).shape                                                           # 混同行列
spk = pd.DataFrame(dict(cl=cl, hit=pr == y)).groupby("cl").hit.mean()                   # 話者（塊）ごと
names = list(z["p5_src_names"].astype(str)); tp = z["p5_src_test_pred"].astype(int)
tu = np.array([100 * np.mean([(tp[j][y == k] == k).mean() for k in range(4)]) for j in range(len(names))])
rho = pd.Series(z["p5_K25_scores"]).corr(pd.Series(tu), method="spearman")              # 採点とテストの順位相関
# Q のテストでの判定（学び直さない）：段 source の W・b と特徴と保存した重みから
df, feats = DD.load_corpus(c, DD.ENC1)
pool = np.flatnonzero(DD.pool_mask(df, f)); test = np.flatnonzero(df["fold"].to_numpy() == f)
L = 0
w = z["p5_K25_w_b5.0"]
for j, nm in enumerate(names):
    s_, m = nm.rsplit("_", 1)
    X = DD.kind_matrix(feats, m).astype(float); mu, sd = X[pool].mean(0), np.maximum(X[pool].std(0, ddof=1), 1e-6)
    q = np.load(ROOT / "results/dai1/source" / f"{s_}_{m}.npz")
    Z = ((X[test] - mu) / sd) @ q["W"].T + q["b"]; Z = Z - Z.max(1, keepdims=True); L = L + w[j] * (Z - np.log(np.exp(Z).sum(1, keepdims=True)))
qt = L.argmax(1)
srcmatch = True
for j, nm in enumerate(names):
    s_, m = nm.rsplit("_", 1)
    X = DD.kind_matrix(feats, m).astype(float); mu, sd = X[pool].mean(0), np.maximum(X[pool].std(0, ddof=1), 1e-6)
    q = np.load(ROOT / "results/dai1/source" / f"{s_}_{m}.npz")
    srcmatch &= bool(((((X[test] - mu) / sd) @ q["W"].T + q["b"]).argmax(1) == tp[j]).all())
# 実験 1 の 1 つのソース × モダリティだけを作り手にした生徒（tt:src）の判定
up = ST.UnitPreds(D5.IN_DIR / "pseudo/units" / name)
tts = [it for it in [json.loads(x) for x in up.p3["p3_A_K25_items"]] if it.get("method") == "tt" and str(it.get("teacher", "")).startswith("src_")]
# 確率（較正・AUC 用）：保存した W・b から
Wb = z["p5_A_K25_W"].shape
print("per-emotion recall finite", np.isfinite(rec).all(), "confusion shape", cm, "clusters", len(spk),
      "spearman finite", bool(np.isfinite(rho)), "Q test pred recomputed n", len(qt), "src test pred recompute matches saved", srcmatch,
      "dai1 tt:src teachers for A student", [t["teacher"] for t in tts], "W shape", Wb)
