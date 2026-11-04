"""gate-code の検算（R1・R2 の (a) の手順を実際に通す）：保存した W・b（map は W·A）と標準化の統計から，生の特徴でテストの点数を作り直し，
保存した判定と一致するかを見る．あわせて，テストの発話を塊（話者の代わり）に対応づけ，DANN・MME の既定の点と mapctl の判定の食い違いを出す。
UAR は出さない。引数：units の置き場，単位の名前"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai_data as DD  # noqa: E402

d0, unit = Path(sys.argv[1]), sys.argv[2]
c = unit.split("_")[1]
df, feats = DD.load_corpus(c)
pos = pd.Series(np.arange(len(df)), index=df["sample_id"])
zl = np.load(d0 / f"{unit}_lin.npz")
te = pos.loc[zl["sample_test"]].to_numpy()
for part, key, s in (("lin", "lin", "A"), ("lin", "nc", "AVT"), ("map", "map", "A"), ("map", "map", "V"), ("xmuda", "xm", "T")):
    z = np.load(d0 / f"{unit}_{part}.npz")
    X = DD.kind_matrix(feats, s)[te].astype(np.float64)
    Xs = (X - z[f"stat_{s}_mu"]) / z[f"stat_{s}_sd"]
    k = f"{key}_{s}_K5" if part == "lin" else f"{key}_{s}"
    W, b, pr = z[f"{k}_W"].astype(np.float64), z[f"{k}_b"].astype(np.float64), z[f"{k}_pred"]
    L = np.einsum("nd,pkd->pnk", Xs, W) + b[:, None, :]
    agree = (L.argmax(-1) == pr).mean(1)
    print(f"{part} {k}：問 {len(pr)}，作り直した判定と保存の判定の一致の最小 {agree.min():.4f}（float32 の W による差のみ）")
# 塊への対応づけ（話者を単位にした区間の手順）
sp = pd.read_csv(f"/home/snics/MER-DA/DAI/splits/{c}.csv").set_index("sample_id")
cl = sp.loc[zl["sample_test"], "cluster"]
print(f"テスト {len(cl)} 本，塊 {cl.nunique()} 個，塊ごとの本数の最小・最大 {cl.value_counts().min()}・{cl.value_counts().max()}")
# DANN・MME の既定の点と mapctl の食い違い（設計 2.3 節の量）
zm = np.load(d0 / f"{unit}_map.npz")
for s in ("A", "V", "T"):
    it = [json.loads(str(x)) for x in zm[f"map_{s}_items"]]
    pr = zm[f"map_{s}_pred"]

    def find(**kw):
        return [j for j, x in enumerate(it) if all(x.get(a) == v for a, v in kw.items())]
    for K in (0, 5, 25):
        jc = find(method="mapctl", lr=1e-3, iters=1000, mu=1.0, K=K)[0]
        jd = find(method="dann", lr=1e-3, iters=1000, lam=0.1, mu=1.0, K=K)[0]
        jm = find(method="mme", lr=1e-3, iters=1000, lam=0.1, mu=1.0, sw=0.0, K=K)[0]
        print(f"{s} K={K}：既定の点の DANN と mapctl の食い違い {1 - (pr[jd] == pr[jc]).mean():.3f}，MME と mapctl {1 - (pr[jm] == pr[jc]).mean():.3f}"
              f"（束 {it[jd]['chunk']}・{it[jm]['chunk']}・{it[jc]['chunk']}，大きさ {it[jd]['chunk_size']}・{it[jm]['chunk_size']}・{it[jc]['chunk_size']}）")
