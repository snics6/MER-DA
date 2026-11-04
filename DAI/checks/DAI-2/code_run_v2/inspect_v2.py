"""gate-code 2 回目の検算：解き直しの行（*_rep），dis_rep・dis_ctl・dis_ctl_rep，分布の量（R2），自己学習の巡ごとの保存（R9）を数える。
UAR は出さない。引数：units の置き場，単位の名前"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d0, unit = Path(sys.argv[1]), sys.argv[2]
p = d0 / f"{unit}_map.csv"
if p.exists():
    d = pd.read_csv(p, low_memory=False)
    z = np.load(str(p).replace(".csv", ".npz"))
    print("map：行", len(d), "手法", d.method.value_counts().to_dict(), "extra_tag", str(z["extra_tag"]))
    rep = d[d.method.str.endswith("_rep")]
    print("解き直しの行：生徒 × 手法", rep.groupby(["student", "method"]).size().unstack(fill_value=0).to_dict("index"))
    print("解き直しの束の大きさ", sorted(rep.chunk_size.unique()), " roles の種類", sorted(rep.roles.unique()))
    cols = ["student", "method", "K", "lr", "iters", "mu", "dis_rep", "dis_ctl", "dis_ctl_rep"]
    print(rep[cols].to_string(index=False))
    main = d[~d.method.str.endswith("_rep")]
    print("本番の行の dis_ctl：NaN の数（mapctl の行）", int(main.dis_ctl.isna().sum()), "/ mapctl の行", int((main.method == "mapctl").sum()))
    for c in ("mdiff_before", "cdiff_before", "dloss_before", "dacc_before", "mdiff_after", "cdiff_after", "dloss_after", "dacc_after", "sec_diag_chunk"):
        v = d[c]
        print(f"{c}: NaN {int(v.isna().sum())}，非有限 {int((~np.isfinite(v.dropna())).sum())}，範囲 {float(v.min()):.4g}〜{float(v.max()):.4g}")
    k0 = main[(main.method == "mapctl") & (main.K == 0)]
    print("mapctl K=0 の mdiff_after − mdiff_before の最大絶対", float((k0.mdiff_after - k0.mdiff_before).abs().max()),
          " cdiff の差", float((k0.cdiff_after - k0.cdiff_before).abs().max()))
    nd = main[main.method != "dann"]
    print("DANN 以外の判別器（学ばない）の dloss_after − dloss_before の最大絶対（A = I の mapctl K=0 だけ）",
          float((k0.dloss_after - k0.dloss_before).abs().max()))
    print("npz の鍵（maprep）", [k for k in z.files if k.startswith("maprep")][:8])
p = d0 / f"{unit}_xmuda.csv"
if p.exists():
    d = pd.read_csv(p, low_memory=False)
    rep = d[d.method == "xmuda_rep"]
    print("\nxmuda：行", len(d), "解き直しの行", len(rep), "roles", sorted(rep.roles.unique()), "束の大きさ", sorted(rep.chunk_size.unique()))
    print(rep[["student", "K", "lr", "iters", "kap", "mu", "dis_rep", "dis_ctl", "dis_ctl_rep"]].to_string(index=False))
    print("本番の行の dis_ctl の NaN", int(d[d.method == "xmuda"].dis_ctl.isna().sum()), "/", int((d.method == "xmuda").sum()))
p = d0 / f"{unit}_lin.npz"
if p.exists():
    z = np.load(p)
    ks = sorted(k for k in z.files if k.startswith("self_A_K5"))
    print("\nlin の自己学習の鍵（A，K=5）", ks, {k: z[k].shape for k in ks})
    print("3 巡のラベルの写しが同じか", bool(np.array_equal(z["self_A_K5_r3_pl"], z["self_A_K5_pl"])))
    print("1 巡と 3 巡の判定が全部同じ問の数", int(sum(np.array_equal(a, b) for a, b in zip(z["self_A_K5_r1_pred"], z["self_A_K5_pred"]))), "/ 12")
