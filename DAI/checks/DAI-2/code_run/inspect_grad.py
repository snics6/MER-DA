"""gate-code の検算：map・xmuda・coral の 1 単位の csv・npz の形，有限，診断の量（UAR の値は出さない）。引数：units の置き場，単位の名前（例 MOSEI_AFEW_f0_r0）"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d0, unit = Path(sys.argv[1]), sys.argv[2]
for part in ("map", "xmuda", "coral"):
    p = d0 / f"{unit}_{part}.csv"
    if not p.exists():
        print(part, "無し")
        continue
    d = pd.read_csv(p, low_memory=False)
    z = np.load(str(p).replace(".csv", ".npz"))
    print(f"\n==== {part}：行 {len(d)}，列 {len(d.columns)}，commit {str(z['commit'])[:7]} dirty {str(z['dirty'])}，sec {float(d.sec.iloc[0]):.0f}")
    print(d.groupby(["method", "student"]).size().unstack(fill_value=0))
    main = d[~d.method.str.startswith("_")]
    print("uar の NaN", int(main.uar.isna().sum()), "0〜100 の外", int(((main.uar < 0) | (main.uar > 100)).sum()), "finite 偽", int((d.finite.astype(str) == "False").sum()))
    bad = [k for k in z.files if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all()]
    print("非有限の npz の配列", bad)
    if part in ("map", "xmuda"):
        for c in ("anorm", "wa_dist", "ent_first", "ent_last", "dom_first", "dom_last", "kl_first", "kl_last", "S", "chunk_size"):
            if c in d:
                g = d.groupby("method")[c]
                print(c, {m: (round(float(v.min()), 3), round(float(v.max()), 3), int(v.isna().sum())) for m, v in g})
        print("束の大きさ", sorted(d.chunk_size.unique()), "束の数", d.chunk.nunique())
        print("K ごとの行", d.groupby(["method", "K"]).size().unstack(fill_value=0).to_dict("index"))
        if part == "map":
            for s in sorted(d.student.unique()):
                it = [json.loads(str(x)) for x in z[f"map_{s}_items"]]
                pr = z[f"map_{s}_pred"]
                k0 = [j for j, x in enumerate(it) if x["method"] == "mapctl" and x["K"] == 0]
                print(f"  {s}：問 {len(it)}，W {z[f'map_{s}_W'].shape}，mapctl K=0 の問 {len(k0)}，その判定が全部同じか {all((pr[j] == pr[k0[0]]).all() for j in k0)}")
                lin = d0 / f"{unit}_lin.npz"
                if lin.exists():
                    zl = np.load(lin)
                    print(f"     mapctl K=0 と none K=0 の判定の一致 {float((pr[k0[0]] == zl[f'none_{s}_K0_pred']).mean()):.4f}")
    if part == "coral":
        print(d[d.method == "_coral_check"][["student", "src", "rel_diff"]].to_string(index=False))
        cs = d[d.method == "_coral_src"]
        print("_coral_src：grad の最大", float(cs.grad.max()), "shift_rel の範囲", float(cs.shift_rel.min()), float(cs.shift_rel.max()),
              "縮小の範囲 src", float(cs.shrink_src.min()), float(cs.shrink_src.max()), "tgt", float(cs.shrink_tgt.min()), float(cs.shrink_tgt.max()))
        cm = d[d.method.isin(["coral", "coral_tether"])]
        print("coral の行（K × r）", cm.groupby(["method", "K"]).size().to_dict(), "grad の最大", float(cm.grad.max()))
