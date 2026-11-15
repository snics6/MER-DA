"""gate-code の検算：B の (a) に分けた解析が，試走の出力から学び直さずに作れることを確かめる．値は出さず，形と件数だけを出す．
1. 感情ごとの再現率（J_none，J_prop，R0，R1．K 本を除いた候補）
2. 性別（IEMOCAP・AFEW）・共有の対話（MELD）・記号の有無を，実験 14 の単位の npz から sample_id で引いて付ける
3. 出発点の判定（R_start）を学び直さずに作り直す（dai1.Unit の出発点の W・b を当てるだけ）"""
import sys, glob
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
from pathlib import Path
import numpy as np, pandas as pd
import dai3, dai1, dai_data as DD, dai_solver as S
out = Path(sys.argv[1])
rows = []
for p in sorted(glob.glob(str(out / "units" / "*.npz"))):
    a = dict(np.load(p))
    T, f, r = Path(p).stem.split("_"); f, r = int(f[1:]), int(r[1:])
    a14 = np.load(DD.ROOT / "results" / "dai14" / "units" / f"{T}_f{f}_r{r}.npz")
    same = bool((a14["sid"] == a["sid"]).all())
    for K in (5, 25):
        m = a[f"nk__K{K}"]
        for nm, key in (("none", f"none__K{K}"), ("pseudo", f"tgt__K{K}"), ("R0", f"K{K}__R0"), ("R1", f"K{K}__R1")):
            for e in range(4):
                sel = m & (a["y"] == e)
                rows.append(dict(T=T, f=f, r=r, K=K, q=nm, emo=e, n=int(sel.sum())))
    sex = a14["sex"]; nons = a14["nonshared"]; mk = a14["marker"]
    print(f"{T}_f{f}_r{r}: sid_same_as_dai14={same} sex_levels={len(set(sex))} nonshared_false={int((~nons).sum())} marker_true={int(mk.sum())}")
R = pd.DataFrame(rows)
print("per-emotion table rows:", len(R), "min n per (unit,K,emo):", int(R.n.min()))
# 出発点の判定
dai1.ENC["enc"] = DD.ENC1
fz1, _ = dai3.load_fz1(DD.ROOT / "results" / "dai1")
sv = dai3.Saved(DD.ROOT / "results" / "dai1", "eval", "AFEW", "AFEW", 0, 0, fz1["AFEW"])
st = S.logits(sv.u.Xp["AVT"], *sv.u.start["AVT"]).argmax(1).cpu().numpy()
print("start__AVT rebuilt without learning: n =", len(st), "classes present =", len(np.unique(st)))
