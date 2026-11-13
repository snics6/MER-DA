"""台本 v1 の生徒の格子の選び方（stage_summary の 791〜802 行と同じ式）と引き戻しの強さの選び方（stage_tether）に人工の表を渡し，
CREMA-D の融合（AV）の扱いを確かめる．"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/snics/MER-DA/DAI/checks/DAI-18/review_ml_sim")
import v1_copy as V
rows = []
for T, c, f, r in V.all_units():
    for K in V.KS:
        for kind in V.kinds_of(c):
            for t_ in V.TEACH:
                for gi in range(len(V.SGRID)):
                    rows.append(dict(T=T, c=c, fold=f, seed=r, K=K, student=kind, teacher=t_, grid=gi, uar=50.0))
sc = pd.DataFrame(rows)
gs = sc[sc.grid.notna() & (sc.grid >= 0)].groupby(["T", "K", "student", "teacher", "grid"]).uar.mean()
sch = {}
for c in V.CORP6:
    for K in V.KS:
        for kind in V.kinds_of(c):
            for t_ in V.TEACH:
                vals = [gs.get((c, K, kind, t_, gi), np.nan) for gi in range(len(V.SGRID))]
                if not np.isfinite(vals).any():
                    continue
                sch[(c, K, kind, t_)] = gi
miss = sorted({(c, kind) for c in V.CORP6 for kind in V.kinds_of(c) if not any((c, K, kind, t_) in sch for K in V.KS for t_ in V.TEACH)})
print("格子の点が選ばれない（生徒の結果が出ない）組：", miss)
# 引き戻しの強さ：T ごとに，種類 AV の行を出す仮のターゲット
src = {}
for T, c, f, r in V.all_units(folds=[0], seeds=[0]):
    for kind in V.kinds_of(c):
        src.setdefault((T, kind), set()).add(c)
print("AV の引き戻しの強さを選ぶ仮のターゲット：", {T: sorted(src.get((T, "AV"), [])) for T in V.CORP6})
print("AVT の引き戻しの強さを選ぶ仮のターゲット：", {T: sorted(src.get((T, "AVT"), [])) for T in V.CORP6})
