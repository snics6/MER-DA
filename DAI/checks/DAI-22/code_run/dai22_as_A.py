"""gate-code の検算：dai22 の run_unit を生徒 A（声）に替えて流し，実験 14 の保存（同じ生徒・同じ目標 prop_one）と
予測の違いの数を比べる（配線の確かめ）．あわせて融合の K 本だけの分類器のテストの UAR を作り直し，実験 1 の保存との差だけを出す．
量の値そのものは出さない．"""
import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
from pathlib import Path
import numpy as np, pandas as pd, torch
import dai22 as D22, dai3, dai1, dai_data as DD, dai_solver as S

dai1.ENC["enc"] = DD.ENC1
T, f, r, stu, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
D22.STU = stu
D22.RUN["commit"], D22.RUN["dirty"] = "gatecheck", True
fz1, _ = dai3.load_fz1(D22.OUT1)
(out / "units").mkdir(parents=True, exist_ok=True)
rp = pd.read_csv(D22.REPRO_SAVED, low_memory=False)
rp = rp[(rp.student == stu) & (rp.subset == "all") & rp.K.isin(D22.KS)]
df = D22.run_unit(T, f, r, fz1[T], out, rp)
a22 = np.load(out / "units" / f"{T}_f{f}_r{r}.npz")
a14 = np.load(D22.OUT14 / "units" / f"{T}_f{f}_r{r}.npz")
for K in D22.KS:
    m = "prop_one"
    pairs = [(f"none__K{K}", f"none__K{K}__{stu}"), (f"tgt__K{K}", f"tgt__K{K}__{m}__true"), (f"nk__K{K}", f"nk__K{K}"),
             (f"K{K}__R0", f"K{K}__main__R0__{stu}__{m}__true"), (f"K{K}__R1", f"K{K}__main__R1__{stu}__{m}__true")]
    for k22, k14 in pairs:
        print(f"K={K} {k22} vs dai14 {k14}: n_diff={(a22[k22].astype(int) != a14[k14].astype(int)).sum()} / {len(a22[k22])}")
    if K == 5:
        for ro in ("R0", "R1"):
            p22 = a22[f"prob__K{K}__{ro}"]; p14 = a14[f"prob__K{K}__main__{ro}__{stu}__{m}__true"]
            print(f"K={K} prob {ro}: max_abs_diff={np.abs(p22 - p14).max():.2e}")
    c22 = a22[f"inner__K{K}"]; c14 = a14[f"inner__main__{stu}__K{K}__{m}"]
    print(f"K={K} inner curve: shape22={c22.shape} shape14={c14.shape} max_abs_diff={np.abs(c22 - c14).max():.2e}")
print("abs_diff vs dai3 saved (R0):", df[df.readout == "R0"].abs_diff.tolist())
print("C0/C1 R1:", df[df.readout == "R1"][["K", "C0", "C1"]].to_dict("records"))
