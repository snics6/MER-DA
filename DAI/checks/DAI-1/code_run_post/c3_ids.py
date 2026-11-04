"""各単位の sample_id を分割の表と突き合わせる：テスト＝分割 f，候補＝分割 f 以外（MELD はテストの対話を除く），K 本は候補の中で
感情ごとに K 本・入れ子，テストと候補で塊が重ならない，K 本は T に依らず同じ（設計 1 節）。"""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/home/snics/MER-DA"); U = ROOT / "results/dai1/pseudo/units"
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
sp = {c: pd.read_csv(ROOT / f"DAI/splits/{c}.csv", keep_default_na=False).set_index("sample_id") for c in C}
bad = []; ksets = {}
for T in C:
    for c in C:
        if c == T: continue
        s = sp[c]
        for f in range(5):
            test_exp = set(s.index[s.fold == f])
            pool_exp = set(s.index[s.fold != f])
            if c == "MELD":
                dl = set(s.loc[list(test_exp), "dialogue"])
                pool_exp = {x for x in pool_exp if s.at[x, "dialogue"] not in dl}
            for r in (0, 1):
                z = np.load(U / f"{T}_{c}_f{f}_r{r}_p1.npz")
                te, po = set(z["sample_test"]), set(z["sample_pool"])
                if te != test_exp: bad.append((T, c, f, r, "test"))
                if po != pool_exp: bad.append((T, c, f, r, "pool"))
                if len(z["sample_test"]) != len(te) or len(z["sample_pool"]) != len(po): bad.append((T, c, f, r, "dup"))
                if set(s.loc[list(te), "cluster"]) & set(s.loc[list(po), "cluster"]): bad.append((T, c, f, r, "cluster overlap"))
                prev = set()
                for K in (1, 5, 10, 25):
                    k = z[f"sample_K{K}"]
                    lab = s.loc[list(k), "label_4"].value_counts()
                    if not (set(k) <= po) or len(set(k)) != 4 * K or not (lab == K).all() or len(lab) != 4: bad.append((T, c, f, r, f"K{K}"))
                    if not prev <= set(k): bad.append((T, c, f, r, f"nest{K}"))
                    prev = set(k)
                    ksets.setdefault((c, f, r, K), set()).add(tuple(sorted(k)))
                # 段 2・3 の npz の引き戻しと指数が tether_w.json と一致するか
print("id problems:", len(bad), bad[:10])
print("K draws differing across T:", sum(len(v) != 1 for v in ksets.values()), "of", len(ksets))
# seeds differ?
same_seed = sum(ksets[(c, f, 0, 5)] == ksets[(c, f, 1, 5)] for c in C for f in range(5))
print("units where seed 0 and 1 give identical K=5 draws:", same_seed)
