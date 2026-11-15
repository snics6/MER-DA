import pandas as pd, numpy as np, glob, os, sys
d = sys.argv[1]
keys = {
 "e76a_1_pairs": ["kind","target","source","K"],
 "e76a_1_kind": ["kind","K"],
 "e76a_1_targets": ["kind","K","target"],
 "e76a_1_kstar": ["kind","target","source","interval","definition"],
 "e76a_2_gender": ["kind","target","K"],
 "e76a_2_subsets": ["kind","target","subset","unit","source","K"],
 "e76a_3_curves": ["kind","target","rule","K"],
 "e76a_3_pairs": ["kind","rule","target","source","K"],
 "e76a_3_gain": ["kind","rule","K"],
 "e76a_3_q_targets": ["kind","target","rule","D_version","fit"],
 "e76a_3_dT": ["kind","target","rule","K","pair"],
 "e76a_3_n_pairs": ["kind","rule","target","source","K"],
 "e76a_4_kind": ["kind","K","quantity"],
 "e76a_4_pairs": ["kind","target","source","K"],
 "e76a_4_do": ["kind","target","source","method","pair"],
 "e76a_4_joint": ["kind","target","source"],
 "e76a_5_hands": ["kind","K","incl_same_corpus","hand"],
 "e76a_5_compare": ["kind","K","subject","other"],
 "e76a_5_alpha": ["kind","target","source","K"],
}
for f in sorted(glob.glob(os.path.join(d,"*.csv"))):
    n = os.path.basename(f)[:-4]
    x = pd.read_csv(f, comment="#", dtype={"K":str,"source":str})
    nan = x.isna().sum()
    nan = nan[nan>0]
    num = x.select_dtypes(include=[np.number])
    inf = int(np.isinf(num.to_numpy(dtype=float)).sum()) if len(num.columns) else 0
    dup = ""
    if n in keys:
        k = [c for c in keys[n] if c in x.columns]
        dup = int(x.duplicated(k).sum())
    print(f"{n}: rows={len(x)} dupkeys={dup} inf={inf} nan_cols={dict(nan)}")
