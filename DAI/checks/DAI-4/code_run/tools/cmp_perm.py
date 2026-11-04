import json, numpy as np
a = np.load("out/g1/pseudo/units/MELD_AFEW_f2_r1_p4.npz"); b = np.load("out/perm/pseudo/units/MELD_AFEW_f2_r1_p4.npz")
mw = mb = 0.0; npred = nm = 0; nitems = 0; same_pred = 0; tot_pred = 0
for k in b.files:
    if k.endswith("_items"):
        ia = [json.loads(x) for x in a[k]]; ib = [json.loads(x) for x in b[k]]
        idx = [next(j for j, y in enumerate(ia) if y == x) for x in ib]
        base = k[:-6]
        mw = max(mw, float(np.abs(a[base + "_W"][idx] - b[base + "_W"]).max())); mb = max(mb, float(np.abs(a[base + "_b"][idx] - b[base + "_b"]).max()))
        same_pred += int((a[base + "_pred"][idx] == b[base + "_pred"]).sum()); tot_pred += b[base + "_pred"].size; nitems += len(ib)
    elif "_mark" in k or k.endswith("_teacher") or k.endswith("_counts"):
        nm += int((a[k] != b[k]).sum())
print("問", nitems, "W 最大差", mw, "b 最大差", mb, "判定の一致", same_pred, "/", tot_pred, "印・教師・本数の不一致", nm)
