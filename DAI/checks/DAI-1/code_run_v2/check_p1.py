"""段 1 の行の数・NaN・収束・保存の並び・判定からの UAR の作り直しを確かめる（UAR の値は出さない）。"""
import sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, "DAI/exp")
import dai_data as DD
O = sys.argv[1]
cache = {}
def uar(y, p):
    return 100 * np.mean([(p[y == k] == k).mean() for k in range(4) if (y == k).any()])
for f in sorted(glob.glob(f"{O}/pseudo/units/*_p1.csv")):
    df = pd.read_csv(f, low_memory=False); z = np.load(f.replace(".csv", ".npz"))
    c = f.split("/")[-1].split("_")[1]
    if c not in cache:
        d, _ = DD.load_corpus(c); cache[c] = dict(zip(d.sample_id, d.label_4))
    y = np.array([cache[c][s] for s in z["sample_test"]])
    cnt = df.groupby("method").size().to_dict()
    mism = 0; tot = 0
    for kind in DD.KINDS:
        items = list(z[f"p1_{kind}_items"]); P = z[f"p1_{kind}_pred"]
        assert P.shape[0] == len(items) == z[f"p1_{kind}_W"].shape[0]
        sub = df[(df.kind == kind) & df.method.isin(["none", "oracle"])]
        assert len(sub) == len(items)
        for (_, r), it, p in zip(sub.iterrows(), items, P):
            K = "all" if r.K == -1 else str(int(r.K))
            assert it == f"{K}|{r.Cw}|{r.Cb}", (it, r.K, r.Cw, r.Cb)
            tot += 1; mism += abs(uar(y, p) - r.uar) > 1e-9
        for m, key in (("start", "start_pred"), ("start_srcstd", "start_srcstd_pred")):
            r = df[(df.kind == kind) & (df.method == m)].iloc[0]; tot += 1; mism += abs(uar(y, z[f"p1_{kind}_{key}"]) - r.uar) > 1e-9
        if kind in DD.STUDENTS:
            for m, key in (("kbias", "kbias"), ("tgtonly", "tgt")):
                sub = df[(df.kind == kind) & (df.method == m)]; it = list(z[f"p1_{kind}_{key}_items"]); P = z[f"p1_{kind}_{key}_pred"]
                assert len(sub) == len(it) == len(P)
                for (_, r), i_, p in zip(sub.iterrows(), it, P):
                    exp = f"{int(r.K)}|{r.Cb}" if m == "kbias" else f"{int(r.K)}|{r.C}"
                    assert i_ == exp, (i_, exp)
                    tot += 1; mism += abs(uar(y, p) - r.uar) > 1e-9
    g = df[df.grad.notna()]
    print(f.split("/")[-1], "行", len(df), cnt, "uar の NaN", int(df[~df.method.str.startswith("_")].uar.isna().sum()),
          "grad 最大", f"{g.grad.max():.2e}", "1e-5 超", int((g.grad > 1e-5).sum()), "stalled", int(g.stalled.astype(bool).sum()),
          "finite 偽", int((~g.finite.astype(bool)).sum()), "照合", f"{mism}/{tot}", "commit", df.commit.unique()[0][:7], "dirty", df.dirty.unique())
