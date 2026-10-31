"""E78 関門 1：小さく流した出力（run_small.py）の点検。引数：出力の場所 評価のターゲット"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out, T = Path(sys.argv[1]), sys.argv[2]
print("ファイル:", sorted(p.name for p in out.iterdir()))
d = pd.read_csv(out / f"{T}.csv", dtype={"K": str, "param": str, "meff": str})
p = pd.read_csv(out / f"{T}_pseudo.csv", dtype={"K": str, "param": str, "meff": str}) if (out / f"{T}_pseudo.csv").stat().st_size > 1 else pd.DataFrame()
print(f"評価の行 {len(d)}，仮のターゲットの行 {len(p)}")
key = ["rep", "student", "method", "param", "meff", "K"]
print("評価の鍵の重複:", int(d.duplicated(key).sum()), " NaN の UAR:", int(d.uar.isna().sum()), " UAR の範囲:", d.uar.min(), d.uar.max())
print("評価の行の数（生徒 × 方法）:\n", d.groupby(["student", "method"]).size().unstack(fill_value=0).to_string())
if len(p):
    print("仮のターゲットの行の数（生徒 × 方法）:\n", p.groupby(["student", "method"]).size().unstack(fill_value=0).to_string())
    print("仮のターゲットの鍵の重複:", int(p.duplicated(["pseudo"] + key).sum()))
ch = pd.read_csv(out / f"{T}_chosen.csv", dtype={"K": str}) if (out / f"{T}_chosen.csv").stat().st_size > 1 else pd.DataFrame()
print("選んだ設定:", len(ch), "行")
if len(ch):
    print(ch.head(12).to_string())
    # 種 1・2 の行が選んだ設定とだけ一致するか
    seeded = d[d.method.isin(["dann", "mme", "mapctl", "xmuda"])].copy()
    seeded["seed"] = seeded.param.str.extract(r"s(\d+)$")[0].astype(int)
    seeded["p0"] = seeded.param.str.replace(r"s\d+$", "", regex=True)
    extra = seeded[seeded.seed > 0]
    chs = set(zip(ch.student, ch.method, ch.K.astype(str), ch.param))
    if "xmuda" in set(ch.method):
        want_x = {(K, prm) for (st, m, K, prm) in chs if m == "xmuda"}
    bad = []
    for _, rw in extra.iterrows():
        if rw.method == "xmuda":
            ok = (rw.K, rw.p0) in want_x
        else:
            ok = (rw.student, rw.method, rw.K, rw.p0) in chs
        if not ok:
            bad.append(tuple(rw[["student", "method", "K", "param"]]))
    print(f"種 1・2 の行 {len(extra)}，選んだ設定に無い行 {len(bad)}", bad[:5])
    miss = []
    for st, m, K, prm in chs:
        sub = seeded[(seeded.method == m) & (seeded.K == K) & (seeded.p0 == prm) & ((seeded.student == st) | (m == "xmuda"))]
        if m == "xmuda":
            sub = sub[sub.student == st]
        if sorted(sub.seed.unique()) != [0, 1, 2]:
            miss.append((st, m, K, prm, sorted(sub.seed.unique())))
    print("選んだ設定で種 0・1・2 がそろわないもの:", miss[:5], len(miss))
    # 選択の再現：仮のターゲットの種 0 の平均の最大
    ps = p[p.method.isin(["dann", "mme", "mapctl", "xmuda"])].copy()
    ps["p0"] = ps.param.str.replace(r"s\d+$", "", regex=True)
    best = ps.groupby(["student", "method", "K", "p0"]).uar.mean().reset_index().sort_values("uar").groupby(["student", "method", "K"]).tail(1)
    ties = ps.groupby(["student", "method", "K", "p0"]).uar.mean().reset_index()
    nt = ties.merge(best[["student", "method", "K", "uar"]], on=["student", "method", "K", "uar"]).groupby(["student", "method", "K"]).size()
    print("仮のターゲットで最大が同点の (生徒, 方法, K) の数:", int((nt > 1).sum()), "/", len(nt))
for f in ("diag", "noise"):
    fp = out / f"{T}_{f}.csv"
    if fp.exists() and fp.stat().st_size > 1:
        x = pd.read_csv(fp, dtype={"K": str})
        print(f"{f}: {len(x)} 行，列 {list(x.columns)}")
        print("  NaN の数:", x.isna().sum()[x.isna().sum() > 0].to_dict())
        if f == "diag" and "method" in x:
            print("  方法ごと:", x.groupby(["method"] + (["pseudo"] if "pseudo" in x else [])).size().to_dict() if "pseudo" not in x else x.method.value_counts().to_dict())
            fx = d[d.method == "cotfix"].param.str.extract(r"^(\w+):tau([\d.]+)")
            print("  cotfix の τ:", sorted(set(zip(d[d.method == "cotfix"].student, fx[0], fx[1].astype(float).round(4))))[:12])
        if f == "noise":
            print("  中央値（生徒 × 参照）:", x.groupby(["student", "teacher"]).err.median().round(4).to_dict())
for r in sorted({int(x.name.split("_r")[1].split("_")[0]) for x in out.glob(f"{T}_r*_logp.npy")}):
    lp = np.load(out / f"{T}_r{r}_logp.npy")
    te = np.load(out / f"{T}_r{r}_te.npy")
    pred = np.load(out / f"{T}_r{r}_pred.npy")
    print(f"反復 {r}：logp {lp.shape} {lp.dtype}，非有限 {int((~np.isfinite(lp)).sum())}，te {te.shape}，pred と argmax の食い違い {int((lp.argmax(2) != pred).sum())}")
    Ws = sorted(out.glob(f"{T}_r{r}_W_*.npy"))
    print("  W のファイル:", [w.name for w in Ws])
    for w in Ws:
        st = w.name.split("_W_")[1][:-4]
        rows_ = np.load(out / f"{T}_r{r}_wrows_{st}.npy")
        W = np.load(w)
        print(f"  {st}: W {W.shape} {W.dtype}，行 {len(rows_)}，その行の方法 {d[(d.rep == r)].set_index('row_id').loc[rows_].method.value_counts().to_dict()}")
    rr = d[d.rep == r]
    print(f"  W を持つ行 / 全行：{sum(len(np.load(out / f'{T}_r{r}_wrows_{w.name.split(chr(95) + 'W' + chr(95))[1][:-4]}.npy')) for w in Ws)} / {len(rr)}")
