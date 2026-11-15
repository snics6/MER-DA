"""実験 20 の本物の posthoc の出力を，集計の段の出力と出力どうしで照らす（単位は開かない）．"""
import json
import numpy as np, pandas as pd
R = "/home/snics/MER-DA/results/dai20_eval"
d = pd.read_csv(f"{R}/posthoc/intercept_direction.csv"); l = pd.read_csv(f"{R}/posthoc/intercept_levels.csv")
pv = json.load(open(f"{R}/posthoc/intercept_provenance.json"))
sl = pd.read_csv(f"{R}/summary/levels.csv", low_memory=False); se = pd.read_csv(f"{R}/summary/effects.csv", low_memory=False)
print("行の数", len(d), "／記録", pv["n_rows"], "水準の行", len(l))
print("重なり", d.duplicated(["c","K","version","quantity","layer","seeds"]).sum(), l.duplicated(["c","K","version","layer","seeds"]).sum())
num = d.select_dtypes("number"); print("NaN・Inf（df を除く）", int((~np.isfinite(num.drop(columns=["df"]))).sum().sum()), "UAR の範囲", l.uar.min(), l.uar.max())
print("区間の向き lo<=est<=hi の外", int(((d.lo > d.est + 1e-12) | (d.hi < d.est - 1e-12)).sum()), int(((l.lo > l.uar + 1e-12) | (l.hi < l.uar - 1e-12)).sum()))
# 水準から差を作り直す
L = l.set_index(["c","K","version","layer","seeds"]).uar
def lv(c,K,v,y,s): return L[(c,K,v,y,s)]
m = []
for r in d.itertuples():
    a = (r.c, r.K); s = r.seeds
    if r.quantity == "total": x = lv(*a,r.version,r.layer,s) - lv(*a,"prod",r.layer,s)
    elif r.quantity == "direction": x = lv(*a,r.version,"ob",s) - lv(*a,"prod","ob",s)
    elif r.quantity == "intercept": x = (lv(*a,r.version,r.layer,s) - lv(*a,"prod",r.layer,s)) - (lv(*a,r.version,"ob",s) - lv(*a,"prod","ob",s))
    else: x = lv(*a,r.version,"ob",s) - lv(*a,r.version,r.layer,s)
    m.append(abs(x - r.est))
print("水準から作り直した差とのずれの最大", max(m))
# 集計の段の水準（点推定と区間の両端．adj・raw）
mine = l[l.layer.isin(["adj","raw"])].assign(version=lambda q: np.where(q.layer=="raw", q.version+"@raw", q.version))
mm = mine.merge(sl[["c","K","version","seeds","uar","lo","hi"]], on=["c","K","version","seeds"], suffixes=("","_s"))
print("水準の照合", len(mm), "行：点", float((mm.uar-mm.uar_s).abs().max()), "端", float(max((mm.lo-mm.lo_s).abs().max(), (mm.hi-mm.hi_s).abs().max())))
print("  コーパスごとの照合の行", mm.groupby("c").size().to_dict())
t = d[(d.quantity=="total")&(d.layer=="adj")].merge(se[se.b=="prod"][["a","c","K","seeds","est","lo","hi","se"]].rename(columns={"a":"version"}), on=["version","c","K","seeds"], suffixes=("","_s"))
print("差の照合", len(t), "行：点", float((t.est-t.est_s).abs().max()), "端", float(max((t.lo-t.lo_s).abs().max(),(t.hi-t.hi_s).abs().max())), "SE", float((t.se-t.se_s).abs().max()))
# raw の差も集計の段（prop@raw − prod@raw，A_VERS@raw は prop@raw との差だけ）と照らす
tr = d[(d.quantity=="total")&(d.layer=="raw")&(d.version=="prop")].merge(se[(se.a=="prop@raw")&(se.b=="prod@raw")][["c","K","seeds","est","lo","hi"]], on=["c","K","seeds"], suffixes=("","_s"))
print("raw の差（prop@raw − prod@raw）の照合", len(tr), "行：点", float((tr.est-tr.est_s).abs().max()), "端", float(max((tr.lo-tr.lo_s).abs().max(),(tr.hi-tr.hi_s).abs().max())))
piv = d[d.quantity.isin(["total","intercept"])].pivot_table(index=["c","K","version","layer","seeds"], columns="quantity", values="est").reset_index().merge(
    d[d.quantity=="direction"][["c","K","version","seeds","est"]].rename(columns={"est":"dir"}), on=["c","K","version","seeds"])
print("足し合わせのずれ", float((piv.total-piv.intercept-piv.dir).abs().max()), "／記録", pv["additivity_dev_max"], "行", len(piv))
print("版", sorted(d.version.unique()), "種", sorted(d.seeds.unique()))
