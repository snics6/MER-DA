"""gate-code 再検査（第 2 版 dai21_posthoc_cremad_sentence.py）．台本の部品（table_block・effect_stats・stack8・assume_of・breakdown）を使わずに作り直して照らす．
値は画面に出さない（ずれ・一致・数だけ）．乱数は区間の引き直しを同じにするため P16.row_rng だけ，d′ の定義は P16.dprime だけを借りる．"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
from scipy import stats as sst

ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai16_probe as P16  # noqa: E402

U = ROOT / "results" / "dai21_eval5" / "cremad"
PH = ROOT / "results" / "dai21_eval5" / "posthoc"
OLD = ROOT / "DAI" / "checks" / "DAI-21" / "post_tools_run" / "out_plain"      # 第 1 版の出力（前の検査で再実行して本番と同じだったもの）
FZ = json.load(open(ROOT / "DAI" / "frozen" / "dai21_frozen.json"))
KS = (5, 10, 25)
SEEDS = {"all": None, "0-1": (0, 1), "2-9": tuple(range(2, 10))}
o = {}


def keymap(K):
    q = FZ["cremad"][str(K)]
    i, gm = int(q["prop_grid"]), int(q["student_main"])
    return {"prop": f"pred_adj_K{K}_M_F6_mix{i}", "prod": f"pred_adj_K{K}_prod", "prop@raw": f"pred_raw_K{K}_M_F6_mix{i}", "tcal": f"pred_adj_K{K}_tcal",
            "stud_prop": f"pred_K{K}_stud_prop_g{gm}", "stud_prod": f"pred_K{K}_stud_prod_g{gm}", "none_AV": f"pred_raw_K{K}_none_AV",
            "none_A": f"pred_raw_K{K}_none_A", "ret_A": f"pred_raw_K{K}_ret_A", "ret_V": f"pred_raw_K{K}_ret_V", "start_A": "pred_raw_K0_start_A", "full_V": "pred_raw_K0_full_V",
            "stud_konly": f"pred_K{K}_stud_konly_g{gm}"}


units = []
for p in sorted(U.glob("*.npz")):
    f, r = int(p.stem.split("_f")[1].split("_r")[0]), int(p.stem.rsplit("_r", 1)[1])
    z = np.load(p)
    sid = z["sid_test"].astype(str)
    d = pd.DataFrame(dict(y=z["y_test"].astype(int), cl=z["cl_test"].astype(str), sen=[s.split("_")[1] for s in sid], inten=[s.split("_")[3] for s in sid]))
    units.append(dict(name=p.stem, f=f, r=r, d=d, pred={(K, v): z[k].astype(int) for K in KS for v, k in keymap(K).items()}, z=z))
SENT = sorted(set(units[0]["d"].sen))


def uar(y, p):
    return 100.0 * np.mean([(p[y == i] == i).mean() for i in range(4)])


def theta(vals):
    fs = sorted({f for f, _ in vals})
    return float(np.mean([np.mean([x for (ff, _), x in vals.items() if ff == f]) for f in fs]))


def vseed(vals):
    fs = sorted({f for f, _ in vals})
    return sum((np.var(xs, ddof=1) / len(xs)) if len(xs) > 1 else 0.0 for xs in ([x for (ff, _), x in vals.items() if ff == f] for f in fs)) / len(fs) ** 2


def sel(sn):
    return [u for u in units if SEEDS[sn] is None or u["r"] in SEEDS[sn]]


def spk(u, K, v, mask):
    d = u["d"][mask]
    p = u["pred"][(K, v)][mask]
    t = pd.crosstab(d.cl, d.y).reindex(columns=range(4), fill_value=0)
    h = p == d.y.to_numpy()
    c = pd.crosstab(d.cl[h], d.y[h]).reindex(index=t.index, columns=range(4), fill_value=0)
    return t.index.to_numpy(), c.to_numpy(float), t.to_numpy(float)


def wuar(W, cnt):
    return 100 * ((W @ cnt[1]) / (W @ cnt[2])).mean(1)


# ---- (b) 差の差の区間：引き直しの 1 回ごとに 4 つの UAR を別々に作る
lv = pd.read_csv(PH / "cremad_no_ieo_levels.csv")
ef = pd.read_csv(PH / "cremad_no_ieo_effects.csv")
devs, n_ = [], 0
cases = [("prop", "prod"), ("stud_prop", "stud_prod"), ("stud_prop", "none_AV"), ("prop", "prop@raw"), ("stud_konly", "stud_prod"), ("prop", "tcal")]
for a, b in cases:
    for K in KS:
        for sn in ("all", "0-1", "2-9"):
            rng = P16.row_rng("dai21", "A", "cremad", a, b, "CREMA-D", K, sn)
            us = sel(sn)
            fs = sorted({u["f"] for u in us})
            B = 2000
            tot, nv = np.zeros(B), np.zeros(B)
            dfr = {}
            for f in fs:
                uf = sorted([u for u in us if u["f"] == f], key=lambda u: u["r"])
                keep = (uf[0]["d"].sen != "IEO").to_numpy()
                allm = np.ones(len(keep), bool)
                names = spk(uf[0], K, a, allm)[0]
                assert np.array_equal(names, spk(uf[0], K, a, keep)[0])
                W = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=B).astype(float)
                ds = []
                for u in uf:
                    ua_n, ub_n, ua_a, ub_a = (wuar(W, spk(u, K, v, m)) for v, m in ((a, keep), (b, keep), (a, allm), (b, allm)))
                    ds.append((ua_n - ub_n) - (ua_a - ub_a))
                    y = u["d"].y.to_numpy()
                    dfr[(f, u["r"])] = (uar(y[keep], u["pred"][(K, a)][keep]) - uar(y[keep], u["pred"][(K, b)][keep])) - (uar(y, u["pred"][(K, a)]) - uar(y, u["pred"][(K, b)]))
                dm = np.mean(np.stack(ds), 0)
                tot += dm
                nv += 1
            bt = tot / nv
            se = float(np.sqrt(np.var(bt, ddof=1) + vseed(dfr)))
            est = theta(dfr)
            r0 = ef[(ef.a == a) & (ef.b == b) & (ef.K == K) & (ef.seeds == sn)].iloc[0]
            devs.append(max(abs(est - r0.d_est), abs(se - r0.d_se), abs(est - 1.96 * se - r0.d_lo), abs(est + 1.96 * se - r0.d_hi)))
            n_ += 1
for v in ("prop", "stud_prop", "none_A", "start_A"):
    for K in KS:
        for sn in ("all", "0-1"):
            rng = P16.row_rng("dai21", "L", "cremad", "CREMA-D", K, v, sn)
            us = sel(sn)
            fs = sorted({u["f"] for u in us})
            tot = np.zeros(2000)
            dfr = {}
            for f in fs:
                uf = sorted([u for u in us if u["f"] == f], key=lambda u: u["r"])
                keep = (uf[0]["d"].sen != "IEO").to_numpy()
                allm = np.ones(len(keep), bool)
                names = spk(uf[0], K, v, allm)[0]
                W = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=2000).astype(float)
                ds = []
                for u in uf:
                    ds.append(wuar(W, spk(u, K, v, keep)) - wuar(W, spk(u, K, v, allm)))
                    y = u["d"].y.to_numpy()
                    dfr[(f, u["r"])] = uar(y[keep], u["pred"][(K, v)][keep]) - uar(y, u["pred"][(K, v)])
                tot += np.mean(np.stack(ds), 0)
            bt = tot / len(fs)
            se = float(np.sqrt(np.var(bt, ddof=1) + vseed(dfr)))
            est = theta(dfr)
            r0 = lv[(lv.version == v) & (lv.K == K) & (lv.seeds == sn)].iloc[0]
            devs.append(max(abs(est - r0.d_uar), abs(se - r0.d_se), abs(est - 1.96 * se - r0.d_lo), abs(est + 1.96 * se - r0.d_hi)))
            n_ += 1
o["dd_interval_independent"] = dict(n=n_, dev_max=float(max(devs)))

# ---- (a) 前の版の値が変わっていないか
old_lv, old_ef = pd.read_csv(OLD / "cremad_no_ieo_levels.csv"), pd.read_csv(OLD / "cremad_no_ieo_effects.csv")
KL, KE = ["K", "version", "seeds"], ["a", "b", "K", "seeds"]
m = lv.merge(old_lv, on=KL, suffixes=("", "_old"))
o["levels_vs_v1"] = dict(n=len(m), dev=max(float(np.abs(m[c] - m[c + "_old"]).max()) for c in ("uar", "lo", "hi", "se", "uar_all", "d_uar")),
                         text=int(sum((m[c].fillna("").astype(str) != m[c + "_old"].fillna("").astype(str)).sum() for c in ("same_as", "same_pred", "method"))))
m = ef.merge(old_ef, on=KE, suffixes=("", "_old"))
o["effects_vs_v1"] = dict(n=len(m), dev=max(float(np.abs(m[c] - m[c + "_old"]).max()) for c in ("est", "lo", "hi", "se", "est_all", "d_est")),
                          text=int(sum((m[c].fillna("").astype(str) != m[c + "_old"].fillna("").astype(str)).sum() for c in ("mark", "mark_all", "same_pred"))))
os_, ns_ = pd.read_csv(OLD / "cremad_sentence_cluster_effects.csv"), pd.read_csv(PH / "cremad_sentence_cluster_effects.csv")
m = ns_.merge(os_, on=KE, suffixes=("", "_old"))
o["sentence_vs_v1"] = dict(n=len(m), est_se_dev=max(float(np.abs(m.est - m.est_old).max()), float(np.abs(m.se - m.se_old).max())),
                           new_lo_vs_old_lo_geff=max(float(np.abs(m.lo - m.lo_geff).max()), float(np.abs(m.hi - m.hi_geff).max())),
                           new_lo_n12_vs_old_lo=max(float(np.abs(m.lo_n12 - m.lo_old).max()), float(np.abs(m.hi_n12 - m.hi_old).max())))
ob, nb = pd.read_csv(OLD / "cremad_by_sentence.csv"), pd.read_csv(PH / "cremad_by_sentence.csv")
m = nb.merge(ob, on=["sentence", "K", "version"], suffixes=("", "_old"))
o["by_sentence_vs_v1"] = dict(n=len(m), versions=sorted(set(m.version)), dev=max([float(np.abs(m.uar_pooled - m.uar).max())] + [float(np.abs(m[f"recall_pooled_{i}"] - m[f"recall_{i}"]).max()) for i in range(4)]
                                                                                      + [float(np.abs(m.n - m.n_old).max())]))
oi, ni = pd.read_csv(OLD / "cremad_by_intensity.csv"), pd.read_csv(PH / "cremad_by_intensity.csv")
m = ni.merge(oi, on=["intensity", "emotion", "K", "version"], suffixes=("", "_old"))
o["by_intensity_vs_v1"] = dict(n=len(m), dev=max(float(np.abs(m.recall_pooled - m.recall).max()), float(np.abs(m.n - m.n_old).max())))
o["removed_vs_v1_identical"] = bool(pd.read_csv(OLD / "cremad_no_ieo_removed.csv").equals(pd.read_csv(PH / "cremad_no_ieo_removed.csv")))

# ---- (c) 文を塊にした区間：自由度 9，印，水準の jackknife
FLOOR = P16.FLOOR
q9, q11 = sst.t.ppf(0.975, 9), sst.t.ppf(0.975, 11)
mk = ns_.apply(lambda r: {"上回る": "↑", "下回る": "↓"}.get(P16.verdict(r.lo, r.hi, r.est), ""), axis=1)
o["sentence_mark_from_main"] = dict(n=len(ns_), mismatch=int((mk.fillna("") != ns_.mark.fillna("")).sum()), dfs=sorted(set(ns_.df.astype(float))),
                                    lo_formula_dev=float(np.abs(ns_.lo - (ns_.est - q9 * ns_.se)).max()), lo_n12_formula_dev=float(np.abs(ns_.lo_n12 - (ns_.est - q11 * ns_.se)).max()))
sl = pd.read_csv(PH / "cremad_sentence_cluster_levels.csv")
jd, nj = 0.0, 0
for v in ("prop", "stud_prop", "none_A", "start_A", "full_V"):
    for K in KS:
        for sn in SEEDS:
            us = sel(sn)
            full = {(u["f"], u["r"]): uar(u["d"].y.to_numpy(), u["pred"][(K, v)]) for u in us}
            th = []
            for s in SENT:
                th.append(theta({(u["f"], u["r"]): uar(u["d"].y.to_numpy()[u["d"].sen.to_numpy() != s], u["pred"][(K, v)][u["d"].sen.to_numpy() != s]) for u in us}))
            th = np.array(th)
            se = float(np.sqrt(11 / 12 * ((th - th.mean()) ** 2).sum() + vseed(full)))
            est = theta(full)
            r0 = sl[(sl.version == v) & (sl.K == K) & (sl.seeds == sn)]
            if len(r0) == 0:
                continue                                                     # K によらない版は引き方の組の行が無いことがある
            r0 = r0.iloc[0]
            jd = max(jd, abs(est - r0.uar), abs(se - r0.se), abs(est - q9 * se - r0.lo), abs(est + q9 * se - r0.hi))
            nj += 1
o["sentence_levels_independent"] = dict(n=nj, dev_max=jd, dfs=sorted(set(sl.df.astype(float))))

# ---- 4. 感情ごとの再現率（文を除く）：点推定と区間の数個
pe = pd.read_csv(PH / "cremad_no_ieo_per_emotion.csv")
pdv, npe = 0.0, 0
for v in ("prop", "stud_prop", "none_A"):
    for K in KS:
        for i in range(4):
            rng = P16.row_rng("dai21", "E", "cremad", "CREMA-D", K, v, i)
            fs = sorted({u["f"] for u in units})
            tot = np.zeros(2000)
            vals = {}
            for f in fs:
                uf = sorted([u for u in units if u["f"] == f], key=lambda u: u["r"])
                keep = (uf[0]["d"].sen != "IEO").to_numpy()
                names = spk(uf[0], K, v, keep)[0]
                W = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=2000).astype(float)
                ds = []
                for u in uf:
                    c = spk(u, K, v, keep)
                    ds.append(100 * (W @ c[1][:, i]) / (W @ c[2][:, i]))
                    y = u["d"].y.to_numpy()[keep]
                    vals[(f, u["r"])] = 100 * (u["pred"][(K, v)][keep][y == i] == i).mean()
                tot += np.mean(np.stack(ds), 0)
            bt = tot / len(fs)
            se = float(np.sqrt(np.var(bt, ddof=1) + vseed(vals)))
            est = theta(vals)
            r0 = pe[(pe.version == v) & (pe.K == K) & (pe.emotion == i)].iloc[0]
            pdv = max(pdv, abs(est - r0.recall), abs(est - 1.96 * se - r0.lo), abs(est + 1.96 * se - r0.hi))
            npe += 1
o["per_emotion_noieo_independent"] = dict(n=npe, dev_max=pdv)

# ---- 内訳 16 版と取り違え
bc = pd.read_csv(PH / "cremad_by_sentence_confusion.csv")
bd, nbc = 0.0, 0
for K in KS:
    for v in ("none_A", "ret_A", "ret_V", "start_A", "full_V", "prop"):
        cat = pd.concat([u["d"].assign(p=u["pred"][(K, v)]) for u in units])
        g = cat.groupby(["sen", "y", "p"]).size()
        sub_ = bc[(bc.K == K) & (bc.version == v)].set_index(["sentence", "true", "pred"]).n
        x = sub_.reindex(g.index.rename(["sentence", "true", "pred"])).to_numpy()
        bd = max(bd, float(np.abs(x - g.to_numpy()).max()), float(sub_.sum() - len(cat)))
        for s in SENT:
            gs = cat[cat.sen == s]
            rec = [100 * (gs[gs.y == i].p == i).mean() for i in range(4)]
            r0 = nb[(nb.K == K) & (nb.version == v) & (nb.sentence == s)].iloc[0]
            bd = max(bd, abs(np.mean(rec) - r0.uar_pooled), max(abs(rec[i] - r0[f"recall_pooled_{i}"]) for i in range(4)))
        nbc += 1
o["breakdown16_independent"] = dict(n_K_version=nbc, dev_max=bd, versions_in_file=sorted(set(nb.version)), n_versions=len(set(nb.version)))
st = nb[nb.version.str.startswith(("start_", "full_"))]
o["start_full_same_across_K"] = int(st.groupby(["sentence", "version"]).uar_pooled.nunique().max())

# ---- (e) K 本の本数と単位ごとの値
ku = pd.read_csv(PH / "cremad_kshot_ieo_units.csv")
ref = pd.read_csv(ROOT / "DAI/checks/DAI-21/post_tools_run/kshot_ieo_counts.csv")
m = ku.merge(ref, on=["name", "K"], suffixes=("", "_r"))
o["kshot_vs_mine"] = dict(n=len(m), dev=int(max(np.abs(m[c] - m[c + "_r"]).max() for c in ["n_ieo"] + [f"n_ieo_{i}" for i in range(4)])))
kd = 0.0
for u in units:
    for K in KS:
        r0 = ku[(ku.name == u["name"]) & (ku.K == K)].iloc[0]
        y = u["d"].y.to_numpy()
        keep = (u["d"].sen != "IEO").to_numpy()
        for a, b in (("prop", "prod"), ("stud_prop", "stud_prod"), ("stud_prop", "none_AV")):
            kd = max(kd, abs(uar(y, u["pred"][(K, a)]) - uar(y, u["pred"][(K, b)]) - r0[f"d_{a}__{b}"]),
                     abs(uar(y[keep], u["pred"][(K, a)][keep]) - uar(y[keep], u["pred"][(K, b)][keep]) - r0[f"d_noieo_{a}__{b}"]))
        kd = max(kd, abs(uar(y, u["pred"][(K, "ret_A")]) - r0.uar_ret_A))
o["kshot_unit_values_independent"] = dict(n_units_K=len(units) * 3, dev_max=kd)


# ---- (d) 仮定の量：自分の Tukey の当てはめと相関で数個
def tk(X):
    g = X.mean(); a = X.mean(1) - g; b = X.mean(0) - g
    den = (a ** 2).sum() * (b ** 2).sum()
    lam = (np.outer(a, b) * X).sum() / den if den > 1e-12 else 0.0
    return X - (g + a[:, None] + b[None, :] + lam * np.outer(a, b))


def cc(x, y):
    x, y = x.ravel(), y.ravel()
    return float(np.corrcoef(x, y)[0, 1])


ass = pd.read_csv(PH / "cremad_assumptions.csv").set_index("K")
acc = {}
rank = []
for u in units:
    z = u["z"]
    for K in KS:
        D = {"ext": z["D_M_ext"], "home": z["D_M_home"], "K": z[f"D_M_K_K{K}"], "orc": z[f"D_M_orc_K{K}"]}
        R = {k: tk(v.astype(float)) for k, v in D.items()}
        for k in ("ext", "home", "K"):
            rank.append(np.linalg.matrix_rank(np.stack([R[k][0], R[k][1]]), tol=1e-10))
            assert np.allclose(R[k][0], -R[k][1]) and abs(R[k][0].sum()) < 1e-9
            bb = D[k].mean(0) - D[k].mean()
            rank.append(float(abs(R[k][0] @ bb)))                     # 列の差の向きと直交（Tukey の 1 自由度を引いた残り）
            acc.setdefault((K, f"intcorr_M_{k}"), []).append(cc(R[k], R["orc"]))
            pr = lambda X: X - X.mean(1, keepdims=True)  # noqa: E731
            acc.setdefault((K, f"profcorr_M_{k}"), []).append(cc(pr(D[k]), pr(D["orc"])))
            acc.setdefault((K, f"emocorr_M_{k}"), []).append(cc(D[k].mean(0), D["orc"].mean(0)))
            acc.setdefault((K, f"topok_M_{k}"), []).append(float(D[k].mean(1).argmax() == D["orc"].mean(1).argmax()))
        dt = np.stack([P16.dprime(z[f"p_test_ret_{m}_K{K}"].astype(np.float64), z["y_test"].astype(int))[0] for m in "AV"])
        acc.setdefault((K, "drift_M"), []).append(float(np.abs(dt - D["orc"]).mean()))
ad = max(abs(np.nanmean(v) - ass.loc[K, c]) for (K, c), v in acc.items())
o["assumptions_independent"] = dict(n_cols_checked=len({c for _, c in acc}), dev_max=float(ad))
o["dcenter_2rows"] = dict(row1_eq_minus_row2=True, row_sum_zero=True, max_abs_dot_with_column_effect=float(max(x for x in rank if isinstance(x, float))))

json.dump(o, open(Path(__file__).parent / "verify_v2_out.json", "w"), indent=1, ensure_ascii=False, default=str)
for k, v in o.items():
    print(k, v)
