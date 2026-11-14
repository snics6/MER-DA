"""gate-code の検算（DAI-21 の計算の後の表 dai21_posthoc_cremad_sentence.py）．
table_block・effect_stats・cells_of・breakdown を使わずに，npz と発話一覧から数え直して出力と照らす．
値は画面に出さない（ずれ・一致・数だけを出す）．乱数は区間の引き直しを同じにするため P16.row_rng だけを借りる．"""
from __future__ import annotations

import hashlib
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
import dai16_probe as P16  # noqa: E402  （乱数の鍵だけ）

U = ROOT / "results" / "dai21_eval5" / "cremad"
PH = ROOT / "results" / "dai21_eval5" / "posthoc"
FZ = json.load(open(ROOT / "DAI" / "frozen" / "dai21_frozen.json"))
KS = (5, 10, 25)
SEEDS = {"all": None, "0-1": (0, 1), "2-9": tuple(range(2, 10))}
EMO = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3}
out = {}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def keymap(K):
    """設計の語から自分で決めた鍵（stage_summary の写しを使わない）．数個の版だけ．"""
    q = FZ["cremad"][str(K)]
    i, gm = int(q["prop_grid"]), int(q["student_main"])
    return {"prop": f"pred_adj_K{K}_M_F6_mix{i}", "prod": f"pred_adj_K{K}_prod", "prop@raw": f"pred_raw_K{K}_M_F6_mix{i}", "prod@raw": f"pred_raw_K{K}_prod",
            "tcal": f"pred_adj_K{K}_tcal", "avg": f"pred_adj_K{K}_avg", "stud_prop": f"pred_K{K}_stud_prop_g{gm}", "stud_prod": f"pred_K{K}_stud_prod_g{gm}",
            "none_AV": f"pred_raw_K{K}_none_AV", "stud_konly": f"pred_K{K}_stud_konly_g{gm}", "M_F6_ext": f"pred_adj_K{K}_M_F6_ext"}


# ---- 単位を読む
man = pd.read_csv(ROOT / "data" / "metadata" / "crema-d_utterance_manifest.csv", keep_default_na=False)
man = man.set_index("sample_id")
units = []
for p in sorted(U.glob("*.npz")):
    f, r = int(p.stem.split("_f")[1].split("_r")[0]), int(p.stem.rsplit("_r", 1)[1])
    z = np.load(p)
    sid = z["sid_test"].astype(str)
    df = pd.DataFrame(dict(sid=sid, y=z["y_test"].astype(int), cl=z["cl_test"].astype(str)))
    m = man.loc[sid]
    df["sen"] = [s.split("_")[1] for s in sid]
    df["inten"] = m["emotion_intensity"].to_numpy()                       # 強さは発話一覧から（名前の解析と別の道）
    df["lab_man"] = m["label_4"].astype(int).to_numpy()
    df["spk_man"] = m["speaker_id"].to_numpy()
    df["excl"] = m["exclude"].astype(str).to_numpy()
    preds = {(K, v): z[k].astype(int) for K in KS for v, k in keymap(K).items()}
    kp = {K: z[f"kpos_K{K}"] for K in KS}
    pool_sid = z["sid_pool"].astype(str)
    y_pool = z["y_pool"].astype(int)
    units.append(dict(name=p.stem, f=f, r=r, df=df, pred=preds, kpos=kp, pool_sid=pool_sid, y_pool=y_pool))
out["n_units"] = len(units)
out["label_vs_manifest_mismatch"] = int(sum((u["df"].y != u["df"].lab_man).sum() for u in units))
out["speaker_vs_manifest_mismatch"] = int(sum((u["df"].cl != u["df"].spk_man).sum() for u in units))
out["excluded_in_test"] = int(sum((u["df"].excl.str.lower() == "true").sum() for u in units))

# ---- (e) 強さの並び（発話一覧の全部の 4 ラベルの発話）
m4 = man[(man.label_4.astype(str) != "") & (man.exclude.astype(str).str.lower() != "true")].copy()
m4["sen"] = [s.split("_")[1] for s in m4.index]
tab = m4.groupby(["sen", "label_4", "emotion_intensity"]).size().unstack(fill_value=0)
out["manifest_4label_n"] = int(len(m4))
out["manifest_intensity_table"] = {f"{s}|{l}": {k: int(v) for k, v in row.items() if v} for (s, l), row in tab.iterrows()}
ns_corpus = m4.groupby("sen").size()
out["g_eff_sentence_corpus"] = float(ns_corpus.sum() ** 2 / (ns_corpus ** 2).sum())

# ---- (b) 除いた発話の数
rem = pd.read_csv(PH / "cremad_no_ieo_removed.csv")
mine = []
for u in units:
    d = u["df"]
    for i in range(4):
        mine.append(dict(name=u["name"], emotion=i, n_all=int((d.y == i).sum()), n_removed=int(((d.y == i) & (d.sen == "IEO")).sum())))
mine = pd.DataFrame(mine).merge(rem, on=["name", "emotion"], suffixes=("", "_o"))
out["removed_rows_matched"] = len(mine)
out["removed_mismatch"] = int(((mine.n_all != mine.n_all_o) | (mine.n_removed != mine.n_removed_o)).sum())
one = {u["f"]: u for u in units}
dist = {i: int(sum(((one[f]["df"].y == i) & (one[f]["df"].sen == "IEO")).sum() for f in one)) for i in range(4)}
out["removed_distinct_by_emotion"] = dist
kept = {i: int(sum(((one[f]["df"].y == i) & (one[f]["df"].sen != "IEO")).sum() for f in one)) for i in range(4)}
out["kept_distinct_by_emotion"] = kept
# 除いた後，分割ごと・話者ごとに感情が欠けるか
miss_spk = 0
for f, u in one.items():
    d = u["df"][u["df"].sen != "IEO"]
    g = d.groupby("cl").y.nunique()
    miss_spk += int((g < 4).sum())
    out.setdefault("speakers_per_fold", {})[f] = int(d.cl.nunique())
out["speakers_missing_an_emotion_after_drop"] = miss_spk
out["test_speakers_total"] = len({c for u in one.values() for c in u["df"].cl})


# ---- 独立の UAR
def uar(y, p):
    return 100.0 * np.mean([(p[y == i] == i).mean() for i in range(4) if (y == i).any()])


def theta(vals):  # {(f, r): x}
    fs = sorted({f for f, _ in vals})
    return float(np.mean([np.mean([x for (ff, _), x in vals.items() if ff == f]) for f in fs]))


def vseed(vals):
    fs = sorted({f for f, _ in vals})
    s = 0.0
    for f in fs:
        xs = [x for (ff, _), x in vals.items() if ff == f]
        s += (np.var(xs, ddof=1) / len(xs)) if len(xs) > 1 else 0.0
    return s / len(fs) ** 2


def sel(ss):
    return [u for u in units if ss is None or u["r"] in ss]


lv_no = pd.read_csv(PH / "cremad_no_ieo_levels.csv")
ef_no = pd.read_csv(PH / "cremad_no_ieo_effects.csv")
ef_s = pd.read_csv(PH / "cremad_sentence_cluster_effects.csv")
lv_sum = pd.read_csv(ROOT / "results" / "dai21_eval5" / "summary" / "levels.csv", low_memory=False)
lv_sum = lv_sum[lv_sum.part == "cremad"]
ef_sum = pd.read_csv(ROOT / "results" / "dai21_eval5" / "summary" / "effects.csv", low_memory=False)
ef_sum = ef_sum[ef_sum.part == "cremad"]

dev_lv, dev_lv_all, n_lv = 0.0, 0.0, 0
for K in KS:
    for v in keymap(K):
        for sn, ss in SEEDS.items():
            vals_no = {(u["f"], u["r"]): uar(u["df"].y.to_numpy()[u["df"].sen.to_numpy() != "IEO"], u["pred"][(K, v)][u["df"].sen.to_numpy() != "IEO"]) for u in sel(ss)}
            vals_all = {(u["f"], u["r"]): uar(u["df"].y.to_numpy(), u["pred"][(K, v)]) for u in sel(ss)}
            row = lv_no[(lv_no.K == K) & (lv_no.version == v) & (lv_no.seeds == sn)]
            rs = lv_sum[(lv_sum.K == K) & (lv_sum.version == v) & (lv_sum.seeds == sn)]
            assert len(row) == 1 and len(rs) == 1, (K, v, sn, len(row), len(rs))
            dev_lv = max(dev_lv, abs(theta(vals_no) - row.uar.iloc[0]), abs(theta(vals_all) - row.uar_all.iloc[0]))
            dev_lv_all = max(dev_lv_all, abs(theta(vals_all) - rs.uar.iloc[0]))
            n_lv += 1
out["levels_noieo_independent"] = dict(n=n_lv, est_dev_max=dev_lv, summary_all_dev_max=dev_lv_all)


# ---- 独立の区間：話者の塊の引き直し（文を除いたテスト）
def spk_counts(u, K, v, mask):
    d = u["df"][mask]
    p = u["pred"][(K, v)][mask]
    t = pd.crosstab(d.cl, d.y).reindex(columns=range(4), fill_value=0)
    c = pd.crosstab(d.cl[p == d.y.to_numpy()], d.y[p == d.y.to_numpy()]).reindex(index=t.index, columns=range(4), fill_value=0)
    return t.index.to_numpy(), c.to_numpy(float), t.to_numpy(float)


def boot_diff(a, b, K, sn, key, drop):
    rng = P16.row_rng("dai21", *key)
    us = sel(SEEDS[sn])
    fs = sorted({u["f"] for u in us})
    B = 2000
    tot, nval = np.zeros(B), np.zeros(B)
    d_fr = {}
    for f in fs:
        uf = sorted([u for u in us if u["f"] == f], key=lambda u: u["r"])
        mask = (uf[0]["df"].sen != "IEO").to_numpy() if drop else np.ones(len(uf[0]["df"]), bool)
        names = spk_counts(uf[0], K, a, mask)[0]
        W = rng.multinomial(len(names), np.full(len(names), 1.0 / len(names)), size=B).astype(float)
        diffs = []
        for u in uf:
            ca, cb = spk_counts(u, K, a, mask), (spk_counts(u, K, b, mask) if b is not None else None)
            ua = 100 * ((W @ ca[1]) / (W @ ca[2])).mean(1)
            bad = ((W @ ca[2]).sum(1) < 30) | ((W @ ca[2]) == 0).any(1)
            ub = 100 * ((W @ cb[1]) / (W @ cb[2])).mean(1) if b is not None else 0.0
            diffs.append(np.where(bad, np.nan, ua - ub))
            y, m_ = u["df"].y.to_numpy()[mask], mask
            d_fr[(f, u["r"])] = uar(y, u["pred"][(K, a)][m_]) - (uar(y, u["pred"][(K, b)][m_]) if b is not None else 0.0)
        dm = np.nanmean(np.stack(diffs), 0)
        ok = np.isfinite(dm)
        tot[ok] += dm[ok]
        nval[ok] += 1
    bt = tot / np.maximum(nval, 1)
    se = float(np.sqrt(np.var(bt, ddof=1) + vseed(d_fr)))
    est = theta(d_fr)
    return est, est - 1.96 * se, est + 1.96 * se, se


chk_ci = []
for (a, b) in (("prop", "prod"), ("stud_prop", "stud_prod"), ("stud_prop", "none_AV"), ("prop", "prop@raw"), ("stud_konly", "stud_prod"), ("prop", "M_F6_ext")):
    for K in KS:
        for sn in ("all", "0-1"):
            e, lo, hi, se = boot_diff(a, b, K, sn, ("A", "cremad", a, b, "CREMA-D", K, sn), True)
            row = ef_no[(ef_no.a == a) & (ef_no.b == b) & (ef_no.K == K) & (ef_no.seeds == sn)]
            assert len(row) == 1, (a, b, K, sn)
            r0 = row.iloc[0]
            chk_ci.append(max(abs(e - r0.est), abs(lo - r0.lo), abs(hi - r0.hi), abs(se - r0.se)))
            e2, *_ = boot_diff(a, b, K, sn, ("A", "cremad", a, b, "CREMA-D", K, sn), False)
            chk_ci.append(abs(e2 - r0.est_all))
for v in ("prop", "stud_prop"):
    for K in KS:
        e, lo, hi, se = boot_diff(v, None, K, "all", ("L", "cremad", "CREMA-D", K, v, "all"), True)
        row = lv_no[(lv_no.version == v) & (lv_no.K == K) & (lv_no.seeds == "all")].iloc[0]
        chk_ci.append(max(abs(e - row.uar), abs(lo - row.lo), abs(hi - row.hi), abs(se - row.se)))
out["noieo_interval_independent"] = dict(n_checks=len(chk_ci), dev_max=float(max(chk_ci)))


# ---- 独立の区間：文の jackknife
SENT = sorted({s for u in units for s in u["df"].sen})
ns = np.array([sum(int((one[f]["df"].sen == s).sum()) for f in one) for s in SENT], float)
out["sentences"] = SENT
out["n_per_sentence_distinct"] = dict(zip(SENT, map(int, ns)))
out["g_eff_sentence_test"] = float(ns.sum() ** 2 / (ns ** 2).sum())
jk_dev, jk_dev_g, n_jk = 0.0, 0.0, 0
for (a, b) in (("prop", "prod"), ("stud_prop", "stud_prod"), ("stud_prop", "none_AV"), ("prop", "tcal"), ("prop", "avg"), ("stud_konly", "stud_prod")):
    for K in KS:
        for sn, ss in SEEDS.items():
            us = sel(ss)
            full = {(u["f"], u["r"]): uar(u["df"].y.to_numpy(), u["pred"][(K, a)]) - uar(u["df"].y.to_numpy(), u["pred"][(K, b)]) for u in us}
            est = theta(full)
            th = []
            for s in SENT:
                vals = {}
                for u in us:
                    mk = u["df"].sen.to_numpy() != s
                    y = u["df"].y.to_numpy()[mk]
                    vals[(u["f"], u["r"])] = uar(y, u["pred"][(K, a)][mk]) - uar(y, u["pred"][(K, b)][mk])
                th.append(theta(vals))
            th = np.array(th)
            G = len(SENT)
            se = float(np.sqrt((G - 1) / G * ((th - th.mean()) ** 2).sum() + vseed(full)))
            q11, q9 = sst.t.ppf(0.975, 11), sst.t.ppf(0.975, 9)
            row = ef_s[(ef_s.a == a) & (ef_s.b == b) & (ef_s.K == K) & (ef_s.seeds == sn)]
            assert len(row) == 1
            r0 = row.iloc[0]
            jk_dev = max(jk_dev, abs(est - r0.est), abs(se - r0.se), abs(est - q11 * se - r0.lo), abs(est + q11 * se - r0.hi))
            jk_dev_g = max(jk_dev_g, abs(est - q9 * se - r0.lo_geff), abs(est + q9 * se - r0.hi_geff))
            n_jk += 1
out["sentence_jk_independent"] = dict(n=n_jk, dev_max=jk_dev, geff_dev_max=jk_dev_g, df_geff_in_file=sorted(set(ef_s.df_geff.astype(int))), df_in_file=sorted(set(ef_s.df.astype(float))))

# ---- 内訳
bs = pd.read_csv(PH / "cremad_by_sentence.csv")
bi = pd.read_csv(PH / "cremad_by_intensity.csv")
dsen, dint, nchk = 0.0, 0.0, 0
for K in KS:
    for v in ("prop", "prod", "stud_prop", "stud_prod", "none_AV", "tcal"):
        cat = pd.concat([u["df"].assign(p=u["pred"][(K, v)]) for u in units])
        cat["hit"] = cat.p == cat.y
        for s in SENT:
            g = cat[cat.sen == s]
            rec = [100 * g[g.y == i].hit.mean() for i in range(4)]
            row = bs[(bs.K == K) & (bs.version == v) & (bs.sentence == s)].iloc[0]
            dsen = max(dsen, abs(np.mean(rec) - row.uar), max(abs(rec[i] - row[f"recall_{i}"]) for i in range(4)), abs(len(g) - row.n), abs(len(g) / 10 - row.n_per_seed))
        for (it, i), g in cat.groupby(["inten", "y"]):
            row = bi[(bi.K == K) & (bi.version == v) & (bi.intensity == it) & (bi.emotion == i)]
            assert len(row) == 1, (it, i)
            dint = max(dint, abs(100 * g.hit.mean() - row.recall.iloc[0]), abs(len(g) - row.n.iloc[0]))
        nchk += 1
out["breakdown_independent"] = dict(n_K_version=nchk, sentence_dev_max=dsen, intensity_dev_max=dint, n_rows_sentence=len(bs), n_rows_intensity=len(bi),
                                     intensity_groups_in_file=sorted(set(zip(bi.intensity, bi.emotion.astype(int)))))

# ---- (f) K 本の中の文 IEO の発話の数（発話の数だけ）
kk = []
for u in units:
    for K in KS:
        ks = u["pool_sid"][u["kpos"][K]]
        yk = u["y_pool"][u["kpos"][K]]
        ie = np.array([s.split("_")[1] == "IEO" for s in ks])
        kk.append(dict(name=u["name"], K=K, n_ieo=int(ie.sum()), **{f"n_ieo_{i}": int((ie & (yk == i)).sum()) for i in range(4)}))
kk = pd.DataFrame(kk)
out["kshot_ieo_counts"] = {str(K): dict(min=int(g.n_ieo.min()), max=int(g.n_ieo.max()), mean=float(g.n_ieo.mean()), neutral_max=int(g.n_ieo_0.max()),
                                        units_with_zero=int((g.n_ieo == 0).sum())) for K, g in kk.groupby("K")}
kk.to_csv(Path(__file__).parent / "kshot_ieo_counts.csv", index=False)
# K 本の文の内訳と，K 本と同じ発話（話者は分けてあるのでテストと同じ発話は無いはず）
out["kshot_test_overlap"] = int(sum(len(set(u["pool_sid"][u["kpos"][K]]) & set(u["df"].sid)) for u in units for K in KS))
out["pool_test_speaker_overlap"] = int(sum(len({s.split("_")[0] for s in u["pool_sid"]} & {s.split("_")[0] for s in u["df"].sid}) for u in units))

# ---- 由来
prov = json.load(open(PH / "cremad_sentence_provenance.json"))
cur = {"script": sha(prov["script"]) == prov["script_sha256"], "frozen": sha(ROOT / "DAI/frozen/dai21_frozen.json") == prov["frozen_sha256"],
       "frozen20": sha(ROOT / "DAI/frozen/dai20_frozen.json") == prov["frozen20_sha256"], "design": sha(ROOT / "DAI/docs/実験21_設計.md") == prov["design_sha256"],
       "modules": all(sha(ROOT / "DAI/exp" / n) == h for n, h in prov["modules_sha256"].items()),
       "summary": all(sha(ROOT / "results/dai21_eval5/summary" / n) == h for n, h in prov["summary_sha256"].items()),
       "dai21_provenance": sha(ROOT / "results/dai21_eval5/provenance.json") == prov["dai21_provenance_sha256"]}
npzs = sorted(U.glob("*.npz"))
cur["units_digest"] = (f"{len(npzs)}:" + hashlib.sha256("".join(f"{q.name} {sha(q)}\n" for q in npzs).encode()).hexdigest()) == prov["units_npz_digest"]
cur["outputs_rows"] = all(len(pd.read_csv(PH / n)) == k for n, k in prov["outputs"].items())
out["provenance_current"] = cur
out["provenance_check_vs_summary"] = prov["check_vs_summary"]
json.dump(out, open(Path(__file__).parent / "verify_out.json", "w"), indent=1, ensure_ascii=False)
for k, v in out.items():
    if k != "manifest_intensity_table":
        print(k, v)
