"""gate-code 第 3 版（DAI-5）：6674496 で足した stats の関数（sensitivity，recall_rows，scores_rows の n_clusters_K，
collect の fixed_agree，重みの表の cv_uar_・score_mean_，K = 0 の identical）を，偽の評価の回で dai5_stats.evaluate ごと流し，
保存した判定から numpy で独立に作り直して突き合わせる。成績の値は出さない（行の数，NaN の数，作り直しとの差だけ）。
偽の評価の回：仮のターゲットの回の試走 AFEW_IEMOCAP_f{0..4}_r0（--test-tether）を AFEW_ALL_f{f}_r0 と名づける（第 2 版の try_stats_v2.py と同じ）。
テストの発話は IEMOCAP なので，DAI/splits/AFEW.csv の代わりに IEMOCAP の分割の表を読み，部分集合は IEMOCAP の marker・nomarker と，
テストの発話から選んだ 40 本を asr_suspect と名づけた偽の表で流す（偽の根 fake/root の下．本物の DAI/frozen と DAI/splits には書かない）。
使い方：CUDA_VISIBLE_DEVICES= ./.venv/bin/python DAI/checks/DAI-5/code_run_v3/try_stats_v3.py <試走の置き場> <偽の根>
"""
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai5 as D5  # noqa: E402
import dai5_stats as S5  # noqa: E402
import dai_data as DD  # noqa: E402

src, fake = Path(sys.argv[1]), Path(sys.argv[2])
if fake.exists():
    shutil.rmtree(fake)
fin, fout, froot = fake / "in", fake / "out5", fake / "root"
(fin / "eval" / "units").mkdir(parents=True); (fout / "eval" / "units").mkdir(parents=True); (fout / "pseudo" / "units").mkdir(parents=True)
(fout / "select").mkdir(parents=True)
for d in ("DAI/splits", "DAI/frozen", "results/dai1/source"):
    (froot / d).mkdir(parents=True)
shutil.copy(ROOT / "DAI" / "splits" / "IEMOCAP.csv", froot / "DAI" / "splits" / "AFEW.csv")
os.symlink(ROOT / "results" / "dai1" / "source" / "summary.csv", froot / "results" / "dai1" / "source" / "summary.csv")

with open(D5.FROZEN1, encoding="utf-8") as fh:
    fz1_real = json.load(fh)
fz1 = json.loads(json.dumps(fz1_real)); fz1["AFEW"]["n_seeds_eval"] = 1
for f in range(DD.N_FOLDS):
    a = ROOT / "results" / "dai1" / "pseudo" / "units" / f"AFEW_IEMOCAP_f{f}_r0"
    b = fin / "eval" / "units" / f"AFEW_ALL_f{f}_r0"
    for w in (1, 2):
        shutil.copy(f"{a}_p{w}.npz", f"{b}_p{w}.npz")
    shutil.copy(f"{a}_p3.csv", f"{b}_p3.csv"); shutil.copy(f"{a}_p1.csv", f"{b}_p1.csv")
    z3 = dict(np.load(f"{a}_p3.npz"))
    for s in DD.STUDENTS:
        for K in (5, 25):
            jj = D5.joint_sel(fz1["AFEW"], s, K)
            it = [json.loads(x) for x in z3[f"p3_{s}_K{K}_items"]]
            jm = [j for j, x in enumerate(it) if x.get("method") == "joint" and x.get("teacher") == jj["teacher"] and np.isclose(x["rho"], jj["rho"]) and np.isclose(x["a"], jj["a"])][0]
            items = [json.dumps(dict(method="joint_tether", tag="prop_sel", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"], Cw=cw, Cb=cb), sort_keys=True) for cw, cb in dai1.TETHER]
            z3[f"p3t_{s}_K{K}_items"] = np.array(items)
            pr = np.repeat(z3[f"p3_{s}_K{K}_pred"][jm][None], len(items), 0).copy()
            fcw, fcb = fz1["AFEW"]["tether"][f"{s}|{K}"]
            jf = [j for j, (cw, cb) in enumerate(dai1.TETHER) if (cw, cb) == (fcw, fcb)][0]
            if s == "A" and K == 5 and f == 0:                     # 固定の点の判定を 3 本だけ替え，判定の一致率が 1 未満になる場合を作る
                pr[jf, :3] = (pr[jf, :3] + 1) % 4
            z3[f"p3t_{s}_K{K}_pred"] = pr
    np.savez(f"{b}_p3.npz", **z3)
    sp = src / "pseudo" / "units" / f"AFEW_IEMOCAP_f{f}_r0"
    shutil.copy(f"{sp}_p5.npz", fout / "eval" / "units" / f"AFEW_ALL_f{f}_r0_p5.npz")
    shutil.copy(f"{sp}_p5.csv", fout / "eval" / "units" / f"AFEW_ALL_f{f}_r0_p5.csv")
    shutil.copy(f"{sp}_p5.npz", fout / "pseudo" / "units" / f"AFEW_IEMOCAP_f{f}_r0_p5.npz")
gpu = str(np.load(fout / "eval" / "units" / "AFEW_ALL_f0_r0_p5.npz")["gpu"])
sel = {"0": 0.0, "1": 2.5, "5": 5.0, "10": 0.0, "25": 5.0}       # p5t は β = 5 で解いたので K = 5・25 は 5．K = 10 は β = 0
frozen = {"_meta": {"pseudo_gpu": [gpu]},
          "AFEW": {"select": {K: {"beta": b} for K, b in sel.items()},
                   "select_student": {f"{s}|{K}": {"beta": b} for s in DD.STUDENTS for K, b in sel.items()},
                   "select_variants": {"srcw_mod|5": {"beta": 2.5}, "srcw_mod|25": {"beta": 0.0}, "srcw_src|5": {"beta": 10.0}, "srcw_src|25": {"beta": 20.0}},
                   "detect": {"cells": {}}}}
with open(fout / "select" / "frozen.json", "w") as fh:
    json.dump(frozen, fh)

# 偽の部分集合の表：IEMOCAP の marker・nomarker を AFEW と名づけ，テストの発話 40 本を asr_suspect にする
sub = pd.read_csv(ROOT / "DAI" / "frozen" / "subsets.csv", keep_default_na=False)
im = sub[sub.corpus == "IEMOCAP"].assign(corpus="AFEW")
tid = np.load(fout / "eval" / "units" / "AFEW_ALL_f0_r0_p5.npz")["sample_test"].astype(str)
asr = pd.DataFrame(dict(sample_id=tid[:40], corpus="AFEW", subset="asr_suspect", value="repeat"))
pd.concat([im, asr]).to_csv(froot / "DAI" / "frozen" / "subsets.csv", index=False)
S5.SUBSETS = {"AFEW": S5.SUBSETS["AFEW"] + S5.SUBSETS["IEMOCAP"]}

# 差し替え（本物の DAI/frozen には書かない）
_orig_others = DD.others
DD.CORPORA = ("AFEW",)
DD.others = lambda T: [c for c in ("AFEW", "IEMOCAP", "MOSEI", "MELD") if c not in (T, "IEMOCAP")] if T == "AFEW" else _orig_others(T)
orig_jload, orig_lm = S5.jload, ST.label_maps


def jload(p):
    p = Path(p)
    if p == D5.FROZEN1:
        return fz1
    if p == D5.GATE5:
        return {"input_sha256": {f"AFEW_ALL_f{f}_r0_p{w}.npz": dai1.sha(fin / "eval" / "units" / f"AFEW_ALL_f{f}_r0_p{w}.npz") for f in range(DD.N_FOLDS) for w in (1, 2, 3)}}
    return orig_jload(p)


S5.jload = jload
S5.ROOT = froot
S5.guard_eval = lambda out5, fz1_: "fakecommit"
S5.pairing = lambda *a: None
S5.recheck_sources = lambda *a: pd.DataFrame()
S5.shutil = types.SimpleNamespace(copy=lambda *a, **k: None)
ST.label_maps = lambda T: orig_lm("IEMOCAP")
S5.evaluate(fout, fin)

st = fout / "stats"
print("== 出力の形")
for nm in ("families7", "desc", "regret", "weights", "scores", "sensitivity", "recall", "confusion", "src_recall"):
    p = st / f"{nm}.csv"
    if not p.exists() or p.stat().st_size < 5:
        print(f"{nm:12s} 無いか空"); continue
    d = pd.read_csv(p, low_memory=False)
    num = d.select_dtypes("number")
    print(f"{nm:12s} 行 {len(d):5d} 列 {d.shape[1]:4d} 数の列の NaN {int(num.isna().sum().sum()):6d} Inf {int(np.isinf(num.to_numpy(float)).sum())}")

# ---- 独立の作り直し
y_of, cl_of = orig_lm("IEMOCAP")
units = [(f, 0) for f in range(DD.N_FOLDS)]
P5s = {u: D5.P5(fout / "eval" / "units" / f"AFEW_ALL_f{u[0]}_r0") for u in units}
UPs = {u: ST.UnitPreds(fin / "eval" / "units" / f"AFEW_ALL_f{u[0]}_r0") for u in units}


def uar_pooled(y, p):
    return 100.0 * float(np.mean([(p[y == k] == k).mean() for k in range(4) if (y == k).any()]))


def pred_of(u, row, s, K):
    if row == "srcw":
        return P5s[u].pred(s, K, sel[str(K)])
    if row == "srcw_b0":
        return P5s[u].pred(s, K, 0.0)
    jj = D5.joint_sel(fz1["AFEW"], s, K)
    return UPs[u].p3item(s, K, method="joint", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"])


# (1) desc の K = 0 の srcw − srcw_b0 に identical_by_construction
d = pd.read_csv(st / "desc.csv", low_memory=False)
k0 = d[(d.K == 0) & (d.B == "srcw_b0") & (d.A == "srcw") & (d.read.fillna("") == "")]
print("== (1) K = 0 の srcw − srcw_b0：行", len(k0), "identical_by_construction が真", int(k0.identical_by_construction.astype(bool).sum()))

# (2) fixed_point_pred_agree
fp = d[d.read == "fixed_point_check"]
mx = 0.0; nrow = 0
for _, rw in fp.iterrows():
    s, K = rw.student, int(rw.K)
    vals = []
    for u in units:
        it = [json.loads(x) for x in P5s[u].z[f"p5t_{s}_K{K}_items"]] if rw.A == "srcw" else [json.loads(x) for x in UPs[u].p3[f"p3t_{s}_K{K}_items"]]
        j = [i for i, x in enumerate(it) if np.isclose(x["Cw"], rw.Cw) and np.isclose(x["Cb"], rw.Cb) and (rw.A == "srcw" or x.get("tag") == "prop_sel")][0]
        tp = P5s[u].z[f"p5t_{s}_K{K}_pred"][j] if rw.A == "srcw" else UPs[u].p3[f"p3t_{s}_K{K}_pred"][j]
        vals.append(float((tp == pred_of(u, rw.A, s, K)).mean()))
    mx = max(mx, abs(np.mean(vals) - rw.fixed_point_pred_agree)); nrow += 1
lt1 = fp[fp.fixed_point_pred_agree < 1]
print("== (2) fixed_point_pred_agree：行", nrow, "NaN", int(fp.fixed_point_pred_agree.isna().sum()), "作り直しとの差の最大", mx,
      "1 未満の行", lt1[["student", "K", "A"]].to_dict("records"), "その行の counts_agree", lt1.fixed_point_counts_agree.tolist())

# (3) sensitivity：部分集合ごとの effect を作り直す（effect = 分割の平均の UAR の差．種は 1 つ）
sn = pd.read_csv(st / "sensitivity.csv")
fsub = pd.read_csv(froot / "DAI" / "frozen" / "subsets.csv", keep_default_na=False)
masks = {"asr_excluded": lambda sid: ~np.isin(sid, fsub[fsub.subset == "asr_suspect"].sample_id),
         "marker": lambda sid: np.isin(sid, fsub[(fsub.subset == "marker") & (fsub.value == "marker")].sample_id),
         "nomarker": lambda sid: np.isin(sid, fsub[(fsub.subset == "marker") & (fsub.value == "nomarker")].sample_id)}
mx, n, ndrop = 0.0, 0, 0
for _, rw in sn.iterrows():
    diffs = []
    for u in units:
        sid = UPs[u].sid
        m = masks[rw.subset](sid)
        y = y_of(sid)[m]
        if m.sum() < ST.MIN_TEST or (np.bincount(y, minlength=4) == 0).any():
            continue
        diffs.append(uar_pooled(y, pred_of(u, "srcw", rw.student, int(rw.K))[m]) - uar_pooled(y, pred_of(u, rw.B, rw.student, int(rw.K))[m]))
    ndrop += len(units) - len(diffs)
    if diffs and np.isfinite(rw.effect):
        mx = max(mx, abs(np.mean(diffs) - rw.effect)); n += 1
nmask = {k: int(sum(masks[k](UPs[u].sid).sum() for u in units)) for k in masks}
print("== (3) sensitivity：行", len(sn), "subset×student", sn.groupby("subset").student.nunique().to_dict(), "作り直した行", n, "effect の差の最大", mx,
      "外した分割の数（作り直し）", ndrop, "（台本の folds_dropped の非空の行）", int((sn.folds_dropped.fillna("").astype(str) != "").sum()),
      "部分集合の発話の数（5 分割の和）", nmask, "srcw_b0 の identical_by_construction", sn[sn.B == "srcw_b0"].identical_by_construction.tolist()[:2])

# (4) recall・confusion：混同行列を作り直す
cf = pd.read_csv(st / "confusion.csv"); rc = pd.read_csv(st / "recall.csv")
mx_cm, mx_rc = 0, 0.0
for _, rw in cf.iterrows():
    cm = np.zeros((4, 4), int)
    for u in units:
        y = y_of(UPs[u].sid)
        np.add.at(cm, (y, pred_of(u, rw.row, rw.student, int(rw.K)).astype(int)), 1)
    mine = np.array([[rw[f"cm_{a}{b}"] for b in range(4)] for a in range(4)])
    mx_cm = max(mx_cm, int(np.abs(cm - mine).max()))
    r2 = rc[(rc.student == rw.student) & (rc.K == rw.K) & (rc.row == rw.row)].iloc[0]
    mx_rc = max(mx_rc, max(abs(cm[k, k] / cm[k].sum() - r2[f"recall_{k}"]) for k in range(4)))
print("== (4) confusion：行", len(cf), "（期待 生徒 4 × K 2 × 行 3 = 24）混同行列の差の最大", mx_cm, "recall：行", len(rc), "再現率の差の最大", mx_rc,
      "n_test_total の一意", sorted(rc.n_test_total.unique().tolist()), "テストの総数", int(sum(len(UPs[u].sid) for u in units)))

# (5) src_recall：9 個（ここでは 6 個）の識別器の感情ごとの再現率と採点 1 位の Neutral
sr = pd.read_csv(st / "src_recall.csv")
names = [str(x) for x in P5s[units[0]].z["p5_src_names"]]
mx, mx_top = 0.0, 0.0
for _, rw in sr.iterrows():
    K = int(rw.K)
    RC, T0, M0 = [], [], []
    for u in units:
        y = y_of(UPs[u].sid); tp = P5s[u].z["p5_src_test_pred"].astype(int); sc = P5s[u].z[f"p5_K{K}_scores"]
        r_ = np.array([[(tp[j][y == k] == k).mean() for k in range(4)] for j in range(len(names))])
        RC.append(r_); T0.append(r_[sc >= sc.max() - 1e-12, 0].mean()); M0.append(np.median(r_[:, 0]))
    RC = np.mean(RC, 0)
    mx = max(mx, max(abs(RC[j, k] - rw[f"recall_{nm}_{k}"]) for j, nm in enumerate(names) for k in range(4)))
    mx_top = max(mx_top, abs(np.mean(T0) - rw.top_neutral_recall), abs(np.mean(M0) - rw.median_neutral_recall))
print("== (5) src_recall：行", len(sr), "K", sr.K.tolist(), "識別器の再現率の列", sum(c.startswith("recall_") for c in sr.columns), "（期待 6 × 4 = 24）",
      "差の最大", mx, "1 位・中央値の Neutral の差の最大", mx_top)

# (6) scores の n_clusters_K
sc_ = pd.read_csv(st / "scores.csv")
spl = pd.read_csv(ROOT / "DAI" / "splits" / "IEMOCAP.csv", keep_default_na=False)
clm = dict(zip(spl.sample_id.astype(str), spl.cluster.astype(str)))
bad = 0
for _, rw in sc_.iterrows():
    v = [len({clm[x] for x in P5s[u].z[f"sample_K{int(rw.K)}"].astype(str)}) for u in units]
    bad += int(not np.isclose(np.mean(v), rw.n_clusters_K_mean) or min(v) != rw.n_clusters_K_min)
print("== (6) scores の n_clusters_K：行", len(sc_), "K", sc_.K.tolist(), "作り直しとの食い違い", bad)

# (7) weights の cv_uar_・score_mean_
w = pd.read_csv(st / "weights.csv")
ss = pd.read_csv(ROOT / "results" / "dai1" / "source" / "summary.csv")
cvm = {f"{c}_{k}": v for c, k, v in ss[["corpus", "kind", "cv_uar"]].itertuples(index=False)}
cvbad = sum(int(not np.isclose(w[f"cv_uar_{nm}"].iloc[i], cvm[nm])) for nm in names for i in range(len(w)))
mx = 0.0
for _, rw in w.iterrows():
    K = int(rw.K)
    for j, nm in enumerate(names):
        if K == 0:
            mx = max(mx, 0.0 if np.isnan(rw[f"score_mean_{nm}"]) else np.inf)
        else:
            mx = max(mx, abs(np.mean([P5s[u].z[f"p5_K{K}_scores"][j] for u in units]) - rw[f"score_mean_{nm}"]))
print("== (7) weights：行", len(w), "cv_uar_ の列", sum(c.startswith("cv_uar_") for c in w.columns), "cv_uar の NaN", int(w[[f"cv_uar_{nm}" for nm in names]].isna().sum().sum()),
      "summary.csv との食い違い", cvbad, "score_mean_ の差の最大（K = 0 は NaN であること）", mx, "名前", names)
