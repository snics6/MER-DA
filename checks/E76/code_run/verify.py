"""gate-code の検算（E76 関門 1）。E76 の小さく流した出力を，台本とは別の道筋で確かめる。

python checks/E76/code_run/verify.py <out_dir> <kind> <target>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import ablate  # noqa: E402
import shift  # noqa: E402
from data import load_any  # noqa: E402

out, kind, T = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
z = np.load(out / kind / f"{T}.npz")
d = pd.read_csv(out / kind / f"{T}.csv", comment="#", dtype={"K": str, "source": str})
d["K"], d["source"] = d.K.fillna(""), d.source.fillna("")
meta = json.loads((out / kind / f"{T}.json").read_text())
Ks = meta["settings"]["K"]
CG = meta["settings"]["C_grid"]
reps = list(z["reps"])
res = {}

# 1. 発話の並びとラベル：load_any と発話一覧（sample_id で貼る）
df, X = load_any(T, kind)
res["sid_same_order"] = bool((df.sample_id.astype(str).to_numpy() == z["sample_id"]).all())
corp = {"IEMOCAP-impro": "iemocap", "IEMOCAP-script": "iemocap", "MOSEI": "mosei", "MELD": "meld",
        "AFEW": "afew", "RAVDESS": "ravdess", "CREMA-D": "crema-d"}[T]
man = pd.read_csv(ROOT / "data" / "metadata" / f"{corp}_utterance_manifest.csv", low_memory=False)
lab = man.set_index("sample_id").label_4.reindex(z["sample_id"]).to_numpy()
res["y_eq_manifest"] = bool((lab == z["y"]).all())
res["speaker_eq"] = bool((df.speaker_group.astype(str).to_numpy() == z["speaker_group"]).all())

# 2. テストと K 本：話者の分離，入れ子，感情ごとの本数
sp = z["speaker_group"]
ok_split = []
for r in reps:
    te = z[f"test_idx_r{r}"]
    prev = set()
    for K in [k for k in Ks if k != "all"]:
        ki = z[f"kidx_r{r}_K{K}"]
        cnt = np.bincount(z["y"][ki], minlength=4)
        ok_split.append(dict(r=r, K=K, disjoint_rows=len(set(ki) & set(te)) == 0,
                             disjoint_spk=len(set(sp[ki]) & set(sp[te])) == 0,
                             per_class=bool((cnt == int(K)).all()), nested=prev <= set(ki)))
        prev = set(ki)
res["split_all_ok"] = all(all(v for k, v in o.items() if k not in ("r", "K")) for o in ok_split)

# 3. 保存した点数から UAR を作り直し，csv の uar と照合．地図（C 固定）を自分で読んで照合
def uar(L, y):
    return shift.uar(y, L.argmax(-1))

fm = pd.read_csv(ROOT / "results" / "fixc_map" / kind / f"{T}.csv", comment="#", dtype={"K": str, "source": str})
fm["K"], fm["source"] = fm.K.fillna(""), fm.source.fillna("")
fmi = fm.set_index(["method", "source", "K", "rep"])
diffs = {"tgt_only": [], "src_only": [], "merged": [], "h": [], "map_tgt": [], "map_src": [], "map_merged": []}
crule = []
y = z["y"]
for r in reps:
    te = z[f"test_idx_r{r}"]
    for kk, K in enumerate(Ks):
        for jc, c in enumerate(CG):
            u = uar(z[f"t_logit_r{r}"][kk, jc], y[te])
            row = d[(d.method == "tgt_only") & (d.K == K) & (d.rep == r)].iloc[0]
            diffs["tgt_only"].append(abs(u - row[f"test_uar_C{c:g}"]))
        # 選択の規則を交差検証の UAR から作り直す（map.choose_c の規則を自分で書く）
        cv = z[f"t_cv_r{r}"][kk]
        if np.isnan(cv).all():
            c_re, rule = 1.0, "fixed"
        else:
            best = cv.max()
            tied = [c for c, v in zip(CG, cv) if abs(v - best) <= 1e-12]
            tied.sort(key=lambda c: (round(abs(np.log10(c)), 9), c))
            c_re, rule = tied[0], "tie" if len(tied) > 1 else "cv"
        crule.append(c_re == z[f"t_C_r{r}"][kk] and rule == z[f"t_rule_r{r}"][kk])
        mrow = fmi.loc[("tgt_only", "", K, r)]
        diffs["map_tgt"].append(abs(uar(z[f"t_logit_r{r}"][kk, CG.index(float(z[f't_C_r{r}'][kk]))], y[te]) - mrow.uar))
sources = sorted({k.split("__")[0][2:] for k in z.files if k.startswith("s_") and k.endswith("__ver")})
for s in sources:
    ver = z[f"s_{s}__ver"]
    for ir, r in enumerate(reps):
        te = z[f"test_idx_r{r}"]
        j = ver[ir]
        C = float(z[f"s_{s}__v{j}_C"])
        Ls = z[f"s_{s}__v{j}_logit"][CG.index(C)]
        u = uar(Ls[te], y[te])
        row = d[(d.method == "src_only") & (d.source == s) & (d.rep == r)].iloc[0]
        diffs["src_only"].append(abs(u - row.uar))
        diffs["map_src"].append(abs(u - fmi.loc[("src_only", s, "", r)].uar))
        Lm = z[f"m_{s}_r{r}_logit"]
        for kk, K in enumerate(Ks):
            L = Lm[kk] if Lm.ndim == 3 else Lm[kk, CG.index(C)]
            u = uar(L, y[te])
            diffs["merged"].append(abs(u - d[(d.method == "merged") & (d.source == s) & (d.K == K) & (d.rep == r)].uar.iloc[0]))
            diffs["map_merged"].append(abs(u - fmi.loc[("merged", s, K, r)].uar))
        # 手 4・5：保存したずらしで作り直す．ずらしを fit_class_bias で解き直す
        if f"h4_{s}_r{r}_b" in z.files:
            L2 = z[f"h2_{s}__v{j}_logit"]
            for kh, K in enumerate(("1", "5", "10")):
                ki = z[f"kidx_r{r}_K{K}"]
                for h, base in ((4, Ls), (5, L2)):
                    b = z[f"h{h}_{s}_r{r}_b"][kh]
                    u = uar(base[te] + b, y[te])
                    diffs["h"].append(abs(u - d[(d.method == f"h{h}") & (d.source == s) & (d.K == K) & (d.rep == r)].uar.iloc[0]))
                    b2 = ablate.fit_class_bias(base[ki].astype(np.float64), y[ki])
                    diffs["h"].append(float(np.abs(b2 - b).max()) * 0)   # b の再現は別に記録
                    res.setdefault("h45_b_refit_maxdiff", 0.0)
                    res["h45_b_refit_maxdiff"] = max(res["h45_b_refit_maxdiff"], float(np.abs(b2 - b).max()))
            for h in (3, 6, 7, 8):
                for kh, K in enumerate(("1", "5", "10")):
                    u = uar(z[f"h{h}_{s}_r{r}_logit"][kh], y[te])
                    diffs["h"].append(abs(u - d[(d.method == f"h{h}") & (d.source == s) & (d.K == K) & (d.rep == r)].uar.iloc[0]))
res.update({f"maxdiff_{k}": (float(max(v)) if v else None) for k, v in diffs.items()})
res["c_rule_reproduced"] = f"{sum(crule)}/{len(crule)}"

# 4. 係数から全発話の点数を作り直す（② と h2．台本の coef_check はテストの ①・③・h3 だけ）
mx = 0.0
for s in sources:
    for j in sorted(set(z[f"s_{s}__ver"])):
        for jc in range(len(CG)):
            L = X @ z[f"s_{s}__v{j}_A"][jc].T.astype(np.float64) + z[f"s_{s}__v{j}_a0"][jc]
            mx = max(mx, float(np.abs(L - z[f"s_{s}__v{j}_logit"][jc]).max()))
        if f"h2_{s}__v{j}_logit" in z.files:
            L = (X - z["mean_t"]) @ z[f"h2_{s}__v{j}_A"].T.astype(np.float64) + z[f"h2_{s}__v{j}_a0"]
            mx = max(mx, float(np.abs(L - z[f"h2_{s}__v{j}_logit"]).max()))
res["coef_rebuild_src_h2_maxabs"] = mx

# 5. csv の健全さ
num = d.select_dtypes("number")
res["csv_rows"] = len(d)
res["csv_nan_uar"] = int(d.uar.isna().sum())
res["csv_inf"] = int(np.isinf(num.to_numpy(dtype=float, na_value=0)).sum())
res["csv_dup_keys"] = int(d.duplicated(["method", "source", "K", "rep"]).sum())
res["npz_nonfinite"] = int(sum((~np.isfinite(z[k])).sum() for k in z.files if z[k].dtype.kind == "f"))
res["rows_by_method"] = d.groupby("method").size().to_dict()
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
