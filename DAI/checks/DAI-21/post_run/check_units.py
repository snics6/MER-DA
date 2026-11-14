"""DAI-21 計算の後の点検 1：単位の出力の数・行・鍵・問・重み・NaN/Inf・由来・検算の量（読むだけ）．値（UAR）は出さない．"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA")
R = ROOT / "results" / "dai21_eval5"
OUT20 = ROOT / "results" / "dai20_eval" / "units"
fz = json.load(open(ROOT / "DAI" / "frozen" / "dai21_frozen.json"))
C4 = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
KS = (5, 10, 25)
EXTRA_T = ("noprof", "noint", "levelfree", "konly", "tcal")
EX1 = ("kd", "agree", "mix")
COMMIT0 = "07c5d6726d3da45ed701ad05abd66629936bfff5"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def specs(gm, ex1):
    s = [f"stud_{t}_g{g}" for t in ("prod", "prop") for g in range(12)] + [f"stud_{t}_g{gm}" for t in EXTRA_T]
    if ex1:
        s += [f"{m}_{t}" for m in EX1 for t in ("prod", "prop")]
    return s


prob = []
res = {}
for part, names, nrow, sub in (("students4", [f"{T}_ALL_f{f}_r{r}" for T in C4 for f in range(5) for r in range(10)], 108, "students4"),
                               ("cremad", [f"CREMA-D_ALL_f{f}_r{r}" for f in range(5) for r in range(10)], 276, "cremad")):
    d = R / sub
    csvs, npzs = sorted(p.stem for p in d.glob("*.csv")), sorted(p.stem for p in d.glob("*.npz"))
    exp_ = sorted(names)
    info = dict(n_csv=len(csvs), n_npz=len(npzs), csv_eq_expected=csvs == exp_, npz_eq_expected=npzs == exp_)
    rows_bad, dup, kset, commits, dirty, nan_uar, inf_any, uar_out, Wnf, keymiss, lenbad, itembad, shabad, gridbad = 0, 0, set(), set(), set(), 0, 0, 0, 0, 0, 0, 0, 0, 0
    nkeys = set()
    allck = []
    gpu_mb = []
    srcbad = 0
    for nm in exp_:
        c = pd.read_csv(d / f"{nm}.csv", low_memory=False)
        if len(c) != nrow:
            rows_bad += 1
        dup += int(c.duplicated(["K", "version", "grid"]).sum()) if "grid" in c else int(c.duplicated(["K", "version"]).sum())
        kset |= set(c.K.unique().tolist())
        commits |= set(c.commit.astype(str)); dirty |= set(c.dirty.astype(str))
        num = c.select_dtypes(include=[np.number])
        inf_any += int(np.isinf(num.to_numpy()).sum())
        for col in ("uar", "uar_raw", "uar_adj", "uar_pool"):
            if col in c:
                v = c[col].dropna()
                uar_out += int(((v < 0) | (v > 100)).sum())
        # 学び直しの行の uar は NaN でない
        st = c[c.version.astype(str).str.startswith(("stud_",) + tuple(f"{m}_" for m in EX1)) & (c.method.notna() if "method" in c else True)]
        nan_uar += int(st[["uar", "uar_pool", "mimic", "teacher_uar"]].isna().sum().sum())
        z = np.load(d / f"{nm}.npz", allow_pickle=False)
        nkeys.add(len(z.files))
        nt = len(z["y_test"])
        for k in z.files:
            a = z[k]
            if a.dtype.kind == "f" and not np.isfinite(a).all():
                Wnf += 1
            if k.startswith(("pred_K", "pred_raw_K", "pred_adj_K")) and len(a) != nt:
                lenbad += 1
        ck = c[c.version == "_check"].copy()
        ck["unit"] = nm
        allck.append(ck)
        if part == "cremad":
            gpu_mb += ck.gpu_max_mb.tolist()
            for kind in ("A", "V", "AV"):
                if sorted(map(str, z[f"srcs_{kind}"])) != sorted(C4):
                    srcbad += 1
        if part == "students4":
            T = nm.split("_ALL_")[0]
            if sha(OUT20 / f"{nm}.npz") != ck.dai20_npz_sha256.iloc[0] or ck.dai20_npz_sha256.nunique() != 1:
                shabad += 1
        for K in KS:
            q = fz["four"][nm.split("_ALL_")[0]][str(K)] if part == "students4" else fz["cremad"][str(K)]
            gm = int(q["student_main"])
            exp_items = specs(gm, part == "students4")
            items = [str(x) for x in z[f"stud_items_K{K}"]]
            if items != exp_items:
                itembad += 1
            W, b = z[f"studW_K{K}"], z[f"studb_K{K}"]
            if W.shape[0] != len(exp_items) or b.shape != (len(exp_items), 4) or W.shape[1] != 4:
                itembad += 1
            for it in exp_items:
                for pre in ("pred_K", "predpool_K"):
                    if f"{pre}{K}_{it}" not in z.files:
                        keymiss += 1
            rk = ck[ck.K == K]
            if len(rk) != 1 or int(rk.student_main.iloc[0]) != gm or int(rk.prop_grid.iloc[0]) != int(q["prop_grid"]):
                gridbad += 1
            sk = c[(c.K == K) & (c.version != "_check")]
            # 学び直しの行の数：stud の 2×12 + 5 (+6)
            nst = int(sk.version.astype(str).str.startswith(("stud_",) + tuple(f"{m}_p" for m in EX1)).sum()) if part == "students4" else int(sk.version.astype(str).str.startswith("stud_").sum())
            if nst != len(exp_items):
                itembad += 1
    ck = pd.concat(allck, ignore_index=True)
    ck["c"] = ck.unit.str.split("_ALL_").str[0]
    info.update(rows_bad=rows_bad, dup_keys=dup, K_values=sorted(int(k) for k in kset), commits=sorted(commits), dirty=sorted(dirty), uar_nan_in_student_rows=nan_uar,
                inf_cells=inf_any, uar_out_of_range=uar_out, npz_nonfinite_arrays=Wnf, npz_n_keys=sorted(nkeys), pred_len_bad=lenbad, items_or_shape_bad=itembad,
                pred_keys_missing=keymiss, check_row_grid_mismatch=gridbad, n_check_rows=len(ck), grad_max=float(ck.grad.max()), stalled=int(ck.stalled.sum()))
    if part == "students4":
        info["dai20_npz_sha_mismatch"] = shabad
        rp = {}
        for col in ("joint_repro", "kd_repro", "agree_repro", "mix_repro", "agree_min"):
            rp[col] = dict(min=float(ck[col].min()), n_below_1=int((ck[col] < 1).sum()), n_below_0995=int((ck[col] < 0.995).sum()),
                           by_c_K_below_1={f"{c_}|{K}": int(n) for (c_, K), n in ck[ck[col] < 1].groupby(["c", "K"]).size().items()})
        info["repro"] = rp
        # 一致の割合を発話の数に直す（1 発話の違いがいくつか）
        info["joint_repro_below_1_detail"] = [dict(unit=r.unit, K=int(r.K), agree=float(r.joint_repro)) for r in ck[ck.joint_repro < 1].itertuples()]
        info["kd_repro_below_1_detail"] = [dict(unit=r.unit, K=int(r.K), agree=float(r.kd_repro)) for r in ck[ck.kd_repro < 1].itertuples()]
    else:
        info.update(gpu_max_mb_max=float(max(gpu_mb)), srcs_not_4corpora=srcbad)
    res[part] = info

json.dump(res, sys.stdout, indent=1, ensure_ascii=False)
print()
