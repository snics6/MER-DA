"""DAI-7 計算の後の点検 (1)(2)(3)：出力の一覧，行の数，鍵の重複，NaN の数と出どころ，provenance．
値そのもの（D・φ・UAR・判定）は出さない．出すのは数だけである．"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA")
OUT = ROOT / "results" / "dai7_probe_v2"
UD = ROOT / "results" / "dai1" / "pseudo" / "units"
CORP = ("AFEW", "IEMOCAP", "MELD", "MOSEI")
KS = (0, 1, 5, 10, 25)
KR = (1, 5, 10, 25)
MODS = "AVT"
RHOS = ("joint_a1_rho0.0625", "joint_a1_rho0.25", "joint_a1_rho1.0", "joint_a1_rho4.0")
SETS3 = ("none", "ret", "joint", "joint_a1", "oracle", *RHOS, "tt_poe_none", "tt_start_AVT", "tt_fused_AVT", "tt_ownret")
SETS2 = {"tt_single_A": "VT", "tt_single_V": "AT", "tt_single_T": "AV"}
TETHER = [(cw, cb) for cw in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0) for cb in (1.0, 10.0)]
TW = [f"none_tw{cw}_{cb}" for cw, cb in TETHER]
PAIRS = ("AV", "AT", "VT")
BOOT = ("none", "joint", "joint_a1", "oracle", RHOS[0], RHOS[2], RHOS[3], "tt_ownret")
POOL = (*BOOT, "ret")
VAR_OF = {"AFEW": ["cent", "keep", "sub_asr_ok"], "IEMOCAP": ["cent", "keep", "scen", "sub_marker", "sub_nomarker"],
          "MELD": ["cent", "keep", "sub_face_single"], "MOSEI": ["cent", "keep", "chan"]}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


units = sorted(f"{T}_{c}_f{f}_r{r}" for c in CORP for T in CORP if T != c for f in range(5) for r in range(2))
assert len(units) == 120
print("== (1) 出力の一覧")
want_top = {"rows", "fits", "cells.csv", "uar_cells.csv", "diffs.csv", "criteria.csv", "tether_weakening.csv",
            "tether_weakening_units.csv", "provenance.json"}
have_top = {p.name for p in OUT.iterdir()}
print("上の階：欠け", sorted(want_top - have_top), "余り", sorted(have_top - want_top))
want_rows = {f"{u}_{k}.csv" for u in units for k in ("units", "uar")}
have_rows = {p.name for p in (OUT / "rows").iterdir()}
print("rows：期待", len(want_rows), "ある", len(have_rows), "欠け", len(want_rows - have_rows), "余り", len(have_rows - want_rows))
want_fits = {f"{u}.npz" for u in units}
have_fits = {p.name for p in (OUT / "fits").iterdir()}
print("fits：期待", len(want_fits), "ある", len(have_fits), "欠け", len(want_fits - have_fits), "余り", len(have_fits - want_fits))
inp = {p.name[: -len("_p3.npz")] for p in UD.glob("*_p3.npz")}
print("入力の単位（p3）の数", len(inp), "期待の単位と同じ集合か", inp == set(units))

print("== (2) 行の数と鍵の重複")
U = pd.concat([pd.read_csv(OUT / "rows" / f"{u}_units.csv") for u in units], ignore_index=True)
A = pd.concat([pd.read_csv(OUT / "rows" / f"{u}_uar.csv") for u in units], ignore_index=True)
exp_u = set()
for u in units:
    T, c, fs, rs = u.split("_")
    for K in KS:
        for s in SETS3:
            for p in PAIRS:
                exp_u.add((T, c, int(fs[1:]), int(rs[1:]), K, s, p))
        for s, p in SETS2.items():
            exp_u.add((T, c, int(fs[1:]), int(rs[1:]), K, s, p))
        if K >= 1:
            for s in TW:
                for p in PAIRS:
                    exp_u.add((T, c, int(fs[1:]), int(rs[1:]), K, s, p))
ku = ["T", "c", "fold", "seed", "K", "set", "pair"]
got_u = set(map(tuple, U[ku].itertuples(index=False)))
print("units 行", len(U), "期待", len(exp_u), "鍵の重複", int(U.duplicated(ku).sum()), "欠け", len(exp_u - got_u), "余り", len(got_u - exp_u))
exp_a = {(k[0], k[1], k[2], k[3], k[4], k[5]) for k in exp_u if k[5] in SETS3}
ka = ["T", "c", "fold", "seed", "K", "set"]
got_a = set(map(tuple, A[ka].itertuples(index=False)))
print("uar 行", len(A), "期待", len(exp_a), "鍵の重複", int(A.duplicated(ka).sum()), "欠け", len(exp_a - got_a), "余り", len(got_a - exp_a))
print("単位ごとの units 行の数の種類", sorted(U.groupby(["T", "c", "fold", "seed"]).size().unique().tolist()),
      "uar 行の数の種類", sorted(A.groupby(["T", "c", "fold", "seed"]).size().unique().tolist()))
print("units の commit の種類", U["commit"].unique().tolist(), "dirty の種類", U["dirty"].unique().tolist())
print("uar の commit の種類", A["commit"].unique().tolist(), "dirty の種類", A["dirty"].unique().tolist())
print("units の列の数", U.shape[1], "uar の列の数", A.shape[1])

cells = pd.read_csv(OUT / "cells.csv")
acells = pd.read_csv(OUT / "uar_cells.csv")
Df = pd.read_csv(OUT / "diffs.csv")
C = pd.read_csv(OUT / "criteria.csv")
TWc = pd.read_csv(OUT / "tether_weakening.csv")
TWu = pd.read_csv(OUT / "tether_weakening_units.csv")
exp_c = set()
for c in CORP:
    for K in KS:
        for s in SETS3:
            for p in (*PAIRS, "all"):
                exp_c.add((c, K, s, p))
        for s, p in SETS2.items():
            exp_c.add((c, K, s, p)); exp_c.add((c, K, s, "all"))
        if K >= 1:
            for s in TW:
                for p in (*PAIRS, "all"):
                    exp_c.add((c, K, s, p))
kc = ["c", "K", "set", "pair"]
got_c = set(map(tuple, cells[kc].itertuples(index=False)))
print("cells 行", len(cells), "期待", len(exp_c), "鍵の重複", int(cells.duplicated(kc).sum()), "欠け", len(exp_c - got_c), "余り", len(got_c - exp_c))
print("cells の n_units の種類（pair が 3 つの対）", sorted(cells.loc[cells.pair != "all", "n_units"].unique().tolist()),
      "（pair = all）", sorted(cells.loc[cells.pair == "all", "n_units"].unique().tolist()))
exp_ac = {(c, K, s) for c in CORP for K in KS for s in SETS3}
got_ac = set(map(tuple, acells[["c", "K", "set"]].itertuples(index=False)))
print("uar_cells 行", len(acells), "期待", len(exp_ac), "鍵の重複", int(acells.duplicated(["c", "K", "set"]).sum()),
      "欠け", len(exp_ac - got_ac), "余り", len(got_ac - exp_ac))

# diffs の期待の鍵（設計 5 節と 4 節から数える）
neff = {}
for c in CORP:
    n = None
    cl = []
    for f in range(5):
        u = next(x for x in units if x.split("_")[1] == c and x.split("_")[2] == f"f{f}")
        cl.append(np.load(OUT / "fits" / f"{u}.npz")["test_cluster"].astype(str))
    v = pd.Series(np.concatenate(cl)).value_counts().to_numpy().astype(float)
    neff[c] = v.sum() ** 2 / (v ** 2).sum()
print("実効の塊の数（設計 3.3 節の 14.99，4.91，8.59，783 と照合）", {c: round(neff[c], 2) for c in CORP})
DDM = ("dD_joint_a1_none", "dD_joint_none", "dD_joint_a1_oracle", "dD_rho_hi_lo", "dD_rho1_ownret", "dD_ownret_none")
DDV = ("dD_joint_a1_none", "dD_joint_none", "dD_joint_a1_oracle", "dD_rho1_ownret")
exp_d = set()
for c in CORP:
    for K in KS:
        for p in (*PAIRS, "all"):
            for q in DDM:
                exp_d.add((c, K, "main", p, q, -1))
            for s in BOOT:
                exp_d.add((c, K, "main", p, f"D_{s}", -1))
        for q in ("dD_joint_a1_none", "dD_joint_a1_oracle"):
            for e in range(4):
                exp_d.add((c, K, "main", "all", q, e))
        for q in ("dphi_joint_a1_none", "dphi_joint_none", "dphi_rho1_ownret", "dphi_ownret_none"):
            exp_d.add((c, K, "main", "all", q, -1))
        for s in ("joint", "joint_a1", RHOS[2]):
            exp_d.add((c, K, "main", "all", f"dLdep_{s}_none", -1))
        for s in ("none", "joint", "joint_a1", RHOS[2]):
            exp_d.add((c, K, "main", "all", f"Ldep_{s}", -1))
        for s in ("joint", "joint_a1", RHOS[2]):
            exp_d.add((c, K, "main", "all", f"G_{s}", -1))
        for s in ("joint_a1", RHOS[2]):
            for q in ("S", "S1", "Scal", "S1b", "dLdep", "dLcal", "share_dep"):
                exp_d.add((c, K, "main", "all", f"{q}_{s}", -1))
        exp_d.add((c, K, "main", "all", "dDt_joint_a1_none", -1))
        if neff[c] < 20:
            for q in ("dD_joint_a1_none", "dD_joint_a1_oracle", "dD_rho1_ownret"):
                exp_d.add((c, K, "jk", "all", q, -1))
        for v in VAR_OF[c] + (["spk"] if c == "IEMOCAP" else []):
            for p in (*PAIRS, "all"):
                for q in DDV:
                    exp_d.add((c, K, v, p, q, -1))
Df["_e"] = Df["emotion"].fillna(-1).astype(int)
kd = ["c", "K", "variant", "pair", "quantity", "_e"]
got_d = set(map(tuple, Df[kd].itertuples(index=False)))
print("diffs 行", len(Df), "期待", len(exp_d), "鍵の重複", int(Df.duplicated(kd).sum()), "欠け", len(exp_d - got_d), "余り", len(got_d - exp_d))
print("diffs の variant の種類", sorted(Df.variant.unique().tolist()))
print("diffs の n_boot_valid（区間のある行）：最小と最大", int(Df.n_boot_valid.dropna().min()), int(Df.n_boot_valid.dropna().max()),
      "main で 1000 未満の行", int(((Df.variant == "main") & (Df.n_boot_valid < 1000) & Df.lo.notna()).sum()))
print("criteria 行", len(C), "c の集合", sorted(C.c.tolist()), "列の数", C.shape[1])
exp_tw = {(c, K, m) for c in CORP for K in KR for m in MODS}
print("tether_weakening 行", len(TWc), "期待", len(exp_tw), "鍵の重複", int(TWc.duplicated(["c", "K", "mod"]).sum()),
      "欠け", len(exp_tw - set(map(tuple, TWc[["c", "K", "mod"]].itertuples(index=False)))))
exp_twu = {(u.split("_")[0], u.split("_")[1], int(u.split("_")[2][1:]), int(u.split("_")[3][1:]), K, m) for u in units for K in KR for m in MODS}
ktw = ["T", "c", "fold", "seed", "K", "mod"]
print("tether_weakening_units 行", len(TWu), "期待", len(exp_twu), "鍵の重複", int(TWu.duplicated(ktw).sum()),
      "欠け", len(exp_twu - set(map(tuple, TWu[ktw].itertuples(index=False)))))
for nm, df in (("cells", cells), ("uar_cells", acells), ("diffs", Df), ("criteria", C), ("tether_weakening", TWc), ("tether_weakening_units", TWu)):
    print(f"{nm}：commit {df['commit'].unique().tolist()} summarize_commit {df['summarize_commit'].unique().tolist()} "
          f"summarize_dirty {df['summarize_dirty'].unique().tolist()} boot_seed {df['boot_seed'].unique().tolist()} n_boot {df['n_boot'].unique().tolist()}")

print("== fits の配列の鍵")
bad = 0
for u in units:
    T, c, fs, rs = u.split("_")
    z = np.load(OUT / "fits" / f"{u}.npz")
    keys = set(z.files)
    e = {"test_sample_id", "test_cluster", "test_group", "test_y", "half", "pool_sample_id", "pool_y", "commit"}
    e |= {f"mask_{v}" for v in VAR_OF[c]}
    if c == "IEMOCAP":
        e.add("test_scenario")
    for K in KS:
        e |= {f"K{K}_pool_kpos", f"K{K}_teacher_t"}
        for s in SETS3:
            e |= {f"K{K}_{s}_z", f"K{K}_{s}_mods", *[f"K{K}_{s}_pred_{q}" for q in (*MODS, "prod", "blk", "full", "lda_A", "lda_V", "lda_T")]}
        for s, p in SETS2.items():
            e |= {f"K{K}_{s}_z", f"K{K}_{s}_mods", *[f"K{K}_{s}_pred_{m}" for m in p]}
        for s in POOL:
            e.add(f"K{K}_{s}_zpool")
        if K >= 1:
            for s in TW:
                e |= {f"K{K}_{s}_z", f"K{K}_{s}_mods", *[f"K{K}_{s}_pred_{m}" for m in MODS]}
    if keys != e:
        bad += 1
        print(u, "欠け", sorted(e - keys)[:5], "余り", sorted(keys - e)[:5])
    if str(z["commit"]) != U["commit"].iloc[0]:
        bad += 1; print(u, "fits の commit が違う")
    nt = len(z["test_y"])
    for K in KS:
        for s in SETS3:
            if z[f"K{K}_{s}_z"].shape != (nt, 9):
                bad += 1
    nonfin = sum(int((~np.isfinite(z[k].astype(float))).sum()) for k in z.files if k.endswith(("_z", "_zpool", "_teacher_t")))
    if nonfin:
        bad += 1; print(u, "fits の点数に有限でない値", nonfin)
print("fits：鍵・形・commit・有限の不一致の単位の数", bad)

print("== NaN の数")
for nm, df in (("units", U), ("uar", A), ("cells", cells), ("uar_cells", acells), ("diffs", Df.drop(columns="_e")), ("criteria", C),
               ("tether_weakening", TWc), ("tether_weakening_units", TWu)):
    num = df.select_dtypes("number")
    print(f"{nm}：NaN {int(df.isna().sum().sum())}，Inf {int(np.isinf(num.to_numpy()).sum())}")
# uar の範囲
uc = [x for x in A.columns if x.startswith("uar_")]
rc = [x for x in A.columns if x.startswith("rec_")]
print("uar：UAR が 0〜100 の外のセル", int(((A[uc] < 0) | (A[uc] > 100)).sum().sum()), "再現率が 0〜1 の外", int(((A[rc] < 0) | (A[rc] > 1)).sum().sum()),
      "NaN（uar_* と rec_*）", int(A[uc + rc].isna().sum().sum()))
print("units：D_raw（d の平均）が 0〜1 の外（有限の行）", int(((U.D_raw < -1e-9) | (U.D_raw > 1 + 1e-9)).sum()),
      "phi が −1〜1 の外", int(((U.phi < -1 - 1e-9) | (U.phi > 1 + 1e-9)).sum()))
for e in range(4):
    de = U[f"d_e{e}"]
    print(f"units：d_e{e} が 0〜1 の外", int(((de < -1e-9) | (de > 1 + 1e-9)).sum()))

print("== (3) provenance")
P = json.load(open(OUT / "provenance.json"))
print("runs の数", len(P["runs"]))
r = P["runs"][0]
print("commit", r["commit"], "dirty", r["dirty"], "argv", r["argv"], "gpu", r["gpu"])
print("版", {k: r[k] for k in ("python", "torch", "numpy", "pandas", "scipy", "cuda")})
print("鍵の一覧", sorted(k for k in r if k != "units"))
print("units の数", len(r["units"]), "名前の集合が期待と同じか", {x["unit"] for x in r["units"]} == set(units))
mis = 0
for x in r["units"]:
    for i in (1, 2, 3):
        if sha(UD / f"{x['unit']}_p{i}.npz") != x["inputs_sha256"][f"p{i}"]:
            mis += 1
print("入力 p1〜p3 の sha256 が今のファイルと違う数（360 個のうち）", mis)
fz = ROOT / "DAI" / "frozen" / "dai1_frozen.json"
print("固定のファイルの sha256 が provenance と同じか", sha(fz) == r["frozen_sha256"])
gfz = subprocess.run(["git", "show", f"{r['commit']}:DAI/frozen/dai1_frozen.json"], cwd=ROOT, capture_output=True).stdout
print("固定のファイルの sha256 がコミットの版と同じか", hashlib.sha256(gfz).hexdigest() == r["frozen_sha256"])
gs = subprocess.run(["git", "show", f"{r['commit']}:DAI/exp/dai7_probe_v2.py"], cwd=ROOT, capture_output=True).stdout
gd = subprocess.run(["git", "show", "d0502d6:DAI/exp/dai7_probe_v2.py"], cwd=ROOT, capture_output=True).stdout
print("台本の sha256：コミット 1c9f7b2 の版 = d0502d6 の版", hashlib.sha256(gs).hexdigest() == hashlib.sha256(gd).hexdigest(),
      "= 今の作業木", hashlib.sha256(gs).hexdigest() == sha(ROOT / "DAI/exp/dai7_probe_v2.py"), hashlib.sha256(gs).hexdigest()[:16])
print("provenance に台本の sha256 の鍵があるか", any("script" in k for k in r))
# pseudo_commit の照合（台本 340 行が止める条件．記録として数え直す）
F = json.load(open(fz))
pc = 0
for x in r["units"]:
    T = x["unit"].split("_")[0]
    for i in (1, 2, 3):
        if str(np.load(UD / f"{x['unit']}_p{i}.npz")["commit"]) != F[T]["pseudo_commit"]:
            pc += 1
print("入力の commit が固定のファイルの pseudo_commit と違う数", pc)
print("所要（単位の段の合計，分）", round(sum(x["sec"] for x in r["units"]) / 60, 1))
