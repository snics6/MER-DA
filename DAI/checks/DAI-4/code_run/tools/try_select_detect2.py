"""最新版の select と detect の経路を，1 単位の保存を 30 単位 × 4 T に複製した偽の置き場で流す（値は意味を持たない．経路・キー・守りの確認だけ）。"""
import sys, json, shutil, os
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd
import dai1, dai4 as D4, dai4_stats as D, dai_data as DD, dai1_stats as ST
CR = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run"); fake = CR / "out2/fake_all"
shutil.rmtree(fake, ignore_errors=True)
srcs = {"IEMOCAP": CR / "out2/a/pseudo/units/IEMOCAP_MOSEI_f4_r1_p4", "AFEW": CR / "out2/b/pseudo/units/AFEW_MELD_f1_r0_p4"}
srcs["MELD"] = srcs["MOSEI"] = srcs["IEMOCAP"]
fz1 = json.load(open(D4.FROZEN1)); (fake / "pseudo/units").mkdir(parents=True)
dai1.git_commit = lambda: ("abc", False)
def write(T, drop_g1, jitter):
    rng = np.random.default_rng(1)
    for n in fz1[T]["units"]:
        c, f, r = n.split("_"); d = pd.read_csv(f"{srcs[T]}.csv", low_memory=False)
        if drop_g1: d = d[~(np.isclose(d.gamma.astype(float), 1.0) & d.method.isin(["cons", "cons_self", "cons_own", "cons_conf", "cons_rand", "_weights"]))]
        d["T"], d["c"], d["fold"], d["seed"] = T, c, int(f[1:]), int(r[1:]); d["commit"] = "abc"; d["dirty"] = False
        d["dai1_frozen_sha256"] = dai1.sha(D4.FROZEN1)
        if jitter: d["uar"] = d["uar"] + rng.normal(0, 0.3, len(d))
        d.to_csv(fake / "pseudo/units" / f"{T}_{n}_p4.csv", index=False)
        p = fake / "pseudo/units" / f"{T}_{n}_p4.npz"
        if not p.exists(): os.symlink(f"{srcs[T]}.npz", p)
for drop in (False, True):
    for T in DD.CORPORA: write(T, drop, True)
    try:
        D4.select_stage(fake, list(DD.CORPORA), fz1); print("drop_g1 =", drop, "-> 通った")
    except SystemExit as e:
        print("drop_g1 =", drop, "-> SystemExit:", e)
fz = json.load(open(fake / "select/frozen.json"))
T = "MELD"; ks = sorted(fz[T]["select"]); print("select キーの数", len(ks), ks[:2], "…", sorted({k.split('|')[2] for k in ks if k.count('|') == 2}))
print("pseudo_unit_sha256 の数", len(fz[T]["pseudo_unit_sha256"]), "最初のキー", next(iter(fz[T]["pseudo_unit_sha256"])))
print("セル数/T", {T: fz[T]["n_cells"] for T in DD.CORPORA}, " 期待: cons 20 + rho25 20 + self 15 + own/conf/rand 60 = 115")
# detect（collect を 1 単位の複製に差し替える）
real_collect = D.collect
cache = {}
def fake_collect(out4, in_dir, tag, T, c, units, y_of, cl_of, sel_cons, fz1T, **kw):
    if "base" not in cache:
        cells, m, t = real_collect(CR / "out2/a", D4.IN_DIR, "pseudo", "IEMOCAP", "MOSEI", [(4, 1)], *ST.label_maps("MOSEI"), {(s, K): (1.0, 0.25) for s in DD.STUDENTS for K in dai1.KS_ALL}, fz1["IEMOCAP"], check_rows=True)
        cache["base"] = (cells, m, t)
    cells, m, t = cache["base"]
    out = {k: {u: v[(4, 1)] for u in units} for k, v in cells.items()}
    return out, m, t
D.collect = fake_collect
D.detect(fake, D4.IN_DIR, list(DD.CORPORA))
fz = json.load(open(fake / "select/frozen.json"))
print("detect 出力のある T:", [T for T in DD.CORPORA if "detect" in fz[T]], " meta:", {k: fz["_meta"][k] for k in ("detect", "detect_commit")})
c0 = fz["AFEW"]["detect"]["cells"]["6|A|5"]; print("detect セルのキー", sorted(c0), " descriptive 数", len(fz["AFEW"]["detect"]["descriptive_cells"]))
