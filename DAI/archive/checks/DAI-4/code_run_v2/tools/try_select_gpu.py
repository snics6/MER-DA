import sys, json, shutil, os
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd
import dai1, dai4 as D4, dai_data as DD
CR = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run_v2"); OLD = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run"); fake = CR / "fake_sel"
shutil.rmtree(fake, ignore_errors=True)
srcs = {"IEMOCAP": OLD / "out2/a/pseudo/units/IEMOCAP_MOSEI_f4_r1_p4", "AFEW": OLD / "out2/b/pseudo/units/AFEW_MELD_f1_r0_p4"}
srcs["MELD"] = srcs["MOSEI"] = srcs["IEMOCAP"]
fz1 = json.load(open(D4.FROZEN1)); (fake / "pseudo/units").mkdir(parents=True)
dai1.git_commit = lambda: ("abc", False)
rng = np.random.default_rng(1)
def write(gpu_of_unit):
    for T in DD.CORPORA:
        for i, n in enumerate(fz1[T]["units"]):
            c, f, r = n.split("_"); d = pd.read_csv(f"{srcs[T]}.csv", low_memory=False)
            d = d[~(np.isclose(d.gamma.astype(float), 1.0) & d.method.isin(["cons", "cons_self", "cons_own", "cons_conf", "cons_rand", "_weights"]))]
            d["T"], d["c"], d["fold"], d["seed"] = T, c, int(f[1:]), int(r[1:]); d["commit"] = "abc"; d["dirty"] = False
            d["dai1_frozen_sha256"] = dai1.sha(D4.FROZEN1); d["uar"] = d["uar"] + rng.normal(0, 0.3, len(d)); d["gpu"] = gpu_of_unit(T, i)
            d.to_csv(fake / "pseudo/units" / f"{T}_{n}_p4.csv", index=False)
            p = fake / "pseudo/units" / f"{T}_{n}_p4.npz"
            if not p.exists(): os.symlink(f"{srcs[T]}.npz", p)
for label, fn in (("全単位 1 種", lambda T, i: "NVIDIA H200"), ("1 単位だけ別の計算機", lambda T, i: "NVIDIA A5000" if (T == "MELD" and i == 3) else "NVIDIA H200")):
    write(fn)
    try:
        D4.select_stage(fake, list(DD.CORPORA), fz1); print(f"[{label}] 通った meta.pseudo_gpu =", json.load(open(fake / "select/frozen.json"))["_meta"]["pseudo_gpu"])
    except SystemExit as e:
        print(f"[{label}] SystemExit:", e)
