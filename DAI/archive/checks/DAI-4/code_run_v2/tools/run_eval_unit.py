"""評価の段の solve（tether_sel あり）を 1 単位で流す。固定のファイルは偽（選んだ点を (s,K) ごとに変えた）。出力は out_e の下。results/ には書かない。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, torch
import dai1, dai4 as D4, dai_data as DD
T = sys.argv[1]; f = int(sys.argv[2]); r = int(sys.argv[3])
out = Path("/home/snics/MER-DA/DAI/checks/DAI-4/code_run_v2/out_e")
fz1 = json.load(open(D4.FROZEN1))
sel = {}
for i, s in enumerate(DD.STUDENTS):
    for j, K in enumerate(dai1.KS_ALL):
        sel[f"{s}|{K}"] = dict(rho=D4.RHO_GRID[(i + j) % 4], gamma=D4.GAMMA_GRID[(i + 2 * j) % 3])
json.dump(sel, open(out.parent / "fake_sel.json", "w"))
tsel = lambda s_, K_: (sel[f"{s_}|{K_}"]["rho"], sel[f"{s_}|{K_}"]["gamma"])
gammas = D4.GAMMA_GRID
run = dict(commit=dai1.git_commit()[0], dirty=dai1.git_commit()[1], frozen_sha256="x", dai1_frozen_sha256=dai1.sha(D4.FROZEN1), gate_sha256="g", gpu=torch.cuda.get_device_name(0))
ctrl_pts = lambda meth, s_, K_: [(float(r_), float(g_)) for r_ in D4.RHO_GRID for g_ in gammas]
tether = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in fz1[T]["tether"].items()}
for c, srcs, ff, rr in dai1.units_for(T, "eval", [f], [r]):
    # eval の再開照合は run["dai1_frozen_sha256"] と z1["frozen_sha256"] の一致を求める（実物と同じ）
    D4.run_unit(D4.IN_DIR, out, "eval", T, c, srcs, ff, rr, tether, gammas, run, ctrl_pts, False, tsel)
