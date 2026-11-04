"""gate-code の検算：族 3 の相手が共訓練・共正則化のときの W・b の読み方（試走の選択では kd と agree だけが選ばれ，この経路を通らなかった）．
作り物の選択で sv.partner を呼び，作り直した判定と保存した int8 の判定の一致を数える．成績は出さない．"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
exec(open(HERE / "run_trial.py").read().split("sys.argv = [")[0])
import numpy as np
import dai3 as D3, dai1, dai_solver as S, dai_data as DD
fz1 = json.load(open(HERE / "d1e" / "select" / "frozen.json"))
sv = D3.Saved(HERE / "d1e", "eval", "AFEW", "AFEW", 0, 0, fz1["AFEW"])
print([k for k in sv.p3.files if k.startswith(("p3c_K5", "p3g_K5"))][:20])
for s in DD.STUDENTS:
    for pm, pv, grid in (("cotrain", "ct", dai1.RHO_GRID), ("coreg", "cr", dai1.COREG_RHO)):
        if pm == "cotrain" and s == "AVT":
            continue
        for rho in grid:
            sv.fz = json.loads(json.dumps(fz1["AFEW"]))
            sv.fz["partner"][f"{s}|5"] = {"method": pm, "version": pv}
            sv.fz["select"][f"{s}|5|{pm}|{pv}"] = {"rho": rho, "pseudo_uar": 0}
            W, b = sv.partner(s, 5)
            Z = S.logits(sv.u.Xt[s], W, b).argmax(1).cpu().numpy()
            key = f"p3c_K5_{s}_pred" if pm == "cotrain" else f"p3g_K5_{s}_pred"
            j = list(grid).index(rho)
            print(s, pm, rho, "一致", float((Z == sv.p3[key][j]).mean()))
