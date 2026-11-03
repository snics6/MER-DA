"""569766b の台本を，格子を絞った設定で動かす（段の分け方・再開・選ぶ段・統計の台本の動作を確かめるため．値は使わない）。
絞り方：K = (1, 5)，分割 2 つ（f = 0, 1），T = MOSEI の仮のターゲットは AFEW と IEMOCAP だけ（ソースは残りの 1 つ），
ρ = (0.25, 1)，a = (1, 4)，引き戻し 4 点，C 3 点，蒸留の温度 (1, 2)，一致の τ (0.7, 0.9)・f (1, 4)，混ぜる手の α (0.2, 0.5)・f (1, 4)。
使い方：run_small.py main <dai1.py の引数…>  /  run_small.py script <台本> <引数…>"""
import sys, runpy, os
from pathlib import Path
SNAP = Path("/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/" + __import__('os').environ.get('SNAPDIR', 'snap569'))
REAL = Path("/home/snics/MER-DA")
sys.path.insert(0, str(SNAP / "DAI" / "exp"))
import dai_data as DD  # noqa: E402
DD.KS = (1, 5)
DD.N_FOLDS = 2
_others = DD.others
DD.others = lambda c: ["AFEW", "IEMOCAP"] if c == "MOSEI" else _others(c)
import dai1  # noqa: E402
dai1.ROOT = REAL
dai1.KS_ALL = (0,) + DD.KS
dai1.RHO_GRID = (0.25, 1.0); dai1.A_GRID = (1, 4)
dai1.TETHER = [(0.01, 1.0), (0.01, 10.0), (0.1, 1.0), (0.1, 10.0)]
dai1.TGT_C = (0.01, 0.1, 1.0)
dai1.KD_T = (1, 2); dai1.AG_TAU = (0.7, 0.9); dai1.AG_F = (1, 4); dai1.MIX_ALPHA = (0.2, 0.5); dai1.MIX_F = (1, 4)
import dai1_select as SEL  # noqa: E402
SEL.GRIDS.update({"joint": [("rho", dai1.RHO_GRID), ("a", dai1.A_GRID)], "rolesep": [("rho", dai1.RHO_GRID), ("a", dai1.A_GRID)],
                  "kd": [("temp", dai1.KD_T), ("rho", dai1.RHO_GRID)], "agree": [("tau", dai1.AG_TAU), ("f", dai1.AG_F)],
                  "mix": [("alpha", dai1.MIX_ALPHA), ("f", dai1.MIX_F)], "cotrain": [("rho", dai1.RHO_GRID)],
                  "coreg": [("rho", dai1.RHO_GRID)], "tgtonly": [("C", dai1.TGT_C)], "joint_scratch": [("C", dai1.TGT_C)]})
import dai1_stats as ST  # noqa: E402
ST.KS_MAIN = (0, 5)
ST.FAMS = {1: ("prop", "none", (0, 5)), 2: ("rolesep", "prop", (5,)), 3: ("prop", "partner", (0, 5)), 4: ("prop", "default", (0, 5))}
if os.environ.get("CAP_LBFGS"):                 # 段の分け方の確かめだけのため，共正則化の反復を打ち切る（値は使わない）
    import torch
    _L = torch.optim.LBFGS
    cap = int(os.environ["CAP_LBFGS"])
    torch.optim.LBFGS = lambda params, **kw: _L(params, **(kw | {"max_iter": cap, "max_eval": 2 * cap}))
assert all(Path(m.__file__).resolve().is_relative_to(SNAP) for m in (DD, dai1, SEL, ST))
if sys.argv[1] == "main":
    sys.argv = ["dai1.py"] + sys.argv[2:]
    dai1.main()
else:
    sys.argv = sys.argv[2:]
    runpy.run_path(sys.argv[0], run_name="__main__")
