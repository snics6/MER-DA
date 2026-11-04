"""dai3.py の試走：実験 1 の試走の出力（Y2．run_small.py の絞った格子：K = (1, 5)，分割 2 つ，T = MOSEI の仮のターゲットは AFEW と
IEMOCAP だけ）に合わせて定数を絞ってから dai3.main を呼ぶ．評価の回の代わりに，仮のターゲット AFEW の単位を評価の回の単位に見立てる
（T = AFEW，ソースは IEMOCAP の 1 つ）．値は使わない．"""
import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai_data as DD  # noqa: E402
DD.KS = (1, 5)
DD.N_FOLDS = 2
_others = DD.others
_FAKE = {"MOSEI": ["AFEW", "IEMOCAP"], "AFEW": ["IEMOCAP"], "IEMOCAP": ["AFEW"]}
DD.others = lambda c: _FAKE.get(c) or _others(c)
import dai1  # noqa: E402
dai1.KS_ALL = (0,) + DD.KS
dai1.RHO_GRID = (0.25, 1.0); dai1.A_GRID = (1, 4); dai1.COREG_RHO = dai1.RHO_GRID     # Y2 は共正則化も RHO_GRID で解いた（3d334ad）
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
import dai3  # noqa: E402
sys.argv = ["dai3.py"] + sys.argv[1:]
dai3.main()
