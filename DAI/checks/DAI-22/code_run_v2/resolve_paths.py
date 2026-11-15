"""gate-code 再検査（DAI-22 第 2 版）：照合が外れたときの経路を AFEW の小さい単位で確かめる．値は出さない．
1. TOL_REPRO を負にして，すべての R0 の行で resolve_r0 を通す（正しく収束した行では予測の違い 0 本になるはず）
2. resolve_r0 を「予測が 1 本違う」と返す偽物に差し替えて，台本が止まるか．その後，元に戻して同じ置き場で流し直すと，止まった単位が飛ばされるか
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai22  # noqa: E402

C = Path("/home/snics/MER-DA/DAI/checks/DAI-22/code_run_v2")
mode = sys.argv[1]
if mode == "all_resolve":
    dai22.TOL_REPRO = -1.0
    sys.argv = ["dai22.py", "--stage", "units", "--out-dir", str(C / "t_resolve"), "--targets", "AFEW", "--folds", "0", "--seeds", "0"]
    dai22.main()
elif mode == "fake_fail":
    dai22.TOL_REPRO = -1.0
    real = dai22.resolve_r0

    def fake(*a, **k):
        o = real(*a, **k)
        for t in dai22.TOL_RESOLVE:
            o[f"n_pred_diff_tol{t:g}"] = 1
        return o
    dai22.resolve_r0 = fake
    sys.argv = ["dai22.py", "--stage", "units", "--out-dir", str(C / "t_fail"), "--targets", "AFEW", "--folds", "0", "--seeds", "0"]
    dai22.main()
elif mode == "restart":
    sys.argv = ["dai22.py", "--stage", "units", "--out-dir", str(C / "t_fail"), "--targets", "AFEW", "--folds", "0", "--seeds", "0"]
    dai22.main()
