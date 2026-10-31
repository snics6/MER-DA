"""E78 関門 1：exp/compare_dai2.py の main を，反復の回数だけ小さくして流す（経路の確かめ．値は本番と違う）。
使い方：./.venv/bin/python checks/E78/code_run/run_small.py <compare_dai2.py の引数>
MAP_ITERS = (10, 20)，CT_EPOCHS = 2 に替える．ほかの格子と手順は台本のまま。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_dai2_9044116 as E  # noqa: E402  （検査するコミットの版を固定）
E.ROOT = ROOT  # 写した版の ROOT は checks/E78 を指すので直す

E.MAP_ITERS = (10, 20)
E.CT_EPOCHS = 2
import os  # noqa: E402
if os.environ.get("CT_SMALL"):
    E.CT_TAUS, E.CT_LRS, E.MAP_ITERS = (0.0, 0.4), (1e-2,), (5, 10)
if os.environ.get("CORAL_R_SMALL"):
    E.CORAL_R = (32,)
sys.argv = ["compare_dai2.py"] + sys.argv[1:]
raise SystemExit(E.main())
