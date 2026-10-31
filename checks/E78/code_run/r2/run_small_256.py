"""E78 関門 1 の再検査：957fa35 の exp/compare_dai2.py（写し）の main を流す。環境変数で格子を縮める（経路の確かめ．値は本番と違う）。
MAP_SMALL=1：DANN 等の反復 (5, 10)．CT_SMALL=1：Co-teaching の τ (0,)・m_eff ("n",)（更新の回数 CT_STEPS と cotzero は本番のまま）．CORAL_R=64 など：r の候補。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_dai2_256d41c as E  # noqa: E402  （検査するコミットの版を固定）
E.ROOT = ROOT
if os.environ.get("MAP_SMALL"):
    E.MAP_ITERS = (5, 10)
if os.environ.get("CT_SMALL"):
    E.CT_TAUS, E.CT_MEFFS = (0.0,), ("n",)
if os.environ.get("CORAL_R"):
    E.CORAL_R = tuple(int(x) for x in os.environ["CORAL_R"].split(","))
sys.argv = ["compare_dai2.py"] + sys.argv[1:]
raise SystemExit(E.main())
