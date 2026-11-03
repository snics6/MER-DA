"""コミット 569766b の台本（git archive で一時の置き場に取り出したもの）を読み込んで動かす（作業ツリーの書き換えの影響を受けないため）。
使い方：run_snap.py main <dai1.py の引数…>  /  run_snap.py script <台本のパス> <引数…>"""
import sys, runpy, os
from pathlib import Path
SNAP = Path("/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/" + __import__('os').environ.get('SNAPDIR', 'snap569'))
REAL = Path("/home/snics/MER-DA")
sys.path.insert(0, str(SNAP / "DAI" / "exp"))
import dai_data, dai1, dai_solver, dai1_select, dai1_stats  # noqa: E402
assert all(Path(m.__file__).resolve().is_relative_to(SNAP) for m in (dai_data, dai1, dai_solver, dai1_select, dai1_stats))
dai1.ROOT = REAL
if os.environ.get('FIXCOMMIT'):                    # 確かめの間に HEAD が動いても照合が止まらないよう，コミットを固定する
    dai1.git_commit = lambda: (os.environ['FIXCOMMIT'], True)                       # git のコミット・置き場の判定は本物の置き場で
mode = sys.argv[1]
if mode == "main":
    sys.argv = ["dai1.py"] + sys.argv[2:]
    dai1.main()
else:
    sys.argv = sys.argv[2:]
    runpy.run_path(sys.argv[0], run_name="__main__")
