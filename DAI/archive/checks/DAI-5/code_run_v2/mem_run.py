"""gate-code 第 2 版（DAI-5）：dai5.py の main を同じ引数で呼び，終わった後に GPU のメモリの最大（確保・予約）と壁時計の所要を出す。
成績は出さない。使い方：./.venv/bin/python DAI/checks/DAI-5/code_run_v2/mem_run.py <dai5.py の引数…>
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai5  # noqa: E402

t0 = time.time()
sys.argv = ["dai5.py"] + sys.argv[1:]
dai5.main()
print("wall sec", round(time.time() - t0, 1), "peak alloc GB", round(torch.cuda.max_memory_allocated() / 1e9, 2),
      "peak reserved GB", round(torch.cuda.max_memory_reserved() / 1e9, 2))
