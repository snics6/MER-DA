"""gate-code（評価の回の起動の台本）：実験 2 の S+T の C の交差検証（ソース 3 つ）の所要とメモリを測る．ソースのデータだけを使う．
出力は一時の置き場（code_run/out/st_cv）に書き，交差検証の UAR は画面に出さない．"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
import torch
import dai1, dai2  # noqa: E402

out = Path(__file__).resolve().parent / "out"
dai2.RUN["commit"] = "gate-code-trial"
srcs, kind = sys.argv[1].split("+"), sys.argv[2]
for s in srcs:
    dai1.Corpus.get(s)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
d = dai2.st_cv(out, srcs, kind)
print(f"st_cv {'+'.join(srcs)} {kind}: {time.time()-t0:.0f} 秒，最大の確保 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB，"
      f"C が格子の端か {d['edge']}，勾配の最大 {d['grad_max']:.1e}", flush=True)
