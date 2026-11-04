"""gate-code の検算：AVT の map の束（12 問）と xMUDA 型の束（30 問）の GPU のメモリの最大と 1 反復の時間。引数：trial3 の置き場"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1  # noqa: E402
import dai2  # noqa: E402
import dai2_grad as G  # noqa: E402

d1 = Path(sys.argv[1])
dai2.RUN["dai1_dir"] = str(d1)
teth, _ = dai2.tether_pseudo(d1, "MOSEI", True)
u = dai1.Unit(d1, "MOSEI", "AFEW", ["IEMOCAP", "MELD"], 0, 0)
pt, ylab, wK = G.kshot(u)
s = "AVT"
Z = u.Xp[s].to(G.F32)
Zs, ys = G.src_feats(u, s)
W0, b0 = (x.to(G.F32) for x in u.start[s])
probs = [("dann", 0.1, 1.0, 0.0, 5)] * 4 + [("mme", 0.1, 1.0, 0.1, 5)] * 8
tw = [teth[(s, 5)]] * 12
base = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
for it in (20, 120):
    torch.cuda.synchronize()
    t0 = time.time()
    G.train_map(Z, Z[pt], ylab, torch.stack([wK[5]] * 12), Zs, ys, W0, b0, [x[0] for x in tw], [x[1] for x in tw],
                [c[0] for c in probs], [c[1] for c in probs], [c[2] for c in probs], [c[3] for c in probs], 1e-3, it, 0)
    torch.cuda.synchronize()
    print(f"AVT の束 12 問，反復 {it}：{time.time()-t0:.1f} 秒")
print("AVT の束の GPU の最大（GB，データを除く）", (torch.cuda.max_memory_allocated() - base) / 1e9, "全体", torch.cuda.max_memory_allocated() / 1e9)
