"""gate-code の所要の見積もり（DAI-6）：MOSEI の分割 0 の大きさで，順の検定の 1 段（並べ替え N 回）と情報の量の行の 1 回（融合，全部の次元）の時間と GPU のメモリを測る。"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai6_probe as P  # noqa: E402

n = 14907
for q in (200, 400):
    torch.cuda.reset_peak_memory_stats()
    Ra = torch.randn(n, 200, dtype=torch.float64, device=P.DEV)
    Rb = torch.randn(n, q, dtype=torch.float64, device=P.DEV)
    perms = P.make_perms(n, 200, None, 0)
    torch.cuda.synchronize(); t0 = time.time()
    P.perm_null_max(Ra, Rb, perms)
    torch.cuda.synchronize()
    print(f"並べ替え 200 回（n={n}, p=200, q={q}）：{time.time() - t0:.1f} 秒，GPU の最大 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
U = P.ProbeUnit("MOSEI", 0)
print("MOSEI 分割 0：候補", U.n, "p", U.p, flush=True)
torch.cuda.reset_peak_memory_stats()
F = torch.cat([U.Xp[m] for m in P.MODS], 1); Ft = torch.cat([U.Xt[m] for m in P.MODS], 1)
torch.cuda.synchronize(); t0 = time.time()
r = P.fit_info(F, Ft, U.y, U.yt, U.cluster)
torch.cuda.synchronize()
print(f"情報の量の行（AVT，全部の次元 {F.shape[1]}）：{time.time() - t0:.1f} 秒，C={r['C']}，収束 {r['ok']}，GPU の最大 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
