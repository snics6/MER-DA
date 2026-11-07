"""gate-code の所要の見積もり（DAI-6）：IEMOCAP の分割 0 で，台本の部品ごとの時間を測る（結果は主張に使わない）。"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai6_probe as P  # noqa: E402


def tic():
    if P.DEV == "cuda":
        torch.cuda.synchronize()
    return time.time()


t0 = tic()
U = P.ProbeUnit("IEMOCAP", 0)
print(f"読み込み {tic() - t0:.1f} 秒：候補 {U.n}，テスト {len(U.test)}，p {U.p}，話者の例 {sorted(set(U.center_group))[:4]}，"
      f"マーカーの欠け {int(U.meta['marker'].isna().sum())}，長さの欠け {int(U.meta['duration_sec'].isna().sum())}，トークンの欠け {int(U.meta['n_token'].isna().sum())}", flush=True)
pc = {m: P.pca(U.Xp[m], U.p) for m in P.MODS}
t0 = tic()
r, rows, *_ = P.seq_test(pc["A"][2], pc["T"][2], 200, None, 1)
print(f"順の検定（音声–言葉，並べ替え 200 回）：{tic() - t0:.1f} 秒，段 {len(rows)}，r = {r}", flush=True)
torch.cuda.reset_peak_memory_stats()
for name, F, Ft in [("AVT 全部の次元", torch.cat([U.Xp[m] for m in P.MODS], 1), torch.cat([U.Xt[m] for m in P.MODS], 1)),
                    ("A 全部の次元", U.Xp["A"], U.Xt["A"]),
                    ("AVT 主成分 p", torch.cat([U.Xp[m] @ (pc[m][0] * pc[m][1].rsqrt()) for m in P.MODS], 1),
                     torch.cat([U.Xt[m] @ (pc[m][0] * pc[m][1].rsqrt()) for m in P.MODS], 1))]:
    t0 = tic()
    res = P.fit_info(F, Ft, U.y, U.yt, U.cluster)
    print(f"情報の量の行（{name}，{F.shape[1]} 次元）：{tic() - t0:.1f} 秒，C={res['C']}，収束 {res['ok']}", flush=True)
print(f"GPU の最大 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
