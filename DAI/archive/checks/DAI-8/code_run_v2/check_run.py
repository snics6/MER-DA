"""gate-code の検査：dai8_probe.run_unit を小さく流し，GPU のピークのメモリを測る（UAR は出さない）。"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai8_probe as P8  # noqa: E402

out = Path(sys.argv[1])
units = sys.argv[2].split(",")
fz = json.loads(P8.FROZEN.read_text())
res = []
for nm in units:
    p3 = P8.UD / f"{nm}_p3.npz"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r = P8.run_unit(p3, out, fz, "gatecode-test")
    r["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2**20
    r["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2**20
    res.append(r)
    print(nm, f"{time.time() - t0:.1f}s", f"peak_alloc {r['peak_alloc_MiB']:.0f} MiB reserved {r['peak_reserved_MiB']:.0f} MiB", flush=True)
(out / "check_run.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
