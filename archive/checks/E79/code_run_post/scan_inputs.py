"""E77・E78・E82 の結果フォルダのファイルが読めるかを調べる（読むだけ．値は出さない）。
npy：ヘッダを読み，ファイルの大きさがヘッダの形 × 要素の大きさと合うか．npz：zip の CRC（testzip）と各配列のヘッダ．
csv：全行を pandas で読む（行の数と列の数だけ残す）．json・jsonl：全部を読む。"""
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

dirs = sys.argv[1:]
bad, n, t0 = [], 0, time.time()
stats = {}
for d in dirs:
    for f in sorted(Path(d).iterdir()):
        n += 1
        try:
            if f.suffix == ".npy":
                a = np.load(f, mmap_mode="r")
                exp = a.offset + a.nbytes
                if f.stat().st_size != exp:
                    bad.append((str(f), f"size {f.stat().st_size} != {exp}"))
            elif f.suffix == ".npz":
                with zipfile.ZipFile(f) as z:
                    r = z.testzip()
                    if r is not None:
                        bad.append((str(f), f"crc {r}"))
            elif f.suffix == ".csv":
                x = pd.read_csv(f, low_memory=False)
                if len(x) == 0:
                    bad.append((str(f), "0 rows"))
                stats[str(f)] = (len(x), x.shape[1])
            elif f.suffix == ".json":
                json.loads(f.read_text())
            elif f.suffix == ".jsonl":
                [json.loads(s) for s in f.read_text().splitlines() if s.strip()]
            else:
                bad.append((str(f), "unknown suffix"))
        except Exception as e:  # noqa: BLE001
            bad.append((str(f), f"{type(e).__name__}: {e}"[:200]))
print(f"files {n}, bad {len(bad)}, {time.time()-t0:.0f}s")
for b in bad:
    print("BAD", *b)
pd.DataFrame([(k, *v) for k, v in stats.items()], columns=["file", "rows", "cols"]).to_csv(
    Path(__file__).parent / f"csv_shapes_{Path(dirs[0]).name}.csv", index=False)
