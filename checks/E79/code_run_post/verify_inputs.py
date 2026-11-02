"""provenance に書かれた入力の sha256（または大きさと時刻）を今のファイルと照らし，E79 p4 と E82 で共通の入力の記録が同じかを見る（読むだけ）。"""
import hashlib
import json
import os
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


rec = {}
for d in ("results/e79_p4", "results/e82_analysis/reselect"):
    for s, k in (("delta", "inputs_sha256"), ("select", "inputs_sha256"), ("measure", "inputs")):
        v = json.load(open(f"{d}/{s}/provenance.json"))[k]
        bad, n_hash, n_other, missing = [], 0, 0, 0
        for path, val in v.items():
            if not os.path.exists(path):
                missing += 1
                continue
            if isinstance(val, str) and len(val) == 64:
                n_hash += 1
                if sha(path) != val:
                    bad.append(path)
            else:
                n_other += 1
                st = os.stat(path)
                if isinstance(val, dict) and (val.get("bytes") != st.st_size or abs(val.get("mtime", 0) - st.st_mtime) > 1):
                    bad.append(path)
                elif isinstance(val, (list, tuple)) and val and val[0] != st.st_size:
                    bad.append(path)
            rec.setdefault(path, {})[f"{d}:{s}"] = val
        print(d, s, "entries", len(v), "hash", n_hash, "size/mtime", n_other, "missing", missing, "changed", len(bad), bad[:3])
# 共通の入力の記録が 2 つの解析で同じか
diff = [p for p, r in rec.items() if len({json.dumps(x, sort_keys=True) for x in r.values()}) > 1]
print("同じファイルで記録が食い違う数", len(diff), diff[:3])
ex = next(iter(v.values()))
print("例（大きさと時刻の記録の形）", [x for x in list(json.load(open("results/e79_p4/measure/provenance.json"))["inputs"].values()) if not isinstance(x, str)][:1])
