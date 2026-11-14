"""関門 1 の統計の確かめ（DAI-21）．メタデータと人工の数だけを読む．GPU を使わない．"""
import sys
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/snics/MER-DA")
from measure import split as SP  # noqa

m = pd.read_csv("/home/snics/MER-DA/data/metadata/crema-d_utterance_manifest.csv")
print(m.columns.tolist())
lab = "label_4" if "label_4" in m.columns else None
d = m[m[lab].notna() & (m[lab].astype(str) != "") & (m[lab].astype(str) != "-1")] if lab else m
print("全行", len(m), "4 ラベル", len(d))
g = d["speaker_group"].astype(str) if "speaker_group" in d.columns else d["speaker_id"].astype(str)
n = g.value_counts().to_numpy()
print("話者", len(n), "g_eff", n.sum() ** 2 / (n ** 2).sum(), "話者あたり 最小/中央/最大", n.min(), np.median(n), n.max())
fo = SP.speaker_folds(g, k=5, seed=0).to_numpy()
for f in range(5):
    s = fo == f
    print("分割", f, "話者", g[s].nunique(), "発話", int(s.sum()), "感情ごと", np.bincount(d[lab].to_numpy().astype(int)[s], minlength=4))
if "sentence" in d.columns or "text" in d.columns:
    pass
print("文の列の候補", [c for c in d.columns if "sent" in c.lower() or "text" in c.lower() or "file" in c.lower() or "sample" in c.lower()])
print("2^(1-5) =", 2.0 ** (1 - 5), " 2^(1-4) =", 2.0 ** (1 - 4))
