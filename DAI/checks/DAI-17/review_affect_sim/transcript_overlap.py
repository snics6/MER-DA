"""感情認識の査読役の検算（DAI-17 第 1 版）：発話一覧の書き起こしだけを使う（ラベルの一致は数えない．特徴も結果も読まない）．
言葉の空間で近い発話を引くとき，ターゲットの発話と同じ文（小文字にして英数字以外を除いた文）がソースの 2 コーパスに何本あるかを数える．
"""
import itertools
import re
import numpy as np
import pandas as pd

ROOT = "/home/snics/MER-DA"
C = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
man = {"AFEW": "afew", "IEMOCAP": "iemocap", "MOSEI": "mosei", "MELD": "meld"}


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower().replace("\x92", "'").replace("’", "'")).strip()


tx = {}
for c in C:
    sp = pd.read_csv(f"{ROOT}/DAI/splits/{c}.csv", dtype=str, keep_default_na=False)
    m = pd.read_csv(f"{ROOT}/data/metadata/{man[c]}_utterance_manifest.csv", dtype=str, keep_default_na=False)
    m = m[m.sample_id.isin(sp.sample_id)].drop_duplicates("sample_id") if c != "IEMOCAP" else m[m.sample_id.isin(sp.sample_id)]
    t = m.set_index("sample_id").transcript.reindex(sp.sample_id).map(norm)
    tx[c] = t
    nw = t.str.split().str.len()
    print(c, len(t), "語数の中央値", int(nw.median()), "3 語以下の割合", round(float((nw <= 3).mean()), 3))
rows = []
for c in C:
    for srcs in itertools.combinations([x for x in C if x != c], 2):
        vc = pd.concat([tx[s] for s in srcs]).value_counts()
        k = tx[c].map(vc).fillna(0).to_numpy()
        rows.append(dict(c=c, srcs="+".join(srcs), share_ge1=float((k >= 1).mean()), share_ge5=float((k >= 5).mean()), share_ge20=float((k >= 20).mean()), share_ge100=float((k >= 100).mean())))
d = pd.DataFrame(rows).round(3)
print(d.to_string(index=False))
d.to_csv(f"{ROOT}/DAI/checks/DAI-17/review_affect_sim/transcript_overlap.csv", index=False)
