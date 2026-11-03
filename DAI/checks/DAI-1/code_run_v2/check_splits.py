"""DAI/splits の表と load_corpus の作り直しの一致，分割ごとのテストの感情の欠け，G_eff（UAR は見ない）。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "DAI/exp")
import dai_data as DD
for c in DD.CORPORA:
    df, _ = DD.load_corpus(c)
    sp = pd.read_csv(f"DAI/splits/{c}.csv", keep_default_na=False)
    same = (len(sp) == len(df)) and (sp.sample_id.astype(str).values == df.sample_id.values).all() and (sp.fold.values == df.fold.values).all() and (sp.cluster.astype(str).values == df.cluster.astype(str).values).all()
    n = sp.cluster.value_counts().to_numpy(); ge = n.sum()**2 / (n**2).sum()
    miss = {f: [k for k in range(4) if ((df.fold == f) & (df.label_4 == k)).sum() == 0] for f in range(5)}
    ntest = [int((df.fold == f).sum()) for f in range(5)]
    ncl = [df.loc[df.fold == f, "cluster"].nunique() for f in range(5)]
    print(c, len(df), "splits一致", bool(same), "G_eff", round(ge, 2), "塊", df.cluster.nunique(), "テスト本数", ntest, "テストの塊数", ncl, "欠けた感情", miss)
