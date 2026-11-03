"""569766b の DAI/splits の表と load_corpus の作り直しの一致，G_eff，subsets.csv の作り直しの一致，jackknife で分割ごと外れる塊（UAR は見ない）。"""
import sys
import numpy as np, pandas as pd
import dai_data as DD, dai_subsets as SB
from pathlib import Path
SNAP = Path(DD.__file__).resolve().parents[2]
for c in DD.CORPORA:
    df, _ = DD.load_corpus(c)
    sp = pd.read_csv(SNAP / f"DAI/splits/{c}.csv", keep_default_na=False)
    same = (len(sp) == len(df)) and (sp.sample_id.astype(str).values == df.sample_id.values).all() and (sp.fold.values == df.fold.values).all() and (sp.cluster.astype(str).values == df.cluster.astype(str).values).all()
    n = sp.cluster.value_counts().to_numpy(); ge = n.sum()**2 / (n**2).sum()
    miss = {f: [k for k in range(4) if ((df.fold == f) & (df.label_4 == k)).sum() == 0] for f in range(5)}
    ntest = [int((df.fold == f).sum()) for f in range(5)]
    ncl = [df.loc[df.fold == f, "cluster"].nunique() for f in range(5)]
    # jackknife：塊 g を除くと分割 f のテストが 30 本未満か感情が欠ける塊
    drop = {}
    for f in range(5):
        t = df[df.fold == f]
        for g, tg in t.groupby("cluster"):
            rest = t[t.cluster != g]
            if len(rest) < 30 or rest.label_4.nunique() < 4:
                drop.setdefault(f, []).append((g, len(tg)))
    print(c, len(df), "splits一致", bool(same), "G_eff", round(ge, 2), "塊", df.cluster.nunique(), "テスト本数", ntest, "テストの塊数", ncl,
          "欠けた感情", {k: v for k, v in miss.items() if v}, "分割ごと外れる塊", {f: len(v) for f, v in drop.items()}, {f: v[:3] for f, v in drop.items()})
    if c == "MELD":
        # MELD の話者の塊に別名が残っていないか
        man_alias = [a for a in DD.MELD_ALIAS if a in set(df.get("speaker", pd.Series(dtype=str)).astype(str))]
        print("MELD 別名の残り", man_alias, "塊の例", sorted(df.cluster.unique())[:5])
old = pd.read_csv(SNAP / "DAI/frozen/subsets.csv", keep_default_na=False)
new = pd.concat([SB.afew_asr(), SB.meld_faces(), SB.iemocap_marker()])[["sample_id", "corpus", "subset", "value"]]
key = ["corpus", "subset", "sample_id"]
m = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
print("subsets 行数", len(old), len(new), "片方だけ", int((m._merge != "both").sum()), "値の違い", int((m.value_old != m.value_new).sum()))
print(new.groupby(["corpus", "subset", "value"]).size().to_string())
