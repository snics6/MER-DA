"""感情認識の査読想定の検査（DAI-16 第 3 版）の数え．git に入っている分割の表 DAI/splits/*.csv だけを読む（results/ は読まない）．
1. ソースの中の交差検証（home）の分割ごとの件数：テストの感情ごとの件数と，学習側の件数（MELD はテストの対話を除く．dai_data.pool_mask と同じ式）．
2. IEMOCAP の即興の筋書きの番号（sample_id の impro01〜08）と 4 感情の表．筋書きの番号だけで，別のセッションの感情をどれだけ当てられるか．
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
for c in ("AFEW", "IEMOCAP", "MOSEI", "MELD"):
    df = pd.read_csv(ROOT / "DAI" / "splits" / f"{c}.csv", dtype={"dialogue": str}, keep_default_na=False)
    print(c, "発話", len(df), "塊", df.cluster.nunique())
    for f in range(5):
        test = (df.fold == f).to_numpy()
        pool = ~test
        if (df.dialogue != "").any():
            pool &= ~df.dialogue.isin(set(df.loc[test, "dialogue"])).to_numpy()
        print(f"  分割 {f}：テスト {test.sum()} 本（感情ごと {np.bincount(df.label_4[test], minlength=4).tolist()}），"
              f"学習側 {pool.sum()} 本（全体の {pool.sum() / len(df):.2f}）")

df = pd.read_csv(ROOT / "DAI" / "splits" / "IEMOCAP.csv", keep_default_na=False)
df["sc"] = df.sample_id.str.extract(r"_(impro\d+)")[0]
print("IEMOCAP の筋書きの番号 × 感情（0=Neutral 1=Happy 2=Sad 3=Angry）")
print(pd.crosstab(df.sc, df.label_4))
uars = []
for f in range(5):
    tr, te = df[df.fold != f], df[df.fold == f]
    cnt = pd.crosstab(tr.sc, tr.label_4)
    best = (cnt / cnt.sum(axis=0)).idxmax(axis=1)          # 筋書きごとに，感情の件数で割った割合が最大の感情（学習側の 4 セッションで決める）
    p, y = te.sc.map(best).to_numpy(), te.label_4.to_numpy()
    uars.append(np.mean([(p[y == k] == k).mean() for k in range(4) if (y == k).any()]))
print(f"筋書きの番号だけで，残りのセッションの感情を当てたときの UAR（5 分割の平均）：{100 * np.mean(uars):.1f}（でたらめなら 25）")
