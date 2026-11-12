"""results/dai14/summary/predictions.csv から，主の条件（K = 5，R1，#shufcl，候補の全体）と，揺れを足さない区間の判定
（verdict_ns）が違うコーパスごとの行を抜き出す．対象の量は P1，P2，P3，P3b，P4a，P4b（P1ctl・P2ctl は含まない）．
数えの規則（前提，MELD の共有の対話，AFEW の年齢）を当てる前の判定どうしを比べる．出力：verdict_diffs_vs_main.csv．"""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[4]
p = pd.read_csv(ROOT / "results/dai14/summary/predictions.csv")
p = p[p.prediction.isin(["P1", "P2", "P3", "P3b", "P4a", "P4b"])]
base = p[(p.K == 5) & (p.readout == "R1") & (p.control == "cl") & (p.subset == "all")].set_index(["T", "prediction"])
out = []
for (K, ro, ct, sb), g in p.groupby(["K", "readout", "control", "subset"]):
    if (K, ro, ct, sb) == (5, "R1", "cl", "all"):
        continue
    g = g.set_index(["T", "prediction"])
    for k in g.index:
        if k in base.index and g.loc[k, "verdict_ns"] != base.loc[k, "verdict_ns"]:
            out.append(dict(K=K, readout=ro, control=ct, subset=sb, T=k[0], prediction=k[1], verdict_ns_main=base.loc[k, "verdict_ns"],
                            verdict_ns=g.loc[k, "verdict_ns"], effect=g.loc[k, "effect"], lo_ns=g.loc[k, "lo_ns"], hi_ns=g.loc[k, "hi_ns"]))
pd.DataFrame(out).to_csv(Path(__file__).with_name("verdict_diffs_vs_main.csv"), index=False)
print(len(out))
