"""gate-code（DAI-1 code_v5）：評価の段の読み方（c = ALL，引き戻しの曲線）を 1 つの T で通す．偽の置き場（AFEW の名前に，
仮のターゲット MOSEI・c = AFEW の 10 単位と引き戻しの曲線の問を置き，固定の設定を MOSEI の値にしたもの）で run_pair を呼ぶ。値は見ない。"""
import sys, json, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1_desc as D
import pandas as pd
place, dest = Path(sys.argv[1]), Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)
r = D.run_pair(dict(mode="eval", out=str(place), T="AFEW", c="AFEW", enc=D.DD.ENC1))
for k, v in r["tables"].items():
    pd.DataFrame(v).to_csv(dest / f"{k}.csv", index=False)
json.dump(dict(chk={f"{a}|{b}": n for (a, b), n in r["chk"].items()}, degen_ok=sum(d["ok"] for d in r["degen"]), degen=len(r["degen"]),
               units=r["units"], sec=r["sec"]), open(dest / "summary.json", "w"), indent=1)
print("済み", round(r["sec"]), "秒", flush=True)
