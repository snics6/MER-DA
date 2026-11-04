"""gate-code の試走の置き場（実装した役の make_trial.py を写し，評価の回に見立てる T を AFEW と IEMOCAP の 2 つにした）．
d1p：仮のターゲットの回だけ．d1e：MOSEI_AFEW と MOSEI_IEMOCAP の単位を，評価の回の AFEW_ALL と IEMOCAP_ALL の単位に見立てる．
実験 1 の固定のファイルに，試走のための種の数と δ（仮の値 1.5）と AFEW・IEMOCAP の項（MOSEI の選択の写し）を足す．成績は読まない．"""
import json, os, sys
from pathlib import Path
TR = Path(sys.argv[1]).resolve(); Y2 = Path(sys.argv[2]).resolve()
fz = json.load(open(Y2 / "select" / "frozen.json"))
for X in ("AFEW", "IEMOCAP"):
    fz[X] = json.loads(json.dumps(fz["MOSEI"]))
for X in ("AFEW", "IEMOCAP", "MOSEI"):
    fz[X]["n_seeds_eval"] = 2
    fz[X]["detect"] = {"delta_same": 1.5, "note": "試走の仮の値"}
for d in ("d1p", "d1e"):
    (TR / d / "select").mkdir(parents=True, exist_ok=True)
    json.dump(fz, open(TR / d / "select" / "frozen.json", "w"), indent=1, ensure_ascii=False)
    for sub in ("source", "pseudo"):
        if not (TR / d / sub).exists():
            os.symlink(Y2 / sub, TR / d / sub)
eu = TR / "d1e" / "eval" / "units"
eu.mkdir(parents=True, exist_ok=True)
for X in ("AFEW", "IEMOCAP"):
    for p in sorted((Y2 / "pseudo" / "units").glob(f"MOSEI_{X}_f*")):
        q = eu / p.name.replace(f"MOSEI_{X}_", f"{X}_ALL_")
        if not q.exists():
            os.symlink(p, q)
print(len(list(eu.iterdir())), "個のリンク")
