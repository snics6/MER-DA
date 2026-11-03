"""1 単位の段 3 を別の処理で解き直し，元の保存とビットで一致するかを見る（並べて流したときに同じ結果になるか）。引数：元の置き場 写しの置き場 c f r"""
import sys, json, shutil, os
from pathlib import Path
import dai1, dai_data as DD
A, X, c, f, r = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
shutil.rmtree(X, ignore_errors=True); (X / "pseudo" / "units").mkdir(parents=True)
os.symlink((A / "source").resolve(), X / "source")
for w in (1, 2):
    for ext in ("npz", "csv"):
        shutil.copy(A / "pseudo" / "units" / f"MOSEI_{c}_f{f}_r{r}_p{w}.{ext}", X / "pseudo" / "units")
shutil.copy(A / "pseudo" / "MOSEI_tether_w.json", X / "pseudo")
tw = json.load(open(X / "pseudo" / "MOSEI_tether_w.json"))
teth = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in tw["tether"].items()}
w = {int(k): tuple(v) for k, v in tw["w"].items()}
dai1.RUN.update(commit=dai1.git_commit()[0], dirty=True, gpu="test")
srcs = [s for s in DD.others("MOSEI") if s != c]
dai1.run_pass(X, "pseudo", "MOSEI", c, srcs, f, r, 3, tether=teth, wsel=w)
