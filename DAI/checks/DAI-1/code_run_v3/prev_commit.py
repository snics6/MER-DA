import sys, json
from pathlib import Path
import dai1
E = Path(sys.argv[1])
tw = json.load(open(Path(sys.argv[2]) / "pseudo" / "MOSEI_tether_w.json"))
teth = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in tw["tether"].items()}
dai1.RUN.update(commit="0" * 40, dirty=True, frozen_sha256="sha-A", gate_sha256="gate-A", gpu="test")
try:
    dai1.run_pass(E, "eval", "AFEW", "AFEW", ["IEMOCAP", "MELD"], 1, 0, 2, tether=teth, wsel={0: (1, 1, 1)}, fsel=None); print("止まらなかった（誤り）")
except SystemExit as e:
    print("前の段のコミットが違う：止まった", str(e)[:70])
