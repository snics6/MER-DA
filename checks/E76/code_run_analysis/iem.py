import sys, numpy as np
from pathlib import Path
ROOT = Path("/home/snics/MER-DA"); sys.path.insert(0, str(ROOT/"exp"))
import analyze_e76 as AE
class C:
    def note_input(self, p): pass
ts = [AE.Target(C(), "text_roberta", g, ROOT/"results/e76") for g in ("IEMOCAP-impro", "IEMOCAP-script", "MELD", "MOSEI")]
dr = AE.Draws("text_roberta", ts, 50, 76)
print({c: len(v) for c, v in dr.persons.items()})
a, b = ts[0], ts[1]
print("same test persons per rep:", [set(a.person[a.te[r]]) == set(b.person[b.te[r]]) for r in AE.REPS])
print("weights shared object:", all((dr.W[a.corpus] is dr.W[b.corpus]) for _ in [0]))
print("MOSEI corpus persons equals union test:", len(dr.persons["MOSEI"]))
for r in AE.REPS:
    print(r, sorted(set(a.person[a.te[r]])), sorted(set(b.person[b.te[r]])))
