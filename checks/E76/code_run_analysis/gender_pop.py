"""性別の 1 段目の母集団（gender_tables の persons）と，反復 0〜9 のテストに出た人物の数を比べる（数だけ）。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/home/snics/MER-DA"); sys.path.insert(0, str(ROOT/"exp"))
import analyze_e76 as AE
class C:
    def note_input(self, p): pass
ctx = type("X", (), {"note_input": lambda self, p: None})()
meta = AE.load_meta(ctx)
for kind in ("audio_wavlm", "video"):
    for g in ("IEMOCAP-impro", "IEMOCAP-script", "AFEW", "RAVDESS"):
        p = ROOT/"results/e76"/kind/f"{g}.npz"
        if not p.exists(): continue
        t = AE.Target(C(), kind, g, ROOT/"results/e76")
        gen = np.array([meta["gender"].get((t.corpus, s), "") for s in t.sid])
        pg = pd.DataFrame({"p": t.person, "g": gen}).groupby("p").g.agg(lambda v: v.iloc[0] if v.nunique() == 1 else "mixed")
        test = set(np.concatenate([t.person[t.te[r]] for r in AE.REPS]))
        for gg in "FM":
            allp = set(pg[pg == gg].index); tp = allp & test
            print(kind, g, gg, "persons_in_draw_pool", len(allp), "persons_in_any_test", len(tp))
        print(kind, g, "unknown/mixed", int((~pg.isin(["F","M"])).sum()), "of", len(pg), "test_persons_total", len(test))
