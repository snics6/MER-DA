"""gate-code（DAI-5）：提案する守り（今の段 source から作った出発点のテストの判定が，実験 1 の p1 の start_pred と一致するか）が成り立つかを 3 単位で試す。"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1, dai5, dai_data as DD, dai_solver as S
for T, c, f, r in [("AFEW", "IEMOCAP", 0, 0), ("MELD", "AFEW", 2, 1), ("IEMOCAP", "MOSEI", 3, 0)]:
    u = dai1.Unit(dai5.IN_DIR, T, c, [s for s in DD.others(T) if s != c], f, r)
    z1 = np.load(ROOT / "results/dai1/pseudo/units" / f"{T}_{c}_f{f}_r{r}_p1.npz")
    agree = [float((S.logits(u.Xt[k], *u.start[k]).argmax(1).cpu().numpy() == z1[f"p1_{k}_start_pred"]).mean()) for k in DD.STUDENTS]
    print(T, c, f, r, "start_pred agreement per student", agree)
