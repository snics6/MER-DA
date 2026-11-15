"""gate-code 再検査（DAI-22 第 4 版）：200 単位すべての p2.csv・p3.csv に，台本が照合に使う行が K ごとにちょうど 1 行あるかと，コミットが 1 つかだけを数える．値は出さない．"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai3  # noqa: E402

R1 = Path("/home/snics/MER-DA/DAI/results/dai1")
fz1, _ = dai3.load_fz1(R1)
cnt = Counter()
for T in ("AFEW", "IEMOCAP", "MELD", "MOSEI"):
    for f in range(5):
        for r in range(int(fz1[T]["n_seeds_eval"])):
            p2 = pd.read_csv(R1 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}_p2.csv")
            p3 = pd.read_csv(R1 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}_p3.csv", low_memory=False)
            t = p3[(p3.method == "_teacher_stats") & p3.student.isna()]
            cnt["units"] += 1
            cnt["p3_commit_not1"] += int(p3.commit.nunique() != 1)
            cnt["p2p3_commit_differ"] += int(set(p2.commit.unique()) != set(p3.commit.unique()))
            for K in (5, 25):
                main = fz1[T]["main"][str(K)]
                cnt[f"main_{main}"] += 1
                cnt["p3_rows_not1"] += int(len(t[(t.teacher == main) & (t.K == K)]) != 1)
                cnt["p2_none_not1"] += int(len(p2[(p2.method == "teacher") & (p2.teacher == "none_AVT") & (p2.K == K)]) != 1)
                cnt["p2_prod_not1"] += int(len(p2[(p2.method == "product") & (p2.teacher == "(1.0, 1.0, 1.0)") & (p2.K == K)]) != 1)
print(dict(cnt))
