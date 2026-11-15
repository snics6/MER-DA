"""gate-code の検算：融合の none（sv.none("AVT", K)）をテストの発話に当てた UAR を作り直し，実験 1 の保存の p1 の csv の
同じ行（method=none，kind=AVT，K，Cw・Cb＝固定の引き戻し）との差だけを出す．値そのものは出さない．"""
import sys
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import numpy as np, pandas as pd
import dai3, dai1, dai_data as DD, dai_solver as S
dai1.ENC["enc"] = DD.ENC1
OUT1 = DD.ROOT / "results" / "dai1"
fz1, _ = dai3.load_fz1(OUT1)
for T, f, r in [("AFEW", 0, 0), ("MOSEI", 0, 1), ("IEMOCAP", 0, 1), ("MELD", 0, 0)]:
    sv = dai3.Saved(OUT1, "eval", T, T, f, r, fz1[T]); u = sv.u
    p1 = pd.read_csv(OUT1 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}_p1.csv")
    for K in (5, 25):
        cw, cb = sv.teth[("AVT", K)]
        pr = S.logits(u.Xt["AVT"], *sv.none("AVT", K)).argmax(1).cpu().numpy()
        mine = 100 * S.uar(np.asarray(u.ytest, int), pr)
        row = p1[(p1.method == "none") & (p1.kind == "AVT") & (p1.K == K) & np.isclose(p1.Cw, cw) & np.isclose(p1.Cb, cb)]
        if len(row) != 1:
            print(T, f, r, K, "rows", len(row)); continue
        sc = float(row.uar.iloc[0]); sc = sc * 100 if sc <= 1 else sc
        print(f"{T} f{f} r{r} K={K}: |rebuilt - saved test UAR| = {abs(mine - sc):.2e}")
    dai1.Corpus.cache.pop(T, None)
