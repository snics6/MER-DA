import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1_stats as S1, dai_data as DD
ROOT = Path("/home/snics/MER-DA"); out = ROOT / "results/dai1"
fz = json.load(open(out / "select/frozen.json"))
mine = pd.read_csv(ROOT / "DAI/checks/DAI-1/code_run_post/detect_rebuild_cells.csv")
T, c = sys.argv[1], sys.argv[2]
g_eff, _ = S1.g_eff_of(c)
units = [(f, r) for f in range(5) for r in (0, 1)]
y_of, cl_of = S1.label_maps(c)
cells, mism, tot = S1.collect(out, "pseudo", T, c, fz[T], units, y_of, cl_of, check_rows=False)
rng = np.random.default_rng(0)
for fam, (A, B, ks) in S1.FAMS.items():
    for s in DD.STUDENTS:
        for K in ks:
            e = S1.effect_stats(cells[(s, K, A)], cells[(s, K, B)], g_eff, rng)
            m = mine[(mine["T"] == T) & (mine.c == c) & (mine.fam == fam) & (mine.s == s) & (mine.K == K)].iloc[0]
            flag = "" if np.isclose(e["v_jk"], m.v_jk) and np.isclose(e["v_seed"], m.v_seed) and np.isclose(e["effect"], m.est) else "  <-- differs"
            print(fam, s, K, f"est {e['effect']:.4f}/{m.est:.4f} vjk {e['v_jk']:.4f}/{m.v_jk:.4f} vseed {e['v_seed']:.4f}/{m.v_seed:.4f} vsa {e['v_seed_all']:.4f}/{m.v_seed_all:.4f} drop {e.get('folds_dropped')}/{m.ndrop}{flag}")
