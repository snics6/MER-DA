"""gate-code の再検査（第 4 版）：行のファイルの中身，(f, r) への足し合わせ，鍵のそろい方，候補の割り当て，Q1 の列。UAR は出さない。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
import dai1_stats as ST  # noqa: E402
import dai8_probe as P8  # noqa: E402

ROWS = ROOT / "DAI" / "checks" / "DAI-8" / "code_run_v3" / "run" / "rows"
files = sorted(ROWS.glob("*.npz"))
print("== (1) 鍵のそろい方（同じ c の単位の間）")
by_c = {}
for p in files:
    c = p.name.split("_")[1]
    z = np.load(p)
    by_c.setdefault(c, []).append((p.name, set(k for k in z.files if z[k].dtype == np.int8 and z[k].shape == z["test_y"].shape)))
for c, lst in by_c.items():
    allk = set.union(*[s for _, s in lst]); common = set.intersection(*[s for _, s in lst])
    diff = sorted(allk - common)
    print(f"{c}：単位 {len(lst)}，判定の鍵 {len(allk)}，一部の単位にだけある鍵 {len(diff)} {diff[:6]}")
    inC = [k for k in diff if any(k == a.format(K=K, s=s) or k == b.format(K=K, s=s) for a, b in P8.COMPARE.values() for K in P8.KS_ALL for s in P8.SPACES)]
    print(f"   そのうち比べ（COMPARE）に使う鍵 {len(inC)} {inC[:4]}")

print("== (2) (f, r) への足し合わせ（IEMOCAP f1 r0 の 2 つの T）")
za, zb = np.load(ROWS / "AFEW_IEMOCAP_f1_r0.npz"), np.load(ROWS / "MOSEI_IEMOCAP_f1_r0.npz")
ks = [k for k in za.files if k in zb.files and za[k].dtype == np.int8 and za[k].shape == za["test_y"].shape and k != "test_y"]
mx = 0.0
for k in ks:
    ca = ST.counts(za["test_y"].astype(int), za[k].astype(int), za["test_cluster"].astype(str))
    cb = ST.counts(zb["test_y"].astype(int), zb[k].astype(int), zb["test_cluster"].astype(str))
    mx = max(mx, abs(ST.uar_full((ca[0], ca[1] + cb[1], ca[2] + cb[2])) - 0.5 * (ST.uar_full(ca) + ST.uar_full(cb))))
print(f"鍵 {len(ks)}：足した数の UAR と T ごとの平均の最大の差 {mx:.2e}")

print("== (3) 行のファイルの中身")
for p in files:
    z = np.load(p)
    fl = z.files
    n = int(z["n_pool"])
    P = {k: len(z[k]) for k in fl if k.startswith("P_")}
    nan = sum(int((~np.isfinite(z[k])).sum()) for k in fl if z[k].dtype in (np.float16, np.float32, np.float64) and z[k].ndim > 0)
    dg = json.loads(str(z["diag"]))
    fixk = [k for k in dg if k.endswith("_em_a0.5_b1.0")]
    cap = sum(dg[k]["n_iter_done"] >= P8.MAX_EM for k in fixk)
    early = sum(dg[k]["n_iter_done"] < P8.N_EM for k in fixk)
    ph = [len(dg[k]["pool_hist"]) for k in fixk]
    print(f"{p.name}: {p.stat().st_size / 2**20:.2f} MiB，P_ の数 {len(P)}（長さが n_pool {n} と違う {sum(v != n for v in P.values())}），"
          f"pool_y/group/dialogue の長さ {len(z['pool_y'])}/{len(z['pool_group'])}/{len(z['pool_dialogue'])}，対話の数 {len(np.unique(z['pool_dialogue']))}，NaN {nan}，"
          f"固定の点で 100 回で止まった {cap}/{len(fixk)}，10 回より前に止まった {early}，pool_hist の長さ {min(ph)}〜{max(ph)}，commit {str(z['commit'])}")
print("== (4) Q1 の列と NaN")
for p in sorted(ROWS.glob("*_knn.csv")):
    kn = pd.read_csv(p)
    print(p.name, kn.shape, "NaN の列", {k: int(v) for k, v in kn.isna().sum().items() if v}, "対話の数", kn.n_dialogues.tolist(), "グループの数", kn.n_groups.tolist())
print("列", list(pd.read_csv(sorted(ROWS.glob('*_knn.csv'))[0]).columns))
