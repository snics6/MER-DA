"""gate-code（E79）：2 段のブートストラップを，発話を重複の数だけ並べて d_o と uar を当てる素朴な計算で作り直し，台本の boot_*.npz と照合する。

台本の検算は「重みを全部 1 にしたとき」だけなので，重みが 0・2 以上の場合（重みつきの分散の式）をここで確かめる。
使い方：python verify_boot.py <E79 の out> <run> <E77 の場所> <E78 の場所> <T…>
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/snics/MER-DA")
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, load_any  # noqa: E402
from explore_pairs import ba  # noqa: E402
from explore_synergy import d_o  # noqa: E402

out, run, d77, d78 = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
Ts = sys.argv[5:]
REPS = [0, 1]
NB = 6
corp = GROUP_SPEC[Ts[0]][0]
keys = pd.read_csv(out / "measure" / f"keys_{run}.csv", dtype={"K": str, "param": str, "meff": str})
Dt = pd.read_csv(out / "measure" / f"D_{run}.csv")
SEEDED = {"dann", "mme", "mapctl", "xmuda", "cot", "cotctl", "cotfix", "cotzero", "cot_g", "cotctl_g", "cotfix_g", "cotzero_g"}

spk_of, te_of, y_of = {}, {}, {}
for T in Ts:
    samp = pd.read_csv(d77 / f"{T}_samples.csv")
    df, _ = load_any(T, run)
    y_of[T] = df.label_4.to_numpy().astype(int)
    spk_of[T] = samp.speaker_group.astype(str).to_numpy()
    te_of[T] = {r: np.load(d77 / f"{T}_r{r}_te.npy") for r in REPS}
speakers = sorted(set().union(*[set(spk_of[T][te_of[T][r]]) for T in Ts for r in REPS]))
rng = np.random.default_rng([0, zlib.crc32(run.encode()), zlib.crc32(corp.encode())])
nS = len(speakers)
B = int(pd.read_csv(out / "measure" / "boot_meta.csv").boot.iloc[0])
C = rng.multinomial(nS, np.full(nS, 1 / nS), size=B).astype(float)
A = rng.multinomial(len(REPS), np.full(len(REPS), 1 / len(REPS)), size=B).astype(float)
sp_ix = {s: i for i, s in enumerate(speakers)}


def rows_for(tab, key):
    st, K, wave, meth, pn, meff, md = key
    t = tab[wave]
    pnorm = t.param.str.replace(r"s\d+$", "", regex=True) if meth in SEEDED else t.param
    m = (t.student == st) & (t.K == K) & (t.method == meth) & (t.meff == meff)
    m &= pnorm.str.startswith(pn[:-1]) if pn.endswith(":*") else (pnorm == pn)
    x = t[m]
    if md == "s0":
        x = x[x.param.str.endswith("s0")]
    return x


def naive(L, y, w, D):
    idx = np.repeat(np.arange(len(y)), w.astype(int))
    Ly, yy = L[idx], y[idx]
    ks = np.unique(yy)
    u = np.mean([(Ly[yy == k].argmax(1) == k).mean() for k in ks])
    dl = []
    for p, (dd, oo) in enumerate(d_o(Ly, yy)) if all((yy == k).sum() >= 2 for k in range(4)) else []:
        dl.append(ba(D[p], 0.0) - ba(dd, 0.0))
    if not dl:                                                               # 感情が欠けたら対ごとに
        from explore_synergy import PAIRS
        for p, (k, j) in enumerate(PAIRS):
            if (yy == k).sum() >= 2 and (yy == j).sum() >= 2:
                zk = Ly[yy == k, k] - Ly[yy == k, j]
                zj = Ly[yy == j, k] - Ly[yy == j, j]
                sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
                dl.append(ba(D[p], 0.0) - ba((zk.mean() - zj.mean()) / sd, 0.0))
    return 100 * u, 100 * np.mean(dl)


worst = 0.0
for T in Ts:
    tabs = {w: pd.read_csv(d / f"{T}.csv", dtype={"K": str, "param": str, "meff": str}) for w, d in (("e77", d77), ("e78", d78))}
    z = np.load(out / "measure" / f"boot_{run}_{T}.npz")
    kt = keys[keys.target == T]
    pick = pd.concat([kt[kt.method == "none"].head(2), kt[kt.seedmode == "agg"].head(3), kt[kt.method == "dist"].head(2),
                      kt[kt.method == "cotfix"].head(1), kt[kt.method == "coral"].head(1)])
    for kr in pick.itertuples():
        key = (kr.student, kr.K, kr.wave, kr.method, kr.param, kr.meff, kr.seedmode)
        x = rows_for(tabs, key)
        for b in range(NB):
            num_u = num_d = den = 0.0
            for ri, r in enumerate(REPS):
                if A[b, ri] == 0:
                    continue
                te = te_of[T][r]
                y = y_of[T][te]
                w = C[b, [sp_ix[s] for s in spk_of[T][te]]]
                if w.sum() == 0:
                    continue
                D = Dt[(Dt.target == T) & (Dt.student == kr.student) & (Dt.rep == r)].sort_values("pair").D.to_numpy()
                Lall = np.load((d77 if kr.wave == "e77" else d78) / f"{T}_r{r}_logp.npy", mmap_mode="r")
                rid = x[x.rep == r].row_id.to_numpy()
                vals = [naive(np.asarray(Lall[i], float), y, w, D) for i in rid]
                vu, vd = np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals])
                num_u += A[b, ri] * vu
                num_d += A[b, ri] * vd
                den += A[b, ri]
            if den == 0:
                continue
            eu = abs(num_u / den - z["uar"][b, kr.key_id])
            ed = abs(num_d / den - z["dirloss"][b, kr.key_id])
            worst = max(worst, eu, ed)
            if eu > 1e-3 or ed > 1e-3:
                print("ずれ", T, key, b, num_u / den, z["uar"][b, kr.key_id], num_d / den, z["dirloss"][b, kr.key_id])
    print(T, "鍵", len(pick), "回", NB, "完了", flush=True)
print("素朴な計算との差の最大（ポイント．npz は float32）：", worst)
