"""gate-code 2 回目の検算：評価の回だけで通る解き直しの経路（sel を渡した pass_map と pass_xmuda）を，仮のターゲットの 1 単位の上で動かす。
sel は試走の選ぶ段の出力（sel_full/select/frozen.json）から作る。UAR は出さない。引数：trial3 の置き場，frozen.json，map の npz を置く一時の場所"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1  # noqa: E402
import dai2  # noqa: E402
import dai2_grad as G  # noqa: E402
import dai_data as DD  # noqa: E402

d1, fzp, tmp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
dai2.RUN["dai1_dir"] = str(d1)
teth, _ = dai2.tether_pseudo(d1, "MOSEI", True)
fz = json.load(open(fzp))["MOSEI"]["select"]
sel = {(s, K, m): fz[f"{s}|{K}|{m}"] for s in DD.STUDENTS for K in G.KS_ALL for m in ("dann", "mme", "mapctl")}
sel.update({(m, K, "xmuda"): fz[f"{m}|{K}|xmuda"] for m in "AVT" for K in G.KS_ALL})
u = dai1.Unit(d1, "MOSEI", "AFEW", ["IEMOCAP", "MELD"], 0, 0)
store, rows = {}, []
G.pass_map(u, store, rows, teth, 0, False, False, students=("A",), sel=sel)
rep = [r for r in rows if r["method"].endswith("_rep")]
print("map の解き直し（A）：問", len(rep), " roles", sorted({r["roles"] for r in rep}))
for K in G.KS_ALL:
    for m in ("dann", "mme", "mapctl"):
        p = sel[("A", K, m)]
        hit = [r for r in rep if r["method"] == f"{m}_rep" and r["K"] == K and r["lr"] == p["lr"] and r["iters"] == p["iters"] and r["mu"] == p["mu"]
               and (m == "mapctl" or r["lam"] == p["lam"]) and (m != "mme" or r["sw"] == p["sw"])]
        assert len(hit) == 1 and "selected" in hit[0]["roles"], (K, m, p)
    px = sel[("A", K, "xmuda")]
    hit = [r for r in rep if r["method"] == "mapctl_rep" and r["K"] == K and r["lr"] == px["lr"] and r["iters"] == px["iters"] and r["mu"] == px["mu"]]
    assert len(hit) == 1 and "ctl_of_xmuda_selected" in hit[0]["roles"], (K, px)
print("選んだ点と xMUDA 型の選んだ点の mapctl の相手が，解き直しに 1 問ずつある（A，K 5 つ）")
tmp.mkdir(parents=True, exist_ok=True)
mp = tmp / "map.npz"
np.savez_compressed(mp, **{k: np.asarray(v) for k, v in store.items()})
st2, rw2 = {}, []
G.pass_xmuda(u, st2, rw2, teth, 0, False, sel=sel, map_npz=mp)
xr = [r for r in rw2 if r["method"] == "xmuda_rep"]
print("xmuda の解き直し：行", len(xr), " roles", sorted({r["roles"] for r in xr}))
a = [r for r in xr if r["student"] == "A"]
print("A の解き直しの行の dis_ctl・dis_ctl_rep が NaN でない数", sum(np.isfinite(r["dis_ctl"]) for r in a), sum(np.isfinite(r["dis_ctl_rep"]) for r in a), "/", len(a))
