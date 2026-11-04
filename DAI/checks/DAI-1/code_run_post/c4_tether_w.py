"""tether_w.json を独立に作り直す：(種類, K) ごとに none の 30 単位の平均 UAR が最大の (C′, C_b)（同点は小さい C′・小さい C_b），
K ごとに積の候補のラベルでの UAR の平均が最大の指数（同点：(1,1,1) からの差の和 → 最大の指数が 1 に近い → 辞書順）。"""
import json, itertools, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/home/snics/MER-DA"); P = ROOT / "DAI/checks/DAI-1/code_run_post"; U = ROOT / "results/dai1/pseudo/units"
p1 = pd.read_pickle(P / "rows_p1.pkl"); p2 = pd.read_pickle(P / "rows_p2.pkl")
C = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
probs = []
edge = []
for T in C:
    tw = json.load(open(ROOT / f"results/dai1/pseudo/{T}_tether_w.json"))
    exp_units = sorted(f"{c}_f{f}_r{r}" for c in C if c != T for f in range(5) for r in (0, 1))
    if sorted(tw["units"]) != exp_units: probs.append((T, "units"))
    if tw["commit"] != "896136c9d00b404409aaed2c7b625a1f2ea8a8f0": probs.append((T, "commit"))
    d = p1[(p1["T"] == T) & (p1.method == "none")]
    for kind in ["A", "V", "T", "AV", "AT", "VT", "AVT"]:
        for K in (1, 5, 10, 25):
            g = d[(d.kind == kind) & (d.K == K)]
            cnt = g.groupby(["Cw", "Cb"]).size()
            assert (cnt == 30).all() and len(cnt) == 14, (T, kind, K)
            m = g.groupby(["Cw", "Cb"]).uar.mean()
            best = m.max()
            top = sorted([ix for ix in m.index if m[ix] == best])
            mine = top[0]
            got = tuple(tw["tether"][f"{kind}|{K}"])
            if tuple(map(float, mine)) != tuple(map(float, got)): probs.append((T, kind, K, mine, got))
            edge.append(dict(T=T, kind=kind, K=K, Cw=mine[0], Cb=mine[1], edge=mine[0] in (0.001, 1.0), ties=len(top)))
    q = p2[(p2["T"] == T) & (p2.method == "product")]
    for K in (0, 1, 5, 10, 25):
        g = q[q.K == K]
        cnt = g.groupby("teacher").size()
        assert (cnt == 30).all() and len(cnt) == 63, (T, K, cnt.size)
        m = g.groupby("teacher").uar_pool.mean()
        best = m.max()
        c = [tuple(float(x) for x in w.strip("()").split(",")) for w in m.index[m == best]]
        mine = min(c, key=lambda w: (sum(abs(x - 1) for x in w), abs(max(w) - 1), w))
        got = tuple(tw["w"][str(K)])
        if mine != tuple(got): probs.append((T, "w", K, mine, got))
        # 1e-9 の許容で同点が増えるか
        c9 = m.index[m >= best - 1e-9]
        if len(c9) != (m == best).sum(): probs.append((T, "w-tol", K))
    # npz に残した引き戻し・指数
    tag_t = [f"{k.split('|')[0]}|{k.split('|')[1]}|{v[0]}|{v[1]}" for k, v in sorted(((k.split('|')[0], int(k.split('|')[1])), v) for k, v in tw["tether"].items()) for k in [f"{k[0]}|{k[1]}"]]
    for u in tw["units"]:
        for p in (2, 3):
            z = np.load(U / f"{T}_{u}_p{p}.npz")
            teth = {(s.split("|")[0], int(s.split("|")[1])): (float(s.split("|")[2]), float(s.split("|")[3])) for s in z["tether"]}
            want = {(k.split("|")[0], int(k.split("|")[1])): tuple(map(float, v)) for k, v in tw["tether"].items()}
            if teth != want: probs.append((T, u, p, "npz tether"))
            if p == 3:
                ws = {int(s.split("|")[0]): tuple(float(x) for x in s.split("|")[1].strip("()").split(",")) for s in z["wsel"]}
                if ws != {int(k): tuple(map(float, v)) for k, v in tw["w"].items()}: probs.append((T, u, "npz wsel"))
print("problems:", probs)
E = pd.DataFrame(edge)
print("tether edge (C'=0.001 or 1) count by T:", E.groupby("T").edge.sum().to_dict(), "of", E.groupby("T").size().to_dict())
print("tether edge by Cw value:", E[E.edge].groupby(["T", "Cw"]).size().to_dict())
print("exact ties in tether:", int((E.ties > 1).sum()))
E.to_csv(P / "tether_choice.csv", index=False)
