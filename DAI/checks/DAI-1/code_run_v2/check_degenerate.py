"""退化した場合の検算（CPU）：疑似ラベルの重みを 0 にした問が比較の基準・ゼロからの版と同じ解になるか．
あわせて，引き戻しの行（p3t）の選んだ引き戻しの点が格子の行（p3）の同じ点と同じ判定か（保存からだけ）。UAR の値は出さない。"""
import sys, json, math
from pathlib import Path
import numpy as np, torch
import dai1, dai_solver as S, dai_data as DD
O = Path(sys.argv[1]); T, c, f, r = "MOSEI", "AFEW", 0, 0
tw = json.load(open(O / "pseudo" / f"{T}_tether_w.json"))
teth = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in tw["tether"].items()}
w = {int(k): tuple(v) for k, v in tw["w"].items()}
base = O / "pseudo" / "units" / f"{T}_{c}_f{f}_r{r}"
p1, p2, p3 = np.load(f"{base}_p1.npz"), np.load(f"{base}_p2.npz"), np.load(f"{base}_p3.npz")
# (1) 保存だけ：p3t の選んだ引き戻しの点 と p3 の (ρ=1, a=1) の点
mis = tot = 0
for s in DD.STUDENTS:
    for K in dai1.KS_ALL:
        cw, cb = teth[(s, max(K, 1))]
        it_t = [json.loads(x) for x in p3[f"p3t_{s}_K{K}_items"]]; it_g = [json.loads(x) for x in p3[f"p3_{s}_K{K}_items"]]
        for j, it in enumerate(it_t):
            if np.isclose(it["Cw"], cw) and np.isclose(it["Cb"], cb):
                h = [i for i, g in enumerate(it_g) if g["method"] == "joint" and g["teacher"] == it["teacher"] and g["rho"] == 1.0 and g["a"] == 1]
                assert len(h) == 1
                tot += 1; mis += int((p3[f"p3t_{s}_K{K}_pred"][j] != p3[f"p3_{s}_K{K}_pred"][h[0]]).any())
                dW = float(np.abs(p3[f"p3t_{s}_K{K}_W"][j] - p3[f"p3_{s}_K{K}_W"][h[0]]).max())
print(f"引き戻しの行と格子の行（同じ問を別の束で解いた）：判定が違う問 {mis}/{tot}，最後の W の差 {dW:.1e}")
# (2) 解き直し：重み 0
dai1.RUN["commit"] = "x"
u = dai1.Unit(O, T, c, [s for s in DD.others(T) if s != c], f, r)
TK = dai1.build_tk(u, p1, p2, teth)
worst = {"none": 0.0, "kd": 0.0, "scratch": 0.0}; flips = {"none": 0, "kd": 0, "scratch": 0}; n = 0
for s in DD.STUDENTS:
    for K in (1, 5, 25):
        tk = TK[K]; gk = u.gK(K); Q = dai1.product(tk, w[K], 0); own = tk[f"none_{s}"][0]
        W0, b0 = u.start[s]; cw, cb = teth[(s, K)]
        Gj, G2j, Tj = dai1.make_G(u, "joint", Q, own, gk, dict(rho=0.0, a=1))
        Gk, G2k, Tk = dai1.make_G(u, "kd", Q, own, gk, dict(temp=2, rho=0.0))
        res = S.fit(u.Xp[s], torch.stack([Gj, Gk]), W0, b0, Cw=cw, Cb=cb, G2=torch.stack([G2j, G2k]), T2=[Tj, Tk])
        Wn, bn = dai1.none_wb(u, p1, s, K, teth)
        zn = S.logits(u.Xt[s], Wn, bn).argmax(1)
        Z = S.logits(u.Xt[s], res["W"], res["b"]).argmax(2)
        for j, name in enumerate(("none", "kd")):
            worst[name] = max(worst[name], float((res["W"][j] - Wn).abs().max())); flips[name] += int((Z[j] != zn).sum())
        # ゼロからの提案（ρ=0）とゼロからの K 本だけ（C=1）
        Gz, _, _ = dai1.make_G(u, "joint", Q, own, gk, dict(rho=0.0, a=1))
        d = u.Xp[s].shape[1]
        rz = S.fit(u.Xp[s], Gz[None], torch.zeros(4, d), torch.zeros(4), Cw=1.0, Cb=math.inf)
        items = [str(x) for x in p1[f"p1_{s}_tgt_items"]]; jj = items.index(f"{K}|1.0")
        Wt = torch.as_tensor(p1[f"p1_{s}_tgt_W"][jj], dtype=torch.float64)
        bt = torch.as_tensor(p1[f"p1_{s}_tgt_b"][jj], dtype=torch.float64)
        worst["scratch"] = max(worst["scratch"], float((rz["W"][0] - Wt).abs().max()))
        flips["scratch"] += int((S.logits(u.Xt[s], rz["W"], rz["b"]).argmax(2)[0] != S.logits(u.Xt[s], Wt, bt).argmax(1)).sum())
        n += 1
print(f"重み 0 の問（生徒 4 × K 3 = {n} 組）：W の差の最大", {k: f"{v:.1e}" for k, v in worst.items()}, "テストの判定が食い違った発話", flips)
