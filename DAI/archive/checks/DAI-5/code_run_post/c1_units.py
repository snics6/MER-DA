"""計算の後の点検 1：評価の回の 200 単位のそろい方・格子・コミット・収束・有限・行の uar の作り直し（台本の関数は使わない．格子の文字列だけ dai5 から読む）。"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA")
U = ROOT / "results/dai5/eval/units"
COMMIT = "f186d02fba35714a310f330180396ade3500f6cc"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
FZ = sha(ROOT / "results/dai5/select/frozen.json"); GATE = sha(ROOT / "DAI/frozen/eval_gate_dai5.json"); FZ1 = sha(ROOT / "DAI/frozen/dai1_frozen.json")
fz5 = json.load(open(ROOT / "results/dai5/select/frozen.json"))
Ts, STU, KS, BET = ["AFEW", "IEMOCAP", "MOSEI", "MELD"], ["A", "V", "T", "AVT"], [0, 1, 5, 10, 25], [0.0, 2.5, 5.0, 10.0, 20.0]
TETHER = None
want = {f"{T}_ALL_f{f}_r{r}_p5.{e}" for T in Ts for f in range(5) for r in range(10) for e in ("npz", "csv")}
have = {p.name for p in U.iterdir()}
print("ファイル：期待", len(want), "ある", len(have), "欠け", sorted(want - have)[:5], "余り", sorted(have - want)[:5])
# 格子（設計 1・2・4 節から数える）
grid = [f"srcw|{K}|{b}" for K in KS for b in ([0.0] if K == 0 else BET)]
grid += [f"{m}|{K}|{b}" for m in ("srcw_mod", "srcw_src") for K in (5, 25) for b in BET[1:]]
grid += [f"srcw_top1|{K}|inf" for K in KS[1:]]
ylab = {T: dict(zip(*[pd.read_csv(ROOT / f"DAI/splits/{T}.csv", keep_default_na=False)[c].astype(str) for c in ("sample_id", "label_4")])) for T in Ts}
bad = dict(meta=0, grid=0, nprob=0, conv=0, finite=0, uar=0, dup=0, predrange=0, wsum=0, tether_beta=0, rows_meta=0, items=0)
n_prob_total = 0; maxgrad = 0.0; maxdiff = 0.0
per_method = {}
for T in Ts:
    for f in range(5):
        for r in range(10):
            nm = f"{T}_ALL_f{f}_r{r}"
            z = np.load(U / f"{nm}_p5.npz"); d = pd.read_csv(U / f"{nm}_p5.csv", low_memory=False)
            ok = str(z["commit"]) == COMMIT and str(z["dirty"]) == "False" and str(z["gpu"]) == "NVIDIA RTX A5000" and str(z["frozen_sha256"]) == FZ \
                and str(z["gate_sha256"]) == GATE and str(z["dai1_frozen_sha256"]) == FZ1
            bad["meta"] += not ok
            okr = set(d.commit.astype(str)) == {COMMIT} and set(d.dirty.astype(str)) == {"False"} and set(d.gpu.astype(str)) == {"NVIDIA RTX A5000"} \
                and set(d.frozen_sha256.astype(str)) == {FZ} and set(d.gate_sha256.astype(str)) == {GATE}
            bad["rows_meta"] += not okr
            g = [str(x) for x in z["grid"]]
            gt = grid + [x for x in g if x.startswith("srcw_tether")]
            bad["grid"] += not (g[:len(grid)] == grid and len(g) == len(grid) + 2 * 14 and len(set(g)) == len(g))
            sol = d[d.method.isin(["srcw", "srcw_mod", "srcw_src", "srcw_top1", "srcw_tether"])]
            for m_, c_ in sol.method.value_counts().items():
                per_method.setdefault(m_, set()).add(c_)
            n_prob_total += len(sol)
            bad["nprob"] += len(sol) != 276
            bad["conv"] += int(((sol.grad.astype(float) > 1e-5) | (sol.stalled.astype(str) == "True")).sum())
            maxgrad = max(maxgrad, float(sol.grad.astype(float).max()))
            bad["finite"] += int((sol.finite.astype(str) != "True").sum())
            key = sol.assign(Cw=sol.Cw.fillna(-1), Cb=sol.Cb.fillna(-1))[["method", "student", "K", "beta", "Cw", "Cb"]].round(9)
            bad["dup"] += int(key.duplicated().sum())
            # npz：判定・W・b・items
            sid = z["sample_test"].astype(str); y = np.array([int(ylab[T][x]) for x in sid])
            n_items = 0
            for s in STU:
                for K in KS:
                    for px in ("p5", "p5m", "p5s", "p5o", "p5t"):
                        k = f"{px}_{s}_K{K}"
                        if f"{k}_items" not in z.files:
                            continue
                        it = [json.loads(x) for x in z[f"{k}_items"]]; P = z[f"{k}_pred"]; W = z[f"{k}_W"]; b = z[f"{k}_b"]
                        n_items += len(it)
                        bad["items"] += not (P.shape == (len(it), len(sid)) and W.shape[0] == len(it) and b.shape[0] == len(it))
                        bad["predrange"] += int(((P < 0) | (P > 3)).any()) + int(not (np.isfinite(W).all() and np.isfinite(b).all()))
                        meth = {"p5": "srcw", "p5m": "srcw_mod", "p5s": "srcw_src", "p5o": "srcw_top1", "p5t": "srcw_tether"}[px]
                        rr = sol[(sol.method == meth) & (sol.student == s) & (sol.K == K)]
                        for j, x in enumerate(it):
                            u_ = 100 * np.mean([(P[j][y == c] == c).mean() for c in range(4) if (y == c).any()])
                            q = rr[np.isclose(rr.beta.astype(float), float(x["beta"])) | (np.isinf(float(x["beta"])) & np.isinf(rr.beta.astype(float)))]
                            if px == "p5t":
                                q = q[np.isclose(q.Cw.astype(float), x["Cw"]) & np.isclose(q.Cb.astype(float), x["Cb"])]
                                bad["tether_beta"] += not np.isclose(float(x["beta"]), float(fz5[T]["select"][str(K)]["beta"]))
                            if len(q) != 1:
                                bad["uar"] += 1; continue
                            dd = abs(float(q.uar.iloc[0]) - u_); maxdiff = max(maxdiff, dd); bad["uar"] += dd > 1e-6
            bad["items"] += n_items != 276
            for K in KS:
                for b_ in ([0.0] if K == 0 else BET):
                    w = z[f"p5_K{K}_w_b{b_}"]; bad["wsum"] += abs(w.sum() - 3) > 1e-9
print("問の総数", n_prob_total, "（期待 200 × 276 =", 200 * 276, "）．版ごとの 1 単位あたりの問の数", {k: sorted(v) for k, v in per_method.items()})
print("勾配の最大", maxgrad, "．行の uar と int8 の判定から作り直した UAR の差の最大", maxdiff)
print("食い違いの数", bad)
