"""gate-code の再検査（E76．c452b38）：手 9〜11 を別の道筋で確かめる。

python checks/E76/code_run/verify2.py <out_dir> <kind> <target>
- 手 9・10・11 の保存した係数から点数を作り直し，保存した点数と比べる（台本の coef_check は手 9〜11 を見ない）
- 手 11 の解が「中心合わせした K 本で，罰則の中心 ＝ 手 2 の解，C′ ＝ ② の C」の目的関数の停留点か（勾配の最大）を，
  出発点（手 2 の解）での勾配と並べる
- 手 10 の α ごとの行の数と，csv の uar と保存した点数の UAR の一致
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "measure"))
import shift  # noqa: E402
from data import load_any  # noqa: E402

out, kind, T = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
z = np.load(out / kind / f"{T}.npz")
d = pd.read_csv(out / kind / f"{T}.csv", comment="#", dtype={"K": str, "source": str})
d["K"], d["source"] = d.K.fillna(""), d.source.fillna("")
meta = json.loads((out / kind / f"{T}.json").read_text())
CG = meta["settings"]["C_grid"]
ALPHAS = meta["settings"]["alphas"]
_, X = load_any(T, kind)
y = z["y"]
reps = list(z["reps"])
KH = ("1", "5", "10")
sources = sorted({k.split("__")[0][2:] for k in z.files if k.startswith("s_") and k.endswith("__ver")})
res = {"coef_h9": 0.0, "coef_h10": 0.0, "coef_h11": 0.0, "uar_csv_h9_h10_h11": 0.0,
       "h11_grad_at_solution_max": 0.0, "h11_grad_at_anchor_min": np.inf}


def softmax(L):
    L = L - L.max(1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(1, keepdims=True)


for s in sources:
    for ir, r in enumerate(reps):
        te = z[f"test_idx_r{r}"]
        j = z[f"s_{s}__ver"][ir]
        C = float(z[f"s_{s}__v{j}_C"])
        for kh, K in enumerate(KH):
            for h, Xf in ((9, X), (11, X - z["mean_t"])):
                key = f"h{h}_{s}_r{r}"
                if f"{key}_logit" not in z.files:
                    continue
                L = Xf[te] @ z[f"{key}_A"][kh].T.astype(float) + z[f"{key}_a0"][kh]
                res[f"coef_h{h}"] = max(res[f"coef_h{h}"], float(np.abs(L - z[f"{key}_logit"][kh]).max()))
                u = shift.uar(y[te], z[f"{key}_logit"][kh].argmax(1))
                row = d[(d.method == f"h{h}") & (d.source == s) & (d.K == K) & (d.rep == r)]
                res["uar_csv_h9_h10_h11"] = max(res["uar_csv_h9_h10_h11"], abs(u - row.uar.iloc[0]))
            key = f"h10_{s}_r{r}"
            if f"{key}_logit" in z.files:
                for ia, a in enumerate(ALPHAS):
                    L = X[te] @ z[f"{key}_A"][ia, kh].T.astype(float) + z[f"{key}_a0"][ia, kh]
                    res["coef_h10"] = max(res["coef_h10"], float(np.abs(L - z[f"{key}_logit"][ia, kh]).max()))
                    u = shift.uar(y[te], z[f"{key}_logit"][ia, kh].argmax(1))
                    row = d[(d.method == "h10") & (d.source == s) & (d.K == K) & (d.rep == r) & np.isclose(d.alpha, a)]
                    res["uar_csv_h9_h10_h11"] = max(res["uar_csv_h9_h10_h11"], abs(u - row.uar.iloc[0]))
            # 手 11 の停留点の確かめ（標準化した座標で．係数は float32 なので勾配は 1e-5 前後までしか下がらない）
            key = f"h11_{s}_r{r}"
            if f"{key}_A" not in z.files:
                continue
            idx = z[f"kidx_r{r}_K{K}"]
            Xc = (X - z["mean_t"])[idx]
            mu, sd = Xc.mean(0), Xc.std(0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            Z = (Xc - mu) / sd
            yk = y[idx]
            cnt = np.bincount(yk, minlength=4)
            sw = (len(yk) / (4 * cnt))[yk]
            S = sw.sum()
            A2, a20 = z[f"h2_{s}__v{j}_A"].astype(float), z[f"h2_{s}__v{j}_a0"]
            W_anc, b_anc = A2 * sd, a20 + mu @ A2.T

            def grad(W, b):
                P = softmax(Z @ W.T + b)
                Y = np.eye(4)[yk]
                G = (sw[:, None] * (P - Y))
                return max(np.abs(G.T @ Z / S + (W - W_anc) / (C * S)).max(), np.abs(G.sum(0) / S).max())

            A11, a110 = z[f"{key}_A"][kh].astype(float), z[f"{key}_a0"][kh]
            W, b = A11 * sd, a110 + mu @ A11.T
            res["h11_grad_at_solution_max"] = max(res["h11_grad_at_solution_max"], float(grad(W, b)))
            res["h11_grad_at_anchor_min"] = min(res["h11_grad_at_anchor_min"], float(grad(W_anc, b_anc)))
res["rows_by_method"] = d.groupby("method").size().to_dict()
res["h10_alpha_counts"] = d[d.method == "h10"].groupby("alpha").size().to_dict()
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
