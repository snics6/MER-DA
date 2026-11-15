"""探索（関門を通していない軽い試し）：ソースが 3 つ以上のとき，「同じコーパスの上で比べた強さ」の重みが働きそうか．
6 コーパスの仮のターゲットの組み方（実験 18 と同じ：評価のターゲット T と仮のターゲット c を除いたコーパスがソース）で，
(1) 重みの大きいソースが，仮のターゲット c で強いソースか（c の全発話・全ラベルで測る．参照），
(2) 重みを付けた出発点（K = 0．学び直し無し）の c の全発話の上の UAR と d′ の合計，を等しい平均と比べる．
分類器は保存済みのソースの分類器を当てるだけ（学び直さない）．CPU で流す．"""
import itertools
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai18_probe as P18  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

dai1.ENC["enc"] = DD.ENC1
CORP = P18.CORP6
MODS = {"A": [c for c in CORP if "A" in P18.MODS_OF[c]], "V": [c for c in CORP if "V" in P18.MODS_OF[c]], "T": [c for c in CORP if "T" in P18.MODS_OF[c]]}
A = {}      # (s, u, m) → s の分類器を u の全発話に当てた d′ の合計
Z = {}      # (s, u, m) → 点（中心化しない logits）


def away(s, u, m):
    if (s, u, m) not in A:
        P = P18.src_probs_on(s, u, m)
        A[(s, u, m)] = float(P16.dprime(P, P18.C6.get(u).y)[0].sum())
        Z[(s, u, m)] = np.log(np.clip(P, 1e-300, None))
    return A[(s, u, m)]


def weights(kind, srcs, m, truth=None):
    n = len(srcs)
    if kind == "orc_prop":     # 参照：仮のターゲットでの強さ（全ラベル）に比例．実際には使えない
        return truth / truth.sum()
    if kind == "orc_best":     # 参照：仮のターゲットで最も強いソースだけ
        w = np.zeros(n); w[truth.argmax()] = 1.0
        return w
    if kind.startswith("wc") and kind != "wc":      # wc を p 乗して差を広げた版（wc4 は 4 乗）
        w = weights("wc", srcs, m) ** float(kind[2:])
        return w / w.sum()
    if kind == "eq" or n < 2:
        return np.full(n, 1.0 / n)
    if kind == "w":            # 別のソースで測った値（平均）に比例（承認済みの形を 3 つ以上のソースに広げたもの）
        q = np.array([np.mean([away(s, u, m) for u in srcs if u != s]) for s in srcs])
    else:                      # wc：同じコーパス u の上で，ほかのソースの分類器の平均と比べた強さの平均（n >= 3 のときだけ意味がある）
        if n < 3:
            return np.full(n, 1.0 / n)
        q = np.array([np.mean([away(s, u, m) / np.mean([away(p, u, m) for p in srcs if p != u]) for u in srcs if u != s]) for s in srcs])
    return q / q.sum() if q.sum() > 0 else np.full(n, 1.0 / n)


rows, pair_rows = [], []
for m, corp in MODS.items():
    for c in corp:                                  # 仮のターゲット
        for T in CORP:                              # 評価のターゲット（ソースから除く）
            if T == c:
                continue
            srcs = [s for s in corp if s not in (T, c)]
            if len(srcs) < 2:
                continue
            y = P18.C6.get(c).y
            truth = np.array([away(s, c, m) for s in srcs])          # 仮のターゲットでの強さ（全ラベル．参照）
            rec = dict(m=m, c=c, T=T, n_src=len(srcs), srcs=",".join(srcs))
            for kind in ("eq", "w", "wc", "wc4", "wc16", "orc_prop", "orc_best"):
                w = weights(kind, srcs, m, truth)
                z = sum(wi * Z[(s, c, m)] for wi, s in zip(w, srcs))   # 重み付きの出発点の点（線形なのでパラメータの重み付き和と同じ）
                Pm = torch.softmax(torch.tensor(z), -1).numpy()
                rec[f"uar_{kind}"] = 100 * S.uar(y, z.argmax(1))
                rec[f"d_{kind}"] = float(P16.dprime(Pm, y)[0].sum())
                rec[f"wmax_{kind}"] = float(w.max())
                if kind in ("w", "wc"):
                    rec[f"top_{kind}"] = bool(w.argmax() == truth.argmax())
                    for i, j in itertools.combinations(range(len(srcs)), 2):
                        if abs(w[i] - w[j]) > 1e-12:
                            pair_rows.append(dict(m=m, c=c, T=T, kind=kind, n_src=len(srcs), agree=bool((w[i] > w[j]) == (truth[i] > truth[j]))))
            rows.append(rec)
d = pd.DataFrame(rows)
p = pd.DataFrame(pair_rows)
pd.set_option("display.width", 250)
out = "/tmp/claude-1000/-home-snics-MER-DA/76085a37-4610-4922-b8ae-ced34f163c39/scratchpad/dai19/"
d.to_csv(out + "quick_common_units.csv", index=False)
p.to_csv(out + "quick_common_pairs.csv", index=False)
g = d[d.n_src >= 3]
print("組の数（ソース 3 つ以上）：", g.groupby("m").size().to_dict(), "／ソース 2 つ：", d[d.n_src == 2].groupby("m").size().to_dict())
print("\n(1) 重みの大きいソースが仮のターゲットで強いソースか（ソースの対ごと．ソース 3 つ以上の組）")
print(p[p.n_src >= 3].groupby(["kind", "m"]).agree.agg(["sum", "size", "mean"]).round(3))
print(p[p.n_src >= 3].groupby("kind").agree.agg(["sum", "size", "mean"]).round(3))
print("\n最も重いソースが，仮のターゲットで最も強いソースと一致した割合（でたらめなら 1 ÷ ソースの数）")
print(g.groupby("m")[["top_w", "top_wc"]].mean().round(3), g[["top_w", "top_wc"]].mean().round(3).to_dict(), "でたらめ", round(float((1 / g.n_src).mean()), 3))
print("\n(2) K = 0 の出発点の UAR（仮のターゲットの全発話）：重み付き − 等しい平均（ソース 3 つ以上の組）")
g = g.assign(dw=g.uar_w - g.uar_eq, dwc=g.uar_wc - g.uar_eq, ddw=g.d_w - g.d_eq, ddwc=g.d_wc - g.d_eq)
print(g.groupby(["m", "c"])[["uar_eq", "dw", "dwc", "wmax_w", "wmax_wc"]].mean().round(2))
print(g.groupby("m")[["uar_eq", "dw", "dwc", "ddw", "ddwc"]].mean().round(3))
for k_ in ("wc4", "wc16", "orc_prop", "orc_best"):
    g[f"d_{k_}_uar"] = g[f"uar_{k_}"] - g.uar_eq
print("\n参照と，差を広げた版：K = 0 の出発点の UAR − 等しい平均")
print(g.groupby("m")[["d_wc4_uar", "d_wc16_uar", "d_orc_prop_uar", "d_orc_best_uar", "wmax_wc4", "wmax_wc16"]].mean().round(2))
print(g.groupby(["m", "c"])[["uar_eq", "d_wc16_uar", "d_orc_prop_uar", "d_orc_best_uar"]].mean().round(2))
print("全体：", g[["d_wc4_uar", "d_wc16_uar", "d_orc_prop_uar", "d_orc_best_uar"]].mean().round(2).to_dict())
print("全体：", g[["dw", "dwc", "ddw", "ddwc"]].mean().round(3).to_dict(), "／UAR が上がった組の割合", {"w": round(float((g.dw > 0).mean()), 3), "wc": round(float((g.dwc > 0).mean()), 3)})
