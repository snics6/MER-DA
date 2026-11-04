"""gate-code の (a) の実演：試走の出力（oute/）と実験 1 の保存（d1e/）だけから，学び直さずに後から出せる量を CPU で作る．値は出さず，作れた数だけを出す．
(1) 予測を付ける側の種類ごと（生徒のモダリティを含まない tt の行）の位置の誤りと方向の誤り（pickle の refit と oracle）
(2) 切片だけの版（kbias）の位置の誤り = UAR(W0, b*(W0)) − UAR(W0, b_kbias)（start の refit と，実験 1 の kbias の判定を 2 組に当てた orig）
(3) 族 5 と同じ形の量を族 3 の相手で（K=5．pickle の partner|5）と，生徒 A・V の筋書きの対（実験 1 の判定から）"""
import sys, json, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
exec(open(HERE / "run_trial.py").read().split("sys.argv = [")[0])
import numpy as np, pandas as pd
import dai3 as D3, dai_data as DD
oute, d1e = HERE / "oute", HERE / "d1e"
fz1 = json.load(open(d1e / "select" / "frozen.json"))
T = "AFEW"
U = {(f, r): pickle.load(open(oute / "eval" / "units" / f"{T}_f{f}_r{r}.pkl", "rb")) for f in (0, 1) for r in (0, 1)}
mode, _ = D3.jk_mode(T); sz = D3.sizes_of(U)
n1 = 0
for s in "AVT":
    for nm in D3.noown_types(s, ["IEMOCAP"]):
        name = f"{s}|tt|5|{nm}"
        for vi in range(7):
            def fn(R):
                a, o = R["refit"][name], R["refit"][f"{s}|oracle"]
                return np.where(a["valid"][vi], o["refit"][vi] - a["refit"][vi], np.nan)
            e = D3.jk_stats(D3.to_vals(U, fn), mode, sz)
            n1 += int(np.isfinite(e["effect"]) and np.isfinite(e["se"]))
print("(1) tt の種類ごとの方向の誤りの区間を作れたセル", n1)
n2 = 0
for (f, r), R in U.items():
    base = d1e / "eval" / "units" / f"{T}_ALL_f{f}_r{r}"
    p1 = np.load(f"{base}_p1.npz")
    df = D3.corpus_df(T); te = df["fold"].to_numpy() == f
    y = df["label_4"].to_numpy()[te]; cl = df["cluster"].to_numpy().astype(str)[te]; sid = df["sample_id"].to_numpy().astype(str)[te]
    assert (p1["sample_test"].astype(str) == sid).all()
    grp = D3.test_halves(y, cl, sid); Yk = D3.onehot(y, 4)
    for s in DD.STUDENTS:
        items = [str(x) for x in p1[f"p1_{s}_kbias_items"]]
        cb = float(fz1[T]["select"][f"{s}|5|kbias|-"]["Cb"])
        pr = p1[f"p1_{s}_kbias_pred"][items.index(f"5|{cb}")]
        hit = (pr == y)[None]
        orig = 0.5 * (D3.uar_of(hit, (grp == 0)[None], Yk)[0] + D3.uar_of(hit, (grp == 1)[None], Yk)[0])
        pos = R["refit"][f"{s}|start"]["refit"][0, 0] - orig
        n2 += int(np.isfinite(pos))
print("(2) 切片だけの版の位置の誤りを作れた (単位，生徒)", n2)
n3 = 0
for s in DD.STUDENTS:
    for vi in range(1, 7):
        e = D3.jk_stats(D3.to_vals(U, D3.refit_diff(f"{s}|partner|5", f"{s}|none|5", vi)), mode, sz)
        n3 += int(np.isfinite(e["effect"]))
print("(3) 族 3 の相手 − none の対の切片を当て直した後の差を作れたセル", n3)
