"""DAI-3 の前提（設計 5 節の最後の行）：研究室で作り直した標準化の特徴に，ポッド（H200）で保存した float32 の W・b を当てた判定が，
保存した int8 の判定と一致するか．(T, c) ごとに 1 単位，生徒 4，K = 5・25 の none・提案・選択なし・tt の全種類（CPU）。"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1, dai3, dai_data as DD, dai_solver as S
out1 = Path("/home/snics/MER-DA/results/dai1"); fz1 = json.load(open(out1 / "select/frozen.json"))
tot = 0; nonmatch = []; worst = 1.0
for T in DD.CORPORA:
    for c in DD.others(T):
        f, r = (DD.CORPORA.index(T) + DD.CORPORA.index(c)) % 5, 1
        sv = dai3.Saved(out1, "pseudo", T, c, f, r, fz1[T])
        for s in DD.STUDENTS:
            for K in (5, 25):
                cw, cb = sv.teth[(s, K)]
                it1 = [str(x) for x in sv.p1[f"p1_{s}_items"]]
                cand = {"none": (sv.none(s, K), sv.p1[f"p1_{s}_pred"][it1.index(f"{K}|{cw}|{cb}")])}
                items = sv.items(s, K)
                for j, it in enumerate(items):
                    if it["method"] in ("tt", "joint", "kd"):
                        cand[json.dumps(it)] = (sv.p3_wb(s, K, j), sv.p3[f"p3_{s}_K{K}_pred"][j])
                for nm, ((W, b), saved) in cand.items():
                    p = S.logits(sv.u.Xt[s], W, b).argmax(1).cpu().numpy()
                    a = float((p == saved).mean()); tot += 1; worst = min(worst, a)
                    if a < 1: nonmatch.append((T, c, f, r, s, K, nm[:60], round(a, 5), int((p != saved).sum())))
        dai1.Corpus.cache.pop(c, None)
print("classifiers checked", tot, "not 100%:", len(nonmatch), "worst agreement", worst)
for x in nonmatch[:15]: print(x)
