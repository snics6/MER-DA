"""ML の査読者の検算（DAI-21 第 2 版）：CREMA-D の単位に足した「出発点（K = 0）」と「上限（候補の全ラベルで学ぶ分類器）」の計算が，CPU で止まらずに解けるか．

台本 unit_cremad の 397〜404 行と同じ呼び方（P18.fit_none に候補の全部の位置を渡す．引き戻しの強さは K = 25 の値）．
fit_none は勾配の最大が 1e-5 を超えると止まる．ここでは解けること，勾配，所要，K 本だけの分類器の候補の上の判定の形だけを見る（UAR は計算しない）．
使い方：CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <この台本>
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1  # noqa: E402
import dai18_probe as P18  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

dai1.ENC["enc"] = DD.ENC1
CD, HOLD, FULL_K = "CREMA-D", "RAVDESS", 25
t18 = json.load(open(DD.ROOT / "results" / "dai18_probe" / "tether.json"))["tether"][CD]      # CREMA-D を触らずに残した単位で選んだ引き戻しの強さ
teth = {f"{k}|{K}": tuple(t18[f"{k}|{K}"]) for K in (5, 10, 25) for k in ("A", "V", "AV")}
for f in (0, 3):
    u = P18.Unit6(HOLD, CD, f, 0)
    for kind in u.kinds:
        t0 = time.time()
        p0 = S.logits(u.Xt[kind], *u.start[kind]).argmax(1).cpu().numpy()
        rf = P18.fit_none(u, kind, FULL_K, teth, [np.arange(u.n)])
        pf = S.logits(u.Xt[kind], rf["W"][0], rf["b"][0]).argmax(1).cpu().numpy()
        rn = P18.fit_none(u, kind, 5, teth)
        ppn = S.logits(u.Xp[kind], rn["W"][0], rn["b"][0]).argmax(1).cpu().numpy()
        print(f"分割 {f} {kind}：候補 {u.n}，引き戻し {teth[f'{kind}|{FULL_K}']}．上限の勾配の最大 {float(rf['grad_max']):.1e}，反復 {int(rf['n_iter'])}，止まり {int(rf['stalled'])}．"
              f"判定の形：出発点 {p0.shape}，上限 {pf.shape}，K 本だけ（候補）{ppn.shape}．出発点と上限で判定が違うテストの発話 {100 * float((p0 != pf).mean()):.1f}%．{time.time() - t0:.0f} 秒", flush=True)
