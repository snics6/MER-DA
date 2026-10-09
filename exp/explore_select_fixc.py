"""【探索的】E24：式でソースを選ぶと，既存の選び方に勝つか（C を固定した地図）。

予言は `docs/exploratory-analyses.md` E22〜E24 の 1 節（計算の前に書いた）。新しい学習はしない。
ターゲットごと・K ごと・反復ごとに，候補のソースを基準で並べて 1 位を選び，
損 = その反復で最良のソースの合併の UAR − 選んだソースの合併の UAR を求めて平均する。

基準：(a) ソースの大きさ，(b) 平均的な良さ（ほかのターゲットでの得の平均．選ぶターゲットは除く），
(c) K 本で試す（ソースのみの分類器の，引いた K 本での UAR），(d) 式の λ（E15），
(e) 式の λ × ラベル無しで見積もった答えの偏り TV_unlab，参考 (f) 式の位置の直り（E15．E13 のラベルを使うので実用ではない）

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_select_fixc.py

出力：`OUT/e24_regret.csv`（種類 × K × 基準），`OUT/e24_rows.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import shift  # noqa: E402
from data import groups_for, load_any  # noqa: E402
from explore_direction import map_draw  # noqa: E402
from explore_synergy import KS, load_map  # noqa: E402
from pair_table import OUT  # noqa: E402

KINDS = {"audio_wavlm": "W7", "audio_hubert": "Hu7", "video": "V", "fuse-audio-video": "AV"}
CRITERIA = ["(a) 大きさ", "(b) 平均的な良さ", "(c) K 本で試す", "(d) 式の λ", "(e) 式の λ × 偏り", "(f) 式の位置の直り（参考）"]


def main() -> int:
    lam = pd.read_csv(OUT / "e15_lambda.csv", dtype={"K": str})
    pairs = pd.read_csv(OUT / "e15_pairs.csv", dtype={"K": str})
    rows = []
    for kind in KINDS:
        mp = load_map(kind)
        groups = groups_for(kind)
        dfs = {g: load_any(g, kind)[0] for g in groups}
        mean_gain = mp.assign(g=mp.m - mp.t).groupby(["target", "source", "K"]).g.mean()
        logits = {s: np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz") for s in groups}
        for t in groups:
            y = dfs[t].label_4.to_numpy()
            for s in groups:
                if s == t:
                    continue
                L = logits[s][f"t_{t}"]
                pred = L.argmax(1)
                share = np.bincount(pred, minlength=4) / len(pred)
                tv = 0.5 * float(np.abs(share - 0.25).sum())
                for K in KS:
                    lr = lam[(lam.kind == kind) & (lam.target == t) & (lam.source == s) & (lam.K == K)].iloc[0]
                    pr = pairs[(pairs.kind == kind) & (pairs.target == t) & (pairs.source == s) & (pairs.K == K)].iloc[0]
                    others = [o for o in groups if o not in (t, s)]
                    avg = float(np.mean([mean_gain[(o, s, K)] for o in others if (o, s, K) in mean_gain.index]))
                    g = mp[(mp.target == t) & (mp.source == s) & (mp.K == K)]
                    for r in g.itertuples():
                        _, idx = map_draw(dfs[t], t, int(r.rep), int(K))
                        rows.append({"kind": kind, "target": t, "source": s, "K": K, "rep": r.rep, "m": r.m,
                                     "(a) 大きさ": float(r.n_src), "(b) 平均的な良さ": avg,
                                     "(c) K 本で試す": shift.uar(y[idx], pred[idx]),
                                     "(d) 式の λ": lr.lam_pred, "(e) 式の λ × 偏り": lr.lam_pred * tv,
                                     "(f) 式の位置の直り（参考）": pr.pos_fix_pred, "tv_unlab": tv})
        print(f"  {kind} 済み", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "e24_rows.csv", index=False)
    out = []
    for (kind, K), g in R.groupby(["kind", "K"], sort=False):
        res = {c: [] for c in CRITERIA + ["でたらめ"]}
        for (t, r), h in g.groupby(["target", "rep"]):
            best = h.m.max()
            for c in CRITERIA:
                res[c].append(best - h.loc[h[c].idxmax(), "m"])
            res["でたらめ"].append(best - h.m.mean())
        for c, v in res.items():
            out.append({"setting": KINDS[kind], "K": K, "criterion": c, "regret_points": 100 * float(np.mean(v)),
                        "n": len(v)})
    O = pd.DataFrame(out)
    O.to_csv(OUT / "e24_regret.csv", index=False)
    pd.set_option("display.width", 250)
    print(O.pivot_table(index=["setting", "K"], columns="criterion", values="regret_points").round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
