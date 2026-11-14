"""ML の査読者の検算（DAI-21）：CREMA-D を評価するコーパスにした単位の作り（学習しない．CPU）．

台本は CREMA-D の単位を P18.Unit6("RAVDESS", "CREMA-D", 分割, 種) で作る．ここでは次を確かめる．
 (1) ソースが，声・顔・声顔の融合のどれでも元の 4 コーパスである（RAVDESS と CREMA-D が入らない）．
 (2) テストの話者が学習側（候補）に現れない．分割ごとの話者の数，発話の数，感情ごとの件数．
 (3) K 本が種 0〜9 で引ける（感情ごとに K 本．K について入れ子）．種どうしで違う K 本になる．
 (4) 区間の作り方を決める実効の塊の数（20 以上なら塊の重みの引き直し）．
 (5) 決まった文が学習側とテストにまたがる数（sample_id の 2 つ目の欄が文の記号）．
使い方：CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <この台本>
"""
import sys

import numpy as np

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai18_probe as P18  # noqa: E402
import dai_data as DD  # noqa: E402

dai1.ENC["enc"] = DD.ENC1
CD, HOLD = "CREMA-D", "RAVDESS"
C4 = sorted(DD.CORPORA)
print("実効の塊の数（CREMA-D）:", round(P18.g_eff_of(CD), 2), "／ テストの発話の下限（dai1_stats.MIN_TEST）:", ST.MIN_TEST)
cp = P18.C6.get(CD)
print("発話", len(cp.df), "話者", cp.df["cluster"].nunique(), "感情ごとの件数", np.bincount(cp.y, minlength=4).tolist(),
      "特徴の次元", {k: int(v.shape[1]) for k, v in cp.X.items()})
sid = cp.df["sample_id"].to_numpy().astype(str)
print("sample_id の例:", sid[:3].tolist())
for f in range(DD.N_FOLDS):
    kp = {}
    for r in range(10):
        u = P18.Unit6(HOLD, CD, f, r)
        assert all(sorted(u.srcs[k]) == C4 for k in u.kinds), u.srcs
        assert u.kinds == ["A", "V", "AV"] and u.fus == "AV" and u.mods == "AV"
        cl = cp.df["cluster"].to_numpy().astype(str)
        assert not (set(cl[u.pool]) & set(cl[u.test])), "テストの話者が学習側にいる"
        assert len(u.pool) + len(u.test) == len(cp.df)
        for K in (5, 10, 25):
            yk = u.ypool[u.kpos[K]]
            assert len(yk) == 4 * K and (np.bincount(yk, minlength=4) == K).all()
        assert set(u.kpos[5]) <= set(u.kpos[10]) <= set(u.kpos[25])
        kp[r] = tuple(u.kpos[25])
    n_distinct = len(set(kp.values()))
    sent = lambda ix: {s.split("_")[1] for s in sid[ix]}  # noqa: E731
    print(f"分割 {f}：候補 {len(u.pool)}，テスト {len(u.test)}，テストの話者 {len(set(cl[u.test]))}，テストの感情ごとの件数 {np.bincount(u.ytest, minlength=4).tolist()}，"
          f"種 0〜9 の K 本（K=25）のうち違う組 {n_distinct}，文の数 候補 {len(sent(u.pool))}・テスト {len(sent(u.test))}・両方 {len(sent(u.pool) & sent(u.test))}")
print("ソース（声・顔・融合）:", {k: u.srcs[k] for k in u.kinds})
print("出発点の形:", {k: tuple(u.start[k][0].shape) for k in u.kinds})
