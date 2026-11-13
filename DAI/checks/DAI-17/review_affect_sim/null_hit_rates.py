"""感情認識の査読役の検算（DAI-17 第 1 版）：人工の数だけを使う（感情の割合だけは DAI/splits/*.csv から取る）．特徴も結果も読まない．
「引いたソースの発話のラベルが，テストの発話の本当の感情と無関係」なとき（発話ごとの感情の情報が 0 のとき）に，
設計 4 節の D1・D3（台本 run_unit / stage_summary の式）がどの値になるかと，
感情ごとに当たる割合を出して 4 感情で平均する数え方（UAR と同じ数え方）ならどうなるかを比べる．
"""
import numpy as np
import pandas as pd

ROOT = "/home/snics/MER-DA"
C = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
sp = {c: pd.read_csv(f"{ROOT}/DAI/splits/{c}.csv") for c in C}
cnt = {c: np.bincount(sp[c].label_4.astype(int), minlength=4).astype(float) for c in C}
RANKS = ((0, 5), (5, 20), (20, 100))
rng = np.random.default_rng(0)


def tilt(q, w):
    q = q * np.asarray(w, float)
    return q / q.sum()


def one(c, srcs, f, scen, info=0.0):
    """scen：近さの順位の区切りごとの，引いた発話のラベルの割合の傾け方（Neutral のオッズの倍率）．info：本当の感情と同じラベルを引く割合の上乗せ（0 なら情報なし）．"""
    y = sp[c].label_4[sp[c].fold == f].astype(int).to_numpy()
    pdb = sum(cnt[s] for s in srcs) / sum(cnt[s].sum() for s in srcs)
    n = len(y)
    Y = np.zeros((n, 100), int)
    for (a, b), w in zip(RANKS, scen):
        q = tilt(pdb, [w, 1, 1, 1])
        Y[:, a:b] = rng.choice(4, size=(n, b - a), p=q)
    if info > 0:
        same = rng.random((n, 100)) < info
        Y = np.where(same, y[:, None], Y)
    hit = Y == y[:, None]
    pt = np.bincount(y, minlength=4) / n
    micro = {f"{a + 1}-{b}": hit[:, a:b].mean() for a, b in RANKS}
    macro = {f"{a + 1}-{b}": np.mean([hit[y == e][:, a:b].mean() for e in range(4) if (y == e).any()]) for a, b in RANKS}
    d1_mi = 100 * (micro["1-5"] - micro["21-100"])
    d3_mi = 100 * (hit.mean() - pt @ pdb)                                  # 台本の D3（表のラベルの割合との内積を引く）
    d1_ma = 100 * (macro["1-5"] - macro["21-100"])
    d3_ma = 100 * (np.mean([hit[y == e].mean() for e in range(4) if (y == e).any()]) - 0.25)
    return d1_mi, d3_mi, d1_ma, d3_ma


SCEN = {"A 情報なし・割合は表どおり": ((1, 1, 1), 0.0),
        "B 情報なし・Neutral が 2 倍引かれやすい": ((2, 2, 2), 0.0),
        "C 情報なし・上位 5 本だけ Neutral が 2 倍": ((2, 1, 1), 0.0),
        "D 情報あり（3% の対で同じ感情を上乗せ）・割合は表どおり": ((1, 1, 1), 0.03)}
rows = []
for name, (scen, info) in SCEN.items():
    for c in C:
        vals = []
        for srcs in [tuple(x for x in C if x not in (c, T)) for T in C if T != c]:
            for f in range(5):
                vals.append(one(c, srcs, f, scen, info))
        v = np.mean(vals, 0)
        rows.append(dict(scenario=name, c=c, D1_script=v[0], D3_script=v[1], D1_macro=v[2], D3_macro=v[3]))
d = pd.DataFrame(rows).round(2)
print(d.to_string(index=False))
cntd = d.assign(s3=d.D3_script >= 2, m3=d.D3_macro >= 2, s1=d.D1_script >= 2, m1=d.D1_macro >= 2).groupby("scenario")[["s1", "s3", "m1", "m3"]].sum()
cntd.columns = ["D1_script：2 点以上のコーパス", "D3_script：同", "D1_macro：同", "D3_macro：同"]
print(cntd.to_string())
d.to_csv(f"{ROOT}/DAI/checks/DAI-17/review_affect_sim/null_hit_rates.csv", index=False)
