"""集計の段の記述の規則（D1〜D3）の行を，台本から文字列のまま取り出して人工の行に当てる（MER-DA のデータ・結果は読まない）．
確かめること：(1) D1・D2・D3 が設計 4 節の定義どおりの値になるか．(2) 種 2 つの行が同じ値の繰り返しでも数えが変わらないか．
(3) 単位の csv の行（true_emotion・neighbor_label）だけから，「本当の感情ごとの当たる割合の平均 − 25」と「引かれたラベルの割合を基準にした差」を出せるか．"""
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

src = Path(sys.argv[1]).read_text().splitlines()
a = next(i for i, l in enumerate(src) if l.strip().startswith('nn = csvs[csvs.version == "_nn"]'))
b = next(i for i, l in enumerate(src) if "neighbors_count.csv" in l)
block = textwrap.dedent("\n".join(src[a:b + 1]))
print(f"取り出した行：{a + 1}〜{b + 1}")

rng = np.random.default_rng(0)
C = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
PT = {"AFEW": [0.26, 0.27, 0.22, 0.25], "IEMOCAP": [0.37, 0.32, 0.21, 0.10], "MOSEI": [0.46, 0.36, 0.10, 0.08], "MELD": [0.57, 0.20, 0.09, 0.14]}
TRUE = {"D1": {"AFEW": 3.0, "IEMOCAP": 1.0, "MOSEI": 4.0, "MELD": 2.5}, "D2": {"AFEW": 5.0, "IEMOCAP": 0.5, "MOSEI": 3.0, "MELD": 2.2}}
rows = []
for c in C:
    for T in [x for x in C if x != c]:
        for f in range(5):
            for sp in ("feat", "logit"):
                for m in "AVT":
                    pt = np.array(PT[c])
                    q = np.array([0.55, 0.25, 0.10, 0.10])                    # 引かれた発話のラベルの割合（Neutral に寄せた）
                    pdb = np.array([0.45, 0.30, 0.13, 0.12])                  # 表のラベルの割合
                    lift = 0.02 + 0.005 * rng.normal()                        # 本当の感情と近い発話のラベルの間の，独立からのずれ（どの感情でも同じ）
                    he = q + lift                                             # 本当の感情 e の発話の当たる割合
                    hit100 = float(pt @ he)
                    d1 = TRUE["D1"][c] / 100
                    h_far = hit100 - 0.05 * d1; h_near = h_far + d1; h_mid = (100 * hit100 - 5 * h_near - 80 * h_far) / 15
                    base = dict(K=-1, version="_nn", space=sp, modality=m, T=T, c=c, fold=f)
                    unit = [dict(base, by="rank", level=lv, hit=h, share=1.0) for lv, h in (("1-5", h_near), ("6-20", h_mid), ("21-100", h_far))]
                    unit.append(dict(base, by="chance", level="all", hit=float(pt @ pdb), share=1.0))
                    for s_ in [x for x in C if x not in (T, c)]:
                        unit.append(dict(base, by="source", level=s_, hit=hit100, share=0.5, db_share=0.5))
                    for e in range(4):
                        unit.append(dict(base, by="true_emotion", level=str(e), hit=float(he[e]), share=float(pt[e])))
                        unit.append(dict(base, by="neighbor_label", level=str(e), hit=0.3, share=float(q[e])))
                        for rl in ("home", "away"):
                            d2 = TRUE["D2"][c] / 100
                            for lv, h in (("low", 0.30), ("mid", 0.31), ("high", 0.30 + d2)):
                                unit.append(dict(base, by=f"rel_{rl}_label{e}", level=lv, hit=h, share=0.08))
                    for seed in (0, 1):                                       # 種に依らない量なので，同じ行を 2 回入れる（台本と同じ）
                        rows += [dict(r, seed=seed) for r in unit]
csvs = pd.DataFrame(rows)
sdir = Path(tempfile.mkdtemp())
D_MIN_PT = 2.0
exec(block)
dr = pd.read_csv(sdir / "neighbors_rules.csv")
dc = pd.read_csv(sdir / "neighbors_count.csv")
one = dr[(dr.space == "feat") & (dr.modality == "A")]
print(one[["c", "quantity", "mean", "std", "count", "min_T", "max_T", "ge2"]].round(3).to_string(index=False))
print(dc[(dc.space == "feat") & (dc.modality == "A")].to_string(index=False))
for c in C:
    g = one[one.c == c].set_index("quantity")["mean"]
    assert abs(g["D1_rank"] - TRUE["D1"][c]) < 1e-6 and abs(g["D2_rel_home"] - TRUE["D2"][c]) < 1e-6 and abs(g["D2_rel_away"] - TRUE["D2"][c]) < 1e-6
print("D1・D2 は置いた値と一致した．count は 30（種 2 つの繰り返しを含む．別々の値は 15）")

# 単位の csv の行だけから出せる 2 つの量
nn = csvs[csvs.version == "_nn"]
key = ["T", "c", "fold", "seed", "space", "modality"]
te = nn[nn.by == "true_emotion"]
bal = 100 * te.groupby(key).hit.mean() - 25                                    # 本当の感情ごとの当たる割合の平均 − 25
pt_ = te.set_index(key + ["level"]).share
q_ = nn[nn.by == "neighbor_label"].set_index(key + ["level"]).share
indep = 100 * (pt_ * q_).groupby(level=key).sum()                              # 引かれた発話のラベルの割合を基準にした，でたらめの当たる割合
pv = nn[nn.by == "rank"].pivot_table(index=key, columns="level", values="hit")
hit100 = 100 * (5 * pv["1-5"] + 15 * pv["6-20"] + 80 * pv["21-100"]) / 100
ch = 100 * nn[nn.by == "chance"].set_index(key).hit
out = pd.DataFrame({"D3（台本）": hit100 - ch, "D3（引かれたラベルの割合を基準）": hit100 - indep, "感情ごとの当たる割合の平均 − 25": bal}).reset_index()
print(out.groupby("c")[["D3（台本）", "D3（引かれたラベルの割合を基準）", "感情ごとの当たる割合の平均 − 25"]].mean().round(2).to_string())
print("置いた本当のずれはどのコーパスも 2.0 点．台本の D3 は，引かれた発話が Neutral に寄った分（表 0.45 → 引かれた 0.55）を含む．")

# 台本の D3 の単位ごとの値と，上で出した値を突き合わせる
qs = q[q.quantity == "D3_vs_chance"].set_index(key).value
mine = (hit100 - ch)
both = pd.concat([qs.rename("script"), mine.rename("mine")], axis=1)
print("単位ごとの D3：台本の行と上の式の差の最大", float((both.script - both.mine).abs().max()), "行の数", len(both), "NaN", int(both.isna().any(axis=1).sum()))
print("c ごとの平均（台本の行）", both.script.groupby(level="c").mean().round(3).to_dict())
print("c ごとの平均（neighbors_rules.csv の全空間・全モダリティ）", dr[dr.quantity == "D3_vs_chance"].groupby("c")["mean"].mean().round(3).to_dict())
