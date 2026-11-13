"""D3（近い 100 本の当たる割合 − でたらめに引いたときの当たる割合）の基準の確かめ．
使うのは DAI/splits/<コーパス>.csv の感情ラベルの割合だけである（特徴・分類器の出力・results/ は読まない）．
問い：近いソースの発話が感情と無関係でも，引かれる発話が片方のソースに寄るだけで，D3 は何点になりうるか．
  chance_all  = (テストの感情の割合) ・ (ソース 2 つを合わせた表のラベルの割合)   … 台本の D3 の基準
  chance_s    = (テストの感情の割合) ・ (ソース s のラベルの割合)               … 引かれる発話が全部 s から来て，s の中ではでたらめのときの当たる割合
"""
import itertools
import numpy as np
import pandas as pd

ROOT = "/home/snics/MER-DA/DAI/splits"
C = ("AFEW", "IEMOCAP", "MOSEI", "MELD")
df = {c: pd.read_csv(f"{ROOT}/{c}.csv", keep_default_na=False) for c in C}
p = {c: np.bincount(df[c].label_4.astype(int), minlength=4) / len(df[c]) for c in C}
n = {c: len(df[c]) for c in C}
print("件数と感情の割合（0=Neutral 1=Happy 2=Sad 3=Angry）")
for c in C:
    print(f"  {c:8s} n={n[c]:6d}  " + " ".join(f"{x:.3f}" for x in p[c]))
print()
print("仮のターゲット c，ソースの組 (s1, s2)：基準 chance_all と，片方のソースだけから引いたときの当たる割合（点），その差")
rows = []
for c in C:
    for s1, s2 in itertools.combinations([x for x in C if x != c], 2):
        pdb = (n[s1] * p[s1] + n[s2] * p[s2]) / (n[s1] + n[s2])
        ca, c1, c2 = 100 * p[c] @ pdb, 100 * p[c] @ p[s1], 100 * p[c] @ p[s2]
        rows.append((c, s1, s2, ca, c1, c2))
        print(f"  c={c:8s} ({s1:8s},{s2:8s})  chance_all={ca:5.1f}  全部 {s1} から={c1:5.1f}（{c1 - ca:+5.1f}）  全部 {s2} から={c2:5.1f}（{c2 - ca:+5.1f}）  表に占める {s1} の割合={n[s1] / (n[s1] + n[s2]):.2f}")
print()
print("c ごとに，3 つのソースの組で「片方のソースに全部寄ったとき」の差の最大を平均したもの（感情と無関係でも D3 がとりうる上の値）")
for c in C:
    d = [max(c1 - ca, c2 - ca) for cc, s1, s2, ca, c1, c2 in rows if cc == c]
    print(f"  c={c:8s}  組ごとの最大 {' '.join(f'{x:+5.1f}' for x in d)}  平均 {np.mean(d):+5.1f}")
print()
print("同じ仮定で，本当の感情ごとの当たる割合の平均（クラスで釣り合わせた当たる割合）は，引かれた発話のラベルの割合 q に依らず 25.0 点になる：")
for c in C:
    for s in [x for x in C if x != c][:1]:
        q = p[s]
        print(f"  c={c:8s} 全部 {s} から：本当の感情 e ごとの当たる割合 = q(e) → 平均 {100 * q.mean():.1f}")
