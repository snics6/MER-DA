"""gate-code 第 2.1 回の検算．値は出さない．"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/snics/MER-DA/DAI")
O, P = Path(sys.argv[1]), Path(sys.argv[2])   # O：第 2.1 版の出力，P：前回の写しの出力
pe = pd.read_csv(O / "select_strongest_per_emotion.csv")
pe0 = pd.read_csv(ROOT / "results/dai20_eval/summary/per_emotion.csv")
m = pe[pe.version.isin(["prop", "prop@raw", "prod"])].merge(pe0, on=["c", "K", "version", "emotion"], suffixes=("", "_0"))
print("集計との照合の行数", len(m), "n・n_fr・n_units の違い", int((m.n != m.n_0).sum()), int((m.n_fr != m.n_fr_0).sum()), int((m.n_units != m.n_units_0).sum()))
g = pe.groupby(["c", "K", "emotion"]).agg(nmin=("n", "min"), nmax=("n", "max"), fr=("n_fr", "nunique"), u=("n_units", "unique"))
print("同じ (コーパス, K, 感情) で版によって n が違う組", int((g.nmin != g.nmax).sum()), "n_units の値", sorted({int(x) for a in g.u for x in a}), "n_fr の値", sorted(pe.n_fr.unique()))
for n in ("select_strongest_effects.csv", "select_strongest_levels.csv", "select_strongest_picks.csv"):
    a, b = pd.read_csv(O / n), pd.read_csv(P / n)
    a = a.drop(columns=[c for c in ("commit", "dirty") if c in a]); b = b.drop(columns=[c for c in ("commit", "dirty") if c in b])
    print(n, "前回の写しの出力と同じ", a.shape == b.shape and a.equals(b))
a = pd.read_csv(O / "select_strongest_per_emotion.csv"); b = pd.read_csv(P / "select_strongest_per_emotion.csv")
print("感情ごとの共通の列が同じ", a[b.columns.drop(["commit", "dirty"])].equals(b[b.columns.drop(["commit", "dirty"])]))
