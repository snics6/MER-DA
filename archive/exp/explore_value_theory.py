"""E67 の下調べ：2 次の近似の 1 次元の模型で，疑似ラベルのラベル換算の価値の K による変わり方を予言し，E65・E66 の値と照らす．

模型（ターゲットのラベル 1 本の情報を 1 とする．単位は 1 感情あたりのラベルの本数）
- 中心の誤差が b，保つ強さ（アンカーへの正則化の精度）が p の生徒モデルを K 本で学び直すと，誤差の 2 乗の期待値は
  R = (p² b² + K) / (p + K)²．ターゲットだけで N 本学ぶと 1/N なので，ラベル換算の総価値は N(p, b; K) = 1/R − K
- 今の手法（A．疑似ラベルと K 本を同時に学ぶ）は，中心 c・保つ強さ p0 + m/4 と同じ（E66 の数理）
- アンカーを動かしてから学び直す形（B．E66，h = h_b = 1）は，中心 c・保つ強さ p0
- 疑似ラベルが足す価値は N(中心 c) − N(アンカー W0)．p0 は W0 の最適な強さ s0 = 1/b0² と置く（仮定）
- s0 は E65 の K=0 の出発点の本数，c の価値 s_c は s0 ＋ E65 の K=0 の疑似ラベルの本数から取る

入力：results/explore_relation/e65_summary.csv（A），e66_value.csv（B）．出力：e67_check.csv．結果を見た後の照合で，予言ではない．
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results" / "explore_relation"


def total_value(p: float, s: float, K: float) -> float:
    """誤差 b = 1/√s の中心を強さ p で保って K 本学んだときのラベル換算の総価値 N(p, b; K)．"""
    return (p + K) ** 2 / (p ** 2 / s + K) - K


def main() -> None:
    a = pd.read_csv(RES / "e65_summary.csv", comment="#")
    a = a[(a.single == True) & (a.belief == "pF")]  # noqa: E712
    b = pd.read_csv(RES / "e66_value.csv", comment="#")
    b = b[(b.belief == "pF") & (b.hold == 1) & (b.hold_b == 1)]
    obs_b = b.groupby(["student", "meff", "K"]).n_mod.median()
    rows = []
    for (student, m), g in a.groupby(["student", "meff"]):
        g = g.set_index("K")
        s0 = g.loc[0, "n_start_med"]
        sc = s0 + g.loc[0, "n_mod_med"]
        for K in sorted(g.index):
            rows.append(dict(
                student=student, meff=m, K=K,
                s0=s0, sc=sc,
                obs_A=g.loc[K, "n_mod_med"],
                pred_A=total_value(s0 + m / 4.0, sc, K) - total_value(s0, s0, K),
                obs_B=obs_b.get((student, m, K), float("nan")),
                pred_B=total_value(s0, sc, K) - total_value(s0, s0, K),
                start_obs=g.loc[K, "n_start_med"],
            ))
    out = pd.DataFrame(rows)
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dst = RES / "e67_check.csv"
    with open(dst, "w") as f:
        f.write("# " + json.dumps({"script": "exp/explore_value_theory.py", "commit": commit,
                                   "inputs": ["e65_summary.csv", "e66_value.csv"]}) + "\n")
        out.to_csv(f, index=False)
    print(out.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
