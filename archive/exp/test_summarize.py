"""`summarize_map.py` の検算。答えが分かっている人工の結果で確かめる（実データは読まない）。

    ./.venv/bin/python exp/test_summarize.py      # 数秒

K ≤ 10 では合併がターゲットのみより 0.05 高く，K ≥ 25 では同じ，という結果を作る。
K* は 10，反復が 30 回そろわない組は provisional，IEMOCAP がターゲットの組は未決の印になるはず。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_map as S  # noqa: E402


def check(name, cond, detail):
    print(("OK   " if cond else "FAIL ") + f"{name}: {detail}")
    return bool(cond)


def fake(target: str, sources: list[str], n_rep: int, rng) -> pd.DataFrame:
    rows = []
    for r in range(n_rep):
        for s in sources:
            rows.append(dict(target=target, source=s, method="src_only", K="", rep=r,
                             kind="fake", uar=0.40 + rng.normal(0, 0.01), uar_oracle=0.41,
                             C_rule="cv"))
        for i, K in enumerate(S.K_ORDER):
            base = 0.30 + 0.03 * i + rng.normal(0, 0.02)          # 反復ごとの揺れ（対で消える）
            rows.append(dict(target=target, source="", method="tgt_only", K=K, rep=r,
                             kind="fake", uar=base, uar_oracle=base + 0.01,
                             C_rule="fixed" if K == "1" else "tie"))
            for s in sources:
                gain = 0.05 + rng.normal(0, 0.005) if i <= 2 else 0.0
                rows.append(dict(target=target, source=s, method="merged", K=K, rep=r,
                                 kind="fake", uar=base + gain, uar_oracle=base + gain + 0.01,
                                 C_rule="cv"))
    return pd.DataFrame(rows)


def main() -> int:
    rng = np.random.default_rng(0)
    ok = True
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fake("T1", ["S1", "S2"], 30, rng).to_csv(d / "T1.csv", index=False)
        fake("T2", ["S1"], 12, rng).to_csv(d / "T2.csv", index=False)
        fake("IEMOCAP-impro", ["S1"], 30, rng).to_csv(d / "IEMOCAP-impro.csv", index=False)
        out = S.summarize(d)
        ks = pd.read_csv(out / "kstar.csv", dtype=str).set_index(["target", "source"])
        pairs = pd.read_csv(out / "pairs.csv", dtype={"K": str})

        ok &= check("K* は 10", (ks["K_star"] == "10").all(), ks["K_star"].to_dict())
        ok &= check("反復の数と状態", ks.loc[("T1", "S1"), "status"] == "final"
                    and ks.loc[("T2", "S1"), "status"] == "provisional"
                    and ks.loc[("IEMOCAP-impro", "S1"), "status"] == "iemocap_interval_undecided",
                    ks[["n_rep", "status"]].to_dict("index"))
        p = pairs[(pairs.target == "T1") & (pairs.source == "S1")].set_index("K")
        ok &= check("差の平均（K=1 で約 0.05，K=50 で 0）",
                    abs(p.loc["1", "diff_mean"] - 0.05) < 0.005 and p.loc["50", "diff_mean"] == 0,
                    f"{p.loc['1', 'diff_mean']:.4f}，{p.loc['50', 'diff_mean']:.4f}")
        ok &= check("対の差の区間は，反復の揺れ（0.02）より十分狭い",
                    (p.loc["1", "diff_hi"] - p.loc["1", "diff_lo"])
                    < (p.loc["1", "tgt_hi"] - p.loc["1", "tgt_lo"]) / 2,
                    f"差の幅 {p.loc['1', 'diff_hi'] - p.loc['1', 'diff_lo']:.4f}，"
                    f"ターゲットのみの幅 {p.loc['1', 'tgt_hi'] - p.loc['1', 'tgt_lo']:.4f}")
        cr = pd.read_csv(out / "c_rule.csv", dtype={"K": str}).set_index(["method", "K"])
        ok &= check("C の決まり方の割合", cr.loc[("tgt_only", "1"), "fixed"] == 1.0
                    and cr.loc[("merged", "5"), "cv"] == 1.0, "tgt_only K=1 は fixed，merged は cv")
        figs = sorted(f.name for f in out.glob("fig_*.png"))
        ok &= check("図ができる", figs == ["fig_IEMOCAP-impro.png", "fig_T1.png", "fig_T2.png",
                                          "fig_gain.png"], f"{figs}")
    print("\nすべて通った" if ok else "\n落ちたものがある")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
