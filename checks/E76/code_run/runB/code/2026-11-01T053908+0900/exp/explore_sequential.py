"""【探索的】E34 の追加の解析：本数をそろえながら分けて選ぶ herding（seq_herding）と，E34 のラベル無しの選び方・ラベルありの参考の比べ（予言 Q1〜Q3）。

規則は `docs/exploratory-analyses.md` E34「E34 の追加」。入力は `results/select_seq_<kind>/`（seq_herding，seq_random），
`results/select_lf_<kind>/`（herding_pred，random_pred，herding_all，random_true），`results/select_herdfeat_<kind>/`（herding_feat）。
単位はターゲット（反復 0〜9 の平均）。

    ./.venv/bin/python exp/explore_sequential.py

出力：`results/explore_relation/e34s_targets.csv`（種類 × ターゲット × K × 選び方），`e34s_tests.csv`，`e34s_balance.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import explore_compare as EC  # noqa: E402

MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
KS = ["1", "5", "10", "25"]
ORDER = ["random_pred", "herding_pred", "herding_all", "seq_random", "seq_herding", "random_true", "herding_feat"]
OUT = ROOT / "results" / "explore_relation"


def load_kind(kind: str):
    seq = ROOT / "results" / f"select_seq_{kind}"
    rows, bal = [], []
    for f in sorted(p for p in seq.glob("*.csv") if not p.stem.endswith("_sel")) if seq.exists() else []:
        t = f.stem
        parts = [pd.read_csv(f, dtype={"K": str})]
        lf = ROOT / "results" / f"select_lf_{kind}" / f"{t}.csv"
        hf = ROOT / "results" / f"select_herdfeat_{kind}" / f"{t}.csv"
        if lf.exists():
            parts.append(pd.read_csv(lf, dtype={"K": str}))
        if hf.exists():
            h = pd.read_csv(hf, dtype={"K": str})
            parts.append(h[h.source == "herding_feat"])
        a = pd.concat(parts)
        a = a[a.rep.isin(EC.REPS)]
        U = a.groupby(["source", "K"]).uar.mean() * 100
        for (st, K), u in U.items():
            rows.append({"kind": kind, "target": t, "strategy": st, "K": K, "uar": u})
        for sf in (seq / f"{t}_sel.csv", ROOT / "results" / f"select_lf_{kind}" / f"{t}_sel.csv"):
            if sf.exists():
                s = pd.read_csv(sf, dtype={"K": str})
                s = s[s.rep.isin(EC.REPS)]
                s["gap"] = s.max_count - s.min_count
                bal.append(s.groupby(["strategy", "K"])[["missing", "min_count", "max_count", "gap"]].mean().reset_index()
                           .assign(kind=kind, target=t))
    return pd.DataFrame(rows), (pd.concat(bal, ignore_index=True) if bal else pd.DataFrame())


def tests(T: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    W = T.pivot_table(index=["kind", "target", "K"], columns="strategy", values="uar")
    G = B.pivot_table(index=["kind", "target", "K"], columns="strategy", values="gap")
    out = []
    for kind in W.index.get_level_values("kind").unique():
        w, g = W.loc[kind], G.loc[kind]
        row = {"kind": kind, "targets": w.index.get_level_values("target").nunique()}
        for K in KS:
            x = w.xs(K, level="K")
            for a, b in (("seq_herding", "herding_pred"), ("seq_herding", "random_true"), ("seq_herding", "herding_feat"),
                         ("seq_herding", "seq_random")):
                if a in x and b in x:
                    row[f"{a}-{b}_K{K}"] = float((x[a] - x[b]).mean())
                    row[f"wins_{a}>{b}_K{K}"] = int((x[a] > x[b]).sum())
            y = g.xs(K, level="K")
            for st in ("herding_pred", "seq_herding"):
                if st in y:
                    row[f"gap_{st}_K{K}"] = float(y[st].mean())
        row["Q1"] = all(row.get(f"gap_seq_herding_K{K}", 1e9) < row.get(f"gap_herding_pred_K{K}", -1e9) for K in ("10", "25"))
        row["Q2"] = all(row.get(f"seq_herding-herding_pred_K{K}", -1) > 0 for K in ("5", "10", "25"))
        row["Q3"] = all(row.get(f"seq_herding-random_true_K{K}", -1) > 0 for K in ("5", "10"))
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    parts = [load_kind(k) for k in MAIN]
    T = pd.concat([p[0] for p in parts], ignore_index=True)
    B = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    R = tests(T, B)
    T.to_csv(OUT / "e34s_targets.csv", index=False)
    R.to_csv(OUT / "e34s_tests.csv", index=False)
    B.to_csv(OUT / "e34s_balance.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(2).T.to_string())
    M = T.pivot_table(index=["kind", "K"], columns="strategy", values="uar", aggfunc="mean")
    print(M[[c for c in ORDER if c in M]].round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
