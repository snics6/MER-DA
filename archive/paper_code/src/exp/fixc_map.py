"""合併の C を同じ反復のソースのみの C に固定した地図を，保存済みの列から作る（新しい学習なし）。

決定は `docs/researchvision.md` 6.2 節（2026-10-09）．理由は `docs/exploratory-analyses.md` E16 の 9 節：
小さい K の合併は交差検証が同点になりやすく，同点の規則で C=1 が選ばれて罰則が弱くなり，向きが悪くなる。
これは C を選ぶ手続きの揺れで，仕組みの解析の対象と混ざる。

作り方：地図（`results/map_<kind>/<ターゲット>.csv`）の合併の行の `uar` を，同じ（ソース，反復）の
ソースのみの行で選ばれた C の列 `test_uar_C<C>` に置き換え，`C` をその値に，`C_rule` を `src` にする。
候補の C ごとのテストの UAR は，地図が学習した分類器の値そのもの（C をテストで選ぶのではない）。
ソースのみとターゲットのみの行は変えない。ほかの列（交差検証の成績，オラクル）は元のまま残す。

    ./.venv/bin/python exp/fixc_map.py            # 全種類 → results/fixc_map/<kind>/<ターゲット>.csv
    ./.venv/bin/python exp/summarize_map.py --dir results/fixc_map/<kind>    # pairs.csv・kstar.csv

比べる表（地図との差）：`results/fixc_map/compare.csv`
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "fixc_map"
KINDS = ("audio_wavlm", "audio_hubert", "video", "fuse-audio-video", "text_roberta", "fuse-audio-text",
         "fuse-video-text", "fuse-audio-video-text")
KS = ["1", "5", "10", "25", "50", "all"]


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def convert(path: Path, dest: Path, prov: dict) -> pd.DataFrame:
    header = [l for l in path.read_text().splitlines() if l.startswith("#")]
    d = pd.read_csv(path, comment="#", dtype={"K": str, "source": str})
    s = d[d.method == "src_only"].set_index(["source", "rep"]).C
    m = d.method == "merged"
    Cs = [s[(src, r)] for src, r in zip(d.loc[m, "source"], d.loc[m, "rep"])]
    d.loc[m, "uar"] = [d.loc[i, f"test_uar_C{c:g}"] for i, c in zip(d.index[m], Cs)]
    d.loc[m, "C"] = Cs
    d.loc[m, "C_rule"] = "src"
    with open(dest, "w") as f:
        f.write("# " + json.dumps({**prov, "derived_from": str(path.relative_to(ROOT))}, ensure_ascii=False) + "\n")
        for h in header:
            f.write(h + "\n")
        d.to_csv(f, index=False)
    return d


def per_rep(d: pd.DataFrame) -> pd.DataFrame:
    t = d[d.method == "tgt_only"][["target", "K", "rep", "uar"]].rename(columns={"uar": "t"})
    s = d[d.method == "src_only"][["target", "source", "rep", "uar"]].rename(columns={"uar": "s"})
    m = d[d.method == "merged"][["target", "source", "K", "rep", "uar"]].rename(columns={"uar": "m"})
    return m.merge(t, on=["target", "K", "rep"]).merge(s, on=["target", "source", "rep"])


def lo(x: np.ndarray) -> float:
    return float(x.mean() - stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def hi(x: np.ndarray) -> float:
    return float(x.mean() + stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def describe(x: pd.DataFrame, kind: str, version: str) -> list[dict]:
    rows = []
    for K, g in x.groupby("K"):
        pairs = []
        for (t, s), h in g.groupby(["target", "source"]):
            mt, ms = (h.m - h.t).to_numpy(), (h.m - h.s).to_numpy()
            pairs.append({"gain": mt.mean(), "m_s": ms.mean(), "up": lo(mt) > 0, "down": hi(mt) < 0,
                          "syn": lo(mt) > 0 and lo(ms) > 0, "t": h.t.mean(), "s": h.s.mean()})
        p = pd.DataFrame(pairs)
        row = {"kind": kind, "version": version, "K": K, "n_pairs": len(p), "n_rep": int(g.rep.nunique()),
               "gain_mean": p.gain.mean(), "m_minus_s_mean": p.m_s.mean(),
               "n_up": int(p.up.sum()), "n_down": int(p.down.sum()), "n_synergy": int(p.syn.sum())}
        if K == "1":
            b = np.polyfit(p.s - p.t, p.gain, 1)
            row["mh1_slope"] = float(b[0])
        rows.append(row)
    return rows


def main() -> int:
    prov = {"script": "exp/fixc_map.py", "commit": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "rule": "merged の C を同じ（source, rep）の src_only の C に固定し，uar を test_uar_C<C> に置き換えた"}
    comp = []
    for kind in KINDS:
        src_dir = ROOT / "results" / f"map_{kind}"
        if not src_dir.exists():
            continue
        dest_dir = OUT / kind
        dest_dir.mkdir(parents=True, exist_ok=True)
        orig, fixed = [], []
        for f in sorted(src_dir.glob("*.csv")):
            orig.append(pd.read_csv(f, comment="#", dtype={"K": str, "source": str}))
            fixed.append(convert(f, dest_dir / f.name, prov))
        o, x = pd.concat(orig), pd.concat(fixed)
        comp += describe(per_rep(o), kind, "cv") + describe(per_rep(x), kind, "fixc")
        print(f"{kind}: {len(fixed)} ターゲット", flush=True)
    C = pd.DataFrame(comp)
    C["K"] = pd.Categorical(C.K, KS, ordered=True)
    C = C.sort_values(["kind", "K", "version"])
    C.to_csv(OUT / "compare.csv", index=False)
    pd.set_option("display.width", 250)
    print(C.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
