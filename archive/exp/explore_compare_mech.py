"""【探索的】E31 の設計 4 項：各手法の「ソースのみからの上がり幅」を位置の分と向きの分に分け，UAR を d・o から共通の雑音の橋で予言する。新しい学習なし。

入力：`results/compare_<kind>/<ターゲット>[_<tag>].npz`（手法のテストの点数），ソースのみの点数は
`results/merged_scores_<kind>/`（言葉系は `results/fixc_text/merged_scores_<kind>/`）の `s_<ソース>_r<r>`，
全ソースは `results/multisource_scores_<kind>/` の `s_all_r<r>`。分け方は E25-4・E30 と同じ（感情の対ごとの 2 つの山の d と o．
位置の分 = A(d_s, o_m) − A(d_s, o_s)，向きの分 = A(d_m, o_m) − A(d_s, o_m)）。

    ./.venv/bin/python exp/explore_compare_mech.py [--kinds …]

出力：`results/explore_relation/e31_mech_pairs.csv`（組 × K × 手法．反復の平均），`e31_mech_summary.csv`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC  # noqa: E402
from explore_pairs import ba  # noqa: E402
from explore_synergy import d_o  # noqa: E402
from explore_uar_link import bridges  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video",
         "text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text"]
TEXT = set(KINDS[4:])
METHODS = ("p1", "p2", "p3", "p4", "h3", "p2t", "p3t", "h3t", "o2")
OUT = ROOT / "results" / "explore_relation"


def src_scores(kind: str, t: str):
    base = ROOT / "results" / ("fixc_text" if kind in TEXT else "") / f"merged_scores_{kind}" / f"{t}.npz"
    ms = ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz"
    return np.load(base), (np.load(ms) if ms.exists() else None)


def split_one(Ls, Lm, y):
    pos, dirc = [], []
    for (ds, os_), (dm, om) in zip(d_o(Ls, y), d_o(Lm, y)):
        pos.append(ba(ds, om) - ba(ds, os_))
        dirc.append(ba(dm, om) - ba(ds, om))
    return 100 * np.mean(pos), 100 * np.mean(dirc)


def run_kind(kind: str) -> list[dict]:
    rows = []
    for f in sorted((ROOT / "results" / f"compare_{kind}").glob("*.npz")):
        t = f.stem.split("_")[0] if f.stem.split("_")[-1] in ("k50all", "trans", "o2") else f.stem
        if f.stem.endswith(("final", "final_k50all")):
            continue
        z = np.load(f)
        zs, zm = src_scores(kind, t)
        for key in z.files:
            m = key.split("_")[0]
            if m not in METHODS:
                continue
            src, rest = key[len(m) + 1:].rsplit("_K", 1)
            K, r = rest.split("_r")
            r = int(r)
            if f"test_idx_r{r}" not in zs.files or not np.array_equal(zs[f"test_idx_r{r}"], z[f"test_idx_r{r}"]):
                raise ValueError(f"テストの行が合わない: {kind} {t} r{r}")
            y = z[f"y_test_r{r}"]
            if src == "ALL":
                if zm is None:
                    continue
                Ls = zm[f"s_all_r{r}"].astype(np.float64)
            else:
                Ls = zs[f"s_{src}_r{r}"].astype(np.float64)
            Lm = z[key].astype(np.float64)
            pos, dirc = split_one(Ls, Lm, y)
            b = bridges(Lm, y)
            rows.append({"kind": kind, "target": t, "source": src, "method": m, "K": K, "rep": r,
                         "uar": 100 * b["uar"], "uar_src": 100 * bridges(Ls, y)["uar"], "pos": pos, "dir": dirc,
                         "bridge": 100 * b["shared_gauss"]})
        print(f"  {kind} {f.stem} 済み", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    args = ap.parse_args()
    kinds = [k for k in args.kinds if (ROOT / "results" / f"compare_{k}").exists()]
    R = pd.DataFrame([x for k in kinds for x in run_kind(k)])
    P = R.groupby(["kind", "target", "source", "method", "K"])[["uar", "uar_src", "pos", "dir", "bridge"]].mean().reset_index()
    P["gain"] = P.uar - P.uar_src
    P["share_pos"] = P.pos / (P.pos + P.dir)
    P["bridge_err"] = P.bridge - P.uar
    P["same"] = [s != "ALL" and GROUP_SPEC[a][0] == GROUP_SPEC[s][0] for a, s in zip(P.target, P.source)]
    P.to_csv(OUT / "e31_mech_pairs.csv", index=False)
    S = P[~P.same].groupby(["kind", "method", "K"]).agg(
        n=("gain", "size"), gain=("gain", "mean"), pos=("pos", "mean"), dir=("dir", "mean"),
        share_pos_median=("share_pos", "median"), bridge_err=("bridge_err", "mean"),
        bridge_abs=("bridge_err", lambda x: x.abs().mean())).reset_index()
    S.to_csv(OUT / "e31_mech_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
