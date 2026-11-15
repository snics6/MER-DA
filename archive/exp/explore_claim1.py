"""【探索的】E29：主張 1（少数ラベルが直すのは主に判断の基準）の検定を 8 種類にそろえる（P29-1〜P29-4）。新しい学習なし。

入力：E25 の組 × K の手ごとの UAR（`OUT/e25_pairs.csv`），E13 の位置による損・向きによる損（`e15_pairs.csv` の
`loss_offset`・`loss_direction`．主な種類は `results/fixc_analysis/`，言葉系は `results/fixc_text/`），C を固定した段階 B の d・o
（`e15b_rows.csv`．同じ 2 か所．反復 0〜9）。同じコーパスの組は除く。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_claim1.py

出力：`OUT/e29_pairs.csv`（組 × K），`OUT/e29_tests.csv`，`OUT/e29_summary.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC  # noqa: E402
from explore_pairs import ba  # noqa: E402
from pair_table import OUT  # noqa: E402

MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
TEXT = ["text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text"]
DIRS = {**{k: ROOT / "results/fixc_analysis/explore_relation" for k in MAIN},
        **{k: ROOT / "results/fixc_text/explore_relation" for k in TEXT}}
KEY = ["kind", "target", "source", "K"]


def decomp() -> pd.DataFrame:
    out = []
    for kind in MAIN + TEXT:
        r = pd.read_csv(DIRS[kind] / "e15b_rows.csv", dtype={"K": str})
        r = r[(r.kind == kind) & (r.rep < 10)]
        r["pos"] = (ba(r.d_s, r.o_m) - ba(r.d_s, r.o_s)) * 100
        r["dir"] = (ba(r.d_m, r.o_m) - ba(r.d_s, r.o_m)) * 100
        out.append(r.groupby(KEY)[["pos", "dir"]].mean().reset_index())
    return pd.concat(out)


def losses() -> pd.DataFrame:
    out = []
    for kind in MAIN + TEXT:
        e = pd.read_csv(DIRS[kind] / "e15_pairs.csv", comment="#", dtype={"K": str})
        e = e[(e.kind == kind) & (e.K == "5")][["kind", "target", "source", "loss_offset", "loss_direction"]]
        out.append(e)
    L = pd.concat(out)
    L["L_crit"], L["L_disc"] = L.loss_offset * 100, L.loss_direction * 100
    return L[["kind", "target", "source", "L_crit", "L_disc"]]


def spear(x, y, alt="greater"):
    ok = x.notna() & y.notna()
    r, p = stats.spearmanr(x[ok], y[ok], alternative=alt)
    return float(r), float(p), int(ok.sum())


def main() -> int:
    P = pd.read_csv(OUT / "e25_pairs.csv", dtype={"K": str})
    P = P[[c for c in P.columns if c in KEY + ["src", "h1", "h2", "h3", "h4", "h5", "t"]]]
    P = P.merge(losses(), on=["kind", "target", "source"], how="left").merge(decomp(), on=KEY, how="left")
    P = P[[GROUP_SPEC[t][0] != GROUP_SPEC[s][0] for t, s in zip(P.target, P.source)]].copy()
    for a, b, name in (("h4", "src", "G_crit"), ("h1", "src", "G_merge"), ("h5", "h2", "G_crit_c"), ("h3", "h2", "G_merge_c"), ("h2", "src", "G_center")):
        P[name] = (P[a] - P[b]) * 100
    P["share_pos"] = P.pos / (P.pos + P.dir)
    P["share_loss"] = P.L_crit / (P.L_crit + P.L_disc)
    P["group"] = np.where(P.kind.isin(TEXT), "text4", P.kind)
    P.to_csv(OUT / "e29_pairs.csv", index=False)
    tests, summ = [], []
    P5 = P[P.K == "5"]
    for grp in ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video", "text4"] + TEXT:
        g = P5[P5.kind == grp] if grp != "text4" else P5[P5.group == "text4"]
        main_grp = grp in ("audio_wavlm", "audio_hubert", "text4")
        r1, p1, n = spear(g.G_crit, g.L_crit)
        tests.append({"group": grp, "test": "P29-1" if main_grp else "P29-1 記述", "rho": r1, "p": p1, "n": n,
                      "holds": (r1 > 0 and p1 < 0.05) if main_grp else np.nan})
        rp, pp, _ = spear(g.pos, g.L_crit)
        rd, pd_, _ = spear(g.dir, g.L_crit, alt="two-sided")
        tests.append({"group": grp, "test": "P29-2" if main_grp else "P29-2 記述", "rho": rp, "p": pp, "rho_dir": rd, "p_dir": pd_, "n": n,
                      "holds": (rp > 0.5 and abs(rd) < 0.3) if main_grp else np.nan})
        summ.append({"group": grp, "n": len(g), "L_crit": g.L_crit.mean(), "L_disc": g.L_disc.mean(), "G_crit": g.G_crit.mean(),
                     "G_merge": g.G_merge.mean(), "G_center": g.G_center.mean(), "G_crit_c": g.G_crit_c.mean(), "G_merge_c": g.G_merge_c.mean(),
                     "pos": g.pos.mean(), "dir": g.dir.mean(), "share_pos_median": g.share_pos.median(), "share_loss_median": g.share_loss.median()})
    r3, p3, n3 = spear(P5.share_pos, P5.share_loss)
    tests.append({"group": "all8", "test": "P29-3", "rho": r3, "p": p3, "n": n3, "holds": r3 > 0.5})
    for grp in ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video", "text4"]:
        g = P5[P5.kind == grp] if grp != "text4" else P5[P5.group == "text4"]
        r, p, n = spear(g.share_pos, g.share_loss)
        tests.append({"group": grp, "test": "P29-3 記述", "rho": r, "p": p, "n": n})
    for grp in MAIN:
        g = P5[P5.kind == grp]
        ok = (g.G_crit_c.mean() <= g.G_crit.mean() / 2) and (g.G_merge_c.mean() <= g.G_merge.mean() / 2)
        tests.append({"group": grp, "test": "P29-4", "G_crit": g.G_crit.mean(), "G_crit_c": g.G_crit_c.mean(),
                      "G_merge": g.G_merge.mean(), "G_merge_c": g.G_merge_c.mean(), "holds": ok, "n": len(g)})
    T, S = pd.DataFrame(tests), pd.DataFrame(summ)
    T.to_csv(OUT / "e29_tests.csv", index=False)
    S.to_csv(OUT / "e29_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
    print(S.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
