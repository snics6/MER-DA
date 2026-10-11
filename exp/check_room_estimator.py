"""B4：K 本で測ったソースの向き d̂_K が，向きの改善の余地 r を見ていないデータでも追うか（E28 の 6 節の追加．B4-1・B4-2）。

B4-1：言葉を含む 4 種類，反復 0〜9 の平均の d̂_10．B4-2：主な 4 種類，見ていない反復 10〜29 の 1 回分の d̂_10。
どちらもターゲットの中の順位相関（種類 × ターゲットの平均を引く）。r は E13（own_d/d_s − 1，6 対の平均）。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/check_room_estimator.py

出力：`OUT/e28t6b_pairs.csv`，`OUT/e28t6b_tests.csv`
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
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_direction import map_draw  # noqa: E402
from explore_pairs import measure  # noqa: E402
from explore_room import d_hat  # noqa: E402
from pair_table import OUT  # noqa: E402

MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
TEXT = ["text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text"]


def rooms() -> pd.DataFrame:
    e13 = pd.read_csv(OUT / "e13_pairs.csv")
    have = set(e13.kind)
    e13 = pd.concat([e13] + [measure(k).assign(kind=k) for k in MAIN + TEXT if k not in have])
    e13 = e13[e13.d > 0.05]                                   # ソースの向きがほぼ 0 の対は比が定まらない
    e13["room"] = e13.own_d / e13.d - 1
    return e13.groupby(["kind", "target", "source"]).room.mean().reset_index()


def dhat_rows(kind: str, reps) -> pd.DataFrame:
    groups = groups_for(kind)
    dfs = {g: load_any(g, kind)[0] for g in groups}
    out = []
    for s in groups:
        z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
        for t in groups:
            if t == s or GROUP_SPEC[t][0] == GROUP_SPEC[s][0]:
                continue
            y = dfs[t].label_4.to_numpy()
            L = z[f"t_{t}"].astype(np.float64)
            for r in reps:
                _, idx = map_draw(dfs[t], t, r, 10)
                out.append({"kind": kind, "target": t, "source": s, "rep": r, "dhat10": d_hat(L[idx], y[idx])})
    return pd.DataFrame(out)


def within(df: pd.DataFrame, x: str, y: str) -> tuple[float, float, int]:
    g = df.dropna(subset=[x, y]).copy()
    for c in (x, y):
        g[c] = g[c] - g.groupby(["kind", "target"])[c].transform("mean")
    r, p = stats.spearmanr(g[x], g[y], alternative="less")
    return float(r), float(p), len(g)


def main() -> int:
    R = rooms()
    A = pd.concat([dhat_rows(k, range(10)) for k in TEXT]).groupby(["kind", "target", "source"]).dhat10.mean().reset_index()
    A = A.merge(R, on=["kind", "target", "source"])
    B = pd.concat([dhat_rows(k, range(10, 30)) for k in MAIN]).merge(R, on=["kind", "target", "source"])
    pd.concat([A.assign(part="B4-1"), B.assign(part="B4-2")]).to_csv(OUT / "e28t6b_pairs.csv", index=False)
    tests = []
    r, p, n = within(A, "dhat10", "room")
    tests.append({"test": "B4-1", "group": "text4", "rho": r, "p": p, "n": n, "holds": r <= -0.4 and p < 0.05})
    for k in TEXT:
        rk, pk, nk = within(A[A.kind == k], "dhat10", "room")
        tests.append({"test": "B4-1 記述", "group": k, "rho": rk, "p": pk, "n": nk})
    for k in MAIN:
        per = [within(B[(B.kind == k) & (B.rep == rr)], "dhat10", "room")[0] for rr in range(10, 30)]
        tests.append({"test": "B4-2" if k in ("audio_wavlm", "audio_hubert") else "B4-2 記述", "group": k,
                      "rho": float(np.mean(per)), "rho_min": float(np.min(per)), "rho_max": float(np.max(per)), "n_reps": len(per),
                      "holds": float(np.mean(per)) <= -0.4 if k in ("audio_wavlm", "audio_hubert") else np.nan})
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "e28t6b_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
