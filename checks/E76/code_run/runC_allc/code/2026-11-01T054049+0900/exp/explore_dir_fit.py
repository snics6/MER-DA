"""【探索的】E20：向きが良くなる量 I の式の形（R1）と，ニュートン法の 1 歩の向きの予言が外れる理由（R2）。

予言は `docs/exploratory-analyses.md` E20 の 1 節（計算の前に書いた）。新しい学習はしない。
R3（曲がりの下限）は `exp/explore_direction.py --kappa-floor 0.01` の出力 `e16_tests_kf0.01.csv` を読んで並べる。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_dir_fit.py

入力（OUT）：`e16_drivers_rows.csv`（`--stage drivers`），`e16_rows.csv`，`e16_tests.csv`，`e16_tests_kf0.01.csv`
出力：`OUT/e20_fit.csv`，`OUT/e20_outliers.csv`，`OUT/e20_floor.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import groups_for, load_any  # noqa: E402
from explore_direction import map_draw  # noqa: E402
from pair_table import OUT  # noqa: E402

KINDS = {"audio_wavlm": "W7", "audio_hubert": "Hu7", "video": "V", "fuse-audio-video": "AV"}
LOW_K = 1e-3


def fit_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    c = float((x * y).sum() / (x * x).sum())
    return c, float(1 - ((y - c * x) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def r1() -> pd.DataFrame:
    D = pd.read_csv(OUT / "e16_drivers_rows.csv", dtype={"K": str})
    D = D.assign(room=D.own_d / D.d_s - 1, Kn=D.K.astype(int))
    forms = {"room*K/(K+q)": D.room * D.Kn / (D.Kn + D.q), "room*K": D.room * D.Kn, "room": D.room}
    rows = []
    for kind, g in D.groupby("kind", sort=False):
        for name, x in forms.items():
            c, r2 = fit_origin(x[g.index].to_numpy(), g.I_obs.to_numpy())
            rows.append({"setting": KINDS[kind], "form": name, "n": len(g), "c": c, "r2": r2,
                         "spearman": stats.spearmanr(x[g.index], g.I_obs)[0]})
    return pd.DataFrame(rows)


def r2() -> pd.DataFrame:
    R = pd.read_csv(OUT / "e16_rows.csv", dtype={"K": str})
    R["err"] = (R.d_pred - R.d_m).abs()
    E = R.groupby(["kind", "target", "source", "K", "rep"]).err.mean().reset_index()
    shares = []
    for kind in E.kind.unique():
        data = {g: load_any(g, kind)[0] for g in groups_for(kind)}
        for (t, s), g in E[E.kind == kind].groupby(["target", "source"]):
            z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
            P = softmax(z[f"t_{t}"], axis=1)
            kap = (1 - (P ** 2).sum(1)) / 3
            for r in g.itertuples():
                _, idx = map_draw(data[t], t, int(r.rep), int(r.K))
                shares.append({"kind": kind, "target": t, "source": s, "K": r.K, "rep": r.rep,
                               "share_lowk": float((kap[idx] < LOW_K).mean())})
    E = E.merge(pd.DataFrame(shares), on=["kind", "target", "source", "K", "rep"])
    rows = []
    for kind, g in E.groupby("kind", sort=False):
        grp = g.groupby(["target", "source", "K"])
        u = g.err - grp.err.transform("mean")
        v = g.share_lowk - grp.share_lowk.transform("mean")
        ok = v.abs() > 0
        r, p = stats.spearmanr(v[ok], u[ok], alternative="greater") if ok.sum() >= 10 else (np.nan, np.nan)
        rows.append({"setting": KINDS[kind], "n_draws": len(g), "n_varying": int(ok.sum()),
                     "mean_share_lowk": g.share_lowk.mean(), "spearman": r, "p_one_sided": p})
    E.to_csv(OUT / "e20_outlier_rows.csv", index=False)
    return pd.DataFrame(rows)


def r3() -> pd.DataFrame:
    a = pd.read_csv(OUT / "e16_tests.csv")[["setting", "spearman_dc", "spearman_dc_emotionpair", "spearman_lam"]]
    b = pd.read_csv(OUT / "e16_tests_kf0.01.csv")[["setting", "spearman_dc", "spearman_dc_emotionpair", "spearman_lam"]]
    return a.merge(b, on="setting", suffixes=("_nofloor", "_floor0.01"))


def main() -> int:
    pd.set_option("display.width", 250)
    F, O, L = r1(), r2(), r3()
    F.to_csv(OUT / "e20_fit.csv", index=False)
    O.to_csv(OUT / "e20_outliers.csv", index=False)
    L.to_csv(OUT / "e20_floor.csv", index=False)
    print(F.round(3).to_string(index=False)); print(O.round(3).to_string(index=False)); print(L.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
