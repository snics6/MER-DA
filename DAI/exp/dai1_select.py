"""DAI 実験 1 の選ぶ段（設計 `DAI/docs/実験1_設計.md` 8 節）。仮のターゲットの出力だけを読み，固定のファイルを書く。

評価のターゲットの出力（`results/dai1/eval/`）は読まない。
"""

from __future__ import annotations

import glob
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dai1
import dai_data as DD

GRIDS = {
    "joint": [("m", [str(x) for x in dai1.M_GRID]), ("a", [str(x) for x in dai1.A_GRID])],
    "rolesep": [("m", [str(x) for x in dai1.M_GRID]), ("a", [str(x) for x in dai1.A_GRID])],
    "kd": [("temp", [str(x) for x in dai1.KD_T]), ("m", [str(x) for x in dai1.M_GRID])],
    "agree": [("tau", [str(x) for x in dai1.AG_TAU]), ("f", [str(x) for x in dai1.AG_F])],
}


def norm(v) -> str:
    """格子の値を文字列にそろえる（CSV では 1 と 1.0，"n" が混ざる）。"""
    s = str(v)
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def load(out: Path, T: str, which: int) -> pd.DataFrame:
    fs = sorted(glob.glob(str(out / "pseudo" / "units" / f"{T}_*_p{which}.csv")))
    assert fs, (T, which)
    return pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)


def delta_sel(p1: pd.DataFrame, tether: dict, n_units: int) -> dict:
    """(生徒, K) ごとの δ_sel = 2√(V/n)．V は比較の基準の UAR の，同じ (c, 分割) の中の種の間の分散の中央値。"""
    out = {}
    for s in DD.STUDENTS:
        for K in DD.KS:
            cw, cb = tether[f"{s}|{K}"]
            d = p1[(p1.stage == 1) & (p1.method == "none") & (p1.kind == s) & (p1.K == K) &
                   np.isclose(p1.Cw, cw) & np.isclose(p1.Cb, cb)]
            v = d.groupby(["c", "fold"])["uar"].var(ddof=1).median()
            out[(s, K)] = float(2 * np.sqrt(v / n_units))
        out[(s, 0)] = out[(s, 1)]                                        # K=0 の比較の基準は出発点で種の揺れが無い
    return out


def pick(g: pd.DataFrame, method: str, delta: float) -> tuple[dict, bool]:
    """格子の点の平均 UAR が最大の点．δ 以内の点が複数なら選択なしの値に最も近い点（格子の番号の差の和）。"""
    keys = [k for k, _ in GRIDS[method]]
    g = g.copy()
    for k in keys:
        g[k] = g[k].map(norm)
    m = g.groupby(keys)["uar"].mean()
    best = m.max()
    cands = m[m >= best - delta].index
    default = dai1.DEFAULTS[method]
    pos = {k: vals for k, vals in GRIDS[method]}

    def dist(ix):
        ix = ix if isinstance(ix, tuple) else (ix,)
        return (sum(abs(pos[k].index(v) - pos[k].index(norm(default[k]))) for k, v in zip(keys, ix)),
                tuple(pos[k].index(v) for k, v in zip(keys, ix)))
    chosen = min(cands, key=dist)
    chosen = chosen if isinstance(chosen, tuple) else (chosen,)
    argmax = m.idxmax()
    argmax = argmax if isinstance(argmax, tuple) else (argmax,)
    return ({k: v for k, v in zip(keys, chosen)} | {"pseudo_uar": float(m.loc[chosen if len(chosen) > 1 else chosen[0]])},
            chosen != argmax)


def main(out: Path) -> None:
    sel_dir = out / "select"
    sel_dir.mkdir(parents=True, exist_ok=True)
    frozen = {"_meta": {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=dai1.ROOT, capture_output=True, text=True).stdout.strip(),
                        "rule": "DAI/docs/実験1_設計.md 8 節"}}
    for T in DD.CORPORA:
        tw = json.load(open(out / "pseudo" / f"{T}_tether_w.json"))
        p1, p3 = load(out, T, 1), load(out, T, 3)
        n_units = p1[["c", "fold", "seed"]].drop_duplicates().shape[0]
        dsel = delta_sel(p1, tw["tether"], n_units)
        d3 = p3[(p3.stage == 3) & p3.method.isin(list(GRIDS))]
        select, tie_used = {}, 0
        for (s, K, method, ver), g in d3.groupby(["student", "K", "method", "teacher"]):
            ch, used = pick(g, method, dsel[(s, int(K))])
            select[f"{s}|{int(K)}|{method}|{ver}"] = ch
            tie_used += int(used)
        main_ver, partner = {}, {}
        for K in (0,) + DD.KS:
            ms = np.mean([select[f"{s}|{K}|joint|sel"]["pseudo_uar"] for s in DD.STUDENTS])
            mo = np.mean([select[f"{s}|{K}|joint|one"]["pseudo_uar"] for s in DD.STUDENTS])
            dm = float(np.mean([dsel[(s, K)] for s in DD.STUDENTS]))
            same = tuple(tw["w"][str(K)]) == (1.0, 1.0, 1.0)
            main_ver[str(K)] = "sel" if (not same and ms - mo > dm) else "one"
            for s in DD.STUDENTS:
                v = main_ver[str(K)]
                kd, ag = select[f"{s}|{K}|kd|{v}"]["pseudo_uar"], select[f"{s}|{K}|agree|{v}"]["pseudo_uar"]
                partner[f"{s}|{K}"] = "agree" if ag - kd > dsel[(s, K)] else "kd"
        frozen[T] = {"tether": tw["tether"], "w": tw["w"], "n_units": n_units,
                     "delta_sel": {f"{s}|{K}": v for (s, K), v in dsel.items()},
                     "select": select, "main": main_ver, "partner": partner, "tie_rule_used": tie_used,
                     "units": tw["units"]}
        print(f"{T}: 単位 {n_units}，主の版 {main_ver}，同点の規則が働いたセル {tie_used}", flush=True)
    json.dump(frozen, open(sel_dir / "frozen.json", "w"), indent=1, ensure_ascii=False)
    fz = dai1.ROOT / "DAI" / "frozen"
    fz.mkdir(exist_ok=True)
    shutil.copy(sel_dir / "frozen.json", fz / "dai1_frozen.json")
    print("書いた:", sel_dir / "frozen.json", "と", fz / "dai1_frozen.json")
