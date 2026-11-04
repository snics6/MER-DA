"""DAI 実験 2 の選ぶ段（設計 `DAI/docs/実験2_設計.md` 第 3 版の 1 節「選び方」と 4 節）。仮のターゲットの出力だけを読み，固定のファイルを書く。

- 規則は実験 1 の 8 節の 1 と同じ（`dai1_select.pick` をそのまま使う）：(T, 生徒, K, 手法) ごとに，格子の点を T の仮のターゲットの単位の UAR の平均で
  選び，最良との差が δ_sel 以内の点が複数なら，選択なしの値に最も近い点（各つまみの格子の番号の差の和．それも同じなら格子の並びで先）．
- δ_sel は実験 1 の固定のファイル（既定 `DAI/frozen/dai1_frozen.json`．試走では引数で与えた置き場の frozen.json）の (T, 生徒, K) の値．
  δ_sel は K 本の抜き方の揺れだけを含み，勾配法の手法の初期化の揺れ（s = r）を含まない（表の注に書く）．
- 選択なしの値が無い xMUDA 型は，δ_sel の中の点のうち格子の並び（学習率，反復，κ，μ の順の辞書順）で先の点を選ぶ．
- 選択なしの値（`dai2.DEFAULTS`・`dai2_grad.DEFAULTS`）は，固定のファイルの _meta に写す．格子の点を持たない saerens は平均だけを書く．
- 止まる条件：評価の回の出力（`<out>/eval/units/`）が 1 つでもある．仮のターゲットの単位の集合が実験 1 の固定のファイルの単位の一覧と違う
  （部分ごと）．格子の点の行の数が単位の数と違う（欠けか重複）．全部の行のコミットが 1 つでない．引き戻しの鍵が実験 1 の固定のファイルの
  引き戻しと違う．本番の置き場（`results/dai2`）で未コミットの変更がある（試走の置き場では --allow-dirty で進め，_meta.dirty に残す）．
- 書き出し：`<out>/select/frozen.json`．本番の置き場のときだけ写しを `DAI/frozen/dai2_frozen.json` に置く（コミットは人が行う）．
  _meta に台本の sha256（評価の段が照らす．ML の査読者の m72），実験 1 の固定のファイルの sha256，勾配法の削り方，選択なしの値と格子を書く．
"""

from __future__ import annotations

import glob
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_select  # noqa: E402
import dai2  # noqa: E402
import dai_data as DD  # noqa: E402

ROOT = DD.ROOT
GRIDS = dict(dai2.GRIDS)
DEFAULTS = dict(dai2.DEFAULTS)
PARTS = {"lin": ("self", "entmin", "im", "t3a", "t3a_b", "lp", "own", "st", "saerens"),
         "map": ("dann", "mme", "mapctl"), "xmuda": ("xmuda",), "coral": ("coral",)}
NO_DEFAULT = ("xmuda",)


def _grad_module():
    import dai2_grad                                                       # torch の TF32 の設定を変えるので，選ぶ段でだけ読む
    return dai2_grad


def load_part(out: Path, T: str, part: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(out / "pseudo" / "units" / f"{T}_*_{part}.csv")))
    if not fs:
        sys.exit(f"{T}: 部分 {part} の仮のターゲットの出力が無い")
    return pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)


def pick_order(g: pd.DataFrame, keys: list, grid: dict, delta: float) -> tuple[dict, bool]:
    """選択なしの値が無い手法（xMUDA 型）：平均が最大の点との差が δ 以内の点のうち，格子の並び（keys の順の辞書順）で先の点。"""
    g = g.copy()
    for k in keys:
        g[k] = g[k].map(float)
    m = g.groupby(keys)["uar"].mean()
    best = m.max()
    cands = list(m[m >= best - delta].index)

    def idx(ix):
        ix = ix if isinstance(ix, tuple) else (ix,)
        return [min(range(len(grid[k])), key=lambda i: abs(grid[k][i] - v)) for k, v in zip(keys, ix)]
    chosen = min(cands, key=idx)
    ch = chosen if isinstance(chosen, tuple) else (chosen,)
    am = m.idxmax()
    am = am if isinstance(am, tuple) else (am,)
    return {k: v for k, v in zip(keys, ch)} | {"pseudo_uar": float(m.loc[chosen])}, ch != am


def grid_present(g: pd.DataFrame, method: str) -> list:
    """削り方（MME の sw，反復 200）で外した点を除いた格子。"""
    out = []
    for k, vals in GRIDS[method]:
        have = set(g[k].map(float))
        out.append((k, tuple(v for v in vals if float(v) in have)))
    return out


def select_T(T: str, d: pd.DataFrame, fz1: dict, n_units: int) -> dict:
    dsel = {tuple([k.split("|")[0], int(k.split("|")[1])]): float(v) for k, v in fz1[T]["delta_sel"].items()}
    select, tie_used, mme_sw0 = {}, {}, 0
    meths = [m for p in PARTS.values() for m in p]
    for (s, K, method), g in d[d.method.isin(meths)].groupby(["student", "K", "method"]):
        K = int(K)
        grid = grid_present(g, method)
        keys = [k for k, _ in grid]
        if keys:                                                           # 欠けと重複（格子の点ごとに単位の数だけ行がある）
            cnt = g.groupby(keys).size()
            npts = int(np.prod([len(v) for _, v in grid]))
            if len(cnt) != npts or (cnt != n_units).any():
                sys.exit(f"{T} {s} K={K} {method}: 格子の点 {len(cnt)}/{npts}，点ごとの行の数 {sorted(set(cnt))}（単位 {n_units}）")
        elif len(g) != n_units:
            sys.exit(f"{T} {s} K={K} {method}: 行の数 {len(g)}（単位 {n_units}）")
        if not keys:
            select[f"{s}|{K}|{method}"] = {"pseudo_uar": float(g["uar"].mean())}
            continue
        delta = dsel[(s, K)]
        if method in NO_DEFAULT:
            ch, used = pick_order(g, keys, dict(grid), delta)
        else:
            dai1_select.GRIDS[f"dai2:{method}"] = grid                        # 同じ規則の関数をそのまま使う（格子だけ渡す）
            ch, used = dai1_select.pick(g, f"dai2:{method}", delta, DEFAULTS[method])
        select[f"{s}|{K}|{method}"] = {k: (float(v) if k not in ("iters", "k", "form", "M", "r", "a") else int(v)) for k, v in ch.items()
                                       if k != "pseudo_uar"} | {"pseudo_uar": ch["pseudo_uar"]}
        tie_used[method] = tie_used.get(method, 0) + int(used)
        if method == "mme" and float(ch.get("sw", 0.0)) == 0.0:
            mme_sw0 += 1
    st = d[(d.method == "st")]
    st_edge = {f"{s}|{srcs}": bool(e) for (s, srcs), e in st.groupby(["student", "srcs"])["C_edge"].first().items()}
    return dict(select=select, tie_used=tie_used, mme_sw0_cells=mme_sw0, st_C_edge=st_edge, st_C_edge_n=int(sum(st_edge.values())),
                delta_sel={f"{s}|{K}": v for (s, K), v in dsel.items()})


def main(out: Path, targets=DD.CORPORA, dai1_frozen: Path = ROOT / "DAI" / "frozen" / "dai1_frozen.json", allow_dirty: bool = False) -> None:
    grad = _grad_module()
    GRIDS.update(grad.GRIDS)
    DEFAULTS.update(grad.DEFAULTS)
    ev = out / "eval" / "units"
    if ev.exists() and any(ev.iterdir()):
        sys.exit("評価の回の出力がすでにある．選ぶ段は評価の前にだけ流す（漏れの検査 L8）")
    prod = out.resolve() == (ROOT / "results" / "dai2").resolve()
    commit, dirty = dai1.git_commit()
    if dirty and (prod or not allow_dirty):
        sys.exit("選ぶ段は未コミットの変更があると流さない（試走の置き場だけ --allow-dirty で進める）")
    fz1 = json.load(open(dai1_frozen))
    sel_dir = out / "select"
    sel_dir.mkdir(parents=True, exist_ok=True)
    frozen = {"_meta": {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit, "dirty": dirty, "trial": not prod,
                        "rule": "DAI/docs/実験2_設計.md 1 節・4 節（実験 1 の 8 節の 1）",
                        "dai1_frozen": str(dai1_frozen), "dai1_frozen_sha256": dai2.sha(dai1_frozen),
                        "script_sha256": {n: dai2.sha(HERE / n) for n in dai2.SCRIPTS},
                        "grids": {m: [(k, list(v)) for k, v in gr] for m, gr in GRIDS.items()}, "defaults": DEFAULTS,
                        "no_default_tie_rule": {m: "δ_sel の中で格子の並び（辞書順）で先" for m in NO_DEFAULT},
                        "note": "δ_sel は K 本の抜き方の揺れだけを含み，勾配法の手法の初期化の揺れを含まない",
                        "grad_seeds_eval": list(grad.GRAD_SEEDS_EVAL)}}
    if list(targets) != list(DD.CORPORA):
        frozen["_meta"]["partial_targets"] = list(targets)
    commits, cuts = set(), set()
    for T in targets:
        want = sorted(fz1[T]["units"])
        parts = {}
        for part in PARTS:
            dp = load_part(out, T, part)
            got = sorted({f"{c}_f{f}_r{r}" for c, f, r in dp[["c", "fold", "seed"]].drop_duplicates().itertuples(index=False)})
            if got != want:
                sys.exit(f"{T}: 部分 {part} の単位が実験 1 の固定のファイルの単位の一覧と違う（m60）")
            parts[part] = dp
            commits |= set(dp.commit.astype(str))
            keyT = dai2.tether_key(dai2.parse_tether(fz1[T]["tether"]))
            if set(dp.tether_key.astype(str)) != {keyT}:
                sys.exit(f"{T}: 部分 {part} の引き戻しが実験 1 の固定のファイルの引き戻しと違う")
        for part in ("map", "xmuda"):
            for f in sorted(glob.glob(str(out / "pseudo" / "units" / f"{T}_*_{part}.npz"))):
                z = np.load(f)
                cuts.add((part, str(z["extra_tag"])))
        d = pd.concat(parts.values(), ignore_index=True)
        r = select_T(T, d, fz1, len(want))
        frozen[T] = r | {"n_units": len(want), "units": want, "n_seeds_eval_linear": fz1[T].get("n_seeds_eval"),
                         "pseudo_commit": sorted(set(d.commit.astype(str)))}
        print(f"{T}: 単位 {len(want)}，同点の規則が働いたセル {r['tie_used']}，MME で sw = 0 を選んだセル {r['mme_sw0_cells']}，"
              f"S+T の C が格子の端 {r['st_C_edge_n']}/{len(r['st_C_edge'])}", flush=True)
    if len(commits) != 1:
        sys.exit(f"仮のターゲットの行のコミットが 1 つでない {commits}（比較の検査 C6）")
    mcut = {json.loads(t)["mme_sw01"] for p, t in cuts if p == "map"}
    icut = {json.loads(t)["iter200"] for p, t in cuts}
    if len(mcut) > 1 or len(icut) > 1:
        sys.exit(f"単位によって勾配法の削り方が違う {cuts}")
    frozen["_meta"]["grad_cut"] = {"mme_sw01": bool(mcut.pop()) if mcut else False, "iter200": bool(icut.pop()) if icut else False}
    frozen["_meta"]["pseudo_commit"] = commits.pop()
    json.dump(frozen, open(sel_dir / "frozen.json", "w"), indent=1, ensure_ascii=False)
    h = hashlib.sha256(open(sel_dir / "frozen.json", "rb").read()).hexdigest()
    print("書いた:", sel_dir / "frozen.json", "sha256", h)
    if prod:                                                               # 本番の置き場のときだけ写しを置く（I8）
        fz = ROOT / "DAI" / "frozen"
        fz.mkdir(exist_ok=True)
        shutil.copy(sel_dir / "frozen.json", fz / "dai2_frozen.json")
        print("写し:", fz / "dai2_frozen.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dai2"))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--dai1-frozen", default=str(ROOT / "DAI" / "frozen" / "dai1_frozen.json"))
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    main(Path(a.out_dir), a.targets, Path(a.dai1_frozen), a.allow_dirty)
