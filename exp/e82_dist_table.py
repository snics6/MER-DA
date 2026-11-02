"""【本番の記述】E82 の設計 2 節の記述の列：entmin・IM・自己学習の解のアンカーからの離れを，C′ の倍率（1・3・10）ごとにまとめる（CPU）。

関門 gate-code の計算の後の点検（`checks/E82/code_post.md` の指摘 1．前回の [I1-dist]）で，倍率ごとの表が出ていないと指摘された．
査読の前提「正則化を弱めると重みが動く」を確かめる記述であり，判定には使わない．評価のテストの値（UAR，予測，対数の確率）は読まない．

入力（どれも学習側の量．テストのラベルを使わない）
  - 倍率 3・10：E82 が行ごとに書いた `results/e82_<実行>/<T>_dist.csv`（評価の単位の行．dW_rel ＝ ‖W − W₀‖_F ÷ ‖W₀‖_F，db ＝ ‖b − b₀‖．
    W₀・b₀ は K=0 の none ＝ E39 のアンカー）
  - 倍率 1：E82 が解き直した none（param '-'）だけがある．entmin・IM・自己学習の倍率 1 の行は E77 にあるが，E77 は W・b を生の特徴の座標で
    保存し（`compare_dai.Rec.add`），E82 は生徒の座標 Z ＝ ((X − c)/sd) @ A_M の上で離れを測るので，相対の離れは同じ量にならない
    （試しで同じ none の行の差が最大 0.12 だった）．このため倍率 1 の手法の行は出さず，選ばれた単位の数だけを数える（2026-11-02 本体の判断）
  - 選んだ設定：`results/e82_analysis/reselect/select/chosen.csv`（selector ＝ 手法，variant ＝ excl．P4・P4′ の選び方）
種のある行（param の末尾 s1・s2）は除き，種 0 の行だけを使う．

出力（`results/e82_analysis/dist/`）
  - `dist_rows.csv`：評価の単位の行ごとの離れ（実行, 生徒, T, 反復, K, 手法, param, meff, 倍率, dW_rel, db, 出所）
  - `dist_grid.csv`：（実行, 生徒, K, 手法, 倍率）ごとに，格子の全設定 × T × 反復の dW_rel・db の数，中央値，四分位
  - `dist_selected.csv`：（実行, 生徒, K, 手法, 選ばれた倍率）ごとに，選ばれた単位の数と，選んだ設定の dW_rel の反復の中央値の中央値
  - `checks.csv`：照合（K=0 の none の離れが 0，行の重複が無い，選ばれた倍率 3・10 の単位に行がある）
  - `provenance.json`

    ./.venv/bin/python exp/e82_dist_table.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ("audio_wavlm", "video", "text_roberta", "fuse-audio-video-text", "fuse-audio-video")
METHODS = ("none", "entmin", "im", "self")
OUT = ROOT / "results" / "e82_analysis" / "dist"
KEYCOLS = ["target", "rep", "row_id", "student", "method", "param", "meff", "K"]


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def mult_of(param: str) -> int:
    m = re.search(r"C(\d+)x", str(param))
    return int(m.group(1)) if m else 1


def seeded(param: str) -> bool:
    return bool(re.search(r"s\d+$", str(param)))


def e82_rows(run: str) -> pd.DataFrame:
    out = []
    for f in sorted((ROOT / "results" / f"e82_{run}").glob("*_dist.csv")):
        x = pd.read_csv(f, dtype=str)
        x = x[x.target.notna() & (x.target != "")]
        out.append(x)
    x = pd.concat(out, ignore_index=True)
    x = x[x.method.isin(METHODS) & ~x.param.map(seeded)].copy()
    x["mult"] = x.param.map(mult_of)
    x["src"] = "e82"
    return x


def main() -> None:
    t0 = time.time()
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"{OUT} が既にある．中身を確かめてから消す")
    bad = [ln for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if re.search(r"\s(exp|measure)/", ln)]
    if bad:
        raise SystemExit(f"exp/ か measure/ に未コミットの変更がある：{bad}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    for run in RUNS:
        x82 = e82_rows(run)
        z = x82[(x82.method == "none") & (x82.param == "-") & (x82.K == "0")].dW_rel.astype(float).abs()
        checks.append({"check": "anchor_zero", "run": run, "n": len(z), "max_abs": float(z.max()) if len(z) else np.nan,
                       "ok": bool(len(z) == 0 or z.max() <= 1e-6)})
        x82 = x82.copy()
        x82["run"] = run
        rows.append(x82)
    R = pd.concat(rows, ignore_index=True)
    R["dW_rel"] = R.dW_rel.astype(float)
    R["db"] = R.db.astype(float)
    dup = R.duplicated(["run", "target", "rep", "student", "method", "param", "meff", "K"]).sum()
    checks.append({"check": "duplicate_rows", "run": "all", "n": int(dup), "max_abs": np.nan, "ok": bool(dup == 0)})
    R[["run", "student", "target", "rep", "K", "method", "param", "meff", "mult", "dW_rel", "db", "src"]].to_csv(OUT / "dist_rows.csv", index=False)

    g = R.groupby(["run", "student", "K", "method", "mult"])
    grid = g.agg(n=("dW_rel", "size"), dW_med=("dW_rel", "median"), dW_q25=("dW_rel", lambda v: v.quantile(.25)),
                 dW_q75=("dW_rel", lambda v: v.quantile(.75)), db_med=("db", "median")).reset_index()
    grid.to_csv(OUT / "dist_grid.csv", index=False)

    ch = pd.read_csv(ROOT / "results" / "e82_analysis" / "reselect" / "select" / "chosen.csv", dtype=str)
    ch = ch[(ch.variant == "excl") & (ch.selector == ch.method) & ch.method.isin(["entmin", "im", "self"])].copy()
    ch = ch.merge(R[["run", "student"]].drop_duplicates(), on=["run", "student"])     # E82 が選び直した生徒だけ（世界の単モダリティの生徒は範囲の外）
    ch = ch[ch.K.isin(["0", "5"])].copy()                                         # E82 の範囲は K ∈ {0, 5}
    ch["mult"] = ch.param.map(mult_of)
    sel = ch.merge(R, left_on=["run", "target", "student", "K", "method", "param", "meff"],
                   right_on=["run", "target", "student", "K", "method", "param", "meff"], how="left")
    s310 = sel[sel.mult_x.isin([3, 10])] if "mult_x" in sel else sel[sel.param.map(mult_of).isin([3, 10])]
    miss = s310.dW_rel.isna().groupby([s310.run, s310.target, s310.student, s310.K, s310.method]).all().sum()
    checks.append({"check": "selected_without_rows", "run": "all", "n": int(miss), "max_abs": np.nan, "ok": bool(miss == 0)})
    per_unit = sel.groupby(["run", "target", "student", "K", "method", "param", "meff"], dropna=False).agg(
        dW_rep_med=("dW_rel", "median"), n_rep=("dW_rel", "count")).reset_index()
    per_unit["mult"] = per_unit.param.map(mult_of)
    selected = per_unit.groupby(["run", "student", "K", "method", "mult"]).agg(
        n_units=("dW_rep_med", "size"), dW_med=("dW_rep_med", "median")).reset_index()
    selected.to_csv(OUT / "dist_selected.csv", index=False)
    per_unit.to_csv(OUT / "dist_selected_units.csv", index=False)
    C = pd.DataFrame(checks)
    C.to_csv(OUT / "checks.csv", index=False)
    (OUT / "provenance.json").write_text(json.dumps({
        "script": "exp/e82_dist_table.py", "commit": git("rev-parse", "HEAD"), "seconds": round(time.time() - t0, 1),
        "runs": RUNS, "methods": METHODS, "n_rows": len(R), "checks_ok": bool(C.ok.all())}, ensure_ascii=False, indent=1))
    print(f"行 {len(R)}，格子の表 {len(grid)} 行，選んだ設定の表 {len(selected)} 行，照合 {int((~C.ok).sum())} 件の外れ（{OUT}）")
    if not C.ok.all():
        print(C[~C.ok].to_string())
        sys.exit(1)


if __name__ == "__main__":
    main()
