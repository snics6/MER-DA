"""【本番の記述】E82 の設計 2 節の記述の列：entmin・IM・自己学習の解のアンカーからの離れを，C′ の倍率（1・3・10）ごとにまとめる（CPU）。

関門 gate-code の計算の後の点検（`checks/E82/code_post.md` の指摘 1．前回の [I1-dist]）で，倍率ごとの表が出ていないと指摘された．
査読の前提「正則化を弱めると重みが動く」を確かめる記述であり，判定には使わない．評価のテストの値（UAR，予測，対数の確率）は読まない．
台本の検査（`checks/E82/code_dist.md`）の指摘で，主の列を生の特徴の座標の離れにし，同じ C′ の none からの離れ，大きさの比，余弦を足した．

入力（どれも学習側の量．テストのラベルを使わない）
  - E77（倍率 1）と E82（倍率 3・10，none の倍率の対照）が保存した生の特徴の座標の重み `results/e7?_<実行>/<T>_r<r>_W_<生徒>.npy`・`_b_`・`_wrows_`
    （`compare_dai.Rec.add`・`Rec.save`．logits ＝ X_raw @ Wᵀ ＋ b）．行の鍵は同じ場所の `<T>.csv` から uar の列を除いて読む
  - 補助の列：E82 が生徒の座標 Z ＝ ((X − c)/sd) @ A_M の上で書いた `results/e82_<実行>/<T>_dist.csv` の dW_rel（倍率 3・10 と none）
  - 選んだ設定：固定の版 `results/e79_p4/select/chosen.csv` と選び直した版 `results/e82_analysis/reselect/select/chosen.csv`
    （variant ＝ excl，selector ＝ 手法．E82 の範囲の生徒，K ∈ {0, 5}）
種のある行（param の末尾 s1・s2）と m_eff が tim の行（E79・E82 は選ばない）は除き，K は E82 の範囲の 0・5 に限る．

量（同じ（実行, T, r, 生徒）の中で）
  - dW_anchor ＝ ‖W − W₀‖_F ÷ ‖W₀‖_F，db_anchor ＝ ‖b − b₀‖（W₀・b₀ は K=0 の none ＝ 正則化の中心）
  - dW_samecp ＝ ‖W − W_none(K, 同じ C′)‖_F ÷ ‖W₀‖_F（K=5 で K 本のラベルによる動きを分ける．K=0 の none は C′ に依らない）
  - norm_ratio ＝ ‖W‖_F ÷ ‖W₀‖_F，cos ＝ ⟨W, W₀⟩_F ÷ (‖W‖_F ‖W₀‖_F)
  - dW_Z（補助）：E82 の `_dist.csv` の値（倍率 3・10 と none だけ）

出力（`results/e82_analysis/dist/`）
  - `dist_rows.csv`：行ごとの量
  - `dist_grid.csv`：（実行, 生徒, K, 手法, 倍率）ごとに格子の全設定 × T × 反復の数，中央値，四分位
  - `dist_pairs.csv`：同じ（実行, T, r, 生徒, K, 手法, 倍率を除いた設定, m_eff）で倍率 1・3・10 をそろえ，1 < 3 < 10 の順に dW_anchor が増えた組の数
  - `dist_selected.csv`・`dist_selected_units.csv`：固定の版と選び直した版の選んだ設定の量（単位ごとの反復の中央値と，その中央値）
  - `checks.csv`：照合（E77 と E82 の none の W・b の一致（K=0・5），アンカーの離れが 0，行の重複，格子と反復のそろい，選んだ単位に行がある）
  - `provenance.json`（入力の指紋：chosen.csv の sha256，E77・E82 の provenance のコミット）

    ./.venv/bin/python exp/e82_dist_table.py
"""
from __future__ import annotations

import hashlib
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
CHOSEN = {"fixed": ROOT / "results" / "e79_p4" / "select" / "chosen.csv",
          "reselect": ROOT / "results" / "e82_analysis" / "reselect" / "select" / "chosen.csv"}
KEYCOLS = ["target", "rep", "row_id", "student", "method", "param", "meff", "K"]
TOL_W = 1e-6        # E77 と E82 の none の W・b の一致（同じ解．float32 の保存）


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mult_of(param: str) -> int:
    m = re.search(r"C(\d+)x", str(param))
    return int(m.group(1)) if m else 1


def base_param(param: str) -> str:
    return re.sub(r"C\d+x", "", str(param))


def seeded(param: str) -> bool:
    return bool(re.search(r"s\d+$", str(param)))


def load_wave(wave: str, run: str, students) -> tuple[pd.DataFrame, dict]:
    """（wave の実行の場所の）全ターゲット × 反復 × 生徒の W・b を行の鍵と結ぶ．戻り値は行の表と {(T, r, 生徒, row_id, wave): (W, b)}。"""
    d = ROOT / "results" / f"{wave}_{run}"
    metas, Wb = [], {}
    for f in sorted(d.glob("*.csv")):
        if not re.fullmatch(r"[A-Za-z-]+\.csv", f.name):
            continue
        T = f.stem
        meta = pd.read_csv(f, usecols=lambda c: c in KEYCOLS, dtype=str)
        meta = meta[meta.target == T]
        for st in students:
            for r in sorted(meta.rep.unique(), key=int):
                wr = d / f"{T}_r{r}_wrows_{st}.npy"
                if not wr.exists():
                    continue
                ids = np.load(wr, allow_pickle=True).astype(int)
                W = np.load(d / f"{T}_r{r}_W_{st}.npy")
                B = np.load(d / f"{T}_r{r}_b_{st}.npy")
                if not (len(ids) == W.shape[0] == B.shape[0]):
                    raise SystemExit(f"{wave} {T} r{r} {st}：wrows {len(ids)}，W {W.shape}，b {B.shape} の行の数が違う")
                m = meta[(meta.rep == r) & (meta.student == st)].copy()
                m["row_id"] = m.row_id.astype(int)
                pos = {int(i): k for k, i in enumerate(ids)}
                m = m[m.row_id.isin(pos) & m.method.isin(METHODS) & ~m.param.map(seeded) & (m.meff != "tim")]
                for _, row in m.iterrows():
                    k = pos[int(row.row_id)]
                    Wb[(T, r, st, int(row.row_id), wave)] = (W[k].astype(np.float64), B[k].astype(np.float64))
                m["wave"] = wave
                metas.append(m)
    x = pd.concat(metas, ignore_index=True)
    x["run"] = run
    return x, Wb


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
        z82 = []
        for f in sorted((ROOT / "results" / f"e82_{run}").glob("*_dist.csv")):
            z = pd.read_csv(f, dtype=str)
            z82.append(z[z.target.notna() & (z.target != "")])
        z82 = pd.concat(z82, ignore_index=True)
        students = sorted(z82.student.unique())
        m77, W77 = load_wave("e77", run, students)
        m82, W82 = load_wave("e82", run, students)
        m77 = m77[((m77.method == "none") & (m77.param == "-")) | (m77.method != "none")]   # E77 は倍率 1 だけ
        m82 = m82[m82.param.map(mult_of).isin([3, 10]) | ((m82.method == "none") & (m82.param == "-"))]
        # 照合 1：E77 と E82 の none（K=0・5，E39 の C′）の W・b が同じ
        n77 = m77[(m77.method == "none") & (m77.param == "-")]
        n82 = m82[(m82.method == "none") & (m82.param == "-")]
        j = n77.merge(n82, on=["target", "rep", "student", "K"], suffixes=("_77", "_82"))
        dmax = 0.0
        for _, q in j.iterrows():
            a = W77[(q.target, q.rep, q.student, int(q.row_id_77), "e77")]
            b = W82[(q.target, q.rep, q.student, int(q.row_id_82), "e82")]
            dmax = max(dmax, float(np.abs(a[0] - b[0]).max()), float(np.abs(a[1] - b[1]).max()))
        checks.append({"check": "none_W_e77_vs_e82", "run": run, "n": len(j), "n_expected": len(n82), "max_abs": dmax,
                       "ok": bool(len(j) == len(n82) and len(j) > 0 and dmax <= TOL_W)})
        m82 = m82[~((m82.method == "none") & (m82.param == "-"))]                          # 倍率 1 の none は E77 の行を使う
        both = pd.concat([m77, m82], ignore_index=True)
        both = both[both.K.isin(["0", "5"])]                                              # E82 の範囲は K ∈ {0, 5}（E77 の K=25 は倍率 1 だけ）
        both["mult"] = both.param.map(mult_of)
        WB = {**W77, **W82}
        out = []
        for (T, r, st), g in both.groupby(["target", "rep", "student"]):
            a = g[(g.method == "none") & (g.K == "0") & (g.mult == 1)]
            if len(a) != 1:
                raise SystemExit(f"{run} {T} r{r} {st}：K=0 の none（倍率 1）が {len(a)} 行")
            W0, b0 = WB[(T, r, st, int(a.row_id.iloc[0]), a.wave.iloc[0])]
            n0 = np.linalg.norm(W0)
            ref = {(q.K, q.mult): WB[(T, r, st, int(q.row_id), q.wave)][0] for _, q in g[g.method == "none"].iterrows()}
            for _, q in g.iterrows():
                W, b = WB[(T, r, st, int(q.row_id), q.wave)]
                Wn = ref.get((q.K, q.mult))
                if Wn is None and q.K == "0":
                    Wn = ref.get(("0", 1))
                out.append({"run": run, "target": T, "rep": r, "student": st, "K": q.K, "method": q.method, "param": q.param,
                            "meff": q.meff, "mult": q.mult, "wave": q.wave,
                            "dW_anchor": float(np.linalg.norm(W - W0) / n0), "db_anchor": float(np.linalg.norm(b - b0)),
                            "dW_samecp": float(np.linalg.norm(W - Wn) / n0) if Wn is not None else np.nan,
                            "norm_ratio": float(np.linalg.norm(W) / n0), "cos": float((W * W0).sum() / (np.linalg.norm(W) * n0))})
        R = pd.DataFrame(out)
        zz = z82[z82.method.isin(METHODS)][["target", "rep", "student", "method", "param", "meff", "K", "dW_rel"]].rename(columns={"dW_rel": "dW_Z"})
        if zz.duplicated(["target", "rep", "student", "method", "param", "meff", "K"]).any():
            raise SystemExit(f"{run}：E82 の _dist.csv に同じ鍵の行が重なる")
        R = R.merge(zz, on=["target", "rep", "student", "method", "param", "meff", "K"], how="left")
        R["dW_Z"] = R.dW_Z.astype(float)
        a0 = R[(R.method == "none") & (R.K == "0") & (R.mult == 1)].dW_anchor.abs()
        checks.append({"check": "anchor_zero", "run": run, "n": len(a0), "n_expected": np.nan, "max_abs": float(a0.max()), "ok": bool(a0.max() == 0)})
        cnt = R[R.method != "none"].groupby(["student", "K", "method", "mult"]).size().unstack("mult")
        okg = bool(cnt.notna().all().all() and (cnt.nunique(axis=1) == 1).all())
        checks.append({"check": "grid_complete", "run": run, "n": int(cnt.size), "n_expected": np.nan, "max_abs": np.nan, "ok": okg})
        rows.append(R)
    R = pd.concat(rows, ignore_index=True)
    dup = int(R.duplicated(["run", "target", "rep", "student", "method", "param", "meff", "K"]).sum())
    checks.append({"check": "duplicate_rows", "run": "all", "n": dup, "n_expected": 0, "max_abs": np.nan, "ok": dup == 0})
    R.to_csv(OUT / "dist_rows.csv", index=False)

    def q(p):
        return lambda v: v.quantile(p)
    grid = R.groupby(["run", "student", "K", "method", "mult"]).agg(
        n=("dW_anchor", "size"), dW_anchor_med=("dW_anchor", "median"), dW_anchor_q25=("dW_anchor", q(.25)), dW_anchor_q75=("dW_anchor", q(.75)),
        dW_samecp_med=("dW_samecp", "median"), norm_ratio_med=("norm_ratio", "median"), cos_med=("cos", "median"),
        db_anchor_med=("db_anchor", "median"), dW_Z_med=("dW_Z", "median")).reset_index()
    grid.to_csv(OUT / "dist_grid.csv", index=False)

    P = R[R.method != "none"].copy()
    P["bp"] = P.param.map(base_param)
    pv = P.pivot_table(index=["run", "target", "rep", "student", "K", "method", "bp", "meff"], columns="mult", values="dW_anchor").dropna()
    inc = (pv[1] < pv[3]) & (pv[3] < pv[10])
    pairs = inc.groupby(level=["run", "student", "K", "method"]).agg(["size", "sum"]).reset_index().rename(columns={"size": "n_triples", "sum": "n_increasing"})
    pairs.to_csv(OUT / "dist_pairs.csv", index=False)
    checks.append({"check": "triples_found", "run": "all", "n": int(len(pv)), "n_expected": np.nan, "max_abs": np.nan, "ok": bool(len(pv) > 0)})

    sels, fp = [], {}
    for ver, path in CHOSEN.items():
        fp[ver] = sha256(path)
        ch = pd.read_csv(path, dtype=str)
        ch = ch[(ch.variant == "excl") & (ch.selector == ch.method) & ch.method.isin(["entmin", "im", "self"]) & ch.K.isin(["0", "5"])]
        ch = ch.merge(R[["run", "student"]].drop_duplicates(), on=["run", "student"])
        s = ch[["run", "target", "student", "K", "method", "param", "meff"]].merge(
            R, on=["run", "target", "student", "K", "method", "param", "meff"], how="left")
        miss = int(s.dW_anchor.isna().groupby([s.run, s.target, s.student, s.K, s.method]).all().sum())
        checks.append({"check": f"selected_without_rows_{ver}", "run": "all", "n": miss, "n_expected": 0, "max_abs": np.nan, "ok": miss == 0})
        u = s.groupby(["run", "target", "student", "K", "method", "param", "meff"]).agg(
            n_rep=("dW_anchor", "count"), dW_anchor=("dW_anchor", "median"), dW_samecp=("dW_samecp", "median"),
            norm_ratio=("norm_ratio", "median"), cos=("cos", "median")).reset_index()
        u["mult"] = u.param.map(mult_of)
        u["version"] = ver
        sels.append(u)
    U = pd.concat(sels, ignore_index=True)
    U.to_csv(OUT / "dist_selected_units.csv", index=False)
    S = U.groupby(["version", "run", "student", "K", "method", "mult"]).agg(
        n_units=("dW_anchor", "size"), dW_anchor_med=("dW_anchor", "median"), dW_samecp_med=("dW_samecp", "median")).reset_index()
    S.to_csv(OUT / "dist_selected.csv", index=False)

    C = pd.DataFrame(checks)
    C.to_csv(OUT / "checks.csv", index=False)
    prov82 = {run: json.loads((ROOT / "results" / f"e82_{run}" / "provenance.json").read_text()).get("commit") for run in RUNS}
    prov77 = {run: json.loads((ROOT / "results" / f"e77_{run}" / "provenance.json").read_text()).get("commit") for run in RUNS}
    (OUT / "provenance.json").write_text(json.dumps({
        "script": "exp/e82_dist_table.py", "commit": git("rev-parse", "HEAD"), "script_sha256": sha256(Path(__file__)),
        "seconds": round(time.time() - t0, 1), "runs": RUNS, "methods": METHODS, "tol_w": TOL_W, "n_rows": len(R),
        "chosen_sha256": fp, "e77_commit": prov77, "e82_commit": prov82, "checks_ok": bool(C.ok.all())}, ensure_ascii=False, indent=1))
    print(f"行 {len(R)}，格子の表 {len(grid)} 行，倍率の 3 つ組 {len(pv)}，選んだ設定の表 {len(S)} 行，照合 {int((~C.ok).sum())} 件の外れ（{OUT}）")
    if not C.ok.all():
        print(C[~C.ok].to_string())
        sys.exit(1)


if __name__ == "__main__":
    main()
