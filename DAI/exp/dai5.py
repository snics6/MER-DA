"""DAI 実験 5：ソース × モダリティの識別器を K 本の採点で重みづけた疑似ラベル `srcw`（設計 `DAI/docs/実験5_設計.md` 第 1 版）。

実験 1 の保存（仮のターゲットの回と評価の回の段 1，段 source の識別器）を入力にして，生徒の学習（実験 1 の段 3 に相当）だけを新しく解く。
段 1・段 2 は流し直さない。DAI/exp のほかの台本は import して使うだけで，書き換えない。

段（--stage）：
  pseudo ：設計 1・4 節。仮のターゲットの回の 120 単位で，生徒 4 × K 5 × β（K = 0 は β = 0 の 1 点，K ≥ 1 は 5 点）= 84 問を解く。
  select ：設計 1 節の規則で (T, K) ごとに β を選び，`results/dai5/select/frozen.json` と写し `DAI/frozen/dai5_frozen.json` に書く。
  detect ：設計 4 節。`dai5_stats.py` の detect（族 7 の検出できる差と記述に回すセル）を呼ぶ。
  eval   ：設計 4 節。評価の回の単位（200 単位）で同じ格子を解く。条件は `check_eval_gate5`（下）。
  集計（stats）は別の台本 `dai5_stats.py`。

実装の対応（設計の節 → このファイル）：
  1 節 集合 C      ：`dai1.Unit.src[m]`（段 source の保存 (W, b)．単位のソースの順）．`source_logprobs` が候補とテストの標準化した特徴に当てる。
  1 節 採点 U_sm   ：K ≥ 1 で，K 本（`u.kpos[K]`）の判定の UAR（%）．`u.ypool[u.kpos[K]]` だけを読む（テストのラベルも K 本以外の候補のラベルも読まない）。
  1 節 重み        ：`weights(scores, beta)` = 3 · softmax(β U / 100)．K = 0 は等しい重み 3/|C|。
  1 節 疑似ラベル Q：softmax(Σ w_sm log p_sm)（対数線形プール）。候補の全発話（K 本を含む）に付ける。
  1 節 生徒        ：`dai1.solve_items`（`dai_solver.fit`）。G = a·gk + soft_rows(Q, ρ n, balance=True)。ρ・a は実験 1 の固定のファイルの `joint`（主の版）の選んだ点，
                     引き戻し (C′, C_b) は実験 1 の固定の値（K = 0 は K = 1 の値），出発点は Unit の start。新しく選ぶのは β だけ。
  4 節 保存        ：`results/dai5/<pseudo|eval>/units/<T>_<c|ALL>_f<f>_r<r>_p5.npz/.csv`。
                     p5_{s}_K{K}_pred（β の全点のテストの判定．int8），_W・_b（float32），_items（json．method・beta・rho・a），
                     p5_src_names（C の名前 "<ソース>_<モダリティ>"），p5_src_test_pred（各識別器のテストの判定．int8 (|C|, n_test)），
                     p5_K{K}_scores（採点 U_sm．K = 0 は NaN），p5_K{K}_w_b{β}（重み），sample_pool・sample_K{K}・sample_test・cluster_test の写し。
                     rows.csv：method=srcw の行（student，K，beta，rho，a，uar，grad，stalled，finite，S），`_teacher_stats`（Q の鋭さ・有効な本数・最も重い識別器），
                     `_teacher_check`（pseudo だけ：各識別器の候補での UAR と実験 1 の `_teacher_stats` の `tt:src_<s>_<m>` の照合），`_teacher_pool`（pseudo だけ：Q の候補での UAR），`_conv`。
                     コミット，dirty，固定のファイルの sha256（frozen_sha256：評価の回は dai5 の固定のファイル，仮のターゲットの回は空），dai1_frozen_sha256。
  4 節 自己検査    ：`--selftest-rho0`（試走だけ）：ρ = 0（疑似ラベルなし）で解いた判定が実験 1 の段 1 の none[K]（K = 0 は出発点）の判定と一致するか（`_rho0check` の行）。
  どの段も成績の値を画面に出さない（行の数・収束・所要だけ）。
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
BETA_GRID = (0.0, 2.5, 5.0, 10.0, 20.0)           # 規則の強さ β．0 は等しい重み（選択なしの値）
DEFAULT_BETA = 0.0
MODS = "AVT"
KS_ALL = dai1.KS_ALL
IN_DIR = ROOT / "results" / "dai1"
OUT_DIR = ROOT / "results" / "dai5"
FROZEN1 = ROOT / "DAI" / "frozen" / "dai1_frozen.json"
GATE5 = ROOT / "DAI" / "frozen" / "eval_gate_dai5.json"
EVAL_REQUIRED = [                                           # 設計 4 節の評価の段の守り 2
    "DAI/docs/実験5_設計.md", "DAI/exp/dai5.py", "DAI/exp/dai5_stats.py",
    "DAI/exp/dai1.py", "DAI/exp/dai_solver.py", "DAI/exp/dai_data.py", "DAI/exp/dai1_select.py", "DAI/exp/dai1_stats.py",
    "DAI/frozen/dai1_frozen.json", "DAI/frozen/dai5_frozen.json",
]
APPROVAL_FOLDER = "DAI/checks/DAI-5/"
CODE_FIXED = ["DAI/exp/dai5.py", "DAI/exp/dai1.py", "DAI/exp/dai_solver.py", "DAI/exp/dai_data.py"]


def betas_for(K: int) -> tuple:
    return (DEFAULT_BETA,) if K == 0 else BETA_GRID


def grid_strings() -> np.ndarray:
    return np.array([f"{K}|{b}" for K in KS_ALL for b in betas_for(K)])


def joint_sel(fz1T: dict, s: str, K: int) -> dict:
    """実験 1 の固定のファイルの `joint`（主の版）の選んだ点（ρ, a）。"""
    v = fz1T["main"][str(K)]
    j = fz1T["select"][f"{s}|{K}|joint|{v}"]
    return dict(teacher=v, rho=float(j["rho"]), a=float(j["a"]))


def weights(scores: np.ndarray, beta: float) -> np.ndarray:
    """w_sm = 3 · softmax(β U_sm / 100)．合計 3（実験 1 の 3 因子の積と同じ鋭さ）。"""
    z = float(beta) * np.asarray(scores, float) / 100.0
    z = z - z.max()
    e = np.exp(z)
    return 3.0 * e / e.sum()


def source_logprobs(u: dai1.Unit):
    """C の各識別器の，候補の対数確率（float64, GPU）とテストの判定（int8）。名前は "<ソース>_<モダリティ>"。"""
    names, LP, TP = [], [], []
    for m in MODS:
        for c, W, b in u.src[m]:
            names.append(f"{c}_{m}")
            LP.append(torch.log_softmax(S.logits(u.Xp[m], W, b), -1))
            TP.append(S.logits(u.Xt[m], W, b).argmax(1).to(torch.int8).cpu().numpy())
    return names, LP, np.stack(TP)


def source_sharp_agree(LP: list) -> tuple[np.ndarray, np.ndarray]:
    """各識別器の候補での鋭さ（最大の確率の平均，エントロピー）(|C|, 2) と，判定の一致率の行列 (|C|, |C|)（ラベルは使わない）。"""
    P = [lp.exp() for lp in LP]
    sharp = np.array([[float(p.max(1).values.mean()), float(-(p * lp).sum(1).mean())] for p, lp in zip(P, LP)])
    am = [p.argmax(1) for p in P]
    agree = np.array([[float((a == b).double().mean()) for b in am] for a in am])
    return sharp, agree


def teacher_stats_row(Q: torch.Tensor, n: int, w: np.ndarray, names: list, K: int, beta: float) -> dict:
    Wt = S.soft_rows(Q, float(n))
    colsum = Q.sum(0).cpu().numpy()
    neff = ((Wt.sum(0) ** 2) / (Wt ** 2).sum(0).clamp(min=1e-30)).cpu().numpy()
    ent = float(-(Q * Q.clamp(min=1e-12).log()).sum(1).mean())
    return dict(stage=3, method="_teacher_stats", K=K, beta=float(beta), maxp=float(Q.max(1).values.mean()), entropy=ent,
                neff_w=float(w.sum() ** 2 / (w ** 2).sum()), heaviest=names[int(np.argmax(w))], w_max=float(w.max()),
                **{f"colsum_{k}": float(colsum[k]) for k in range(4)}, **{f"neff_{k}": float(neff[k]) for k in range(4)})


def solve_unit(u: dai1.Unit, tether: dict, fz1T: dict, store: dict, rows: list, z1, p3rows: pd.DataFrame | None = None,
               selftest_rho0: bool = False) -> None:
    """p3rows があれば（仮のターゲットの回だけ）候補のラベルで教師の検算の行を作る。評価の回は候補のラベルを読まない。"""
    names, LP, TP = source_logprobs(u)
    store["p5_src_names"] = np.array(names)
    store["p5_src_test_pred"] = TP
    store["p5_src_sharp"], store["p5_src_agree"] = source_sharp_agree(LP)
    zero = torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
    for K in KS_ALL:
        gk = u.gK(K) if K else zero
        kp = u.kpos[K]
        nk = np.setdiff1d(np.arange(u.n), kp)
        if K:
            yk = u.ypool[kp]                                          # 採点に使うのは K 本のラベルだけ
            scores = np.array([100.0 * S.uar(yk, LP[j][kp].argmax(1).cpu().numpy()) for j in range(len(names))])
        else:
            scores = np.full(len(names), np.nan)
        store[f"p5_K{K}_scores"] = scores
        teachers = {}
        for beta in betas_for(K):
            w = weights(scores, beta) if K else np.full(len(names), 3.0 / len(names))
            L = sum(float(wj) * LP[j] for j, wj in enumerate(w))
            Q = torch.softmax(L, -1)
            teachers[beta] = Q
            store[f"p5_K{K}_w_b{beta}"] = w
            rows.append(teacher_stats_row(Q, u.n, w, names, K, beta))
            if p3rows is not None:
                rows.append(dict(stage=3, method="_teacher_pool", K=K, beta=float(beta),
                                 uar_pool=100.0 * S.uar(u.ypool[nk], Q.argmax(1).cpu().numpy()[nk])))
        if p3rows is not None:                                             # 教師の検算（候補のラベルは仮のターゲットの回だけ読む）
            for j, nm in enumerate(names):
                mine = 100.0 * S.uar(u.ypool[nk], LP[j].argmax(1).cpu().numpy()[nk])
                ref = p3rows[(p3rows.method == "_teacher_stats") & (p3rows.teacher.astype(str) == f"tt:src_{nm}") & (p3rows.K == K)]
                theirs = float(ref.uar_pool.iloc[0]) if len(ref) else float("nan")
                rows.append(dict(stage=3, method="_teacher_check", K=K, src=nm, uar_pool_mine=mine, uar_pool_dai1=theirs,
                                 uar_pool_diff=mine - theirs, match=bool(np.isfinite(theirs) and abs(mine - theirs) < 1e-6)))
        for s in DD.STUDENTS:
            W0, b0 = u.start[s]
            cw, cb = tether[(s, max(K, 1))]
            js = joint_sel(fz1T, s, K)
            specs = []
            for beta, Q in teachers.items():
                it = dict(method="srcw", beta=float(beta), rho=js["rho"], a=js["a"])
                G = js["a"] * gk + S.soft_rows(Q, dai1.mval(js["rho"], u.n), balance=True)
                specs.append((it, G, zero, 1.0))
            dai1.solve_items(u, s, K, specs, cw, cb, W0, b0, rows, store, f"p5_{s}_K{K}")
            if selftest_rho0:                                              # ρ = 0 は none[K]（K = 0 は出発点）と同じ解になるはず（I7）
                it = dict(method="rho0", beta=0.0, rho=0.0, a=1.0)
                dai1.solve_items(u, s, K, [(it, gk.clone(), zero, 1.0)], cw, cb, W0, b0, rows, store, f"p5z_{s}_K{K}")
                if K == 0:
                    ref = z1[f"p1_{s}_start_pred"]
                else:
                    items = [str(x) for x in z1[f"p1_{s}_items"]]
                    ref = z1[f"p1_{s}_pred"][items.index(f"{K}|{cw}|{cb}")]
                rows.append(dict(stage=3, method="_rho0check", student=s, K=K,
                                 pred_agree=float((store[f"p5z_{s}_K{K}_pred"][0] == ref).mean())))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if selftest_rho0:
        ag = min(x["pred_agree"] for x in rows if x.get("method") == "_rho0check")
        if ag < 0.999:
            sys.exit(f"--selftest-rho0：ρ = 0 の判定と実験 1 の none の一致が 99.9% 未満（最小 {ag:.4f}）。止める")


def check_inputs(u: dai1.Unit, z1) -> None:
    ids = u.cp.df["sample_id"].to_numpy().astype(str)
    ok = (ids[u.pool] == z1["sample_pool"]).all() and (ids[u.test] == z1["sample_test"]).all()
    ok = ok and all((ids[u.pool[u.kpos[K]]] == z1[f"sample_K{K}"]).all() for K in DD.KS)
    ok = ok and all(np.allclose(u.stats[k][0], z1[f"stat_{k}_mu"], rtol=1e-4, atol=1e-6) and
                    np.allclose(u.stats[k][1], z1[f"stat_{k}_sd"], rtol=1e-4, atol=1e-6) for k in DD.KINDS)
    if not ok:
        sys.exit(f"{u.name()}: 実験 1 の保存の分割・K 本・標準化の統計が今の Unit と違う")


def run_unit(in_dir: Path, out_dir: Path, tag: str, T: str, c: str, srcs: list, f: int, r: int, tether: dict, fz1T: dict,
             run: dict, selftest_rho0: bool = False) -> pd.DataFrame:
    cname = "ALL" if tag == "eval" else c
    name = f"{T}_{cname}_f{f}_r{r}"
    ib = in_dir / tag / "units" / name
    base = out_dir / tag / "units" / name
    path, rpath = Path(f"{base}_p5.npz"), Path(f"{base}_p5.csv")
    grid = grid_strings()
    if path.exists() and rpath.exists():
        old = pd.read_csv(rpath, low_memory=False)
        z = np.load(path)
        fs = str(old["frozen_sha256"].iloc[0]) if "frozen_sha256" in old and pd.notna(old["frozen_sha256"].iloc[0]) else ""
        bad = str(old["commit"].iloc[0]) != run["commit"] or fs != run["frozen_sha256"]
        bad |= str(z["dai1_frozen_sha256"]) != run["dai1_frozen_sha256"] or list(z["grid"]) != list(grid)
        bad |= "finite" in old and (old["finite"].astype(str) == "False").any()
        if bad:
            sys.exit(f"保存済みの単位 {rpath.name} のコミット・固定のファイル・格子・有限のどれかが今の実行と違う（消してから流し直す）")
        return old
    for ext in ("npz", "csv"):
        if not Path(f"{ib}_p1.{ext}").exists():
            sys.exit(f"入力の保存が無い：{ib.name}_p1.{ext}")
    z1 = np.load(f"{ib}_p1.npz")
    if tag == "eval" and str(z1["frozen_sha256"]) != run["dai1_frozen_sha256"]:
        sys.exit(f"{ib.name} の実験 1 の保存の frozen_sha256 が dai1_frozen.json と違う")
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    u = dai1.Unit(in_dir, T, c, srcs, f, r)
    check_inputs(u, z1)
    p3rows = None
    if tag == "pseudo":
        if not Path(f"{ib}_p3.csv").exists():
            sys.exit(f"入力の保存が無い：{ib.name}_p3.csv（教師の検算に要る）")
        p3rows = pd.read_csv(f"{ib}_p3.csv", low_memory=False)
    store, rows = {}, []
    solve_unit(u, tether, fz1T, store, rows, z1, p3rows, selftest_rho0)
    store.update(sample_pool=z1["sample_pool"], sample_test=z1["sample_test"], cluster_test=u.cp.df["cluster"].to_numpy()[u.test].astype(str),
                 grid=grid, commit=np.array(run["commit"]), dirty=np.array(run["dirty"]), frozen_sha256=np.array(run["frozen_sha256"]),
                 dai1_frozen_sha256=np.array(run["dai1_frozen_sha256"]), gate_sha256=np.array(run["gate_sha256"]), gpu=np.array(run["gpu"]),
                 **{f"sample_K{K}": z1[f"sample_K{K}"] for K in DD.KS})
    if any(("finite" in x and x["finite"] is False) for x in rows):
        sys.exit(f"{rpath.name}: 勾配か解に NaN・Inf の問がある（計算の失敗として止める．保存しない）")
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in store.items()})
    df = pd.DataFrame(rows).assign(T=T, c=cname, fold=f, seed=r, sec=time.time() - t0, commit=run["commit"], dirty=run["dirty"],
                                   frozen_sha256=run["frozen_sha256"], dai1_frozen_sha256=run["dai1_frozen_sha256"], gpu=run["gpu"],
                                   gate_sha256=run["gate_sha256"])
    df.to_csv(rpath, index=False)
    cv = df[df.method == "srcw"]
    nconv = int(((cv.grad <= 1e-5) & ~cv.stalled.astype(bool)).sum())
    print(f"{tag} {name} p5: 問 {len(cv)}，収束 {nconv}，{time.time()-t0:.0f} 秒", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


# ------------------------------------------------------------------ 保存の読み出し（select・stats が使う）

class P5:
    """1 単位の p5 の保存。"""

    def __init__(self, base: Path):
        self.z = np.load(f"{base}_p5.npz")
        self._items = {}

    def items(self, s: str, K: int) -> list:
        if (s, K) not in self._items:
            self._items[(s, K)] = [json.loads(x) for x in self.z[f"p5_{s}_K{K}_items"]]
        return self._items[(s, K)]

    def pred(self, s: str, K: int, beta) -> np.ndarray:
        b = DEFAULT_BETA if K == 0 else float(beta)
        hits = [j for j, it in enumerate(self.items(s, K)) if np.isclose(it["beta"], b)]
        if len(hits) != 1:
            raise KeyError((s, K, beta, len(hits)))
        return self.z[f"p5_{s}_K{K}_pred"][hits[0]]


def load_rows(out_dir: Path, tag: str, T: str) -> pd.DataFrame:
    fs = sorted((out_dir / tag / "units").glob(f"{T}_*_p5.csv"))
    if not fs:
        sys.exit(f"{T}: {out_dir / tag / 'units'} に p5 の行が無い")
    return pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)


def pick5(g: pd.DataFrame, delta: float) -> tuple[float, bool, float]:
    """β の点ごとの平均 UAR が最大の点．最良との差が δ 以内の点が複数なら β = 0 に最も近い点（格子の番号）。
    戻り：(β, 同点の規則が働いたか, 選んだ点の平均)。"""
    m = g.groupby("beta")["uar"].mean()
    best = m.max()
    cands = [float(b) for b in m.index if m.loc[b] >= best - delta]
    grid = list(BETA_GRID)
    idx = lambda b: min(range(len(grid)), key=lambda i: abs(grid[i] - b))   # noqa: E731
    d0 = idx(DEFAULT_BETA)
    chosen = min(cands, key=lambda b: (abs(idx(b) - d0), idx(b)))
    return chosen, chosen != float(m.idxmax()), float(m.loc[chosen])


def commit_of(raw: pd.DataFrame) -> str:
    return str(raw.commit.astype(str).iloc[0])


# ------------------------------------------------------------------ 段 select

def select_stage(out_dir: Path, targets, fz1: dict) -> None:
    ev = out_dir / "eval" / "units"
    if ev.exists() and any(ev.iterdir()):
        sys.exit("評価の回の出力がすでにある．選ぶ段は評価の前にだけ流す")
    commit, dirty = dai1.git_commit()
    if dirty:
        sys.exit("選ぶ段は未コミットの変更があると流さない")
    sel_dir = out_dir / "select"
    sel_dir.mkdir(parents=True, exist_ok=True)
    frozen = {"_meta": {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit, "dirty": dirty,
                        "rule": "DAI/docs/実験5_設計.md 1 節（(T, K) ごとに 1 つの β．生徒 4 × 30 単位の平均．同点は β = 0 に近い点）",
                        "default_beta": DEFAULT_BETA, "beta_grid": list(BETA_GRID), "dai1_frozen_sha256": dai1.sha(FROZEN1)}}
    if list(targets) != list(DD.CORPORA):
        frozen["_meta"]["partial_targets"] = list(targets)
    gpus: set = set()
    for T in targets:
        raw = load_rows(out_dir, "pseudo", T)
        d = raw[raw.method == "srcw"]
        names = sorted({f"{c}_f{f}_r{r}" for c, f, r in d[["c", "fold", "seed"]].drop_duplicates().itertuples(index=False)})
        if names != sorted(fz1[T]["units"]):
            sys.exit(f"{T}: 単位が実験 1 の固定のファイルの units と違う")
        if len(names) != 30 and "partial_targets" not in frozen["_meta"]:
            sys.exit(f"{T}: 単位の数 {len(names)} が 30 でない")
        commits = set(raw.commit.astype(str))
        if len(commits) != 1:
            sys.exit(f"{T}: 行のコミットが 1 つでない {commits}")
        if (raw.dirty.astype(str) == "True").any() and "partial_targets" not in frozen["_meta"]:
            sys.exit(f"{T}: dirty の行がある")
        if set(raw.dai1_frozen_sha256.astype(str)) != {dai1.sha(FROZEN1)}:
            sys.exit(f"{T}: 行の dai1_frozen_sha256 が今の固定のファイルと違う")
        gpus |= set(raw.gpu.astype(str))
        if not set(d.beta.round(6)) <= set(BETA_GRID):
            sys.exit(f"{T}: 格子の外の β がある")
        for K in KS_ALL:
            nb = len(betas_for(K))
            g = d[d.K == K]
            if (g.groupby(["student", "beta"]).size() != len(names)).any() or g.groupby("student").ngroups != 4 or g.beta.nunique() != nb:
                sys.exit(f"{T}: K = {K} の β の点ごとの単位が {len(names)} でないか，生徒・β の数が違う")
        select, select_student, dbar, tie_used, edge = {}, {}, {}, 0, 0
        for K in KS_ALL:
            g = d[d.K == K]
            dmean = float(np.mean([fz1[T]["delta_sel"][f"{s}|{K}"] for s in DD.STUDENTS]))
            dbar[str(K)] = dmean
            b, used, mu = pick5(g, dmean)
            select[str(K)] = dict(beta=b, pseudo_uar=mu)
            tie_used += int(used)
            edge += int(K > 0 and b in (min(BETA_GRID), max(BETA_GRID)))
            for s in DD.STUDENTS:
                bs, _, mus = pick5(g[g.student == s], float(fz1[T]["delta_sel"][f"{s}|{K}"]))
                select_student[f"{s}|{K}"] = dict(beta=bs, pseudo_uar=mus)
        frozen[T] = dict(select=select, select_student=select_student, delta_bar=dbar, delta_sel=fz1[T]["delta_sel"], n_units=len(names),
                         tie_rule_used=tie_used, grid_edge_cells=edge, n_seeds_eval=fz1[T]["n_seeds_eval"], units=names,
                         pseudo_commit=commit_of(raw),
                         pseudo_unit_sha256={f"{T}_{n.split('_', 1)[0]}_{n.split('_', 1)[1]}_p5.npz":
                                             dai1.sha(out_dir / "pseudo" / "units" / f"{T}_{n}_p5.npz") for n in names})
        print(f"{T}: 単位 {len(names)}，同点の規則が働いた K {tie_used}，格子の端の K {edge}", flush=True)
    if len(gpus) != 1 and "partial_targets" not in frozen["_meta"]:
        sys.exit(f"pseudo の全単位の計算機名が 1 種でない：{sorted(gpus)}")
    frozen["_meta"]["pseudo_gpu"] = sorted(gpus)
    fzp = sel_dir / "frozen.json"
    json.dump(frozen, open(fzp, "w"), indent=1, ensure_ascii=False)
    print("書いた:", fzp, "sha256", dai1.sha(fzp))
    if out_dir.resolve().parent == (ROOT / "results").resolve():
        fzd = ROOT / "DAI" / "frozen"
        fzd.mkdir(exist_ok=True)
        shutil.copy(fzp, fzd / f"{out_dir.name}_frozen.json")
        print("写し:", fzd / f"{out_dir.name}_frozen.json")


# ------------------------------------------------------------------ 評価の段の関門

def check_eval_gate5(frozen_rel: str, out_dir: Path) -> str:
    """評価の段の守り（設計 4 節．実験 4 の `check_eval_gate4` と同じ形）．戻り：eval_gate_dai5.json の sha256。"""
    if not GATE5.exists() or not dai1.committed_clean(GATE5):
        sys.exit("DAI/frozen/eval_gate_dai5.json が無いかコミットされていない")
    gate = json.load(open(GATE5))
    req = gate["required"]
    missing = [r for r in EVAL_REQUIRED if r not in req]
    if missing:
        sys.exit(f"eval_gate_dai5.json に必須のファイルが無い：{missing}")
    for rel, h in req.items():
        fp = ROOT / rel
        if not fp.exists() or dai1.sha(fp) != h or not dai1.committed_clean(fp):
            sys.exit(f"eval_gate_dai5.json の {rel} が無いか，sha256 が違うか，コミットされていない")
    if set(gate.get("approvals", {})) != {"DAI-5"}:
        sys.exit("eval_gate_dai5.json の approvals は DAI-5 だけを持つ")
    files = gate["approvals"]["DAI-5"]
    reqset = {(ROOT / r).as_posix() for r in req}
    for role in dai1.APPROVAL_ROLES:
        if role not in files:
            sys.exit(f"DAI-5 の {role} の報告が approvals に無い")
        if not files[role].startswith(APPROVAL_FOLDER):
            sys.exit(f"DAI-5 の {role} の報告 {files[role]} が {APPROVAL_FOLDER} に無い")
        fp = ROOT / files[role]
        if fp.as_posix() not in reqset:
            sys.exit(f"DAI-5 の {role} の報告が required に無い")
        latest = dai1.latest_report(fp.parent, role)
        if latest != fp.name:
            sys.exit(f"DAI-5 の {role} は新しい版 {latest} がある（{fp.name} を並べている）")
        head = "\n".join(open(fp, encoding="utf-8").read().splitlines()[:12])
        word = "OK" if role.startswith("review_") else "通す"
        ok = rf"^判定(?:（DAI-5）)?[：:]\s*\**{word}(?:\**)?(?:$|[．。（])"
        sec = re.search(rf"^## 判定\s*\n+\s*\**{word}(?:$|[．。（])", head, re.M)
        if not re.search(ok, head, re.M) and not sec:
            sys.exit(f"DAI-5 の {role} の報告（{files[role]}）が「判定：{word}」でない")
    fz1 = json.load(open(FROZEN1))
    ish = gate.get("input_sha256")
    if not ish:
        sys.exit("eval_gate_dai5.json に input_sha256（実験 1 の評価の回の p1・p2・p3 の sha256）が無い")
    want = {f"{T}_ALL_f{f}_r{r}_p{w}.npz" for T in DD.CORPORA for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"])) for w in (1, 2, 3)}
    if set(ish) != want:
        sys.exit("input_sha256 の単位が 4 T × 50 単位 × p1・p2・p3 でない")
    for nm, h in ish.items():
        fp = IN_DIR / "eval" / "units" / nm
        if not fp.exists() or dai1.sha(fp) != h:
            sys.exit(f"実験 1 の評価の回の保存 {nm} が無いか，記録の sha256 と違う")
    fzj = json.load(open(out_dir / "select" / "frozen.json"))
    for T in DD.CORPORA:
        pc = fzj[T]["pseudo_commit"]
        if subprocess.run(["git", "diff", "--quiet", pc, "HEAD", "--", *CODE_FIXED], cwd=ROOT).returncode != 0:
            sys.exit(f"{T}: 固定のファイルの pseudo_commit {pc[:10]} と HEAD の間で {CODE_FIXED} に差がある")
        for nm, h in fzj[T]["pseudo_unit_sha256"].items():
            fp = out_dir / "pseudo" / "units" / nm
            if not fp.exists() or dai1.sha(fp) != h:
                sys.exit(f"{T}: pseudo の保存 {nm} が無いか，固定のファイルの sha256 と違う")
    ct = int(subprocess.run(["git", "log", "-1", "--format=%ct", "--", frozen_rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() or 0)
    sv = [p.stat().st_mtime for p in (out_dir / "eval" / "units").glob("*_p5.npz")] if (out_dir / "eval" / "units").exists() else []
    if not ct or (sv and min(sv) < ct):
        sys.exit("dai5_frozen.json のコミット時刻が無いか，評価の段の最初の保存より後である")
    return dai1.sha(GATE5)


# ------------------------------------------------------------------ 進め方

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("pseudo", "select", "detect", "eval"))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--in-dir", default=str(IN_DIR))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(DD.N_FOLDS)))
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="既定は pseudo で 0 1，eval で実験 1 の固定のファイルの n_seeds_eval")
    ap.add_argument("--cs", nargs="*", default=None, help="pseudo の仮のターゲット c を絞る（試走だけ）")
    ap.add_argument("--allow-dirty", action="store_true", help="試走だけ（pseudo のみ）")
    ap.add_argument("--selftest-rho0", action="store_true", help="ρ = 0 の問を足して実験 1 の none と照合する（試走だけ．本番の置き場以外）")
    args = ap.parse_args()
    out_dir, in_dir = Path(args.out_dir), Path(args.in_dir)
    fz1 = json.load(open(FROZEN1))
    if args.selftest_rho0 and (not args.allow_dirty or out_dir.resolve() == OUT_DIR.resolve() or args.stage != "pseudo"):
        sys.exit("--selftest-rho0 は --allow-dirty の pseudo で本番の置き場以外にだけ使える")
    if args.stage == "select":
        select_stage(out_dir, args.targets, fz1); return
    if args.stage == "detect":
        import dai5_stats
        dai5_stats.detect(out_dir, in_dir, args.targets); return
    commit, dirty = dai1.git_commit()
    if args.stage == "eval" and dirty:
        sys.exit("評価の段は未コミットの変更があると流さない（--allow-dirty も受け付けない）")
    if dirty and not args.allow_dirty:
        sys.exit("DAI/exp・measure・features に未コミットの変更がある（試しは --allow-dirty）")
    run = dict(commit=commit, dirty=dirty, frozen_sha256="", dai1_frozen_sha256=dai1.sha(FROZEN1), gate_sha256="",
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    dai1.RUN["commit"], dai1.RUN["dirty"] = commit, dirty
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, host=platform.node(), gpu=run["gpu"], torch=torch.__version__,
                script_sha256=dai1.sha(Path(__file__)), time=time.strftime("%Y-%m-%d %H:%M:%S"), dai1_frozen_sha256=run["dai1_frozen_sha256"],
                beta_grid=list(BETA_GRID))
    if args.stage == "pseudo":
        for T in args.targets:
            tw = json.load(open(in_dir / "pseudo" / f"{T}_tether_w.json"))
            if tw["tether"] != fz1[T]["tether"]:
                sys.exit(f"{T}: tether_w.json の引き戻しが実験 1 の固定のファイルと違う")
        seeds = args.seeds if args.seeds is not None else [0, 1]
    else:
        fzp = out_dir / "select" / "frozen.json"
        cp = ROOT / "DAI" / "frozen" / f"{out_dir.name}_frozen.json"
        if not fzp.exists() or not cp.exists() or dai1.sha(fzp) != dai1.sha(cp):
            sys.exit("固定のファイルが無いか，DAI/frozen/dai5_frozen.json と一致しない")
        if not dai1.committed_clean(cp):
            sys.exit("固定のファイルの写しがコミットされていない")
        fzj = json.load(open(fzp))
        if any("detect" not in fzj.get(T, {}) for T in DD.CORPORA):
            sys.exit("固定のファイルに detect の出力が無い（dai5_stats の detect を先に流す）")
        gate_sha = check_eval_gate5(f"DAI/frozen/{out_dir.name}_frozen.json", out_dir)
        run["gate_sha256"], run["frozen_sha256"] = gate_sha, dai1.sha(fzp)
        prov.update(eval_gate_sha256=gate_sha, frozen_sha256=run["frozen_sha256"])
        seeds = None
    with open(out_dir / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    n = 0
    for T in args.targets:
        tether = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in fz1[T]["tether"].items()}
        sd = seeds if seeds is not None else (args.seeds if args.seeds is not None else list(range(int(fz1[T]["n_seeds_eval"]))))
        for c, srcs, f, r in dai1.units_for(T, args.stage, args.folds, sd):
            if args.cs is not None and c not in args.cs:
                continue
            run_unit(in_dir, out_dir, args.stage, T, c, srcs, f, r, tether, fz1[T], run, args.selftest_rho0)
            n += 1
        dai1.Corpus.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"{args.stage}: {n} 単位", flush=True)


if __name__ == "__main__":
    main()
