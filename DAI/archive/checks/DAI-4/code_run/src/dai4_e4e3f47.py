"""DAI 実験 4：信頼度で重みづけた別のモダリティの疑似ラベル `cons`（設計 `DAI/docs/実験4_設計.md` 第 1 版）。

実験 1 の保存（仮のターゲットの回と評価の回の段 1・段 2）を入力にして，生徒の学習（実験 1 の段 3 に相当）だけを新しく解く。
段 1・段 2 は流し直さない。DAI/exp のほかの台本は import して使うだけで，書き換えない。

段（--stage）：
  pseudo ：設計 1・4 節。仮のターゲットの回の 120 単位で，生徒 4 × K 5 × 格子 12 点（ρ ∈ {1/16, 1/4, 1, 4} × γ ∈ {0, 1/4, 1/2}）を解く。
  select ：設計 4 節。実験 1 の 8 節の規則（`dai1_select.pick` と同じ．δ_sel は実験 1 の固定のファイルの値）で (T, 生徒, K) ごとに (ρ, γ) を選び，
           `results/dai4/select/frozen.json` と写し `DAI/frozen/dai4_frozen.json` に書く。
  detect ：設計 4・5 節。`dai4_stats.py` の detect（族 6 の検出できる差と記述に回すセル）を呼ぶ。
  eval   ：設計 4 節。評価の回の単位（200 単位）で同じ格子を解く。条件は `check_eval_gate4`（下）。
  集計（stats）は別の台本 `dai4_stats.py`。

実装の対応（設計の節 → このファイル）：
  1 節 教師 Q       ：保存した `p2_K{K}_ret_{m}_W・_b`（float32）を候補の標準化した特徴に当て，`dai1.product(tk, (1,1,1), 0)`（確率の積の softmax．
                      実験 1 の `teacher = one`）で作る。K = 0 も保存の K = 0 の ret（出発点の識別器から作った）を使う。
  1 節 信頼の印 c_i ：生徒 s 以外のモダリティの ret の判定（argmax）がすべて一致すれば 1。A→(V,T)，V→(A,T)，T→(A,V)，AVT→(A,V,T) の 3 つ。
  1 節 重み・格子   ：m_i = 1（印あり）か γ（印なし）。G = a·gk + `soft_rows(diag(m) Q, ρ n, balance=True)`，a = 1（K 本の重みは `gK`）。
                      γ = 1 は実験 1 の同時に解く版（a = 1）に一致する（`--selftest-g1` の試走で検算する。本番の格子には入れない）。
  1 節 学習器       ：`dai1.solve_items`（`dai_solver.fit`）を使い回す。引き戻し (C′, C_b) は実験 1 の固定のファイルの tether（K = 0 は K = 1 の値．
                      実験 1 と同じ），出発点は Unit の start。
  4 節 保存         ：`results/dai4/<pseudo|eval>/units/<T>_<c|ALL>_f<f>_r<r>_p4.npz/.csv`。
                      p4_{s}_K{K}_pred（格子の全点のテストの判定．int8），_W・_b（float32），_items（json．rho・gamma），
                      p4_K{K}_teacher（候補の教師の判定．int8），p4_K{K}_{s}_mark（印．int8），
                      p4_K{K}_{s}_counts（[候補の本数 n，印ありの数，K 本を除いた候補の数，その印ありの数]）。
                      sample_pool・sample_K{K} を写す（集計がラベルを引いて印の有無ごとの教師の正解率を数えるため。学習にラベルは使わない）。
                      rows.csv：method=cons の行（student，K，rho，gamma，uar，grad，stalled，finite，S），`_conv` と `_marks` の行。
                      コミット，dirty，固定のファイルの sha256（frozen_sha256：評価の回は dai4 の固定のファイル，仮のターゲットの回は空。
                      dai1_frozen_sha256：実験 1 の固定のファイル）を持つ。

実装で決めた細部：
  - 対照 4 つ（設計第 2 版の M2・比べ方の関門）を，本番と同じ ρ 4 × γ 3 の格子の全点で，仮のターゲットの回にも評価の回にも解く：
    cons_self（印＝生徒を含む全モダリティの ret の判定の一致．AVT の生徒は本番と同じなので省く），cons_own（印＝生徒自身の none[K] の判定と教師 Q の
    判定の一致．K = 0 は出発点），cons_conf（印＝教師 Q の最大の確率が上位の発話．数は本番の印と同じ．同点は番号の若い方），cons_rand（印＝本番と同じ数の乱数．
    種は crc32("T|c|f|r|生徒|K")）。保存は接頭辞 p4s・p4o・p4c・p4r（本番は p4）で，形は本番と同じ。評価の集計は，本番の選んだ (ρ, γ) の点で比べる。
  - 行 `_weights`：感情ごとの重みの合計と有効な本数 (Σw)²/Σw²，重みが 0 の感情の数（γ = 0 で印の付いた発話が無い感情）。止めずに残す。
    行 `_marks`：印の数，感情（教師の判定）ごとの印の数と割合，印の付いた発話が 0 の感情の数。`_marks_ctrl`：対照の印の数と本番の印との一致の割合。
  - 行 `_teacher_check`：作り直した教師の候補の UAR（K 本を除く）が実験 1 の段 3 の `_teacher_stats`（teacher = one）の uar_pool と一致するか。
  - `--selftest-g1`：γ = 1 の点も解き，実験 1 の p3 の joint（a = 1，同じ ρ，teacher = one）との判定の一致と W・b の差を `_g1check` の行に残す
    （本番の格子には入れない．--allow-dirty の pseudo で本番の置き場以外だけ）。
  - 計算機の名前を行（gpu）と npz（gpu）に残す。問の名前の JSON（_items）に method・rho・gamma を入れる。
  - 入力の単位が実験 1 の保存と同じ分割・K 本・標準化かを，sample_pool・sample_test・sample_K{K}・標準化の統計で照合し，違えば止まる。
  - tether は実験 1 の固定のファイル（`DAI/frozen/dai1_frozen.json`）から読む。仮のターゲットの回では `<T>_tether_w.json` と一致を確かめる。
  - 印の定義は ret の判定だけを使う（生徒自身のモダリティの判定は使わない）。印の有無ごとの数えは K 本を除いた候補で行う。
  - select の δ_sel は実験 1 の固定のファイルの `delta_sel`（n = 30 単位）。同点の規則は `pick4`（実験 1 の `pick` と同じ：最良との差が δ 以内の点のうち，
    選択なしの値（ρ = 1，γ = 1/4）に最も近い点．格子の番号の差の和，それも同じなら番号の並び）。
  - 選択なしの値は ρ = 1，γ = 1/4。格子の端は ρ の 1/16 と 4，γ の 0 と 1/2。
  - 評価の段の関門は `check_eval_gate4`：`DAI/frozen/eval_gate_dai4.json` の required に EVAL_REQUIRED（と固定のファイル）が並び，すべてコミット済みで
    sha256 が一致し，`DAI/checks/DAI-4/` の 7 役の最新の版が OK（査読）・通す（関門）で，最新の版を並べていること。
    EVAL_REQUIRED には，設計の指定（設計，dai4.py，dai4_stats.py，dai1_frozen.json，dai4_frozen.json）に，結果が依る import 先の台本
    （dai1.py，dai_solver.py，dai_data.py，dai1_select.py，dai1_stats.py）と実験 1 の `eval_gate.json` を足した。
  - どの段も成績の値を画面に出さない（行の数・件数・収束・所要だけ）。
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import sys
import time
import zlib
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
RHO_GRID = dai1.RHO_GRID                      # (1/16, 1/4, 1, 4)
GAMMA_GRID = (0.0, 0.25, 0.5)
DEFAULT = dict(rho=1.0, gamma=0.25)            # 選択なしの値
KS_ALL = dai1.KS_ALL
IN_DIR = ROOT / "results" / "dai1"
OUT_DIR = ROOT / "results" / "dai4"
FROZEN1 = ROOT / "DAI" / "frozen" / "dai1_frozen.json"
GATE4 = ROOT / "DAI" / "frozen" / "eval_gate_dai4.json"

EVAL_REQUIRED = [
    "DAI/docs/実験4_設計.md", "DAI/exp/dai4.py", "DAI/exp/dai4_stats.py",
    "DAI/frozen/dai1_frozen.json", "DAI/frozen/dai4_frozen.json",
    "DAI/exp/dai1.py", "DAI/exp/dai_solver.py", "DAI/exp/dai_data.py", "DAI/exp/dai1_select.py", "DAI/exp/dai1_stats.py",
    "DAI/frozen/eval_gate.json",
]
CTRL_PREFIX = {"cons_self": "p4s", "cons_own": "p4o", "cons_conf": "p4c", "cons_rand": "p4r"}   # 対照の保存の接頭辞
APPROVAL_FOLDER = "DAI/checks/DAI-4/"


def others_of(s: str) -> str:
    return {"A": "VT", "V": "AT", "T": "AV", "AVT": "AVT"}[s]


def grid_points(gammas) -> list[dict]:
    return [dict(method="cons", rho=float(r), gamma=float(g)) for r in RHO_GRID for g in gammas]


# ------------------------------------------------------------------ 1 単位

def mark_and_teacher(u: dai1.Unit, z2, K: int):
    """教師 Q（候補の確率の積の softmax）と，モダリティごとの判定。"""
    tk, am = {}, {}
    for m in "AVT":
        W, b = S.t(z2[f"p2_K{K}_ret_{m}_W"]), S.t(z2[f"p2_K{K}_ret_{m}_b"])
        P = dai1.probs(u.Xp[m], W, b)
        tk[f"ret_{m}"] = (P, P)
        am[m] = P.argmax(1)
    return dai1.product(tk, (1, 1, 1), 0), am


def build_specs(u, s: str, K: int, mark, Q, gk, method: str, gammas, rows: list) -> list:
    """(ρ, γ) の格子の問を作る。印なしの確率に γ を掛け，感情ごとの総量をそろえて ρ n に（`soft_rows`）。
    あわせて感情ごとの重みの合計と有効な本数（実験 1 の D11 と同じ式）を `_weights` の行に残す。"""
    zero = torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
    specs = []
    for rho in RHO_GRID:
        for gam in gammas:
            m = torch.where(mark, torch.ones((), device=S.DEV, dtype=S.DT), torch.full((), float(gam), device=S.DEV, dtype=S.DT))
            P = S.soft_rows(Q * m[:, None], dai1.mval(rho, u.n), balance=True)
            cs = P.sum(0).cpu().numpy()
            neff = ((P.sum(0) ** 2) / (P ** 2).sum(0).clamp(min=1e-30)).cpu().numpy()
            rows.append(dict(stage=3, method="_weights", variant=method, student=s, K=K, rho=float(rho), gamma=float(gam),
                             n_zero_weight_classes=int((P.sum(0) < 1e-12).sum()),
                             **{f"colsum_{k}": float(cs[k]) for k in range(4)}, **{f"neff_{k}": float(neff[k]) for k in range(4)}))
            specs.append((dict(method=method, rho=float(rho), gamma=float(gam)), gk + P, zero, 1.0))
    return specs


def seed_of(u, s: str, K: int) -> int:
    return zlib.crc32(f"{u.T}|{u.c}|{u.f}|{u.r}|{s}|{K}".encode())


def solve_unit(u: dai1.Unit, tether: dict, gammas, store: dict, rows: list, z1, z3=None, p3rows=None) -> None:
    """対照 4 つ（cons_self・cons_own・cons_conf・cons_rand）も本番と同じ格子の全点で解く（仮のターゲットの回も評価の回も）。z1 は実験 1 の段 1 の保存。z3 と p3rows（実験 1 の p3 の保存と行）があれば教師の検算をする。"""
    z2 = u.z2
    for K in KS_ALL:
        gk = u.gK(K) if K else torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
        Q, am = mark_and_teacher(u, z2, K)
        nk = np.setdiff1d(np.arange(u.n), u.kpos[K])
        tpred = Q.argmax(1)
        store[f"p4_K{K}_teacher"] = tpred.to(torch.int8).cpu().numpy()
        # 教師の検算：作り直した教師の候補の UAR（K 本を除く）が実験 1 の段 3 の `_teacher_stats`（teacher = one）の uar_pool と一致するか
        if p3rows is not None:
            mine = S.uar(u.ypool[nk], tpred.cpu().numpy()[nk]) * 100
            ref = p3rows[(p3rows.method == "_teacher_stats") & (p3rows.teacher == "one") & (p3rows.K == K)]
            theirs = float(ref.uar_pool.iloc[0]) if len(ref) == 1 else float("nan")
            rows.append(dict(stage=3, method="_teacher_check", K=K, uar_pool_mine=mine, uar_pool_dai1=theirs,
                             match=bool(np.isfinite(theirs) and abs(mine - theirs) < 1e-6)))
        allm = torch.ones(u.n, dtype=torch.bool, device=S.DEV)
        for o in "AVT":
            allm &= am["A"] == am[o]
        for s in DD.STUDENTS:
            oth = others_of(s)
            mark = torch.ones(u.n, dtype=torch.bool, device=S.DEV)
            for o in oth[1:]:
                mark &= am[oth[0]] == am[o]
            mk = mark.cpu().numpy()
            store[f"p4_K{K}_{s}_mark"] = mk.astype(np.int8)
            store[f"p4_K{K}_{s}_counts"] = np.array([u.n, int(mk.sum()), len(nk), int(mk[nk].sum())])
            tp = tpred.cpu().numpy()
            cls_marked = [int((mk & (tp == k)).sum()) for k in range(4)]          # 感情（教師の判定）ごとの印の付いた発話の数
            rows.append(dict(stage=3, method="_marks", student=s, K=K, n_pool=u.n, n_pool_marked=int(mk.sum()),
                             n_cand=len(nk), n_cand_marked=int(mk[nk].sum()), zero_marked_classes=int(sum(c == 0 for c in cls_marked)),
                             **{f"marked_cls{k}": cls_marked[k] for k in range(4)},
                             **{f"frac_cls{k}": cls_marked[k] / max(1, int((tp == k).sum())) for k in range(4)}))
            W0, b0 = u.start[s]
            cw, cb = tether[(s, max(K, 1))]
            specs = build_specs(u, s, K, mark, Q, gk, "cons", gammas, rows)
            dai1.solve_items(u, s, K, specs, cw, cb, W0, b0, rows, store, f"p4_{s}_K{K}")
            ctrl = {}
            if s != "AVT":                                                          # (a) 生徒を含む全モダリティの一致（AVT の生徒は本番と同じなので省く）
                ctrl["cons_self"] = ("self", allm)
            Wn, bn = dai1.none_wb(u, z1, s, K, tether)                              # (b) 生徒自身の none[K] の判定と教師 Q の判定の一致
            ctrl["cons_own"] = ("own", dai1.probs(u.Xp[s], Wn, bn).argmax(1) == tpred)
            order = np.argsort(-Q.max(1).values.cpu().numpy(), kind="stable")       # (c) 教師 Q の最大の確率が上位の発話（本番の印と同じ数．同点は番号の若い方）
            mc = np.zeros(u.n, bool)
            mc[order[:int(mk.sum())]] = True
            ctrl["cons_conf"] = ("conf", torch.as_tensor(mc, device=S.DEV))
            rng = np.random.default_rng(seed_of(u, s, K))                          # (d) 本番の印と同じ割合の乱数の印
            mr = np.zeros(u.n, bool)
            mr[rng.permutation(u.n)[:int(mk.sum())]] = True
            ctrl["cons_rand"] = ("rand", torch.as_tensor(mr, device=S.DEV))
            for meth, (tag_, mkc) in ctrl.items():
                mkn = mkc.cpu().numpy().astype(bool)
                store[f"p4_K{K}_{s}_mark{tag_}"] = mkn.astype(np.int8)
                rows.append(dict(stage=3, method="_marks_ctrl", variant=meth, student=s, K=K, n_pool_marked=int(mkn.sum()),
                                 n_cand_marked=int(mkn[nk].sum()), agree_with_cons=float((mkn == mk).mean())))
                sp = build_specs(u, s, K, mkc, Q, gk, meth, gammas, rows)
                dai1.solve_items(u, s, K, sp, cw, cb, W0, b0, rows, store, f"p4{tag_[0]}_{s}_K{K}" if tag_ != "own" else f"p4o_{s}_K{K}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if z3 is not None:                                                                  # 式の検算（--selftest-g1）：γ = 1 は実験 1 の joint（a = 1，teacher = one）
        for K in KS_ALL:
            for s in DD.STUDENTS:
                it3 = [json.loads(x) for x in z3[f"p3_{s}_K{K}_items"]]
                it4 = [json.loads(x) for x in store[f"p4_{s}_K{K}_items"]]
                for j4, it in enumerate(it4):
                    if not np.isclose(it["gamma"], 1.0):
                        continue
                    j3 = [j for j, x in enumerate(it3) if x.get("method") == "joint" and x.get("teacher") == "one"
                          and np.isclose(float(x["rho"]), it["rho"]) and float(x["a"]) == 1.0]
                    if len(j3) != 1:
                        sys.exit("実験 1 の p3 に対応する joint の点が 1 つでない")
                    j3 = j3[0]
                    rows.append(dict(stage=3, method="_g1check", student=s, K=K, rho=it["rho"],
                                     pred_agree=float((store[f"p4_{s}_K{K}_pred"][j4] == z3[f"p3_{s}_K{K}_pred"][j3]).mean()),
                                     W_maxdiff=float(np.abs(store[f"p4_{s}_K{K}_W"][j4] - z3[f"p3_{s}_K{K}_W"][j3]).max()),
                                     b_maxdiff=float(np.abs(store[f"p4_{s}_K{K}_b"][j4] - z3[f"p3_{s}_K{K}_b"][j3]).max())))


def check_inputs(u: dai1.Unit, z1) -> None:
    ids = u.cp.df["sample_id"].to_numpy().astype(str)
    ok = (ids[u.pool] == z1["sample_pool"]).all() and (ids[u.test] == z1["sample_test"]).all()
    ok = ok and all((ids[u.pool[u.kpos[K]]] == z1[f"sample_K{K}"]).all() for K in DD.KS)
    ok = ok and all(np.allclose(u.stats[k][0], z1[f"stat_{k}_mu"], rtol=1e-4, atol=1e-6) and
                    np.allclose(u.stats[k][1], z1[f"stat_{k}_sd"], rtol=1e-4, atol=1e-6) for k in DD.KINDS)
    if not ok:
        sys.exit(f"{u.name()}: 実験 1 の保存の分割・K 本・標準化の統計が今の Unit と違う")


def run_unit(in_dir: Path, out_dir: Path, tag: str, T: str, c: str, srcs: list, f: int, r: int, tether: dict, gammas,
             run: dict, selftest_g1: bool = False) -> pd.DataFrame:
    cname = "ALL" if tag == "eval" else c
    name = f"{T}_{cname}_f{f}_r{r}"
    ib = in_dir / tag / "units" / name
    base = out_dir / tag / "units" / name
    path, rpath = Path(f"{base}_p4.npz"), Path(f"{base}_p4.csv")
    grid = np.array([f"{it['rho']}|{it['gamma']}" for it in grid_points(gammas)])
    if path.exists() and rpath.exists():
        old = pd.read_csv(rpath, low_memory=False)
        z = np.load(path)
        bad = str(old["commit"].iloc[0]) != run["commit"] or str(old["frozen_sha256"].iloc[0] if "frozen_sha256" in old and
                                                                 pd.notna(old["frozen_sha256"].iloc[0]) else "") != run["frozen_sha256"]
        bad |= str(z["dai1_frozen_sha256"]) != run["dai1_frozen_sha256"] or list(z["grid"]) != list(grid)
        bad |= "finite" in old and (old["finite"].astype(str) == "False").any()
        if bad:
            sys.exit(f"保存済みの単位 {rpath.name} のコミット・固定のファイル・格子・有限のどれかが今の実行と違う（消してから流し直す）")
        return old
    for w in (1, 2):
        for ext in ("npz", "csv"):
            if not Path(f"{ib}_p{w}.{ext}").exists():
                sys.exit(f"入力の保存が無い：{ib.name}_p{w}.{ext}")
    z1, z2 = np.load(f"{ib}_p1.npz"), np.load(f"{ib}_p2.npz")
    if tag == "eval" and str(z1["frozen_sha256"]) != run["dai1_frozen_sha256"]:
        sys.exit(f"{ib.name} の実験 1 の保存の frozen_sha256 が dai1_frozen.json と違う")
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    u = dai1.Unit(in_dir, T, c, srcs, f, r)
    check_inputs(u, z1)
    u.z2 = z2
    z3 = p3rows = None
    if selftest_g1:                                                      # 式の検算：実験 1 の p3 が要る
        z3, p3rows = np.load(f"{ib}_p3.npz"), pd.read_csv(f"{ib}_p3.csv", low_memory=False)
    elif Path(f"{ib}_p3.csv").exists():
        p3rows = pd.read_csv(f"{ib}_p3.csv", low_memory=False)           # 教師の検算だけ
    store, rows = {}, []
    solve_unit(u, tether, gammas, store, rows, z1, z3, p3rows)
    store.update(sample_pool=z1["sample_pool"], grid=grid, commit=np.array(run["commit"]), dirty=np.array(run["dirty"]),
                 frozen_sha256=np.array(run["frozen_sha256"]), dai1_frozen_sha256=np.array(run["dai1_frozen_sha256"]),
                 gate_sha256=np.array(run["gate_sha256"]), gpu=np.array(run["gpu"]),
                 **{f"sample_K{K}": z1[f"sample_K{K}"] for K in DD.KS}, sample_test=z1["sample_test"])
    if any(("finite" in x and x["finite"] is False) for x in rows):
        sys.exit(f"{rpath.name}: 勾配か解に NaN・Inf の問がある（計算の失敗として止める．保存しない）")
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in store.items()})
    df = pd.DataFrame(rows).assign(T=T, c=cname, fold=f, seed=r, sec=time.time() - t0, commit=run["commit"], dirty=run["dirty"],
                                   frozen_sha256=run["frozen_sha256"], dai1_frozen_sha256=run["dai1_frozen_sha256"], gpu=run["gpu"],
                                   gate_sha256=run["gate_sha256"])
    df.to_csv(rpath, index=False)
    cv = df[df.method == "cons"]
    nconv = int(((cv.grad <= 1e-5) & ~cv.stalled.astype(bool)).sum())
    print(f"{tag} {name} p4: 問 {len(cv)}，収束 {nconv}，{time.time()-t0:.0f} 秒", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


# ------------------------------------------------------------------ 保存の読み出し（段 select・stats・detect が使う）

class P4:
    """1 単位の p4 の保存。"""

    def __init__(self, base: Path):
        self.z = np.load(f"{base}_p4.npz")
        self._items = {}

    def items(self, s: str, K: int, prefix: str = "p4") -> list:
        k = (s, K, prefix)
        if k not in self._items:
            self._items[k] = [json.loads(x) for x in self.z[f"{prefix}_{s}_K{K}_items"]]
        return self._items[k]

    def has(self, s: str, K: int, prefix: str) -> bool:
        return f"{prefix}_{s}_K{K}_items" in self.z

    def pred(self, s: str, K: int, rho, gamma, prefix: str = "p4") -> np.ndarray:
        hits = [j for j, it in enumerate(self.items(s, K, prefix)) if np.isclose(it["rho"], float(rho)) and np.isclose(it["gamma"], float(gamma))]
        if len(hits) != 1:
            raise KeyError((s, K, rho, gamma, len(hits)))
        return self.z[f"{prefix}_{s}_K{K}_pred"][hits[0]]


def load_rows(out_dir: Path, tag: str, T: str) -> pd.DataFrame:
    fs = sorted((out_dir / tag / "units").glob(f"{T}_*_p4.csv"))
    if not fs:
        sys.exit(f"{T}: {out_dir / tag / 'units'} に p4 の行が無い")
    return pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)


def pick4(g: pd.DataFrame, delta: float) -> tuple[dict, bool]:
    """(ρ, γ) の格子の点ごとの平均 UAR が最大の点．最良との差が δ 以内の点が複数なら，選択なしの値に最も近い点
    （各つまみの格子の番号の差の和．それも同じなら番号の並びで先）。`dai1_select.pick` と同じ規則。"""
    grid = {"rho": list(RHO_GRID), "gamma": list(GAMMA_GRID)}
    keys = ["rho", "gamma"]
    m = g.groupby(keys)["uar"].mean()
    best = m.max()
    cands = list(m[m >= best - delta].index)

    def idx(k, v):
        return min(range(len(grid[k])), key=lambda i: abs(grid[k][i] - float(v)))
    dflt = [idx(k, DEFAULT[k]) for k in keys]
    chosen = min(cands, key=lambda ix: (sum(abs(idx(k, v) - d) for k, v, d in zip(keys, ix, dflt)), [idx(k, v) for k, v in zip(keys, ix)]))
    am = m.idxmax()
    return dict(rho=float(chosen[0]), gamma=float(chosen[1]), pseudo_uar=float(m.loc[chosen])), chosen != am


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
                        "rule": "DAI/docs/実験4_設計.md 4 節（実験 1 の 8 節と同じ pick）", "default": DEFAULT,
                        "dai1_frozen_sha256": dai1.sha(FROZEN1)}}
    if list(targets) != list(DD.CORPORA):
        frozen["_meta"]["partial_targets"] = list(targets)
    for T in targets:
        d = load_rows(out_dir, "pseudo", T)
        d = d[d.method == "cons"]
        names = sorted({f"{c}_f{f}_r{r}" for c, f, r in d[["c", "fold", "seed"]].drop_duplicates().itertuples(index=False)})
        if names != sorted(fz1[T]["units"]):
            sys.exit(f"{T}: 単位が実験 1 の固定のファイルの units と違う")
        if len(names) != 30 and "partial_targets" not in frozen["_meta"]:
            sys.exit(f"{T}: 単位の数 {len(names)} が 30 でない")
        raw = load_rows(out_dir, "pseudo", T)
        commits = set(raw.commit.astype(str))
        if len(commits) != 1:
            sys.exit(f"{T}: 行のコミットが 1 つでない {commits}")
        if (raw.dirty.astype(str) == "True").any() and "partial_targets" not in frozen["_meta"]:
            sys.exit(f"{T}: dirty の行がある")
        if set(raw.dai1_frozen_sha256.astype(str)) != {dai1.sha(FROZEN1)}:
            sys.exit(f"{T}: 行の dai1_frozen_sha256 が今の固定のファイルと違う")
        if not set(d.gamma.round(6)) <= set(GAMMA_GRID) or not set(d.rho.round(6)) <= set(RHO_GRID):
            sys.exit(f"{T}: 格子の外の点がある（γ = 1 の検算の出力を混ぜない）")
        if (d.groupby(["student", "K", "rho", "gamma"]).size() != 30).any():
            sys.exit(f"{T}: 格子の点ごとの単位が 30 でない")
        select, tie_used, edge = {}, 0, dict(rho=0, gamma=0, any=0)
        for (s, K), g in d.groupby(["student", "K"]):
            ch, used = pick4(g, float(fz1[T]["delta_sel"][f"{s}|{int(K)}"]))
            select[f"{s}|{int(K)}"] = ch
            tie_used += int(used)
            er, eg = ch["rho"] in (min(RHO_GRID), max(RHO_GRID)), ch["gamma"] in (min(GAMMA_GRID), max(GAMMA_GRID))
            edge["rho"] += int(er); edge["gamma"] += int(eg); edge["any"] += int(er or eg)
        frozen[T] = dict(select=select, delta_sel=fz1[T]["delta_sel"], n_units=len(names), tie_rule_used=tie_used, grid_edge_cells=edge,
                         n_cells=len(select), n_seeds_eval=fz1[T]["n_seeds_eval"], units=names, pseudo_commit=commit_of(raw))
        print(f"{T}: 単位 {len(names)}，セル {len(select)}，同点の規則が働いたセル {tie_used}，格子の端のセル {edge}", flush=True)
    fzp = sel_dir / "frozen.json"
    json.dump(frozen, open(fzp, "w"), indent=1, ensure_ascii=False)
    print("書いた:", fzp, "sha256", dai1.sha(fzp))
    if out_dir.resolve().parent == (ROOT / "results").resolve():
        fzd = ROOT / "DAI" / "frozen"
        fzd.mkdir(exist_ok=True)
        shutil.copy(fzp, fzd / f"{out_dir.name}_frozen.json")
        print("写し:", fzd / f"{out_dir.name}_frozen.json")


def commit_of(raw: pd.DataFrame) -> str:
    return str(raw.commit.astype(str).iloc[0])


# ------------------------------------------------------------------ 評価の段の関門

def check_eval_gate4(frozen_rel: str) -> str:
    """評価の段を流す条件：`DAI/frozen/eval_gate_dai4.json` がコミット済みで，required に EVAL_REQUIRED と固定のファイルが並び，
    並べたファイルがすべてコミット済みで sha256 が一致し，`DAI/checks/DAI-4/` の 7 役の最新の版が OK（査読）・通す（関門）であること。
    戻り：eval_gate_dai4.json の sha256。"""
    if not GATE4.exists() or not dai1.committed_clean(GATE4):
        sys.exit("DAI/frozen/eval_gate_dai4.json が無いかコミットされていない")
    gate = json.load(open(GATE4))
    req = gate["required"]
    missing = [r for r in EVAL_REQUIRED + [frozen_rel] if r not in req]
    if missing:
        sys.exit(f"eval_gate_dai4.json に必須のファイルが無い：{missing}（必須の一覧は台本の EVAL_REQUIRED）")
    for rel, h in req.items():
        fp = ROOT / rel
        if not fp.exists() or dai1.sha(fp) != h or not dai1.committed_clean(fp):
            sys.exit(f"eval_gate_dai4.json の {rel} が無いか，sha256 が違うか，コミットされていない")
    if set(gate.get("approvals", {})) != {"DAI-4"}:
        sys.exit("eval_gate_dai4.json の approvals は DAI-4 だけを持つ")
    files = gate["approvals"]["DAI-4"]
    reqset = {(ROOT / r).as_posix() for r in req}
    for role in dai1.APPROVAL_ROLES:
        if role not in files:
            sys.exit(f"DAI-4 の {role} の報告が approvals に無い")
        if not files[role].startswith(APPROVAL_FOLDER):
            sys.exit(f"DAI-4 の {role} の報告 {files[role]} が {APPROVAL_FOLDER} に無い")
        fp = ROOT / files[role]
        if fp.as_posix() not in reqset:
            sys.exit(f"DAI-4 の {role} の報告が required に無い")
        latest = dai1.latest_report(fp.parent, role)
        if latest != fp.name:
            sys.exit(f"DAI-4 の {role} は新しい版 {latest} がある（{fp.name} を並べている）")
        head = "\n".join(open(fp, encoding="utf-8").read().splitlines()[:12])
        word = "OK" if role.startswith("review_") else "通す"
        ok = rf"^判定(?:（DAI-4）)?[：:]\s*\**{word}(?:\**)?(?:$|[．。（])"
        sec = re.search(rf"^## 判定\s*\n+\s*\**{word}(?:$|[．。（])", head, re.M)
        if not re.search(ok, head, re.M) and not sec:
            sys.exit(f"DAI-4 の {role} の報告（{files[role]}）が「判定：{word}」でない")
    return dai1.sha(GATE4)


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
    ap.add_argument("--selftest-g1", action="store_true", help="γ = 1 の点を足す（実験 1 の joint との式の検算．試走だけ）")
    args = ap.parse_args()
    out_dir, in_dir = Path(args.out_dir), Path(args.in_dir)
    fz1 = json.load(open(FROZEN1))
    if args.selftest_g1 and (not args.allow_dirty or out_dir.resolve() == OUT_DIR.resolve() or args.stage != "pseudo"):
        sys.exit("--selftest-g1 は --allow-dirty の pseudo で本番の置き場以外にだけ使える")
    if args.stage == "select":
        select_stage(out_dir, args.targets, fz1); return
    if args.stage == "detect":
        import dai4_stats
        dai4_stats.detect(out_dir, in_dir, args.targets); return
    commit, dirty = dai1.git_commit()
    if args.stage == "eval" and dirty:
        sys.exit("評価の段は未コミットの変更があると流さない（--allow-dirty も受け付けない）")
    if dirty and not args.allow_dirty:
        sys.exit("DAI/exp・measure・features に未コミットの変更がある（試しは --allow-dirty）")
    gammas = GAMMA_GRID + ((1.0,) if args.selftest_g1 else ())
    run = dict(commit=commit, dirty=dirty, frozen_sha256="", dai1_frozen_sha256=dai1.sha(FROZEN1), gate_sha256="",
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    dai1.RUN["commit"], dai1.RUN["dirty"] = commit, dirty
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, host=platform.node(), gpu=run["gpu"], torch=torch.__version__,
                script_sha256=dai1.sha(Path(__file__)), time=time.strftime("%Y-%m-%d %H:%M:%S"), dai1_frozen_sha256=run["dai1_frozen_sha256"],
                gammas=list(gammas))
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
            sys.exit("固定のファイルが無いか，DAI/frozen/dai4_frozen.json と一致しない")
        if not dai1.committed_clean(cp):
            sys.exit("固定のファイルの写しがコミットされていない")
        gate_sha = check_eval_gate4(f"DAI/frozen/{out_dir.name}_frozen.json")
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
            run_unit(in_dir, out_dir, args.stage, T, c, srcs, f, r, tether, gammas, run, args.selftest_g1)
            n += 1
    print(f"{args.stage}: {n} 単位", flush=True)


if __name__ == "__main__":
    main()
