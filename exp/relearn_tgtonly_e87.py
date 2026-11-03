"""E87 の物差し：ターゲットだけで学んだ分類器（E76 の ①）を，世界（AVT・AV）の発話の集合と E77 と同じテストの分け方で学び直す．

経緯：E76 の ① は，そのモダリティ単独で使える発話の集合で作られている．E77・E79（世界の実行）の手法と none は，世界のモダリティが
そろう発話の集合で作られている．音声と言葉では MELD・MOSEI などでテストの発話が違うので，E76 の ① を物差しにできない（関門 1 の gate-code，2026-11-03）．
この台本は，世界の群 `map.load(T, 世界)` にモダリティの列を切り出した特徴の上で，E77 と同じ `map.test_mask`・`map.kshot_order`・`map.take_k` で
テストと K 本を作り，E76 の ① と同じ `map.train`（交差検証で C を選ぶ）で K = 1, 5, 10, 25, 50, 全件 を学ぶ．出力の鍵は E76 と同じ（`t_*`）．

  ./.venv/bin/python exp/relearn_tgtonly_e87.py [--runs fuse-audio-video-text fuse-audio-video] [--targets ...] [--reps 0 1 ...] [--out results/e87_tgtonly]

出力：`<out>/<世界>/<配備する分類器>/<T>.npz`（sample_id，speaker_group，y，reps，test_idx_r{r}，kidx_r{r}_K{K}，t_logit_r{r} (K 6, C 4, テスト, 4)，
      t_A_r{r}，t_a0_r{r}，t_C_r{r}，t_rule_r{r}，t_cv_r{r}，_meta），`<T>.csv`（tgt_only の行），`<T>_e76_check.csv`（E76 の ① と発話の集合が
      同じ組で，点数の差の最大と argmax の一致率．I14），`provenance.json`．
検算：反復ごとのテストの行番号が E77 の `results/e77_<世界>/<T>_r{r}_te.npy` と一致しなければ止まる（I18）．
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import logreg  # noqa: E402
import map as M  # noqa: E402
import relearn_e76 as R  # noqa: E402  （raw_coef・logits_np・uar_of・code_state・KEYS_DOC を使う）
from data import load_any  # noqa: E402
from multimodal_distill import WORLDS, blocks  # noqa: E402

RUNS = ("fuse-audio-video-text", "fuse-audio-video")
K_GRID = M.K_GRID                                                              # ("1", "5", "10", "25", "50", "all")
C_GRID = M.C_GRID
MEM_FRACTION = R.MEM_FRACTION


def world_targets(run):
    """E77 の世界の実行で評価のターゲットになった群（`results/e77_<世界>/<T>_r0_te.npy` があるもの）．"""
    d = ROOT / "results" / f"e77_{run}"
    return sorted(f.name[:-len("_r0_te.npy")] for f in d.glob("*_r0_te.npy"))


def student_group(g, run, st):
    if st == run:
        return g
    for m, a, b in blocks(run):
        if m == st:
            return M.Group(g.name, g.X[:, a:b], g.y, g.speaker, g.person)
    raise KeyError(st)


def e76_check(run, st, T, A, reps):
    """E76 の ① と発話の集合（sample_id）が同じ組で，同じ K・反復の点数を比べる（I14）．集合が違う組は 'set_differs' と記録する．"""
    f = ROOT / "results" / "e76" / st / f"{T}.npz"
    rows = []
    if not f.exists():
        return rows
    z = np.load(f)
    s76 = np.asarray(z["sample_id"]).astype(str)
    sid = A["sample_id"]
    same_set = set(s76) == set(sid)
    for r in reps:
        te76 = np.asarray(z[f"test_idx_r{r}"])
        te = A[f"test_idx_r{r}"]
        if not same_set or set(s76[te76]) != set(sid[te]):
            rows.append({"run": run, "student": st, "target": T, "rep": r, "status": "set_differs", "n_world": len(sid), "n_e76": len(s76),
                         "n_test_world": len(te), "n_test_e76": len(te76)})
            continue
        pos76 = {s: i for i, s in enumerate(s76[te76])}
        order = np.array([pos76[s] for s in sid[te]])
        L76 = np.asarray(z[f"t_logit_r{r}"])[:, :, order]                      # (K, C, テスト, 4) を世界の並びに
        Lw = A[f"t_logit_r{r}"]
        C76, Cw = np.asarray(z[f"t_C_r{r}"]), A[f"t_C_r{r}"]
        k76 = {K: set(s76[np.asarray(z[f"kidx_r{r}_K{K}"])]) for K in K_GRID if K != "all"}
        kw = {K: set(sid[A[f"kidx_r{r}_K{K}"]]) for K in K_GRID if K != "all"}
        rows.append({"run": run, "student": st, "target": T, "rep": r, "status": "compared",
                     "max_abs_logit_diff": float(np.abs(L76 - Lw).max()), "argmax_agree_min": float(min((L76[i, j].argmax(1) == Lw[i, j].argmax(1)).mean() for i in range(6) for j in range(4))),
                     "C_equal": bool(np.allclose(C76, Cw)), "kidx_equal": bool(all(k76[K] == kw[K] for K in k76))})
    return rows


def run_target(run, st, T, g, reps, out_dir):
    gs = student_group(g, run, st)
    df, _ = load_any(T, run)
    A = {"sample_id": df.sample_id.astype(str).to_numpy(), "speaker_group": df.speaker_group.astype(str).to_numpy(),
         "y": gs.y.astype(np.int8), "reps": np.array(reps)}
    d77 = ROOT / "results" / f"e77_{run}"
    rows, timing = [], {}
    t0 = time.perf_counter()
    for r in reps:
        te = M.test_mask(gs, r)
        te77 = np.load(d77 / f"{T}_r{r}_te.npy")
        if not np.array_equal(np.flatnonzero(te), np.sort(np.asarray(te77).ravel())):
            raise SystemExit(f"{run} {st} {T} r{r}：テストの行番号が E77 と違う（I18）")
        order = M.kshot_order(gs, np.flatnonzero(~te), r)
        idx = {K: M.take_k(order, K) for K in K_GRID}
        A[f"test_idx_r{r}"] = np.flatnonzero(te).astype(np.int32)
        for K in K_GRID:
            if K != "all":
                A[f"kidx_r{r}_K{K}"] = idx[K].astype(np.int32)
        Xte, yte = M.rows(gs.X, te), gs.y[te]
        Ls, As, a0s, Cs, rules, cvs = [], [], [], [], [], []
        for K in K_GRID:
            i = idx[K]
            t = M.train(M.rows(gs.X, i), gs.y[i], gs.person[i], np.ones(len(i), bool), r)
            L = np.stack([R.logits_np(t.final, Xte, c) for c in C_GRID])
            co = [R.raw_coef(t.final, c) for c in C_GRID]
            Ls.append(L.astype(np.float32))
            As.append(np.stack([a for a, _ in co]))
            a0s.append(np.stack([b for _, b in co]))
            Cs.append(t.C)
            rules.append(t.rule)
            cvs.append([np.nan if t.cv is None else t.cv[c] for c in C_GRID])
            test = {c: R.uar_of(L[j], yte) for j, c in enumerate(C_GRID)}
            pool_counts = np.bincount(gs.y[~te], minlength=4)
            rows.append({"run": run, "kind": st, "target": T, "method": "tgt_only", "K": K, "rep": r, "C": t.C, "C_rule": t.rule,
                         "n_tgt": int(len(i)), "n_test": int(te.sum()), "n_pool_min_class": int(pool_counts.min()), "uar": test[t.C],
                         "n_iter": int(t.final.n_iter[t.C]), "grad": float(max(t.final.grad.values())),
                         **{f"test_uar_C{c:g}": test[c] for c in C_GRID}})
        A[f"t_logit_r{r}"] = np.stack(Ls)
        A[f"t_A_r{r}"] = np.stack(As)
        A[f"t_a0_r{r}"] = np.stack(a0s)
        A[f"t_C_r{r}"] = np.array(Cs)
        A[f"t_rule_r{r}"] = np.array(rules)
        A[f"t_cv_r{r}"] = np.array(cvs)
        R.free_gpu()
    timing["tgt_only"] = time.perf_counter() - t0
    chk = e76_check(run, st, T, A, reps)
    return A, pd.DataFrame(rows), pd.DataFrame(chk), timing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--students", nargs="+", help="既定：世界のモダリティと融合")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e87_tgtonly")
    ap.add_argument("--gpu-fraction", type=float, default=MEM_FRACTION)
    ap.add_argument("--allow-dirty", action="store_true", help="読み込むコードにコミットしていない変更があっても流す（試しだけ）")
    args = ap.parse_args()
    state = R.code_state()
    if state["code_dirty"] and not args.allow_dirty:
        raise SystemExit("読み込むコードにコミットしていない変更がある（--allow-dirty は試しだけ）:\n" + "\n".join(state["code_status"]))
    partial = bool(args.targets or args.students or args.reps != list(range(10)))
    out = args.out.resolve()
    if partial and out.is_relative_to(ROOT / "results"):
        raise SystemExit("試し（--targets，--students，--reps）の出力は results/ の下に書かない（I8）")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    settings = {"reps": args.reps, "K": list(K_GRID), "C_grid": list(C_GRID), "n_test_folds": M.N_TEST_FOLDS, "n_cv_folds": M.N_CV_FOLDS,
                "solver": {"tol": logreg.TOL, "max_iter": logreg.MAX_ITER, "history": logreg.HISTORY, "dtype": str(logreg.DTYPE)}, "mem_fraction": args.gpu_fraction,
                "split": "map.test_mask・map.kshot_order・map.take_k を世界の群（map.load(T, 世界)）に当てる（E77 の compare_b1.Ctx と同じ）"}
    prov_path = out / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {"runs": []}
    run_rec = {"script": "exp/relearn_tgtonly_e87.py", "argv": sys.argv[1:], **state, "started": started, "settings": settings,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__, "done": []}
    prov["runs"].append(run_rec)
    for run in args.runs:
        targets = [t for t in world_targets(run) if (not args.targets or t in args.targets)]
        students = args.students or (WORLDS[run] + [run])
        data = {T: M.load(T, run) for T in targets}
        for T in targets:
            for st in students:
                kdir = out / run / st
                kdir.mkdir(parents=True, exist_ok=True)
                if (kdir / f"{T}.npz").exists():
                    z = np.load(kdir / f"{T}.npz")
                    meta = json.loads(str(z["_meta"]))
                    if meta.get("code_hash") == state["code_hash"] and meta.get("settings") == settings:
                        print(f"{run} {st} {T}：済み（同じコードと設定）．飛ばす", flush=True)
                        continue
                    raise SystemExit(f"{kdir / T}.npz が別のコードか設定で作られている．消すか別の --out にする")
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                A, d, chk, timing = run_target(run, st, T, data[T], args.reps, kdir)
                meta = {**{k: state[k] for k in ("commit", "code_hash", "code_dirty")}, "settings": settings, "started": started, "run": run, "kind": st, "target": T,
                        "elapsed_sec": time.perf_counter() - t0, "timing_sec": timing,
                        "gpu_peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None, "keys_doc": R.KEYS_DOC}
                A["_meta"] = np.array(json.dumps({k: meta[k] for k in ("commit", "code_hash", "code_dirty", "settings", "run", "kind", "target", "started")}))
                tmp = kdir / f".{T}.tmp.npz"
                np.savez_compressed(tmp, **A)
                os.replace(tmp, kdir / f"{T}.npz")
                with open(kdir / f".{T}.tmp.csv", "w") as f:
                    f.write("# " + json.dumps({k: meta[k] for k in ("commit", "code_hash", "code_dirty", "settings", "started")}, ensure_ascii=False) + "\n")
                    d.to_csv(f, index=False)
                os.replace(kdir / f".{T}.tmp.csv", kdir / f"{T}.csv")
                if len(chk):
                    chk.to_csv(kdir / f"{T}_e76_check.csv", index=False)
                run_rec["done"].append({"run": run, "kind": st, "target": T, "elapsed_sec": meta["elapsed_sec"], "gpu_peak_alloc_gib": meta["gpu_peak_alloc_gib"],
                                        "e76_check": chk.status.value_counts().to_dict() if len(chk) else {}})
                prov_path.write_text(json.dumps(prov, ensure_ascii=False, indent=1))
                print(f"{run} {st} {T}：{meta['elapsed_sec']:.0f} 秒", flush=True)
        del data
        R.free_gpu()
    run_rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov_path.write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    print(out)


if __name__ == "__main__":
    main()
