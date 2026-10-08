"""ずれを消す・足す操作の実験（`docs/methods.md` 5 節，`researchvision.md` 8.2）．2026-09-30 から後回し．速報（反復 1 回）だけを回した。

地図（`map.py`）と**同じテスト・同じ K 本**（乱数はターゲットと反復番号だけで決まる）で，
ソースに操作を施して学び直す。比べる相手（操作なし）は，地図の結果（`results/map_<kind>/`）の
同じ (target, source, K, rep) の行である。操作はすべて `measure/ablate.py` を通す。

速報で回す操作（`researchvision.md` 6.1．事前登録の主判定に使う設定）：

| op | method | 施すこと | 予言 |
|---|---|---|---|
| I1 | merged，tgt_only | 群ごとの標準化．ソースはソース（使う行）の，ターゲットはターゲットのラベルなし全件の平均・標準偏差で | P1-1・P1-3 |
| C1 | merged | ソースを，ターゲット（ラベルなし全件）の近傍から n/2 本選ぶ（2 標本を合わせて標準化した空間） | P2-2 |
| C1ctrl | merged | 同じ本数を無作為に選ぶ（対照 R1） | P2-2 |
| P2 | merged | ソースを，ターゲットの推定割合（MLLS＋BCTS．`results/shift_<kind>/label.csv`）に合わせて抜き出す | P3-1 |
| P2ctrl | merged | 同じ本数を無作為に選ぶ（対照 R1） | P3-1 |
| D3s0.4 | merged，src_only | ソースのラベルの 40% を，ほかの 3 感情へ一様に付け替える | P4-2 |
| D3s0.2 | src_only | 同じく 20% | P4-3 |
| D3a0.2 | merged，src_only | ソースの Neutral の 20% を Sad へ付け替える | P4-3 |
| none・D3s0.2・D3s0.4・D3a0.2 | src_only+D1 | ソースのみの分類器の出力に，K 本で学んだ感情ごとのずらしを足す（D1） | P4-2・P4-3 |
| none | M5 | ターゲットの K 本で，ソースのみの分類器の係数へ引き寄せて学ぶ（C は引き寄せの弱さ．K 本の中の交差検証で選ぶ） | P4-4 |

付け替えの乱数はソースと反復番号で決め，どのターゲットと組んでも同じ付け替えにする。
操作ありの学習は温め直さない（操作で学習データが変わるため）。

    ./.venv/bin/python exp/ops.py --kind audio_wavlm --reps 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
import map as M  # noqa: E402
import ablate  # noqa: E402  （map が measure/ を探す場所に足している）
import common  # noqa: E402
import shift  # noqa: E402
from data import groups_for  # noqa: E402

OPS_MERGED = ("I1", "C1", "C1ctrl", "P2", "P2ctrl", "D3s0.4", "D3a0.2")
FLIPS = {"D3s0.2": ("s", 0.2), "D3s0.4": ("s", 0.4), "D3a0.2": ("a", 0.2)}
D3A_PAIR = (0, 2)                 # Neutral → Sad（`docs/methods.md` 5.3.4 の例．`researchvision.md` 8.2 で登録）
C1_FRAC = 0.5                     # n' = 0.5 n（`researchvision.md` 8.2 の主判定）
COLUMNS = ["op"] + M.COLUMNS


def expected_rows(n_src: int, n_k: int) -> int:
    """反復 1 回・ターゲット 1 つぶんの行の数（再開の検査に使う）。"""
    per_src = len(FLIPS) + (1 + len(FLIPS)) * n_k + n_k + len(OPS_MERGED) * n_k
    return n_src * per_src + n_k                          # 最後の n_k は I1 のターゲットのみ


def to_np(T: torch.Tensor) -> np.ndarray:
    return T.detach().cpu().numpy()


def flip(y: np.ndarray, name: str, source: str, r: int) -> np.ndarray:
    kind, rho = FLIPS[name]
    rng = common.rng_for(f"ops-{name}:{source}", r)
    if kind == "s":
        return ablate.flip_labels(y, rho, rng)
    return ablate.flip_labels_pair(y, D3A_PAIR[0], D3A_PAIR[1], rho, rng)


def d1_rows(model: M.Trained, X_test: torch.Tensor, y_test: np.ndarray, Xk_all: dict,
            yk_all: dict) -> dict[str, float]:
    """D1：ソースのみの分類器の出力に，K 本で学んだずらしを足してテストの UAR を出す。"""
    L_test = to_np(model.final.logits(X_test, model.C))
    out = {}
    for K, Xk in Xk_all.items():
        b = ablate.fit_class_bias(to_np(model.final.logits(Xk, model.C)), yk_all[K])
        out[K] = shift.uar(y_test, (L_test + b).argmax(1))
    return out


def run_rep(tgt: M.Group, sources: list[M.Group], r: int, Ks: list[str], kind: str,
            cache: dict, q_prior: dict) -> list[dict]:
    te = M.test_mask(tgt, r)
    pool = np.flatnonzero(~te)
    order = M.kshot_order(tgt, pool, r)
    test_persons = np.unique(tgt.person[te])
    y_test = tgt.y[te]
    base = {"target": tgt.name, "rep": r, "kind": kind, "n_test": int(te.sum())}
    idx_k = {K: M.take_k(order, K) for K in Ks}
    Xk = {K: M.rows(tgt.X, i) for K, i in idx_k.items()}
    yk = {K: tgt.y[i] for K, i in idx_k.items()}
    pk = {K: tgt.person[i] for K, i in idx_k.items()}
    X_test = M.rows(tgt.X, te)

    if ("I1t", tgt.name) not in cache:                    # ターゲットのラベルなし全件の統計で標準化
        Xn = to_np(tgt.X)
        cache[("I1t", tgt.name)] = M.logreg.to_device(ablate.standardize_with(Xn, Xn))
    Xt_I1 = cache[("I1t", tgt.name)]
    X_test_I1 = M.rows(Xt_I1, te)
    rows = []
    for K in Ks:                                          # I1 のターゲットのみ（K*' の相手）
        t = M.train(M.rows(Xt_I1, idx_k[K]), yk[K], pk[K], np.ones(len(yk[K]), bool), r)
        rows.append({**base, "op": "I1", "source": "", "method": "tgt_only", "K": K,
                     "n_src": 0, "n_tgt": len(yk[K]), **M.score(t, X_test_I1, y_test, False)})

    for s in sources:
        keep = ~np.isin(s.person, test_persons)
        ex = tuple(np.intersect1d(test_persons, s.person))
        Xs, ys, ps = M.rows(s.X, keep), s.y[keep], s.person[keep]
        ones_s = np.ones(len(ys), bool)
        sbase = {**base, "source": s.name}

        key = ("src", s.name, ex)                         # 操作なしのソースのみ（地図の ② と同じ）
        if key not in cache:
            cache[key] = M.train(Xs, ys, ps, ones_s, M.SRC_CV_SEED)
        src_models = {"none": cache[key]}
        flipped = {}
        for name in FLIPS:                                # 付け替えたソースのみ
            fk = ("flip", s.name, ex, name, r)
            if fk not in cache:
                yf = flip(ys, name, s.name, r)
                cache[fk] = (yf, M.train(Xs, yf, ps, ones_s, M.SRC_CV_SEED))
            flipped[name], src_models[name] = cache[fk]
            rows.append({**sbase, "op": name, "method": "src_only", "K": "", "n_src": len(ys),
                         "n_tgt": 0, **M.score(src_models[name], X_test, y_test, False)})
        for name, model in src_models.items():            # D1
            for K, u in d1_rows(model, X_test, y_test, Xk, yk).items():
                rows.append({**sbase, "op": name, "method": "src_only+D1", "K": K,
                             "n_src": len(ys), "n_tgt": len(yk[K]), "uar": u})
        for K in Ks:                                      # M5
            t = M.train(Xk[K], yk[K], pk[K], np.ones(len(yk[K]), bool), r,
                        anchor=src_models["none"])
            rows.append({**sbase, "op": "none", "method": "M5", "K": K, "n_src": len(ys),
                         "n_tgt": len(yk[K]), **M.score(t, X_test, y_test, False)})

        # ---- 合併に施す操作の，ソース側の行と特徴
        ik = ("I1s", s.name, ex)
        if ik not in cache:
            Xn = to_np(Xs)
            cache[ik] = M.logreg.to_device(ablate.standardize_with(Xn, Xn))
        ck = ("C1", s.name, tgt.name, ex)
        if ck not in cache:
            A, Bt = shift.pooled_standardize(to_np(Xs), to_np(tgt.X))
            cache[ck] = ablate.near_target_subset(A, Bt, int(C1_FRAC * len(ys)))
        sel_c1 = cache[ck]
        sel_c1c = ablate.random_subset_like(len(ys), len(sel_c1),
                                            common.rng_for(f"ops-C1ctrl:{s.name}:{tgt.name}", r))
        sel_p2 = ablate.prior_matched_subset(ys, q_prior[(s.name, tgt.name)],
                                             common.rng_for(f"ops-P2:{s.name}:{tgt.name}", r))
        sel_p2c = ablate.random_subset_like(len(ys), len(sel_p2),
                                            common.rng_for(f"ops-P2ctrl:{s.name}:{tgt.name}", r))
        variants = {                                      # op → (ソースの特徴, ラベル, 人物, テスト, ターゲットの特徴)
            "I1": (cache[ik], ys, ps, X_test_I1, Xt_I1),
            "C1": (M.rows(Xs, sel_c1), ys[sel_c1], ps[sel_c1], X_test, tgt.X),
            "C1ctrl": (M.rows(Xs, sel_c1c), ys[sel_c1c], ps[sel_c1c], X_test, tgt.X),
            "P2": (M.rows(Xs, sel_p2), ys[sel_p2], ps[sel_p2], X_test, tgt.X),
            "P2ctrl": (M.rows(Xs, sel_p2c), ys[sel_p2c], ps[sel_p2c], X_test, tgt.X),
            "D3s0.4": (Xs, flipped["D3s0.4"], ps, X_test, tgt.X),
            "D3a0.2": (Xs, flipped["D3a0.2"], ps, X_test, tgt.X),
        }
        assert tuple(variants) == OPS_MERGED
        for K in Ks:
            n = len(yk[K])
            for op, (XS, yS, pS, XT, XTall) in variants.items():
                first = np.r_[np.ones(n, bool), np.zeros(len(yS), bool)]
                t = M.train(torch.cat([M.rows(XTall, idx_k[K]), XS]), np.r_[yk[K], yS],
                            np.r_[pk[K], pS], first, r, M.SRC_CV_SEED)
                rows.append({**sbase, "op": op, "method": "merged", "K": K, "n_src": len(yS),
                             "n_tgt": n, **M.score(t, XT, y_test, False)})
    return rows


def load_prior(kind: str) -> dict:
    p = ROOT / "results" / f"shift_{kind}" / "label.csv"
    if not p.exists():
        raise SystemExit(f"{p} が無い．先に `exp/shift_map.py gpu --kind {kind}` を回す（P2 に要る）")
    d = pd.read_csv(p, comment="#")
    return {(r.source, r.target): np.array([getattr(r, f"q_t_{c}") for c in range(4)])
            for r in d.itertuples()}


def main() -> None:
    ap = argparse.ArgumentParser(description="ずれを消す・足す操作（docs/methods.md 5 節）")
    ap.add_argument("--kind", default=M.MODALITIES["audio"][0])
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--source", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(M.N_REPS)))
    ap.add_argument("--K", nargs="+", default=list(M.K_GRID), choices=M.K_GRID)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.cold = True                                      # provenance 用（操作ありは温め直さない）

    avail = groups_for(args.kind)
    args.target, args.source = args.target or avail, args.source or avail
    out_dir = args.out_dir or ROOT / "results" / f"ops_{args.kind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    q_prior = load_prior(args.kind)

    done: dict[str, set[int]] = {}
    for tname in args.target:
        path = out_dir / f"{tname}.csv"
        want = expected_rows(len([s for s in args.source if s != tname]), len(args.K))
        done[tname] = set()
        if path.exists():
            if not args.resume:
                raise SystemExit(f"{path} がすでにある。続きを足すなら --resume")
            cnt = pd.read_csv(path, comment="#", usecols=["rep"]).groupby("rep").size()
            if (cnt != want).any():
                raise SystemExit(f"{path}：行の数が {want} でない反復がある {cnt[cnt != want].to_dict()}")
            done[tname] = set(int(r) for r in cnt.index)

    prov = M.provenance(args)
    prov.update({"script": "exp/ops.py", "ops_merged": list(OPS_MERGED), "flips": FLIPS,
                 "d3a_pair": list(D3A_PAIR), "c1_frac": C1_FRAC,
                 "prior_from": f"results/shift_{args.kind}/label.csv"})
    code = M.snapshot_code(out_dir.resolve(), prov["started"])
    prov["code"] = str(code.relative_to(ROOT) if code.is_relative_to(ROOT) else code)
    data = {g: M.load(g, args.kind) for g in dict.fromkeys(args.target + args.source)}
    for tname in args.target:
        with open(out_dir / f"{tname}.csv", "a") as f:
            f.write("# " + json.dumps(prov, ensure_ascii=False) + "\n")
            if not done[tname]:
                pd.DataFrame(columns=COLUMNS).to_csv(f, index=False)

    cache: dict = {}
    t_all = time.perf_counter()
    for r in args.reps:
        for tname in args.target:
            if r in done[tname]:
                continue
            sources = [data[s] for s in args.source if s != tname]
            t0 = time.perf_counter()
            rows = run_rep(data[tname], sources, r, args.K, args.kind, cache, q_prior)
            pd.DataFrame(rows, columns=COLUMNS).to_csv(out_dir / f"{tname}.csv", mode="a",
                                                       header=False, index=False)
            print(f"{tname} r={r}: {len(rows)} 行 {time.perf_counter() - t0:.1f} 秒"
                  f"（累計 {time.perf_counter() - t_all:.0f} 秒）", flush=True)
        for k in [k for k in cache if k[0] == "flip" and k[-1] == r]:
            del cache[k]                                  # 付け替えは反復ごと．全ターゲットで使った後に捨てる
    print(f"終わり: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
