"""得の仕組みの仮説 H3・H4 を，学び直して確かめる（`docs/exploratory-analyses.md` E12）。

地図（`map.py`）と**同じテスト・同じ K 本**（乱数はターゲットと反復番号だけで決まる）で学び直す。
比べる相手（操作なし）は，地図の結果（`results/map_<kind>/`）の同じ (target, source, K, rep) の行。

**H3（中心のずれが境目の位置をずらす）**：ラベルを使わずに中心をそろえると，ソースのみの成績が上がるか。

| op | method | 施すこと |
|---|---|---|
| center | src_only | ソースは学習に使う行の平均を，ターゲットはラベルなし全件の平均を引く |
| I1 | src_only | 平均を引き，さらにそれぞれの標準偏差で割る（`ops.py` の I1 と同じ） |
| none・center | src_only+D1 | ソースのみの分類器の出力に，K 本で学んだ感情ごとのずらしを足す（`ops.py` の D1） |
| center | merged | 中心をそろえた特徴で合併を学ぶ |

**H4（ターゲットがソースに埋もれる）**：学習データのうちターゲットが占める重みの割合 α を選ぶと，
K が大きいときの負の得が消えるか。

| op | method | 施すこと |
|---|---|---|
| wt | merged | α を候補 `ALPHAS` から C と一緒に交差検証で選ぶ |

α の候補の nat は地図と同じ重み（ソースとターゲットを合わせて感情の釣り合いをとる．ターゲットの
割合は本数の比で決まる）。nat 以外は，ターゲットの行の重みの和を α N，ソースの行を (1−α) N にし
（N は学習に使う行の数），それぞれの中で感情の釣り合いをとる。交差検証の袋・成績の測り方は地図と
同じ（成績は保留側のターゲットの行だけで測る）。最良の (α, C) を選ぶ。同点なら nat，次に α の小さい方を
採り，C の同点は地図の規則（`map.choose_c`）。全候補のテストの UAR も残す（オラクル．分析専用）。

    ./.venv/bin/python exp/mech.py --kind audio_wavlm --reps 0 1 2
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
import shift  # noqa: E402
from data import groups_for  # noqa: E402

ALPHAS = ("nat", 0.25, 0.5, 0.75)
K_SRC = M.K_GRID                          # D1 の K
K_CENTER = ("1", "5", "25", "all")        # 中心をそろえた合併の K
K_WT = ("5", "25", "all")                 # α を選ぶ合併の K
EXTRA = (["alpha", "alpha_cv"]
         + [f"cv_a{a}_C{c:g}" for a in ALPHAS for c in M.C_GRID]
         + [f"test_a{a}_C{c:g}" for a in ALPHAS for c in M.C_GRID])
COLUMNS = ["op"] + M.COLUMNS + EXTRA


def expected_rows(n_src: int, k_src: int, k_center: int, k_wt: int) -> int:
    """反復 1 回・ターゲット 1 つぶんの行の数（再開の検査に使う）。"""
    return n_src * (2 + 2 * k_src + k_center + k_wt)


def to_np(T: torch.Tensor) -> np.ndarray:
    return T.detach().cpu().numpy()


def centered(X: torch.Tensor, scale: bool) -> torch.Tensor:
    """X 自身の平均を引く（scale なら標準偏差でも割る）。ラベルは使わない。"""
    Z = X - X.mean(0)
    if scale:
        sd = X.std(0, correction=0)
        Z = Z / torch.where(sd > 1e-12, sd, torch.ones_like(sd))
    return Z


def d1_rows(model: M.Trained, X_test: torch.Tensor, y_test: np.ndarray, Xk: dict,
            yk: dict) -> dict[str, float]:
    """D1：ソースのみの分類器の出力に，K 本で学んだずらしを足してテストの UAR を出す（`ops.py` と同じ）。"""
    L_test = to_np(model.final.logits(X_test, model.C))
    out = {}
    for K, X in Xk.items():
        b = ablate.fit_class_bias(to_np(model.final.logits(X, model.C)), yk[K])
        out[K] = shift.uar(y_test, (L_test + b).argmax(1))
    return out


def alpha_weights(y: np.ndarray, first: np.ndarray, m: np.ndarray, a) -> torch.Tensor | None:
    """α の重み。nat なら None（`balanced_weights` を使う）。"""
    if a == "nat":
        return None
    w = np.zeros(len(y))
    N = int(m.sum())
    for part, share in ((first & m, a), (~first & m, 1.0 - a)):
        cls = np.unique(y[part])
        for c in cls:
            sel = part & (y == c)
            w[sel] = share * N / (len(cls) * sel.sum())
    return torch.as_tensor(w, dtype=M.logreg.DTYPE, device=M.logreg.DEVICE)


def train_alpha(X: torch.Tensor, y: np.ndarray, person: np.ndarray, first: np.ndarray,
                seed: int, other_seed: int, init: M.Trained) -> dict:
    """α と C を交差検証で選ぶ。袋の作り方と温め直しは `map.train` と同じ。"""
    t0 = time.perf_counter()
    yt = M.labels(y)
    folds = M.cv_folds(person, first, seed, other_seed)
    kind = "folds" if np.array_equal(folds[~first], init.folds) else "final"

    def start(j: int):
        return init.fold_fits[j] if kind == "folds" and j in init.fold_fits else init.final

    used = [j for j in range(M.N_CV_FOLDS) if (first & (folds == j)).any()]
    if not all(len(np.unique(y[folds != j])) == M.N_CLASSES for j in used):
        used = []
    masks, ws, inits, tags = [], [], [], []
    for a in ALPHAS:
        for j in used + [None]:
            m = np.ones(len(y), bool) if j is None else folds != j
            masks.append(m)
            ws.append(alpha_weights(y, first, m, a))
            inits.append(init.final if j is None else start(j))
            tags.append((a, j))
    fits = M.logreg.fit_batch(X, yt, masks, M.C_GRID, inits, weights=ws)
    finals = {a: f for (a, j), f in zip(tags, fits) if j is None}
    cv = None
    if used:
        cv = {}
        for a in ALPHAS:
            pred = {c: np.full(len(y), -1) for c in M.C_GRID}
            for (aa, j), f in zip(tags, fits):
                if aa != a or j is None:
                    continue
                va = first & (folds == j)
                Xva = M.rows(X, va)
                for c in M.C_GRID:
                    pred[c][va] = f.predict(Xva, c)
            for c in M.C_GRID:
                cv[(a, c)] = shift.uar(y[first], pred[c][first])
        best = (None, None, -np.inf)
        for a in ALPHAS:                                   # 同点なら先の候補（nat，小さい α）
            c, _ = M.choose_c({c: cv[(a, c)] for c in M.C_GRID})
            if cv[(a, c)] > best[2] + 1e-12:
                best = (a, c, cv[(a, c)])
        a_sel, c_sel, rule = best[0], best[1], "cv"
    else:
        a_sel, c_sel, rule = "nat", M.C_DEFAULT, "fixed"
    return {"finals": finals, "alpha": a_sel, "C": c_sel, "rule": rule, "cv": cv,
            "sec": time.perf_counter() - t0, "init": kind,
            "n_iter": sum(sum(f.n_iter.values()) for f in fits),
            "n_loop": max(max(f.n_iter.values()) for f in fits)}


def score_alpha(t: dict, X_test: torch.Tensor, y_test: np.ndarray) -> dict:
    test = {(a, c): shift.uar(y_test, f.predict(X_test, c))
            for a, f in t["finals"].items() for c in M.C_GRID}
    (a_or, c_or) = max(test, key=lambda k: test[k])
    row = {"C": t["C"], "C_rule": t["rule"], "uar": test[(t["alpha"], t["C"])],
           "C_oracle": c_or, "uar_oracle": test[(a_or, c_or)], "sec": t["sec"],
           "n_iter": t["n_iter"], "n_loop": t["n_loop"], "init": t["init"], "cached": False,
           "alpha": t["alpha"], "alpha_cv": t["rule"]}
    for (a, c), u in test.items():
        row[f"test_a{a}_C{c:g}"] = u
        row[f"cv_a{a}_C{c:g}"] = np.nan if t["cv"] is None else t["cv"][(a, c)]
    for c in M.C_GRID:                                     # 地図と同じ列には nat の値
        row[f"cv_uar_C{c:g}"] = np.nan if t["cv"] is None else t["cv"][("nat", c)]
        row[f"test_uar_C{c:g}"] = test[("nat", c)]
    return row


def run_rep(tgt: M.Group, sources: list[M.Group], r: int, kind: str, cache: dict,
            k_src, k_center, k_wt) -> list[dict]:
    te = M.test_mask(tgt, r)
    pool = np.flatnonzero(~te)
    order = M.kshot_order(tgt, pool, r)
    test_persons = np.unique(tgt.person[te])
    y_test = tgt.y[te]
    base = {"target": tgt.name, "rep": r, "kind": kind, "n_test": int(te.sum())}
    Ks = list(dict.fromkeys(list(k_src) + list(k_center) + list(k_wt)))
    idx_k = {K: M.take_k(order, K) for K in Ks}
    yk = {K: tgt.y[i] for K, i in idx_k.items()}
    pk = {K: tgt.person[i] for K, i in idx_k.items()}
    Xt = {"none": tgt.X, "center": centered(tgt.X, False), "I1": centered(tgt.X, True)}
    rows = []
    for s in sources:
        keep = ~np.isin(s.person, test_persons)
        ex = tuple(np.intersect1d(test_persons, s.person))
        Xs0, ys, ps = M.rows(s.X, keep), s.y[keep], s.person[keep]
        ones_s = np.ones(len(ys), bool)
        sbase = {**base, "source": s.name}
        Xs = {"none": Xs0, "center": centered(Xs0, False), "I1": centered(Xs0, True)}
        src = {}
        for op in ("none", "center", "I1"):               # ソースのみ（none は地図の ② と同じ）
            key = ("src", op, s.name, ex)
            cached = key in cache
            if not cached:
                cache[key] = M.train(Xs[op], ys, ps, ones_s, M.SRC_CV_SEED)
            src[op] = cache[key]
            if op != "none":
                rows.append({**sbase, "op": op, "method": "src_only", "K": "", "n_src": len(ys),
                             "n_tgt": 0, **M.score(src[op], M.rows(Xt[op], te), y_test, cached)})
        for op in ("none", "center"):                     # D1
            Xk = {K: M.rows(Xt[op], idx_k[K]) for K in k_src}
            for K, u in d1_rows(src[op], M.rows(Xt[op], te), y_test, Xk,
                                {K: yk[K] for K in k_src}).items():
                rows.append({**sbase, "op": op, "method": "src_only+D1", "K": K,
                             "n_src": len(ys), "n_tgt": len(yk[K]), "uar": u})
        for K in k_center:                                # 中心をそろえた合併
            n = len(idx_k[K])
            first = np.r_[np.ones(n, bool), np.zeros(len(ys), bool)]
            t = M.train(torch.cat([M.rows(Xt["center"], idx_k[K]), Xs["center"]]),
                        np.r_[yk[K], ys], np.r_[pk[K], ps], first, r, M.SRC_CV_SEED,
                        src["center"])
            rows.append({**sbase, "op": "center", "method": "merged", "K": K, "n_src": len(ys),
                         "n_tgt": n, **M.score(t, M.rows(Xt["center"], te), y_test, False)})
        for K in k_wt:                                    # α を選ぶ合併
            n = len(idx_k[K])
            first = np.r_[np.ones(n, bool), np.zeros(len(ys), bool)]
            t = train_alpha(torch.cat([M.rows(tgt.X, idx_k[K]), Xs0]), np.r_[yk[K], ys],
                            np.r_[pk[K], ps], first, r, M.SRC_CV_SEED, src["none"])
            rows.append({**sbase, "op": "wt", "method": "merged", "K": K, "n_src": len(ys),
                         "n_tgt": n, **score_alpha(t, M.rows(tgt.X, te), y_test)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="得の仕組みの仮説 H3・H4 の確かめ（E12）")
    ap.add_argument("--kind", default=M.MODALITIES["audio"][0])
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--source", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--K-src", nargs="+", default=list(K_SRC), choices=M.K_GRID)
    ap.add_argument("--K-center", nargs="+", default=list(K_CENTER), choices=M.K_GRID)
    ap.add_argument("--K-wt", nargs="+", default=list(K_WT), choices=M.K_GRID)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.cold, args.K = False, sorted(set(args.K_src + args.K_center + args.K_wt),
                                      key=M.K_GRID.index)

    avail = groups_for(args.kind)
    args.target, args.source = args.target or avail, args.source or avail
    out_dir = args.out_dir or ROOT / "results" / f"mech_{args.kind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    done: dict[str, set[int]] = {}
    for tname in args.target:
        path = out_dir / f"{tname}.csv"
        want = expected_rows(len([s for s in args.source if s != tname]), len(args.K_src),
                             len(args.K_center), len(args.K_wt))
        done[tname] = set()
        if path.exists():
            if not args.resume:
                raise SystemExit(f"{path} がすでにある。続きを足すなら --resume")
            cnt = pd.read_csv(path, comment="#", usecols=["rep"]).groupby("rep").size()
            if (cnt != want).any():
                raise SystemExit(f"{path}：行の数が {want} でない反復がある {cnt[cnt != want].to_dict()}")
            done[tname] = set(int(r) for r in cnt.index)

    prov = M.provenance(args)
    prov.update({"script": "exp/mech.py", "alphas": list(ALPHAS), "K_src": args.K_src,
                 "K_center": args.K_center, "K_wt": args.K_wt})
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
            rows = run_rep(data[tname], sources, r, args.kind, cache, args.K_src,
                           args.K_center, args.K_wt)
            pd.DataFrame(rows, columns=COLUMNS).to_csv(out_dir / f"{tname}.csv", mode="a",
                                                       header=False, index=False)
            print(f"{tname} r={r}: {len(rows)} 行 {time.perf_counter() - t0:.1f} 秒"
                  f"（累計 {time.perf_counter() - t_all:.0f} 秒）", flush=True)
    print(f"終わり: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
