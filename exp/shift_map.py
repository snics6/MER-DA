"""ずれの指標を全組で測る（`docs/methods.md` 4 節，`researchvision.md` 8.2）。3 段に分ける。

    ./.venv/bin/python exp/shift_map.py gpu   --kind audio_wavlm            # 軸 3・4 と対数オッズ（GPU）
    ./.venv/bin/python exp/shift_map.py input --kind audio_wavlm --workers 14  # 軸 1・2（CPU）
    ./.venv/bin/python exp/shift_map.py space --kind audio_wavlm            # 空間の選択（登録した規則）

指標はすべて `measure/shift.py` を通す（書き直さない）。結果は `results/shift_<kind>/`：

| ファイル | 中身 |
|---|---|
| `labeling.csv` | 軸 4：ソースのみの分類器（本研究の固定学習器．C はソース内の話者交差検証）の，ソース側（交差検証の保留側）と ターゲット全件での UAR・混同行列，Δ_CE，BER，P4-1 の検算，K 本（反復 0 の抽出）での実用版 |
| `label.csv` | 軸 3：割合の差（記述用＝全ラベル，実用＝MLLS＋BCTS でターゲットのラベルなしから推定） |
| `logits/<ソース>.npz` | 空間 (b) の点数：ソースは交差検証の保留側の予測，ターゲットは学習済みの分類器の出力．軸 1・2 を測るときは和 0 の 3 次元の座標に直す（`sum_zero_coords`） |
| `input.csv` | 軸 1・2：空間 raw（参考）・pca64（候補 a）・logit（候補 b），割合をそろえた版（C9，logit と pca64） |
| `zero.csv` | 同じ群を話者で 2 つに割ったゼロ点（raw・pca64 は群ごと，logit は組ごと） |
| `space_choice.json` | 登録した規則（`researchvision.md` 8.2 共通の決まり 6）の当てはめ |

**同じ人物が両側にいる組**（IEMOCAP-impro と -script）では，ターゲットの採点に，その話者を除いた
ソースで学び直した分類器を使う（話者 5 分割の交差当てはめ．`cross_fit` 列が真）。
"""

from __future__ import annotations

import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "input":
    os.environ.setdefault("OMP_NUM_THREADS", "1")          # 並列の各プロセスは 1 スレッド

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "measure"))
sys.path.insert(0, str(ROOT / "features"))
sys.path.insert(0, str(ROOT / "exp"))
import shift  # noqa: E402
from data import groups_for, load_any  # noqa: E402

N_BAL = 160                  # 割合をそろえた版の本数（感情ごとに 40．RAVDESS の半分でも足りる）
PAD_SATURATED = 1.9          # 登録した規則の「飽和していない」の境
K_GRID = ("1", "5", "10", "25", "50", "all")


def out_dir(kind: str) -> Path:
    d = ROOT / "results" / f"shift_{kind}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def provenance(kind: str, stage: str) -> dict:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"script": "exp/shift_map.py", "stage": stage, "kind": kind, "argv": sys.argv[1:],
            "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "n": shift.N_DEFAULT, "n_balanced": N_BAL, "reps": shift.R_DEFAULT,
            "pca_dim": shift.PCA_DIM, "pad_tol": shift.PAD_TOL}


def write_csv(path: Path, rows: list[dict], prov: dict) -> None:
    with open(path, "w") as f:
        f.write("# " + json.dumps(prov, ensure_ascii=False) + "\n")
        pd.DataFrame(rows).to_csv(f, index=False)


# ---------------------------------------------------------------- GPU 段：軸 3・4 と対数オッズ

def stage_gpu(kind: str) -> None:
    import map as M                                   # torch を使うのはこの段だけ
    from split import speaker_folds

    out = out_dir(kind)
    (out / "logits").mkdir(exist_ok=True)
    prov = provenance(kind, "gpu")
    prov["code"] = str(M.snapshot_code(out, prov["started"]).relative_to(ROOT))
    groups = groups_for(kind)
    data = {g: M.load(g, kind) for g in groups}
    ones = lambda n: np.ones(n, bool)  # noqa: E731
    lab_rows, label_rows = [], []
    for sname in groups:
        t0 = time.perf_counter()
        s = data[sname]
        tr = M.train(s.X, s.y, s.person, ones(len(s.y)), M.SRC_CV_SEED)
        C = tr.C
        oof = np.zeros((len(s.y), shift.N_CLASSES))
        for j, f in tr.fold_fits.items():
            m = tr.folds == j
            oof[m] = f.logits(M.rows(s.X, m), C).cpu().numpy()
        pred_s = oof.argmax(1)
        conf_s = shift.confusion(s.y, pred_s)
        uar_s = shift.uar(s.y, pred_s)
        temp, bias = shift.fit_bcts(oof, s.y)
        prior_cal = shift.apply_bcts(oof, temp, bias).mean(0)
        p_s = shift.label_prior(s.y)
        saved = {"oof": oof}
        for tname in groups:
            if tname == sname:
                continue
            t = data[tname]
            cross = bool(np.isin(t.person, s.person).any())
            if cross:                                     # 同じ人物が両側にいる → 交差当てはめ
                lt = np.zeros((len(t.y), shift.N_CLASSES))
                tf = speaker_folds(t.speaker, k=5, seed=0).to_numpy()
                for fo in range(5):
                    mt = tf == fo
                    ks = ~np.isin(s.person, np.unique(t.person[mt]))
                    trf = M.train(M.rows(s.X, ks), s.y[ks], s.person[ks], ones(int(ks.sum())),
                                  M.SRC_CV_SEED)
                    lt[mt] = trf.final.logits(M.rows(t.X, mt), trf.C).cpu().numpy()
            else:
                lt = tr.final.logits(t.X, C).cpu().numpy()
            saved[f"t_{tname}"] = lt
            pred_t = lt.argmax(1)
            conf_t = shift.confusion(t.y, pred_t)
            uar_t = shift.uar(t.y, pred_t)
            dce, diff = shift.delta_ce(conf_s, conf_t)
            row = {"source": sname, "target": tname, "kind": kind, "C": C, "cross_fit": cross,
                   "uar_s": uar_s, "uar_t": uar_t, "r_src": 1 - uar_t,
                   "delta_ce": dce, "ber_s": shift.ber(conf_s), "ber_t": shift.ber(conf_t),
                   "p41_holds": bool(uar_s - uar_t <= 3 * dce + 1e-12),
                   "p41_ratio": (uar_s - uar_t) / (3 * dce) if dce > 0 else np.nan}
            for a in range(4):
                for b in range(4):
                    row[f"conf_s_{a}{b}"], row[f"conf_t_{a}{b}"] = conf_s[a, b], conf_t[a, b]
            te0 = M.test_mask(t, 0)                       # 実用版：反復 0 の K 本
            order = M.kshot_order(t, np.flatnonzero(~te0), 0)
            for K in K_GRID:
                idx = M.take_k(order, K)
                ck = shift.confusion(t.y[idx], pred_t[idx])
                row[f"uar_t_K{K}"] = shift.uar(t.y[idx], pred_t[idx])
                row[f"delta_ce_K{K}"] = shift.delta_ce(conf_s, ck)[0]
            lab_rows.append(row)
            q = shift.mlls_prior(shift.apply_bcts(lt, temp, bias), prior_cal)
            p_t = shift.label_prior(t.y)
            lr = {"source": sname, "target": tname, "kind": kind,
                  "l1_true": shift.label_l1(s.y, t.y),
                  "l1_est": float(np.abs(p_s - q).sum()),
                  "est_err_l1": float(np.abs(q - p_t).sum()), "bcts_temp": temp}
            for c in range(4):
                lr[f"p_s_{c}"], lr[f"p_t_{c}"], lr[f"q_t_{c}"] = p_s[c], p_t[c], q[c]
            label_rows.append(lr)
        np.savez_compressed(out / "logits" / f"{sname}.npz", **saved)
        print(f"{kind} ソース {sname}: {time.perf_counter() - t0:.1f} 秒", flush=True)
    write_csv(out / "labeling.csv", lab_rows, prov)
    write_csv(out / "label.csv", label_rows, prov)


# ---------------------------------------------------------------- CPU 段：軸 1・2

_Q = np.linalg.svd(np.eye(4) - 1 / 4)[0][:, :3]   # 和が 0 の部分空間の正規直交基底（4×3）


def sum_zero_coords(L: np.ndarray) -> np.ndarray:
    """4 つの対数オッズを，平均を引いて和 0 の 3 次元の座標で表す。

    多クラスのロジスティック回帰では 4 つの点数の和がほぼ一定になり（係数の和が 0 に寄る），
    空間は実質 3 次元である。一定の方向だけを捨てるので，点どうしの距離は変わらない
    （平均を引く射影の後の正規直交座標）。4 次元のままだと，共分散の支配が広がり 0 の方向で落ちた。
    """
    return (L - L.mean(1, keepdims=True)) @ _Q

_G: dict = {}                                            # 並列の子プロセスが fork で受け継ぐ


def _task(job: tuple) -> dict:
    """1 つの組（またはゼロ点）を測る。失敗しても全体を止めず，失敗の印を返す。"""
    try:
        return _task_body(job)
    except Exception as e:                                # noqa: BLE001
        return {"_error": f"{job}: {type(e).__name__}: {e}"}


def _task_body(job: tuple) -> dict:
    what, key, space, bal = job
    n = N_BAL if bal else shift.N_DEFAULT
    if what == "pair":
        sname, tname = key
        if space == "logit":
            Xs, Xt = _G["logits"][sname]["oof"], _G["logits"][sname][f"t_{tname}"]
            sp = "raw"
        else:
            Xs, Xt, sp = _G["X"][sname], _G["X"][tname], space
        ys, yt = (_G["y"][sname], _G["y"][tname]) if bal else (None, None)
        m = shift.input_metrics(Xs, _G["g"][sname], Xt, _G["g"][tname], n=n, space=sp,
                                ys=ys, yt=yt, reps=_G["reps"])
        return {**m, "source": sname, "target": tname, "space": space, "balanced": bal}
    # ゼロ点：raw・pca64 は群ごと，logit は組ごと（ソースの分類器の空間でターゲットを割る）
    if space == "logit":
        sname, tname = key
        X, sp = _G["logits"][sname][f"t_{tname}"], "raw"
    else:
        sname, tname, X, sp = "", key, _G["X"][key], space
    m = shift.zero_point(X, _G["g"][tname], n=n, space=sp, y=_G["y"][tname] if bal else None,
                         reps=_G["reps"])
    return {**m, "_zero": True, "source": sname, "target": tname, "space": space, "balanced": bal}


def stage_input(kind: str, workers: int, reps: int, out_name: str) -> None:
    out = out_dir(kind)
    prov = provenance(kind, "input")
    prov["reps"] = _G["reps"] = reps
    groups = groups_for(kind)
    for g in groups:
        df, X = load_any(g, kind)
        _G.setdefault("X", {})[g] = X
        _G.setdefault("y", {})[g] = df["label_4"].to_numpy()
        _G.setdefault("g", {})[g] = df["speaker_group"].astype(str).to_numpy()
    _G["logits"] = {}
    for g in groups:
        z = np.load(out / "logits" / f"{g}.npz")
        _G["logits"][g] = {k: sum_zero_coords(z[k]) for k in z.files}
        if len(_G["logits"][g]["oof"]) != len(_G["y"][g]):
            raise SystemExit(f"{g} の対数オッズの行数が特徴と合わない（gpu 段をやり直す）")
    pairs = [(s, t) for s in groups for t in groups if s != t]
    jobs = ([("pair", p, "raw", False) for p in pairs]
            + [("pair", p, sp, b) for p in pairs for sp in ("pca64", "logit") for b in (False, True)]
            + [("zero", g, sp, b) for g in groups for sp, b in
               (("raw", False), ("pca64", False), ("pca64", True))]
            + [("zero", p, "logit", b) for p in pairs for b in (False, True)])
    t0 = time.perf_counter()
    rows_in, rows_zero, errors = [], [], []
    with get_context("fork").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_task, jobs), 1):
            if "_error" in r:
                errors.append(r["_error"])
                print("失敗:", r["_error"], flush=True)
                continue
            (rows_zero if r.pop("_zero", False) else rows_in).append(r)
            if k % 20 == 0 or k == len(jobs):
                print(f"{kind} 軸 1・2：{k}/{len(jobs)}（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    prov["failed"] = errors
    write_csv(out / f"{out_name}input.csv", rows_in, prov)
    write_csv(out / f"{out_name}zero.csv", rows_zero, prov)
    if errors:
        raise SystemExit(f"{len(errors)} 件が失敗した（設定行の failed に記録．結果は書いた）")


# ---------------------------------------------------------------- 選択段：登録した規則

def stage_space(kind: str) -> None:
    out = out_dir(kind)
    inp = pd.read_csv(out / "input.csv", comment="#")
    zero = pd.read_csv(out / "zero.csv", comment="#", keep_default_na=False)
    res = {"kind": kind, "rule": "researchvision.md 8.2 共通の決まり 6", "pad_saturated": PAD_SATURATED}
    for space in ("raw", "pca64", "logit"):
        d = inp[(inp.space == space) & (~inp.balanced)]
        if space == "logit":
            z = zero[(zero.space == "logit") & (~zero.balanced)].set_index(["source", "target"])["pad"]
            zmed = float(np.median([z.loc[(r.source, r.target)] for r in d.itertuples()]))
        else:
            z = zero[(zero.space == space) & (~zero.balanced)].set_index("target")["pad"]
            zmed = float(np.median([z.loc[r.target] for r in d.itertuples()]))
        res[space] = {"pairs": int(len(d)), "unsaturated": int((d.pad < PAD_SATURATED).sum()),
                      "zero_pad_median": zmed, "pad_median": float(d.pad.median())}
    a, b = res["pca64"], res["logit"]
    if a["unsaturated"] != b["unsaturated"]:
        choice = "pca64" if a["unsaturated"] > b["unsaturated"] else "logit"
    elif a["zero_pad_median"] != b["zero_pad_median"]:
        choice = "pca64" if a["zero_pad_median"] < b["zero_pad_median"] else "logit"
    else:
        choice = "pca64"
    res["choice"] = choice
    res["decided"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "space_choice.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="ずれの指標を全組で測る")
    ap.add_argument("stage", choices=("gpu", "input", "space"))
    ap.add_argument("--kind", default="audio_wavlm")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--reps", type=int, default=shift.R_DEFAULT,
                    help="抜き出しの回数（既定 20．試しに流すときだけ減らし，出力名に try_ が付く）")
    a = ap.parse_args()
    name = "" if a.reps == shift.R_DEFAULT else "try_"
    {"gpu": lambda: stage_gpu(a.kind),
     "input": lambda: stage_input(a.kind, a.workers, a.reps, name),
     "space": lambda: stage_space(a.kind)}[a.stage]()


if __name__ == "__main__":
    main()
