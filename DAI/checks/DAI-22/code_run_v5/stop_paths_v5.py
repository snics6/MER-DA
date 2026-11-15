"""gate-code 再検査（DAI-22 第 5 版）：境目の入れ替わりの規則（near_tie）を偽の入力で確かめる．量の値は出さない（数え・種類・止まったかだけ）．
mode:
  real       MOSEI 分割 1・種 3 を偽の入力なしで流す（本番の 1 回目で止まった単位．near_tie で通るはず）
  same       MOSEI 分割 0・種 0 で実験 3 の保存の一致を 0.01 ずらし，本物の解き直しを通す（same で通るはず）
  over       同じずらしで，解き直しの戻り値を「厳しい 2 解は同じ，本番との違い floor(0.001 n)+1」にする（止まるはず）
  at         同じで違い floor(0.001 n)（near_tie で通るはず）
  tight      同じで違い (1, 2)，厳しい 2 解どうし 1（止まるはず）
  diff_lo / diff_hi  保存の一致を 0.09 / 0.11 ずらし，偽の解き直しで (1, 1, 0)（保存との差の条件．通る / 止まるはず）
  bias       同じずらしで，本物の解き直しのうち 1e-10 の解だけ b を動かす（n_pred_diff_tight が実際に 0 でなくなり，止まるはず）
  corp       MOSEI 分割 0・種 0〜3．種 0〜2 だけずらし，偽の解き直しで same 2 行・near_tie 4 行（違い (3,5),(7,2),(1,1),(4,4)）．
             種 3 は解き直しなし．コーパスを流し終えた時点で 6 / 8 > 4 で止まるはず
  corp_sum   corp の置き場で格子を分割 0 × 種 0〜3 に縮めた非 partial の summary（コーパスの上限で止まるはず）
  corp_part  corp の置き場で --allow-partial の summary（convergence.json を書くはず）
  real_part  real の置き場で --allow-partial の summary
"""
import json
import numpy as np
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai22  # noqa: E402

C = Path("/home/snics/MER-DA/DAI/checks/DAI-22/code_run_v5")
mode = sys.argv[1]
SHOW = ["T", "fold", "seed", "K", "n_cand", "tol_resolved", "resolve_kind", "n_pred_diff_tol1e-08", "n_pred_diff_tol1e-10", "n_pred_diff_tight",
        "stalled_tol1e-08", "stalled_tol1e-10"]


def run(argv):
    sys.argv = argv
    try:
        dai22.main()
        print(f"[{mode}] 最後まで通った")
    except SystemExit as e:
        print(f"[{mode}] 止まった：{e}")


def shift_saved(seeds, d=0.01):
    real_run = dai22.run_unit

    def run_unit(T, f, r, fz, out, repro):
        if r in seeds:
            repro = repro.copy(); repro["agree"] = repro["agree"] + d
        return real_run(T, f, r, fz, out, repro)
    dai22.run_unit = run_unit


def fake_resolve(seq):
    it = iter(seq)

    def rr(sv, K, qy, halves, unit_pred):
        a, b, t = next(it)
        n = len(unit_pred)
        a = a(n) if callable(a) else a; b = b(n) if callable(b) else b
        out = {}
        for tol, d in zip(dai22.TOL_RESOLVE, (a, b)):
            out[f"n_iter_tol{tol:g}"] = 1; out[f"stalled_tol{tol:g}"] = 0; out[f"n_pred_diff_tol{tol:g}"] = int(d); out[f"grad_tol{tol:g}"] = 0.0
        out["n_pred_diff_tight"] = int(t)
        out["_pred_tight"] = unit_pred.copy()          # 540bba2 の戻り値の形（偽の解き直しでは単位の予測をそのまま返す）
        return out
    dai22.resolve_r0 = rr


def show(od):
    for p in sorted((od / "units").glob("*.csv")):
        x = pd.read_csv(p); x = x[x.readout == "R0"]
        print(x[[c for c in SHOW if c in x]].to_string(index=False))
    print("check_fail：", sorted(p.name for p in od.glob("check_fail_*.csv")))
    import numpy as np
    sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
    import dai_solver as S
    for p in sorted((od / "units").glob("*.csv")):
        x = pd.read_csv(p); x = x[x.readout == "R0"]
        a = np.load(p.with_suffix(".npz"))
        for _, row in x.iterrows():
            K = int(row.K); key = f"K{K}__R0_tight"
            has = key in a.files
            res = "resolve_kind" in x and isinstance(row.get("resolve_kind"), str)
            if not res:
                print(f"  {p.stem} K={K}：解き直しなし．npz の {key} の有無 {has}，uar_diff 列が NaN {pd.isna(row.get('uar_diff_tight_vs_unit', np.nan))}")
                continue
            nk = a[f"nk__K{K}"].astype(bool); y = a["y"].astype(int)
            nd = int((a[key] != a[f"K{K}__R0"]).sum())
            ud = 100 * S.uar(y[nk], a[key].astype(int)[nk]) - 100 * S.uar(y[nk], a[f"K{K}__R0"].astype(int)[nk])
            print(f"  {p.stem} K={K}：npz に {key} {has}，R0_tight と R0 の違いの本数 {nd}（csv の 1e-08 の列と同じ {nd == row['n_pred_diff_tol1e-08']}），"
                  f"uar_diff を npz から作り直した差 < 1e-12 {abs(ud - row.uar_diff_tight_vs_unit) < 1e-12}，uar_diff が有限 {np.isfinite(row.uar_diff_tight_vs_unit)}，"
                  f"保存との差 × n / 100 ≤ max(2, 0.001 n) {row.abs_diff * row.n_cand / 100 <= max(2, 1e-3 * row.n_cand)}")


lim = lambda n: math.floor(1e-3 * n)
od = C / f"t_{mode}"
if mode == "real":
    run(["dai22.py", "--stage", "units", "--out-dir", str(od), "--targets", "MOSEI", "--folds", "1", "--seeds", "3", "--allow-dirty"])
    show(od)
elif mode in ("same", "over", "at", "tight", "bias", "diff_lo", "diff_hi"):
    shift_saved({0}, {"diff_lo": 0.09, "diff_hi": 0.11}.get(mode, 0.01))
    if mode in ("diff_lo", "diff_hi"):
        fake_resolve([(1, 1, 0)] * 2)
    if mode == "over":
        fake_resolve([(lambda n: lim(n) + 1, lambda n: lim(n) + 1, 0)] * 2)
    elif mode == "at":
        fake_resolve([(lim, lim, 0)] * 2)
    elif mode == "tight":
        fake_resolve([(1, 2, 1)] * 2)
    elif mode == "bias":
        real_fit = dai22.S.fit

        def fit(*a, **k):
            res = real_fit(*a, **k)
            if k.get("tol") == 1e-10:
                res["b"] = res["b"].clone(); res["b"][:, 0] += 3.0
            return res
        dai22.S.fit = fit
    run(["dai22.py", "--stage", "units", "--out-dir", str(od), "--targets", "MOSEI", "--folds", "0", "--seeds", "0", "--allow-dirty"])
    show(od)
elif mode == "corp":
    shift_saved({0, 1, 2})
    fake_resolve([(0, 0, 0), (0, 0, 0), (3, 5, 0), (7, 2, 0), (1, 1, 0), (4, 4, 0)])
    run(["dai22.py", "--stage", "units", "--out-dir", str(C / "t_corp"), "--targets", "MOSEI", "--folds", "0", "--seeds", "0", "1", "2", "3", "--allow-dirty"])
    show(C / "t_corp")
elif mode in ("corp_sum", "corp_part", "real_part"):
    d = C / ("t_real" if mode == "real_part" else "t_corp")
    if mode == "corp_sum":
        real_load = dai22.dai3.load_fz1

        def load(p):
            fz, sh = real_load(p)
            fz["MOSEI"]["n_seeds_eval"] = 4
            return fz, sh
        dai22.dai3.load_fz1 = load
        dai22.DD.N_FOLDS = 1
        dai22.TOL_PSEUDO = 1e9
        argv = ["dai22.py", "--targets", "MOSEI", "--stage", "summary", "--out-dir", str(d), "--allow-dirty"]
    else:
        argv = ["dai22.py", "--targets", "MOSEI", "--stage", "summary", "--out-dir", str(d), "--allow-dirty", "--allow-partial"]
    run(argv)
    cj = d / "summary" / "convergence.json"
    if cj.exists():
        j = json.load(open(cj))
        print({k: j[k] for k in ("n_rows", "n_units", "r0_saved_n_over_tol", "r0_n_resolved", "r0_n_near_tie", "r0_near_tie_pred_diff_max")})
        import glob
        cv = pd.concat([pd.read_csv(q) for q in sorted(glob.glob(str(d / "units" / "*.csv")))]); cv = cv[cv.readout == "R0"]
        m = float(cv.uar_diff_tight_vs_unit.abs().max()) if "uar_diff_tight_vs_unit" in cv else 0.0
        print("r0_uar_diff_tight_abs_max が単位の csv の絶対値の最大と同じ", j["r0_uar_diff_tight_abs_max"] == m, "有限", bool(np.isfinite(j["r0_uar_diff_tight_abs_max"])) if True else None)
    else:
        print("convergence.json なし．summary の中身：", sorted(p.name for p in (d / "summary").glob("*")) if (d / "summary").exists() else "置き場なし")
