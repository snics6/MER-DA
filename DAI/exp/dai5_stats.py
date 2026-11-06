"""DAI 実験 5 の detect と集計（設計 `DAI/docs/実験5_設計.md` 2・4・5・9 節）。区間・p 値は `dai1_stats.effect_stats` をそのまま使う。

使い方：
  python DAI/exp/dai5_stats.py detect [出力の置き場] [T,T,...]  ：選ぶ段の後・評価の段の前。実験 4 の detect と同じ手順を族 7（srcw − joint の 8 セル．K ∈ {5, 25}）に当てる。
      効果の見込みは X のテストを使わず，T = X の仮のターゲットの回で仮のターゲット c を 1 つ除いた 20 単位で β と joint の設定を選び直し，除いた c の 10 単位で測って 3 つの c で平均する。
  python DAI/exp/dai5_stats.py eval [出力の置き場]  ：評価の回の後。
      families7.csv（族 7．コーパスごとに Holm．探索の印つき），desc.csv（記述：srcw − none・kd_one・partner・poe_none・joint_default・srcw_b0・srcw_student・srcw_mod・srcw_src・srcw_top1・joint，
      β ごとの srcw(β) − joint と srcw(β) − srcw_b0，軸を分けた版の β の曲線，K = 0〜25 の全部．β = 0 が選ばれたセルは identical の印），
      regret.csv（取りこぼし：生徒ごと／4 生徒の平均で最良の β，2 組に分けた帰無の水準．joint も），weights.csv（(T, K) ごとの指数の割合・影響の割合（大きさ，判定を変える）・
      ばらつき（分割の中・間）・最も重い識別器の頻度・採点の順位・鋭さ．モダリティごと・ソースごとの和も），scores.csv（採点の妥当性：採点で選んだ識別器の取りこぼしと
      でたらめの水準（全体，モダリティの中，ソースの中），順位相関，同点の数．分割を単位にした区間），src_agree.csv（識別器どうしの候補での判定の一致率．群ごとの平均と採点の上位 2 個），
      poe_check.csv（K = 0 の srcw と実験 1 の tt:poe_none の判定の一致率．ラベルなし），rule_b_chance.csv（3 節 (b) の偶然の見込み．分割を単位にした符号の反転），
      teacher_check(.summary).csv（各識別器の候補での UAR と実験 1 の `_teacher_stats` の照合），teacher_sharp.csv（Q の鋭さと実験 1 の教師の鋭さ），gpu_counts.csv，pairing.csv（対の照合）。
      判定は int8 の保存から作り直し，行の uar と全部の問で照合した数を残す。
  python DAI/exp/dai5_stats.py premise [出力の置き場]  ：規則の前提の確かめ（実験 1 の仮のターゲットの回の段 3 の行だけを読む．評価の回は読まない．仮のターゲット c ごとに平均）。
"""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_select as SEL  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai3 as D3  # noqa: E402
import dai5 as D5  # noqa: E402
import dai_data as DD  # noqa: E402

KS_FAM = (5, 25)
KS_ALL = dai1.KS_ALL
ROOT = DD.ROOT
counts = ST.counts
MODS = D5.MODS


def jload(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sels_from(fz5T: dict) -> dict:
    return {int(K): float(v["beta"]) for K, v in fz5T["select"].items()}


def sels_student(fz5T: dict) -> dict:
    return {(k.split("|")[0], int(k.split("|")[1])): float(v["beta"]) for k, v in fz5T["select_student"].items()}


def sels_variants(fz5T: dict) -> dict:
    return {(k.split("|")[0], int(k.split("|")[1])): float(v["beta"]) for k, v in fz5T.get("select_variants", {}).items()}


def uar_fast(y: np.ndarray, p: np.ndarray) -> float:
    return 100.0 * float(np.mean([(p[y == k] == k).mean() for k in range(4) if (y == k).any()]))


def collect(out5: Path, in_dir: Path, tag: str, T: str, c: str, units, y_of, cl_of, sel_beta: dict, fz1T: dict, sel_joint=None,
            extras: bool = False, check_rows: bool = False, sel_student=None, sel_var=None, mask_fn=None, halves: dict | None = None):
    """単位の判定を集める：(生徒, K, 行の名前) → {(f, r): 塊ごとの数}。halves（dict）を渡すと，2 組 A・B での UAR も (生徒, K, 行) → {(f, r): (uarA, uarB)} に溜める。"""
    cells, mism, total = {}, 0, 0
    cname = "ALL" if tag == "eval" else c
    for f, r in units:
        name = f"{T}_{cname}_f{f}_r{r}"
        up = ST.UnitPreds(in_dir / tag / "units" / name)
        p5 = D5.P5(out5 / tag / "units" / name)
        y, cl = y_of(up.sid), cl_of(up.sid)
        mk_ = mask_fn(up.sid) if mask_fn is not None else np.ones(len(y), bool)
        if not (up.sid == p5.z["sample_test"].astype(str)).all():
            sys.exit(f"{name}: dai5 と実験 1 の保存のテストの発話が違う")
        hv = D3.test_halves(y, cl, up.sid) if halves is not None else None
        rows = pd.read_csv(out5 / tag / "units" / f"{name}_p5.csv", low_memory=False) if check_rows else None
        for s in DD.STUDENTS:
            sp = up.none(s, 0, None)
            for K in KS_ALL:
                def put(nm, pr):
                    cells.setdefault((s, K, nm), {})[(f, r)] = counts(y[mk_], pr[mk_], cl[mk_])
                    if hv is not None:
                        halves.setdefault((s, K, nm), {})[(f, r)] = (uar_fast(y[hv == 0], pr[hv == 0]), uar_fast(y[hv == 1], pr[hv == 1]))
                beta = sel_beta[K]
                put("srcw", p5.pred(s, K, beta))
                jj = (sel_joint or {}).get((s, K)) or D5.joint_sel(fz1T, s, K)
                put("joint", up.p3item(s, K, method="joint", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"]))
                if extras:
                    P6 = ST.unit_preds(fz1T, up, s, K, sp)
                    put("none", P6["none"]); put("joint_default", P6["default"]); put("partner", P6["partner"])
                    kd = fz1T["select"][f"{s}|{K}|kd|one"]
                    put("kd_one", up.p3item(s, K, method="kd", teacher="one", temp=kd["temp"], rho=kd["rho"]))
                    put("poe_none", up.p3item(s, K, method="tt", teacher="poe_none"))
                    put("srcw_b0", p5.pred(s, K, D5.DEFAULT_BETA))
                    if sel_student is not None:
                        put("srcw_student", p5.pred(s, K, sel_student[(s, K)]))
                    for b in D5.betas_for(K):
                        put(f"b|{b}", p5.pred(s, K, b))
                    for vj in ("one", "sel"):                              # joint の全格子（取りこぼし用．ρ 4 × a 3 × 版 2 = 24 点）
                        for rj in dai1.RHO_GRID:
                            for aj in dai1.A_GRID:
                                put(f"jg|{vj}|{rj}|{aj}", up.p3item(s, K, method="joint", teacher=vj, rho=rj, a=aj))
                    if K in D5.KS_VAR and sel_var is not None:
                        for meth, px, tagc in (("srcw_mod", "p5m", "m"), ("srcw_src", "p5s", "s")):
                            put(meth, p5.pred(s, K, sel_var[(meth, K)], px))
                            for b in D5.BETA_GRID:
                                put(f"{tagc}|{b}", p5.pred(s, K, b, px))
                    if K in D5.KS_TOP1:
                        put("srcw_top1", p5.pred(s, K, float("inf"), "p5o"))
                    if p5.has(s, K, D5.TETHER_PREFIX):                        # 引き戻しの曲線：srcw の選んだ β（p5t）と joint の選んだ点（実験 1 の p3t）の 14 点
                        it5 = [json.loads(x) for x in p5.z[f"{D5.TETHER_PREFIX}_{s}_K{K}_items"]]
                        it3 = [json.loads(x) for x in up.p3[f"p3t_{s}_K{K}_items"]]
                        for tcw, tcb in dai1.TETHER:
                            j5 = [j for j, x in enumerate(it5) if np.isclose(x["Cw"], tcw) and np.isclose(x["Cb"], tcb)]
                            j3 = [j for j, x in enumerate(it3) if x.get("method") == "joint_tether" and x.get("tag") == "prop_sel"
                                  and np.isclose(x["Cw"], tcw) and np.isclose(x["Cb"], tcb)]
                            if len(j5) != 1 or len(j3) != 1:
                                sys.exit(f"{name}: 引き戻しの点が 1 つに決まらない ({s}, K={K}, {tcw}, {tcb})")
                            put(f"tc|{tcw}|{tcb}", p5.z[f"{D5.TETHER_PREFIX}_{s}_K{K}_pred"][j5[0]])
                            put(f"tj|{tcw}|{tcb}", up.p3[f"p3t_{s}_K{K}_pred"][j3[0]])
                if check_rows:
                    mine = np.sort([ST.uar_full(counts(y, pr, cl)) for pr in p5.z[f"p5_{s}_K{K}_pred"]])
                    rr = rows[(rows.method == "srcw") & (rows.student == s) & (rows.K == K)]
                    theirs = np.sort(rr.uar.to_numpy(float))
                    total += 1
                    mism += int(len(mine) != len(theirs) or not np.allclose(mine, theirs, atol=1e-6))
    return cells, mism, total


def expected_effects(out5: Path, in_dir: Path, X: str, fz1: dict) -> dict:
    """評価のターゲット X の効果の見込み（X のテストを使わない）：T = X の仮のターゲットの回で，c を 1 つ除いた 20 単位で
    joint（`dai1_select.select_T`）と β（`pick5`）を選び直し，除いた c の 10 単位で srcw − joint を測って 3 つの c で平均する。"""
    tw = jload(in_dir / "pseudo" / f"{X}_tether_w.json")
    p1, p2, p3 = (SEL.load(in_dir, X, w) for w in (1, 2, 3))
    r5 = D5.load_rows(out5, "pseudo", X)
    r5 = r5[r5.method == "srcw"]
    acc = {}
    for c in DD.others(X):
        keep = [d[d.c != c] for d in (p1, p2, p3)]
        n_keep = len(SEL.unit_names(keep[2]))
        rj = SEL.select_T(tw, *keep, n_units=n_keep)
        dsel = SEL.delta_sel(keep[0], tw["tether"], n_keep)
        k5 = r5[r5.c != c]
        sel_b, sel_j = {}, {}
        for K in KS_ALL:
            dmean = float(np.mean([dsel[(s, K)] for s in DD.STUDENTS]))
            sel_b[K], _, _ = D5.pick5(k5[k5.K == K], dmean)
            v = rj["main"][str(K)]
            for s in DD.STUDENTS:
                j = rj["select"][f"{s}|{K}|joint|{v}"]
                sel_j[(s, K)] = dict(teacher=v, rho=j["rho"], a=j["a"])
        units = [(f, r) for f in range(DD.N_FOLDS) for r in (0, 1)]
        y_of, cl_of = ST.label_maps(c)
        cells, _, _ = collect(out5, in_dir, "pseudo", X, c, units, y_of, cl_of, sel_b, fz1[X], sel_joint=sel_j)
        for s in DD.STUDENTS:
            for K in KS_FAM:
                d = [ST.uar_full(cells[(s, K, "srcw")][u]) - ST.uar_full(cells[(s, K, "joint")][u]) for u in units]
                acc.setdefault(f"7|{s}|{K}", []).append(float(np.mean(d)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def detect(out5: Path, in_dir: Path, targets=DD.CORPORA) -> None:
    fzp = out5 / "select" / "frozen.json"
    frozen = jload(fzp)
    fz1 = jload(D5.FROZEN1)
    ev = out5 / "eval" / "units"
    if ev.exists() and any(ev.iterdir()):
        sys.exit("評価の回の出力がすでにある")
    commit, dirty = dai1.git_commit()
    if dirty:
        sys.exit("detect は未コミットの変更があると流さない")
    rng = np.random.default_rng(20261105)
    acc = {}
    for T in targets:
        sb = sels_from(frozen[T])
        for c in DD.others(T):
            g_eff, _ = ST.g_eff_of(c)
            units = [(f, r) for f in range(DD.N_FOLDS) for r in (0, 1)]
            y_of, cl_of = ST.label_maps(c)
            cells, mism, total = collect(out5, in_dir, "pseudo", T, c, units, y_of, cl_of, sb, fz1[T], check_rows=True, extras=True)
            for s in DD.STUDENTS:
                for K in KS_FAM:
                    e = ST.effect_stats(cells[(s, K, "srcw")], cells[(s, K, "joint")], g_eff, rng)
                    e2 = ST.effect_stats(cells[(s, K, "srcw")], cells[(s, K, "srcw_b0")], g_eff, rng)      # srcw − srcw_b0 の揺れ（3 節 (d) の半幅の見込み）
                    acc.setdefault(c, []).append(dict(T=T, family=7, student=s, K=K, **e, mismatch=mism, checked=total,
                                                      v_jk_b0=e2["v_jk"], v_seed_b0=e2["v_seed"], v_seed_all_b0=e2["v_seed_all"]))
            print(f"detect {T} c={c}: 照合の食い違い {mism}/{total}", flush=True)
    for X, lst in acc.items():
        df = pd.DataFrame(lst)
        g_eff, _ = ST.g_eff_of(X)
        crit95 = 1.959964 if g_eff >= 20 else float(st.t.ppf(0.975, max(1, int(np.floor(g_eff)) - 1)))
        df["v_cl"] = (df.v_jk - (df.v_seed_all - df.v_seed)).clip(lower=0)
        n_seeds = int(fz1[X]["n_seeds_eval"])
        delta = float(fz1[X]["detect"]["delta_same"])
        exp_eff = expected_effects(out5, in_dir, X, fz1) if X in targets else {}
        cells = {}
        for (s, K), g in df.groupby(["student", "K"]):
            vjk = float(g.v_cl.mean())
            vseed = float(g.v_seed_all.mean()) * 2 / n_seeds if K > 0 else 0.0
            se = float(np.sqrt(vjk + vseed))
            key = f"7|{s}|{K}"
            vjk0 = float((g.v_jk_b0 - (g.v_seed_all_b0 - g.v_seed_b0)).clip(lower=0).mean())
            vseed0 = float(g.v_seed_all_b0.mean()) * 2 / n_seeds if K > 0 else 0.0
            cells[key] = dict(se=se, v_cl_or_jk=vjk, v_seed=vseed, detectable_first=ST.crit_first(g_eff, len(KS_FAM) * 4) * se,
                              detectable_last=ST.crit_first(g_eff, 1) * se, halfwidth=crit95 * se,
                              halfwidth_b0=crit95 * float(np.sqrt(vjk0 + vseed0)), same_level_possible_b0=bool(crit95 * float(np.sqrt(vjk0 + vseed0)) < delta),
                              expected_effect=exp_eff.get(key, float("nan")))
        for key, v in cells.items():
            v["same_level_possible"] = bool(int(key.split("|")[2]) > 0 and v["halfwidth"] < delta)
            v["descriptive"] = bool(not np.isfinite(v["expected_effect"]) or v["detectable_first"] > abs(v["expected_effect"]))
        frozen.setdefault(X, {})["detect"] = dict(
            n_seeds_eval=n_seeds, g_eff=g_eff, delta_same=delta, cells=cells, n_cells=len(KS_FAM) * 4,
            delta_same_note="実験 1 の固定のファイルの値（比較の基準の種の間の分散だけに依る）",
            descriptive_cells=[k for k, v in cells.items() if v["descriptive"]],
            same_level_cells=[k for k, v in cells.items() if v["same_level_possible"]])
        print(f"{X}: 種 {n_seeds}，記述に回すセル {len(frozen[X]['detect']['descriptive_cells'])}/{len(cells)}，同じ水準を出しうるセル "
              f"{len(frozen[X]['detect']['same_level_cells'])}", flush=True)
    frozen["_meta"]["detect"] = "dai5_stats.py detect"
    frozen["_meta"]["detect_commit"] = commit
    json.dump(frozen, open(fzp, "w"), indent=1, ensure_ascii=False)
    if out5.resolve().parent == (ROOT / "results").resolve():
        shutil.copy(fzp, ROOT / "DAI" / "frozen" / f"{out5.name}_frozen.json")
    print("書いた:", fzp, dai1.sha(fzp))


# ------------------------------------------------------------------ 評価の回の集計

def guard_eval(out5: Path, fz1: dict) -> str:
    fzp = out5 / "select" / "frozen.json"
    cp = ROOT / "DAI" / "frozen" / f"{out5.name}_frozen.json"
    if not fzp.exists() or not cp.exists() or dai1.sha(fzp) != dai1.sha(cp):
        sys.exit(f"固定のファイルが無いか，{cp.relative_to(ROOT)} と一致しない")
    if not dai1.committed_clean(cp):
        sys.exit("固定のファイルの写しがコミットされていない")
    h, commits = dai1.sha(fzp), set()
    for T in DD.CORPORA:
        for f in range(DD.N_FOLDS):
            for r in range(int(fz1[T]["n_seeds_eval"])):
                base = out5 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}"
                for ext in ("npz", "csv"):
                    if not Path(f"{base}_p5.{ext}").exists():
                        sys.exit(f"評価の回の単位がそろっていない：{base.name}_p5.{ext}")
                z = np.load(f"{base}_p5.npz")
                if str(z["frozen_sha256"]) != h:
                    sys.exit(f"{base.name} の frozen_sha256 が固定のファイルと違う")
                commits.add(str(z["commit"]))
    if len(commits) != 1:
        sys.exit(f"評価の回の単位のコミットが 1 つでない：{sorted(commits)}")
    ucommit = commits.pop()
    if subprocess.run(["git", "merge-base", "--is-ancestor", ucommit, "HEAD"], cwd=ROOT).returncode != 0:
        sys.exit(f"評価の回の単位のコミット {ucommit[:10]} が HEAD の祖先でない")
    fixed = D5.CODE_FIXED + ["DAI/frozen/dai5_frozen.json", "DAI/frozen/dai1_frozen.json"]
    if subprocess.run(["git", "diff", "--quiet", ucommit, "HEAD", "--", *fixed], cwd=ROOT).returncode != 0 or \
            subprocess.run(["git", "diff", "--quiet", ucommit, "--", *fixed], cwd=ROOT).returncode != 0:
        sys.exit(f"評価の回の単位のコミット {ucommit[:10]} と今の間で {fixed} に差がある（stats の台本自身の変更は許す）")
    return ucommit


def pairing(out5: Path, in_dir: Path, fz1: dict) -> None:
    """同じ (T, f, r) で，テストの sample_id・K 本・塊の名前が実験 1 の保存と srcw の保存で一致するか（設計 5 節）。"""
    rows, bad = [], 0
    for T in DD.CORPORA:
        sp = pd.read_csv(ROOT / "DAI" / "splits" / f"{T}.csv", keep_default_na=False)
        cl = dict(zip(sp["sample_id"].astype(str), sp["cluster"].astype(str)))
        ymap = dict(zip(sp["sample_id"].astype(str), sp["label_4"].astype(int)))
        for f in range(DD.N_FOLDS):
            for r in range(int(fz1[T]["n_seeds_eval"])):
                name = f"{T}_ALL_f{f}_r{r}"
                z1, z5 = np.load(in_dir / "eval" / "units" / f"{name}_p1.npz"), np.load(out5 / "eval" / "units" / f"{name}_p5.npz")
                ok_t = bool((z1["sample_test"] == z5["sample_test"]).all())
                ok_k = all(bool((z1[f"sample_K{K}"] == z5[f"sample_K{K}"]).all()) for K in DD.KS)
                ok_c = [cl[x] for x in z1["sample_test"].astype(str)] == list(z5["cluster_test"].astype(str))
                up = ST.UnitPreds(in_dir / "eval" / "units" / name)
                y = np.array([ymap[x] for x in up.sid])
                r3 = pd.read_csv(in_dir / "eval" / "units" / f"{name}_p3.csv", low_memory=False)
                nbad = 0
                for s in DD.STUDENTS:
                    for K in KS_ALL:
                        jj = D5.joint_sel(fz1[T], s, K)
                        mine = dai1.S.uar(y, up.p3item(s, K, method="joint", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"]).astype(int)) * 100
                        q = r3[(r3.stage == 3) & (r3.student == s) & (r3.K == K) & (r3.method == "joint") & (r3.teacher == jj["teacher"])]
                        q = q[np.isclose(q.rho.astype(float), float(jj["rho"])) & np.isclose(q.a.astype(float), float(jj["a"]))]
                        nbad += int(len(q) != 1 or abs(float(q.uar.iloc[0]) - mine) > 1e-6)
                rows.append(dict(T=T, fold=f, seed=r, test_ok=ok_t, K_ok=ok_k, cluster_ok=ok_c, joint_uar_mismatch=nbad))
                bad += int(not (ok_t and ok_k and ok_c) or nbad > 0)
    (out5 / "stats").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out5 / "stats" / "pairing.csv", index=False)
    print(f"対の照合：一致しない単位 {bad}/{len(rows)}", flush=True)
    if bad:
        sys.exit("実験 1 の保存と srcw の保存でテスト・K 本・塊が一致しない単位がある（stats/pairing.csv）")


def holm(p):
    return ST.holm(np.asarray(p, float))


def theta_of(cell: dict) -> float:
    d = {}
    for (f, r), v in cell.items():
        d.setdefault(f, []).append(ST.uar_full(v))
    return ST.theta(d)


def theta_half(hcell: dict, which: int, folds_ok: set) -> float:
    d = {}
    for (f, r), v in hcell.items():
        if f in folds_ok:
            d.setdefault(f, []).append(v[which])
    return ST.theta(d) if d else float("nan")


def regret_rows(cells: dict, halves: dict, T: str, s: str, K: int, names_pts: list, chosen: str, label: str, g_eff: float, rng,
                folds_ok: set, delta: float, n_grid: int, th_override: dict | None = None, half_override=None) -> dict:
    """取りこぼし（設計 2 節）：(i) 同じテストで測った取りこぼし＝評価のテストで平均が最良の点 − 選んだ点（UAR の差，0 以上．選びの楽観を含む），
    (ii) 2 組で測った取りこぼし＝テストの組 A で最良を選んで組 B で測り，入れ替えて平均（選びの楽観を除いた推定）．規則は (ii) が δ 以上かで読む（統計の第 2 回 S8）。
    th_override・half_override を渡すと，最良の点の選び方だけを替えられる（4 生徒の平均で選ぶ版）。"""
    th = th_override if th_override is not None else {p: theta_of(cells[(s, K, p)]) for p in names_pts}
    best = max(names_pts, key=lambda p: (th[p], p))
    e = ST.effect_stats(cells[(s, K, best)], cells[(s, K, chosen)], g_eff, rng)
    sh = []
    if folds_ok:
        if half_override is not None:
            tA, tB = half_override
        else:
            tA = {p: theta_half(halves[(s, K, p)], 0, folds_ok) for p in names_pts}
            tB = {p: theta_half(halves[(s, K, p)], 1, folds_ok) for p in names_pts}
        bA = max(names_pts, key=lambda p: (tA[p], p)); bB = max(names_pts, key=lambda p: (tB[p], p))
        mA = {p: theta_half(halves[(s, K, p)], 0, folds_ok) for p in names_pts}      # 測るのは必ずこの生徒の値
        mB = {p: theta_half(halves[(s, K, p)], 1, folds_ok) for p in names_pts}
        sh = [mB[bA] - mB[chosen], mA[bB] - mA[chosen]]
    shr = float(np.mean(sh)) if sh else np.nan
    return dict(T=T, student=s, K=K, which=label, chosen=chosen, best=best, best_is_chosen=best == chosen, n_grid_points=n_grid,
                split_half_regret=shr, selection_optimism=(e["effect"] - shr) if sh else np.nan, split_half_folds_used=len(folds_ok), delta_same=delta,
                split_half_regret_ge_delta=bool(sh and shr >= delta),
                **{k: v for k, v in e.items() if k in ("effect", "lo", "hi", "se", "p", "method", "df", "n_units", "identical")})


def fold_vectors(cells: dict, s: str, K: int, A: str = "srcw", B: str = "srcw_b0") -> np.ndarray:
    """分割を単位にした A − B の差（種で平均）。"""
    byf = {}
    for (f, r) in cells[(s, K, A)]:
        byf.setdefault(f, []).append(ST.uar_full(cells[(s, K, A)][(f, r)]) - ST.uar_full(cells[(s, K, B)][(f, r)]))
    return np.array([np.mean(byf[f]) for f in sorted(byf)])


def rule_b_chance(vecs: dict) -> dict:
    """3 節 (b) の偶然の見込み（統計の第 2 回 S10）：vecs = {(T, s): 分割ごとの srcw − srcw_b0}（AFEW・MELD の音声・顔・言葉の K = 25．identical を除く）．
    同じコーパスの 3 セルには同じ符号の反転を，2 つのコーパスには別々の反転を当て（5 分割なら 32 × 32 通り），「4 以上が正」と「2 以下が正」の割合を出す。
    自己検査：独立な 6 セル（コーパスごとの反転でなくセルごとの反転）なら「4 以上」は 22/64 = 34.4%。"""
    Ts = sorted({t for t, _ in vecs})
    if not vecs:
        return dict(n_cells=0, chance_ge4=np.nan, chance_le2=np.nan, observed_positive=np.nan, selfcheck_indep_ge4=np.nan)
    F = len(next(iter(vecs.values())))
    pats = list(itertools.product([1, -1], repeat=F))
    ge4 = le2 = 0
    for combo in itertools.product(pats, repeat=len(Ts)):
        sg = {t: np.array(c) for t, c in zip(Ts, combo)}
        npos = sum(int((v * sg[t]).mean() > 0) for (t, _), v in vecs.items())
        ge4 += int(npos >= 4); le2 += int(npos <= 2)
    total = len(pats) ** len(Ts)
    # 自己検査：セルごとに独立に反転（6 セルなら 2^(6F) は多いので，各セルの符号は分割の反転で決まる 2 通りの平均の符号 → 近似せず，独立の二項で理論値を出す）
    n = len(vecs)
    indep = sum(__import__("math").comb(n, k) for k in range(4, n + 1)) / 2 ** n if n >= 4 else np.nan
    return dict(n_cells=n, chance_ge4=ge4 / total, chance_le2=le2 / total, observed_positive=int(sum(int(v.mean() > 0) for v in vecs.values())),
                selfcheck_indep_ge4=indep, n_patterns=total)


def scores_rows(p5: D5.P5, names: list, yt: np.ndarray, T: str, f: int, r: int, n_src: int) -> list:
    """採点の妥当性（設計 2 節）：単位ごとに，採点で選んだ識別器の取りこぼし（0 以下）とでたらめの水準，順位相関（全体，モダリティの中，ソースの中）。"""
    tp = p5.z["p5_src_test_pred"].astype(int)
    tu = np.array([uar_fast(yt, tp[j]) for j in range(len(names))])
    out = []
    nm_ = len(names) // n_src
    for K in KS_ALL:
        if K == 0:
            continue
        sc = p5.z[f"p5_K{K}_scores"]

        def pick(ix):                                                  # 採点の 1 位（同点は平均）のテストの UAR − その集合の最良
            top = ix[np.flatnonzero(sc[ix] >= sc[ix].max() - 1e-12)]
            return float(tu[top].mean() - tu[ix].max()), len(top)
        allix = np.arange(len(names))
        rg, nt = pick(allix)
        null_all = tu - tu.max()                                       # でたらめに 1 つ選んだときの取りこぼしの分布（一様）
        blocks_mod = [np.arange(m * n_src, (m + 1) * n_src) for m in range(nm_)]            # モダリティの中（ソースを選ぶ）
        blocks_src = [np.arange(si, len(names), n_src) for si in range(n_src)]              # ソースの中（モダリティを選ぶ）
        rg_mod = float(np.mean([pick(b)[0] for b in blocks_mod])); null_mod = float(np.mean([(tu[b] - tu[b].max()).mean() for b in blocks_mod]))
        rg_src = float(np.mean([pick(b)[0] for b in blocks_src])); null_src = float(np.mean([(tu[b] - tu[b].max()).mean() for b in blocks_src]))
        sp_all = float(pd.Series(sc).corr(pd.Series(tu), method="spearman")) if sc.std() > 0 and tu.std() > 0 else np.nan
        sp_mod = [float(pd.Series(sc[b]).corr(pd.Series(tu[b]), method="spearman")) for b in blocks_mod if sc[b].std() > 0 and tu[b].std() > 0]
        sp_src = [float(pd.Series(sc[b]).corr(pd.Series(tu[b]), method="spearman")) for b in blocks_src if sc[b].std() > 0 and tu[b].std() > 0]
        out.append(dict(T=T, fold=f, seed=r, K=K, regret_sel=rg, n_top_tied=nt, null_mean=float(null_all.mean()), null_q05=float(np.percentile(null_all, 5)),
                        sign_note="regret_sel は損（0 以下）．null_* はでたらめに 1 つ選んだときの損の平均と 5% 点（並べ替えと同じ分布）",
                        regret_sel_within_mod=rg_mod, null_within_mod=null_mod, regret_sel_within_src=rg_src, null_within_src=null_src,
                        spearman_all=sp_all, spearman_within_mod=float(np.mean(sp_mod)) if sp_mod else np.nan, n_blocks_mod_defined=len(sp_mod),
                        spearman_within_src=float(np.mean(sp_src)) if sp_src else np.nan, n_blocks_src_defined=len(sp_src),
                        score_all_tied=bool(sc.std() == 0), score_range=float(sc.max() - sc.min()), score_sd=float(sc.std()),
                        test_range=float(tu.max() - tu.min()), **{f"score_rank_{nm}": float(st.rankdata(-sc)[j]) for j, nm in enumerate(names)}))
    return out


def jk_folds(vals_by_fold: dict) -> dict:
    """分割を単位にした平均と jackknife の区間（自由度 = 分割の数 − 1）。"""
    v = np.array([np.mean(x) for x in vals_by_fold.values()], float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return dict(mean=float(v.mean()) if len(v) else np.nan, lo=np.nan, hi=np.nan, n_folds=len(v))
    G = len(v); jk = np.array([np.delete(v, i).mean() for i in range(G)])
    se = float(np.sqrt((G - 1) / G * ((jk - jk.mean()) ** 2).sum())); q = float(st.t.ppf(0.975, G - 1))
    return dict(mean=float(v.mean()), lo=float(v.mean() - q * se), hi=float(v.mean() + q * se), n_folds=G)


def evaluate(out5: Path, in_dir: Path) -> None:
    fz1 = jload(D5.FROZEN1)
    ucommit = guard_eval(out5, fz1)
    pairing(out5, in_dir, fz1)
    frozen = jload(out5 / "select" / "frozen.json")
    rng = np.random.default_rng(20261105)
    fam, desc, regret, wrows, srows, tcheck, sharp, agrows, poe = [], [], [], [], [], [], [], [], []
    bvecs = {}
    for T in DD.CORPORA:
        g_eff, _ = ST.g_eff_of(T)
        R = int(fz1[T]["n_seeds_eval"])
        units = [(f, r) for f in range(DD.N_FOLDS) for r in range(R)]
        y_of, cl_of = ST.label_maps(T)
        sb, ss, sv = sels_from(frozen[T]), sels_student(frozen[T]), sels_variants(frozen[T])
        halves = {}
        cells, mism, total = collect(out5, in_dir, "eval", T, T, units, y_of, cl_of, sb, fz1[T], extras=True, check_rows=True,
                                     sel_student=ss, sel_var=sv, halves=halves)
        det = frozen[T].get("detect", {}).get("cells", {})
        delta = float(fz1[T]["detect"]["delta_same"])
        ident = set()
        res = []
        for s in DD.STUDENTS:
            for K in KS_FAM:
                e = ST.effect_stats(cells[(s, K, "srcw")], cells[(s, K, "joint")], g_eff, rng)
                d = det.get(f"7|{s}|{K}", {})
                res.append(dict(family=7, exploratory=True, T=T, student=s, K=K, A="srcw", B="joint", beta=sb[K], g_eff=g_eff, mismatch=mism, checked=total,
                                descriptive=d.get("descriptive"), detectable_first=d.get("detectable_first"), **e))
        for x, a in zip(res, holm([x["p"] for x in res])):
            x["p_holm"] = float(a)
        fam += res
        # 2 組に分けた帰無の水準に使える分割（両組が 30 本以上で 4 感情がそろう）
        folds_ok = set()
        for f in range(DD.N_FOLDS):
            up = ST.UnitPreds(in_dir / "eval" / "units" / f"{T}_ALL_f{f}_r0")
            y = y_of(up.sid); hv = D3.test_halves(y, cl_of(up.sid), up.sid)
            if all(((hv == h).sum() >= ST.MIN_TEST) and (np.bincount(y[hv == h], minlength=4) > 0).all() for h in (0, 1)):
                folds_ok.add(f)
        for s in DD.STUDENTS:
            for K in KS_ALL:
                b_sel = sb[K]
                if K > 0 and b_sel == D5.DEFAULT_BETA:
                    ident.add((s, K))
                part = fz1[T]["partner"][f"{s}|{K}"]
                comps = ["none", "kd_one", "partner", "poe_none", "joint_default", "srcw_b0", "srcw_student", "joint"]
                if K in D5.KS_VAR:
                    comps += ["srcw_mod", "srcw_src"]
                if K in D5.KS_TOP1:
                    comps += ["srcw_top1"]
                for B in comps:
                    e = ST.effect_stats(cells[(s, K, "srcw")], cells[(s, K, B)], g_eff, rng)
                    if K == 0:
                        e["n_units"] = DD.N_FOLDS
                    rule_d = ""
                    if B == "srcw_b0" and s == "AVT" and K in KS_FAM and (s, K) not in ident:      # 3 節 (d)：1 通りの式（統計の第 2 回 S4）
                        rule_d = ("差が小さい" if (-delta < e["lo"] and e["hi"] < delta) else
                                  "見込みと違う" if ((e["lo"] > 0 or e["hi"] < 0) and abs(e["effect"]) > delta) else "判定できない")
                    desc.append(dict(T=T, student=s, K=K, A="srcw", B=B, beta=b_sel, beta_student=ss[(s, K)], read="", rule_d=rule_d,
                                     beta_variant=sv.get((B, K), np.nan) if B in ("srcw_mod", "srcw_src") else np.nan,
                                     identical_by_construction=bool((B == "srcw_b0" and (s, K) in ident) or
                                                                    (B in ("srcw_mod", "srcw_src") and (s, K) in ident and sv.get((B, K)) == D5.DEFAULT_BETA)),
                                     partner_method=part["method"] if B == "partner" else "", partner_version=part["version"] if B == "partner" else "",
                                     delta_same=delta, interval_within_delta=bool(K > 0 and -delta < e["lo"] and e["hi"] < delta), g_eff=g_eff, **e))
                for b in D5.betas_for(K):                                   # β の曲線（各 β − joint，各 β − β = 0）
                    for B, tag in (("joint", "beta_curve"), ("srcw_b0", "beta_vs_b0")):
                        e = ST.effect_stats(cells[(s, K, f"b|{b}")], cells[(s, K, B)], g_eff, rng)
                        if K == 0:
                            e["n_units"] = DD.N_FOLDS
                        desc.append(dict(T=T, student=s, K=K, A=f"srcw_beta{b}", B=B, beta=b, beta_student=ss[(s, K)], read=tag, beta_variant=np.nan,
                                         identical_by_construction=bool(B == "srcw_b0" and b == D5.DEFAULT_BETA), partner_method="", partner_version="",
                                         delta_same=delta, interval_within_delta=bool(K > 0 and -delta < e["lo"] and e["hi"] < delta), g_eff=g_eff, **e))
                if K in D5.KS_VAR:                                           # 軸を分けた版（選んだ β）− β = 0．両方 β = 0 なら同じ識別器
                    for meth in ("srcw_mod", "srcw_src"):
                        e = ST.effect_stats(cells[(s, K, meth)], cells[(s, K, "srcw_b0")], g_eff, rng)
                        desc.append(dict(T=T, student=s, K=K, A=meth, B="srcw_b0", beta=sv.get((meth, K), np.nan), beta_student=np.nan, read="variant_vs_b0",
                                         beta_variant=sv.get((meth, K), np.nan), identical_by_construction=bool(sv.get((meth, K)) == D5.DEFAULT_BETA),
                                         partner_method="", partner_version="", delta_same=delta, interval_within_delta=bool(-delta < e["lo"] and e["hi"] < delta), g_eff=g_eff, **e))
                if K in D5.KS_VAR:                                           # 軸を分けた版の β の曲線（各 β − β = 0）
                    for tagc, meth in (("m", "srcw_mod"), ("s", "srcw_src")):
                        for b in D5.BETAS_VAR:
                            e = ST.effect_stats(cells[(s, K, f"{tagc}|{b}")], cells[(s, K, "srcw_b0")], g_eff, rng)
                            desc.append(dict(T=T, student=s, K=K, A=f"{meth}_beta{b}", B="srcw_b0", beta=b, beta_student=np.nan, read="variant_curve",
                                             beta_variant=sv.get((meth, K), np.nan), identical_by_construction=False, partner_method="", partner_version="",
                                             delta_same=delta, interval_within_delta=bool(-delta < e["lo"] and e["hi"] < delta), g_eff=g_eff, **e))
                # 取りこぼし：生徒ごとに最良の β，4 生徒の平均で最良の β（選び方と同じ自由度），joint の格子
                pts = [f"b|{b}" for b in D5.betas_for(K)]
                chosen = f"b|{b_sel}" if K > 0 else "b|0.0"
                regret.append(regret_rows(cells, halves, T, s, K, pts, chosen, "srcw_per_student", g_eff, rng, folds_ok, delta, len(pts)))
                jpts = [f"jg|{vj}|{rj}|{aj}" for vj in ("one", "sel") for rj in dai1.RHO_GRID for aj in dai1.A_GRID]
                jj = D5.joint_sel(fz1[T], s, K)
                ja = int(jj["a"]) if float(jj["a"]).is_integer() else jj["a"]
                regret.append(regret_rows(cells, halves, T, s, K, jpts, f"jg|{jj['teacher']}|{jj['rho']}|{ja}", "joint_per_student", g_eff, rng, folds_ok, delta, len(jpts)))
                if (s, K, "tc|0.001|1.0") in cells:                              # 固定の点の照合（比較の第 2 回の記録 3）：固定の (C′, C_b) の点は本番の srcw・joint と同じ問
                    fcw, fcb = fz1[T]["tether"][f"{s}|{max(K, 1)}"]
                    for A_, B_ in (("srcw", f"tc|{fcw}|{fcb}"), ("joint", f"tj|{fcw}|{fcb}")):
                        ag = float(np.mean([(cells[(s, K, A_)][u][1] == cells[(s, K, B_)][u][1]).all() and (cells[(s, K, A_)][u][2] == cells[(s, K, B_)][u][2]).all() for u in cells[(s, K, A_)]]))
                        desc.append(dict(T=T, student=s, K=K, A=A_, B=B_, beta=b_sel, beta_student=np.nan, read="fixed_point_check", Cw=fcw, Cb=fcb,
                                         beta_variant=np.nan, identical_by_construction=True, partner_method="", partner_version="", delta_same=delta,
                                         interval_within_delta=False, g_eff=g_eff, effect=0.0, n_units=len(cells[(s, K, A_)]), fixed_point_counts_agree=ag))
                if (s, K, "tc|0.001|1.0") in cells:                              # 引き戻しの曲線（記述）：srcw の選んだ β − joint の選んだ点，14 点
                    for tcw, tcb in dai1.TETHER:
                        e = ST.effect_stats(cells[(s, K, f"tc|{tcw}|{tcb}")], cells[(s, K, f"tj|{tcw}|{tcb}")], g_eff, rng)
                        desc.append(dict(T=T, student=s, K=K, A="srcw_tether", B="joint_tether", beta=b_sel, beta_student=np.nan, read="tether_curve", Cw=tcw, Cb=tcb,
                                         beta_variant=np.nan, identical_by_construction=False, partner_method="", partner_version="", delta_same=delta,
                                         interval_within_delta=bool(-delta < e["lo"] and e["hi"] < delta), g_eff=g_eff, **e))
        # 4 生徒の平均で最良の β（選び方と同じ自由度）：最良の点を 4 生徒の平均で決め，各生徒の区間と 2 組の値を出す
        for K in KS_ALL:
            pts = [f"b|{b}" for b in D5.betas_for(K)]
            mean_th = {p: float(np.mean([theta_of(cells[(s, K, p)]) for s in DD.STUDENTS])) for p in pts}
            hA = {p: float(np.mean([theta_half(halves[(s, K, p)], 0, folds_ok) for s in DD.STUDENTS])) for p in pts} if folds_ok else None
            hB = {p: float(np.mean([theta_half(halves[(s, K, p)], 1, folds_ok) for s in DD.STUDENTS])) for p in pts} if folds_ok else None
            chosen = f"b|{sb[K]}" if K > 0 else "b|0.0"
            for s in DD.STUDENTS:
                regret.append(regret_rows(cells, halves, T, s, K, pts, chosen, "srcw_shared_best", g_eff, rng, folds_ok, delta, len(pts),
                                          th_override=mean_th, half_override=(hA, hB) if folds_ok else None))
        # 3 節 (b) の偶然の見込み（AFEW・MELD の K = 25 の音声・顔・言葉．ループの外で合わせて数える）
        if T in ("AFEW", "MELD"):
            for s_ in ("A", "V", "T"):
                if (s_, 25) not in ident:
                    bvecs[(T, s_)] = fold_vectors(cells, s_, 25)
        # 指数の割合の表，影響の割合，採点の妥当性，教師の検算，Q の鋭さ，識別器の一致率，K = 0 の tt:poe_none との一致
        acc_w, acc_i, acc_sh, acc_ag, acc_s, acc_top2, acc_top2n = {}, {}, [], [], [], {}, {}
        names = None
        for f, r in units:
            name = f"{T}_ALL_f{f}_r{r}"
            p5 = D5.P5(out5 / "eval" / "units" / name)
            up = ST.UnitPreds(in_dir / "eval" / "units" / name)
            names = [str(x) for x in p5.z["p5_src_names"]]
            n_src = len(DD.others(T))
            yt = y_of(p5.z["sample_test"].astype(str))
            ids = p5.z["sample_pool"].astype(str)
            p3rows = pd.read_csv(in_dir / "eval" / "units" / f"{name}_p3.csv", low_memory=False)
            rows5 = pd.read_csv(out5 / "eval" / "units" / f"{name}_p5.csv", low_memory=False)
            acc_sh.append(p5.z["p5_src_sharp"]); acc_ag.append(p5.z["p5_src_agree"])
            acc_s += scores_rows(p5, names, yt, T, f, r, n_src)
            for s in DD.STUDENTS:                                         # K = 0 の srcw と実験 1 の tt:poe_none の判定の一致（ラベルなし）
                jj = D5.joint_sel(fz1[T], s, 0)
                poe.append(dict(T=T, fold=f, seed=r, student=s, same_problem=bool(jj["rho"] == 1.0 and jj["a"] == 1.0),
                                pred_agree=float((p5.pred(s, 0, 0.0) == up.p3item(s, 0, method="tt", teacher="poe_none")).mean())))
            for K in KS_ALL:
                b = sb[K] if K else D5.DEFAULT_BETA
                w = p5.z[f"p5_K{K}_w_b{b}"]
                acc_w.setdefault(K, {}).setdefault(f, []).append(w / 3.0)
                acc_i.setdefault(K, []).append(p5.z[f"p5_K{K}_infl_b{b}"])
                if K:
                    sc = p5.z[f"p5_K{K}_scores"]
                    second = np.sort(sc)[-2]                                 # 重複を数えて 2 番目に高い採点以上の識別器の集合（同点は全部入れる．定義の第 3 回 1）
                    top = np.flatnonzero(sc >= second - 1e-12)
                    prs = [(i, j) for i in top for j in top if i < j]
                    acc_top2.setdefault(K, []).append(float(np.mean([p5.z["p5_src_agree"][i, j] for i, j in prs])) if prs else np.nan)
                    acc_top2n.setdefault(K, []).append(int(len(top) >= 3))
                nk = ~np.isin(ids, p5.z[f"sample_K{K}"].astype(str)) if K else np.ones(len(ids), bool)
                for j, nm in enumerate(names):
                    ref = p3rows[(p3rows.method == "_teacher_stats") & (p3rows.teacher.astype(str) == f"tt:src_{nm}") & (p3rows.K == K)]
                    if len(ref):
                        tcheck.append(dict(T=T, fold=f, seed=r, K=K, src=nm, uar_pool_dai1=float(ref.uar_pool.iloc[0])))
                ts = rows5[(rows5.method == "_teacher_stats") & (rows5.variant == "srcw") & (rows5.K == K) & np.isclose(rows5.beta.astype(float), float(b))]
                t0 = rows5[(rows5.method == "_teacher_stats") & (rows5.variant == "srcw") & (rows5.K == K) & np.isclose(rows5.beta.astype(float), 0.0)]
                ref1 = p3rows[(p3rows.method == "_teacher_stats") & (p3rows.teacher.astype(str) == "one") & (p3rows.K == K)]
                if len(ts) and len(ref1) and len(t0):
                    sharp.append(dict(T=T, fold=f, seed=r, K=K, beta=b, maxp_srcw=float(ts.maxp.iloc[0]), entropy_srcw=float(ts.entropy.iloc[0]),
                                      maxp_b0=float(t0.maxp.iloc[0]), entropy_b0=float(t0.entropy.iloc[0]), neff_w=float(ts.neff_w.iloc[0]),
                                      maxp_joint=float(ref1.maxp.iloc[0]), entropy_joint=float(ref1.entropy.iloc[0]),
                                      **{f"neff_{k}_srcw": float(ts[f"neff_{k}"].iloc[0]) for k in range(4)}))
        SH = np.stack(acc_sh).mean(0); AG = np.stack(acc_ag).mean(0)
        mod_of = lambda nm: nm.split("_")[-1]; src_of = lambda nm: nm.rsplit("_", 1)[0]       # noqa: E731
        groups = {"same_source": [], "same_modality": [], "other": []}
        for i, a in enumerate(names):
            for j, b_ in enumerate(names):
                if j > i:
                    g = "same_source" if src_of(a) == src_of(b_) else ("same_modality" if mod_of(a) == mod_of(b_) else "other")
                    groups[g].append(float(AG[i, j]))
                    agrows.append(dict(T=T, a=a, b=b_, agree=float(AG[i, j]), group=g, n_units=len(acc_ag)))
        gmean = {g: float(np.mean(v)) for g, v in groups.items() if v}
        for K, byf in acc_w.items():
            fold_means = {f: np.mean(np.stack(v), 0) for f, v in byf.items()}
            W = np.stack([x for v in byf.values() for x in v]); Wf = np.stack(list(fold_means.values()))
            sd_within = float(np.mean([np.std(np.stack(v), axis=0, ddof=1).mean() for v in byf.values() if len(v) > 1])) if R > 1 else np.nan
            I = np.stack(acc_i[K])                                             # (単位, |C|, 2)
            mag, flip = I[:, :, 0], I[:, :, 1]
            flip_share = flip / np.clip(flip.sum(1, keepdims=True), 1e-12, None)        # C の中で正規化した割合．生の割合（判定が変わる候補の割合）は flipraw_
            heavy = np.zeros(len(names))
            for x in W:
                top = np.flatnonzero(x >= x.max() - 1e-12)
                if len(top) < len(x):
                    heavy[top] += 1.0 / len(top)
            heavy /= len(W)
            ranks = np.zeros(len(names)) if K == 0 else np.mean([st.rankdata(-D5.P5(out5 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}").z[f"p5_K{K}_scores"]) for f, r in units], 0)
            row = dict(T=T, K=K, beta=sb[K] if K else 0.0, n_units=len(W), uniform_by_construction=bool(K == 0 or sb[K] == D5.DEFAULT_BETA),
                       neff_w_mean=float(np.mean([(x.sum() ** 2) / (x ** 2).sum() for x in W])),
                       sd_within_fold=sd_within, sd_between_fold=float(Wf.std(0, ddof=1).mean()) if len(Wf) > 1 else np.nan,
                       top2_agree=float(np.mean(acc_top2[K])) if K in acc_top2 else np.nan, n_top2_set_ge3=int(np.sum(acc_top2n[K])) if K in acc_top2n else 0,
                       group_agree_max=max(gmean.values()),
                       top2_more_than_groups=bool(K in acc_top2 and np.mean(acc_top2[K]) > max(gmean.values())),
                       **{f"agree_{g}": v for g, v in gmean.items()})
            for j, nm in enumerate(names):
                row.update({f"share_{nm}": float(W[:, j].mean()), f"mag_{nm}": float(mag[:, j].mean()), f"flip_{nm}": float(flip_share[:, j].mean()),
                            f"flipraw_{nm}": float(flip[:, j].mean()), f"sd_between_fold_{nm}": float(Wf[:, j].std(ddof=1)) if len(Wf) > 1 else np.nan,
                            f"heavy_{nm}": float(heavy[j]), f"rank_{nm}": float(ranks[j]), f"maxp_{nm}": float(SH[j, 0]), f"entropy_{nm}": float(SH[j, 1]),
                            f"logit_sd_{nm}": float(SH[j, 2])})
            for kind, M in (("share", W), ("mag", mag), ("flip", flip_share)):
                for m in MODS:
                    row[f"{kind}_mod_{m}"] = float(sum(M[:, j].mean() for j, nm in enumerate(names) if mod_of(nm) == m))
                for c in DD.others(T):
                    row[f"{kind}_src_{c}"] = float(sum(M[:, j].mean() for j, nm in enumerate(names) if src_of(nm) == c))
            row["max_abs_share_minus_mag"] = float(np.max([abs(W[:, j].mean() - mag[:, j].mean()) for j in range(len(names))]))
            wrows.append(row)
        # 採点の妥当性：分割を単位に（種で平均してから jackknife）．仮のターゲットの回の採点の範囲も添える
        dsc = pd.DataFrame(acc_s)
        pse = []
        for f_ in sorted((out5 / "pseudo" / "units").glob(f"{T}_*_p5.npz")):
            zz = np.load(f_)
            for K in KS_ALL:
                if K:
                    sc = zz[f"p5_K{K}_scores"]; pse.append(dict(K=K, rng=float(sc.max() - sc.min()), sd=float(sc.std())))
        pse = pd.DataFrame(pse).groupby("K").mean() if pse else None
        for K, g in dsc.groupby("K"):
            rec = dict(T=T, K=K, n_units=len(g), n_score_all_tied=int(g.score_all_tied.sum()), n_top_tied_mean=float(g.n_top_tied.mean()),
                       n_spearman_nan=int(g.spearman_all.isna().sum()))
            for col in ("regret_sel", "null_mean", "null_q05", "regret_sel_within_mod", "null_within_mod", "regret_sel_within_src", "null_within_src",
                        "spearman_all", "spearman_within_mod", "spearman_within_src", "score_range", "score_sd", "test_range"):
                j = jk_folds({f: g[g.fold == f][col].dropna().tolist() for f in sorted(g.fold.unique()) if g[g.fold == f][col].notna().any()})
                rec.update({f"{col}_mean": j["mean"], f"{col}_lo": j["lo"], f"{col}_hi": j["hi"]})
            for a_, b_, nm_ in (("regret_sel", "null_mean", "diff_vs_null"), ("regret_sel_within_mod", "null_within_mod", "diff_vs_null_within_mod"),
                                ("regret_sel_within_src", "null_within_src", "diff_vs_null_within_src")):
                j = jk_folds({f: (g[g.fold == f][a_] - g[g.fold == f][b_]).dropna().tolist() for f in sorted(g.fold.unique())})
                rec.update({f"{nm_}_mean": j["mean"], f"{nm_}_lo": j["lo"], f"{nm_}_hi": j["hi"]})
            rec["score_range_pseudo"] = float(pse.loc[K, "rng"]) if pse is not None and K in pse.index else np.nan
            rec["range_ratio_eval_over_pseudo"] = rec["score_range_mean"] / rec["score_range_pseudo"] if rec.get("score_range_pseudo") else np.nan
            rec["usable_rule"] = bool(K == 25 and np.isfinite(rec["regret_sel_mean"]) and rec["regret_sel_mean"] >= 0.5 * rec["null_mean_mean"]
                                      and np.isfinite(rec["diff_vs_null_lo"]) and rec["diff_vs_null_lo"] > 0)   # 取りこぼしは 0 以下．半分以下＝ −0.5·null 以上．分割ごとの対の差（採点 − でたらめ）の区間が 0 より上
            rec.update({f"rank_{nm}": float(g[f"score_rank_{nm}"].mean()) for nm in names})
            srows.append(rec)
        print(f"{T}: 単位 {len(units)}，実効の数 {g_eff:.1f}，照合の食い違い {mism}/{total}，identical {len(ident)}", flush=True)
    dest = out5 / "stats"
    dest.mkdir(exist_ok=True)
    shas = {f"{T}_ALL_f{f}_r{r}_p{w}.npz": dai1.sha(in_dir / "eval" / "units" / f"{T}_ALL_f{f}_r{r}_p{w}.npz")
            for T in DD.CORPORA for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"])) for w in (1, 2, 3)}
    gate_sha = jload(D5.GATE5).get("input_sha256", {})
    if gate_sha != shas:
        sys.exit("実験 1 の評価の回の p1・p2・p3 の sha256 が eval_gate_dai5.json の input_sha256 と違う")
    ish = dest / "eval_input_sha256.json"
    if ish.exists() and jload(ish) != shas:
        sys.exit("実験 1 の評価の回の保存の sha256 が前の集計のときと違う")
    json.dump(shas, open(ish, "w"), indent=1)
    shutil.copy(ish, ROOT / "DAI" / "frozen" / "dai5_eval_inputs_sha256.json")
    cm, dt = dai1.git_commit()
    allrows = pd.concat([pd.read_csv(out5 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}_p5.csv", low_memory=False)
                         for T in DD.CORPORA for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"]))], ignore_index=True)
    gpu = allrows.drop_duplicates(["T", "fold", "seed"]).groupby(["T", "gpu"]).size().rename("n_units").reset_index()
    gpu["pseudo_gpu"] = "|".join(frozen["_meta"].get("pseudo_gpu", []))
    gpu["same_as_pseudo"] = gpu.apply(lambda x: x["gpu"] in frozen["_meta"].get("pseudo_gpu", []), axis=1)
    chance = [rule_b_chance(bvecs)]
    tc = pd.DataFrame(tcheck)
    tcs = recheck_sources(out5, in_dir, fz1, tc) if len(tc) else pd.DataFrame()
    poe_s = pd.DataFrame(poe).groupby(["T", "student", "same_problem"]).agg(n=("pred_agree", "size"), pred_agree_mean=("pred_agree", "mean"),
                                                                           pred_agree_min=("pred_agree", "min")).reset_index()
    print(f"計算機ごとの単位の数：{gpu.groupby('gpu').n_units.sum().to_dict()}", flush=True)
    for nm, rows in (("families7", fam), ("desc", desc), ("regret", regret), ("weights", wrows), ("scores", srows), ("teacher_check", tcs),
                     ("teacher_sharp", sharp), ("src_agree", agrows), ("poe_check", poe_s), ("rule_b_chance", chance), ("gpu_counts", gpu)):
        pd.DataFrame(rows).assign(commit=cm, dirty=dt, unit_commit=ucommit).to_csv(dest / f"{nm}.csv", index=False)
    print("書いた:", dest, "（families7.csv，desc.csv，regret.csv，weights.csv，scores.csv）")


def recheck_sources(out5: Path, in_dir: Path, fz1: dict, tc: pd.DataFrame) -> pd.DataFrame:
    """評価の回の各ソースの識別器の候補での UAR（K 本を除く）を特徴から作り直し，実験 1 の tt:src の行と照合する（設計 4 節の教師の検算）。"""
    import torch
    out = []
    for T in DD.CORPORA:
        srcs = DD.others(T)
        for f in range(DD.N_FOLDS):
            for r in range(int(fz1[T]["n_seeds_eval"])):
                u = dai1.Unit(in_dir, T, T, srcs, f, r)
                names, LP, _ = D5.source_logprobs(u)
                for K in KS_ALL:
                    nk = np.setdiff1d(np.arange(u.n), u.kpos[K])
                    for j, nm in enumerate(names):
                        mine = 100.0 * dai1.S.uar(u.ypool[nk], LP[j].argmax(1).cpu().numpy()[nk])
                        ref = tc[(tc["T"] == T) & (tc.fold == f) & (tc.seed == r) & (tc.K == K) & (tc.src == nm)]
                        theirs = float(ref.uar_pool_dai1.iloc[0]) if len(ref) else float("nan")
                        out.append(dict(T=T, fold=f, seed=r, K=K, src=nm, uar_pool_mine=mine, uar_pool_dai1=theirs, diff=mine - theirs,
                                        match=bool(np.isfinite(theirs) and abs(mine - theirs) < 1e-6)))
                del u
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        dai1.Corpus.cache.clear()
    d = pd.DataFrame(out)
    s = d[d.uar_pool_dai1.notna()].groupby(["T", "K"]).agg(n=("match", "size"), n_mismatch=("match", lambda x: int((~x).sum())),
                                                           max_absdiff=("diff", lambda x: float(np.abs(x).max()))).reset_index()
    d.to_csv(out5 / "stats" / "teacher_check_units.csv", index=False)
    return s


def premise(out5: Path, in_dir: Path) -> None:
    """規則の前提の確かめ（設計 2 節・9 節．関門 0 の報告の 5 節の 1）：実験 1 の仮のターゲットの回の段 3 の行（`method=tt`，`teacher=src_<c>_<m>`）で，
    教師（1 つのソース × モダリティの識別器）の候補での UAR と，その疑似ラベルで学んだ生徒の UAR の，単位の中の順位相関（Spearman）と，
    教師の UAR が最も高い識別器が生徒の UAR も最も高い割合（同点は 1/t）を，仮のターゲット c ごとに平均してから (T, 生徒, K) にまとめる。
    単一モダリティの識別器だけの版（uni）と AVT を含む版（all）。評価の回の行は読まない。出力：<out5>/stats/premise_pseudo(.by_c).csv。"""
    rows = []
    for T in DD.CORPORA:
        for f in sorted((in_dir / "pseudo" / "units").glob(f"{T}_*_p3.csv")):
            r = pd.read_csv(f, low_memory=False)
            ts = r[(r.method == "_teacher_stats") & r.teacher.astype(str).str.startswith("tt:src_")]
            tt = r[(r.method == "tt") & r.teacher.astype(str).str.startswith("src_")]
            c, fold, seed = str(r.c.iloc[0]), int(r.fold.iloc[0]), int(r.seed.iloc[0])
            for (s, K), g in tt.groupby(["student", "K"]):
                pairs = []
                for _, row in g.iterrows():
                    nm = str(row.teacher)
                    ref = ts[(ts.student == s) & (ts.K == K) & (ts.teacher.astype(str) == "tt:" + nm)]
                    if len(ref) == 1:
                        pairs.append((nm, float(ref.uar_pool.iloc[0]), float(row.uar)))
                for tag, sel in (("all", pairs), ("uni", [p for p in pairs if not p[0].endswith("_AVT")])):
                    if len(sel) < 3:
                        continue
                    tu = np.array([p[1] for p in sel]); su = np.array([p[2] for p in sel])
                    rho = float(pd.Series(tu).corr(pd.Series(su), method="spearman")) if tu.std() > 0 and su.std() > 0 else np.nan
                    top_t = np.flatnonzero(tu >= tu.max() - 1e-12); top_s = int(np.argmax(su))
                    rows.append(dict(T=T, c=c, fold=fold, seed=seed, student=s, K=int(K), which=tag, n_teachers=len(sel), spearman=rho,
                                     top_agree=(1.0 / len(top_t)) if top_s in top_t else 0.0, n_top_tied=len(top_t), chance=1.0 / len(sel),
                                     top_teacher="|".join(sel[i][0] for i in top_t), top_student=sel[top_s][0],
                                     teacher_range=float(tu.max() - tu.min()), student_range=float(su.max() - su.min())))
    d = pd.DataFrame(rows)
    (out5 / "stats").mkdir(parents=True, exist_ok=True)
    d.to_csv(out5 / "stats" / "premise_pseudo_units.csv", index=False)
    byc = d.groupby(["T", "c", "student", "K", "which"]).agg(n_units=("seed", "size"), spearman_mean=("spearman", "mean"), n_spearman_nan=("spearman", lambda x: int(x.isna().sum())),
                                                             top_agree=("top_agree", "mean"), chance=("chance", "mean"), n_top_tied=("n_top_tied", "mean")).reset_index()
    agg = byc.groupby(["T", "student", "K", "which"]).agg(n_c=("c", "size"), spearman_mean=("spearman_mean", "mean"), top_agree_mean=("top_agree", "mean"),
                                                         top_agree_min_over_c=("top_agree", "min"), chance=("chance", "mean"),
                                                         n_spearman_nan=("n_spearman_nan", "sum")).reset_index()
    agg["below_chance_rule"] = (agg.K == 25) & (agg.top_agree_min_over_c <= agg.chance)
    cm, dt = dai1.git_commit()
    byc.assign(commit=cm, dirty=dt).to_csv(out5 / "stats" / "premise_pseudo_by_c.csv", index=False)
    agg.assign(commit=cm, dirty=dt).to_csv(out5 / "stats" / "premise_pseudo.csv", index=False)
    print(f"premise: 単位の行 {len(d)}，c ごとの行 {len(byc)}，セル {len(agg)}．書いた: {out5 / 'stats' / 'premise_pseudo.csv'}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else D5.OUT_DIR
    in_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else D5.IN_DIR          # 実験 1 の保存の置き場（既定は results/dai1）
    if mode == "detect":
        detect(out, in_dir, sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else list(DD.CORPORA))
    elif mode == "eval":
        evaluate(out, in_dir)
    elif mode == "premise":
        premise(out, in_dir)
    else:
        sys.exit("detect か eval か premise")
