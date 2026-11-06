"""DAI 実験 5 の detect と集計（設計 `DAI/docs/実験5_設計.md` 2・4・5・9 節）。区間・p 値は `dai1_stats.effect_stats` をそのまま使う。

使い方：
  python DAI/exp/dai5_stats.py detect [出力の置き場] [T,T,...]  ：選ぶ段の後・評価の段の前。実験 4 の detect と同じ手順を族 7（srcw − joint の 12 セル）に当てる。
      効果の見込みは X のテストを使わず，T = X の仮のターゲットの回で仮のターゲット c を 1 つ除いた 20 単位で β と joint の設定を選び直し，除いた c の 10 単位で測って 3 つの c で平均する。
  python DAI/exp/dai5_stats.py eval [出力の置き場]  ：評価の回の後。
      families7.csv（族 7．コーパスごとに Holm），desc.csv（記述：srcw − none・kd_one・partner・poe_none・joint_default・srcw_b0・srcw_student，β ごとの srcw(β) − joint，
      取りこぼし（srcw と joint の両方），K = 0〜25 の全部），weights.csv（(T, K) ごとの参照の割合 w_sm/3 の単位平均と最も重い識別器），scores.csv（採点の妥当性：
      単位の中の採点とテストの UAR の順位相関，最も高い識別器の一致），teacher_check(.summary).csv（評価の回の各識別器の候補での UAR と実験 1 の `_teacher_stats` の照合），
      teacher_sharp.csv（Q の鋭さと実験 1 の教師の鋭さ），gpu_counts.csv，pairing.csv（対の照合）。
      判定は int8 の保存から作り直し，行の uar と全部の問で照合した数を残す。
"""

from __future__ import annotations

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
import dai5 as D5  # noqa: E402
import dai_data as DD  # noqa: E402

KS_FAM = (0, 5, 25)
KS_ALL = dai1.KS_ALL
ROOT = DD.ROOT
counts = ST.counts


def sels_from(fz5T: dict) -> dict:
    return {int(K): float(v["beta"]) for K, v in fz5T["select"].items()}


def sels_student(fz5T: dict) -> dict:
    return {(k.split("|")[0], int(k.split("|")[1])): float(v["beta"]) for k, v in fz5T["select_student"].items()}


def collect(out5: Path, in_dir: Path, tag: str, T: str, c: str, units, y_of, cl_of, sel_beta: dict, fz1T: dict, sel_joint=None,
            extras: bool = False, check_rows: bool = False, sel_student=None, mask_fn=None):
    """単位の判定を集める：(生徒, K, 行の名前) → {(f, r): 塊ごとの数}。"""
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
        rows = pd.read_csv(out5 / tag / "units" / f"{name}_p5.csv", low_memory=False) if check_rows else None
        for s in DD.STUDENTS:
            sp = up.none(s, 0, None)
            for K in KS_ALL:
                put = lambda nm, pr: cells.setdefault((s, K, nm), {}).__setitem__((f, r), counts(y[mk_], pr[mk_], cl[mk_]))   # noqa: E731
                beta = sel_beta[K]
                put("srcw", p5.pred(s, K, beta))
                jj = (sel_joint or {}).get((s, K)) or D5.joint_sel(fz1T, s, K)
                put("joint", up.p3item(s, K, method="joint", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"]))
                if extras:
                    P6 = ST.unit_preds(fz1T, up, s, K, sp)
                    put("none", P6["none"])
                    put("joint_default", P6["default"])
                    put("partner", P6["partner"])
                    kd = fz1T["select"][f"{s}|{K}|kd|one"]
                    put("kd_one", up.p3item(s, K, method="kd", teacher="one", temp=kd["temp"], rho=kd["rho"]))
                    put("poe_none", up.p3item(s, K, method="tt", teacher="poe_none"))
                    put("srcw_b0", p5.pred(s, K, D5.DEFAULT_BETA))
                    if sel_student is not None:
                        put("srcw_student", p5.pred(s, K, sel_student[(s, K)]))
                    for b in D5.betas_for(K):
                        put(f"b|{b}", p5.pred(s, K, b))
                    vj = fz1T["main"][str(K)]
                    for rj in dai1.RHO_GRID:
                        for aj in dai1.A_GRID:
                            put(f"jg|{rj}|{aj}", up.p3item(s, K, method="joint", teacher=vj, rho=rj, a=aj))
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
    tw = json.load(open(in_dir / "pseudo" / f"{X}_tether_w.json"))
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
    frozen = json.load(open(fzp))
    fz1 = json.load(open(D5.FROZEN1))
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
            cells, mism, total = collect(out5, in_dir, "pseudo", T, c, units, y_of, cl_of, sb, fz1[T], check_rows=True)
            for s in DD.STUDENTS:
                for K in KS_FAM:
                    e = ST.effect_stats(cells[(s, K, "srcw")], cells[(s, K, "joint")], g_eff, rng)
                    acc.setdefault(c, []).append(dict(T=T, family=7, student=s, K=K, **e, mismatch=mism, checked=total))
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
            cells[key] = dict(se=se, v_cl_or_jk=vjk, v_seed=vseed, detectable_first=ST.crit_first(g_eff, 12) * se,
                              detectable_last=ST.crit_first(g_eff, 1) * se, halfwidth=crit95 * se,
                              expected_effect=exp_eff.get(key, float("nan")))
        for key, v in cells.items():
            v["same_level_possible"] = bool(int(key.split("|")[2]) > 0 and v["halfwidth"] < delta)
            v["descriptive"] = bool(not np.isfinite(v["expected_effect"]) or v["detectable_first"] > abs(v["expected_effect"]))
        frozen.setdefault(X, {})["detect"] = dict(
            n_seeds_eval=n_seeds, g_eff=g_eff, delta_same=delta, cells=cells,
            delta_same_note="実験 1 の固定のファイルの値（比較の基準の種の間の分散だけに依る）",
            descriptive_cells=[k for k, v in cells.items() if v["descriptive"]],
            same_level_cells=[k for k, v in cells.items() if v["same_level_possible"]])
        print(f"{X}: 種 {n_seeds}，記述に回すセル {len(frozen[X]['detect']['descriptive_cells'])}/12，同じ水準を出しうるセル "
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


def evaluate(out5: Path, in_dir: Path) -> None:
    fz1 = json.load(open(D5.FROZEN1))
    ucommit = guard_eval(out5, fz1)
    pairing(out5, in_dir, fz1)
    frozen = json.load(open(out5 / "select" / "frozen.json"))
    rng = np.random.default_rng(20261105)
    fam, desc, wrows, srows, tcheck, sharp = [], [], [], [], [], []
    for T in DD.CORPORA:
        g_eff, _ = ST.g_eff_of(T)
        R = int(fz1[T]["n_seeds_eval"])
        units = [(f, r) for f in range(DD.N_FOLDS) for r in range(R)]
        y_of, cl_of = ST.label_maps(T)
        sb, ss = sels_from(frozen[T]), sels_student(frozen[T])
        cells, mism, total = collect(out5, in_dir, "eval", T, T, units, y_of, cl_of, sb, fz1[T], extras=True, check_rows=True, sel_student=ss)
        det = frozen[T].get("detect", {}).get("cells", {})
        res = []
        for s in DD.STUDENTS:
            for K in KS_FAM:
                e = ST.effect_stats(cells[(s, K, "srcw")], cells[(s, K, "joint")], g_eff, rng)
                if K == 0:
                    e["n_units"] = DD.N_FOLDS
                d = det.get(f"7|{s}|{K}", {})
                res.append(dict(family=7, T=T, student=s, K=K, A="srcw", B="joint", beta=sb[K], g_eff=g_eff, mismatch=mism, checked=total,
                                descriptive=d.get("descriptive"), detectable_first=d.get("detectable_first"), **e))
        for x, a in zip(res, holm([x["p"] for x in res])):
            x["p_holm"] = float(a)
        fam += res
        delta = float(fz1[T]["detect"]["delta_same"])
        for s in DD.STUDENTS:
            for K in KS_ALL:
                means = {k[2]: np.mean([ST.uar_full(v) for v in cells[k].values()]) for k in cells if k[:2] == (s, K) and k[2].startswith("b|")}
                oracle = max(means, key=lambda g: (means[g], g))
                jmeans = {k[2]: np.mean([ST.uar_full(v) for v in cells[k].values()]) for k in cells if k[:2] == (s, K) and k[2].startswith("jg|")}
                joracle = max(jmeans, key=lambda g: (jmeans[g], g))
                part = fz1[T]["partner"][f"{s}|{K}"]
                comps = ["none", "kd_one", "partner", "poe_none", "joint_default", "srcw_b0", "srcw_student", "joint", oracle]
                for B in comps:
                    nm = "oracle" if B == oracle else B
                    e = ST.effect_stats(cells[(s, K, "srcw")], cells[(s, K, B)], g_eff, rng)
                    if K == 0:
                        e["n_units"] = DD.N_FOLDS
                    desc.append(dict(T=T, student=s, K=K, A="srcw", B=nm, beta=sb[K], beta_student=ss[(s, K)], oracle_point=oracle if nm == "oracle" else "",
                                     n_grid_points=len(means), partner_method=part["method"] if B == "partner" else "",
                                     partner_version=part["version"] if B == "partner" else "", delta_same=delta,
                                     interval_within_delta=bool(K > 0 and -delta < e["lo"] and e["hi"] < delta), g_eff=g_eff, **e))
                for b in D5.betas_for(K):                                   # β の曲線（各 β − joint）と，β − β = 0
                    for B, tag in (("joint", "beta_curve"), ("srcw_b0", "beta_vs_b0")):
                        e = ST.effect_stats(cells[(s, K, f"b|{b}")], cells[(s, K, B)], g_eff, rng)
                        if K == 0:
                            e["n_units"] = DD.N_FOLDS
                        desc.append(dict(T=T, student=s, K=K, A=f"srcw_beta{b}", B=B, beta=b, beta_student=ss[(s, K)], oracle_point="", n_grid_points=0,
                                         read=tag, delta_same=delta, interval_within_delta=bool(K > 0 and -delta < e["lo"] and e["hi"] < delta), g_eff=g_eff, **e))
                e = ST.effect_stats(cells[(s, K, "joint")], cells[(s, K, joracle)], g_eff, rng)     # joint の取りこぼし
                if K == 0:
                    e["n_units"] = DD.N_FOLDS
                desc.append(dict(T=T, student=s, K=K, A="joint", B="joint_oracle", beta=sb[K], beta_student=ss[(s, K)], oracle_point=joracle,
                                 n_grid_points=len(jmeans), read="regret", delta_same=delta,
                                 interval_within_delta=bool(K > 0 and -delta < e["lo"] and e["hi"] < delta), g_eff=g_eff, **e))
        # 重みの表，採点の妥当性，教師の検算（候補のラベルはここでだけ読む），Q の鋭さ
        acc_w, acc_s = {}, {}
        for f, r in units:
            name = f"{T}_ALL_f{f}_r{r}"
            p5 = D5.P5(out5 / "eval" / "units" / name)
            names = [str(x) for x in p5.z["p5_src_names"]]
            yt = y_of(p5.z["sample_test"].astype(str))
            tp = p5.z["p5_src_test_pred"].astype(int)
            test_uar = np.array([100.0 * dai1.S.uar(yt, tp[j]) for j in range(len(names))])
            ids = p5.z["sample_pool"].astype(str)
            yp = y_of(ids)
            p3rows = pd.read_csv(in_dir / "eval" / "units" / f"{name}_p3.csv", low_memory=False)
            rows5 = pd.read_csv(out5 / "eval" / "units" / f"{name}_p5.csv", low_memory=False)
            for K in KS_ALL:
                b = sb[K]
                w = p5.z[f"p5_K{K}_w_b{D5.DEFAULT_BETA if K == 0 else b}"]
                sc = p5.z[f"p5_K{K}_scores"]
                acc_w.setdefault(K, []).append(w / 3.0)
                if K:
                    rho = float(pd.Series(sc).corr(pd.Series(test_uar), method="spearman")) if np.std(sc) > 0 else np.nan
                    top_s, top_t = int(np.argmax(sc)), int(np.argmax(test_uar))
                    acc_s.setdefault(K, []).append(dict(spearman=rho, top_agree=float(top_s == top_t), top_score=names[top_s], top_test=names[top_t],
                                                        score_range=float(sc.max() - sc.min()), test_range=float(test_uar.max() - test_uar.min())))
                nk = ~np.isin(ids, p5.z[f"sample_K{K}"].astype(str)) if K else np.ones(len(ids), bool)
                for j, nm in enumerate(names):                              # 各識別器の候補での UAR を，実験 1 の tt:src 行と照合（判定は候補の特徴から作り直す）
                    ref = p3rows[(p3rows.method == "_teacher_stats") & (p3rows.teacher.astype(str) == f"tt:src_{nm}") & (p3rows.K == K)]
                    if len(ref):
                        tcheck.append(dict(T=T, fold=f, seed=r, K=K, src=nm, uar_pool_dai1=float(ref.uar_pool.iloc[0])))
                ts = rows5[(rows5.method == "_teacher_stats") & (rows5.K == K) & np.isclose(rows5.beta.astype(float), float(D5.DEFAULT_BETA if K == 0 else b))]
                ref1 = p3rows[(p3rows.method == "_teacher_stats") & (p3rows.teacher.astype(str) == "one") & (p3rows.K == K)]
                if len(ts) and len(ref1):
                    sharp.append(dict(T=T, fold=f, seed=r, K=K, beta=b, maxp_srcw=float(ts.maxp.iloc[0]), entropy_srcw=float(ts.entropy.iloc[0]),
                                      neff_w=float(ts.neff_w.iloc[0]), heaviest=str(ts.heaviest.iloc[0]),
                                      maxp_joint=float(ref1.maxp.iloc[0]), entropy_joint=float(ref1.entropy.iloc[0]),
                                      **{f"neff_{k}_srcw": float(ts[f"neff_{k}"].iloc[0]) for k in range(4)}))
            names_T = names
        for K, lst in acc_w.items():
            W = np.stack(lst)
            wrows.append(dict(T=T, K=K, beta=sb[K], n_units=len(lst), **{f"share_{nm}": float(W[:, j].mean()) for j, nm in enumerate(names_T)},
                              **{f"sd_{nm}": float(W[:, j].std(ddof=1)) for j, nm in enumerate(names_T)},
                              heaviest_mode=pd.Series([names_T[int(np.argmax(x))] for x in lst]).mode().iloc[0],
                              neff_w_mean=float(np.mean([(x.sum() ** 2) / (x ** 2).sum() for x in lst]))))
        for K, lst in acc_s.items():
            dfk = pd.DataFrame(lst)
            srows.append(dict(T=T, K=K, n_units=len(dfk), spearman_mean=float(dfk.spearman.mean()), top_agree=float(dfk.top_agree.mean()),
                              top_score_mode=dfk.top_score.mode().iloc[0], top_test_mode=dfk.top_test.mode().iloc[0],
                              score_range=float(dfk.score_range.mean()), test_range=float(dfk.test_range.mean())))
        print(f"{T}: 単位 {len(units)}，実効の数 {g_eff:.1f}，照合の食い違い {mism}/{total}", flush=True)
    dest = out5 / "stats"
    dest.mkdir(exist_ok=True)
    shas = {f"{T}_ALL_f{f}_r{r}_p{w}.npz": dai1.sha(in_dir / "eval" / "units" / f"{T}_ALL_f{f}_r{r}_p{w}.npz")
            for T in DD.CORPORA for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"])) for w in (1, 2, 3)}
    gate_sha = json.load(open(D5.GATE5)).get("input_sha256", {})
    if gate_sha != shas:
        sys.exit("実験 1 の評価の回の p1・p2・p3 の sha256 が eval_gate_dai5.json の input_sha256 と違う")
    ish = dest / "eval_input_sha256.json"
    if ish.exists() and json.load(open(ish)) != shas:
        sys.exit("実験 1 の評価の回の保存の sha256 が前の集計のときと違う")
    json.dump(shas, open(ish, "w"), indent=1)
    shutil.copy(ish, ROOT / "DAI" / "frozen" / "dai5_eval_inputs_sha256.json")
    cm, dt = dai1.git_commit()
    allrows = pd.concat([pd.read_csv(out5 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}_p5.csv", low_memory=False)
                         for T in DD.CORPORA for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"]))], ignore_index=True)
    gpu = allrows.drop_duplicates(["T", "fold", "seed"]).groupby(["T", "gpu"]).size().rename("n_units").reset_index()
    # 評価の回の教師の検算：各識別器の候補での UAR を候補のラベルから作り直して照合する（p5 の保存には候補の判定が無いので，特徴から作り直す）
    tc = pd.DataFrame(tcheck)
    tcs = recheck_sources(out5, in_dir, fz1, tc) if len(tc) else pd.DataFrame()
    print(f"計算機ごとの単位の数：{gpu.groupby('gpu').n_units.sum().to_dict()}", flush=True)
    for nm, rows in (("families7", fam), ("desc", desc), ("weights", wrows), ("scores", srows), ("teacher_check", tcs),
                     ("teacher_sharp", sharp), ("gpu_counts", gpu)):
        pd.DataFrame(rows).assign(commit=cm, dirty=dt, unit_commit=ucommit).to_csv(dest / f"{nm}.csv", index=False)
    print("書いた:", dest, "（families7.csv，desc.csv，weights.csv，scores.csv）")


def recheck_sources(out5: Path, in_dir: Path, fz1: dict, tc: pd.DataFrame) -> pd.DataFrame:
    """評価の回の各ソースの識別器の候補での UAR（K 本を除く）を特徴から作り直し，実験 1 の tt:src の行と照合する（設計 4 節の教師の検算）。"""
    import torch
    out = []
    for T in DD.CORPORA:
        srcs = DD.others(T)
        y_of, _ = ST.label_maps(T)
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


if __name__ == "__main__":
    mode = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else D5.OUT_DIR
    if mode == "detect":
        detect(out, D5.IN_DIR, sys.argv[3].split(",") if len(sys.argv) > 3 else list(DD.CORPORA))
    elif mode == "eval":
        evaluate(out, D5.IN_DIR)
    else:
        sys.exit("detect か eval")
