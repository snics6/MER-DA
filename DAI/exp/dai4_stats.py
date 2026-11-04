"""DAI 実験 4 の detect と集計（設計 `DAI/docs/実験4_設計.md` 2・3・5 節）。区間・p 値は `dai1_stats.effect_stats` をそのまま使う。

使い方：
  python DAI/exp/dai4_stats.py detect [出力の置き場] [T,T,...]  ：選ぶ段の後・評価の段の前。実験 1 の 9.2 節と同じ手順（`dai1_stats.detect` を族 6 に当てた）。
      族 6 = cons − joint の 12 セル（生徒 4 × K ∈ {0, 5, 25}）。評価のターゲット X ごとに，ほかの T の仮のターゲットの回の c = X の単位で
      標準誤差の見込み（塊の項と種の項）を測り，検出できる差（Holm の最初の段と最後の段）を出す。効果の見込みは X のテストを使わず，
      T = X の仮のターゲットの回で仮のターゲット c を 1 つ除いた 20 単位で (ρ, γ) と joint の設定を選び直し，除いた c の 10 単位で測って 3 つの c で平均する。
      検出できる差（最初の段）が効果の見込みの大きさを超えるセルは記述に回す。同じ水準の δ と種の数は実験 1 の固定のファイルの値を使う
      （比較の基準の種の間の分散だけに依り，この実験で変わらない）。結果は frozen.json の X の `detect` に書き，写しを置く。
  python DAI/exp/dai4_stats.py eval [出力の置き場]  ：評価の回の後。
      families6.csv（族 6．コーパスごとに Holm），desc.csv（記述．cons_tether − joint_tether の引き戻しの曲線（14 点），cons − none，kd_one，partner（セルごとに相手の名前と版），agree，cotrain，coreg，cons_g1，
      cons0（選択なしどうしは cons0 − joint_default の行），joint，joint_a1，ρ だけ選んだ版 cons_rho25 − joint_a1，ρ = 1 での γ の曲線，取りこぼし
      （cons − 格子で平均が最良の点，joint − joint の格子で最良の点．n_grid_points を添える），同じ水準の δ との比べ，対照 4 つとの差を 2 通り：
      B = cons_* は各対照が自分の選んだ点（read = own_selected），B = cons_*@cons は cons の選んだ (ρ, γ) の点（read = cons_selected．印だけの差），K = 0〜25 の全部），
      marks.csv（印の割合，印の有無ごとの教師の正解率・誤り・再現率・直す・壊す，正解の感情に置かれた重みの割合（cons と joint），D8 の測る量：生徒以外の ret どうしの
      判定の一致率と ret と AVT の none[K] の一致率，誤りの φ 係数と「同じ誤りをする割合 ÷ 独立なときの見込み」），marks_variants.csv（本番と対照の印の割合，感情ごと），
      sensitivity.csv（漏れの感度：subsets.csv の部分集合と実験 3 の筋書きの対での cons − joint），teacher_check(.summary).csv（評価の回の教師の UAR と実験 1 の
      `_teacher_stats` の照合），gpu_counts.csv，zero_weight_classes.csv，neff_lt30.csv，pairing.csv（対の照合：テスト・K 本・塊の名前）。
      判定は int8 の保存から作り直し，行の uar と全部の問で照合した数を残す。評価の回の単位のコミットが HEAD の祖先でないか，dai4.py・dai1.py・dai_solver.py・dai_data.py・両方の固定のファイルが単位のコミットと違えば止まる（stats の台本自身の変更は許す。結果の csv には今のコミットを書く）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_select as SEL  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai4 as D4  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

KS_FAM = (0, 5, 25)
KS_ALL = dai1.KS_ALL
ROOT = DD.ROOT


def joint_sel(fz1T: dict, s: str, K: int) -> dict:
    v = fz1T["main"][str(K)]
    j = fz1T["select"][f"{s}|{K}|joint|{v}"]
    return dict(teacher=v, rho=j["rho"], a=j["a"])


def collect(out4: Path, in_dir: Path, tag: str, T: str, c: str, units, y_of, cl_of, sel_cons: dict, fz1T: dict, sel_joint=None,
            extras: bool = False, check_rows: bool = False, sel_ctrl=None, sel_rho25=None, mask_fn=None):
    """単位の判定を集める：(生徒, K, 行の名前) → {(f, r): 塊ごとの数}。
    行の名前：cons（sel_cons[(s,K)] = (ρ, γ)），joint（実験 1 の主の版・選んだ設定．sel_joint で替えられる）．extras なら
    none，kd（実験 1 の固定のファイルの select の kd|one），cons0（選択なし ρ = 1，γ = 1/4），g|ρ|γ（格子の全点）も。"""
    cells, mism, total = {}, 0, 0
    cname = "ALL" if tag == "eval" else c
    for f, r in units:
        name = f"{T}_{cname}_f{f}_r{r}"
        up = ST.UnitPreds(in_dir / tag / "units" / name)
        p4 = D4.P4(out4 / tag / "units" / name)
        y, cl = y_of(up.sid), cl_of(up.sid)
        mk_ = mask_fn(up.sid) if mask_fn is not None else np.ones(len(y), bool)       # 部分集合（テストの発話の選び方）
        if not (up.sid == p4.z["sample_test"].astype(str)).all():
            sys.exit(f"{name}: dai4 と実験 1 の保存のテストの発話が違う")
        rows = pd.read_csv(out4 / tag / "units" / f"{name}_p4.csv", low_memory=False) if check_rows else None
        for s in DD.STUDENTS:
            sp = up.none(s, 0, None)
            for K in KS_ALL:
                put = lambda nm, pr: cells.setdefault((s, K, nm), {}).__setitem__((f, r), counts(y[mk_], pr[mk_], cl[mk_]))   # noqa: E731
                rho, gam = sel_cons[(s, K)]
                put("cons", p4.pred(s, K, rho, gam))
                jj = (sel_joint or {}).get((s, K)) or joint_sel(fz1T, s, K)
                put("joint", up.p3item(s, K, method="joint", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"]))
                if extras:
                    P6 = ST.unit_preds(fz1T, up, s, K, sp)                       # 実験 1 の none・default（joint の選択なし）・partner
                    put("none", P6["none"])
                    put("joint_default", P6["default"])
                    put("partner", P6["partner"])
                    kd = fz1T["select"][f"{s}|{K}|kd|one"]
                    put("kd_one", up.p3item(s, K, method="kd", teacher="one", temp=kd["temp"], rho=kd["rho"]))
                    ag = fz1T["select"][f"{s}|{K}|agree|one"]
                    put("agree", up.p3item(s, K, method="agree", teacher="one", tau=ag["tau"], f=ag["f"]))
                    if s == "AVT":
                        ct = fz1T["select"][f"{s}|{K}|cotrain|ct"]
                        put("cotrain", up.p3item(s, K, method="cotrain", teacher="ct", rho=ct["rho"]))
                    else:
                        put("cotrain", up.cotrain_single(s, K, fz1T["select"][f"{s}|{K}|cotrain|ct"]["rho"]))
                    put("coreg", up.coreg(s, K, fz1T["select"][f"{s}|{K}|coreg|cr"]["rho"]))
                    ja1 = fz1T["select"][f"{s}|{K}|joint_a1|one"]
                    put("joint_a1", up.p3item(s, K, method="joint", teacher="one", rho=ja1["rho"], a=1))
                    put("cons_g1", up.p3item(s, K, method="joint", teacher="one", rho=rho, a=1))      # γ = 1 に当たる参照（cons の選んだ ρ）
                    put("cons0", p4.pred(s, K, D4.DEFAULT["rho"], D4.DEFAULT["gamma"]))
                    rr25 = sel_rho25[(s, K)]
                    put("cons_rho25", p4.pred(s, K, rr25[0], rr25[1]))
                    for it in p4.items(s, K):
                        put(f"g|{it['rho']}|{it['gamma']}", p4.pred(s, K, it["rho"], it["gamma"]))
                    if p4.has(s, K, "p4t"):                                 # 引き戻しの曲線：cons の選んだ点（p4t）と joint の選んだ点（実験 1 の p3t）の 14 点
                        it4 = [json.loads(x) for x in p4.z[f"p4t_{s}_K{K}_items"]]
                        it3 = [json.loads(x) for x in up.p3[f"p3t_{s}_K{K}_items"]]
                        for tcw, tcb in dai1.TETHER:
                            j4 = [j for j, x in enumerate(it4) if np.isclose(x["Cw"], tcw) and np.isclose(x["Cb"], tcb)]
                            j3 = [j for j, x in enumerate(it3) if x.get("method") == "joint_tether" and x.get("tag") == "prop_sel"
                                  and np.isclose(x["Cw"], tcw) and np.isclose(x["Cb"], tcb)]
                            if len(j4) != 1 or len(j3) != 1:
                                sys.exit(f"{name}: 引き戻しの点が 1 つに決まらない ({s}, K={K}, {tcw}, {tcb})")
                            put(f"tc|{tcw}|{tcb}", p4.z[f"p4t_{s}_K{K}_pred"][j4[0]])
                            put(f"tj|{tcw}|{tcb}", up.p3[f"p3t_{s}_K{K}_pred"][j3[0]])
                    vj = fz1T["main"][str(K)]
                    for rj in dai1.RHO_GRID:                               # joint の全格子（取りこぼし用．主の版の ρ 4 × a 3）
                        for aj in dai1.A_GRID:
                            put(f"jg|{rj}|{aj}", up.p3item(s, K, method="joint", teacher=vj, rho=rj, a=aj))
                    for cn, px in D4.CTRL_PREFIX.items():                  # 対照（AVT の cons_self は無い）：(A) それぞれが選んだ点，(B) cons の選んだ (ρ, γ) の点（印だけの差）
                        if p4.has(s, K, px):
                            rc, gc = sel_ctrl[cn][(s, K)]
                            put(cn, p4.pred(s, K, rc, gc, px))
                            put(cn + "@cons", p4.pred(s, K, rho, gam, px))
                if check_rows:
                    mine = np.sort([ST.uar_full(counts(y, pr, cl)) for pr in p4.z[f"p4_{s}_K{K}_pred"]])
                    rr = rows[(rows.method == "cons") & (rows.student == s) & (rows.K == K)]
                    theirs = np.sort(rr.uar.to_numpy(float))
                    total += 1
                    mism += int(len(mine) != len(theirs) or not np.allclose(mine, theirs, atol=1e-6))
    return cells, mism, total


counts = ST.counts


def sels_from(fz4T: dict, meth: str = "cons") -> dict:
    """固定のファイルの select から (生徒, K) → (ρ, γ)。meth は cons（キー "s|K"），cons_self など，cons_rho25。"""
    out = {}
    for k, v in fz4T["select"].items():
        p = k.split("|")
        if (len(p) == 2 and meth == "cons") or (len(p) == 3 and p[2] == meth):
            out[(p[0], int(p[1]))] = (v["rho"], v["gamma"])
    return out


def expected_effects(out4: Path, in_dir: Path, X: str, fz1: dict) -> dict:
    """評価のターゲット X の効果の見込み（X のテストを使わない）：T = X の仮のターゲットの回で，c を 1 つ除いた 20 単位で
    joint（`dai1_select.select_T`）と cons（`pick4`）を選び直し，除いた c の 10 単位で cons − joint を測って 3 つの c で平均する。"""
    tw = json.load(open(in_dir / "pseudo" / f"{X}_tether_w.json"))
    p1, p2, p3 = (SEL.load(in_dir, X, w) for w in (1, 2, 3))
    r4 = D4.load_rows(out4, "pseudo", X)
    r4 = r4[r4.method == "cons"]
    acc = {}
    for c in DD.others(X):
        keep = [d[d.c != c] for d in (p1, p2, p3)]
        n_keep = len(SEL.unit_names(keep[2]))
        rj = SEL.select_T(tw, *keep, n_units=n_keep)
        dsel = SEL.delta_sel(keep[0], tw["tether"], n_keep)
        sel_c, sel_j = {}, {}
        k4 = r4[r4.c != c]
        for (s, K), g in k4.groupby(["student", "K"]):
            ch, _ = D4.pick4(g, dsel[(s, int(K))])
            sel_c[(s, int(K))] = (ch["rho"], ch["gamma"])
            v = rj["main"][str(int(K))]
            j = rj["select"][f"{s}|{int(K)}|joint|{v}"]
            sel_j[(s, int(K))] = dict(teacher=v, rho=j["rho"], a=j["a"])
        units = [(f, r) for f in range(DD.N_FOLDS) for r in (0, 1)]
        y_of, cl_of = ST.label_maps(c)
        cells, _, _ = collect(out4, in_dir, "pseudo", X, c, units, y_of, cl_of, sel_c, fz1[X], sel_joint=sel_j)
        for s in DD.STUDENTS:
            for K in KS_FAM:
                d = [ST.uar_full(cells[(s, K, "cons")][u]) - ST.uar_full(cells[(s, K, "joint")][u]) for u in units]
                acc.setdefault(f"6|{s}|{K}", []).append(float(np.mean(d)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def detect(out4: Path, in_dir: Path, targets=DD.CORPORA) -> None:
    fzp = out4 / "select" / "frozen.json"
    frozen = json.load(open(fzp))
    fz1 = json.load(open(D4.FROZEN1))
    ev = out4 / "eval" / "units"
    if ev.exists() and any(ev.iterdir()):
        sys.exit("評価の回の出力がすでにある")
    commit, dirty = dai1.git_commit()
    if dirty:
        sys.exit("detect は未コミットの変更があると流さない")
    rng = np.random.default_rng(20261105)
    acc = {}
    for T in targets:
        sc = sels_from(frozen[T])
        for c in DD.others(T):
            g_eff, _ = ST.g_eff_of(c)
            units = [(f, r) for f in range(DD.N_FOLDS) for r in (0, 1)]
            y_of, cl_of = ST.label_maps(c)
            cells, mism, total = collect(out4, in_dir, "pseudo", T, c, units, y_of, cl_of, sc, fz1[T], check_rows=True)
            for s in DD.STUDENTS:
                for K in KS_FAM:
                    e = ST.effect_stats(cells[(s, K, "cons")], cells[(s, K, "joint")], g_eff, rng)
                    acc.setdefault(c, []).append(dict(T=T, family=6, student=s, K=K, **e, mismatch=mism, checked=total))
            print(f"detect {T} c={c}: 照合の食い違い {mism}/{total}", flush=True)
    for X, lst in acc.items():
        df = pd.DataFrame(lst)
        g_eff, _ = ST.g_eff_of(X)
        crit95 = 1.959964 if g_eff >= 20 else float(st.t.ppf(0.975, max(1, int(np.floor(g_eff)) - 1)))
        df["v_cl"] = (df.v_jk - (df.v_seed_all - df.v_seed)).clip(lower=0)
        n_seeds = int(fz1[X]["n_seeds_eval"])
        delta = float(fz1[X]["detect"]["delta_same"])
        exp_eff = expected_effects(out4, in_dir, X, fz1) if X in targets else {}
        cells = {}
        for (s, K), g in df.groupby(["student", "K"]):
            vjk = float(g.v_cl.mean())
            vseed = float(g.v_seed_all.mean()) * 2 / n_seeds if K > 0 else 0.0
            se = float(np.sqrt(vjk + vseed))
            key = f"6|{s}|{K}"
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
    frozen["_meta"]["detect"] = "dai4_stats.py detect"
    frozen["_meta"]["detect_commit"] = commit
    json.dump(frozen, open(fzp, "w"), indent=1, ensure_ascii=False)
    if out4.resolve().parent == (ROOT / "results").resolve():
        shutil.copy(fzp, ROOT / "DAI" / "frozen" / f"{out4.name}_frozen.json")
    print("書いた:", fzp, dai1.sha(fzp))


# ------------------------------------------------------------------ 評価の回の集計

def guard_eval(out4: Path, fz1: dict) -> str:
    fzp = out4 / "select" / "frozen.json"
    cp = ROOT / "DAI" / "frozen" / f"{out4.name}_frozen.json"
    if not fzp.exists() or not cp.exists() or dai1.sha(fzp) != dai1.sha(cp):
        sys.exit(f"固定のファイルが無いか，{cp.relative_to(ROOT)} と一致しない")
    if not dai1.committed_clean(cp):
        sys.exit("固定のファイルの写しがコミットされていない")
    commit, dirty = dai1.git_commit()
    h, commits = dai1.sha(fzp), set()
    for T in DD.CORPORA:
        for f in range(DD.N_FOLDS):
            for r in range(int(fz1[T]["n_seeds_eval"])):
                base = out4 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}"
                for ext in ("npz", "csv"):
                    if not Path(f"{base}_p4.{ext}").exists():
                        sys.exit(f"評価の回の単位がそろっていない：{base.name}_p4.{ext}")
                z = np.load(f"{base}_p4.npz")
                if str(z["frozen_sha256"]) != h:
                    sys.exit(f"{base.name} の frozen_sha256 が固定のファイルと違う")
                commits.add(str(z["commit"]))
    if len(commits) != 1:
        sys.exit(f"評価の回の単位のコミットが 1 つでない：{sorted(commits)}")
    ucommit = commits.pop()
    if subprocess.run(["git", "merge-base", "--is-ancestor", ucommit, "HEAD"], cwd=ROOT).returncode != 0:
        sys.exit(f"評価の回の単位のコミット {ucommit[:10]} が HEAD の祖先でない")
    fixed = ["DAI/exp/dai4.py", "DAI/exp/dai1.py", "DAI/exp/dai_solver.py", "DAI/exp/dai_data.py", "DAI/frozen/dai4_frozen.json", "DAI/frozen/dai1_frozen.json"]
    if subprocess.run(["git", "diff", "--quiet", ucommit, "HEAD", "--", *fixed], cwd=ROOT).returncode != 0 or \
            subprocess.run(["git", "diff", "--quiet", ucommit, "--", *fixed], cwd=ROOT).returncode != 0:
        sys.exit(f"評価の回の単位のコミット {ucommit[:10]} と今の間で {fixed} に差がある（stats の台本自身の変更は許す）")
    return ucommit


def pairing(out4: Path, in_dir: Path, fz1: dict) -> None:
    """同じ (T, f, r) で，テストの sample_id・K 本の番号・塊の名前が実験 1 の保存と cons の保存で一致するか（設計 5 節）。
    塊の名前は cons の保存が持つ Unit の塊と，分割の表 `DAI/splits/<T>.csv` を比べる。一致しない単位の数を `stats/pairing.csv` に残し，あれば止まる。"""
    rows, bad = [], 0
    for T in DD.CORPORA:
        sp = pd.read_csv(ROOT / "DAI" / "splits" / f"{T}.csv", keep_default_na=False)
        cl = dict(zip(sp["sample_id"].astype(str), sp["cluster"].astype(str)))
        ymap = dict(zip(sp["sample_id"].astype(str), sp["label_4"].astype(int)))
        for f in range(DD.N_FOLDS):
            for r in range(int(fz1[T]["n_seeds_eval"])):
                name = f"{T}_ALL_f{f}_r{r}"
                z1, z4 = np.load(in_dir / "eval" / "units" / f"{name}_p1.npz"), np.load(out4 / "eval" / "units" / f"{name}_p4.npz")
                ok_t = bool((z1["sample_test"] == z4["sample_test"]).all())
                ok_k = all(bool((z1[f"sample_K{K}"] == z4[f"sample_K{K}"]).all()) for K in DD.KS)
                ok_c = [cl[x] for x in z1["sample_test"].astype(str)] == list(z4["cluster_test"].astype(str))
                # joint 側：p1 の sample_test の順で解釈した p3 の選んだ点の判定から作った UAR が，p3 の csv の行の uar と一致するか
                up = ST.UnitPreds(in_dir / "eval" / "units" / name)
                y = np.array([ymap[x] for x in up.sid])
                r3 = pd.read_csv(in_dir / "eval" / "units" / f"{name}_p3.csv", low_memory=False)
                nbad = 0
                for s in DD.STUDENTS:
                    for K in dai1.KS_ALL:
                        jj = joint_sel(fz1[T], s, K)
                        mine = S.uar(y, up.p3item(s, K, method="joint", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"]).astype(int)) * 100
                        q = r3[(r3.stage == 3) & (r3.student == s) & (r3.K == K) & (r3.method == "joint") & (r3.teacher == jj["teacher"])]
                        q = q[np.isclose(q.rho.astype(float), float(jj["rho"])) & np.isclose(q.a.astype(float), float(jj["a"]))]
                        nbad += int(len(q) != 1 or abs(float(q.uar.iloc[0]) - mine) > 1e-6)
                rows.append(dict(T=T, fold=f, seed=r, test_ok=ok_t, K_ok=ok_k, cluster_ok=ok_c, joint_uar_mismatch=nbad))
                bad += int(not (ok_t and ok_k and ok_c) or nbad > 0)
    (out4 / "stats").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out4 / "stats" / "pairing.csv", index=False)
    print(f"対の照合：一致しない単位 {bad}/{len(rows)}", flush=True)
    if bad:
        sys.exit("実験 1 の保存と cons の保存でテスト・K 本・塊が一致しない単位がある（stats/pairing.csv）")


def err_stats(a: np.ndarray, b: np.ndarray, y: np.ndarray, nk: np.ndarray):
    """2 つの識別器 a，b の判定の誤り（真のラベルと違う）の 0/1 の φ 係数と，「両方が同じ誤りをする割合 ÷ 誤りがラベルの下で独立なときの見込み」。
    見込み = Σ_k π_k Σ_{j≠k} P(a = j | y = k) P(b = j | y = k)（候補で数える。K 本を除く）。"""
    a, b, y = a[nk], b[nk], y[nk]
    ea, eb = (a != y).astype(float), (b != y).astype(float)
    phi = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 0 and eb.std() > 0 else float("nan")
    obs = float(((a == b) & (a != y)).mean())
    exp_ = 0.0
    for k in range(4):
        g = y == k
        if g.any():
            exp_ += g.mean() * sum((a[g] == j).mean() * (b[g] == j).mean() for j in range(4) if j != k)
    return phi, (obs / exp_ if exp_ > 0 else float("nan"))


SUBSETS = {"AFEW": [("asr_excluded", "asr_suspect", None, ("T", "AVT"))],
           "MELD": [("face_single", "face_compete", "single", ("V", "AVT"))],
           "IEMOCAP": [("marker", "marker", "marker", dai1.DD.STUDENTS), ("nomarker", "marker", "nomarker", dai1.DD.STUDENTS)]}
PAIRS = [("05", {5}, (3, 0)), ("02_06", {2, 6}, (2, 0)), ("03_07", {3, 7}, (1, 0))]            # 筋書きの番号と感情の対（Angry-Neutral，Sad-Neutral，Happy-Neutral）
MIN_SUB = ST.MIN_TEST


def sensitivity(out4, in_dir, T, units, y_of, cl_of, fz4T, fz1T, g_eff, rng, sel_ctrl) -> list:
    """漏れの感度（設計 2 節の L12）：実験 1 の 9.4 節の部分集合（`DAI/frozen/subsets.csv`）と，IEMOCAP 即興の実験 3 の筋書きの対（生徒 T・AVT，K = 5）で
    cons − joint を記述として出す。部分集合の 1 回のテストが 30 本未満か感情が欠ける分割は，その量から外す（実験 1 の 9.4 節の規則）．全部が外れたセルは行だけ残す。"""
    out = []
    sc = sels_from(fz4T)
    sub = pd.read_csv(ROOT / "DAI" / "frozen" / "subsets.csv", keep_default_na=False)
    sub = sub[sub["corpus"] == T]
    for name, sname, val, studs in SUBSETS.get(T, []):
        d = sub[sub["subset"] == sname]
        if name == "asr_excluded":
            drop = set(d["sample_id"])
            fn = lambda sid, drop=drop: np.array([x not in drop for x in sid])      # noqa: E731
        else:
            keep = set(d[d["value"] == val]["sample_id"])
            fn = lambda sid, keep=keep: np.array([x in keep for x in sid])          # noqa: E731
        cells, _, _ = collect(out4, in_dir, "eval", T, T, units, y_of, cl_of, sc, fz1T, mask_fn=fn)
        for s in studs:
            for K in KS_FAM:
                A, B = cells[(s, K, "cons")], cells[(s, K, "joint")]
                bad = set()
                for (f, r), (_, cor, tot) in A.items():
                    tc = tot.sum(0)
                    if tc.sum() < MIN_SUB or (tc == 0).any():
                        bad.add(f)
                ks = [u for u in A if u[0] not in bad]
                row = dict(T=T, subset=name, student=s, K=K, folds_dropped=",".join(map(str, sorted(bad))), n_units=len(ks),
                           n_test_by_fold=json.dumps({f: int(A[(f, 0)][2].sum()) for f in sorted({u[0] for u in A})}))
                if ks:
                    e = ST.effect_stats({u: A[u] for u in ks}, {u: B[u] for u in ks}, g_eff, rng)
                    row.update({k: v for k, v in e.items() if k in ("effect", "lo", "hi", "se", "method", "df", "n_folds")})
                out.append(row)
    if T == "IEMOCAP":                                                                      # 筋書きの対（実験 3 の設計 3 節の 5）
        import re
        for s in ("T", "AVT"):
            K = 5
            rows_ = {pn: {} for pn, _, _ in PAIRS}
            for f, r in units:
                name = f"{T}_ALL_f{f}_r{r}"
                up = ST.UnitPreds(in_dir / "eval" / "units" / name)
                p4 = D4.P4(out4 / "eval" / "units" / name)
                y = y_of(up.sid)
                scr = np.array([int(re.search(r"impro(\d\d)", x).group(1)) for x in up.sid])
                pc = p4.pred(s, K, *sc[(s, K)])
                jj = joint_sel(fz1T, s, K)
                pj = up.p3item(s, K, method="joint", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"])
                for pn, scs, (c1, c2) in PAIRS:
                    m = np.isin(scr, list(scs))
                    n1, n2 = int((m & (y == c1)).sum()), int((m & (y == c2)).sum())
                    if min(n1, n2) < 10:
                        rows_[pn][(f, r)] = None
                        continue
                    rec = lambda p: np.mean([(p[m & (y == c)] == c).mean() for c in (c1, c2)]) * 100       # noqa: E731
                    rows_[pn][(f, r)] = (rec(pc) - rec(pj), n1 + n2)
            for pn, _, _ in PAIRS:
                byf = {}
                for (f, r), v in rows_[pn].items():
                    if v is not None:
                        byf.setdefault(f, []).append(v)
                th = {f: (float(np.mean([x[0] for x in v])), v[0][1]) for f, v in byf.items()}
                row = dict(T=T, subset=f"script_pair_{pn}", student=s, K=K, n_folds=len(th), folds_kept=",".join(map(str, sorted(th))))
                if len(th) >= 2:
                    vals = np.array([v[0] for v in th.values()]); ns = np.array([v[1] for v in th.values()], float)
                    est = float(vals.mean())
                    jk = np.array([np.delete(vals, i).mean() for i in range(len(vals))])
                    G = len(vals)
                    se = float(np.sqrt((G - 1) / G * ((jk - jk.mean()) ** 2).sum()))
                    geff = float(ns.sum() ** 2 / (ns ** 2).sum())
                    df = max(1, int(np.floor(geff)) - 1)
                    q = float(st.t.ppf(0.975, df))
                    row.update(effect=est, se=se, lo=est - q * se, hi=est + q * se, df=df, method="jk_folds")
                elif th:
                    row.update(effect=float(list(th.values())[0][0]))
                out.append(row)
    return out


def holm(p):
    return ST.holm(np.asarray(p, float))


def evaluate(out4: Path, in_dir: Path) -> None:
    fz1 = json.load(open(D4.FROZEN1))
    ucommit = guard_eval(out4, fz1)
    pairing(out4, in_dir, fz1)
    frozen = json.load(open(out4 / "select" / "frozen.json"))
    rng = np.random.default_rng(20261105)
    fam, desc, marks, tcheck, marks_var, sens = [], [], [], [], [], []
    for T in DD.CORPORA:
        g_eff, _ = ST.g_eff_of(T)
        R = int(fz1[T]["n_seeds_eval"])
        units = [(f, r) for f in range(DD.N_FOLDS) for r in range(R)]
        y_of, cl_of = ST.label_maps(T)
        sc = sels_from(frozen[T])
        sel_ctrl = {m: sels_from(frozen[T], m) for m in D4.CTRL_PREFIX}
        cells, mism, total = collect(out4, in_dir, "eval", T, T, units, y_of, cl_of, sc, fz1[T], extras=True, check_rows=True,
                                     sel_ctrl=sel_ctrl, sel_rho25=sels_from(frozen[T], "cons_rho25"))
        det = frozen[T].get("detect", {}).get("cells", {})
        res = []
        for s in DD.STUDENTS:
            for K in KS_FAM:
                e = ST.effect_stats(cells[(s, K, "cons")], cells[(s, K, "joint")], g_eff, rng)
                if K == 0:
                    e["n_units"] = DD.N_FOLDS
                d = det.get(f"6|{s}|{K}", {})
                res.append(dict(family=6, T=T, student=s, K=K, A="cons", B="joint", g_eff=g_eff, mismatch=mism, checked=total,
                                descriptive=d.get("descriptive"), detectable_first=d.get("detectable_first"), **e))
        for x, a in zip(res, holm([x["p"] for x in res])):
            x["p_holm"] = float(a)
        fam += res
        delta = float(fz1[T]["detect"]["delta_same"])
        for s in DD.STUDENTS:
            for K in KS_ALL:
                means = {k[2]: np.mean([ST.uar_full(v) for v in cells[k].values()]) for k in cells if k[:2] == (s, K) and k[2].startswith("g|")}
                oracle = max(means, key=lambda g: (means[g], g))
                pj = {k[2]: v for k, v in cells.items() if k[:2] == (s, K) and k[2].startswith("jg|")}
                jmeans = {k: np.mean([ST.uar_full(c_) for c_ in v.values()]) for k, v in pj.items()}
                joracle = max(jmeans, key=lambda g: (jmeans[g], g))
                part = fz1[T]["partner"][f"{s}|{K}"]
                comps = ["none", "kd_one", "partner", "agree", "cotrain", "coreg", "cons_g1", "cons0", "joint", "joint_default", "joint_a1", oracle]
                for cn in D4.CTRL_PREFIX:
                    if (s, K, cn) in cells:
                        comps += [cn, cn + "@cons"]
                for B in comps:
                    nm = "oracle" if B == oracle else B
                    e = ST.effect_stats(cells[(s, K, "cons")], cells[(s, K, B)], g_eff, rng)
                    if K == 0:
                        e["n_units"] = DD.N_FOLDS
                    desc.append(dict(T=T, student=s, K=K, A="cons", B=nm, oracle_point=oracle if nm == "oracle" else "", n_grid_points=len(means),
                                     read=("own_selected" if B in D4.CTRL_PREFIX else "cons_selected" if B.endswith("@cons") else ""),
                                     partner_method=part["method"] if B == "partner" else "", partner_version=part["version"] if B == "partner" else "",
                                     delta_same=delta, interval_within_delta=bool(K > 0 and -delta < e["lo"] and e["hi"] < delta),
                                     g_eff=g_eff, **e))
                # 取りこぼし（評価のテストで平均が最良の格子の点 − 選んだ点）を joint でも出す（格子の実効の点の数を添える：joint の選択は大半が (ρ = 1, a = 1)）
                e = ST.effect_stats(cells[(s, K, "joint")], cells[(s, K, joracle)], g_eff, rng)
                if K == 0:
                    e["n_units"] = DD.N_FOLDS
                desc.append(dict(T=T, student=s, K=K, A="joint", B="joint_oracle", oracle_point=joracle, n_grid_points=len(jmeans), read="regret",
                                 delta_same=delta, interval_within_delta=bool(K > 0 and -delta < e["lo"] and e["hi"] < delta), g_eff=g_eff, **e))
                for tcw, tcb in dai1.TETHER:                              # cons − joint の引き戻しの曲線（記述）
                    e = ST.effect_stats(cells[(s, K, f"tc|{tcw}|{tcb}")], cells[(s, K, f"tj|{tcw}|{tcb}")], g_eff, rng)
                    if K == 0:
                        e["n_units"] = DD.N_FOLDS
                    desc.append(dict(T=T, student=s, K=K, A="cons_tether", B="joint_tether", Cw=tcw, Cb=tcb, read="tether_curve", oracle_point="",
                                     n_grid_points=0, delta_same=delta, interval_within_delta=False, g_eff=g_eff, **e))
                # 選択の自由度をそろえた比べ：γ = 1/4 の列で ρ だけ選んだ cons − joint_a1，選択なしどうし cons0 − joint_default
                for A_, B_ in (("cons_rho25", "joint_a1"), ("cons0", "joint_default")):
                    e = ST.effect_stats(cells[(s, K, A_)], cells[(s, K, B_)], g_eff, rng)
                    if K == 0:
                        e["n_units"] = DD.N_FOLDS
                    desc.append(dict(T=T, student=s, K=K, A=A_, B=B_, oracle_point="", n_grid_points=0, delta_same=delta,
                                     interval_within_delta=bool(K > 0 and -delta < e["lo"] and e["hi"] < delta), g_eff=g_eff, **e))
                for gm in (0.0, 0.25, 0.5):                              # ρ = 1 での γ の曲線（γ = 1 は joint_default）
                    e = ST.effect_stats(cells[(s, K, f"g|1.0|{gm}")], cells[(s, K, "joint_default")], g_eff, rng)
                    if K == 0:
                        e["n_units"] = DD.N_FOLDS
                    desc.append(dict(T=T, student=s, K=K, A=f"cons_rho1_gamma{gm}", B="joint_default", oracle_point="", n_grid_points=0,
                                     delta_same=delta, interval_within_delta=False, g_eff=g_eff, **e))
        # 印の割合・印の有無ごとの教師の正解率と直す・壊す・正解の感情に置かれた重みの割合（候補のラベルは集計のときだけ引く．K 本を除いた候補で数える）
        acc = {}
        fz1T = fz1[T]
        for f, r in units:
            name = f"{T}_ALL_f{f}_r{r}"
            p4 = D4.P4(out4 / "eval" / "units" / name)
            z2 = np.load(in_dir / "eval" / "units" / f"{name}_p2.npz")
            p3rows = pd.read_csv(in_dir / "eval" / "units" / f"{name}_p3.csv", low_memory=False)
            ids = p4.z["sample_pool"].astype(str)
            y = y_of(ids)
            n = len(ids)
            for K in KS_ALL:
                nk = ~np.isin(ids, p4.z[f"sample_K{K}"].astype(str)) if K else np.ones(n, bool)
                tp = p4.z[f"p4_K{K}_teacher"].astype(int)
                tr = tp == y
                mine = 100 * float(np.mean([(tp[nk & (y == k)] == k).mean() for k in range(4) if (nk & (y == k)).any()]))
                ref = p3rows[(p3rows.method == "_teacher_stats") & (p3rows.teacher == "one") & (p3rows.K == K)]
                tcheck.append(dict(T=T, fold=f, seed=r, K=K, uar_pool_mine=mine, uar_pool_dai1=float(ref.uar_pool.iloc[0]),
                                   diff=mine - float(ref.uar_pool.iloc[0])))
                Q = torch.as_tensor(p4.z[f"p4_K{K}_Q"].astype(np.float64), device=S.DEV)
                yy = torch.as_tensor(y, device=S.DEV)
                for s in DD.STUDENTS:
                    mk = p4.z[f"p4_K{K}_{s}_mark"].astype(bool)
                    sp = z2[f"p2_K{K}_none_{s}_pool"].astype(np.float32).argmax(1)
                    sr = sp == y
                    rho_c, gam_c = sels_from(frozen[T])[(s, K)]
                    jj = joint_sel(fz1T, s, K)
                    m = torch.as_tensor(np.where(mk, 1.0, gam_c), device=S.DEV)
                    Pc = S.soft_rows(Q * m[:, None], dai1.mval(rho_c, n), balance=True)
                    Pj = S.soft_rows(Q, dai1.mval(jj["rho"], n), balance=True)

                    nkt = torch.as_tensor(nk, device=S.DEV)

                    def wfrac(P):                                           # K 本を除いた候補で数える
                        return float(np.mean([(P[nkt & (yy == k), k].sum() / P[nkt, k].sum().clamp(min=1e-30)).item() for k in range(4)]))
                    d = dict(share=float(mk[nk].mean()), n_marked=int((nk & mk).sum()), n_unmarked=int((nk & ~mk).sum()),
                             wfrac_correct_cons=wfrac(Pc), wfrac_correct_joint=wfrac(Pj))
                    for nm_, g in (("marked", nk & mk), ("unmarked", nk & ~mk)):
                        d[f"acc_{nm_}"] = float(tr[g].mean()) if g.any() else np.nan
                        d[f"err_{nm_}"] = float((~tr[g]).mean()) if g.any() else np.nan
                        d[f"fix_{nm_}"] = float((tr & ~sr)[g].mean()) if g.any() else np.nan
                        d[f"harm_{nm_}"] = float((~tr & sr)[g].mean()) if g.any() else np.nan
                        for k in range(4):
                            gk_ = g & (y == k)
                            d[f"recall_{nm_}_{k}"] = float(tr[gk_].mean()) if gk_.any() else np.nan
                    for k in range(4):
                        gk_ = nk & (tp == k)
                        d[f"share_cls{k}"] = float(mk[gk_].mean()) if gk_.any() else np.nan
                    am = {m_: p4.z[f"p4_K{K}_ret_{m_}_am"].astype(int) for m_ in "AVT"}
                    avt = z2[f"p2_K{K}_none_AVT_pool"].astype(np.float32).argmax(1)
                    oth = D4.others_of(s)
                    prs = [(oth[i], oth[j]) for i in range(len(oth)) for j in range(i + 1, len(oth))]
                    d["agree_others"] = float(np.mean([(am[a_] == am[b_])[nk].mean() for a_, b_ in prs]))          # 生徒以外の ret どうしの判定の一致率
                    d["agree_other_vs_avt_none"] = float(np.mean([(am[m_] == avt)[nk].mean() for m_ in oth]))     # それぞれの ret と AVT の none[K] の一致率
                    d["agree_diff"] = d["agree_others"] - d["agree_other_vs_avt_none"]
                    es = [err_stats(am[a_], am[b_], y, nk) for a_, b_ in prs]
                    d["err_phi"], d["err_same_ratio"] = float(np.nanmean([e_[0] for e_ in es])), float(np.nanmean([e_[1] for e_ in es]))
                    acc.setdefault((s, K), []).append(d)
                    for vn, tg in (("cons", ""), *[(cn, px) for cn, px in (("cons_self", "self"), ("cons_own", "own"), ("cons_conf", "conf"), ("cons_rand", "rand"))]):
                        key = f"p4_K{K}_{s}_mark{tg}"
                        if key in p4.z:
                            mv = p4.z[key].astype(bool)
                            marks_var.append(dict(T=T, student=s, K=K, variant=vn, share=float(mv[nk].mean()),
                                                  **{f"share_cls{k}": (float(mv[nk & (tp == k)].mean()) if (nk & (tp == k)).any() else np.nan) for k in range(4)}))
        for (s, K), a in acc.items():
            marks.append(dict(T=T, student=s, K=K, n_units=len(a), **pd.DataFrame(a).mean().to_dict()))
        sens += sensitivity(out4, in_dir, T, units, y_of, cl_of, frozen[T], fz1[T], g_eff, rng, sel_ctrl)
        print(f"{T}: 単位 {len(units)}，実効の数 {g_eff:.1f}，照合の食い違い {mism}/{total}", flush=True)
    dest = out4 / "stats"
    dest.mkdir(exist_ok=True)
    # joint・kd・none の保存（実験 1 の評価の回の段 1・段 3）の sha256 を書く（m7）。固定のファイルは評価の後に書き換えないので別のファイルに置く
    shas = {f"{T}_ALL_f{f}_r{r}_p{w}.npz": dai1.sha(in_dir / "eval" / "units" / f"{T}_ALL_f{f}_r{r}_p{w}.npz")
            for T in DD.CORPORA for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"])) for w in (1, 2, 3)}
    gate_sha = json.load(open(D4.GATE4)).get("input_sha256", {})
    if gate_sha != shas:
        sys.exit("実験 1 の評価の回の p1・p2・p3 の sha256 が eval_gate_dai4.json の input_sha256 と違う")
    ish = dest / "eval_input_sha256.json"
    if ish.exists() and json.load(open(ish)) != shas:
        sys.exit("実験 1 の評価の回の保存の sha256 が前の集計のときと違う")
    json.dump(shas, open(ish, "w"), indent=1)
    shutil.copy(ish, ROOT / "DAI" / "frozen" / "dai4_eval_inputs_sha256.json")
    cm, dt = dai1.git_commit()
    # 計算機ごとの単位の数（M8），重みが 0 になった感情の数（m9），教師の一致の集計（m10）
    allrows = pd.concat([pd.read_csv(out4 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}_p4.csv", low_memory=False)
                         for T in DD.CORPORA for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"]))], ignore_index=True)
    gpu = allrows.drop_duplicates(["T", "fold", "seed"]).groupby(["T", "gpu"]).size().rename("n_units").reset_index()
    w = allrows[allrows.method == "_weights"]
    zero = w.groupby(["T", "variant", "student", "K"]).agg(n_zero_weight_classes=("n_zero_weight_classes", "sum"),
                                                          n_points_with_zero=("n_zero_weight_classes", lambda x: int((x > 0).sum())),
                                                          n_rows=("n_zero_weight_classes", "size")).reset_index()
    tcs = pd.DataFrame(tcheck)
    tcs = tcs.assign(absdiff=tcs["diff"].abs()).groupby(["T", "K"]).agg(
        n=("absdiff", "size"), n_mismatch=("absdiff", lambda x: int((x > 1e-6).sum())), max_absdiff=("absdiff", "max")).reset_index()
    mvd = pd.DataFrame(marks_var).groupby(["T", "student", "K", "variant"]).mean(numeric_only=True).reset_index()
    ch = pd.DataFrame([dict(T=T, student=s, K=K, rho=v[0], gamma=v[1]) for T in DD.CORPORA for (s, K), v in sels_from(frozen[T]).items()])
    wc = w[w.variant == "cons"].merge(ch, on=["T", "student", "K", "rho", "gamma"])
    neff = wc.assign(**{f"small{k}": wc[f"neff_{k}"] < 30 for k in range(4)}).assign(n_small=lambda d: d[[f"small{k}" for k in range(4)]].sum(1))
    neff = neff.groupby(["T", "student", "K"]).agg(n_units=("n_small", "size"), n_unit_class_lt30=("n_small", "sum")).reset_index()
    print(f"計算機ごとの単位の数：{gpu.groupby('gpu').n_units.sum().to_dict()}，教師の不一致 {int(tcs.n_mismatch.sum())}/{int(tcs.n.sum())}", flush=True)
    for nm, rows in (("families6", fam), ("desc", desc), ("marks", marks), ("teacher_check", tcheck), ("gpu_counts", gpu),
                     ("zero_weight_classes", zero), ("teacher_check_summary", tcs), ("marks_variants", mvd), ("neff_lt30", neff), ("sensitivity", sens)):
        pd.DataFrame(rows).assign(commit=cm, dirty=dt, unit_commit=ucommit).to_csv(dest / f"{nm}.csv", index=False)
    print("書いた:", dest, "（families6.csv，desc.csv，marks.csv）")


if __name__ == "__main__":
    mode = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else D4.OUT_DIR
    if mode == "detect":
        detect(out, D4.IN_DIR, sys.argv[3].split(",") if len(sys.argv) > 3 else list(DD.CORPORA))
    elif mode == "eval":
        evaluate(out, D4.IN_DIR)
    else:
        sys.exit("detect か eval")
