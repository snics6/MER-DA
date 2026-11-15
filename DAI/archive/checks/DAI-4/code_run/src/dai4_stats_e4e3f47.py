"""DAI 実験 4 の detect と集計（設計 `DAI/docs/実験4_設計.md` 2・3・5 節）。区間・p 値は `dai1_stats.effect_stats` をそのまま使う。

使い方：
  python DAI/exp/dai4_stats.py detect [出力の置き場] [T,T,...]  ：選ぶ段の後・評価の段の前。実験 1 の 9.2 節と同じ手順（`dai1_stats.detect` を族 6 に当てた）。
      族 6 = cons − joint の 12 セル（生徒 4 × K ∈ {0, 5, 25}）。評価のターゲット X ごとに，ほかの T の仮のターゲットの回の c = X の単位で
      標準誤差の見込み（塊の項と種の項）を測り，検出できる差（Holm の最初の段と最後の段）を出す。効果の見込みは X のテストを使わず，
      T = X の仮のターゲットの回で仮のターゲット c を 1 つ除いた 20 単位で (ρ, γ) と joint の設定を選び直し，除いた c の 10 単位で測って 3 つの c で平均する。
      検出できる差（最初の段）が効果の見込みの大きさを超えるセルは記述に回す。同じ水準の δ と種の数は実験 1 の固定のファイルの値を使う
      （比較の基準の種の間の分散だけに依り，この実験で変わらない）。結果は frozen.json の X の `detect` に書き，写しを置く。
  python DAI/exp/dai4_stats.py eval [出力の置き場]  ：評価の回の後。
      families6.csv（族 6．コーパスごとに Holm），desc.csv（記述：cons − none，cons − kd，cons − cons（選択なし），cons − joint の K = 1・10，
      対照 cons_self・cons_own・cons_conf・cons_rand との差（同じ選んだ (ρ, γ)），
      取りこぼし = 評価のテストで平均が最良の格子の点 − 選んだ点，同じ水準の δ との比べ），marks.csv（印の割合，印の有無ごとの教師の正解率）。
      判定は int8 の保存から作り直し，行の uar と全部の問で照合した数を残す。今のコミットが評価の回の単位のコミットと違えば止まる。
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
import dai4 as D4  # noqa: E402
import dai_data as DD  # noqa: E402

KS_FAM = (0, 5, 25)
KS_ALL = dai1.KS_ALL
ROOT = DD.ROOT


def joint_sel(fz1T: dict, s: str, K: int) -> dict:
    v = fz1T["main"][str(K)]
    j = fz1T["select"][f"{s}|{K}|joint|{v}"]
    return dict(teacher=v, rho=j["rho"], a=j["a"])


def collect(out4: Path, in_dir: Path, tag: str, T: str, c: str, units, y_of, cl_of, sel_cons: dict, fz1T: dict, sel_joint=None,
            extras: bool = False, check_rows: bool = False):
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
        if not (up.sid == p4.z["sample_test"].astype(str)).all():
            sys.exit(f"{name}: dai4 と実験 1 の保存のテストの発話が違う")
        rows = pd.read_csv(out4 / tag / "units" / f"{name}_p4.csv", low_memory=False) if check_rows else None
        for s in DD.STUDENTS:
            sp = up.none(s, 0, None)
            for K in KS_ALL:
                put = lambda nm, pr: cells.setdefault((s, K, nm), {}).__setitem__((f, r), counts(y, pr, cl))   # noqa: E731
                rho, gam = sel_cons[(s, K)]
                put("cons", p4.pred(s, K, rho, gam))
                jj = (sel_joint or {}).get((s, K)) or joint_sel(fz1T, s, K)
                put("joint", up.p3item(s, K, method="joint", teacher=jj["teacher"], rho=jj["rho"], a=jj["a"]))
                if extras:
                    put("none", sp if K == 0 else up.none(s, K, fz1T["tether"]))
                    kd = fz1T["select"][f"{s}|{K}|kd|one"]
                    put("kd", up.p3item(s, K, method="kd", teacher="one", temp=kd["temp"], rho=kd["rho"]))
                    put("cons0", p4.pred(s, K, D4.DEFAULT["rho"], D4.DEFAULT["gamma"]))
                    for cn, px in D4.CTRL_PREFIX.items():                  # 対照：本番と同じ選んだ (ρ, γ) の点（AVT の cons_self は無い）
                        if p4.has(s, K, px):
                            put(cn, p4.pred(s, K, rho, gam, px))
                    for it in p4.items(s, K):
                        put(f"g|{it['rho']}|{it['gamma']}", p4.pred(s, K, it["rho"], it["gamma"]))
                if check_rows:
                    mine = np.sort([ST.uar_full(counts(y, pr, cl)) for pr in p4.z[f"p4_{s}_K{K}_pred"]])
                    rr = rows[(rows.method == "cons") & (rows.student == s) & (rows.K == K)]
                    theirs = np.sort(rr.uar.to_numpy(float))
                    total += 1
                    mism += int(len(mine) != len(theirs) or not np.allclose(mine, theirs, atol=1e-6))
    return cells, mism, total


counts = ST.counts


def sels_from(fz4T: dict) -> dict:
    return {(k.split("|")[0], int(k.split("|")[1])): (v["rho"], v["gamma"]) for k, v in fz4T["select"].items()}


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
    if dirty:
        sys.exit("作業木に未コミットの変更がある．評価の集計は流さない")
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
    if commit != ucommit:
        sys.exit(f"今のコミット {commit[:10]} が評価の回の単位のコミット {ucommit[:10]} と違う．評価の回のコミットの作業木で流す")
    return ucommit


def holm(p):
    return ST.holm(np.asarray(p, float))


def evaluate(out4: Path, in_dir: Path) -> None:
    fz1 = json.load(open(D4.FROZEN1))
    ucommit = guard_eval(out4, fz1)
    frozen = json.load(open(out4 / "select" / "frozen.json"))
    rng = np.random.default_rng(20261105)
    fam, desc, marks = [], [], []
    for T in DD.CORPORA:
        g_eff, _ = ST.g_eff_of(T)
        R = int(fz1[T]["n_seeds_eval"])
        units = [(f, r) for f in range(DD.N_FOLDS) for r in range(R)]
        y_of, cl_of = ST.label_maps(T)
        sc = sels_from(frozen[T])
        cells, mism, total = collect(out4, in_dir, "eval", T, T, units, y_of, cl_of, sc, fz1[T], extras=True, check_rows=True)
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
                for B in ("none", "kd", "cons0", "joint", oracle, *[cn for cn in D4.CTRL_PREFIX if (s, K, cn) in cells]):
                    nm = "oracle" if B == oracle else B
                    e = ST.effect_stats(cells[(s, K, "cons")], cells[(s, K, B)], g_eff, rng)
                    if K == 0:
                        e["n_units"] = DD.N_FOLDS
                    desc.append(dict(T=T, student=s, K=K, A="cons", B=nm, oracle_point=oracle if nm == "oracle" else "",
                                     delta_same=delta, interval_within_delta=bool(K > 0 and -delta < e["lo"] and e["hi"] < delta),
                                     g_eff=g_eff, **e))
        # 印の割合と，印の有無ごとの教師の正解率（K 本を除いた候補．ラベルは集計のときだけ引く）
        acc = {}
        for f, r in units:
            p4 = D4.P4(out4 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}")
            ids = p4.z["sample_pool"].astype(str)
            y = y_of(ids)
            for K in KS_ALL:
                nk = ~np.isin(ids, p4.z[f"sample_K{K}"].astype(str)) if K else np.ones(len(ids), bool)
                tr = p4.z[f"p4_K{K}_teacher"].astype(int) == y
                for s in DD.STUDENTS:
                    mk = p4.z[f"p4_K{K}_{s}_mark"].astype(bool)
                    a = acc.setdefault((s, K), [])
                    a.append(dict(share=float(mk[nk].mean()), acc_marked=float(tr[nk & mk].mean()) if (nk & mk).any() else np.nan,
                                  acc_unmarked=float(tr[nk & ~mk].mean()) if (nk & ~mk).any() else np.nan,
                                  n_marked=int((nk & mk).sum()), n_unmarked=int((nk & ~mk).sum())))
        for (s, K), a in acc.items():
            marks.append(dict(T=T, student=s, K=K, n_units=len(a), **pd.DataFrame(a).mean().to_dict()))
        print(f"{T}: 単位 {len(units)}，実効の数 {g_eff:.1f}，照合の食い違い {mism}/{total}", flush=True)
    dest = out4 / "stats"
    dest.mkdir(exist_ok=True)
    # joint・kd・none の保存（実験 1 の評価の回の段 1・段 3）の sha256 を書く（m7）。固定のファイルは評価の後に書き換えないので別のファイルに置く
    shas = {f"{T}_ALL_f{f}_r{r}_p{w}.npz": dai1.sha(in_dir / "eval" / "units" / f"{T}_ALL_f{f}_r{r}_p{w}.npz")
            for T in DD.CORPORA for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"])) for w in (1, 3)}
    ish = dest / "eval_input_sha256.json"
    if ish.exists() and json.load(open(ish)) != shas:
        sys.exit("実験 1 の評価の回の保存の sha256 が前の集計のときと違う")
    json.dump(shas, open(ish, "w"), indent=1)
    shutil.copy(ish, ROOT / "DAI" / "frozen" / "dai4_eval_inputs_sha256.json")
    cm, dt = dai1.git_commit()
    for nm, rows in (("families6", fam), ("desc", desc), ("marks", marks)):
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
