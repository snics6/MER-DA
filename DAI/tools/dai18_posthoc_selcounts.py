"""実験 18 の下調べ（DAI-18）の計算の後の数え：選び方に関わる 4 つの量を，保存から学び直さずに数える．

照合の第 1 回（`checks/DAI-18/claim.md`）が結果の文に求めた量である．検査役の台本（`checks/DAI-18/claim_run/`）は読まずに書いた．
`results/dai18_probe/units/`・`students/` と `summary/sel.csv` を読むだけ．GPU は使わない．学び直しはしない．

数える量：
  1. 自然な会話のコーパスだけで選び直したときの点（設計 8 節の 6）．
     版の格子の点を，仮のターゲットが自然な会話のコーパス（AFEW・IEMOCAP・MOSEI・MELD）の単位だけで選び直す．
     元の選び方は CREMA-D を仮のターゲットにした単位も使う（RAVDESS を評価するときは 50 単位，ほかは 40 単位）．
     選ぶ規則は集計の段と同じ `dai18_probe.select_points`（テストの UAR の平均が最大．同点は並びで先）．
     出力：posthoc/natural_only_reselect.csv（105 行＝評価するコーパス × K × 選ぶ版）
  2. 選択を通した依存（設計 8 節）．
     コーパス c を評価する生徒の格子の点は，c を評価のターゲットにした単位（仮のターゲット c′）で選ぶ．その単位の疑似ラベルの版の点は c′ を評価するときの点で，
     その点を選んだ単位には c を仮のターゲットにした単位（c のテストのラベルを使う）が入る．
     c を仮のターゲットにした単位を除いて c′ の点を選び直し，点が変わる (c, c′, K, 版) を数える．
     数える範囲は，c を仮のターゲットにした単位が c′ の点の選択に入っている組だけ（`sel.csv` の `pseudo_targets` に c がある組）．
     出力：posthoc/selection_dependency.csv
  3. mix2 と mix3 の判定の一致．units の保存した判定の配列（adj・raw・候補の 3 層）で，`M_F6_mix2` と `M_F6_mix3` の判定が違う発話を数える．
  4. 融合の生徒の格子の点を選ぶ単位の内訳（評価するコーパスごとに，声・顔・言葉の融合の単位と声・顔の融合の単位の数）．
  3 と 4 は posthoc/provenance_selcounts.json に書く．

使い方：.venv/bin/python tools/dai18_posthoc_selcounts.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                    # GPU を使わない（別の実験が動いている）
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True                                            # exp に __pycache__ を書かない

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
import dai18_probe as P18  # noqa: E402   集計の段と同じ選び方（select_points）を使う．読み込むだけ

OUT = ROOT / "results" / "dai18_probe"
PDIR = OUT / "posthoc"
NAT = list(P18.DD.CORPORA)                                                # 自然な会話のコーパス（元の 4 コーパス）
TEACH_SEL = ("M_F6_mixsel", "MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel", "MS_eq_sel")   # 生徒の疑似ラベルに使う版のうち，格子の点を選ぶもの
ROLE_OF = lambda kind: "fusion" if len(kind) >= 2 else kind              # noqa: E731


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def roles(c: str) -> set:
    return {ROLE_OF(k) for k in P18.kinds_of(c)}


def main() -> None:
    t0 = time.time()
    assert NAT == ["AFEW", "IEMOCAP", "MOSEI", "MELD"], NAT
    ufiles = sorted((OUT / "units").glob("*.csv"))
    assert len(ufiles) == 300, len(ufiles)
    csvs = pd.concat([pd.read_csv(p, usecols=["K", "version", "uar_adj", "T", "c", "fold", "seed"], low_memory=False) for p in ufiles], ignore_index=True)
    sel = pd.read_csv(OUT / "summary" / "sel.csv")
    sel0 = sel[sel.tied_to.isna()]

    # 元の選び方を作り直し，集計の sel.csv と同じ点になることを確かめる
    ch0, rows0 = P18.select_points(csvs)
    r0 = pd.DataFrame(rows0)
    r0 = r0[r0.tied_to.isna()] if "tied_to" in r0 else r0
    assert {(r.c, int(r.K), r.version): r.chosen for r in sel.itertuples()} == ch0, "選び直した点が sel.csv と違う"
    key0 = {(r.c, int(r.K), r.version): r for r in r0.itertuples()}
    assert len(key0) == len(sel0) == 105, (len(key0), len(sel0))

    # 1. 自然な会話のコーパスだけで選び直す
    ch_n, rows_n = P18.select_points(csvs[csvs.c.isin(NAT)])
    rn = pd.DataFrame(rows_n)
    keyn = {(r.c, int(r.K), r.version): r for r in rn[rn.tied_to.isna()].itertuples()}
    assert set(keyn) == set(key0)
    nat_rows = []
    for (c, K, v), a in sorted(key0.items(), key=lambda x: (P18.CORP6.index(x[0][0]), x[0][1], x[0][2])):
        b = keyn[(c, K, v)]
        assert set(b.pseudo_targets.split(",")) <= set(NAT)
        nat_rows.append(dict(c=c, K=K, version=v, chosen_orig=a.chosen, chosen_natural=b.chosen, changed=bool(a.chosen != b.chosen),
                             n_units_orig=int(a.n_units), n_units_natural=int(b.n_units), pseudo_targets_orig=a.pseudo_targets, pseudo_targets_natural=b.pseudo_targets,
                             n_tied_orig=int(a.n_tied), n_tied_natural=int(b.n_tied), same_units=bool(a.pseudo_targets == b.pseudo_targets)))
    nat = pd.DataFrame(nat_rows)
    assert not nat[nat.same_units].changed.any(), "単位が同じなのに点が変わった"

    # 2. 選択を通した依存：c を仮のターゲットにした単位を除いて，c′ の点を選び直す
    dep_rows = []
    for c in P18.CORP6:
        ch_x, _ = P18.select_points(csvs[csvs.c != c])
        for (c2, K, v), a in key0.items():
            if c2 == c or v not in TEACH_SEL:
                continue
            if v == "M_F6_mixsel" and len(P18.MODS_OF[c]) < 2:             # c が RAVDESS のとき，提案の手法の疑似ラベルの生徒は無い
                continue
            in_sel = c in a.pseudo_targets.split(",")
            new = ch_x[(c2, K, v)]
            if not in_sel:
                assert new == a.chosen, (c, c2, K, v)                      # c の単位が選択に入っていなければ，点は変わらない
                continue
            t_old, t_new = P18.tied(v, a.chosen), P18.tied(v, new)
            dep_rows.append(dict(c=c, c_prime=c2, K=K, version=v, chosen_orig=a.chosen, chosen_without_c=new, changed=bool(new != a.chosen),
                                 tied_version=P18.TIED_NAME.get(v, ""), tied_changed=bool(t_old != t_new),
                                 roles_shared=",".join(r for r in ("fusion", "A", "V", "T") if r in roles(c) & roles(c2))))
    dep = pd.DataFrame(dep_rows).sort_values(["version", "c", "K", "c_prime"], key=lambda s: s.map(P18.CORP6.index) if s.name in ("c", "c_prime") else s).reset_index(drop=True)

    # 3. mix2 と mix3 の判定の一致（判定の配列で比べる）
    n_arr, n_diff_arr, n_diff_utt, n_utt = 0, 0, 0, 0
    for p in ufiles:
        c = p.stem.rsplit("_", 3)[1]
        if len(P18.MODS_OF[c]) < 2:
            continue
        z = np.load(p.with_suffix(".npz"), allow_pickle=False)
        for K in P18.KS:
            for w in ("adj", "raw", "pool"):
                a, b = z[f"pred_{w}_K{K}_M_F6_mix2"], z[f"pred_{w}_K{K}_M_F6_mix3"]
                assert a.shape == b.shape
                d = int((a != b).sum())
                n_arr += 1
                n_utt += a.size
                n_diff_utt += d
                n_diff_arr += d > 0

    # 4. 融合の生徒の格子の点を選ぶ単位の内訳
    sfiles = sorted((OUT / "students").glob("*.csv"))
    assert len(sfiles) == 300
    sc = pd.concat([pd.read_csv(p, usecols=["K", "student", "teacher", "grid", "T", "c", "fold", "seed"], low_memory=False) for p in sfiles], ignore_index=True)
    fu = sc[sc.student.isin(["AVT", "AV"]) & (sc.grid == 0) & (sc.K == P18.KS[0])]
    fusion_units = {t: {c: {k: int(n) for k, n in fu[(fu.teacher == t) & (fu["T"] == c)].groupby("student").size().items()} for c in P18.CORP6 if len(P18.MODS_OF[c]) >= 2}
                    for t in ("prod", "M_F6_mixsel", "MN_F6K_sel")}
    for K in P18.KS[1:]:                                                    # 内訳は K で変わらない
        fk = sc[sc.student.isin(["AVT", "AV"]) & (sc.grid == 0) & (sc.K == K)]
        assert {c: {k: int(n) for k, n in fk[(fk.teacher == "prod") & (fk["T"] == c)].groupby("student").size().items()} for c in fusion_units["prod"]} == fusion_units["prod"]

    PDIR.mkdir(exist_ok=True)
    nat.to_csv(PDIR / "natural_only_reselect.csv", index=False)
    dep.to_csv(PDIR / "selection_dependency.csv", index=False)
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                                     # noqa: BLE001
        commit = ""
    r5 = dep[dep.version != "MS_eq_sel"]
    json.dump(dict(script="tools/dai18_posthoc_selcounts.py", script_sha256=sha(Path(__file__)), time=time.strftime("%Y-%m-%d %H:%M:%S"),
                   head_commit_when_run=commit, dai18_probe_sha256=sha(ROOT / "exp" / "dai18_probe.py"), sel_sha256=sha(OUT / "summary" / "sel.csv"),
                   n_unit_files=len(ufiles), n_student_files=len(sfiles), device="cpu",
                   natural=dict(n=len(nat), n_changed=int(nat.changed.sum()), by_version={v: [int(g.changed.sum()), len(g)] for v, g in nat.groupby("version")}),
                   dependency=dict(n_r5_versions=len(r5), n_changed_r5_versions=int(r5.changed.sum()),
                                   by_version={v: [int(g.changed.sum()), len(g)] for v, g in dep.groupby("version")}),
                   mix2_vs_mix3=dict(n_arrays=n_arr, n_arrays_with_diff=int(n_diff_arr), n_utterances=int(n_utt), n_utterances_diff=int(n_diff_utt)),
                   fusion_selection_units=fusion_units, sec=time.time() - t0),
              open(PDIR / "provenance_selcounts.json", "w"), indent=1, ensure_ascii=False)
    print(f"自然な会話のコーパスだけ：{len(nat)} 組のうち {int(nat.changed.sum())} 組で点が変わる")
    print(f"選択を通した依存（R5 の疑似ラベルの 4 版）：{len(r5)} のうち {int(r5.changed.sum())}")
    print(f"mix2 と mix3：配列 {n_arr}，違う発話 {n_diff_utt}．{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
