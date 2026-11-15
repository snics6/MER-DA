"""実験 18 の下調べ（DAI-18）の計算の後の量：声の生徒で，積の疑似ラベルの生徒の格子の点を，提案の手法（`M_F6_mixsel`）の生徒と同じ 40 単位で選び直した値．

設計 `docs/設計/実験18_下調べ_設計.md` 5 節の R5 が「結果の文に感度として並べる」と決めた量である．
学び直さない．`results/dai18_probe/students/` の保存（格子の全点のテストの判定と UAR）だけを読む．GPU は使わない．

何をするか：
  1. 評価するコーパス c（RAVDESS を除く 5 つ）と K ごとに，声の生徒の格子の点（12 点）を選ぶ．
     選ぶ単位は c を評価のターゲット T にした単位（c を含まない）．集計の段と同じ規則（テストの UAR の平均が最大．同点は並びで先）．
       - 積の生徒・元の選び方：仮のターゲットが 5 つの 50 単位（`summary/students_sel.csv` と同じ点になることを確かめる）
       - 積の生徒・選び直し：仮のターゲットから RAVDESS を除いた 40 単位（`M_F6_mixsel` の生徒が選ばれるのと同じ単位）
       - `M_F6_mixsel` の生徒：40 単位（`students_sel.csv` と同じ点になることを確かめる）
  2. c を仮のターゲットにした単位のテストの判定から，「`M_F6_mixsel` の生徒 − 積の生徒」の差と区間を，集計の段と同じ部品
     （`dai1_stats.effect_stats`，`dai12_probe._add`・`_finish`，`dai16_probe.verdict`，`dai18_probe.row_rng`）で作る．
     元の選び方の差は `summary/students_effects.csv` の行と一致することを確かめる（一致しなければ止まる）．
  3. `results/dai18_probe/posthoc/voice_student_prod40.csv` と `provenance_voice40.json` に書く．

使い方：.venv/bin/python tools/dai18_posthoc_voice40.py
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
import dai18_probe as P18  # noqa: E402   集計の段と同じ部品を使う（読み込むだけ．台本は変えない）

ST, P12, P16 = P18.ST, P18.P12, P18.P16
OUT = ROOT / "results" / "dai18_probe"
SDIR = OUT / "students"
PDIR = OUT / "posthoc"
KIND, A, B = "A", "M_F6_mixsel", "prod"
CORP = [c for c in P18.CORP6 if len(P18.MODS_OF[c]) >= 2]                 # `M_F6_mixsel` の生徒がある 5 コーパス
NG = len(P18.SGRID)


def g_eff() -> dict:
    """実効の塊の数．4 コーパスは実験 1 の分割の表，CREMA-D は実験 18 の分割の表から数える（集計の段は読み込んだ発話の表から数える．
    区間の作り方と自由度が `students_effects.csv` と同じになることを下で確かめる）．"""
    g = {c: ST.g_eff_of(c)[0] for c in P18.DD.CORPORA}
    sp = pd.read_csv(OUT / "splits_new.csv", keep_default_na=False)
    n = sp[sp.corpus == "CREMA-D"]["cluster"].value_counts().to_numpy()
    g["CREMA-D"] = float(n.sum() ** 2 / (n ** 2).sum())
    return g


def main() -> None:
    t0 = time.time()
    files = sorted(SDIR.glob("*.csv"))
    assert len(files) == 300, len(files)
    sc = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
    assert sc.commit.nunique() == 1
    g = sc[(sc.student == KIND) & sc.teacher.isin([A, B]) & sc.grid.notna() & (sc.grid >= 0)].copy()
    g["grid"] = g.grid.astype(int)
    sel = pd.read_csv(OUT / "summary" / "students_sel.csv")
    eff0 = pd.read_csv(OUT / "summary" / "students_effects.csv")
    geff = g_eff()

    def pick(c: str, K: int, teacher: str, drop_ravdess: bool):
        q = g[(g["T"] == c) & (g.K == K) & (g.teacher == teacher)]
        if drop_ravdess:
            q = q[q.c != "RAVDESS"]
        vals = q.groupby("grid").uar.mean().reindex(range(NG)).to_numpy()
        n_units = len(q[["c", "fold", "seed"]].drop_duplicates())
        return int(np.nanargmax(vals)), n_units, vals

    def saved_point(c: str, K: int, teacher: str) -> int:
        q = sel[(sel.c == c) & (sel.K == K) & (sel.student == KIND) & (sel.teacher == teacher)]
        assert len(q) == 1
        return int(q.iloc[0].grid)

    choice = {}
    for c in CORP:
        for K in P18.KS:
            gm, nm, _ = pick(c, K, A, drop_ravdess=False)                 # `M_F6_mixsel` の生徒は RAVDESS の単位に無いので，除かなくても 40 単位
            g50, n50, v50 = pick(c, K, B, drop_ravdess=False)
            g40, n40, v40 = pick(c, K, B, drop_ravdess=True)
            assert (nm, n50, n40) == (40, 50, 40), (c, K, nm, n50, n40)
            assert gm == saved_point(c, K, A) and g50 == saved_point(c, K, B), (c, K)   # 集計の段と同じ点
            choice[(c, K)] = dict(grid_mix=gm, grid_prod50=g50, grid_prod40=g40, mean40_at_g40=float(v40[g40]), mean40_at_g50=float(v40[g50]),
                                  mean50_at_g50=float(v50[g50]), mean50_at_g40=float(v50[g40]))

    cells = {}
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        if c not in CORP:
            continue
        f, r = int(f[1:]), int(r[1:])
        z = np.load(p.with_suffix(".npz"), allow_pickle=False)
        yt = z["y_test"].astype(int)
        for K in P18.KS:
            ch = choice[(c, K)]
            for name, t_, gi in (("mix", A, ch["grid_mix"]), ("prod50", B, ch["grid_prod50"]), ("prod40", B, ch["grid_prod40"])):
                cnt = ST.counts(yt, z[f"pred_K{K}_{KIND}_{t_}_g{gi}"].astype(int), z["cl_test"])
                P12._add(cells.setdefault((c, K, name), {}), (f, r), cnt, p.stem)
    P12._finish(cells)

    def level(dd: dict) -> float:
        return float(np.mean([np.mean([ST.uar_full(dd[k]) for k in dd if k[0] == f]) for f in sorted({k[0] for k in dd})]))

    def eff(c: str, K: int, b_name: str) -> dict:
        es = ST.effect_stats(cells[(c, K, "mix")], cells[(c, K, b_name)], geff[c], P18.row_rng("C", c, K, KIND, A, B))   # 集計の段と同じ乱数の鍵
        return dict(est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], df=es.get("df", np.nan), method=es["method"], n_fr=es["n_units"],
                    verdict=P16.verdict(es["lo"], es["hi"], es["effect"]))

    rows, worst = [], 0.0
    for c in CORP:
        for K in P18.KS:
            ch = choice[(c, K)]
            e50, e40 = eff(c, K, "prod50"), eff(c, K, "prod40")
            s = eff0[(eff0.c == c) & (eff0.K == K) & (eff0.student == KIND) & (eff0.a == A) & (eff0.b == B)]
            assert len(s) == 1
            s = s.iloc[0]
            d = max(abs(e50[k] - s[k]) for k in ("est", "lo", "hi", "se"))
            worst = max(worst, d)
            if d > 1e-9 or e50["method"] != s["method"] or e50["verdict"] != s["verdict"] or not (np.isnan(e50["df"]) and np.isnan(s["df"]) or e50["df"] == s["df"]):
                raise SystemExit(f"{c} K={K}：元の選び方の差が students_effects.csv と合わない（最大の違い {d:.3g}）")
            rows.append(dict(c=c, K=K, student=KIND, a=A, b=B, **{k: v for k, v in ch.items()}, changed=bool(ch["grid_prod40"] != ch["grid_prod50"]),
                             rho_prod50=P18.SGRID[ch["grid_prod50"]][0], a_prod50=P18.SGRID[ch["grid_prod50"]][1],
                             rho_prod40=P18.SGRID[ch["grid_prod40"]][0], a_prod40=P18.SGRID[ch["grid_prod40"]][1],
                             uar_mix=level(cells[(c, K, "mix")]), uar_prod50=level(cells[(c, K, "prod50")]), uar_prod40=level(cells[(c, K, "prod40")]),
                             **{f"{k}_sel50": v for k, v in e50.items()}, **{f"{k}_sel40": v for k, v in e40.items()}))
    PDIR.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(PDIR / "voice_student_prod40.csv", index=False)
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                                     # noqa: BLE001
        commit = ""
    json.dump(dict(script="tools/dai18_posthoc_voice40.py", script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   time=time.strftime("%Y-%m-%d %H:%M:%S"), head_commit_when_run=commit, students_commit=str(sc.commit.iloc[0]),
                   dai18_probe_sha256=hashlib.sha256((ROOT / "exp" / "dai18_probe.py").read_bytes()).hexdigest(),
                   students_sel_sha256=hashlib.sha256((OUT / "summary" / "students_sel.csv").read_bytes()).hexdigest(),
                   students_effects_sha256=hashlib.sha256((OUT / "summary" / "students_effects.csv").read_bytes()).hexdigest(),
                   n_student_files=len(files), n_rows=len(out), n_changed=int(out.changed.sum()), reproduce_max_abs_diff=worst,
                   g_eff={k: float(v) for k, v in geff.items()}, device="cpu", sec=time.time() - t0),
              open(PDIR / "provenance_voice40.json", "w"), indent=1, ensure_ascii=False)
    print(f"書いた：{PDIR / 'voice_student_prod40.csv'}（{len(out)} 行．点が変わった組 {int(out.changed.sum())}．元の選び方の差の再現の違いの最大 {worst:.3g}．{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
