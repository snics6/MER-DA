"""実験 1 の評価の回の族 1・3・4 を，K = 10 でも保存済みの単位から集計する（計算の後の解析．2026-11-16．学び直しはしない）．

研究者の指示（2026-11-16）：論文で K = 5 だけを主にせず，K = 10，25 も同等に扱う．実験 1 の族は計算の前に K = 0，5，25 に決めたので
（`exp/dai1_stats.py` の KS_MAIN），K = 10 の差と区間は集計していなかった．K = 10 の設定（疑似ラベルの版，ρ，a，引き戻し，相手）は評価の回の
前に固定のファイル `results/dai1/select/frozen.json` に入っている．この台本はその固定の設定のまま，`dai1_stats.collect` と
`dai1_stats.effect_stats` で K = 10 の族 1（提案 − K 本の基準），族 3（提案 − 相手），族 4（選んだ設定 − 既定の設定）の差と区間を作る．

照合：同じ台本で K = 5 も作り，`results/dai1/stats/families_flagged.csv` の K = 5 の行と，差（effect）が 1e-9 以内，区間の端が
AFEW・IEMOCAP・MELD（塊の jackknife で乱数を使わない）で 1e-9 以内に一致することを確かめる．MOSEI の区間は塊の重みの引き直しの乱数の
列が元の集計と違うので，端の差を記録だけする．一致しなければ止める．

Holm の補正の後の p は書かない（族の数え方が計算の前の族と違うため）．補正していない 95% 区間だけを書く．

出力：results/dai1/stats/families_k10.csv（K = 10 の行），results/dai1/stats/families_k10_check.csv（K = 5 の照合）．
使い方：.venv/bin/python tools/dai1_k10_families.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai_data as DD  # noqa: E402

OUT1 = ROOT / "results" / "dai1"
TOL = 1e-9
RNG_SEED = 20261116
FAMS = {1: ("prop", "none"), 3: ("prop", "partner"), 4: ("prop", "default")}


def run(Ks: tuple) -> pd.DataFrame:
    import json
    frozen = json.load(open(OUT1 / "select" / "frozen.json"))
    old_ks = ST.KS_MAIN
    ST.KS_MAIN = Ks                                            # collect が回す K（この台本の中だけ）
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    try:
        for T in DD.CORPORA:
            fz = frozen[T]
            g_eff, _ = ST.g_eff_of(T)
            R = int(fz["n_seeds_eval"])
            units = [(f, r) for f in range(DD.N_FOLDS) for r in range(R)
                     if Path(f"{OUT1}/eval/units/{T}_ALL_f{f}_r{r}_p3.npz").exists()]
            if len(units) != DD.N_FOLDS * R:
                raise SystemExit(f"{T}：単位が {len(units)} 個（見込み {DD.N_FOLDS * R}）")
            y_of, cl_of = ST.label_maps(T)
            cells, mism, total = ST.collect(OUT1, "eval", T, T, fz, units, y_of, cl_of)
            if mism:
                raise SystemExit(f"{T}：保存した判定と rows.csv の UAR が {mism}/{total} で食い違う")
            for fam, (A, B) in FAMS.items():
                for s in DD.STUDENTS:
                    for K in Ks:
                        e = ST.effect_stats(cells[(s, K, A)], cells[(s, K, B)], g_eff, rng)
                        rows.append(dict(family=fam, T=T, student=s, K=K, A=A, B=B, g_eff=g_eff, mismatch=mism, checked=total, **e))
            print(f"{T}：単位 {len(units)}，照合の食い違い {mism}/{total}", flush=True)
    finally:
        ST.KS_MAIN = old_ks
    cm, dt = dai1.git_commit()
    return pd.DataFrame(rows).assign(commit=cm, dirty=dt, units_commit="ba98974", posthoc=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT1 / "stats"), help="書き出す置き場（試しは一時の置き場）．照合の元の families_flagged.csv はいつも results/dai1/stats から読む")
    args = ap.parse_args()
    sdir = Path(args.out_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    for name in ("families_k10.csv", "families_k10_check.csv"):
        if (sdir / name).exists():
            raise SystemExit(f"{sdir / name} が既にある（上書きしない）")
    chk = run((5,))
    ref = pd.read_csv(OUT1 / "stats" / "families_flagged.csv", comment="#")
    ref = ref[ref.K == 5]
    m = chk.merge(ref, on=["family", "T", "student", "K"], suffixes=("", "_ref"))
    if len(m) != len(chk):
        raise SystemExit(f"K = 5 の照合の行が {len(m)}／{len(chk)}")
    m["d_effect"] = (m.effect - m.effect_ref).abs()
    m["d_lo"] = (m.lo - m.lo_ref).abs()
    m["d_hi"] = (m.hi - m.hi_ref).abs()
    det = m[m["T"] != "MOSEI"]
    bad = pd.concat([m[m.d_effect > TOL], det[(det.d_lo > TOL) | (det.d_hi > TOL)]])
    m[["family", "T", "student", "K", "effect", "effect_ref", "d_effect", "lo", "lo_ref", "d_lo", "hi", "hi_ref", "d_hi"]].to_csv(
        sdir / "families_k10_check.csv", index=False)
    print(f"K = 5 の照合：{len(m)} 行，差の最大 {m.d_effect.max():.2e}，jackknife の区間の端の差の最大 {det[['d_lo', 'd_hi']].max().max():.2e}，"
          f"MOSEI の区間の端の差の最大 {m[m['T'] == 'MOSEI'][['d_lo', 'd_hi']].max().max():.3f}", flush=True)
    if len(bad):
        raise SystemExit(f"K = 5 の照合で {len(bad)} 行が食い違う（{sdir / 'families_k10_check.csv'}）")
    out = run((10,))
    out.to_csv(sdir / "families_k10.csv", index=False)
    print(f"書いた：{sdir / 'families_k10.csv'}（{len(out)} 行）", flush=True)


if __name__ == "__main__":
    main()
