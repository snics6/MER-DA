"""DAI-5 の軸を分けた版（srcw_mod・srcw_src）の「作り直した最良の β」と，その β での srcw_x − srcw_b0 の数え上げ（評価の後の記述）。

固定のファイル DAI/frozen/dai5_frozen.json の select_variants には同点の規則で選んだ β しか無い（srcw の beta_argmax に当たる値が無い）．
そこで仮のターゲットの回の保存した行（results/dai5/pseudo/units/*_p5.csv．評価のテストを含まない）から，選ぶ段（dai5.py の select_stage）と
同じ組み方（srcw の β = 0 の行を β = 0 の候補に足し，δ̄ を δ にして pick5）で選び直し，固定のファイルの β・平均・同点の印を再現することを確かめてから，
β ごとの平均が最大の β（作り直した最良の β）を書く．評価の前に固定した値ではない（gate-claim の第 4 回，docs/methods.md 9.3 節 A20）．
数え上げは results/dai5/stats/desc.csv の β の曲線の行（A = <版>_beta<β>，B = srcw_b0）から引く．作り直した β が 0 のセルは srcw_b0 と同じ識別器（identical）．
使い方：./.venv/bin/python DAI/exp/dai5_variant_argmax.py  → results/dai5/stats/variant_argmax.csv，variant_argmax_cells.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DAI" / "exp"))
sys.path.insert(0, str(ROOT))
import dai1  # noqa: E402
import dai5 as D5  # noqa: E402
import dai_data as DD  # noqa: E402

OUT = ROOT / "results" / "dai5"
AXIS_VARIANTS = ("srcw_mod", "srcw_src")                                  # 軸を分けた版（srcw_top1 は選ぶ点が無い）


def main() -> None:
    with open(ROOT / "DAI" / "frozen" / "dai5_frozen.json", encoding="utf-8") as fh:
        fz = json.load(fh)
    desc = pd.read_csv(OUT / "stats" / "desc.csv", low_memory=False)
    sel_rows, cell_rows = [], []
    for T in DD.CORPORA:
        raw = D5.load_rows(OUT, "pseudo", T)
        d = raw[raw.method == "srcw"]
        for meth in AXIS_VARIANTS:
            dm = raw[raw.method == meth]
            b0 = d[np.isclose(d.beta.astype(float), 0.0) & d.K.isin(D5.KS_VAR)].assign(method=meth)
            dv = pd.concat([dm, b0], ignore_index=True)
            for K in D5.KS_VAR:
                g = dv[dv.K == K]
                b, used, mu = D5.pick5(g, fz[T]["delta_bar"][str(K)])
                sv = fz[T]["select_variants"][f"{meth}|{K}"]
                if not (np.isclose(sv["beta"], b) and np.isclose(sv["pseudo_uar"], mu) and bool(sv["tie_used"]) == bool(used)):
                    sys.exit(f"{T} {meth} K={K}: 固定のファイルの select_variants を再現しない")
                am = float(g.groupby("beta")["uar"].mean().idxmax())
                sel_rows.append(dict(T=T, method=meth, K=K, beta_selected=b, tie_used=bool(used), beta_best_rebuilt=am))
                for s in DD.STUDENTS:
                    q = desc[(desc["T"] == T) & (desc.student == s) & (desc.K == K) & (desc.B == "srcw_b0")]
                    q = q[q.A == (meth if am == 0 else f"{meth}_beta{am}")]
                    if len(q) != 1:
                        sys.exit(f"{T} {meth} {s} K={K}: desc.csv の行が {len(q)}")
                    q = q.iloc[0]
                    cell_rows.append(dict(T=T, method=meth, student=s, K=K, beta_best_rebuilt=am, identical=bool(am == 0),
                                          effect=float(q.effect), lo=float(q.lo), hi=float(q.hi)))
    cm, dt = dai1.git_commit()
    sel = pd.DataFrame(sel_rows).assign(commit=cm, dirty=dt)
    cells = pd.DataFrame(cell_rows).assign(commit=cm, dirty=dt)
    sel.to_csv(OUT / "stats" / "variant_argmax.csv", index=False)
    cells.to_csv(OUT / "stats" / "variant_argmax_cells.csv", index=False)
    for meth in AXIS_VARIANTS:
        c = cells[(cells.method == meth) & ~cells.identical]
        print(f"{meth}: 固定のファイルを 8/8 再現．作り直した最良の β での分母 {len(c)}，下端 > 0 {int((c.lo > 0).sum())}，上端 < 0 {int((c.hi < 0).sum())}")
    print("書いた:", OUT / "stats" / "variant_argmax.csv", OUT / "stats" / "variant_argmax_cells.csv")


if __name__ == "__main__":
    main()
