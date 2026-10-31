"""【探索的・下調べ】E49 の後：生徒の出発点（E39 の下書き）を仲間に入れた見込みは，出発点よりどれだけ当たるか。

E49 の結果を見て，ソースの票の見込み q が出発点 p より当たるターゲットでは教える選び方が勝ち，当たらないターゲットでは負けることに気づいた．
次の実験（見込みに出発点を仲間として入れる）の前に，等しい重みで掛け合わせただけの見込み（PoE）の当たり方を候補の上で測る．
  p    ：E39 の下書きの確率
  q    ：E44 の C（同じエンコーダのソースの票）
  q_o  ：別のエンコーダ（A なら HuBERT）のソースの票の C
  pq   ：softmax(log p + log q)
  pqo  ：softmax(log p + log q + log q_o)
A と A(HuBERT) の 14 ターゲット，反復 0〜9．

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/check_belief_stack.py

出力：`results/explore_relation/belief_stack_check.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from check_lll_cues import PAIR, committee_q  # noqa: E402
from data import groups_for  # noqa: E402
from teach_student39 import Student  # noqa: E402

OUT_REL = ROOT / "results" / "explore_relation"


def sm(L):
    L = L - L.max(1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(1, keepdims=True)


def main() -> int:
    rows = []
    sel = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"])
    for kind in PAIR:
        other = PAIR[kind]
        groups = groups_for(kind)
        data = {g: M.load(g, kind) for g in groups}
        data_o = {g: M.load(g, other) for g in groups}
        f, Cs, P, Y = CM.prepare(kind, data)
        f_o, Cs_o, P_o, Y_o = CM.prepare(other, data_o)
        cache = {}
        for T in groups:
            S = [s for s in data if CM.corp(s) != CM.corp(T)]
            s_ = sel.loc[(kind, T)]
            C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
            src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
            ev, ev_o = data[T], data_o[T]
            ctx, ctx_o = CB.Ctx(ev, range(10), ["5"], center="ind"), CB.Ctx(ev_o, range(10), ["5"], center="ind")
            for r in range(10):
                pool = np.flatnonzero(~ctx.te[r])
                yq = ev.y[pool]
                Xc = M.rows(ev.X, pool) - ctx.ct[r]
                Xc_o = M.rows(ev_o.X, pool) - ctx_o.ct[r]
                st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, float(s_.gamma), float(s_.rho_s), [1.0])
                p = torch.softmax(st.L0, 1).cpu().numpy()
                q, _ = committee_q(kind, T, data, f, Cs, P, Y, Xc)
                q_o, _ = committee_q(other, T, data_o, f_o, Cs_o, P_o, Y_o, Xc_o)
                lg = lambda a: np.log(np.clip(a, 1e-12, None))  # noqa: E731
                bel = {"p": p, "q": q, "q_o": q_o, "pq": sm(lg(p) + lg(q)), "pqo": sm(lg(p) + lg(q) + lg(q_o))}
                row = {"kind": kind, "target": T, "rep": r}
                for k, b in bel.items():
                    row[f"uar_{k}"] = CB.uar(yq, b.argmax(1))
                    row[f"dis_{k}"] = float((b.argmax(1) != p.argmax(1)).mean())
                rows.append(row)
            print(kind, T, "済み", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT_REL / "belief_stack_check.csv", index=False)
    g = d.groupby(["kind", "target"]).mean(numeric_only=True).drop(columns="rep")
    pd.set_option("display.width", 250)
    print("== ターゲットごと（％）")
    print((g * 100).round(1).to_string())
    print("== 種類の平均（％）")
    print((g.groupby("kind").mean() * 100).round(2).to_string())
    e49 = pd.read_csv(OUT_REL / "e49_targets.csv", dtype={"K": str})
    e49 = e49[e49.K == "5"].set_index(["kind", "target"])
    gain = (e49["C:0000"] - e49.random_pool).reindex(g.index)
    adv = (g.uar_q - g.uar_p) * 100
    print("== q − p と，E49 の C:0000 − random_pool（K=5）の順位相関：", round(float(spearmanr(adv, gain).statistic), 2), f"（{len(adv)} ターゲット）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
