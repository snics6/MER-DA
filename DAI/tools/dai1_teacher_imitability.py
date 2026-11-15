"""単独のモダリティの先生について，先生と自分の差（margin）と，先生の判断を生徒の特徴で再現できる度合い（実験 14 の R・D）を並べる
（計算の後の解析．2026-11-16．学び直しはしない）．

問い：先生のほうが当たるのに学び直しで下がった点（`docs/結果/計算の後の解析_先生と自分の差とK10_2026-11-16.md` の要点 6）は，
先生の判断を生徒の特徴で再現しにくい点か．論文の 6.5 節は，主の先生（積）についてだけ「再現できる分 P」を測っている．単独のモダリティの
先生（生徒 s に対する single_m．m ≠ s．実験 1 の none_m と同じ分類器で，実験 14 の判断 J_m と同じ）については，実験 14 が R（J_m を
正解とみなした交差の予測の UAR．候補の全体）と D（R から，感情・話者の単位・性別の層の中で J_m を入れ替えた目標の R を引いた量）を
測っている．この台本はそれを margin と gain の点に結びつける．

入力：results/dai1/stats/teacher_margin.csv（teacher = single_m の点．生徒 A・V・T．K = 5・25），
      results/dai14/summary/descriptive.csv（readout = R1（主），subset = all，quantity = R と D_shufcl）．
出力：results/dai1/stats/teacher_imitability.csv（48 点），teacher_imitability_summary.csv（記述の数え）．
記述の解析で予言は置かない．点どうしは同じ候補とテストを共有するので，順位相関と数え上げは検定ではない．
使い方：.venv/bin/python tools/dai1_teacher_imitability.py [--out-dir 一時の置き場]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
import dai1  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

TM = ROOT / "results" / "dai1" / "stats" / "teacher_margin.csv"
DESC14 = ROOT / "results" / "dai14" / "summary" / "descriptive.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dai1" / "stats"))
    args = ap.parse_args()
    sdir = Path(args.out_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    for name in ("teacher_imitability.csv", "teacher_imitability_summary.csv"):
        if (sdir / name).exists():
            raise SystemExit(f"{sdir / name} が既にある（上書きしない）")
    tm = pd.read_csv(TM)
    tm = tm[(tm.excluded.isna() | (tm.excluded == "")) & tm.teacher.str.match(r"^single_[AVT]$") & tm.student.isin(["A", "V", "T"])]
    d = pd.read_csv(DESC14)
    d = d[(d.readout == "R1") & (d.subset == "all") & d.quantity.isin(["R", "D_shufcl"])]
    w = d.pivot_table(index=["T", "K", "student", "target"], columns="quantity", values="effect").reset_index()
    lohi = d[d.quantity == "D_shufcl"][["T", "K", "student", "target", "lo", "hi"]].rename(columns={"lo": "D_lo", "hi": "D_hi"})
    w = w.merge(lohi, on=["T", "K", "student", "target"]).rename(columns={"target": "teacher", "D_shufcl": "D"})
    m = tm.merge(w, on=["T", "K", "student", "teacher"], how="left")
    if m[["R", "D"]].isna().any().any() or len(m) != 48:
        raise SystemExit(f"結びつけられない点がある（{len(m)} 点，欠け {int(m[['R', 'D']].isna().any(axis=1).sum())}）")
    # 検査の 2：margin と同じ K 本を除いた候補の上で R と D を作り直す（実験 14 の単位の保存の予測から．学び直さない）．R1 と R0 の両方
    import json
    frozen = json.load(open(ROOT / "results" / "dai1" / "select" / "frozen.json"))
    nk_vals = []
    for _, r in m.iterrows():
        vals = {ro: [] for ro in ("R1", "R0")}
        for f in range(DD.N_FOLDS):
            for rr in range(int(frozen[r.T]["n_seeds_eval"])):
                a = np.load(ROOT / "results" / "dai14" / "units" / f"{r.T}_f{f}_r{rr}.npz", allow_pickle=False)
                nk = a[f"nk__K{r.K}"].astype(bool)
                for ro in ("R1", "R0"):
                    def uar(ver):
                        tg = a[f"tgt__K{r.K}__{r.teacher}__{ver}"][nk].astype(int)
                        pr = a[f"K{r.K}__main__{ro}__{r.student}__{r.teacher}__{ver}"][nk].astype(int)
                        return 100.0 * S.uar(tg, pr)
                    vals[ro].append((uar("true"), uar("true") - uar("shufcl")))
        nk_vals.append({f"R_nk_{ro}": float(np.mean([v[0] for v in vals[ro]])) for ro in vals} |
                       {f"D_nk_{ro}": float(np.mean([v[1] for v in vals[ro]])) for ro in vals})
    m = pd.concat([m.reset_index(drop=True), pd.DataFrame(nk_vals)], axis=1)
    # 検査の 1：入れ替えで目標が変わった割合の平均が 0.40 未満の点（D が作りの上で 0 に寄る．実験 14 の assumptions.csv）
    asm = pd.read_csv(ROOT / "results" / "dai14" / "summary" / "assumptions.csv")
    fr = asm[(asm.kind == "frac_changed") & (asm.version == "shufcl")][["T", "K", "target", "frac_mean"]].rename(columns={"target": "teacher"})
    m = m.merge(fr, on=["T", "K", "teacher"], how="left")
    m["frac_low"] = m.frac_mean < 0.40
    # margin で説明しきれない gain の部分（K ごとの最小二乗の残り．記述）
    m["gain_resid"] = np.nan
    for K, x in m.groupby("K"):
        b = np.polyfit(x.margin, x.gain, 1)
        m.loc[x.index, "gain_resid"] = x.gain - np.polyval(b, x.margin)
    cm, dt = dai1.git_commit()
    m.assign(commit=cm, dirty=dt).to_csv(sdir / "teacher_imitability.csv", index=False)
    rows = []
    sets = [(K, "all_points", x) for K, x in list(m.groupby("K")) + [("all", m)]] + \
           [(K, "frac_ok", x[~x.frac_low]) for K, x in list(m.groupby("K")) + [("all", m)]]
    for K, sub, x in sets:
        pos = x[x.margin > 0]
        # 検査の 3：（コーパス，K，生徒）の中で 2 つの先生を比べる（D_nk_R1 の大小と gain・残りの大小がそろうか）
        agree_gain = agree_resid = npair = 0
        for _, g in x.groupby(["T", "K", "student"]):
            if len(g) != 2:
                continue
            npair += 1
            dD = g.D_nk_R1.iloc[0] - g.D_nk_R1.iloc[1]
            agree_gain += int(np.sign(dD) == np.sign(g.gain.iloc[0] - g.gain.iloc[1]))
            agree_resid += int(np.sign(dD) == np.sign(g.gain_resid.iloc[0] - g.gain_resid.iloc[1]))
        rows.append(dict(K=K, subset=sub, n=len(x), n_margin_pos=len(pos), n_pairs=npair, pairs_D_gain_agree=agree_gain, pairs_D_resid_agree=agree_resid,
                         spearman_gain_Dnk=float(st.spearmanr(x.D_nk_R1, x.gain).statistic) if len(x) > 2 else np.nan,
                         spearman_resid_Dnk=float(st.spearmanr(x.D_nk_R1, x.gain_resid).statistic) if len(x) > 2 else np.nan,
                         spearman_gain_margin=float(st.spearmanr(x.margin, x.gain).statistic),
                         spearman_gain_D=float(st.spearmanr(x.D, x.gain).statistic),
                         spearman_resid_D=float(st.spearmanr(x.D, x.gain_resid).statistic),
                         spearman_margin_D=float(st.spearmanr(x.D, x.margin).statistic),
                         pos_hurt_n=int((pos.gain_hi < 0).sum()), pos_hurt_D_median=float(pos[pos.gain_hi < 0].D.median()) if (pos.gain_hi < 0).any() else np.nan,
                         pos_gain_ci_pos_n=int((pos.gain_lo > 0).sum()), pos_gain_ci_pos_D_median=float(pos[pos.gain_lo > 0].D.median()) if (pos.gain_lo > 0).any() else np.nan,
                         pos_other_D_median=float(pos[(pos.gain_lo <= 0) & (pos.gain_hi >= 0)].D.median()) if ((pos.gain_lo <= 0) & (pos.gain_hi >= 0)).any() else np.nan))
    pd.DataFrame(rows).assign(commit=cm, dirty=dt).to_csv(sdir / "teacher_imitability_summary.csv", index=False)
    print(f"書いた：{sdir / 'teacher_imitability.csv'}（{len(m)} 点）と要約", flush=True)


if __name__ == "__main__":
    main()
