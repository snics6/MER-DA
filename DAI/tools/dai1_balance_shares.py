"""感情ごとの重みの合計をそろえる操作と，テストの上の答えの感情の割合・感情ごとの再現率（計算の後の解析．2026-11-16．学び直しはしない）．

問い：計算の後の解析（`docs/結果/計算の後の解析_感情ごとの当たり方_2026-11-16.md` の要点 17）で，主の疑似ラベルで学び直した分類器の
テストの上の答えの 4 感情の割合は 0.21〜0.29 に寄り，IEMOCAP と MELD では Neutral と答える割合が下がった．照合（`checks/posthoc_2026-11-16/
claim_emotion_v3.md`）は，これが形 (a) の「疑似ラベルの感情ごとの重みの合計をそろえる」作り（`dai_solver.soft_rows` の balance=True）から
見込まれる向きで，そろえない形と比べていないと指摘した．実験 1 の評価の回は，除去実験の行 d（「感情ごとの総量をそろえる操作」．
`exp/dai1_desc.py` の t_ablation：joint(one, ρ = 1, a = 1) − kd(one, T = 1, ρ = 1)．同じ疑似ラベル q で，kd は温度 1 の交差エントロピーで
重みの合計をそろえない）を評価の前に決めて計算してある．この台本は，その 2 つの学び直しと K 本の基準の分類器について，
(i) テストの上の答えの 4 感情の割合，(ii) 感情ごとの再現率の K 本の基準の分類器との差と区間，(iii) 感情ごとの再現率の joint − kd の差と区間
を並べる．

量：答えの割合は 1 単位の取り違えの数の列の和 ÷ 全体（単位で平均）．差と区間は `exp/dai1_desc.py` の t_recall と同じ作り（emo_of，
ST.effect_stats，_fmt(scale=4)．乱数の列は表の名前 "ablation" の番号．MOSEI だけが乱数に依る）．K は 5，10，25．分類器は声・顔・言葉・AVT．
照合：3 つの差のどれも，4 感情の差の平均が同じ部品で作った UAR の差と 1e-9 で合い，
joint − kd の UAR の差は ablation.csv の行 d の effect と 1e-9 で合う．集計の予測の照合（`Unit.pred` の rows.csv との照合）に欠けと食い違いが無い．
出力：results/dai1/stats/balance_shares.csv（コーパス × 分類器 × K × 版 × 感情：答えの割合，正解の割合），balance_recall.csv（感情ごとの差と区間），
balance_check.csv．記述の解析で予言は置かない（値は計算の後に見る）．
使い方：.venv/bin/python tools/dai1_balance_shares.py [--out-dir 一時の置き場] [--targets AFEW ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
import dai1  # noqa: E402
import dai1_desc as DS  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai_data as DD  # noqa: E402

OUT1 = ROOT / "results" / "dai1"
KS = (5, 10, 25)
STUDENTS = ("A", "V", "T", "AVT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT1 / "stats"))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    args = ap.parse_args()
    sdir = Path(args.out_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    for name in ("balance_shares.csv", "balance_recall.csv", "balance_check.csv"):
        if (sdir / name).exists():
            raise SystemExit(f"{sdir / name} が既にある（上書きしない）")
    frozen = json.load(open(OUT1 / "select" / "frozen.json"))
    abl = pd.read_csv(OUT1 / "stats" / "desc" / "ablation.csv", comment="#")
    abl = abl[abl.row == "d"]
    shares, recs, checks = [], [], []
    for T in args.targets:
        fz = frozen[T]
        R = int(fz["n_seeds_eval"])
        lab = DS.Labels(T, DD.ENC1)
        chk = Counter()
        units = [DS.Unit(OUT1 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}", f, r, lab, chk) for f in range(DD.N_FOLDS) for r in range(R)]
        for u in units:
            if not (np.bincount(u.y, minlength=4) > 0).all():
                raise SystemExit(f"{u.base.name}：テストに欠ける感情がある")
        g_eff, _ = ST.g_eff_of(T)
        ctx = DS.Ctx("eval", OUT1, T, T, fz, units, lab, g_eff, defaultdict(list), chk)
        for s in STUDENTS:
            for K in KS:
                specs = {"none": DS.fam_rows(fz, s, K)["none"],
                         "joint": DS.P3(s, K, method="joint", teacher="one", rho=1.0, a=1),
                         "kd": DS.P3(s, K, method="kd", teacher="one", temp=1, rho=1.0)}
                for ver, spec in specs.items():
                    ans, tru = [], []
                    for u in units:
                        p = u.pred(spec)
                        ans.append(np.bincount(p, minlength=4) / len(p))
                        tru.append(np.bincount(u.y, minlength=4) / len(u.y))
                    ans, tru = np.mean(ans, 0), np.mean(tru, 0)
                    for e in range(4):
                        shares.append(dict(T=T, student=s, K=K, version=ver, emotion=e, answer_share=float(ans[e]), true_share=float(tru[e])))
                rng = ctx.rng("ablation")
                cells = {v: ctx.cell(sp) for v, sp in specs.items()}
                for a_, b_ in (("joint", "none"), ("kd", "none"), ("joint", "kd")):
                    es = [DS._fmt(ST.effect_stats(DS.emo_of(cells[a_], e), DS.emo_of(cells[b_], e), ctx.g_eff, rng), K, scale=4.0) for e in range(4)]
                    uar = DS._fmt(ST.effect_stats(cells[a_], cells[b_], ctx.g_eff, rng), K)
                    d = abs(np.mean([x["effect"] for x in es]) - uar["effect"])
                    checks.append(dict(T=T, student=s, K=K, pair=f"{a_}-{b_}", check="emotion_mean_vs_uar", value=float(d)))
                    if d > 1e-9:
                        raise SystemExit(f"{T}/{s}/K{K}/{a_}-{b_}：4 感情の差の平均が UAR の差と {d:.2e} 違う")
                    if (a_, b_) == ("joint", "kd"):
                        ref = abl[(abl["T"] == T) & (abl.student == s) & (abl.K == K)]
                        if len(ref) != 1:
                            raise SystemExit(f"{T}/{s}/K{K}：ablation.csv の行 d が {len(ref)} 個")
                        d2 = abs(uar["effect"] - float(ref.effect.iloc[0]))
                        checks.append(dict(T=T, student=s, K=K, pair="joint-kd", check="uar_vs_ablation_d", value=float(d2)))
                        if d2 > 1e-9:
                            raise SystemExit(f"{T}/{s}/K{K}：joint − kd が ablation.csv の行 d と {d2:.2e} 違う")
                    for e in range(4):
                        recs.append(dict(T=T, student=s, K=K, pair=f"{a_} - {b_}", emotion=e, effect=float(es[e]["effect"]),
                                         lo=float(es[e]["lo"]), hi=float(es[e]["hi"]), n_units=int(es[e]["n_units"]),
                                         uar_effect=float(uar["effect"]), uar_lo=float(uar["lo"]), uar_hi=float(uar["hi"])))
                ctx.clear()
        if any(k[1] in ("missing_row", "mismatch") and v > 0 for k, v in chk.items()):
            raise SystemExit(f"{T}：集計の予測の照合で欠けか食い違いがある {dict(chk)}")
        checks += [dict(T=T, check=f"pred_{a}_{b}", value=v) for (a, b), v in chk.items()]
        print(f"{T} 済み", flush=True)
    cm, dt = dai1.git_commit()
    pd.DataFrame(shares).assign(commit=cm, dirty=dt, units_commit="ba98974").to_csv(sdir / "balance_shares.csv", index=False)
    pd.DataFrame(recs).assign(commit=cm, dirty=dt, units_commit="ba98974").to_csv(sdir / "balance_recall.csv", index=False)
    pd.DataFrame(checks).assign(commit=cm, dirty=dt).to_csv(sdir / "balance_check.csv", index=False)
    print(f"書いた：{sdir / 'balance_shares.csv'} ほか 2 つ", flush=True)


if __name__ == "__main__":
    main()
