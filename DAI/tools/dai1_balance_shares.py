"""感情ごとの重みの合計をそろえる操作と，テストの上の答えの感情の割合・感情ごとの再現率（計算の後の解析．2026-11-16．学び直しはしない）．

問い：計算の後の解析（`docs/結果/計算の後の解析_感情ごとの当たり方_2026-11-16.md` の要点 17）で，主の疑似ラベルで学び直した分類器の
テストの上の答えの 4 感情の割合は 0.21〜0.29 に寄り，IEMOCAP と MELD では Neutral と答える割合が下がった．照合（`checks/posthoc_2026-11-16/
claim_emotion_v3.md`）は，これが形 (a) の「疑似ラベルの感情ごとの重みの合計をそろえる」作り（`dai_solver.soft_rows` の balance=True）から
見込まれる向きで，そろえない形と比べていないと指摘した．実験 1 の評価の回は，除去実験の行 d（「感情ごとの総量をそろえる操作」．
`exp/dai1_desc.py` の t_ablation：joint(one, ρ = 1, a = 1) − kd(one, T = 1, ρ = 1)．同じ疑似ラベル q で，kd は温度 1 の交差エントロピーで
重みの合計をそろえない）を評価の前に決めて計算してある．この台本は，その 2 つの学び直しと K 本の基準の分類器について，
(i) テストの上の答えの 4 感情の割合，(ii) 感情ごとの再現率の K 本の基準の分類器との差と区間，(iii) 感情ごとの再現率の joint − kd の差と区間
を並べる．主の対（版 one，ρ = 1）のほかに，疑似ラベルの版 v ∈ {one, sel} と ρ ∈ {1/16, 1/4, 1, 4} の対 joint(v, ρ, a = 1) − kd(v, T = 1, ρ) も
記述として出す（検査 checks/dai1_balance_shares/code.md の [B-足すもの]）．

量：答えの割合は，全単位の答えの数を足して全体で割る（`emotion_margin_confusion.csv.gz` の要点 17 と同じ作り）．正解の割合も同じ．
取り違えの表は単位あたりの平均の数．差と区間は `exp/dai1_desc.py` の t_recall と同じ作り（emo_of，ST.effect_stats，_fmt(scale=4)）．
乱数の列は表の名前 "ablation" の番号から，(分類器，K，対，感情) の順に引く．AFEW・IEMOCAP・MELD の区間（jackknife）は乱数に依らず，
MOSEI の区間（塊の重みの引き直し）だけが保存済みの recall.csv・ablation.csv と別の乱数の列になる（端が最大 0.15 点ほど違う）．
K は 5，10，25．分類器は声・顔・言葉・AVT．
照合（合わなければ止まる）：
  1. 主の提案 prop の指定が 48 条件とも joint(one, ρ = 1, a = 1) と同じ．
  2. 主の対の joint − none の感情ごとの差が recall.csv の family1 の effect と 1e-9 で合い，MOSEI 以外は lo・hi も 1e-9 で合う．
  3. 主の対の joint − kd の UAR の差が ablation.csv の行 d の effect と 1e-9 で合い，MOSEI 以外は lo・hi も 1e-9 で合う．
  4. none と主の joint の答えの割合が，emotion_margin_confusion.csv.gz の none_test と retrained_test（teacher = main_q）の割合と 1e-9 で合う．
  5. 3 つの差のどれも，4 感情の差の平均が同じ部品で作った UAR の差と 1e-9 で合う（作りの確かめ）．
  6. 全単位のテストに 4 感情があり，単位の commit がすべて同じで，集計の予測の照合（`Unit.pred` の rows.csv との照合）に欠けと食い違いが無い．
出力：results/dai1/stats/balance_shares.csv（コーパス × 分類器 × K × 版 × 感情：答えの割合，正解の割合），balance_recall.csv（感情ごとの差と区間．
se・df・interval の列を含む），balance_confusion.csv.gz，balance_check.csv，balance_provenance.json．記述の解析で予言は置かない（値は計算の後に見る）．
使い方：.venv/bin/python tools/dai1_balance_shares.py [--out-dir 一時の置き場] [--targets AFEW ...]
"""
from __future__ import annotations

import argparse
import hashlib
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
VERS = ("one", "sel")
INPUTS = {"frozen": OUT1 / "select" / "frozen.json", "ablation": OUT1 / "stats" / "desc" / "ablation.csv",
          "recall": OUT1 / "stats" / "desc" / "recall.csv", "emotion_margin_confusion": OUT1 / "stats" / "emotion_margin_confusion.csv.gz",
          "dai1_desc": ROOT / "exp" / "dai1_desc.py", "dai1_stats": ROOT / "exp" / "dai1_stats.py"}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT1 / "stats"))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    args = ap.parse_args()
    sdir = Path(args.out_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    outs = ("balance_shares.csv", "balance_recall.csv", "balance_confusion.csv.gz", "balance_check.csv", "balance_provenance.json")
    for name in outs:
        if (sdir / name).exists():
            raise SystemExit(f"{sdir / name} が既にある（上書きしない）")
    frozen = json.load(open(INPUTS["frozen"]))
    abl = pd.read_csv(INPUTS["ablation"], comment="#", low_memory=False)
    abl = abl[abl.row == "d"]
    rec = pd.read_csv(INPUTS["recall"], comment="#", low_memory=False)
    rec = rec[rec.kind == "family1"]
    emc = pd.read_csv(INPUTS["emotion_margin_confusion"])
    shares, recs, confs, checks, ucommits = [], [], [], [], {}
    for T in args.targets:
        fz = frozen[T]
        R = int(fz["n_seeds_eval"])
        lab = DS.Labels(T, DD.ENC1)
        chk = Counter()
        units = [DS.Unit(OUT1 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}", f, r, lab, chk) for f in range(DD.N_FOLDS) for r in range(R)]
        for u in units:
            if not (np.bincount(u.y, minlength=4) > 0).all():
                raise SystemExit(f"{u.base.name}：テストに欠ける感情がある")
        uc = sorted({(u.commit, u.frozen_sha) for u in units})
        if len(uc) != 1:
            raise SystemExit(f"{T}：単位の commit か固定のファイルが揃わない {uc}")
        ucommits[T] = dict(commit=uc[0][0], frozen_sha256=uc[0][1])
        g_eff, _ = ST.g_eff_of(T)
        ctx = DS.Ctx("eval", OUT1, T, T, fz, units, lab, g_eff, defaultdict(list), chk)
        for s in STUDENTS:
            for K in KS:
                Rf = DS.fam_rows(fz, s, K)
                main_joint = DS.P3(s, K, method="joint", teacher="one", rho=1.0, a=1)
                if Rf["prop"] != main_joint:                                           # 照合 1
                    raise SystemExit(f"{T}/{s}/K{K}：prop が joint(one, 1, 1) と違う（{Rf['prop']}）")
                specs = {"none": Rf["none"]}
                for v in VERS:
                    for rho in dai1.RHO_GRID:
                        specs[f"joint|{v}|{rho:g}"] = DS.P3(s, K, method="joint", teacher=v, rho=rho, a=1)
                        specs[f"kd|{v}|{rho:g}"] = DS.P3(s, K, method="kd", teacher=v, temp=1, rho=rho)
                for ver, spec in specs.items():
                    M = np.zeros((4, 4))
                    for u in units:
                        np.add.at(M, (u.y, u.pred(spec)), 1.0)
                    ans, tru = M.sum(0) / M.sum(), M.sum(1) / M.sum()
                    for e in range(4):
                        shares.append(dict(T=T, student=s, K=K, version=ver, emotion=e, answer_share=float(ans[e]), true_share=float(tru[e])))
                    confs += [dict(T=T, student=s, K=K, version=ver, true=i, pred=j, count_mean=float(M[i, j] / len(units)))
                              for i in range(4) for j in range(4)]
                    if ver in ("none", "joint|one|1"):                                 # 照合 4
                        who = "none_test" if ver == "none" else "retrained_test"
                        x = emc[(emc["T"] == T) & (emc.student == s) & (emc.K == K) & (emc.who == who)]
                        x = x[x.teacher == ("(self)" if ver == "none" else "main_q")]
                        Mr = x.pivot_table(index="true", columns="pred", values="count_mean", aggfunc="sum").reindex(index=range(4), columns=range(4)).fillna(0).values
                        d4 = float(np.abs(Mr.sum(0) / Mr.sum() - ans).max())
                        checks.append(dict(T=T, student=s, K=K, pair=ver, check="answer_share_vs_emotion_margin_confusion", value=d4))
                        if d4 > 1e-9:
                            raise SystemExit(f"{T}/{s}/K{K}/{ver}：答えの割合が emotion_margin_confusion.csv.gz と {d4:.2e} 違う")
                rng = ctx.rng("ablation")
                cells = {v: ctx.cell(sp) for v, sp in specs.items()}
                pairs = []
                for v in VERS:
                    for rho in dai1.RHO_GRID:
                        j_, k_ = f"joint|{v}|{rho:g}", f"kd|{v}|{rho:g}"
                        pairs += [(j_, "none"), (k_, "none"), (j_, k_)]
                for a_, b_ in pairs:
                    es = [DS._fmt(ST.effect_stats(DS.emo_of(cells[a_], e), DS.emo_of(cells[b_], e), ctx.g_eff, rng), K, scale=4.0) for e in range(4)]
                    uar = DS._fmt(ST.effect_stats(cells[a_], cells[b_], ctx.g_eff, rng), K)
                    d5 = abs(np.mean([x["effect"] for x in es]) - uar["effect"])          # 照合 5
                    checks.append(dict(T=T, student=s, K=K, pair=f"{a_} - {b_}", check="emotion_mean_vs_uar", value=float(d5)))
                    if d5 > 1e-9:
                        raise SystemExit(f"{T}/{s}/K{K}/{a_}-{b_}：4 感情の差の平均が UAR の差と {d5:.2e} 違う")
                    if (a_, b_) == ("joint|one|1", "none"):                               # 照合 2
                        rr = rec[(rec["T"] == T) & (rec.student == s) & (rec.K == K)].sort_values("emotion")
                        if len(rr) != 4:
                            raise SystemExit(f"{T}/{s}/K{K}：recall.csv の family1 の行が {len(rr)} 個")
                        de = max(abs(es[e]["effect"] - float(rr.effect.iloc[e])) for e in range(4))
                        dl = max(max(abs(es[e]["lo"] - float(rr.lo.iloc[e])), abs(es[e]["hi"] - float(rr.hi.iloc[e]))) for e in range(4))
                        checks += [dict(T=T, student=s, K=K, pair="joint|one|1 - none", check="vs_recall_family1_effect", value=float(de)),
                                   dict(T=T, student=s, K=K, pair="joint|one|1 - none", check="vs_recall_family1_lohi", value=float(dl))]
                        if de > 1e-9 or (T != "MOSEI" and dl > 1e-9):
                            raise SystemExit(f"{T}/{s}/K{K}：joint − none が recall.csv の family1 と違う（{de:.2e}，{dl:.2e}）")
                    if (a_, b_) == ("joint|one|1", "kd|one|1"):                           # 照合 3
                        ref = abl[(abl["T"] == T) & (abl.student == s) & (abl.K == K)]
                        if len(ref) != 1:
                            raise SystemExit(f"{T}/{s}/K{K}：ablation.csv の行 d が {len(ref)} 個")
                        de = abs(uar["effect"] - float(ref.effect.iloc[0]))
                        dl = max(abs(uar["lo"] - float(ref.lo.iloc[0])), abs(uar["hi"] - float(ref.hi.iloc[0])))
                        checks += [dict(T=T, student=s, K=K, pair="joint|one|1 - kd|one|1", check="vs_ablation_d_effect", value=float(de)),
                                   dict(T=T, student=s, K=K, pair="joint|one|1 - kd|one|1", check="vs_ablation_d_lohi", value=float(dl))]
                        if de > 1e-9 or (T != "MOSEI" and dl > 1e-9):
                            raise SystemExit(f"{T}/{s}/K{K}：joint − kd が ablation.csv の行 d と違う（{de:.2e}，{dl:.2e}）")
                    for e in range(4):
                        recs.append(dict(T=T, student=s, K=K, pair=f"{a_} - {b_}", main_pair=(a_, b_) in (("joint|one|1", "none"), ("kd|one|1", "none"), ("joint|one|1", "kd|one|1")),
                                         emotion=e, effect=float(es[e]["effect"]), lo=float(es[e]["lo"]), hi=float(es[e]["hi"]),
                                         se=float(es[e].get("se", np.nan)), df=es[e].get("df"), interval=es[e].get("interval"), n_units=int(es[e]["n_units"]),
                                         uar_effect=float(uar["effect"]), uar_lo=float(uar["lo"]), uar_hi=float(uar["hi"]), uar_se=float(uar.get("se", np.nan))))
                ctx.clear()
        if any(k[1] in ("missing_row", "mismatch") and v > 0 for k, v in chk.items()):           # 照合 6
            raise SystemExit(f"{T}：集計の予測の照合で欠けか食い違いがある {dict(chk)}")
        checks += [dict(T=T, check=f"pred_{a}_{b}", value=v) for (a, b), v in chk.items()]
        print(f"{T} 済み", flush=True)
    cm, dt = dai1.git_commit()
    pd.DataFrame(shares).assign(commit=cm, dirty=dt).to_csv(sdir / "balance_shares.csv", index=False)
    pd.DataFrame(recs).assign(commit=cm, dirty=dt).to_csv(sdir / "balance_recall.csv", index=False)
    pd.DataFrame(confs).to_csv(sdir / "balance_confusion.csv.gz", index=False)
    pd.DataFrame(checks).assign(commit=cm, dirty=dt).to_csv(sdir / "balance_check.csv", index=False)
    prov = dict(script=str(Path(__file__).resolve().relative_to(ROOT)), script_sha256=sha(Path(__file__)), commit=cm, dirty=dt,
                targets=args.targets, units=ucommits, inputs={k: dict(path=str(v.relative_to(ROOT)), sha256=sha(v)) for k, v in INPUTS.items()},
                numpy=np.__version__, pandas=pd.__version__)
    json.dump(prov, open(sdir / "balance_provenance.json", "w"), indent=1, ensure_ascii=False)
    print(f"書いた：{sdir / 'balance_shares.csv'} ほか 4 つ", flush=True)


if __name__ == "__main__":
    main()
