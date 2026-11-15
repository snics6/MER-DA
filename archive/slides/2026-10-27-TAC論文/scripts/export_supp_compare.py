"""補足 3「比較の詳細」（parts/supp_compare.tex）の表と図の値を CSV に書き出す（資料 2026-10-27-TAC論文）．

台帳（paper/results/numbers.json）に無い値だけをここで作る．集計は台帳の comparison.py の _world と，
論文の図の台本 paper/scripts/figures/fig_compare.py（world_table・best_of・best_label・diffs）と同じ．
台帳に同じ量の key があれば照合し，食い違えば止める（見本は scripts/export_figdata.py の check）．
各 CSV の先頭の # 行に出所を書く（pgfplots は # の行を読み飛ばす）．

    cd 2026-10-27-TAC論文 && ../../.venv/bin/python scripts/export_supp_compare.py

書くもの（figures/）
  supp_c_range.csv      11 の生徒 × K ∈ {0, 5}：最も強い比較手法の略称と，3 つの手法の出発点との差のターゲットの間の最小・最大
  supp_c_kbest.csv      AVT の世界（1 つ目・2 つ目のエンコーダ）の 4 生徒 × K：K ごとに選び直した最も強い比較手法の略称
  supp_c_stats.csv      11 の生徒 × K ∈ {0, 5}：同じテストの話者の集合の反復を平均した後の Holm 後の p（results/stats_compare/tests.csv）
  supp_c_testsets.csv   ターゲットごとの，反復 10 回のテストの話者の集合の数（results/stats_compare/testsets.csv）
  supp_c_collapse.csv   AVT の単独の生徒（K=0）：出発点と，固定版のエントロピー最小化・共正則化・事前確率の調整の UAR
  supp_c_entmin.csv     融合の生徒（K=0）：世界 × ターゲットの固定版のエントロピー最小化と拡張版の，出発点との差
  supp_c_entpick.csv    融合の生徒：調整版のエントロピー最小化が「使わない」を選んだターゲットの数（世界 × K）
  supp_c_negcells.csv   拡張版が出発点を下回ったターゲットとセル（生徒と K ∈ {0, 5}）の組（results/stats_compare/tests.csv）
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

DECK = pathlib.Path(__file__).resolve().parents[1]
MERDA = DECK.parents[1]
PAPER = MERDA / "paper"
RESULTS = MERDA / "results"
sys.path.insert(0, str(PAPER / "scripts" / "figures"))
sys.path.insert(0, str(PAPER / "scripts"))
sys.dont_write_bytecode = True

import fig_compare as F  # noqa: E402
import fig_entmin  # noqa: E402
import fig_kcurves  # noqa: E402

OUT = DECK / "figures"
LEDGER = json.loads((PAPER / "results" / "numbers.json").read_text())
N_CHECKED = [0]
SHORT = {"audio_wavlm": "a", "video": "v", "text_roberta": "t", "audio_hubert": "a", "video_clip": "v", "text_deberta": "t"}
TGT_JA = {"IEMOCAP-impro": "IEMOCAP 即興", "IEMOCAP-script": "IEMOCAP 台本", "MELD": "MELD", "MOSEI": "CMU-MOSEI",
          "AFEW": "AFEW", "CREMA-D": "CREMA-D"}
TGT_KEY = {"IEMOCAP-impro": "impro", "IEMOCAP-script": "script", "MELD": "meld", "MOSEI": "mosei", "AFEW": "afew", "CREMA-D": "cremad"}
STUDENT_JA = {"a": "音声", "v": "顔", "t": "言葉", "f": "融合"}
WORLD_JA = {"avt": "AVT", "av": "AV", "vt": "VT", "at": "AT", "w2": "AVT（2 つ目）"}


def short(s: str) -> str:
    return SHORT.get(s, "f")


def check(key: str, v: float, tol: float = 1e-6) -> None:
    """台帳に key があれば値を照合する（無ければ何もしない）．"""
    if key not in LEDGER:
        return
    w = float(LEDGER[key]["value"])
    if abs(w - float(v)) > tol:
        raise SystemExit(f"export_supp_compare: {key} は計算した値 {v} と台帳の値 {w} が食い違う")
    N_CHECKED[0] += 1


def must(key: str, v: float, tol: float = 1e-6) -> None:
    """台帳に必ずある key と照合する．"""
    if key not in LEDGER:
        raise SystemExit(f"export_supp_compare: 照合する key {key} が台帳に無い")
    check(key, v, tol)


def write(name: str, df: pd.DataFrame, src: str) -> None:
    OUT.mkdir(exist_ok=True)
    head = "".join(f"# {line}\n" for line in src.strip().splitlines())
    (OUT / name).write_text(head + df.to_csv(index=False, float_format="%.6g"))
    print(f"  {name}: {len(df)} 行")


def f1(v: float, nd: int = 1) -> str:
    """小数 nd 桁の TeX の文字列（台帳の \\NUM と同じく負の符号は \\ensuremath{-}．−0.0 は 0.0 にする）．"""
    t = f"{abs(v):.{nd}f}"
    return ("\\ensuremath{-}" + t) if v < 0 and float(t) != 0 else t


def rng(lo: float, hi: float) -> str:
    return f"{f1(lo)}〜{f1(hi)}"


def label(wk: str, s: str, K: str) -> str:
    b = F.best_label(wk, s, K)
    return "なし" if b == "none" else b


# ---- ターゲットの間の範囲 ------------------------------------------------------
GROUP_ORDER = [("avt", "a"), ("avt", "v"), ("avt", "t"), ("avt", "f"), ("av", "f"), ("vt", "f"), ("at", "f"),
               ("w2", "a"), ("w2", "v"), ("w2", "t"), ("w2", "f")]     # parts/supp_compare.tex はこの行の順で値を読む


def ranges() -> None:
    if [(wk, short(s)) for wk, s, _, _ in F.GROUPS] != GROUP_ORDER:
        raise SystemExit("export_supp_compare: fig_compare.GROUPS の順が TeX の決め打ちと違う")
    rows = []
    for wk, s, lab, _ in F.GROUPS:
        st = short(s)
        r = dict(world=WORLD_JA[wk], student=STUDENT_JA[st] if st != "f" else f"融合（{lab}）")
        for K in ("0", "5"):
            b = F.best_of(wk, s, K)
            r[f"lab{K}"] = label(wk, s, K)
            for mk, col, key in (("b", f"{b}:tuned", "cmp-best-vs-e39"), ("o", "ours", "cmp-ours-vs-e39"),
                                 ("e", "oursx:tuned", "cmp-oursx-vs-e39")):
                v = F.diffs(wk, s, K, col)
                must(f"{key}-{wk}-{st}-k{K}", v.mean())
                r[f"{mk}{K}lo"], r[f"{mk}{K}hi"] = float(v.min()), float(v.max())
                r[f"{mk}{K}n"] = int(v.size)
                r[f"{mk}{K}r"] = rng(v.min(), v.max())
        rows.append(r)
    write("supp_c_range.csv", pd.DataFrame(rows), """出所：results/baselines_<世界>/<T>.csv と <T>_pseudo.csv，results/multimodal_distill_<世界>/<T>.csv と <T>_pseudo.csv
（台帳 paper/scripts/ledger/comparison.py の _world を paper/scripts/figures/fig_compare.py の world_table が呼ぶ）．行は fig_compare.GROUPS の順の 11 の生徒．
lab0・lab5 は K=0・5 で最も強い比較手法の略称（調整版がどのターゲットでも「使わない」を選んだときは「なし」．fig_compare.best_label）．
b・o・e は最も強い比較手法の調整版・基本版・拡張版の，5 つの値を選んだ出発点との差（UAR のポイント）．<手法><K>lo・hi はターゲットの間の最小・最大，n はターゲットの数，r は「最小〜最大」を小数 1 桁にした TeX の文字列．
各ターゲットの値は反復 0〜9 の平均（仮のターゲットは反復 0〜4）．ターゲットの平均は台帳 cmp-best-vs-e39-*・cmp-ours-vs-e39-*・cmp-oursx-vs-e39-* と照合した""")


# ---- K ごとの最も強い比較手法 ---------------------------------------------------
W2_STUDENTS = [("audio_hubert", "a"), ("video_clip", "v"), ("text_deberta", "t"), ("fuse2-audio-video-text", "f")]


def kbest() -> None:
    rows = []
    for K in fig_kcurves.KS:
        r = dict(K=K)
        for wk, studs in (("avt", [(s, short(s)) for s, _ in fig_kcurves.STUDENTS]), ("w2", W2_STUDENTS)):
            for s, st in studs:
                b = F.best_of(wk, s, K)
                must(f"cmp-best-vs-e39-{wk}-{st}-k{K}", F.diffs(wk, s, K, f"{b}:tuned").mean())
                r[f"{wk}{st}"] = label(wk, s, K)
        rows.append(r)
    write("supp_c_kbest.csv", pd.DataFrame(rows), """出所：supp_c_range.csv と同じ（台帳 comparison.py の _world）．AVT の世界の 4 生徒（列 avt<生徒>）と，2 つ目のエンコーダの AVT の世界の 4 生徒（列 w2<生徒>）．
生徒は a 音声，v 顔，t 言葉，f 融合．値は K ごとに選び直した最も強い比較手法の略称（fig_compare.best_label．「なし」は調整版がどのターゲットでも使わないを選んだこと）．
最も強い比較手法の差は台帳 cmp-best-vs-e39-<世界>-<生徒>-k<K> と照合した""")


# ---- 検定：同じテストの話者の集合の反復を平均した後の p ----------------------------
def pfmt(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def stats() -> None:
    t = pd.read_csv(RESULTS / "stats_compare" / "tests.csv", comment="#", dtype={"K": str}, keep_default_na=False)
    a = t[(t.experiment.isin(["E61", "E62"])) & (t.comparison == "a")]
    rows = []
    for wk, s, lab, _ in F.GROUPS:
        st = short(s)
        r = dict(world=WORLD_JA[wk], student=STUDENT_JA[st])
        for K in ("0", "5"):
            g = a[(a.world == wk) & (a.student == s) & (a.K == K)]
            if len(g) != 1:
                raise SystemExit(f"export_supp_compare: tests.csv の比較 (a) の {wk} {s} K={K} が {len(g)} 行")
            g = g.iloc[0]
            b = f"st-oursx-best-{wk}-{st}-k{K}"
            must(f"{b}-mean", float(g["mean"]))
            must(f"{b}-dz", float(g["dz"]))
            must(f"{b}-pholm", float(g["p_holm"]))
            must(f"{b}-tgtpos", int(g["tgt_pos"]))
            pd_ = float(g["p_dedup_holm"])
            r[f"pd{K}"] = pfmt(pd_)
            r[f"sd{K}"] = int(pd_ < 0.05)
            r[f"nu{K}"] = int(g["n_units_dedup"])
            r[f"nt{K}"] = int(g["n_targets"])
        rows.append(r)
    d = pd.DataFrame(rows)
    for K in ("0", "5"):
        must(f"st-a-k{K}-sigdedup-n", int(d[f"sd{K}"].sum()))
    write("supp_c_stats.csv", d, """出所：results/stats_compare/tests.csv（exp/stats_compare.py．E64）の比較 (a)（拡張版の調整版 − 最も強い比較手法の調整版），実験 E61・E62．行は fig_compare.GROUPS の順の 11 の生徒．
pd<K> は，同じテストの話者の集合を持つ反復を平均した単位での両側の Wilcoxon の符号順位検定の p に，族 F-a（55 検定）の Holm の補正をかけた値（列 p_dedup_holm）．
sd<K> は pd<K> が 0.05 未満なら 1，nu<K> はまとめた後の単位の数（列 n_units_dedup），nt<K> はターゲットの数．
同じ行の mean・dz・p_holm・tgt_pos は台帳 st-oursx-best-<世界>-<生徒>-k<K>-* と，sd の合計は台帳 st-a-k<K>-sigdedup-n と照合した""")


def testsets() -> None:
    t = pd.read_csv(RESULTS / "stats_compare" / "testsets.csv", comment="#")
    t = t[t.world != "ft"]                                   # 微調整の確かめ（反復 0〜2）は除く
    rows = []
    for T in ("IEMOCAP-impro", "IEMOCAP-script", "MELD", "MOSEI", "AFEW", "CREMA-D"):
        g = t[t.target == T]
        n = g.groupby("world").testset.nunique()
        if n.nunique() != 1:
            raise SystemExit(f"export_supp_compare: {T} のテストの話者の集合の数が世界で違う：{n.to_dict()}")
        sp = g.groupby("testset").n_test_speakers.first()
        rows.append(dict(target=TGT_JA[T], nsets=int(n.iloc[0]), reps=int(g.groupby("world").rep.nunique().iloc[0]),
                         spk_lo=int(sp.min()), spk_hi=int(sp.max()), worlds="・".join(WORLD_JA[w] for w in sorted(n.index))))
    d = pd.DataFrame(rows)
    write("supp_c_testsets.csv", d, """出所：results/stats_compare/testsets.csv（exp/stats_compare.py が exp/map.py の test_mask で作った，反復ごとのテストの話者の集合）．
nsets は反復 0〜9（reps 回）のテストの話者の集合の数（世界によらず同じことを確かめた），spk_lo・spk_hi は集合ごとのテストの話者の数の最小・最大，worlds はそのターゲットを使う世界（微調整の確かめの行 world=ft は除いた）""")


# ---- 固定版が 1 つの感情に潰れる例 -------------------------------------------------
COLLAPSE = [("e39", "出発点"), ("entmin:fixed", "EntM"), ("coreg:fixed", "CoR"), ("saerens:fixed", "PA")]


def collapse() -> None:
    d = F.world_table("avt")
    rows = []
    for i, (s, st) in enumerate((("audio_wavlm", "a"), ("video", "v"), ("text_roberta", "t"))):
        g = d[(d.student == s) & (d.K == "0")]
        r = dict(x=i, student=STUDENT_JA[st])
        start = float(g["e39"].mean())
        must(f"tab-avt-{st}-k0-e39", start)
        r["start"] = start
        for col, ab in COLLAPSE[1:]:
            v = float(g[col].mean())
            m = col.split(":")[0]
            must(f"cmp-{m}-fixed-vs-e39-avt-{st}-k0", v - start)
            r[ab] = v
        rows.append(r)
    write("supp_c_collapse.csv", pd.DataFrame(rows), """出所：results/baselines_fuse-audio-video-text/<T>.csv（台帳 comparison.py の _world）．AVT の世界の単独の生徒，K=0，4 ターゲット，反復 0〜9．
start は 5 つの値を選んだ出発点，EntM・CoR・PA は元の論文の値を使う固定版（エントロピー最小化 entmin|-|n，共正則化 coreg|-|n，事前確率の調整 saerens|-|0）の UAR（%）のターゲットの平均．
出発点は台帳 tab-avt-<生徒>-k0-e39 と，固定版 − 出発点は台帳 cmp-<手法>-fixed-vs-e39-avt-<生徒>-k0 と照合した""")


# ---- エントロピー最小化のターゲットごとの値 ------------------------------------------
def entmin() -> None:
    rows, picks = [], []
    for wk, s, lab in fig_entmin.WORLDS:
        d = F.world_table(wk)
        g = d[(d.student == s) & (d.K == "0")].set_index("target")
        for T in g.index:
            ent = float(g.loc[T, "entmin:fixed"] - g.loc[T, "e39"])
            ox = float(g.loc[T, "oursx:tuned"] - g.loc[T, "e39"])
            must(f"cmp-entmin-fixed-vs-e39-{wk}-f-k0-{TGT_KEY[T]}", ent)
            check(f"cmp-oursx-vs-e39-{wk}-f-k0-{TGT_KEY[T]}", ox)
            rows.append(dict(world=lab, target=TGT_JA[T], tkey=TGT_KEY[T], ent=ent, ox=ox, oxs=f1(ox)))
        r = dict(world=lab, n=int(g.shape[0]))
        for K in fig_kcurves.KS:
            gk = d[(d.student == s) & (d.K == K)]
            r[f"none{K}"] = int(gk["entmin:pick"].str.startswith("none|").sum())
        picks.append(r)
    order = [(r["world"], r["tkey"]) for r in rows]
    expect = ([("AVT", t) for t in ("impro", "script", "meld", "mosei")] + [("VT", t) for t in ("impro", "script", "meld", "mosei")]
              + [("AT", t) for t in ("impro", "script", "meld", "mosei")]
              + [("AV", t) for t in ("afew", "cremad", "impro", "script", "meld", "mosei")])
    if order != expect:                               # parts/supp_compare.tex はこの行の順で値を読む
        raise SystemExit(f"export_supp_compare: supp_c_entmin.csv の行の順が TeX の決め打ちと違う：{order}")
    write("supp_c_entmin.csv", pd.DataFrame(rows), """出所：results/baselines_<世界>/<T>.csv（台帳 comparison.py の _world）．融合の生徒，K=0，反復 0〜9 の平均．
ent は固定版のエントロピー最小化（候補 1 本あたり重み 1．entmin|-|n）− 5 つの値を選んだ出発点，ox は拡張版の調整版 − 出発点（UAR のポイント）．
oxs は ox を小数 1 桁にした TeX の文字列．ent は台帳 cmp-entmin-fixed-vs-e39-<世界>-f-k0-<ターゲット> と，AVT の ox は台帳 cmp-oursx-vs-e39-avt-f-k0-<ターゲット> と照合した（ほかの世界の ox は台帳に無い）""")
    write("supp_c_entpick.csv", pd.DataFrame(picks), """出所：results/baselines_<世界>/<T>.csv と <T>_pseudo.csv（台帳 comparison.py の _world が作り直した選択．列 entmin:pick）．融合の生徒．
none<K> は，調整版のエントロピー最小化が仮のターゲットで「使わない」（none）を選んだターゲットの数（全 n のうち）""")


# ---- 拡張版が出発点を下回った組 ----------------------------------------------------
def negcells() -> None:
    t = pd.read_csv(RESULTS / "stats_compare" / "tests.csv", comment="#", dtype={"K": str}, keep_default_na=False)
    b = t[(t.experiment.isin(["E61", "E62"])) & (t.comparison == "b") & (t.K.isin(["0", "5"]))]
    rows, allv = [], []
    for _, r in b.iterrows():
        for x in r.tgt_means.split("；"):
            T, v = x.rsplit(" ", 1)
            v = float(v)
            allv.append(v)
            if v < 0:
                st = r.student_key
                rows.append(dict(world=WORLD_JA[r.world], student=STUDENT_JA[st], K=r.K, target=TGT_JA[T], diff=v, diffs=f1(v, 2)))
    must("st-b-tgt-n", len(allv))
    must("st-b-tgtneg-n", len(rows))
    must("st-b-tgtneg-worst", min(allv))
    d = pd.DataFrame(rows).sort_values("diff", kind="stable").reset_index(drop=True)
    if len(d) != 10:                                  # parts/supp_compare.tex の表は 10 行を決め打ちで読む
        raise SystemExit(f"export_supp_compare: 出発点を下回る組が {len(d)} 個（TeX の表は 10 行）")
    write("supp_c_negcells.csv", d, """出所：results/stats_compare/tests.csv の比較 (b)（拡張版の調整版 − 5 つの値を選んだ出発点），実験 E61・E62，K ∈ {0, 5} の 22 セル，列 tgt_means（ターゲットごとに反復 0〜9 を平均した差．小数 2 桁）．
差が負のターゲットとセルの組だけを差の小さい順に並べた（diffs は小数 2 桁にした TeX の文字列．小数 1 桁では −0.04 が 0.0 になるため）．組の総数・負の数・最小は台帳 st-b-tgt-n・st-b-tgtneg-n・st-b-tgtneg-worst と照合した""")


def main() -> None:
    print("export_supp_compare: 書き出す")
    ranges()
    kbest()
    stats()
    testsets()
    collapse()
    entmin()
    negcells()
    print(f"export_supp_compare: 台帳と {N_CHECKED[0]} 個の値を照合した（食い違いなし）")


if __name__ == "__main__":
    main()
