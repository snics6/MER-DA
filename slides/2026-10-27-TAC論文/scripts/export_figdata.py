"""本編の図の値を CSV に書き出す（資料 2026-10-27-TAC論文）．

論文の図の台本（paper/scripts/figures/fig_*.py）の関数と同じ集計で値を作り，figures/ に CSV で置く．
台帳（paper/results/numbers.json）に同じ量の key があるものは照合し，食い違えば止める．
各 CSV の先頭の # 行に出所を書く（pgfplots は # の行を読み飛ばす）．

    cd slides && make data        （= cd 2026-10-27-TAC論文 && ../../.venv/bin/python scripts/export_figdata.py）
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
sys.path.insert(0, str(PAPER / "scripts" / "figures"))
sys.path.insert(0, str(PAPER / "scripts"))
sys.dont_write_bytecode = True

import style  # noqa: E402
from ledger import part1 as P1  # noqa: E402
import fig_twocloud  # noqa: E402
import fig_centering  # noqa: E402
import fig_compare as F  # noqa: E402
import fig_ladder  # noqa: E402
import fig_kcurves  # noqa: E402
import fig_entmin  # noqa: E402
import fig_finetune  # noqa: E402
import fig_capture  # noqa: E402

OUT = DECK / "figures"
LEDGER = json.loads((PAPER / "results" / "numbers.json").read_text())
N_CHECKED = [0]


def check(key: str, v: float, tol: float = 1e-6) -> None:
    """台帳に key があれば値を照合する（無ければ何もしない）．"""
    if key not in LEDGER:
        return
    w = float(LEDGER[key]["value"])
    if abs(w - float(v)) > tol:
        raise SystemExit(f"export_figdata: {key} は図の値 {v} と台帳の値 {w} が食い違う")
    N_CHECKED[0] += 1


def write(name: str, df: pd.DataFrame, src: str) -> None:
    OUT.mkdir(exist_ok=True)
    head = "".join(f"# {line}\n" for line in src.strip().splitlines())
    (OUT / name).write_text(head + df.to_csv(index=False, float_format="%.6g"))
    print(f"  {name}: {len(df)} 行")


# ---- 第 I 部 ---------------------------------------------------------------
KS_MAP = ["1", "5", "10", "25", "50", "all"]
KLAB_MAP = ["1", "5", "10", "25", "50", "全件"]


def gainmap() -> None:
    data = {}
    for k in style.CONFIG_ORDER:
        p = pd.read_csv(style.RESULTS / "fixc_map" / style.CONFIG[k]["kind"] / "summary" / "pairs.csv",
                        dtype={"K": str, "source": str})
        p["gain"] = p["diff_mean"] * 100
        data[k] = p
    g = pd.DataFrame({"x": range(len(KS_MAP)), "K": KLAB_MAP})
    for k in style.CONFIG_ORDER:
        m = data[k].groupby("K")["gain"].mean().reindex(KS_MAP)
        g[k] = m.to_numpy()
        for K, v in m.items():
            check(f"map-gain-{k}-k{K}", v)
    write("gain_mean.csv", g, """出所：results/fixc_map/<種類>/summary/pairs.csv の diff_mean（合併 − ターゲットのみ）の組の平均 ×100．
同じコーパスの組を含む全組，各組 30 回．論文の図 gainmap (a)（paper/scripts/figures/fig_gainmap.py）と同じ集計．台帳 map-gain-<種類>-k<K>""")
    rows = []
    for gi, k in enumerate(["a", "v", "av", "t"]):
        p = data[k]
        for j, K in enumerate(KS_MAP):
            q = p[p["K"] == K]
            n = len(q)
            npos, nneg = int((q["diff_lo"] > 0).sum()), int((q["diff_hi"] < 0).sum())
            check(f"map-gain-pos-{k}-k{K}", npos)
            check(f"map-gain-neg-{k}-k{K}", nneg)
            rows.append(dict(kind=k, x=gi * 7 + j, K=KLAB_MAP[j], n=n, npos=npos, nneg=nneg,
                             pos=npos / n, mid=1 - (npos + nneg) / n, neg=nneg / n))
    write("gain_sign.csv", pd.DataFrame(rows), """出所：同じ pairs.csv の diff_lo・diff_hi（反復 30 回の対の差の 95% t 区間）．
pos は区間が 0 を上回る組，neg は 0 を下回る組の割合．論文の図 gainmap (b) と同じ数え方．台帳 map-gain-pos-*・map-gain-neg-*""")


def losses() -> None:
    loss = fig_twocloud.e13_losses()
    rows = []
    for i, k in enumerate(style.CONFIG_ORDER):
        th, di = loss[k]
        suf = "" if k == "ah" else "-xsame"
        check(f"p1-e13-loss-offset-{k}{suf}", th)
        check(f"p1-e13-loss-dir-{k}{suf}", di)
        rows.append(dict(x=i, kind=k, thr=th, dir=di))
    write("loss_split.csv", pd.DataFrame(rows), """出所：results/rerun_fixc/explore_relation/snapshots/20261018-0213_fixc/e13_pairs.csv（loss_offset・loss_direction ×100）．
A(HuBERT) は同 check_mh/e13.csv．同じコーパスの組を除く（A(HuBERT) は全組）．論文の図 twocloud (b)・表 2 と同じ．台帳 p1-e13-loss-*""")


def thrshare() -> None:
    p = pd.read_csv(P1.E29 / "e29_pairs.csv", dtype={"K": str})
    rows = []
    for x, K in enumerate(["1", "5", "10"]):
        r = dict(x=x, K=K)
        for k in style.CONFIG_ORDER:
            g = p[(p.kind == style.CONFIG[k]["kind"]) & (p.K == K)]
            r[k] = float(g.share_pos.median())
            check(f"p1-e29-critshare-k{K}-{k}", r[k])
        rows.append(r)
    write("thrshare.csv", pd.DataFrame(rows), """出所：results/explore_relation/snapshots/20261011-1524_e29/e29_pairs.csv の share_pos（組の中央値）．
合併がソースのみを上回った分のうち，判断基準のずれ o の変化による分の割合．同じコーパスの組を除く，反復 0〜9．
論文の図 thresholdshare (a)・表 2 と同じ．台帳 p1-e29-critshare-k<K>-<種類>""")
    T = pd.read_csv(P1.E28T9 / "e28t9_targets.csv")
    q = []
    for k in ("a", "ah", "av", "v"):
        v = float(T[T.kind == style.CONFIG[k]["kind"]].q_dir.median())
        check(f"p1-qdir-{k}", v)
        q.append(dict(kind=k, q=v))
    write("qdir.csv", pd.DataFrame(q), """出所：results/explore_relation/snapshots/20261011-0139_e28t9/e28t9_targets.csv の q_dir（ターゲットの中央値）．台帳 p1-qdir-<種類>""")


def centering() -> None:
    rows = []
    for x, k in enumerate(("a", "ah", "v")):
        t = fig_centering.per_pair(k)
        r = dict(x=x, kind=k, **{c: float(t[c].mean()) for c in fig_centering.COLS + ["c_d1_x", "c_pooled_x"]})
        check(f"p1-center-srconly-{k}", r["center"])
        check(f"p1-d1-gain-{k}-k5", r["d1"])
        check(f"p1-merged-gain-{k}-k5", r["pooled"])
        check(f"p1-center-d1-gain-{k}-k5", r["c_d1_x"])
        check(f"p1-center-merged-gain-{k}-k5", r["c_pooled_x"])
        rows.append(r)
    write("centering.csv", pd.DataFrame(rows), """出所：results/mech_<種類>/*.csv（中心化 center と none，src_only・src_only+D1・merged，学び直しの反復 0〜2，C は交差検証）と
results/fixc_map/<種類>/*.csv（src_only，merged の K=5）．ソースのみからの差 ×100 の組の平均．論文の補足の図 S2（fig_centering.py）と同じ．
c_d1・c_pooled は「中心化の分 ＋ 中心化の後の上乗せ（c_d1_x・c_pooled_x）」．台帳 p1-center-*・p1-d1-gain-*・p1-merged-gain-*""")


def lam() -> None:
    e = pd.read_csv(P1.STAGEB / "e15b_pairs.csv", dtype={"K": str})
    rr = []
    for k in ("a", "ah", "v", "av"):
        g = e[e.kind == style.CONFIG[k]["kind"]]
        rho = P1._spearman(g.lam_obs, g.lam_pred_rep09)
        check(f"p1-e15-lam-rho-{k}", rho)
        rr.append(dict(kind=k, rho=rho, n=len(g)))
        write(f"lambda_{k}.csv", pd.DataFrame(dict(pred=g.lam_pred_rep09.to_numpy(), obs=g.lam_obs.to_numpy())),
              f"""出所：results/fixc_analysis/explore_relation/snapshots/20261009-0105_fixc/stageB/e15b_pairs.csv（{k}）．
pred は式の λ = ρ/(1+ρ)（lam_pred_rep09），obs は測った λ（lam_obs）．組 × K=1・5・10，同じコーパスの組を含む，反復 0〜9．論文の補足の図 S1 と同じ""")
    write("lambda_rho.csv", pd.DataFrame(rr), "出所：上の lambda_<種類>.csv の順位相関（P1._spearman）．台帳 p1-e15-lam-rho-<種類>")


# ---- 第 II 部 --------------------------------------------------------------
def ladder() -> None:
    lv = fig_ladder.levels()
    d = pd.DataFrame({"K": fig_ladder.KS, **{k: v.to_numpy() for k, v in lv.items()}})
    d.insert(0, "x", range(len(d)))
    write("ladder_levels.csv", d, """出所：results/explore_relation/snapshots/20261017-1654_e31final/e31_finals30.csv（B1 ターゲットのみ，B4 合併．ソースで平均）と
同 20261018-0213_ext30/ext30_targets.csv（span=all の P4 仮説転移，E39 の 5 つの値の出発点）．A（WavLM），7 ターゲット，反復 0〜29．論文の図 ladder (a) と同じ""")
    rows = {"K": fig_ladder.KS}
    ns = []
    for key, v, n in fig_ladder.gains():
        rows[key] = v.to_numpy()
        ns.append(f"{key}={n}")
    d = pd.DataFrame(rows)
    d.insert(0, "x", range(len(d)))
    write("ladder_gains.csv", d, f"""出所：e59 は 20261022-0055_e59k/e59k_targets.csv（meta − e39，音声だけの世界），e60av・e60avt は 20261022-0810_e60/e60_targets.csv（meta − e39，音声の生徒），
oursx は results/baselines_fuse-audio-video-text/<T>.csv（拡張版の調整版 − e39．台帳 comparison.py と同じ）．反復 0〜9．論文の図 ladder (b) と同じ．
ターゲットの数：{' '.join(ns)}""")
    for K in ("0", "5"):
        check(f"p2-e60-av-a-k{K}", float(d.loc[d.K == K, "e60av"].iloc[0]))
        check(f"p2-e60-avt-a-k{K}", float(d.loc[d.K == K, "e60avt"].iloc[0]))
        check(f"p2-e59k-a-k{K}", float(d.loc[d.K == K, "e59"].iloc[0]))


def capture() -> None:
    d = pd.read_csv(fig_capture.SRC, comment="#")
    R = d.groupby("kind")[[f"r_{s}" for s, _ in fig_capture.RULES] + ["r_random_true"]].mean()
    rows = []
    unlab = []
    for i, (s, lab) in enumerate(fig_capture.RULES):
        r = dict(y=len(fig_capture.RULES) - 1 - i, rule=s)
        for kind, tok in fig_capture.KINDS:
            g = d[d.kind == kind]
            per = g[f"r_{s}"] / g["r_random_true"]
            r[tok] = float(R.loc[kind, f"r_{s}"] / R.loc[kind, "r_random_true"])
            r[f"{tok}_lo"], r[f"{tok}_hi"] = float(per.min()), float(per.max())
            if s != "LLL":
                unlab.append(r[tok])
            else:
                check(f"p2-e58-ratio-lll-{tok}", r[tok])
        rows.append(r)
    check("p2-e58-ratio-unlab-lo", min(unlab))
    check("p2-e58-ratio-unlab-hi", max(unlab))
    write("capture.csv", pd.DataFrame(rows), """出所：results/explore_relation/snapshots/20261021-2245_e58/e58_targets.csv．
値は「ターゲットの平均の r(選び方)」÷「ターゲットの平均の r(random_true)」．_lo・_hi はターゲットごとの比の最小・最大．
r は出発点の向きの誤差 E0 のうち K=5 のラベル 20 本が張る空間に入る割合．A 7，V 6，T 4 ターゲット，反復 0〜9．論文の補足の図 S5 と同じ．
rule の LLL は候補のラベルを全部使う教える選び方（上の参考）．台帳 p2-e58-ratio-*""")


# ---- 比較 -------------------------------------------------------------------
SHORT = {"audio_wavlm": "a", "video": "v", "text_roberta": "t", "audio_hubert": "a", "video_clip": "v",
         "text_deberta": "t"}


def short(s: str) -> str:
    return SHORT.get(s, "f")


def compare() -> None:
    for K in ("0", "5"):
        rows = []
        for i, (wk, s, lab, _) in enumerate(F.GROUPS):
            b = F.best_of(wk, s, K)
            bl = F.best_label(wk, s, K)
            r = dict(pos=i + (0.5 if i >= 4 else 0) + (0.5 if i >= 7 else 0), world=wk, student=lab,
                     best=bl, tick=f"{lab}（{'なし' if bl == 'none' else bl}）")
            for mk, col in (("b", f"{b}:tuned"), ("o", "ours"), ("e", "oursx:tuned")):
                v = F.diffs(wk, s, K, col)
                r[mk], r[f"{mk}_lo"], r[f"{mk}_hi"] = float(v.mean()), float(v.min()), float(v.max())
            st = short(s)
            check(f"cmp-best-vs-e39-{wk}-{st}-k{K}", r["b"])
            check(f"cmp-ours-vs-e39-{wk}-{st}-k{K}", r["o"])
            check(f"cmp-oursx-vs-e39-{wk}-{st}-k{K}", r["e"])
            rows.append(r)
        write(f"compare_k{K}.csv", pd.DataFrame(rows), f"""出所：results/baselines_<世界>/<T>.csv と <T>_pseudo.csv，results/multimodal_distill_<世界>/<T>.csv と <T>_pseudo.csv（台帳 comparison.py の _world）．
K={K}．pos は横の位置．b は最も強い相手の調整版，o は基本版，e は拡張版の，5 つの値の出発点との差（ポイント）．値はターゲットの平均，_lo・_hi はターゲットの間の最小・最大（95% 区間ではない）．
反復 0〜9，仮のターゲットは反復 0〜4．論文の補足の図 S6（fig_compare.py）と同じ．台帳 cmp-best-vs-e39-*・cmp-ours-vs-e39-*・cmp-oursx-vs-e39-*""")


def kcurves() -> None:
    rows = []
    for x, K in enumerate(fig_kcurves.KS):
        r = dict(x=x, K=K)
        for s, _ in fig_kcurves.STUDENTS:
            st = short(s)
            b = F.best_of("avt", s, K)
            r[f"b_{st}"] = float(F.diffs("avt", s, K, f"{b}:tuned").mean())
            r[f"lab_{st}"] = F.best_label("avt", s, K)
            r[f"o_{st}"] = float(F.diffs("avt", s, K, "ours").mean())
            r[f"x_{st}"] = float(F.diffs("avt", s, K, "oursx:tuned").mean())
            check(f"cmp-oursx-vs-e39-avt-{st}-k{K}", r[f"x_{st}"])
        rows.append(r)
    write("kcurves.csv", pd.DataFrame(rows), """出所：results/baselines_fuse-audio-video-text/ と results/multimodal_distill_fuse-audio-video-text/（台帳 comparison.py の _world）．
AVT の世界の 4 つの生徒（a 音声，v 顔，t 言葉，f 融合），4 ターゲット，反復 0〜9．b は K ごとに選び直した最も強い相手（lab がその略称），o は基本版，x は拡張版．
5 つの値の出発点との差（ポイント）．論文の補足の図 S7 と同じ""")


def entmin() -> None:
    rows = []
    for i, (wk, s, lab) in enumerate(fig_entmin.WORLDS):
        d = F.world_table(wk)
        g = d[(d.student == s) & (d.K == "0")].set_index("target")
        ent = g["entmin:fixed"] - g["e39"]
        ox = g["oursx:tuned"] - g["e39"]
        check(f"cmp-entmin-fixed-vs-e39-{wk}-f-k0", ent.mean())
        check(f"cmp-oursx-vs-e39-{wk}-f-k0", ox.mean())
        for t in g.index:
            rows.append(dict(x=i, world=lab, target=t, ent=float(ent[t]), ox=float(ox[t])))
    m = pd.DataFrame(rows).groupby("x")[["ent", "ox"]].mean().reset_index()
    write("entmin_mean.csv", m, "entmin.csv の世界ごとのターゲットの平均（台帳 cmp-entmin-fixed-vs-e39-<世界>-f-k0，cmp-oursx-vs-e39-<世界>-f-k0 と照合済み）")
    write("entmin.csv", pd.DataFrame(rows), """出所：results/baselines_<世界>/<T>.csv（融合の生徒，K=0）．ent はエントロピー最小化の論文どおりの設定（entmin|-|n），ox は拡張版の調整版．
5 つの値の出発点との差（ポイント），反復 0〜9 の平均．論文の補足の図 S8 と同じ．台帳 cmp-entmin-fixed-vs-e39-*""")


def finetune() -> None:
    ft, fc, fr = fig_finetune.tables()
    tm, LAM = fig_finetune.tmean, fig_finetune.LAM
    rows = [
        ("0", "凍結", "出発点", tm(fr, "frozen_e39", "0"), "cmp-frozen-uar-e39-k0"),
        ("0", "凍結", "弱いラベル", tm(fr, "frozen_ours", "0"), "cmp-frozen-uar-ours-k0"),
        ("0", "微調整", "出発点", tm(ft, "start", "0"), "cmp-ft-uar-start-k0"),
        ("0", "微調整", "弱いラベル", tm(ft, "ours", "0", LAM), "cmp-ft-uar-ours-k0"),
        ("0", "微調整＋中心化", "出発点", tm(fc, "start", "0"), "cmp-ftc-uar-start-k0"),
        ("0", "微調整＋中心化", "弱いラベル", tm(fc, "ours", "0", LAM), "cmp-ftc-uar-ours-k0"),
        ("5", "凍結", "K 本＋罰則", tm(fr, "frozen_e39", "5"), "cmp-frozen-uar-e39-k5"),
        ("5", "凍結", "弱いラベル", tm(fr, "frozen_ours", "5"), "cmp-frozen-uar-ours-k5"),
        ("5", "微調整", "K 本だけ", tm(ft, "k_only", "5"), "cmp-ft-uar-konly-k5"),
        ("5", "微調整", "K 本＋罰則", tm(ft, "anchor", "5", LAM), "cmp-ft-uar-anchor-k5"),
        ("5", "微調整", "弱いラベル", tm(ft, "ours", "5", LAM), "cmp-ft-uar-ours-k5"),
        ("5", "微調整＋中心化", "K 本だけ", tm(fc, "k_only", "5"), "cmp-ftc-uar-konly-k5"),
        ("5", "微調整＋中心化", "K 本＋罰則", tm(fc, "anchor", "5", LAM), "cmp-ftc-uar-anchor-k5"),
        ("5", "微調整＋中心化", "弱いラベル", tm(fc, "ours", "5", LAM), "cmp-ftc-uar-ours-k5"),
    ]
    for *_, v, key in rows:
        check(key, v)
    df = pd.DataFrame([dict(K=K, group=g, bar=b, uar=v) for K, g, b, v, _ in rows])
    for K in ("0", "5"):
        w = df[df.K == K].pivot(index="group", columns="bar", values="uar").reindex(["凍結", "微調整", "微調整＋中心化"])
        w = w.rename(columns={"出発点": "start", "弱いラベル": "weak", "K 本だけ": "konly", "K 本＋罰則": "anchor"})
        w.insert(0, "x", range(len(w)))
        write(f"finetune_k{K}.csv", w.reset_index().fillna("nan"), f"finetune.csv の K={K} を横に並べたもの（行は凍結・微調整・微調整＋中心化）")
    write("finetune.csv", df, """出所：results/finetune_check/<T>.csv（中心化なし），<T>_c.csv（中心化あり），teacher_<T>.csv（凍結）．
T は IEMOCAP 即興・台本，MELD，MOSEI，反復 0〜2，罰則 λ=0.01．音声の生徒の UAR（%）の 4 ターゲットの平均．論文の補足の図 S9 と同じ．台帳 cmp-frozen-uar-*・cmp-ft-uar-*・cmp-ftc-uar-*""")


def analytic() -> None:
    """式の値（実験の値ではない）：2 つの感情の正解率 a(d, o)，K 本で減る 2 つの誤差，ソースの価値の模式図．"""
    from scipy.stats import norm
    o = np.round(np.arange(-3, 3.0001, 0.05), 4)
    acc = pd.DataFrame({"o": o, **{f"d{d}": 100 * 0.5 * (norm.cdf(d / 2 + o) + norm.cdf(d / 2 - o)) for d in (1, 2, 3)}})
    write("acc_do.csv", acc, "式の値：a(d,o) = [Φ(d/2+o) + Φ(d/2−o)]/2 ×100（論文の式 (1)）．実験の値ではない")
    K = np.geomspace(1, 50, 80)
    q = pd.read_csv(OUT / "qdir.csv", comment="#").set_index("kind").q
    cost = pd.DataFrame({"K": K, "thr": 1 / (2 * K), **{k: q[k] / K for k in ("a", "ah", "av", "v")}})
    write("cost_curves.csv", cost, "式の値：判断基準のずれの誤差 1/(2K) と向きの誤差 q/K（q は qdir.csv のターゲットの中央値）．論文の図 thresholdshare (b) と同じ")
    # ソースの価値の模式図：A の q で d(K) = Δ/sqrt(1+q/K) の正解率の曲線（Δ=2.6 は説明用の値）
    qa, D = float(q["a"]), 2.6
    Ks = np.geomspace(1, 100, 120)
    curve = pd.DataFrame({"K": Ks, "acc": 100 * norm.cdf(D / np.sqrt(1 + qa / Ks) / 2)})
    pts = pd.DataFrame({"K": [5.0, 15.0], "acc": [float(100 * norm.cdf(D / np.sqrt(1 + qa / k) / 2)) for k in (5.0, 15.0)]})
    write("value_points.csv", pts, "模式図の値：上の曲線の K=5 と K=5+n（n=10 は説明用の値）の点")
    write("value_curve.csv", curve, f"模式図の値：2 つの感情の正解率 Φ(d(K)/2)，d(K) = Δ/sqrt(1+q/K)，q = {qa:.2f}（A），Δ = {D}（説明用の値）．実験の値ではない")


def from_ledger() -> None:
    """台帳の値をそのまま図にするもの（再計算しない）．"""
    rows = [dict(x=i, kind=k, bound=LEDGER[f"p1-dirloss-recov-bound-k10-{k}"]["value"],
                 ht=LEDGER[f"p1-dirloss-recov-ht-k10-{k}"]["value"]) for i, k in enumerate(("a", "ah", "av", "v"))]
    write("dirrecov.csv", pd.DataFrame(rows), """出所：台帳 p1-dirloss-recov-bound-k10-<種類>（式 d(K) のもとでソースの向きと K=10 本を最適に組んだときに取り戻せる向きの損失の割合，%）と
p1-dirloss-recov-ht-k10-<種類>（仮説転移で実際に取り戻した割合，%）．組の中央値，同じコーパスの組を除く""")
    c = pd.read_csv(OUT / "centering.csv", comment="#").set_index("kind")
    t = pd.DataFrame({"x": range(5), "method": fig_centering.COLS,
                      **{k: [float(c.loc[k, m]) for m in fig_centering.COLS] for k in ("a", "ah", "v")},
                      **{f"{k}_center": [float(c.loc[k, "center"])] * 5 for k in ("a", "ah", "v")}})
    write("centering_t.csv", t, "centering.csv を図の向き（行が方法，列が種類）に並べ替えたもの．*_center は中心化だけの高さ")


def main() -> None:
    print("export_figdata: 書き出す")
    gainmap()
    losses()
    thrshare()
    centering()
    lam()
    ladder()
    capture()
    compare()
    kcurves()
    entmin()
    finetune()
    analytic()
    from_ledger()
    print(f"export_figdata: 台帳と {N_CHECKED[0]} 個の値を照合した（食い違いなし）")


if __name__ == "__main__":
    main()
