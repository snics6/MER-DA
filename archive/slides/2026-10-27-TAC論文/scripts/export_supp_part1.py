"""補足 0（設定の詳細）と補足 1（第 I 部の式と導出）の図と表の値を CSV に書き出す（資料 2026-10-27-TAC論文）．

台帳（paper/results/numbers.json）に無い値だけをここで作る．集計は論文の台帳の関数（paper/scripts/ledger/*.py）か
論文の図の台本（paper/scripts/figures/fig_*.py）と同じにし，台帳に同じ量の key があれば照合して，食い違えば止める
（見本は scripts/export_figdata.py の check）．各 CSV の先頭の # 行に出所を書く（pgfplots は # の行を読み飛ばす）．
新しい学習はしない．結果ファイルを読むだけで，全体で 1 分ほど（特徴の件数を数える子プロセスが大半）．

    cd 2026-10-27-TAC論文 && ../../.venv/bin/python scripts/export_supp_part1.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import subprocess
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

import style  # noqa: E402
from ledger import part1 as P1  # noqa: E402
from ledger import data_map as DM  # noqa: E402
import fig_cvmap  # noqa: E402
import fig_sourcevalue  # noqa: E402

OUT = DECK / "figures"
LEDGER = json.loads((PAPER / "results" / "numbers.json").read_text())
N_CHECKED = [0]


def check(key: str, v: float, tol: float = 1e-6) -> None:
    """台帳の key の値と照合する（key が無ければ止める．照合のために呼ぶので，無いのは書き間違い）．"""
    if key not in LEDGER:
        raise SystemExit(f"export_supp_part1: 台帳に key {key} が無い")
    w = float(LEDGER[key]["value"])
    if abs(w - float(v)) > tol * max(1.0, abs(w)):
        raise SystemExit(f"export_supp_part1: {key} は図の値 {v} と台帳の値 {w} が食い違う")
    N_CHECKED[0] += 1


def write(name: str, df: pd.DataFrame, src: str) -> None:
    OUT.mkdir(exist_ok=True)
    head = "".join(f"# {line}\n" for line in src.strip().splitlines())
    (OUT / name).write_text(head + df.to_csv(index=False, float_format="%.6g"))
    print(f"  {name}: {len(df)} 行")


# ---- 補足 0：融合で残る発話の数（AT・VT は台帳に無い） ---------------------------------
COUNT_CODE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import data as D
out = {}
for g in D.GROUPS_WITH_TEXT:
    ids = {k: set(D.load_group(g, k)[0]["sample_id"]) for k in ("audio_wavlm", "video", "text_roberta")}
    a, v, t = ids["audio_wavlm"], ids["video"], ids["text_roberta"]
    out[g] = {"av": len(a & v), "at": len(a & t), "vt": len(v & t), "avt": len(a & v & t)}
print(json.dumps(out))
"""
GKEY = {"IEMOCAP-impro": "impro", "IEMOCAP-script": "script", "MOSEI": "mosei", "MELD": "meld"}


def counts() -> None:
    py = MERDA / ".venv" / "bin" / "python"
    r = subprocess.run([str(py), "-I", "-c", COUNT_CODE, str(MERDA / "measure")], capture_output=True, text=True,
                       cwd=str(MERDA), check=False)
    if r.returncode != 0:
        raise SystemExit(f"export_supp_part1: 件数を数える子プロセスが失敗した\n{r.stderr}")
    n = json.loads(r.stdout.strip().splitlines()[-1])
    rows = []
    for g, gk in GKEY.items():
        check(f"data-n-{gk}-av", n[g]["av"])
        check(f"data-n-{gk}-avt", n[g]["avt"])
        rows.append(dict(group=g, key=gk, av=n[g]["av"], at=n[g]["at"], vt=n[g]["vt"], avt=n[g]["avt"]))
    write("supp_set_counts.csv", pd.DataFrame(rows), """出所：measure/data.py の load_group(群, 'audio_wavlm'／'video'／'text_roberta') の sample_id の共通部分の数
（load_fused と同じ突き合わせ）．言葉のある 4 群．av と avt は台帳 data-n-<群>-av・-avt と照合した．at と vt は台帳に無い""")


# ---- 補足 0：C の決め方（交差検証の地図 results/map_<種類>/） ------------------------------
CKINDS = ("a", "ah", "v", "av")
KS_MAP = ["1", "5", "10", "25", "50", "all"]
KLAB = ["1", "5", "10", "25", "50", "全件"]


def _read_cvmap(kind: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(RESULTS / f"map_{kind}" / "*.csv")))
    return pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str},
                                  usecols=["target", "source", "method", "K", "rep", "C", "C_rule"]) for f in fs],
                     ignore_index=True)


def crule() -> None:
    rows = [dict(x=i, K=KLAB[i]) for i in range(len(KS_MAP))]
    src_rows = []
    for k in CKINDS:
        kind = style.CONFIG[k]["kind"]
        d = _read_cvmap(kind)
        m, t, s = (d[d.method == x] for x in ("merged", "tgt_only", "src_only"))
        ref = pd.read_csv(RESULTS / f"map_{kind}" / "summary" / "c_rule.csv", dtype={"K": str}).set_index(["method", "K"])
        for i, K in enumerate(KS_MAP):
            mk, tk = m[m.K == K], t[t.K == K]
            tie_m = float((mk.C_rule == "tie").mean())
            if abs(tie_m - float(ref.loc[("merged", K), "tie"])) > 1e-9:
                raise SystemExit(f"export_supp_part1: {kind} の合併の同点の割合が summary/c_rule.csv と食い違う（K={K}）")
            rows[i][f"{k}_mtie"] = tie_m
            rows[i][f"{k}_mc1"] = float((mk.C == 1.0).mean())
            rows[i][f"{k}_ttie"] = float((tk.C_rule == "tie").mean())
            rows[i][f"{k}_tfix"] = float((tk.C_rule == "fixed").mean())
        cs = s.C.value_counts(normalize=True)
        src_rows.append(dict(kind=k, n=len(s), **{f"c{c:g}".replace(".", "p"): float(cs.get(c, 0.0))
                                                  for c in (0.01, 0.1, 1.0, 10.0)}))
    write("supp_set_crule.csv", pd.DataFrame(rows), """出所：results/map_<種類>/*.csv（交差検証の地図：合併の C も K 本の交差検証で選んだ版．反復 0〜29）の C_rule と C．
<種類>_mtie：合併の行のうち同点の規則（tie．1.0 に最も近い候補）で C が決まった割合．<種類>_mc1：合併で C=1 になった割合．
<種類>_ttie・<種類>_tfix：ターゲットのみの行のうち同点の規則・欠けた感情の規則（fixed．C=1）で決まった割合．
合併の同点の割合は results/map_<種類>/summary/c_rule.csv と照合した．台帳に同じ量は無い""")
    write("supp_set_csrc.csv", pd.DataFrame(src_rows), """出所：results/map_<種類>/*.csv の src_only の行の C（ソースの話者で 3 分割の交差検証．反復 0〜29）．
c0p01 などは C=0.01，0.1，1，10 が選ばれた割合．C を固定した地図では，合併はこの C を使う．台帳に同じ量は無い""")


def cvmap() -> None:
    g = pd.DataFrame({"x": range(len(KS_MAP)), "K": KLAB})
    reps = {}
    for k in style.CONFIG_ORDER:
        kind = style.CONFIG[k]["kind"]
        cv = fig_cvmap.mean_gain("map", k)
        fx = fig_cvmap.mean_gain("fixc_map", k)
        for K, v in zip(KS_MAP, fx):
            check(f"map-gain-{k}-k{K}", v)
        g[f"{k}_cv"] = cv
        g[f"{k}_fix"] = fx
        p = pd.read_csv(RESULTS / f"map_{kind}" / "summary" / "pairs.csv", dtype={"K": str, "source": str})
        reps[k] = (int(p.n_rep.min()), int(p.n_rep.max()))
    rtxt = "，".join(f"{k} {a}" if a == b else f"{k} {a}〜{b}" for k, (a, b) in reps.items())
    write("supp_set_cvmap.csv", g, f"""出所：results/map_<種類>/summary/pairs.csv（交差検証の地図．<種類>_cv）と results/fixc_map/<種類>/summary/pairs.csv（C を固定した地図．<種類>_fix）の
diff_mean（合併 − ターゲットのみ）の組の平均 ×100．論文の補足の図 cvmap（paper/scripts/figures/fig_cvmap.py の mean_gain）と同じ集計．
<種類>_fix は台帳 map-gain-<種類>-k<K> と照合した．交差検証の地図の組ごとの反復の数：{rtxt}（C を固定した地図はどれも 30 回）""")


# ---- 補足 1：q のあてはめ（E28 の 9） ---------------------------------------------------
QKINDS = ("a", "ah", "av", "v")


def qdir() -> None:
    T = pd.read_csv(P1.E28T9 / "e28t9_targets.csv")
    rows = []
    for i, k in enumerate(QKINDS):
        tt = T[T.kind == style.CONFIG[k]["kind"]].sort_values("q_dir")
        check(f"p1-qdir-{k}", float(tt.q_dir.median()))
        n = len(tt)
        for j, (_, r) in enumerate(tt.iterrows()):
            rows.append(dict(kind=k, x=i, xj=i + 0.32 * (j - (n - 1) / 2) / max(n - 1, 1), target=r.target,
                             q=float(r.q_dir), D=float(r.own_d),
                             dT1=float(r.dT_K1), dT5=float(r.dT_K5), dT10=float(r.dT_K10)))
    # あてはめの例：A で q がターゲットの中央値に当たるターゲット（7 ターゲットなので中央のもの）
    ta = T[T.kind == "audio_wavlm"].sort_values("q_dir").reset_index(drop=True)
    ex = ta.iloc[len(ta) // 2]
    assert abs(float(ex.q_dir) - float(ta.q_dir.median())) < 1e-12
    write("supp_p1_qfit_example.csv",
          pd.DataFrame([dict(K=K, dT=float(ex[f"dT_K{K}"]), fit=float(ex[f"dT_fit_K{K}"]), D=float(ex.own_d), q=float(ex.q_dir),
                             target=ex.target) for K in (1, 5, 10)]),
          f"""出所：e28t9_targets.csv の A（WavLM）で q_dir がターゲットの中央値に当たる {ex.target} の行．dT は測ったターゲットのみの識別力，fit は D/√(1 + q/K)．
D は own_d（ターゲットの最良の向きの識別力）で固定し，q だけをあてはめた値．台帳 p1-qdir-a と同じ q""")
    write("supp_p1_qdir.csv", pd.DataFrame(rows), """出所：results/explore_relation/snapshots/20261011-0139_e28t9/e28t9_targets.csv．ターゲットごとの q_dir（ターゲットのみの分類器の識別力 d を
K=1・5・10 で測り，d_T(K) = D/√(1 + q/K) を当てた値．D は own_d），D，測った d_T（dT1・dT5・dT10）．xj は図で点を横にずらす位置．
種類ごとの中央値は台帳 p1-qdir-<種類> と照合した""")


# ---- 補足 1：上乗せの判断基準の分と向きの分（E29） -----------------------------------------
def share() -> None:
    p = pd.read_csv(P1.E29 / "e29_pairs.csv", dtype={"K": str})
    rows = []
    for x, k in enumerate(style.CONFIG_ORDER):
        r = dict(x=x, kind=k)
        for K in ("1", "5", "10"):
            g = p[(p.kind == style.CONFIG[k]["kind"]) & (p.K == K)]
            check(f"p1-e29-critshare-k{K}-{k}", float(g.share_pos.median()))
            r[f"pos{K}"], r[f"dir{K}"] = float(g.pos.mean()), float(g.dir.mean())
            r[f"n{K}"] = len(g)
        rows.append(r)
    write("supp_p1_share.csv", pd.DataFrame(rows), """出所：results/explore_relation/snapshots/20261011-1524_e29/e29_pairs.csv の pos（判断基準の分 a(d_s,o_m) − a(d_s,o_s)）と
dir（向きの分 a(d_m,o_m) − a(d_s,o_m)）の組の平均（2 つの感情の正解率の点．6 対の平均）．同じコーパスの組を除く，反復 0〜9，合併の C はソースのみの C．
台帳の割合 p1-e29-critshare-k<K>-<種類>（組ごとの pos/(pos+dir) の中央値）と同じ行で照合した．平均どうしの比はこの割合と一致しない""")
    long = [dict(x=4 * i + j, kind=r["kind"], K=K, pos=r[f"pos{K}"], dir=r[f"dir{K}"])
            for i, r in enumerate(rows) for j, K in enumerate(("1", "5", "10"))]
    write("supp_p1_share_long.csv", pd.DataFrame(long), """supp_p1_share.csv を積み上げの棒の形（1 行が 種類 × K）に並べ替えたもの．x = 4 ×（種類の番号．A，V，T，AV，VT，AT，AVT，A(HuBERT) の順）+（K=1，5，10 の番号）""")


# ---- 補足 1：加法の模型で説明できる割合（E2） ----------------------------------------------
VLAB = {"gain_K1": "利得 K=1", "gain_K5": "利得 K=5", "g_small": "利得 K=1・5 の平均", "gain_all_Kall": "利得 全件",
        "kstar": "K*"}


def mapshare() -> None:
    rows = []
    for k in DM.SHARE_KINDS:
        kind = DM.KIND[k]
        vals = DM._share_values(kind, DM.SHARE_DECOMP, DM.SHARE_H7)
        check(f"map-share-{k}-lo", 100 * min(vals.values()))
        check(f"map-share-{k}-hi", 100 * max(vals.values()))
        for path, vars_ in ((DM.SHARE_DECOMP, DM.SHARE_VARS_DECOMP), (DM.SHARE_H7, DM.SHARE_VARS_H7)):
            t = pd.read_csv(path)
            for v in vars_:
                r = t[(t.kind == kind) & (t.variable == v)]
                if not len(r):
                    continue
                r = r.iloc[0]
                rows.append(dict(kind=k, file="check_h7" if "check_h7" in str(path) else "decomp_share", variable=v,
                                 label=VLAB[v], n_pairs=int(r.n_pairs), r2_target=float(r.r2_target),
                                 r2_source=float(r.r2_source), r2_both=float(r.r2_both)))
    d = pd.DataFrame(rows).drop_duplicates(subset=["kind", "variable"], keep="first")
    write("supp_p1_mapshare.csv", d, """出所：results/fixc_analysis/explore_relation/snapshots/20261009-0105_fixc/decomp_share.csv と results/rerun_fixc/check_h7/share.csv（C を固定した地図）．
組ごとの値 v を v = μ + a_t + b_s + e（ターゲットの効果 a_t とソースの効果 b_s）で当てた決定係数 r2_both．r2_target・r2_source はターゲットだけ・ソースだけの模型の値．
変数は台帳 data_map.py の SHARE_VARS_*（K* は IEMOCAP がターゲットの組を除く）．種類ごとの最小・最大は台帳 map-share-<種類>-lo・-hi と照合した．
同じ変数が 2 つのファイルにあるとき（A の K*）は decomp_share.csv の値（2 つは同じ）""")
    # 表に組むための横長の版（百分率に丸めた文字列．無い組み合わせは --）
    order = ["gain_K1", "gain_K5", "g_small", "gain_all_Kall", "kstar"]
    wide = []
    for v in order:
        r = dict(variable=v, label=VLAB[v])
        for k in DM.SHARE_KINDS:
            x = d[(d.kind == k) & (d.variable == v)]
            for col, tag in (("r2_both", "b"), ("r2_target", "t"), ("r2_source", "s")):
                r[f"{k}_{tag}"] = f"{100 * float(x[col].iloc[0]):.0f}" if len(x) else "--"
            r[f"{k}_n"] = f"{int(x.n_pairs.iloc[0])}" if len(x) else "--"
        wide.append(r)
    write("supp_p1_mapshare_wide.csv", pd.DataFrame(wide), """supp_p1_mapshare.csv を表の形（行が変数，列が種類）に並べ替え，百分率に丸めた文字列にしたもの．
<種類>_b は両方の効果（r2_both），_t はターゲットだけ，_s はソースだけ，_n は組の数．-- はその種類に値が無い""")


# ---- 補足 1：λ の予言と実測（E15，C を固定した段階 B） ---------------------------------------
LKINDS = ("a", "ah", "v", "av")
SETTING = {"a": "W7", "ah": "Hu7", "v": "V", "av": "AV"}


def lam() -> None:
    e = pd.read_csv(P1.STAGEB / "e15b_pairs.csv", dtype={"K": str})
    t15 = pd.read_csv(P1.STAGEB / "e15b_tests.csv").set_index("setting")
    rows = []
    for k in LKINDS:
        g = e[e.kind == style.CONFIG[k]["kind"]]
        rho = P1._spearman(g.lam_obs, g.lam_pred_rep09)
        check(f"p1-e15-lam-rho-{k}", rho)
        tt = t15.loc[SETTING[k]]
        r = dict(kind=k, n=len(g), rho=rho, obs=float(g.lam_obs.mean()), pred=float(g.lam_pred_rep09.mean()))
        if abs(r["obs"] - tt.mean_lam_obs) > 1e-9 or abs(r["pred"] - tt.mean_lam_pred) > 1e-9:
            raise SystemExit(f"export_supp_part1: {k} の λ の平均が e15b_tests.csv と食い違う")
        for K in ("1", "5", "10"):
            h = g[g.K == K]
            r[f"obs{K}"], r[f"pred{K}"] = float(h.lam_obs.mean()), float(h.lam_pred_rep09.mean())
            r[f"rho{K}"] = P1._spearman(h.lam_obs, h.lam_pred_rep09)
            if abs(r[f"obs{K}"] - tt[f"mean_lam_obs_K{K}"]) > 1e-9 or abs(r[f"rho{K}"] - tt[f"spearman_K{K}"]) > 1e-9:
                raise SystemExit(f"export_supp_part1: {k} の K={K} の λ が e15b_tests.csv と食い違う")
        rows.append(r)
    write("supp_p1_lam.csv", pd.DataFrame(rows), """出所：results/fixc_analysis/explore_relation/snapshots/20261009-0105_fixc/stageB/e15b_pairs.csv．obs は測った λ（lam_obs），pred は式の λ（lam_pred_rep09）の
組 × K の平均（K=1・5・10 をまとめた値と K ごとの値），rho は順位相関（P1._spearman）．同じコーパスの組を含む，反復 0〜9，合併の C はソースのみの C．
rho は台帳 p1-e15-lam-rho-<種類>，平均と K ごとの順位相関は stageB/e15b_tests.csv と照合した""")


# ---- 補足 1：ソースの価値の予言と実測（E32．論文の補足の図 sourcevalue） ------------------------
VKINDS = ("a", "ah", "v", "av")


def value() -> None:
    P = pd.read_csv(P1.E32 / "e32_pairs.csv", dtype={"K": str})
    A = pd.read_csv(P1.E32 / "e32_allsrc.csv", dtype={"K": str})
    rng = np.random.default_rng(0)
    main, top, bot, cnt = [], [], [], []
    for gi, k in enumerate(VKINDS):
        kind = style.CONFIG[k]["kind"]
        for si, (src, tab, ucol, flag) in enumerate((("one", P, "uar_P1", "flag_meas"), ("all", A, "P4", "flag"))):
            d = fig_sourcevalue.measured(kind, tab[(tab.kind == kind) & tab.K.isin(fig_sourcevalue.KUSE)], ucol)
            assert (d.f2 == d[flag]).all() and np.allclose(d.n2.fillna(-999), d.n_meas.fillna(-999))
            assert (d.f2 != -1).all()
            g = 2 * gi + si + 1
            ok = (d.f2 == 0) & (d.n2 > 0)
            hi = d[d.f2 == 1]
            le0 = d[(d.f2 == 0) & (d.n2 <= 0)]
            main += [dict(g=g, kind=k, src=src, K=K, x=x, y=y) for K, x, y in zip(d[ok].K, d[ok].n_pred, d[ok].n2)]
            top += [dict(g=g, kind=k, src=src, K=K, x=x, y=y) for K, x, y in zip(hi.K, hi.n_pred, rng.uniform(0.25, 0.75, len(hi)))]
            bot += [dict(g=g, kind=k, src=src, K=K, x=x, y=y) for K, x, y in zip(le0.K, le0.n_pred, rng.uniform(0.25, 0.75, len(le0)))]
            cnt.append(dict(g=g, kind=k, src=src, n=len(d), main=int(ok.sum()), top=len(hi), bot=len(le0)))
            if src == "one":
                for K in fig_sourcevalue.KUSE:
                    h = d[d.K == K]
                    rho, ratio = P1._n_test(h.n2.to_numpy(), h.f2.to_numpy(), h.n_pred.to_numpy(), h.flag_pred.to_numpy())
                    check(f"p1-e32-rho-{k}-k{K}", rho)
                    check(f"p1-e32-ratio-{k}-k{K}", ratio)
            else:
                rho, ratio = P1._n_test(d.n2.to_numpy(), d.f2.to_numpy(), d.n_pred.to_numpy(), d.flag_pred.to_numpy())
                check(f"p1-e32-allsrc-rho-{k}", rho)
                check(f"p1-e32-allsrc-ratio-{k}", ratio)
    src = """出所：results/explore_relation/snapshots/20261016-0954_e32/e32_pairs.csv（1 ソースの仮説転移 uar_P1，同じコーパスの組を除く）と e32_allsrc.csv（全ソース P4），
results/fixc_map/<種類>/*.csv の tgt_only（学習曲線）．x は予言した n（n_pred），y は測った n（学習曲線で UAR に当たる本数 − K）．K=5・10．
論文の補足の図 sourcevalue（paper/scripts/figures/fig_sourcevalue.py の measured）と同じ集計．g は 2×(種類の番号)+(1 ソース 1，全ソース 2)，種類は a，ah，v，av の順．
台帳 p1-e32-rho-*・p1-e32-ratio-*・p1-e32-allsrc-rho-*・p1-e32-allsrc-ratio-* を同じ行から計算し直して照合した"""
    write("supp_p1_val_main.csv", pd.DataFrame(main), src + "\n測った n が 0 より大きく学習曲線の範囲の中にある点")
    write("supp_p1_val_top.csv", pd.DataFrame(top), src + "\n学習曲線の範囲を超えた点（K=50 のターゲットのみより良い）．y は帯の中の縦の位置（乱数の種 0．値に意味はない）")
    write("supp_p1_val_bot.csv", pd.DataFrame(bot), src + "\n測った n が 0 以下の点．y は帯の中の縦の位置（乱数の種 0．値に意味はない）")
    write("supp_p1_val_count.csv", pd.DataFrame(cnt), src + "\n点の数（main：主な枠，top：上の帯，bot：下の帯）")


# ---- 補足 1：式の値（実験の値ではない） -----------------------------------------------------
ILL = dict(d_s=1.1, D=1.7, o_s=0.8)   # 損失の分け方の模式図の説明用の値


def analytic() -> None:
    from scipy.stats import norm

    def a(d, o):
        return 100 * 0.5 * (norm.cdf(d / 2 + o) + norm.cdf(d / 2 - o))

    d = np.round(np.arange(0, 3.0001, 0.05), 4)
    write("supp_p1_acc.csv", pd.DataFrame({"d": d, "o0": a(d, 0), "os": a(d, ILL["o_s"])}),
          f"式の値：a(d,o) = [Φ(d/2+o) + Φ(d/2−o)]/2 ×100 を o=0 と o={ILL['o_s']}（説明用の値）で．実験の値ではない")
    pts = pd.DataFrame([dict(name="src", d=ILL["d_s"], acc=a(ILL["d_s"], ILL["o_s"])),
                        dict(name="mid", d=ILL["d_s"], acc=a(ILL["d_s"], 0.0)),
                        dict(name="best", d=ILL["D"], acc=a(ILL["D"], 0.0))])
    write("supp_p1_acc_pts.csv", pts, f"式の値：模式図の 3 点（d_s={ILL['d_s']}，Δ={ILL['D']}，o_s={ILL['o_s']} は説明用の値）．実験の値ではない")
    o = np.round(np.arange(-2, 2.0001, 0.05), 4)
    c = 2 / 4 * norm.pdf(1.0)
    write("supp_p1_threxp.csv", pd.DataFrame({"o": o, "exact": a(2, 0) - a(2, o), "approx": 100 * c * o ** 2}),
          "式の値：d=2 の判断基準による損失 a(2,0) − a(2,o)（exact）と 2 次の近似 (d/4)φ(d/2)o² ×100（approx）．実験の値ではない")
    # 向きの取り戻しの模式図：q は A の中央値（台帳 p1-qdir-a），e は説明用の値
    q = float(LEDGER["p1-qdir-a"]["value"])
    e = 0.6
    n = q * e ** 2 / (1 - e ** 2)
    K = np.geomspace(1, 200, 100)
    write("supp_p1_dirrec_curve.csv", pd.DataFrame({"K": K, "tonly": 1 / np.sqrt(1 + q / K), "comb": 1 / np.sqrt(1 + q / (K + n)),
                                                     "src": e}),
          f"""式の値：ターゲットのみ d_T(K)/Δ = 1/√(1 + q/K)（tonly）と，ソースの向きと K 本を最良に組んだ d*(K)/Δ = 1/√(1 + q/(K + n))（comb），
n = q e²/(1 − e²)．q = {q:.4g}（台帳 p1-qdir-a．A のターゲットの中央値），e = {e}（説明用の値）．実験の値ではない""")


def main() -> None:
    print("export_supp_part1: 書き出す")
    analytic()
    counts()
    crule()
    cvmap()
    qdir()
    share()
    mapshare()
    lam()
    value()
    print(f"export_supp_part1: 台帳と {N_CHECKED[0]} 個の値を照合した（食い違いなし）")


if __name__ == "__main__":
    main()
