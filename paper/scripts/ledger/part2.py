"""台帳のモジュール（part2）．entries() は key → common.entry(...) を返す．

論文 5 節（Part II：仕組みから手法へ）の数値．`docs/paper-claim.md` 4 節の「最小の証拠 2」
（E58，E49〜E54，E59 の追加）と，手法の積み上げ（E31・E39・E59・E60）と，E60 の下調べの純相補性．

値はすべて MER-DA の結果ファイルからその場で計算する．`docs/` の値は検算の比べる相手にだけ使い，
食い違いは note に書く．読むのは集計済みの表（`results/explore_relation/snapshots/` の
`*_targets.csv` など．1 行 = ターゲット × K で，反復を平均したもの）で，反復の数は元の出力
（`results/<実験>_<種類>/<ターゲット>.csv` の rep 列）で数えて source に書く．
重い計算は無い（全体で数秒）ので `.cache/part2.json` は使わない．

UAR の差の単位はポイント．「ターゲットの平均」は，ターゲットごとに反復を平均した値を
ターゲットについて平均したもの．「組の平均」は（ターゲット, ソース）の組についての平均．
"""
from __future__ import annotations

import re

import pandas as pd

from .common import entry, rel, RESULTS, SNAP

KTOK = {"audio_wavlm": "a", "audio_hubert": "ah", "video": "v", "text_roberta": "t",
        "fuse-audio-video": "av", "fuse-video-text": "vt", "fuse-audio-text": "at", "fuse-audio-video-text": "avt"}
STOK = {"audio_wavlm": "a", "video": "v", "text_roberta": "t"}           # E60 の生徒（それ以外は融合 f）
JA = {"audio_wavlm": "A（WavLM）", "audio_hubert": "A(HuBERT)", "video": "V", "text_roberta": "T",
      "fuse-audio-video": "AV", "fuse-video-text": "VT", "fuse-audio-text": "AT", "fuse-audio-video-text": "AVT"}

F_E58 = SNAP / "20261021-2245_e58" / "e58_targets.csv"
F_E49 = SNAP / "20261021-0440_e49" / "e49_targets.csv"
F_E50 = SNAP / "20261021-1220_e50" / "e50_targets.csv"
F_E52 = SNAP / "20261021-1355_e52" / "e52_targets.csv"
F_E52B = SNAP / "20261021-1355_e52" / "e52_belief.csv"
F_E53 = SNAP / "20261021-1450_e53" / "e53_targets.csv"
F_E54 = SNAP / "20261021-1625_e54" / "e54_targets.csv"
F_E31 = SNAP / "20261017-1654_e31final" / "e31_finals30.csv"
F_X30 = SNAP / "20261018-0213_ext30" / "ext30_targets.csv"
F_X30T = SNAP / "20261018-0213_ext30" / "ext30_tests.csv"
F_E59K = SNAP / "20261022-0055_e59k" / "e59k_targets.csv"
F_E60P = RESULTS / "explore_relation" / "e60_mm_check.csv"           # スナップショットは無い（直下だけ）
F_E60 = SNAP / "20261022-0810_e60" / "e60_targets.csv"

SIDE = re.compile(r"_(meta|pseudo|q|sel)$")


def _read(path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", **kw)


def _reps(dirs: list[str]) -> str:
    """元の出力の rep 列から反復の数を数える．どのターゲットでも 0〜n−1 がそろうことを確かめる．"""
    ns = []
    for d in dirs:
        for p in sorted((RESULTS / d).glob("*.csv")):
            if SIDE.search(p.stem):
                continue
            r = set(_read(p, usecols=["rep"]).rep.astype(int).unique())
            if r != set(range(len(r))):
                raise ValueError(f"part2: {p} の反復が 0 から連続しない: {sorted(r)}")
            ns.append(len(r))
    if not ns:
        raise FileNotFoundError(f"part2: 元の出力が無い: {dirs}")
    if min(ns) != max(ns):
        raise ValueError(f"part2: {dirs} の反復の数がターゲットでそろわない（{min(ns)}〜{max(ns)}）")
    n = ns[0]
    pre = sorted({next((d[: -len(k) - 1] for k in KTOK if d.endswith("_" + k)), d) for d in dirs})
    return f"反復 0〜{n - 1}（{n} 回．元の出力 {'・'.join(f'results/{p}_<種類>/' for p in pre)} の {len(ns)} ファイルの rep 列で確認）"


def _by_kind(d: pd.DataFrame, col: str) -> pd.Series:
    return d.groupby("kind")[col].mean()


# ---------------------------------------------------------------- E58：E_0 の幾何

def _e58(out: dict) -> None:
    D = _read(F_E58)
    reps = _reps([f"geom_e0_{k}" for k in ("audio_wavlm", "video", "text_roberta")])
    src = f"{rel(F_E58)}"
    kinds = ["audio_wavlm", "video", "text_roberta"]
    unlab = ["random_pool", "margin", "herding_pred", "coreset_raw", "typiclust", "kcenter_white", "norm_white"]
    R = D.groupby("kind")[[c for c in D.columns if c.startswith(("r_", "d_eff", "dim_"))]].mean()   # ターゲットの平均
    ratio = pd.DataFrame({s: R[f"r_{s}"] / R["r_random_true"] for s in unlab + ["LLL"]})

    base = f"{reps}，K=5 の 20 本 | ターゲットの平均の r(S) どうしの比 r(S)／r(random_true)（geom_e0.py --analyze と同じ）"
    note9 = ("文書（E58 の『わかったこと』と results.md 3.6）は『ラベル無しの選び方 9 通り』と書くが，r(S) を測った 9 つの S のうち "
             "random_true と LLL は候補のラベルを使う．ラベル無しは 7 通りで，この 7 通りの比が文書の 0.5〜1.2 倍に当たる")
    x = ratio[unlab]
    lo_s, hi_s = x.stack().idxmin(), x.stack().idxmax()
    out["p2-e58-ratio-unlab-lo"] = entry(x.values.min(), f"{src} | r_* 列．ラベル無しの 7 通り（{', '.join(unlab)}），A・V・T，最小は {JA[lo_s[0]]} の {lo_s[1]}，{base}", "%.1f", note9)
    out["p2-e58-ratio-unlab-hi"] = entry(x.values.max(), f"{src} | 同じ 7 通り，A・V・T，最大は {JA[hi_s[0]]} の {hi_s[1]}，{base}", "%.1f", note9)
    for k in kinds:
        t = KTOK[k]
        out[f"p2-e58-ratio-unlab-{t}-lo"] = entry(x.loc[k].min(), f"{src} | {JA[k]}，ラベル無しの 7 通りの最小（{x.loc[k].idxmin()}），{base}", "%.2f")
        out[f"p2-e58-ratio-unlab-{t}-hi"] = entry(x.loc[k].max(), f"{src} | {JA[k]}，ラベル無しの 7 通りの最大（{x.loc[k].idxmax()}），{base}", "%.2f")
        out[f"p2-e58-ratio-lll-{t}"] = entry(ratio.loc[k, "LLL"], f"{src} | {JA[k]}，S = LLL（条件 1 の教える選び方），{base}", "%.2f")
        out[f"p2-e58-r-random-{t}"] = entry(100 * R.loc[k, "r_random_true"], f"{src} | {JA[k]}，r_random_true 列（百分率） {reps} | ターゲットの平均", "%.1f")
        out[f"p2-e58-r-ua-{t}"] = entry(100 * R.loc[k, "r_UA"], f"{src} | {JA[k]}，r_UA 列（ソース間の向きの散らばりの空間．百分率） {reps} | ターゲットの平均", "%.1f")
        out[f"p2-e58-r-ub-{t}"] = entry(100 * R.loc[k, "r_UB"], f"{src} | {JA[k]}，r_UB 列（仮のターゲットの直し向きの空間．百分率） {reps} | ターゲットの平均", "%.1f")
        dmin, dmax = D[D.kind == k].d_eff.min(), D[D.kind == k].d_eff.max()
        out[f"p2-e58-deff-z-{t}"] = entry(
            R.loc[k, "d_eff"], f"{src} | {JA[k]}，d_eff 列（生徒の座標 z＝標準化 × E39 の計量 M_ρM^(-1/2) の候補の共分散の参加率 (Σλ)²/Σλ²） {reps} | ターゲットの平均（ターゲットごとは {dmin:.1f}〜{dmax:.1f}）", "%.0f",
            "文書（E58 の命題 2）の『候補の特徴の実効次元は標準化した座標で A 6〜9，V 11，T 65』は E58 の計算の前に見た値で，結果ファイルに無い（標準化しただけの座標の参加率はどのファイルにも保存されていない）．"
            "ここに置くのは E58 の結果の表『d，z の実効次元』（A 40，V 36，T 180）に当たる z の座標の値．標準化した座標の値を出すには特徴を読み直す計算が要る")
    rl = ratio["LLL"]
    out["p2-e58-ratio-lll-lo"] = entry(rl.min(), f"{src} | S = LLL，A・V・T の最小（{JA[rl.idxmin()]}），{base}", "%.1f")
    out["p2-e58-ratio-lll-hi"] = entry(rl.max(), f"{src} | S = LLL，A・V・T の最大（{JA[rl.idxmax()]}），{base}", "%.1f")
    ub = 100 * R[["r_UA", "r_UB"]]
    out["p2-e58-r-uaub-lo"] = entry(ub.values.min(), f"{src} | r_UA・r_UB 列，A・V・T の 6 つの最小（{JA[ub.stack().idxmin()[0]]} の {ub.stack().idxmin()[1]}．百分率） {reps} | ターゲットの平均", "%.1f")
    out["p2-e58-r-uaub-hi"] = entry(ub.values.max(), f"{src} | r_UA・r_UB 列，A・V・T の 6 つの最大（{JA[ub.stack().idxmax()[0]]} の {ub.stack().idxmax()[1]}．百分率） {reps} | ターゲットの平均", "%.1f")

    # 条件 1 の教える選び方の得の分け方（E58 の記述 1．値は E49 から）
    e49 = _read(F_E49, dtype={"K": str})
    k5 = e49[e49.K == "5"]
    r49 = _reps([f"teach_student39_{k}" for k in ("audio_wavlm", "audio_hubert", "video", "fuse-audio-video")])
    for k in ("audio_wavlm", "audio_hubert", "video", "fuse-audio-video"):
        g = k5[k5.kind == k]
        t = KTOK[k]
        s = f"{rel(F_E49)} | {JA[k]}，K=5（20 本），生徒は E39 の学習器"
        out[f"p2-e58-lll-gain-{t}"] = entry((g.LLL - g.random_pool).mean(), f"{s}，LLL − random_pool {r49} | ターゲットの平均", "%.1f")
        out[f"p2-e58-lll-gain-sel-{t}"] = entry((g["LLL@r"] - g.random_pool).mean(),
                                                f"{s}，LLL@r − random_pool（random の組の小さい C′ のまま選び方だけを替えた分） {r49} | ターゲットの平均", "%.1f")
        out[f"p2-e58-lll-gain-cp-{t}"] = entry((g.LLL - g["LLL@r"]).mean(),
                                               f"{s}，LLL − LLL@r（teach の組の C′ に緩めた分） {r49} | ターゲットの平均", "%.1f")


# ---------------------------------------------------------------- E49〜E54：条件 2 の選び方

NOT_SELECT = {"random_pool", "random_pool@r", "random_true", "random_true@r", "LLL", "LLL@r"}


def _label_free(cols) -> list[str]:
    return [c for c in cols if c not in NOT_SELECT and not c.startswith(("P4:", "chk_", "bel_"))
            and c not in ("kind", "target", "K")]


def _cond2(out: dict) -> None:
    files = {"E49": _read(F_E49, dtype={"K": str}), "E50": _read(F_E50, dtype={"K": str}),
             "E52": _read(F_E52, dtype={"K": str}), "E53": _read(F_E53, dtype={"K": str}).assign(kind="text_roberta")}
    paths = {"E49": F_E49, "E50": F_E50, "E52": F_E52, "E53": F_E53}
    reps = _reps([f"{p}_{k}" for p in ("teach_student39", "al_student39") for k in ("audio_wavlm", "audio_hubert", "video", "fuse-audio-video")]
                 + ["teach_stack_audio_wavlm", "teach_stack_audio_hubert", "text_cond2_text_roberta"])
    rows = []
    for name, d in files.items():
        x = d[d.K == "5"]
        for c in _label_free(x.columns):
            ref = "random_pool@r" if c.endswith("@r") and "random_pool@r" in x else "random_pool"
            for k, v in (x[c] - x[ref]).groupby(x.kind).mean().dropna().items():
                rows.append((name, k, c, float(v)))
    R = pd.DataFrame(rows, columns=["exp", "kind", "sel", "gain"])
    allsrc = "，".join(rel(p) for p in paths.values())
    defn = ("ラベル無しの選び方＝各表の版のうち random_pool・random_true・LLL（と @r の同じもの）・P4: の列を除く全部"
            "（herding_pred 2 通り，見込みで選ぶ教える選び方，margin・badge・coreset・typiclust など）．@r の版は random_pool@r と比べる")
    top = R.loc[R.gain.idxmax()]
    out["p2-cond2-e39-maxgain-k5"] = entry(top.gain, f"{allsrc} | K=5，生徒は E39 の学習器，A・A(HuBERT)・V・AV・T．{defn} {reps}．"
                                                     f"最大は {top.exp} の {JA[top.kind]} の {top.sel} | 版ごとに『版 − random_pool』のターゲットの平均をとった最大", "%.2f")
    for k in ("audio_wavlm", "audio_hubert", "video", "fuse-audio-video", "text_roberta"):
        g = R[R.kind == k]
        b = g.loc[g.gain.idxmax()]
        out[f"p2-cond2-e39-maxgain-k5-{KTOK[k]}"] = entry(
            b.gain, f"{allsrc} | {JA[k]}，K=5，生徒は E39 の学習器．{defn}．{len(g)} 版 {reps}．最大は {b.exp} の {b.sel} | 『版 − random_pool』のターゲットの平均の最大", "%.2f")

    # 生徒を P4 から E39 に替えるとラベル無しのランダムが上がる幅（E49，T は E53）
    gains = {}
    for name, d in (("E49", files["E49"]), ("E53", files["E53"])):
        x = d[d.K == "5"]
        gains.update({k: (name, float(v)) for k, v in (x.random_pool - x["P4:random_pool"]).groupby(x.kind).mean().items()})
    main4 = ("audio_wavlm", "audio_hubert", "video", "fuse-audio-video")
    rs = {"E49": f"E39 の生徒は {_reps([f'teach_student39_{k}' for k in main4])}，P4 の生徒（E46 の値を写したもの）は {_reps([f'select_baselines_{k}' for k in main4])}",
          "E53": f"両方の生徒とも {_reps(['text_cond2_text_roberta'])}"}
    sg = "K=5，random_pool（E39 の生徒）− P4:random_pool（P4 の生徒．ラベル無しのランダム）"
    for k, (name, v) in gains.items():
        out[f"p2-cond2-student-random-k5-{KTOK[k]}"] = entry(v, f"{rel(paths[name])} | {JA[k]}，{sg}．{rs[name]} | ターゲットの平均", "%.1f")
    vals = pd.Series({k: v for k, (_, v) in gains.items()})
    both = f"E49 の 4 種類：{rs['E49']}．E53 の T：{rs['E53']}"
    out["p2-cond2-student-random-k5-lo"] = entry(vals.min(), f"{rel(F_E49)}，{rel(F_E53)} | A・A(HuBERT)・V・AV・T の最小（{JA[vals.idxmin()]}），{sg}．{both} | ターゲットの平均", "%.1f")
    out["p2-cond2-student-random-k5-hi"] = entry(vals.max(), f"{rel(F_E49)}，{rel(F_E53)} | 同じ 5 種類の最大（{JA[vals.idxmax()]}），{sg}．{both} | ターゲットの平均", "%.1f")

    # E54：P4 の生徒で，K=1 に出発点＋見込み（pq）の版がランダムを上回る幅
    e54 = _read(F_E54, dtype={"K": str})
    r54 = _reps([f"p4_cond2_{k}" for k in ("audio_wavlm", "video", "text_roberta")])
    x = e54[e54.K == "1"]
    pq = pd.DataFrame({c: (x[c] - x.random_pool).groupby(x.kind).mean() for c in ("pq:cycle", "pq:count")})
    best = pq.max(axis=1)
    for k, v in best.items():
        out[f"p2-e54-pq-k1-{KTOK[k]}"] = entry(v, f"{rel(F_E54)} | {JA[k]}，K=1（4 本），P4 の生徒，pq:cycle と pq:count のうち高い方（{pq.loc[k].idxmax()}）− random_pool {r54} | ターゲットの平均", "%.1f")
    out["p2-e54-pq-k1-lo"] = entry(best.min(), f"{rel(F_E54)} | A（WavLM）・V・T の最小（{JA[best.idxmin()]}），K=1，P4 の生徒，pq の 2 版の高い方 − random_pool {r54} | ターゲットの平均", "%.1f")
    out["p2-e54-pq-k1-hi"] = entry(best.max(), f"{rel(F_E54)} | 同じ 3 種類の最大（{JA[best.idxmax()]}），K=1，P4 の生徒，pq の 2 版の高い方 − random_pool {r54} | ターゲットの平均", "%.1f")


# ---------------------------------------------------------------- E31・E39：出発点の積み上げ

def _e31_e39(out: dict) -> None:
    F = _read(F_E31, dtype={"K": str})
    F = F[~F.same]
    n = int(F[["n_P4_def", "n_H3", "n_P3_def"]].min().min())
    nx = int(F[["n_P4_def", "n_H3", "n_P3_def"]].max().max())
    if n != nx:
        raise ValueError(f"part2: e31_finals30.csv の反復の数がそろわない（{n}〜{nx}）")
    S = F.groupby(["kind", "K"])[["B1", "B4", "H3", "O1_meta", "P4_def"]].mean()
    npair = F.groupby("kind").size() // F.K.nunique()
    rep31 = f"反復 0〜{n - 1}（{n} 回．n_P4_def・n_H3・n_P3_def 列で確認．B1・B4 は C を固定した地図の 30 回）"
    for k in ("audio_wavlm", "audio_hubert", "video", "fuse-audio-video"):
        out[f"p2-e31-p4-uar-k5-{KTOK[k]}"] = entry(
            S.loc[(k, "5"), "P4_def"], f"{rel(F_E31)} | {JA[k]}，K=5，P4_def 列（仮説転移 4 の既定），同じコーパスの組を除く {npair[k]} 組 {rep31} | 組の平均（P4 はターゲットの値を組に写したもの）", "%.1f",
            "ターゲットの平均にすると値が変わる（A（WavLM）の K=5 で 53.8．ext30_targets.csv の P4）．文書（E31 の段 3 の表と results.md 3.6）の 53.6 は組の平均" if k == "audio_wavlm" else None)
    m = {}
    for k in ("audio_wavlm", "audio_hubert"):
        for K in ("1", "5", "10"):
            r = S.loc[(k, K)]
            best = r[["B1", "B4", "H3", "O1_meta"]]
            m[(k, K)] = (float(r.P4_def - best.max()), best.idxmax())
            out[f"p2-e31-p4-vs-best-{KTOK[k]}-k{K}"] = entry(
                m[(k, K)][0], f"{rel(F_E31)} | {JA[k]}，K={K}，P4_def − max(B1, B4, H3, O1_meta)（提案以外の最終候補の最良．ここでは {best.idxmax()}），{npair[k]} 組 {rep31} | 組の平均どうしの差", "%.1f")
    vs = pd.Series({kk: v for kk, (v, _) in m.items()})
    s = f"{rel(F_E31)} | 音声 2 種（A（WavLM）・A(HuBERT)）× K=1・5・10 の 6 つ，P4_def − 提案以外の最終候補（B1・B4・H3・O1_meta）の最良 {rep31} | 組の平均どうしの差"
    out["p2-e31-p4-vs-best-audio-lo"] = entry(vs.min(), f"{s}．最小は {JA[vs.idxmin()[0]]} の K={vs.idxmin()[1]}", "%.1f")
    out["p2-e31-p4-vs-best-audio-hi"] = entry(vs.max(), f"{s}．最大は {JA[vs.idxmax()[0]]} の K={vs.idxmax()[1]}", "%.1f")

    X = _read(F_X30, dtype={"K": str})
    X = X[(X.span == "all") & (X.K == "5")]
    T = _read(F_X30T, dtype={"K": str})
    T = T[(T.span == "all") & (T.K == "5")].set_index("kind")
    kinds = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
    d1 = pd.Series({k: float((X[X.kind == k].E39 - X[X.kind == k].D1).mean()) for k in kinds})
    p4 = pd.Series({k: float((X[X.kind == k].E39 - X[X.kind == k].P4).mean()) for k in kinds})
    doc_d1 = {"audio_wavlm": 1.00, "audio_hubert": 1.62, "video": 1.64, "fuse-audio-video": 1.21}
    doc_p4 = {"audio_wavlm": 3.0, "audio_hubert": 3.3, "video": 3.8, "fuse-audio-video": 4.0}
    for k in kinds:
        reps = f"反復 0〜29（30 回．ext30_tests.csv の min_reps = {int(T.loc[k, 'min_reps'])}）"
        nt = int(T.loc[k, "targets"])
        out[f"p2-e39-vs-d1-k5-{KTOK[k]}"] = entry(
            d1[k], f"{rel(F_X30)} | {JA[k]}，span = all，K=5，E39 − D1，{nt} ターゲット {reps} | ターゲットの平均", "%.1f",
            f"文書（E39 の節，results.md 3.6・3.9，paper-claim.md 4 節）の +{doc_d1[k]:.2f} は反復 0〜9 の値．30 回では {d1[k]:+.2f}（『P4 を出発点にした実験を反復 30 回に延ばす』の節の値と一致）")
        out[f"p2-e39-vs-p4-k5-{KTOK[k]}"] = entry(
            p4[k], f"{rel(F_X30)} | {JA[k]}，span = all，K=5，E39 − P4（E31 の段 3 の HT_ALL 既定），{nt} ターゲット {reps} | ターゲットの平均", "%.1f",
            f"文書（E39 の節の記述 1）の +{doc_p4[k]:.1f} は反復 0〜9 の値．30 回では {p4[k]:+.2f}")
    s30 = f"{rel(F_X30)} | 主な 4 種類（A（WavLM）・A(HuBERT)・V・AV），span = all，K=5，反復 0〜29（30 回） | ターゲットの平均"
    out["p2-e39-vs-d1-k5-lo"] = entry(d1.min(), f"{s30}，E39 − D1 の最小（{JA[d1.idxmin()]}）", "%.1f",
                                      "文書（results.md 3.9，paper-claim.md 4 節の付随する証拠）の『D1 を K=5 で +1.0〜+1.6』は反復 0〜9 の範囲（+1.00〜+1.64）．30 回では +0.74〜+1.72")
    out["p2-e39-vs-d1-k5-hi"] = entry(d1.max(), f"{s30}，E39 − D1 の最大（{JA[d1.idxmax()]}）", "%.1f",
                                      "文書の +1.6 は反復 0〜9 の値．30 回では最大 +1.72（V）")
    out["p2-e39-vs-p4-k5-lo"] = entry(p4.min(), f"{s30}，E39 − P4 の最小（{JA[p4.idxmin()]}）", "%.1f",
                                      "文書（results.md 3.9，paper-claim.md 4 節）の『仮説転移 4 を +3〜+4』は反復 0〜9 の範囲（+3.0〜+4.0）．30 回では +2.57（A（WavLM））〜+3.83 で，A は +3 に届かない")
    out["p2-e39-vs-p4-k5-hi"] = entry(p4.max(), f"{s30}，E39 − P4 の最大（{JA[p4.idxmax()]}）", "%.1f",
                                      "文書の +4 は反復 0〜9 の AV の +4.03．30 回では最大 +3.83（V）")


# ---------------------------------------------------------------- E59 の追加：K ごとに総重みを選ぶ見込みの蒸留

def _e59k(out: dict) -> None:
    d = _read(F_E59K, dtype={"K": str})
    reps = _reps([f"distill_beliefk_{k}" for k in ("audio_wavlm", "video", "text_roberta")])
    d = d.assign(gain=d.meta - d.e39, ctrl=d.p_ctrl - d.e39, info=d.meta - d.p_ctrl)
    G = d.groupby(["kind", "K"])[["gain", "ctrl", "info"]].mean()
    nt = d.groupby("kind").target.nunique()
    src = rel(F_E59K)
    for k in ("audio_wavlm", "video", "text_roberta"):
        t = KTOK[k]
        for K in ("0", "1", "5", "10", "25"):
            out[f"p2-e59k-{t}-k{K}"] = entry(G.loc[(k, K), "gain"], f"{src} | {JA[k]}，K={K}，meta − e39（K ごとに仮のターゲットで選んだ蒸留 − E39），{nt[k]} ターゲット {reps} | ターゲットの平均", "%.1f")
        if k == "text_roberta":
            continue
        for K in ("0", "1", "5"):
            out[f"p2-e59k-ctrl-{t}-k{K}"] = entry(G.loc[(k, K), "ctrl"], f"{src} | {JA[k]}，K={K}，p_ctrl − e39（出発点自身の予測を弱いラベルにした対照＝釣り合い直し），{nt[k]} ターゲット {reps} | ターゲットの平均", "%.1f")
            out[f"p2-e59k-info-{t}-k{K}"] = entry(G.loc[(k, K), "info"], f"{src} | {JA[k]}，K={K}，meta − p_ctrl（見込みの情報の上乗せ），{nt[k]} ターゲット {reps} | ターゲットの平均", "%.1f")
        inf = G.loc[k].loc[["0", "1", "5"], "info"]
        out[f"p2-e59k-info-{t}-lo"] = entry(inf.min(), f"{src} | {JA[k]}，K=0・1・5 の meta − p_ctrl の最小（K={inf.idxmin()}） {reps} | ターゲットの平均", "%.1f")
        out[f"p2-e59k-info-{t}-hi"] = entry(inf.max(), f"{src} | {JA[k]}，K=0・1・5 の meta − p_ctrl の最大（K={inf.idxmax()}） {reps} | ターゲットの平均", "%.1f")

    b = _read(F_E52B)
    r52 = _reps(["teach_stack_audio_wavlm", "teach_stack_audio_hubert"])
    jnote = ("この差は候補の上の UAR の差である．paper-claim.md 1・3 節はこれを『同じモダリティの J ≈ +0.6』と呼ぶが，J（純相補性）は正解率の差に等しく，"
             "同じモダリティの正解率は結果ファイルに無い（distill_belief*_*/ の acc_p・acc_q の行も中身は UAR）．J と書くなら UAR の差で代えたと断るか，正解率を測り直す")
    for k in ("audio_wavlm", "audio_hubert"):
        g = b[b.kind == k]
        out[f"p2-e52-q-minus-p-{KTOK[k]}"] = entry(
            (g.q - g.p).mean(), f"{rel(F_E52B)} | {JA[k]}，q − p（見込み q（E44 の C）と E39 の出発点 p の，候補の上の UAR の差．直す前），{len(g)} ターゲット {r52} | ターゲットの平均", "%.1f",
            jnote if k == "audio_wavlm" else None)


# ---------------------------------------------------------------- E60 の下調べ：純相補性

def _e60pre(out: dict) -> None:
    d = _read(F_E60P)
    nrep = d.groupby(["world", "target"]).rep.nunique()
    if nrep.min() != nrep.max():
        raise ValueError("part2: e60_mm_check.csv の反復の数がそろわない")
    reps = f"反復 0〜{int(d.rep.max())}（{int(nrep.min())} 回．下調べ．rep 列で確認）"
    src = rel(F_E60P)
    # 恒等式 net = acc(見込み) − acc(出発点) を，正解率が保存されている組で確かめる
    acc = {"fused": "acc_fused", "video": "acc_video", "text_roberta": "acc_text_roberta", "A": "acc_audio_wavlm"}
    err = 0.0
    for q in ("fused", "video", "text_roberta"):
        c = f"net_{q}|A"
        m = d[c].notna()
        err = max(err, float((d.loc[m, c] - (d.loc[m, acc[q]] - d.loc[m, acc["A"]])).abs().max()))
    ident = (f"純相補性 net = P(出発点が外れ ∧ 見込みが正解) − P(出発点が当たり ∧ 見込みが外れ) は，正解率が保存されている見込み"
             f"（融合・映像・言葉の出発点）では全 {int(d['net_fused|A'].notna().sum())} 行で acc(見込み) − acc(出発点) と一致した（差の最大 {err:.1e}）．"
             "票の積・E44 型・MLP の見込みは正解率が保存されていないが，恒等式なので同じく正解率の差である．"
             "このファイルの J_* 列は別の量（平均 ⟨q − p, e_y − p⟩）で，paper-claim.md の J ではない")
    names = {"fused": "融合の出発点", "poe": "票の積", "poolC": "E44 型のまとめ方", "mlp": "小さな MLP のまとめ方", "video": "映像の出発点", "text_roberta": "言葉の出発点"}
    btok = {"fused": "fused", "poe": "poe", "poolC": "pool", "mlp": "mlp", "video": "v", "text_roberta": "t"}
    M = d.groupby("world").mean(numeric_only=True)
    nt = d.groupby("world").target.nunique()
    for w in M.index:
        wt = KTOK[w]
        for q in ("fused", "poe", "poolC", "mlp", "video", "text_roberta"):
            c = f"net_{q}|A"
            if c in M and pd.notna(M.loc[w, c]):
                out[f"p2-e60pre-net-{wt}-a-{btok[q]}"] = entry(
                    100 * M.loc[w, c], f"{src} | {JA[w]} の世界，{c} 列（音声の出発点に対する {names[q]} の純相補性．正解率のポイント），{nt[w]} ターゲット {reps} | ターゲット × 反復の平均",
                    "%.1f", ident if (w == "fuse-audio-video" and q == "fused") else None)
        for q in ("poe", "poolC", "mlp"):
            c = f"net_{q}|fused"
            out[f"p2-e60pre-net-{wt}-f-{btok[q]}"] = entry(
                100 * M.loc[w, c], f"{src} | {JA[w]} の世界，{c} 列（融合の出発点に対する {names[q]} の純相補性．正解率のポイント），{nt[w]} ターゲット {reps} | ターゲット × 反復の平均", "%.1f")
    # 範囲
    a = 100 * M.loc["fuse-audio-video", ["net_fused|A", "net_poe|A", "net_poolC|A"]]
    out["p2-e60pre-net-av-a-lo"] = entry(a.min(), f"{src} | AV の世界，音声の出発点に対する別のモダリティの見込み 3 つ（融合の出発点・票の積・E44 型）の最小（{a.idxmin()}） {reps} | ターゲット × 反復の平均", "%.1f",
                                         "paper-claim.md 1・3 節の『別のモダリティでは J +7〜9』の下端は E44 型の +6.66（四捨五入で 6.7）")
    out["p2-e60pre-net-av-a-hi"] = entry(a.max(), f"{src} | AV の世界，同じ 3 つの最大（{a.idxmax()}） {reps} | ターゲット × 反復の平均", "%.1f")
    f = 100 * M[["net_poe|fused", "net_poolC|fused", "net_mlp|fused"]]
    out["p2-e60pre-net-f-lo"] = entry(f.values.min(), f"{src} | AV・AVT の世界，融合の出発点に対する見込み 3 つ（票の積・E44 型・MLP）の 6 つの最小（{JA[f.stack().idxmin()[0]]} の {f.stack().idxmin()[1]}） {reps} | ターゲット × 反復の平均", "%.1f")
    out["p2-e60pre-net-f-hi"] = entry(f.values.max(), f"{src} | AV・AVT の世界，同じ 6 つの最大（{JA[f.stack().idxmax()[0]]} の {f.stack().idxmax()[1]}） {reps} | ターゲット × 反復の平均", "%.1f",
                                      "文書（E60 の下調べ）は『どの見込みも融合の出発点を上回らない』と書くが，AVT の票の積は +0.81 で上回る（文書の括弧の中の値と一致）．範囲を『−0.8〜−3.5』と書けるのは AV だけか，AVT の票の積を除いたとき")


# ---------------------------------------------------------------- E60：モダリティをまたぐ蒸留

def _e60(out: dict) -> None:
    d = _read(F_E60, dtype={"K": str})
    worlds = ["fuse-audio-video", "fuse-video-text", "fuse-audio-text", "fuse-audio-video-text"]
    reps = _reps([f"multimodal_distill_{w}" for w in worlds])
    d = d.assign(gain=d.meta - d.e39, ctrl=d.p_ctrl - d.e39)
    G = d.groupby(["world", "student", "K"]).agg(gain=("gain", "mean"), ctrl=("ctrl", "mean"), n=("target", "nunique"))
    src = rel(F_E60)
    for (w, s, K), r in G.iterrows():
        st = STOK.get(s, "f")
        kind = "融合" if st == "f" else JA[s]
        out[f"p2-e60-{KTOK[w]}-{st}-k{K}"] = entry(
            r.gain, f"{src} | {JA[w]} の世界の {kind} の生徒，K={K}，meta − e39（K ごとに仮のターゲットで選んだ蒸留 − E39），{int(r.n)} ターゲット {reps} | ターゲットの平均", "%.1f")
    mod = G.reset_index()
    mod = mod.assign(fused=mod.student.str.startswith("fuse"))
    out["p2-e60-av-f-k0"]["note"] = (f"値は {out['p2-e60-av-f-k0']['value']:.3f} で，小数 1 桁では 1.1．文書（results.md 3.6・3.9，paper-claim.md 4 節）の『融合 AV +1.2』"
                                     "は E60 の表の +1.15 をさらに丸めた値")
    f0 = mod[mod.fused & (mod.K == "0")].set_index("world").gain
    fsrc = f"{src} | 融合の生徒 4 世界（AV・VT・AT・AVT），K=0，meta − e39 の{{}}（{{}}） {reps} | ターゲットの平均"
    out["p2-e60-f-k0-lo"] = entry(f0.min(), fsrc.format("最小", JA[f0.idxmin()]), "%.1f",
                                  "文書（results.md 3.9）の『融合は 4 世界中 3 つで +1.2〜+3.0』の下端は AV の 1.146（小数 1 桁では 1.1）")
    out["p2-e60-f-k0-hi"] = entry(f0.max(), fsrc.format("最大", JA[f0.idxmax()]), "%.1f")
    m1 = mod[~mod.fused]
    n_mod = m1[["world", "student"]].drop_duplicates().shape[0]
    for K in ("0", "5"):
        x = m1[m1.K == K]
        out[f"p2-e60-m1-pos-k{K}-n"] = entry(int((x.gain > 0).sum()), f"{src} | K={K}，モダリティの生徒 {n_mod} 組（世界, モダリティ）のうち meta − e39 が正の組の数 {reps} | ターゲットの平均の符号", "%d")
    out["p2-e60-m1-n"] = entry(n_mod, f"{src} | モダリティの生徒の（世界, モダリティ）の組（AV 2，VT 2，AT 2，AVT 3） | 数え上げ", "%d")
    fz = mod[mod.fused]
    ok = fz.pivot_table(index="world", columns="K", values="gain")
    m2 = int(((ok["0"] >= 0.5) & (ok["5"] >= 0.5)).sum())
    out["p2-e60-m2-n"] = entry(m2, f"{src} | 融合の生徒 4 世界のうち K=0 と K=5 の両方で meta − e39 ≥ 0.5 の世界の数（予言 M2 の基準） {reps} | ターゲットの平均", "%d")
    hi = mod[mod.K.isin(["10", "25"])]
    worst = hi.loc[hi.gain.idxmin()]
    loss = hi[hi.gain < -0.1].groupby(["world", "student"]).size()
    lossall = mod[mod.gain < -0.1]
    n_st = mod[["world", "student"]].drop_duplicates().shape[0]
    out["p2-e60-m3-loss-n"] = entry(len(loss), f"{src} | {n_st} の生徒のうち，K=10・25 のどこかで meta − e39 < −0.1 の生徒の数（K ≤ 5 で −0.1 未満は {len(lossall) - int(loss.sum())} 件） {reps} | ターゲットの平均", "%d")
    out["p2-e60-m3-students-n"] = entry(n_st, f"{src} | E60 の生徒（モダリティ 9 ＋ 融合 4） | 数え上げ", "%d")
    wst = STOK.get(worst.student, "f")
    out["p2-e60-m3-worst"] = entry(worst.gain, f"{src} | K=10・25 の meta − e39 の最小（{JA[worst.world]} の世界の {'融合' if wst == 'f' else JA[worst.student]} の生徒，K={worst.K}） {reps} | ターゲットの平均", "%.1f")
    c = mod[mod.K.isin(["0", "5"])]
    out["p2-e60-ctrl-lo"] = entry(c.ctrl.min(), f"{src} | 13 の生徒 × K=0・5 の p_ctrl − e39（釣り合い直しだけの対照）の最小 {reps} | ターゲットの平均", "%.1f",
                                  "文書（E60 の M1 の段）の『対照 p の得は −0.4〜+1.4』は K=0 と K=5 を合わせた範囲（K=0 だけなら −0.08〜+1.40）")
    out["p2-e60-ctrl-hi"] = entry(c.ctrl.max(), f"{src} | 13 の生徒 × K=0・5 の p_ctrl − e39 の最大 {reps} | ターゲットの平均", "%.1f")


def entries() -> dict[str, dict]:
    out: dict[str, dict] = {}
    _e58(out)
    _cond2(out)
    _e31_e39(out)
    _e59k(out)
    _e60pre(out)
    _e60(out)
    return out
