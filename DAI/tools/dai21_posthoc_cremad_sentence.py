"""実験 21（DAI-21）の計算の後の表：CREMA-D の文についての感度と内訳（単位の保存を数え直すだけ．学び直さない．GPU は使わない）．
設計 `docs/設計/実験21_設計.md` 4 節末の「計算の後に保存から出す表」を出す．第 2 版（検査 `checks/DAI-21/code_post_tools.md` の指摘を入れた）．

CREMA-D は俳優 91 人が決まった 12 文を読むコーパスで，発話の名前は「話者_文_感情_強さ」（例 `1001_IEO_ANG_HI`）である．
文 `IEO`（It's eleven o'clock.）だけが強さ LO・MD・HI を持ち，4 ラベルの発話では Neutral が 10 本に 1 本である（ほかの 11 文は 4 本に 1 本）．
同じ 12 文が学習側（K 本とラベル無しの発話）とテストにまたがる．

出すもの（置き場は既定で `results/dai21_eval5/posthoc/`．確かめが全部通ってから書く）：
  1. 照合（ファイルは書かない）：テストの全発話・話者の塊で，集計の段（`dai21_eval5.stage_summary` の CREMA-D の部分）の水準・差・感情ごとの再現率を作り直し，
     `summary/levels.csv`・`effects.csv`・`per_emotion.csv` の `part == "cremad"` の行と照らす．点推定のずれが 1e-9，区間の両端と標準誤差のずれが 1e-6 を
     超えるか，照らした行の数が期待（版の数 × K 3 つ × 引き方の組 3 つ，差の対 78 × 3 × 3，版の数 × 3 × 感情 4 つ）と違えば止まる．`same_as`・`same_pred`
     （判定の並びの sha256 が同じ版の印）・`method`・印も照らす．判定そのものの照合として，取り違えの数（本当の感情 × 答えた感情）を集計の段の
     `confusion.csv`（5 版）と照らす．集計の段は判定の並びの sha256 を保存していないので，sha256 どうしは照らせない（この台本の値を由来の記録に残す）．
  2. 文 `IEO` を除いたテスト（話者の塊．集計の段と同じ乱数の鍵）：`cremad_no_ieo_levels.csv`（水準），`cremad_no_ieo_effects.csv`（差），
     `cremad_no_ieo_per_emotion.csv`（感情ごとの再現率），`cremad_no_ieo_removed.csv`（除いた発話の数．単位ごと・感情ごと）．
     `*_all` の列は全発話の値（1 で集計と照らした値）．`d_uar`・`d_est`・`d_recall` は「除いた値 − 全発話の値」．`same_as`・`same_pred` は除いたテストの上の印．
     `d_lo`・`d_hi`・`d_se` は「除いた値 − 全発話の値」の区間である．作り方：除いたテストと全発話では件数が違うので，正解の数を足し引きせず，話者の塊の
     引き直しの 1 回ごとに UAR を別々に作ってから差を取る．水準は `effect_stats`（除いた数，全発話の数）をそのまま呼ぶ．差の差
     [UAR(a, 除) − UAR(b, 除)] − [UAR(a, 全) − UAR(b, 全)] は，(a の除いた数 | b の全発話の数) と (b の除いた数 | a の全発話の数) を列の方向につないだ
     8 列の数を `effect_stats` に渡す．8 列の再現率の平均は 2 つの UAR の平均になるので，戻り値を 2 倍すると差の差になる（引き直しの 1 回ごとに，列ごとに
     割ってから平均する）．乱数の鍵は集計の段のその行の鍵と同じで，話者の並びも同じなので，引き直しの話者の組は全発話の区間・除いた区間と同じである．
  3. 文を塊にした区間（テストの全発話．12 塊）：`cremad_sentence_cluster_effects.csv`（差），`cremad_sentence_cluster_levels.csv`（水準）．
     `dai1_stats.effect_stats` に，文を塊にした数と，文の実効の数 (Σn)² / Σn²（`dai1_stats` の規則．テストの別々の発話で数える．文 `IEO` の発話が多いので
     12 より小さい）を渡す．塊が 20 未満の作りになる：文を 1 つずつ，全部の分割のテストから同時に除く jackknife の分散（(G − 1)/G 倍．G ＝ 12）に，
     K 本の引き方の項を足す．自由度は ⌊実効の数⌋ − 1 で，主の列 `lo`・`hi`・`mark` はこの自由度の区間である．`lo_n12`・`hi_n12` は，自由度を塊の数 12 から
     決めた場合（11）の区間（参考．標準誤差は同じ）．点推定は話者の塊のときと同じになる（確かめる）．`*_speaker` の列は話者の塊の区間（1 の値）．
     この区間は，話者と学習した分類器を固定し，テストの 12 文を同じ文の集まりから取り直したときの揺れと，K 本の引き方の揺れを含む．テストの話者の揺れは
     含まない．学習側にも同じ 12 文があるので，学習に現れない文への区間ではない．
  4. 内訳（記述．区間は付けない）：`cremad_by_sentence.csv`（文ごとの UAR と感情ごとの再現率），`cremad_by_intensity.csv`（強さ × 感情ごとの再現率），
     `cremad_by_sentence_confusion.csv`（文 × 本当の感情 × 答えた感情の数）．版は，提案の手法・積・学び直した融合 2 つ・K 本だけの融合・分類器ごとの温度の較正と，
     声だけ・顔だけの分類器（K 本だけの版 `none_*`，微調整した版 `ret_*`），出発点 `start_*`（CREMA-D のラベルを見ていない．K によらない），上限 `full_*`．
     どれも，全部の単位（分割 5 × K 本の引き方 10 ＝ 50）の正解の数と発話の数を足し合わせてから割った値である（列の名前は `*_pooled`）．
     水準の表の UAR（単位ごとの UAR を引き方で平均してから分割で平均した値）とは作りが違い，文ごとの UAR を 12 文で平均しても水準の値にはならない．
     `n` は 50 単位の合計，`n_per_seed` は別々の発話の数（同じ分割のテストは K 本の引き方によらず同じ）．強さ LO・MD・HI は文 `IEO` の Happy・Sad・Angry
     だけにあり，Neutral は XX だけなので，強さごとの UAR は作らない．
  5. `cremad_kshot_ieo_units.csv`：単位 × K ごとに，K 本の中の文 `IEO` の本数（感情ごと．npz の `sid_pool[kpos_K{K}]` から数える）と，主の差の単位ごとの値
     （全発話と，文 `IEO` を除いたテスト），内訳の版の単位ごとの UAR．テストから文を除く表（2）は学習側（K 本に文 `IEO` が入ること）の影響を測らないので，
     K 本の側の本数と単位ごとの値を並べられるようにする．検査役が数えた本数（`checks/DAI-21/post_tools_run/kshot_ieo_counts.csv`）があれば照らす．
  6. `cremad_assumptions.csv`：仮定の表（K ごとの 50 単位の平均）．単位の csv の `_check` の行には仮定の量の列が無い（`check_row_assumption_columns` に
     「無い」と書く）ので，npz の表（`D_M_ext`・`D_M_home`・`D_M_K_K{K}`・`D_M_orc_K{K}`・`ret_M`）と確率（`p_test_ret_*`）から，`dai16_probe` の関数
     （`corr`・`dcenter`・`perm_share`・`dprime`）を呼んで，`dai16_probe.run_unit` と `dai20_eval.stage_summary` の「仮定の表」と同じ式で作る（`assume_of`）．
     式が同じであることは，同じ関数を実験 20 の単位（AFEW の 50 単位）に当て，その csv の `_check` の行と集計の `assumptions.csv` に一致することで確かめる．
     分類器が 2 つなので，分類器 × 感情の成分（`intcorr`・`intperm` の材料）は声の行と顔の行が符号だけ違う 4 個の数になり，相関は偶然でも大きく振れる．
     作らない量：`levelcorr_M_*`（分類器が 2 つだと 2 点の相関で ±1 にしかならない），`excess_AV`・`sdratio_M`（`run_unit` の中の複数行の式を書き写さないと
     作れない．npz から作れる），`*_away`（CREMA-D の npz に表が無い），`profcorr_noV`（声の行だけになり `profcorr_row_A` と同じ）．
  7. `cremad_sentence_provenance.json`：由来の記録．

版の名前 → npz の鍵の対応（`names_of`）と差の対（`con_of`）は，集計の段の関数の中の変数（`names`・`con`）で import できないので，同じ式を写した．
写しが集計の段と同じであることは，1 の照合（版の集合，差の対の集合，値）で確かめる．それ以外の定数と式（`A2`・`EXTRA_T`・`ABL`・`GRID`・`same_but`・
`table_block`・`seed_rng`・`SEED_SETS`・`g_eff_of`・`load_frozen`・`todo_cremad`・`effect_stats`）は `dai21_eval5` とその部品を import して使う．

使い方：CUDA_VISIBLE_DEVICES="" .venv/bin/python tools/dai21_posthoc_cremad_sentence.py [--units …] [--summary …] [--out …] [--allow-partial]
  既定は本番の置き場（単位 50 個・名前・鍵・K・コミット・集計との一致・式の確かめを行い，合わなければ何も書かずに止まる）．
  --allow-partial は検査用（単位の数と名前，単位のコミット，集計との照合，実験 20 での式の確かめ，検査役の本数との照合を省く）．
  本番の置き場（`results/dai21_eval5/posthoc/`）に書けるのは，本番の単位と集計を既定の確かめつきで読んだときだけである．
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                    # GPU は使わない（部品が torch を読み込む前に決める）

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy import stats as sst  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai15  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai18_probe as P18  # noqa: E402
import dai20_eval as P20  # noqa: E402
import dai21_eval5 as E21  # noqa: E402
import dai_data as DD  # noqa: E402

ROOT = DD.ROOT
CD, KS, GRID, A2 = E21.CD, E21.KS, E21.GRID, E21.A2
PART = "cremad"                                                            # 集計の段が CREMA-D の行に付ける名前（乱数の鍵にも入る）
OUT21 = E21.OUT_PROD
DROP = "IEO"                                                               # 文「It's eleven o'clock.」の符号
SENT = ("DFA", "IEO", "IOM", "ITH", "ITS", "IWL", "IWW", "MTI", "TAI", "TIE", "TSI", "WSI")     # CREMA-D の 12 文の符号
EMO = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3}                              # 発話の名前の感情の符号 → ラベル（データの取り決め 8）
INTENS = ("XX", "LO", "MD", "HI")
N_CON = 78                                                                 # CREMA-D の差の対の数（設計 4 節の「区間の数」）
N_SPK = 91                                                                 # CREMA-D の話者の数
TOL_EST, TOL_CI = 1e-9, 1e-6                                               # 集計の段との照合の許し（点推定，区間の両端と標準誤差）
TOL_F32 = 1e-3                                                             # float32 で保存した確率から d′ を作り直したときの許し（保存の d′ は float64 の確率から作ってある）
LK, EK = ["part", "c", "K", "version", "seeds"], ["part", "a", "b", "c", "K", "seeds"]      # 水準の行と差の行を決める列
PK = ["part", "c", "K", "version", "emotion"]                              # 感情ごとの再現率の行を決める列
ASSUME_PREFIX = ("intcorr_M_", "intperm_M_", "profcorr_M_", "levelcorr_M_", "topok_M_", "emocorr_M_", "ret_", "excess_", "level_M_")   # dai20_eval.stage_summary の「仮定の表」の列の選び方（写し）
ASSUME_EXACT = ("drift_M", "sdratio_M")
REF20 = "AFEW"                                                             # 仮定の量の式を確かめるのに使う実験 20 のコーパス（50 単位）
KSHOT_REF = ROOT / "checks" / "DAI-21" / "post_tools_run" / "kshot_ieo_counts.csv"   # 検査役が数えた K 本の中の文 IEO の本数（あれば照らす．写さない）
NOT_MADE = {"levelcorr_M_*": "分類器が 2 つだと 2 点の相関で ±1 にしかならない",
            "excess_AV": "作っていない（npz から作れる）．dai16_probe.run_unit の中の複数行の式を書き写さないと作れない（p_pool_ret_*・y_pool・kpos_K* から）",
            "sdratio_M": "作っていない（npz から作れる）．同上",
            "*_away": "CREMA-D の npz に away の表が無い",
            "profcorr_noV": "言葉の行が無いので声の行だけになり，profcorr_row_A と同じ"}


def names_of(K: int, q: dict, fk: str) -> dict:
    """版の名前 → 単位の npz の鍵．`dai21_eval5.stage_summary` の CREMA-D の部分の `names` と同じ式（並びも同じ）．q：固定のファイルの cremad[K]．"""
    i, gm, gs = int(q["prop_grid"]), int(q["student_main"]), int(q["student_prop_sel"])
    names = {v: f"pred_adj_K{K}_{v}" for v in A2 + ("M_F6_orc", "M_F6_ext")}
    names.update({f"{v}@raw": f"pred_raw_K{K}_{v}" for v in A2})
    names.update(prop=f"pred_adj_K{K}_M_F6_mix{i}", **{"prop@raw": f"pred_raw_K{K}_M_F6_mix{i}"}, prop_noprof=f"pred_adj_K{K}_M_F6_mix{P20.same_but(i, beta=0.0)}",
                 prop_noprof_sel=f"pred_adj_K{K}_M_F6_mix{int(q['noprof_sel'])}", prop_noint=f"pred_adj_K{K}_M_F6n_mix{i}",
                 prop_n0zero=f"pred_adj_K{K}_M_F6_mix{P20.same_but(i, n0=0.0)}", prop_home=f"pred_adj_K{K}_M_F6h_mix{i}",
                 prop_levelfree=f"pred_adj_K{K}_M_F6_mix{GRID.index((float('inf'), 1.0, 'eq'))}",
                 stud_prod=f"pred_K{K}_stud_prod_g{gm}", stud_prop=f"pred_K{K}_stud_prop_g{gm}", stud_prop_sel=f"pred_K{K}_stud_prop_g{gs}",
                 **{f"stud_{t_}": f"pred_K{K}_stud_{t_}_g{gm}" for t_ in E21.EXTRA_T},
                 **{f"none_{k}": f"pred_raw_K{K}_none_{k}" for k in ("A", "V", fk)}, ret_A=f"pred_raw_K{K}_ret_A", ret_V=f"pred_raw_K{K}_ret_V",
                 **{f"full_{k}": f"pred_raw_K0_full_{k}" for k in ("A", "V", fk)}, **{f"start_{k}": f"pred_raw_K0_start_{k}" for k in ("A", "V", fk)})
    return names


def con_of(fk: str) -> list:
    """差の対 (a, b)．`dai21_eval5.stage_summary` の CREMA-D の部分の `con` と同じ式．"""
    nf = f"none_{fk}"
    st_ex = tuple(f"stud_{t_}" for t_ in E21.EXTRA_T)
    return ([("prop", b) for b in A2 + P20.ABL + (nf, "stud_prod", "M_F6_ext")] + [(a, "prod") for a in A2[1:] + ("prop_noprof", "prop_levelfree", "M_F6_orc", "M_F6_ext")]
            + [("prop_noint", "prop_noprof"), ("prop", "prop@raw")] + [("prop@raw", f"{b}@raw") for b in A2] + [("prop", f"{b}@raw") for b in A2[1:]]
            + [("stud_prop", b) for b in ("stud_prod", nf, "prop", "prod") + st_ex] + [("stud_prop_sel", "stud_prod"), ("stud_prod", nf), ("stud_prod", "prod")]
            + [(s_, "stud_prod") for s_ in st_ex])


def parse_sid(sid: np.ndarray, label: str) -> tuple:
    """発話の名前「話者_文_感情_強さ」を分ける．戻り：(文の符号, 名前の感情から決めたラベル, 強さ)．形か符号が想定と違えば止まる．"""
    parts = [s.split("_") for s in sid]
    if any(len(q) != 4 for q in parts) or len(set(sid)) != len(sid):
        raise SystemExit(f"{label}：発話の名前が「話者_文_感情_強さ」の形でないか，重なりがある")
    sen, emo, inten = (np.array([q[j] for q in parts]) for j in (1, 2, 3))
    if not (set(sen) <= set(SENT) and set(emo) <= set(EMO) and set(inten) <= set(INTENS)):
        raise SystemExit(f"{label}：文・感情・強さの符号が想定と違う")
    return sen, np.array([EMO[e] for e in emo]), inten


def assume_of(z, K: int, mods: str, ext: str, home: str, ret: str) -> tuple[dict, float]:
    """1 単位・1 つの K の仮定の量．`dai16_probe.run_unit` の「仮定の表」（intcorr・intperm・profcorr・emocorr・levelcorr・topok・drift_M・ret・level）と
    `dai20_eval.stage_summary` の「仮定の表」（dk_*・profcorr_row）と同じ式で，計算は `dai16_probe` の関数（corr・dcenter・perm_share・dprime）を呼ぶ．
    表はソースで測った表（ext・home）と K 本で測った表を，候補の全ラベルで測った表（orc．参照）と比べる．levelcorr は分類器が 3 つ以上のときだけ作る．
    戻りの 2 つ目：保存の確率（float32）から作り直した orc の d′ と，保存の orc の d′ の差の最大（確率の読み方と K 本以外の候補の取り方の確かめ）．"""
    D = {"ext": z[ext].astype(float), "home": z[home].astype(float), "K": z[f"D_M_K_K{K}"].astype(float), "orc": z[f"D_M_orc_K{K}"].astype(float)}
    prof = lambda X: X - X.mean(1, keepdims=True)  # noqa: E731
    cap = math.sqrt(2.0) * float(sst.norm.ppf(P16.AUC_CLIP[1])) - 1e-9          # d′ の上限（dai20_eval.stage_summary と同じ式）
    a = {}
    for r in ("home", "ext", "K"):
        a[f"intcorr_M_{r}"] = P16.corr(P16.dcenter(D[r]), P16.dcenter(D["orc"]))
        a[f"intperm_M_{r}"] = P16.perm_share(D[r], D["orc"])
        a[f"profcorr_M_{r}"] = P16.corr(prof(D[r]), prof(D["orc"]))
        a[f"emocorr_M_{r}"] = P16.corr(D[r].mean(0), D["orc"].mean(0))
        if len(mods) > 2:
            a[f"levelcorr_M_{r}"] = P16.corr(D[r].mean(1), D["orc"].mean(1))
        a[f"topok_M_{r}"] = float(D[r].mean(1).argmax() == D["orc"].mean(1).argmax())
    yt, yp = z["y_test"].astype(int), z["y_pool"].astype(int)
    a["drift_M"] = float(np.abs(np.stack([P16.dprime(z[f"p_test_ret_{m}_K{K}"].astype(np.float64), yt)[0] for m in mods]) - D["orc"]).mean())
    a.update({f"ret_{m}": float(z[ret][i]) for i, m in enumerate(mods)})
    a.update({f"level_M_{r}_{m}": float(D[r][i].mean()) for r in ("home", "ext", "K", "orc") for i, m in enumerate(mods)})
    a["dk_ceiling_share"], a["dk_zero_share"] = float((D["K"] >= cap).mean()), float((D["K"] <= 0).mean())
    a.update({f"profcorr_row_{m}": P16.corr(prof(D["ext"])[j], prof(D["orc"])[j]) for j, m in enumerate(mods)})
    nk = np.setdiff1d(np.arange(len(yp)), z[f"kpos_K{K}"])
    orc2 = np.stack([P16.dprime(z[f"p_pool_ret_{m}_K{K}"].astype(np.float64)[nk], yp[nk])[0] for m in mods])
    return a, float(np.abs(orc2 - D["orc"]).max())


def check_assume_formulas(strict: bool):
    """`assume_of` の式が元の台本と同じであることの確かめ：実験 20 の単位（REF20 の 50 単位．分類器 3 つ）に当て，単位の csv の `_check` の行の同じ名前の列と，
    集計の `assumptions.csv` の dk_*・profcorr_row_*（単位の平均）に一致するかを見る．drift_M だけは保存の確率が float32 なので許しを広げる．"""
    ud, sd = E21.OUT20 / "units", E21.OUT20 / "summary" / "assumptions.csv"
    files = sorted(ud.glob(f"{REF20}_ALL_f*_r*.npz"))
    if len(files) != DD.N_FOLDS * 10 or not sd.exists() or not all(p.with_suffix(".csv").exists() for p in files):
        if strict:
            raise SystemExit(f"仮定の量の式を確かめる実験 20 の単位（{REF20} の 50 単位）か assumptions.csv が無い")
        return "省いた（実験 20 の単位が無い）"
    rows, dev_t, dev_d, dev_o, seen = [], 0.0, 0.0, 0.0, set()
    summary_cols = ["dk_ceiling_share", "dk_zero_share"] + [f"profcorr_row_{m}" for m in P16.MODS]
    for p in files:
        z = np.load(p, allow_pickle=False)
        ck = pd.read_csv(p.with_suffix(".csv"), low_memory=False)
        ck = ck[ck.version == "_check"].set_index("K")
        for K in KS:
            a, od = assume_of(z, K, P16.MODS, "SR_M_ext", "SR_M_home", "SR_ret")
            dev_o = max(dev_o, od) if np.isfinite(od) else float("inf")
            for k_, v in a.items():
                if k_ in summary_cols:
                    continue
                if k_ not in ck.columns:
                    raise SystemExit(f"仮定の量 {k_} が実験 20 の単位の csv に無い")
                w = float(ck.loc[K, k_])
                dv = 0.0 if (np.isnan(v) and np.isnan(w)) else (abs(v - w) if (np.isfinite(v) and np.isfinite(w)) else float("inf"))
                if k_ == "drift_M":
                    dev_d = max(dev_d, dv)
                else:
                    dev_t = max(dev_t, dv)
                seen.add(k_)
            rows.append(dict(K=K, **{k_: a[k_] for k_ in summary_cols}))
    m = pd.DataFrame(rows).groupby("K")[summary_cols].mean()
    A = pd.read_csv(sd)
    A = A[A.c == REF20].set_index("K")
    dev_s = float(np.abs(m.loc[list(KS)].to_numpy(float) - A.loc[list(KS), summary_cols].to_numpy(float)).max())
    res = dict(corpus=REF20, n_units=len(files), n_columns_vs_check_rows=len(seen), table_dev_max=dev_t, drift_dev_max=dev_d, summary_columns=summary_cols, summary_dev_max=dev_s,
               orc_recheck_dev_max=dev_o)
    if not (dev_t <= TOL_EST and dev_d <= TOL_F32 and dev_s <= TOL_EST and dev_o <= TOL_F32):
        raise SystemExit(f"仮定の量の式が実験 20 の保存と合わない：{res}")
    return res


def load_units(udir: Path, fz: dict, fk: str, mods: str, strict: bool) -> tuple[list, pd.DataFrame]:
    """単位の csv と npz を読み，テストのラベル・話者の塊・文・強さ，全部の版の判定，K 本の中の文 IEO の本数，仮定の量を取り出す．
    名前・鍵・K・発話の名前の形が合わなければ止まる．"""
    exp = sorted(n for n, _ in E21.todo_cremad(None, None))
    csvs = sorted(udir.glob("*.csv"))
    have = sorted(p.stem for p in csvs)
    if have != sorted(p.stem for p in udir.glob("*.npz")):
        raise SystemExit("csv と npz の名前がそろわない（csv が済みの印．片方だけの単位は書きかけ）")
    if strict and have != exp:
        raise SystemExit(f"単位の数か名前が合わない：{len(have)} 個（{len(exp)} 個のはず）")
    if not csvs:
        raise SystemExit(f"{udir} に単位が無い")
    units, frames, sid_of = [], [], {}
    for p in csvs:
        head, r = p.stem.rsplit("_r", 1)
        head, f = head.rsplit("_f", 1)
        if head != f"{CD}_ALL" or not (f.isdigit() and r.isdigit()):
            raise SystemExit(f"{p.name}：CREMA-D の評価の回の単位の名前ではない")
        f, r = int(f), int(r)
        d = pd.read_csv(p, low_memory=False)
        if set(d.name.astype(str)) != {p.stem}:
            raise SystemExit(f"{p.name}：csv の name の列がファイルの名前と違う")
        if sorted(d[d.version == "_check"].K.astype(int)) != sorted(KS):
            raise SystemExit(f"{p.name}：_check の行の K が {KS} でない")
        frames.append(d)
        z = np.load(p.with_suffix(".npz"), allow_pickle=False)
        keys = set(z.files)
        nm = {K: names_of(K, fz["cremad"][str(K)], fk) for K in KS}
        need = ({kk for K in KS for kk in nm[K].values()} | {"y_test", "cl_test", "sid_test", "y_pool", "cl_pool", "sid_pool", "D_M_ext", "D_M_home", "ret_M"}
                | {f"{k}_K{K}" for K in KS for k in ("D_M_K", "D_M_orc", "kpos")} | {f"p_{w}_ret_{m}_K{K}" for K in KS for w in ("test", "pool") for m in mods})
        miss = sorted(need - keys)
        if miss:
            raise SystemExit(f"{p.name}：npz に鍵が無い：{miss[:5]}（計 {len(miss)}）")
        y, cl, sid = z["y_test"].astype(int), z["cl_test"].astype(str), z["sid_test"].astype(str)
        yp, clp, sidp = z["y_pool"].astype(int), z["cl_pool"].astype(str), z["sid_pool"].astype(str)
        sen, ye, inten = parse_sid(sid, p.name)
        if not (len(y) == len(cl) == len(sid)) or set(sen) != set(SENT):
            raise SystemExit(f"{p.name}：テストの長さがそろわないか，テストに 12 文がそろわない")
        if not (ye == y).all():
            raise SystemExit(f"{p.name}：発話の名前の感情とラベルが食い違う")
        if not ((inten == "XX") | ((sen == DROP) & (y != 0))).all() or ((sen == DROP) & (y != 0) & (inten == "XX")).any():
            raise SystemExit(f"{p.name}：強さ LO・MD・HI が文 {DROP} の Happy・Sad・Angry 以外にあるか，その発話に XX がある")
        if not np.array_equal(sid_of.setdefault(f, sid), sid):
            raise SystemExit(f"{p.name}：同じ分割のテストが K 本の引き方によって違う")
        if set(clp) & set(cl) or set(sidp) & set(sid):
            raise SystemExit(f"{p.name}：候補とテストに同じ話者か同じ発話がある")
        pred, kieo, ass, orc_dev = {}, {}, {}, 0.0
        for K in KS:
            kpos = z[f"kpos_K{K}"]
            sen_k, ye_k, _ = parse_sid(sidp[kpos], f"{p.name} の K = {K} の K 本")
            if len(kpos) != 4 * K or (np.bincount(yp[kpos], minlength=4) != K).any() or not (ye_k == yp[kpos]).all():
                raise SystemExit(f"{p.name}：K = {K} の K 本が感情ごとに {K} 本でないか，名前の感情とラベルが食い違う")
            kieo[K] = np.array([int(((sen_k == DROP) & (yp[kpos] == i)).sum()) for i in range(4)])
            ass[K], od = assume_of(z, K, mods, "D_M_ext", "D_M_home", "ret_M")
            orc_dev = max(orc_dev, od) if np.isfinite(od) else float("inf")
            for v, kk in nm[K].items():
                a = z[kk].astype(int)
                if a.shape != y.shape or a.min() < 0 or a.max() > 3:
                    raise SystemExit(f"{p.name}：{kk} の形か値が判定ではない")
                pred[(K, v)] = a
        if not (orc_dev <= TOL_F32):
            raise SystemExit(f"{p.name}：保存の確率から作り直した全ラベルの d′ が保存の表と {orc_dev:.2e} 違う")
        units.append(dict(name=p.stem, f=f, r=r, y=y, cl=cl, sen=sen, inten=inten, pred=pred, kieo=kieo, ass=ass, orc_dev=orc_dev))
    return units, pd.concat(frames, ignore_index=True)


def cells_of(units: list, cluster: str, drop: str | None = None) -> tuple[dict, dict]:
    """{(コーパス, K, 版): {(分割, 引き方): 塊ごとの数}} と，判定の並びの sha256．集計の段と同じ作り（`ST.counts` → `P12._add` → `P12._finish`）．
    cluster："cl" は話者の塊，"sen" は文の塊．drop：テストから除く文の符号（None なら全発話）．"""
    cells, psig = {}, {}
    for u in units:
        m = np.ones(len(u["y"]), bool) if drop is None else (u["sen"] != drop)
        y, g = u["y"][m], u[cluster][m]
        for (K, v), pred in u["pred"].items():
            d = cells.setdefault((CD, K, v), {})
            if (u["f"], u["r"]) in d:
                raise SystemExit(f"{u['name']}：同じ (分割, 引き方) の単位が重なっている")
            P12._add(d, (u["f"], u["r"]), ST.counts(y, pred[m], g), f"{u['name']} {v}")
            psig.setdefault((CD, K, v), hashlib.sha256()).update(np.ascontiguousarray(pred[m].astype(np.int8)).tobytes())
    P12._finish(cells)
    return cells, psig


def sub(A: dict, sn: str) -> dict:
    """K 本の引き方の組 sn の単位だけ（集計の段の table_block の sub と同じ絞り方）．"""
    ss = E21.SEED_SETS[sn]
    return A if ss is None else {k_: x for k_, x in A.items() if k_[1] in ss}


def stack8(P: dict, Q: dict) -> dict:
    """2 つの数（件数が違ってよい．塊の並びは同じ）を列の方向につなぐ．8 列の再現率の平均は (UAR(P) ＋ UAR(Q)) ÷ 2 になる．"""
    if set(P) != set(Q) or any(not np.array_equal(P[k_][0], Q[k_][0]) for k_ in P):
        raise SystemExit("つなぐ 2 つの数の単位か塊の並びがそろわない")
    return {k_: (P[k_][0], np.concatenate([P[k_][1], Q[k_][1]], 1), np.concatenate([P[k_][2], Q[k_][2]], 1)) for k_ in P}


def dev(a, b) -> float:
    """差の絶対値の最大．NaN が 1 つでもあれば NaN を返す（pandas の max は NaN を飛ばすので numpy で取る）．"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max()) if len(a) else float("nan")


def txt_diff(m: pd.DataFrame, cols: tuple) -> int:
    """文字の列（空は ""）が集計の段と違う行の数．"""
    return int(sum((m[c_].fillna("").astype(str) != m[f"{c_}_s"].fillna("").astype(str)).sum() for c_ in cols))


def check_summary(lv: pd.DataFrame, ef: pd.DataFrame, pe: pd.DataFrame, sdir: Path, n_ver: int, con: list) -> dict:
    """作り直した水準・差・感情ごとの再現率（全発話・話者の塊）を，集計の段の `levels.csv`・`effects.csv`・`per_emotion.csv` の CREMA-D の行と照らす．合わなければ止まる．"""
    for q in ("levels.csv", "effects.csv", "per_emotion.csv"):
        if not (sdir / q).exists():
            raise SystemExit(f"{sdir / q} が無い")
    sl, se, sp = (pd.read_csv(sdir / q, low_memory=False) for q in ("levels.csv", "effects.csv", "per_emotion.csv"))
    sl, se, sp = sl[sl.part == PART], se[se.part == PART], sp[sp.part == PART]
    n_set = len(E21.SEED_SETS)
    if any(d.duplicated(k_).any() for d, k_ in ((lv, LK), (ef, EK), (pe, PK), (sl, LK), (se, EK), (sp, PK))):
        raise SystemExit("水準・差・感情ごとの行が重なっている")
    ml, me, mp = lv.merge(sl, on=LK, suffixes=("", "_s")), ef.merge(se, on=EK, suffixes=("", "_s")), pe.merge(sp, on=PK, suffixes=("", "_s"))
    res = dict(levels=dict(n=len(ml), n_mine=len(lv), n_summary=len(sl), n_expected=n_ver * len(KS) * n_set, est_dev_max=dev(ml.uar, ml.uar_s),
                           ci_dev_max=max(dev(ml.lo, ml.lo_s), dev(ml.hi, ml.hi_s), dev(ml.se, ml.se_s)), count_dev_max=dev(ml.n_fr, ml.n_fr_s),
                           text_mismatch=txt_diff(ml, ("role", "method", "same_as", "same_pred"))),
               effects=dict(n=len(me), n_mine=len(ef), n_summary=len(se), n_expected=len(con) * len(KS) * n_set, est_dev_max=dev(me.est, me.est_s),
                            ci_dev_max=max(dev(me.lo, me.lo_s), dev(me.hi, me.hi_s), dev(me.se, me.se_s)), count_dev_max=dev(me.n_fr, me.n_fr_s),
                            text_mismatch=txt_diff(me, ("role_a", "role_b", "method", "a_same_as", "b_same_as", "same_pred", "mark"))),
               per_emotion=dict(n=len(mp), n_mine=len(pe), n_summary=len(sp), n_expected=n_ver * len(KS) * 4, est_dev_max=max(dev(mp.recall, mp.recall_s), dev(mp.recall_pooled, mp.recall_pooled_s)),
                                ci_dev_max=max(dev(mp.lo, mp.lo_s), dev(mp.hi, mp.hi_s)), count_dev_max=max(dev(mp.n, mp.n_s), dev(mp.n_units, mp.n_units_s), dev(mp.n_per_seed, mp.n_per_seed_s)),
                                text_mismatch=txt_diff(mp, ("role",))),
               pairs_equal=bool(set(zip(se.a, se.b)) == set(con)), versions_equal=bool(set(sl.version) == set(lv.version)))
    print("集計の段との照合：" + "，".join(f"{lab} {res[k_]['n']} 行（期待 {res[k_]['n_expected']}）で点推定のずれの最大 {res[k_]['est_dev_max']:.2e}・区間のずれの最大 {res[k_]['ci_dev_max']:.2e}・"
                                  f"文字の列の食い違い {res[k_]['text_mismatch']}" for lab, k_ in (("水準", "levels"), ("差", "effects"), ("感情ごと", "per_emotion"))), flush=True)
    for k_ in ("levels", "effects", "per_emotion"):
        q = res[k_]
        if not (q["n"] == q["n_mine"] == q["n_summary"] == q["n_expected"]):
            raise SystemExit(f"集計の段と照らす行の数が違う（{k_}：照らせた {q['n']}，作り直し {q['n_mine']}，集計 {q['n_summary']}，期待 {q['n_expected']}）")
        if not (q["est_dev_max"] <= TOL_EST and q["ci_dev_max"] <= TOL_CI and q["count_dev_max"] == 0 and q["text_mismatch"] == 0):
            raise SystemExit(f"集計の段と合わない（{k_}：{q}）")
    if not (res["pairs_equal"] and res["versions_equal"]):
        raise SystemExit("版の集合か差の対の集合が集計の段と違う")
    return res


def check_confusion(conf: dict, sdir: Path, vers: tuple) -> dict:
    """文ごとの取り違えの数を文で足した値を，集計の段の `confusion.csv`（判定から直接数えた数．5 版）と照らす．合わなければ止まる．"""
    if not (sdir / "confusion.csv").exists():
        raise SystemExit(f"{sdir / 'confusion.csv'} が無い")
    sc = pd.read_csv(sdir / "confusion.csv", low_memory=False)
    sc = sc[sc.c == CD]
    mine = pd.DataFrame([dict(K=K, version=v, true=i, pred=j, n=float(conf[(K, v)][:, i, j].sum())) for K in KS for v in vers for i in range(4) for j in range(4)])
    m = mine.merge(sc, on=["K", "version", "true", "pred"], suffixes=("", "_s"))
    res = dict(n=len(m), n_summary=len(sc), n_expected=len(vers) * len(KS) * 16, count_dev_max=dev(m.n, m.n_s), versions_equal=bool(set(sc.version) == set(vers)))
    if not (res["n"] == res["n_summary"] == res["n_expected"] and res["count_dev_max"] == 0 and res["versions_equal"]):
        raise SystemExit(f"取り違えの数が集計の段と合わない：{res}")
    return res


def breakdown(units: list, vers: tuple, by: str, groups: tuple) -> dict:
    """{(K, 版): 数 (群, 本当の感情, 答えた感情)}．全部の単位の数を足し合わせる．by："sen"（文）か "inten"（強さ）．"""
    idx = {g: j for j, g in enumerate(groups)}
    acc = {}
    for u in units:
        gi, y = np.array([idx[x] for x in u[by]]), u["y"]
        for K in KS:
            for v in vers:
                np.add.at(acc.setdefault((K, v), np.zeros((len(groups), 4, 4))), (gi, y, u["pred"][(K, v)]), 1)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=str(OUT21 / "cremad"))
    ap.add_argument("--summary", default=str(OUT21 / "summary"))
    ap.add_argument("--out", default=str(OUT21 / "posthoc"))
    ap.add_argument("--allow-partial", action="store_true", help="検査用：単位の数と名前，単位のコミット，集計の段との照合，式の確かめ，検査役の本数との照合を省く（本番の置き場には書けない）")
    args = ap.parse_args()
    t0 = time.time()
    strict = not args.allow_partial
    udir, sdir, out = Path(args.units).resolve(), Path(args.summary).resolve(), Path(args.out).resolve()
    if out == (OUT21 / "posthoc").resolve() and not (strict and udir == (OUT21 / "cremad").resolve() and sdir == (OUT21 / "summary").resolve()):
        raise SystemExit("本番の置き場に書くのは，本番の単位と集計を既定の確かめつきで読んだときだけ（--allow-partial や別の置き場の単位・集計では書かない）")
    fz = E21.load_frozen()                                                   # 固定のファイルの元の sha256 と格子を確かめて読む（集計の段と同じ読み方）
    fk, mods = P18.fus(CD), P18.MODS_OF[CD]
    nf = f"none_{fk}"
    con = con_of(fk)
    if len(con) != N_CON or len(set(con)) != len(con):
        raise SystemExit(f"差の対が {len(con)} 個（重なりを除くと {len(set(con))}．{N_CON} のはず）")
    form_chk = check_assume_formulas(strict) if strict else "省いた（--allow-partial）"
    units, cs = load_units(udir, fz, fk, mods, strict)
    ucom = sorted(map(str, cs.commit.unique()))
    if strict and not all(E21.same_code(x, ucom[0]) for x in ucom):          # 集計の段の check_commits と同じ確かめ
        raise SystemExit("計算に関わるファイルの違うコミットで作った単位が混ざっている")
    if strict and not all(E21.same_code(x, "HEAD", E21.FIXED) for x in ucom):
        raise SystemExit("単位を作った後に，固定のファイルか格子の台本が変わっている")
    if cs.duplicated(["name", "K", "version", "grid"]).any():
        raise SystemExit("単位の csv の行が重なっている")
    folds = sorted({u["f"] for u in units})
    one = [next(u for u in units if u["f"] == f) for f in folds]             # 分割ごとに 1 単位（同じ分割のテストは引き方によらず同じ．load_units が確かめた）
    per_seed = len(folds) / len(units)                                       # 50 単位の合計を，別々の発話の数に直す倍率
    n_spk = len({c_ for u in one for c_ in u["cl"]})
    if strict and n_spk != N_SPK:
        raise SystemExit(f"テストの話者の塊が {n_spk}（{N_SPK} のはず）")
    geff = P18.g_eff_of(CD)                                                  # 集計の段と同じ関数（話者の塊の実効の数．20 以上なので塊の引き直しになる）
    n_ver = len({v for _, v in units[0]["pred"]})
    print(f"読み込み済み（{time.time() - t0:.0f} 秒）", flush=True)

    # ---- 1. 全発話・話者の塊：集計の段を作り直して照らす
    cells_all, psig_all = cells_of(units, "cl")
    lv_all, ef_all, pe_all = (pd.DataFrame(x) for x in E21.table_block(cells_all, con, {CD: geff}, PART, psig_all))
    chk = check_summary(lv_all, ef_all, pe_all, sdir, n_ver, con) if strict else "省いた（--allow-partial）"
    print(f"1 済み（{time.time() - t0:.0f} 秒）", flush=True)

    # ---- 2. 文 IEO を除いたテスト（話者の塊．同じ乱数の鍵）
    cells_no, psig_no = cells_of(units, "cl", DROP)
    for k_, dd in cells_no.items():                                          # 塊の並びが全発話のときと同じ（話者が欠けない）なら，塊の引き直しは全発話のときと同じ話者の組を引く
        if any(not np.array_equal(dd[u_][0], cells_all[k_][u_][0]) for u_ in dd):
            raise SystemExit(f"文 {DROP} を除くとテストの話者の塊が変わる：{k_}")
    lv_no, ef_no, pe_no = (pd.DataFrame(x) for x in E21.table_block(cells_no, con, {CD: geff}, PART, psig_no))
    lv_no = lv_no.merge(lv_all[LK + ["uar", "lo", "hi"]].rename(columns={"uar": "uar_all", "lo": "lo_all", "hi": "hi_all"}), on=LK, how="left", validate="one_to_one")
    ef_no = ef_no.merge(ef_all[EK + ["est", "lo", "hi", "mark"]].rename(columns={"est": "est_all", "lo": "lo_all", "hi": "hi_all", "mark": "mark_all"}), on=EK, how="left",
                        validate="one_to_one")
    pe_no = pe_no.merge(pe_all[PK + ["recall", "lo", "hi", "recall_pooled", "n"]].rename(columns={"recall": "recall_all", "lo": "lo_all", "hi": "hi_all", "recall_pooled": "recall_pooled_all",
                                                                                            "n": "n_all"}), on=PK, how="left", validate="one_to_one")
    lv_no["d_uar"], ef_no["d_est"], pe_no["d_recall"] = lv_no.uar - lv_no.uar_all, ef_no.est - ef_no.est_all, pe_no.recall - pe_no.recall_all
    # 「除いた値 − 全発話の値」の区間：引き直しの 1 回ごとに UAR を別々に作ってから差を取る（冒頭の 2 の説明）．乱数の鍵はその行の鍵と同じ
    dl = [ST.effect_stats(sub(cells_no[(CD, int(r_.K), r_.version)], r_.seeds), sub(cells_all[(CD, int(r_.K), r_.version)], r_.seeds), geff,
                          E21.seed_rng("L", PART, CD, int(r_.K), r_.version, r_.seeds)) for r_ in lv_no.itertuples()]
    de = [ST.effect_stats(stack8(sub(cells_no[(CD, int(r_.K), r_.a)], r_.seeds), sub(cells_all[(CD, int(r_.K), r_.b)], r_.seeds)),
                          stack8(sub(cells_no[(CD, int(r_.K), r_.b)], r_.seeds), sub(cells_all[(CD, int(r_.K), r_.a)], r_.seeds)), geff,
                          E21.seed_rng("A", PART, r_.a, r_.b, CD, int(r_.K), r_.seeds)) for r_ in ef_no.itertuples()]
    for d, es, mult in ((lv_no, dl, 1.0), (ef_no, de, 2.0)):                 # 差の差は 8 列の平均の差なので 2 倍する
        d["d_lo"], d["d_hi"], d["d_se"] = [mult * e["lo"] for e in es], [mult * e["hi"] for e in es], [mult * e["se"] for e in es]
        d["d_method"] = [e["method"] for e in es]
    dd_chk = dict(levels_est_dev_max=dev(lv_no.d_uar, [e["effect"] for e in dl]), effects_est_dev_max=dev(ef_no.d_est, [2.0 * e["effect"] for e in de]),
                  methods=sorted(set(lv_no.d_method) | set(ef_no.d_method)), n_levels=len(dl), n_effects=len(de))
    if not (dd_chk["levels_est_dev_max"] <= TOL_EST and dd_chk["effects_est_dev_max"] <= TOL_EST and dd_chk["methods"] == ["boot_cl+seed"]):
        raise SystemExit(f"「除いた値 − 全発話の値」の区間の確かめが合わない：{dd_chk}")
    if len(lv_no) != len(lv_all) or len(ef_no) != len(ef_all) or len(pe_no) != len(pe_all) or not np.isfinite(lv_no[["uar", "lo", "hi", "d_uar", "d_lo", "d_hi"]].to_numpy(float)).all() \
            or not np.isfinite(ef_no[["est", "lo", "hi", "d_est", "d_lo", "d_hi"]].to_numpy(float)).all() or not np.isfinite(pe_no[["recall", "lo", "hi", "d_recall"]].to_numpy(float)).all():
        raise SystemExit("文を除いた表の行の数が全発話の表と違うか，値に NaN がある")
    rem = pd.DataFrame([dict(name=u["name"], fold=u["f"], seed=u["r"], emotion=i, n_all=int((u["y"] == i).sum()), n_removed=int(((u["y"] == i) & (u["sen"] == DROP)).sum()))
                        for u in units for i in range(4)])
    rem["n_kept"] = rem.n_all - rem.n_removed
    if (rem.n_kept <= 0).any() or (rem.n_removed <= 0).any():
        raise SystemExit(f"文 {DROP} の発話が無い単位か，除くと感情が欠ける単位がある")
    rem1 = rem[rem.name.isin([u["name"] for u in one])]                      # 別々の発話の数（分割ごとに 1 単位）
    removed = dict(sentence=DROP, units_sum={str(i): int(rem[rem.emotion == i].n_removed.sum()) for i in range(4)}, units_sum_all=int(rem.n_removed.sum()),
                   distinct={str(i): int(rem1[rem1.emotion == i].n_removed.sum()) for i in range(4)}, distinct_all=int(rem1.n_removed.sum()),
                   distinct_test_all=int(rem1.n_all.sum()), distinct_test_by_emotion={str(i): int(rem1[rem1.emotion == i].n_all.sum()) for i in range(4)},
                   neutral_share_in_dropped=float(rem1[rem1.emotion == 0].n_removed.sum() / rem1.n_removed.sum()),
                   neutral_share_in_kept=float(rem1[rem1.emotion == 0].n_kept.sum() / rem1.n_kept.sum()))
    print(f"2 済み（{time.time() - t0:.0f} 秒）", flush=True)

    # ---- 3. 文を塊にした区間（全発話．12 塊．自由度は文の実効の数から）
    n_sent = len(SENT)
    cells_s, _ = cells_of(units, "sen")
    if any(len(x[0]) != n_sent for dd in cells_s.values() for x in dd.values()):
        raise SystemExit("文の塊が 12 でない単位がある")
    ns = np.array([sum(int((u["sen"] == s_).sum()) for u in one) for s_ in SENT], float)        # 文ごとの別々の発話の数（テスト）
    g_sent = float(ns.sum() ** 2 / (ns ** 2).sum())                          # 文の実効の数（dai1_stats の G_eff と同じ式）
    df_g = max(1, int(np.floor(g_sent)) - 1)                                 # effect_stats が使う自由度と同じ式
    if g_sent >= 20:
        raise SystemExit(f"文の実効の数が {g_sent:.2f} で，jackknife の道にならない")
    lv_s, ef_s, _ = (pd.DataFrame(x) for x in E21.table_block(cells_s, con, {CD: g_sent}, PART, psig_all))
    lv_s, ef_s = lv_s.drop(columns=["same_as"]), ef_s.drop(columns=["a_same_as", "b_same_as"])   # 文の塊ごとの正解の数が同じ版の印になるので外す（same_pred は判定の並びの印で，塊によらない）
    lv_s = lv_s.merge(lv_all[LK + ["uar", "lo", "hi", "se", "method"]].rename(columns=lambda c_: c_ if c_ in LK else f"{c_}_speaker"), on=LK, how="left", validate="one_to_one")
    ef_s = ef_s.merge(ef_all[EK + ["est", "lo", "hi", "se", "method", "mark"]].rename(columns=lambda c_: c_ if c_ in EK else f"{c_}_speaker"), on=EK, how="left", validate="one_to_one")
    qg, q12 = float(sst.t.ppf(0.975, df_g)), float(sst.t.ppf(0.975, n_sent - 1))
    sent_chk = dict(n_clusters=n_sent, g_eff_sentence=g_sent, df_main=df_g, df_n12=n_sent - 1, n_per_sentence_distinct=dict(zip(SENT, map(int, ns))))
    for lab, d, est in (("levels", lv_s, "uar"), ("effects", ef_s, "est")):
        d["n_clusters"], d["g_eff_sentence"], d["df_n12"] = n_sent, g_sent, n_sent - 1
        d["lo_n12"], d["hi_n12"] = d[est] - q12 * d.se, d[est] + q12 * d.se   # 自由度を塊の数 12 から決めた場合（参考）
        q = dict(n=len(d), est_dev_vs_speaker_max=dev(d[est], d[f"{est}_speaker"]), ci_formula_dev_max=max(dev(d.lo, d[est] - qg * d.se), dev(d.hi, d[est] + qg * d.se)),
                 methods=sorted(map(str, d.method.unique())), dfs=sorted(set(d.df.astype(float))))
        sent_chk[lab] = q
        if not (q["est_dev_vs_speaker_max"] <= TOL_EST and q["ci_formula_dev_max"] <= TOL_EST and q["methods"] == ["jk+seed"] and q["dfs"] == [float(df_g)]):
            raise SystemExit(f"文を塊にした区間の確かめが合わない（{lab}）：{q}")
    sent_chk["folds_dropped"] = sorted(set(ef_s.folds_dropped.fillna("").astype(str)))
    if not (len(ef_s) == len(ef_all) and len(lv_s) == len(lv_all) and sent_chk["folds_dropped"] == [""]):
        raise SystemExit(f"文を塊にした表の行の数が違うか，文を除くと外れる分割がある：{sent_chk}")
    print(f"3 済み（{time.time() - t0:.0f} 秒）", flush=True)

    # ---- 4. 文ごと・強さごとの内訳（記述．50 単位の数を足し合わせてから割る）
    bvers = ("prop", "prod", "stud_prop", "stud_prod", nf, "tcal", "none_A", "ret_A", "none_V", "ret_V", "start_A", "start_V", f"start_{fk}", "full_A", "full_V", f"full_{fk}")
    tot_ref = {(K, v): (sum(x[1].sum(0) for x in cells_all[(CD, K, v)].values()), sum(x[2].sum(0) for x in cells_all[(CD, K, v)].values())) for K in KS for v in bvers}
    conf_s, conf_i = breakdown(units, bvers, "sen", SENT), breakdown(units, bvers, "inten", INTENS)
    for lab, acc in (("文", conf_s), ("強さ", conf_i)):
        for (K, v), M in acc.items():                                        # 対角の合計と件数の合計が，1 で集計と照らした全発話の数と同じ
            if not (np.array_equal(np.einsum("gii->i", M), tot_ref[(K, v)][0]) and np.array_equal(M.sum((0, 2)), tot_ref[(K, v)][1])):
                raise SystemExit(f"内訳の数の合計が，照合した全発話の数と違う：{(lab, K, v)}")
    conf_chk = check_confusion(conf_s, sdir, ("stud_prop", "stud_prod", nf, "prop", "prod")) if strict else "省いた（--allow-partial）"
    srows, irows, crows = [], [], []
    for (K, v), M in conf_s.items():
        for j, g in enumerate(SENT):
            tot = M[j].sum(1)
            if (tot == 0).any():
                raise SystemExit(f"文 {g} に感情が欠ける")
            rec = 100.0 * np.diag(M[j]) / tot
            srows.append(dict(c=CD, sentence=g, K=K, version=v, role=E21.role_of(v), uar_pooled=float(rec.mean()), n=float(tot.sum()), n_per_seed=float(tot.sum()) * per_seed,
                              **{f"recall_pooled_{i}": float(rec[i]) for i in range(4)}, **{f"n_{i}": float(tot[i]) for i in range(4)}, n_units=len(units)))
            crows += [dict(c=CD, sentence=g, K=K, version=v, role=E21.role_of(v), true=i, pred=k_, n=float(M[j, i, k_]), n_per_seed=float(M[j, i, k_]) * per_seed, n_units=len(units))
                      for i in range(4) for k_ in range(4)]
    for (K, v), M in conf_i.items():
        for j, g in enumerate(INTENS):
            tot = M[j].sum(1)
            irows += [dict(c=CD, intensity=g, emotion=i, K=K, version=v, role=E21.role_of(v), recall_pooled=float(100.0 * M[j, i, i] / tot[i]), n=float(tot[i]),
                           n_per_seed=float(tot[i]) * per_seed, n_units=len(units)) for i in range(4) if tot[i] > 0]
    by_sent, by_int, by_conf = pd.DataFrame(srows), pd.DataFrame(irows), pd.DataFrame(crows)
    if len(by_sent) != len(SENT) * len(KS) * len(bvers) or by_int.duplicated(["intensity", "emotion", "K", "version"]).any() \
            or len(by_int) != (4 + 3 * 3) * len(KS) * len(bvers) or len(by_conf) != len(by_sent) * 16:   # 強さ × 感情の組：XX は 4 感情，LO・MD・HI は Happy・Sad・Angry
        raise SystemExit(f"内訳の行の数が違う：文 {len(by_sent)}，強さ {len(by_int)}，取り違え {len(by_conf)}")

    # ---- 5. K 本の中の文 IEO の本数と，単位ごとの値
    main_pairs = (("prop", "prod"), ("stud_prop", "stud_prod"), ("stud_prop", nf), ("stud_prod", nf), ("prop", nf))
    if not set(main_pairs) <= set(con):
        raise SystemExit("単位ごとの値を出す差の対が，集計の段の差の対に無い")
    krows = []
    for u in units:
        fr = (u["f"], u["r"])
        for K in KS:
            ua, un = {v: ST.uar_full(cells_all[(CD, K, v)][fr]) for v in bvers}, {v: ST.uar_full(cells_no[(CD, K, v)][fr]) for v in bvers}
            krows.append(dict(name=u["name"], fold=u["f"], seed=u["r"], K=K, n_kshot=4 * K, n_ieo=int(u["kieo"][K].sum()), **{f"n_ieo_{i}": int(u["kieo"][K][i]) for i in range(4)},
                              **{f"d_{a}__{b}": ua[a] - ua[b] for a, b in main_pairs}, **{f"d_noieo_{a}__{b}": un[a] - un[b] for a, b in main_pairs},
                              **{f"uar_{v}": ua[v] for v in bvers}, **{f"uar_noieo_{v}": un[v] for v in bvers}))
    ku = pd.DataFrame(krows)
    th = lambda d, col: float(d.groupby("fold")[col].mean().mean())  # noqa: E731     分割ごとに引き方で平均してから分割で平均（dai1_stats.theta と同じ）
    row = lambda d, **kw: d[np.logical_and.reduce([d[k_] == v for k_, v in kw.items()])].iloc[0]  # noqa: E731
    k_dev = 0.0
    for K in KS:                                                             # 単位ごとの値の平均が，照合した表の値（引き方の組 all）と同じ
        g = ku[ku.K == K]
        for a, b in main_pairs:
            k_dev = max(k_dev, abs(th(g, f"d_{a}__{b}") - row(ef_all, a=a, b=b, K=K, seeds="all").est), abs(th(g, f"d_noieo_{a}__{b}") - row(ef_no, a=a, b=b, K=K, seeds="all").est))
        for v in bvers:
            k_dev = max(k_dev, abs(th(g, f"uar_{v}") - row(lv_all, version=v, K=K, seeds="all").uar), abs(th(g, f"uar_noieo_{v}") - row(lv_no, version=v, K=K, seeds="all").uar))
    kshot_chk = dict(n=len(ku), unit_mean_dev_max=float(k_dev), reference=str(KSHOT_REF), reference_compared=False)
    if not (k_dev <= TOL_EST and len(ku) == len(units) * len(KS) and not ku.duplicated(["name", "K"]).any()):
        raise SystemExit(f"単位ごとの値の平均が表の値と合わない：{kshot_chk}")
    if strict and KSHOT_REF.exists():                                        # 検査役が別に数えた本数と照らす（値は写さない）
        cols = ["n_ieo"] + [f"n_ieo_{i}" for i in range(4)]
        m = ku[["name", "K"] + cols].merge(pd.read_csv(KSHOT_REF), on=["name", "K"], suffixes=("", "_ref"))
        kshot_chk.update(reference_compared=True, reference_sha256=dai15.sha(KSHOT_REF), n_matched=len(m), count_dev_max=dev(m[cols], m[[f"{c_}_ref" for c_ in cols]]))
        if not (len(m) == len(ku) and kshot_chk["count_dev_max"] == 0):
            raise SystemExit(f"K 本の中の文 {DROP} の本数が，検査役の数えた値と合わない：{kshot_chk}")

    # ---- 6. 仮定の表（K ごとの単位の平均．NaN は平均から外れる）
    ck = cs[cs.version == "_check"]
    acols = [x for x in ck.columns if x.startswith(ASSUME_PREFIX) or x in ASSUME_EXACT]
    if (ck.groupby("K").size() != len(units)).any():
        raise SystemExit("_check の行の数が K ごとに単位の数と違う")
    au = pd.DataFrame([dict(K=K, **u["ass"][K]) for u in units for K in KS])
    qcols = [x for x in au.columns if x != "K"]
    ass = pd.concat([pd.DataFrame(dict(c=CD, K=list(KS), n_units=len(units), check_row_assumption_columns=(",".join(acols) if acols else "無い"))),
                     pd.DataFrame({x: [float(ck[ck.K == K][x].mean()) for K in KS] for x in acols}),
                     au.groupby("K")[qcols].mean().loc[list(KS)].reset_index(drop=True)], axis=1)
    ass_finite = {x: {str(K): int(np.isfinite(au[au.K == K][x]).sum()) for K in KS} for x in qcols if not np.isfinite(au[x]).all()}

    # ---- 7. 書き出し（ここまでの確かめが全部通ったときだけ）
    out.mkdir(parents=True, exist_ok=True)
    files = {"cremad_no_ieo_levels.csv": lv_no, "cremad_no_ieo_effects.csv": ef_no, "cremad_no_ieo_per_emotion.csv": pe_no, "cremad_no_ieo_removed.csv": rem,
             "cremad_sentence_cluster_effects.csv": ef_s, "cremad_sentence_cluster_levels.csv": lv_s,
             "cremad_by_sentence.csv": by_sent, "cremad_by_intensity.csv": by_int, "cremad_by_sentence_confusion.csv": by_conf,
             "cremad_kshot_ieo_units.csv": ku, "cremad_assumptions.csv": ass}
    for n_, d in files.items():
        d.to_csv(out / n_, index=False)
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    me = Path(__file__).resolve()
    watch = list(E21.WATCH) + ["tools"]
    mods_py = tuple(E21.SCRIPTS) + ("dai21_eval5.py",)
    sums = {q.name: dai15.sha(q) for q in (sdir / "levels.csv", sdir / "effects.csv", sdir / "per_emotion.csv", sdir / "confusion.csv", sdir / "convergence.json") if q.exists()}
    npzs = sorted(udir.glob("*.npz"))
    pred_sha = {f"{K}|{v}": psig_all[(CD, K, v)].hexdigest() for (_, K, v) in psig_all}
    json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv, commit=git("rev-parse", "HEAD"),
                   dirty_tracked=bool(git("status", "--porcelain", "--untracked-files=no", "--", *watch)), dirty_paths=watch,       # 追跡しているファイルだけを見る
                   script=str(me), script_sha256=dai15.sha(me), script_committed_clean=bool(dai1.committed_clean(me)),
                   modules_sha256={n_: dai15.sha(ROOT / "exp" / n_) for n_ in mods_py}, frozen_sha256=dai15.sha(E21.FROZEN), frozen20_sha256=dai15.sha(E21.FROZEN20),
                   design_sha256=dai15.sha(E21.DESIGN), summary_dir=str(sdir), summary_sha256=sums,
                   dai21_provenance_sha256=(dai15.sha(udir.parent / "provenance.json") if (udir.parent / "provenance.json").exists() else ""),
                   units_dir=str(udir), n_units=len(units), n_folds=len(folds), unit_commits=ucom,
                   units_npz_digest=f"{len(npzs)}:" + hashlib.sha256("".join(f"{q.name} {dai15.sha(q)}\n" for q in npzs).encode()).hexdigest(),
                   allow_partial=args.allow_partial, host=platform.node(), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                   consts=dict(KS=KS, PART=PART, DROP=DROP, SENT=SENT, N_CON=N_CON, N_SPK=N_SPK, TOL_EST=TOL_EST, TOL_CI=TOL_CI, TOL_F32=TOL_F32, N_BOOT=ST.N_BOOT, MIN_TEST=ST.MIN_TEST,
                               SEED_SETS={k_: v for k_, v in E21.SEED_SETS.items()}, breakdown_versions=bvers, main_pairs=main_pairs, REF20=REF20),
                   cremad_points={str(K): {k_: v for k_, v in fz["cremad"][str(K)].items() if k_ in ("prop_grid", "noprof_sel", "student_main", "student_prop_sel")} for K in KS},
                   n_versions=n_ver, n_pairs=len(con), g_eff_speaker=geff, n_speaker_clusters=n_spk,
                   check_vs_summary=chk, confusion_check=conf_chk, d_interval_check=dd_chk, sentence_cluster_check=sent_chk, removed=removed, kshot_check=kshot_chk,
                   assumptions=dict(check_row_columns=(acols if acols else "無い"), rule_prefix=ASSUME_PREFIX, rule_exact=ASSUME_EXACT, check_row_columns_present=sorted(ck.dropna(axis=1, how="all").columns),
                                    from_npz=qcols, not_made=NOT_MADE, formula_check_vs_dai20=form_chk, orc_recheck_dev_max=float(max(u["orc_dev"] for u in units)),
                                    n_finite_where_some_nan=ass_finite, dk_cap=math.sqrt(2.0) * float(sst.norm.ppf(P16.AUC_CLIP[1])) - 1e-9),
                   pred_sha256_digest=hashlib.sha256("".join(f"{k_} {v}\n" for k_, v in sorted(pred_sha.items())).encode()).hexdigest(), pred_sha256=pred_sha,
                   outputs={n_: len(d) for n_, d in files.items()},
                   notes={"cremad_by_sentence.csv・cremad_by_intensity.csv・cremad_by_sentence_confusion.csv":
                          "50 単位（分割 5 × K 本の引き方 10）の正解の数と発話の数を足し合わせてから割った値（列の名前は *_pooled）．水準の表の UAR（単位ごとの UAR を引き方で平均してから分割で"
                          "平均した値）とは作りが違い，文ごとの UAR を 12 文で平均しても水準の値にはならない．区間は無い．n は 50 単位の合計，n_per_seed は別々の発話の数．"
                          "start_*・full_* は K によらない判定で，K ごとの行に同じ値が入る．強さ LO・MD・HI は文 IEO の Happy・Sad・Angry だけにあり，Neutral は XX だけ．"
                          "XX には，ほかの 11 文の全部と文 IEO の Neutral が入る",
                          "cremad_no_ieo_*.csv": "区間は話者の塊の引き直し（集計の段と同じ乱数の鍵）．same_as・same_pred は文 IEO を除いたテストの上の印．d_lo・d_hi・d_se は「除いた値 − 全発話の値」の"
                                                 "区間で，同じ乱数の鍵の同じ話者の組の上で，引き直しの 1 回ごとに UAR を別々に作ってから差を取った（差の差は 8 列につないだ数を effect_stats に渡して 2 倍）．"
                                                 "テストから文を除く表は，学習側（K 本に文 IEO が入ること）の影響を測っていない",
                          "cremad_sentence_cluster_*.csv": "lo・hi・mark は自由度 ⌊文の実効の数⌋ − 1 の区間（dai1_stats の規則）．lo_n12・hi_n12 は自由度を塊の数 12 から決めた場合（参考）．"
                                                           "話者と学習した分類器を固定し，テストの 12 文を取り直したときの揺れと K 本の引き方の揺れを含む．テストの話者の揺れは含まない．"
                                                           "学習側にも同じ 12 文があるので，学習に現れない文への区間ではない",
                          "cremad_kshot_ieo_units.csv": "n_ieo は K 本（4K 本）の中の文 IEO の本数，n_ieo_0〜3 は感情ごと．d_*・uar_* はその単位のテストの全発話の値，*_noieo_* は文 IEO を除いたテストの値．"
                                                        "差の名前は d_<a>__<b> ＝ UAR(a) − UAR(b)．start_*・full_* の UAR は K によらない",
                          "cremad_assumptions.csv": "K ごとの 50 単位の平均（NaN は平均から外れる）．ソースで測った表（ext・home）と K 本で測った表を，候補の全ラベルで測った表（orc．参照）と比べた量．"
                                                    "分類器が 2 つなので，分類器 × 感情の成分（intcorr・intperm の材料）は声の行と顔の行が符号だけ違う 4 個の数になり，相関は偶然でも大きく振れる．"
                                                    "profcorr_row_* と emocorr は 4 個の数の相関．drift_M は float32 で保存した確率から作った"},
                   seconds=time.time() - t0),
              open(out / "cremad_sentence_provenance.json", "w"), indent=1, ensure_ascii=False)
    print(f"単位 {len(units)}，版 {n_ver}，差の対 {len(con)}．除いた発話（別々の発話．感情 0〜3）{removed['distinct']}．文を塊にした区間：文の実効の数 {g_sent:.2f}，自由度 {df_g}．"
          f"取り違えの照合 {conf_chk}．K 本の本数の照合 {kshot_chk}．仮定の量の式の確かめ {form_chk}．仮定の量の列（_check の行）：{acols if acols else '無い'}．"
          f"出力の行の数 {({n_: len(d) for n_, d in files.items()})}．{time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
