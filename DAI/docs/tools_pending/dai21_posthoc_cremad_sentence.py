"""実験 21（DAI-21）の計算の後の表：CREMA-D の文についての感度と内訳（単位の保存の判定を数え直すだけ．学び直さない．GPU は使わない）．
設計 `DAI/docs/実験21_設計.md` 4 節末の「計算の後に保存から出す表」を出す．

CREMA-D は俳優 91 人が決まった 12 文を読むコーパスで，発話の名前は「話者_文_感情_強さ」（例 `1001_IEO_ANG_HI`）である．
文 `IEO`（It's eleven o'clock.）だけが強さ LO・MD・HI を持ち，4 ラベルの発話では Neutral が 10 本に 1 本である（ほかの 11 文は 4 本に 1 本）．
同じ 12 文が学習側とテストにまたがる．

出すもの（置き場は既定で `results/dai21_eval5/posthoc/`．確かめが全部通ってから書く）：
  1. 照合（ファイルは書かない）：テストの全発話・話者の塊で，集計の段（`dai21_eval5.stage_summary` の CREMA-D の部分）の水準と差を作り直し，
     `summary/levels.csv`・`effects.csv` の `part == "cremad"` の行と照らす．点推定のずれが 1e-9，区間の両端と標準誤差のずれが 1e-6 を超えるか，
     照らした行の数が「版の数 × K 3 つ × K 本の引き方の組 3 つ」「差の対 78 × 3 × 3」と違えば止まる．`same_as`・`same_pred`・`method`・印も照らす．
  2. `cremad_no_ieo_levels.csv`・`cremad_no_ieo_effects.csv`：テストから文 `IEO` の発話を除き，同じ版・同じ差の対・同じ区間の作り方（話者の塊．
     集計の段と同じ乱数の鍵）で出した水準と差．`*_all` の列は全発話の値（1 で集計と照らした値），`d_uar`・`d_est` は「除いた値 − 全発話の値」．
     `same_as`・`same_pred` は，文 `IEO` を除いたテストの上での印である．`cremad_no_ieo_removed.csv` は除いた発話の数（単位ごと・感情ごと）．
  3. `cremad_sentence_cluster_effects.csv`：テストの全発話で，区間の塊を話者から文（12 塊）に替えた差の区間．`dai1_stats.effect_stats` に，文を塊にした数と
     塊の数 12 を渡す（塊が 20 未満の作り：文を 1 つずつ，全部の分割から同時に除く jackknife の分散に，K 本の引き方の項を足す．自由度は 12 − 1 ＝ 11）．
     点推定は話者の塊のときと同じになる（確かめる）．`*_speaker` の列は話者の塊の区間（1 の値）．
     `lo_geff`・`hi_geff` は，自由度を文の実効の数 (Σn)² / Σn²（文 `IEO` の発話が多いので 12 より小さい）から決めたときの区間である（標準誤差は同じ）．
     この区間は話者を固定して文だけを揺らした区間で，話者の揺れを含まない（感度を見るための値）．
  4. `cremad_by_sentence.csv`・`cremad_by_intensity.csv`：文ごと（12）の UAR と，強さ × 感情ごとの再現率（記述．区間は付けない）．
     どちらも，全部の単位（分割 5 × K 本の引き方 10 ＝ 50）の正解の数と発話の数を足し合わせてから割った値である．`n` は 50 単位の合計で，
     別々の発話の数は `n_per_seed`（同じ分割のテストは K 本の引き方によらず同じ）．強さ LO・MD・HI は文 `IEO` の Happy・Sad・Angry だけにあり，
     Neutral は XX だけなので，強さごとの UAR は作らない．
  5. `cremad_assumptions.csv`：仮定の表．実験 20 の集計の段（`dai20_eval.stage_summary` の「仮定の表」）と同じ選び方で，単位の csv の `_check` の行から
     仮定の量の列を選んで K ごとに平均する．CREMA-D の単位の csv にその列が 1 つも無ければ `check_row_assumption_columns` に「無い」と書く．
     実験 20 の集計の段が npz の表から作る量のうち，K 本で測った d′ が上限に当たった割合と 0 だった割合（`dk_ceiling_share`・`dk_zero_share`）は，
     CREMA-D の npz の `D_M_K_K{K}` から同じ式で作る．行ごとの相関（`profcorr_row_*`）は，分類器が 2 つだと 4 個の数の相関になるので作らない．
  6. `cremad_sentence_provenance.json`：由来の記録．

版の名前 → npz の鍵の対応（`names_of`）と差の対（`con_of`）は，集計の段の関数の中の変数（`names`・`con`）で import できないので，同じ式を写した．
写しが集計の段と同じであることは，1 の照合（版の集合，差の対の集合，値）で確かめる．それ以外の定数と式（`A2`・`EXTRA_T`・`ABL`・`GRID`・`same_but`・
`table_block`・`seed_rng`・`SEED_SETS`・`g_eff_of`・`load_frozen`・`todo_cremad`）は `dai21_eval5` とその部品を import して使う．

使い方：CUDA_VISIBLE_DEVICES="" .venv/bin/python DAI/docs/tools_pending/dai21_posthoc_cremad_sentence.py [--units …] [--summary …] [--out …] [--allow-partial]
  既定は本番の置き場（単位 50 個・名前・鍵・K・コミット・集計との一致を確かめ，合わなければ何も書かずに止まる）．
  --allow-partial は検査用（単位の数と名前，単位のコミット，集計との照合を省く）．
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

sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
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
LK, EK = ["part", "c", "K", "version", "seeds"], ["part", "a", "b", "c", "K", "seeds"]      # 水準の行と差の行を決める列
ASSUME_PREFIX = ("intcorr_M_", "intperm_M_", "profcorr_M_", "levelcorr_M_", "topok_M_", "emocorr_M_", "ret_", "excess_", "level_M_")   # dai20_eval.stage_summary の「仮定の表」の列の選び方（写し）
ASSUME_EXACT = ("drift_M", "sdratio_M")


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


def load_units(udir: Path, fz: dict, fk: str, strict: bool) -> tuple[list, pd.DataFrame]:
    """単位の csv と npz を読み，テストのラベル・話者の塊・文・強さと，全部の版の判定を取り出す．名前・鍵・K・発話の名前の形が合わなければ止まる．"""
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
        need = {kk for K in KS for kk in nm[K].values()} | {"y_test", "cl_test", "sid_test"} | {f"{k}_K{K}" for K in KS for k in ("D_M_K", "kpos")}
        miss = sorted(need - keys)
        if miss:
            raise SystemExit(f"{p.name}：npz に鍵が無い：{miss[:5]}（計 {len(miss)}）")
        y, cl, sid = z["y_test"].astype(int), z["cl_test"].astype(str), z["sid_test"].astype(str)
        parts = [s.split("_") for s in sid]
        if any(len(q) != 4 for q in parts) or len(set(sid)) != len(sid) or not (len(y) == len(cl) == len(sid)):
            raise SystemExit(f"{p.name}：発話の名前が「話者_文_感情_強さ」の形でないか，重なりがある")
        sen, emo, inten = (np.array([q[j] for q in parts]) for j in (1, 2, 3))
        if not (set(sen) == set(SENT) and set(emo) <= set(EMO) and set(inten) <= set(INTENS)):
            raise SystemExit(f"{p.name}：文・感情・強さの符号が想定と違う（テストに 12 文がそろわない場合を含む）")
        if not (np.array([EMO[e] for e in emo]) == y).all():
            raise SystemExit(f"{p.name}：発話の名前の感情とラベルが食い違う")
        if not ((inten == "XX") | ((sen == DROP) & (y != 0))).all() or ((sen == DROP) & (y != 0) & (inten == "XX")).any():
            raise SystemExit(f"{p.name}：強さ LO・MD・HI が文 {DROP} の Happy・Sad・Angry 以外にあるか，その発話に XX がある")
        if not np.array_equal(sid_of.setdefault(f, sid), sid):
            raise SystemExit(f"{p.name}：同じ分割のテストが K 本の引き方によって違う")
        pred = {}
        for K in KS:
            if len(z[f"kpos_K{K}"]) != 4 * K:
                raise SystemExit(f"{p.name}：K = {K} の K 本が {4 * K} 本でない")
            for v, kk in nm[K].items():
                a = z[kk].astype(int)
                if a.shape != y.shape or a.min() < 0 or a.max() > 3:
                    raise SystemExit(f"{p.name}：{kk} の形か値が判定ではない")
                pred[(K, v)] = a
        units.append(dict(name=p.stem, f=f, r=r, y=y, cl=cl, sen=sen, inten=inten, pred=pred, DK={K: z[f"D_M_K_K{K}"].astype(float) for K in KS}))
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


def dev(a: pd.Series, b: pd.Series) -> float:
    """差の絶対値の最大．NaN が 1 つでもあれば NaN を返す（pandas の max は NaN を飛ばすので numpy で取る）．"""
    return float(np.abs(a.to_numpy(float) - b.to_numpy(float)).max()) if len(a) else float("nan")


def txt_diff(m: pd.DataFrame, cols: tuple) -> int:
    """文字の列（空は ""）が集計の段と違う行の数．"""
    return int(sum((m[c_].fillna("").astype(str) != m[f"{c_}_s"].fillna("").astype(str)).sum() for c_ in cols))


def check_summary(lv: pd.DataFrame, ef: pd.DataFrame, sdir: Path, n_ver: int, con: list) -> dict:
    """作り直した水準と差（全発話・話者の塊）を，集計の段の `levels.csv`・`effects.csv` の CREMA-D の行と照らす．合わなければ止まる．"""
    for q in (sdir / "levels.csv", sdir / "effects.csv"):
        if not q.exists():
            raise SystemExit(f"{q} が無い")
    sl, se = pd.read_csv(sdir / "levels.csv", low_memory=False), pd.read_csv(sdir / "effects.csv", low_memory=False)
    sl, se = sl[sl.part == PART], se[se.part == PART]
    n_lv_exp, n_ef_exp = n_ver * len(KS) * len(E21.SEED_SETS), len(con) * len(KS) * len(E21.SEED_SETS)
    if lv.duplicated(LK).any() or ef.duplicated(EK).any() or sl.duplicated(LK).any() or se.duplicated(EK).any():
        raise SystemExit("水準か差の行が重なっている")
    ml, me = lv.merge(sl, on=LK, suffixes=("", "_s")), ef.merge(se, on=EK, suffixes=("", "_s"))
    res = dict(levels=dict(n=len(ml), n_mine=len(lv), n_summary=len(sl), n_expected=n_lv_exp, est_dev_max=dev(ml.uar, ml.uar_s),
                           ci_dev_max=max(dev(ml.lo, ml.lo_s), dev(ml.hi, ml.hi_s), dev(ml.se, ml.se_s)), n_fr_dev_max=dev(ml.n_fr, ml.n_fr_s),
                           text_mismatch=txt_diff(ml, ("role", "method", "same_as", "same_pred"))),
               effects=dict(n=len(me), n_mine=len(ef), n_summary=len(se), n_expected=n_ef_exp, est_dev_max=dev(me.est, me.est_s),
                            ci_dev_max=max(dev(me.lo, me.lo_s), dev(me.hi, me.hi_s), dev(me.se, me.se_s)), n_fr_dev_max=dev(me.n_fr, me.n_fr_s),
                            text_mismatch=txt_diff(me, ("role_a", "role_b", "method", "a_same_as", "b_same_as", "same_pred", "mark"))),
               pairs_equal=bool(set(zip(se.a, se.b)) == set(con)), versions_equal=bool(set(sl.version) == set(lv.version)))
    print(f"集計の段との照合：水準 {len(ml)} 行（期待 {n_lv_exp}）で点推定のずれの最大 {res['levels']['est_dev_max']:.2e}・区間のずれの最大 {res['levels']['ci_dev_max']:.2e}，"
          f"差 {len(me)} 行（期待 {n_ef_exp}）で点推定のずれの最大 {res['effects']['est_dev_max']:.2e}・区間のずれの最大 {res['effects']['ci_dev_max']:.2e}．"
          f"文字の列の食い違い 水準 {res['levels']['text_mismatch']}・差 {res['effects']['text_mismatch']}", flush=True)
    for k_ in ("levels", "effects"):
        q = res[k_]
        if not (q["n"] == q["n_mine"] == q["n_summary"] == q["n_expected"]):
            raise SystemExit(f"集計の段と照らす行の数が違う（{k_}：照らせた {q['n']}，作り直し {q['n_mine']}，集計 {q['n_summary']}，期待 {q['n_expected']}）")
        if not (q["est_dev_max"] <= TOL_EST and q["ci_dev_max"] <= TOL_CI and q["n_fr_dev_max"] == 0 and q["text_mismatch"] == 0):
            raise SystemExit(f"集計の段と合わない（{k_}：{q}）")
    if not (res["pairs_equal"] and res["versions_equal"]):
        raise SystemExit("版の集合か差の対の集合が集計の段と違う")
    return res


def breakdown(units: list, vers: tuple, by: str, groups: tuple) -> dict:
    """{(K, 版): (正解の数 (群, 4), 発話の数 (群, 4))}．全部の単位の数を足し合わせる．by："sen"（文）か "inten"（強さ）．"""
    idx = {g: j for j, g in enumerate(groups)}
    acc = {}
    for u in units:
        gi, y = np.array([idx[x] for x in u[by]]), u["y"]
        for K in KS:
            for v in vers:
                cor, tot = acc.setdefault((K, v), (np.zeros((len(groups), 4)), np.zeros((len(groups), 4))))
                hit = u["pred"][(K, v)] == y
                np.add.at(tot, (gi, y), 1)
                np.add.at(cor, (gi[hit], y[hit]), 1)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=str(OUT21 / "cremad"))
    ap.add_argument("--summary", default=str(OUT21 / "summary"))
    ap.add_argument("--out", default=str(OUT21 / "posthoc"))
    ap.add_argument("--allow-partial", action="store_true", help="検査用：単位の数と名前，単位のコミット，集計の段との照合を省く（本番の置き場には書けない）")
    args = ap.parse_args()
    t0 = time.time()
    strict = not args.allow_partial
    udir, sdir, out = Path(args.units).resolve(), Path(args.summary).resolve(), Path(args.out).resolve()
    if out == (OUT21 / "posthoc").resolve() and not (strict and udir == (OUT21 / "cremad").resolve() and sdir == (OUT21 / "summary").resolve()):
        raise SystemExit("本番の置き場に書くのは，本番の単位と集計を既定の確かめつきで読んだときだけ（--allow-partial や別の置き場の単位・集計では書かない）")
    fz = E21.load_frozen()                                                   # 固定のファイルの元の sha256 と格子を確かめて読む（集計の段と同じ読み方）
    fk = P18.fus(CD)
    con = con_of(fk)
    if len(con) != N_CON or len(set(con)) != len(con):
        raise SystemExit(f"差の対が {len(con)} 個（重なりを除くと {len(set(con))}．{N_CON} のはず）")
    units, cs = load_units(udir, fz, fk, strict)
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

    # ---- 1. 全発話・話者の塊：集計の段を作り直して照らす
    cells_all, psig_all = cells_of(units, "cl")
    lv_all, ef_all, _ = (pd.DataFrame(x) for x in E21.table_block(cells_all, con, {CD: geff}, PART, psig_all))
    chk = check_summary(lv_all, ef_all, sdir, n_ver, con) if strict else {}
    print(f"1 済み（{time.time() - t0:.0f} 秒）", flush=True)

    # ---- 2. 文 IEO を除いたテスト（話者の塊．同じ乱数の鍵）
    cells_no, psig_no = cells_of(units, "cl", DROP)
    for k_, dd in cells_no.items():                                          # 塊の並びが全発話のときと同じ（話者が欠けない）なら，塊の引き直しは全発話のときと同じ話者の組を引く
        if any(not np.array_equal(dd[u_][0], cells_all[k_][u_][0]) for u_ in dd):
            raise SystemExit(f"文 {DROP} を除くとテストの話者の塊が変わる：{k_}")
    lv_no, ef_no, _ = (pd.DataFrame(x) for x in E21.table_block(cells_no, con, {CD: geff}, PART, psig_no))
    lv_no = lv_no.merge(lv_all[LK + ["uar", "lo", "hi"]].rename(columns={"uar": "uar_all", "lo": "lo_all", "hi": "hi_all"}), on=LK, how="left", validate="one_to_one")
    ef_no = ef_no.merge(ef_all[EK + ["est", "lo", "hi", "mark"]].rename(columns={"est": "est_all", "lo": "lo_all", "hi": "hi_all", "mark": "mark_all"}), on=EK, how="left",
                        validate="one_to_one")
    lv_no["d_uar"], ef_no["d_est"] = lv_no.uar - lv_no.uar_all, ef_no.est - ef_no.est_all
    if len(lv_no) != len(lv_all) or len(ef_no) != len(ef_all) or not np.isfinite(lv_no[["uar", "lo", "hi", "d_uar"]].to_numpy(float)).all() \
            or not np.isfinite(ef_no[["est", "lo", "hi", "d_est"]].to_numpy(float)).all():
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

    # ---- 3. 文を塊にした区間（全発話．12 塊）
    n_sent = len(SENT)
    cells_s, _ = cells_of(units, "sen")
    if any(len(x[0]) != n_sent for dd in cells_s.values() for x in dd.values()):
        raise SystemExit("文の塊が 12 でない単位がある")
    _, ef_s, _ = (pd.DataFrame(x) for x in E21.table_block(cells_s, con, {CD: float(n_sent)}, PART, psig_all))
    ef_s = ef_s.drop(columns=["a_same_as", "b_same_as"])                    # 文の塊ごとの正解の数が同じ版の印になるので外す（same_pred は判定の並びの印で，塊によらない）
    ef_s = ef_s.merge(ef_all[EK + ["est", "lo", "hi", "se", "method", "mark"]].rename(columns=lambda c_: c_ if c_ in EK else f"{c_}_speaker"), on=EK, how="left",
                      validate="one_to_one")
    ns = np.array([sum(int((u["sen"] == s_).sum()) for u in one) for s_ in SENT], float)        # 文ごとの別々の発話の数
    g_sent = float(ns.sum() ** 2 / (ns ** 2).sum())                          # 文の実効の数（dai1_stats の G_eff と同じ式）
    df_g = max(1, int(np.floor(g_sent)) - 1)
    q11, qg = float(sst.t.ppf(0.975, n_sent - 1)), float(sst.t.ppf(0.975, df_g))
    ef_s["n_clusters"], ef_s["g_eff_sentence"], ef_s["df_geff"] = n_sent, g_sent, df_g
    ef_s["lo_geff"], ef_s["hi_geff"] = ef_s.est - qg * ef_s.se, ef_s.est + qg * ef_s.se
    sent_chk = dict(n=len(ef_s), est_dev_vs_speaker_max=dev(ef_s.est, ef_s.est_speaker), ci_formula_dev_max=max(dev(ef_s.lo, ef_s.est - q11 * ef_s.se), dev(ef_s.hi, ef_s.est + q11 * ef_s.se)),
                    methods=sorted(map(str, ef_s.method.unique())), dfs=sorted(set(ef_s.df.astype(float))), folds_dropped=sorted(set(ef_s.folds_dropped.fillna("").astype(str))),
                    n_clusters=n_sent, g_eff_sentence=g_sent, df_geff=df_g, n_per_sentence_distinct=dict(zip(SENT, map(int, ns))))
    if not (len(ef_s) == len(ef_all) and sent_chk["est_dev_vs_speaker_max"] <= TOL_EST and sent_chk["ci_formula_dev_max"] <= TOL_EST and sent_chk["methods"] == ["jk+seed"]
            and sent_chk["dfs"] == [float(n_sent - 1)] and sent_chk["folds_dropped"] == [""]):
        raise SystemExit(f"文を塊にした区間の確かめが合わない：{sent_chk}")
    print(f"3 済み（{time.time() - t0:.0f} 秒）", flush=True)

    # ---- 4. 文ごと・強さごとの内訳（記述．50 単位の数を足し合わせてから割る）
    bvers = ("prop", "prod", "stud_prop", "stud_prod", f"none_{fk}", "tcal")
    tot_ref = {(K, v): (sum(x[1].sum(0) for x in cells_all[(CD, K, v)].values()), sum(x[2].sum(0) for x in cells_all[(CD, K, v)].values())) for K in KS for v in bvers}
    brows = {}
    for by, groups in (("sen", SENT), ("inten", INTENS)):
        acc = breakdown(units, bvers, by, groups)
        rows = []
        for (K, v), (cor, tot) in acc.items():
            if not (np.array_equal(cor.sum(0), tot_ref[(K, v)][0]) and np.array_equal(tot.sum(0), tot_ref[(K, v)][1])):
                raise SystemExit(f"内訳の数の合計が，照合した全発話の数と違う：{(by, K, v)}")
            for j, g in enumerate(groups):
                with np.errstate(invalid="ignore", divide="ignore"):
                    rec = 100.0 * cor[j] / tot[j]
                if by == "sen":
                    if (tot[j] == 0).any():
                        raise SystemExit(f"文 {g} に感情が欠ける")
                    rows.append(dict(c=CD, sentence=g, K=K, version=v, role=E21.role_of(v), uar=float(rec.mean()), n=float(tot[j].sum()), n_per_seed=float(tot[j].sum()) * per_seed,
                                     **{f"recall_{i}": float(rec[i]) for i in range(4)}, **{f"n_{i}": float(tot[j, i]) for i in range(4)}, n_units=len(units)))
                else:
                    rows += [dict(c=CD, intensity=g, emotion=i, K=K, version=v, role=E21.role_of(v), recall=float(rec[i]), n=float(tot[j, i]), n_per_seed=float(tot[j, i]) * per_seed,
                                  n_units=len(units)) for i in range(4) if tot[j, i] > 0]
        brows[by] = pd.DataFrame(rows)
    by_sent, by_int = brows["sen"], brows["inten"]
    if len(by_sent) != len(SENT) * len(KS) * len(bvers) or by_int.duplicated(["intensity", "emotion", "K", "version"]).any() \
            or len(by_int) != (4 + 3 * 3) * len(KS) * len(bvers):                # 強さ × 感情の組：XX は 4 感情，LO・MD・HI は Happy・Sad・Angry
        raise SystemExit(f"内訳の行の数が違う：文 {len(by_sent)}，強さ {len(by_int)}")

    # ---- 5. 仮定の表
    ck = cs[cs.version == "_check"]
    acols = [x for x in ck.columns if x.startswith(ASSUME_PREFIX) or x in ASSUME_EXACT]
    if (ck.groupby("K").size() != len(units)).any():
        raise SystemExit("_check の行の数が K ごとに単位の数と違う")
    cap = math.sqrt(2.0) * float(sst.norm.ppf(P16.AUC_CLIP[1])) - 1e-9           # d′ の上限（dai20_eval.stage_summary と同じ式）
    ass = pd.DataFrame([dict(c=CD, K=K, n_units=len(units), check_row_assumption_columns=(",".join(acols) if acols else "無い"),
                             **{x: float(ck[ck.K == K][x].mean()) for x in acols},
                             dk_ceiling_share=float(np.mean([(u["DK"][K] >= cap).mean() for u in units])), dk_zero_share=float(np.mean([(u["DK"][K] <= 0).mean() for u in units])))
                        for K in KS])

    # ---- 6. 書き出し（ここまでの確かめが全部通ったときだけ）
    out.mkdir(parents=True, exist_ok=True)
    files = {"cremad_no_ieo_levels.csv": lv_no, "cremad_no_ieo_effects.csv": ef_no, "cremad_no_ieo_removed.csv": rem, "cremad_sentence_cluster_effects.csv": ef_s,
             "cremad_by_sentence.csv": by_sent, "cremad_by_intensity.csv": by_int, "cremad_assumptions.csv": ass}
    for n_, d in files.items():
        d.to_csv(out / n_, index=False)
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    me = Path(__file__).resolve()
    watch = list(E21.WATCH) + ["DAI/docs/tools_pending"]
    mods = tuple(E21.SCRIPTS) + ("dai21_eval5.py",)
    sums = {q.name: dai15.sha(q) for q in (sdir / "levels.csv", sdir / "effects.csv", sdir / "convergence.json") if q.exists()}
    npzs = sorted(udir.glob("*.npz"))
    json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv, commit=git("rev-parse", "HEAD"),
                   dirty_tracked=bool(git("status", "--porcelain", "--untracked-files=no", "--", *watch)), dirty_paths=watch,       # 追跡しているファイルだけを見る
                   script=str(me), script_sha256=dai15.sha(me), script_committed_clean=bool(dai1.committed_clean(me)),
                   modules_sha256={n_: dai15.sha(ROOT / "DAI" / "exp" / n_) for n_ in mods}, frozen_sha256=dai15.sha(E21.FROZEN), frozen20_sha256=dai15.sha(E21.FROZEN20),
                   design_sha256=dai15.sha(E21.DESIGN), summary_dir=str(sdir), summary_sha256=sums,
                   dai21_provenance_sha256=(dai15.sha(udir.parent / "provenance.json") if (udir.parent / "provenance.json").exists() else ""),
                   units_dir=str(udir), n_units=len(units), n_folds=len(folds), unit_commits=ucom,
                   units_npz_digest=f"{len(npzs)}:" + hashlib.sha256("".join(f"{q.name} {dai15.sha(q)}\n" for q in npzs).encode()).hexdigest(),
                   allow_partial=args.allow_partial, host=platform.node(), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                   consts=dict(KS=KS, PART=PART, DROP=DROP, SENT=SENT, N_CON=N_CON, N_SPK=N_SPK, TOL_EST=TOL_EST, TOL_CI=TOL_CI, N_BOOT=ST.N_BOOT, MIN_TEST=ST.MIN_TEST,
                               SEED_SETS={k_: v for k_, v in E21.SEED_SETS.items()}, breakdown_versions=bvers),
                   cremad_points={str(K): {k_: v for k_, v in fz["cremad"][str(K)].items() if k_ in ("prop_grid", "noprof_sel", "student_main", "student_prop_sel")} for K in KS},
                   n_versions=n_ver, n_pairs=len(con), g_eff_speaker=geff, n_speaker_clusters=n_spk,
                   check_vs_summary=(chk if strict else "省いた（--allow-partial）"), sentence_cluster_check=sent_chk, removed=removed,
                   assumptions=dict(check_row_columns=(acols if acols else "無い"), rule_prefix=ASSUME_PREFIX, rule_exact=ASSUME_EXACT, check_row_columns_present=sorted(ck.dropna(axis=1, how="all").columns),
                                    from_npz=["dk_ceiling_share", "dk_zero_share"], dk_cap=cap),
                   outputs={n_: len(d) for n_, d in files.items()},
                   notes={"cremad_by_sentence.csv": "50 単位（分割 5 × K 本の引き方 10）の正解の数と発話の数を足し合わせてから割った値．区間は無い．n は 50 単位の合計，n_per_seed は別々の発話の数",
                          "cremad_by_intensity.csv": "同上．強さ LO・MD・HI は文 IEO の Happy・Sad・Angry だけにあり，Neutral は XX だけ．XX には，ほかの 11 文の全部と文 IEO の Neutral が入る",
                          "cremad_no_ieo_*.csv": "区間は話者の塊の引き直し（集計の段と同じ乱数の鍵）．same_as・same_pred は文 IEO を除いたテストの上の印",
                          "cremad_sentence_cluster_effects.csv": "lo・hi は自由度 11（塊の数 12）．lo_geff・hi_geff は自由度を文の実効の数から決めた区間．話者の揺れを含まない"},
                   seconds=time.time() - t0),
              open(out / "cremad_sentence_provenance.json", "w"), indent=1, ensure_ascii=False)
    print(f"単位 {len(units)}，版 {n_ver}，差の対 {len(con)}．除いた発話（別々の発話．感情 0〜3）{removed['distinct']}．文を塊にした区間：点推定のずれの最大 {sent_chk['est_dev_vs_speaker_max']:.2e}，"
          f"文の実効の数 {g_sent:.2f}．仮定の量の列（_check の行）：{acols if acols else '無い'}．出力の行の数 {({n_: len(d) for n_, d in files.items()})}．{time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
