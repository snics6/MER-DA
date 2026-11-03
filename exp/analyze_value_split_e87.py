"""E87：疑似ラベルのモデルの価値（ターゲットのラベル何本ぶん）を，向きの分と位置の分に分ける（第 3 版．関門 1 の第 2 版の指摘で直した）．

設計：docs/exploratory-analyses.md の E87（規則は値を見る前に書いた）．評価のターゲットの値で判定するので本番（関門 1 は 5 体）．

  物差し：exp/relearn_tgtonly_e87.py が世界の発話の集合と E77 の分け方で学び直したターゲットだけの分類器（K = 1, 5, 10, 25, 50, 全件．選んだ C）．
  座標：S_uar（UAR），S_dir（6 対の A(d, 0) の平均），S_pos（6 対の o の 2 乗平均の平方根）．副：S_posloss．点だけ：S_dir_auc，S_dir_bfix，S_pos_bfix．
  読み：反復で平均した曲線に，反復で平均した none と手法の値を当てて本数 K' を読み，n_x = K'_x(手法) − K'_x(none)．
       読めない読み：上下の打ち切り（向きを別に記録），整える前の曲線が値を 2 回以上横切る（あいまい）．読めるかは主の右端（n_tgt/4）で決める．
  単位：（比較，世界，配備する分類器，K，コーパス）．抽出の種は（種，コーパス）だけで，世界と比較相手で共有する．
  コーパスの値：世界の平均（複製も平均）．区間は百分位と，G_eff < 20 なら jackknife の t 区間との和集合．
  族：読めるコーパスの中央値．複製は，G_eff < 20 のコーパスの複製を jackknife の幅まで広げてから中央値．抽出の中の打ち切りが 2.5% を超える区間は使わない．
  判定：予言ごとに当たり・外れ・判定できない（P87-2 は負も）．読めるコーパスの版と全コーパスの版（打ち切りは端に置く）の両方で同じ側に出たときだけ当たり・外れにする．

使い方：./.venv/bin/python exp/analyze_value_split_e87.py [--out results/e87] [--curve-root results/e87_tgtonly] [--boot 1000] [--runs ...] [--targets ...] [--smoke]
  試走（--smoke，--targets，--runs）の出力は results/ の下に書けない．本番は既定の引数で 1 回で流す．CSV の先頭の # の行は JSON（コミットと設定）．pd.read_csv(..., comment="#") で読む．
"""
import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.optimize import minimize
from scipy.special import logsumexp

warnings.filterwarnings("ignore", message="Mean of empty slice")                  # 抽出でテストの話者が全員抜けた反復は NaN になり，rep_mean が重みから外す（意図した扱い）

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import analyze_pairs_e83 as E  # noqa: E402  （d・o，行の対応，選んだ鍵，jackknife の区間は E83 と同じ関数）
from explore_pairs import ba  # noqa: E402
from explore_synergy import PAIRS  # noqa: E402
from multimodal_distill import WORLDS  # noqa: E402

RUNS = ("fuse-audio-video-text", "fuse-audio-video")
KS = ("0", "5", "25")
REPS = tuple(range(10))
K_GRID = ("1", "5", "10", "25", "50", "all")                                   # 物差しの主の実行の格子（results/e87_tgtonly）
K_EXTRA = ("100",)                                                             # 足した点（results/e87_tgtonly_k100．K = 50 と全件の間の補間を短くする）
K_ALL = ("1", "5", "10", "25", "50", "100", "all")                             # 解析の曲線の並び（--without-k100 なら K_GRID）
C_GRID = (0.01, 0.1, 1.0, 10.0)
KALL_RULES = ("mean", "min")                                                   # 右端の置き方：n_tgt/4（主）と，学習プールで最も少ない感情の本数（感度）
MAIN_RULE = "mean"
QUANTS_B = ("uar", "dir", "pos", "posloss")                                    # 複製を持つ座標（pos が位置の主，posloss は感度）
QUANTS_PT = ("dir_auc", "dir_bfix", "pos_bfix")                                # 点の値だけの座標（模型に依らない感度）
DIFFS = ("dir_minus_pos", "dir_minus_posloss")                                 # 同じ単位の中の差（複製あり）
DS = ("uar", "dir", "pos")                                                     # 換算する前の差 ΔS（複製あり）
BETTER_HIGH = {"uar": True, "dir": True, "dir_auc": True, "dir_bfix": True, "pos": False, "posloss": False, "pos_bfix": False}
SELECTORS = (("oursx", "none-oursx"), ("kd", "none-kd"), ("abl_ownK", "none-abl_ownK"), ("groupB", "none-groupB"), ("abl_refb", "none-abl_refb"), ("abl_ref0", "none-abl_ref0"))
FIXED = (("saerens", "none-saerens"), ("massbal", "none-massbal"))             # 位置だけを直す標準の手（設定は 1 つ．S_dir は d だけで決まるので n_dir は定義から 0）
COMPARISONS = SELECTORS + FIXED
MAIN_COMP = "none-oursx"
DIRECT = (("none-oursx", "none-abl_ownK", "oursx-abl_ownK"), ("none-oursx", "none-kd", "oursx-kd"), ("none-oursx", "none-groupB", "oursx-groupB"),
          ("none-abl_refb", "none-abl_ownK", "abl_refb-abl_ownK"), ("none-abl_ref0", "none-abl_ownK", "abl_ref0-abl_ownK"))
GEFF_MIN = E.GEFF_MIN
HOLD_RATIO = 0.5                                                               # P87-1・P87-3 の比の基準（E66 の P2 と同じ）
POS_RES_FRAC = 0.5                                                             # P87-2 の帯：|med_pos(K)| の区間が ±0.5·K に入る
POS_BAND_FIXED = 2.5                                                           # P87-2 の帯の感度 (ii)：±2.5 本（K = 5 の帯を K = 25 にも当てる）
CENS_DRAW_MAX = 0.025                                                          # 抽出の中で打ち切り・あいまいになった割合がこれを超える区間は使わない
CENS_DRAW_MAX_SENS = 0.10                                                      # 同上の感度 (n)：族の判定だけ基準を 10% にする
NOT_READABLE_FRAC_563188d = 0.5                                                # 直す前の規則：打ち切りの反復が半数以上なら読めない
EXCL_MA = ("MOSEI", "AFEW")
NOMINAL = {"audio": 5, "video": 5, "text": 3, "fused:fuse-audio-video-text": 3, "fused:fuse-audio-video": 5}   # 族の名目のコーパス数（判定の下限の分母）
VARIANTS = ("readable", "all", "readable_excl_mosei_afew", "readable_impro_only")
SIGN_AGREE_LOW = 0.75                                                          # 仮定 3・4：符号の一致率がこれ未満なら別の座標の版を並べる
HEADER = {}
SCRIPT_DEPS = ("exp/analyze_value_split_e87.py", "exp/relearn_tgtonly_e87.py", "exp/analyze_pairs_e83.py", "exp/explore_pairs.py", "exp/explore_synergy.py",
               "exp/pair_table.py", "exp/multimodal_distill.py", "measure/data.py", "measure/split.py")


def write_csv(df, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# " + json.dumps(HEADER, ensure_ascii=False) + "\n")
        df.to_csv(fh, index=False)


def family_of(st):
    return {"audio_wavlm": "audio", "video": "video", "text_roberta": "text"}.get(st, f"fused:{st}")


def nominal_of(fam, variant="readable"):
    """族の名目のコーパス数．MOSEI・AFEW を除く版では，除いた後の名目の数（音声・顔 3，言葉 2，融合 AVT 2・AV 3）．"""
    n = NOMINAL[fam]
    if variant == "readable_excl_mosei_afew":
        n -= {"audio": 2, "video": 2, "text": 1, "fused:fuse-audio-video-text": 1, "fused:fuse-audio-video": 2}[fam]
    return n


def need_of(fam, variant="readable"):
    return max(2, int(np.ceil(nominal_of(fam, variant) / 2)))


def pct(b, p):
    return float(np.nanpercentile(b, p)) if b is not None and np.isfinite(b).any() else np.nan


# ---------------------------------------------------------------- 座標

def present_uar(y, pred):
    return float(np.nanmean([float((pred[y == e] == e).mean()) if (y == e).any() else np.nan for e in range(4)]))


def uar_bias_fixed(L, y):
    """E79 と同じ：向き（L の差）を変えずに，テストの感情を釣り合わせた交差エントロピーを最小にする 4 つの切片を足した UAR．"""
    n = len(y)
    w = 1.0 / np.bincount(y, minlength=4)[y]
    oh = np.eye(4)[y]

    def f(b3):
        b = np.r_[0.0, b3]
        Z = L + b
        lse = logsumexp(Z, 1)
        P = np.exp(Z - lse[:, None])
        return float(((lse - Z[np.arange(n), y]) * w).sum()), ((P - oh) * w[:, None]).sum(0)[1:]
    res = minimize(f, np.zeros(3), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return 100 * present_uar(y, (L + np.r_[0.0, res.x]).argmax(1)), bool(res.success)


def d_auc_pairs(L, y):
    out = np.full(6, np.nan)
    for p, (k, j) in enumerate(PAIRS):
        z = L[:, k] - L[:, j]
        zk, zj = z[y == k], z[y == j]
        if len(zk) < 2 or len(zj) < 2:
            continue
        ranks = stats.rankdata(np.concatenate([zk, zj]))
        auc = (ranks[:len(zk)].sum() - len(zk) * (len(zk) + 1) / 2) / (len(zk) * len(zj))
        out[p] = np.sqrt(2) * stats.norm.ppf(np.clip(auc, 1e-12, 1 - 1e-12))
    return out


def scores_point(L, y):
    d, o = E.d_o_pairs(L, y)
    a0 = ba(d, 0.0)
    uar = 100 * present_uar(y, L.argmax(1))
    bfix, ok = uar_bias_fixed(L, y)
    return {"uar": uar, "dir": 100 * float(np.nanmean(a0)), "pos": float(np.sqrt(np.nanmean(o ** 2))), "posloss": 100 * float(np.nanmean(a0 - ba(d, o))),
            "dir_auc": 100 * float(np.nanmean(ba(d_auc_pairs(L, y), 0.0))), "dir_bfix": bfix, "pos_bfix": bfix - uar, "bfix_ok": ok}, d, o


def feats_all(L, y, sidx, nls):
    F = E.pair_feats(L, y, sidx, nls)
    GT = sparse.csr_matrix((np.ones(len(y)), (sidx, np.arange(len(y)))), shape=(nls, len(y)))
    pred = L.argmax(1)
    U = np.zeros((nls, 4, 2))
    for e in range(4):
        m = (y == e).astype(np.float64)
        U[:, e, 0] = GT @ (m * (pred == e))
        U[:, e, 1] = GT @ m
    return F, U


def scores_w(F, U, W):
    """W：(B, nls)．戻り：座標ごとに (B,)．"""
    d, o = E.pair_d_o_from_feats(F, W)
    corr, tot = W @ U[:, :, 0], W @ U[:, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(tot > 0, corr / tot, np.nan)
        a0 = ba(d, 0.0)
        return {"uar": 100 * np.nanmean(rec, 1), "dir": 100 * np.nanmean(a0, 1), "pos": np.sqrt(np.nanmean(o ** 2, 1)), "posloss": 100 * np.nanmean(a0 - ba(d, o), 1)}


# ---------------------------------------------------------------- 曲線と読み

def read_K(curve, logk, v, better_high, adjust=True):
    """曲線（…, 6）に値 v（…）を当てて本数 K' を読む（log K について線形．平らな区間では届く最小の K）．
    戻り：(K', 下の打ち切り（v が K=1 の値より悪い．等号は含まない），上の打ち切り（全件の値より良い），あいまい（整える前の曲線が v を 2 回以上横切る）)．"""
    s = np.asarray(curve if better_high else -curve, dtype=np.float64)
    vv = np.asarray(v if better_high else -v, dtype=np.float64)
    raw = s
    if adjust:
        s = np.maximum.accumulate(s, axis=-1)
    logk = np.broadcast_to(np.asarray(logk, dtype=np.float64), s.shape)
    ge = s >= vv[..., None]
    any_ge = ge.any(-1)
    i0 = np.argmax(ge, -1)
    i = np.clip(i0, 1, s.shape[-1] - 1)
    s_prev, s_cur = np.take_along_axis(s, (i - 1)[..., None], -1)[..., 0], np.take_along_axis(s, i[..., None], -1)[..., 0]
    lk_prev, lk_cur = np.take_along_axis(logk, (i - 1)[..., None], -1)[..., 0], np.take_along_axis(logk, i[..., None], -1)[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(s_cur > s_prev, (vv - s_prev) / (s_cur - s_prev), 1.0)
    lk = lk_prev + np.clip(frac, 0, 1) * (lk_cur - lk_prev)
    lk = np.where(ge[..., 0], logk[..., 0], lk)
    lk = np.where(any_ge, lk, logk[..., -1])
    below = raw < vv[..., None]
    crossings = (below[..., 1:] != below[..., :-1]).sum(-1)
    bad = ~np.isfinite(vv) | ~np.isfinite(s).all(-1)
    K = np.where(bad, np.nan, np.exp(lk))
    return K, (vv < s[..., 0]) & ~bad, (~any_ge) & ~bad, (crossings >= 2) & ~bad


def load_one(root, run, st, T, yall, te, prov, kgrid):
    """1 つの物差しの実行（格子 kgrid）のテストの点数（K × 選んだ C）を E77 の _te の並びで返す．"""
    f = root / run / st / f"{T}.npz"
    fc = root / run / st / f"{T}.csv"
    if not f.exists():
        raise SystemExit(f"{run} {st} {T}：物差しの npz が無い（{f}）")
    E.note_input(prov, f)
    E.note_input(prov, fc, with_sha=True)
    z = np.load(f)
    meta = json.loads(str(z["_meta"]))
    prov.setdefault("yardstick_meta", {})[f"{root.name}|{run}|{st}|{T}"] = {k: meta.get(k) for k in ("commit", "code_hash", "code_dirty", "started")}
    st_ = meta["settings"]
    if list(st_["K"]) != list(kgrid) or [float(c) for c in st_["C_grid"]] != list(C_GRID) or list(st_["reps"]) != list(REPS):
        raise SystemExit(f"{run} {st} {T}：物差し（{root.name}）の K・C・反復の並びが台本の約束と違う")
    if not np.array_equal(np.asarray(z["y"]).astype(int), yall):
        raise SystemExit(f"{run} {st} {T}：物差し（{root.name}）の y が load_any のラベルと違う")
    csv = pd.read_csv(fc, comment="#", dtype={"K": str, "rep": int})
    tg = csv[csv.method == "tgt_only"]
    out = {}
    for r in REPS:
        idx = np.asarray(z[f"test_idx_r{r}"]).ravel()
        te77 = np.sort(np.asarray(te[r]).ravel())
        if not np.array_equal(np.sort(idx), te77):
            raise SystemExit(f"{run} {st} {T} r{r}：物差し（{root.name}）のテストの行が E77 と違う（curve_mismatch）")
        pos = {int(a): i for i, a in enumerate(idx)}
        order = np.array([pos[int(a)] for a in np.asarray(te[r]).ravel()])
        Cs = np.asarray(z[f"t_C_r{r}"], dtype=np.float64)
        Lk, cv_uar = [], []
        for iK, K in enumerate(kgrid):
            row = tg[(tg.K == K) & (tg.rep == r)]
            if len(row) != 1 or not np.isclose(float(row.C.iloc[0]), Cs[iK]):
                raise SystemExit(f"{run} {st} {T} r{r} K{K}：物差し（{root.name}）の csv の行か C が npz と合わない")
            iC = int(np.flatnonzero(np.isclose(np.array(C_GRID), Cs[iK]))[0])
            L = np.asarray(z[f"t_logit_r{r}"][iK, iC], dtype=np.float64)[order]
            if not np.isfinite(L).all():
                raise SystemExit(f"{run} {st} {T} r{r} K{K}：物差し（{root.name}）の点数に NaN/Inf")
            Lk.append(L)
            cv_uar.append({c: float(row[f"test_uar_C{c:g}"].iloc[0]) for c in C_GRID})
        rows_r = tg[tg.rep == r]
        out[r] = {"L": Lk, "C": list(Cs), "rule": [str(x) for x in z[f"t_rule_r{r}"]], "test_uar_allC": cv_uar, "labels": list(kgrid),
                  "n_all": int(rows_r[rows_r.K == "all"].n_tgt.iloc[0]) if (rows_r.K == "all").any() else None, "n_min": int(rows_r.n_pool_min_class.iloc[0])}
    return out


def load_curve(root, root_k100, run, st, T, yall, te, prov):
    """学び直した物差しの ① の曲線：主の実行（K = 1, 5, 10, 25, 50, 全件）に，K = 100 の実行があれば差し込む（並びは K_ALL）．"""
    main = load_one(root, run, st, T, yall, te, prov, K_GRID)
    if root_k100 is None:
        return main
    ext = load_one(root_k100, run, st, T, yall, te, prov, K_EXTRA)
    out = {}
    for r in REPS:
        m, e = main[r], ext[r]
        if e["n_min"] != m["n_min"]:
            raise SystemExit(f"{run} {st} {T} r{r}：K = 100 の実行の学習プールが主の実行と違う")
        i = list(K_GRID).index("all")
        out[r] = {"L": m["L"][:i] + e["L"] + m["L"][i:], "C": m["C"][:i] + e["C"] + m["C"][i:], "rule": m["rule"][:i] + e["rule"] + m["rule"][i:],
                  "test_uar_allC": m["test_uar_allC"][:i] + e["test_uar_allC"] + m["test_uar_allC"][i:], "labels": list(K_ALL), "n_all": m["n_all"], "n_min": m["n_min"]}
    return out


def rep_mean(arr_by_rep, A=None):
    """反復ごとの配列（R 個，同じ形）を，A（B, R）の重みで混ぜる（A が無ければ平均）．NaN は重みから外す．"""
    arr = np.stack(arr_by_rep, 0)                                              # (R, ...)
    if A is None:
        return np.nanmean(arr, 0)
    w = A.T.reshape((len(arr_by_rep), A.shape[0]) + (1,) * (arr.ndim - 2))    # (R, B, 1...)
    fin = np.isfinite(arr)
    return (np.where(fin, arr, 0) * w).sum(0) / np.where(fin, w, 0).sum(0)


def k_values(labels, n_all):
    return np.array([float(l) if l != "all" else float(n_all) for l in labels], dtype=np.float64)


def group_curve(cv, yte_by_rep, sidx_by_rep, ssidx_by_rep, C, A, Wj, need_jk, Wj_s):
    """1 群・1 配備する分類器の曲線：反復の平均（点），抽出の重みで混ぜた複製（B），jackknife（J，S），反復ごとの曲線（直す前の規則）．"""
    R = len(REPS)
    labels = cv[REPS[0]]["labels"]
    nK = len(labels)
    pt = {q: np.zeros((R, nK)) for q in QUANTS_B + QUANTS_PT}
    allc = np.zeros((R, nK, 4))
    B_, J_, S_ = ({q: [] for q in QUANTS_B} for _ in range(3))
    for ri, r in enumerate(REPS):
        e = cv[r]
        bB = {q: np.zeros((len(C), nK)) for q in QUANTS_B}
        bJ = {q: np.zeros((Wj.shape[0], nK)) for q in QUANTS_B} if need_jk else None
        bS = {q: np.zeros((Wj_s.shape[0], nK)) for q in QUANTS_B} if Wj_s is not None else None
        for iK in range(nK):
            sc, _, _ = scores_point(e["L"][iK], yte_by_rep[r])
            for q in QUANTS_B + QUANTS_PT:
                pt[q][ri, iK] = sc[q]
            allc[ri, iK] = [e["test_uar_allC"][iK][c] for c in C_GRID]
            F, U = feats_all(e["L"][iK], yte_by_rep[r], sidx_by_rep[r], C.shape[1])
            s_ = scores_w(F, U, C)
            for q in QUANTS_B:
                bB[q][:, iK] = s_[q]
            if need_jk:
                s_ = scores_w(F, U, Wj)
                for q in QUANTS_B:
                    bJ[q][:, iK] = s_[q]
            if Wj_s is not None:
                Fs, Us = feats_all(e["L"][iK], yte_by_rep[r], ssidx_by_rep[r], Wj_s.shape[0])
                s_ = scores_w(Fs, Us, Wj_s)
                for q in QUANTS_B:
                    bS[q][:, iK] = s_[q]
        for q in QUANTS_B:
            B_[q].append(bB[q])
            if need_jk:
                J_[q].append(bJ[q])
            if Wj_s is not None:
                S_[q].append(bS[q])
    n_all_rep = np.array([cv[r]["n_all"] for r in REPS], dtype=np.float64)
    kv = {"mean": k_values(labels, n_all_rep.mean() / 4), "min": k_values(labels, np.mean([cv[r]["n_min"] for r in REPS]))}
    for rule, k in kv.items():
        if not (np.diff(k) > 0).all():
            raise SystemExit(f"右端の本数（{rule}）が左の点以下：{k}")
    return {"pt_rep": pt, "pt": {q: pt[q].mean(0) for q in pt}, "allc_rep": allc, "kv": kv, "logk": {r_: np.log(k) for r_, k in kv.items()}, "Aw": A, "labels": labels,
            "logk_rep": [np.log(k_values(labels, n / 4)) for n in n_all_rep],
            "B": {q: rep_mean(B_[q], A) for q in QUANTS_B}, "J": {q: rep_mean(J_[q]) for q in QUANTS_B} if need_jk else None,
            "S": {q: rep_mean(S_[q]) for q in QUANTS_B} if Wj_s is not None else None}


def corpus_draws(corpus, speakers, B, seed, n_reps):
    """抽出の種は（種，コーパス）だけ（同じコーパスの世界と比較相手で抽出を共有する）．"""
    rng = np.random.default_rng([seed, zlib.crc32(corpus.encode())])
    nS = len(speakers)
    C = rng.multinomial(nS, np.full(nS, 1 / nS), size=B).astype(np.float64)
    A = rng.multinomial(n_reps, np.full(n_reps, 1 / n_reps), size=B).astype(np.float64)
    return C, A


def fixed_setting(tab, st, K, method):
    """chosen.csv に無い固定の比較相手（saerens・massbal）：行の表から (wave, method, pnorm, meff) を引く．設定が 1 つでなければ "missing"．"""
    g = tab[(tab.student == st) & (tab.K == K) & (tab.method == method)][["wave", "method", "pnorm", "meff"]].drop_duplicates()
    if len(g) != 1:
        return "missing"
    r = g.iloc[0]
    return (r.wave, r.method, str(r.pnorm), str(r.meff))


# ---------------------------------------------------------------- 本体

def run_all(args, out, prov):
    sel_path = ROOT / "results" / "e79" / "select" / "chosen.csv"
    E.note_input(prov, sel_path, with_sha=True)
    sel = pd.read_csv(sel_path, comment="#", dtype={"K": str, "param": str, "meff": str})
    curve_rows, curve_rep_rows, val_rows, unit_rows, check_rows, seed_rows = [], [], [], [], [], []
    boot_store, jk_store = {}, {}
    for run in (args.runs or RUNS):
        d77 = ROOT / "results" / f"e77_{run}"
        pf = ROOT / "results" / "e79" / "measure" / f"pairs_{run}.csv.gz"
        rf = ROOT / "results" / "e79" / "measure" / f"rows_{run}.csv.gz"
        E.note_input(prov, pf)
        E.note_input(prov, rf)
        P = pd.read_csv(pf, dtype={"row_id": int, "rep": int, "pair": int})
        RW = pd.read_csv(rf, dtype={"row_id": int, "rep": int, "K": str}, usecols=["target", "rep", "row_id", "wave", "uar_bfix"])
        targets = [t for t in sorted(P.target.unique()) if (not args.targets or t in args.targets)]
        students = WORLDS[run] + [run]
        by_corpus = {}
        for T in targets:
            by_corpus.setdefault(E.corp(T), []).append(T)
        for cp, Ts in by_corpus.items():
            prep = {}
            for T in Ts:
                tab = E.row_table(run, T, prov)
                X = E.dedupe_pairs(P[P.target == T], f"{run} {T}").merge(tab, on=["wave", "rep", "row_id", "student"], how="left")
                if X.method.isna().any():
                    raise SystemExit(f"{run} {T}：pairs の行が <T>.csv に無い")
                bf = RW[RW.target == T].set_index(["wave", "rep", "row_id"]).uar_bfix.to_dict()
                yall, spk_all = E.load_labels_speakers(run, T, d77, prov)
                te = {r: np.load(d77 / f"{T}_r{r}_te.npy") for r in REPS}
                for r in REPS:
                    E.note_input(prov, d77 / f"{T}_r{r}_te.npy")
                sets = {}
                for r in REPS:
                    sets.setdefault(te[r].tobytes(), []).append(r)
                prov.setdefault("distinct_test_sets", {})[f"{run}|{T}"] = len(sets)
                logp = {}
                for wave in ("e77", "e78", "e80c"):
                    dd = ROOT / "results" / f"{wave}_{run}"
                    if (dd / f"{T}_r0_logp.npy").exists():
                        logp[wave] = {r: (np.load(dd / f"{T}_r{r}_logp.npy", mmap_mode="r"), np.load(dd / f"{T}_r{r}_rowids.npy")) for r in REPS}
                        for r in REPS:
                            E.note_input(prov, dd / f"{T}_r{r}_logp.npy")
                            E.note_input(prov, dd / f"{T}_r{r}_rowids.npy")
                            if not np.array_equal(np.load(dd / f"{T}_r{r}_te.npy"), te[r]):
                                raise SystemExit(f"{run} {T} r{r}：{wave} の _te が E77 と違う")
                curves_raw = {st: load_curve(args.curve_root, None if args.without_k100 else args.curve_root_k100, run, st, T, yall, te, prov) for st in students}
                prep[T] = {"X": X, "tab": tab, "bf": bf, "y": yall, "spk": spk_all, "te": te, "logp": logp, "curves_raw": curves_raw}
            speakers = sorted(set().union(*[set(p["spk"][p["te"][r]]) for p in prep.values() for r in REPS]))
            sid = {s: i for i, s in enumerate(speakers)}
            nls = len(speakers)
            C, A = corpus_draws(cp, speakers, args.boot, args.seed, len(REPS))
            prov.setdefault("speakers_crc32", {})[f"{run}|{cp}"] = zlib.crc32("\n".join(speakers).encode())   # 世界の間で同じ並びか（抽出の共有の前提）を後で確かめられるように
            Wj = np.ones((nls, nls)) - np.eye(nls)
            sess, Wj_s = None, None
            if cp == "IEMOCAP":
                sessions = sorted({E.session_of(s) for s in speakers})
                sess = {s: sessions.index(E.session_of(s)) for s in speakers}
                Wj_s = np.ones((len(sessions), len(sessions))) - np.eye(len(sessions))
                prov.setdefault("sessions", {})[f"{run}|{cp}"] = sessions
            geff = {}
            for T, p in prep.items():
                cnt = pd.Series(np.concatenate([p["spk"][p["te"][r]] for r in REPS])).value_counts().to_numpy().astype(float)
                geff[T] = cnt.sum() ** 2 / (cnt ** 2).sum()
            prov.setdefault("g_eff", {}).update({f"{run}|{T}": float(v) for T, v in geff.items()})
            need_jk = min(geff.values()) < GEFF_MIN
            n_sess = len(set(sess.values())) if sess else 0
            # ---- 曲線（配備する分類器 × 群）
            curves = {}
            for T, p in prep.items():
                yte = {r: p["y"][p["te"][r]] for r in REPS}
                sidx = {r: np.array([sid[s] for s in p["spk"][p["te"][r]]]) for r in REPS}
                ssidx = {r: np.array([sess[s] for s in p["spk"][p["te"][r]]]) for r in REPS} if sess is not None else None
                p["yte"], p["sidx"], p["ssidx"] = yte, sidx, ssidx
                for st in students:
                    cv = group_curve(p["curves_raw"][st], yte, sidx, ssidx, C, A, Wj, need_jk, Wj_s)
                    curves[(T, st)] = cv
                    labels = cv["labels"]
                    for q in QUANTS_B + QUANTS_PT:
                        raw = cv["pt"][q]
                        s = raw if BETTER_HIGH[q] else -raw
                        adj = np.maximum.accumulate(s)
                        adj = adj if BETTER_HIGH[q] else -adj
                        for rule in KALL_RULES:
                            for iK in range(len(labels)):
                                curve_rows.append({"run": run, "target": T, "student": st, "quantity": q, "kall_rule": rule, "K_label": labels[iK], "K_value": cv["kv"][rule][iK],
                                                   "S_raw": raw[iK], "S_adj": adj[iK], "adjusted": bool(abs(adj[iK] - raw[iK]) > 1e-12)})
                    for ri, r in enumerate(REPS):
                        raw_r = p["curves_raw"][st][r]
                        for iK in range(len(labels)):
                            curve_rep_rows.append({"run": run, "target": T, "student": st, "rep": r, "K_label": labels[iK], "C": raw_r["C"][iK], "C_rule": raw_r["rule"][iK],
                                                   "n_all": raw_r["n_all"], "n_min": raw_r["n_min"],
                                                   **{f"S_{q}": cv["pt_rep"][q][ri, iK] for q in QUANTS_B + QUANTS_PT}, **{f"uar_C{c:g}": cv["allc_rep"][ri, iK, j] for j, c in enumerate(C_GRID)}})
            # ---- none と手法
            for selector, comp in COMPARISONS:
                for st in students:
                    for K in KS:
                        if comp in dict(FIXED).values():
                            chs = {T: fixed_setting(prep[T]["tab"], st, K, selector) for T in Ts}
                        else:
                            chs = {T: E.chosen_setting(sel, run, T, st, K, selector) for T in Ts}
                        if any(c == "missing" for c in chs.values()):
                            expected = selector in E.PARTIAL_SELECTORS and E.PARTIAL_SELECTORS[selector] != E.role_of(run, st)
                            check_rows.append({"check": "chosen_missing_expected" if expected else "chosen_missing", "comparison": comp, "run": run, "corpus": cp, "student": st, "K": K})
                            continue
                        if len({c for c in chs.values()}) > 1:
                            check_rows.append({"check": "chosen_differs_in_corpus", "comparison": comp, "run": run, "corpus": cp, "student": st, "K": K,
                                               "chosen": ";".join(f"{T}:{'none' if c is None else '|'.join(c)}" for T, c in chs.items())})
                            if comp == MAIN_COMP:
                                raise SystemExit(f"{run} {cp} {st} K{K}：IEMOCAP の 2 群で {selector} の選択が違う（主の比較なので止める）")
                        ncands = {T: (E.n_cand_of(sel, run, T, st, K, selector) if selector in dict(SELECTORS) else 1) for T in Ts}
                        prov.setdefault("n_cand", {})[f"{comp}|{run}|{st}|{K}|{cp}"] = ncands
                        prov.setdefault("n_groups_none", {})[f"{comp}|{run}|{st}|{K}|{cp}"] = sum(c is None for c in chs.values())
                        per_T = {T: values_group(run, T, prep[T], curves[(T, st)], chs[T], st, K, comp, nls, C, Wj, need_jk, sess, Wj_s, val_rows, check_rows, seed_rows) for T in Ts}
                        all_none = all(c is None for c in chs.values())
                        chosen_short = "none" if all_none else "|".join(next(c for c in chs.values() if c is not None))
                        meta = {"chosen": chosen_short, "n_groups_none": sum(c is None for c in chs.values()), "n_cand_min": min(ncands.values()),
                                "n_seeds": max((per_T[T]["n_seeds"] for T in Ts if per_T[T] is not None), default=0)}
                        specs = [("main", Ts, "jk")]
                        if cp == "IEMOCAP":
                            specs.append(("iemocap_impro_only", [T for T in Ts if T != "IEMOCAP-script"], "jk"))
                        if sess is not None:
                            specs.append(("session_jk", Ts, "jk_sess"))
                        for stag, Ts_use, jkkind in specs:
                            if Ts_use:
                                aggregate_unit(run, cp, st, K, comp, stag, jkkind, Ts_use, per_T, geff, n_sess, all_none, meta, unit_rows, boot_store, jk_store)
    return (pd.DataFrame(curve_rows), pd.DataFrame(curve_rep_rows), pd.DataFrame(val_rows), pd.DataFrame(unit_rows), pd.DataFrame(check_rows), pd.DataFrame(seed_rows),
            boot_store, jk_store)


def method_scores(run, T, p, X, chT, st, K, comp, nls, C, Wj, need_jk, Wj_s, check_rows, seed_rows):
    """none か手法の，反復ごとの点数（種の平均）と，抽出の重みでの点数（B，J，S）．戻りは反復のリスト．種ごと・反復ごとの点数は seed_rows に残す．"""
    out = {"pt": [], "B": [], "J": [], "S": [], "seeds": set(), "chosen": "none" if chT is None else "|".join(chT), "n_bfix_fail": 0}
    for r in REPS:
        yte = p["yte"][r]
        if chT is None:
            xo_all = X[(X.rep == r) & (X.student == st) & (X.K == K) & (X.method == "none") & (X.param == "-")]
            seeds = [None]
        else:
            wave, meth, par, me = chT
            xo_all = X[(X.rep == r) & (X.student == st) & (X.K == K) & (X.wave == wave) & (X.method == meth) & (X.pnorm == par) & (X.meff == me)]
            seeds = sorted(xo_all.seed.unique())
        if not seeds or any(len(xo_all[(xo_all.seed == s_) if s_ is not None else slice(None)]) != 6 for s_ in seeds):
            check_rows.append({"check": "none_rows" if chT is None else "ours_rows", "comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "n": len(xo_all)})
            return None
        acc = {"pt": [], "B": [], "J": [], "S": []}
        for s_ in seeds:
            xo = (xo_all if s_ is None else xo_all[xo_all.seed == s_]).sort_values("pair")
            wave = "e77" if chT is None else chT[0]
            Lo = E.logp_row(p["logp"], wave, r, int(xo.row_id.iloc[0]))
            sc, do_, oo = scores_point(Lo, yte)
            mx, mism = E.max_abs_or_nan_mismatch(np.concatenate([do_, oo]), np.concatenate([xo.d.to_numpy(), xo.o.to_numpy()]))
            if mx > 1e-6 or mism:
                raise SystemExit(f"{run} {T} {st} K{K} r{r} {comp}：対数の確率から計算し直した d・o が pairs の表と合わない（最大 {mx:.2e}，NaN {mism}）")
            bfix_ref = p["bf"].get((wave, r, int(xo.row_id.iloc[0])))                   # rows_*.csv.gz の uar_bfix は 0〜100 の尺度
            if bfix_ref is not None and np.isfinite(bfix_ref) and abs(float(bfix_ref) - sc["dir_bfix"]) > 1e-6:
                raise SystemExit(f"{run} {T} {st} K{K} r{r} {comp}：計算し直した uar_bfix が rows の表と合わない（{sc['dir_bfix']:.6f} 対 {float(bfix_ref):.6f}）")
            if not sc["bfix_ok"]:
                out["n_bfix_fail"] += 1
            seed_rows.append({"comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "role": "none" if chT is None else "method",
                              "seed": -1 if s_ is None else int(s_), "wave": wave, "row_id": int(xo.row_id.iloc[0]), **{f"S_{q}": sc[q] for q in QUANTS_B + QUANTS_PT}})
            acc["pt"].append(sc)
            F, U = feats_all(Lo, yte, p["sidx"][r], nls)
            acc["B"].append(scores_w(F, U, C))
            if need_jk:
                acc["J"].append(scores_w(F, U, Wj))
            if Wj_s is not None:
                Fs, Us = feats_all(Lo, yte, p["ssidx"][r], Wj_s.shape[0])
                acc["S"].append(scores_w(Fs, Us, Wj_s))
            if s_ is not None:
                out["seeds"].add(int(s_))
        out["pt"].append({q: float(np.mean([a[q] for a in acc["pt"]])) for q in QUANTS_B + QUANTS_PT})
        out["B"].append({q: np.mean(np.stack([a[q] for a in acc["B"]]), 0) for q in QUANTS_B})
        if need_jk:
            out["J"].append({q: np.mean(np.stack([a[q] for a in acc["J"]]), 0) for q in QUANTS_B})
        if Wj_s is not None:
            out["S"].append({q: np.mean(np.stack([a[q] for a in acc["S"]]), 0) for q in QUANTS_B})
    return out


def values_group(run, T, p, cv, chT, st, K, comp, nls, C, Wj, need_jk, sess, Wj_s, val_rows, check_rows, seed_rows):
    """1 群：反復で平均した曲線に，反復で平均した none と手法の値を当てて読む．点，複製（B）と打ち切りの印，jackknife（J，S），反復ごとの読み（直す前の規則）．"""
    X = p["X"]
    res = {"n_seeds": 0, "pt": {}, "pt_nomono": {}, "flag": {}, "readable": {}, "B": {}, "mask": {}, "J": {}, "S": {}, "cens": {}, "rep": {}, "dS": {}, "dS_B": {}, "dS_J": {}, "dS_S": {}}
    N = method_scores(run, T, p, X, None, st, K, comp, nls, C, Wj, need_jk, Wj_s, check_rows, seed_rows)
    if N is None:
        return None
    Mth = N if chT is None else method_scores(run, T, p, X, chT, st, K, comp, nls, C, Wj, need_jk, Wj_s, check_rows, seed_rows)
    if Mth is None:
        return None
    is_none = chT is None
    res["n_seeds"] = len(Mth["seeds"])
    pt_n = {q: float(np.mean([a[q] for a in N["pt"]])) for q in QUANTS_B + QUANTS_PT}
    pt_m = {q: float(np.mean([a[q] for a in Mth["pt"]])) for q in QUANTS_B + QUANTS_PT}
    B_n = {q: rep_mean([a[q] for a in N["B"]], cv["Aw"]) for q in QUANTS_B}
    B_m = {q: rep_mean([a[q] for a in Mth["B"]], cv["Aw"]) for q in QUANTS_B}
    J_n = {q: rep_mean([a[q] for a in N["J"]]) for q in QUANTS_B} if need_jk else None
    J_m = {q: rep_mean([a[q] for a in Mth["J"]]) for q in QUANTS_B} if need_jk else None
    S_n = {q: rep_mean([a[q] for a in N["S"]]) for q in QUANTS_B} if Wj_s is not None else None
    S_m = {q: rep_mean([a[q] for a in Mth["S"]]) for q in QUANTS_B} if Wj_s is not None else None
    base = {"comparison": comp, "run": run, "target": T, "student": st, "K": K, "chosen": Mth["chosen"], "n_seeds": res["n_seeds"], "n_bfix_fail": N["n_bfix_fail"] + Mth["n_bfix_fail"]}
    row = dict(base)
    for q in QUANTS_B + QUANTS_PT:
        row[f"S_{q}_none"], row[f"S_{q}_method"] = pt_n[q], pt_m[q]
    for rule in KALL_RULES:
        logk = cv["logk"][rule]
        for q in QUANTS_B + QUANTS_PT:
            Kn, lo_n, hi_n, am_n = read_K(cv["pt"][q], logk, pt_n[q], BETTER_HIGH[q])
            Km, lo_m, hi_m, am_m = read_K(cv["pt"][q], logk, pt_m[q], BETTER_HIGH[q])
            Kn2 = read_K(cv["pt"][q], logk, pt_n[q], BETTER_HIGH[q], adjust=False)[0]
            Km2 = read_K(cv["pt"][q], logk, pt_m[q], BETTER_HIGH[q], adjust=False)[0]
            n = 0.0 if is_none else float(Km - Kn)
            res["pt"][(rule, q)] = n
            res["pt_nomono"][(rule, q)] = 0.0 if is_none else float(Km2 - Kn2)
            fl = {"cens_lo_none": bool(lo_n), "cens_hi_none": bool(hi_n), "amb_none": bool(am_n), "cens_lo_method": bool(lo_m), "cens_hi_method": bool(hi_m), "amb_method": bool(am_m)}
            res["flag"][(rule, q)] = fl
            res["readable"][(rule, q)] = True if is_none else not any(fl.values())      # none を選んだ群は定義から 0（読める）
            row.update({f"K_{q}_none_{rule}": float(Kn), f"K_{q}_method_{rule}": float(Km), f"n_{q}_{rule}": n, f"n_{q}_nomono_{rule}": res["pt_nomono"][(rule, q)],
                        f"readable_{q}_{rule}": res["readable"][(rule, q)], **{f"{k}_{q}_{rule}": v for k, v in fl.items()}})
            if q in QUANTS_B:
                if is_none:
                    res["B"][(rule, q)] = np.zeros(len(C))
                    res["mask"][(rule, q)] = np.zeros(len(C), bool)
                    res["cens"][(rule, q)] = {k: 0.0 for k in ("lo_none", "hi_none", "amb_none", "lo_method", "hi_method", "amb_method")}
                    if need_jk:
                        res["J"][(rule, q)] = np.zeros(nls)
                    if Wj_s is not None:
                        res["S"][(rule, q)] = np.zeros(Wj_s.shape[0])
                else:
                    KBn, lo1, hi1, a1 = read_K(cv["B"][q], logk, B_n[q], BETTER_HIGH[q])
                    KBm, lo2, hi2, a2 = read_K(cv["B"][q], logk, B_m[q], BETTER_HIGH[q])
                    res["B"][(rule, q)] = KBm - KBn
                    res["mask"][(rule, q)] = lo1 | hi1 | a1 | lo2 | hi2 | a2
                    res["cens"][(rule, q)] = {"lo_none": float(np.mean(lo1)), "hi_none": float(np.mean(hi1)), "amb_none": float(np.mean(a1)),
                                              "lo_method": float(np.mean(lo2)), "hi_method": float(np.mean(hi2)), "amb_method": float(np.mean(a2))}
                    if need_jk:
                        res["J"][(rule, q)] = read_K(cv["J"][q], logk, J_m[q], BETTER_HIGH[q])[0] - read_K(cv["J"][q], logk, J_n[q], BETTER_HIGH[q])[0]
                    if Wj_s is not None:
                        res["S"][(rule, q)] = read_K(cv["S"][q], logk, S_m[q], BETTER_HIGH[q])[0] - read_K(cv["S"][q], logk, S_n[q], BETTER_HIGH[q])[0]
        # 向きの分 − 位置の分（同じ単位の中の差．位置は S_pos が主，S_posloss が感度 (a)）
        for pq in ("pos", "posloss"):
            dq = f"dir_minus_{pq}"
            res["pt"][(rule, dq)] = res["pt"][(rule, "dir")] - res["pt"][(rule, pq)]
            res["pt_nomono"][(rule, dq)] = res["pt_nomono"][(rule, "dir")] - res["pt_nomono"][(rule, pq)]
            res["readable"][(rule, dq)] = res["readable"][(rule, "dir")] and res["readable"][(rule, pq)]
            res["flag"][(rule, dq)] = {k: (res["flag"][(rule, "dir")][k] or res["flag"][(rule, pq)][k]) for k in res["flag"][(rule, "dir")]}
            res["B"][(rule, dq)] = res["B"][(rule, "dir")] - res["B"][(rule, pq)]
            res["mask"][(rule, dq)] = res["mask"][(rule, "dir")] | res["mask"][(rule, pq)]
            res["cens"][(rule, dq)] = {k: max(res["cens"][(rule, "dir")][k], res["cens"][(rule, pq)][k]) for k in res["cens"][(rule, "dir")]}
            if need_jk:
                res["J"][(rule, dq)] = res["J"][(rule, "dir")] - res["J"][(rule, pq)]
            if Wj_s is not None:
                res["S"][(rule, dq)] = res["S"][(rule, "dir")] - res["S"][(rule, pq)]
            row[f"n_{dq}_{rule}"] = res["pt"][(rule, dq)]
    # 反復ごとの読み（直す前の規則 563188d：反復ごとの 6 点の曲線（K = 100 を除く）に反復ごとの値を当て，右端は n_tgt_r/4）
    sub6 = np.array([i for i, l in enumerate(cv["labels"]) if l in K_GRID])
    for q in QUANTS_B + QUANTS_PT:
        ns, cens = [], []
        for ri, r in enumerate(REPS):
            crv, lk = cv["pt_rep"][q][ri][sub6], cv["logk_rep"][ri][sub6]
            Kn, lo_n, hi_n, _ = read_K(crv, lk, N["pt"][ri][q], BETTER_HIGH[q])
            Km, lo_m, hi_m, _ = read_K(crv, lk, Mth["pt"][ri][q], BETTER_HIGH[q])
            ns.append(0.0 if is_none else float(Km - Kn))
            cens.append(bool(lo_n or hi_n) or (not is_none and bool(lo_m or hi_m)))
        res["rep"][q] = {"n_mean": float(np.mean(ns)), "frac_cens": float(np.mean(cens))}
        row[f"n_{q}_rep"], row[f"frac_cens_{q}_rep"] = res["rep"][q]["n_mean"], res["rep"][q]["frac_cens"]
    for q in DS:                                                               # 換算する前の差（右端の置き方に依らない）
        res["dS"][q] = 0.0 if is_none else pt_m[q] - pt_n[q]
        res["dS_B"][q] = np.zeros(len(C)) if is_none else B_m[q] - B_n[q]
        if need_jk:
            res["dS_J"][q] = np.zeros(nls) if is_none else J_m[q] - J_n[q]
        if Wj_s is not None:
            res["dS_S"][q] = np.zeros(Wj_s.shape[0]) if is_none else S_m[q] - S_n[q]
        row[f"dS_{q}"] = res["dS"][q]
    val_rows.append(row)
    return res


def aggregate_unit(run, cp, st, K, comp, stag, jkkind, Ts_use, per_T, geff, n_sess, all_none, meta, unit_rows, boot_store, jk_store):
    if any(per_T[T] is None for T in Ts_use):
        return
    gmin = float(min(geff[T] for T in Ts_use))
    items = [(rule, q) for rule in KALL_RULES for q in QUANTS_B + QUANTS_PT + DIFFS] + [("-", f"dS_{q}") for q in DS]

    def stack_mean(key, qq):
        return np.mean(np.stack([per_T[T][key][qq] for T in Ts_use]), 0) if all(qq in per_T[T][key] for T in Ts_use) else None
    for rule, q in items:
        is_ds = q.startswith("dS_")
        qq = q[3:] if is_ds else q
        mask = None
        rep_n = rep_cens = np.nan
        if is_ds:
            x = float(np.mean([per_T[T]["dS"][qq] for T in Ts_use]))
            x_nm = np.nan
            readable = True
            flags = {}
            b, th, ths = stack_mean("dS_B", qq), stack_mean("dS_J", qq), stack_mean("dS_S", qq)
            cens = {}
        else:
            x = float(np.mean([per_T[T]["pt"][(rule, q)] for T in Ts_use]))
            x_nm = float(np.mean([per_T[T]["pt_nomono"][(rule, q)] for T in Ts_use]))
            readable = all(per_T[T]["readable"][(MAIN_RULE, q)] for T in Ts_use)                  # 読めるかは主の右端の置き方で決める
            flags = {k: any(per_T[T]["flag"][(rule, q)][k] for T in Ts_use) for k in per_T[Ts_use[0]]["flag"][(rule, q)]}
            has_b = q in QUANTS_B or q in DIFFS
            b = stack_mean("B", (rule, q)) if has_b else None
            th = stack_mean("J", (rule, q)) if has_b else None
            ths = stack_mean("S", (rule, q)) if has_b else None
            if has_b:
                mask = np.any(np.stack([per_T[T]["mask"][(rule, q)] for T in Ts_use]), 0)
            cens = {k: float(np.mean([per_T[T]["cens"][(rule, q)][k] for T in Ts_use])) for k in per_T[Ts_use[0]]["cens"][(rule, q)]} if has_b else {}
            if q in QUANTS_B + QUANTS_PT:
                rep_n = float(np.mean([per_T[T]["rep"][q]["n_mean"] for T in Ts_use]))
                rep_cens = float(np.mean([per_T[T]["rep"][q]["frac_cens"] for T in Ts_use]))
        lo = hi = lo_j = hi_j = lo_s = hi_s = np.nan
        rule_i, nvalid = "point_only", 0
        frac_mask = float(mask.mean()) if mask is not None else 0.0
        interval_ok = bool(frac_mask <= CENS_DRAW_MAX)
        if b is not None:
            lo, hi = pct(b, 2.5), pct(b, 97.5)
            nvalid = int(np.isfinite(b).sum())
            rule_i = "pct"
            if gmin < GEFF_MIN:
                if th is not None:
                    lo_j, hi_j = E.jk_interval(x, th, gmin)
                    lo, hi = np.fmin(lo, lo_j), np.fmax(hi, hi_j)
                    rule_i = "union_pct_jk"
                else:
                    rule_i, lo, hi = "pct_jk_missing", np.nan, np.nan
            if jkkind == "jk_sess" and rule_i != "pct_jk_missing":
                if ths is not None:
                    lo_s, hi_s = E.jk_interval(x, ths, float(n_sess))
                    lo, hi = np.fmin(lo, lo_s), np.fmax(hi, hi_s)
                    rule_i = "union_main_session_jk"
        key = f"{comp}|{run}|{st}|{K}|{cp}|{stag}|{rule}|{q}"
        if b is not None:
            boot_store[key] = b.astype(np.float32)
        if mask is not None:
            boot_store[key + "|mask"] = mask.astype(np.uint8)
        if th is not None:
            jk_store[key] = th.astype(np.float32)
        if all_none:
            jd = "same_by_definition"
        elif b is None:
            jd = "point_only"
        elif not (np.isfinite(x) and np.isfinite(lo) and np.isfinite(hi)):
            jd = "missing"
        elif not interval_ok:
            jd = "cens_interval"
        elif lo > 0:
            jd = "pos"
        elif hi < 0:
            jd = "neg"
        else:
            jd = "undet"
        unit_rows.append({"comparison": comp, "run": run, "role": E.role_of(run, st), "student": st, "family": family_of(st), "K": K, "corpus": cp, "n_groups": len(Ts_use),
                          "groups": ";".join(Ts_use), "sensitivity": stag, "kall_rule": rule, "quantity": q, "point": x, "point_nomono": x_nm, "lo": lo, "hi": hi,
                          "lo_jk": lo_j, "hi_jk": hi_j, "lo_session_jk": lo_s, "hi_session_jk": hi_s, "interval_rule": rule_i, "g_eff_min": gmin, "n_boot_valid": nvalid,
                          "is_readable": bool(readable), **{f"flag_{k}": v for k, v in flags.items()}, **{f"frac_cens_{k}": v for k, v in cens.items()},
                          "frac_draw_cens_any": frac_mask, "interval_ok": interval_ok, "point_rep": rep_n, "frac_cens_rep": rep_cens,
                          "judgment": jd, "boot_key": key, **meta})


# ---------------------------------------------------------------- 直接の差，コーパスの値，族，判定

def direct_units(U, boot_store, jk_store):
    rows, store, jstore = [], {}, {}
    keys = ["run", "student", "K", "corpus", "sensitivity", "kall_rule", "quantity"]
    for a, b_, name in DIRECT:
        Ua = U[(U.comparison == a)].set_index(keys)
        Ub = U[(U.comparison == b_)].set_index(keys)
        for idx in Ua.index.intersection(Ub.index):
            ra, rb = Ua.loc[idx], Ub.loc[idx]
            x = float(ra.point - rb.point)
            ba_, bb = boot_store.get(ra.boot_key), boot_store.get(rb.boot_key)
            lo = hi = lo_j = hi_j = np.nan
            jd, rule_i, frac_mask = "point_only", "point_only", 0.0
            key = f"{name}|{'|'.join(map(str, idx))}"
            if ba_ is not None and bb is not None:
                d = ba_.astype(np.float64) - bb.astype(np.float64)
                store[key] = d.astype(np.float32)
                ma, mb = boot_store.get(ra.boot_key + "|mask"), boot_store.get(rb.boot_key + "|mask")
                mask = ((ma if ma is not None else 0) | (mb if mb is not None else 0)).astype(np.uint8) if (ma is not None or mb is not None) else np.zeros(len(d), np.uint8)
                store[key + "|mask"] = mask
                frac_mask = float(mask.mean())
                lo, hi = pct(d, 2.5), pct(d, 97.5)
                rule_i = "pct"
                ja, jb = jk_store.get(ra.boot_key), jk_store.get(rb.boot_key)
                if ra.g_eff_min < GEFF_MIN and ja is not None and jb is not None:
                    th = ja.astype(np.float64) - jb.astype(np.float64)
                    jstore[key] = th.astype(np.float32)
                    lo_j, hi_j = E.jk_interval(x, th, float(ra.g_eff_min))
                    lo, hi = np.fmin(lo, lo_j), np.fmax(hi, hi_j)
                    rule_i = "union_pct_jk"
                elif ra.g_eff_min < GEFF_MIN:
                    rule_i, lo, hi = "pct_jk_missing", np.nan, np.nan
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    jd = "missing"
                elif frac_mask > CENS_DRAW_MAX:
                    jd = "cens_interval"
                else:
                    jd = "pos" if lo > 0 else ("neg" if hi < 0 else "undet")
            run, st, K, cp, stag, rule, q = idx
            rows.append({"comparison": name, "run": run, "role": E.role_of(run, st), "student": st, "family": family_of(st), "K": K, "corpus": cp, "n_groups": int(ra.n_groups), "groups": ra.groups,
                         "sensitivity": stag, "kall_rule": rule, "quantity": q, "point": x, "point_nomono": float(ra.point_nomono - rb.point_nomono) if np.isfinite(ra.point_nomono) and np.isfinite(rb.point_nomono) else np.nan,
                         "lo": lo, "hi": hi, "lo_jk": lo_j, "hi_jk": hi_j, "interval_rule": rule_i, "g_eff_min": float(ra.g_eff_min), "is_readable": bool(ra.is_readable and rb.is_readable),
                         "frac_draw_cens_any": frac_mask, "interval_ok": bool(frac_mask <= CENS_DRAW_MAX), "judgment": jd, "boot_key": key})
    return pd.DataFrame(rows), store, jstore


def corpus_level(U, store, jstore):
    """コーパスの値：そのコーパスを含む世界の単位の平均（抽出を共有しているので複製も平均できる．jackknife も同じ話者なので平均できる）．
    戻り CORP[(comp, fam, K, rule, q, uv)][corpus] = {...}．uv は main と impro_only（AVT と AV の IEMOCAP の単位を即興だけの行に差し替える）．"""
    CORP, rows = {}, []
    U = U[U.sensitivity.isin(["main", "iemocap_impro_only"])]
    for (comp, fam, K, rule, q), g0 in U.groupby(["comparison", "family", "K", "kall_rule", "quantity"]):
        gm = g0[g0.sensitivity == "main"]
        per_uv = {"main": gm}
        gi = g0[g0.sensitivity == "iemocap_impro_only"]
        if len(gi):
            drop = gm.set_index(["run", "corpus"]).index.isin(gi.set_index(["run", "corpus"]).index)
            per_uv["impro_only"] = pd.concat([gm[~drop], gi])
        for uv, g in per_uv.items():
            cd = {}
            for cp, gc in g.groupby("corpus"):
                bs = [store[k] for k in gc.boot_key if k in store]
                ms = [store[k + "|mask"] for k in gc.boot_key if (k + "|mask") in store]
                js = [jstore[k] for k in gc.boot_key if k in jstore]
                x = float(gc.point.mean())
                cb = np.mean(np.stack(bs), 0).astype(np.float64) if len(bs) == len(gc) else None
                cm = np.any(np.stack(ms), 0) if len(ms) == len(gc) and ms else (np.zeros(len(cb), bool) if cb is not None else None)
                geff = float(gc.g_eff_min.min())
                lo, hi = (pct(cb, 2.5), pct(cb, 97.5)) if cb is not None else (np.nan, np.nan)
                lo_j = hi_j = np.nan
                rule_i = "pct" if cb is not None else "point_only"
                if cb is not None and geff < GEFF_MIN:
                    if len(js) == len(gc) and js:
                        th = np.mean(np.stack(js), 0).astype(np.float64)
                        lo_j, hi_j = E.jk_interval(x, th, geff)
                        lo, hi = float(np.fmin(lo, lo_j)), float(np.fmax(hi, hi_j))
                        rule_i = "union_pct_jk"
                    else:
                        rule_i, lo, hi = "pct_jk_missing", np.nan, np.nan
                frac_mask = float(cm.mean()) if cm is not None else 0.0
                ok = bool(frac_mask <= CENS_DRAW_MAX) and bool(gc.interval_ok.all()) if "interval_ok" in gc.columns else bool(frac_mask <= CENS_DRAW_MAX)
                if (gc.judgment == "same_by_definition").all():
                    jd = "same_by_definition"
                elif cb is None:
                    jd = "point_only"
                elif not (np.isfinite(lo) and np.isfinite(hi)):
                    jd = "missing"
                elif not ok:
                    jd = "cens_interval"
                else:
                    jd = "pos" if lo > 0 else ("neg" if hi < 0 else "undet")
                fl = {k: bool(gc[f"flag_{k}"].any()) for k in ("cens_lo_none", "cens_hi_none", "amb_none", "cens_lo_method", "cens_hi_method", "amb_method") if f"flag_{k}" in gc.columns}
                cd[cp] = {"point": x, "point_nomono": float(gc.point_nomono.mean()) if "point_nomono" in gc.columns else np.nan, "B": cb, "mask": cm, "geff": geff,
                          "lo": lo, "hi": hi, "lo_jk": lo_j, "hi_jk": hi_j, "ok": ok, "frac_mask": frac_mask, "judgment": jd, "readable": bool(gc.is_readable.all()),
                          "n_units": len(gc), "flags": fl, "point_rep": float(gc.point_rep.mean()) if "point_rep" in gc.columns else np.nan}
                if uv == "main":
                    rows.append({"comparison": comp, "family": fam, "K": K, "kall_rule": rule, "quantity": q, "corpus": cp, "n_units": len(gc), "runs": ";".join(sorted(gc.run)),
                                 "point": x, "point_nomono": cd[cp]["point_nomono"], "lo": lo, "hi": hi, "lo_jk": lo_j, "hi_jk": hi_j, "interval_rule": rule_i, "g_eff_min": geff,
                                 "frac_draw_cens_any": frac_mask, "interval_ok": ok, "judgment": jd, "is_readable": cd[cp]["readable"], **{f"flag_{k}": v for k, v in fl.items()},
                                 "n_unit_pos": int((gc.judgment == "pos").sum()), "n_unit_neg": int((gc.judgment == "neg").sum())})
            CORP[(comp, fam, K, rule, q, uv)] = cd
    return CORP, pd.DataFrame(rows)


def family_stats(cd, cps, inflate=True, cens_max=CENS_DRAW_MAX):
    """族の統計量：cps の中央値．複製は，G_eff < 20 のコーパスの複製を点の値のまわりで jackknife の幅まで広げてから，複製ごとに中央値．
    打ち切り・あいまいの複製がどれかのコーパスにある割合が cens_max を超えれば区間は使わない（ok = False）．コーパスの ok は単位の規則（2.5%）のまま．"""
    out = {"n_used": len(cps), "corpora": ";".join(cps), "median": np.nan, "median_nomono": np.nan, "lo": np.nan, "hi": np.nan, "ok": False, "frac_mask": np.nan, "b": None,
           "n_pos": int(sum(cd[c]["judgment"] == "pos" for c in cps)), "n_neg": int(sum(cd[c]["judgment"] == "neg" for c in cps)), "inflation": ""}
    if not cps:
        return out
    out["median"] = float(np.median([cd[c]["point"] for c in cps]))
    out["median_nomono"] = float(np.median([cd[c]["point_nomono"] for c in cps]))
    out["median_rep"] = float(np.median([cd[c]["point_rep"] for c in cps]))
    if len(cps) > 1:
        loo = {c: float(np.median([cd[d]["point"] for d in cps if d != c])) for c in cps}
        out["loo_medians"] = ";".join(f"{c}:{v:.3f}" for c, v in loo.items())
        out["loo_min"], out["loo_max"] = min(loo.values()), max(loo.values())
    if all(cd[c]["B"] is not None for c in cps):
        reps, infl = [], []
        for c in cps:
            b, x = cd[c]["B"], cd[c]["point"]
            r = 1.0
            if inflate and cd[c]["geff"] < GEFF_MIN and np.isfinite(cd[c]["lo_jk"]) and np.isfinite(cd[c]["hi_jk"]):
                pw = (pct(b, 97.5) - pct(b, 2.5)) / 2
                jw = (cd[c]["hi_jk"] - cd[c]["lo_jk"]) / 2
                r = max(1.0, jw / pw) if pw > 0 else 1.0
            infl.append(f"{c}:{r:.2f}")
            reps.append(x + r * (b - x))
        fb = np.nanmedian(np.stack(reps), 0)
        masks = [cd[c]["mask"] for c in cps if cd[c]["mask"] is not None]
        fm = np.any(np.stack(masks), 0) if masks else np.zeros(len(fb), bool)
        out.update({"b": fb, "mask": fm, "lo": pct(fb, 2.5), "hi": pct(fb, 97.5), "frac_mask": float(fm.mean()), "inflation": ";".join(infl),
                    "ok": bool(fm.mean() <= cens_max) and all(cd[c]["frac_mask"] <= cens_max for c in cps)})
    return out


def corpora_for(cd, variant):
    if variant == "all":
        return sorted(cd)
    if variant == "readable_excl_mosei_afew":
        return sorted(c for c in cd if cd[c]["readable"] and c not in EXCL_MA)
    return sorted(c for c in cd if cd[c]["readable"])


def family_table(CORP, fstore):
    rows = []
    for (comp, fam, K, rule, q, uv), cd in CORP.items():
        for variant in VARIANTS:
            if (variant == "readable_impro_only") != (uv == "impro_only"):
                continue
            cps = corpora_for(cd, variant)
            for inflate in (True, False):
                fs = family_stats(cd, cps, inflate)
                key = f"{comp}|{fam}|{K}|{rule}|{q}|{variant}|{'infl' if inflate else 'noinfl'}"
                if fs["b"] is not None:
                    fstore[key] = fs["b"].astype(np.float32)
                rows.append({"comparison": comp, "family": fam, "K": K, "kall_rule": rule, "quantity": q, "variant": variant, "jk_inflation": inflate, "n_corpora_nominal": nominal_of(fam, variant),
                             "n_corpora_present": len(cd), "n_corpora_used": fs["n_used"], "corpora_used": fs["corpora"], "judgeable": bool(fs["n_used"] >= need_of(fam, variant)),
                             "median": fs["median"], "median_nomono": fs["median_nomono"], "median_rep": fs.get("median_rep", np.nan), "lo": fs["lo"], "hi": fs["hi"],
                             "interval_ok": fs["ok"], "frac_draw_cens_any": fs["frac_mask"], "inflation": fs["inflation"], "n_corp_pos": fs["n_pos"], "n_corp_neg": fs["n_neg"],
                             "loo_medians": fs.get("loo_medians", ""), "loo_median_min": fs.get("loo_min", np.nan), "loo_median_max": fs.get("loo_max", np.nan),
                             "n_corp_cens_hi_method": int(sum(cd[c]["flags"].get("cens_hi_method", False) for c in cd)),
                             "n_corp_cens_lo_method": int(sum(cd[c]["flags"].get("cens_lo_method", False) for c in cd)),
                             "n_corp_cens_none": int(sum(cd[c]["flags"].get("cens_lo_none", False) or cd[c]["flags"].get("cens_hi_none", False) for c in cd)),
                             "n_corp_amb": int(sum(cd[c]["flags"].get("amb_none", False) or cd[c]["flags"].get("amb_method", False) for c in cd)),
                             "boot_key": key if fs["b"] is not None else ""})
    return pd.DataFrame(rows)


def status3(hold, fail):
    return "holds" if hold else ("fails" if fail else "undet")


def judge_one(CORP, comp, fam, rule, variant, dirq, posq, uarq, band_mode="frac", cens_max=CENS_DRAW_MAX):
    """1 つの（比較，族，右端の置き方，版，座標の組）について P87-1〜3 と対比の条件を判定する．戻りは列の辞書．"""
    uv = "impro_only" if variant == "readable_impro_only" else "main"
    need = need_of(fam, variant)
    pointonly = dirq in QUANTS_PT or posq in QUANTS_PT

    def FS(cd_, cps_):
        return family_stats(cd_, cps_, cens_max=cens_max)

    def cd_of(q, K):
        return CORP.get((comp, fam, K, rule, q, uv))

    def common(q, K1, K2):
        c1, c2 = cd_of(q, K1), cd_of(q, K2)
        if c1 is None or c2 is None:
            return None, None, []
        cps = sorted(set(corpora_for(c1, variant)) & set(corpora_for(c2, variant)))
        return c1, c2, cps
    rec = {"family": fam, "comparison": comp, "kall_rule": rule, "variant": variant, "coordinates": f"{dirq}/{posq}/{uarq}", "band": band_mode, "cens_max": cens_max}
    # ---- P87-1：向きの分は K によらず残る
    c0, c25, cps = common(dirq, "0", "25")
    rec["P87_1_n_corpora"] = len(cps)
    if c0 is None or len(cps) < need:
        rec["P87_1_status"] = "not_judgeable"
    else:
        f0, f25 = FS(c0, cps), FS(c25, cps)
        rec.update({"P87_1_med_K0": f0["median"], "P87_1_med_K25": f25["median"], "P87_1_corpora": ";".join(cps),
                    "P87_1_ratio": f25["median"] / f0["median"] if f0["median"] > 0 else np.nan})
        if pointonly or f0["b"] is None or f25["b"] is None:
            rec["P87_1_status"] = "point_only"
            rec["P87_1_point_holds"] = bool(f0["median"] > 0 and f25["median"] >= HOLD_RATIO * f0["median"])
        elif not (f0["ok"] and f25["ok"]):
            rec["P87_1_status"] = "not_judgeable_cens"
        else:
            diff = f25["b"] - HOLD_RATIO * f0["b"]
            lo0, hi0, lod, hid = pct(f0["b"], 2.5), pct(f0["b"], 97.5), pct(diff, 2.5), pct(diff, 97.5)
            rec.update({"P87_1_lo_K0": lo0, "P87_1_hi_K0": hi0, "P87_1_lo_diff": lod, "P87_1_hi_diff": hid,
                        "P87_1_status": status3(lo0 > 0 and lod >= 0, hi0 <= 0 or hid < 0)})
    # ---- P87-2：位置の分は K ≥ 5 で消える（帯 ±band(K) に区間が入る）
    st2, parts = [], {}
    for K_ in ("5", "25"):
        cK = cd_of(posq, K_)
        cps_K = corpora_for(cK, variant) if cK is not None else []
        parts[f"P87_2_n_corpora_K{K_}"] = len(cps_K)
        if cK is None or len(cps_K) < need:
            st2.append("not_judgeable")
            continue
        fK = FS(cK, cps_K)
        if band_mode == "frac":
            band = POS_RES_FRAC * float(K_)
        elif band_mode == "fixed":
            band = POS_BAND_FIXED
        else:                                                                  # "half_dir"：向きの分の中央値の半分（同じコーパスの集合）
            cdK = cd_of(dirq, K_)
            fd = FS(cdK, [c for c in cps_K if c in cdK]) if cdK is not None else None
            band = 0.5 * fd["median"] if fd is not None and np.isfinite(fd["median"]) and fd["median"] > 0 else np.nan
        parts.update({f"P87_2_med_pos_K{K_}": fK["median"], f"P87_2_band_K{K_}": band, f"P87_2_npos_K{K_}": fK["n_pos"], f"P87_2_corpora_K{K_}": ";".join(cps_K)})
        if pointonly or fK["b"] is None:
            parts[f"P87_2_point_K{K_}"] = bool(abs(fK["median"]) <= band) if np.isfinite(band) else None
            st2.append("point_only")
        elif not fK["ok"] or not np.isfinite(band):
            st2.append("not_judgeable_cens" if not fK["ok"] else "not_judgeable")
        else:
            lo, hi = pct(fK["b"], 2.5), pct(fK["b"], 97.5)
            parts.update({f"P87_2_lo_K{K_}": lo, f"P87_2_hi_K{K_}": hi})
            s = "holds" if (lo >= -band and hi <= band) else ("fails" if lo > band else ("neg" if hi < -band else "undet"))
            parts[f"P87_2_status_K{K_}"] = s
            st2.append(s)
    rec.update(parts)
    if "fails" in st2:
        rec["P87_2_status"] = "fails"
    elif "neg" in st2:
        rec["P87_2_status"] = "neg"
    elif all(s == "holds" for s in st2):
        rec["P87_2_status"] = "holds"
    elif all(s in ("holds", "point_only") for s in st2) and "point_only" in st2:
        rec["P87_2_status"] = "point_only"
    elif any(s.startswith("not_judgeable") for s in st2):
        rec["P87_2_status"] = "not_judgeable"
    else:
        rec["P87_2_status"] = "undet"
    c0p = cd_of(posq, "0")
    if c0p is not None:
        f0p = FS(c0p, corpora_for(c0p, variant))
        rec["P87_2_med_pos_K0"] = f0p["median"]
    # ---- 向きの分 − 位置の分（対比の条件 (C) に使う．P87-2 の中身ではない）
    dq = f"dir_minus_{posq}" if posq in ("pos", "posloss") and dirq == "dir" else None
    if dq is not None:
        cdm = cd_of(dq, "25")
        if cdm is not None:
            cps_d = corpora_for(cdm, variant)
            fdm = FS(cdm, cps_d)
            rec.update({"C_dir_minus_pos_med_K25": fdm["median"], "C_dir_minus_pos_lo_K25": pct(fdm["b"], 2.5) if fdm["b"] is not None else np.nan,
                        "C_dir_minus_pos_ok": bool(fdm["ok"] and len(cps_d) >= need)})
    # ---- P87-3：UAR で読んだ分は K とともに減る
    u0, u25, cps_u = common(uarq, "0", "25")
    rec["P87_3_n_corpora"] = len(cps_u)
    if u0 is None or len(cps_u) < need:
        rec["P87_3_status"] = "not_judgeable"
    else:
        f0, f25 = FS(u0, cps_u), FS(u25, cps_u)
        rec.update({"P87_3_med_K0": f0["median"], "P87_3_med_K25": f25["median"], "P87_3_corpora": ";".join(cps_u),
                    "P87_3_ratio": f25["median"] / f0["median"] if f0["median"] > 0 else np.nan})
        if f0["b"] is None or f25["b"] is None:
            rec["P87_3_status"] = "point_only"
            rec["P87_3_point_holds"] = bool(f0["median"] > 0 and f25["median"] < HOLD_RATIO * f0["median"])
        elif not (f0["ok"] and f25["ok"]):
            rec["P87_3_status"] = "not_judgeable_cens"
        else:
            diff = f25["b"] - HOLD_RATIO * f0["b"]
            lo0, hi0, lod, hid = pct(f0["b"], 2.5), pct(f0["b"], 97.5), pct(diff, 2.5), pct(diff, 97.5)
            rec.update({"P87_3_lo_K0": lo0, "P87_3_hi_K0": hi0, "P87_3_lo_diff": lod, "P87_3_hi_diff": hid,
                        "P87_3_status": status3(lo0 > 0 and hid < 0, hi0 <= 0 or lod >= 0)})
    # ---- 対比の条件 (A)(B)(C)：「向きは残り，位置は消える」と書けるか
    p0, p25, cps_p = common(posq, "0", "25")
    if p0 is not None and len(cps_p) >= need and not pointonly:
        f0, f25 = FS(p0, cps_p), FS(p25, cps_p)
        if f0["b"] is not None and f25["b"] is not None and f0["ok"] and f25["ok"]:
            condA = bool(pct(f0["b"], 2.5) > 0 and pct(f25["b"] - HOLD_RATIO * f0["b"], 97.5) < 0)
            rec["contrast_A_pos_present_then_drops"] = condA
    d25 = cd_of(dirq, "25")
    p25b = cd_of(posq, "25")
    if d25 is not None and p25b is not None:
        cps_d, cps_pp = corpora_for(d25, variant), corpora_for(p25b, variant)
        if len(cps_d) >= need and len(cps_pp) >= need:
            fd, fp = FS(d25, cps_d), FS(p25b, cps_pp)
            rec["contrast_B_counts"] = bool(fd["n_pos"] * 2 >= len(cps_d) and fp["n_pos"] * 2 < len(cps_pp))
    rec["contrast_C_dir_gt_pos"] = bool(rec.get("C_dir_minus_pos_ok", False) and np.isfinite(rec.get("C_dir_minus_pos_lo_K25", np.nan)) and rec.get("C_dir_minus_pos_lo_K25", np.nan) > 0)
    rec["contrast_allowed"] = bool(rec.get("P87_1_status") == "holds" and rec.get("P87_2_status") == "holds" and rec.get("contrast_A_pos_present_then_drops") is True
                                   and rec.get("contrast_B_counts") is True and rec["contrast_C_dir_gt_pos"])
    return rec


def judge(CORP, CORP_d):
    rows = []
    combos = {c[:2] for c in CORP}
    for fam, comp in sorted(set((c[1], c[0]) for c in CORP)):
        for rule in KALL_RULES:
            for variant in VARIANTS:
                if not any(k[0] == comp and k[1] == fam and k[3] == rule and (k[5] == ("impro_only" if variant == "readable_impro_only" else "main")) for k in CORP):
                    continue
                coord_sets = [("dir", "pos", "uar", "main")]
                if comp == MAIN_COMP and variant == "readable":
                    coord_sets += [("dir", "posloss", "uar", "posloss_as_position"), ("dir_bfix", "pos_bfix", "uar", "bfix_point"), ("dir_auc", "pos", "uar", "dir_auc_point")]
                for dirq, posq, uarq, vname in coord_sets:
                    is_main_set = (vname == "main" and comp == MAIN_COMP and variant in ("readable", "all"))
                    settings = [("frac", CENS_DRAW_MAX)]
                    if is_main_set:
                        settings += [("fixed", CENS_DRAW_MAX), ("half_dir", CENS_DRAW_MAX), ("frac", CENS_DRAW_MAX_SENS)]
                    for band, cmax in settings:
                        rec = judge_one(CORP, comp, fam, rule, variant, dirq, posq, uarq, band, cmax)
                        rec["coord_set"] = vname
                        rec["is_main_judgment"] = bool(fam == "audio" and comp == MAIN_COMP and rule == MAIN_RULE and variant == "readable" and vname == "main" and band == "frac" and cmax == CENS_DRAW_MAX)
                        # 教師に帰す規則（主の比較だけ）：その K で oursx − abl_ownK の n_dir の族の区間が 0 より上，族が判定できる，区間が使える
                        if comp == MAIN_COMP and vname == "main" and band == "frac" and cmax == CENS_DRAW_MAX:
                            for K_ in KS:
                                for name in ("oursx-abl_ownK", "abl_refb-abl_ownK", "abl_ref0-abl_ownK"):
                                    cdd = CORP_d.get((name, fam, K_, rule, dirq, "impro_only" if variant == "readable_impro_only" else "main"))
                                    if cdd is None:
                                        continue
                                    cps = corpora_for(cdd, variant)
                                    fs = family_stats(cdd, cps)
                                    tag = {"oursx-abl_ownK": "teacher", "abl_refb-abl_ownK": "refb_vs_ownK", "abl_ref0-abl_ownK": "ref0_vs_ownK"}[name]
                                    rec[f"{tag}_med_K{K_}"] = fs["median"]
                                    rec[f"{tag}_lo_K{K_}"] = pct(fs["b"], 2.5) if fs["b"] is not None else np.nan
                                    rec[f"{tag}_judgeable_K{K_}"] = bool(len(cps) >= need_of(fam, variant) and fs["ok"] and fs["b"] is not None)
                                    if name == "oursx-abl_ownK":
                                        rec[f"teacher_attrib_K{K_}"] = bool(rec[f"{tag}_judgeable_K{K_}"] and rec[f"{tag}_lo_K{K_}"] > 0)
                        rows.append(rec)
    J = pd.DataFrame(rows)
    # 最終の区分：readable の版と all の版（打ち切りは端に置く）の両方で同じ側に出たときだけ holds / fails / neg．右端の置き方で変わるかも付ける
    keys = ["family", "comparison", "coord_set", "band", "cens_max", "kall_rule", "variant"]
    for p in ("P87_1", "P87_2", "P87_3"):
        col = f"{p}_status"
        if col not in J.columns:
            continue
        lookup = {tuple(k): v for k, v in zip(J[keys].itertuples(index=False), J[col])}
        fin, sens = [], []
        for k in J[keys].itertuples(index=False):
            fam, comp, cs, band, cm, rule, variant = k
            s_r, s_a = lookup.get((fam, comp, cs, band, cm, rule, "readable")), lookup.get((fam, comp, cs, band, cm, rule, "all"))
            if variant != "readable":
                fin.append("")
            elif s_r in ("holds", "fails", "neg"):
                fin.append(s_r if s_r == s_a else "undet_censored_version")
            else:
                fin.append(s_r if s_r is not None else "")
            s_m, s_n = lookup.get((fam, comp, cs, band, cm, "mean", variant)), lookup.get((fam, comp, cs, band, cm, "min", variant))
            sens.append(bool(s_m != s_n))
        J[f"{p}_final"] = fin
        J[f"{p}_rule_sensitive"] = sens
    return J


def judge_dS(CORP):
    """換算する前の差 ΔS で P87-1（dS_dir）と P87-3（dS_uar）の比の判定（感度）．P87-2 は当てない（o の単位に帯が無い）．"""
    rows = []
    for fam, comp in sorted(set((c[1], c[0]) for c in CORP)):
        rec = {"family": fam, "comparison": comp}
        for p, q in (("P87_1", "dS_dir"), ("P87_3", "dS_uar")):
            c0, c25 = CORP.get((comp, fam, "0", "-", q, "main")), CORP.get((comp, fam, "25", "-", q, "main"))
            if c0 is None or c25 is None:
                rec[f"{p}_status"] = "not_judgeable"
                continue
            cps = sorted(set(c0) & set(c25))
            if len(cps) < need_of(fam):
                rec[f"{p}_status"] = "not_judgeable"
                continue
            f0, f25 = family_stats(c0, cps), family_stats(c25, cps)
            rec.update({f"{p}_med_K0": f0["median"], f"{p}_med_K25": f25["median"], f"{p}_n_corpora": len(cps)})
            if f0["b"] is None or f25["b"] is None:
                rec[f"{p}_status"] = "point_only"
                continue
            diff = f25["b"] - HOLD_RATIO * f0["b"]
            lo0, hi0, lod, hid = pct(f0["b"], 2.5), pct(f0["b"], 97.5), pct(diff, 2.5), pct(diff, 97.5)
            rec.update({f"{p}_lo_K0": lo0, f"{p}_lo_diff": lod, f"{p}_hi_diff": hid})
            rec[f"{p}_status"] = status3(lo0 > 0 and lod >= 0, hi0 <= 0 or hid < 0) if p == "P87_1" else status3(lo0 > 0 and hid < 0, hi0 <= 0 or lod >= 0)
        rows.append(rec)
    return pd.DataFrame(rows)


def judge_563188d(U):
    """直す前の規則（コミット 563188d）を第 2 版以後の推定量に当てる．
    563188d の規則：族は音声＝audio_wavlm の全単位（AVT 3 + AV 5 = 8），ほかは (役, 分類器, 世界) ごと．単位の点は反復ごとの読みの平均（反復ごとの曲線，右端 n_tgt_r/4），
    読めない単位＝打ち切りの反復が半数以上，判定できる＝読めない単位が半数以下．P1：med0 > 0 かつ med25 ≥ 0.5·med0（点の中央値．全単位）．
    P2：K = 5・25 で区分「正」の単位が半数未満（区分は第 3 版の単位の区間から取る．563188d の抽出とは違う）．P3：med0 > 0 かつ med25 < 0.5·med0．"""
    rows = []
    U = U[(U.sensitivity == "main") & (U.kall_rule == MAIN_RULE)].copy()
    U["fam_v1"] = ["audio" if s == "audio_wavlm" else f"{ro}:{s}:{ru}" for s, ro, ru in zip(U.student, U.role, U.run)]
    U["readable_v1"] = U.frac_cens_rep.fillna(1.0) < NOT_READABLE_FRAC_563188d
    for (comp, fam), g in U.groupby(["comparison", "fam_v1"]):
        def grp(q, K):
            return g[(g.quantity == q) & (g.K == K)]

        def judgeable(*gs):
            return all(len(s) and (~s.readable_v1).sum() * 2 <= len(s) for s in gs)

        def med(s):
            return float(np.nanmedian(s.point_rep)) if len(s) else np.nan
        d0, d25, u0, u25, p5, p25 = grp("dir", "0"), grp("dir", "25"), grp("uar", "0"), grp("uar", "25"), grp("pos", "5"), grp("pos", "25")
        rec = {"family_563188d": fam, "comparison": comp, "n_units": len(d0), "n_not_readable_dir_K0": int((~d0.readable_v1).sum()), "n_not_readable_dir_K25": int((~d25.readable_v1).sum())}
        if judgeable(d0, d25):
            rec.update({"P87_1_med_K0": med(d0), "P87_1_med_K25": med(d25), "P87_1_holds": bool(med(d0) > 0 and med(d25) >= HOLD_RATIO * med(d0)), "P87_1_status": "judged"})
        else:
            rec["P87_1_status"] = "not_judgeable"
        if judgeable(p5, p25):
            rec.update({"P87_2_npos_K5": int((p5.judgment == "pos").sum()), "P87_2_n_K5": len(p5), "P87_2_npos_K25": int((p25.judgment == "pos").sum()), "P87_2_n_K25": len(p25),
                        "P87_2_holds": bool((p5.judgment == "pos").sum() * 2 < len(p5) and (p25.judgment == "pos").sum() * 2 < len(p25)), "P87_2_status": "judged"})
        else:
            rec["P87_2_status"] = "not_judgeable"
        if judgeable(u0, u25):
            rec.update({"P87_3_med_K0": med(u0), "P87_3_med_K25": med(u25), "P87_3_holds": bool(med(u0) > 0 and med(u25) < HOLD_RATIO * med(u0)), "P87_3_status": "judged"})
        else:
            rec["P87_3_status"] = "not_judgeable"
        rows.append(rec)
    return pd.DataFrame(rows)


def assumptions(out, Cv, U, V, Cp, prov):
    rows = []
    Cv2 = Cv.copy()
    Cv2["family"] = [family_of(s) for s in Cv2.student]
    for (fam, q, rule), g in Cv2[Cv2.K_label != "1"].groupby(["family", "quantity", "kall_rule"]):   # K = 1 の点は整えようがないので分母から外す
        rows.append({"assumption": 1, "what": "monotone_adjustment", "family": fam, "quantity": q, "kall_rule": rule, "n_points": len(g), "frac_points_adjusted": float(g.adjusted.mean()),
                     "max_adjustment": float((g.S_adj - g.S_raw).abs().max()), "ok": bool(g.adjusted.mean() < 0.25)})
    Um = U[(U.sensitivity == "main")]
    for (comp, fam, K, rule, q), g in Um.groupby(["comparison", "family", "K", "kall_rule", "quantity"]):
        if q.startswith("dS_"):
            continue

        def cnt(col):
            return int(g[col].sum()) if col in g.columns else 0

        def nanmed(col):
            s = g[col].to_numpy(dtype=np.float64) if col in g.columns else np.array([])
            return float(np.nanmedian(s)) if np.isfinite(s).any() else np.nan
        rows.append({"assumption": 2, "what": "readable_units", "comparison": comp, "family": fam, "K": K, "kall_rule": rule, "quantity": q, "n_units": len(g), "n_not_readable": int((~g.is_readable).sum()),
                     "n_cens_hi_method": cnt("flag_cens_hi_method"), "n_cens_lo_method": cnt("flag_cens_lo_method"), "n_cens_hi_none": cnt("flag_cens_hi_none"), "n_cens_lo_none": cnt("flag_cens_lo_none"),
                     "n_amb": cnt("flag_amb_none") + cnt("flag_amb_method"), "n_interval_not_ok": int((~g.interval_ok).sum()) if "interval_ok" in g.columns else 0,
                     "median_frac_draw_cens_any": nanmed("frac_draw_cens_any"), "median_frac_cens_hi_method": nanmed("frac_cens_hi_method"), "median_frac_cens_none": float(np.nanmax([nanmed("frac_cens_lo_none"), nanmed("frac_cens_hi_none")]))})
    # 3・4：符号の一致率（コーパスごと．定義から 0 の単位と読めないコーパスを除く）
    for (comp, fam, K, rule), g in Cp.groupby(["comparison", "family", "K", "kall_rule"]):
        def sign_agree(qa, qb):
            a = g[(g.quantity == qa) & (g.judgment != "same_by_definition") & g.is_readable].set_index("corpus").point
            b = g[(g.quantity == qb) & (g.judgment != "same_by_definition") & g.is_readable].set_index("corpus").point
            ix = a.index.intersection(b.index)
            return (float(np.mean(np.sign(a[ix]) == np.sign(b[ix]))) if len(ix) else np.nan), len(ix)
        for qa, qb, n_ in (("dir", "dir_auc", 3), ("dir", "dir_bfix", 3), ("pos", "pos_bfix", 4), ("pos", "posloss", 4)):
            ag, n = sign_agree(qa, qb)
            rows.append({"assumption": n_, "what": f"sign_agreement_{qa}_vs_{qb}", "comparison": comp, "family": fam, "K": K, "kall_rule": rule, "n_corpora": n, "agreement": ag,
                         "ok": bool(ag >= SIGN_AGREE_LOW) if np.isfinite(ag) else None})
    rows.append({"assumption": 4, "what": "E75_position_model_fit", "note": "E75 7 節：位置の側の 2 つの山の模型は 8 種類とも基準 1/3 を満たさなかった（S_pos_bfix を並べる）"})
    # 5：右端の置き方と，手法の読みが K = 50 と全件の間の補間に入ったコーパスの数
    if len(V):
        for rule in KALL_RULES:
            for q in ("dir", "uar", "pos"):
                col = f"K_{q}_method_{rule}"
                vm = V[V.comparison == MAIN_COMP]
                for K in KS:
                    s = vm[vm.K == K]
                    rows.append({"assumption": 5, "what": "method_reading_in_long_segments", "quantity": q, "kall_rule": rule, "K": K, "n_groups": int(s[col].notna().sum()),
                                 "n_above_50": int((s[col] > 50).sum()), "n_above_100": int((s[col] > 100).sum()),
                                 "n_none_above_50": int((s[f"K_{q}_none_{rule}"] > 50).sum()), "n_none_above_100": int((s[f"K_{q}_none_{rule}"] > 100).sum())})
    for (fam, rule), g in Cv2[Cv2.K_label == "all"].groupby(["family", "kall_rule"]):
        rows.append({"assumption": 5, "what": "K_all_value", "family": fam, "kall_rule": rule, "K_all_median": float(g.K_value.median()), "K_all_min": float(g.K_value.min()), "K_all_max": float(g.K_value.max())})
    # 6：種・反復・群の数・G_eff
    for k, v in prov.get("distinct_test_sets", {}).items():
        rows.append({"assumption": 6, "what": "distinct_test_sets", "run_target": k, "n_distinct": v, "g_eff": prov.get("g_eff", {}).get(k)})
    for (comp, fam, K), g in Um[Um.quantity == "dir"].groupby(["comparison", "family", "K"]):
        rows.append({"assumption": 6, "what": "seeds_and_groups", "comparison": comp, "family": fam, "K": K, "n_units": len(g), "n_seeds_max": int(g.n_seeds.max()), "n_groups_none": int(g.n_groups_none.sum()),
                     "g_eff_min": float(g.g_eff_min.min())})
    rows.append({"assumption": 7, "what": "theory_assumptions", "note": "教師（oursx が選ぶ疑似ラベルのモデル）は配備する分類器のモダリティと K 本を使う（条件つき独立と K 本との独立は保証されない）"})
    rows.append({"assumption": 8, "what": "four_class_direction", "note": "「疑似ラベルは向きを保つ」は 2 クラスか一様な雑音の結果．4 感情で教師の混同が一様でなければ，条件つき独立でも対ごとの向きが偏りうる（P87-1 が外れたときの別の説明）"})
    write_csv(pd.DataFrame(rows), out / "assumptions.csv")


def write_summary_md(out, Fm, J, Jd):
    lines = ["# E87 の出力の表（解釈は書かない）", ""]
    sub = Fm[(Fm.variant == "readable") & (Fm.kall_rule == MAIN_RULE) & (Fm.jk_inflation)]
    for comp in sorted(sub.comparison.unique()):
        lines.append(f"## {comp}")
        for fam in sorted(sub.family.unique()):
            s2 = sub[(sub.comparison == comp) & (sub.family == fam)]
            if s2.empty:
                continue
            lines.append(f"### {fam}")
            lines.append("| 量 | K | 読めるコーパス／名目 | 中央値 | 2.5% | 97.5% | 区間が使える | 正のコーパス |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r_ in s2.sort_values(["quantity", "K"]).itertuples():
                lines.append(f"| {r_.quantity} | {r_.K} | {r_.n_corpora_used}/{r_.n_corpora_nominal} | {r_.median:.2f} | {r_.lo:.2f} | {r_.hi:.2f} | {r_.interval_ok} | {r_.n_corp_pos} |")
            lines.append("")
    lines.append("## 判定")
    if len(J):
        cols = [c for c in ("family", "comparison", "kall_rule", "variant", "coord_set", "band", "P87_1_status", "P87_1_final", "P87_2_status", "P87_2_final", "P87_3_status", "P87_3_final", "contrast_allowed") if c in J.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r_ in J[cols].itertuples(index=False):
            lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r_) + " |")
    lines.append("")
    lines.append("## ΔS での比の判定（感度）")
    if len(Jd):
        cols = list(Jd.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r_ in Jd[cols].itertuples(index=False):
            lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r_) + " |")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def snapshot_code(out, stamp):
    dest = out / "code" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for rel in SCRIPT_DEPS:
        (dest / Path(rel).parent).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest / rel)
    diff = subprocess.run(["git", "diff", "HEAD", "--"] + list(SCRIPT_DEPS), cwd=ROOT, capture_output=True, text=True).stdout
    (dest / "git_diff_HEAD.patch").write_text(diff, encoding="utf-8")
    return dest


def main():
    global HEADER
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e87")
    ap.add_argument("--curve-root", type=Path, default=ROOT / "results" / "e87_tgtonly")
    ap.add_argument("--curve-root-k100", type=Path, default=ROOT / "results" / "e87_tgtonly_k100")
    ap.add_argument("--without-k100", action="store_true", help="試走用．K = 100 の点を差し込まない（6 点の曲線）")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", nargs="+", help="試走用．評価のターゲットを限る")
    ap.add_argument("--runs", nargs="+", choices=RUNS, help="試走用．世界を限る")
    ap.add_argument("--smoke", action="store_true", help="試走（--boot 50 と同じ扱い）")
    args = ap.parse_args()
    if args.smoke:
        args.boot = min(args.boot, 50)
    out = args.out.resolve()
    partial = bool(args.smoke or args.targets or args.runs or args.without_k100)
    if partial and out.is_relative_to(ROOT / "results"):
        raise SystemExit("試走（--smoke，--targets，--runs，--without-k100）の出力は results/ の下に書かない．--out を checks/E87/ か scratchpad にする（I8）")
    if not partial and out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} が空でない．消すか別の --out にしてから流す")
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--"] + list(SCRIPT_DEPS), cwd=ROOT, capture_output=True, text=True).stdout.strip()
    HEADER = {"E": "E87", "commit": commit, "dirty": bool(dirty), "script_sha256": E.sha256(Path(__file__))[:16], "boot": args.boot, "seed": args.seed, "smoke": args.smoke,
              "runs": args.runs, "targets": args.targets, "curve_root": str(args.curve_root), "curve_root_k100": None if args.without_k100 else str(args.curve_root_k100),
              "started": started.isoformat(timespec="seconds")}
    prov = {"commit": commit, "dirty_deps": bool(dirty), "dirty_deps_status": dirty, "script_sha256": E.sha256(Path(__file__)),
            "deps_sha256": {rel: E.sha256(ROOT / rel) for rel in SCRIPT_DEPS}, "args": vars(args) | {"out": str(out), "curve_root": str(args.curve_root)},
            "constants": {"GEFF_MIN": GEFF_MIN, "HOLD_RATIO": HOLD_RATIO, "POS_RES_FRAC": POS_RES_FRAC, "POS_BAND_FIXED": POS_BAND_FIXED, "CENS_DRAW_MAX": CENS_DRAW_MAX,
                          "NOT_READABLE_FRAC_563188d": NOT_READABLE_FRAC_563188d, "SIGN_AGREE_LOW": SIGN_AGREE_LOW, "NOMINAL": NOMINAL,
                          "K_GRID": list(K_GRID), "K_EXTRA": list(K_EXTRA), "K_ALL": list(K_GRID) if args.without_k100 else list(K_ALL), "C_GRID": list(C_GRID), "KALL_RULES": list(KALL_RULES), "MAIN_RULE": MAIN_RULE,
                          "K_all_rule": {"mean": "n_tgt/4（反復の平均．E65 と同じ）", "min": "学習プールで最も少ない感情の本数（反復の平均）"},
                          "interpolation": "log K について線形．平らな区間では届く最小の K．K=1 の値より悪い（等号を含まない）→ 1 に置いて下の打ち切り．全件の値より良い → 全件に置いて上の打ち切り．整える前の曲線が値を 2 回以上横切れば「あいまい」",
                          "monotone": "S_uar・S_dir・S_dir_* は累積の最大，S_pos・S_posloss・S_pos_bfix は累積の最小（反復で平均した曲線に当てる）",
                          "readable": "主の右端の置き方で，none と手法の読みがどの群でも打ち切りでもあいまいでもない（none を選んだ群は定義から 0 で読める）",
                          "draws": "種は (seed, crc32(コーパス))．話者の多項分布 (B, nls) と反復の多項分布 (B, R)．世界と比較相手で共有",
                          "corpus_value": "世界の単位の平均（複製・jackknife も平均）．区間は百分位と，G_eff < 20 なら jackknife の t 区間の和集合",
                          "family": "読めるコーパスの中央値．複製は G_eff < 20 のコーパスの複製を点のまわりで jackknife の幅まで広げてから複製ごとに中央値．打ち切り・あいまいの複製がどれかのコーパスにある割合が CENS_DRAW_MAX を超えれば区間は使わない",
                          "judgment": "予言ごとに holds / fails / undet（P87-2 は neg も）．final は readable と all（打ち切りを端に置く）の両方で同じ側のときだけ holds / fails",
                          "QUANTS_B": list(QUANTS_B), "QUANTS_PT": list(QUANTS_PT), "DIFFS": list(DIFFS), "DS": list(DS), "BETTER_HIGH": BETTER_HIGH, "KS": list(KS), "REPS": list(REPS),
                          "SEEDED": sorted(E.SEEDED), "PARTIAL_SELECTORS": E.PARTIAL_SELECTORS, "EXCL_MA": list(EXCL_MA), "family_min_readable": "max(2, ceil(名目のコーパス数/2))"},
            "comparisons": [c for _, c in COMPARISONS], "direct": [n for _, _, n in DIRECT], "main_comparison": MAIN_COMP, "started": started.isoformat(timespec="seconds"), "inputs": {}}
    prov["code_snapshot"] = str(snapshot_code(out, started.strftime("%Y%m%d-%H%M%S")).relative_to(out))
    Cv, Cr, V, U, Ch, Sd, boot_store, jk_store = run_all(args, out, prov)
    write_csv(Cv, out / "curves.csv")
    write_csv(Cr, out / "curves_rep.csv")
    write_csv(V, out / "values.csv")
    write_csv(Sd, out / "values_seed.csv")
    write_csv(U, out / "units.csv")
    write_csv(Ch if len(Ch) else pd.DataFrame([{"check": "none"}]), out / "checks.csv")
    Ud, dstore, djstore = direct_units(U, boot_store, jk_store)
    write_csv(Ud, out / "direct.csv")
    CORP, Cp = corpus_level(U, boot_store, jk_store)
    CORP_d, Cpd = corpus_level(Ud, dstore, djstore) if len(Ud) else ({}, pd.DataFrame())
    fstore = {}
    Fm = family_table(CORP, fstore)
    Fd = family_table(CORP_d, fstore) if CORP_d else pd.DataFrame()
    write_csv(Cp, out / "corpus.csv")
    write_csv(Fm, out / "family.csv")
    if len(Cpd):
        write_csv(Cpd, out / "direct_corpus.csv")
        write_csv(Fd, out / "direct_family.csv")
    J = judge(CORP, CORP_d)
    write_csv(J, out / "judgments.csv")
    Jd = judge_dS(CORP)
    write_csv(Jd, out / "judgments_dS.csv")
    write_csv(judge_563188d(U), out / "judgments_563188d.csv")
    np.savez_compressed(out / "boot_units.npz", **boot_store, **{f"direct|{k}": v for k, v in dstore.items()})
    np.savez_compressed(out / "boot_family.npz", **fstore)
    if jk_store or djstore:
        np.savez_compressed(out / "jk_units.npz", **jk_store, **{f"direct|{k}": v for k, v in djstore.items()})
    assumptions(out, Cv, U, V, Cp, prov)
    write_summary_md(out, Fm, J, Jd)
    prov["n_units"] = int(len(U))
    prov["checks"] = Ch.to_dict("records") if len(Ch) else []
    prov["n_adjusted_points"] = int(Cv.adjusted.sum()) if len(Cv) else 0
    prov["n_units_not_readable"] = int((~U.is_readable).sum()) if len(U) else 0
    prov["n_units_interval_not_ok"] = int((~U.interval_ok).sum()) if len(U) else 0
    prov["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
