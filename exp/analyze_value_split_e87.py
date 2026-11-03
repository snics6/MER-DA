"""E87：疑似ラベルのモデルの価値（ターゲットのラベル何本ぶん）を，向きの分と位置の分に分ける（第 2 版．関門 1 の指摘で直した）．

設計：docs/exploratory-analyses.md の E87（規則は値を見る前に書いた）．評価のターゲットの値で判定するので本番（関門 1 は 5 体）．

  物差し：exp/relearn_tgtonly_e87.py が世界の発話の集合と E77 の分け方で学び直したターゲットだけの分類器（K = 1, 5, 10, 25, 50, 全件．選んだ C）．
  座標：S_uar（UAR），S_dir（6 対の A(d, 0) の平均），S_pos（6 対の o の 2 乗平均の平方根）．副：S_posloss（E79 の位置の損）．点だけ：S_dir_auc，S_dir_bfix（切片だけを直した UAR），S_pos_bfix（S_dir_bfix − S_uar）．
  読み：反復で平均した曲線に，反復で平均した none と手法の値を当てて本数 K' を読み，n_x = K'_x(手法) − K'_x(none)．平らにした区間に入った読みと打ち切りの読みは「読めない」．
  単位：（比較，世界，配備する分類器，K，コーパス）．抽出の種は（種，コーパス）だけで，世界と比較相手で共有する．族はコーパスの値（世界の平均）の中央値で，複製から区間を出す．

使い方：./.venv/bin/python exp/analyze_value_split_e87.py [--out results/e87] [--curve-root results/e87_tgtonly] [--boot 1000] [--targets ...] [--smoke]
  試走（--smoke，--targets）の出力は results/ の下に書けない．本番は既定の引数で 1 回で流す．CSV の先頭の # の行は JSON（コミットと設定）．pd.read_csv(..., comment="#") で読む．
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

warnings.filterwarnings("ignore", message="Mean of empty slice")                  # 抽出でテストの話者が全員抜けた反復は NaN になり，rep_mean が重みから外す（意図した扱い）

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.optimize import minimize
from scipy.special import logsumexp

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
K_GRID = ("1", "5", "10", "25", "50", "all")
C_GRID = (0.01, 0.1, 1.0, 10.0)
KALL_RULES = ("mean", "min")                                                   # 右端の置き方：n_tgt/4（主）と，学習プールで最も少ない感情の本数（感度）
QUANTS_B = ("uar", "dir", "pos", "posloss")                                    # 複製を持つ座標（pos が位置の主，posloss は感度）
QUANTS_PT = ("dir_auc", "dir_bfix", "pos_bfix")                                # 点の値だけの座標（模型に依らない感度）
DS = ("uar", "dir", "pos")                                                     # 換算する前の差 ΔS（複製あり）
BETTER_HIGH = {"uar": True, "dir": True, "dir_auc": True, "dir_bfix": True, "pos": False, "posloss": False, "pos_bfix": False}
SELECTORS = (("oursx", "none-oursx"), ("kd", "none-kd"), ("abl_ownK", "none-abl_ownK"), ("groupB", "none-groupB"))
FIXED = (("saerens", "none-saerens"), ("massbal", "none-massbal"))             # 位置だけを直す標準の手（設定は 1 つ．chosen.csv に無いので行の表から引く）
COMPARISONS = SELECTORS + FIXED
MAIN_COMP = "none-oursx"
DIRECT = (("none-oursx", "none-abl_ownK", "oursx-abl_ownK"), ("none-oursx", "none-kd", "oursx-kd"), ("none-oursx", "none-groupB", "oursx-groupB"))
GEFF_MIN = E.GEFF_MIN
HOLD_RATIO = 0.5                                                               # P87-1・P87-3 の比の基準（E66 の P2 と同じ）
POS_RES_FRAC = 0.5                                                             # P87-2 の分解能：hi[med_pos(K)] ≤ 0.5·K
EXCL_MA = ("MOSEI", "AFEW")
HEADER = {}
SCRIPT_DEPS = ("exp/analyze_value_split_e87.py", "exp/relearn_tgtonly_e87.py", "exp/analyze_pairs_e83.py", "exp/explore_pairs.py", "exp/explore_synergy.py",
               "exp/pair_table.py", "exp/multimodal_distill.py", "measure/data.py", "measure/split.py")


def write_csv(df, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# " + json.dumps(HEADER, ensure_ascii=False) + "\n")
        df.to_csv(fh, index=False)


def family_of(st):
    return {"audio_wavlm": "audio", "video": "video", "text_roberta": "text"}.get(st, f"fused:{st}")


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
    """曲線（…, 6）に値 v（…）を当てて本数 K' を読む．平らな区間では届く最小の K．
    戻り：(K', 下の打ち切り（v が K=1 の値より悪い．等号は含まない），上の打ち切り（全件の値より良い），平らに落ちた（読みの位置が単調に整えて持ち上げた点）)．"""
    s = np.asarray(curve if better_high else -curve, dtype=np.float64)
    vv = np.asarray(v if better_high else -v, dtype=np.float64)
    raw = s
    if adjust:
        s = np.maximum.accumulate(s, axis=-1)
    adjusted = s > raw + 1e-12
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
    # 読みに使った区間の左端が，単調に整えて持ち上げた点（平らにした区間の端）なら「平らに落ちた」
    in_flat = np.take_along_axis(adjusted, (i - 1)[..., None], -1)[..., 0] & any_ge & ~ge[..., 0]
    bad = ~np.isfinite(vv) | ~np.isfinite(s).all(-1)
    K = np.where(bad, np.nan, np.exp(lk))
    return K, (vv < s[..., 0]) & ~bad, (~any_ge) & ~bad, in_flat & ~bad


def load_curve(root, run, st, T, yall, te, prov):
    """学び直した物差しの ① のテストの点数（K 6 × 選んだ C）を E77 の _te の並びで返す．"""
    f = root / run / st / f"{T}.npz"
    fc = root / run / st / f"{T}.csv"
    if not f.exists():
        raise SystemExit(f"{run} {st} {T}：物差しの npz が無い（{f}）")
    E.note_input(prov, f)
    E.note_input(prov, fc, with_sha=True)
    z = np.load(f)
    meta = json.loads(str(z["_meta"]))
    st_ = meta["settings"]
    if list(st_["K"]) != list(K_GRID) or [float(c) for c in st_["C_grid"]] != list(C_GRID) or list(st_["reps"]) != list(REPS):
        raise SystemExit(f"{run} {st} {T}：物差しの K・C・反復の並びが台本の約束と違う")
    if not np.array_equal(np.asarray(z["y"]).astype(int), yall):
        raise SystemExit(f"{run} {st} {T}：物差しの y が load_any のラベルと違う")
    csv = pd.read_csv(fc, comment="#", dtype={"K": str, "rep": int})
    tg = csv[csv.method == "tgt_only"]
    out = {}
    for r in REPS:
        idx = np.asarray(z[f"test_idx_r{r}"]).ravel()
        te77 = np.sort(np.asarray(te[r]).ravel())
        if not np.array_equal(np.sort(idx), te77):
            raise SystemExit(f"{run} {st} {T} r{r}：物差しのテストの行が E77 と違う（curve_mismatch）")
        pos = {int(a): i for i, a in enumerate(idx)}
        order = np.array([pos[int(a)] for a in np.asarray(te[r]).ravel()])
        Cs = np.asarray(z[f"t_C_r{r}"], dtype=np.float64)
        Lk, cv_uar = [], []
        for iK, K in enumerate(K_GRID):
            row = tg[(tg.K == K) & (tg.rep == r)]
            if len(row) != 1 or not np.isclose(float(row.C.iloc[0]), Cs[iK]):
                raise SystemExit(f"{run} {st} {T} r{r} K{K}：物差しの csv の行か C が npz と合わない")
            iC = int(np.flatnonzero(np.isclose(np.array(C_GRID), Cs[iK]))[0])
            L = np.asarray(z[f"t_logit_r{r}"][iK, iC], dtype=np.float64)[order]
            if not np.isfinite(L).all():
                raise SystemExit(f"{run} {st} {T} r{r} K{K}：物差しの点数に NaN/Inf")
            Lk.append(L)
            cv_uar.append({c: float(row[f"test_uar_C{c:g}"].iloc[0]) for c in C_GRID})
        ra = tg[(tg.K == "all") & (tg.rep == r)].iloc[0]
        out[r] = {"L": Lk, "C": Cs, "rule": [str(x) for x in z[f"t_rule_r{r}"]], "n_all": int(ra.n_tgt), "n_min": int(ra.n_pool_min_class), "test_uar_allC": cv_uar}
    return out


def rep_mean(arr_by_rep, A=None):
    """反復ごとの配列（R 個，同じ形）を，A（B, R）の重みで混ぜる（A が無ければ平均）．NaN は重みから外す．"""
    arr = np.stack(arr_by_rep, 0)                                              # (R, ...)
    if A is None:
        return np.nanmean(arr, 0)
    w = A.T                                                                     # (R, B)
    w = w.reshape((len(arr_by_rep), A.shape[0]) + (1,) * (arr.ndim - 2))
    fin = np.isfinite(arr)
    return np.where(fin, arr, 0).__mul__(w).sum(0) / np.where(fin, w, 0).sum(0)


def group_curve(cv, yte_by_rep, sidx_by_rep, ssidx_by_rep, C, A, Wj, need_jk, Wj_s):
    """1 群・1 配備する分類器の曲線：反復の平均（点），抽出の重みで混ぜた複製（B），jackknife（J，S）．"""
    R = len(REPS)
    pt = {q: np.zeros((R, 6)) for q in QUANTS_B + QUANTS_PT}
    allc = np.zeros((R, 6, 4))
    B_ = {q: [] for q in QUANTS_B}
    J_ = {q: [] for q in QUANTS_B}
    S_ = {q: [] for q in QUANTS_B}
    for ri, r in enumerate(REPS):
        e = cv[r]
        bB = {q: np.zeros((len(C), 6)) for q in QUANTS_B}
        bJ = {q: np.zeros((Wj.shape[0], 6)) for q in QUANTS_B} if need_jk else None
        bS = {q: np.zeros((Wj_s.shape[0], 6)) for q in QUANTS_B} if Wj_s is not None else None
        for iK in range(6):
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
    kv = {"mean": np.array([1, 5, 10, 25, 50, np.mean([cv[r]["n_all"] / 4 for r in REPS])], dtype=np.float64),
          "min": np.array([1, 5, 10, 25, 50, np.mean([cv[r]["n_min"] for r in REPS])], dtype=np.float64)}
    for rule, k in kv.items():
        if not (np.diff(k) > 0).all():
            raise SystemExit(f"右端の本数（{rule}）が 50 以下：{k}")
    return {"pt_rep": pt, "pt": {q: pt[q].mean(0) for q in pt}, "allc_rep": allc, "kv": kv, "logk": {r_: np.log(k) for r_, k in kv.items()}, "Aw": A,
            "B": {q: rep_mean(B_[q], A) for q in QUANTS_B}, "J": {q: rep_mean(J_[q]) for q in QUANTS_B} if need_jk else None,
            "S": {q: rep_mean(S_[q]) for q in QUANTS_B} if Wj_s is not None else None}


def corpus_draws(corpus, speakers, B, seed, n_reps):
    """抽出の種は（種，コーパス）だけ（E79 の種は実行を含むので，同じコーパスの世界と比較相手で抽出を共有するために変える）．"""
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
                curves_raw = {st: load_curve(args.curve_root, run, st, T, yall, te, prov) for st in students}
                prep[T] = {"X": X, "tab": tab, "bf": bf, "y": yall, "spk": spk_all, "te": te, "logp": logp, "curves_raw": curves_raw}
            speakers = sorted(set().union(*[set(p["spk"][p["te"][r]]) for p in prep.values() for r in REPS]))
            sid = {s: i for i, s in enumerate(speakers)}
            nls = len(speakers)
            C, A = corpus_draws(cp, speakers, args.boot, args.seed, len(REPS))
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
                    for q in QUANTS_B + QUANTS_PT:
                        for rule in KALL_RULES:
                            raw = cv["pt"][q]
                            s = raw if BETTER_HIGH[q] else -raw
                            adj = np.maximum.accumulate(s)
                            adj = adj if BETTER_HIGH[q] else -adj
                            for iK in range(6):
                                curve_rows.append({"run": run, "target": T, "student": st, "quantity": q, "kall_rule": rule, "K_label": K_GRID[iK], "K_value": cv["kv"][rule][iK],
                                                   "S_raw": raw[iK], "S_adj": adj[iK], "adjusted": bool(abs(adj[iK] - raw[iK]) > 1e-12)})
                    for ri, r in enumerate(REPS):
                        raw_r = p["curves_raw"][st][r]
                        for iK in range(6):
                            curve_rep_rows.append({"run": run, "target": T, "student": st, "rep": r, "K_label": K_GRID[iK], "C": raw_r["C"][iK], "C_rule": raw_r["rule"][iK],
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
                            check_rows.append({"check": "chosen_missing", "comparison": comp, "run": run, "corpus": cp, "student": st, "K": K})
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
                                "n_seeds": max((per_T[T]["n_seeds"] for T in Ts), default=0)}
                        specs = [("main", Ts, "jk")]
                        if run == "fuse-audio-video-text" and cp == "IEMOCAP":
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
    """1 群：反復で平均した曲線に，反復で平均した none と手法の値を当てて読む．点，複製（B），jackknife（J，S）．"""
    X = p["X"]
    res = {"n_seeds": 0, "pt": {}, "pt_nomono": {}, "flag": {}, "B": {}, "J": {}, "S": {}, "cens": {}, "readable": {}, "dS": {}, "dS_B": {}, "dS_J": {}, "dS_S": {}}
    N = method_scores(run, T, p, X, None, st, K, comp, nls, C, Wj, need_jk, Wj_s, check_rows, seed_rows)
    if N is None:
        return None
    Mth = N if chT is None else method_scores(run, T, p, X, chT, st, K, comp, nls, C, Wj, need_jk, Wj_s, check_rows, seed_rows)
    if Mth is None:
        return None
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
            Kn, lo_n, hi_n, fl_n = read_K(cv["pt"][q], logk, pt_n[q], BETTER_HIGH[q])
            Km, lo_m, hi_m, fl_m = read_K(cv["pt"][q], logk, pt_m[q], BETTER_HIGH[q])
            Kn2, _, _, _ = read_K(cv["pt"][q], logk, pt_n[q], BETTER_HIGH[q], adjust=False)
            Km2, _, _, _ = read_K(cv["pt"][q], logk, pt_m[q], BETTER_HIGH[q], adjust=False)
            n = 0.0 if chT is None else float(Km - Kn)
            res["pt"][(rule, q)] = n
            res["pt_nomono"][(rule, q)] = 0.0 if chT is None else float(Km2 - Kn2)
            res["flag"][(rule, q)] = {"cens_none": bool(lo_n or hi_n), "flat_none": bool(fl_n), "cens_method": bool(lo_m or hi_m), "flat_method": bool(fl_m)}
            res["readable"][(rule, q)] = not (lo_n or hi_n or fl_n or lo_m or hi_m or fl_m)
            row.update({f"K_{q}_none_{rule}": float(Kn), f"K_{q}_method_{rule}": float(Km), f"n_{q}_{rule}": n, f"n_{q}_nomono_{rule}": res["pt_nomono"][(rule, q)],
                        f"readable_{q}_{rule}": res["readable"][(rule, q)], **{f"{k}_{q}_{rule}": v for k, v in res["flag"][(rule, q)].items()}})
            if q in QUANTS_B:
                if chT is None:
                    res["B"][(rule, q)] = np.zeros(len(C))
                    res["cens"][(rule, q)] = {"none": 0.0, "method": 0.0, "flat_none": 0.0, "flat_method": 0.0}
                    if need_jk:
                        res["J"][(rule, q)] = np.zeros(nls)
                    if Wj_s is not None:
                        res["S"][(rule, q)] = np.zeros(Wj_s.shape[0])
                else:
                    KBn, lo1, hi1, f1 = read_K(cv["B"][q], logk, B_n[q], BETTER_HIGH[q])
                    KBm, lo2, hi2, f2 = read_K(cv["B"][q], logk, B_m[q], BETTER_HIGH[q])
                    res["B"][(rule, q)] = KBm - KBn
                    res["cens"][(rule, q)] = {"none": float(np.nanmean(lo1 | hi1)), "method": float(np.nanmean(lo2 | hi2)), "flat_none": float(np.nanmean(f1)), "flat_method": float(np.nanmean(f2))}
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
            res["cens"][(rule, dq)] = {k: max(res["cens"][(rule, "dir")][k], res["cens"][(rule, pq)][k]) for k in res["cens"][(rule, "dir")]}
            if need_jk:
                res["J"][(rule, dq)] = res["J"][(rule, "dir")] - res["J"][(rule, pq)]
            if Wj_s is not None:
                res["S"][(rule, dq)] = res["S"][(rule, "dir")] - res["S"][(rule, pq)]
            row[f"n_{dq}_{rule}"] = res["pt"][(rule, dq)]
    for q in DS:                                                               # 換算する前の差（右端の置き方に依らない）
        res["dS"][q] = 0.0 if chT is None else pt_m[q] - pt_n[q]
        res["dS_B"][q] = np.zeros(len(C)) if chT is None else B_m[q] - B_n[q]
        if need_jk:
            res["dS_J"][q] = np.zeros(nls) if chT is None else J_m[q] - J_n[q]
        if Wj_s is not None:
            res["dS_S"][q] = np.zeros(Wj_s.shape[0]) if chT is None else S_m[q] - S_n[q]
        row[f"dS_{q}"] = res["dS"][q]
    val_rows.append(row)
    return res


def aggregate_unit(run, cp, st, K, comp, stag, jkkind, Ts_use, per_T, geff, n_sess, all_none, meta, unit_rows, boot_store, jk_store):
    if any(per_T[T] is None for T in Ts_use):
        return
    gmin = float(min(geff[T] for T in Ts_use))
    items = [(rule, q) for rule in KALL_RULES for q in QUANTS_B + QUANTS_PT + ("dir_minus_pos", "dir_minus_posloss")] + [("-", f"dS_{q}") for q in DS]

    def stack_mean(key, qq):
        return np.mean(np.stack([per_T[T][key][qq] for T in Ts_use]), 0) if all(qq in per_T[T][key] for T in Ts_use) else None
    for rule, q in items:
        is_ds = q.startswith("dS_")
        qq = q[3:] if is_ds else q
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
            readable = all(per_T[T]["readable"][(rule, q)] for T in Ts_use)
            flags = {k: any(per_T[T]["flag"][(rule, q)][k] for T in Ts_use) for k in per_T[Ts_use[0]]["flag"][(rule, q)]}
            has_b = q in QUANTS_B or q.startswith("dir_minus_")
            b = stack_mean("B", (rule, q)) if has_b else None
            th = stack_mean("J", (rule, q)) if has_b else None
            ths = stack_mean("S", (rule, q)) if has_b else None
            cens = {k: float(np.mean([per_T[T]["cens"][(rule, q)][k] for T in Ts_use])) for k in per_T[Ts_use[0]]["cens"][(rule, q)]} if has_b else {}
        lo = hi = lo_j = hi_j = lo_s = hi_s = np.nan
        rule_i, nvalid = "point_only", 0
        if b is not None:
            lo, hi = np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)
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
        if th is not None:
            jk_store[key] = th.astype(np.float32)
        if all_none:
            jd = "same_by_definition"
        elif b is None:
            jd = "point_only"
        elif not (np.isfinite(x) and np.isfinite(lo) and np.isfinite(hi)):
            jd = "missing"
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
                          "judgment": jd, "boot_key": key, **meta})


# ---------------------------------------------------------------- 直接の差，コーパスの値，族，判定

def direct_units(U, boot_store):
    rows, store = [], {}
    for a, b_, name in DIRECT:
        Ua = U[(U.comparison == a)].set_index(["run", "student", "K", "corpus", "sensitivity", "kall_rule", "quantity"])
        Ub = U[(U.comparison == b_)].set_index(["run", "student", "K", "corpus", "sensitivity", "kall_rule", "quantity"])
        for idx in Ua.index.intersection(Ub.index):
            ra, rb = Ua.loc[idx], Ub.loc[idx]
            x = float(ra.point - rb.point)
            ba_, bb = boot_store.get(ra.boot_key), boot_store.get(rb.boot_key)
            lo = hi = np.nan
            jd = "point_only"
            key = f"{name}|{'|'.join(map(str, idx))}"
            if ba_ is not None and bb is not None:
                d = ba_.astype(np.float64) - bb.astype(np.float64)
                store[key] = d.astype(np.float32)
                lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
                jd = "pos" if lo > 0 else ("neg" if hi < 0 else "undet")
            run, st, K, cp, stag, rule, q = idx
            rows.append({"comparison": name, "run": run, "student": st, "family": family_of(st), "K": K, "corpus": cp, "sensitivity": stag, "kall_rule": rule, "quantity": q,
                         "point": x, "lo": lo, "hi": hi, "judgment": jd, "is_readable": bool(ra.is_readable and rb.is_readable), "boot_key": key})
    return pd.DataFrame(rows), store


VARIANTS = ("readable", "all", "readable_excl_mosei_afew", "readable_impro_only")


def corpus_and_family(U, store):
    """コーパスの値（そのコーパスを含む世界の単位の平均．抽出を共有しているので複製も平均できる）と，族の中央値（読めるコーパス）．
    版：readable（主），all（感度 (i)），readable_excl_mosei_afew（感度 (g)），readable_impro_only（感度 (e)：AVT の IEMOCAP の単位を即興だけの行に差し替える）．"""
    cr, fr, fstore = [], [], {}
    U = U[U.sensitivity.isin(["main", "iemocap_impro_only"])]
    for (comp, fam, K, rule, q), g0 in U.groupby(["comparison", "family", "K", "kall_rule", "quantity"]):
        per_variant_units = {"main": g0[g0.sensitivity == "main"]}
        if (g0.sensitivity == "iemocap_impro_only").any():
            gi = g0[g0.sensitivity == "iemocap_impro_only"]
            gm = g0[g0.sensitivity == "main"]
            drop = gm.set_index(["run", "corpus"]).index.isin(gi.set_index(["run", "corpus"]).index)
            per_variant_units["impro_only"] = pd.concat([gm[~drop], gi])
        corp = {}
        for uv, g in per_variant_units.items():
            corp[uv] = {}
            for cp, gc in g.groupby("corpus"):
                bs = [store[k] for k in gc.boot_key if k in store]
                cb = np.mean(np.stack(bs), 0).astype(np.float64) if len(bs) == len(gc) else None
                lo, hi = (float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5))) if cb is not None else (np.nan, np.nan)
                jd = "same_by_definition" if (gc.judgment == "same_by_definition").all() else ("point_only" if cb is None else ("pos" if lo > 0 else ("neg" if hi < 0 else "undet")))
                corp[uv][cp] = {"point": float(gc.point.mean()), "point_nomono": float(gc.point_nomono.mean()) if "point_nomono" in gc.columns else np.nan, "B": cb,
                                "readable": bool(gc.is_readable.all()), "lo": lo, "hi": hi, "judgment": jd, "n_units": len(gc)}
                if uv == "main":
                    cr.append({"comparison": comp, "family": fam, "K": K, "kall_rule": rule, "quantity": q, "corpus": cp, "n_units": len(gc), "runs": ";".join(sorted(gc.run)),
                               "point": corp[uv][cp]["point"], "point_nomono": corp[uv][cp]["point_nomono"], "lo": lo, "hi": hi, "judgment": jd, "is_readable": corp[uv][cp]["readable"],
                               "n_unit_pos": int((gc.judgment == "pos").sum()), "n_unit_neg": int((gc.judgment == "neg").sum())})
        for variant in VARIANTS:
            uv = "impro_only" if variant == "readable_impro_only" else "main"
            if uv not in corp:
                continue
            cv_ = corp[uv]
            if variant == "all":
                cps = sorted(cv_)
            elif variant == "readable_excl_mosei_afew":
                cps = sorted(c for c in cv_ if cv_[c]["readable"] and c not in EXCL_MA)
            else:
                cps = sorted(c for c in cv_ if cv_[c]["readable"])
            n_total = len(cv_)
            need = max(2, int(np.ceil(n_total / 2)))
            rec = {"comparison": comp, "family": fam, "K": K, "kall_rule": rule, "quantity": q, "variant": variant, "n_corpora_total": n_total, "n_corpora_used": len(cps),
                   "corpora_used": ";".join(cps), "judgeable": bool(len(cps) >= need), "n_corp_pos": int(sum(cv_[c]["judgment"] == "pos" for c in cps)),
                   "n_corp_neg": int(sum(cv_[c]["judgment"] == "neg" for c in cps))}
            if cps:
                vals = np.array([cv_[c]["point"] for c in cps])
                rec["median"] = float(np.median(vals))
                rec["median_nomono"] = float(np.median([cv_[c]["point_nomono"] for c in cps]))
                if len(cps) > 1:
                    loo = {c: float(np.median([cv_[d]["point"] for d in cps if d != c])) for c in cps}
                    rec["loo_medians"] = ";".join(f"{c}:{v:.3f}" for c, v in loo.items())
                    rec["loo_median_min"], rec["loo_median_max"] = min(loo.values()), max(loo.values())
                if all(cv_[c]["B"] is not None for c in cps):
                    fb = np.nanmedian(np.stack([cv_[c]["B"] for c in cps]), 0)
                    key = f"{comp}|{fam}|{K}|{rule}|{q}|{variant}"
                    fstore[key] = fb.astype(np.float32)
                    rec["lo"], rec["hi"] = float(np.nanpercentile(fb, 2.5)), float(np.nanpercentile(fb, 97.5))
                    rec["boot_key"] = key
            fr.append(rec)
    return pd.DataFrame(cr), pd.DataFrame(fr), fstore


def q_of(F, fstore, comp, fam, rule, q, K, variant="readable"):
    r = F[(F.comparison == comp) & (F.family == fam) & (F.kall_rule == rule) & (F.quantity == q) & (F.K == K) & (F.variant == variant)]
    if len(r) != 1 or not bool(r.iloc[0].judgeable) or "median" not in r.columns or pd.isna(r.iloc[0].get("median", np.nan)):
        return None
    r = r.iloc[0]
    return {"med": float(r["median"]), "b": fstore.get(r.get("boot_key")), "n_pos": int(r.get("n_corp_pos", 0)), "n_used": int(r.n_corpora_used), "lo": r.get("lo", np.nan), "hi": r.get("hi", np.nan)}


def pct(b, p):
    return float(np.nanpercentile(b, p)) if b is not None else np.nan


def judge(F, fstore, Fd, fdstore):
    rows = []
    fams = sorted(F.family.unique())
    for fam in fams:
        for comp in sorted(F.comparison.unique()):
            for rule in KALL_RULES:
                for variant in VARIANTS:
                    for posq, dirq, uarq, vname in (("pos", "dir", "uar", "main"), ("posloss", "dir", "uar", "posloss_as_position"), ("pos_bfix", "dir_bfix", "uar", "bfix_point"), ("pos", "dir_auc", "uar", "dir_auc_point")):
                        if vname != "main" and (comp != MAIN_COMP or variant != "readable"):
                            continue
                        if not len(F[(F.comparison == comp) & (F.family == fam) & (F.variant == variant)]):
                            continue
                        rec = {"family": fam, "comparison": comp, "kall_rule": rule, "variant": variant, "coordinates": vname,
                               "is_main_judgment": bool(fam == "audio" and comp == MAIN_COMP and rule == "mean" and variant == "readable" and vname == "main")}
                        d0, d25 = q_of(F, fstore, comp, fam, rule, dirq, "0", variant), q_of(F, fstore, comp, fam, rule, dirq, "25", variant)
                        u0, u25 = q_of(F, fstore, comp, fam, rule, uarq, "0", variant), q_of(F, fstore, comp, fam, rule, uarq, "25", variant)
                        p0, p5, p25 = (q_of(F, fstore, comp, fam, rule, posq, K_, variant) for K_ in ("0", "5", "25"))
                        pointonly = dirq in QUANTS_PT or posq in QUANTS_PT
                        # P87-1
                        if d0 and d25:
                            rec.update({"P87_1_med_K0": d0["med"], "P87_1_med_K25": d25["med"], "P87_1_ratio": d25["med"] / d0["med"] if d0["med"] > 0 else np.nan})
                            if pointonly or d0["b"] is None:
                                rec.update({"P87_1_holds_point": bool(d0["med"] > 0 and d25["med"] >= HOLD_RATIO * d0["med"]), "P87_1_status": "point_only"})
                            else:
                                diff = d25["b"] - HOLD_RATIO * d0["b"]
                                rec.update({"P87_1_lo_K0": pct(d0["b"], 2.5), "P87_1_lo_diff": pct(diff, 2.5), "P87_1_hi_diff": pct(diff, 97.5),
                                            "P87_1_holds": bool(pct(d0["b"], 2.5) > 0 and pct(diff, 2.5) >= 0), "P87_1_status": "judged"})
                        else:
                            rec.update({"P87_1_status": "not_judgeable"})
                        # P87-2
                        ok2, parts = True, {}
                        for K_, pK in (("5", p5), ("25", p25)):
                            dm = q_of(F, fstore, comp, fam, rule, f"dir_minus_{posq}", K_, variant) if posq in ("pos", "posloss") and dirq == "dir" else None
                            if pK is None:
                                ok2 = None
                                break
                            parts[f"P87_2_med_pos_K{K_}"] = pK["med"]
                            parts[f"P87_2_npos_K{K_}"] = pK["n_pos"]
                            parts[f"P87_2_n_K{K_}"] = pK["n_used"]
                            if pointonly or pK["b"] is None or dm is None or dm["b"] is None:
                                ok2 = None
                                parts[f"P87_2_point_K{K_}"] = bool(pK["med"] <= POS_RES_FRAC * float(K_))
                                continue
                            parts[f"P87_2_hi_pos_K{K_}"] = pct(pK["b"], 97.5)
                            parts[f"P87_2_lo_dmp_K{K_}"] = pct(dm["b"], 2.5)
                            cond = bool(pct(dm["b"], 2.5) > 0 and pct(pK["b"], 97.5) <= POS_RES_FRAC * float(K_))
                            parts[f"P87_2_holds_K{K_}"] = cond
                            ok2 = ok2 and cond
                        rec.update(parts)
                        rec.update({"P87_2_holds": ok2, "P87_2_status": "judged" if ok2 is not None else ("point_only" if (p5 and p25) else "not_judgeable")})
                        if p0:
                            rec["P87_2_med_pos_K0"] = p0["med"]
                        # P87-3
                        if u0 and u25:
                            rec.update({"P87_3_med_K0": u0["med"], "P87_3_med_K25": u25["med"], "P87_3_ratio": u25["med"] / u0["med"] if u0["med"] > 0 else np.nan})
                            if u0["b"] is None:
                                rec.update({"P87_3_holds_point": bool(u0["med"] > 0 and u25["med"] < HOLD_RATIO * u0["med"]), "P87_3_status": "point_only"})
                            else:
                                diff = u25["b"] - HOLD_RATIO * u0["b"]
                                rec.update({"P87_3_lo_K0": pct(u0["b"], 2.5), "P87_3_hi_diff": pct(diff, 97.5),
                                            "P87_3_holds": bool(pct(u0["b"], 2.5) > 0 and pct(diff, 97.5) < 0), "P87_3_status": "judged"})
                        else:
                            rec.update({"P87_3_status": "not_judgeable"})
                        # 対比の条件 (A)(B)
                        if p0 and p25 and d25 and p25["b"] is not None and p0["b"] is not None:
                            condA = bool(pct(p25["b"] - HOLD_RATIO * p0["b"], 97.5) < 0 or pct(p0["b"], 97.5) <= 0)
                            condB = bool(d25["n_pos"] * 2 >= d25["n_used"] and p25["n_pos"] * 2 < p25["n_used"])
                            rec.update({"contrast_A_pos_ratio_fails": condA, "contrast_B_counts": condB,
                                        "contrast_allowed": bool(rec.get("P87_1_holds") is True and rec.get("P87_2_holds") is True and condA and condB)})
                        # 教師に帰す規則（主の比較だけ）
                        if comp == MAIN_COMP and vname == "main":
                            for K_ in KS:
                                t = Fd[(Fd.comparison == "oursx-abl_ownK") & (Fd.family == fam) & (Fd.kall_rule == rule) & (Fd.quantity == "dir") & (Fd.K == K_) & (Fd.variant == variant)]
                                if len(t) == 1 and pd.notna(t.iloc[0].get("lo", np.nan)):
                                    rec[f"teacher_attrib_K{K_}"] = bool(t.iloc[0].lo > 0)
                                    rec[f"teacher_attrib_lo_K{K_}"] = float(t.iloc[0].lo)
                        rows.append(rec)
    return pd.DataFrame(rows)


def judge_563188d(U):
    """直す前の規則（コミット 563188d）：主の族は音声の 8 単位（AVT 3，AV 5）の点の中央値．P1：med25 ≥ 0.5·med0 かつ med0 > 0．P2：K=5・25 で「正」の単位が半数未満．P3：med25 < 0.5·med0．"""
    rows = []
    U = U[(U.sensitivity == "main") & (U.kall_rule == "mean")]
    for (comp, fam), g in U.groupby(["comparison", "family"]):
        def med(q, K):
            s = g[(g.quantity == q) & (g.K == K)]
            return float(np.nanmedian(s.point)) if len(s) else np.nan, s
        d0, _ = med("dir", "0"); d25, _ = med("dir", "25"); u0, _ = med("uar", "0"); u25, _ = med("uar", "25")
        _, p5 = med("pos", "5"); _, p25 = med("pos", "25")
        rows.append({"family": fam, "comparison": comp, "n_units_K0": int(((g.K == "0") & (g.quantity == "dir")).sum()),
                     "P87_1_holds": bool(d0 > 0 and d25 >= HOLD_RATIO * d0), "P87_1_med_K0": d0, "P87_1_med_K25": d25,
                     "P87_2_holds": bool((p5.judgment == "pos").sum() * 2 < len(p5) and (p25.judgment == "pos").sum() * 2 < len(p25)) if len(p5) and len(p25) else None,
                     "P87_3_holds": bool(u0 > 0 and u25 < HOLD_RATIO * u0), "P87_3_med_K0": u0, "P87_3_med_K25": u25})
    return pd.DataFrame(rows)


def assumptions(out, Cv, U, V, Fm):
    rows = []
    Cv2 = Cv.copy()
    Cv2["family"] = [family_of(s) for s in Cv2.student]
    for (fam, q, rule), g in Cv2.groupby(["family", "quantity", "kall_rule"]):
        rows.append({"assumption": 1, "what": "monotone_adjustment", "family": fam, "quantity": q, "kall_rule": rule, "frac_points_adjusted": float(g.adjusted.mean()),
                     "max_adjustment": float((g.S_adj - g.S_raw).abs().max()), "ok": bool(g.adjusted.mean() < 0.25)})
    Um = U[(U.sensitivity == "main")]
    for (comp, fam, K, rule, q), g in Um.groupby(["comparison", "family", "K", "kall_rule", "quantity"]):
        if q.startswith("dS_"):
            continue
        def nanmed(col):
            s = g[col].to_numpy(dtype=np.float64) if col in g.columns else np.array([])
            return float(np.nanmedian(s)) if np.isfinite(s).any() else np.nan
        rows.append({"assumption": 2, "what": "readable_units", "comparison": comp, "family": fam, "K": K, "kall_rule": rule, "quantity": q, "n_units": len(g), "n_not_readable": int((~g.is_readable).sum()),
                     "n_flat": int(g.get("flag_flat_method", pd.Series(False, index=g.index)).sum() + g.get("flag_flat_none", pd.Series(False, index=g.index)).sum()),
                     "median_frac_cens_method": nanmed("frac_cens_method"), "median_frac_cens_none": nanmed("frac_cens_none")})
    for (comp, fam, K, rule), g in Um.groupby(["comparison", "family", "K", "kall_rule"]):
        def sign_agree(qa, qb):
            a = g[g.quantity == qa].set_index(["run", "corpus"]).point
            b = g[g.quantity == qb].set_index(["run", "corpus"]).point
            ix = a.index.intersection(b.index)
            return (float(np.mean(np.sign(a[ix]) == np.sign(b[ix]))) if len(ix) else np.nan), len(ix)
        for qa, qb, n_ in (("dir", "dir_auc", 3), ("dir", "dir_bfix", 3), ("pos", "pos_bfix", 4), ("pos", "posloss", 4)):
            ag, n = sign_agree(qa, qb)
            rows.append({"assumption": n_, "what": f"sign_agreement_{qa}_vs_{qb}", "comparison": comp, "family": fam, "K": K, "kall_rule": rule, "n": n, "agreement": ag})
    rows.append({"assumption": 4, "what": "E75_position_model_fit", "note": "E75 7 節：位置の側の 2 つの山の模型は 8 種類とも基準 1/3 を満たさなかった（S_pos_bfix を並べる）"})
    if len(V):
        for rule in KALL_RULES:
            col = f"K_dir_method_{rule}"
            vm = V[V.comparison == MAIN_COMP]
            rows.append({"assumption": 5, "what": f"method_K_between_50_and_all_{rule}", "quantity": "dir", "n_groups": int(vm[col].notna().sum()), "n_above_50": int((vm[col] > 50).sum())})
    for (fam, rule), g in Cv2[Cv2.K_label == "all"].groupby(["family", "kall_rule"]):
        rows.append({"assumption": 5, "what": "K_all_value", "family": fam, "kall_rule": rule, "K_all_median": float(g.K_value.median()), "K_all_min": float(g.K_value.min()), "K_all_max": float(g.K_value.max())})
    rows.append({"assumption": 7, "what": "theory_assumptions", "note": "教師（oursx が選ぶ疑似ラベルのモデル）は配備する分類器のモダリティと K 本を使う（条件つき独立と K 本との独立は保証されない）"})
    write_csv(pd.DataFrame(rows), out / "assumptions.csv")


def write_summary_md(out, Fm, J):
    lines = ["# E87 の出力の表（解釈は書かない）", ""]
    sub = Fm[(Fm.variant == "readable") & (Fm.kall_rule == "mean")]
    for comp in sorted(sub.comparison.unique()):
        lines.append(f"## {comp}")
        for fam in sorted(sub.family.unique()):
            s2 = sub[(sub.comparison == comp) & (sub.family == fam)]
            if s2.empty:
                continue
            lines.append(f"### {fam}")
            lines.append("| 座標 | K | 読めるコーパス | 中央値 | 2.5% | 97.5% | 正のコーパス |")
            lines.append("|---|---|---|---|---|---|---|")
            for r_ in s2.sort_values(["quantity", "K"]).itertuples():
                lines.append(f"| {r_.quantity} | {r_.K} | {r_.n_corpora_used}/{r_.n_corpora_total} | {getattr(r_, 'median', float('nan')):.2f} | {getattr(r_, 'lo', float('nan')):.2f} | {getattr(r_, 'hi', float('nan')):.2f} | {getattr(r_, 'n_corp_pos', 0)} |")
            lines.append("")
    lines.append("## 判定")
    if len(J):
        cols = [c for c in J.columns if c in ("family", "comparison", "kall_rule", "variant", "coordinates", "P87_1_holds", "P87_1_holds_point", "P87_1_status", "P87_2_holds", "P87_2_status", "P87_3_holds", "P87_3_holds_point", "P87_3_status", "contrast_allowed")]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r_ in J[cols].itertuples(index=False):
            lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r_) + " |")
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
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", nargs="+", help="試走用．評価のターゲットを限る")
    ap.add_argument("--runs", nargs="+", choices=RUNS, help="試走用．世界を限る")
    ap.add_argument("--smoke", action="store_true", help="試走（--boot 50 と同じ扱い）")
    args = ap.parse_args()
    if args.smoke:
        args.boot = min(args.boot, 50)
    out = args.out.resolve()
    partial = bool(args.smoke or args.targets or args.runs)
    if partial and out.is_relative_to(ROOT / "results"):
        raise SystemExit("試走（--smoke，--targets）の出力は results/ の下に書かない．--out を checks/E87/ か scratchpad にする（I8）")
    if not partial and out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} が空でない．消すか別の --out にしてから流す")
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--"] + list(SCRIPT_DEPS), cwd=ROOT, capture_output=True, text=True).stdout.strip()
    HEADER = {"E": "E87", "commit": commit, "dirty": bool(dirty), "script_sha256": E.sha256(Path(__file__))[:16], "boot": args.boot, "seed": args.seed, "smoke": args.smoke,
              "targets": args.targets, "curve_root": str(args.curve_root), "started": started.isoformat(timespec="seconds")}
    prov = {"commit": commit, "dirty_deps": bool(dirty), "dirty_deps_status": dirty, "script_sha256": E.sha256(Path(__file__)),
            "deps_sha256": {rel: E.sha256(ROOT / rel) for rel in SCRIPT_DEPS}, "args": vars(args) | {"out": str(out), "curve_root": str(args.curve_root)},
            "constants": {"GEFF_MIN": GEFF_MIN, "HOLD_RATIO": HOLD_RATIO, "POS_RES_FRAC": POS_RES_FRAC, "K_GRID": list(K_GRID), "C_GRID": list(C_GRID), "KALL_RULES": list(KALL_RULES),
                          "K_all_rule": {"mean": "n_tgt/4（反復の平均．E65 と同じ）", "min": "学習プールで最も少ない感情の本数（反復の平均）"},
                          "interpolation": "log K について線形．平らな区間では届く最小の K．K=1 の値より悪い（等号を含まない）→ 1 に置いて下の打ち切り．全件の値より良い → 全件に置いて上の打ち切り．整えて持ち上げた点で読んだら平らに落ちた印",
                          "monotone": "S_uar・S_dir・S_dir_* は累積の最大，S_pos・S_posloss・S_pos_bfix は累積の最小（反復で平均した曲線に当てる）",
                          "draws": "種は (seed, crc32(コーパス))．話者の多項分布 (B, nls) と反復の多項分布 (B, R)．世界と比較相手で共有",
                          "QUANTS_B": list(QUANTS_B), "QUANTS_PT": list(QUANTS_PT), "DS": list(DS), "BETTER_HIGH": BETTER_HIGH, "KS": list(KS), "REPS": list(REPS),
                          "SEEDED": sorted(E.SEEDED), "EXCL_MA": list(EXCL_MA), "readable_rule": "主の右端の置き方で，none と手法の読みがどの群でも打ち切りでも平らに落ちてもいない",
                          "family_min_readable": "max(2, ceil(コーパス数/2))"},
            "comparisons": [c for _, c in COMPARISONS], "direct": [n for _, _, n in DIRECT], "main_comparison": MAIN_COMP, "started": started.isoformat(timespec="seconds"), "inputs": {}}
    prov["code_snapshot"] = str(snapshot_code(out, started.strftime("%Y%m%d-%H%M%S")).relative_to(out))
    Cv, Cr, V, U, Ch, Sd, boot_store, jk_store = run_all(args, out, prov)
    write_csv(Cv, out / "curves.csv")
    write_csv(Cr, out / "curves_rep.csv")
    write_csv(V, out / "values.csv")
    write_csv(Sd, out / "values_seed.csv")
    write_csv(U, out / "units.csv")
    write_csv(Ch if len(Ch) else pd.DataFrame([{"check": "none"}]), out / "checks.csv")
    Ud, dstore = direct_units(U, boot_store)
    write_csv(Ud, out / "direct.csv")
    Cp, Fm, fstore = corpus_and_family(U, boot_store)
    Cpd, Fd, fdstore = corpus_and_family(Ud, dstore) if len(Ud) else (pd.DataFrame(), pd.DataFrame(), {})
    write_csv(Cp, out / "corpus.csv")
    write_csv(Fm, out / "family.csv")
    if len(Fd):
        write_csv(Cpd, out / "direct_corpus.csv")
        write_csv(Fd, out / "direct_family.csv")
    J = judge(Fm, fstore, Fd, fdstore)
    write_csv(J, out / "judgments.csv")
    write_csv(judge_563188d(U), out / "judgments_563188d.csv")
    np.savez_compressed(out / "boot_units.npz", **boot_store)
    np.savez_compressed(out / "boot_family.npz", **fstore, **{f"direct|{k}": v for k, v in fdstore.items()})
    if jk_store:
        np.savez_compressed(out / "jk_units.npz", **jk_store)
    assumptions(out, Cv, U, V, Fm)
    write_summary_md(out, Fm, J)
    prov["n_units"] = int(len(U))
    prov["checks"] = Ch.to_dict("records") if len(Ch) else []
    prov["n_adjusted_points"] = int(Cv.adjusted.sum()) if len(Cv) else 0
    prov["n_units_not_readable"] = int((~U.is_readable).sum()) if len(U) else 0
    prov["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
