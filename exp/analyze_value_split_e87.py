"""E87：疑似ラベルのモデルの価値（ターゲットのラベル何本ぶん）を，向きの分と位置の分に分ける．

設計：docs/exploratory-analyses.md の E87（規則は値を見る前に書いた）．新しい学習はしない．評価のターゲットの値で判定するので本番（関門 1 は 5 体）．

  曲線：ターゲットだけで学んだ分類器（E76 の ①．K = 1, 5, 10, 25, 50, 全件．選んだ C）のテストの点数から，座標ごとの学習曲線を反復ごとに作る．
        S_uar（UAR），S_dir（6 対の A(d, 0) の平均．向きの正解率），S_pos（6 対の o の 2 乗平均の平方根．閾値のずれ）．副：S_posloss（E79 の位置の損），S_dir_auc（AUC から作った d．点だけ）．
  読み：none と手法（E79 の行．K = 0, 5, 25）の同じ座標の値を曲線に当てて本数 K' に読み，n_x = K'_x(手法) − K'_x(none) を出す．
  単位と区間は E79・E83 と同じ（世界，配備する分類器，K，コーパス．話者と反復の 2 段のブートストラップ 1,000 回．G_eff < 20 は話者の jackknife との和集合）．

使い方：./.venv/bin/python exp/analyze_value_split_e87.py [--out results/e87] [--boot 1000] [--targets ...] [--smoke]
  試走（--smoke，--targets）の出力は results/ の下に書けない（--out を checks/E87/ か scratchpad にする）．本番は既定の引数で 1 回で流す．
  出力の CSV は先頭に `#` の行（コミットと設定）を持つので，読むときは pd.read_csv(..., comment="#") にする．
"""
import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import analyze_pairs_e83 as E  # noqa: E402  （d・o，ブートストラップの抽出，行の対応，選んだ鍵は E83 と同じ関数を使う）
from explore_pairs import ba  # noqa: E402
from explore_synergy import PAIRS  # noqa: E402
from multimodal_distill import WORLDS  # noqa: E402

RUNS = ("fuse-audio-video-text", "fuse-audio-video")
KS = ("0", "5", "25")
REPS = tuple(range(10))
K_GRID = ("1", "5", "10", "25", "50", "all")                                   # E76 の ① の K の並び（_meta と照合する）
C_GRID = (0.01, 0.1, 1.0, 10.0)                                                # E76 の C の格子（_meta と照合する）
QUANTS = ("uar", "dir", "pos", "posloss")                                      # 区間を作る座標（pos が位置の主，posloss は感度）
QUANTS_PT = ("dir_auc",)                                                       # 点の値だけの座標（感度 (b)）
BETTER_HIGH = {"uar": True, "dir": True, "dir_auc": True, "pos": False, "posloss": False}
COMPARISONS = (("oursx", "none-oursx"), ("kd", "none-kd"), ("abl_ownK", "none-abl_ownK"), ("groupB", "none-groupB"))
MAIN_COMP = "none-oursx"
GEFF_MIN = E.GEFF_MIN
MAIN_FAMILY = ("audio_wavlm",)                                                 # 主の族：音声の配備する分類器（AVT と AV を合わせる）
HOLD_RATIO = 0.5                                                               # P87-1・P87-3 の基準（E66 の P2 と同じ 0.5）
NOT_READABLE_FRAC = 0.5                                                        # 読めない単位：打ち切りの反復が半数以上
HEADER = "E87"
SCRIPT_DEPS = ("exp/analyze_value_split_e87.py", "exp/analyze_pairs_e83.py", "exp/explore_pairs.py", "exp/explore_synergy.py", "exp/multimodal_distill.py", "measure/data.py")


def write_csv(df, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {HEADER}\n")
        df.to_csv(fh, index=False)


# ---------------------------------------------------------------- 点数（座標）

def d_auc_pairs(L, y):
    """対ごとの AUC（z = L_k − L_j の順位）から d_AUC = √2·Φ⁻¹(AUC)（模型に依らない向きの感度．点の値だけ）．"""
    out = np.full(6, np.nan)
    for p, (k, j) in enumerate(PAIRS):
        z = L[:, k] - L[:, j]
        zk, zj = z[y == k], z[y == j]
        if len(zk) < 2 or len(zj) < 2:
            continue
        ranks = stats.rankdata(np.concatenate([zk, zj]))
        auc = (ranks[:len(zk)].sum() - len(zk) * (len(zk) + 1) / 2) / (len(zk) * len(zj))
        out[p] = np.sqrt(2) * stats.norm.ppf(np.clip(auc, 1e-6, 1 - 1e-6))
    return out


def scores_point(L, y):
    rec = E.recall_by_emotion(L, y)
    d, o = E.d_o_pairs(L, y)
    a0 = ba(d, 0.0)
    return {"uar": 100 * float(np.nanmean(rec)), "dir": 100 * float(np.nanmean(a0)), "pos": float(np.sqrt(np.nanmean(o ** 2))),
            "posloss": 100 * float(np.nanmean(a0 - ba(d, o))), "dir_auc": 100 * float(np.nanmean(ba(d_auc_pairs(L, y), 0.0)))}, d, o


def feats_all(L, y, sidx, nls):
    """話者ごとの集計：対ごとの d・o の材料（E83 の pair_feats）と，感情ごとの正解と総数（UAR の材料）．"""
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
    """W：(B, nls) の話者の重み．戻り：座標ごとに (B,)．空の抽出や感情の欠けは nan（E79 と同じ扱い）．"""
    d, o = E.pair_d_o_from_feats(F, W)
    corr, tot = W @ U[:, :, 0], W @ U[:, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(tot > 0, corr / tot, np.nan)
        a0 = ba(d, 0.0)
        return {"uar": 100 * np.nanmean(rec, 1), "dir": 100 * np.nanmean(a0, 1), "pos": np.sqrt(np.nanmean(o ** 2, 1)),
                "posloss": 100 * np.nanmean(a0 - ba(d, o), 1)}


# ---------------------------------------------------------------- 曲線と読み

def monotone(curve, better_high):
    """K について単調に整える（良くなる向きに累積）．戻り：整えた曲線と整えた点の印．"""
    s = curve if better_high else -curve
    s_adj = np.maximum.accumulate(s, axis=-1)
    adjusted = ~np.isclose(s_adj, s, rtol=0, atol=1e-12)
    return (s_adj if better_high else -s_adj), adjusted


def read_K(curve, logk, v, better_high, adjust=True):
    """曲線（…, 6）に値 v（…）を当てて本数 K' を読む．平らな区間では届く最小の K．
    K = all の値より良ければ K = all に置いて上の打ち切り，K = 1 の値より悪ければ 1 に置いて下の打ち切り．戻り：(K', 下の打ち切り, 上の打ち切り)．"""
    s = curve if better_high else -curve
    vv = v if better_high else -v
    if adjust:
        s = np.maximum.accumulate(s, axis=-1)
    logk = np.broadcast_to(logk, s.shape)
    vv = np.asarray(vv, dtype=np.float64)
    ge = s >= vv[..., None]
    any_ge = ge.any(-1)
    i = np.clip(np.argmax(ge, -1), 1, s.shape[-1] - 1)
    s_prev, s_cur = np.take_along_axis(s, (i - 1)[..., None], -1)[..., 0], np.take_along_axis(s, i[..., None], -1)[..., 0]
    lk_prev, lk_cur = np.take_along_axis(logk, (i - 1)[..., None], -1)[..., 0], np.take_along_axis(logk, i[..., None], -1)[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(s_cur > s_prev, (vv - s_prev) / (s_cur - s_prev), 1.0)
    lk = lk_prev + np.clip(frac, 0, 1) * (lk_cur - lk_prev)
    lk = np.where(ge[..., 0], logk[..., 0], lk)                                # K = 1 の値以下（含む）→ K' = 1
    lk = np.where(any_ge, lk, logk[..., -1])                                   # 右端より良い → K = all
    cens_lo = vv < s[..., 0]
    cens_hi = ~any_ge
    K = np.exp(lk)
    bad = ~np.isfinite(vv) | ~np.isfinite(s).all(-1)
    K = np.where(bad, np.nan, K)
    return K, cens_lo & ~bad, cens_hi & ~bad


# ---------------------------------------------------------------- E76（曲線の材料）

def load_e76(st, T, yall, te, prov, check_rows):
    """E76 の ① のテストの点数（K 6 × 選んだ C）を E77 の _te の並びで返す．K = all の本数（n_tgt/4）と選んだ C も返す．"""
    f = ROOT / "results" / "e76" / st / f"{T}.npz"
    fc = ROOT / "results" / "e76" / st / f"{T}.csv"
    E.note_input(prov, f)
    E.note_input(prov, fc, with_sha=True)
    z = np.load(f)
    meta = json.loads(str(z["_meta"]))
    st_ = meta["settings"]
    if list(st_["K"]) != list(K_GRID) or [float(c) for c in st_["C_grid"]] != list(C_GRID) or list(st_["reps"]) != list(REPS):
        raise SystemExit(f"{st} {T}：E76 の K・C・反復の並びが台本の約束と違う：{st_['K']} {st_['C_grid']} {st_['reps']}")
    if not np.array_equal(np.asarray(z["y"]).astype(int), yall):
        raise SystemExit(f"{st} {T}：E76 の y が load_any のラベルと違う")
    csv = pd.read_csv(fc, comment="#", dtype={"K": str, "rep": int})
    tg = csv[csv.method == "tgt_only"]
    out = {}
    for r in REPS:
        idx = np.asarray(z[f"test_idx_r{r}"]).ravel()
        if sorted(idx.tolist()) != sorted(np.asarray(te[r]).ravel().tolist()):
            check_rows.append({"check": "e76_mismatch", "student": st, "target": T, "rep": r})
            raise SystemExit(f"{st} {T} r{r}：E76 のテストの行が E77 と違う")
        pos = {int(a): i for i, a in enumerate(idx)}
        order = np.array([pos[int(a)] for a in np.asarray(te[r]).ravel()])
        Cs = np.asarray(z[f"t_C_r{r}"], dtype=np.float64)
        rule = [str(x) for x in z[f"t_rule_r{r}"]]
        Lk = []
        for iK, K in enumerate(K_GRID):
            row = tg[(tg.K == K) & (tg.rep == r)]
            if len(row) != 1:
                raise SystemExit(f"{st} {T} r{r} K{K}：E76 の csv の tgt_only の行が {len(row)} 行")
            if not np.isclose(float(row.C.iloc[0]), Cs[iK]):
                raise SystemExit(f"{st} {T} r{r} K{K}：E76 の csv の C と npz の C が違う")
            iC = int(np.flatnonzero(np.isclose(np.array(C_GRID), Cs[iK]))[0])
            L = np.asarray(z[f"t_logit_r{r}"][iK, iC], dtype=np.float64)[order]
            if not np.isfinite(L).all():
                raise SystemExit(f"{st} {T} r{r} K{K}：E76 の点数に NaN/Inf")
            Lk.append(L)
        n_all = int(tg[(tg.K == "all") & (tg.rep == r)].n_tgt.iloc[0])
        kv = np.array([1.0, 5.0, 10.0, 25.0, 50.0, n_all / 4.0])
        if not (np.diff(kv) > 0).all():
            raise SystemExit(f"{st} {T} r{r}：K = all の本数 {n_all / 4} が 50 以下")
        out[r] = {"L": Lk, "kv": kv, "logk": np.log(kv), "C": Cs, "rule": rule, "n_all": n_all}
    return out


# ---------------------------------------------------------------- 本体

def run_all(args, out, prov):
    sel_path = ROOT / "results" / "e79" / "select" / "chosen.csv"
    E.note_input(prov, sel_path, with_sha=True)
    sel = pd.read_csv(sel_path, comment="#", dtype={"K": str, "param": str, "meff": str})
    curve_rows, val_rows, unit_rows, check_rows, adj_rows = [], [], [], [], []
    boot_store, jk_store = {}, {}
    for run in RUNS:
        d77 = ROOT / "results" / f"e77_{run}"
        pf = ROOT / "results" / "e79" / "measure" / f"pairs_{run}.csv.gz"
        E.note_input(prov, pf)
        P = pd.read_csv(pf, dtype={"row_id": int, "rep": int, "pair": int})
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
                e76 = {st: load_e76(st, T, yall, te, prov, check_rows) for st in students}
                prep[T] = {"X": X, "y": yall, "spk": spk_all, "te": te, "logp": logp, "e76": e76}
            speakers = sorted(set().union(*[set(p["spk"][p["te"][r]]) for p in prep.values() for r in REPS]))
            sid = {s: i for i, s in enumerate(speakers)}
            nls = len(speakers)
            C, A = E.boot_draws(run, cp, speakers, args.boot, args.seed, len(REPS))
            Wj = np.ones((nls, nls)) - np.eye(nls)
            sess, Wj_s = None, None
            if cp == "IEMOCAP":
                sessions = sorted({E.session_of(s) for s in speakers})
                sess = {s: sessions.index(E.session_of(s)) for s in speakers}
                Wj_s = np.ones((len(sessions), len(sessions))) - np.eye(len(sessions))
            geff = {}
            for T, p in prep.items():
                cnt = pd.Series(np.concatenate([p["spk"][p["te"][r]] for r in REPS])).value_counts().to_numpy().astype(float)
                geff[T] = cnt.sum() ** 2 / (cnt ** 2).sum()
            prov.setdefault("g_eff", {}).update({f"{run}|{T}": float(v) for T, v in geff.items()})
            need_jk = min(geff.values()) < GEFF_MIN
            n_sess = len(set(sess.values())) if sess else 0
            # ---- 曲線（配備する分類器 × T × 反復．K と比較に依らない）
            curves = {}
            for T, p in prep.items():
                for st in students:
                    for r in REPS:
                        yte = p["y"][p["te"][r]]
                        sidx = np.array([sid[s] for s in p["spk"][p["te"][r]]])
                        ssidx = np.array([sess[s] for s in p["spk"][p["te"][r]]]) if sess is not None else None
                        e = p["e76"][st][r]
                        pts = {q: np.zeros(6) for q in QUANTS + QUANTS_PT}
                        FU = []
                        FU_s = []
                        for iK in range(6):
                            sc, _, _ = scores_point(e["L"][iK], yte)
                            for q in pts:
                                pts[q][iK] = sc[q]
                            FU.append(feats_all(e["L"][iK], yte, sidx, nls))
                            if sess is not None:
                                FU_s.append(feats_all(e["L"][iK], yte, ssidx, Wj_s.shape[0]))
                            curve_rows.append({"run": run, "target": T, "student": st, "rep": r, "K_label": K_GRID[iK], "K_value": e["kv"][iK], "C": e["C"][iK],
                                               "C_rule": e["rule"][iK], **{f"S_{q}": sc[q] for q in pts}})
                        adj = {}
                        for q in pts:
                            m_, a_ = monotone(pts[q], BETTER_HIGH[q])
                            adj[q] = m_
                            adj_rows.append({"run": run, "target": T, "student": st, "rep": r, "quantity": q, "n_adjusted": int(a_.sum()),
                                             "max_adjustment": float(np.abs(m_ - pts[q]).max())})
                        wB = {q: np.zeros((len(C), 6)) for q in QUANTS}
                        wJ = {q: np.zeros((nls, 6)) for q in QUANTS} if need_jk else None
                        wS = {q: np.zeros((Wj_s.shape[0], 6)) for q in QUANTS} if sess is not None else None
                        for iK in range(6):
                            s_ = scores_w(FU[iK][0], FU[iK][1], C)
                            for q in QUANTS:
                                wB[q][:, iK] = s_[q]
                            if need_jk:
                                s_ = scores_w(FU[iK][0], FU[iK][1], Wj)
                                for q in QUANTS:
                                    wJ[q][:, iK] = s_[q]
                            if sess is not None:
                                s_ = scores_w(FU_s[iK][0], FU_s[iK][1], Wj_s)
                                for q in QUANTS:
                                    wS[q][:, iK] = s_[q]
                        curves[(T, st, r)] = {"pts": pts, "adj": adj, "B": wB, "J": wJ, "S": wS, "logk": e["logk"], "kv": e["kv"], "yte": yte, "sidx": sidx, "ssidx": ssidx}
            # ---- none と手法
            for selector, comp in COMPARISONS:
                for st in students:
                    for K in KS:
                        chs = {T: E.chosen_setting(sel, run, T, st, K, selector) for T in Ts}
                        if any(c == "missing" for c in chs.values()):
                            check_rows.append({"check": "chosen_missing", "comparison": comp, "run": run, "corpus": cp, "student": st, "K": K})
                            continue
                        if len({c for c in chs.values()}) > 1:
                            check_rows.append({"check": "chosen_differs_in_corpus", "comparison": comp, "run": run, "corpus": cp, "student": st, "K": K,
                                               "chosen": ";".join(f"{T}:{'none' if c is None else '|'.join(c)}" for T, c in chs.items())})
                            if comp == MAIN_COMP:
                                raise SystemExit(f"{run} {cp} {st} K{K}：IEMOCAP の 2 群で {selector} の選択が違う（主の比較なので止める）")
                        ncands = {T: E.n_cand_of(sel, run, T, st, K, selector) for T in Ts}
                        per_T = {}
                        for T in Ts:
                            per_T[T] = values_target(run, T, prep[T], curves, chs[T], st, K, comp, sid, nls, C, Wj, need_jk, sess, Wj_s, val_rows, check_rows)
                        all_none = all(c is None for c in chs.values())
                        chosen_short = "none" if all_none else "|".join(next(c for c in chs.values() if c is not None))
                        meta = {"chosen": chosen_short, "n_groups_none": sum(c is None for c in chs.values()), "n_cand_min": min(ncands.values()),
                                "n_seeds": max((len(per_T[T]["seeds"]) for T in Ts), default=0)}
                        specs = [("main", Ts, "jk")]
                        if run == "fuse-audio-video-text":
                            specs.append(("iemocap_impro_only", [T for T in Ts if T != "IEMOCAP-script"], "jk"))
                        if sess is not None:
                            specs.append(("session_jk", Ts, "jk_sess"))
                        for stag, Ts_use, jkkind in specs:
                            if Ts_use:
                                aggregate_unit(run, cp, st, K, comp, stag, jkkind, Ts_use, per_T, A, geff, n_sess, all_none, meta, unit_rows, boot_store, jk_store)
    return (pd.DataFrame(curve_rows), pd.DataFrame(val_rows), pd.DataFrame(unit_rows), pd.DataFrame(check_rows), pd.DataFrame(adj_rows), boot_store, jk_store)


def values_target(run, T, p, curves, chT, st, K, comp, sid, nls, C, Wj, need_jk, sess, Wj_s, val_rows, check_rows):
    """1 つのターゲット（群）：反復ごとの none・手法の点数と K'，n_x（点，ブートストラップ，jackknife）．"""
    X = p["X"]
    R = len(REPS)
    allq = QUANTS + QUANTS_PT
    res = {"pts": {q: np.full(R, np.nan) for q in allq}, "pts_nomono": {q: np.full(R, np.nan) for q in allq},
           "readable": {q: np.full(R, np.nan) for q in allq}, "boot": {q: {} for q in QUANTS}, "jk": {q: {} for q in QUANTS}, "jk_sess": {q: {} for q in QUANTS},
           "cens_boot": {q: {} for q in QUANTS}, "seeds": set()}
    for r in REPS:
        cv = curves[(T, st, r)]
        yte = cv["yte"]
        xn = X[(X.rep == r) & (X.student == st) & (X.K == K) & (X.method == "none") & (X.param == "-")].sort_values("pair")
        if len(xn) != 6:
            check_rows.append({"check": "none_rows", "comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "n": len(xn)})
            continue
        Ln = E.logp_row(p["logp"], "e77", r, int(xn.row_id.iloc[0]))
        sn, dn, on = scores_point(Ln, yte)
        mx, mism = E.max_abs_or_nan_mismatch(np.concatenate([dn, on]), np.concatenate([xn.d.to_numpy(), xn.o.to_numpy()]))
        if mx > 1e-6 or mism:
            raise SystemExit(f"{run} {T} {st} K{K} r{r} none：対数の確率から計算し直した d・o が pairs の表と合わない（最大 {mx:.2e}，NaN の食い違い {mism}）")
        Kn, Kn_nomono, cens_n = {}, {}, {}
        for q in allq:
            Kn[q], lo_, hi_ = read_K(cv["adj"][q], cv["logk"], np.array(sn[q]), BETTER_HIGH[q], adjust=False)
            Kn_nomono[q], _, _ = read_K(cv["pts"][q], cv["logk"], np.array(sn[q]), BETTER_HIGH[q], adjust=False)
            cens_n[q] = bool(lo_ or hi_)
        FnU = feats_all(Ln, yte, cv["sidx"], nls)
        FnU_s = feats_all(Ln, yte, cv["ssidx"], Wj_s.shape[0]) if sess is not None else None
        base = {"comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "n_test": int(len(yte)), "K_all_value": float(cv["kv"][-1])}
        if chT is None:
            for q in allq:
                res["pts"][q][r] = 0.0
                res["pts_nomono"][q][r] = 0.0
                res["readable"][q][r] = 1.0
            for q in QUANTS:
                res["boot"][q][r] = np.zeros(len(C))
                res["cens_boot"][q][r] = 0.0
                if need_jk:
                    res["jk"][q][r] = np.zeros(nls)
                if sess is not None:
                    res["jk_sess"][q][r] = np.zeros(Wj_s.shape[0])
            val_rows.append({**base, "chosen": "none", "n_seeds": 0, **{f"S_{q}_none": sn[q] for q in allq}, **{f"K_{q}_none": float(Kn[q]) for q in allq},
                             **{f"n_{q}": 0.0 for q in allq}, **{f"cens_{q}_none": cens_n[q] for q in allq}, **{f"cens_{q}_method": False for q in allq}})
            continue
        wave, meth, par, me = chT
        xo_all = X[(X.rep == r) & (X.student == st) & (X.K == K) & (X.wave == wave) & (X.method == meth) & (X.pnorm == par) & (X.meff == me)]
        seeds = sorted(xo_all.seed.unique())
        if not seeds or any(len(xo_all[xo_all.seed == s_]) != 6 for s_ in seeds):
            check_rows.append({"check": "ours_rows", "comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "n": len(xo_all), "chosen": "|".join(chT)})
            continue
        res["seeds"].update(int(s_) for s_ in seeds)
        acc = {"pts": {q: [] for q in allq}, "nomono": {q: [] for q in allq}, "cens": {q: [] for q in allq}, "S": {q: [] for q in allq},
               "boot": {q: [] for q in QUANTS}, "jk": {q: [] for q in QUANTS}, "jk_sess": {q: [] for q in QUANTS}, "cens_boot": {q: [] for q in QUANTS}}
        sB_n = scores_w(FnU[0], FnU[1], C)
        KB_n = {q: read_K(cv["B"][q], cv["logk"], sB_n[q], BETTER_HIGH[q]) for q in QUANTS}
        if need_jk:
            sJ_n = scores_w(FnU[0], FnU[1], Wj)
            KJ_n = {q: read_K(cv["J"][q], cv["logk"], sJ_n[q], BETTER_HIGH[q])[0] for q in QUANTS}
        if sess is not None:
            sS_n = scores_w(FnU_s[0], FnU_s[1], Wj_s)
            KS_n = {q: read_K(cv["S"][q], cv["logk"], sS_n[q], BETTER_HIGH[q])[0] for q in QUANTS}
        for s_ in seeds:
            xo = xo_all[xo_all.seed == s_].sort_values("pair")
            Lo = E.logp_row(p["logp"], wave, r, int(xo.row_id.iloc[0]))
            so, do_, oo = scores_point(Lo, yte)
            mx, mism = E.max_abs_or_nan_mismatch(np.concatenate([do_, oo]), np.concatenate([xo.d.to_numpy(), xo.o.to_numpy()]))
            if mx > 1e-6 or mism:
                raise SystemExit(f"{run} {T} {st} K{K} r{r} {comp} seed{s_}：対数の確率から計算し直した d・o が pairs の表と合わない（最大 {mx:.2e}，NaN の食い違い {mism}）")
            for q in allq:
                Ko, lo_, hi_ = read_K(cv["adj"][q], cv["logk"], np.array(so[q]), BETTER_HIGH[q], adjust=False)
                Ko2, _, _ = read_K(cv["pts"][q], cv["logk"], np.array(so[q]), BETTER_HIGH[q], adjust=False)
                acc["pts"][q].append(float(Ko - Kn[q]))
                acc["nomono"][q].append(float(Ko2 - Kn_nomono[q]))
                acc["cens"][q].append(bool(lo_ or hi_))
                acc["S"][q].append(so[q])
            FoU = feats_all(Lo, yte, cv["sidx"], nls)
            sB_o = scores_w(FoU[0], FoU[1], C)
            for q in QUANTS:
                Ko, lo_, hi_ = read_K(cv["B"][q], cv["logk"], sB_o[q], BETTER_HIGH[q])
                acc["boot"][q].append(Ko - KB_n[q][0])
                acc["cens_boot"][q].append(float(np.nanmean(lo_ | hi_ | KB_n[q][1] | KB_n[q][2])))
            if need_jk:
                sJ_o = scores_w(FoU[0], FoU[1], Wj)
                for q in QUANTS:
                    acc["jk"][q].append(read_K(cv["J"][q], cv["logk"], sJ_o[q], BETTER_HIGH[q])[0] - KJ_n[q])
            if sess is not None:
                FoU_s = feats_all(Lo, yte, cv["ssidx"], Wj_s.shape[0])
                sS_o = scores_w(FoU_s[0], FoU_s[1], Wj_s)
                for q in QUANTS:
                    acc["jk_sess"][q].append(read_K(cv["S"][q], cv["logk"], sS_o[q], BETTER_HIGH[q])[0] - KS_n[q])
        for q in allq:
            res["pts"][q][r] = float(np.mean(acc["pts"][q]))                   # 種は平均する（E79・E83 と同じ）
            res["pts_nomono"][q][r] = float(np.mean(acc["nomono"][q]))
            res["readable"][q][r] = float(not (cens_n[q] or any(acc["cens"][q])))
        for q in QUANTS:
            res["boot"][q][r] = np.mean(np.stack(acc["boot"][q]), 0)
            res["cens_boot"][q][r] = float(np.mean(acc["cens_boot"][q]))
            if need_jk:
                res["jk"][q][r] = np.mean(np.stack(acc["jk"][q]), 0)
            if sess is not None:
                res["jk_sess"][q][r] = np.mean(np.stack(acc["jk_sess"][q]), 0)
        val_rows.append({**base, "chosen": "|".join(chT), "n_seeds": len(seeds),
                         **{f"S_{q}_none": sn[q] for q in allq}, **{f"S_{q}_method": float(np.mean(acc["S"][q])) for q in allq},
                         **{f"K_{q}_none": float(Kn[q]) for q in allq}, **{f"K_{q}_method": float(Kn[q] + res["pts"][q][r]) for q in allq},
                         **{f"n_{q}": res["pts"][q][r] for q in allq}, **{f"n_{q}_nomono": res["pts_nomono"][q][r] for q in allq},
                         **{f"cens_{q}_none": cens_n[q] for q in allq}, **{f"cens_{q}_method": any(acc["cens"][q]) for q in allq},
                         **{f"n_start_{q}": float(Kn[q]) - float(K) for q in allq}})
    return res


def aggregate_unit(run, cp, st, K, comp, stag, jkkind, Ts_use, per_T, A, geff, n_sess, all_none, meta, unit_rows, boot_store, jk_store):
    """単位（コーパス）に集める：反復の平均を群で平均．区間は反復の重み A で反復を混ぜ，群で平均した複製の百分位．G_eff < 20 は話者の jackknife との和集合（E79・E83 と同じ）．"""
    gmin = float(min(geff[T] for T in Ts_use))
    for q in QUANTS + QUANTS_PT:
        x = float(np.nanmean([np.nanmean(per_T[T]["pts"][q]) for T in Ts_use]))
        x_nomono = float(np.nanmean([np.nanmean(per_T[T]["pts_nomono"][q]) for T in Ts_use]))
        readable = float(np.nanmean([np.nanmean(per_T[T]["readable"][q]) for T in Ts_use]))
        lo = hi = lo_j = hi_j = lo_s = hi_s = np.nan
        rule, nvalid, cens_b = "point_only", 0, np.nan
        b = th = None
        if q in QUANTS:
            if all(len(per_T[T]["boot"][q]) == len(REPS) for T in Ts_use):
                per = []
                for T in Ts_use:
                    arr = np.stack([per_T[T]["boot"][q][r] for r in REPS], 1)   # (B, R)
                    fin = np.isfinite(arr)
                    per.append(np.where(fin, arr, 0).__mul__(A).sum(1) / np.where(fin, A, 0).sum(1))
                b = np.nanmean(np.stack(per), 0)
                lo, hi = np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)
                nvalid = int(np.isfinite(b).sum())
                rule = "pct"
                cens_b = float(np.nanmean([np.nanmean([per_T[T]["cens_boot"][q][r] for r in REPS]) for T in Ts_use]))
                if gmin < GEFF_MIN:
                    if all(len(per_T[T]["jk"][q]) == len(REPS) for T in Ts_use):
                        th = np.nanmean(np.stack([np.nanmean(np.stack([per_T[T]["jk"][q][r] for r in REPS], 1), 1) for T in Ts_use]), 0)
                        lo_j, hi_j = E.jk_interval(x, th, gmin)
                        lo, hi = np.fmin(lo, lo_j), np.fmax(hi, hi_j)
                        rule = "union_pct_jk"
                    else:
                        rule, lo, hi = "pct_jk_missing", np.nan, np.nan
                if jkkind == "jk_sess" and rule != "pct_jk_missing":
                    if all(len(per_T[T]["jk_sess"][q]) == len(REPS) for T in Ts_use):
                        th = np.nanmean(np.stack([np.nanmean(np.stack([per_T[T]["jk_sess"][q][r] for r in REPS], 1), 1) for T in Ts_use]), 0)
                        lo_s, hi_s = E.jk_interval(x, th, float(n_sess))
                        lo, hi = np.fmin(lo, lo_s), np.fmax(hi, hi_s)
                        rule = "union_main_session_jk"
                    else:
                        rule, lo, hi = "pct_jk_missing", np.nan, np.nan
            else:
                rule = "no_replicates"
        key = f"{comp}|{run}|{st}|{K}|{cp}|{stag}|{q}"
        if b is not None:
            boot_store[key] = b.astype(np.float32)
        if th is not None:
            jk_store[key] = th.astype(np.float32)
        if all_none:
            jd = "same_by_definition"
        elif q in QUANTS_PT:
            jd = "point_only"
        elif not (np.isfinite(x) and np.isfinite(lo) and np.isfinite(hi)):
            jd = "missing"
        elif lo > 0:
            jd = "pos"
        elif hi < 0:
            jd = "neg"
        else:
            jd = "undet"
        unit_rows.append({"comparison": comp, "run": run, "role": E.role_of(run, st), "student": st, "K": K, "corpus": cp, "n_groups": len(Ts_use), "groups": ";".join(Ts_use),
                          "sensitivity": stag, "quantity": q, "point": x, "point_nomono": x_nomono, "lo": lo, "hi": hi, "lo_jk": lo_j, "hi_jk": hi_j,
                          "lo_session_jk": lo_s, "hi_session_jk": hi_s, "interval_rule": rule, "g_eff_min": gmin, "n_boot_valid": nvalid,
                          "readable_frac": readable, "is_readable": bool(readable >= NOT_READABLE_FRAC) if np.isfinite(readable) else False,
                          "frac_cens_boot": cens_b, "judgment": jd, "boot_key": key, **meta})


# ---------------------------------------------------------------- まとめと判定

def family_of(u):
    if u.student in MAIN_FAMILY:
        return "audio"
    return f"{u.role}:{u.student}:{u.run}"


def summarize(out, U, V, adj):
    U = U.copy()
    U["family"] = [family_of(u) for u in U.itertuples()]
    rows = []
    for (comp, fam, stag, q, K), g in U.groupby(["comparison", "family", "sensitivity", "quantity", "K"]):
        pts = g.point.to_numpy()
        rows.append({"comparison": comp, "family": fam, "sensitivity": stag, "quantity": q, "K": K, "n_units": len(g), "n_corpora_distinct": g.corpus.nunique(),
                     "median": float(np.nanmedian(pts)), "q1": float(np.nanpercentile(pts, 25)), "q3": float(np.nanpercentile(pts, 75)),
                     "min": float(np.nanmin(pts)), "max": float(np.nanmax(pts)), "median_nomono": float(np.nanmedian(g.point_nomono.to_numpy())),
                     **{f"n_{j}": int((g.judgment == j).sum()) for j in ("pos", "neg", "undet", "missing", "same_by_definition", "point_only")},
                     "n_not_readable": int((~g.is_readable).sum()), "median_readable_only": float(np.nanmedian(pts[g.is_readable.to_numpy()])) if g.is_readable.any() else np.nan,
                     "median_frac_cens_boot": float(np.nanmedian(g.frac_cens_boot.to_numpy()))})
    S = pd.DataFrame(rows)
    write_csv(S, out / "summary.csv")
    # 判定（3 節）
    J = []
    fams = sorted(S.family.unique())
    for fam in fams:
        for comp in sorted(S.comparison.unique()):
            for stag in sorted(S.sensitivity.unique()):
                for variant, qd, qp, qu in (("main", "dir", "pos", "uar"), ("posloss_as_position", "dir", "posloss", "uar"), ("dir_auc_point", "dir_auc", "pos", "uar")):
                    if variant != "main" and comp != MAIN_COMP:
                        continue
                    sub = S[(S.family == fam) & (S.comparison == comp) & (S.sensitivity == stag)]
                    def row(q, K):
                        r_ = sub[(sub.quantity == q) & (sub.K == K)]
                        return r_.iloc[0] if len(r_) == 1 else None
                    d0, d25, p5, p25, u0, u25 = row(qd, "0"), row(qd, "25"), row(qp, "5"), row(qp, "25"), row(qu, "0"), row(qu, "25")
                    def judgeable(*rs):
                        return all(r_ is not None and r_.n_not_readable * 2 <= r_.n_units for r_ in rs)
                    rec = {"family": fam, "comparison": comp, "sensitivity": stag, "variant": variant, "is_main_judgment": bool(fam == "audio" and comp == MAIN_COMP and stag == "main" and variant == "main")}
                    # P87-1
                    if judgeable(d0, d25):
                        rec.update({"P87_1_med_K0": d0["median"], "P87_1_med_K25": d25["median"], "P87_1_ratio": d25["median"] / d0["median"] if d0["median"] > 0 else np.nan,
                                    "P87_1_holds": bool(d0["median"] > 0 and d25["median"] >= HOLD_RATIO * d0["median"]), "P87_1_status": "judged"})
                    else:
                        rec.update({"P87_1_holds": None, "P87_1_status": "not_judgeable"})
                    # P87-2（区間の区分を使うので点だけの座標では判定できない）
                    if qp in QUANTS and judgeable(p5, p25):
                        rec.update({"P87_2_npos_K5": int(p5["n_pos"]), "P87_2_n_K5": int(p5["n_units"]), "P87_2_npos_K25": int(p25["n_pos"]), "P87_2_n_K25": int(p25["n_units"]),
                                    "P87_2_holds": bool(p5["n_pos"] * 2 < p5["n_units"] and p25["n_pos"] * 2 < p25["n_units"]), "P87_2_status": "judged",
                                    "P87_2_med_K0": row(qp, "0")["median"] if row(qp, "0") is not None else np.nan, "P87_2_med_K5": p5["median"], "P87_2_med_K25": p25["median"]})
                    else:
                        rec.update({"P87_2_holds": None, "P87_2_status": "not_judgeable"})
                    # P87-3
                    if judgeable(u0, u25):
                        rec.update({"P87_3_med_K0": u0["median"], "P87_3_med_K25": u25["median"], "P87_3_ratio": u25["median"] / u0["median"] if u0["median"] > 0 else np.nan,
                                    "P87_3_holds": bool(u0["median"] > 0 and u25["median"] < HOLD_RATIO * u0["median"]), "P87_3_status": "judged"})
                    else:
                        rec.update({"P87_3_holds": None, "P87_3_status": "not_judgeable"})
                    J.append(rec)
    Jd = pd.DataFrame(J)
    write_csv(Jd, out / "judgments.csv")
    # 仮定の表
    Aro = []
    adj2 = adj.copy()
    adj2["family"] = ["audio" if s in MAIN_FAMILY else f"{s}:{r}" for s, r in zip(adj2.student, adj2.run)]
    for (fam, q), g in adj2.groupby(["family", "quantity"]):
        Aro.append({"assumption": 1, "what": "monotone_adjustment", "family": fam, "quantity": q, "frac_points_adjusted": float(g.n_adjusted.sum() / (6 * len(g))),
                    "max_adjustment": float(g.max_adjustment.max()), "ok": bool(g.n_adjusted.sum() / (6 * len(g)) < 0.25)})
    for (comp, fam, stag, q, K), g in U.groupby(["comparison", "family", "sensitivity", "quantity", "K"]):
        if stag != "main":
            continue
        Aro.append({"assumption": 2, "what": "readable_units", "comparison": comp, "family": fam, "quantity": q, "K": K, "n_units": len(g), "n_not_readable": int((~g.is_readable).sum()),
                    "median_frac_cens_boot": float(np.nanmedian(g.frac_cens_boot.to_numpy())), "ok": bool((~g.is_readable).sum() * 2 < len(g))})
    um = U[(U.sensitivity == "main")]
    for (comp, fam, K), g in um.groupby(["comparison", "family", "K"]):
        a_ = g[g.quantity == "dir"].set_index("corpus").point
        b_ = g[g.quantity == "dir_auc"].set_index("corpus").point
        c_ = g[g.quantity == "pos"].set_index("corpus").point
        d_ = g[g.quantity == "posloss"].set_index("corpus").point
        ix = a_.index.intersection(b_.index)
        iy = c_.index.intersection(d_.index)
        Aro.append({"assumption": 3, "what": "sign_agreement_dir_vs_dir_auc", "comparison": comp, "family": fam, "K": K, "n": len(ix),
                    "agreement": float(np.mean(np.sign(a_[ix]) == np.sign(b_[ix]))) if len(ix) else np.nan})
        Aro.append({"assumption": 6, "what": "sign_agreement_pos_vs_posloss", "comparison": comp, "family": fam, "K": K, "n": len(iy),
                    "agreement": float(np.mean(np.sign(c_[iy]) == np.sign(d_[iy]))) if len(iy) else np.nan})
    if len(V):
        for (run, T), g in V.groupby(["run", "target"]):
            Aro.append({"assumption": 4, "what": "K_all_value", "run": run, "target": T, "K_all_value_median": float(np.nanmedian(g.K_all_value))})
        vm = V[V.comparison == MAIN_COMP]
        for q in QUANTS + QUANTS_PT:
            col = f"K_{q}_method"
            if col in vm.columns:
                n_hi = int(((vm[col] > 50) & vm[col].notna()).sum())
                Aro.append({"assumption": 4, "what": f"method_K_between_50_and_all_{q}", "n_reps_rows": int(vm[col].notna().sum()), "n_above_50": n_hi})
    write_csv(pd.DataFrame(Aro), out / "assumptions.csv")
    # summary.md（表だけ）
    lines = ["# E87 の出力の表（解釈は書かない）", ""]
    for comp in sorted(S.comparison.unique()):
        lines.append(f"## {comp}")
        for fam in fams:
            sub = S[(S.comparison == comp) & (S.family == fam) & (S.sensitivity == "main")]
            if sub.empty:
                continue
            lines.append(f"### {fam}")
            lines.append("| 座標 | K | 単位 | 中央値 | q1 | q3 | 正 | 負 | 判定できない | 読めない |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for r_ in sub.sort_values(["quantity", "K"], key=lambda s: s.map(lambda v: (QUANTS + QUANTS_PT).index(v) if v in QUANTS + QUANTS_PT else float(v))).itertuples():
                lines.append(f"| {r_.quantity} | {r_.K} | {r_.n_units} | {r_.median:.2f} | {r_.q1:.2f} | {r_.q3:.2f} | {r_.n_pos} | {r_.n_neg} | {r_.n_undet} | {r_.n_not_readable} |")
            lines.append("")
    lines.append("## 判定")
    if len(Jd):
        cols = list(Jd.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r_ in Jd.itertuples(index=False):
            lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r_) + " |")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return S, Jd


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
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)                             # E79・E83 と同じ種
    ap.add_argument("--targets", nargs="+", help="試走用．評価のターゲットを限る")
    ap.add_argument("--smoke", action="store_true", help="試走（--boot 50 と同じ扱い．provenance に残す）")
    args = ap.parse_args()
    if args.smoke:
        args.boot = min(args.boot, 50)
    out = args.out.resolve()
    partial = bool(args.smoke or args.targets)
    if partial and out.is_relative_to(ROOT / "results"):
        raise SystemExit("試走（--smoke，--targets）の出力は results/ の下に書かない．--out を checks/E87/ か scratchpad にする（I8）")
    if not partial and out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} が空でない．消すか別の --out にしてから流す（上書きで出所が消えないように）")
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--"] + list(SCRIPT_DEPS), cwd=ROOT, capture_output=True, text=True).stdout.strip()
    HEADER = f"E87 commit={commit} dirty_deps={bool(dirty)} script_sha256={E.sha256(Path(__file__))[:16]} boot={args.boot} seed={args.seed} smoke={args.smoke} targets={args.targets} started={started.isoformat(timespec='seconds')}"
    prov = {"commit": commit, "dirty_deps": bool(dirty), "dirty_deps_status": dirty, "script_sha256": E.sha256(Path(__file__)),
            "deps_sha256": {rel: E.sha256(ROOT / rel) for rel in SCRIPT_DEPS}, "args": vars(args) | {"out": str(out)},
            "constants": {"GEFF_MIN": GEFF_MIN, "HOLD_RATIO": HOLD_RATIO, "NOT_READABLE_FRAC": NOT_READABLE_FRAC, "K_GRID": list(K_GRID), "C_GRID": list(C_GRID),
                          "K_all_rule": "n_tgt/4（E65 と同じ）", "QUANTS": list(QUANTS), "QUANTS_PT": list(QUANTS_PT), "BETTER_HIGH": BETTER_HIGH, "KS": list(KS), "REPS": list(REPS),
                          "MAIN_FAMILY": list(MAIN_FAMILY), "SEEDED": sorted(E.SEEDED)},
            "comparisons": [c for _, c in COMPARISONS], "main_comparison": MAIN_COMP, "started": started.isoformat(timespec="seconds"), "inputs": {}}
    prov["code_snapshot"] = str(snapshot_code(out, started.strftime("%Y%m%d-%H%M%S")).relative_to(out))
    Cv, V, U, Ch, Adj, boot_store, jk_store = run_all(args, out, prov)
    write_csv(Cv, out / "curves_rep.csv")
    write_csv(V, out / "values_rep.csv")
    write_csv(U, out / "units.csv")
    write_csv(Adj, out / "monotone_adjust_rep.csv")
    write_csv(Ch if len(Ch) else pd.DataFrame([{"check": "none"}]), out / "checks.csv")
    np.savez_compressed(out / "boot_units.npz", **{k: v for k, v in boot_store.items()})
    if jk_store:
        np.savez_compressed(out / "jk_units.npz", **{k: v for k, v in jk_store.items()})
    summarize(out, U, V, Adj)
    prov["n_units"] = int(len(U))
    prov["checks"] = Ch.to_dict("records") if len(Ch) else []
    prov["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
