"""E83：向きの情報はどの感情の対にあるか．対ごとの方向の損の分解（評価のターゲット．E79 の保存物）と，仮のターゲットでの補い合い（E80 の units）．

設計：docs/exploratory-analyses.md の E83（規則は値を見る前に書いた）．新しい学習はしない．

  (A) 対ごとの D と none の向きの損（results/e79/measure/D_*.csv，pairs_*.csv.gz）
  (B) 対ごとの手法（oursx．none を含む版の選択）− none の向きの損の差を，E79 と同じ単位・2 段のブートストラップ・jackknife で判定する．
      δ_d は対ごとに仮のターゲットの none の揺れから作る．模型に依る対の印も対ごとに付ける
  (C) 仮のターゲットの候補の上で，f のアンカーと予測モデル g の 2 感情の判定を比べ，対ごとの補い合い J_kl，感情ごとの再現率，誤りの φ を出す

使い方：./.venv/bin/python exp/analyze_pairs_e83.py [--out results/e83] [--boot 1000] [--targets ...] [--smoke]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, load_any  # noqa: E402
from explore_pairs import ba  # noqa: E402
from explore_synergy import PAIRS  # noqa: E402
from multimodal_distill import WORLDS  # noqa: E402

RUNS = ("fuse-audio-video-text", "fuse-audio-video")
KINDS = ("audio_wavlm", "video", "text_roberta")                               # (A) の none の行だけ（results/e79_p4 にあれば）
KS = ("0", "5", "25")
REPS = tuple(range(10))
PSEUDO_REPS = tuple(range(5))
EMO = ("平静", "喜び", "悲しみ", "怒り")
PAIR_NAME = {p: f"{EMO[k]}–{EMO[j]}" for p, (k, j) in enumerate(PAIRS)}
ANGER_PAIRS = tuple(p for p, (k, j) in enumerate(PAIRS) if 3 in (k, j))       # 怒りを含む 3 対（P3a の主）
AROUSAL_PAIRS = tuple(p for p, (k, j) in enumerate(PAIRS) if (k, j) in ((1, 3), (0, 2)))   # 覚醒を共有する対（P3b の副．喜び–怒り，平静–悲しみ）
NEU_ANG = next(p for p, (k, j) in enumerate(PAIRS) if (k, j) == (0, 3))
GEFF_MIN = 20                                                                  # E79・E76 の規則
MODEL_UNIT_FRAC = 0.5                                                          # E79 の規則（模型に依る印）
CHANCE = 0.025                                                                 # 差が 0 のとき偶然「減らした」に当たる割合（両側 5% の片側）


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def corp(g):
    return GROUP_SPEC[g][0]


def role_of(run, student):
    return "fused" if student == run else "uni"


def zone(x, dl):
    return "pos" if x >= dl else ("neg" if x <= -dl else "mid")


def judge_unit(x, lo, hi, dl):
    """E79 と同じ 4 区分．"""
    if not (np.isfinite(x) and np.isfinite(lo) and np.isfinite(hi) and np.isfinite(dl)):
        return "missing"
    if lo > 0 and x >= dl:
        return "pos"
    if hi < 0 and x <= -dl:
        return "neg"
    if lo >= -dl and hi <= dl:
        return "same"
    return "undet"


def jk_interval(theta_hat, th, geff_min):
    """E79 と同じ（人物を 1 人ずつ抜いた値から SE_jk と自由度 max(1, ⌊G_eff⌋ − 1) の t 区間）．"""
    th = th[np.isfinite(th)]
    G = len(th)
    if G < 2 or not np.isfinite(theta_hat) or not np.isfinite(geff_min):
        return np.nan, np.nan
    se = np.sqrt((G - 1) / G * ((th - th.mean()) ** 2).sum())
    t = stats.t.ppf(0.975, max(1, int(np.floor(geff_min)) - 1))
    return theta_hat - t * se, theta_hat + t * se


def boot_draws(run, corpus, speakers, B, seed, n_reps):
    """E79 の boot_draws と同じ乱数の流れ（同じ種・同じ順）．"""
    rng = np.random.default_rng([seed, zlib.crc32(run.encode()), zlib.crc32(corpus.encode())])
    nS = len(speakers)
    C = rng.multinomial(nS, np.full(nS, 1 / nS), size=B).astype(np.float64)
    A = rng.multinomial(n_reps, np.full(n_reps, 1 / n_reps), size=B).astype(np.float64)
    return C, A


# ---------------------------------------------------------------- 対ごとの d（重みつき）

def pair_feats(L, y, sidx, nls):
    """L：(n, 4)．話者ごとの対ごとの和（S1・S2・N を感情 k・j について）．戻り：(nls, 6, 6) [S1k, S2k, Nk, S1j, S2j, Nj]．"""
    GT = sparse.csr_matrix((np.ones(len(y)), (sidx, np.arange(len(y)))), shape=(nls, len(y)))
    F = np.zeros((nls, 6, 6))
    for p, (k, j) in enumerate(PAIRS):
        z = L[:, k] - L[:, j]
        for q, c in enumerate((k, j)):
            m = (y == c).astype(np.float64)
            F[:, p, 3 * q] = GT @ (z * m)
            F[:, p, 3 * q + 1] = GT @ (z * z * m)
            F[:, p, 3 * q + 2] = GT @ m
    return F


def pair_d_from_feats(F, W):
    """W：(B, nls) の話者の重み．戻り：(B, 6) の d（感情が 2 本未満か sd が 0 の対は nan）．"""
    M = np.einsum("bs,spq->bpq", W, F)                                       # B × 6 × 6
    with np.errstate(divide="ignore", invalid="ignore"):
        nk, nj = M[:, :, 2], M[:, :, 5]
        mk, mj = M[:, :, 0] / nk, M[:, :, 3] / nj
        vk = (M[:, :, 1] - nk * mk ** 2) / (nk - 1)
        vj = (M[:, :, 4] - nj * mj ** 2) / (nj - 1)
        sd = np.sqrt(np.clip((vk + vj) / 2, 0, None))
        ok = (nk >= 2) & (nj >= 2) & (sd > 0)
        return np.where(ok, (mk - mj) / sd, np.nan)


def d_o_pairs(L, y):
    """E13・E79 と同じ式（反復 1 つ，重みなし）．戻り (6,) の d と o．"""
    d, o = np.full(6, np.nan), np.full(6, np.nan)
    for p, (k, j) in enumerate(PAIRS):
        zk, zj = L[y == k, k] - L[y == k, j], L[y == j, k] - L[y == j, j]
        if len(zk) < 2 or len(zj) < 2:
            continue
        sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
        if sd > 0:
            d[p] = (zk.mean() - zj.mean()) / sd
            o[p] = -(zk.mean() + zj.mean()) / 2 / sd
    return d, o


# ---------------------------------------------------------------- 入力の読み込み

def read_rows_csv(path):
    return pd.read_csv(path, dtype={"K": str, "param": str, "meff": str, "row_id": int, "rep": int}, float_precision="round_trip")


def dedupe_pairs(X, where):
    """pairs_*.csv.gz は同じ行が複数の族の鍵として重複して書かれている．(wave, rep, row_id, pair) で 1 つにする（値が同じことを確かめる）．"""
    key = ["wave", "rep", "row_id", "pair"]
    g = X.groupby(key)[["d", "o", "D"]].agg(["min", "max"])
    bad = ((g[("d", "max")] - g[("d", "min")]).abs() > 1e-9) | ((g[("o", "max")] - g[("o", "min")]).abs() > 1e-9) | ((g[("D", "max")] - g[("D", "min")]).abs() > 1e-9)
    if bad.any():
        raise SystemExit(f"{where}：pairs の重複した行で値が違う（{int(bad.sum())} 組）")
    return X.drop_duplicates(key).copy()


def row_table(run, T):
    """(wave, rep, row_id) → (student, method, param, meff, K)．E77・E78・E80 の段 c の <T>.csv から．"""
    out = []
    for wave in ("e77", "e78", "e80c"):
        f = ROOT / "results" / f"{wave}_{run}" / f"{T}.csv"
        if f.exists():
            d = read_rows_csv(f)
            d["wave"] = wave
            out.append(d[["wave", "rep", "row_id", "student", "method", "param", "meff", "K"]])
    if not out:
        raise SystemExit(f"{run} {T}：<T>.csv が無い")
    tab = pd.concat(out, ignore_index=True)
    if tab.duplicated(["wave", "rep", "row_id"]).any():
        raise SystemExit(f"{run} {T}：(wave, rep, row_id) が重複している")
    return tab


def chosen_ours(sel, run, T, student, K):
    """E79 の Claim 5 と同じ：selector = ours，variant = incl（none を含めて選ぶ版）．戻り (wave, method, param, meff) か None（none が選ばれた）．"""
    ch = sel[(sel.run == run) & (sel.target == T) & (sel.student == student) & (sel.K == K) & (sel.selector == "ours") & (sel.variant == "incl")]
    if ch.empty:
        return "missing"
    r = ch.iloc[0]
    cn = r.chose_none
    if (cn is True or str(cn) == "True") or r.method == "none" or pd.isna(r.method):
        return None
    return (r.wave, r.method, str(r.param), str(r.meff))


def load_labels_speakers(run, T, d77):
    samp = pd.read_csv(d77 / f"{T}_samples.csv")
    df, _ = load_any(T, run)                                                   # 取り決め 1〜3 を満たすラベル（E79 と同じ）
    if not np.array_equal(df.sample_id.astype(str).to_numpy(), samp.sample_id.astype(str).to_numpy()):
        raise SystemExit(f"{run} {T}：_samples.csv の sample_id の並びが load_any と違う")
    return df.label_4.to_numpy().astype(int), samp.speaker_group.astype(str).to_numpy()


# ---------------------------------------------------------------- (A)(B)

def stage_ab(args, out, prov):
    sel = pd.read_csv(ROOT / "results" / "e79" / "select" / "chosen.csv", dtype={"K": str, "param": str, "meff": str})
    prov["inputs"]["chosen.csv"] = sha256(ROOT / "results" / "e79" / "select" / "chosen.csv")
    rep_rows, unit_rows, delta_rows, check_rows = [], [], [], []
    # ---- (A) D と none の向きの損（種類の実行も，あれば）
    a_rows = []
    for mdir in (ROOT / "results" / "e79" / "measure", ROOT / "results" / "e79_p4" / "measure"):
        for f in sorted(mdir.glob("pairs_*.csv.gz")):
            run = f.name[len("pairs_"):-len(".csv.gz")]
            if run not in RUNS + KINDS:
                continue
            prov["inputs"][str(f.relative_to(ROOT))] = sha256(f)
            P = pd.read_csv(f, dtype={"row_id": int, "rep": int, "pair": int})
            for T in sorted(P.target.unique()):
                if args.targets and T not in args.targets:
                    continue
                tab = row_table(run, T)
                X = dedupe_pairs(P[P.target == T], f"{run} {T}").merge(tab, on=["wave", "rep", "row_id", "student"], how="left")
                if X.method.isna().any():
                    raise SystemExit(f"{run} {T}：pairs の行が <T>.csv に無い")
                N = X[(X.method == "none") & (X.param == "-")]
                for (st, K), g in N.groupby(["student", "K"]):
                    if K not in KS:
                        continue
                    for p, gg in g.groupby("pair"):
                        a_rows.append({"run": run, "target": T, "student": st, "K": K, "pair": int(p), "pair_name": PAIR_NAME[int(p)],
                                       "n_reps": len(gg), "A_D0": 100 * float(np.mean(ba(gg.D.to_numpy(), 0.0))),
                                       "none_dirloss": 100 * float(np.mean(ba(gg.D.to_numpy(), 0.0) - ba(gg.d.to_numpy(), 0.0))),
                                       "none_posloss": 100 * float(np.mean(ba(gg.d.to_numpy(), 0.0) - ba(gg.d.to_numpy(), gg.o.to_numpy()))),
                                       "none_d": float(gg.d.mean()), "D": float(gg.D.mean()), "none_o_abs": float(gg.o.abs().mean())})
            if run in RUNS and mdir == ROOT / "results" / "e79" / "measure":   # (B) は E79 の主の実行だけ（e79_p4 は P4 の判定用で oursx の行が無い）
                stage_b_run(args, run, P, sel, rep_rows, unit_rows, delta_rows, check_rows, prov)
    pd.DataFrame(a_rows).to_csv(out / "a_none_pairs.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(out / "b_pairs_rep.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(out / "b_delta_pairs.csv", index=False)
    pd.DataFrame(check_rows).to_csv(out / "checks.csv", index=False)
    U = pd.DataFrame(unit_rows)
    U.to_csv(out / "b_units_pairs.csv", index=False)
    return pd.DataFrame(a_rows), U


def delta_pairs(run, d77, targets, prov):
    """対ごとの δ_d：仮のターゲットの none の 100·A(d, 0) の反復 0〜4 の SD ÷ √10 を (T, c) ごとに，(student, K, pair) で中央値 × 2．"""
    rows = []
    for T in targets:
        for f in sorted(d77.glob(f"{T}_ps_*_none.npz")):
            m = f.name[len(T) + 4:-len("_none.npz")]
            c, r = m.rsplit("_r", 1)
            z = np.load(f)
            y = np.asarray(z["y"]).astype(int)
            for i, key in enumerate(z["keys"]):
                st, K = str(key).split("|")
                if K not in KS:
                    continue
                d, _ = d_o_pairs(np.asarray(z[f"lp{i}"], dtype=np.float64), y)
                for p in range(6):
                    rows.append({"run": run, "target": T, "pseudo": c, "rep": int(r), "student": st, "K": K, "pair": p, "a_d0": 100 * float(ba(d[p], 0.0))})
    R = pd.DataFrame(rows)
    if R.empty:
        raise SystemExit(f"{run}：δ の材料（_ps_ のファイル）が無い")
    g = R.groupby(["run", "target", "pseudo", "student", "K", "pair"]).agg(n_reps=("rep", "nunique"), sd=("a_d0", lambda x: x.std(ddof=1))).reset_index()
    g = g[g.n_reps >= 2]
    g["s_d"] = g.sd / np.sqrt(10)
    D = g.groupby(["run", "student", "K", "pair"]).agg(n_units=("s_d", "size"), s_d=("s_d", "median")).reset_index()
    D["delta_d_pair"] = 2 * D.s_d
    return D, R


def stage_b_run(args, run, P, sel, rep_rows, unit_rows, delta_rows, check_rows, prov):
    d77 = ROOT / "results" / f"e77_{run}"
    targets = [t for t in sorted(P.target.unique()) if (not args.targets or t in args.targets)]
    D, Rd = delta_pairs(run, d77, targets, prov)
    Rd.to_csv(args.out / f"b_delta_reps_{run}.csv", index=False)
    delta_rows.extend(D.assign(run=run).to_dict("records"))
    dmap = {(r.student, r.K, int(r.pair)): float(r.delta_d_pair) for r in D.itertuples()}
    mods = WORLDS[run]
    students = mods + [run]
    # ---- 点の値（pairs の表から），行の対応，反復ごとの対数確率
    by_corpus = {}
    for T in targets:
        by_corpus.setdefault(corp(T), []).append(T)
    for cp, Ts in by_corpus.items():
        prep = {}
        for T in Ts:
            tab = row_table(run, T)
            X = dedupe_pairs(P[P.target == T], f"{run} {T}").merge(tab, on=["wave", "rep", "row_id", "student"], how="left")
            if X.method.isna().any():
                raise SystemExit(f"{run} {T}：pairs の行が <T>.csv に無い")
            yall, spk_all = load_labels_speakers(run, T, d77)
            te = {r: np.load(d77 / f"{T}_r{r}_te.npy") for r in REPS}
            rowids = {r: np.load(d77 / f"{T}_r{r}_rowids.npy") for r in REPS}
            logp = {}
            for wave in ("e77", "e78", "e80c"):
                dd = ROOT / "results" / f"{wave}_{run}"
                if (dd / f"{T}_r0_logp.npy").exists():
                    logp[wave] = {r: (np.load(dd / f"{T}_r{r}_logp.npy", mmap_mode="r"), np.load(dd / f"{T}_r{r}_rowids.npy")) for r in REPS}
            for wave in logp:
                for r in REPS:
                    if not np.array_equal(np.load(ROOT / "results" / f"{wave}_{run}" / f"{T}_r{r}_te.npy"), te[r]):
                        raise SystemExit(f"{run} {T} r{r}：{wave} の _te が E77 と違う")
            prep[T] = {"X": X, "y": yall, "spk": spk_all, "te": te, "logp": logp}
        speakers = sorted(set().union(*[set(p["spk"][p["te"][r]]) for p in prep.values() for r in REPS]))
        sid = {s: i for i, s in enumerate(speakers)}
        nls = len(speakers)
        C, A = boot_draws(run, cp, speakers, args.boot, args.seed, len(REPS))
        # G_eff（E79：反復 0〜9 のテストの発話数を人物ごとに足す．コーパスの中の最小）
        geffs = []
        for T, p in prep.items():
            cnt = pd.Series(np.concatenate([p["spk"][p["te"][r]] for r in REPS])).value_counts().to_numpy().astype(float)
            geffs.append(cnt.sum() ** 2 / (cnt ** 2).sum())
        gmin = float(min(geffs))
        for st in students:
            for K in KS:
                ch = chosen_ours(sel, run, Ts[0], st, K)
                if ch == "missing":
                    check_rows.append({"check": "chosen_missing", "run": run, "corpus": cp, "student": st, "K": K})
                    continue
                pts = {T: np.full((len(REPS), 6), np.nan) for T in Ts}             # 反復 × 対の点の値（pairs の表）
                aucs = {T: np.full((len(REPS), 6), np.nan) for T in Ts}
                des = {T: np.full((len(REPS), 6), np.nan) for T in Ts}
                dos = {T: np.full((len(REPS), 6), np.nan) for T in Ts}
                nonelosses = {T: np.full((len(REPS), 6), np.nan) for T in Ts}
                boot_diff = {T: [] for T in Ts}                                   # 反復ごとの (B, 6)
                jk_diff = {T: [] for T in Ts}
                for T in Ts:
                    p = prep[T]
                    X = p["X"]
                    chT = chosen_ours(sel, run, T, st, K)
                    if chT != ch:
                        check_rows.append({"check": "chosen_differs_in_corpus", "run": run, "corpus": cp, "student": st, "K": K, "target": T})
                    for r in REPS:
                        xn = X[(X.rep == r) & (X.student == st) & (X.K == K) & (X.method == "none") & (X.param == "-")].sort_values("pair")
                        if len(xn) != 6:
                            check_rows.append({"check": "none_rows", "run": run, "target": T, "student": st, "K": K, "rep": r, "n": len(xn)})
                            continue
                        if chT is None:
                            pts[T][r] = 0.0
                            aucs[T][r] = 0.0
                            des[T][r] = 0.0
                            dos[T][r] = 0.0
                            nonelosses[T][r] = 100 * (ba(xn.D.to_numpy(), 0.0) - ba(xn.d.to_numpy(), 0.0))
                            continue
                        wave, meth, par, me = chT
                        xo = X[(X.rep == r) & (X.student == st) & (X.K == K) & (X.wave == wave) & (X.method == meth) & (X.param == par) & (X.meff == me)].sort_values("pair")
                        if len(xo) != 6:
                            check_rows.append({"check": "ours_rows", "run": run, "target": T, "student": st, "K": K, "rep": r, "n": len(xo), "chosen": "|".join(chT)})
                            continue
                        dn, do_ = xn.d.to_numpy(), xo.d.to_numpy()
                        pts[T][r] = 100 * (ba(do_, 0.0) - ba(dn, 0.0))
                        aucs[T][r] = 100 * (ba(xo.d_auc.to_numpy(), 0.0) - ba(xn.d_auc.to_numpy(), 0.0))
                        des[T][r] = 100 * ((xo.acc_model.to_numpy() - xo.acc_actual.to_numpy()) - (xn.acc_model.to_numpy() - xn.acc_actual.to_numpy()))
                        dos[T][r] = xo.o.to_numpy() - xn.o.to_numpy()
                        nonelosses[T][r] = 100 * (ba(xn.D.to_numpy(), 0.0) - ba(dn, 0.0))
                        # 対数の確率から計算し直して照合し，ブートストラップの材料にする
                        yte = p["y"][p["te"][r]]
                        sidx = np.array([sid[s] for s in p["spk"][p["te"][r]]])
                        Ln = logp_row(p["logp"], "e77", r, int(xn.row_id.iloc[0]))
                        Lo = logp_row(p["logp"], wave, r, int(xo.row_id.iloc[0]))
                        dn2, _ = d_o_pairs(Ln, yte)
                        do2, _ = d_o_pairs(Lo, yte)
                        rec = 100 * (ba(do2, 0.0) - ba(dn2, 0.0))
                        if np.nanmax(np.abs(rec - pts[T][r])) > 1e-6:
                            raise SystemExit(f"{run} {T} {st} K{K} r{r}：対ごとの点の値が pairs の表と合わない（最大 {np.nanmax(np.abs(rec - pts[T][r])):.2e}）")
                        Fn, Fo = pair_feats(Ln, yte, sidx, nls), pair_feats(Lo, yte, sidx, nls)
                        boot_diff[T].append((r, 100 * (ba(pair_d_from_feats(Fo, C), 0.0) - ba(pair_d_from_feats(Fn, C), 0.0))))
                        if gmin < GEFF_MIN:
                            Wj = np.ones((nls, nls)) - np.eye(nls)
                            jk_diff[T].append((r, 100 * (ba(pair_d_from_feats(Fo, Wj), 0.0) - ba(pair_d_from_feats(Fn, Wj), 0.0))))
                        for pidx in range(6):
                            rep_rows.append({"run": run, "target": T, "student": st, "K": K, "rep": r, "pair": pidx, "pair_name": PAIR_NAME[pidx],
                                             "diff_dir": pts[T][r, pidx], "diff_dir_auc": aucs[T][r, pidx], "diff_model_err": des[T][r, pidx],
                                             "diff_o": dos[T][r, pidx], "none_dirloss": nonelosses[T][r, pidx], "chosen": "|".join(chT)})
                # ---- 単位（コーパス）に集める：反復の平均をターゲットで平均（E79 と同じ）
                x = np.nanmean(np.stack([np.nanmean(pts[T], 0) for T in Ts]), 0)   # (6,)
                xa = np.nanmean(np.stack([np.nanmean(aucs[T], 0) for T in Ts]), 0)
                de = np.nanmean(np.stack([np.nanmean(des[T], 0) for T in Ts]), 0)
                dO = np.nanmean(np.stack([np.nanmean(dos[T], 0) for T in Ts]), 0)
                nl = np.nanmean(np.stack([np.nanmean(nonelosses[T], 0) for T in Ts]), 0)
                # ブートストラップ：反復の重み A で反復を混ぜ，ターゲットで平均
                lo = hi = np.full(6, np.nan)
                nvalid = 0
                lo_j = hi_j = np.full(6, np.nan)
                rule = "pct"
                if ch is not None and all(len(boot_diff[T]) == len(REPS) for T in Ts):
                    per_T = []
                    for T in Ts:
                        arr = np.stack([v for _, v in sorted(boot_diff[T])], 1)   # (B, R, 6)
                        w = A[:, :, None]
                        fin = np.isfinite(arr)
                        per_T.append(np.where(fin, arr, 0).__mul__(w).sum(1) / np.where(fin, w, 0).sum(1))
                    b = np.nanmean(np.stack(per_T), 0)                             # (B, 6)
                    lo, hi = np.nanpercentile(b, 2.5, 0), np.nanpercentile(b, 97.5, 0)
                    nvalid = int(np.isfinite(b).all(1).sum())
                    if gmin < GEFF_MIN:
                        th = np.nanmean(np.stack([np.nanmean(np.stack([v for _, v in sorted(jk_diff[T])], 1), 1) for T in Ts]), 0)   # (G, 6)
                        lo_j, hi_j = np.array([jk_interval(x[pp], th[:, pp], gmin) for pp in range(6)]).T
                        lo, hi = np.fmin(lo, lo_j), np.fmax(hi, hi_j)
                        rule = "union_pct_jk"
                for pidx in range(6):
                    dl = dmap.get((st, K, pidx), np.nan)
                    for scale, stag in ((1.0, "main"), (0.5, "delta_half"), (2.0, "delta_double")):
                        dls = dl * scale
                        md = np.nan
                        if np.isfinite(x[pidx]) and np.isfinite(xa[pidx]) and np.isfinite(dls):
                            md = bool(abs(x[pidx] - xa[pidx]) > MODEL_UNIT_FRAC * dls or abs(de[pidx]) > MODEL_UNIT_FRAC * dls or zone(x[pidx], dls) != zone(xa[pidx], dls))
                        jd = "same_by_definition" if ch is None else judge_unit(x[pidx], lo[pidx], hi[pidx], dls)
                        unit_rows.append({"run": run, "role": role_of(run, st), "student": st, "K": K, "corpus": cp, "n_groups": len(Ts), "pair": pidx,
                                          "pair_name": PAIR_NAME[pidx], "point": x[pidx], "lo": lo[pidx], "hi": hi[pidx], "lo_jk": lo_j[pidx], "hi_jk": hi_j[pidx],
                                          "interval_rule": rule, "g_eff_min": gmin, "n_boot_valid": nvalid, "delta_d_pair": dls, "sensitivity": stag,
                                          "judgment": jd, "model_dependent": md, "point_auc": xa[pidx], "model_err_diff": de[pidx], "diff_o": dO[pidx],
                                          "none_dirloss": nl[pidx], "chosen": "none" if ch is None else "|".join(ch)})


def logp_row(logp, wave, r, row_id):
    arr, rid = logp[wave][r]
    pos = np.flatnonzero(rid == row_id)
    if len(pos) != 1:
        raise SystemExit(f"{wave} r{r}：row_id {row_id} が logp に {len(pos)} 行")
    L = np.asarray(arr[pos[0]], dtype=np.float64)
    if not np.isfinite(L).all():
        raise SystemExit(f"{wave} r{r} row {row_id}：NaN/Inf")
    return L


# ---------------------------------------------------------------- (C)

def stage_c(args, out, prov):
    rows, rec_rows, phi_rows = [], [], []
    for w, run in (("avt", "fuse-audio-video-text"), ("av", "fuse-audio-video")):
        mods = WORLDS[run]
        udir = ROOT / "results" / "e80" / w / "units" / run
        if not udir.exists():
            raise SystemExit(f"{udir} が無い")
        for Tdir in sorted(p for p in udir.iterdir() if p.is_dir()):
            T = Tdir.name
            if args.targets and T not in args.targets:
                continue
            for f in sorted(Tdir.glob("*.npz")):
                if f.name.endswith("_agg.npz"):
                    continue
                c, r = f.stem.rsplit("_r", 1)
                z = np.load(f)
                y = np.asarray(z["y"]).astype(int)
                for K in KS:
                    keep = np.ones(len(y), bool)
                    lab = np.asarray(z[f"lab__{K}"]).astype(int)
                    if len(lab):
                        if lab.max() >= len(y):
                            raise SystemExit(f"{f}：lab__{K} が候補の位置の範囲を超える")
                        keep[lab] = False                                      # K 本を除く（R2）
                    yk = y[keep]
                    preds = {f"other_{m}": np.asarray(z[f"P__{m}"])[keep] for m in mods}
                    preds["pF"] = np.asarray(z[f"P__{run}"])[keep]
                    preds["poe0"] = np.asarray(z["poe0"])[keep]
                    preds["agree"] = np.asarray(z["agree"])[keep]
                    preds["ref"] = np.asarray(z[f"ref__{K}"])[keep]
                    preds["ref0"] = np.asarray(z[f"ref0__{K}"])[keep]
                    tag = {"world": w, "run": run, "target": T, "pseudo": c, "pseudo_corpus": corp(c), "rep": int(r), "K": K, "n_cand": int(keep.sum())}
                    # 感情ごとの再現率（4 クラスの argmax）
                    for name, Pm in preds.items():
                        am = Pm.argmax(1)
                        for e in range(4):
                            m = yk == e
                            rec_rows.append({**tag, "model": name, "emotion": EMO[e], "n": int(m.sum()), "recall": float((am[m] == e).mean()) if m.any() else np.nan})
                    # 誤りの φ（モダリティの対ごと，感情ごと．当たったかの相関）
                    corr = {m: (preds[f"other_{m}"].argmax(1) == yk) for m in mods}
                    for i, m1 in enumerate(mods):
                        for m2 in mods[i + 1:]:
                            for e in range(4):
                                mm = yk == e
                                a, b = corr[m1][mm].astype(float), corr[m2][mm].astype(float)
                                phi = np.nan
                                if mm.sum() >= 10 and a.std() > 0 and b.std() > 0:
                                    phi = float(np.corrcoef(a, b)[0, 1])
                                phi_rows.append({**tag, "mod1": m1, "mod2": m2, "emotion": EMO[e], "n": int(mm.sum()), "phi": phi,
                                                 "both_wrong": float(((a == 0) & (b == 0)).mean()) if mm.any() else np.nan,
                                                 "indep_both_wrong": float((1 - a.mean()) * (1 - b.mean())) if mm.any() else np.nan})
                    # 対ごとの補い合い J_kl(f, g)
                    for st in mods:
                        S = np.asarray(z[f"none__{st}__{K}"])[keep]
                        for name, G in preds.items():
                            if name == f"other_{st}":
                                continue
                            for p, (k, j) in enumerate(PAIRS):
                                m = (yk == k) | (yk == j)
                                if m.sum() < 10:
                                    continue
                                yy = yk[m]
                                sdec = np.where(S[m, k] >= S[m, j], k, j)
                                gdec = np.where(G[m, k] >= G[m, j], k, j)
                                sr, gr = sdec == yy, gdec == yy
                                rows.append({**tag, "student": st, "g": name, "pair": p, "pair_name": PAIR_NAME[p], "n_pair": int(m.sum()),
                                             "acc_student": float(sr.mean()), "acc_g": float(gr.mean()),
                                             "J": 100 * float((gr & ~sr).mean() - (~gr & sr).mean()),
                                             "g_right_s_wrong": float((gr & ~sr).mean()), "g_wrong_s_right": float((~gr & sr).mean())})
    J = pd.DataFrame(rows)
    J.to_csv(out / "c_complementarity_units.csv", index=False)
    pd.DataFrame(rec_rows).to_csv(out / "c_recall_units.csv", index=False)
    pd.DataFrame(phi_rows).to_csv(out / "c_phi_units.csv", index=False)
    return J, pd.DataFrame(rec_rows), pd.DataFrame(phi_rows)


# ---------------------------------------------------------------- まとめと予言の判定

def summarize(out, A_, U, J, R, Ph):
    lines = ["# E83 のまとめ（台本が自動で書く．読み方は docs/exploratory-analyses.md の E83 3 節）", ""]
    # (A)
    if len(A_):
        a = A_[A_.K == "0"].groupby(["run", "student", "pair_name"]).agg(A_D0_med=("A_D0", "median"), none_dirloss_med=("none_dirloss", "median"),
                                                                           none_posloss_med=("none_posloss", "median"), n=("target", "nunique")).reset_index()
        a.to_csv(out / "a_summary_K0.csv", index=False)
        lines += ["## (A) K=0 の none：対ごとの A(D,0) と向きの損の中央値（ターゲットと反復の中央値）", "", a.to_string(index=False), ""]
    # (B) 判定の数
    if len(U):
        main = U[U.sensitivity == "main"]
        cnt = main.groupby(["run", "role", "student", "pair_name", "judgment"]).size().unstack(fill_value=0).reset_index()
        cnt.to_csv(out / "b_judgments_by_pair.csv", index=False)
        nomd = main[main.model_dependent != True].groupby(["run", "role", "student", "pair_name", "judgment"]).size().unstack(fill_value=0).reset_index()  # noqa: E712
        nomd.to_csv(out / "b_judgments_by_pair_nomd.csv", index=False)
        for sname in ("delta_half", "delta_double"):
            U[U.sensitivity == sname].groupby(["run", "role", "student", "pair_name", "judgment"]).size().unstack(fill_value=0).reset_index().to_csv(out / f"b_judgments_by_pair_{sname}.csv", index=False)
        lines += ["## (B) 単位 × 対の判定の数（主．pos＝減らした，neg＝増やした，same＝同じ水準，undet＝判定できない）", "", cnt.to_string(index=False), ""]
        lines += ["## (B) 同じく，模型に依る対を除いた数", "", nomd.to_string(index=False), ""]
        # 族ごとの予言の判定（P3a・P3b）
        fam_rows = []
        for fam, stus, sel_pairs, base, label in (("audio", ["audio_wavlm"], ANGER_PAIRS, 0.5, "怒りを含む 3 対"), ("video", ["video"], (NEU_ANG,), 1 / 6, "平静–怒り"),
                                                  ("text", ["text_roberta"], ANGER_PAIRS, 0.5, "怒りを含む 3 対（記述）"), ("fused", RUNS, ANGER_PAIRS, 0.5, "怒りを含む 3 対（記述）")):
            for use_md in (True, False):
                m = main[main.student.isin(stus)]
                if not use_md:
                    m = m[m.model_dependent != True]  # noqa: E712
                n_units = len(m[["run", "student", "K", "corpus"]].drop_duplicates())
                n_tests = len(m)
                pos = m[m.judgment == "pos"]
                n_pos = len(pos)
                share = float(pos.pair.isin(sel_pairs).mean()) if n_pos else np.nan
                share_ar = float(pos.pair.isin(AROUSAL_PAIRS).mean()) if n_pos else np.nan
                fam_rows.append({"family": fam, "with_model_dependent": use_md, "n_units": n_units, "n_tests": n_tests, "chance_pos": CHANCE * n_tests,
                                 "n_pos": n_pos, "n_neg": int((m.judgment == "neg").sum()), "n_same": int((m.judgment == "same").sum()),
                                 "n_undet": int((m.judgment == "undet").sum()), "selected_pairs": label, "share_selected_among_pos": share, "base_share": base,
                                 "share_arousal_pairs_among_pos": share_ar, "base_arousal": 2 / 6,
                                 "P3a_holds": bool(n_pos > CHANCE * n_tests and np.isfinite(share) and share > base) if fam in ("audio", "video") else None,
                                 "pos_pairs": ";".join(f"{PAIR_NAME[p]}:{n}" for p, n in pos.pair.value_counts().sort_index().items())})
        F = pd.DataFrame(fam_rows)
        F.to_csv(out / "b_p3_checks.csv", index=False)
        lines += ["## (B) 予言 P3a の判定（族ごと．偶然の見込み＝検定の数 × 0.025）", "", F.to_string(index=False), ""]
    # (C)
    if len(J):
        Jc = J.groupby(["world", "student", "K", "g", "pair_name", "pseudo_corpus"]).J.median().reset_index()      # コーパスの中央値を先に取る（S7）
        Js = Jc.groupby(["world", "student", "K", "g", "pair_name"]).agg(J_med=("J", "median"), J_q1=("J", lambda x: x.quantile(0.25)),
                                                                         J_q3=("J", lambda x: x.quantile(0.75)), n_corpora=("pseudo_corpus", "nunique")).reset_index()
        Js.to_csv(out / "c_complementarity_summary.csv", index=False)
        Jc.to_csv(out / "c_complementarity_by_corpus.csv", index=False)
        lines += ["## (C) 対ごとの補い合い J_kl(f, g) の中央値（仮のターゲットのコーパスの中央値を取ってから中央値）", "",
                  Js[Js.g.isin(["pF", "ref", "poe0"])].to_string(index=False), ""]
        # P3c：(B) で減った対について，同じコーパスを仮のターゲットにした J(f, ref) の中央値が正か
        if len(U):
            main = U[U.sensitivity == "main"]
            wmap = {"fuse-audio-video-text": "avt", "fuse-audio-video": "av"}
            jref = Jc[Jc.g == "ref"].set_index(["world", "student", "K", "pair_name", "pseudo_corpus"]).J
            rows = []
            for r in main.itertuples():
                if r.role != "uni":
                    continue
                key = (wmap[r.run], r.student, r.K, r.pair_name, r.corpus)
                rows.append({"run": r.run, "student": r.student, "K": r.K, "corpus": r.corpus, "pair_name": r.pair_name, "judgment": r.judgment,
                             "model_dependent": r.model_dependent, "J_ref_same_corpus": float(jref.get(key, np.nan))})
            Q = pd.DataFrame(rows)
            Q.to_csv(out / "b_c_p3c_rows.csv", index=False)
            base = float((Q.J_ref_same_corpus > 0).mean())
            red = Q[Q.judgment == "pos"]
            share = float((red.J_ref_same_corpus > 0).mean()) if len(red) else np.nan
            P3c = pd.DataFrame([{"n_rows": len(Q), "base_share_J_pos": base, "n_reduced": len(red), "share_J_pos_among_reduced": share,
                                 "P3c_holds": bool(len(red) and share > base)}])
            P3c.to_csv(out / "b_c_p3c.csv", index=False)
            lines += ["## P3c：減った対のうち J(f, ref) > 0 の割合と土台", "", P3c.to_string(index=False), ""]
    if len(R):
        Rc = R.groupby(["world", "model", "emotion", "pseudo_corpus"]).recall.median().reset_index()
        Rs = Rc.groupby(["world", "model", "emotion"]).recall.median().unstack("emotion").reset_index()
        Rs.to_csv(out / "c_recall_summary.csv", index=False)
        lines += ["## (C) 感情ごとの再現率の中央値（モダリティの分類器と予測モデル）", "", Rs.to_string(index=False), ""]
    if len(Ph):
        Pc = Ph.groupby(["world", "mod1", "mod2", "emotion", "pseudo_corpus"])[["phi", "both_wrong", "indep_both_wrong"]].median().reset_index()
        Ps = Pc.groupby(["world", "mod1", "mod2", "emotion"])[["phi", "both_wrong", "indep_both_wrong"]].median().reset_index()
        Ps.to_csv(out / "c_phi_summary.csv", index=False)
        lines += ["## (C) 誤りの φ（感情ごと）", "", Ps.to_string(index=False), ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e83")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)                             # E79 と同じ種
    ap.add_argument("--targets", nargs="+", help="試走用．評価のターゲットを限る")
    ap.add_argument("--stages", nargs="+", default=["ab", "c"], choices=["ab", "c"])
    ap.add_argument("--smoke", action="store_true", help="試走（--boot 50 と同じ扱い．provenance に残す）")
    args = ap.parse_args()
    if args.smoke:
        args.boot = min(args.boot, 50)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "exp", "measure"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    prov = {"commit": commit, "dirty_exp_measure": bool(dirty), "script_sha256": sha256(Path(__file__)), "args": vars(args) | {"out": str(out)}, "inputs": {}}
    A_ = U = J = R = Ph = pd.DataFrame()
    if "ab" in args.stages:
        A_, U = stage_ab(args, out, prov)
    if "c" in args.stages:
        J, R, Ph = stage_c(args, out, prov)
    summarize(out, A_, U, J, R, Ph)
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
