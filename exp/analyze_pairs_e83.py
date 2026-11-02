"""E83：向きの情報はどの感情の対にあるか．対ごとの方向の損の分解（評価のターゲット．E79 の保存物）と，仮のターゲットでの補い合い（E80 の units）．

設計：docs/exploratory-analyses.md の E83（規則は値を見る前に書いた）．新しい学習はしない．(A)(B) は評価のターゲットの値を対に分けて判定するので本番として扱う．

  (A) 対ごとの D と none の向きの損（results/e79/measure/D_*.csv，pairs_*.csv.gz．種類の実行の none は results/e79_p4/measure から）．
      反復ごとの行と，反復・ターゲットの中央値と範囲，d_AUC から作った向きの損，模型の誤差 acc_model − acc_actual を残す
  (B) 対ごとの手法（oursx．none を含む版の選択．E79 の Claim 5 の P5-3 と同じ鍵）− none の向きの損の差を，E79 と同じ単位・2 段のブートストラップ・jackknife で判定する．
      δ_d は対ごとに仮のターゲットの none の揺れから作る．模型に依る対の印も対ごとに付ける．
      感度の行は δ の半分・2 倍，逆の順序（A(D,o) − A(d,o)．rev_order），IEMOCAP 即興だけ（iemocap_impro_only）．E79 の P5-3 と同じ作り．
      kd と abl_own（none を含む版の選択）の行も同じ計算で出す（comparison 列．文の規則 (3) の補助）．単位ごとのブートストラップの複製は npz に残す
  (C) 仮のターゲットの候補の上で，f のアンカー（単モダリティと融合）と予測モデル g の 2 感情の判定を比べ，対ごとの補い合い J_kl，感情ごとの再現率，誤りの φ を出す．
      agree は pF の行ごとの正の定数倍で判定が pF と同じなので出さない

使い方：./.venv/bin/python exp/analyze_pairs_e83.py [--out results/e83] [--boot 1000] [--perm 1000] [--targets ...] [--stages ab c] [--smoke]
"""
import argparse
import datetime as dt
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
KINDS = ("audio_wavlm", "video", "text_roberta")                               # (A) の none の行だけ（results/e79_p4 から）
KS = ("0", "5", "25")
REPS = tuple(range(10))
PSEUDO_REPS = tuple(range(5))
EMO = ("平静", "喜び", "悲しみ", "怒り")
PAIR_NAME = {p: f"{EMO[k]}–{EMO[j]}" for p, (k, j) in enumerate(PAIRS)}
PAIR_IDX = {(k, j): p for p, (k, j) in enumerate(PAIRS)}
ANGER_PAIRS = tuple(p for p, (k, j) in enumerate(PAIRS) if 3 in (k, j))       # 怒りを含む 3 対（P3a の主の集合．E13 から立てた）
P3A_SIDE_PAIRS = (PAIR_IDX[(0, 3)], PAIR_IDX[(2, 3)])                          # 平静–怒り，悲しみ–怒り（P3a の側に帰属する対）
P3B_SIDE_PAIRS = (PAIR_IDX[(0, 2)],)                                           # 平静–悲しみ（P3b の側に帰属する対）
HAPPY_ANGER = PAIR_IDX[(1, 3)]                                                 # 喜び–怒り（どちらの側にも数えず，名前を書く）
AROUSAL_PAIRS = (PAIR_IDX[(1, 3)], PAIR_IDX[(0, 2)])                           # 覚醒を共有する対（P3b の記述．喜び–怒り，平静–悲しみ）
NEU_ANG = PAIR_IDX[(0, 3)]
GEFF_MIN = 20                                                                  # E79・E76 の規則
MODEL_UNIT_FRAC = 0.5                                                          # E79 の規則（模型に依る印）
CHANCE = 0.025                                                                 # 差が 0 のとき偶然「減らした」に当たる割合（両側 5% の片側）
COMPARISONS = (("oursx", "none-oursx"), ("kd", "none-kd"), ("abl_own", "none-abl_own"))   # 先頭が主（P3a・P3c の判定に使う）
MAIN_COMP = "none-oursx"
JUDG_COLS = ["pos", "neg", "same", "undet", "missing", "same_by_definition"]
PERM_SEED = 12345


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def note_input(prov, path, with_sha=False):
    """読んだファイルの大きさと時刻（と小さいものは sha256）を provenance に残す（I9）．"""
    p = Path(path)
    key = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    if key in prov["inputs"]:
        return
    st_ = p.stat()
    rec = {"bytes": st_.st_size, "mtime": dt.datetime.fromtimestamp(st_.st_mtime).isoformat(timespec="seconds")}
    if with_sha:
        rec["sha256"] = sha256(p)
    prov["inputs"][key] = rec


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


# ---------------------------------------------------------------- 対ごとの d・o（重みつき）

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


def pair_d_o_from_feats(F, W):
    """W：(B, nls) の話者の重み．戻り：(B, 6) の d と o（感情が 2 本未満か sd が 0 の対は nan）．E79 の boot_rep と同じ式．"""
    M = np.einsum("bs,spq->bpq", W, F)                                       # B × 6 × 6
    with np.errstate(divide="ignore", invalid="ignore"):
        nk, nj = M[:, :, 2], M[:, :, 5]
        mk, mj = M[:, :, 0] / nk, M[:, :, 3] / nj
        vk = (M[:, :, 1] - nk * mk ** 2) / (nk - 1)
        vj = (M[:, :, 4] - nj * mj ** 2) / (nj - 1)
        sd = np.sqrt(np.clip((vk + vj) / 2, 0, None))
        ok = (nk >= 2) & (nj >= 2) & (sd > 0)
        return np.where(ok, (mk - mj) / sd, np.nan), np.where(ok, -(mk + mj) / 2 / sd, np.nan)


def pair_d_from_feats(F, W):
    return pair_d_o_from_feats(F, W)[0]


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


def dir_diff(d_o, d_n):
    """向きの損の差（none − 手法）．D が打ち消し合うので 100·[A(d_手法, 0) − A(d_none, 0)]．"""
    return 100 * (ba(d_o, 0.0) - ba(d_n, 0.0))


def dir_diff_rev(D, d_o, o_o, d_n, o_n):
    """逆の順序（E79 の dir_rev．向きを o の下で測る）：100·{[A(D, o_none) − A(d_none, o_none)] − [A(D, o_手法) − A(d_手法, o_手法)]}．D は固定．"""
    return 100 * ((ba(D, o_n) - ba(d_n, o_n)) - (ba(D, o_o) - ba(d_o, o_o)))


def max_abs_or_nan_mismatch(a, b):
    """1e−6 の照合：値の差の最大と，片方だけ NaN の対の数（I11）．"""
    na, nb = np.isnan(a), np.isnan(b)
    mism = int((na != nb).sum())
    both = ~na & ~nb
    return (float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0), mism


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


def row_table(run, T, prov=None):
    """(wave, rep, row_id) → (student, method, param, meff, K)．E77・E78・E80 の段 c の <T>.csv から．"""
    out = []
    for wave in ("e77", "e78", "e80c"):
        f = ROOT / "results" / f"{wave}_{run}" / f"{T}.csv"
        if f.exists():
            if prov is not None:
                note_input(prov, f, with_sha=True)
            d = read_rows_csv(f)
            d["wave"] = wave
            out.append(d[["wave", "rep", "row_id", "student", "method", "param", "meff", "K"]])
    if not out:
        raise SystemExit(f"{run} {T}：<T>.csv が無い")
    tab = pd.concat(out, ignore_index=True)
    if tab.duplicated(["wave", "rep", "row_id"]).any():
        raise SystemExit(f"{run} {T}：(wave, rep, row_id) が重複している")
    return tab


def chosen_setting(sel, run, T, student, K, selector):
    """results/e79/select/chosen.csv の selector（oursx は E79 の Claim 5 の P5-3 と同じ鍵），variant = incl（none を含めて選ぶ版）．
    戻り (wave, method, param, meff) か None（none が選ばれた）か "missing"．"""
    ch = sel[(sel.run == run) & (sel.target == T) & (sel.student == student) & (sel.K == K) & (sel.selector == selector) & (sel.variant == "incl")]
    if ch.empty:
        return "missing"
    if len(ch) > 1:
        raise SystemExit(f"chosen.csv：{run} {T} {student} K{K} {selector} incl の行が {len(ch)} 行")
    r = ch.iloc[0]
    cn = r.chose_none
    if (cn is True or str(cn) == "True") or r.method == "none" or pd.isna(r.method):
        return None
    return (r.wave, r.method, str(r.param), str(r.meff))


def chosen_ours(sel, run, T, student, K):
    """主の手法の行（oursx・incl）．"""
    return chosen_setting(sel, run, T, student, K, "oursx")


def load_labels_speakers(run, T, d77, prov):
    note_input(prov, d77 / f"{T}_samples.csv", with_sha=True)
    samp = pd.read_csv(d77 / f"{T}_samples.csv")
    df, _ = load_any(T, run)                                                   # 取り決め 1〜3 を満たすラベル（E79 と同じ）
    if not np.array_equal(df.sample_id.astype(str).to_numpy(), samp.sample_id.astype(str).to_numpy()):
        raise SystemExit(f"{run} {T}：_samples.csv の sample_id の並びが load_any と違う")
    return df.label_4.to_numpy().astype(int), samp.speaker_group.astype(str).to_numpy()


# ---------------------------------------------------------------- (A)(B)

def a_rows_from(run, T, X):
    """(A)：none の行の反復ごとの対ごとの量（ポイント）．pairs の acc_actual・acc_model はすでにポイント．"""
    rows = []
    N = X[(X.method == "none") & (X.param == "-")]
    for (st, K), g in N.groupby(["student", "K"]):
        if K not in KS:
            continue
        for r_ in g.itertuples():
            rows.append({"run": run, "target": T, "student": st, "K": K, "rep": int(r_.rep), "pair": int(r_.pair), "pair_name": PAIR_NAME[int(r_.pair)],
                         "A_D0": 100 * float(ba(r_.D, 0.0)), "none_dirloss": 100 * float(ba(r_.D, 0.0) - ba(r_.d, 0.0)),
                         "none_dirloss_auc": 100 * float(ba(r_.D, 0.0) - ba(r_.d_auc, 0.0)),
                         "none_posloss": 100 * float(ba(r_.d, 0.0) - ba(r_.d, r_.o)),
                         "none_model_err": float(r_.acc_model - r_.acc_actual), "none_d": float(r_.d), "none_d_auc": float(r_.d_auc), "D": float(r_.D),
                         "none_o": float(r_.o)})
    return rows


def stage_ab(args, out, prov):
    sel_path = ROOT / "results" / "e79" / "select" / "chosen.csv"
    sel = pd.read_csv(sel_path, dtype={"K": str, "param": str, "meff": str})
    note_input(prov, sel_path, with_sha=True)
    rep_rows, unit_rows, delta_rows, check_rows, boot_store = [], [], [], [], {}
    # ---- (A) D と none の向きの損．融合の実行は e79/measure から，種類の実行は e79_p4/measure から（二重に読まない．I8）
    a_rep = []
    for mdir, runs in ((ROOT / "results" / "e79" / "measure", RUNS), (ROOT / "results" / "e79_p4" / "measure", KINDS)):
        for run in runs:
            f = mdir / f"pairs_{run}.csv.gz"
            if not f.exists():
                check_rows.append({"check": "pairs_missing", "run": run, "path": str(f.relative_to(ROOT))})
                continue
            note_input(prov, f, with_sha=True)
            P = pd.read_csv(f, dtype={"row_id": int, "rep": int, "pair": int})
            for T in sorted(P.target.unique()):
                if args.targets and T not in args.targets:
                    continue
                tab = row_table(run, T, prov)
                X = dedupe_pairs(P[P.target == T], f"{run} {T}").merge(tab, on=["wave", "rep", "row_id", "student"], how="left")
                if X.method.isna().any():
                    raise SystemExit(f"{run} {T}：pairs の行が <T>.csv に無い")
                a_rep.extend([{**row, "source": str(mdir.relative_to(ROOT))} for row in a_rows_from(run, T, X)])
            if run in RUNS:                                                    # (B) は E79 の主の実行だけ（e79_p4 は P4 の判定用で oursx の行が無い）
                stage_b_run(args, run, P, sel, rep_rows, unit_rows, delta_rows, check_rows, boot_store, prov)
    AR = pd.DataFrame(a_rep)
    AR.to_csv(out / "a_none_pairs_rep.csv", index=False)
    A_ = pd.DataFrame()
    if len(AR):
        if AR.duplicated(["run", "target", "student", "K", "rep", "pair"]).any():
            raise SystemExit("(A)：(run, target, student, K, rep, pair) が重複している")
        qcols = ["A_D0", "none_dirloss", "none_dirloss_auc", "none_posloss", "none_model_err"]
        A_ = AR.groupby(["run", "target", "student", "K", "pair", "pair_name"]).agg(
            n_reps=("rep", "nunique"), **{c: (c, "mean") for c in qcols},
            **{f"{c}_rep_med": (c, "median") for c in ("A_D0", "none_dirloss")}, **{f"{c}_rep_min": (c, "min") for c in ("A_D0", "none_dirloss")},
            **{f"{c}_rep_max": (c, "max") for c in ("A_D0", "none_dirloss")},
            none_d=("none_d", "mean"), none_d_auc=("none_d_auc", "mean"), D=("D", "mean"), none_o_abs=("none_o", lambda x: float(x.abs().mean()))).reset_index()
    A_.to_csv(out / "a_none_pairs.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(out / "b_pairs_rep.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(out / "b_delta_pairs.csv", index=False)
    pd.DataFrame(check_rows).to_csv(out / "checks.csv", index=False)
    U = pd.DataFrame(unit_rows)
    U.to_csv(out / "b_units_pairs.csv", index=False)
    if boot_store:                                                             # 単位ごとのブートストラップの複製 (B, 6)（R1）
        np.savez_compressed(out / "b_boot_units.npz", **boot_store)
        pd.DataFrame([{"key": k, "B": v.shape[0]} for k, v in boot_store.items()]).to_csv(out / "b_boot_units_index.csv", index=False)
    return A_, U


def delta_pairs(run, d77, targets, prov):
    """対ごとの δ_d：仮のターゲットの none の 100·A(d, 0) の反復 0〜4 の SD ÷ √10 を (T, c) ごとに，(student, K, pair) で中央値 × 2．
    感度の列 delta_d_pair_corpus：仮のターゲットのコーパスごとに中央値を取ってからコーパスの間で中央値（E79 の delta_d_corpus と同じ作り．判定には使わない）．"""
    rows = []
    for T in targets:
        for f in sorted(d77.glob(f"{T}_ps_*_none.npz")):
            m = f.name[len(T) + 4:-len("_none.npz")]
            c, r = m.rsplit("_r", 1)
            if int(r) not in PSEUDO_REPS:
                continue
            note_input(prov, f)
            z = np.load(f)
            y = np.asarray(z["y"]).astype(int)
            for i, key in enumerate(z["keys"]):
                st, K = str(key).split("|")
                if K not in KS:
                    continue
                lp = np.asarray(z[f"lp{i}"], dtype=np.float64)
                if not np.isfinite(lp).all():
                    raise SystemExit(f"NaN/Inf：{f} {key}")
                d, _ = d_o_pairs(lp, y)
                for p in range(6):
                    rows.append({"run": run, "target": T, "pseudo": c, "pseudo_corpus": corp(c), "rep": int(r), "student": st, "K": K, "pair": p,
                                 "a_d0": 100 * float(ba(d[p], 0.0))})
    R = pd.DataFrame(rows)
    if R.empty:
        raise SystemExit(f"{run}：δ の材料（_ps_ のファイル）が無い")
    if not np.isfinite(R.a_d0).all():
        raise SystemExit(f"{run}：δ の材料の A(d, 0) に NaN/Inf（{int((~np.isfinite(R.a_d0)).sum())} 行）")
    g = R.groupby(["run", "target", "pseudo", "pseudo_corpus", "student", "K", "pair"]).agg(n_reps=("rep", "nunique"), sd=("a_d0", lambda x: x.std(ddof=1))).reset_index()
    g = g[g.n_reps >= 2]
    if g.empty:
        raise SystemExit(f"{run}：δ の材料に反復が 2 つ以上ある (T, c) が無い")
    g["s_d"] = g.sd / np.sqrt(10)
    D = g.groupby(["run", "student", "K", "pair"]).agg(n_units=("s_d", "size"), n_targets=("target", "nunique"), s_d=("s_d", "median")).reset_index()
    D["delta_d_pair"] = 2 * D.s_d
    pc = g.groupby(["run", "student", "K", "pair", "pseudo_corpus"]).s_d.median().groupby(["run", "student", "K", "pair"]).median().rename("s_d_corpus").reset_index()
    cn = g.groupby(["run", "student", "K", "pair", "pseudo_corpus"]).size().reset_index(name="n")
    cnt = pd.DataFrame([{"run": a, "student": b, "K": c, "pair": p, "pairs_by_corpus": ";".join(f"{x}:{y}" for x, y in zip(gg.pseudo_corpus, gg.n))}
                        for (a, b, c, p), gg in cn.groupby(["run", "student", "K", "pair"])])
    D = D.merge(pc, how="left").merge(cnt, how="left")
    D["delta_d_pair_corpus"] = 2 * D.s_d_corpus
    if not np.isfinite(D.delta_d_pair).all() or not np.isfinite(D.delta_d_pair_corpus).all():
        raise SystemExit(f"{run}：δ_d,対 に NaN/Inf")
    return D, R


def stage_b_run(args, run, P, sel, rep_rows, unit_rows, delta_rows, check_rows, boot_store, prov):
    d77 = ROOT / "results" / f"e77_{run}"
    targets = [t for t in sorted(P.target.unique()) if (not args.targets or t in args.targets)]
    D, Rd = delta_pairs(run, d77, targets, prov)
    Rd.to_csv(args.out / f"b_delta_reps_{run}.csv", index=False)
    delta_rows.extend(D.assign(run=run).to_dict("records"))
    dmap = {(r.student, r.K, int(r.pair)): float(r.delta_d_pair) for r in D.itertuples()}
    mods = WORLDS[run]
    students = mods + [run]
    # ---- 行の対応，ラベルと話者，反復ごとの対数確率
    by_corpus = {}
    for T in targets:
        by_corpus.setdefault(corp(T), []).append(T)
    for cp, Ts in by_corpus.items():
        prep = {}
        for T in Ts:
            tab = row_table(run, T, prov)
            X = dedupe_pairs(P[P.target == T], f"{run} {T}").merge(tab, on=["wave", "rep", "row_id", "student"], how="left")
            if X.method.isna().any():
                raise SystemExit(f"{run} {T}：pairs の行が <T>.csv に無い")
            yall, spk_all = load_labels_speakers(run, T, d77, prov)
            te = {r: np.load(d77 / f"{T}_r{r}_te.npy") for r in REPS}
            for r in REPS:
                note_input(prov, d77 / f"{T}_r{r}_te.npy")
            logp = {}
            for wave in ("e77", "e78", "e80c"):
                dd = ROOT / "results" / f"{wave}_{run}"
                if (dd / f"{T}_r0_logp.npy").exists():
                    logp[wave] = {r: (np.load(dd / f"{T}_r{r}_logp.npy", mmap_mode="r"), np.load(dd / f"{T}_r{r}_rowids.npy")) for r in REPS}
                    for r in REPS:
                        note_input(prov, dd / f"{T}_r{r}_logp.npy")
                        note_input(prov, dd / f"{T}_r{r}_rowids.npy")
            for wave in logp:
                for r in REPS:
                    if not np.array_equal(np.load(ROOT / "results" / f"{wave}_{run}" / f"{T}_r{r}_te.npy"), te[r]):
                        raise SystemExit(f"{run} {T} r{r}：{wave} の _te が E77 と違う")
            prep[T] = {"X": X, "y": yall, "spk": spk_all, "te": te, "logp": logp}
        speakers = sorted(set().union(*[set(p["spk"][p["te"][r]]) for p in prep.values() for r in REPS]))
        sid = {s: i for i, s in enumerate(speakers)}
        nls = len(speakers)
        C, A = boot_draws(run, cp, speakers, args.boot, args.seed, len(REPS))
        Wj = np.ones((nls, nls)) - np.eye(nls)
        # G_eff（E79：反復 0〜9 のテストの発話数を人物ごとに足す．群ごと）
        geff = {}
        for T, p in prep.items():
            cnt = pd.Series(np.concatenate([p["spk"][p["te"][r]] for r in REPS])).value_counts().to_numpy().astype(float)
            geff[T] = cnt.sum() ** 2 / (cnt ** 2).sum()
        for selector, comp in COMPARISONS:
            for st in students:
                for K in KS:
                    chs = {T: chosen_setting(sel, run, T, st, K, selector) for T in Ts}
                    if any(c == "missing" for c in chs.values()):
                        check_rows.append({"check": "chosen_missing", "comparison": comp, "run": run, "corpus": cp, "student": st, "K": K})
                        continue
                    if len({c for c in chs.values()}) > 1:                     # IEMOCAP の 2 群で選択が違う（I1）
                        check_rows.append({"check": "chosen_differs_in_corpus", "comparison": comp, "run": run, "corpus": cp, "student": st, "K": K,
                                           "chosen": ";".join(f"{T}:{'none' if c is None else '|'.join(c)}" for T, c in chs.items())})
                        if comp == MAIN_COMP:
                            raise SystemExit(f"{run} {cp} {st} K{K}：IEMOCAP の 2 群で {selector} の選択が違う（主の比較なので止める）")
                    per_T = {}
                    for T in Ts:
                        per_T[T] = pair_diffs_target(run, T, prep[T], chs[T], st, K, comp, sid, nls, C, Wj, min(geff.values()) < GEFF_MIN, rep_rows, check_rows)
                    all_none = all(c is None for c in chs.values())
                    chosen_str = "none" if all_none else ";".join(f"{T}:{'none' if c is None else '|'.join(c)}" for T, c in chs.items())
                    chosen_short = "none" if all_none else "|".join(next(c for c in chs.values() if c is not None))
                    # 単位（コーパス）に集める：主（全群），逆の順序，IEMOCAP 即興だけ（AVT．E79 の P5-3 と同じ）
                    specs = [("main", "dir", Ts), ("rev_order", "dir_rev", Ts)]
                    if run == "fuse-audio-video-text":
                        specs.append(("iemocap_impro_only", "dir", [T for T in Ts if T != "IEMOCAP-script"]))
                    for stag, metric, Ts_use in specs:
                        if not Ts_use:
                            continue
                        aggregate_unit(run, cp, st, K, comp, metric, stag, Ts_use, per_T, A, geff, dmap, all_none, chosen_short, chosen_str, unit_rows, boot_store)


def pair_diffs_target(run, T, p, chT, st, K, comp, sid, nls, C, Wj, need_jk, rep_rows, check_rows):
    """1 つのターゲット（群）：反復ごとの対ごとの点の値（pairs の表），対数の確率からの照合，ブートストラップと jackknife の複製．
    chT が None（none を選んだ群）なら差を 0 とする（E79 と同じ）．"""
    X = p["X"]
    R = len(REPS)
    res = {k: np.full((R, 6), np.nan) for k in ("pts", "aucs", "des", "dos", "nonelosses", "revs")}
    res["boot"], res["boot_rev"], res["jk"], res["jk_rev"] = {}, {}, {}, {}
    for r in REPS:
        xn = X[(X.rep == r) & (X.student == st) & (X.K == K) & (X.method == "none") & (X.param == "-")].sort_values("pair")
        if len(xn) != 6:
            check_rows.append({"check": "none_rows", "comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "n": len(xn)})
            continue
        dn, on, Dn = xn.d.to_numpy(), xn.o.to_numpy(), xn.D.to_numpy()
        res["nonelosses"][r] = 100 * (ba(Dn, 0.0) - ba(dn, 0.0))
        if chT is None:
            for k in ("pts", "aucs", "des", "dos", "revs"):
                res[k][r] = 0.0
            res["boot"][r] = np.zeros((len(C), 6))
            res["boot_rev"][r] = np.zeros((len(C), 6))
            if need_jk:
                res["jk"][r] = np.zeros((nls, 6))
                res["jk_rev"][r] = np.zeros((nls, 6))
            chosen_lab = "none"
        else:
            wave, meth, par, me = chT
            xo = X[(X.rep == r) & (X.student == st) & (X.K == K) & (X.wave == wave) & (X.method == meth) & (X.param == par) & (X.meff == me)].sort_values("pair")
            if len(xo) != 6:
                check_rows.append({"check": "ours_rows", "comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "n": len(xo), "chosen": "|".join(chT)})
                continue
            if not np.allclose(xo.D.to_numpy(), Dn, rtol=0, atol=1e-9):
                raise SystemExit(f"{run} {T} {st} K{K} r{r}：手法の行と none の行で D が違う")
            do_, oo = xo.d.to_numpy(), xo.o.to_numpy()
            res["pts"][r] = dir_diff(do_, dn)
            res["aucs"][r] = 100 * (ba(xo.d_auc.to_numpy(), 0.0) - ba(xn.d_auc.to_numpy(), 0.0))
            res["des"][r] = (xo.acc_model.to_numpy() - xo.acc_actual.to_numpy()) - (xn.acc_model.to_numpy() - xn.acc_actual.to_numpy())   # すでにポイント（I7）
            res["dos"][r] = oo - on
            res["revs"][r] = dir_diff_rev(Dn, do_, oo, dn, on)
            # 対数の確率から計算し直して照合し，ブートストラップの材料にする
            yte = p["y"][p["te"][r]]
            sidx = np.array([sid[s] for s in p["spk"][p["te"][r]]])
            Ln = logp_row(p["logp"], "e77", r, int(xn.row_id.iloc[0]))
            Lo = logp_row(p["logp"], wave, r, int(xo.row_id.iloc[0]))
            dn2, on2 = d_o_pairs(Ln, yte)
            do2, oo2 = d_o_pairs(Lo, yte)
            for name, rec, ref in (("dir", dir_diff(do2, dn2), res["pts"][r]), ("dir_rev", dir_diff_rev(Dn, do2, oo2, dn2, on2), res["revs"][r])):
                mx, mism = max_abs_or_nan_mismatch(rec, ref)
                if mx > 1e-6 or mism:
                    raise SystemExit(f"{run} {T} {st} K{K} r{r} {comp} {name}：対ごとの点の値が pairs の表と合わない（最大 {mx:.2e}，NaN の食い違い {mism}）")
            Fn, Fo = pair_feats(Ln, yte, sidx, nls), pair_feats(Lo, yte, sidx, nls)
            for W, key in ((C, "boot"), (Wj, "jk")):
                if key == "jk" and not need_jk:
                    continue
                dBn, oBn = pair_d_o_from_feats(Fn, W)
                dBo, oBo = pair_d_o_from_feats(Fo, W)
                res[key][r] = dir_diff(dBo, dBn)
                res[key + "_rev"][r] = dir_diff_rev(Dn[None, :], dBo, oBo, dBn, oBn)
            chosen_lab = "|".join(chT)
        for pidx in range(6):
            rep_rows.append({"comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "pair": pidx, "pair_name": PAIR_NAME[pidx],
                             "diff_dir": res["pts"][r, pidx], "diff_dir_rev": res["revs"][r, pidx], "diff_dir_auc": res["aucs"][r, pidx],
                             "diff_model_err": res["des"][r, pidx], "diff_o": res["dos"][r, pidx], "none_dirloss": res["nonelosses"][r, pidx], "chosen": chosen_lab})
    return res


def aggregate_unit(run, cp, st, K, comp, metric, stag, Ts_use, per_T, A, geff, dmap, all_none, chosen_short, chosen_str, unit_rows, boot_store):
    """単位（コーパス）に集める：反復の平均をターゲットで平均（E79 と同じ）．区間は反復の重み A で反復を混ぜ，ターゲットで平均した複製の百分位．
    G_eff < 20 の群を含む単位は jackknife の t 区間との和集合．"""
    pk, bk, jk = ("pts", "boot", "jk") if metric == "dir" else ("revs", "boot_rev", "jk_rev")
    x = np.nanmean(np.stack([np.nanmean(per_T[T][pk], 0) for T in Ts_use]), 0)   # (6,)
    xa = np.nanmean(np.stack([np.nanmean(per_T[T]["aucs"], 0) for T in Ts_use]), 0)
    de = np.nanmean(np.stack([np.nanmean(per_T[T]["des"], 0) for T in Ts_use]), 0)
    dO = np.nanmean(np.stack([np.nanmean(per_T[T]["dos"], 0) for T in Ts_use]), 0)
    nl = np.nanmean(np.stack([np.nanmean(per_T[T]["nonelosses"], 0) for T in Ts_use]), 0)
    gmin = float(min(geff[T] for T in Ts_use))
    lo = hi = np.full(6, np.nan)
    lo_j = hi_j = np.full(6, np.nan)
    nvalid = 0
    rule = "pct"
    b = None
    if all(len(per_T[T][bk]) == len(REPS) for T in Ts_use):
        per = []
        for T in Ts_use:
            arr = np.stack([per_T[T][bk][r] for r in REPS], 1)                 # (B, R, 6)
            w = A[:, :, None]
            fin = np.isfinite(arr)
            per.append(np.where(fin, arr, 0).__mul__(w).sum(1) / np.where(fin, w, 0).sum(1))
        b = np.nanmean(np.stack(per), 0)                                       # (B, 6)
        lo, hi = np.nanpercentile(b, 2.5, 0), np.nanpercentile(b, 97.5, 0)
        nvalid = int(np.isfinite(b).all(1).sum())
        if gmin < GEFF_MIN:
            if all(len(per_T[T][jk]) == len(REPS) for T in Ts_use):
                th = np.nanmean(np.stack([np.nanmean(np.stack([per_T[T][jk][r] for r in REPS], 1), 1) for T in Ts_use]), 0)   # (G, 6)
                lo_j, hi_j = np.array([jk_interval(x[pp], th[:, pp], gmin) for pp in range(6)]).T
                lo, hi = np.fmin(lo, lo_j), np.fmax(hi, hi_j)
                rule = "union_pct_jk"
            else:
                rule = "pct_jk_missing"
                lo = hi = np.full(6, np.nan)
    else:
        rule = "no_replicates"
    if b is not None:
        boot_store[f"{comp}|{run}|{st}|{K}|{cp}|{stag}"] = b.astype(np.float32)
    scales = ((1.0, stag),) if stag != "main" else ((1.0, "main"), (0.5, "delta_half"), (2.0, "delta_double"))
    for pidx in range(6):
        dl = dmap.get((st, K, pidx), np.nan)
        for scale, sname in scales:
            dls = dl * scale
            md = np.nan
            if metric == "dir" and np.isfinite(x[pidx]) and np.isfinite(xa[pidx]) and np.isfinite(dls):   # 模型に依る印は向きの損（主の順序）だけ（E79 と同じ）
                md = bool(abs(x[pidx] - xa[pidx]) > MODEL_UNIT_FRAC * dls or abs(de[pidx]) > MODEL_UNIT_FRAC * dls or zone(x[pidx], dls) != zone(xa[pidx], dls))
            jd = "same_by_definition" if all_none else judge_unit(x[pidx], lo[pidx], hi[pidx], dls)
            unit_rows.append({"comparison": comp, "run": run, "role": role_of(run, st), "student": st, "K": K, "corpus": cp, "n_groups": len(Ts_use),
                              "groups": ";".join(Ts_use), "metric": metric, "pair": pidx, "pair_name": PAIR_NAME[pidx], "point": x[pidx],
                              "lo": lo[pidx], "hi": hi[pidx], "lo_jk": lo_j[pidx], "hi_jk": hi_j[pidx], "interval_rule": rule, "g_eff_min": gmin,
                              "n_boot_valid": nvalid, "delta_d_pair": dls, "sensitivity": sname, "judgment": jd, "model_dependent": md,
                              "point_auc": xa[pidx] if metric == "dir" else np.nan, "model_err_diff": de[pidx] if metric == "dir" else np.nan,
                              "diff_o": dO[pidx], "none_dirloss": nl[pidx], "chosen": chosen_short, "chosen_by_group": chosen_str})


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
                if int(r) not in PSEUDO_REPS:
                    continue
                note_input(prov, f)
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
                    preds["ref"] = np.asarray(z[f"ref__{K}"])[keep]            # agree は pF の行ごとの正の定数倍で判定が pF と同じなので出さない（I1）
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
                    # 対ごとの補い合い J_kl(f, g)．f は単モダリティと融合（none__<実行>__<K>）．g は other_m（f と別のモダリティ），pF，poe0，ref，ref0
                    for st in mods + [run]:
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
                                rows.append({**tag, "student": st, "role": role_of(run, st), "g": name, "pair": p, "pair_name": PAIR_NAME[p], "n_pair": int(m.sum()),
                                             "acc_student": float(sr.mean()), "acc_g": float(gr.mean()),
                                             "J": 100 * float((gr & ~sr).mean() - (~gr & sr).mean()),
                                             "g_right_s_wrong": float((gr & ~sr).mean()), "g_wrong_s_right": float((~gr & sr).mean())})
    J = pd.DataFrame(rows)
    J.to_csv(out / "c_complementarity_units.csv", index=False)
    pd.DataFrame(rec_rows).to_csv(out / "c_recall_units.csv", index=False)
    pd.DataFrame(phi_rows).to_csv(out / "c_phi_units.csv", index=False)
    return J, pd.DataFrame(rec_rows), pd.DataFrame(phi_rows)


# ---------------------------------------------------------------- まとめと予言の判定

def judgment_table(m, path):
    keys = ["comparison", "run", "role", "student", "pair_name"]
    if m.empty:
        cnt = pd.DataFrame(columns=keys + JUDG_COLS)
    else:
        cnt = m.groupby(keys + ["judgment"]).size().unstack(fill_value=0)
        cnt = cnt.reindex(columns=JUDG_COLS, fill_value=0).reset_index()       # 列を固定する（I1）
    cnt.to_csv(path, index=False)
    return cnt


def perm_share_null(m, sel_pairs, n_perm, seed):
    """割合の偶然の分布（S2）：コーパスの塊の中で対の名前を並べ替える（コーパスごとに 6 対の名前の並べ替えを 1 つ引き，その塊の全行に当てる）．
    判定の列は固定．「減らした」行のうち sel_pairs の割合の分布を返す（n_perm,）．"""
    rng = np.random.default_rng(seed)
    pos = m.judgment.to_numpy() == "pos"
    if not pos.any():
        return np.full(n_perm, np.nan)
    corpora = m.corpus.to_numpy()
    pair = m.pair.to_numpy().astype(int)
    cps = sorted(set(corpora))
    sel = np.zeros(6, bool)
    sel[list(sel_pairs)] = True
    out = np.empty(n_perm)
    for i in range(n_perm):
        newpair = np.empty_like(pair)
        for cp in cps:
            perm = rng.permutation(6)
            mk = corpora == cp
            newpair[mk] = perm[pair[mk]]
        out[i] = sel[newpair[pos]].mean()
    return out


def p3a_table(main, out, n_perm):
    """族ごとの P3a の判定（3 節）．数は（コーパス，対）の塊で数え，土台は none の損が δ 以上の（単位，対）に条件づけた割合，割合の偶然の分布は並べ替え．"""
    fam_rows, perm_store = [], {}
    fams = (("audio", ["audio_wavlm"], ANGER_PAIRS, 0.5, "怒りを含む 3 対", True),
            ("video", ["video"], (NEU_ANG,), 1 / 6, "平静–怒り", False),
            ("text", ["text_roberta"], ANGER_PAIRS, 0.5, "怒りを含む 3 対（記述）", True),
            ("fused", list(RUNS), ANGER_PAIRS, 0.5, "怒りを含む 3 対（記述）", True))
    for fam, stus, sel_pairs, base, label, main_with_md in fams:
        for use_md in (True, False):
            m = main[main.student.isin(stus)]
            if not use_md:
                m = m[m.model_dependent != True]  # noqa: E712
            ukey = ["run", "student", "K", "corpus"]
            n_units = len(m[ukey].drop_duplicates())
            n_tests = len(m)
            m_nn = m[m.chosen != "none"]                                       # none を選んだ単位を除いた数（S4・D11）
            n_units_nonone = len(m_nn[ukey].drop_duplicates())
            n_tests_nonone = len(m_nn)
            pos = m[m.judgment == "pos"]
            n_pos = len(pos)
            # 塊で数える（S1・S7・S4）：（コーパス，対）に畳む．どの世界・K かで 1 回でも減らせば 1
            blk = m.assign(is_pos=(m.judgment == "pos")).groupby(["corpus", "pair"]).agg(pos_any=("is_pos", "any"), n=("is_pos", "size")).reset_index()
            blk["pos_any"] = blk.pos_any.astype(bool)
            n_blocks = len(blk)
            n_pos_blocks = int(blk.pos_any.sum()) if n_blocks else 0
            chance_blocks = CHANCE * n_blocks
            chance_blocks_anyof = float((1 - (1 - CHANCE) ** blk.n).sum())
            pos_corp = pos.groupby("corpus").size()
            n_pos_corpora = int((blk.groupby("corpus").pos_any.any()).sum())
            share = float(pos.pair.isin(sel_pairs).mean()) if n_pos else np.nan
            share_blk = float(blk[blk.pos_any].pair.isin(sel_pairs).mean()) if n_pos_blocks else np.nan
            elig = m[(m.none_dirloss >= m.delta_d_pair) & (m.chosen != "none") & (m.judgment != "missing")]   # 条件つきの土台（D6・L1）
            base_cond = float(elig.pair.isin(sel_pairs).mean()) if len(elig) else np.nan
            null = perm_share_null(m, sel_pairs, n_perm, [PERM_SEED, zlib.crc32(fam.encode()), int(use_md)])
            perm_store[f"{fam}|md{int(use_md)}"] = null.astype(np.float32)
            q95 = float(np.nanpercentile(null, 95)) if np.isfinite(null).any() else np.nan
            p_perm = float(np.nanmean(null >= share)) if (np.isfinite(null).any() and np.isfinite(share)) else np.nan
            c_count = bool(n_pos_blocks > chance_blocks)
            c_corp = bool(n_pos_corpora >= 2)
            c_cond = bool(np.isfinite(share) and np.isfinite(base_cond) and share > base_cond)
            c_fixed = bool(np.isfinite(share) and share > base)
            c_perm = bool(np.isfinite(share) and np.isfinite(q95) and share > q95)
            judged = fam in ("audio", "video")
            fam_rows.append({"family": fam, "with_model_dependent": use_md, "is_main_version": use_md == main_with_md,
                             "n_units": n_units, "n_tests": n_tests, "chance_pos": CHANCE * n_tests,
                             "n_units_nonone": n_units_nonone, "n_tests_nonone": n_tests_nonone, "chance_pos_nonone": CHANCE * n_tests_nonone,
                             "n_pos": n_pos, "n_neg": int((m.judgment == "neg").sum()), "n_same": int((m.judgment == "same").sum()),
                             "n_undet": int((m.judgment == "undet").sum()), "n_missing": int((m.judgment == "missing").sum()),
                             "n_same_by_definition": int((m.judgment == "same_by_definition").sum()),
                             "n_blocks": n_blocks, "n_pos_blocks": n_pos_blocks, "chance_pos_blocks": chance_blocks, "chance_pos_blocks_anyof": chance_blocks_anyof,
                             "n_pos_corpora": n_pos_corpora, "pos_corpora": ";".join(f"{c}:{n}" for c, n in pos_corp.items()),
                             "selected_pairs": label, "share_selected_among_pos": share, "share_selected_among_pos_blocks": share_blk,
                             "base_share": base, "base_share_cond": base_cond, "n_eligible_cond": len(elig),
                             "perm_q95": q95, "perm_p": p_perm, "n_perm": n_perm,
                             "share_p3a_side_among_pos": float(pos.pair.isin(P3A_SIDE_PAIRS).mean()) if n_pos else np.nan,
                             "share_p3b_side_among_pos": float(pos.pair.isin(P3B_SIDE_PAIRS).mean()) if n_pos else np.nan,
                             "n_pos_happy_anger": int((pos.pair == HAPPY_ANGER).sum()),
                             "share_arousal_pairs_among_pos": float(pos.pair.isin(AROUSAL_PAIRS).mean()) if n_pos else np.nan, "base_arousal": 2 / 6,
                             "cond_count_blocks": c_count, "cond_two_corpora": c_corp, "cond_share_gt_base_cond": c_cond, "cond_share_gt_base_fixed": c_fixed,
                             "cond_share_gt_perm_q95": c_perm,
                             "P3a_holds": bool(c_count and c_corp and c_cond and c_perm) if judged else None,
                             "P3a_holds_base_fixed": bool(c_count and c_corp and c_fixed and c_perm) if judged else None,
                             "pos_pairs": ";".join(f"{PAIR_NAME[p]}:{n}" for p, n in pos.pair.value_counts().sort_index().items())})
    F = pd.DataFrame(fam_rows)
    F.to_csv(out / "b_p3_checks.csv", index=False)
    np.savez_compressed(out / "b_p3a_perm.npz", **perm_store)
    return F


def summarize(out, A_, U, J, R, Ph, n_perm):
    lines = ["# E83 のまとめ（台本が自動で書く．読み方は docs/exploratory-analyses.md の E83 3 節）", ""]
    # (A)
    if len(A_):
        qs = {"A_D0": "A_D0", "none_dirloss": "none_dirloss", "none_dirloss_auc": "none_dirloss_auc", "none_model_err": "none_model_err", "none_posloss": "none_posloss"}
        agg = {}
        for c in qs:
            agg[f"{c}_med"] = (c, "median")
            agg[f"{c}_min"] = (c, "min")
            agg[f"{c}_max"] = (c, "max")
            agg[f"{c}_q1"] = (c, lambda x: x.quantile(0.25))
            agg[f"{c}_q3"] = (c, lambda x: x.quantile(0.75))
        a_all = A_.groupby(["run", "student", "K", "pair_name"]).agg(n_targets=("target", "nunique"), **agg).reset_index()
        a_all.to_csv(out / "a_summary_by_K.csv", index=False)
        a = a_all[a_all.K == "0"].drop(columns="K")
        a.to_csv(out / "a_summary_K0.csv", index=False)
        show = ["run", "student", "pair_name", "n_targets", "A_D0_med", "A_D0_min", "A_D0_max", "none_dirloss_med", "none_dirloss_min", "none_dirloss_max",
                "none_dirloss_auc_med", "none_model_err_med"]
        lines += ["## (A) K=0 の none：対ごとの A(D,0) と向きの損（反復の平均のターゲットの中央値と範囲．d_AUC 版と模型の誤差も）", "", a[show].to_string(index=False), ""]
    # (B) 判定の数
    if len(U):
        main = U[(U.comparison == MAIN_COMP) & (U.sensitivity == "main")]
        cnt = judgment_table(main, out / "b_judgments_by_pair.csv")
        nomd = judgment_table(main[main.model_dependent != True], out / "b_judgments_by_pair_nomd.csv")  # noqa: E712
        for sname in ("delta_half", "delta_double", "rev_order", "iemocap_impro_only"):
            judgment_table(U[(U.comparison == MAIN_COMP) & (U.sensitivity == sname)], out / f"b_judgments_by_pair_{sname}.csv")
        for _, comp in COMPARISONS[1:]:                                        # kd・abl_own（補助．主の数え上げには入れない）
            judgment_table(U[(U.comparison == comp) & (U.sensitivity == "main")], out / f"b_judgments_by_pair_{comp.split('-')[-1]}.csv")
        lines += ["## (B) 単位 × 対の判定の数（主：none − oursx．pos＝減らした，neg＝増やした，same＝同じ水準，undet＝判定できない）", "", cnt.to_string(index=False), ""]
        lines += ["## (B) 同じく，模型に依る対を除いた数", "", nomd.to_string(index=False), ""]
        F = p3a_table(main, out, n_perm)
        showF = ["family", "with_model_dependent", "is_main_version", "n_tests", "n_tests_nonone", "n_pos", "n_blocks", "n_pos_blocks", "chance_pos_blocks",
                 "n_pos_corpora", "share_selected_among_pos", "base_share", "base_share_cond", "perm_q95", "P3a_holds", "P3a_holds_base_fixed", "pos_pairs"]
        lines += ["## (B) 予言 P3a の判定（族ごと．数は（コーパス，対）の塊，土台は条件つき，割合の偶然は並べ替え）", "", F[showF].to_string(index=False), ""]
    # (C)
    if len(J):
        Jc = J.groupby(["world", "student", "role", "K", "g", "pair_name", "pseudo_corpus"]).J.median().reset_index()      # コーパスの中央値を先に取る（S7）
        Js = Jc.groupby(["world", "student", "role", "K", "g", "pair_name"]).agg(J_med=("J", "median"), J_q1=("J", lambda x: x.quantile(0.25)),
                                                                                 J_q3=("J", lambda x: x.quantile(0.75)), n_corpora=("pseudo_corpus", "nunique")).reset_index()
        Js.to_csv(out / "c_complementarity_summary.csv", index=False)
        Jc.to_csv(out / "c_complementarity_by_corpus.csv", index=False)
        lines += ["## (C) 対ごとの補い合い J_kl(f, g) の中央値（仮のターゲットのコーパスの中央値を取ってから中央値）", "",
                  Js[Js.g.isin(["pF", "ref", "poe0"])].to_string(index=False), ""]
        # P3c：(B) で減った対について，同じコーパスを仮のターゲットにした J(f, ref) の中央値が正か（単モダリティと融合の f．none を選んだ単位と J の無い行は除く）
        if len(U):
            main = U[(U.comparison == MAIN_COMP) & (U.sensitivity == "main")]
            wmap = {"fuse-audio-video-text": "avt", "fuse-audio-video": "av"}
            jref = Jc[Jc.g == "ref"].set_index(["world", "student", "K", "pair_name", "pseudo_corpus"]).J
            rows = []
            for r in main.itertuples():
                key = (wmap[r.run], r.student, r.K, r.pair_name, r.corpus)
                rows.append({"run": r.run, "role": r.role, "student": r.student, "K": r.K, "corpus": r.corpus, "pair_name": r.pair_name, "judgment": r.judgment,
                             "model_dependent": r.model_dependent, "chosen": r.chosen, "J_ref_same_corpus": float(jref.get(key, np.nan))})
            Q = pd.DataFrame(rows)
            Q.to_csv(out / "b_c_p3c_rows.csv", index=False)
            n_all = len(Q)
            Qk = Q[(Q.chosen != "none") & Q.J_ref_same_corpus.notna()]
            p3c_rows = []
            for scope, Qs in (("all", Qk), ("uni", Qk[Qk.role == "uni"]), ("fused", Qk[Qk.role == "fused"]), ("nomd", Qk[Qk.model_dependent != True])):  # noqa: E712
                base = float((Qs.J_ref_same_corpus > 0).mean()) if len(Qs) else np.nan
                red = Qs[Qs.judgment == "pos"]
                share = float((red.J_ref_same_corpus > 0).mean()) if len(red) else np.nan
                p3c_rows.append({"scope": scope, "n_rows_all": n_all, "n_rows_used": len(Qs), "n_excluded_none_chosen": int((Q.chosen == "none").sum()),
                                 "n_excluded_J_nan": int(Q.J_ref_same_corpus.isna().sum()), "base_share_J_pos": base, "n_reduced": len(red),
                                 "share_J_pos_among_reduced": share, "P3c_holds": bool(len(red) and np.isfinite(share) and share > base)})
            P3c = pd.DataFrame(p3c_rows)
            P3c.to_csv(out / "b_c_p3c.csv", index=False)
            lines += ["## P3c：減った対のうち J(f, ref) > 0 の割合と土台（主は scope = all）", "", P3c.to_string(index=False), ""]
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
    ap.add_argument("--perm", type=int, default=1000, help="P3a の割合の並べ替えの回数")
    ap.add_argument("--seed", type=int, default=0)                             # E79 と同じ種
    ap.add_argument("--targets", nargs="+", help="試走用．評価のターゲットを限る")
    ap.add_argument("--stages", nargs="+", default=["ab", "c"], choices=["ab", "c"])
    ap.add_argument("--smoke", action="store_true", help="試走（--boot 50・--perm 100 と同じ扱い．provenance に残す）")
    args = ap.parse_args()
    if args.smoke:
        args.boot = min(args.boot, 50)
        args.perm = min(args.perm, 100)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "exp", "measure"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    prov = {"commit": commit, "dirty_exp_measure": bool(dirty), "script_sha256": sha256(Path(__file__)), "args": vars(args) | {"out": str(out)},
            "comparisons": [c for _, c in COMPARISONS], "main_comparison": MAIN_COMP, "perm_seed": PERM_SEED,
            "started": dt.datetime.now().isoformat(timespec="seconds"), "inputs": {}}
    A_ = U = J = R = Ph = pd.DataFrame()
    if "ab" in args.stages:
        A_, U = stage_ab(args, out, prov)
    if "c" in args.stages:
        J, R, Ph = stage_c(args, out, prov)
    summarize(out, A_, U, J, R, Ph, args.perm)
    prov["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
