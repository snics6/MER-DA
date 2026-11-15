"""E83：向きの情報はどの感情の対にあるか．対ごとの方向の損の分解（評価のターゲット．E79 の保存物）と，仮のターゲットでの補い合い（E80 の units）．

設計：docs/exploratory-analyses.md の E83（規則は値を見る前に書いた）．新しい学習はしない．(A)(B) は評価のターゲットの値を対に分けて判定するので本番として扱う．

  (A) 対ごとの D と none の向きの損（results/e79/measure/D_*.csv，pairs_*.csv.gz．種類の実行の none は results/e79_p4/measure から）．
      反復ごとの行と，反復・ターゲットの中央値と範囲，d_AUC から作った向きの損，模型の誤差 acc_model − acc_actual，負の対の数，分散の比を残す
  (B) 対ごとの手法（oursx．none を含む版の選択．E79 の Claim 5 の P5-3 と同じ鍵）− none の向きの損の差を，E79 と同じ単位・2 段のブートストラップ・jackknife で判定する．
      δ_d は対ごとに仮のターゲットの none の揺れから作る．模型に依る対の印も対ごとに付ける．
      感度の行は δ の半分・2 倍，逆の順序（A(D,o) − A(d,o)．rev_order），IEMOCAP 即興だけ（iemocap_impro_only），IEMOCAP のセッションを 1 つずつ抜く jackknife（session_jk）．
      比較相手 kd・abl_own・abl_ownK・groupB（none を含む版の選択．種のある手法は種 0〜2 の平均）の行も同じ計算で出し，手法どうしの直接の差（oursx − 相手）は
      同じ抽出の複製の差から判定する．単位ごとのブートストラップと jackknife の複製は npz に残す
  (C) 仮のターゲットの候補の上で，f のアンカー（単モダリティと融合）と予測モデル g の 2 感情の判定を比べ，対ごとの補い合い J_kl（切片を含む）と向きだけの版 J_dir，
      感情ごとの再現率，誤りの φ を出す．agree は pF の行ごとの正の定数倍で判定が pF と同じなので出さない
  P3a の判定：数は複製から作った帰無の 95% 点，広がりは怒りの対の「減らした」が出たコーパスの数，偏りは効果の出うるセルに限った感情の名前の置換（24 通り）の厳密な p．

使い方：./.venv/bin/python exp/analyze_pairs_e83.py [--out results/e83] [--boot 1000] [--targets ...] [--stages ab c] [--smoke]
  試走（--smoke，--targets，--stages の一部）の出力は results/ の下に書けない（--out を checks/E83/ か scratchpad にする）．本番は既定の引数で 1 回で流す．
  出力の CSV は先頭に `#` の行（コミットと設定）を持つので，読むときは pd.read_csv(..., comment="#") にする．
"""
import argparse
import datetime as dt
import hashlib
import itertools
import json
import re
import shutil
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
CHANCE = 0.025                                                                 # 差が 0 のとき偶然「減らした」に当たる割合（両側 5% の片側）．記述の列にだけ使う
DELTA_SCALES = ((1.0, "main"), (0.5, "delta_half"), (2.0, "delta_double"))
MIN_PAIR_N = 10                                                                # (C) の J と φ に要る最小の本数
P_ALPHA = 0.05                                                                 # P3a (iii) の厳密な p の基準
COMPARISONS = (("oursx", "none-oursx"), ("kd", "none-kd"), ("abl_own", "none-abl_own"), ("abl_ownK", "none-abl_ownK"), ("groupB", "none-groupB"),
               ("abl_refb", "none-abl_refb"), ("abl_ref0", "none-abl_ref0"))   # abl_refb は単モダリティの f だけ，abl_ref0 は融合の f だけにある（候補 20 個．D12）
PARTIAL_SELECTORS = {"abl_refb": "uni", "abl_ref0": "fused"}                   # 無い f では chosen_missing_expected として記録して飛ばす
MAIN_COMP = "none-oursx"
DIRECT = (("none-oursx", "none-kd", "oursx-kd"), ("none-oursx", "none-abl_own", "oursx-abl_own"), ("none-oursx", "none-abl_ownK", "oursx-abl_ownK"),
          ("none-oursx", "none-groupB", "oursx-groupB"), ("none-abl_refb", "none-abl_ownK", "abl_refb-abl_ownK"), ("none-abl_ref0", "none-abl_ownK", "abl_ref0-abl_ownK"))
CEBD64B_CHANCE = {"audio": 144 * CHANCE, "video": 144 * CHANCE, "text": 54 * CHANCE, "fused": 144 * CHANCE}   # cebd64b の規則の数の基準（族ごとに固定．顔も 3.6）
NEAR_BOUNDARY = 2.0                                                            # 効果の出うるセルの境界の近く（ポイント．仮定の表の 5 行目）
JUDG_COLS = ["pos", "neg", "same", "undet", "missing", "same_by_definition"]
SEEDED = {"dann", "mme", "mapctl", "xmuda", "entmin", "im"}                   # E79 と同じ（評価は種 0・1・2 の平均．選択は種 0）
NC_SEEDED = {"entmin", "im"}                                                   # E77 の非凸の手法：種 0 の param は末尾に s が無い
EXCL_MA = ("MOSEI", "AFEW")                                                    # 話者が代用の定義のコーパス（E79 の excl_mosei_afew）
TEACHER_G = {"ref": "ref", "ref0": "ref0", "agree": "pF", "pF": "pF"}          # oursx が選んだ教師 → (C) の g（pFK は units に無いので欠け）
CLASS_PERM_MAPS = np.array([[PAIR_IDX[tuple(sorted((s[k], s[j])))] for (k, j) in PAIRS] for s in itertools.permutations(range(4))])   # (24, 6)
PAIR_PERM_MAPS = np.array(list(itertools.permutations(range(6))))             # (720, 6)．感度の列にだけ使う
HEADER = "E83"                                                                 # 各 CSV の先頭の # 行（main で設定する）
SCRIPT_DEPS = ("exp/analyze_pairs_e83.py", "exp/explore_pairs.py", "exp/explore_synergy.py", "exp/multimodal_distill.py", "measure/data.py")


def strip_seed(p):
    return re.sub(r"s\d+$", "", p)


def seed_of(method, p):
    """E79 と同じ．種の番号．E78 の手法は param の末尾 `s<種>`．E77 の entmin・im は種 0 の param に末尾の s が無い．"""
    if method not in SEEDED:
        return -1
    m = re.search(r"s(\d+)$", p)
    if m:
        return int(m.group(1))
    return 0 if method in NC_SEEDED else -1


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


def write_csv(df, path):
    """各 CSV の先頭に # の行（コミットと設定）を書く（methods.md 6.3）．読むときは comment="#"．"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {HEADER}\n")
        df.to_csv(fh, index=False)


def corp(g):
    return GROUP_SPEC[g][0]


def session_of(speaker_group):
    """IEMOCAP の話者群 Ses0x_F/M → セッション Ses0x（セッションの jackknife に使う）．"""
    return str(speaker_group)[:5]


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


def recall_by_emotion(L, y):
    am = L.argmax(1)
    return np.array([float((am[y == e] == e).mean()) if (y == e).any() else np.nan for e in range(4)])


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
    """(wave, rep, row_id) → (student, method, param, pnorm, seed, meff, K)．E77・E78・E80 の段 c の <T>.csv から．"""
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
    tab["pnorm"] = [strip_seed(p) if m in SEEDED else p for m, p in zip(tab.method, tab.param)]   # 種を除いた param（E79 と同じ）
    tab["seed"] = [seed_of(m, p) for m, p in zip(tab.method, tab.param)]
    return tab


def chosen_setting(sel, run, T, student, K, selector):
    """results/e79/select/chosen.csv の selector（oursx は E79 の Claim 5 の P5-3 と同じ鍵），variant = incl（none を含めて選ぶ版）．
    戻り (wave, method, pnorm, meff) か None（none が選ばれた）か "missing"．chosen.csv の param は種を除いた形（E79 の pnorm）．"""
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


def n_cand_of(sel, run, T, student, K, selector):
    ch = sel[(sel.run == run) & (sel.target == T) & (sel.student == student) & (sel.K == K) & (sel.selector == selector) & (sel.variant == "incl")]
    return int(ch.iloc[0].n_cand) if len(ch) == 1 and pd.notna(ch.iloc[0].n_cand) else -1


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
                         "none_o": float(r_.o), "none_var_ratio": float(getattr(r_, "var_ratio", np.nan))})
    return rows


def stage_ab(args, out, prov):
    sel_path = ROOT / "results" / "e79" / "select" / "chosen.csv"
    sel = pd.read_csv(sel_path, dtype={"K": str, "param": str, "meff": str})
    note_input(prov, sel_path, with_sha=True)
    # 比較相手ごとの候補の数と none を選んだ数（C2・C6．選択の事実で，評価の値ではない）
    cand = sel[(sel.variant == "incl") & sel.K.isin(KS) & sel.run.isin(RUNS) & sel.selector.isin([s for s, _ in COMPARISONS])]
    cand = cand[["run", "target", "student", "K", "selector", "n_cand", "chose_none", "wave", "method", "param", "meff"]].copy()
    write_csv(cand, out / "b_candidates.csv")
    prov["candidates"] = {s: {"n_rows": int((cand.selector == s).sum()), "n_none": int(((cand.selector == s) & (cand.chose_none.astype(str) == "True")).sum()),
                              "n_cand_min": int(cand[cand.selector == s].n_cand.astype(float).min()) if (cand.selector == s).any() else -1,
                              "n_cand_max": int(cand[cand.selector == s].n_cand.astype(float).max()) if (cand.selector == s).any() else -1}
                         for s, _ in COMPARISONS}
    rep_rows, unit_rows, delta_rows, check_rows, recall_rows, nd_rows = [], [], [], [], [], []
    boot_store, jk_store = {}, {}
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
                stage_b_run(args, run, P, sel, rep_rows, unit_rows, delta_rows, check_rows, recall_rows, boot_store, jk_store, prov, nd_rows)
    AR = pd.DataFrame(a_rep)
    write_csv(AR, out / "a_none_pairs_rep.csv")
    A_ = pd.DataFrame()
    if len(AR):
        if AR.duplicated(["run", "target", "student", "K", "rep", "pair"]).any():
            raise SystemExit("(A)：(run, target, student, K, rep, pair) が重複している")
        qcols = ["A_D0", "none_dirloss", "none_dirloss_auc", "none_posloss", "none_model_err", "none_var_ratio"]
        A_ = AR.groupby(["run", "target", "student", "K", "pair", "pair_name"]).agg(
            n_reps=("rep", "nunique"), **{c: (c, "mean") for c in qcols},
            **{f"{c}_rep_med": (c, "median") for c in ("A_D0", "none_dirloss")}, **{f"{c}_rep_min": (c, "min") for c in ("A_D0", "none_dirloss")},
            **{f"{c}_rep_max": (c, "max") for c in ("A_D0", "none_dirloss")}, n_neg_reps=("none_dirloss", lambda x: int((x < 0).sum())),
            none_d=("none_d", "mean"), none_d_auc=("none_d_auc", "mean"), D=("D", "mean"), none_o_abs=("none_o", lambda x: float(x.abs().mean()))).reset_index()
    write_csv(A_, out / "a_none_pairs.csv")
    write_csv(pd.DataFrame(rep_rows), out / "b_pairs_rep.csv")
    write_csv(pd.DataFrame(recall_rows), out / "b_recall_rep.csv")
    write_csv(pd.DataFrame(delta_rows), out / "b_delta_pairs.csv")
    write_csv(pd.DataFrame(check_rows), out / "checks.csv")
    ND = pd.DataFrame(nd_rows)
    write_csv(ND, out / "b_pool_nd.csv")
    if len(ND):
        prov["pool_nd"] = {f"{a}|{b}|{c}": {"n_pool_min": int(g.n_pool.min()), "n_pool_max": int(g.n_pool.max()), "dim": int(g.dim.iloc[0]), "n_over_d_min": float(g.n_over_d.min())}
                           for (a, b, c), g in ND.groupby(["run", "target", "student"])}
    U = pd.DataFrame(unit_rows)
    write_csv(U, out / "b_units_pairs.csv")
    if boot_store:                                                             # 単位ごとのブートストラップと jackknife の複製（R1，S4）
        np.savez_compressed(out / "b_boot_units.npz", **boot_store)
        np.savez_compressed(out / "b_jk_units.npz", **jk_store)
        write_csv(pd.DataFrame([{"key": k, "B": v.shape[0], "has_jk": k in jk_store, "G_jk": jk_store[k].shape[0] if k in jk_store else 0} for k, v in boot_store.items()]),
                  out / "b_boot_units_index.csv")
    return A_, AR, U, pd.DataFrame(recall_rows), boot_store, jk_store


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


def stage_b_run(args, run, P, sel, rep_rows, unit_rows, delta_rows, check_rows, recall_rows, boot_store, jk_store, prov, nd_rows):
    d77 = ROOT / "results" / f"e77_{run}"
    targets = [t for t in sorted(P.target.unique()) if (not args.targets or t in args.targets)]
    D, Rd = delta_pairs(run, d77, targets, prov)
    write_csv(Rd, args.out / f"b_delta_reps_{run}.csv")
    delta_rows.extend(D.assign(run=run).to_dict("records"))
    dmap = {(r.student, r.K, int(r.pair)): float(r.delta_d_pair) for r in D.itertuples()}
    dmap_c = {(r.student, r.K, int(r.pair)): float(r.delta_d_pair_corpus) for r in D.itertuples()}
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
            for st_ in students:                                               # D を学ぶプール（テストの外の発話）の本数と特徴の次元（D17）
                dim = int(load_any(T, st_)[1].shape[1])
                for r in REPS:
                    n_pool = int(len(yall) - len(te[r]))
                    nd_rows.append({"run": run, "target": T, "student": st_, "rep": r, "n_pool": n_pool, "dim": dim, "n_over_d": n_pool / dim})
            # 反復ごとのテストの集合が異なる数（I15・S6・S9）
            sets = {}
            for r in REPS:
                sets.setdefault(te[r].tobytes(), []).append(r)
            same = [v for v in sets.values() if len(v) > 1]
            check_rows.append({"check": "distinct_test_sets", "run": run, "target": T, "n_distinct": len(sets), "identical_reps": ";".join("=".join(map(str, v)) for v in same)})
            prov.setdefault("distinct_test_sets", {})[f"{run}|{T}"] = {"n_distinct": len(sets), "identical_reps": [v for v in same]}
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
        sess, Wj_s = None, None
        if cp == "IEMOCAP":                                                    # セッションを 1 つずつ抜く jackknife（S3）
            sessions = sorted({session_of(s) for s in speakers})
            sess = {s: sessions.index(session_of(s)) for s in speakers}
            Wj_s = np.ones((len(sessions), len(sessions))) - np.eye(len(sessions))
            prov.setdefault("sessions", {})[f"{run}|{cp}"] = sessions
        # G_eff（E79：反復 0〜9 のテストの発話数を人物ごとに足す．群ごと）
        geff = {}
        for T, p in prep.items():
            cnt = pd.Series(np.concatenate([p["spk"][p["te"][r]] for r in REPS])).value_counts().to_numpy().astype(float)
            geff[T] = cnt.sum() ** 2 / (cnt ** 2).sum()
        prov.setdefault("g_eff", {}).update({f"{run}|{T}": float(v) for T, v in geff.items()})
        need_jk = min(geff.values()) < GEFF_MIN
        n_sess = len(set(sess.values())) if sess else 0
        for selector, comp in COMPARISONS:
            for st in students:
                for K in KS:
                    chs = {T: chosen_setting(sel, run, T, st, K, selector) for T in Ts}
                    if any(c == "missing" for c in chs.values()):
                        expected = selector in PARTIAL_SELECTORS and PARTIAL_SELECTORS[selector] != role_of(run, st)
                        check_rows.append({"check": "chosen_missing_expected" if expected else "chosen_missing", "comparison": comp, "run": run, "corpus": cp, "student": st, "K": K})
                        continue
                    if len({c for c in chs.values()}) > 1:                     # IEMOCAP の 2 群で選択が違う（I1）
                        check_rows.append({"check": "chosen_differs_in_corpus", "comparison": comp, "run": run, "corpus": cp, "student": st, "K": K,
                                           "chosen": ";".join(f"{T}:{'none' if c is None else '|'.join(c)}" for T, c in chs.items())})
                        if comp == MAIN_COMP:
                            raise SystemExit(f"{run} {cp} {st} K{K}：IEMOCAP の 2 群で {selector} の選択が違う（主の比較なので止める）")
                    ncands = {T: n_cand_of(sel, run, T, st, K, selector) for T in Ts}
                    per_T = {}
                    for T in Ts:
                        per_T[T] = pair_diffs_target(run, T, prep[T], chs[T], st, K, comp, sid, nls, C, Wj, need_jk, sess, Wj_s, rep_rows, check_rows, recall_rows)
                    all_none = all(c is None for c in chs.values())
                    chosen_str = "none" if all_none else ";".join(f"{T}:{'none' if c is None else '|'.join(c)}" for T, c in chs.items())
                    chosen_short = "none" if all_none else "|".join(next(c for c in chs.values() if c is not None))
                    meta = {"chosen": chosen_short, "chosen_by_group": chosen_str, "n_groups_none": sum(c is None for c in chs.values()),
                            "n_cand_min": min(ncands.values()), "n_cand_by_group": ";".join(f"{T}:{n}" for T, n in ncands.items()),
                            "n_seeds": max((len(per_T[T]["seeds"]) for T in Ts), default=0)}
                    # 単位（コーパス）に集める：主（全群），逆の順序，IEMOCAP 即興だけ（AVT），セッションの jackknife（IEMOCAP）
                    specs = [("main", "dir", Ts, "jk"), ("rev_order", "dir_rev", Ts, "jk")]
                    if run == "fuse-audio-video-text":
                        specs.append(("iemocap_impro_only", "dir", [T for T in Ts if T != "IEMOCAP-script"], "jk"))
                    if sess is not None:
                        specs.append(("session_jk", "dir", Ts, "jk_sess"))
                    for stag, metric, Ts_use, jkkind in specs:
                        if not Ts_use:
                            continue
                        aggregate_unit(run, cp, st, K, comp, metric, stag, jkkind, Ts_use, per_T, A, geff, n_sess, dmap, dmap_c, all_none, meta, unit_rows, boot_store, jk_store)


def pair_diffs_target(run, T, p, chT, st, K, comp, sid, nls, C, Wj, need_jk, sess, Wj_s, rep_rows, check_rows, recall_rows):
    """1 つのターゲット（群）：反復ごとの対ごとの点の値（pairs の表），対数の確率からの照合，ブートストラップと jackknife の複製，感情ごとの再現率の差．
    chT が None（none を選んだ群）なら差を 0 とする（E79 と同じ）．種のある手法は種ごとに計算して平均する（E79 と同じ）．"""
    X = p["X"]
    R = len(REPS)
    res = {k: np.full((R, 6), np.nan) for k in ("pts", "aucs", "des", "dos", "nonelosses", "nonelosses_rev", "revs")}
    res["boot"], res["boot_rev"], res["jk"], res["jk_rev"], res["jk_sess"], res["seeds"] = {}, {}, {}, {}, {}, set()
    for r in REPS:
        xn = X[(X.rep == r) & (X.student == st) & (X.K == K) & (X.method == "none") & (X.param == "-")].sort_values("pair")
        if len(xn) != 6:
            check_rows.append({"check": "none_rows", "comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "n": len(xn)})
            continue
        dn, on, Dn = xn.d.to_numpy(), xn.o.to_numpy(), xn.D.to_numpy()
        res["nonelosses"][r] = 100 * (ba(Dn, 0.0) - ba(dn, 0.0))
        res["nonelosses_rev"][r] = 100 * (ba(Dn, on) - ba(dn, on))              # none の逆の順序の損（逆の順序の表の効果の出うるセルに使う．D15）
        yte = p["y"][p["te"][r]]
        sidx = np.array([sid[s] for s in p["spk"][p["te"][r]]])
        ssidx = np.array([sess[s] for s in p["spk"][p["te"][r]]]) if sess is not None else None
        Ln = logp_row(p["logp"], "e77", r, int(xn.row_id.iloc[0]))
        rec_n = recall_by_emotion(Ln, yte)
        if chT is None:
            for k in ("pts", "aucs", "des", "dos", "revs"):
                res[k][r] = 0.0
            res["boot"][r] = np.zeros((len(C), 6))
            res["boot_rev"][r] = np.zeros((len(C), 6))
            if need_jk:
                res["jk"][r] = np.zeros((nls, 6))
                res["jk_rev"][r] = np.zeros((nls, 6))
            if sess is not None:
                res["jk_sess"][r] = np.zeros((Wj_s.shape[0], 6))
            chosen_lab, rec_o = "none", rec_n.copy()
        else:
            wave, meth, par, me = chT
            xo_all = X[(X.rep == r) & (X.student == st) & (X.K == K) & (X.wave == wave) & (X.method == meth) & (X.pnorm == par) & (X.meff == me)]
            seeds = sorted(xo_all.seed.unique())
            if meth in SEEDED and len(seeds) != 3:
                check_rows.append({"check": "seed_rows", "comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "n_seeds": len(seeds), "chosen": "|".join(chT)})
            if not seeds or any(len(xo_all[xo_all.seed == s_]) != 6 for s_ in seeds):
                check_rows.append({"check": "ours_rows", "comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "n": len(xo_all), "chosen": "|".join(chT)})
                continue
            res["seeds"].update(int(s_) for s_ in seeds)
            acc = {k: [] for k in ("pts", "aucs", "des", "dos", "revs", "boot", "boot_rev", "jk", "jk_rev", "jk_sess", "rec")}
            Fn = pair_feats(Ln, yte, sidx, nls)
            Fn_s = pair_feats(Ln, yte, ssidx, Wj_s.shape[0]) if sess is not None else None
            dn2, on2 = d_o_pairs(Ln, yte)
            for s_ in seeds:
                xo = xo_all[xo_all.seed == s_].sort_values("pair")
                if not np.allclose(xo.D.to_numpy(), Dn, rtol=0, atol=1e-9):
                    raise SystemExit(f"{run} {T} {st} K{K} r{r}：手法の行と none の行で D が違う")
                do_, oo = xo.d.to_numpy(), xo.o.to_numpy()
                acc["pts"].append(dir_diff(do_, dn))
                acc["aucs"].append(100 * (ba(xo.d_auc.to_numpy(), 0.0) - ba(xn.d_auc.to_numpy(), 0.0)))
                acc["des"].append((xo.acc_model.to_numpy() - xo.acc_actual.to_numpy()) - (xn.acc_model.to_numpy() - xn.acc_actual.to_numpy()))   # すでにポイント（I7）
                acc["dos"].append(oo - on)
                acc["revs"].append(dir_diff_rev(Dn, do_, oo, dn, on))
                # 対数の確率から計算し直して照合し，ブートストラップの材料にする
                Lo = logp_row(p["logp"], wave, r, int(xo.row_id.iloc[0]))
                do2, oo2 = d_o_pairs(Lo, yte)
                for name, rec, ref in (("dir", dir_diff(do2, dn2), acc["pts"][-1]), ("dir_rev", dir_diff_rev(Dn, do2, oo2, dn2, on2), acc["revs"][-1])):
                    mx, mism = max_abs_or_nan_mismatch(rec, ref)
                    if mx > 1e-6 or mism:
                        raise SystemExit(f"{run} {T} {st} K{K} r{r} {comp} seed{s_} {name}：対ごとの点の値が pairs の表と合わない（最大 {mx:.2e}，NaN の食い違い {mism}）")
                Fo = pair_feats(Lo, yte, sidx, nls)
                for W, key, FN, FO in ((C, "boot", Fn, Fo), (Wj, "jk", Fn, Fo)):
                    if key == "jk" and not need_jk:
                        continue
                    dBn, oBn = pair_d_o_from_feats(FN, W)
                    dBo, oBo = pair_d_o_from_feats(FO, W)
                    acc[key].append(dir_diff(dBo, dBn))
                    acc[key + "_rev"].append(dir_diff_rev(Dn[None, :], dBo, oBo, dBn, oBn))
                if sess is not None:
                    Fo_s = pair_feats(Lo, yte, ssidx, Wj_s.shape[0])
                    acc["jk_sess"].append(dir_diff(pair_d_from_feats(Fo_s, Wj_s), pair_d_from_feats(Fn_s, Wj_s)))
                acc["rec"].append(recall_by_emotion(Lo, yte))
            for k in ("pts", "aucs", "des", "dos", "revs"):
                res[k][r] = np.mean(np.stack(acc[k]), 0)                       # 種は平均する（E79 と同じ．合奏ではない）
            for k in ("boot", "boot_rev", "jk", "jk_rev", "jk_sess"):
                if acc[k]:
                    res[k][r] = np.mean(np.stack(acc[k]), 0)
            chosen_lab, rec_o = "|".join(chT), np.mean(np.stack(acc["rec"]), 0)
        for pidx in range(6):
            rep_rows.append({"comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "pair": pidx, "pair_name": PAIR_NAME[pidx],
                             "diff_dir": res["pts"][r, pidx], "diff_dir_rev": res["revs"][r, pidx], "diff_dir_auc": res["aucs"][r, pidx],
                             "diff_model_err": res["des"][r, pidx], "diff_o": res["dos"][r, pidx], "none_dirloss": res["nonelosses"][r, pidx], "chosen": chosen_lab,
                             "n_seeds": len(res["seeds"])})
        for e in range(4):                                                     # 感情ごとの再現率の差（UAR への橋．記述．D2）
            recall_rows.append({"comparison": comp, "run": run, "target": T, "student": st, "K": K, "rep": r, "emotion": EMO[e],
                                "recall_none": rec_n[e], "recall_method": rec_o[e], "diff_recall": rec_o[e] - rec_n[e], "chosen": chosen_lab})
    return res


def aggregate_unit(run, cp, st, K, comp, metric, stag, jkkind, Ts_use, per_T, A, geff, n_sess, dmap, dmap_c, all_none, meta, unit_rows, boot_store, jk_store):
    """単位（コーパス）に集める：反復の平均をターゲットで平均（E79 と同じ）．区間は反復の重み A で反復を混ぜ，ターゲットで平均した複製の百分位．
    G_eff < 20 の群を含む単位は jackknife の t 区間との和集合（jkkind = jk_sess はセッションを抜く jackknife．自由度 セッション数 − 1）．"""
    pk, bk = ("pts", "boot") if metric == "dir" else ("revs", "boot_rev")
    jk = "jk" if metric == "dir" else "jk_rev"
    x = np.nanmean(np.stack([np.nanmean(per_T[T][pk], 0) for T in Ts_use]), 0)   # (6,)
    xa = np.nanmean(np.stack([np.nanmean(per_T[T]["aucs"], 0) for T in Ts_use]), 0)
    de = np.nanmean(np.stack([np.nanmean(per_T[T]["des"], 0) for T in Ts_use]), 0)
    dO = np.nanmean(np.stack([np.nanmean(per_T[T]["dos"], 0) for T in Ts_use]), 0)
    nl = np.nanmean(np.stack([np.nanmean(per_T[T]["nonelosses"], 0) for T in Ts_use]), 0)
    nlr = np.nanmean(np.stack([np.nanmean(per_T[T]["nonelosses_rev"], 0) for T in Ts_use]), 0)
    gmin = float(min(geff[T] for T in Ts_use))
    geff_jk = float(n_sess) if jkkind == "jk_sess" else gmin
    lo = hi = np.full(6, np.nan)
    lo_j = hi_j = np.full(6, np.nan)
    lo_s = hi_s = np.full(6, np.nan)
    nvalid = 0
    rule = "pct"
    b = th = None

    def jk_mean(kind):
        return np.nanmean(np.stack([np.nanmean(np.stack([per_T[T][kind][r] for r in REPS], 1), 1) for T in Ts_use]), 0)   # (G, 6)
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
        if gmin < GEFF_MIN:                                                    # 主の規則：話者を 1 人ずつ抜く jackknife との和集合
            if all(len(per_T[T][jk]) == len(REPS) for T in Ts_use):
                th = jk_mean(jk)
                lo_j, hi_j = np.array([jk_interval(x[pp], th[:, pp], gmin) for pp in range(6)]).T
                lo, hi = np.fmin(lo, lo_j), np.fmax(hi, hi_j)
                rule = "union_pct_jk"
            else:
                rule = "pct_jk_missing"
                lo = hi = np.full(6, np.nan)
        if jkkind == "jk_sess" and rule != "pct_jk_missing":                   # 感度：主の区間とセッションを 1 つずつ抜く jackknife（自由度 セッション数 − 1）の和集合（S3）
            if all(len(per_T[T]["jk_sess"]) == len(REPS) for T in Ts_use):
                th = jk_mean("jk_sess")
                lo_s, hi_s = np.array([jk_interval(x[pp], th[:, pp], geff_jk) for pp in range(6)]).T
                lo, hi = np.fmin(lo, lo_s), np.fmax(hi, hi_s)
                rule = "union_main_session_jk"
            else:
                rule = "pct_jk_missing"
                lo = hi = np.full(6, np.nan)
    else:
        rule = "no_replicates"
    key = f"{comp}|{run}|{st}|{K}|{cp}|{stag}"
    if b is not None:
        boot_store[key] = b.astype(np.float32)
    if th is not None:
        jk_store[key] = th.astype(np.float32)
    scales = DELTA_SCALES if stag == "main" else ((1.0, stag),)
    for pidx in range(6):
        dl = dmap.get((st, K, pidx), np.nan)
        dlc = dmap_c.get((st, K, pidx), np.nan)
        for scale, sname in scales:
            dls = dl * scale
            md = np.nan
            if metric == "dir" and np.isfinite(x[pidx]) and np.isfinite(xa[pidx]) and np.isfinite(dls):   # 模型に依る印は向きの損（主の順序）だけ（E79 と同じ）
                md = bool(abs(x[pidx] - xa[pidx]) > MODEL_UNIT_FRAC * dls or abs(de[pidx]) > MODEL_UNIT_FRAC * dls or zone(x[pidx], dls) != zone(xa[pidx], dls))
            jd = "same_by_definition" if all_none else judge_unit(x[pidx], lo[pidx], hi[pidx], dls)
            unit_rows.append({"comparison": comp, "run": run, "role": role_of(run, st), "student": st, "K": K, "corpus": cp, "n_groups": len(Ts_use),
                              "groups": ";".join(Ts_use), "metric": metric, "pair": pidx, "pair_name": PAIR_NAME[pidx], "point": x[pidx],
                              "lo": lo[pidx], "hi": hi[pidx], "lo_jk": lo_j[pidx], "hi_jk": hi_j[pidx], "lo_session_jk": lo_s[pidx], "hi_session_jk": hi_s[pidx],
                              "interval_rule": rule, "g_eff_min": gmin, "g_eff_jk": geff_jk,
                              "n_boot_valid": nvalid, "delta_d_pair": dls, "delta_d_pair_corpus": dlc * scale, "sensitivity": sname, "boot_key": key,
                              "judgment": jd, "model_dependent": md,
                              "point_auc": xa[pidx] if metric == "dir" else np.nan, "model_err_diff": de[pidx] if metric == "dir" else np.nan,
                              "diff_o": dO[pidx], "none_dirloss": nl[pidx], "none_dirloss_rev": nlr[pidx], **meta})


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

def d_from_probs(Pm, yy, k, j):
    """候補の確率から 2 感情の d（E13 と同じ式．z = log p_k − log p_j）．"""
    z = np.log(np.clip(Pm[:, k], 1e-12, None)) - np.log(np.clip(Pm[:, j], 1e-12, None))
    zk, zj = z[yy == k], z[yy == j]
    if len(zk) < 2 or len(zj) < 2:
        return np.nan
    sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
    return (zk.mean() - zj.mean()) / sd if sd > 0 else np.nan


def stage_c(args, out, prov):
    rows, rec_rows, phi_rows, anchor_rows = [], [], [], []
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
                                if mm.sum() >= MIN_PAIR_N and a.std() > 0 and b.std() > 0:
                                    phi = float(np.corrcoef(a, b)[0, 1])
                                phi_rows.append({**tag, "mod1": m1, "mod2": m2, "emotion": EMO[e], "n": int(mm.sum()), "phi": phi,
                                                 "both_wrong": float(((a == 0) & (b == 0)).mean()) if mm.any() else np.nan,
                                                 "indep_both_wrong": float((1 - a.mean()) * (1 - b.mean())) if mm.any() else np.nan})
                    # 対ごとの補い合い J_kl(f, g)．f は単モダリティと融合（none__<実行>__<K>）．g は other_m（f と別のモダリティ），pF，poe0，ref，ref0
                    for st in mods + [run]:
                        S = np.asarray(z[f"none__{st}__{K}"])[keep]
                        d_f = {}
                        for p, (k, j) in enumerate(PAIRS):
                            m = (yk == k) | (yk == j)
                            d_f[p] = d_from_probs(S[m], yk[m], k, j) if m.sum() >= MIN_PAIR_N else np.nan
                            anchor_rows.append({**tag, "student": st, "role": role_of(run, st), "pair": p, "pair_name": PAIR_NAME[p], "n_pair": int(m.sum()),
                                                "d_f": d_f[p], "a_d0_f": 100 * float(ba(d_f[p], 0.0)) if np.isfinite(d_f[p]) else np.nan})
                        for name, G in preds.items():
                            if name == f"other_{st}":
                                continue
                            for p, (k, j) in enumerate(PAIRS):
                                m = (yk == k) | (yk == j)
                                if m.sum() < MIN_PAIR_N:
                                    continue
                                yy = yk[m]
                                sdec = np.where(S[m, k] >= S[m, j], k, j)
                                gdec = np.where(G[m, k] >= G[m, j], k, j)
                                sr, gr = sdec == yy, gdec == yy
                                d_g = d_from_probs(G[m], yy, k, j)
                                rows.append({**tag, "student": st, "role": role_of(run, st), "g": name, "pair": p, "pair_name": PAIR_NAME[p], "n_pair": int(m.sum()),
                                             "acc_student": float(sr.mean()), "acc_g": float(gr.mean()),
                                             "J": 100 * float((gr & ~sr).mean() - (~gr & sr).mean()),
                                             "g_right_s_wrong": float((gr & ~sr).mean()), "g_wrong_s_right": float((~gr & sr).mean()),
                                             "J_dir": 100 * float(ba(d_g, 0.0) - ba(d_f[p], 0.0)) if np.isfinite(d_g) and np.isfinite(d_f[p]) else np.nan})
    J = pd.DataFrame(rows)
    write_csv(J, out / "c_complementarity_units.csv")
    write_csv(pd.DataFrame(rec_rows), out / "c_recall_units.csv")
    write_csv(pd.DataFrame(phi_rows), out / "c_phi_units.csv")
    write_csv(pd.DataFrame(anchor_rows), out / "c_anchor_pairs.csv")
    return J, pd.DataFrame(rec_rows), pd.DataFrame(phi_rows), pd.DataFrame(anchor_rows)


# ---------------------------------------------------------------- まとめと予言の判定

def judgment_table(m, path):
    keys = ["comparison", "run", "role", "student", "pair_name"]
    if m.empty:
        cnt = pd.DataFrame(columns=keys + JUDG_COLS)
    else:
        cnt = m.groupby(keys + ["judgment"]).size().unstack(fill_value=0)
        cnt = cnt.reindex(columns=JUDG_COLS, fill_value=0).reset_index()       # 列を固定する（I1）
    write_csv(cnt, path)
    return cnt


def exact_perm_p(m, cells, elig, sel_pairs, perm_maps, conditional):
    """厳密な p（S10・D14）：コーパスごとに置換（perm_maps：感情の名前の置換 24 通りか，感度の列の対の名前の置換 720 通り）を 1 つ引いて塊の全行の対に当てる．
    cells：数える行（減らした [かつ効果の出うるセル]）．conditional なら，各コーパスで cells のすべてを同じ単位の効果の出うるセルに移す置換だけから一様に引く．
    コーパスの間は独立なので数の分布を畳み込む．戻り：観測した選んだ対の数，p ＝ P(数 ≥ 観測)，分布，置換を引けたコーパスの数．"""
    sel = np.zeros(6, bool)
    sel[list(sel_pairs)] = True
    ukey = m.run.astype(str) + "|" + m.student.astype(str) + "|" + m.K.astype(str) + "|" + m.corpus.astype(str)
    pair = m.pair.to_numpy().astype(int)
    pmf, obs, n_random = np.array([1.0]), 0, 0
    for cp in sorted(set(m.corpus)):
        mk = (m.corpus == cp).to_numpy() & cells
        if not mk.any():
            continue
        cu, cpair = ukey[mk].to_numpy(), pair[mk]
        ek = elig & (m.corpus == cp).to_numpy()
        elig_set = {(u, int(q)) for u, q in zip(ukey[ek], pair[ek])}
        counts = []
        for perm in perm_maps:
            newp = perm[cpair]
            if conditional and not all((u, int(q)) in elig_set for u, q in zip(cu, newp)):
                continue
            counts.append(int(sel[newp].sum()))
        pmf_cp = np.bincount(counts, minlength=7) / len(counts)
        if len(set(counts)) > 1:
            n_random += 1
        pmf = np.convolve(pmf, pmf_cp)
        obs += int(sel[cpair].sum())
    p = float(pmf[obs:].sum()) if cells.any() else np.nan
    return obs, p, pmf, n_random


def null_block_counts(m, rep_lookup):
    """数の帰無（S10・S11）：保存した複製 b (B, 6) を点で引いて中心を 0 に寄せ，抽出ごとに「x* + (lo − x) > 0 かつ x* ≥ δ」を帰無の「減らした」とし，
    （コーパス，対）の塊に畳んだ数の分布を返す．同じ (実行, コーパス) の中では抽出が f・K・対で共通なので相関が保たれる（世界の間は独立）．"""
    hits, B = {}, None
    for r_ in m.itertuples():
        b = rep_lookup(r_)
        if b is None or not (np.isfinite(r_.point) and np.isfinite(r_.lo) and np.isfinite(r_.delta_d_pair)):
            continue
        xs = b[:, int(r_.pair)].astype(np.float64) - r_.point
        posn = (xs + (r_.lo - r_.point) > 0) & (xs >= r_.delta_d_pair)
        B = len(posn)
        k = (r_.corpus, int(r_.pair))
        hits[k] = hits[k] | posn if k in hits else posn
    if B is None:
        return None
    return np.sum(np.stack(list(hits.values())), 0) if hits else np.zeros(B)


def elig_and_keys(m):
    """効果の出うるセル（D14）と単位の鍵．逆の順序は none の逆の順序の損で決める（D15）．"""
    metric = str(m.metric.iloc[0]) if len(m) and "metric" in m.columns else "dir"
    loss = (m.none_dirloss_rev if (metric == "dir_rev" and "none_dirloss_rev" in m.columns) else m.none_dirloss).to_numpy(dtype=float)
    dl = m.delta_d_pair.to_numpy(dtype=float)
    observable = (m.judgment != "same_by_definition").to_numpy()               # 観測で数えうる行（D19．両方が none の単位は入らない）
    elig = (loss >= dl) & (m.chosen != "none").to_numpy() & (m.judgment != "missing").to_numpy() & observable
    ukey = (m.run.astype(str) + "|" + m.student.astype(str) + "|" + m.K.astype(str) + "|" + m.corpus.astype(str)).to_numpy()
    full_units = {u for u in set(ukey) if int((ukey == u).sum()) == 6 and bool(elig[ukey == u].all())}   # 6 対すべてが効果の出うるセルの単位
    return loss, dl, observable, elig, ukey, full_units


def by_corpus_str(units):
    c = pd.Series([u.split("|")[-1] for u in units]).value_counts().sort_index() if units else pd.Series(dtype=int)
    return ";".join(f"{k}:{n}" for k, n in c.items())


def p3a_eval(m, rep_lookup, sel_pairs, base, label, judged, old_chance, m_ref=None):
    """1 つの族・1 つの版の P3a の判定（3 節）．m は（単位，対）の行（judgment・point・lo・delta_d_pair・none_dirloss(_rev)・chosen・corpus・pair を持つ）．
    m_ref は印を含む行（同じ族）：(iii) の対象の単位と到達しうる最小の p と judgeable はこれで決める（印は手法の結果から作る量なので．gate-stats の再検査 2）．
    印を除く版（m が m_ref の部分集合）では，対象のうち印の付いた行を含まない単位で p を計算し，印で落ちた単位の数を出す．"""
    ukey_cols = ["run", "student", "K", "corpus"]
    if m_ref is None:
        m_ref = m
    loss, dl, observable, elig, ukey, full_units_used = elig_and_keys(m)
    _, _, _, _, ukey_ref, full_units_ref = elig_and_keys(m_ref)
    full_units_used = full_units_used & full_units_ref                         # 対象は印を含む行で決めた単位のうち，印を除いても 6 行そろう単位
    pos = (m.judgment == "pos").to_numpy()
    pos_elig = pos & elig
    n_pos, n_pos_elig = int(pos.sum()), int(pos_elig.sum())
    pairs_arr = m.pair.to_numpy().astype(int)
    sel = np.zeros(6, bool)
    sel[list(sel_pairs)] = True

    def perm_on(mm, keys, units, cells_fn):
        inu = np.isin(keys, list(units)) if units else np.zeros(len(mm), bool)
        if not inu.any():
            return 0, np.nan, 0, inu
        mf = mm[inu]
        obs_, p_, _, nr_ = exact_perm_p(mf, cells_fn(inu), np.ones(int(inu.sum()), bool), sel_pairs, CLASS_PERM_MAPS, False)
        return obs_, p_, nr_, inu
    # 到達しうる最小の p：印を含む行の対象の単位で（配置だけで決まる）
    pairs_ref = m_ref.pair.to_numpy().astype(int)
    _, p_min_ref, _, _ = perm_on(m_ref, ukey_ref, full_units_ref, lambda inu: sel[pairs_ref[inu]])
    judgeable_ref = bool(np.isfinite(p_min_ref) and p_min_ref < P_ALPHA)
    # 使う行（印を除く版なら印の無い単位）での p と，印の後の到達しうる最小の p
    obs_f, p_full, n_rand_f, in_full = perm_on(m, ukey, full_units_used, lambda inu: pos[inu])
    _, p_min_used, _, _ = perm_on(m, ukey, full_units_used, lambda inu: sel[pairs_arr[inu]])
    judgeable_used = bool(np.isfinite(p_min_used) and p_min_used < P_ALPHA)
    n_pos_full = int(pos[in_full].sum())
    n_dropped = len(full_units_ref) - len(full_units_used)
    # (i) 数の帰無：観測で数えうる行の複製から
    mo = m[observable]
    blk = mo.assign(is_pos=pos[observable]).groupby(["corpus", "pair"]).agg(pos_any=("is_pos", "any"), n=("is_pos", "size")).reset_index()
    blk["pos_any"] = blk.pos_any.astype(bool)
    n_blocks = len(blk)
    n_pos_blocks = int(blk.pos_any.sum()) if n_blocks else 0
    chance_anyof = float((1 - (1 - CHANCE) ** blk.n).sum()) if n_blocks else 0.0   # 塊の中の判定を独立とみなしたときの平均（上限ではない）
    null = null_block_counts(mo, rep_lookup)
    null_q95 = float(np.percentile(null, 95)) if null is not None else np.nan
    null_mean = float(null.mean()) if null is not None else np.nan
    thr = float(np.nanmax([null_q95, chance_anyof]))
    c_count = bool(n_pos_blocks > thr)
    # (ii) 広がり：効果の出うるセルの怒りの対の「減らした」が出たコーパスの数
    sel_pos_elig = pos_elig & sel[pairs_arr]
    n_corp_sel_elig = int(m.corpus[sel_pos_elig].nunique())
    c_corp = bool(n_corp_sel_elig >= 2)
    c_p = bool(judgeable_ref and judgeable_used and np.isfinite(p_full) and p_full < P_ALPHA)
    # 感度と記述：条件つきの置換（前の版の主），無条件の置換，対の名前の置換，固定と条件つきの土台
    obs_c, p_cond, _, n_rand = exact_perm_p(m, pos_elig, elig, sel_pairs, CLASS_PERM_MAPS, True)
    obs_u, p_class, _, _ = exact_perm_p(m, pos, elig, sel_pairs, CLASS_PERM_MAPS, False)
    _, p_pairperm, _, _ = exact_perm_p(m, pos, elig, sel_pairs, PAIR_PERM_MAPS, False)
    share_full = obs_f / n_pos_full if n_pos_full else np.nan
    share_elig = obs_c / n_pos_elig if n_pos_elig else np.nan
    share_all = obs_u / n_pos if n_pos else np.nan
    base_cond = float(sel[pairs_arr[elig]].mean()) if elig.any() else np.nan
    near = observable & (np.abs(loss - dl) < NEAR_BOUNDARY)                     # 境界の近く（仮定の表の 5 行目）
    n_tests = len(m)
    pos_rows = m[pos]
    holds_old = bool(n_pos > old_chance and np.isfinite(share_all) and share_all > base)   # cebd64b の規則（数の基準は族ごとに固定）
    verdict = None if not judged else ("not_judgeable" if not judgeable_ref else ("not_judgeable_by_flag" if not judgeable_used else ("holds" if (c_count and c_corp and c_p) else "fails")))
    return {"n_units": len(m[ukey_cols].drop_duplicates()), "n_tests": n_tests, "n_tests_observable": int(observable.sum()), "n_tests_nonone": int((m.chosen != "none").sum()),
            "chance_pos": CHANCE * n_tests, "cebd64b_chance": old_chance,
            "n_pos": n_pos, "n_neg": int((m.judgment == "neg").sum()), "n_same": int((m.judgment == "same").sum()), "n_undet": int((m.judgment == "undet").sum()),
            "n_missing": int((m.judgment == "missing").sum()), "n_same_by_definition": int((m.judgment == "same_by_definition").sum()),
            "n_blocks": n_blocks, "n_pos_blocks": n_pos_blocks, "chance_pos_blocks_anyof": chance_anyof, "null_count_q95": null_q95, "null_count_mean": null_mean,
            "count_threshold": thr, "cond_count": c_count,
            "n_eligible": int(elig.sum()), "n_pos_elig": n_pos_elig, "n_pos_not_elig": n_pos - n_pos_elig, "n_pos_not_elig_selected": int((pos & ~elig & sel[pairs_arr]).sum()),
            "n_near_boundary": int(near.sum()), "n_near_boundary_selected": int((near & sel[pairs_arr]).sum()),
            "n_pos_corpora": int(m.corpus[pos].nunique()), "n_pos_corpora_selected": int(m.corpus[pos & sel[pairs_arr]].nunique()),
            "n_pos_corpora_selected_elig": n_corp_sel_elig, "cond_two_corpora": c_corp,
            "pos_corpora": ";".join(f"{c}:{n}" for c, n in pos_rows.groupby("corpus").size().items()),
            "selected_pairs": label,
            "n_units_full_elig": len(full_units_ref), "full_elig_units_by_corpus": by_corpus_str(full_units_ref),
            "n_units_full_elig_used": len(full_units_used), "full_elig_units_by_corpus_used": by_corpus_str(full_units_used), "n_full_elig_dropped_by_flag": n_dropped,
            "n_pos_full_elig": n_pos_full, "n_pos_elig_not_full": int((pos_elig & ~in_full).sum()), "n_pos_elig_not_full_selected": int((pos_elig & ~in_full & sel[pairs_arr]).sum()),
            "n_selected_among_pos_full_elig": obs_f, "share_selected_among_pos_full_elig": share_full, "p_class_perm_full_elig": p_full,
            "n_corpora_with_random_perm_full_elig": n_rand_f, "p_min_attainable": p_min_ref, "judgeable": judgeable_ref,
            "p_min_attainable_after_flag": p_min_used, "judgeable_after_flag": judgeable_used, "cond_p": c_p,
            "n_selected_among_pos_elig": obs_c, "share_selected_among_pos_elig": share_elig, "p_class_perm_cond": p_cond, "n_corpora_with_random_perm": n_rand,
            "share_selected_among_pos": share_all, "base_share": base, "base_share_cond": base_cond, "p_class_perm_uncond": p_class, "p_pair_perm_uncond": p_pairperm,
            "share_p3a_side_among_pos": float(pos_rows.pair.isin(P3A_SIDE_PAIRS).mean()) if n_pos else np.nan,
            "share_p3b_side_among_pos": float(pos_rows.pair.isin(P3B_SIDE_PAIRS).mean()) if n_pos else np.nan,
            "n_pos_happy_anger": int((pos_rows.pair == HAPPY_ANGER).sum()),
            "share_arousal_pairs_among_pos": float(pos_rows.pair.isin(AROUSAL_PAIRS).mean()) if n_pos else np.nan, "base_arousal": 2 / 6,
            "P3a_holds": (verdict == "holds") if judged else None, "P3a_verdict": verdict, "P3a_holds_cebd64b": holds_old if judged else None,
            "pos_pairs": ";".join(f"{PAIR_NAME[p]}:{n}" for p, n in pos_rows.pair.value_counts().sort_index().items())}


FAMILIES = (("audio", ["audio_wavlm"], ANGER_PAIRS, 0.5, "怒りを含む 3 対", True),
            ("video", ["video"], (NEU_ANG,), 1 / 6, "平静–怒り", False),
            ("text", ["text_roberta"], ANGER_PAIRS, 0.5, "怒りを含む 3 対（記述）", True),
            ("fused", list(RUNS), ANGER_PAIRS, 0.5, "怒りを含む 3 対（記述）", True))


def p3a_table(rows, rep_lookup, tag, judged_comp=MAIN_COMP):
    """族 × 印の有無の P3a の表．rows は 1 つの比較・1 つの感度の（単位，対）の行．"""
    out = []
    for fam, stus, sel_pairs, base, label, main_with_md in FAMILIES:
        for use_md in (True, False):
            m_all = rows[rows.student.isin(stus)]                              # 印を含む行（(iii) の対象と最小の p はこれで決める）
            m = m_all if use_md else m_all[m_all.model_dependent != True]  # noqa: E712
            judged = fam in ("audio", "video") and (rows.comparison.iloc[0] == judged_comp if len(rows) else False)
            out.append({"table": tag, "comparison": rows.comparison.iloc[0] if len(rows) else "", "family": fam, "with_model_dependent": use_md,
                        "is_main_version": use_md == main_with_md, **p3a_eval(m, rep_lookup, sel_pairs, base, label, judged, CEBD64B_CHANCE[fam], m_ref=m_all)})
    return pd.DataFrame(out)


def direct_comparisons(U, boot_store, jk_store):
    """手法どうしの直接の差（a − b．none は打ち消し合う）．同じ (実行, コーパス) の抽出が共通なので複製の差がそのまま複製になる（S14）．
    両方が none を選んだ単位だけ「定義から同じ水準」．片方だけ none の単位は差が 0 でないので主と同じ規則で判定する（D19）．"""
    main = U[U.sensitivity == "main"]
    rows, store = [], {}
    for comp_a, comp_b, name in DIRECT:
        A_ = main[main.comparison == comp_a].set_index(["run", "student", "K", "corpus", "pair"])
        B_ = main[main.comparison == comp_b].set_index(["run", "student", "K", "corpus", "pair"])
        for key in A_.index.intersection(B_.index):
            ra, rb = A_.loc[key], B_.loc[key]
            run, st, K, cp, pidx = key
            x = ra.point - rb.point
            lo = hi = np.nan
            rule = "no_replicates"
            gmin = float(min(ra.g_eff_min, rb.g_eff_min))
            ba_, bb_ = boot_store.get(ra.boot_key), boot_store.get(rb.boot_key)
            skey = f"{name}|{run}|{st}|{K}|{cp}|main"
            if ba_ is not None and bb_ is not None:
                b = ba_.astype(np.float64) - bb_.astype(np.float64)
                store[skey] = b.astype(np.float32)
                lo, hi = np.nanpercentile(b[:, pidx], 2.5), np.nanpercentile(b[:, pidx], 97.5)
                rule = "pct"
                if gmin < GEFF_MIN:
                    ta, tb = jk_store.get(ra.boot_key), jk_store.get(rb.boot_key)
                    if ta is not None and tb is not None:
                        th = ta.astype(np.float64) - tb.astype(np.float64)
                        lj, hj = jk_interval(x, th[:, pidx], gmin)
                        lo, hi = np.fmin(lo, lj), np.fmax(hi, hj)
                        rule = "union_pct_jk"
                    else:
                        rule, lo, hi = "pct_jk_missing", np.nan, np.nan
            a_none, b_none = ra.chosen == "none", rb.chosen == "none"
            both_none, both = a_none and b_none, (not a_none) and (not b_none)
            none_side = "both" if both_none else ("a" if a_none else ("b" if b_none else ""))
            rows.append({"comparison": name, "run": run, "role": ra.role, "student": st, "K": K, "corpus": cp, "n_groups": ra.n_groups, "metric": "dir", "pair": pidx,
                         "pair_name": PAIR_NAME[pidx], "point": x, "lo": lo, "hi": hi, "interval_rule": rule, "g_eff_min": gmin, "delta_d_pair": ra.delta_d_pair,
                         "sensitivity": "main", "boot_key": skey, "judgment": "same_by_definition" if both_none else judge_unit(x, lo, hi, ra.delta_d_pair),
                         "model_dependent": np.nan, "none_dirloss": ra.none_dirloss, "none_dirloss_rev": ra.none_dirloss_rev,
                         "chosen": "none" if both_none else "both", "chosen_a": ra.chosen, "chosen_b": rb.chosen, "none_side": none_side,
                         "both_chosen": both, "judgment_a": ra.judgment, "judgment_b": rb.judgment, "same_pos": bool(ra.judgment == "pos" and rb.judgment == "pos"),
                         "n_cand_min_a": ra.n_cand_min, "n_cand_min_b": rb.n_cand_min})
    return pd.DataFrame(rows), store


def level_fit_table(AR):
    """仮定の表の 3 行目：（f，K，対）ごとの |acc_model − acc_actual| の中央値 ÷ none の向きの損の中央値（1/3 未満なら水準の当てはまりが良い）．"""
    lv = AR.groupby(["run", "student", "K", "pair_name"]).agg(med_abs_err=("none_model_err", lambda x: float(x.abs().median())), med_dirloss=("none_dirloss", "median")).reset_index()
    lv["level_fit_ratio"] = lv.med_abs_err / lv.med_dirloss.where(lv.med_dirloss > 0)
    lv["level_fit_ok"] = lv.level_fit_ratio < 1 / 3
    return lv


def assumptions_table(A_, AR, U, anchors, out):
    """E83 の仮定の表（define の D8）．基準は 3 節．"""
    rows = []
    main = U[(U.comparison == MAIN_COMP) & (U.sensitivity == "main")] if len(U) else U
    if len(main):
        g_flag = (main.point - main.point_auc).abs() > MODEL_UNIT_FRAC * main.delta_d_pair
        e_flag = main.model_err_diff.abs() > MODEL_UNIT_FRAC * main.delta_d_pair
        rows.append({"assumption": 1, "name": "z が各感情の中で正規（形）", "measure": "|g_対| > δ/2 の（単位，対）の数（区分の違いは含まない）", "value": int(g_flag.sum()), "n": len(main), "criterion": "印（E79 の 1 行目）"})
        rows.append({"assumption": "1+2", "name": "模型に依る印（1・2 行目と区分の違いを合わせた数）", "measure": "model_dependent が真の（単位，対）の数",
                     "value": int((main.model_dependent == True).sum()), "n": len(main), "criterion": "印"})  # noqa: E712
        rows.append({"assumption": 2, "name": "模型の誤差が none と手法で打ち消し合う", "measure": "|Δe_対| > δ/2 の（単位，対）の数", "value": int(e_flag.sum()), "n": len(main), "criterion": "印（E79 の 2 行目）"})
        hw = (main.hi - main.lo) / 2
        ratio = main.delta_d_pair / hw.where(hw > 0)
        per_fam = ratio.groupby(main.student).median().to_dict()
        rows.append({"assumption": 6, "name": "仮のターゲットの揺れを評価のターゲットの基準に移せる", "measure": "δ_d,対 ÷ 区間の半分の幅の中央値（f ごと）",
                     "value": ";".join(f"{k}:{v:.2f}" for k, v in per_fam.items()), "n": int(ratio.notna().sum()), "criterion": "記述（比が 1 から大きく外れる族を書く）"})
        near = (main.none_dirloss - main.delta_d_pair).abs() < NEAR_BOUNDARY
        rows.append({"assumption": 5, "name": "D が最良の向きの近く（土台の境界）", "measure": "|none の向きの損 − δ_d,対| < 2 ポイントの（単位，対）の数（f ごと）",
                     "value": ";".join(f"{k}:{int(v)}" for k, v in near.groupby(main.student).sum().items()), "n": len(main),
                     "criterion": "記述（族ごとの数は b_p3_checks.csv の n_near_boundary・n_near_boundary_selected）"})
    if len(AR):
        neg = AR.groupby(["run", "student", "K", "pair_name"]).none_dirloss.apply(lambda x: int((x < 0).sum())).reset_index(name="n_neg")
        write_csv(neg, out / "assumption5_negative_counts.csv")
        rows.append({"assumption": 5, "name": "D が最良の向きの近く（負の対）", "measure": "none の向きの損が負の（ターゲット，反復，対）の数", "value": int((AR.none_dirloss < 0).sum()),
                     "n": len(AR), "criterion": "記述（f × K × 対の内訳は assumption5_negative_counts.csv）"})
        lv = level_fit_table(AR)
        write_csv(lv, out / "assumption3_level_fit.csv")
        rows.append({"assumption": 3, "name": "水準の当てはまり（対ごと）", "measure": "|acc_model − acc_actual| の中央値 ÷ none の向きの損の中央値 ≥ 1/3 の（f，K，対）の数",
                     "value": int((~lv.level_fit_ok.fillna(False)).sum()), "n": len(lv), "criterion": "1/3 未満（E75 の決まり）．満たさない (f, 対) は (A) の表に注を付ける"})
        vr = AR.dropna(subset=["none_var_ratio"])
        if len(vr):
            share = ((vr.none_var_ratio >= 0.5) & (vr.none_var_ratio <= 2)).groupby([vr.student, vr.pair_name]).mean().reset_index(name="share_in_0.5_2")
            write_csv(share, out / "assumption4_var_ratio.csv")
            rows.append({"assumption": 4, "name": "等分散（記述）", "measure": "v_k/v_l が [0.5, 2] に入る割合（全体）", "value": float(((vr.none_var_ratio >= 0.5) & (vr.none_var_ratio <= 2)).mean()),
                         "n": len(vr), "criterion": "記述（E79 の 4 行目と同じ理由）"})
    rows.append({"assumption": 7, "name": "6 対の名前は交換できない（4 感情の名前は交換できる）", "measure": "p_class_perm_cond（主）と p_pair_perm_uncond（感度）", "value": "b_p3_checks.csv",
                 "n": "", "criterion": "感情の名前の置換を主にする"})
    # 8：J の f が (B) の none の対ごとの弱さを写す（順位の一致）
    if len(anchors) and len(A_):
        wmap = {"fuse-audio-video-text": "avt", "fuse-audio-video": "av"}
        cA = anchors.groupby(["world", "student", "K", "pseudo_corpus", "pair"]).a_d0_f.median().reset_index()
        bB = A_[A_.run.isin(RUNS)].assign(world=lambda d: d.run.map(wmap), corpus=lambda d: d.target.map(corp), a_d0_none=lambda d: d.A_D0 - d.none_dirloss)
        bB = bB.groupby(["world", "student", "K", "corpus", "pair"]).a_d0_none.mean().reset_index().rename(columns={"corpus": "pseudo_corpus"})
        mm = cA.merge(bB, on=["world", "student", "K", "pseudo_corpus", "pair"])
        rk = []
        for key, g in mm.groupby(["world", "student", "K", "pseudo_corpus"]):
            if len(g) == 6 and g.a_d0_f.notna().all():
                rho = stats.spearmanr(g.a_d0_f, g.a_d0_none).correlation
                rk.append({"world": key[0], "student": key[1], "K": key[2], "corpus": key[3], "spearman": float(rho)})
        rk = pd.DataFrame(rk)
        write_csv(rk, out / "assumption8_rank.csv")
        if len(rk):
            rows.append({"assumption": 8, "name": "J の f が (B) の none の対ごとの弱さを写す", "measure": "(C) の f の 6 対の A(d,0) と (B) の none の A(d_none,0) の順位相関の中央値（f ごと）",
                         "value": ";".join(f"{k}:{v:.2f}" for k, v in rk.groupby("student").spearman.median().items()), "n": len(rk), "criterion": "記述（一致が低い族では P3c を読まない）"})
    T_ = pd.DataFrame(rows)
    write_csv(T_, out / "assumptions.csv")
    return T_


def summarize(out, A_, AR, U, RC, J, R, Ph, anchors, boot_store, jk_store):
    lines = ["# E83 のまとめ（台本が自動で書く．読み方は docs/exploratory-analyses.md の E83 3 節）", ""]
    # (A)
    if len(A_):
        qs = ["A_D0", "none_dirloss", "none_dirloss_auc", "none_model_err", "none_posloss"]
        agg = {}
        for c in qs:
            agg[f"{c}_med"] = (c, "median")
            agg[f"{c}_min"] = (c, "min")
            agg[f"{c}_max"] = (c, "max")
            agg[f"{c}_q1"] = (c, lambda x: x.quantile(0.25))
            agg[f"{c}_q3"] = (c, lambda x: x.quantile(0.75))
        a_all = A_.groupby(["run", "student", "K", "pair_name"]).agg(n_targets=("target", "nunique"), n_neg_targets=("none_dirloss", lambda x: int((x < 0).sum())),
                                                                     n_neg_reps=("n_neg_reps", "sum"), **agg).reset_index()
        if len(AR):                                                            # 仮定の表の 3 行目（水準の当てはまり）を (A) の表に結合する（D8）
            a_all = a_all.merge(level_fit_table(AR)[["run", "student", "K", "pair_name", "level_fit_ratio", "level_fit_ok"]], how="left")
        write_csv(a_all, out / "a_summary_by_K.csv")
        a = a_all[a_all.K == "0"].drop(columns="K")
        write_csv(a, out / "a_summary_K0.csv")
        show = ["run", "student", "pair_name", "n_targets", "n_neg_targets", "A_D0_med", "A_D0_min", "A_D0_max", "none_dirloss_med", "none_dirloss_min", "none_dirloss_max",
                "none_dirloss_auc_med", "none_model_err_med"] + (["level_fit_ok"] if "level_fit_ok" in a.columns else [])
        lines += ["## (A) K=0 の none：対ごとの A(D,0)（縮小 LDA の向き．最良の向きの推定値）と向きの損（反復の平均のターゲットの中央値と範囲．d_AUC 版・模型の誤差・負の数も）", "",
                  a[show].to_string(index=False), ""]
    # (B)
    if len(U):
        def rep_lookup(r_):
            return boot_store.get(r_.boot_key)
        main = U[(U.comparison == MAIN_COMP) & (U.sensitivity == "main")]
        cnt = judgment_table(main, out / "b_judgments_by_pair.csv")
        nomd = judgment_table(main[main.model_dependent != True], out / "b_judgments_by_pair_nomd.csv")  # noqa: E712

        def with_replaced(rows_sub, mask_fn):
            """感度の表は主の行から，感度の操作を受ける行だけを差し替えて作る（S16）．"""
            return pd.concat([main[~mask_fn(main)], rows_sub]) if len(rows_sub) else main
        sj = U[(U.comparison == MAIN_COMP) & (U.sensitivity == "session_jk")]
        imp = U[(U.comparison == MAIN_COMP) & (U.sensitivity == "iemocap_impro_only") & (U.corpus == "IEMOCAP")]   # 差し替えるのは AVT の IEMOCAP の行だけ
        main_sj = with_replaced(sj, lambda d: d.corpus.isin(set(sj.corpus)))
        main_imp = with_replaced(imp, lambda d: (d.run == "fuse-audio-video-text") & (d.corpus == "IEMOCAP"))
        for sname in ("delta_half", "delta_double", "rev_order"):
            judgment_table(U[(U.comparison == MAIN_COMP) & (U.sensitivity == sname)], out / f"b_judgments_by_pair_{sname}.csv")
        judgment_table(main_imp.assign(sensitivity="iemocap_impro_only"), out / "b_judgments_by_pair_iemocap_impro_only.csv")
        judgment_table(main_sj.assign(sensitivity="session_jk"), out / "b_judgments_by_pair_session_jk.csv")
        for _, comp in COMPARISONS[1:]:
            judgment_table(U[(U.comparison == comp) & (U.sensitivity == "main")], out / f"b_judgments_by_pair_{comp.split('-')[-1]}.csv")
        lines += ["## (B) 単位 × 対の判定の数（主：none − oursx．pos＝減らした，neg＝増やした，same＝同じ水準，undet＝判定できない）", "", cnt.to_string(index=False), ""]
        lines += ["## (B) 同じく，模型に依る対を除いた数", "", nomd.to_string(index=False), ""]
        # 手法どうしの直接の差（S14・C8）
        DU, dstore = direct_comparisons(U, boot_store, jk_store)
        write_csv(DU, out / "b_units_direct.csv")
        if len(dstore):
            np.savez_compressed(out / "b_boot_direct.npz", **dstore)
        for name in DU.comparison.unique() if len(DU) else []:
            judgment_table(DU[DU.comparison == name], out / f"b_judgments_by_pair_{name}.csv")
        # P3a：主の表，感度の表，比較相手ごとの表
        F = p3a_table(main, rep_lookup, "main")
        write_csv(F, out / "b_p3_checks.csv")
        showF = ["family", "with_model_dependent", "is_main_version", "n_tests", "n_pos", "n_pos_elig", "n_pos_blocks", "count_threshold", "null_count_q95",
                 "n_pos_corpora_selected_elig", "n_units_full_elig", "n_full_elig_dropped_by_flag", "n_pos_full_elig", "share_selected_among_pos_full_elig", "p_class_perm_full_elig",
                 "p_min_attainable", "judgeable", "judgeable_after_flag", "P3a_verdict", "P3a_holds_cebd64b", "pos_pairs"]
        lines += ["## (B) 予言 P3a の判定（族ごと．数は複製の帰無の 95% 点，広がりは怒りの対の減らしたが出たコーパス，偏りは 6 対すべてが効果の出うるセルの単位での感情の名前の置換の厳密な p）", "",
                  F[showF].to_string(index=False), ""]
        sens = []
        for sname in ("delta_half", "delta_double", "rev_order"):
            sens.append(p3a_table(U[(U.comparison == MAIN_COMP) & (U.sensitivity == sname)], rep_lookup, sname))
        sens.append(p3a_table(main_imp, rep_lookup, "iemocap_impro_only"))
        sens.append(p3a_table(main_sj, rep_lookup, "session_jk"))
        mc = main.copy()
        mc["judgment"] = [("same_by_definition" if j == "same_by_definition" else judge_unit(x, lo, hi, dl)) for j, x, lo, hi, dl in zip(mc.judgment, mc.point, mc.lo, mc.hi, mc.delta_d_pair_corpus)]
        mc["delta_d_pair"] = mc.delta_d_pair_corpus
        sens.append(p3a_table(mc, rep_lookup, "delta_corpus"))
        sens.append(p3a_table(main[~main.corpus.isin(EXCL_MA)], rep_lookup, "excl_mosei_afew"))
        S_ = pd.concat(sens, ignore_index=True)
        write_csv(S_, out / "b_p3_checks_sens.csv")
        byc = [p3a_table(U[(U.comparison == comp) & (U.sensitivity == "main")], rep_lookup, "main") for _, comp in COMPARISONS[1:]]
        if len(DU):
            def dlookup(r_):
                return dstore.get(r_.boot_key)
            for name in DU.comparison.unique():
                d_ = DU[DU.comparison == name]
                byc.append(p3a_table(d_, dlookup, "direct_all"))
                byc.append(p3a_table(d_[d_.both_chosen], dlookup, "direct_both_chosen"))
        Bc = pd.concat(byc, ignore_index=True)
        write_csv(Bc, out / "b_p3_checks_by_comparison.csv")
        lines += ["## (B) 比較相手ごとの同じ表（groupB で P3a の 3 条件が成り立てば偏りは手法に特有とは書かない）", "",
                  Bc[Bc.is_main_version][["table", "comparison", "family", "n_pos", "n_pos_elig", "n_pos_blocks", "count_threshold", "n_pos_corpora_selected_elig",
                                          "n_units_full_elig", "share_selected_among_pos_full_elig", "p_class_perm_full_elig", "p_min_attainable", "cond_count",
                                          "cond_two_corpora", "cond_p", "pos_pairs"]].to_string(index=False), ""]
    # 再現率の差（UAR への橋）
    if len(RC):
        ru = RC.groupby(["comparison", "run", "student", "K", "target", "emotion"]).diff_recall.mean().reset_index()
        ru["corpus"] = ru.target.map(corp)
        ru = ru.groupby(["comparison", "run", "student", "K", "corpus", "emotion"]).diff_recall.mean().reset_index()
        write_csv(ru, out / "b_recall_units.csv")
    # (C)
    if len(J):
        def summ(Jx, name):
            Jc = Jx.groupby(["world", "student", "role", "K", "g", "pair_name", "pseudo_corpus"])[["J", "J_dir"]].median().reset_index()   # コーパスの中央値を先に取る（S7）
            Js = Jc.groupby(["world", "student", "role", "K", "g", "pair_name"]).agg(J_med=("J", "median"), J_q1=("J", lambda x: x.quantile(0.25)),
                                                                                     J_q3=("J", lambda x: x.quantile(0.75)), J_dir_med=("J_dir", "median"),
                                                                                     J_dir_q1=("J_dir", lambda x: x.quantile(0.25)), J_dir_q3=("J_dir", lambda x: x.quantile(0.75)),
                                                                                     n_corpora=("pseudo_corpus", "nunique")).reset_index()
            write_csv(Js, out / f"c_complementarity_summary{name}.csv")
            write_csv(Jc, out / f"c_complementarity_by_corpus{name}.csv")
            return Jc, Js
        Jc, Js = summ(J, "")
        Jc_ns, _ = summ(J[J.pseudo != "IEMOCAP-script"], "_noscript")          # 仮のターゲットの IEMOCAP 台本を除いた版（L6）
        lines += ["## (C) 対ごとの補い合い J_kl(f, g)（切片を含む）と向きだけの J_dir の中央値（仮のターゲットのコーパスの中央値を取ってから中央値）", "",
                  Js[Js.g.isin(["pF", "ref", "poe0"])].to_string(index=False), ""]
        # P3c（記述．判定はしない）：(B) で減った対と，同じコーパスを仮のターゲットにした J（g = ref と，oursx がその単位で選んだ教師）
        if len(U):
            main = U[(U.comparison == MAIN_COMP) & (U.sensitivity == "main")]
            wmap = {"fuse-audio-video-text": "avt", "fuse-audio-video": "av"}
            jix = Jc.set_index(["world", "student", "K", "g", "pair_name", "pseudo_corpus"])
            jix_ns = Jc_ns.set_index(["world", "student", "K", "g", "pair_name", "pseudo_corpus"])
            rows = []
            for r in main.itertuples():
                w = wmap[r.run]
                teacher = r.chosen.split("|")[2].split(":")[0] if r.chosen != "none" else ""
                gt = TEACHER_G.get(teacher)
                def jget(ix, g, col):
                    try:
                        return float(ix.loc[(w, r.student, r.K, g, r.pair_name, r.corpus), col])
                    except KeyError:
                        return np.nan
                rows.append({"run": r.run, "role": r.role, "student": r.student, "K": r.K, "corpus": r.corpus, "pair_name": r.pair_name, "judgment": r.judgment,
                             "model_dependent": r.model_dependent, "chosen": r.chosen, "teacher": teacher, "teacher_g": gt or "missing",
                             "elig": bool(r.none_dirloss >= r.delta_d_pair and r.chosen != "none" and r.judgment != "missing"),
                             "J_ref_same_corpus": jget(jix, "ref", "J"), "J_dir_ref_same_corpus": jget(jix, "ref", "J_dir"),
                             "J_teacher_same_corpus": jget(jix, gt, "J") if gt else np.nan, "J_dir_teacher_same_corpus": jget(jix, gt, "J_dir") if gt else np.nan,
                             "J_ref_same_corpus_noscript": jget(jix_ns, "ref", "J")})
            Q = pd.DataFrame(rows)
            write_csv(Q, out / "b_c_p3c_rows.csv")
            p3c_rows = []
            for jcol in ("J_ref_same_corpus", "J_dir_ref_same_corpus", "J_teacher_same_corpus", "J_dir_teacher_same_corpus", "J_ref_same_corpus_noscript"):
                Qk = Q[(Q.chosen != "none") & Q[jcol].notna()]
                for scope, Qs in (("all", Qk), ("elig", Qk[Qk.elig]), ("uni", Qk[Qk.role == "uni"]), ("fused", Qk[Qk.role == "fused"]), ("nomd", Qk[Qk.model_dependent != True])):  # noqa: E712
                    red = Qs[Qs.judgment == "pos"]
                    strata = Qs.groupby(["run", "student", "K"])[jcol].apply(lambda x: float((x > 0).mean()))
                    wbase = float(np.mean([strata.get((a, b, c), np.nan) for a, b, c in zip(red.run, red.student, red.K)])) if len(red) else np.nan
                    p3c_rows.append({"J_column": jcol, "scope": scope, "n_rows_all": len(Q), "n_rows_used": len(Qs), "n_excluded_none_chosen": int((Q.chosen == "none").sum()),
                                     "n_excluded_J_nan": int(Q[jcol].isna().sum()), "base_share_J_pos": float((Qs[jcol] > 0).mean()) if len(Qs) else np.nan,
                                     "base_share_J_pos_stratified": wbase, "n_reduced": len(red),
                                     "share_J_pos_among_reduced": float((red[jcol] > 0).mean()) if len(red) else np.nan, "descriptive_only": True})
            P3c = pd.DataFrame(p3c_rows)
            write_csv(P3c, out / "b_c_p3c.csv")
            lines += ["## P3c（記述．判定はしない）：減った対のうち J > 0 の割合と土台（全体と（世界，f，K）の層で重みづけた土台）", "", P3c.to_string(index=False), ""]
    if len(R):
        Rc = R.groupby(["world", "model", "emotion", "pseudo_corpus"]).recall.median().reset_index()
        Rs = Rc.groupby(["world", "model", "emotion"]).agg(recall_med=("recall", "median"), recall_q1=("recall", lambda x: x.quantile(0.25)),
                                                            recall_q3=("recall", lambda x: x.quantile(0.75)), n_corpora=("pseudo_corpus", "nunique")).reset_index()
        write_csv(Rs, out / "c_recall_summary.csv")
        lines += ["## (C) 感情ごとの再現率の中央値と四分位（モダリティの分類器と予測モデル）", "", Rs.to_string(index=False), ""]
    if len(Ph):
        Pc = Ph.groupby(["world", "mod1", "mod2", "emotion", "pseudo_corpus"])[["phi", "both_wrong", "indep_both_wrong"]].median().reset_index()
        Ps = Pc.groupby(["world", "mod1", "mod2", "emotion"]).agg(phi_med=("phi", "median"), phi_q1=("phi", lambda x: x.quantile(0.25)), phi_q3=("phi", lambda x: x.quantile(0.75)),
                                                                  both_wrong_med=("both_wrong", "median"), indep_both_wrong_med=("indep_both_wrong", "median"),
                                                                  n_corpora=("pseudo_corpus", "nunique")).reset_index()
        write_csv(Ps, out / "c_phi_summary.csv")
        lines += ["## (C) 誤りの φ（感情ごと．中央値と四分位）", "", Ps.to_string(index=False), ""]
    T_ = assumptions_table(A_, AR, U, anchors, out)
    if len(T_):
        lines += ["## 仮定の表（3 節．基準は値を見る前に決めた）", "", T_.to_string(index=False), ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def snapshot_code(out, stamp):
    """実行したコードの写しと git diff HEAD を results/e83/code/<開始時刻>/ に置く（methods.md 6.3）．"""
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
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e83")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)                             # E79 と同じ種
    ap.add_argument("--targets", nargs="+", help="試走用．評価のターゲットを限る")
    ap.add_argument("--stages", nargs="+", default=["ab", "c"], choices=["ab", "c"])
    ap.add_argument("--smoke", action="store_true", help="試走（--boot 50 と同じ扱い．provenance に残す）")
    args = ap.parse_args()
    if args.smoke:
        args.boot = min(args.boot, 50)
    out = args.out.resolve()
    partial = bool(args.smoke or args.targets or sorted(args.stages) != ["ab", "c"])
    if partial and out.is_relative_to(ROOT / "results"):
        raise SystemExit("試走（--smoke，--targets，--stages の一部）の出力は results/ の下に書かない．--out を checks/E83/ か scratchpad にする（I8）")
    if not partial and out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} が空でない（前の本番か途中で落ちた回の出力）．消すか別の --out にしてから流す（上書きで出所が消えないように）")
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--"] + list(SCRIPT_DEPS), cwd=ROOT, capture_output=True, text=True).stdout.strip()
    untracked = [l_[3:] for l_ in subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "exp", "measure"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
                 if l_.startswith("??")]
    HEADER = f"E83 commit={commit} dirty_deps={bool(dirty)} script_sha256={sha256(Path(__file__))[:16]} boot={args.boot} seed={args.seed} smoke={args.smoke} targets={args.targets} stages={args.stages} started={started.isoformat(timespec='seconds')}"
    prov = {"commit": commit, "dirty_deps": bool(dirty), "dirty_deps_status": dirty, "untracked_exp_measure": untracked, "script_sha256": sha256(Path(__file__)),
            "deps_sha256": {rel: sha256(ROOT / rel) for rel in SCRIPT_DEPS}, "args": vars(args) | {"out": str(out)},
            "constants": {"GEFF_MIN": GEFF_MIN, "MODEL_UNIT_FRAC": MODEL_UNIT_FRAC, "CHANCE": CHANCE, "DELTA_SCALES": list(DELTA_SCALES), "P_ALPHA": P_ALPHA,
                          "MIN_PAIR_N": MIN_PAIR_N, "REPS": list(REPS), "PSEUDO_REPS": list(PSEUDO_REPS), "KS": list(KS), "delta_rule": "2 × median_(T,c) SD_r(100·A(d_none,0))/√10",
                          "ANGER_PAIRS": [PAIR_NAME[p] for p in ANGER_PAIRS], "P3A_SIDE_PAIRS": [PAIR_NAME[p] for p in P3A_SIDE_PAIRS],
                          "P3B_SIDE_PAIRS": [PAIR_NAME[p] for p in P3B_SIDE_PAIRS], "VIDEO_PAIR": PAIR_NAME[NEU_ANG], "SEEDED": sorted(SEEDED), "EXCL_MA": list(EXCL_MA),
                          "class_perms": int(len(CLASS_PERM_MAPS)), "pair_perms_sensitivity": int(len(PAIR_PERM_MAPS)), "TEACHER_G": TEACHER_G},
            "comparisons": [c for _, c in COMPARISONS], "direct": [n for _, _, n in DIRECT], "main_comparison": MAIN_COMP,
            "started": started.isoformat(timespec="seconds"), "inputs": {}}
    prov["code_snapshot"] = str(snapshot_code(out, started.strftime("%Y%m%d-%H%M%S")).relative_to(out))
    A_ = AR = U = RC = J = R = Ph = AN = pd.DataFrame()
    boot_store, jk_store = {}, {}
    if "ab" in args.stages:
        A_, AR, U, RC, boot_store, jk_store = stage_ab(args, out, prov)
    if "c" in args.stages:
        J, R, Ph, AN = stage_c(args, out, prov)
    summarize(out, A_, AR, U, RC, J, R, Ph, AN, boot_store, jk_store)
    prov["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
