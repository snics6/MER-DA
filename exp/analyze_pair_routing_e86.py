"""E86：怒りの確率の出どころを替える振り分けの余地——感情の対ごとの補いを，仮のターゲットの候補の上で，自分のモダリティを除いたモデルについて数える（下調べ）．

設計（正本）：research_notes/研究提案_2026-11-04/段0a_設計案.md（E86 1〜5 節．規則は値を見る前に書いた）．背景は同じフォルダの 案_2_改.md．
下調べ（仮のターゲットの候補だけ．評価のターゲット T は置き場の名前として使うだけで，T のテストの値もラベルも読まない）．

入力（すべて保存済み）
  - E80 の単位ファイル results/e80/<world>/units/<fused>/<T>/<c>_r<r>.npz（候補の行番号 cand，テストの行番号 te，ラベル y・yte，K 本の位置 lab__K，
    候補の上の確率 P__<kind>（ソースだけの出発点．P__<fused> は融合のモデル），ref__K（学び直して掛けたモデル），none__<f>__K（K 本で学び直した配備する分類器＝比較の基準．log_softmax））
  - samples_<c>.csv（行番号 → sample_id・speaker_group）．区間のブートストラップの話者群に使う
  - 特徴：measure/data.load_any(c, <fused>) の行を cand で引き，配備する分類器 f のブロック（kind の順に連結．音声 1024・顔 512・言葉 768）だけを使う（土台の D にだけ使う）
新しい学習：土台の D を決める Ledoit–Wolf の LDA（exp/explore_pairs.own_separation．話者群で 2 分）だけ．

モデル（候補の発話ごとの 4 感情の確率．0 平静・1 喜び・2 悲しみ・3 怒り）
  比較の基準 a = none__<f>__K（指数を取って正規化）．既定 q⁰：主 ref = ref__K，副 pF = P__<fused>．候補 s = f 以外の P__<m> の要素ごとの積を正規化（av は残り 1 つそのもの）．
  記述 other_<m>（avt だけ）．階層の形 h(s, α)：h(怒) = α·s(怒) + (1−α)·q⁰(怒)，怒り以外 c：h(c) = (1−h(怒))·q⁰(c)/(1−q⁰(怒))．α ∈ {1 (h1), 0.5 (h05)}．
対ごとの補い J_kl = 100·[P(モデルが当たり ∧ a が外す) − P(モデルが外し ∧ a が当たる)]（= 2 感情の正解率の差．E83 (C) と同じ）．Δ_kl = J_kl(モデル) − J_kl(q⁰)．
δ_J，区間（話者群の復元抽出），判定（pos/neg/same/undet），土台（D_kl，d_kl，方向の誤り），数え上げ，コーパスの塊，怒りと答える割合，階層の切り分けは設計 2・3 節のとおり．

出力（--out．既定 results/e86）：pairs.csv（反復ごと），base.csv（反復ごとの土台），units.csv（単位 × 対 × 候補 × 既定．区間・δ_J・判定・土台の条件），delta.csv，
  counts.csv，corpus_counts.csv，anger_share.csv，hier_counts.csv，checks.csv（照合の記録），provenance.json，script_copy.py．各 CSV の先頭に `# E86 commit=...` の行（読むときは comment="#"）．

使い方：./.venv/bin/python exp/analyze_pair_routing_e86.py [--out results/e86] [--worlds avt av] [--reps 5] [--boot 1000] [--targets T ...] [--smoke] [--selftest]
  --smoke：avt の 1 ターゲット・反復 1 つ・ブートストラップ 50 回．出力は --out か checks/E86/code_run/smoke．
  --selftest：人工データでの退化の検算（設計 3 節）．ファイルは読まず書かない．
  部分的な流し方（--smoke，--targets，--reps < 5，--worlds の一部）の出力は results/ の下に書けない．出力先に古い CSV があれば止まる．

設計に無い決め事（設計を変えずに台本で決めたこと．検査役はここを見る）
  D1 対数で保存された確率の見分け方：配列の最小値が負なら対数（log_softmax）とみなして指数を取る（設計 2 節「値が負」）．行の和の検査（10⁻⁴）は指数を取った後の和に当てる．
     d_kl に使う対数確率は，対数で保存された配列はそのまま，確率で保存された配列は log(max(p, 1e−12))．
  D2 階層の形で q⁰(怒) が 1 に極めて近い行：q⁰ を 1e−12 で下から切って正規化してから使う（1 − q⁰(怒) ≥ 3e−12）．怒り以外の比 q⁰(c)/(1 − q⁰(怒)) の分母は
     怒り以外の 3 つの和で計算する（同じ値．1 − q⁰(怒) の桁落ちを避ける）．α = 0 の検算はこの切った q⁰ と比べる（差は 1e−9 以内）．
  D3 2 感情の判定の同点（k と l の確率が等しい）は番号の小さい k とする（E83 (C) の `>=` と同じ）．
  D4 対の最小の本数は設けない（E83 (C) は 10）．対のどちらかの感情が候補に無い反復は J・Δ を NaN にし，n_pair を列に残す．点の値は NaN を除いた反復の平均（n_reps_finite を列に残す）．
     E83 との照合は E83 が値を出す n_pair ≥ 10 の行だけで行う．
  D5 土台の 2 分割は，K 本を除いた候補の話者群に speaker_folds(k=2, seed=0) を当てる（K ごとに分け直す）．どちらかの半分でどれかの感情が 2 本未満なら D は NaN（checks.csv に記録）．
     特徴の標準化の統計も K 本を除いた候補から取る（標準偏差に 1e−8 を足す．E85 と同じ）．
  D6 ブートストラップの重みは，反復ごとに候補（K 本を除く前）の話者群の全体に多項分布（群の数 G，確率 1/G）を 1 回引き，同じ重みを全 f・K・既定・候補・対に使う
     （K 本を除いて空になった群は分母に入らない）．種は [86, crc32(world), crc32(T), crc32(c)]．反復 r の重みは r 番目の引き（反復の処理の順は 0, 1, 2, …）．
     復元抽出で対の分母が 0 になった引きは NaN とし，反復の平均と百分位から除く（n_boot_valid を列に残す）．
  D7 δ_J の標準偏差は ddof = 1．反復が 2 未満の (T, c, 対) は δ_J の材料から除く（試走の反復 1 つでは δ_J が無く判定は missing）．÷√5 は ÷√(反復の数)（本番は 5）．
  D8 判定の呼び方は E79・E83 と同じ pos（上回った）・neg（下回った）・same（同じ水準）・undet（判定できない）．点・区間・δ_J のどれかが無ければ missing．
  D9 コーパスの塊の「過半」は，塊の中の (T, K) のセルのうち pos が厳密に半分を超えること．土台で条件づけた版も並べる（塊の中の土台の条件を満たすセルだけで数える）．
  D10 怒りと答える割合は反復ごとの値も列に残す（share_anger_r<r>）．other_<m>（avt）についても出す．R86-4 の ±2 点は s との差 diff_vs_s の列で読む．
  D11 候補のモデル s の積が全て 0 の行があれば止まる．
  D12 偶然の見込みは 1 セルあたり 0.025（両側 5% の片側）．塊の偶然の見込みは 塊の数 × 0.025（注：T・K をまたぐ数え上げの偶然の見込みはこれより大きい）．
  D13 --targets（T を絞る）と --boot（回数）を足した（試走と検査のため）．
  D14 BLAS のスレッド数を 4 に固定する（環境変数 OMP_NUM_THREADS 等が既に設定されていればそれに従う）．WSL2 の 16 スレッドの既定では 1024×1024 の固有値分解と solve が
     スレッドの取り合いで 8〜80 倍遅く（3.4 秒・1.6 秒 → 0.43 秒・0.02 秒），Ledoit–Wolf の共分散の値は同一であることを確かめた．provenance に記録する．
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")                                      # numpy を読む前に決める（D14）

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, fused_parts, load_any, load_group  # noqa: E402
from explore_pairs import PAIRS, ba, own_separation  # noqa: E402
from split import speaker_folds  # noqa: E402

WORLDS = {"avt": "fuse-audio-video-text", "av": "fuse-audio-video"}
KS = (0, 5, 25)
EMO = ("平静", "喜び", "悲しみ", "怒り")
ANGER = 3
PAIR_NAME = [f"{EMO[k]}–{EMO[j]}" for k, j in PAIRS]
ANGER_PAIRS = tuple(p for p, (k, j) in enumerate(PAIRS) if ANGER in (k, j))
DEFAULTS = ("ref", "pF")                                   # 既定 q⁰：主 = ref__K，副 = P__<fused>
HIER = ((1.0, "h1"), (0.5, "h05"))                         # 階層の形 h(s, α)
CHANCE = 0.025                                             # 1 セルあたりの偶然の見込み（両側 5% の片側）
DELTA_SCALES = ((1.0, "main"), (0.5, "half"), (2.0, "double"))
PCT = (2.5, 97.5)
MIN_GROUPS = 10                                            # 話者群がこれ未満の単位は区間が名目より狭いので印
SUM_TOL = 1e-4
EPS = 1e-12
E83_TOL = 1e-6
E83_MIN_PAIR_N = 10
BOOT_SEED = 86
DEFAULT_REPS = 5
DEFAULT_BOOT = 1000
SMOKE_BOOT = 50
SCRIPT_DEPS = ("exp/analyze_pair_routing_e86.py", "exp/explore_pairs.py", "exp/explore_geometry.py", "exp/explore_mechanism.py", "exp/pair_table.py",
               "exp/summarize_map.py", "measure/data.py", "measure/split.py", "features/common.py", "data/metadata")
CONSTANTS = dict(KS=KS, DEFAULTS=DEFAULTS, HIER=HIER, CHANCE=CHANCE, DELTA_SCALES=DELTA_SCALES, PCT=PCT, MIN_GROUPS=MIN_GROUPS, SUM_TOL=SUM_TOL, EPS=EPS,
                 E83_TOL=E83_TOL, E83_MIN_PAIR_N=E83_MIN_PAIR_N, BOOT_SEED=BOOT_SEED, ANGER=ANGER, PAIRS=PAIRS, ANGER_PAIRS=ANGER_PAIRS,
                 judge="pos: lo>0 & x>=dl / neg: hi<0 & x<=-dl / same: lo>=-dl & hi<=dl / undet: else / missing: NaN",
                 delta_J="2 * median_{T,c,pair}( sd_{reps}(J_kl(q0)) ) / sqrt(n_reps)", base="dir_err = 100*(Phi(D/2) - Phi(d/2)) >= delta_J")
HEADER = "E86"


# ---------------------------------------------------------------- 道具

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def corp(g):
    return GROUP_SPEC[g][0]


def write_csv(df, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {HEADER}\n")
        df.to_csv(fh, index=False)


def to_prob(arr, where):
    """保存された確率（または log_softmax）を (n, 4) の確率にする（D1）．行の和が 1 から SUM_TOL 以上ずれれば止まる．戻り (P, is_log)．"""
    A = np.asarray(arr, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != 4:
        raise SystemExit(f"{where}：形が (n, 4) でない {A.shape}")
    if not np.isfinite(A).all():
        raise SystemExit(f"{where}：NaN/Inf")
    is_log = bool(A.min() < 0)
    if is_log:
        A = np.exp(A)
    s = A.sum(1)
    dev = float(np.abs(s - 1).max())
    if dev > SUM_TOL:
        raise SystemExit(f"{where}：確率の行の和が 1 から {dev:.2e} ずれる（許容 {SUM_TOL}）")
    return A / s[:, None], is_log


def to_logp(arr):
    """d_kl に使う対数確率（D1）．"""
    A = np.asarray(arr, dtype=np.float64)
    return A if A.min() < 0 else np.log(np.clip(A, EPS, None))


def product_model(Ps):
    """残りの出発点の積を正規化（D11）．"""
    M = np.ones_like(Ps[0])
    for P in Ps:
        M = M * P
    s = M.sum(1)
    if not (s > 0).all():
        raise SystemExit("候補のモデル s の積が全て 0 の行がある")
    return M / s[:, None]


def hier(s, q0, alpha):
    """階層の形 h(s, α)（設計 2 節の表）．q⁰ は EPS で切って正規化（D2）．"""
    q = np.clip(q0, EPS, None)
    q = q / q.sum(1, keepdims=True)
    h = np.empty_like(q)
    h[:, ANGER] = alpha * s[:, ANGER] + (1 - alpha) * q[:, ANGER]
    rest = [c for c in range(4) if c != ANGER]
    ratio = q[:, rest] / q[:, rest].sum(1, keepdims=True)                # = q⁰(c)/(1 − q⁰(怒))．1 − q⁰(怒) の桁落ちを避けて和で割る（D2）
    h[:, rest] = (1 - h[:, ANGER])[:, None] * ratio
    return h


def pair_right(P, y, p):
    """対 p の 2 感情の判定（k と l の確率の大きい方．同点は k．D3）が正解か．P・y は対の発話に絞ったもの．"""
    k, j = PAIRS[p]
    dec = np.where(P[:, k] >= P[:, j], k, j)
    return dec == y


def j_stat(mr, ar):
    """J_kl = 100·[P(モデルが当たり ∧ a が外す) − P(モデルが外し ∧ a が当たる)]．"""
    return 100 * (float(np.mean(mr & ~ar)) - float(np.mean(~mr & ar)))


def pair_d(L, y):
    """比較の基準の離れ方 d_kl（E13・E83 の d_o_pairs と同じ式．z = L_k − L_l）．戻り (6,)．"""
    d = np.full(6, np.nan)
    for p, (k, j) in enumerate(PAIRS):
        zk, zj = L[y == k, k] - L[y == k, j], L[y == j, k] - L[y == j, j]
        if len(zk) < 2 or len(zj) < 2:
            continue
        sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
        if sd > 0:
            d[p] = (zk.mean() - zj.mean()) / sd
    return d


def dir_err(D, d):
    """方向の誤り = 100·[Φ(D/2) − Φ(d/2)]（2 つの山の模型．切片を最良に直した 2 感情の正解率の差）．"""
    D, d = np.asarray(D, dtype=np.float64), np.asarray(d, dtype=np.float64)
    return 100 * (ba(D, 0.0) - ba(d, 0.0))


def base_D(Z, y, groups):
    """最良の方向での離れ方 D_kl：話者群で 2 分（speaker_folds k=2 seed=0）し，explore_pairs.own_separation（Ledoit–Wolf の LDA）で測る（D5）．
    戻り (D (6,), 半分ごとの本数, 半分ごとの話者群の数, 検算の記録)．"""
    half = speaker_folds(pd.Series(groups), k=2, seed=0).to_numpy()
    n_half = [int((half == a).sum()) for a in (0, 1)]
    g_half = [int(len(np.unique(groups[half == a]))) for a in (0, 1)]
    ok = all(((y[half == a] == k).sum() >= 2) for a in (0, 1) for k in range(4))
    if not ok:
        return np.full(6, np.nan), n_half, g_half, "D skipped: some emotion < 2 in a half"
    return own_separation(Z, y, half), n_half, g_half, "ok"


def judge(x, lo, hi, dl):
    """E79・E83 と同じ 4 区分（D8）．"""
    if not (np.isfinite(x) and np.isfinite(lo) and np.isfinite(hi) and np.isfinite(dl)):
        return "missing"
    if lo > 0 and x >= dl:
        return "pos"
    if hi < 0 and x <= -dl:
        return "neg"
    if lo >= -dl and hi <= dl:
        return "same"
    return "undet"


def feats_for(c, fused):
    """E85 と同じ：融合の特徴と，kind ごとのブロックの位置．"""
    df, X = load_any(c, fused)
    mods, kinds = fused_parts(fused)
    dims = [load_group(c, kd)[1].shape[1] for kd in kinds]
    blocks, lo = {}, 0
    for kd, dm in zip(kinds, dims):
        blocks[kd] = (lo, lo + dm)
        lo += dm
    if lo != X.shape[1]:
        raise SystemExit(f"{c} {fused}：ブロックの次元の和 {lo} が特徴 {X.shape[1]} と合わない")
    return df, X, blocks


# ---------------------------------------------------------------- 単位ファイル

class Unit:
    def __init__(self, world, T, c, r, feats, samples, chk):
        fused = WORLDS[world]
        self.world, self.T, self.c, self.r, self.fused = world, T, c, r, fused
        self.path = ROOT / "results" / "e80" / world / "units" / fused / T / f"{c}_r{r}.npz"
        self.z = np.load(self.path)
        self.cand, self.te = np.asarray(self.z["cand"]).astype(int), np.asarray(self.z["te"]).astype(int)
        self.y, self.yte = np.asarray(self.z["y"]).astype(int), np.asarray(self.z["yte"]).astype(int)
        df, X, blocks = feats
        tag = dict(world=world, T=T, c=c, r=r)
        if len(self.cand) != len(self.y) or len(self.te) != len(self.yte) or int(max(self.cand.max(), self.te.max())) >= len(df) or min(self.cand.min(), self.te.min()) < 0:
            raise SystemExit(f"{world} {T} {c} r{r}：候補・テストの行番号が発話一覧と合わない")
        lab = df.label_4.to_numpy().astype(int)
        if not np.array_equal(lab[self.cand], self.y) or not np.array_equal(lab[self.te], self.yte):
            raise SystemExit(f"{world} {T} {c} r{r}：ラベルが発話一覧と合わない")
        self.y, self.yte = lab[self.cand], lab[self.te]                  # 作業用のラベルは発話一覧から貼り直したもの（npz の y と一致を確かめた後．CLAUDE.md 1）
        if len(np.intersect1d(self.cand, self.te)):
            raise SystemExit(f"{world} {T} {c} r{r}：候補とテストの行が重なる")
        chk.append({**tag, "check": "rows_labels_vs_manifest", "status": "ok", "detail": f"n_cand={len(self.cand)} n_te={len(self.te)}"})
        self.X, self.blocks, self.df = X, blocks, df
        self.mods = fused_parts(fused)[1]
        for m in self.mods:
            if f"P__{m}" not in self.z.files:
                raise SystemExit(f"{world} {T} {c} r{r}：P__{m} が無い")
        chk.append({**tag, "check": "P_keys_present", "status": "ok", "detail": ",".join(self.mods)})
        self.labs = {}
        for K in KS:
            lab_K = np.asarray(self.z[f"lab__{K}"]).astype(int)
            if len(lab_K) != 4 * K or (len(lab_K) and (lab_K.min() < 0 or lab_K.max() >= len(self.cand) or len(np.unique(lab_K)) != len(lab_K))):
                raise SystemExit(f"{world} {T} {c} r{r}：lab__{K} が候補の位置の部分集合でない（{len(lab_K)} 本）")
            self.labs[K] = lab_K
        chk.append({**tag, "check": "labK_subset_of_cand", "status": "ok", "detail": ";".join(f"K{K}:{len(self.labs[K])}" for K in KS)})
        self.groups = samples.speaker_group.astype(str).to_numpy()[self.cand]
        self.n_groups = int(len(np.unique(self.groups)))
        # 確率（候補の全体）
        self.P = {}
        self.is_log = {}
        for key in [f"P__{m}" for m in self.mods] + [f"P__{fused}"] + [f"ref__{K}" for K in KS] + [f"none__{m}__{K}" for m in self.mods for K in KS]:
            self.P[key], self.is_log[key] = to_prob(self.z[key], f"{world} {T} {c} r{r} {key}")
            if self.P[key].shape[0] != len(self.cand):
                raise SystemExit(f"{world} {T} {c} r{r} {key}：行数が候補と合わない")
        chk.append({**tag, "check": "prob_rows_sum_to_one", "status": "ok", "detail": "log:" + ",".join(k for k, v in self.is_log.items() if v)})

    def feat(self, f, rows):
        lo, hi = self.blocks[f]
        return self.X[rows, lo:hi]


def process_unit(u, B, W, rows_p, rows_b, rows_a, boot, chk):
    """1 つの単位ファイルで，f × K × 既定 × モデル × 対の J・Δ（反復の行），土台，怒りの割合，ブートストラップの材料を出す．W：(B, G) の話者群の重み（この反復の引き）．"""
    fused = u.fused
    ug, gi_all = np.unique(u.groups, return_inverse=True)
    G = len(ug)
    if W.shape != (B, G):
        raise SystemExit(f"{u.world} {u.T} {u.c} r{u.r}：重みの形 {W.shape} が (B, G)=({B}, {G}) と合わない")
    tag0 = dict(world=u.world, T=u.T, c=u.c, unit=corp(u.c), r=u.r)
    for f in u.mods:
        for K in KS:
            keep = np.ones(len(u.cand), bool)
            keep[u.labs[K]] = False                                      # K 本を候補から除く（E83 (C) と同じ）
            yk, gi = u.y[keep], gi_all[keep]
            n_kept, n_groups = int(keep.sum()), int(len(np.unique(gi)))
            A = u.P[f"none__{f}__{K}"][keep]                             # 比較の基準 a
            others = [m for m in u.mods if m != f]
            s = product_model([u.P[f"P__{m}"][keep] for m in others])
            base_models = {"s": s}
            if len(others) >= 2:                                         # 記述：残りの単モダリティ（avt だけ）
                for m in others:
                    base_models[f"other_{m}"] = u.P[f"P__{m}"][keep]
            masks = [(yk == k) | (yk == j) for k, j in PAIRS]
            a_right = [pair_right(A[m], yk[m], p) for p, m in enumerate(masks)]
            tag = {**tag0, "f": f, "K": K, "n_cand": len(u.cand), "n_cand_kept": n_kept, "n_groups": n_groups}
            share_true = float((yk == ANGER).mean())
            for dflt in DEFAULTS:
                q0 = (u.P[f"ref__{K}"] if dflt == "ref" else u.P[f"P__{fused}"])[keep]
                models = {"q0": q0, **base_models}
                for alpha, name in HIER:
                    models[name] = hier(s, q0, alpha)
                # 対ごとの J と，話者群ごとの当たりの数（ブートストラップの材料）
                right = {name: [pair_right(P[m], yk[m], p) for p, m in enumerate(masks)] for name, P in models.items()}
                J = {name: [j_stat(right[name][p], a_right[p]) if masks[p].any() else np.nan for p in range(6)] for name in models}
                for name, P in models.items():
                    rows_a.append({**tag, "default": dflt, "model": name, "share_anger": float((P.argmax(1) == ANGER).mean()), "share_anger_true": share_true})
                    for p in range(6):
                        m = masks[p]
                        rows_p.append({**tag, "default": dflt, "model": name, "pair": p, "pair_name": PAIR_NAME[p], "anger_pair": p in ANGER_PAIRS,
                                       "n_pair": int(m.sum()), "acc_a": float(a_right[p].mean()) if m.any() else np.nan,
                                       "acc_model": float(right[name][p].mean()) if m.any() else np.nan,
                                       "J": J[name][p], "J_q0": J["q0"][p], "Delta": J[name][p] - J["q0"][p]})
                        if name == "q0" or not m.any():
                            continue
                        den = np.bincount(gi[m], minlength=G).astype(np.float64)
                        num = np.bincount(gi[m], weights=right[name][p].astype(np.float64) - right["q0"][p].astype(np.float64), minlength=G)
                        wd = W @ den
                        with np.errstate(divide="ignore", invalid="ignore"):
                            db = np.where(wd > 0, 100 * (W @ num) / wd, np.nan)
                        key = (f, K, dflt, name, p)
                        if key not in boot:
                            boot[key] = {"sum": np.zeros(B), "n": 0}
                        boot[key]["sum"] += db                           # NaN は伝播させ，最後に除く（D6）
                        boot[key]["n"] += 1
            # 土台：D（特徴の LDA）と d（比較の基準の対数確率）
            Z = u.feat(f, u.cand[keep])
            mu, sd = Z.mean(0), Z.std(0) + 1e-8
            D, n_half, g_half, note = base_D((Z - mu) / sd, yk, u.groups[keep])
            if note != "ok":
                chk.append({**tag0, "f": f, "K": K, "check": "base_D", "status": "skipped", "detail": note})
            d = pair_d(to_logp(u.z[f"none__{f}__{K}"])[keep], yk)
            de = dir_err(D, d)
            for p in range(6):
                rows_b.append({**tag, "pair": p, "pair_name": PAIR_NAME[p], "anger_pair": p in ANGER_PAIRS, "n_pair": int(masks[p].sum()),
                               "n_half0": n_half[0], "n_half1": n_half[1], "n_groups_half0": g_half[0], "n_groups_half1": g_half[1],
                               "D": float(D[p]), "d": float(d[p]), "dir_err": float(de[p])})


def finalize_boot(world, T, c, boot, rows_i):
    for (f, K, dflt, name, p), v in boot.items():
        vals = v["sum"] / v["n"]
        ok = np.isfinite(vals)
        lo, hi = (np.percentile(vals[ok], PCT) if ok.sum() >= 2 else (np.nan, np.nan))
        rows_i.append(dict(world=world, T=T, c=c, f=f, K=K, default=dflt, model=name, pair=p, lo=float(lo), hi=float(hi), n_boot_valid=int(ok.sum()), n_reps_boot=v["n"]))


# ---------------------------------------------------------------- まとめ

def delta_table(Pp, n_reps):
    """δ_J：（世界，f，K，既定）ごとに J_kl(q⁰) の反復の SD（T，c，対ごと）の中央値 × 2 ÷ √(反復の数)（D7）．"""
    q = Pp[Pp.model == "q0"]
    g = q.groupby(["world", "f", "K", "default", "T", "c", "pair"]).agg(n=("r", "nunique"), sd=("J", lambda x: x.std(ddof=1))).reset_index()
    g = g[(g.n >= 2) & np.isfinite(g.sd)]
    if g.empty:
        D = pd.DataFrame(columns=["world", "f", "K", "default", "n_cells", "n_reps", "median_sd", "delta_J", "delta_J_half", "delta_J_double"])
        return D
    D = g.groupby(["world", "f", "K", "default"]).agg(n_cells=("sd", "size"), median_sd=("sd", "median")).reset_index()
    D["n_reps"] = n_reps
    D["delta_J"] = 2 * D.median_sd / np.sqrt(n_reps)
    D["delta_J_half"] = 0.5 * D.delta_J
    D["delta_J_double"] = 2.0 * D.delta_J
    return D


def units_table(Pp, Bb, Ii, Dd, n_reps):
    """単位（世界，T，c，K）× f × 既定 × 候補 × 対の表：点の値（反復の平均），区間，δ_J，判定，土台の条件．"""
    P = Pp[Pp.model != "q0"].copy()
    keys = ["world", "T", "c", "unit", "f", "K", "default", "model", "pair", "pair_name", "anger_pair"]
    agg = P.groupby(keys).agg(n_reps=("r", "nunique"), n_reps_finite=("Delta", lambda x: int(np.isfinite(x).sum())), n_pair_mean=("n_pair", "mean"),
                             n_groups_min=("n_groups", "min"), n_cand_kept_mean=("n_cand_kept", "mean"),
                             J_model_mean=("J", "mean"), J_q0_mean=("J_q0", "mean"), Delta_mean=("Delta", "mean")).reset_index()
    wide = P.pivot_table(index=keys, columns="r", values="Delta", aggfunc="first")
    wide.columns = [f"Delta_r{r}" for r in wide.columns]
    U = agg.merge(wide.reset_index(), on=keys, how="left")
    U = U.merge(Ii, on=["world", "T", "c", "f", "K", "default", "model", "pair"], how="left")
    U["few_groups"] = U.n_groups_min < MIN_GROUPS
    # 土台（反復の平均．f × K × 対）
    bk = ["world", "T", "c", "f", "K", "pair"]
    Bm = Bb.groupby(bk).agg(D_mean=("D", "mean"), d_mean=("d", "mean"), dir_err_mean=("dir_err", "mean"),
                            n_reps_base=("dir_err", lambda x: int(np.isfinite(x).sum()))).reset_index()
    U = U.merge(Bm, on=bk, how="left")
    U = U.merge(Dd[["world", "f", "K", "default", "delta_J"]], on=["world", "f", "K", "default"], how="left")
    for col in ("delta_J", "lo", "hi", "Delta_mean", "dir_err_mean"):
        U[col] = pd.to_numeric(U[col], errors="coerce")                  # δ_J の表が空（反復 1 つ）でも数値型にそろえる
    for scale, sname in DELTA_SCALES:
        dl = scale * U.delta_J
        U[f"judge_{sname}"] = [judge(x, lo, hi, d) for x, lo, hi, d in zip(U.Delta_mean, U.lo, U.hi, dl)]
        U[f"base_ok_{sname}"] = np.where(np.isfinite(U.dir_err_mean) & np.isfinite(dl), U.dir_err_mean >= dl, False)
    U["dir_err_neg"] = np.where(np.isfinite(U.dir_err_mean), U.dir_err_mean < 0, False)
    return U


def long_judgments(U):
    """判定の長い表（1 行 = 単位 × 対 × 候補 × 既定 × δ の尺度）．"""
    out = []
    for scale, sname in DELTA_SCALES:
        x = U[["world", "T", "c", "unit", "f", "K", "default", "model", "pair", "pair_name", "anger_pair", "dir_err_neg", f"judge_{sname}", f"base_ok_{sname}"]].copy()
        x = x.rename(columns={f"judge_{sname}": "judge", f"base_ok_{sname}": "base_ok"})
        x["scale"] = sname
        out.append(x)
    L = pd.concat(out, ignore_index=True)
    L["pair_type"] = np.where(L.anger_pair, "anger", "nonanger")
    L["K"] = L.K.astype(str)
    return L


def counts_table(L):
    """数え上げ（世界 × f × K（と all）× 既定 × 候補 × 対の種類 × 尺度）：pos/neg/same/undet/missing，分母，土台後の分母，偶然の見込み，判定の規則の列（R86-1・R86-2）．"""
    rows = []
    for with_all in (False, True):
        X = L.copy()
        if with_all:
            X["K"] = "all"
        for key, g in X.groupby(["world", "f", "K", "default", "model", "pair_type", "scale"]):
            b = g[g.base_ok]
            n, nb = len(g), len(b)
            row = dict(zip(["world", "f", "K", "default", "model", "pair_type", "scale"], key))
            row.update(n_cells=n, chance=n * CHANCE, n_pos=int((g.judge == "pos").sum()), n_neg=int((g.judge == "neg").sum()), n_same=int((g.judge == "same").sum()),
                       n_undet=int((g.judge == "undet").sum()), n_missing=int((g.judge == "missing").sum()),
                       n_base=nb, chance_base=nb * CHANCE, n_pos_base=int((b.judge == "pos").sum()), n_neg_base=int((b.judge == "neg").sum()),
                       n_same_base=int((b.judge == "same").sum()), n_undet_base=int((b.judge == "undet").sum()), n_missing_base=int((b.judge == "missing").sum()),
                       share_pos_base=(float((b.judge == "pos").mean()) if nb else np.nan),
                       n_dir_err_neg=int(g.dir_err_neg.sum()), share_dir_err_neg=float(g.dir_err_neg.mean()) if n else np.nan)
            row["exceeds_chance_base"] = bool(row["n_pos_base"] > row["chance_base"]) if nb else False      # R86-1／R86-2 の規則
            row["exceeds_chance_all"] = bool(row["n_pos"] > row["chance"]) if n else False
            rows.append(row)
    return pd.DataFrame(rows)


def corpus_table(L):
    """コーパスの塊（c のコーパス，対）に畳む：どの (T, K) かで 1 回でも pos なら any_pos，過半で pos なら majority_pos（D9）．塊の数 × 0.025 を添える（D12）．"""
    blocks, totals = [], []
    key_b = ["world", "f", "default", "model", "scale", "unit", "pair", "pair_name", "pair_type"]
    for key, g in L.groupby(key_b):
        b = g[g.base_ok]
        n, npos, nb, npb = len(g), int((g.judge == "pos").sum()), len(b), int((b.judge == "pos").sum())
        blocks.append({**dict(zip(key_b, key)), "level": "block", "n_cells": n, "n_pos": npos, "any_pos": npos >= 1, "majority_pos": npos > n / 2,
                       "n_cells_base": nb, "n_pos_base": npb, "any_pos_base": npb >= 1, "majority_pos_base": (npb > nb / 2) if nb else False})
    Bk = pd.DataFrame(blocks)
    key_t = ["world", "f", "default", "model", "scale", "pair_type"]
    for key, g in Bk.groupby(key_t):
        totals.append({**dict(zip(key_t, key)), "level": "total", "n_blocks": len(g), "chance_blocks": len(g) * CHANCE,
                       "n_any_pos": int(g.any_pos.sum()), "n_majority_pos": int(g.majority_pos.sum()),
                       "n_blocks_base": int((g.n_cells_base > 0).sum()), "chance_blocks_base": int((g.n_cells_base > 0).sum()) * CHANCE,
                       "n_any_pos_base": int(g.any_pos_base.sum()), "n_majority_pos_base": int(g.majority_pos_base.sum()),
                       "note": "chance_blocks is the per-comparison chance; the chance of a count across T and K is larger"})
    return pd.concat([Bk, pd.DataFrame(totals)], ignore_index=True)


def hier_table(L):
    """階層の形の切り分け（R86-3）：s が pos の（単位，対）のうち h1・h05 も pos の数と割合（全体と土台後．K ごとと all）．"""
    idx = ["world", "T", "c", "f", "K", "default", "pair", "pair_type", "scale"]
    S = L[L.model.isin(["s", "h1", "h05"])].pivot_table(index=idx, columns="model", values="judge", aggfunc="first").reset_index()
    base = L[L.model == "s"][idx + ["base_ok"]]
    S = S.merge(base, on=idx, how="left")
    rows = []
    for with_all in (False, True):
        X = S.copy()
        if with_all:
            X["K"] = "all"
        for key, g in X.groupby(["world", "f", "K", "default", "pair_type", "scale"]):
            for cond, gg in (("all", g), ("base", g[g.base_ok])):
                sp = gg[gg.s == "pos"]
                n = len(sp)
                h1 = int((sp.h1 == "pos").sum()) if n else 0
                h05 = int((sp.h05 == "pos").sum()) if n else 0
                rows.append({**dict(zip(["world", "f", "K", "default", "pair_type", "scale"], key)), "cond": cond, "n_cells": len(gg), "n_s_pos": n,
                             "n_h1_also_pos": h1, "n_h05_also_pos": h05, "share_h1": h1 / n if n else np.nan, "share_h05": h05 / n if n else np.nan,
                             "majority_h1": (h1 / n > 0.5) if n else False, "h05_le_h1": (h05 <= h1) if n else False,
                             "n_h1_pos_total": int((gg.h1 == "pos").sum()), "n_h05_pos_total": int((gg.h05 == "pos").sum())})
    return pd.DataFrame(rows)


def anger_table(Aa, n_reps):
    """怒りと答える割合（単位の表．反復の平均と反復ごとの値．D10）．R86-4：s との差 diff_vs_s と ±2 点の印．"""
    keys = ["world", "T", "c", "unit", "f", "K", "default", "model"]
    agg = Aa.groupby(keys).agg(n_reps=("r", "nunique"), share_anger_mean=("share_anger", "mean"), share_anger_true_mean=("share_anger_true", "mean")).reset_index()
    wide = Aa.pivot_table(index=keys, columns="r", values="share_anger", aggfunc="first")
    wide.columns = [f"share_anger_r{r}" for r in wide.columns]
    A = agg.merge(wide.reset_index(), on=keys, how="left")
    s = A[A.model == "s"][["world", "T", "c", "f", "K", "default", "share_anger_mean"]].rename(columns={"share_anger_mean": "share_s"})
    A = A.merge(s, on=["world", "T", "c", "f", "K", "default"], how="left")
    A["diff_vs_s"] = 100 * (A.share_anger_mean - A.share_s)
    A["within_2pt_of_s"] = A.diff_vs_s.abs() <= 2
    return A


def compare_e83(Pp, path, chk):
    """results/e83/ に (C) の出力があれば，J_kl(ref__K)（model q0，既定 ref）と J_kl(other_<m>) が E83_TOL 以内で一致することを確かめる（無ければ飛ばす）．"""
    if not path.exists():
        chk.append({"world": "", "T": "", "c": "", "r": "", "check": "e83_compare", "status": "skipped", "detail": f"{path.relative_to(ROOT)} absent"})
        return None
    E = pd.read_csv(path, comment="#", dtype={"K": str})
    E = E[(E.n_pair >= E83_MIN_PAIR_N)]
    E = E[E.g.isin(["ref"]) | E.g.str.startswith("other_")]
    E = E.rename(columns={"target": "T", "pseudo": "c", "rep": "r", "student": "f", "J": "J_e83"})
    E["model"] = np.where(E.g == "ref", "q0", E.g)
    E["K"] = E.K.astype(int)
    M = Pp[(Pp.default == "ref") & (Pp.model.isin(E.model.unique())) & (Pp.n_pair >= E83_MIN_PAIR_N)]
    J = M.merge(E[["world", "T", "c", "r", "f", "K", "model", "pair", "J_e83"]], on=["world", "T", "c", "r", "f", "K", "model", "pair"], how="inner")
    if J.empty:
        chk.append({"world": "", "T": "", "c": "", "r": "", "check": "e83_compare", "status": "skipped", "detail": "no overlapping rows"})
        return None
    mx = float((J.J - J.J_e83).abs().max())
    chk.append({"world": "", "T": "", "c": "", "r": "", "check": "e83_compare", "status": "ok" if mx <= E83_TOL else "FAIL", "detail": f"n_rows={len(J)} max_abs_diff={mx:.3e}"})
    if mx > E83_TOL:
        raise SystemExit(f"E83 (C) の J と {mx:.3e} 食い違う（許容 {E83_TOL}）")
    return sha256(path)


# ---------------------------------------------------------------- 退化の検算

def selftest():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 4, n)
    logits = rng.normal(size=(n, 4)) + np.eye(4)[y] * 1.5
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    # (1) 比較の基準と同じ確率を候補に置くと J_kl = 0
    for p in range(6):
        m = (y == PAIRS[p][0]) | (y == PAIRS[p][1])
        r = pair_right(P[m], y[m], p)
        assert j_stat(r, r) == 0.0, "J(a, a) != 0"
        # J は 2 感情の正解率の差
        Q = rng.dirichlet(np.ones(4), m.sum())
        rq = pair_right(Q, y[m], p)
        assert abs(j_stat(rq, r) - 100 * (rq.mean() - r.mean())) < 1e-9, "J != acc difference"
    print("selftest 1: J(a, a) = 0 and J = 100*(acc_model - acc_a)  ok")
    # (2) D = d なら方向の誤り = 0
    Dv = rng.uniform(0.2, 3, 6)
    assert np.allclose(dir_err(Dv, Dv), 0), "dir_err(D, D) != 0"
    assert (dir_err(Dv + 0.5, Dv) > 0).all(), "dir_err not positive for D > d"
    # 既知の離れ方の人工データで base_D が真の値に近い（等方の散らばり，20 次元，群は 40）
    d_dim, n2 = 20, 4000
    y2 = rng.integers(0, 4, n2)
    mu = np.zeros((4, d_dim))
    mu[1, 0], mu[2, 1], mu[3, 2] = 2.0, 2.0, 2.0
    Z2 = rng.normal(size=(n2, d_dim)) + mu[y2]
    groups = np.array([f"g{i % 40}" for i in range(n2)])
    D2, _, _, note = base_D(Z2, y2, groups)
    truth = np.array([np.linalg.norm(mu[k] - mu[j]) for k, j in PAIRS])
    assert note == "ok" and np.all(np.abs(D2 / truth - 1) < 0.1), f"base_D off: {D2} vs {truth}"
    # 2 つの山の模型の d：対数確率の差が正規の 2 つの山なら d_kl は既知の値に近い
    L3 = rng.normal(size=(n2, 4))
    L3[np.arange(n2), y2] += 1.5
    d3 = pair_d(L3, y2)
    assert np.all(np.abs(d3 - 1.5 * 2 / np.sqrt(2)) < 0.15), f"pair_d off: {d3}"
    print("selftest 2: dir_err(D, D) = 0; base_D and pair_d recover known separations  ok")
    # (3)(4)(5) 階層の形：α = 0 は q⁰，α = 1 の h(怒) = s(怒)，行の和が 1，怒りを含まない対の判定は q⁰ と同じ
    s = rng.dirichlet(np.ones(4), n)
    q0 = rng.dirichlet(np.ones(4) * 0.3, n)
    q0[:5] = np.eye(4)[[3, 3, 0, 1, 2]]                                 # 端の行（確率 1 と 0）
    qc = np.clip(q0, EPS, None)
    qc = qc / qc.sum(1, keepdims=True)
    h0 = hier(s, q0, 0.0)
    assert np.abs(h0 - qc).max() < 1e-9, "alpha=0 != q0"
    for alpha in (0.0, 0.5, 1.0):
        h = hier(s, q0, alpha)
        assert np.abs(h.sum(1) - 1).max() < 1e-9, f"rows of h(alpha={alpha}) do not sum to 1"
        assert (h >= 0).all(), "negative probability in h"
        if alpha == 1.0:
            assert np.abs(h[:, ANGER] - s[:, ANGER]).max() < 1e-12, "alpha=1: h(anger) != s(anger)"
        for p, (k, j) in enumerate(PAIRS):
            if ANGER in (k, j):
                continue
            m = (y == k) | (y == j)
            assert np.array_equal(pair_right(h[m], y[m], p), pair_right(qc[m], y[m], p)), "non-anger pair decision changed by h"
    print("selftest 3-5: hier alpha=0 == q0, alpha=1 h(anger) == s(anger), rows sum to 1, non-anger pairs unchanged  ok")
    # (6) to_prob：対数と確率の両方で同じ確率に戻る．和がずれれば止まる
    Pl, is_log = to_prob(np.log(P).astype(np.float32), "selftest")
    Pp_, is_log2 = to_prob(P.astype(np.float32), "selftest")
    assert is_log and not is_log2 and np.abs(Pl - Pp_).max() < 1e-5, "to_prob log/prob mismatch"
    try:
        to_prob(P * 1.01, "selftest")
        raise AssertionError("to_prob did not stop on bad row sums")
    except SystemExit:
        pass
    print("selftest 6: to_prob handles log_softmax and probabilities; stops on bad sums  ok")
    # (7) ブートストラップ：モデルと q⁰ が同じなら Δ の全ての引きが 0．判定の 4 区分
    G, B = 12, 200
    gi = rng.integers(0, G, n)
    W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(np.float64)
    p = ANGER_PAIRS[0]
    m = (y == PAIRS[p][0]) | (y == PAIRS[p][1])
    r1 = pair_right(P[m], y[m], p)
    den = np.bincount(gi[m], minlength=G).astype(np.float64)
    num = np.bincount(gi[m], weights=r1.astype(np.float64) - r1.astype(np.float64), minlength=G)
    db = 100 * (W @ num) / (W @ den)
    assert np.all(db == 0), "bootstrap Delta != 0 for identical models"
    assert judge(1.0, 0.2, 1.8, 0.5) == "pos" and judge(-1.0, -1.8, -0.2, 0.5) == "neg" and judge(0.1, -0.3, 0.3, 0.5) == "same" \
        and judge(1.0, -0.2, 1.8, 0.5) == "undet" and judge(np.nan, 0, 1, 0.5) == "missing", "judge rules"
    print("selftest 7: bootstrap Delta = 0 for identical models; judge rules  ok")
    print("selftest: all passed")
    return 0


# ---------------------------------------------------------------- 本体

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e86")
    ap.add_argument("--worlds", nargs="*", default=list(WORLDS))
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS, help="反復 0〜reps−1（既定 5）")
    ap.add_argument("--boot", type=int, default=DEFAULT_BOOT, help="ブートストラップの回数（既定 1000）")
    ap.add_argument("--targets", nargs="*", default=None, help="評価のターゲット T を絞る（試走と検査のため．D13）")
    ap.add_argument("--smoke", action="store_true", help="試走（avt の 1 ターゲット，反復 1，ブートストラップ 50．出力は --out か checks/E86/code_run/smoke）")
    ap.add_argument("--selftest", action="store_true", help="人工データでの退化の検算（設計 3 節）")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.smoke:
        args.worlds, args.reps, args.boot = ["avt"], 1, SMOKE_BOOT
        if args.out == "results/e86":
            args.out = "checks/E86/code_run/smoke"
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    partial = args.smoke or bool(args.targets) or args.reps < DEFAULT_REPS or set(args.worlds) != set(WORLDS)
    if partial and (ROOT / "results") in out.resolve().parents:
        raise SystemExit("部分的な流し方（--smoke，--targets，--reps < 5，--worlds の一部）の出力は results/ の下に書けない．--out を checks/ か scratchpad にする")
    if out.exists() and any(out.glob("*.csv")):
        raise SystemExit(f"{out} に古い出力がある．消してから流す")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), out / "script_copy.py")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty_txt = subprocess.run(["git", "status", "--porcelain", "--", *SCRIPT_DEPS], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    global HEADER
    HEADER = f"E86 commit={commit} dirty={bool(dirty_txt)} worlds={args.worlds} reps={args.reps} boot={args.boot} targets={args.targets} smoke={args.smoke}"
    t0 = time.time()
    prov = {"experiment": "E86", "commit": commit, "dirty": bool(dirty_txt), "dirty_files": dirty_txt.splitlines(), "dirty_scope": list(SCRIPT_DEPS),
            "script_sha256": sha256(Path(__file__)), "args": vars(args) | {"out": str(out)}, "constants": CONSTANTS, "inputs": {},
            "started": dt.datetime.now().isoformat(timespec="seconds"),
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__}}
    rows_p, rows_b, rows_a, rows_i, chk, timing = [], [], [], [], [], []
    for world in args.worlds:
        fused = WORLDS[world]
        udir = ROOT / "results" / "e80" / world / "units" / fused
        if not udir.exists():
            raise SystemExit(f"{udir} が無い")
        Ts = sorted(p.name for p in udir.iterdir() if p.is_dir())
        if args.targets:
            Ts = [T for T in Ts if T in args.targets]
        if args.smoke:
            Ts = Ts[:1]
        feats, samples = {}, {}
        for T in Ts:
            cs = sorted({f.stem.rsplit("_r", 1)[0] for f in (udir / T).glob("*_r*.npz") if not f.name.endswith("_agg.npz")})
            for c in cs:
                if c not in feats:
                    feats[c] = feats_for(c, fused)
                    sp_path = udir / f"samples_{c}.csv"
                    sp = pd.read_csv(sp_path)
                    if not np.array_equal(sp.sample_id.astype(str).to_numpy(), feats[c][0].sample_id.astype(str).to_numpy()) or not np.array_equal(sp.row.to_numpy(), np.arange(len(sp))):
                        raise SystemExit(f"{world} {c}：samples の sample_id が発話一覧と合わない")
                    chk.append({"world": world, "T": "", "c": c, "r": "", "check": "samples_sample_id_vs_manifest", "status": "ok", "detail": f"n={len(sp)}"})
                    samples[c] = sp
                    prov["inputs"][str(sp_path.relative_to(ROOT))] = sha256(sp_path)
                rng = np.random.default_rng([BOOT_SEED, zlib.crc32(world.encode()), zlib.crc32(T.encode()), zlib.crc32(c.encode())])
                boot = {}
                for r in range(args.reps):
                    fpath = udir / T / f"{c}_r{r}.npz"
                    if not fpath.exists():
                        raise SystemExit(f"{fpath} が無い（反復がそろっていない）")
                    tu = time.time()
                    prov["inputs"][str(fpath.relative_to(ROOT))] = sha256(fpath)
                    u = Unit(world, T, c, r, feats[c], samples[c], chk)
                    W = rng.multinomial(u.n_groups, np.full(u.n_groups, 1 / u.n_groups), size=args.boot).astype(np.float64)
                    process_unit(u, args.boot, W, rows_p, rows_b, rows_a, boot, chk)
                    timing.append(dict(world=world, T=T, c=c, r=r, n_cand=len(u.cand), n_groups=u.n_groups, seconds=round(time.time() - tu, 2)))
                    print(world, T, c, r, f"n_cand={len(u.cand)} groups={u.n_groups} {time.time() - tu:.1f}s", flush=True)
                finalize_boot(world, T, c, boot, rows_i)
    Pp, Bb, Aa, Ii = pd.DataFrame(rows_p), pd.DataFrame(rows_b), pd.DataFrame(rows_a), pd.DataFrame(rows_i)
    if Pp.empty:
        raise SystemExit("単位ファイルが 1 つも無い")
    for name, df in (("pairs", Pp), ("base", Bb)):
        bad = df[["J", "Delta"] if name == "pairs" else ["D", "d", "dir_err"]]
        if np.isinf(bad.to_numpy(dtype=float)).any():
            raise SystemExit(f"{name}：Inf がある")
    e83_sha = compare_e83(Pp, ROOT / "results" / "e83" / "c_complementarity_units.csv", chk)
    if e83_sha:
        prov["inputs"]["results/e83/c_complementarity_units.csv"] = e83_sha
    Dd = delta_table(Pp, args.reps)
    U = units_table(Pp, Bb, Ii, Dd, args.reps)
    L = long_judgments(U)
    write_csv(Pp, out / "pairs.csv")
    write_csv(Bb, out / "base.csv")
    write_csv(U, out / "units.csv")
    write_csv(Dd, out / "delta.csv")
    write_csv(counts_table(L), out / "counts.csv")
    write_csv(corpus_table(L), out / "corpus_counts.csv")
    write_csv(anger_table(Aa, args.reps), out / "anger_share.csv")
    write_csv(hier_table(L), out / "hier_counts.csv")
    write_csv(pd.DataFrame(chk), out / "checks.csv")
    prov["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    prov["elapsed_seconds"] = round(time.time() - t0, 1)
    prov["timing_per_unit_file"] = timing
    prov["n_rows"] = {"pairs": len(Pp), "base": len(Bb), "units": len(U), "anger_share_reps": len(Aa), "checks": len(chk)}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("wrote", out, f"pairs={len(Pp)} base={len(Bb)} units={len(U)} elapsed={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
