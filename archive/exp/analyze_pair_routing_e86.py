"""E86：怒りの確率の出どころを替える振り分けの余地——感情の対ごとの補いを，仮のターゲットの候補の上で，自分のモダリティを除いたモデルについて数える（下調べ）．

設計の正本：docs/exploratory-analyses.md の「## E86．…」の節（1〜5 節．規則は値を見る前に書いた．関門 1 の直しを入れた版）．
下書きは research_notes/研究提案_2026-11-04/段0a_設計案.md（c60746e 時点の写し）．背景は同じフォルダの 案_2_改.md．
下調べ（仮のターゲットの候補だけ．評価のターゲット T は置き場の名前として使うだけで，T のテストの値もラベルも読まない）．

入力（すべて保存済み．場所は --e80・--e83 で渡す）
  - E80 の単位ファイル <e80>/<world>/units/<fused>/<T>/<c>_r<r>.npz（候補の行番号 cand，テストの行番号 te，ラベル y・yte，K 本の位置 lab__K，
    候補の上の確率 P__<kind>（ソースだけの出発点．P__<fused> は融合のモデル），ref__K（学び直して掛けたモデル），none__<f>__K（K 本で学び直した配備する分類器＝比較の基準．候補の上では確率））
  - samples_<c>.csv（行番号 → sample_id・speaker_group）．区間のブートストラップと jackknife の話者群に使う（両世界の samples を読み，コーパスごとの話者群の一覧を作る）
  - 特徴：measure/data.load_any(c, <fused>) の行を cand で引き，配備する分類器 f のブロック（kind の順に連結．音声 1024・顔 512・言葉 768）だけを使う（土台の D にだけ使う）
新しい学習：土台の D を決める Ledoit–Wolf の LDA（exp/explore_pairs.own_separation．話者群で 2 分）だけ．

モデル（候補の発話ごとの 4 感情の確率．0 平静・1 喜び・2 悲しみ・3 怒り）
  比較の基準 a = none__<f>__K．既定 q⁰：主 ref = ref__K，副 pF = P__<fused>．候補 s = f 以外の P__<m> の要素ごとの積を正規化（av は残り 1 つそのもの）．
  記述 other_<m>（avt だけ）．対照 own = P__<f>（自分のモダリティのソースだけの出発点）．
  階層の形 h(s, α)：h(怒) = α·s(怒) + (1−α)·q⁰(怒)，怒り以外 c：h(c) = (1−h(怒))·q⁰(c)/(1−q⁰(怒))．α ∈ {1 (h1), 0.5 (h05)}．対照 h_own1 = h(own, 1)．
量：J_kl（E83 (C) と同じ 2 感情の判定の補い），Δ_kl = J_kl(モデル) − J_kl(q⁰)（2 感情の正解率の差），Δ^AUC_kl = 100·[AUC_kl(モデル) − AUC_kl(q⁰)]（対数オッズ L_k − L_l の 2 感情の AUC．切片に依らない），
  δ_J，区間（コーパスの話者群の復元抽出．実効的な群の数が 10 未満の単位は jackknife の t 区間との和集合），判定（pos/neg/same/undet），土台（D_kl，d_kl，d_AUC，方向の誤り），
  数え上げと帰無の 95% 点（中心を 0 に寄せた複製から），コーパスの塊・(T, c, 対) の塊・T ごとの表，既定の主と副の比べ，感情を要にした割合，怒りと答える割合と 4 クラスの質，階層の切り分け．

出力（--out．既定 results/e86）：pairs.csv（反復ごと），base.csv（反復ごとの土台），units.csv（単位 × 対 × 候補 × 既定），delta.csv，counts.csv，corpus_counts.csv，target_counts.csv，
  default_compare.csv，pivot_counts.csv，assumptions.csv，anger_share.csv（反復ごと．4 クラスの質つき），anger_share_units.csv，hier_counts.csv，checks.csv，
  boot_replicates.npz + boot_index.csv（セルごとの複製），jackknife.npz，group_counts.npz（話者群ごとの当たりの数），provenance.json，script_copy.py，code/（依存のモジュールの写し）．
  各 CSV の先頭に `# E86 commit=... script_sha256=... seed=...` の行（読むときは comment="#"）．

使い方：./.venv/bin/python exp/analyze_pair_routing_e86.py [--out results/e86] [--e80 results/e80] [--e83 results/e83] [--worlds avt av] [--reps 5] [--boot 1000] [--targets T ...] [--smoke] [--selftest]
  --smoke：avt の 1 ターゲット・反復 1 つ・ブートストラップ 50 回．出力は --out か checks/E86/code_run/smoke．  --selftest：人工データでの退化の検算（設計 3 節）．
  部分的な流し方（--smoke，--targets，--reps < 5，--worlds の一部）の出力は results/ の下に書けない．出力先に古い CSV があれば止まる．

設計に無い決め事（設計 2 節に書いたもの以外の，台本で決めた細部）
  D1 対数で保存された確率の見分け方：配列の最小値が負なら対数とみなして指数を取る．行の和の検査（10⁻⁴）は指数を取った後の和に当てる．d_kl と AUC に使う対数確率は，対数で保存された配列はそのまま，
     確率の配列は log(max(p, 1e−12))．
  D3 2 感情の判定の同点（k と l の確率が等しい）は番号の小さい k とする（E83 (C) の `>=` と同じ）．AUC の同点は ½（E79 の rankdata と同じ）．
  D7 δ_J の標準偏差は ddof = 1．反復が 2 未満の (T, c, 対) は δ_J の材料から除く（試走の反復 1 つでは δ_J が無く判定は missing）．÷√5 は ÷√(反復の数)（本番は 5）．
  D8 判定の呼び方は E79・E83 と同じ pos（上回った）・neg（下回った）・same（同じ水準）・undet（判定できない）．点・区間・δ_J のどれかが無ければ missing．
  D10 怒りと答える割合と 4 クラスの質は反復ごとの行（anger_share.csv）と単位の表（anger_share_units.csv．反復の平均）の両方に出す．
  D11 候補のモデル s の積が全て 0 の行があれば止まる．
  D13 --targets（T を絞る）と --boot（回数）と --e80・--e83（場所）を引数にした．
  D14 BLAS のスレッド数を 4 に固定する（環境変数が既にあればそれに従う）．既定の 16 スレッドでは 1024×1024 の固有値分解と solve がスレッドの取り合いで 8〜80 倍遅く，値は同一であることを確かめた．
  D15 ブートストラップと jackknife の重みは，コーパスの話者群の一覧（両世界の samples_<c>.csv の和集合．IEMOCAP は即興と台本で同じ役者名）の順に引く．復元抽出で分母が 0 になった引きは NaN とし，
     反復の平均と百分位から除く（n_boot_valid）．jackknife で分母が 0 になる抜き方は NaN とし，SE から除く．
  D16 G_eff は反復 0〜4 の候補（K 本を除く前）を話者群ごとに足した本数から (Σn)²/Σn² で出す（E79 と同じ）．jackknife の t の自由度は，話者群なら max(1, ⌊G_eff⌋ − 1)，セッションならセッションの数 − 1．
  D17 AUC の区間と判定にも同じ δ_J を当てる（AUC も 2 感情の正解率と同じ 0〜100 の尺度．記述の量）．
  D18 帰無の数の 95% 点は，複製の数（B 本）の 95 パーセンタイル（np.percentile の線形補間）．帰無の平均も出す．
  D19 本番（部分でない流し）では，(T, c) の組が avt 8・av 24 と一致しなければ止まる．
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
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, fused_parts, load_any, load_group  # noqa: E402
from explore_pairs import PAIRS, ba, own_separation  # noqa: E402
from split import speaker_folds  # noqa: E402

WORLDS = {"avt": "fuse-audio-video-text", "av": "fuse-audio-video"}
EXPECTED_PAIRS = {"avt": 8, "av": 24}                            # (T, c) の組の数（設計 2 節．D19）
KS = (0, 5, 25)
EMO = ("平静", "喜び", "悲しみ", "怒り")
ANGER = 3
PAIR_NAME = [f"{EMO[k]}–{EMO[j]}" for k, j in PAIRS]
ANGER_PAIRS = tuple(p for p, (k, j) in enumerate(PAIRS) if ANGER in (k, j))
DEFAULTS = ("ref", "pF")                                   # 既定 q⁰：主 = ref__K，副 = P__<fused>
MAIN_DEFAULT = "ref"
HIER = ((1.0, "h1"), (0.5, "h05"))                         # 階層の形 h(s, α)
MODEL_ROLE = {"q0": "default", "s": "candidate", "own": "control", "h1": "hier", "h05": "hier", "h_own1": "control"}   # other_<m> は description
CHANCE = 0.025                                             # 1 セルあたりの偶然の見込み（下限の参照だけ．判定は帰無の 95% 点）
NULL_PCT = 95
DELTA_SCALES = ((1.0, "main"), (0.5, "half"), (2.0, "double"))
BASE_RULES = ("fixed", "scaled", "auc")                    # 土台の規則：fixed = 主の δ_base で固定，scaled = 尺度と一緒に変える（感度），auc = d_AUC の方向の誤りで固定（感度）
PCT = (2.5, 97.5)
MIN_GEFF = 10                                              # 実効的な話者群の数がこれ未満の単位は jackknife の t 区間との和集合（設計 2 節）
SESSION_CORPORA = ("IEMOCAP",)                             # jackknife の単位をセッション（話者群の先頭 5 文字 Ses0x）にするコーパス
SUM_TOL = 1e-4
EPS = 1e-12
E83_TOL = 1e-6
E83_MIN_PAIR_N = 10
BOOT_SEED = 86
DEFAULT_REPS = 5
DEFAULT_BOOT = 1000
SMOKE_BOOT = 50
AUC_CHUNK = 250                                            # 重みつき AUC を一度に計算する行の数（メモリ）
NEG_SHARE_LIMIT = 0.5                                      # (世界, f) で方向の誤りが負のセルがこの割合を超えたら R86-1 を全セルでも読む（設計 4 節の仮定の表）
SCRIPT_DEPS = ("exp/analyze_pair_routing_e86.py", "exp/explore_pairs.py", "exp/explore_geometry.py", "exp/explore_mechanism.py", "exp/explore_factor.py", "exp/pair_table.py",
               "exp/summarize_map.py", "measure/data.py", "measure/split.py", "features/common.py", "data/metadata")
CODE_COPY = ("exp/explore_pairs.py", "measure/split.py", "measure/data.py", "features/common.py")
CONSTANTS = dict(KS=KS, DEFAULTS=DEFAULTS, MAIN_DEFAULT=MAIN_DEFAULT, HIER=HIER, CHANCE=CHANCE, NULL_PCT=NULL_PCT, DELTA_SCALES=DELTA_SCALES, BASE_RULES=BASE_RULES, PCT=PCT,
                 MIN_GEFF=MIN_GEFF, SESSION_CORPORA=SESSION_CORPORA, SUM_TOL=SUM_TOL, EPS=EPS, E83_TOL=E83_TOL, E83_MIN_PAIR_N=E83_MIN_PAIR_N, BOOT_SEED=BOOT_SEED,
                 ANGER=ANGER, PAIRS=PAIRS, ANGER_PAIRS=ANGER_PAIRS, NEG_SHARE_LIMIT=NEG_SHARE_LIMIT,
                 judge="pos: lo>0 & x>=dl / neg: hi<0 & x<=-dl / same: lo>=-dl & hi<=dl / undet: else / missing: NaN",
                 delta_J="2 * median_{T,c,pair}( sd_{reps}(J_kl(q0)) ) / sqrt(n_reps)  per (world, f, K, default)",
                 delta_base="delta_J of default=ref (per world, f, K); base: dir_err >= delta_base (fixed) / >= scale*delta_base (scaled) / dir_err_auc >= delta_base (auc)",
                 null="per draw i: (b_i - x) + (lo - x) > 0 and (b_i - x) >= dl  -> count over cells; 95th percentile and mean",
                 interval="percentile of corpus-speaker-group bootstrap; union with jackknife t interval when G_eff < MIN_GEFF (IEMOCAP: sessions)")
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


def session_of(group):
    return str(group)[:5]


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
    """d_kl と AUC に使う対数確率（D1）．"""
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
    """階層の形 h(s, α)（設計 2 節の表）．q⁰ は EPS で切って正規化し，怒り以外の比の分母は 3 つの和で計算する（設計 2 節）．"""
    q = np.clip(q0, EPS, None)
    q = q / q.sum(1, keepdims=True)
    h = np.empty_like(q)
    h[:, ANGER] = alpha * s[:, ANGER] + (1 - alpha) * q[:, ANGER]
    rest = [c for c in range(4) if c != ANGER]
    ratio = q[:, rest] / q[:, rest].sum(1, keepdims=True)
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
    """比較の基準の離れ方 d_kl（E13・E83 の d_o_pairs と同じ式．z = L_k − L_l）と分散の比 v_k/v_l．戻り (d (6,), var_ratio (6,))．"""
    d, vr = np.full(6, np.nan), np.full(6, np.nan)
    for p, (k, j) in enumerate(PAIRS):
        zk, zj = L[y == k, k] - L[y == k, j], L[y == j, k] - L[y == j, j]
        if len(zk) < 2 or len(zj) < 2:
            continue
        vk, vj = zk.var(ddof=1), zj.var(ddof=1)
        sd = np.sqrt((vk + vj) / 2)
        if sd > 0:
            d[p] = (zk.mean() - zj.mean()) / sd
        if vj > 0:
            vr[p] = vk / vj
    return d, vr


def pair_auc(z, isk):
    """2 感情の AUC（E79 の pair_moments と同じ：rankdata の平均の同点）．z：対の発話の対数オッズ L_k − L_l，isk：正解が k．"""
    n1, n2 = int(isk.sum()), int((~isk).sum())
    if n1 == 0 or n2 == 0:
        return np.nan
    rk = stats.rankdata(np.concatenate([z[isk], z[~isk]]))[:n1]
    return float((rk.sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def weighted_auc(z, isk, Wn):
    """重みつきの 2 感情の AUC（復元抽出の回数を重みにした標本の AUC．同点は ½）．z (n,)，isk (n,) bool，Wn (R, n)．戻り (R,)．分母が 0 の行は NaN．"""
    order = np.argsort(z, kind="stable")
    zs, ks = z[order], isk[order]
    _, start = np.unique(zs, return_index=True)
    kf = ks.astype(np.float32)
    out = np.empty(Wn.shape[0])
    for a in range(0, Wn.shape[0], AUC_CHUNK):
        Ws = Wn[a:a + AUC_CHUNK][:, order].astype(np.float32)           # 重みは整数（復元抽出の回数か 0/1）なので float32 でも塊の和は正確
        seg_k = np.add.reduceat(Ws * kf, start, axis=1)                  # 同点の塊ごとの k の重み
        seg_l = np.add.reduceat(Ws, start, axis=1) - seg_k               # 同点の塊ごとの l の重み
        F = np.cumsum(seg_l, axis=1) - 0.5 * seg_l                       # より小さい l の重み ＋ ½ 同点の重み
        num = np.sum(seg_k * F, axis=1, dtype=np.float64)
        den = np.sum(seg_k, axis=1, dtype=np.float64) * np.sum(seg_l, axis=1, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[a:a + AUC_CHUNK] = np.where(den > 0, num / den, np.nan)
    return out


def dir_err(D, d):
    """方向の誤り = 100·[Φ(D/2) − Φ(d/2)]（2 つの山の模型．切片を最良に直した 2 感情の正解率の差）．"""
    D, d = np.asarray(D, dtype=np.float64), np.asarray(d, dtype=np.float64)
    return 100 * (ba(D, 0.0) - ba(d, 0.0))


def d_from_auc(auc):
    return np.sqrt(2) * stats.norm.ppf(np.clip(auc, 1e-12, 1 - 1e-12))


def base_D(Z, y, groups):
    """最良の方向での離れ方 D_kl：話者群で 2 分（speaker_folds k=2 seed=0）し，explore_pairs.own_separation（Ledoit–Wolf の LDA）で測る（設計 2 節）．
    戻り (D (6,), 半分ごとの本数, 半分ごとの話者群の数, 半分が分けたセッションの数, 検算の記録)．"""
    half = speaker_folds(pd.Series(groups), k=2, seed=0).to_numpy()
    n_half = [int((half == a).sum()) for a in (0, 1)]
    g_half = [int(len(np.unique(groups[half == a]))) for a in (0, 1)]
    sess = pd.DataFrame({"s": [session_of(g) for g in groups], "h": half}).groupby("s").h.nunique()
    n_sess_split = int((sess > 1).sum())
    ok = all(((y[half == a] == k).sum() >= 2) for a in (0, 1) for k in range(4))
    if not ok:
        return np.full(6, np.nan), n_half, g_half, n_sess_split, "D skipped: some emotion < 2 in a half"
    return own_separation(Z, y, half), n_half, g_half, n_sess_split, "ok"


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


def jk_interval(theta_hat, th, df):
    """E79 と同じ jackknife の t 区間：SE_jk = √((G−1)/G · Σ(θ_(g) − θ̄)²)，自由度 df．"""
    th = th[np.isfinite(th)]
    G = len(th)
    if G < 2 or not np.isfinite(theta_hat):
        return np.nan, np.nan
    se = np.sqrt((G - 1) / G * ((th - th.mean()) ** 2).sum())
    t = stats.t.ppf(0.975, max(1, int(df)))
    return theta_hat - t * se, theta_hat + t * se


def quality(P, y):
    """4 クラスの質：感情ごとの再現率，UAR，怒りの適合率，怒りと答える割合．"""
    am = P.argmax(1)
    rec = [float((am[y == e] == e).mean()) if (y == e).any() else np.nan for e in range(4)]
    pa = am == ANGER
    prec = float((y[pa] == ANGER).mean()) if pa.any() else np.nan
    return rec, float(np.nanmean(rec)), prec, float(pa.mean())


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
    def __init__(self, world, T, c, r, e80, feats, samples, chk):
        fused = WORLDS[world]
        self.world, self.T, self.c, self.r, self.fused = world, T, c, r, fused
        self.path = e80 / world / "units" / fused / T / f"{c}_r{r}.npz"
        self.z = np.load(self.path)
        self.cand, self.te = np.asarray(self.z["cand"]).astype(int), np.asarray(self.z["te"]).astype(int)
        y_npz, yte_npz = np.asarray(self.z["y"]).astype(int), np.asarray(self.z["yte"]).astype(int)
        df, X, blocks = feats
        tag = dict(world=world, T=T, c=c, r=r)
        if len(self.cand) != len(y_npz) or len(self.te) != len(yte_npz) or int(max(self.cand.max(), self.te.max())) >= len(df) or min(self.cand.min(), self.te.min()) < 0:
            raise SystemExit(f"{world} {T} {c} r{r}：候補・テストの行番号が発話一覧と合わない")
        lab = df.label_4.to_numpy().astype(int)
        if not np.array_equal(lab[self.cand], y_npz) or not np.array_equal(lab[self.te], yte_npz):
            raise SystemExit(f"{world} {T} {c} r{r}：ラベルが発話一覧と合わない")
        self.y, self.yte = lab[self.cand], lab[self.te]                  # 作業用のラベルは発話一覧から貼り直したもの（CLAUDE.md 1）
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
        self.P, self.is_log = {}, {}
        for key in [f"P__{m}" for m in self.mods] + [f"P__{fused}"] + [f"ref__{K}" for K in KS] + [f"none__{m}__{K}" for m in self.mods for K in KS]:
            self.P[key], self.is_log[key] = to_prob(self.z[key], f"{world} {T} {c} r{r} {key}")
            if self.P[key].shape[0] != len(self.cand):
                raise SystemExit(f"{world} {T} {c} r{r} {key}：行数が候補と合わない")
        chk.append({**tag, "check": "prob_rows_sum_to_one", "status": "ok", "detail": "log:" + ",".join(k for k, v in self.is_log.items() if v)})

    def feat(self, f, rows):
        lo, hi = self.blocks[f]
        return self.X[rows, lo:hi]


class UnitAcc:
    """(世界, T, c) の反復をまとめる入れ物：ブートストラップと jackknife の複製の和，反復ごとの点．"""

    def __init__(self, world, T, c, ug, cg, W_corpus, geff, need_jk, jk_kind):
        self.world, self.T, self.c, self.ug, self.geff, self.need_jk, self.jk_kind = world, T, c, ug, geff, need_jk, jk_kind
        pos = {g: i for i, g in enumerate(cg)}
        idx = np.array([pos[g] for g in ug])
        self.W = W_corpus[:, idx]                                         # (B, G_u)．コーパスの話者群の重みをこの単位の群に写す（D15）
        if jk_kind == "session":
            units = sorted({session_of(g) for g in ug})
            self.jk_names = units
            self.JW = np.array([[0.0 if session_of(g) == s else 1.0 for g in ug] for s in units])
        else:
            self.jk_names = list(ug)
            self.JW = 1.0 - np.eye(len(ug))
        self.jk_df = (len(self.jk_names) - 1) if jk_kind == "session" else max(1, int(np.floor(geff)) - 1)
        self.cells = {}                                                   # key -> dict(boot=sum (B,), boot_auc, jk=sum (G_jk,), jk_auc, pts=[], pts_auc=[], n)

    def add(self, key, db, dauc_b, jk, jk_auc, pt, pt_auc):
        c = self.cells.setdefault(key, {"boot": np.zeros(self.W.shape[0]), "boot_auc": np.zeros(self.W.shape[0]),
                                        "jk": np.zeros(self.JW.shape[0]), "jk_auc": np.zeros(self.JW.shape[0]), "pts": [], "pts_auc": [], "n": 0})
        c["boot"] += db
        c["boot_auc"] += dauc_b
        if jk is not None:
            c["jk"] += jk
            c["jk_auc"] += jk_auc
        c["pts"].append(pt)
        c["pts_auc"].append(pt_auc)
        c["n"] += 1


def process_unit(u, acc, rows_p, rows_b, rows_a, gc_store, chk):
    """1 つの単位ファイルで，f × K × 既定 × モデル × 対の J・Δ・AUC（反復の行），土台，4 クラスの質，ブートストラップと jackknife の材料を出す．"""
    fused = u.fused
    gpos = {g: i for i, g in enumerate(acc.ug)}
    gi_all = np.array([gpos[g] for g in u.groups])
    G = len(acc.ug)
    W, JW = acc.W, (acc.JW if acc.need_jk else None)
    tag0 = dict(world=u.world, T=u.T, c=u.c, unit=corp(u.c), r=u.r)
    for f in u.mods:
        for K in KS:
            keep = np.ones(len(u.cand), bool)
            keep[u.labs[K]] = False                                      # K 本を候補から除く（E83 (C) と同じ）
            yk, gi = u.y[keep], gi_all[keep]
            n_kept, n_groups = int(keep.sum()), int(len(np.unique(gi)))
            A = u.P[f"none__{f}__{K}"][keep]                             # 比較の基準 a
            LA = to_logp(u.z[f"none__{f}__{K}"])[keep]
            others = [m for m in u.mods if m != f]
            s = product_model([u.P[f"P__{m}"][keep] for m in others])
            own = u.P[f"P__{f}"][keep]
            free_models = {"s": s, "own": own}
            if len(others) >= 2:                                         # 記述：残りの単モダリティ（avt だけ）
                for m in others:
                    free_models[f"other_{m}"] = u.P[f"P__{m}"][keep]
            masks = [(yk == k) | (yk == j) for k, j in PAIRS]
            a_right = [pair_right(A[m], yk[m], p) for p, m in enumerate(masks)]
            tag = {**tag0, "f": f, "K": K, "n_cand": len(u.cand), "n_cand_kept": n_kept, "n_groups": n_groups}
            # 話者群ごとの当たりの数（保存．区間の作り直しに使える）
            den_g = np.stack([np.bincount(gi[m], minlength=G) for m in masks]).astype(np.int32)
            a_g = np.stack([np.bincount(gi[m], weights=a_right[p].astype(float), minlength=G) for p, m in enumerate(masks)]).astype(np.int32)
            gc_key = f"{u.world}|{u.T}|{u.c}|r{u.r}|{f}|K{K}"
            gc_store[gc_key + "|den"], gc_store[gc_key + "|a_right"] = den_g, a_g
            right_entries, right_arrays = [], []
            # AUC の材料：既定に依らないモデルの z と AUC の複製は 1 回だけ
            free_auc = {}
            for name, P in free_models.items():
                LP = np.log(np.clip(P, EPS, None))
                free_auc[name] = []
                for p, (k, j) in enumerate(PAIRS):
                    m = masks[p]
                    z, isk = LP[m, k] - LP[m, j], yk[m] == k
                    free_auc[name].append(auc_bundle(z, isk, W[:, gi[m]], JW[:, gi[m]] if JW is not None else None))
            rec_a = quality(A, yk)
            for dflt in DEFAULTS:
                q0 = (u.P[f"ref__{K}"] if dflt == "ref" else u.P[f"P__{fused}"])[keep]
                models = {"q0": q0, **free_models}
                for alpha, name in HIER:
                    models[name] = hier(s, q0, alpha)
                models["h_own1"] = hier(own, q0, 1.0)
                right = {name: [pair_right(P[m], yk[m], p) for p, m in enumerate(masks)] for name, P in models.items()}
                J = {name: [j_stat(right[name][p], a_right[p]) if masks[p].any() else np.nan for p in range(6)] for name in models}
                auc = {}
                for name, P in models.items():
                    if name in free_auc:
                        auc[name] = free_auc[name]
                        continue
                    LP = np.log(np.clip(P, EPS, None))
                    auc[name] = []
                    for p, (k, j) in enumerate(PAIRS):
                        m = masks[p]
                        z, isk = LP[m, k] - LP[m, j], yk[m] == k
                        auc[name].append(auc_bundle(z, isk, W[:, gi[m]], JW[:, gi[m]] if JW is not None else None))
                # 4 クラスの質（比較の基準 a も並べる）
                for name, P in [("a", A)] + list(models.items()):
                    rec, uar, prec, share = quality(P, yk) if name != "a" else rec_a
                    rows_a.append({**tag, "default": dflt, "model": name, "share_anger": share, "share_anger_true": float((yk == ANGER).mean()),
                                   "recall_neutral": rec[0], "recall_happy": rec[1], "recall_sad": rec[2], "recall_angry": rec[3], "uar": uar, "precision_anger": prec})
                for name, P in models.items():
                    if not (name in free_models and dflt != DEFAULTS[0]):   # 既定に依らないモデルの当たりの数は最初の既定でだけ保存する
                        right_entries.append(f"{dflt if name not in free_models else '-'}|{name}")
                        right_arrays.append(np.stack([np.bincount(gi[m], weights=right[name][p].astype(float), minlength=G) for p, m in enumerate(masks)]).astype(np.int32))
                    for p in range(6):
                        m = masks[p]
                        pt_auc = 100 * (auc[name][p]["pt"] - auc["q0"][p]["pt"]) if m.any() else np.nan
                        rows_p.append({**tag, "default": dflt, "model": name, "role": MODEL_ROLE.get(name, "description"), "pair": p, "pair_name": PAIR_NAME[p],
                                       "anger_pair": p in ANGER_PAIRS, "n_pair": int(m.sum()), "acc_a": float(a_right[p].mean()) if m.any() else np.nan,
                                       "acc_model": float(right[name][p].mean()) if m.any() else np.nan, "J": J[name][p], "J_q0": J["q0"][p], "Delta": J[name][p] - J["q0"][p],
                                       "auc_model": auc[name][p]["pt"], "auc_q0": auc["q0"][p]["pt"], "Delta_auc": pt_auc})
                        if name == "q0" or not m.any():
                            continue
                        den = den_g[p].astype(np.float64)
                        num = np.bincount(gi[m], weights=right[name][p].astype(np.float64) - right["q0"][p].astype(np.float64), minlength=G)
                        wd = W @ den
                        with np.errstate(divide="ignore", invalid="ignore"):
                            db = np.where(wd > 0, 100 * (W @ num) / wd, np.nan)
                        dauc_b = 100 * (auc[name][p]["boot"] - auc["q0"][p]["boot"])
                        jk = jk_auc = None
                        if JW is not None:
                            jd = JW @ den
                            with np.errstate(divide="ignore", invalid="ignore"):
                                jk = np.where(jd > 0, 100 * (JW @ num) / jd, np.nan)
                            jk_auc = 100 * (auc[name][p]["jk"] - auc["q0"][p]["jk"])
                        acc.add((f, K, dflt, name, p), db, dauc_b, jk, jk_auc, J[name][p] - J["q0"][p], pt_auc)
            gc_store[gc_key + "|right"] = np.stack(right_arrays)
            gc_store[gc_key + "|right_entries"] = np.array(right_entries)
            # 土台：D（特徴の LDA），d と d_AUC（比較の基準の対数確率），分散の比，半分の本数 ÷ 次元
            Z = u.feat(f, u.cand[keep])
            mu, sd = Z.mean(0), Z.std(0) + 1e-8
            D, n_half, g_half, n_sess_split, note = base_D((Z - mu) / sd, yk, u.groups[keep])
            if note != "ok":
                chk.append({**tag0, "f": f, "K": K, "check": "base_D", "status": "skipped", "detail": note})
            d, vr = pair_d(LA, yk)
            auc_a = np.array([pair_auc(LA[m, k] - LA[m, j], yk[m] == k) if m.any() else np.nan for (k, j), m in zip(PAIRS, masks)])
            d_auc = d_from_auc(auc_a)
            de, de_auc = dir_err(D, d), dir_err(D, d_auc)
            for p in range(6):
                rows_b.append({**tag, "pair": p, "pair_name": PAIR_NAME[p], "anger_pair": p in ANGER_PAIRS, "n_pair": int(masks[p].sum()),
                               "n_half0": n_half[0], "n_half1": n_half[1], "n_groups_half0": g_half[0], "n_groups_half1": g_half[1], "n_sessions_split": n_sess_split,
                               "dim": Z.shape[1], "n_half_min_over_dim": min(n_half) / Z.shape[1],
                               "D": float(D[p]), "d": float(d[p]), "auc_a": float(auc_a[p]), "d_auc": float(d_auc[p]), "var_ratio": float(vr[p]),
                               "dir_err": float(de[p]), "dir_err_auc": float(de_auc[p]), "model_err": float(100 * (ba(d[p], 0.0) - ba(d_auc[p], 0.0)))})


def auc_bundle(z, isk, Wn, JWn):
    """点の AUC と，ブートストラップの重み（と jackknife の重み）での AUC．"""
    if len(z) == 0 or not isk.any() or isk.all():
        B = Wn.shape[0]
        return {"pt": np.nan, "boot": np.full(B, np.nan), "jk": (np.full(JWn.shape[0], np.nan) if JWn is not None else None)}
    return {"pt": pair_auc(z, isk), "boot": weighted_auc(z, isk, Wn), "jk": (weighted_auc(z, isk, JWn) if JWn is not None else None)}


def finalize_unit(acc, rows_i, rep_store, jk_store, gc_store):
    """(世界, T, c) の区間：百分位（反復の平均の複製），G_eff < MIN_GEFF なら jackknife の t 区間との和集合．複製と jackknife の値を保存する．"""
    gc_store[f"{acc.world}|{acc.T}|{acc.c}|groups"] = np.array(acc.ug)
    if acc.need_jk:
        jk_store[f"{acc.world}|{acc.T}|{acc.c}|jk_names"] = np.array(acc.jk_names)
    for (f, K, dflt, name, p), v in acc.cells.items():
        base = dict(world=acc.world, T=acc.T, c=acc.c, f=f, K=K, default=dflt, model=name, pair=p, G_eff=acc.geff, few_groups=acc.need_jk, jk_unit=(acc.jk_kind if acc.need_jk else "none"),
                    n_jk=(len(acc.jk_names) if acc.need_jk else 0), jk_df=(acc.jk_df if acc.need_jk else np.nan), n_reps_boot=v["n"])
        key = f"{acc.world}|{acc.T}|{acc.c}|{f}|K{K}|{dflt}|{name}|p{p}"
        for metric, bkey, jkey, pkey in (("", "boot", "jk", "pts"), ("_auc", "boot_auc", "jk_auc", "pts_auc")):
            vals = v[bkey] / v["n"]
            ok = np.isfinite(vals)
            lo_p, hi_p = (np.percentile(vals[ok], PCT) if ok.sum() >= 2 else (np.nan, np.nan))
            x = float(np.nanmean(v[pkey])) if np.isfinite(v[pkey]).any() else np.nan
            lo_j = hi_j = np.nan
            rule = "pct"
            if acc.need_jk:
                th = v[jkey] / v["n"]
                lo_j, hi_j = jk_interval(x, th, acc.jk_df)
                rule = "union_pct_jk" if np.isfinite(lo_j) else "pct_jk_missing"
                jk_store[key + metric] = th.astype(np.float32)
            lo, hi = (float(np.fmin(lo_p, lo_j)), float(np.fmax(hi_p, hi_j))) if acc.need_jk and np.isfinite(lo_j) else (float(lo_p), float(hi_p))
            base.update({f"lo_pct{metric}": float(lo_p), f"hi_pct{metric}": float(hi_p), f"lo_jk{metric}": float(lo_j), f"hi_jk{metric}": float(hi_j),
                         f"lo{metric}": lo, f"hi{metric}": hi, f"interval_rule{metric}": rule, f"n_boot_valid{metric}": int(ok.sum())})
            rep_store[key + metric] = vals.astype(np.float32)
        rows_i.append(base)


# ---------------------------------------------------------------- まとめ

def delta_table(Pp, n_reps):
    """δ_J：（世界，f，K，既定）ごとに J_kl(q⁰) の反復の SD（T，c，対ごと）の中央値 × 2 ÷ √(反復の数)（D7）．δ_base は既定 ref の δ_J．Δ の反復の SD の中央値も並べる（記述）．"""
    q = Pp[Pp.model == "q0"]
    g = q.groupby(["world", "f", "K", "default", "T", "c", "pair"]).agg(n=("r", "nunique"), sd=("J", lambda x: x.std(ddof=1))).reset_index()
    g = g[(g.n >= 2) & np.isfinite(g.sd)]
    cols = ["world", "f", "K", "default", "n_cells", "n_reps", "median_sd_Jq0", "delta_J", "delta_J_half", "delta_J_double", "delta_base"]
    if g.empty:
        return pd.DataFrame(columns=cols)
    D = g.groupby(["world", "f", "K", "default"]).agg(n_cells=("sd", "size"), median_sd_Jq0=("sd", "median")).reset_index()
    D["n_reps"] = n_reps
    D["delta_J"] = 2 * D.median_sd_Jq0 / np.sqrt(n_reps)
    D["delta_J_half"], D["delta_J_double"] = 0.5 * D.delta_J, 2.0 * D.delta_J
    ref = D[D.default == MAIN_DEFAULT][["world", "f", "K", "delta_J"]].rename(columns={"delta_J": "delta_base"})
    D = D.merge(ref, on=["world", "f", "K"], how="left")
    for name in sorted(Pp.model.unique()):
        if name == "q0":
            continue
        gm = Pp[Pp.model == name].groupby(["world", "f", "K", "default", "T", "c", "pair"]).agg(n=("r", "nunique"), sd=("Delta", lambda x: x.std(ddof=1))).reset_index()
        gm = gm[(gm.n >= 2) & np.isfinite(gm.sd)].groupby(["world", "f", "K", "default"]).sd.median().rename(f"median_sd_Delta_{name}").reset_index()
        D = D.merge(gm, on=["world", "f", "K", "default"], how="left")
    return D


def units_table(Pp, Bb, Ii, Dd):
    """単位（世界，T，c，K）× f × 既定 × 候補 × 対の表：点の値，区間，δ_J，判定（正解率と AUC），土台の条件（3 つの規則）．"""
    P = Pp[Pp.model != "q0"].copy()
    keys = ["world", "T", "c", "unit", "f", "K", "default", "model", "role", "pair", "pair_name", "anger_pair"]
    agg = P.groupby(keys).agg(n_reps=("r", "nunique"), n_reps_finite=("Delta", lambda x: int(np.isfinite(x).sum())), n_pair_mean=("n_pair", "mean"),
                             n_groups_min=("n_groups", "min"), n_cand_kept_mean=("n_cand_kept", "mean"),
                             J_model_mean=("J", "mean"), J_q0_mean=("J_q0", "mean"), Delta_mean=("Delta", "mean"), Delta_auc_mean=("Delta_auc", "mean")).reset_index()
    for col in ("Delta", "Delta_auc"):
        wide = P.pivot_table(index=keys, columns="r", values=col, aggfunc="first")
        wide.columns = [f"{col}_r{r}" for r in wide.columns]
        agg = agg.merge(wide.reset_index(), on=keys, how="left")
    U = agg.merge(Ii, on=["world", "T", "c", "f", "K", "default", "model", "pair"], how="left")
    bk = ["world", "T", "c", "f", "K", "pair"]
    Bm = Bb.groupby(bk).agg(D_mean=("D", "mean"), d_mean=("d", "mean"), d_auc_mean=("d_auc", "mean"), dir_err_mean=("dir_err", "mean"), dir_err_auc_mean=("dir_err_auc", "mean"),
                            model_err_mean=("model_err", "mean"), var_ratio_mean=("var_ratio", "mean"), n_half_min_over_dim=("n_half_min_over_dim", "min"),
                            n_reps_base=("dir_err", lambda x: int(np.isfinite(x).sum()))).reset_index()
    U = U.merge(Bm, on=bk, how="left")
    U = U.merge(Dd[["world", "f", "K", "default", "delta_J", "delta_base"]], on=["world", "f", "K", "default"], how="left")
    for col in ("delta_J", "delta_base", "lo", "hi", "lo_auc", "hi_auc", "Delta_mean", "Delta_auc_mean", "dir_err_mean", "dir_err_auc_mean", "model_err_mean"):
        U[col] = pd.to_numeric(U[col], errors="coerce")
    fin = np.isfinite(U.delta_base)
    for scale, sname in DELTA_SCALES:
        dl = scale * U.delta_J
        U[f"judge_{sname}"] = [judge(x, lo, hi, d) for x, lo, hi, d in zip(U.Delta_mean, U.lo, U.hi, dl)]
        U[f"judge_auc_{sname}"] = [judge(x, lo, hi, d) for x, lo, hi, d in zip(U.Delta_auc_mean, U.lo_auc, U.hi_auc, dl)]
        U[f"base_ok_scaled_{sname}"] = np.where(np.isfinite(U.dir_err_mean) & fin, U.dir_err_mean >= scale * U.delta_base, False)
    U["base_ok_fixed"] = np.where(np.isfinite(U.dir_err_mean) & fin, U.dir_err_mean >= U.delta_base, False)
    U["base_ok_auc"] = np.where(np.isfinite(U.dir_err_auc_mean) & fin, U.dir_err_auc_mean >= U.delta_base, False)
    U["base_nan"] = ~np.isfinite(U.dir_err_mean)
    U["dir_err_neg"] = np.where(np.isfinite(U.dir_err_mean), U.dir_err_mean < 0, False)
    U["model_dep"] = np.where(np.isfinite(U.model_err_mean) & fin, U.model_err_mean.abs() > 0.5 * U.delta_base, False)   # 2 つの山の模型に依る印（E79 と同じ形）
    U["n_half_lt_dim"] = U.n_half_min_over_dim < 1
    return U


def null_pos_matrix(U, rep_store):
    """帰無の「上回った」：セル i・引き b で (b_i − x_i) + (lo_i − x_i) > 0 かつ (b_i − x_i) ≥ dl_i（設計 2 節．E83 4 節 (i) と同じ作り）．
    戻り：{(metric, scale): (n_cells, B) bool}．欠けのセルは全部 False．"""
    out = {}
    keys = [f"{r.world}|{r.T}|{r.c}|{r.f}|K{r.K}|{r.default}|{r.model}|p{r.pair}" for r in U.itertuples()]
    for metric, xcol, locol in (("", "Delta_mean", "lo"), ("_auc", "Delta_auc_mean", "lo_auc")):
        reps = [rep_store.get(k + metric) for k in keys]
        B = max((len(b) for b in reps if b is not None), default=0)
        M = np.zeros((len(U), B))
        for i, b in enumerate(reps):
            M[i] = b if b is not None else np.nan
        x, lo = U[xcol].to_numpy(dtype=float), U[locol].to_numpy(dtype=float)
        for scale, sname in DELTA_SCALES:
            dl = scale * U.delta_J.to_numpy(dtype=float)
            xs = M - x[:, None]
            with np.errstate(invalid="ignore"):
                pos = (xs + (lo - x)[:, None] > 0) & (xs >= dl[:, None])
            pos &= np.isfinite(x)[:, None] & np.isfinite(lo)[:, None] & np.isfinite(dl)[:, None] & np.isfinite(M)
            out[(metric, sname)] = pos
    return out


def long_judgments(U):
    """判定の長い表（1 行 = 単位 × 対 × 候補 × 既定 × 尺度 × 土台の規則）．row_id は U の行番号（帰無の行列の行）．"""
    out = []
    for scale, sname in DELTA_SCALES:
        for brule in BASE_RULES:
            bcol = {"fixed": "base_ok_fixed", "scaled": f"base_ok_scaled_{sname}", "auc": "base_ok_auc"}[brule]
            x = U[["world", "T", "c", "unit", "f", "K", "default", "model", "role", "pair", "pair_name", "anger_pair", "dir_err_neg", "base_nan", "model_dep",
                   f"judge_{sname}", f"judge_auc_{sname}", bcol]].copy()
            x = x.rename(columns={f"judge_{sname}": "judge", f"judge_auc_{sname}": "judge_auc", bcol: "base_ok"})
            x["scale"], x["base_rule"], x["row_id"] = sname, brule, np.arange(len(U))
            out.append(x)
    L = pd.concat(out, ignore_index=True)
    L["pair_type"] = np.where(L.anger_pair, "anger", "nonanger")
    L["K"] = L.K.astype(str)
    return L


def null_stats(NP, metric, sname, rows, blocks=None):
    """帰無の数の分布：rows（U の行番号）のセルの帰無の pos を引きごとに数える．blocks を渡すと塊（同じ値の行は any で畳む）の数．戻り (95% 点, 平均)．"""
    if len(rows) == 0:
        return 0.0, 0.0
    M = NP[(metric, sname)][np.asarray(rows)]
    if blocks is not None:
        bl = np.asarray(blocks)
        M = np.stack([M[bl == b].any(0) for b in np.unique(bl)])
    cnt = M.sum(0)
    return float(np.percentile(cnt, NULL_PCT)), float(cnt.mean())


def counts_table(L, NP):
    """数え上げ（世界 × f × K（と all）× 既定 × 候補 × 対の種類 × 尺度 × 土台の規則）：pos/neg/same/undet/missing，分母，土台後，帰無の 95% 点と平均，n×0.025（下限の参照），AUC との重なり．"""
    rows = []
    for with_all in (False, True):
        X = L.copy()
        if with_all:
            X["K"] = "all"
        for key, g in X.groupby(["world", "f", "K", "default", "model", "pair_type", "scale", "base_rule"]):
            b = g[g.base_ok]
            n, nb = len(g), len(b)
            q95, mu = null_stats(NP, "", key[6], b.row_id.to_numpy())
            q95a, mua = null_stats(NP, "", key[6], g.row_id.to_numpy())
            pb = b[b.judge == "pos"]
            row = dict(zip(["world", "f", "K", "default", "model", "pair_type", "scale", "base_rule"], key))
            row.update(n_cells=n, n_pos=int((g.judge == "pos").sum()), n_neg=int((g.judge == "neg").sum()), n_same=int((g.judge == "same").sum()),
                       n_undet=int((g.judge == "undet").sum()), n_missing=int((g.judge == "missing").sum()), chance_all=n * CHANCE, null95_all=q95a, null_mean_all=mua,
                       n_base=nb, n_pos_base=len(pb), n_neg_base=int((b.judge == "neg").sum()), n_same_base=int((b.judge == "same").sum()),
                       n_undet_base=int((b.judge == "undet").sum()), n_missing_base=int((b.judge == "missing").sum()),
                       share_pos_base=(len(pb) / nb if nb else np.nan), chance_base=nb * CHANCE, null95_base=q95, null_mean_base=mu,
                       n_pos_base_auc_pos=int((pb.judge_auc == "pos").sum()), n_pos_base_auc_same=int((pb.judge_auc == "same").sum()),
                       n_pos_base_auc_other=int((~pb.judge_auc.isin(["pos", "same"])).sum()),
                       n_base_model_dep=int(b.model_dep.sum()), n_base_nan=int(g.base_nan.sum()),
                       n_dir_err_neg=int(g.dir_err_neg.sum()), share_dir_err_neg=(float(g.dir_err_neg.mean()) if n else np.nan))
            row["exceeds_null95_base"] = bool(row["n_pos_base"] > q95) if nb else False          # R86-1／R86-2 の規則（帰無の 95% 点）
            row["exceeds_null95_all"] = bool(row["n_pos"] > q95a) if n else False               # 全セル（仮定の表の読み替え）
            row["exceeds_chance_base"] = bool(row["n_pos_base"] > row["chance_base"]) if nb else False   # 下限の参照だけ
            rows.append(row)
    return pd.DataFrame(rows)


def corpus_table(L, NP):
    """塊に畳む：(c のコーパス，対)（どの T・K かで 1 回でも pos；過半で pos），(T，c，対)（K をまたがない）．帰無の 95% 点と平均，塊の数 × 0.025（参照）．"""
    blocks, totals = [], []
    key_t = ["world", "f", "default", "model", "scale", "base_rule", "pair_type"]
    for key, Lg in L.groupby(key_t):
        g_rows = []
        for (unit, pair, pname), g in Lg.groupby(["unit", "pair", "pair_name"]):
            b = g[g.base_ok]
            n, npos, nb, npb = len(g), int((g.judge == "pos").sum()), len(b), int((b.judge == "pos").sum())
            g_rows.append({**dict(zip(key_t, key)), "unit": unit, "pair": pair, "pair_name": pname, "level": "block", "n_cells": n, "n_pos": npos, "any_pos": npos >= 1,
                           "majority_pos": npos > n / 2, "n_cells_base": nb, "n_pos_base": npb, "any_pos_base": npb >= 1, "majority_pos_base": (npb > nb / 2) if nb else False})
        blocks.extend(g_rows)
        g = pd.DataFrame(g_rows)
        base = Lg[Lg.base_ok]
        q95c, muc = null_stats(NP, "", key[4], base.row_id.to_numpy(), (base.unit + "|" + base.pair.astype(str)).to_numpy())
        q95t, mut = null_stats(NP, "", key[4], base.row_id.to_numpy(), (base["T"] + "|" + base.c + "|" + base.pair.astype(str)).to_numpy())
        tc = base.groupby(["T", "c", "pair"]).judge.apply(lambda s: bool((s == "pos").any())).astype(bool) if len(base) else pd.Series(dtype=bool)
        totals.append({**dict(zip(key_t, key)), "level": "total_corpus", "n_blocks": len(g), "chance_blocks": len(g) * CHANCE, "n_blocks_base": int((g.n_cells_base > 0).sum()),
                       "n_any_pos": int(g.any_pos.sum()), "n_majority_pos": int(g.majority_pos.sum()), "n_any_pos_base": int(g.any_pos_base.sum()),
                       "n_majority_pos_base": int(g.majority_pos_base.sum()), "null95_any_base": q95c, "null_mean_any_base": muc,
                       "exceeds_null95_any_base": bool(int(g.any_pos_base.sum()) > q95c) if int((g.n_cells_base > 0).sum()) else False,
                       "note": "chance_blocks = per-comparison lower reference; judge with null95 (shared speaker-group draws)"})
        totals.append({**dict(zip(key_t, key)), "level": "total_Tc", "n_blocks": int(len(tc)), "chance_blocks": len(tc) * CHANCE, "n_blocks_base": int(len(tc)),
                       "n_any_pos": np.nan, "n_majority_pos": np.nan, "n_any_pos_base": int(tc.sum()), "n_majority_pos_base": np.nan,
                       "null95_any_base": q95t, "null_mean_any_base": mut, "exceeds_null95_any_base": bool(int(tc.sum()) > q95t) if len(tc) else False,
                       "note": "(T, c, pair) blocks over K (base cells only)"})
    return pd.concat([pd.DataFrame(blocks), pd.DataFrame(totals)], ignore_index=True)


def target_table(L, NP):
    """評価のターゲット T ごとの数え上げ（R86-1T．E80 の R3′ と同じ扱い）：その T の単位（T の仮のターゲット c）だけで数え，帰無の 95% 点も同じ複製から出す．"""
    rows = []
    for with_all in (False, True):
        X = L.copy()
        if with_all:
            X["K"] = "all"
        for key, g in X.groupby(["world", "T", "f", "K", "default", "model", "pair_type", "scale", "base_rule"]):
            b = g[g.base_ok]
            q95, mu = null_stats(NP, "", key[7], b.row_id.to_numpy())
            npb = int((b.judge == "pos").sum())
            rows.append({**dict(zip(["world", "T", "f", "K", "default", "model", "pair_type", "scale", "base_rule"], key)), "n_cells": len(g), "n_pos": int((g.judge == "pos").sum()),
                         "n_base": len(b), "n_pos_base": npb, "chance_base": len(b) * CHANCE, "null95_base": q95, "null_mean_base": mu,
                         "exceeds_null95_base": bool(npb > q95) if len(b) else False, "n_units": int(g[["c"]].drop_duplicates().shape[0])})
    return pd.DataFrame(rows)


def default_compare(L):
    """P86-2：同じ（単位，対）の組で，主（ref）と副（pF）の判定を並べる（両方 pos・主だけ・副だけ・どちらでもない）．土台は既定に依らない（fixed）ので同じ集合．K ごとと all，(T, c, 対) に畳んだ版．"""
    idx = ["world", "T", "c", "f", "K", "model", "pair", "pair_type", "scale", "base_rule"]
    S = L.pivot_table(index=idx, columns="default", values="judge", aggfunc="first").reset_index()
    base = L[L.default == MAIN_DEFAULT][idx + ["base_ok"]]
    S = S.merge(base, on=idx, how="left")
    rows = []
    for with_all in (False, True):
        X = S.copy()
        if with_all:
            X["K"] = "all"
        for key, g in X.groupby(["world", "f", "K", "model", "pair_type", "scale", "base_rule"]):
            b = g[g.base_ok]
            mp, sp = (b[MAIN_DEFAULT] == "pos"), (b["pF"] == "pos")
            tc = (b.groupby(["T", "c", "pair"]).agg(m=(MAIN_DEFAULT, lambda s: bool((s == "pos").any())), s=("pF", lambda s: bool((s == "pos").any()))).astype(bool)
                  if len(b) else pd.DataFrame({"m": pd.Series(dtype=bool), "s": pd.Series(dtype=bool)}))
            rows.append({**dict(zip(["world", "f", "K", "model", "pair_type", "scale", "base_rule"], key)), "n_base": len(b),
                         "n_both": int((mp & sp).sum()), "n_main_only": int((mp & ~sp).sum()), "n_sub_only": int((~mp & sp).sum()), "n_neither": int((~mp & ~sp).sum()),
                         "n_pos_main": int(mp.sum()), "n_pos_sub": int(sp.sum()), "sub_ge_main": bool(sp.sum() >= mp.sum()),
                         "n_Tc_blocks": int(len(tc)), "n_Tc_both": int((tc.m & tc.s).sum()), "n_Tc_main_only": int((tc.m & ~tc.s).sum()), "n_Tc_sub_only": int((~tc.m & tc.s).sum())})
    return pd.DataFrame(rows)


def pivot_table_emotions(L):
    """P86-3：4 つの感情のそれぞれを要にして，要を含む 3 対と含まない 3 対の土台後の「上回った」の割合を並べ，怒りの順位を付ける（記述．4 通りしかないので偶然を超えるとは書かない）．"""
    rows = []
    for with_all in (False, True):
        X = L.copy()
        if with_all:
            X["K"] = "all"
        for key, g in X.groupby(["world", "f", "K", "default", "model", "scale", "base_rule"]):
            b = g[g.base_ok]
            diffs = {}
            for e in range(4):
                inc = b.pair.map(lambda p: e in PAIRS[int(p)]).to_numpy(dtype=bool) if len(b) else np.zeros(0, bool)
                w, wo = b[inc], b[~inc]
                sw = float((w.judge == "pos").mean()) if len(w) else np.nan
                swo = float((wo.judge == "pos").mean()) if len(wo) else np.nan
                diffs[e] = sw - swo if np.isfinite(sw) and np.isfinite(swo) else np.nan
                rows.append({**dict(zip(["world", "f", "K", "default", "model", "scale", "base_rule"], key)), "pivot": EMO[e], "pivot_id": e,
                             "n_base_with": len(w), "n_pos_with": int((w.judge == "pos").sum()), "share_pos_with": sw,
                             "n_base_without": len(wo), "n_pos_without": int((wo.judge == "pos").sum()), "share_pos_without": swo, "diff_share": diffs[e]})
            vals = np.array([diffs[e] for e in range(4)])
            if np.isfinite(vals).all():
                order = (-vals).argsort(kind="stable")
                rank = {int(e): int(i + 1) for i, e in enumerate(order)}
                for rr in rows[-4:]:
                    rr["rank"] = rank[rr["pivot_id"]]
                    rr["anger_first"] = rank[ANGER] == 1 and vals[ANGER] > np.max(np.delete(vals, ANGER))
            else:
                for rr in rows[-4:]:
                    rr["rank"], rr["anger_first"] = np.nan, False
    return pd.DataFrame(rows)


def assumptions_table(U):
    """土台の仮定を測る表（設計 4 節）：(世界, f, K) ごとに，方向の誤りが負のセルの割合，半分の本数 ÷ 次元，2 つの山の模型に依るセルの割合，等分散の比，IEMOCAP の半分が分けたセッション．"""
    rows = []
    for key, g in U[(U.default == MAIN_DEFAULT) & (U.model == "s")].groupby(["world", "f", "K"]):
        base = g[g.base_ok_fixed]
        rows.append({"world": key[0], "f": key[1], "K": key[2], "n_cells": len(g), "n_base_fixed": len(base),
                     "share_dir_err_neg": float(g.dir_err_neg.mean()), "read_unconditioned": bool(g.dir_err_neg.mean() > NEG_SHARE_LIMIT),
                     "n_half_min_over_dim_median": float(g.n_half_min_over_dim.median()), "share_units_n_half_lt_dim": float(g.n_half_lt_dim.mean()),
                     "share_model_dep_base": (float(base.model_dep.mean()) if len(base) else np.nan),
                     "share_var_ratio_in_half_2": float(((g.var_ratio_mean >= 0.5) & (g.var_ratio_mean <= 2)).mean()),
                     "median_abs_model_err": float(g.model_err_mean.abs().median()), "delta_base": float(g.delta_base.iloc[0]) if len(g) else np.nan})
    return pd.DataFrame(rows)


def hier_table(L):
    """階層の形の切り分け（R86-3）：s が pos の（単位，対）のうち h1・h05 も pos の数と割合（全体と土台後．K ごとと all）．"""
    idx = ["world", "T", "c", "f", "K", "default", "pair", "pair_type", "scale", "base_rule"]
    S = L[L.model.isin(["s", "h1", "h05"])].pivot_table(index=idx, columns="model", values="judge", aggfunc="first").reset_index()
    base = L[L.model == "s"][idx + ["base_ok"]]
    S = S.merge(base, on=idx, how="left")
    rows = []
    for with_all in (False, True):
        X = S.copy()
        if with_all:
            X["K"] = "all"
        for key, g in X.groupby(["world", "f", "K", "default", "pair_type", "scale", "base_rule"]):
            for cond, gg in (("all", g), ("base", g[g.base_ok])):
                sp = gg[gg.s == "pos"]
                n = len(sp)
                h1 = int((sp.h1 == "pos").sum()) if n else 0
                h05 = int((sp.h05 == "pos").sum()) if n else 0
                rows.append({**dict(zip(["world", "f", "K", "default", "pair_type", "scale", "base_rule"], key)), "cond": cond, "n_cells": len(gg), "n_s_pos": n,
                             "n_h1_also_pos": h1, "n_h05_also_pos": h05, "share_h1": h1 / n if n else np.nan, "share_h05": h05 / n if n else np.nan,
                             "readable": n >= 3, "majority_h1": (h1 / n > 0.5) if n else False, "h05_le_h1": (h05 <= h1) if n else False,
                             "n_h1_pos_total": int((gg.h1 == "pos").sum()), "n_h05_pos_total": int((gg.h05 == "pos").sum())})
    return pd.DataFrame(rows)


def anger_units(Aa):
    """怒りと答える割合と 4 クラスの質の単位の表（反復の平均）．R86-4：s との差 diff_vs_s と ±2 点の印．"""
    keys = ["world", "T", "c", "unit", "f", "K", "default", "model"]
    qcols = ["share_anger", "share_anger_true", "recall_neutral", "recall_happy", "recall_sad", "recall_angry", "uar", "precision_anger"]
    A = Aa.groupby(keys).agg(n_reps=("r", "nunique"), **{f"{q}_mean": (q, "mean") for q in qcols}).reset_index()
    s = A[A.model == "s"][["world", "T", "c", "f", "K", "default", "share_anger_mean"]].rename(columns={"share_anger_mean": "share_s"})
    A = A.merge(s, on=["world", "T", "c", "f", "K", "default"], how="left")
    A["diff_vs_s"] = 100 * (A.share_anger_mean - A.share_s)
    A["within_2pt_of_s"] = A.diff_vs_s.abs() <= 2
    return A


def compare_e83(Pp, path, chk):
    """results/e83/ に (C) の出力があれば，J_kl(ref__K)（既定 ref の q⁰），J_kl(P__<fused>)（既定 pF の q⁰），J_kl(other_<m>) が E83_TOL 以内で一致することを確かめる（無ければ飛ばす）．"""
    if not path.exists():
        chk.append({"world": "", "T": "", "c": "", "r": "", "check": "e83_compare", "status": "skipped", "detail": f"{path} absent"})
        return None
    E = pd.read_csv(path, comment="#", dtype={"K": str})
    E = E[(E.n_pair >= E83_MIN_PAIR_N) & (E.g.isin(["ref", "pF"]) | E.g.str.startswith("other_"))]
    E = E.rename(columns={"target": "T", "pseudo": "c", "rep": "r", "student": "f", "J": "J_e83"})
    E["model"] = np.where(E.g.isin(["ref", "pF"]), "q0", E.g)
    E["default"] = np.where(E.g == "pF", "pF", "ref")
    E["K"] = E.K.astype(int)
    M = Pp[(Pp.n_pair >= E83_MIN_PAIR_N) & ((Pp.model == "q0") | Pp.model.str.startswith("other_"))]
    J = M.merge(E[["world", "T", "c", "r", "f", "K", "default", "model", "pair", "J_e83"]], on=["world", "T", "c", "r", "f", "K", "default", "model", "pair"], how="inner")
    if J.empty:
        chk.append({"world": "", "T": "", "c": "", "r": "", "check": "e83_compare", "status": "skipped", "detail": "no overlapping rows"})
        return None
    mx = float((J.J - J.J_e83).abs().max())
    chk.append({"world": "", "T": "", "c": "", "r": "", "check": "e83_compare", "status": "ok" if mx <= E83_TOL else "FAIL",
                "detail": f"n_rows={len(J)} (ref {int((J.default == 'ref').sum())}, pF {int((J.default == 'pF').sum())}) max_abs_diff={mx:.3e}"})
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
    # (1) 比較の基準と同じ確率を候補に置くと J_kl = 0．J は 2 感情の正解率の差
    for p in range(6):
        m = (y == PAIRS[p][0]) | (y == PAIRS[p][1])
        r = pair_right(P[m], y[m], p)
        assert j_stat(r, r) == 0.0, "J(a, a) != 0"
        Q = rng.dirichlet(np.ones(4), m.sum())
        rq = pair_right(Q, y[m], p)
        assert abs(j_stat(rq, r) - 100 * (rq.mean() - r.mean())) < 1e-9, "J != acc difference"
    print("selftest 1: J(a, a) = 0 and J = 100*(acc_model - acc_a)  ok")
    # (2) D = d なら方向の誤り = 0．base_D と pair_d が既知の離れ方を再現
    Dv = rng.uniform(0.2, 3, 6)
    assert np.allclose(dir_err(Dv, Dv), 0) and (dir_err(Dv + 0.5, Dv) > 0).all(), "dir_err"
    d_dim, n2 = 20, 4000
    y2 = rng.integers(0, 4, n2)
    mu = np.zeros((4, d_dim))
    mu[1, 0], mu[2, 1], mu[3, 2] = 2.0, 2.0, 2.0
    Z2 = rng.normal(size=(n2, d_dim)) + mu[y2]
    groups = np.array([f"g{i % 40}" for i in range(n2)])
    D2, _, _, _, note = base_D(Z2, y2, groups)
    truth = np.array([np.linalg.norm(mu[k] - mu[j]) for k, j in PAIRS])
    assert note == "ok" and np.all(np.abs(D2 / truth - 1) < 0.1), f"base_D off: {D2} vs {truth}"
    L3 = rng.normal(size=(n2, 4))
    L3[np.arange(n2), y2] += 1.5
    d3, vr3 = pair_d(L3, y2)
    assert np.all(np.abs(d3 - 1.5 * 2 / np.sqrt(2)) < 0.15) and np.all(np.abs(vr3 - 1) < 0.2), f"pair_d off: {d3} {vr3}"
    print("selftest 2: dir_err(D, D) = 0; base_D and pair_d recover known separations  ok")
    # (3)(4)(5) 階層の形：α = 0 は q⁰，α = 1 の h(怒) = s(怒)，行の和が 1，怒りを含まない対の判定は q⁰ と同じ
    s = rng.dirichlet(np.ones(4), n)
    q0 = rng.dirichlet(np.ones(4) * 0.3, n)
    q0[:5] = np.eye(4)[[3, 3, 0, 1, 2]]
    qc = np.clip(q0, EPS, None)
    qc = qc / qc.sum(1, keepdims=True)
    assert np.abs(hier(s, q0, 0.0) - qc).max() < 1e-9, "alpha=0 != q0"
    for alpha in (0.0, 0.5, 1.0):
        h = hier(s, q0, alpha)
        assert np.abs(h.sum(1) - 1).max() < 1e-9 and (h >= 0).all(), f"rows of h(alpha={alpha})"
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
    # (7) ブートストラップ：同じモデルなら Δ の全ての引きが 0．判定の 4 区分．jackknife の区間は点を含む
    G, B = 12, 200
    gi = rng.integers(0, G, n)
    W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(np.float64)
    p = ANGER_PAIRS[0]
    m = (y == PAIRS[p][0]) | (y == PAIRS[p][1])
    r1 = pair_right(P[m], y[m], p)
    den = np.bincount(gi[m], minlength=G).astype(np.float64)
    num = np.bincount(gi[m], weights=r1.astype(np.float64) - r1.astype(np.float64), minlength=G)
    assert np.all(100 * (W @ num) / (W @ den) == 0), "bootstrap Delta != 0 for identical models"
    assert judge(1.0, 0.2, 1.8, 0.5) == "pos" and judge(-1.0, -1.8, -0.2, 0.5) == "neg" and judge(0.1, -0.3, 0.3, 0.5) == "same" \
        and judge(1.0, -0.2, 1.8, 0.5) == "undet" and judge(np.nan, 0, 1, 0.5) == "missing", "judge rules"
    th = rng.normal(size=10)
    lo_j, hi_j = jk_interval(0.0, th, 9)
    assert lo_j < 0 < hi_j and np.isnan(jk_interval(np.nan, th, 9)[0]), "jk_interval"
    print("selftest 7: bootstrap Delta = 0 for identical models; judge rules; jackknife interval  ok")
    # (8) 重みつき AUC：重み 1 なら E79 の rankdata の式と一致（同点あり）．復元抽出の回数の重みは，複製した標本の AUC と一致
    z = np.round(rng.normal(size=300), 1)
    isk = rng.random(300) < 0.4
    isk[:2] = [True, False]
    a_ref = pair_auc(z, isk)
    a_w = weighted_auc(z, isk, np.ones((3, 300)))
    assert np.allclose(a_w, a_ref, atol=1e-6), "weighted_auc(ones) != rankdata AUC"
    w = rng.integers(0, 3, 300).astype(float)
    rep = np.repeat(np.arange(300), w.astype(int))
    a_rep = pair_auc(z[rep], isk[rep])
    assert abs(weighted_auc(z, isk, w[None, :])[0] - a_rep) < 1e-6, "weighted_auc(counts) != AUC of replicated sample"
    assert np.isnan(weighted_auc(z, isk, (~isk).astype(float)[None, :])[0]), "weighted_auc with no k-weight should be NaN"
    print("selftest 8: weighted AUC matches rankdata AUC and replicated-sample AUC  ok")
    # (9) 帰無の規則：複製が点と同じなら帰無の pos は 0．G_eff
    assert abs((lambda nn: nn.sum() ** 2 / (nn ** 2).sum())(np.array([5, 5, 5, 5])) - 4) < 1e-12, "G_eff"
    print("selftest 9: G_eff  ok")
    print("selftest: all passed")
    return 0


# ---------------------------------------------------------------- 本体

def corpus_group_lists(e80):
    """コーパスごとの話者群の一覧（両世界の samples_<c>.csv の和集合．順は名前順）．ブートストラップと jackknife の重みはこの順に引く（D15）．"""
    lists = {}
    for sp_path in sorted(e80.glob("*/units/*/samples_*.csv")):
        c = sp_path.stem[len("samples_"):]
        if c not in GROUP_SPEC:
            continue
        g = set(pd.read_csv(sp_path).speaker_group.astype(str))
        lists.setdefault(corp(c), set()).update(g)
    return {cp: sorted(gs) for cp, gs in lists.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e86")
    ap.add_argument("--e80", default="results/e80", help="E80 の出力の場所（単位ファイルと samples）")
    ap.add_argument("--e83", default="results/e83", help="E83 の出力の場所（(C) との照合）")
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
    e80 = (ROOT / args.e80) if not Path(args.e80).is_absolute() else Path(args.e80)
    e83 = (ROOT / args.e83) if not Path(args.e83).is_absolute() else Path(args.e83)
    partial = args.smoke or bool(args.targets) or args.reps < DEFAULT_REPS or set(args.worlds) != set(WORLDS)
    if partial and (ROOT / "results") in out.resolve().parents:
        raise SystemExit("部分的な流し方（--smoke，--targets，--reps < 5，--worlds の一部）の出力は results/ の下に書けない．--out を checks/ か scratchpad にする")
    if out.exists() and any(out.glob("*.csv")):
        raise SystemExit(f"{out} に古い出力がある．消してから流す")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), out / "script_copy.py")
    (out / "code").mkdir(exist_ok=True)
    for rel in CODE_COPY:
        (out / "code" / Path(rel).parent).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, out / "code" / rel)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty_txt = subprocess.run(["git", "status", "--porcelain", "--", *SCRIPT_DEPS], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    script_sha = sha256(Path(__file__))
    global HEADER
    HEADER = (f"E86 commit={commit} dirty={bool(dirty_txt)} script_sha256={script_sha[:16]} seed={BOOT_SEED} worlds={args.worlds} reps={args.reps} boot={args.boot} "
              f"targets={args.targets} smoke={args.smoke} e80={args.e80}")
    t0 = time.time()
    prov = {"experiment": "E86", "commit": commit, "dirty": bool(dirty_txt), "dirty_files": dirty_txt.splitlines(), "dirty_scope": list(SCRIPT_DEPS),
            "script_sha256": script_sha, "code_copy": {rel: sha256(ROOT / rel) for rel in CODE_COPY}, "args": vars(args) | {"out": str(out), "e80": str(e80), "e83": str(e83)},
            "constants": CONSTANTS, "inputs": {}, "started": dt.datetime.now().isoformat(timespec="seconds"),
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__}}
    # コーパスごとの話者群の一覧と重み（両世界で共通．反復・T・f・K・既定・候補・対で共通）
    cg = corpus_group_lists(e80)
    W_corpus, geff_info = {}, {}
    for cp, gs in cg.items():
        rng = np.random.default_rng([BOOT_SEED, zlib.crc32(cp.encode())])
        W_corpus[cp] = rng.multinomial(len(gs), np.full(len(gs), 1 / len(gs)), size=args.boot).astype(np.float64)
    prov["corpus_groups"] = {cp: len(gs) for cp, gs in cg.items()}
    rows_p, rows_b, rows_a, rows_i, chk, timing = [], [], [], [], [], []
    rep_store, jk_store, gc_store = {}, {}, {}
    for world in args.worlds:
        fused = WORLDS[world]
        udir = e80 / world / "units" / fused
        if not udir.exists():
            raise SystemExit(f"{udir} が無い")
        Ts = sorted(p.name for p in udir.iterdir() if p.is_dir())
        if args.targets:
            Ts = [T for T in Ts if T in args.targets]
        if args.smoke:
            Ts = Ts[:1]
        pairs_tc = [(T, c) for T in Ts for c in sorted({f.stem.rsplit("_r", 1)[0] for f in (udir / T).glob("*_r*.npz") if not f.name.endswith("_agg.npz")})]
        if not partial and len(pairs_tc) != EXPECTED_PAIRS[world]:
            raise SystemExit(f"{world}：(T, c) の組が {len(pairs_tc)} で設計の {EXPECTED_PAIRS[world]} と合わない（D19）")
        chk.append({"world": world, "T": "", "c": "", "r": "", "check": "n_Tc_pairs", "status": "ok" if len(pairs_tc) == EXPECTED_PAIRS[world] else "partial", "detail": f"{len(pairs_tc)} (design {EXPECTED_PAIRS[world]})"})
        feats, samples = {}, {}
        for T, c in pairs_tc:
            if c not in feats:
                feats[c] = feats_for(c, fused)
                sp_path = udir / f"samples_{c}.csv"
                sp = pd.read_csv(sp_path)
                if not np.array_equal(sp.sample_id.astype(str).to_numpy(), feats[c][0].sample_id.astype(str).to_numpy()) or not np.array_equal(sp.row.to_numpy(), np.arange(len(sp))):
                    raise SystemExit(f"{world} {c}：samples の sample_id が発話一覧と合わない")
                if not np.array_equal(sp.speaker_group.astype(str).to_numpy(), feats[c][0].speaker_group.astype(str).to_numpy()):
                    raise SystemExit(f"{world} {c}：samples の speaker_group が発話一覧と合わない")
                chk.append({"world": world, "T": "", "c": c, "r": "", "check": "samples_vs_manifest", "status": "ok", "detail": f"n={len(sp)}"})
                samples[c] = sp
                prov["inputs"][str(sp_path.relative_to(ROOT)) if sp_path.is_relative_to(ROOT) else str(sp_path)] = sha256(sp_path)
            # 反復の候補の話者群の本数 → G_eff と jackknife の単位（反復の処理の前に決める）
            files = []
            for r in range(args.reps):
                fpath = udir / T / f"{c}_r{r}.npz"
                if not fpath.exists():
                    raise SystemExit(f"{fpath} が無い（反復がそろっていない）")
                files.append(fpath)
            grp_all = samples[c].speaker_group.astype(str).to_numpy()
            cnt = {}
            for fpath in files:
                for g in grp_all[np.asarray(np.load(fpath)["cand"]).astype(int)]:
                    cnt[g] = cnt.get(g, 0) + 1
            ug = sorted(cnt)
            nn = np.array([cnt[g] for g in ug], dtype=float)
            geff = float(nn.sum() ** 2 / (nn ** 2).sum())
            need_jk = geff < MIN_GEFF
            jk_kind = "session" if corp(c) in SESSION_CORPORA else "speaker"
            acc = UnitAcc(world, T, c, ug, cg[corp(c)], W_corpus[corp(c)], geff, need_jk, jk_kind)
            geff_info[f"{world}|{T}|{c}"] = {"G": len(ug), "G_eff": geff, "few_groups": need_jk, "jk_unit": jk_kind if need_jk else "none", "n_jk": len(acc.jk_names) if need_jk else 0}
            for r, fpath in enumerate(files):
                tu = time.time()
                prov["inputs"][str(fpath.relative_to(ROOT)) if fpath.is_relative_to(ROOT) else str(fpath)] = sha256(fpath)
                u = Unit(world, T, c, r, e80, feats[c], samples[c], chk)
                process_unit(u, acc, rows_p, rows_b, rows_a, gc_store, chk)
                timing.append(dict(world=world, T=T, c=c, r=r, n_cand=len(u.cand), n_groups=len(ug), seconds=round(time.time() - tu, 2)))
                print(world, T, c, r, f"n_cand={len(u.cand)} groups={len(ug)} G_eff={geff:.1f} jk={jk_kind if need_jk else 'none'} {time.time() - tu:.1f}s", flush=True)
            finalize_unit(acc, rows_i, rep_store, jk_store, gc_store)
    Pp, Bb, Aa, Ii = pd.DataFrame(rows_p), pd.DataFrame(rows_b), pd.DataFrame(rows_a), pd.DataFrame(rows_i)
    if Pp.empty:
        raise SystemExit("単位ファイルが 1 つも無い")
    if np.isinf(Pp[["J", "Delta", "Delta_auc"]].to_numpy(dtype=float)).any() or np.isinf(Bb[["D", "d", "dir_err", "d_auc"]].to_numpy(dtype=float)).any():
        raise SystemExit("pairs/base に Inf がある")
    e83_sha = compare_e83(Pp, e83 / "c_complementarity_units.csv", chk)
    if e83_sha:
        prov["inputs"][str(args.e83) + "/c_complementarity_units.csv"] = e83_sha
    Dd = delta_table(Pp, args.reps)
    U = units_table(Pp, Bb, Ii, Dd)
    NP = null_pos_matrix(U, rep_store)
    L = long_judgments(U)
    write_csv(Pp, out / "pairs.csv")
    write_csv(Bb, out / "base.csv")
    write_csv(U, out / "units.csv")
    write_csv(Dd, out / "delta.csv")
    write_csv(counts_table(L, NP), out / "counts.csv")
    write_csv(corpus_table(L, NP), out / "corpus_counts.csv")
    write_csv(target_table(L, NP), out / "target_counts.csv")
    write_csv(default_compare(L), out / "default_compare.csv")
    write_csv(pivot_table_emotions(L), out / "pivot_counts.csv")
    write_csv(assumptions_table(U), out / "assumptions.csv")
    write_csv(Aa, out / "anger_share.csv")
    write_csv(anger_units(Aa), out / "anger_share_units.csv")
    write_csv(hier_table(L), out / "hier_counts.csv")
    write_csv(pd.DataFrame(chk), out / "checks.csv")
    # 複製と jackknife と話者群ごとの当たりの数（区間と帰無の作り直しに使える）
    keys = [f"{r.world}|{r.T}|{r.c}|{r.f}|K{r.K}|{r.default}|{r.model}|p{r.pair}" for r in U.itertuples()]
    B = args.boot
    np.savez_compressed(out / "boot_replicates.npz", delta=np.stack([rep_store.get(k, np.full(B, np.nan, np.float32)) for k in keys]),
                        delta_auc=np.stack([rep_store.get(k + "_auc", np.full(B, np.nan, np.float32)) for k in keys]))
    pd.DataFrame({"row_id": np.arange(len(keys)), "key": keys}).to_csv(out / "boot_index.csv", index=False)
    np.savez_compressed(out / "jackknife.npz", **{k.replace("|", "__"): v for k, v in jk_store.items()})      # セルごとの θ_(g)（few_groups の単位だけ）と単位ごとの抜く単位の名前
    np.savez_compressed(out / "group_counts.npz", **{k.replace("|", "__"): v for k, v in gc_store.items()})  # (単位ファイル, f, K) ごとの den・a_right・right（話者群の順は <world>__<T>__<c>__groups）
    prov["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    prov["elapsed_seconds"] = round(time.time() - t0, 1)
    prov["timing_per_unit_file"] = timing
    prov["units_geff"] = geff_info
    prov["n_rows"] = {"pairs": len(Pp), "base": len(Bb), "units": len(U), "anger_share": len(Aa), "checks": len(chk), "replicate_cells": len(keys)}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("wrote", out, f"pairs={len(Pp)} base={len(Bb)} units={len(U)} elapsed={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
