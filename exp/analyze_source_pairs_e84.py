"""E84：ソース × 感情の対 × モダリティ．ソースだけで学んだ分類器の方向は，どのターゲットのどの感情の対で使えるか．

設計：docs/exploratory-analyses.md の E84（規則は値を見る前に書いた．関門 1 の直しを 2026-11-03 に入れた）．新しい学習はしない．
  (A) kind × ターゲット × 対 × ソース（全ソースの合併 ALL を含む）の a = 100·Φ(d/2)（切片を直した後の 2 感情の正解率）．
      全発話と，話者（IEMOCAP はセッション）の半分ごと
  (B) 対ごとに最良のソースを選ぶ余地：半分 A で選び，半分 B で測る（入れ替えて両方向）．主の相手は合併 ALL（G_all），副は 6 対の平均で最良の 1 ソース（G_best）．
      区間は測る半分の話者（セッション）のブートストラップ．UAR への橋（投票の分類器）．a_AUC・実際の閾値での正解率も出す
  (C) 世界 AV・AVT で，対ごとに最良の単モダリティを半分 A で選び，半分 B で融合と比べる（R_oos．符号つき）．
      主の版は単モダリティのソース候補を融合の群に限る．副は全群と，cebd64b の規則の R（半分 B で選び 0 で切る）
  (D) 記述：音声の s* をコーパスの種類で集計
  (E) 偶然の見込み：P84-1 はソース名の置換，R1・R2 の数え上げは二項分布

入力は results/shift_<kind>/logits/<ソース>.npz の t_<ターゲット>（E13 と同じ）と，results/multisource_scores_<kind>/<t>.npz の p_tgt_all（全ソースの合併）．
ラベルと話者は measure/data.load_any で読む．同点は群の名前の辞書順で先のものを取る．

使い方：./.venv/bin/python exp/analyze_source_pairs_e84.py [--out results/e84] [--kinds ...] [--worlds ...] [--seeds 5] [--boot 500] [--perm 1000] [--smoke]
"""
import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_pairs import ACTED, NATURAL, PAIR_NAMES, PAIRS, ba  # noqa: E402

KINDS_MAIN = ("audio_wavlm", "video", "text_roberta")
KINDS_ROBUST = ("audio_hubert",)
WORLDS = {"AV": (("audio_wavlm", "video"), "fuse-audio-video"),
          "AVT": (("audio_wavlm", "video", "text_roberta"), "fuse-audio-video-text")}
E13 = ROOT / "results" / "explore_relation" / "snapshots" / "20261002-1339_e13" / "e13_pairs.csv"
ANGER = [PAIR_NAMES[i] for i, (k, j) in enumerate(PAIRS) if 3 in (k, j)]                      # 怒りを含む 3 対（P84-1）
AUDIO_WEAK = [PAIR_NAMES[i] for i, (k, j) in enumerate(PAIRS) if (k, j) in ((1, 3), (0, 2))]  # 喜び–怒り，平静–悲しみ（P84-2）
ALL = "ALL"                 # 全ソースの合併（発話数の比重．results/multisource_scores_<kind>/<t>.npz の p_tgt_all）
SCRIPT = "IEMOCAP-script"   # 台詞の共有があるので主の集計から外す（副に入れる）
TOL = 1e-6
NP = len(PAIRS)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def corp(g):
    return GROUP_SPEC[g][0]


def ctype(g):
    return "acted" if g in ACTED else ("natural" if g in NATURAL else "other")


def unit_of(t):
    return "IEMOCAP(script)" if t == SCRIPT else corp(t)


def sources_for(kind, t, pool=None):
    pool = groups_for(kind) if pool is None else [g for g in pool if g in groups_for(kind)]
    return sorted(s for s in pool if s != t and corp(s) != corp(t))


def binom_majority(n, p=0.5):
    """n 単位で，差が無いとき（各単位が独立に確率 p で正）に過半が正になる確率．"""
    k = n // 2 + 1
    return float(1 - stats.binom.cdf(k - 1, n, p)) if n > 0 else np.nan


# ---------------------------------------------------------------- 点数と分割

def log_scores(kind, s, t):
    if s == ALL:
        z = np.load(ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz")
        p = z["p_tgt_all"].astype(np.float64)
        return np.log(np.clip(p, 1e-12, 1.0)), sorted(z["src_names"].tolist())
    z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
    return z[f"t_{t}"].astype(np.float64), None


def check_mixture_rows(kind, t, y, logp):
    """合併の点数 p_tgt_all の行がターゲットの発話一覧の順と合うか：保存されたテストの添字でラベルと argmax を照合する（反復 0〜9）．"""
    z = np.load(ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz")
    for r in range(10):
        if f"test_idx_r{r}" not in z.files:
            continue
        ix = z[f"test_idx_r{r}"]
        if not np.array_equal(y[ix], z[f"y_test_r{r}"]):
            raise SystemExit(f"{kind} {t} r{r}：合併のテストのラベルが発話一覧と合わない")
        if not np.array_equal(logp[ix].argmax(1), z[f"s_all_r{r}"].argmax(1)):
            raise SystemExit(f"{kind} {t} r{r}：合併のテストの argmax が p_tgt_all の行と合わない")


def make_splits(df, target, seeds):
    """(名前, half∈{0,1}, 半分 0 の群) の一覧と，分割の単位（session か speaker）と，ブートストラップの塊．"""
    grp = df.speaker_group.astype(str)
    if target.startswith("IEMOCAP"):
        sess = grp.str[:5]
        S = sorted(sess.unique())
        out = [(f"ses{a[3:]}+{b[3:]}", np.where(sess.isin((a, b)), 0, 1), (a, b)) for a, b in itertools.combinations(S, 2)]
        return out, "session", sess.to_numpy()
    counts = grp.value_counts()
    names = counts.index.to_numpy()
    out = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        load, assign = [0, 0], {}
        for i in rng.permutation(len(names)):
            h = 0 if load[0] <= load[1] else 1
            assign[names[i]] = h
            load[h] += int(counts.iloc[i])
        half = grp.map(assign).to_numpy().astype(int)
        out.append((f"seed{sd}", half, tuple(sorted(n for n, h in assign.items() if h == 0))))
    return out, "speaker", grp.to_numpy()


# ---------------------------------------------------------------- 対ごとの量

def wstat(z, mk, ml, w=None):
    """重み w（復元抽出の多重度．None なら 1）での d，o，平均，標準偏差，分散比．"""
    if w is None:
        zk, zl, wk, wl = z[mk], z[ml], None, None
        nk, nl = len(zk), len(zl)
        if nk < 2 or nl < 2:
            return None
        mk_, ml_ = zk.mean(), zl.mean()
        vk, vl = zk.var(ddof=1), zl.var(ddof=1)
    else:
        wk, wl = w[mk], w[ml]
        nk, nl = wk.sum(), wl.sum()
        if nk < 2 or nl < 2:
            return None
        zk, zl = z[mk], z[ml]
        mk_, ml_ = (wk * zk).sum() / nk, (wl * zl).sum() / nl
        vk, vl = (wk * (zk - mk_) ** 2).sum() / (nk - 1), (wl * (zl - ml_) ** 2).sum() / (nl - 1)
    sd = np.sqrt((vk + vl) / 2)
    if not np.isfinite(sd) or sd <= 0:
        return None
    d = (mk_ - ml_) / sd
    return dict(d=d, o=-(mk_ + ml_) / 2 / sd, a=100 * ba(d, 0), mean_k=mk_, mean_l=ml_, sd=sd,
                var_ratio=vk / vl if vl > 0 else np.nan, n_k=int(mk.sum()), n_l=int(ml.sum()))


def auc_a(z, mk, ml):
    zk, zl = z[mk], z[ml]
    if len(zk) < 2 or len(zl) < 2:
        return np.nan, np.nan
    auc = roc_auc_score(np.r_[np.ones(len(zk)), np.zeros(len(zl))], np.r_[zk, zl])
    auc = float(np.clip(auc, 1e-6, 1 - 1e-6))
    d = np.sqrt(2) * stats.norm.ppf(auc)
    return d, 100 * ba(d, 0)


def best_thr_acc(z, mk, ml):
    """全発話で最良の閾値を引いたときの 2 感情の平均の正解率（内側．上限）．"""
    zk, zl = z[mk], z[ml]
    allz = np.r_[zk, zl]
    lab = np.r_[np.ones(len(zk)), np.zeros(len(zl))]
    labs = lab[np.argsort(allz, kind="stable")]
    ck, cl = np.cumsum(labs), np.cumsum(1 - labs)
    acc = 0.5 * ((len(zk) - ck) / len(zk) + cl / len(zl))
    return 100 * max(0.5, float(acc.max()))


def thr_acc(z, mk, ml, thr):
    return 100 * 0.5 * ((z[mk] > thr).mean() + (z[ml] < thr).mean())


def cell(z, mk, ml, full=False):
    r = wstat(z, mk, ml)
    if r is None:
        r = dict(d=np.nan, o=np.nan, a=np.nan, mean_k=np.nan, mean_l=np.nan, sd=np.nan, var_ratio=np.nan, n_k=int(mk.sum()), n_l=int(ml.sum()))
    r["d_auc"], r["a_auc"] = auc_a(z, mk, ml)
    r["a_emp_in"] = best_thr_acc(z, mk, ml) if (full and np.isfinite(r["a"])) else np.nan
    return r


class Block:
    """1 つの (kind, ターゲット) の点数・ラベル・分割．(C) では共通の発話に絞った版を作る．"""

    def __init__(self, kind, t, seeds, pool=None, idx=None, base=None):
        self.kind, self.t = kind, t
        df, _ = load_any(t, kind)
        if idx is not None:
            df = df.iloc[idx].reset_index(drop=True)
        self.df = df
        self.y = df.label_4.to_numpy()
        base = self if base is None else base
        if base is self:
            self.splits, self.split_unit, self.cluster = make_splits(df, t, seeds)
        else:
            self.splits, self.split_unit, self.cluster = base.splits, base.split_unit, base.cluster
        self.srcs = sources_for(kind, t, pool)
        self.Z, self.P = {}, {}
        n = len(df)
        for s in self.srcs + [ALL]:
            L, names = log_scores(kind, s, t)
            if idx is not None:
                L = L[idx]
            if len(L) != n:
                raise SystemExit(f"{kind} {s}→{t}：点数 {len(L)} 行と発話 {n} 本が合わない")
            if s == ALL and names != sources_for(kind, t):
                raise SystemExit(f"{kind} {t}：合併のソース {names} が候補 {sources_for(kind, t)} と合わない")
            if s == ALL and idx is None:
                check_mixture_rows(kind, t, self.y, L)
            self.P[s] = L
            self.Z[s] = np.stack([L[:, k] - L[:, j] for k, j in PAIRS], 1)
        self.masks = [(self.y == k, self.y == j) for k, j in PAIRS]

    def cells(self, s, sel, full=False):
        return [cell(self.Z[s][:, p], self.masks[p][0] & sel, self.masks[p][1] & sel, full) for p in range(NP)]

    def uar(self, s, sel):
        pred = self.P[s].argmax(1)
        return 100 * np.mean([(pred[(self.y == c) & sel] == c).mean() for c in range(4) if ((self.y == c) & sel).any()])


# ---------------------------------------------------------------- (A) 表

def cell_rows(blk):
    rows = []
    allm = np.ones(len(blk.y), bool)
    for s in blk.srcs + [ALL]:
        for p, r in enumerate(blk.cells(s, allm, full=True)):
            rows.append(dict(kind=blk.kind, target=blk.t, source=s, split="full", half=-1, pair=PAIR_NAMES[p], **r))
        for name, half, _ in blk.splits:
            for h in (0, 1):
                for p, r in enumerate(blk.cells(s, half == h)):
                    rows.append(dict(kind=blk.kind, target=blk.t, source=s, split=name, half=h, pair=PAIR_NAMES[p], **r))
    return rows


def split_rows(blk):
    seen = set()
    rows = []
    for name, half, g0 in blk.splits:
        seen.add(g0)
        # 群の名前に '#' が入ると pd.read_csv(comment="#") で行が切れるので置き換える（MELD の話者名）
        rows.append(dict(kind=blk.kind, target=blk.t, split=name, unit=blk.split_unit, n_half0=int((half == 0).sum()),
                         n_half1=int((half == 1).sum()), n_groups_half0=len(g0), n_distinct_so_far=len(seen),
                         groups_half0="|".join(g0).replace("#", "＃")))
    return rows


# ---------------------------------------------------------------- (B) 選んで測る

def argmax_name(values):
    """{名前: 値} で最大の名前．NaN は除く．同点は名前の辞書順で先のもの．無ければ None．"""
    best, bv = None, -np.inf
    for k in sorted(values):
        v = values[k]
        if np.isfinite(v) and v > bv:
            best, bv = k, v
    return best


def select_eval(blk, A, B):
    """半分 A で選び，半分 B で測る．戻り値：対ごとの行と，まとめの行と，ブートストラップ用の選択．"""
    cA = {s: blk.cells(s, A) for s in blk.srcs + [ALL]}
    cB = {s: blk.cells(s, B) for s in blk.srcs + [ALL]}
    star = [argmax_name({s: cA[s][p]["a"] for s in blk.srcs}) for p in range(NP)]
    star_auc = [argmax_name({s: cA[s][p]["a_auc"] for s in blk.srcs}) for p in range(NP)]
    circ = argmax_name({s: (np.mean([cA[s][p]["a"] for p in range(NP)]) if all(np.isfinite(cA[s][p]["a"]) for p in range(NP)) else np.nan) for s in blk.srcs})
    circ_uar = argmax_name({s: blk.uar(s, A) for s in blk.srcs})
    ok = circ is not None and all(x is not None for x in star)
    pr = []
    for p in range(NP):
        s = star[p]
        aB = cB[s][p]["a"] if s else np.nan
        thr = (cA[s][p]["mean_k"] + cA[s][p]["mean_l"]) / 2 if s else np.nan
        thr_all = (cA[ALL][p]["mean_k"] + cA[ALL][p]["mean_l"]) / 2
        pr.append(dict(pair=PAIR_NAMES[p], s_star_A=s, s_star_auc_A=star_auc[p], s_circ_A=circ, s_circ_uar_A=circ_uar,
                       a_B_star=aB, a_B_all=cB[ALL][p]["a"], a_B_circ=cB[circ][p]["a"] if circ else np.nan,
                       gap_all=aB - cB[ALL][p]["a"], gap_best=(aB - cB[circ][p]["a"]) if circ else np.nan,
                       a_auc_B_star=cB[s][p]["a_auc"] if s else np.nan, a_auc_B_all=cB[ALL][p]["a_auc"],
                       gap_all_auc=(cB[star_auc[p]][p]["a_auc"] - cB[ALL][p]["a_auc"]) if star_auc[p] else np.nan,
                       a_emp_B_star=thr_acc(blk.Z[s][:, p], blk.masks[p][0] & B, blk.masks[p][1] & B, thr) if s else np.nan,
                       a_emp_B_all=thr_acc(blk.Z[ALL][:, p], blk.masks[p][0] & B, blk.masks[p][1] & B, thr_all),
                       star_ne_circ=(s != circ) if ok else np.nan))
    P = pd.DataFrame(pr)
    summ = dict(G_all=P.gap_all.mean() if ok else np.nan, G_best=P.gap_best.mean() if ok else np.nan,
                G_all_auc=P.gap_all_auc.mean() if ok else np.nan,
                G_emp_all=(P.a_emp_B_star - P.a_emp_B_all).mean() if ok else np.nan,
                s_circ_A=circ, s_circ_uar_A=circ_uar, s_star_A="|".join(str(x) for x in star),
                n_distinct_star=len({x for x in star if x}))
    # UAR への橋：対ごとの 2 感情の判定（半分 A の中点の閾値）を 6 対で投票させた 4 クラスの分類器の，半分 B の UAR
    if ok:
        summ["uar_vote_star"] = vote_uar(blk, cA, star, B)
        summ["uar_vote_circ"] = vote_uar(blk, cA, [circ] * NP, B)
        summ["uar_vote_all"] = vote_uar(blk, cA, [ALL] * NP, B)
        summ["uar_all_raw"] = blk.uar(ALL, B)
        summ["uar_circ_uar_raw"] = blk.uar(circ_uar, B)
        summ["G_UAR"] = summ["uar_vote_star"] - summ["uar_vote_all"]
    else:
        for k in ("uar_vote_star", "uar_vote_circ", "uar_vote_all", "uar_all_raw", "uar_circ_uar_raw", "G_UAR"):
            summ[k] = np.nan
    return P, summ, (star if ok else None)


def vote_uar(blk, cA, choice, B):
    n = len(blk.y)
    votes, margin = np.zeros((n, 4)), np.zeros((n, 4))
    for p, (k, j) in enumerate(PAIRS):
        s = choice[p]
        c = cA[s][p]
        if not np.isfinite(c["a"]):
            return np.nan
        m = (blk.Z[s][:, p] - (c["mean_k"] + c["mean_l"]) / 2) / c["sd"]
        votes[:, k] += m > 0
        votes[:, j] += m <= 0
        margin[:, k] += np.maximum(m, 0)
        margin[:, j] += np.maximum(-m, 0)
    pred = (votes + 1e-3 * margin / (1 + margin.max())).argmax(1)
    return 100 * np.mean([(pred[(blk.y == c) & B] == c).mean() for c in range(4) if ((blk.y == c) & B).any()])


def boot_gain(blk, picks, n_draws, rng):
    """picks：(B のマスク, 対ごとの選んだソース, 相手のソース) の一覧（分け方 × 向き）．B の塊を復元抽出して G を引き直す．"""
    draws = []
    for _ in range(n_draws):
        B, star, ref = picks[rng.integers(len(picks))]
        cl = blk.cluster[B]
        uniq = np.unique(cl)
        samp = rng.choice(uniq, len(uniq), replace=True)
        mult = pd.Series(samp).value_counts()
        w = np.zeros(len(blk.y))
        w[B] = pd.Series(cl).map(mult).fillna(0).to_numpy()
        vals = []
        for p in range(NP):
            a1 = wstat(blk.Z[star[p]][:, p], blk.masks[p][0], blk.masks[p][1], w)
            a0 = wstat(blk.Z[ref[p]][:, p], blk.masks[p][0], blk.masks[p][1], w)
            vals.append((a1["a"] - a0["a"]) if (a1 and a0) else np.nan)
        draws.append(np.nanmean(vals) if np.isfinite(vals).any() else np.nan)
    d = np.array(draws)
    d = d[np.isfinite(d)]
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(d)) if len(d) else (np.nan, np.nan, 0)


def gain_for_block(blk, n_boot, rng):
    prow, grow, picks = [], [], []
    for name, half, _ in blk.splits:
        for hs, he in ((0, 1), (1, 0)):
            A, B = half == hs, half == he
            P, summ, star = select_eval(blk, A, B)
            for r in P.to_dict("records"):
                prow.append(dict(kind=blk.kind, target=blk.t, unit=unit_of(blk.t), split=name, dir=f"{hs}>{he}", **r))
            grow.append(dict(kind=blk.kind, target=blk.t, unit=unit_of(blk.t), split=name, dir=f"{hs}>{he}", **summ))
            if star is not None:
                picks.append((B, star, [ALL] * NP))
    lo, hi, nb = boot_gain(blk, picks, n_boot, rng) if picks else (np.nan, np.nan, 0)
    # δ_u：この単位の (分け方, ソース, 対) での |a_h0 − a_h1| の中央値．a_AUC 版も
    diffs, diffs_auc = [], []
    for name, half, _ in blk.splits:
        for s in blk.srcs:
            c0, c1 = blk.cells(s, half == 0), blk.cells(s, half == 1)
            diffs += [abs(c0[p]["a"] - c1[p]["a"]) for p in range(NP) if np.isfinite(c0[p]["a"]) and np.isfinite(c1[p]["a"])]
            diffs_auc += [abs(c0[p]["a_auc"] - c1[p]["a_auc"]) for p in range(NP) if np.isfinite(c0[p]["a_auc"]) and np.isfinite(c1[p]["a_auc"])]
    delta_u = float(np.median(diffs)) if diffs else np.nan
    delta_u_auc = float(np.median(diffs_auc)) if diffs_auc else np.nan
    # 全発話で選んで全発話で測った値（上限ではない）
    allm = np.ones(len(blk.y), bool)
    _, sf, _ = select_eval(blk, allm, allm)
    g = pd.DataFrame(grow)
    unit = dict(kind=blk.kind, target=blk.t, unit=unit_of(blk.t), main=blk.t != SCRIPT, n_splits=len(blk.splits),
                n_distinct_splits=len({g0 for _, _, g0 in blk.splits}), split_unit=blk.split_unit,
                G_all=g.G_all.mean(), G_all_lo=lo, G_all_hi=hi, n_boot=nb, G_all_min=g.G_all.min(), G_all_max=g.G_all.max(),
                G_best=g.G_best.mean(), G_all_auc=g.G_all_auc.mean(), G_emp_all=g.G_emp_all.mean(), G_UAR=g.G_UAR.mean(),
                uar_vote_star=g.uar_vote_star.mean(), uar_vote_all=g.uar_vote_all.mean(), uar_all_raw=g.uar_all_raw.mean(),
                uar_vote_circ=g.uar_vote_circ.mean(), uar_circ_uar_raw=g.uar_circ_uar_raw.mean(),
                G_full_all=sf["G_all"], G_full_best=sf["G_best"], delta_u=delta_u, delta_u_auc=delta_u_auc, n_distinct_star_full=sf["n_distinct_star"])
    return prow, grow, unit


def best_full_rows(blk):
    """全発話の表：対ごとの最良のソース s*，6 対の平均で最良 s°，合併との差．"""
    allm = np.ones(len(blk.y), bool)
    c = {s: blk.cells(s, allm) for s in blk.srcs + [ALL]}
    star = [argmax_name({s: c[s][p]["a"] for s in blk.srcs}) for p in range(NP)]
    star_auc = [argmax_name({s: c[s][p]["a_auc"] for s in blk.srcs}) for p in range(NP)]
    circ = argmax_name({s: (np.mean([c[s][p]["a"] for p in range(NP)]) if all(np.isfinite(c[s][p]["a"]) for p in range(NP)) else np.nan) for s in blk.srcs})
    rows = []
    for p in range(NP):
        s = star[p]
        rows.append(dict(kind=blk.kind, target=blk.t, unit=unit_of(blk.t), target_type=ctype(blk.t), pair=PAIR_NAMES[p],
                         s_star=s, s_star_type=ctype(s) if s else None, a_star=c[s][p]["a"] if s else np.nan,
                         s_circ=circ, a_circ=c[circ][p]["a"] if circ else np.nan, a_all=c[ALL][p]["a"],
                         gap_best=(c[s][p]["a"] - c[circ][p]["a"]) if (s and circ) else np.nan,
                         gap_all=(c[s][p]["a"] - c[ALL][p]["a"]) if s else np.nan,
                         star_ne_circ=(s != circ) if (s and circ) else np.nan, s_star_auc=star_auc[p],
                         agree_auc=(s == star_auc[p]) if s else np.nan, n_sources=len(blk.srcs), anger_pair=PAIR_NAMES[p] in ANGER))
    return rows, c


def perm_p1(full_cells_by_target, n_perm, rng):
    """P84-1 の置換の基準：ターゲットと対ごとに独立にソース名を並べ替え，s* ≠ s° の割合（怒り − その他）の帰無分布．"""
    def share_diff(tabs):
        ne = {True: [], False: []}
        for t, A in tabs.items():          # A: (ソース × 対) の a
            srcs = sorted(A)
            M = np.array([[A[s][p] for p in range(NP)] for s in srcs])
            if not np.isfinite(M).all():
                continue
            circ = srcs[int(np.argmax(M.mean(1)))]
            for p in range(NP):
                ne[PAIR_NAMES[p] in ANGER].append(srcs[int(np.argmax(M[:, p]))] != circ)
        return (np.mean(ne[True]) if ne[True] else np.nan) - (np.mean(ne[False]) if ne[False] else np.nan)
    obs = share_diff(full_cells_by_target)
    null = []
    for _ in range(n_perm):
        tabs = {}
        for t, A in full_cells_by_target.items():
            srcs = sorted(A)
            M = np.array([[A[s][p] for p in range(NP)] for s in srcs])
            for p in range(NP):
                M[:, p] = M[rng.permutation(len(srcs)), p]
            tabs[t] = {s: list(M[i]) for i, s in enumerate(srcs)}
        null.append(share_diff(tabs))
    null = np.array(null)
    null = null[np.isfinite(null)]
    return dict(p1_obs_diff=obs, p1_null_mean=float(null.mean()) if len(null) else np.nan,
                p1_null_p95=float(np.percentile(null, 95)) if len(null) else np.nan,
                p1_null_share_ge_obs=float((null >= obs).mean()) if len(null) and np.isfinite(obs) else np.nan, n_perm=len(null))


def summarize_kind(kind, units, prow, best, p1):
    u = units[(units.kind == kind) & units.main]
    n = len(u)
    out = dict(kind=kind, n_units=n, units="|".join(u.unit),
               n_units_G_all_ge_delta=int((u.G_all >= u.delta_u).sum()), n_units_G_all_pos=int((u.G_all > 0).sum()),
               binom_majority_pos=binom_majority(n), G_all_median=float(u.G_all.median()), delta_u_median=float(u.delta_u.median()),
               R1_room=bool((u.G_all >= u.delta_u).sum() > n / 2),
               n_units_G_all_auc_ge_delta=int((u.G_all_auc >= u.delta_u_auc).sum()), n_units_G_emp_ge_delta=int((u.G_emp_all >= u.delta_u).sum()),
               G_UAR_median=float(u.G_UAR.median()), n_units_G_UAR_pos=int((u.G_UAR > 0).sum()), G_emp_all_median=float(u.G_emp_all.median()))
    # 副：相手を最良の 1 ソースにした G_best（δ_u で数える）．cebd64b の規則そのものの値は凍結した台本 analyze_source_pairs_e84_cebd64b.py の出力で並べる
    out.update(old_G_best_median=float(u.G_best.median()), old_n_units_G_best_ge_delta_u=int((u.G_best >= u.delta_u).sum()))
    # P84-1：全発話（E13 の d の再集計）と，外の半分で確かめた版（gap_best ≥ δ_u の s* ≠ s°）
    b = best[(best.kind == kind) & (best.unit != "IEMOCAP(script)")]
    sh = b.groupby(["unit", "pair", "anger_pair"]).star_ne_circ.mean().reset_index().groupby("anger_pair").star_ne_circ.mean()
    pr = prow[(prow.kind == kind) & (prow.unit != "IEMOCAP(script)")].merge(u[["unit", "delta_u"]], on="unit")
    pr = pr.assign(anger_pair=pr.pair.isin(ANGER), ne_ge=(pr.star_ne_circ.astype(float) > 0) & (pr.gap_best >= pr.delta_u))
    sh2 = pr.groupby(["unit", "pair", "anger_pair"]).ne_ge.mean().reset_index().groupby("anger_pair").ne_ge.mean()
    out.update(p1_full_share_ne_anger=float(sh.get(True, np.nan)), p1_full_share_ne_other=float(sh.get(False, np.nan)),
               p1_oos_share_ne_ge_delta_anger=float(sh2.get(True, np.nan)), p1_oos_share_ne_ge_delta_other=float(sh2.get(False, np.nan)),
               agree_auc_share=float(b.agree_auc.mean()), **p1)
    return out


# ---------------------------------------------------------------- (C) 世界ごとのモダリティ

def world_blocks(world, t, seeds, restrict):
    singles, fused = WORLDS[world]
    kinds = tuple(singles) + (fused,)
    dfs = {k: load_any(t, k)[0] for k in kinds}
    common = None
    for k in kinds:
        ids = set(dfs[k].sample_id)
        common = ids if common is None else common & ids
    common = sorted(common)
    idx = {k: pd.Index(dfs[k].sample_id).get_indexer(common) for k in kinds}
    if any((ix < 0).any() for ix in idx.values()):
        raise SystemExit(f"{world} {t}：sample_id の対応が取れない")
    pool = groups_for(fused) if restrict else None
    base = Block(fused, t, seeds, pool=None, idx=idx[fused])
    blocks = {fused: base}
    for k in singles:
        b = Block(k, t, seeds, pool=pool, idx=idx[k], base=base)
        if not np.array_equal(b.y, base.y):
            raise SystemExit(f"{world} {t} {k}：ラベルが融合と合わない")
        blocks[k] = b
    return blocks, len(common)


def world_rows(world, seeds, restrict, n_boot, rng):
    singles, fused = WORLDS[world]
    kinds = tuple(singles) + (fused,)
    ver = "main" if restrict else "allsrc"
    crow, srow, urow = [], [], []
    for t in groups_for(fused):
        blocks, n_common = world_blocks(world, t, seeds, restrict)
        base = blocks[fused]
        for k in kinds:
            for r in cell_rows(blocks[k]):
                crow.append(dict(world=world, version=ver, **r))
        picks, rrows = [], []
        for name, half, _ in base.splits:
            for hs, he in ((0, 1), (1, 0)):
                A, B = half == hs, half == he
                circA, cA, cB = {}, {}, {}
                for k in kinds:
                    b = blocks[k]
                    cA[k] = {s: b.cells(s, A) for s in b.srcs}
                    cB[k] = {s: b.cells(s, B) for s in b.srcs}
                    circA[k] = argmax_name({s: (np.mean([cA[k][s][p]["a"] for p in range(NP)]) if all(np.isfinite(cA[k][s][p]["a"]) for p in range(NP)) else np.nan) for s in b.srcs})
                if any(circA[k] is None for k in kinds):
                    continue
                aA = {k: [cA[k][circA[k]][p]["a"] for p in range(NP)] for k in kinds}
                aB = {k: [cB[k][circA[k]][p]["a"] for p in range(NP)] for k in kinds}
                aucB = {k: [cB[k][circA[k]][p]["a_auc"] for p in range(NP)] for k in kinds}

                def emp_B(k, p):   # 半分 A の中点の閾値を，半分 B で測る
                    c = cA[k][circA[k]][p]
                    b = blocks[k]
                    return thr_acc(b.Z[circA[k]][:, p], base.masks[p][0] & B, base.masks[p][1] & B, (c["mean_k"] + c["mean_l"]) / 2)
                empB = {k: [emp_B(k, p) for p in range(NP)] for k in kinds}
                mstar = [argmax_name({k: aA[k][p] for k in singles}) for p in range(NP)]        # 単モダリティの最良（R_oos 用）
                mstar_all = [argmax_name({k: aA[k][p] for k in kinds}) for p in range(NP)]      # 融合を含む最良（P84-2 用．cebd64b と同じ候補）
                r_oos = [aB[mstar[p]][p] - aB[fused][p] for p in range(NP)]
                r_oos_auc = [aucB[mstar[p]][p] - aucB[fused][p] for p in range(NP)]
                r_oos_emp = [empB[mstar[p]][p] - empB[fused][p] for p in range(NP)]
                # cebd64b の規則の近似：測る半分 B で単モダリティの最良を選び，0 で切る（向きごと．cebd64b そのものの値は凍結した台本で別に出す）
                r_old = [max(0.0, max(aB[k][p] for k in singles) - aB[fused][p]) for p in range(NP)]
                for p in range(NP):
                    srow.append(dict(world=world, version=ver, target=t, unit=unit_of(t), split=name, dir=f"{hs}>{he}", pair=PAIR_NAMES[p],
                                     m_star_A=mstar[p], m_star_all_A=mstar_all[p], r_oos=r_oos[p], r_oos_auc=r_oos_auc[p], r_oos_emp=r_oos_emp[p],
                                     r_old=r_old[p], a_B_fused=aB[fused][p],
                                     **{f"a_B_{k}": aB[k][p] for k in singles}, **{f"s_circ_A_{k}": circA[k] for k in kinds},
                                     audio_weak_pair=PAIR_NAMES[p] in AUDIO_WEAK, n_common=n_common))
                rrows.append(dict(R_oos=np.mean(r_oos), R_oos_auc=np.mean(r_oos_auc), R_oos_emp=np.mean(r_oos_emp), R_old=np.mean(r_old)))
                picks.append((B, [(mstar[p], circA[mstar[p]]) for p in range(NP)], circA[fused]))
        # ブートストラップ（測る半分の塊を復元抽出）
        draws = []
        for _ in range(n_boot if picks else 0):
            B, ch, cf = picks[rng.integers(len(picks))]
            cl = base.cluster[B]
            uniq = np.unique(cl)
            mult = pd.Series(rng.choice(uniq, len(uniq), replace=True)).value_counts()
            w = np.zeros(len(base.y))
            w[B] = pd.Series(cl).map(mult).fillna(0).to_numpy()
            vals = []
            for p in range(NP):
                k, s = ch[p]
                a1 = wstat(blocks[k].Z[s][:, p], base.masks[p][0], base.masks[p][1], w)
                a0 = wstat(base.Z[cf][:, p], base.masks[p][0], base.masks[p][1], w)
                vals.append((a1["a"] - a0["a"]) if (a1 and a0) else np.nan)
            draws.append(np.nanmean(vals) if np.isfinite(vals).any() else np.nan)
        d = np.array(draws)
        d = d[np.isfinite(d)]
        # δ_u：世界の全 kind（その版の候補のソース）の (分け方, ソース, 対) での |a_h0 − a_h1| の中央値（共通の発話の上）．a_AUC 版も
        diffs, diffs_auc = [], []
        for k in kinds:
            b = blocks[k]
            for name, half, _ in b.splits:
                for s in b.srcs:
                    c0, c1 = b.cells(s, half == 0), b.cells(s, half == 1)
                    diffs += [abs(c0[p]["a"] - c1[p]["a"]) for p in range(NP) if np.isfinite(c0[p]["a"]) and np.isfinite(c1[p]["a"])]
                    diffs_auc += [abs(c0[p]["a_auc"] - c1[p]["a_auc"]) for p in range(NP) if np.isfinite(c0[p]["a_auc"]) and np.isfinite(c1[p]["a_auc"])]
        rr = pd.DataFrame(rrows)
        urow.append(dict(world=world, version=ver, target=t, unit=unit_of(t), main=t != SCRIPT, n_common=n_common,
                         n_splits=len(base.splits), n_distinct_splits=len({g0 for _, _, g0 in base.splits}),
                         R_oos=rr.R_oos.mean() if len(rr) else np.nan, R_oos_lo=float(np.percentile(d, 2.5)) if len(d) else np.nan,
                         R_oos_hi=float(np.percentile(d, 97.5)) if len(d) else np.nan, n_boot=len(d),
                         R_oos_auc=rr.R_oos_auc.mean() if len(rr) else np.nan, R_oos_emp=rr.R_oos_emp.mean() if len(rr) else np.nan,
                         R_old=rr.R_old.mean() if len(rr) else np.nan,
                         delta_u=float(np.median(diffs)) if diffs else np.nan, delta_u_auc=float(np.median(diffs_auc)) if diffs_auc else np.nan,
                         **{f"n_src_{k}": len(blocks[k].srcs) for k in kinds}))
    return crow, srow, urow


def summarize_world(world, ver, srow, urow):
    singles, fused = WORLDS[world]
    u = urow[(urow.world == world) & (urow.version == ver) & urow.main]
    n = len(u)
    s = srow[(srow.world == world) & (srow.version == ver) & (srow.unit != "IEMOCAP(script)")]
    # P84-2：融合を含む候補の中で音声が最良になった割合（半分 A で選んだ m*_all）
    sh = s.assign(is_audio=s.m_star_all_A == "audio_wavlm").groupby(["unit", "pair", "audio_weak_pair"]).is_audio.mean().reset_index().groupby("audio_weak_pair").is_audio.mean()
    sh1 = s.assign(is_audio=s.m_star_A == "audio_wavlm").groupby(["unit", "pair", "audio_weak_pair"]).is_audio.mean().reset_index().groupby("audio_weak_pair").is_audio.mean()
    return dict(world=world, version=ver, n_units=n, units="|".join(u.unit),
                n_units_R_oos_ge_delta=int((u.R_oos >= u.delta_u).sum()), n_units_R_oos_pos=int((u.R_oos > 0).sum()),
                binom_majority_pos=binom_majority(n), R_oos_median=float(u.R_oos.median()) if n else np.nan,
                R2_room=bool((u.R_oos >= u.delta_u).sum() > n / 2),
                n_units_R_oos_auc_ge_delta=int((u.R_oos_auc >= u.delta_u_auc).sum()), n_units_R_oos_emp_ge_delta=int((u.R_oos_emp >= u.delta_u).sum()),
                old_R_median=float(u.R_old.median()) if n else np.nan, old_n_units_R_ge_delta_u=int((u.R_old >= u.delta_u).sum()),
                p2_share_audio_best_weak=float(sh.get(True, np.nan)), p2_share_audio_best_other=float(sh.get(False, np.nan)),
                p2_chance_share=1 / (len(singles) + 1),
                p2_singles_share_audio_best_weak=float(sh1.get(True, np.nan)), p2_singles_share_audio_best_other=float(sh1.get(False, np.nan)))


# ---------------------------------------------------------------- main

def write(df, path, header):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {header}\n")
        df.to_csv(fh, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e84")
    ap.add_argument("--kinds", nargs="*", default=list(KINDS_MAIN + KINDS_ROBUST))
    ap.add_argument("--worlds", nargs="*", default=list(WORLDS))
    ap.add_argument("--seeds", type=int, default=5, help="IEMOCAP 以外の話者の分け方の数（IEMOCAP はセッションの 10 通り）")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="試走（seed 1，boot 50，perm 100，音声だけ，世界は AVT だけ．出力は --out か checks/E84/code_run/smoke）")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.boot, args.perm, args.kinds, args.worlds = 1, 50, 100, ["audio_wavlm"], ["AVT"]
        if args.out == "results/e84":
            args.out = "checks/E84/code_run/smoke"
    out = ROOT / args.out
    if out.exists() and any(out.glob("*.csv")):
        raise SystemExit(f"{out} に古い出力がある．消してから流す")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), out / "script_copy.py")
    seeds = list(range(args.seeds))
    rng = np.random.default_rng(84)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "exp", "measure", "data/metadata"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    header = f"E84 commit={commit} dirty={bool(dirty)} seeds={args.seeds} boot={args.boot} perm={args.perm} tie=群の名前の辞書順で先"
    prov = {"commit": commit, "dirty_exp_measure_metadata": bool(dirty), "script_sha256": sha256(Path(__file__)),
            "args": vars(args) | {"out": str(out)}, "inputs": {}, "checks": {}, "tie_rule": "群の名前の辞書順で先のもの"}
    for f in sorted((ROOT / "data" / "metadata").glob("*.csv")):
        prov["inputs"][f"data/metadata/{f.name}"] = sha256(f)

    # (A)(B)(D)
    cells, splits, prow, grow, units, best, full_tabs = [], [], [], [], [], [], {}
    for kind in args.kinds:
        full_tabs[kind] = {}
        for t in groups_for(kind):
            blk = Block(kind, t, seeds)
            cells += cell_rows(blk)
            splits += split_rows(blk)
            pr, gr, un = gain_for_block(blk, args.boot, rng)
            prow += pr
            grow += gr
            units.append(un)
            br, c = best_full_rows(blk)
            best += br
            if t != SCRIPT:
                full_tabs[kind][t] = {s: [c[s][p]["a"] for p in range(NP)] for s in blk.srcs}
            for s in blk.srcs:
                prov["inputs"][f"shift_{kind}/logits/{s}.npz"] = sha256(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
            prov["inputs"][f"multisource_scores_{kind}/{t}.npz"] = sha256(ROOT / "results" / f"multisource_scores_{kind}" / f"{t}.npz")
    cells, prow, grow, units, best = (pd.DataFrame(x) for x in (cells, prow, grow, units, best))
    # E13 との照合（全発話の d．合併の行は E13 に無い）
    if E13.exists():
        e13 = pd.read_csv(E13, comment="#")
        m = cells[(cells.split == "full") & (cells.source != ALL)][["kind", "target", "source", "pair", "d"]].merge(
            e13[["kind", "target", "source", "pair", "d"]], on=["kind", "target", "source", "pair"], suffixes=("", "_e13"))
        diff = float((m.d - m.d_e13).abs().max()) if len(m) else np.nan
        if len(m) and diff > TOL:
            raise SystemExit(f"E13 の d と合わない：最大の差 {diff:.3g} > {TOL}（{len(m)} 行）")
        prov["checks"]["e13"] = {"rows": int(len(m)), "max_abs_diff": diff}
        prov["inputs"]["e13_pairs.csv"] = sha256(E13)
    write(cells, out / "cells.csv", header)
    write(pd.DataFrame(splits), out / "splits.csv", header)
    write(prow, out / "select_pairs.csv", header)
    write(grow, out / "gain_splits.csv", header)
    write(units, out / "gain_units.csv", header)
    write(best, out / "best_full.csv", header)
    summs = []
    for kind in args.kinds:
        p1 = perm_p1(full_tabs[kind], args.perm, rng)
        sm = summarize_kind(kind, units, prow, best, p1)
        summs.append(sm)
    write(pd.DataFrame(summs), out / "summary_kind.csv", header)
    if "audio_wavlm" in args.kinds:
        b = best[(best.kind == "audio_wavlm") & (best.unit != "IEMOCAP(script)")]
        d = b.groupby(["target_type", "pair"]).apply(
            lambda g: pd.Series(dict(n=len(g), share_star_acted=(g.s_star_type == "acted").mean(),
                                     share_star_natural=(g.s_star_type == "natural").mean(),
                                     share_circ_acted=(g.s_circ.map(ctype) == "acted").mean()))).reset_index()
        write(d, out / "source_type_audio.csv", header)

    # (C)
    wc, ws, wu, wsum = [], [], [], []
    for w in args.worlds:
        for restrict in (True, False):
            c, s, u = world_rows(w, seeds, restrict, args.boot, rng)
            wc += c
            ws += s
            wu += u
        singles, fused = WORLDS[w]
        for k in tuple(singles) + (fused,):
            for g in groups_for(k):
                prov["inputs"][f"shift_{k}/logits/{g}.npz"] = sha256(ROOT / "results" / f"shift_{k}" / "logits" / f"{g}.npz")
            for t in groups_for(fused):
                prov["inputs"][f"multisource_scores_{k}/{t}.npz"] = sha256(ROOT / "results" / f"multisource_scores_{k}" / f"{t}.npz")
    if wc:
        wc, ws, wu = (pd.DataFrame(x) for x in (wc, ws, wu))
        write(wc, out / "modality_cells.csv", header)
        write(ws, out / "modality_select.csv", header)
        write(wu, out / "modality_units.csv", header)
        for w in args.worlds:
            for ver in ("main", "allsrc"):
                wsum.append(summarize_world(w, ver, ws, wu))
        write(pd.DataFrame(wsum), out / "summary_world.csv", header)

    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out, "| E13 check:", prov["checks"].get("e13"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
