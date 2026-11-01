"""【本番の感度】E81（CPU）：G3 の分割（`exp/g3_split.py` の出力）と今の分割（E79 の段 (c)・(d) の出力）で「手法 − none」を比べ，
E81 の 4 節（関門 1 の後に書き直した版．コミット 4a52140）の読み方を機械的に当てる。

設計の正本は `docs/exploratory-analyses.md` の E81 の 4・5 節．E79 の関数（`exp/analyze_e79.py` の boot_draws・boot_rep・pair_moments・lda_sep・
jk_interval・judge_unit・cluster_labels・statements_p5・Store）をそのまま使う．値は画面に出さない（行の数とファイルの場所だけ）。

単位：E79 の P5 の単位（世界，生徒，K ∈ {0, 5}，コーパス）．E81 が測り直すのは IEMOCAP（即興と台本を 1 コーパス）と MELD．
  Δ_cur ＝ 今の分割の「手法 − none」（E79 の measure の点の値と複製．`--e77` などを渡せば同じ手順で作り直したものと照合する），
  Δ_G3 ＝ E81 の分割の同じ量（E81 の出力から E79 と同じ手順で作る），ΔΔ ＝ Δ_G3 − Δ_cur（同じ抽出ごとの差）．
主の読み方（4 節）：E79 の `judge/units.csv` の P5-1・P5-2・P5-3・P5-3u の主の単位のうち IEMOCAP と MELD を E81 の判定に差し替え，
  MOSEI・AFEW・CREMA-D は E79 の判定のまま，E79 の `statements_p5` と同じ母数（世界の全コーパス）で文の鍵を作り直し，E79 の文の鍵と比べる．
  P5-3u は E79 に文の行が無いので，P5-1・P5-2 と同じ 3 つの区分の鍵（none が同じ水準か上回った／oursx が上回った／判定できない）を作る（impl.md に記録）．
  2 コーパスだけの鍵は記述（statements_2corpus.csv）．
ΔΔ の判定：ΔΔ を E79 と同じ δ（UAR は δ_u，向きの損・位置の損は δ_d）と judge_unit で判定する．Δ_cur と Δ_G3 の判定の移りを，
  変わらない・両方で判定できない（「同じ」に数えない）・隣の区分への移り・向きの逆転（上回った ↔ 下回った）に分ける．
  選択が両方の腕で同じ設定（none どうしなど）の単位は ΔΔ が定義から 0 なので，数から除いて別に数える．台本の反復 3・8 を除いた版も作る．
書き方の規則（4 節）：(i) 作り直した鍵が全部で E79 の鍵と同じ，(ii) oursx − none の UAR（P5-3u）の ΔΔ が IEMOCAP と MELD の単位
  （定義から 0 の単位を除く）の過半で「同じ水準」で，Δ の判定の向きの逆転が 0．両方を満たすときだけ限定を外す（rule.json）．
区間：主は E79 と同じ（人物と反復の 2 段の百分位．G_eff < 20 は人物の jackknife の t 区間との和集合）．感度は E79 の cluster_boot と同じ
  （セッション・対話の塊の百分位，塊が 10 未満なら塊の jackknife の t，主の区間との和集合）．ほかの感度：IEMOCAP を即興だけにした版，台本の反復 3・8 を除いた版，K ごとの表．
模型に依る単位の印（a_auc，e_model）は E81 の行で作り直す（P5-3 と向きの損の行）．位置の損の行には E75 の注を付ける．逆の順序の量（dirloss_rev・posloss_rev）は
  D が分割で違うので分割の間で比べない（rows_*.csv.gz に残すだけ）。

出力（--out．既定 results/e81_analysis/）：units_dd.csv（単位ごとの Δ_cur・Δ_G3・ΔΔ と判定），transitions.csv（移りと ΔΔ の判定の数），
  statements_main.csv（主の読み方の鍵），statements_sens.csv（感度の行の鍵），statements_2corpus.csv（記述），rule.json（(i)・(ii) と結論），
  rows_g3_<世界>.csv.gz・rows_cur_<世界>.csv.gz（反復ごとの点の値），checks.csv，identity_same_split.csv，provenance.json

段取り：E79 の段 (d)（judge）の後に流す（主の読み方は `judge/units.csv`・`judge/statements.csv` を読む）．
結論を書かない条件（gate-code の I12・I14）：`checks.csv` に偽の行がある，`identity_same_split.csv` に不一致がある，2 世界 × 3 群 × 反復 0〜9 がそろわない，
  E79 の judge の出力が無い，E81 の出力が試しの印（--allow-incomplete・--allow-dirty・--check）つきか照合を飛ばした記録（unchecked）を持つ．
  このとき rule.json の decision は null で，終了の値は 1．rule.json に判定の範囲（世界・群・反復）と偽になった照合の名前を書く．

    ./.venv/bin/python exp/analyze_e81.py --e81 fuse-audio-video-text=results/e81_fuse-audio-video-text fuse-audio-video=results/e81_fuse-audio-video \\
        --e79 results/e79 --e77 世界=場所 … --e78 世界=場所 … --e80c 世界=場所 …
  --e77・--e78・--e80c（E77・E78・E80c の本番の出力）は，台本の反復 3・8 を除いた版の Δ_cur と，学習側が変わらない単位の照合に使う．
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import analyze_e79 as A9  # noqa: E402
from analyze_e79 import (CLUSTER_COLS, CLUSTER_JK_MAX, EMPTY, EXCLUDED, GEFF_MIN, METRIC, MODEL_UNIT_FRAC, QN, SEEDED, boot_draws,  # noqa: E402
                         boot_rep, corp, jk_interval, judge_unit, lda_sep, majority, pair_moments, seed_of, strip_seed, student_feats)
from data import groups_for, load_any  # noqa: E402
from explore_pairs import ba  # noqa: E402
from multimodal_distill import WORLDS  # noqa: E402
from shift import uar as shift_uar  # noqa: E402

WORLDS81 = ("fuse-audio-video-text", "fuse-audio-video")
TARGETS = ("IEMOCAP-impro", "IEMOCAP-script", "MELD")
CORPORA81 = ("IEMOCAP", "MELD")
KS = ("0", "5")
SELS = ("oursx", "kd", "groupB")
SAME_SPLIT = {("IEMOCAP-script", 3), ("IEMOCAP-script", 8)}                     # 学習側が変わらない単位（AVT・AV）
NONE = ("e77", "none", "-", "0", "-")
TOL_POINT = 1e-4                                                              # 学習側が変わらない単位の反復ごとの点の値（UAR は割合，損はポイント）．1 発話の予測の変化（約 2e-3）を拾う
TOL_LOGP = 1e-4                                                               # 同じ単位の対数の確率．凸の解き（dist・kd・lp）は束の中身で 2.4e-5 まで動いた（checks/E81/impl.md）
TOL_CUR = 1e-6                                                                # E79 の measure と，本番の出力から作り直した今の分割の値の差（同じ手順なので浮動小数点の差だけ）
FAMS = {  # 族 → (比べ, x の族, y の族（None は none）, 量, 教師なしの扱い)．E79 の stage_judge の P5 の主の行と同じ
    "P5-1": ("oursx-kd", "oursx", "kd", "uar", "exclude"),
    "P5-2": ("oursx-groupB", "oursx", "groupB", "uar", "count"),
    "P5-3": ("none-oursx", "oursx", None, "dir", "exclude"),
    "P5-3u": ("oursx-none", "oursx", None, "uar", "count"),
}
DTAB = [("uar", "uar"), ("dirloss", "dir"), ("posloss", "pos")]               # Δ の表の量（手法 − none）
NOTE_POS = "位置の損は 2 つの山の模型で測った量で，模型の誤差と同じ大きさ（E75 の決まり）．主張の根拠に使わない"
NOTE_DETECT = "この設計で検出できる ΔΔ はおよそ 1.5δ 以上（E79 の 9035 行と同じ見積もり）"
VARIANTS = ("main", "ex_script38", "impro_only")                              # 単位の作り方：主，IEMOCAP 台本の反復 3・8 を除く，IEMOCAP を即興だけ
NAN_PT = {q: np.nan for q in QN + ("a_auc", "e_model")}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def kv(xs):
    return dict(x.split("=", 1) for x in (xs or []))


def read_tab(f, wave=None):
    t = pd.read_csv(f, dtype={"K": str, "param": str, "meff": str, "pnorm": str, "student": str, "method": str, "wave": str, "target": str})
    if wave is not None:
        t["wave"] = wave
    t["pnorm"] = [strip_seed(p) if m in SEEDED else p for m, p in zip(t.method, t.param)]
    t["seed"] = [seed_of(m, p) for m, p in zip(t.method, t.param)]
    return t


def chosen_pick(ch):
    """(run, T, 生徒, K, 族) → E79 の鍵 (波, 方法, pnorm, m_eff, 種の扱い)．族 None は none．候補が無い行は None（E79 の stage_judge の pick と同じ）。"""
    SEL = {(r.run, r.target, r.student, r.K, r.selector): r for r in ch[ch.variant == "incl"].itertuples(index=False)}

    def pick(run, T, st, K, sel):
        if sel is None:
            return NONE
        r = SEL.get((run, T, st, K, sel))
        if r is None or pd.isna(r.method) or r.method == EMPTY:
            return None
        if r.method == "none":
            return NONE
        return (r.wave, r.method, r.param, r.meff, "agg" if r.method in SEEDED else "-")
    return pick


# ---------------------------------------------------------------- 点の値と複製（E79 の measure と同じ手順）

class SrcStore:
    """選んだ鍵の点の値と複製（boot・jk・cl・cljk）を，E79 の stage_measure・measure_target と同じ手順で作る．
    src：T → {波: 出力の場所}（E81 の出力は {"*": 場所} で，行の wave の列で波を分ける），pool："saved"（E81 の `_pool.npy`）か "complement"（テスト以外．今の分割）．
    複製は "all"（全反復）と "ex_script38"（IEMOCAP 台本の反復 3・8 を足さない）を同じ抽出で同時に作る（抽出は話者・塊の並びと反復の数だけで決まる）。"""

    def __init__(self, run, src, pool, pick, B, seed, reps, checks, name):
        self.run, self.src, self.pool_mode, self.B, self.seed, self.reps, self.name = run, src, pool, B, seed, list(reps), name
        self.point, self.rp, self.geff, self.speakers, self.clusters, self.ncl, self.rows = {}, {}, {}, {}, {}, {}, []
        self.targets = list(src)
        students = WORLDS[run] + [run]
        prep = {T: self.measure_target(T, pick, students, checks) for T in self.targets}
        by = {}
        for T in self.targets:
            by.setdefault(corp(T), []).append(T)
        for cp, Ts in by.items():
            speakers = sorted(set().union(*[prep[T]["speakers"] for T in Ts]))
            self.speakers[cp] = speakers
            C, Aw = boot_draws(run, cp, speakers, B, seed, len(self.reps))
            rep_col = {r: i for i, r in enumerate(self.reps)}
            sp_ix = {s: i for i, s in enumerate(speakers)}
            geff_min = min(prep[T]["geff"]["all"] for T in Ts)
            specs = [("boot", sp_ix, None, C, Aw)]
            if geff_min < GEFF_MIN and len(speakers) >= 2:
                specs.append(("jk", sp_ix, None, 1.0 - np.eye(len(speakers)), None))
            if cp in CLUSTER_COLS and all(prep[T]["clusters"] is not None for T in Ts):
                clusters = sorted(set().union(*[prep[T]["cl_units"] for T in Ts]))
                self.clusters[cp] = clusters
                Cc, Ac = boot_draws(run, cp, clusters, B, seed, len(self.reps), tag="cluster")
                cl_ix = {c: i for i, c in enumerate(clusters)}
                specs.append(("cl", cl_ix, "clusters", Cc, Ac))
                if len(clusters) < CLUSTER_JK_MAX:
                    specs.append(("cljk", cl_ix, "clusters", 1.0 - np.eye(len(clusters)), None))
            for T in Ts:
                p = prep[T]
                nk = len(p["keys"])
                acc = {(v, nm): ({q: np.zeros((len(Cm), nk)) for q in QN}, np.zeros((len(Cm), nk)))
                       for v in ("all", "ex_script38") for nm, _, _, Cm, _ in specs}
                for r in p["reps"]:
                    L, y, spk, te, Drows, krow, pt = p["load"](r)
                    vs = ("all",) if (T, r) in SAME_SPLIT else ("all", "ex_script38")
                    for nm, ix, labcol, Cm, Am in specs:
                        labels = spk if labcol is None else p["clusters"][te]
                        gi = np.array([ix[s] for s in labels])
                        ls, sidx = np.unique(gi, return_inverse=True)
                        Cb = Cm[:, ls]
                        if nm == "boot" and len(L):                              # E79 と同じ検算：重みを全部 1 にすると点の値に戻る
                            u1, d1, *_ = boot_rep(L, y, sidx, np.ones((1, len(ls))), Drows)
                            e1 = max(float(np.nanmax(np.abs(u1[0] - pt[:, 0]))), float(np.nanmax(np.abs(d1[0] - pt[:, 1]))))
                            checks.append({"check": "boot_identity", "store": name, "run": run, "target": T, "rep": r, "absdiff_max": e1, "ok": e1 <= 1e-6})
                            if not e1 <= 1e-6:
                                raise SystemExit(f"ブートストラップの式が点の値に戻らない（{name} {run} {T} 反復 {r}）")
                        if not len(L):
                            continue
                        *vals, empty, miss = boot_rep(L, y, sidx, Cb, Drows)
                        V = dict(zip(QN, vals))
                        ar = Am[:, rep_col[r]] if Am is not None else np.ones(len(Cm))
                        w = ar * (~empty)
                        for v in vs:
                            num, den = acc[(v, nm)]
                            for j, rows_j in enumerate(krow):
                                if not rows_j:
                                    continue
                                ok = np.isfinite(V["uar"][:, rows_j].mean(1))
                                for q in QN:
                                    vq = V[q][:, rows_j].mean(1)
                                    num[q][:, j] += np.where(np.isfinite(vq), w * vq, 0)
                                den[:, j] += np.where(ok, w, 0)
                for (v, nm), (num, den) in acc.items():
                    with np.errstate(invalid="ignore", divide="ignore"):
                        res = {q: (num[q] / den).astype(np.float32) for q in QN}  # E79 と同じく float32 で持つ
                    for j, key in enumerate(p["keys"]):
                        self.rp.setdefault((v, T, key), {})[nm] = {q: res[q][:, j] for q in QN}
                self.ncl[(run, T)] = len(self.clusters.get(cp, []))
                for v in ("all", "ex_script38"):
                    self.geff[(v, T)] = p["geff"][v]

    def measure_target(self, T, pick, students, checks):
        run = self.run
        waves = dict(self.src[T])
        tabs = {}
        if "*" in waves:                                                         # E81 の出力：1 つの表を波で分ける（行の番号は 1 つの表の番号）
            d = waves.pop("*")
            t = read_tab(Path(d) / f"{T}.csv")
            for w in sorted(set(t.wave)):
                tabs[w], waves[w] = t[t.wave == w], d
        else:
            for w, d in waves.items():
                tabs[w] = read_tab(Path(d) / f"{T}.csv", w)
        dir0 = waves.get("e77", next(iter(waves.values())))
        samp = pd.read_csv(Path(dir0) / f"{T}_samples.csv")
        df, X = load_any(T, run)
        if not np.array_equal(df.sample_id.astype(str).to_numpy(), samp.sample_id.astype(str).to_numpy()):
            raise SystemExit(f"{run} {T}：_samples.csv の sample_id の並びが load_any と違う")
        yall = df.label_4.to_numpy().astype(int)
        spk_all = samp.speaker_group.astype(str).to_numpy()
        clusters = A9.cluster_labels(T, samp.sample_id.astype(str).to_numpy(), [], run)
        feats = student_feats(run, X)
        keys = sorted({(st, K, *k_) for st in students for K in KS for sel in SELS + (None,) for k_ in [pick(run, T, st, K, sel)] if k_ is not None})
        reps = [r for r in self.reps if (Path(dir0) / f"{T}_r{r}_te.npy").exists()]
        if reps != self.reps:
            checks.append({"check": "reps_incomplete", "store": self.name, "run": run, "target": T, "have": str(reps), "ok": False})
        te_of, D_of, krows, recs = {}, {}, {}, []
        for r in reps:
            te = np.load(Path(dir0) / f"{T}_r{r}_te.npy")
            for w, d in waves.items():
                if not np.array_equal(np.load(Path(d) / f"{T}_r{r}_te.npy"), te):
                    raise SystemExit(f"{self.name} {run} {T} 反復 {r}：波 {w} のテストの行が違う")
            pool = np.load(Path(dir0) / f"{T}_r{r}_pool.npy") if self.pool_mode == "saved" else np.setdiff1d(np.arange(len(yall)), te)
            assert not np.isin(pool, te).any()
            te_of[r] = te
            y = yall[te]
            for st in students:                                                  # D：この単位の候補のラベルで学んだ LDA の向き（E79 と同じ式．前向きの差では打ち消し合う）
                Xs = np.asarray(feats[st], dtype=np.float64)
                mu, sd = Xs[pool].mean(0), Xs[pool].std(0)
                sd = np.where(sd > 1e-12, sd, 1.0)
                D_of[(st, r)] = lda_sep((Xs[pool] - mu) / sd, yall[pool], (Xs[te] - mu) / sd, y)
            rows_r = []                                                          # 鍵ごとの (波, 行の番号)
            for key in keys:
                st, K, wave, meth, pn, meff, md = key
                t = tabs.get(wave)
                if t is None:
                    rows_r.append([])
                    continue
                x = t[(t.rep == r) & (t.student == st) & (t.K == K) & (t.method == meth) & (t.pnorm == pn) & (t.meff == meff)]
                if md == "s0":
                    x = x[x.seed == 0]
                rows_r.append([(wave, int(i)) for i in x.row_id])
            flat = sorted({wi for lst in rows_r for wi in lst})
            pos = {wi: i for i, wi in enumerate(flat)}
            L = np.zeros((len(flat), len(te), 4))
            P = np.zeros((len(flat), len(te)), int)
            parts_ = []
            for w in sorted({wv for wv, _ in flat}):
                ids = [i for wv, i in flat if wv == w]
                a = np.asarray(np.load(Path(waves[w]) / f"{T}_r{r}_logp.npy", mmap_mode="r")[ids], dtype=np.float64)
                pr_ = np.asarray(np.load(Path(waves[w]) / f"{T}_r{r}_pred.npy", mmap_mode="r")[ids]).astype(int)
                if not np.isfinite(a).all():
                    raise SystemExit(f"NaN/Inf：{self.name} {run} {T} {w} 反復 {r}")
                for i, rid in enumerate(ids):
                    L[pos[(w, rid)]], P[pos[(w, rid)]] = a[i], pr_[i]
                tw = tabs[w]
                parts_.append(tw[(tw.rep == r) & tw.row_id.isin(ids)])
            tr = pd.concat(parts_) if parts_ else tabs[next(iter(tabs))].iloc[:0]
            if len(tr):                                                          # 行を flat の並び（L の行）にそろえる
                tr = tr.assign(_o=[pos[(w, int(i))] for w, i in zip(tr.wave, tr.row_id)]).sort_values("_o")
                assert tr._o.tolist() == list(range(len(flat)))
                tr = tr.drop(columns="_o").reset_index(drop=True)
            if len(tr):
                u_pred = np.array([shift_uar(y, p_) for p_ in P])
                ud = float(np.abs(u_pred - tr.uar.to_numpy()).max())
                checks.append({"check": "uar_vs_csv", "store": self.name, "run": run, "target": T, "rep": r, "absdiff_max": ud, "ok": ud <= 1e-7})
                d, o, acc_, _, auc = pair_moments(L, y)
                Dm = np.stack([D_of[(st, r)] for st in tr.student])
                a0 = ba(d, 0.0)
                dauc = np.sqrt(2) * stats.norm.ppf(np.clip(auc, 1e-12, 1 - 1e-12))
                tr = tr.assign(dirloss=100 * np.nanmean(ba(Dm, 0.0) - a0, 1), posloss=100 * np.nanmean(a0 - ba(d, o), 1),
                               dirloss_rev=100 * np.nanmean(ba(Dm, o) - ba(d, o), 1), posloss_rev=100 * np.nanmean(ba(Dm, 0.0) - ba(Dm, o), 1),
                               a_auc=100 * np.nanmean(ba(dauc, 0.0), 1), e_model=100 * np.nanmean(ba(d, o) - acc_, 1))
                recs.append(tr.assign(run=run, store=self.name))
                pt = np.stack([tr.uar.to_numpy() * 100, tr.dirloss.to_numpy()], 1)
            else:
                Dm, pt = np.zeros((0, 6)), np.zeros((0, 2))
            krows[r] = ([[pos[wi] for wi in lst] for lst in rows_r], L, Dm, pt)
        R = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["student", "K", "wave", "method", "pnorm", "meff", "seed", "rep"])
        self.rows.append(R)
        excl = [r for (t_, r) in SAME_SPLIT if t_ == T]
        for key in keys:                                                         # 点の値：種を反復ごとに平均してから反復で平均（E79 の Store.get と同じ）
            st, K, wave, meth, pn, meff, md = key
            x = R[(R.student == st) & (R.K == K) & (R.wave == wave) & (R.method == meth) & (R.pnorm == pn) & (R.meff == meff)]
            if md == "s0":
                x = x[x.seed == 0]
            for v in ("all", "ex_script38"):
                xv = x[~x.rep.isin(excl)] if v == "ex_script38" else x
                pt = dict(NAN_PT)
                if len(xv):
                    g = xv.groupby("rep")
                    for q in QN + ("a_auc", "e_model"):
                        pt[q] = g[q].mean().mean() * (100 if q == "uar" else 1)
                elif v == "all":
                    checks.append({"check": "key_missing", "store": self.name, "run": run, "target": T, "key": "|".join(key), "ok": False})
                self.point[(v, T, key)] = pt

        def geff_of(rs):
            if not rs:
                return np.nan
            vc = pd.Series(spk_all[np.concatenate([te_of[r] for r in rs])]).value_counts()
            return float(vc.sum() ** 2 / (vc ** 2).sum())
        geff = {"all": geff_of(reps), "ex_script38": geff_of([r for r in reps if r not in excl])}
        cl_units = set(clusters[np.concatenate([te_of[r] for r in reps])]) if clusters is not None else set()

        def load(r):
            krow, L, Dm, pt = krows[r]
            return L, yall[te_of[r]], spk_all[te_of[r]], te_of[r], Dm, krow, pt
        return {"keys": keys, "reps": reps, "speakers": set(spk_all[np.concatenate([te_of[r] for r in reps])]), "geff": geff,
                "clusters": clusters, "cl_units": cl_units, "load": load}

    def get(self, T, key, variant="all"):
        v = "ex_script38" if variant == "ex_script38" else "all"
        pt = self.point.get((v, T, key))
        if pt is None:
            return dict(NAN_PT), None
        return pt, self.rp.get((v, T, key))

    def geff_of(self, T, variant):
        return self.geff.get(("ex_script38" if variant == "ex_script38" else "all", T), np.nan)


class E79Store:
    """今の分割の主：E79 の measure の出力（analyze_e79.Store）．get は SrcStore と同じ形（台本の反復 3・8 を除いた版は持たない）。"""

    def __init__(self, mdir, run):
        self.S, self.run, self.mdir = A9.Store(mdir, [run]), run, Path(mdir)
        self.ncl = self.S.ncl

    def get(self, T, key, variant="all"):
        if variant == "ex_script38":
            return dict(NAN_PT), None
        return self.S.get(self.run, T, key[0], key[1], *key[2:])

    def geff_of(self, T, variant):
        return self.S.geff.get((self.run, T), np.nan)

    def names(self, T, fn):
        f = self.mdir / (f"jk_{self.run}_{T}.npz" if fn == "jk" else f"jkcl_{self.run}_{T}.npz")
        if not f.exists():
            return None
        z = np.load(f)
        return [str(x) for x in z["persons" if fn == "jk" else "clusters"]]

    def meta(self, T):
        f = self.mdir / "boot_meta.csv"
        if not f.exists():
            return None
        m = pd.read_csv(f)
        m = m[(m.run == self.run) & (m.target == T)]
        return m.iloc[0].to_dict() if len(m) else None


class CurStore:
    """今の分割：主の版は E79 の measure（あれば），台本の反復 3・8 を除いた版は本番の出力から作り直した SrcStore（あれば）。"""

    def __init__(self, e79, prod):
        self.e79, self.prod = e79, prod
        self.ncl = (e79 or prod).ncl

    def get(self, T, key, variant="all"):
        if variant == "ex_script38" or self.e79 is None:
            return self.prod.get(T, key, variant) if self.prod is not None else (dict(NAN_PT), None)
        return self.e79.get(T, key, variant)

    def geff_of(self, T, variant):
        if (variant == "ex_script38" or self.e79 is None) and self.prod is not None:
            return self.prod.geff_of(T, variant)
        return self.e79.geff_of(T, variant) if self.e79 is not None else np.nan


# ---------------------------------------------------------------- 比べ（E79 の stage_judge の compare と同じ区間の作り方）

def parts(store, run, Ts, st, K, kx, ky, metric, variant):
    q, sign, _, _ = METRIC[metric]
    out = []
    for T in Ts:
        a, b = kx(T), ky(T)
        base = {"geff": store.geff_of(T, variant), "ncl": store.ncl.get((run, T), 0)}
        if a is None or b is None:
            out.append({"x": np.nan, "auc": np.nan, "de": np.nan, "rp": None, **base})
            continue
        px, bx = store.get(T, (st, K, *a), variant)
        py, by = store.get(T, (st, K, *b), variant)
        rr = None
        if bx is not None and by is not None:
            rr = {}
            for kind in ("boot", "jk", "cl", "cljk"):
                xa, ya = bx.get(kind), by.get(kind)
                rr[kind] = None if (xa is None or ya is None or q not in xa or q not in ya) else \
                    sign * (np.asarray(xa[q], np.float64) - np.asarray(ya[q], np.float64))
        out.append({"x": sign * (px[q] - py[q]), "auc": sign * -(px.get("a_auc", np.nan) - py.get("a_auc", np.nan)) if metric == "dir" else np.nan,
                    "de": px.get("e_model", np.nan) - py.get("e_model", np.nan) if metric == "dir" else np.nan, "rp": rr, **base})
    return out


def intervals(x, P):
    """E79 の compare：主（百分位．G_eff < 20 は人物の jackknife の t 区間との和集合）と cluster_boot（主 ∪ 塊の百分位 ∪ 塊の jackknife の t）。"""
    def pct(kind):
        if not P or any(p["rp"] is None or p["rp"].get(kind) is None for p in P):
            return np.nan, np.nan, 0
        b = np.mean(np.stack([p["rp"][kind] for p in P]), 0)
        b = b[np.isfinite(b)]
        return (*(np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)), len(b))

    def union(*iv):
        los, his = [a for a, _ in iv if np.isfinite(a)], [b for _, b in iv if np.isfinite(b)]
        return (min(los) if los else np.nan), (max(his) if his else np.nan)
    lo_p, hi_p, nvalid = pct("boot")
    geffs = [p["geff"] for p in P]
    gmin = float(np.nanmin(geffs)) if np.isfinite(geffs).any() else np.nan
    lo_j = hi_j = np.nan
    rule = "pct"
    if nvalid == 0:
        rule = "no_replicates"
    elif np.isfinite(gmin) and gmin < GEFF_MIN:
        if all(p["rp"].get("jk") is not None for p in P):
            lo_j, hi_j = jk_interval(x, np.mean(np.stack([p["rp"]["jk"] for p in P]), 0), gmin)
            rule = "union_pct_jk"
        else:
            rule = "pct_jk_missing"
    lo, hi = union((lo_p, hi_p), (lo_j, hi_j))
    if rule == "pct_jk_missing":
        lo = hi = np.nan
    lo_c, hi_c, _ = pct("cl")
    lo_cj = hi_cj = np.nan
    if P and all(p["rp"] is not None and p["rp"].get("cljk") is not None for p in P):
        lo_cj, hi_cj = jk_interval(x, np.mean(np.stack([p["rp"]["cljk"] for p in P]), 0), float(min(p["ncl"] for p in P)))
    lo_s, hi_s = union((lo, hi), (lo_c, hi_c), (lo_cj, hi_cj)) if np.isfinite([lo, hi]).all() else (np.nan, np.nan)
    return {"main": (lo, hi, rule), "cluster_boot": (lo_s, hi_s, "union_main_cluster"), "n_boot_valid": nvalid, "gmin": gmin}


def compare(cur, g3, run, Ts, st, K, kx, ky, metric, variant):
    pc, pg = parts(cur, run, Ts, st, K, kx, ky, metric, variant), parts(g3, run, Ts, st, K, kx, ky, metric, variant)
    out = {}
    for name, P in (("cur", pc), ("g3", pg)):
        x = float(np.mean([p["x"] for p in P]))
        out[name] = {"x": x, **intervals(x, P), "auc": float(np.mean([p["auc"] for p in P])), "de": float(np.mean([p["de"] for p in P]))}
    pd_ = []
    for a, b in zip(pg, pc):                                                     # ΔΔ：同じ抽出ごとの差（話者・塊の並びと複製の数は照合済み）
        rr = None
        if a["rp"] is not None and b["rp"] is not None:
            rr = {k: (None if a["rp"].get(k) is None or b["rp"].get(k) is None or len(a["rp"][k]) != len(b["rp"][k]) else a["rp"][k] - b["rp"][k])
                  for k in ("boot", "jk", "cl", "cljk")}
        pd_.append({"x": a["x"] - b["x"], "rp": rr, "geff": a["geff"], "ncl": a["ncl"]})
    x = out["g3"]["x"] - out["cur"]["x"]
    out["dd"] = {"x": x, **intervals(x, pd_)}
    return out


def transition(jc, jg):
    """Δ_cur と Δ_G3 の判定の移り（4 節）：両方で判定できない は「同じ」に数えない．向きの逆転は 上回った ↔ 下回った だけ。"""
    if "missing" in (jc, jg) or "no_teacher" in (jc, jg):
        return "missing"
    if jc == jg:
        return "both_undet" if jc == "undet" else "unchanged"
    if {jc, jg} == {"pos", "neg"}:
        return "reversal"
    return "adjacent"


def zone(x, dl):
    return "pos" if x >= dl else ("neg" if x <= -dl else "mid")


def model_dep(r, dl):
    """E79 の stage_judge と同じ「模型に依る」の印（向きの損の行だけ）。"""
    if not (np.isfinite(r["x"]) and np.isfinite(r["auc"]) and np.isfinite(dl)):
        return np.nan
    return bool(abs(r["x"] - r["auc"]) > MODEL_UNIT_FRAC * dl or abs(r["de"]) > MODEL_UNIT_FRAC * dl or zone(r["x"], dl) != zone(r["auc"], dl))


# ---------------------------------------------------------------- 文の鍵（E79 の summarize の数え上げと statements_p5）

def combos(U):
    """E79 の summarize の組み合わせの行（主の tag，MOSEI・AFEW を除かない版）．U の列は E79 の units.csv と同じ。"""
    rows = []
    for (fam, comp, run, st, K, metric), g in U.groupby(["family", "comparison", "run", "student", "K", "metric"]):
        cps = {corp(t) for t in groups_for(run)}
        c = g.judgment.value_counts()
        nnt, nid, noc = int(c.get("no_teacher", 0)), int(c.get("identical", 0)), int(c.get("other_class", 0))
        n_tot = len(cps) - nnt - nid - noc
        rows.append({"family": fam, "comparison": comp, "metric": metric, "run": run, "student": st, "K": K, "n_corpora": n_tot,
                     "majority": majority(n_tot), **{f"n_{k}": int(c.get(k, 0)) for k in ("pos", "neg", "same", "undet", "missing", "inactive")},
                     "n_no_teacher": nnt, "n_identical": nid, "n_other_class": noc,
                     "n_oursx_none": int((g.no_teacher.map(lambda v: v is True or str(v) == "True") & ~g.judgment.isin(EXCLUDED)).sum()),
                     "n_units": len(g)})
    return pd.DataFrame(rows)


def keys_of(U):
    """(族, 世界, 生徒, K) → 文の鍵．P5-1・P5-2・P5-3 は E79 の statements_p5 そのもの．P5-3u は P5-1 と同じ 3 つの区分（E79 に文の行が無いため）。"""
    C = combos(U)
    out = {}
    if C.empty:
        return out
    for r in A9.statements_p5("main", C):
        if r["family"] not in ("P5-1", "P5-2", "P5-3") or r.get("rule") in ("summary",) or "|K" not in str(r.get("combos", "")):
            continue
        run, st, K = str(r["combos"]).split("|")
        out[(r["family"], run, st, K[1:])] = r["statement_key"]
    for r in C[C.family == "P5-3u"].itertuples(index=False):
        if r.n_corpora < 2 and r.n_no_teacher > 0:
            key = f"対象外（教師の残ったコーパスが {r.n_corpora} つ）"
        elif r.n_same + r.n_neg >= r.majority:
            key = "none が oursx と同じ水準か上回った"
        elif r.n_pos >= r.majority:
            key = "oursx が none を上回った"
        else:
            key = "判定できない（受け皿の行）"
        out[("P5-3u", r.run, r.student, r.K)] = key
    return out


def key_2corpus(fam, js):
    """記述：IEMOCAP と MELD の 2 コーパスだけの鍵（教師なしの単位を除いた過半）。"""
    c = pd.Series(js).value_counts()
    nnt = int(c.get("no_teacher", 0))
    n = len(js) - nnt
    if n < 1:
        return "対象外"
    maj = majority(n)
    pos, neg, same = (int(c.get(k, 0)) for k in ("pos", "neg", "same"))
    if fam == "P5-3":
        return "減らした" if pos >= maj else "判定できない"
    if same + neg >= maj:
        return "相手が同じ水準か上回った"
    return "上回った" if pos >= maj else "判定できない"


_IDENT: list = []


def ident_rows(stores, runs, e81, prod, args):
    """反復ごとの点の値の保存と，学習側が変わらない単位（IEMOCAP 台本の反復 3・8）の照合（1 回だけ計算して使い回す）。"""
    if _IDENT:
        return _IDENT[0]
    ident = []
    for run in runs:
        cur, g3, pstore = stores[run]
        R = pd.concat(g3.rows, ignore_index=True)
        R.to_csv(args.out / f"rows_g3_{run}.csv.gz", index=False)
        P = None
        if pstore is not None:
            P = pd.concat(pstore.rows, ignore_index=True)
            P.to_csv(args.out / f"rows_cur_{run}.csv.gz", index=False)
        elif cur.e79 is not None and run in cur.e79.S.point:
            P = cur.e79.S.point[run]
        if P is None:
            continue
        for (T, r) in sorted(SAME_SPLIT):
            a = R[(R.target == T) & (R.rep == r)] if len(R) else R
            if a.empty:
                continue
            b = P[(P.target == T) & (P.rep == r)]
            kk_ = ["student", "K", "wave", "method", "param", "meff"]
            jj = a.merge(b[kk_ + ["uar", "dirloss", "posloss"]], on=kk_, how="left", suffixes=("", "_cur"), indicator=True)
            row = {"run": run, "target": T, "rep": r, "check": "points", "n_rows": len(a), "n_unmatched": int((jj._merge != "both").sum())}
            for q in ("uar", "dirloss", "posloss"):
                v = (jj[q] - jj[f"{q}_cur"]).abs()
                row[f"{q}_maxabs"] = float(v.max()) if v.notna().any() else np.nan
                row[f"{q}_n_over_tol"] = int((v > TOL_POINT).sum())
            row["ok"] = row["n_unmatched"] == 0 and all(row[f"{q}_n_over_tol"] == 0 for q in ("uar", "dirloss", "posloss"))
            ident.append(row)
            t81 = read_tab(e81[run] / f"{T}.csv")
            t81 = t81[t81.rep == r]
            f81 = e81[run] / f"{T}_r{r}_logp.npy"
            for w in ("e77", "e78", "e80c"):
                dp = prod[w].get(run)
                if not f81.exists() or not dp or not (Path(dp) / f"{T}_r{r}_logp.npy").exists():
                    continue
                L81, Lp = np.load(f81, mmap_mode="r"), np.load(Path(dp) / f"{T}_r{r}_logp.npy", mmap_mode="r")
                tp = read_tab(Path(dp) / f"{T}.csv")
                tp = tp[tp.rep == r]
                src = t81[(t81.wave == w) | ((t81.method == "none") & (w == "e77"))]
                m = src.merge(tp[["student", "method", "param", "meff", "K", "row_id"]], on=["student", "method", "param", "meff", "K"], suffixes=("", "_p"))
                for meth, gm in m.groupby("method"):
                    v = max(float(np.abs(np.asarray(L81[int(a_)]) - np.asarray(Lp[int(b_)])).max()) for a_, b_ in zip(gm.row_id, gm.row_id_p))
                    pa = min(float((np.asarray(L81[int(a_)]).argmax(1) == np.asarray(Lp[int(b_)]).argmax(1)).mean()) for a_, b_ in zip(gm.row_id, gm.row_id_p))
                    ident.append({"run": run, "target": T, "rep": r, "check": "logp", "wave": w, "method": meth, "n_rows": len(gm), "n_src": int(len(src)),
                                  "logp_maxabs": v, "pred_agree_min": pa, "ok": v <= TOL_LOGP})
    _IDENT.append(ident)
    return ident


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e81", nargs="+", required=True, help="世界=場所（g3_split.py の出力．G3 の分割）")
    ap.add_argument("--e79", type=Path, default=ROOT / "results" / "e79", help="E79 の出力（delta/・select/・measure/・judge/）")
    ap.add_argument("--e77", nargs="+", help="世界=E77 の本番の出力（台本の反復 3・8 を除いた今の分割，学習側が変わらない単位の照合）")
    ap.add_argument("--e78", nargs="+")
    ap.add_argument("--e80c", nargs="+")
    ap.add_argument("--cur-e81", nargs="+", help="試し：世界=場所（g3_split.py --no-exclude の出力）を今の分割に使う（E79 の measure の代わり）")
    ap.add_argument("--chosen", type=Path, help="既定 <e79>/select/chosen.csv")
    ap.add_argument("--delta", type=Path, help="既定 <e79>/delta/delta.csv")
    ap.add_argument("--kept-used", type=Path, help="既定 <e79>/select/kept_used.csv")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--boot", type=int, help="既定は E79 の measure の provenance の値")
    ap.add_argument("--seed", type=int, help="既定は E79 の measure の provenance の値")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e81_analysis")
    args = ap.parse_args()
    t_start = time.time()
    e81 = {k: Path(v) for k, v in kv(args.e81).items()}
    runs = [r for r in WORLDS81 if r in e81]
    chosen = args.chosen or args.e79 / "select" / "chosen.csv"
    ch = pd.read_csv(chosen, dtype={"K": str, "param": str, "meff": str, "wave": str, "method": str, "student": str, "target": str})
    pick = chosen_pick(ch)
    mprov = args.e79 / "measure" / "provenance.json"
    mp = json.loads(mprov.read_text()) if mprov.exists() else {}
    B = args.boot if args.boot is not None else int(mp.get("boot", 1000))
    seed = args.seed if args.seed is not None else int(mp.get("seed", 0))
    args.out.mkdir(parents=True, exist_ok=True)
    checks = []
    dpath = args.delta or args.e79 / "delta" / "delta.csv"
    DEL = {(x.run, x.student, x.K): x for x in pd.read_csv(dpath, dtype={"K": str}).itertuples(index=False)}
    kpath = args.kept_used or args.e79 / "select" / "kept_used.csv"
    kk = pd.read_csv(kpath, dtype={"K": str, "teachers": str}).fillna({"teachers": ""})
    NOKEPT = {(a, b, c, d) for a, b, c, d, e in zip(kk.run, kk.target, kk.student, kk.K, kk.teachers) if not [t for t in e.split(";") if t]}
    uf, sf = args.e79 / "judge" / "units.csv", args.e79 / "judge" / "statements.csv"
    U79 = None
    if uf.exists():
        U79 = pd.read_csv(uf, dtype={"K": str})
        U79 = U79[U79.family.isin(list(FAMS)) & U79.K.isin(KS) & U79.run.isin(runs)]
    else:
        checks.append({"check": "e79_judge_units_missing", "path": str(uf), "ok": False})
    S79 = pd.read_csv(sf) if sf.exists() else None
    prod = {w: kv(getattr(args, w)) for w in ("e77", "e78", "e80c")}
    curd = kv(args.cur_e81)
    stores = {}
    for run in runs:
        prov81 = json.loads((e81[run] / "provenance.json").read_text())
        checks.append({"check": "chosen_sha256", "run": run, "ok": prov81.get("chosen_sha256") == sha256(chosen)})
        checks.append({"check": "e81_split_is_g3", "run": run, "split": prov81.get("split"), "ok": prov81.get("split") == "g3"})
        trial = {k_: prov81.get(k_) for k_ in ("allow_incomplete", "allow_dirty", "check_only") if prov81.get(k_)}
        checks.append({"check": "e81_not_trial", "run": run, "flags": json.dumps(trial), "unchecked": json.dumps(prov81.get("unchecked", []), ensure_ascii=False),
                       "ok": not trial and not prov81.get("unchecked") and "unchecked" in prov81})
        tg = [t for t in TARGETS if (e81[run] / f"{t}.csv").exists()]
        g3 = SrcStore(run, {T: {"*": e81[run]} for T in tg}, "saved", pick, B, seed, args.reps, checks, "g3")
        pstore = None
        if run in curd:
            pstore = SrcStore(run, {T: {"*": Path(curd[run])} for T in tg}, "saved", pick, B, seed, args.reps, checks, "cur_e81")
        elif run in prod["e77"]:
            src = {T: {w: Path(prod[w][run]) for w in ("e77", "e78", "e80c") if run in prod[w] and (Path(prod[w][run]) / f"{T}.csv").exists()}
                   for T in tg}
            pstore = SrcStore(run, src, "complement", pick, B, seed, args.reps, checks, "cur_prod")
        e79s = E79Store(args.e79 / "measure", run) if (args.e79 / "measure").exists() and run not in curd else None
        if e79s is None and pstore is None:
            raise SystemExit(f"{run}：今の分割の値が無い（--e79 の measure か --e77 などか --cur-e81）")
        cur = CurStore(e79s, pstore)
        for T in tg:                                                             # 対にした抽出の照合（話者・塊の並びと数）
            cp = corp(T)
            sp, cl = g3.speakers.get(cp), g3.clusters.get(cp, [])
            if e79s is not None:
                m = e79s.meta(T)
                pj, pc = e79s.names(T, "jk"), e79s.names(T, "cljk")
                ok = m is not None and int(m["n_speakers_corpus"]) == len(sp) and int(m.get("n_clusters_corpus", 0) or 0) == len(cl) \
                    and (pj is None or pj == sp) and (pc is None or pc == cl) and int(m.get("boot", B)) == B
                checks.append({"check": "paired_draws_e79", "run": run, "target": T, "ok": bool(ok)})
                if not ok:
                    raise SystemExit(f"{run} {T}：E79 と E81 の話者・塊の並びか複製の数が違う（抽出を対にできない）")
            if pstore is not None:
                ok = pstore.speakers.get(cp) == sp and pstore.clusters.get(cp, []) == cl
                checks.append({"check": "paired_draws_cur_store", "run": run, "target": T, "ok": bool(ok)})
                if not ok:
                    raise SystemExit(f"{run} {T}：今の分割と G3 の話者・塊の並びが違う")
        if e79s is not None and pstore is not None:                             # 本番の出力から作り直した今の分割が E79 の measure と同じか
            mx, n = 0.0, 0
            for (v, T, key), pt in pstore.point.items():
                if v != "all":
                    continue
                p9, r9 = e79s.get(T, key)
                for q in QN:
                    if np.isfinite(pt[q]) != np.isfinite(p9[q]):
                        mx = np.inf
                    elif np.isfinite(pt[q]):
                        mx = max(mx, abs(pt[q] - p9[q]))
                    for kind in ("boot", "jk", "cl", "cljk"):
                        a = (pstore.rp.get((v, T, key), {}).get(kind) or {}).get(q)
                        b = ((r9 or {}).get(kind) or {}).get(q)
                        if a is not None and b is not None:
                            mx = max(mx, float(np.nanmax(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))))
                n += 1
            checks.append({"check": "cur_store_vs_e79_measure", "run": run, "n_keys": n, "absdiff_max": mx, "ok": mx <= TOL_CUR})
        stores[run] = (cur, g3, pstore)
    # 単位ごとの Δ_cur・Δ_G3・ΔΔ と判定
    U = []
    for run in runs:
        cur, g3, _ = stores[run]
        studs = sorted(set(WORLDS[run] + [run]))
        for variant in VARIANTS:
            byc = {}
            for T in g3.targets:
                if variant == "impro_only" and T == "IEMOCAP-script":
                    continue
                byc.setdefault(corp(T), []).append(T)
            for st in studs:
                role = "fused" if st == run else "uni"
                for K in KS:
                    dr = DEL.get((run, st, K))
                    for cp, Ts in byc.items():
                        if variant != "main" and cp != "IEMOCAP":
                            continue
                        nt = [(run, T, st, K) in NOKEPT for T in Ts]
                        all_nt = bool(nt and all(nt))
                        comps = [(fam, comp, (lambda T, s_=sx: pick(run, T, st, K, s_)), (lambda T, s_=sy: pick(run, T, st, K, s_)), metric, sx, ntm)
                                 for fam, (comp, sx, sy, metric, ntm) in FAMS.items()]
                        for sel in SELS:                                         # Δ の表：手法 − none（METRIC の dir・pos は y − x なので x = none，y = 手法）
                            for mname, metric in DTAB:
                                if metric == "uar":
                                    comps.append(("delta", f"{sel}-none", (lambda T, s_=sel: pick(run, T, st, K, s_)), (lambda T: NONE), metric, sel, None))
                                else:
                                    comps.append(("delta", f"{sel}-none", (lambda T: NONE), (lambda T, s_=sel: pick(run, T, st, K, s_)), metric, sel, None))
                        for fam, comp, kx, ky, metric, msel, ntm in comps:
                            res = compare(cur, g3, run, Ts, st, K, kx, ky, metric, variant)
                            dl = np.nan if dr is None else float(dr.delta_d if METRIC[metric][2] == "d" else dr.delta_u)
                            kxs, kys = [kx(T) for T in Ts], [ky(T) for T in Ts]
                            zero = all(a is not None and a == b for a, b in zip(kxs, kys))
                            ms = [pick(run, T, st, K, msel) for T in Ts]
                            for ivk in ("main", "cluster_boot"):
                                row = {"variant": variant, "interval": ivk, "family": fam, "comparison": comp, "metric": metric, "run": run, "student": st,
                                       "role": role, "K": K, "corpus": cp, "n_targets": len(Ts), "targets": ";".join(Ts), "delta": dl,
                                       "no_teacher": all_nt, "structural_zero": zero,
                                       "chosen": ";".join("|".join(map(str, k_)) if k_ else "missing" for k_ in ms),
                                       "meff_kind": ";".join(("n" if k_[3] == "n" else ("fixed" if k_[3] in ("100", "500", "2000") else "-")) if k_ else "missing"
                                                             for k_ in ms),
                                       "n_reps": len(args.reps) * len(Ts) - (2 if variant == "ex_script38" and "IEMOCAP-script" in Ts else 0),
                                       "n_speakers": len(g3.speakers.get(cp, [])), "n_clusters": len(g3.clusters.get(cp, [])), "note_detect": NOTE_DETECT,
                                       "note": NOTE_POS if metric == "pos" else ""}
                                for split in ("cur", "g3", "dd"):
                                    r_ = res[split]
                                    lo, hi, rule = r_[ivk]
                                    j = judge_unit(r_["x"], lo, hi, dl)
                                    if ntm == "exclude" and all_nt:
                                        j = "no_teacher"
                                    row.update({f"{split}_point": r_["x"], f"{split}_lo": lo, f"{split}_hi": hi, f"{split}_rule": rule,
                                                f"{split}_judgment": j, f"{split}_n_boot_valid": r_["n_boot_valid"], f"{split}_g_eff_min": r_["gmin"]})
                                    if split != "dd" and metric == "dir":
                                        row[f"{split}_model_dependent"] = model_dep(r_, dl)
                                        row[f"{split}_point_auc"] = r_["auc"]
                                        row[f"{split}_model_err_diff"] = r_["de"]
                                row["transition"] = transition(row["cur_judgment"], row["g3_judgment"])
                                row["dd_excludes_0"] = bool(np.isfinite(row["dd_lo"]) and np.isfinite(row["dd_hi"]) and (row["dd_lo"] > 0 or row["dd_hi"] < 0))
                                U.append(row)
    U = pd.DataFrame(U)
    U.to_csv(args.out / "units_dd.csv", index=False)
    KEYC = ["family", "comparison", "metric", "run", "student", "K", "corpus"]
    if U79 is not None:                                                          # 照合：今の分割の IEMOCAP・MELD の単位が E79 の units.csv と同じか
        m9 = U79[U79.sensitivity.isin(["main", "cluster_boot"]) & U79.corpus.isin(CORPORA81)].assign(interval=lambda d: d.sensitivity)
        mine = U[(U.variant == "main") & (U.family != "delta")]
        j = mine.merge(m9, on=KEYC + ["interval"], how="inner", suffixes=("", "_79"))
        dp = float((j.cur_point - j.point).abs().max()) if len(j) else np.nan
        dlh = float(np.nanmax(np.abs(np.r_[j.cur_lo - j.lo, j.cur_hi - j.hi]))) if len(j) else np.nan
        nj = int((j.cur_judgment != j.judgment).sum())
        checks.append({"check": "cur_units_vs_e79_units", "n": len(j), "n_expected": len(mine), "point_absdiff_max": dp, "interval_absdiff_max": dlh,
                       "n_judgment_diff": nj, "ok": bool(len(j) == len(mine) and nj == 0 and (not np.isfinite(dp) or dp <= 1e-6))})
    # 移りと ΔΔ の判定の数（定義から 0 の単位は除いて別に数える）
    T_rows = []
    for (variant, ivk, fam, comp, metric, K), g in U.groupby(["variant", "interval", "family", "comparison", "metric", "K"]):
        for scope, gg in (("all", g), ("IEMOCAP", g[g.corpus == "IEMOCAP"]), ("MELD", g[g.corpus == "MELD"])):
            if gg.empty:
                continue
            x = gg[~gg.structural_zero]
            tc, dc = x.transition.value_counts(), x.dd_judgment.value_counts()
            T_rows.append({"variant": variant, "interval": ivk, "family": fam, "comparison": comp, "metric": metric, "K": K, "scope": scope,
                           "n_units": len(gg), "n_structural_zero": int(gg.structural_zero.sum()), "n_counted": len(x),
                           **{f"n_{k}": int(tc.get(k, 0)) for k in ("unchanged", "both_undet", "adjacent", "reversal", "missing")},
                           **{f"n_dd_{k}": int(dc.get(k, 0)) for k in ("pos", "same", "neg", "undet", "missing", "no_teacher")},
                           "n_dd_excludes_0": int(x.dd_excludes_0.sum()), "note_detect": NOTE_DETECT})
    TR = pd.DataFrame(T_rows)
    TR.to_csv(args.out / "transitions.csv", index=False)
    # 主の読み方：E79 の単位の IEMOCAP・MELD を E81 の判定に差し替えて，E79 の母数で文の鍵を作り直す
    st_rows, sens_rows = [], []
    if U79 is not None:
        for variant, ivk in (("main", "main"), ("main", "cluster_boot"), ("ex_script38", "main"), ("impro_only", "main")):
            base = U79[U79.sensitivity == ivk].copy()
            mine = U[(U.variant == variant) & (U.interval == ivk) & (U.family != "delta")]
            if variant != "main":                                                # 感度の版は IEMOCAP だけ作り方が変わる．MELD は主の版の単位
                mine = pd.concat([mine, U[(U.variant == "main") & (U.interval == ivk) & (U.family != "delta") & (U.corpus == "MELD")]])
            mi = mine.set_index(KEYC)
            idx = base.set_index(KEYC).index
            hit = idx.isin(mi.index)
            rep_cur, rep_g3 = base.copy(), base.copy()
            rep_g3.loc[hit, "judgment"] = mi.loc[idx[hit], "g3_judgment"].to_numpy()
            if variant != "main":                                                # 感度の版は今の分割の側も同じ作り方の判定に替える
                rep_cur.loc[hit, "judgment"] = mi.loc[idx[hit], "cur_judgment"].to_numpy()
            n_missing_unit = int(len(mine) - hit.sum()) if variant == "main" else 0
            kc, kg = keys_of(rep_cur), keys_of(rep_g3)
            for key in sorted(set(kc) | set(kg)):
                fam, run, st, K = key
                cu = mine[(mine.family == fam) & (mine.run == run) & (mine.student == st) & (mine.K == K)]
                chg = cu[cu.cur_judgment != cu.g3_judgment]
                row = {"variant": variant, "interval": ivk, "family": fam, "run": run, "student": st, "K": K, "key_cur": kc.get(key), "key_g3": kg.get(key),
                       "same": kc.get(key) == kg.get(key),
                       "both_undetermined": bool(str(kc.get(key)).startswith("判定できない") and str(kg.get(key)).startswith("判定できない")),
                       "n_units_replaced": int(len(cu)), "n_units_changed": int(len(chg)), "n_units_changed_dd_excl0": int(chg.dd_excludes_0.sum()),
                       "changed_units": ";".join(f"{a}:{b}->{c}" for a, b, c in zip(chg.corpus, chg.cur_judgment, chg.g3_judgment))}
                (st_rows if (variant, ivk) == ("main", "main") else sens_rows).append(row)
            if (variant, ivk) == ("main", "main"):
                checks.append({"check": "replaced_units_found_in_e79", "n_missing": n_missing_unit, "ok": n_missing_unit == 0})
                if S79 is not None:                                              # 今の分割の鍵が E79 の statements.csv と同じか（同じ規則で作り直せたか）
                    s9 = S79[(S79.tag == "main") & S79.family.isin(["P5-1", "P5-2", "P5-3"]) & S79.combos.astype(str).str.contains(r"\|K")]
                    n_cmp = n_bad = 0
                    for r in s9.itertuples(index=False):
                        run, st, K = str(r.combos).split("|")
                        k_ = (r.family, run, st, K[1:])
                        if k_ in kc:
                            n_cmp += 1
                            n_bad += int(kc[k_] != r.statement_key)
                    checks.append({"check": "cur_keys_vs_e79_statements", "n": n_cmp, "n_diff": n_bad, "ok": n_bad == 0 and n_cmp > 0})
    ST = pd.DataFrame(st_rows)
    ST.to_csv(args.out / "statements_main.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(args.out / "statements_sens.csv", index=False)
    two = []                                                                     # 記述：IEMOCAP・MELD の 2 コーパスだけの鍵
    for (variant, ivk, fam, run, st, K), g in U[U.family.isin(["P5-1", "P5-2", "P5-3"])].groupby(["variant", "interval", "family", "run", "student", "K"]):
        kc_, kg_ = key_2corpus(fam, list(g.cur_judgment)), key_2corpus(fam, list(g.g3_judgment))
        two.append({"variant": variant, "interval": ivk, "family": fam, "run": run, "student": st, "K": K, "key_cur": kc_, "key_g3": kg_,
                    "same": kc_ == kg_, "both_undetermined": kc_ == kg_ == "判定できない", "n_units": len(g)})
    pd.DataFrame(two).to_csv(args.out / "statements_2corpus.csv", index=False)

    def rule_ii(variant="main", ivk="main", K=None):
        g = U[(U.variant == variant) & (U.interval == ivk) & (U.family == "P5-3u")]
        if K is not None:
            g = g[g.K == K]
        x = g[~g.structural_zero]
        n, n_same, n_rev = len(x), int((x.dd_judgment == "same").sum()), int((x.transition == "reversal").sum())
        return {"n_units": n, "n_structural_zero": int(g.structural_zero.sum()), "n_dd_same": n_same, "majority": majority(n) if n else None,
                "n_reversal": n_rev, "n_dd_undet": int((x.dd_judgment == "undet").sum()), "ok": bool(n > 0 and n_same >= majority(n) and n_rev == 0)}
    ri = {"n_keys": int(len(ST)), "n_changed": int((~ST.same).sum()) if len(ST) else None,
          "n_both_undetermined": int(ST.both_undetermined.sum()) if len(ST) else None,
          "changed_by_family": ST[~ST.same].groupby("family").size().to_dict() if len(ST) else {},
          "n_changed_with_dd_excl0": int(((~ST.same) & (ST.n_units_changed_dd_excl0 > 0)).sum()) if len(ST) else None,
          "ok": (bool(len(ST) and ST.same.all()) if U79 is not None else None)}
    rii = rule_ii()
    scope = {run: {"targets": stores[run][1].targets, "reps": args.reps} for run in runs}
    failed = sorted({c_["check"] for c_ in checks if not c_["ok"]})
    if any(not x_.get("ok", True) for x_ in ident_rows(stores, runs, e81, prod, args)):
        failed.append("identity_same_split")
    if sorted(runs) != sorted(WORLDS81):
        failed.append("missing_world")
    for run in runs:
        if stores[run][1].targets != list(TARGETS):
            failed.append(f"missing_target:{run}")
    if list(args.reps) != list(range(10)):
        failed.append("reps_not_0_9")
    if U79 is None:
        failed.append("e79_judge_missing")
    flags = {"rule": "E81 の 4 節：(i) 作り直した文の鍵が世界・生徒・K・P5-1・P5-2・P5-3・P5-3u の全部で E79 の鍵と同じ，"
                     "(ii) oursx − none の UAR の ΔΔ が IEMOCAP と MELD の単位（定義から 0 の単位を除く）の過半で同じ水準で，Δ の判定の向きの逆転が 0．"
                     "両方を満たすときだけ K=0・5 の P5-1〜P5-3 の文から限定を外す",
             "i": ri, "ii": rii,
             "scope": scope, "failed_checks": failed,
             "decision": (None if (failed or ri["ok"] is None) else ("限定を外す" if (ri["ok"] and rii["ok"]) else "限定を付ける")),
             "sensitivity": {"ii_K0": rule_ii(K="0"), "ii_K5": rule_ii(K="5"), "ii_ex_script38": rule_ii("ex_script38"), "ii_impro_only": rule_ii("impro_only"),
                             "ii_cluster_boot": rule_ii("main", "cluster_boot")},
             "notes": [NOTE_DETECT, "P5-3u の文の鍵は E79 に無いので P5-1・P5-2 と同じ 3 つの区分で作った（impl.md）",
                       "両方の分割で判定できない鍵も (i) では同じに数える（4 節の (i) の文のまま）．数は i.n_both_undetermined",
                       "IEMOCAP の ΔΔ は即興で 4 通り，台本で 6 通りの学習側から来る（checks/E81/stats.md）"]}
    (args.out / "rule.json").write_text(json.dumps(flags, ensure_ascii=False, indent=1, default=str))
    ident = ident_rows(stores, runs, e81, prod, args)
    pd.DataFrame(ident).to_csv(args.out / "identity_same_split.csv", index=False)
    C = pd.DataFrame(checks)
    C.to_csv(args.out / "checks.csv", index=False)
    prov = {"script": "exp/analyze_e81.py", "experiment": "E81", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": git("status", "--porcelain", "--", "exp/analyze_e81.py", "exp/analyze_e79.py").splitlines(), "script_sha256": sha256(__file__),
            "analyze_e79_sha256": sha256(A9.__file__), "e81": {k: str(v) for k, v in e81.items()},
            "e81_commits": {r: json.loads((e81[r] / "provenance.json").read_text()).get("commit") for r in runs},
            "chosen": str(chosen), "chosen_sha256": sha256(chosen), "delta": str(dpath), "delta_sha256": sha256(dpath), "kept_used": str(kpath),
            "e79_units_sha256": sha256(uf) if uf.exists() else None, "e79_statements_sha256": sha256(sf) if sf.exists() else None,
            "e79_measure_provenance_sha256": sha256(mprov) if mprov.exists() else None, "prod": prod, "cur_e81": curd, "boot": B, "seed": seed,
            "reps": args.reps, "tol": {"point": TOL_POINT, "logp": TOL_LOGP, "cur": TOL_CUR}, "seconds": round(time.time() - t_start, 1)}
    (args.out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))
    nbad = int((~C.ok.astype(bool)).sum()) if len(C) else 0
    nid_bad = sum(1 for x in ident if not x.get("ok", True))
    print(f"単位 {len(U)} 行，移りの表 {len(TR)} 行，主の鍵 {len(ST)} 行，照合 {len(C)} 項目（食い違い {nbad}），同じ単位の照合 {len(ident)} 項目（食い違い {nid_bad}）",
          flush=True)
    print(f"書いた場所：{args.out}（値と結論は画面に出さない．rule.json に (i)・(ii) と結論）", flush=True)
    if failed:
        print(f"照合に落ちたか入力が欠けたので結論を書かない（rule.json の failed_checks：{len(failed)} 件）", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
