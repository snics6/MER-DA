"""【本番の感度】E81（CPU）：G3 の分割（`exp/g3_split.py` の出力）と今の分割（E79 の段 (c) の出力）で「手法 − none」を比べ，
P5-1〜P5-3 の判定を 3 つのターゲット（IEMOCAP 即興・台本，MELD）の単位に当て直す。

設計は `docs/exploratory-analyses.md` の E81 の 4 節（値を見る前に決めた）．E79 の関数（`exp/analyze_e79.py` の boot_draws・boot_rep・pair_moments・
lda_sep・jk_interval・judge_unit・Store）をそのまま使う．値は画面に出さない（行の数とファイルの場所だけ）。

単位：(世界, 生徒, コーパス, K)．コーパスは IEMOCAP（即興と台本を 1 つにまとめる．E79 と同じ）と MELD．
  Δ_cur ＝ 今の分割の「手法 − none」（E79 の measure の点の値と複製），Δ_G3 ＝ G3 の分割の同じ量（E81 の出力から E79 と同じ手順で作る），ΔΔ ＝ Δ_G3 − Δ_cur．
  手法は oursx・kd・groupB（chosen.csv の incl の行．none を選んだ単位は差が 0）．量は UAR・向きの損・位置の損（どれも「手法 − none」）．
  区間：人物と反復の 2 段の百分位のブートストラップ（E79 と同じ抽出：boot_draws(実行, コーパス, 話者, B, 種, 反復の数)．G3 と今の分割はテストが同じなので
  同じ抽出になり，ΔΔ は抽出ごとの差で作る）．G_eff < 20 のターゲットを含む単位は，人物を 1 人ずつ抜く jackknife の t 区間との和集合（E79 と同じ）。
判定の再適用：P5-1（oursx − kd の UAR），P5-2（oursx − groupB の UAR），P5-3（none − oursx の向きの損．P5-3u：oursx − none の UAR）を
  E79 の judge_unit と δ（delta.csv）で単位ごとに判定し，E79 の statements_p5 と同じ文の鍵を，2 つのコーパスの過半（教師なしの単位を除く）で決める．
  文の鍵が全部の（族, 世界, 生徒, K）で今の分割と同じなら「同じ」（筋 5 の主張に限定を付けない），1 つでも違えば「変わった」（E81 の 4 節）。
照合：E81 の出力の chosen.csv の sha256，話者の並びと複製の数が E79 と同じこと（対にした抽出），学習側が変わらない単位（IEMOCAP 台本の反復 3・8）の
  反復ごとの点の値が E79 の行と同じこと（--e77 などを渡せば対数の確率も），ブートストラップの式が全部 1 の重みで点の値に戻ること。

出力（--out．既定 results/e81_analysis/）：delta_table.csv，units.csv，judgment_changes.csv，statements.csv，statement_flag.json，
  rows_g3_<世界>.csv.gz（G3 の行ごと・反復ごとの点の値），checks.csv，identity_same_split.csv，provenance.json

    ./.venv/bin/python exp/analyze_e81.py --e81 fuse-audio-video-text=results/e81_fuse-audio-video-text fuse-audio-video=results/e81_fuse-audio-video \\
        --e79 results/e79 [--e77 世界=場所 …] [--e78 世界=場所 …] [--e80c 世界=場所 …]
  試し：--cur-e81 世界=場所（今の分割を，E81 の --no-exclude の出力から同じ手順で作る．E79 の measure の代わり）
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import analyze_e79 as A9  # noqa: E402
from analyze_e79 import (EMPTY, GEFF_MIN, METRIC, QN, SEEDED, boot_draws, boot_rep, corp, jk_interval, judge_unit, lda_sep,  # noqa: E402
                         majority, pair_moments, seed_of, strip_seed, student_feats)
from data import groups_for, load_any  # noqa: E402
from explore_pairs import ba  # noqa: E402
from multimodal_distill import WORLDS  # noqa: E402
from shift import uar as shift_uar  # noqa: E402

WORLDS81 = ("fuse-audio-video-text", "fuse-audio-video")
TARGETS = ("IEMOCAP-impro", "IEMOCAP-script", "MELD")
KS = ("0", "5")
SELS = ("oursx", "kd", "groupB")
SAME_SPLIT = {("IEMOCAP-script", 3), ("IEMOCAP-script", 8)}
NONE = ("e77", "none", "-", "0", "-")
TOL_POINT = 1e-4                                                              # 学習側が変わらない単位の反復ごとの点の値（UAR は割合，損はポイント）の許容幅．1 発話の予測の変化（約 2e-3）を拾う
TOL_LOGP = 1e-4                                                               # 同じ単位の対数の確率．凸の解き（dist・kd・lp）は束の中身で 2.4e-5 まで動いた（checks/E81/impl.md）．none・教師・CORAL・DANN・entmin・IM・self はビット単位で同じ
FAMS = {  # 族 → (比べ, x の族, y の族（None は none）, 量, 教師なしの扱い)
    "P5-1": ("oursx-kd", "oursx", "kd", "uar", "exclude"),
    "P5-2": ("oursx-groupB", "oursx", "groupB", "uar", "count"),
    "P5-3": ("none-oursx", "oursx", None, "dir", "exclude"),
    "P5-3u": ("oursx-none", "oursx", None, "uar", "count"),
}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def kv(xs):
    return dict(x.split("=", 1) for x in (xs or []))


# ---------------------------------------------------------------- E81 の出力を E79 と同じ手順で測る

def chosen_keys(ch, run, T, students):
    """(生徒, K, 族) → E79 の鍵 (波, 方法, pnorm, m_eff, 種の扱い)．none を選んだ行は NONE，候補が無い行は None。"""
    out = {}
    x = ch[(ch.run == run) & (ch.target == T) & (ch.variant == "incl") & ch.selector.isin(SELS) & ch.K.isin(KS)]
    for r in x.itertuples(index=False):
        if pd.isna(r.method) or r.method == EMPTY:
            out[(r.student, r.K, r.selector)] = None
        elif r.method == "none":
            out[(r.student, r.K, r.selector)] = NONE
        else:
            out[(r.student, r.K, r.selector)] = (r.wave, r.method, r.param, r.meff, "agg" if r.method in SEEDED else "-")
    for st in students:
        for K in KS:
            out[(st, K, "none")] = NONE
    return out


class E81Store:
    """E81 の出力（g3_split.py）から，E79 の measure と同じ手順で鍵ごとの点の値と複製（boot・jk）を作る．get は analyze_e79.Store.get と同じ形。"""

    def __init__(self, run, d, ch, B, seed, reps, checks, split_name):
        self.run, self.d, self.B, self.seed, self.reps = run, Path(d), B, seed, list(reps)
        self.point, self.reps_, self.keys, self.geff, self.speakers, self.rows_all = {}, {}, {}, {}, {}, []
        prov = json.loads((self.d / "provenance.json").read_text())
        self.prov = prov
        checks.append({"check": "chosen_sha256", "run": run, "split": split_name, "dir": str(d),
                       "ok": prov.get("chosen_sha256") == A9.sha256(args_chosen_path[0])})
        students = WORLDS[run] + [run]
        targets = [t for t in TARGETS if (self.d / f"{t}.csv").exists()]
        self.targets = targets
        prep = {}
        for T in targets:
            prep[T] = self.measure_target(T, ch, students, checks, split_name)
        by_corpus = {}
        for T in targets:
            by_corpus.setdefault(corp(T), []).append(T)
        for cp, Ts in by_corpus.items():
            speakers = sorted(set().union(*[prep[T]["speakers"] for T in Ts]))
            self.speakers[cp] = speakers
            C, Aw = boot_draws(run, cp, speakers, B, seed, len(self.reps))
            rep_col = {r: i for i, r in enumerate(self.reps)}
            sp_ix = {s: i for i, s in enumerate(speakers)}
            geff_min = min(prep[T]["geff"] for T in Ts)
            specs = [("boot", C, Aw)]
            if geff_min < GEFF_MIN and len(speakers) >= 2:
                specs.append(("jk", 1.0 - np.eye(len(speakers)), None))
            for T in Ts:
                p = prep[T]
                nk = len(p["keys"])
                acc = {nm: ({q: np.zeros((len(Cm), nk)) for q in QN}, np.zeros((len(Cm), nk))) for nm, Cm, _ in specs}
                for r in p["reps"]:
                    L, y, spk, Drows, krow, pt = p["load"](r)
                    gi = np.array([sp_ix[s] for s in spk])
                    ls, sidx = np.unique(gi, return_inverse=True)
                    for nm, Cm, Am in specs:
                        Cb = Cm[:, ls]
                        if nm == "boot":                                         # E79 と同じ検算：重みを全部 1 にすると点の値に戻る
                            u1, d1, *_ = boot_rep(L.astype(np.float64), y, sidx, np.ones((1, len(ls))), Drows)
                            e1 = max(float(np.nanmax(np.abs(u1[0] - pt[:, 0]))), float(np.nanmax(np.abs(d1[0] - pt[:, 1]))))
                            if not e1 <= 1e-6:
                                raise SystemExit(f"ブートストラップの式が点の値に戻らない（{run} {T} 反復 {r}：{e1}）")
                            checks.append({"check": "boot_identity", "run": run, "split": split_name, "target": T, "rep": r, "absdiff_max": e1, "ok": True})
                        V = {q: np.full((len(Cm), L.shape[0]), np.nan) for q in QN}
                        *vals, empty, miss = boot_rep(L.astype(np.float64), y, sidx, Cb, Drows)
                        for q, v_ in zip(QN, vals):
                            V[q][:, :] = v_
                        ar = Am[:, rep_col[r]] if Am is not None else np.ones(len(Cm))
                        w = ar * (~empty)
                        num, den = acc[nm]
                        for j, rows_j in enumerate(krow):
                            if not rows_j:
                                continue
                            ok = np.isfinite(V["uar"][:, rows_j].mean(1))
                            for q in QN:
                                vq = V[q][:, rows_j].mean(1)
                                num[q][:, j] += np.where(np.isfinite(vq), w * vq, 0)
                            den[:, j] += np.where(ok, w, 0)
                res = {}
                for nm, *_ in specs:
                    num, den = acc[nm]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        res[nm] = {q: (num[q] / den).astype(np.float32) for q in QN}  # E79 と同じく float32 で持つ
                for j, key in enumerate(p["keys"]):
                    self.reps_[(T, *key)] = {nm: {q: res[nm][q][:, j] for q in QN} for nm in res}
                self.geff[(run, T)] = p["geff"]

    def measure_target(self, T, ch, students, checks, split_name):
        run = self.run
        tab = pd.read_csv(self.d / f"{T}.csv", dtype={"K": str, "param": str, "meff": str, "pnorm": str, "student": str, "method": str, "wave": str})
        tab["pnorm"] = [strip_seed(p) if m in SEEDED else p for m, p in zip(tab.method, tab.param)]
        tab["seed"] = [seed_of(m, p) for m, p in zip(tab.method, tab.param)]
        samp = pd.read_csv(self.d / f"{T}_samples.csv")
        df, X = load_any(T, run)
        if not np.array_equal(df.sample_id.astype(str).to_numpy(), samp.sample_id.astype(str).to_numpy()):
            raise SystemExit(f"{run} {T}：_samples.csv の sample_id の並びが load_any と違う")
        yall = df.label_4.to_numpy().astype(int)
        spk_all = samp.speaker_group.astype(str).to_numpy()
        feats = student_feats(run, X)
        ck = chosen_keys(ch, run, T, students)
        keys = sorted({(st, K, *k_) for (st, K, _), k_ in ck.items() if k_ is not None})
        reps = [r for r in self.reps if (self.d / f"{T}_r{r}_logp.npy").exists()]
        if reps != self.reps:
            checks.append({"check": "reps_incomplete", "run": run, "split": split_name, "target": T, "have": str(reps), "ok": False})
        te_of, rows_rep, D_of = {}, {}, {}
        for r in reps:
            te = np.load(self.d / f"{T}_r{r}_te.npy")
            pool = np.load(self.d / f"{T}_r{r}_pool.npy")
            assert not np.isin(pool, te).any()
            te_of[r] = te
            y = yall[te]
            L = np.load(self.d / f"{T}_r{r}_logp.npy").astype(np.float64)
            P = np.load(self.d / f"{T}_r{r}_pred.npy").astype(int)
            tr = tab[tab.rep == r].sort_values("row_id").reset_index(drop=True)
            if len(tr) != L.shape[0] or L.shape[1] != len(te) or not np.array_equal(tr.row_id.to_numpy(), np.arange(len(tr))):
                raise SystemExit(f"{run} {T} 反復 {r}：行の数かテストの発話の数が合わない")
            if not np.isfinite(L).all():
                raise SystemExit(f"NaN/Inf：{run} {T} 反復 {r}")
            u = np.array([shift_uar(y, pc) for pc in P])
            ud = float(np.abs(u - tr.uar.to_numpy()).max())
            checks.append({"check": "uar_vs_csv", "run": run, "split": split_name, "target": T, "rep": r, "absdiff_max": ud, "ok": ud <= 1e-7})
            d, o, _, _, _ = pair_moments(L, y, full=False)
            for st in students:                                                  # D：この単位の候補（学習側）のラベルで学んだ LDA の向き（E79 と同じ式．Δ では打ち消し合う）
                Xs = np.asarray(feats[st], dtype=np.float64)
                mu, sd = Xs[pool].mean(0), Xs[pool].std(0)
                sd = np.where(sd > 1e-12, sd, 1.0)
                D_of[(st, r)] = lda_sep((Xs[pool] - mu) / sd, yall[pool], (Xs[te] - mu) / sd, y)
            Dm = np.stack([D_of[(st, r)] for st in tr.student])
            a0 = ba(d, 0.0)
            tr["uar"] = u
            tr["dirloss"] = 100 * np.nanmean(ba(Dm, 0.0) - a0, 1)
            tr["posloss"] = 100 * np.nanmean(a0 - ba(d, o), 1)
            tr["dirloss_rev"] = 100 * np.nanmean(ba(Dm, o) - ba(d, o), 1)
            tr["posloss_rev"] = 100 * np.nanmean(ba(Dm, 0.0) - ba(Dm, o), 1)
            rows_rep[r] = tr
            self.rows_all.append(tr.assign(run=run))
        allrows = pd.concat(rows_rep.values(), ignore_index=True) if rows_rep else tab.iloc[:0]
        for key in keys:                                                         # 点の値：種を反復ごとに平均してから反復で平均（E79 の Store.get と同じ）
            st, K, wave, meth, pn, meff, md = key
            x = allrows[(allrows.student == st) & (allrows.K == K) & (allrows.wave == wave) & (allrows.method == meth) & (allrows.pnorm == pn)
                        & (allrows.meff == meff)]
            if md == "s0":
                x = x[x.seed == 0]
            pt = {q: np.nan for q in QN}
            if len(x):
                g = x.groupby("rep")
                for q in QN:
                    pt[q] = g[q].mean().mean() * (100 if q == "uar" else 1)
                if x.groupby("rep").size().nunique() != 1 or set(x.rep) != set(reps):
                    checks.append({"check": "key_rows_uneven", "run": run, "split": split_name, "target": T, "key": "|".join(key), "ok": False})
            else:
                checks.append({"check": "key_missing", "run": run, "split": split_name, "target": T, "key": "|".join(key), "ok": False})
            self.point[(T, *key)] = pt
        vc = pd.Series(spk_all[np.concatenate([te_of[r] for r in reps])]).value_counts()
        geff = float(vc.sum() ** 2 / (vc ** 2).sum())

        def load(r):
            tr = rows_rep[r]
            krow = []
            for key in keys:
                st, K, wave, meth, pn, meff, md = key
                m = (tr.student == st) & (tr.K == K) & (tr.wave == wave) & (tr.method == meth) & (tr.pnorm == pn) & (tr.meff == meff)
                if md == "s0":
                    m &= tr.seed == 0
                krow.append(tr.index[m].tolist())
            L = np.load(self.d / f"{T}_r{r}_logp.npy")
            Drows = np.stack([D_of[(st, r)] for st in tr.student])
            pt = np.stack([tr.uar.to_numpy() * 100, tr.dirloss.to_numpy()], 1)
            return L, yall[te_of[r]], spk_all[te_of[r]], Drows, krow, pt
        return {"keys": keys, "reps": reps, "speakers": set(vc.index), "geff": geff, "load": load}

    def get(self, run, T, st, K, wave, meth, pn, meff, md):
        key = (T, st, K, wave, meth, pn, meff, md)
        pt = self.point.get(key)
        if pt is None:
            return {q: np.nan for q in QN}, None
        rp = self.reps_.get(key)
        return pt, (None if rp is None else {"boot": rp.get("boot"), "jk": rp.get("jk")})


args_chosen_path = [None]


class E79Cur:
    """今の分割：E79 の measure の出力（analyze_e79.Store）．get は同じ形。"""

    def __init__(self, mdir, runs):
        self.S = A9.Store(mdir, runs)
        self.geff = self.S.geff
        self.mdir = Path(mdir)

    def get(self, run, T, st, K, wave, meth, pn, meff, md):
        return self.S.get(run, T, st, K, wave, meth, pn, meff, md)

    def persons(self, run, T):
        f = self.mdir / f"jk_{run}_{T}.npz"
        return [str(x) for x in np.load(f)["persons"]] if f.exists() else None

    def n_speakers(self, run, T):
        f = self.mdir / "boot_meta.csv"
        if not f.exists():
            return None
        m = pd.read_csv(f)
        m = m[(m.run == run) & (m.target == T)]
        return int(m.n_speakers_corpus.iloc[0]) if len(m) else None


# ---------------------------------------------------------------- 比べ（E79 の compare と同じ区間の作り方）

def diff_parts(store, run, Ts, st, K, kx, ky, metric):
    """コーパスの各ターゲットの差（METRIC の向き）の点の値と複製。"""
    q, sign, _, _ = METRIC[metric]
    pts, reps_, geffs = [], [], []
    for T in Ts:
        kxx, kyy = kx(T), ky(T)
        if kxx is None or kyy is None:
            pts.append(np.nan)
            reps_.append({"boot": None, "jk": None})
            geffs.append(store.geff.get((run, T), np.nan))
            continue
        px, bx = store.get(run, T, st, K, *kxx)
        py, by = store.get(run, T, st, K, *kyy)
        pts.append(sign * (px[q] - py[q]))
        rr = {}
        for kind in ("boot", "jk"):
            a_ = None if (bx is None or by is None or bx.get(kind) is None or by.get(kind) is None) else (bx[kind], by[kind])
            rr[kind] = None if a_ is None else sign * (np.asarray(a_[0][q], np.float64) - np.asarray(a_[1][q], np.float64))
        reps_.append(rr)
        geffs.append(store.geff.get((run, T), np.nan))
    return pts, reps_, geffs


def interval(x, reps_, geffs):
    """E79 の compare と同じ：百分位（boot）と，G_eff < 20 なら jackknife の t 区間との和集合．"""
    if not np.isfinite(x) or not reps_ or any(r_["boot"] is None for r_ in reps_):
        return np.nan, np.nan, "no_replicates"
    b = np.mean(np.stack([r_["boot"] for r_ in reps_]), 0)
    b = b[np.isfinite(b)]
    if not len(b):
        return np.nan, np.nan, "no_replicates"
    lo_p, hi_p = np.percentile(b, [2.5, 97.5])
    gmin = float(np.nanmin(geffs)) if np.isfinite(geffs).any() else np.nan
    if np.isfinite(gmin) and gmin < GEFF_MIN:
        if all(r_["jk"] is not None for r_ in reps_):
            lo_j, hi_j = jk_interval(x, np.mean(np.stack([r_["jk"] for r_ in reps_]), 0), gmin)
            los = [v for v in (lo_p, lo_j) if np.isfinite(v)]
            his = [v for v in (hi_p, hi_j) if np.isfinite(v)]
            return min(los), max(his), "union_pct_jk"
        return np.nan, np.nan, "pct_jk_missing"
    return float(lo_p), float(hi_p), "pct"


def compare_both(cur, g3, run, cp, Ts, st, K, kx, ky, metric):
    pc, rc, gc = diff_parts(cur, run, Ts, st, K, kx, ky, metric)
    pg, rg, gg = diff_parts(g3, run, Ts, st, K, kx, ky, metric)
    xc, xg = float(np.mean(pc)), float(np.mean(pg))
    out = {"cur": (xc, *interval(xc, rc, gc)), "g3": (xg, *interval(xg, rg, gg))}
    rd = []                                                                       # ΔΔ：同じ抽出ごとの差
    for a, b in zip(rg, rc):
        rd.append({k: (None if a[k] is None or b[k] is None or len(a[k]) != len(b[k]) else a[k] - b[k]) for k in ("boot", "jk")})
    xd = xg - xc
    out["dd"] = (xd, *interval(xd, rd, gg))
    out["gmin"] = float(np.nanmin(gg)) if np.isfinite(gg).any() else np.nan
    return out


# ---------------------------------------------------------------- 文の鍵（E79 の statements_p5 と同じ規則を 2 つのコーパスに当てる）

def statement(fam, c, u=None):
    maj, ntot = c["majority"], c["n_corpora"]
    if ntot < 2 and c["n_no_teacher"] > 0:
        return f"対象外（教師の残ったコーパスが {ntot} つ．教師なし {c['n_no_teacher']}）"
    den = f"（教師の残った {ntot} コーパスのうち）" if c["n_no_teacher"] else ""
    if fam in ("P5-1", "P5-2"):
        if c["n_same"] + c["n_neg"] >= maj:
            key = "相手が oursx と同じ水準か上回った"
        elif c["n_pos"] >= maj:
            key = "oursx が相手を上回った"
        else:
            key = "判定できない（受け皿の行）"
        if fam == "P5-2" and c.get("n_oursx_none", 0):
            key += f"．うち {c['n_oursx_none']} コーパスは教師なしで oursx ＝ none"
    elif fam == "P5-3":
        if c["n_pos"] >= maj:
            key = "oursx は none より向きの損を減らした"
        elif u is not None and u["n_pos"] >= u["majority"]:
            key = "向きの損は減らさず UAR は上回った：筋の 5 を書き直し ours を o の補正として書く"
        else:
            key = "判定できない（受け皿の行）"
    else:
        return None
    return key + den


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e81", nargs="+", required=True, help="世界=場所（g3_split.py の出力．G3 の分割）")
    ap.add_argument("--e79", type=Path, default=ROOT / "results" / "e79", help="E79 の出力（measure/・delta/・select/）")
    ap.add_argument("--cur-e81", nargs="+", help="試し：世界=場所（g3_split.py --no-exclude の出力）を今の分割に使う（E79 の measure の代わり）")
    ap.add_argument("--chosen", type=Path, help="既定 <e79>/select/chosen.csv")
    ap.add_argument("--delta", type=Path, help="既定 <e79>/delta/delta.csv")
    ap.add_argument("--kept-used", type=Path, help="既定 <e79>/select/kept_used.csv（無ければ --kept から作る）")
    ap.add_argument("--kept", type=Path, default=ROOT / "results" / "e80" / "kept.csv")
    ap.add_argument("--e77", nargs="+", help="照合：世界=E77 の場所（学習側が変わらない単位の対数の確率）")
    ap.add_argument("--e78", nargs="+")
    ap.add_argument("--e80c", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--boot", type=int, help="既定は E79 の measure の provenance の値（無ければ 1000）")
    ap.add_argument("--seed", type=int, help="既定は E79 の measure の provenance の値（無ければ 0）")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e81_analysis")
    args = ap.parse_args()
    t_start = time.time()
    e81 = {k: Path(v) for k, v in kv(args.e81).items()}
    runs = [r for r in WORLDS81 if r in e81]
    chosen = args.chosen or args.e79 / "select" / "chosen.csv"
    args_chosen_path[0] = chosen
    ch = pd.read_csv(chosen, dtype={"K": str, "param": str, "meff": str, "wave": str, "method": str, "student": str, "target": str})
    mprov = args.e79 / "measure" / "provenance.json"
    mp = json.loads(mprov.read_text()) if mprov.exists() else {}
    B = args.boot if args.boot is not None else int(mp.get("boot", 1000))
    seed = args.seed if args.seed is not None else int(mp.get("seed", 0))
    args.out.mkdir(parents=True, exist_ok=True)
    checks = []
    if args.cur_e81:
        cur = {r: E81Store(r, Path(d), ch, B, seed, args.reps, checks, "cur") for r, d in kv(args.cur_e81).items() if r in runs}
        cur_src = "cur_e81"
    else:
        if not (args.e79 / "measure").exists():
            raise SystemExit(f"{args.e79 / 'measure'} が無い（E79 の段 (c) を先に流す．試しは --cur-e81）")
        c79 = E79Cur(args.e79 / "measure", runs)
        cur = {r: c79 for r in runs}
        cur_src = "e79_measure"
    g3 = {r: E81Store(r, e81[r], ch, B, seed, args.reps, checks, "g3") for r in runs}
    for r in runs:                                                               # 対にした抽出の照合：話者の並び（jk の人物）と複製の数
        if cur_src == "e79_measure":
            for T in g3[r].targets:
                ns = cur[r].n_speakers(r, T)                                     # 複製の抽出は話者の数と並びで決まる（boot_draws）
                if ns is not None and ns != len(g3[r].speakers.get(corp(T), [])):
                    raise SystemExit(f"{r} {T}：E79 の話者の数と E81 の話者の数が違う（抽出を対にできない）")
                checks.append({"check": "paired_n_speakers", "run": r, "target": T, "ok": ns is not None})
                pc = cur[r].persons(r, T)
                if pc is not None:
                    same = pc == g3[r].speakers.get(corp(T))
                    checks.append({"check": "paired_speakers", "run": r, "target": T, "ok": bool(same)})
                    if not same:
                        raise SystemExit(f"{r} {T}：E79 の jk の人物の並びと E81 の話者の並びが違う（抽出を対にできない）")
        else:
            for cp, sp in g3[r].speakers.items():
                same = cur[r].speakers.get(cp) == sp
                checks.append({"check": "paired_speakers", "run": r, "corpus": cp, "ok": bool(same)})
                if not same:
                    raise SystemExit(f"{r} {cp}：今の分割と G3 の話者の並びが違う")
    # δ と教師なしの単位
    dpath = args.delta or args.e79 / "delta" / "delta.csv"
    delta = pd.read_csv(dpath, dtype={"K": str}) if dpath.exists() else None
    DEL = {} if delta is None else {(x.run, x.student, x.K): x for x in delta.itertuples(index=False)}
    if delta is None:
        checks.append({"check": "delta_missing", "path": str(dpath), "ok": False})
    kpath = args.kept_used or args.e79 / "select" / "kept_used.csv"
    if kpath.exists() and kpath.stat().st_size > 1:
        kk = pd.read_csv(kpath, dtype={"K": str, "teachers": str}).fillna({"teachers": ""})
        KEPTU = {(a, b, c, d): tuple(t for t in e.split(";") if t) for a, b, c, d, e in zip(kk.run, kk.target, kk.student, kk.K, kk.teachers)}
        kept_src = str(kpath)
    else:
        kw = pd.read_csv(args.kept, dtype={"K": str})
        kw["keep"] = kw.keep.map(lambda v: v is True or str(v).strip().lower() in ("true", "1"))
        KEPTU = {(w, T, st, K): tuple(sorted(g.teacher[g.keep])) for (w, T, st, K), g in kw.groupby(["world", "target", "student", "K"])}
        for (w, T, st, K) in list(KEPTU):                                         # 台本は即興の行（E79 の load_kept と同じ）
            if T == "IEMOCAP-impro" and (w, "IEMOCAP-script", st, K) not in KEPTU:
                KEPTU[(w, "IEMOCAP-script", st, K)] = KEPTU[(w, T, st, K)]
        kept_src = str(args.kept)
    NOKEPT = {k_ for k_, v_ in KEPTU.items() if not v_}
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

    dtab, units = [], []
    for run in runs:
        studs = sorted(set(WORLDS[run] + [run]))
        targets = [t for t in g3[run].targets if t in groups_for(run)]
        byc = {}
        for T in targets:
            byc.setdefault(corp(T), []).append(T)
        for st in studs:
            role = "fused" if st == run else "uni"
            for K in KS:
                for cp, Ts in byc.items():
                    nt = [(run, T, st, K) in NOKEPT for T in Ts]
                    all_nt = bool(nt and all(nt))
                    for sel in SELS:                                             # Δ の表：手法 − none（3 つの量）
                        for metric, mname in (("uar", "uar"), ("dir", "dirloss"), ("pos", "posloss")):
                            # METRIC の dir・pos は y − x なので，手法 − none にするには x = none，y = 手法で呼ぶ
                            kx, ky = ((lambda T, s_=sel: pick(run, T, st, K, s_)), (lambda T: NONE)) if metric == "uar" else \
                                     ((lambda T: NONE), (lambda T, s_=sel: pick(run, T, st, K, s_)))
                            res = compare_both(cur[run], g3[run], run, cp, Ts, st, K, kx, ky, metric)
                            dtab.append({"run": run, "student": st, "role": role, "K": K, "corpus": cp, "n_targets": len(Ts), "method": sel,
                                         "metric": mname, "definition": "method - none", "no_teacher": all_nt,
                                         "chosen": ";".join("|".join(map(str, pick(run, T, st, K, sel) or ("missing",))) for T in Ts),
                                         **{f"{a}_{b}": v for a in ("cur", "g3", "dd") for b, v in zip(("point", "lo", "hi", "interval_rule"), res[a])},
                                         "g_eff_min": res["gmin"]})
                    for fam, (comp, sx, sy, metric, ntmode) in FAMS.items():      # 判定の再適用（E79 の compare と同じ向き：uar は x − y，dir は y − x）
                        res = compare_both(cur[run], g3[run], run, cp, Ts, st, K, lambda T, s_=sx: pick(run, T, st, K, s_),
                                           lambda T, s_=sy: pick(run, T, st, K, s_), metric)
                        dkind = METRIC[metric][2]
                        dr = DEL.get((run, st, K))
                        dl = np.nan if dr is None else float(dr.delta_d if dkind == "d" else dr.delta_u)
                        for split in ("cur", "g3"):
                            x, lo, hi, rule = res[split]
                            jd = jraw = judge_unit(x, lo, hi, dl)
                            if ntmode == "exclude" and all_nt:
                                jd = "no_teacher"
                            units.append({"family": fam, "comparison": comp, "metric": metric, "run": run, "student": st, "role": role, "K": K,
                                          "corpus": cp, "n_groups": len(Ts), "split": split, "point": x, "lo": lo, "hi": hi, "interval_rule": rule,
                                          "delta": dl, "judgment": jd, "judgment_raw": jraw, "no_teacher": all_nt, "g_eff_min": res["gmin"],
                                          "dd_point": res["dd"][0], "dd_lo": res["dd"][1], "dd_hi": res["dd"][2]})
    D = pd.DataFrame(dtab)
    U = pd.DataFrame(units)
    D.to_csv(args.out / "delta_table.csv", index=False)
    U.to_csv(args.out / "units.csv", index=False)
    # 判定の変わった単位の数
    piv = U.pivot_table(index=["family", "run", "student", "K", "corpus"], columns="split", values="judgment", aggfunc="first").reset_index()
    piv["changed"] = piv["cur"] != piv["g3"]
    chg = piv.groupby(["family", "run"]).agg(n_units=("changed", "size"), n_changed=("changed", "sum")).reset_index()
    tr = piv.groupby(["family", "run", "cur", "g3"]).size().rename("n").reset_index()
    chg.to_csv(args.out / "judgment_changes.csv", index=False)
    tr.to_csv(args.out / "judgment_transitions.csv", index=False)
    # 文の鍵（2 つのコーパスの過半）
    srows = []
    for (fam, run, st, K, split), g in U.groupby(["family", "run", "student", "K", "split"]):
        c = g.judgment.value_counts()
        nnt = int(c.get("no_teacher", 0))
        n_tot = len(g) - nnt
        rec = {"n_corpora": n_tot, "majority": majority(n_tot), "n_no_teacher": nnt,
               **{f"n_{k}": int(c.get(k, 0)) for k in ("pos", "neg", "same", "undet", "missing")},
               "n_oursx_none": int(g.no_teacher.sum()) if fam == "P5-2" else 0}
        srows.append({"family": fam, "run": run, "student": st, "K": K, "split": split, **rec})
    SC = pd.DataFrame(srows)
    st_rows = []
    for (fam, run, st, K), g in SC.groupby(["family", "run", "student", "K"]):
        if fam == "P5-3u":
            continue
        keys = {}
        for split, r_ in g.set_index("split").iterrows():
            u = None
            if fam == "P5-3":
                uu = SC[(SC.family == "P5-3u") & (SC.run == run) & (SC.student == st) & (SC.K == K) & (SC.split == split)]
                u = uu.iloc[0].to_dict() if len(uu) else None
            keys[split] = statement(fam, r_.to_dict(), u)
        st_rows.append({"family": fam, "run": run, "student": st, "K": K, "statement_cur": keys.get("cur"), "statement_g3": keys.get("g3"),
                        "same": keys.get("cur") == keys.get("g3")})
    ST = pd.DataFrame(st_rows)
    ST.to_csv(args.out / "statements.csv", index=False)
    SC.to_csv(args.out / "statement_counts.csv", index=False)
    n_diff = int((~ST.same).sum()) if len(ST) else 0
    flag = {"rule": "E81 の 4 節：G3 の分割での P5-1〜P5-3 の文の鍵（2 つのコーパスの過半）が今の分割と全部同じなら same（限定を付けず，Δ_G3 と ΔΔ を感度の表に載せる），"
                    "1 つでも違えば changed（筋 5 の主張を「候補がテストと同じ対話を含む設定で」と限り，要旨にも書く）",
            "flag": "same" if (len(ST) and n_diff == 0 and delta is not None) else ("changed" if n_diff else "undetermined"),
            "n_combos": int(len(ST)), "n_changed_combos": n_diff,
            "changed_by_family": ST[~ST.same].groupby("family").size().to_dict() if len(ST) else {},
            "delta_available": delta is not None}
    (args.out / "statement_flag.json").write_text(json.dumps(flag, ensure_ascii=False, indent=1))
    # 学習側が変わらない単位（IEMOCAP 台本の反復 3・8）：反復ごとの点の値が今の分割と同じか
    ident = []
    for run in runs:
        R = pd.concat(g3[run].rows_all, ignore_index=True) if g3[run].rows_all else pd.DataFrame()
        R.to_csv(args.out / f"rows_g3_{run}.csv.gz", index=False)
        if R.empty:
            continue
        if cur_src == "e79_measure":
            P = cur[run].S.point.get(run)
            if P is None:
                continue
            P = P[P.wave.isin(["e77", "e78", "e80c"])]
        else:
            P = pd.concat(cur[run].rows_all, ignore_index=True)
        for (T, r) in sorted(SAME_SPLIT):
            a = R[(R.target == T) & (R.rep == r)]
            if a.empty:
                continue
            b = P[(P.target == T) & (P.rep == r)]
            kk = ["student", "K", "wave", "method", "param", "meff"]
            j = a.merge(b, on=kk, how="left", suffixes=("", "_cur"), indicator=True)
            nmiss = int((j._merge != "both").sum())
            row = {"run": run, "target": T, "rep": r, "n_rows": len(a), "n_unmatched": nmiss}
            for q in ("uar", "dirloss", "posloss"):
                v = (j[q] - j[f"{q}_cur"]).abs()
                row[f"{q}_maxabs"] = float(v.max()) if v.notna().any() else np.nan
                row[f"{q}_n_over_tol"] = int((v > TOL_POINT).sum())
            row["ok"] = nmiss == 0 and all(row[f"{q}_n_over_tol"] == 0 for q in ("uar", "dirloss", "posloss"))
            ident.append(row)
        prod = {w: kv(getattr(args, w)).get(run) for w in ("e77", "e78", "e80c")}   # 対数の確率の照合（本番の E77・E78・E80c の場所を渡したとき）
        for (T, r) in sorted(SAME_SPLIT):
            f81 = e81[run] / f"{T}_r{r}_logp.npy"
            if not f81.exists():
                continue
            L81 = np.load(f81, mmap_mode="r")
            t81 = pd.read_csv(e81[run] / f"{T}.csv", dtype={"K": str, "param": str, "meff": str})
            t81 = t81[t81.rep == r]
            for w, dp in prod.items():
                if not dp or not (Path(dp) / f"{T}_r{r}_logp.npy").exists():
                    continue
                if not np.array_equal(np.load(Path(dp) / f"{T}_r{r}_te.npy"), np.load(e81[run] / f"{T}_r{r}_te.npy")):
                    ident.append({"run": run, "target": T, "rep": r, "wave": w, "check": "logp", "status": "te が違う", "ok": False})
                    continue
                Lp = np.load(Path(dp) / f"{T}_r{r}_logp.npy", mmap_mode="r")
                tp = pd.read_csv(Path(dp) / f"{T}.csv", dtype={"K": str, "param": str, "meff": str})
                tp = tp[tp.rep == r]
                src = t81[(t81.wave == w) | ((t81.method == "none") & (w == "e77"))]
                m = src.merge(tp, on=["student", "method", "param", "meff", "K"], suffixes=("", "_p"))
                if m.empty:
                    continue
                for meth, gm in m.groupby("method"):
                    v = max(float(np.abs(np.asarray(L81[int(a_)]) - np.asarray(Lp[int(b_)])).max()) for a_, b_ in zip(gm.row_id, gm.row_id_p))
                    pa = min(float((np.asarray(L81[int(a_)]).argmax(1) == np.asarray(Lp[int(b_)]).argmax(1)).mean()) for a_, b_ in zip(gm.row_id, gm.row_id_p))
                    ident.append({"run": run, "target": T, "rep": r, "wave": w, "check": "logp", "method": meth, "n_rows": len(gm), "n_src": int(len(src)),
                                  "logp_maxabs": v, "pred_agree_min": pa, "ok": v <= TOL_LOGP})
    pd.DataFrame(ident).to_csv(args.out / "identity_same_split.csv", index=False)
    C = pd.DataFrame(checks)
    C.to_csv(args.out / "checks.csv", index=False)
    prov = {"script": "exp/analyze_e81.py", "experiment": "E81", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": git("status", "--porcelain", "--", "exp", "measure").splitlines(), "script_sha256": sha256(__file__),
            "analyze_e79_sha256": sha256(A9.__file__), "cur_source": cur_src, "e81": {k: str(v) for k, v in e81.items()},
            "e81_provenance": {r: g3[r].prov.get("commit") for r in runs}, "chosen": str(chosen), "chosen_sha256": sha256(chosen),
            "delta": str(dpath), "delta_sha256": sha256(dpath) if dpath.exists() else None, "kept_source": kept_src, "boot": B, "seed": seed,
            "reps": args.reps, "e79_measure_provenance_sha256": sha256(mprov) if mprov.exists() else None, "geff_min_rule": GEFF_MIN,
            "seconds": round(time.time() - t_start, 1)}
    (args.out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))
    nbad = int((~C.ok.astype(bool)).sum()) if len(C) and "ok" in C else 0
    nid = len(ident)
    nid_bad = sum(1 for x in ident if not x.get("ok", True))
    print(f"Δ の表 {len(D)} 行，単位 {len(U)} 行，文 {len(ST)} 行，照合 {len(C)} 項目（食い違い {nbad}），同じ単位の照合 {nid} 項目（食い違い {nid_bad}）", flush=True)
    print(f"書いた場所：{args.out}（値は画面に出さない．statement_flag.json に文の鍵の同じ・変わったの判定）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
