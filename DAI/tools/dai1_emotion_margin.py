"""感情ごとに見た「疑似ラベルの当たり方 − 自分の当たり方」と，学び直しの感情ごとの再現率の差（計算の後の解析．2026-11-16．学び直しはしない）．

問い：計算の後の解析（`docs/結果/計算の後の解析_先生と自分の差とK10_2026-11-16.md` の要点 4〜6）は，疑似ラベルを作る分類器（先生）が
学び直す分類器（生徒）の K 本の基準の分類器より当たるか（UAR の差 margin）で，学び直しの UAR の差（gain）の向きがそろうことを示した．
ただし，先生のほうが UAR で当たるのに gain の区間が 0 を下回った点が 36（K = 5・25，teacher_margin.csv の除外の印の無い点）あった．
この台本は，同じ関係を感情ごとに見る：
  (a) 1 つの点の中で，先生が自分より当たる感情ほど，その感情の再現率が学び直しで上がったか．
  (b) UAR では先生のほうが当たるのに下がった点では，先生が自分より当たらない感情の再現率が下がったか．

量（単位 (分割, 種) ごとに作り，単位で平均する．感情 e = 0 Neutral，1 Happy，2 Sad，3 Angry）：
  t_rec_e  = 先生の判定の感情 e の再現率（K 本を除いた候補の上．%）
  s_rec_e  = 自分（K 本の基準の分類器）の判定の感情 e の再現率（同じ候補の上）
  margin_e = t_rec_e − s_rec_e
  gain_e   = 学び直し（tt の行．ρ = 1，a = 1）の感情 e の再現率 − K 本の基準の分類器の感情 e の再現率（テストの上）．
             区間は実験 1 の集計の recall.csv と同じ作り（`exp/dai1_desc.py` の t_recall．emo_of で感情 e の列だけを残した数の
             UAR の差を 4 倍．AFEW・IEMOCAP・MELD は塊の jackknife と種の項，MOSEI は塊の重みの引き直しと種の項）．
             乱数の列は表の名前 "teacher_types" の番号から作る（MOSEI の区間だけが乱数に依り，recall.csv の区間と最大 0.2 点ほど違う）．
先生：teacher_types.csv（B = none）の tt の先生の全部（AVT の生徒の fused_AVT は自分そのものなので除く）と主の先生．
  候補の上の判定は，保存した確率（p2 の *_pool は float16 で，同点の丸めで判定が実験 14 と食い違う発話がある）ではなく，
  保存した重み（float32）と作り直した標準化の特徴（`dai3.Saved`．実験 14 と同じ）から float64 で作る（`exp/dai1.py` 469〜481 行と同じ式）．
  single_m・pair_XY・fused_AVT・自分 → p1 の none[K] の W・b（`dai1.none_wb`），
  poe_none → none[K] の A・V・T の確率（1e-12 で下から切る）の対数の和，noret_sel → 同じ対数の指数 wsel[K] の重みつきの和，
  ret_m → p2 の p2_K<K>_ret_<m>_W・_b，主の先生（teacher = one）→ ret_A・ret_V・ret_T の確率の対数の和（指数 (1,1,1)），
  start_AVT → AVT の出発点（`dai1.Unit.start`），src_<c>_<kind> → ソースのコーパス c の kind の分類器（`dai1.Unit.src`）．
  主の先生の gain_e は族 1（prop − none．results/dai1/stats/desc/recall.csv の kind = family1 と同じ作り）．
  指数 wsel[K] が (1,1,1) の (T, K) の noret_sel は poe_none と同じ先生なので，excluded = same_as_poe_none の印を付けて要約から外す
  （teacher_margin.csv と同じ）．
K は 5，10，25（K = 0 は K 本の基準が出発点で，問いの margin が別の量になるので除く）．

照合（走らせるたびに確かめ，合わなければ止まる）：
  1. 作り直した先生の候補の上の UAR が，単位の p3.csv の _teacher_stats の uar_pool と合う（差が 0.05 点を超えたら止まる．差の最大と
     1e-6 を超えた単位の数を照合の表に残す）．
  2. 自分の候補の上の判定と K 本の印を，K = 5・25 で実験 14・22 の単位の none__K<K>（__<s>）と nk__K<K> と比べ，全部一致する．
     K 本の印は dai1.Unit の kpos とも一致する．同じ W・b で作ったテストの上の判定が，集計の NONE の行の予測と全部一致する．
     全単位のテストに 4 感情がある．
  3. 4 感情の gain_e の平均が，teacher_types.csv（B = none）の effect と 1e-9 で合う．主の先生は族 1 と合う．
  4. K = 5・25 の 4 感情の margin_e の平均が，teacher_margin.csv の margin と 1e-6 で合い，(T, 生徒, K) ごとに teacher_margin.csv の
     除外の印の無い先生の集合と，この台本の印の無い先生の集合が同じ．noret_sel の印も teacher_margin.csv と同じ．
  5. 主の先生の gain_e が recall.csv の family1 の effect と 1e-9 で合う．AFEW・IEMOCAP・MELD は lo・hi も合う．
  6. (T, 生徒, K) ごとの先生の数が，teacher_types.csv の行の数（AVT は fused_AVT を除く）＋ 1（主の先生）と同じ．

出力（--out-dir の下．既にあれば止まる）：
  emotion_margin.csv（点 × 感情），emotion_margin_summary.csv（記述の数え．(K, 生徒)，(T, K, 生徒)，(T, K) ほか），
  emotion_margin_units.csv.gz（単位 × 点 × 感情：t_rec，s_rec，候補の本数，先生と自分の真の感情の確率の平均），
  emotion_margin_confusion.csv.gz（4 × 4 の取り違えの単位の平均：先生と自分の候補の上，学び直しと K 本の基準のテストの上），
  emotion_margin_check.csv（照合），emotion_margin_provenance.json（台本と入力の sha256，単位のコミット，git の状態）．
記述の解析で予言は置かない．点どうしと感情どうしは同じ候補とテストを共有するので，順位相関と数え上げは検定ではない．
要約の top_emotion_same は，点の中で margin_e が最大の感情と gain_e が最大の感情が同じ点の数（偶然なら 1/4）．
hurt_* は「margin_uar > 0 かつ gain の UAR の区間の上端 < 0」の点の数えで，K と先生の集合で行を分ける（"5+25" の行が冒頭の 36 点に当たる）．
使い方：CUDA_VISIBLE_DEVICES="" .venv/bin/python tools/dai1_emotion_margin.py [--out-dir 一時の置き場] [--targets AFEW ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
import dai1  # noqa: E402
import dai1_desc as DS  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai3  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402
import torch  # noqa: E402

OUT1 = ROOT / "results" / "dai1"
KS = (5, 10, 25)
STUDENTS = ("A", "V", "T", "AVT")
TOL_POOL = 0.05                     # 照合 1：作り直した先生の候補の上の UAR と uar_pool の差の上限（点）
INPUTS = {"teacher_types": OUT1 / "stats" / "desc" / "teacher_types.csv", "families_flagged": OUT1 / "stats" / "families_flagged.csv",
          "outside_families": OUT1 / "stats" / "desc" / "outside_families.csv", "recall": OUT1 / "stats" / "desc" / "recall.csv",
          "teacher_margin": OUT1 / "stats" / "teacher_margin.csv", "frozen": OUT1 / "select" / "frozen.json"}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clog(L: torch.Tensor) -> torch.Tensor:
    """点数を確率に直し，1e-12 で下から切った確率の対数（`dai1.product`・`dai1.poe` の clamp(min=1e-12).log() と同じ）。"""
    return torch.softmax(L, -1).clamp(min=1e-12).log()


def source_logprobs(sv: "dai3.Saved") -> dict:
    """K に依らない先生（AVT の出発点とソースのコーパスごとの分類器）の候補の上の対数確率（`exp/dai1.py` 479〜481 行）。"""
    Xp = sv.u.Xp
    out = {"start_AVT": torch.log_softmax(S.logits(Xp["AVT"], *sv.u.start["AVT"]), -1).cpu().numpy()}
    for kind in ("AVT", "A", "V", "T"):
        for c, Wc, bc in sv.u.src[kind]:
            out[f"src_{c}_{kind}"] = torch.log_softmax(S.logits(Xp[kind], Wc, bc), -1).cpu().numpy()
    return out


def pool_logprobs(sv: "dai3.Saved", K: int, src_lp: dict) -> dict:
    """1 単位・1 つの K の候補の上の先生と自分の対数確率（正規化済み．float64 の numpy）。"""
    Xp = sv.u.Xp
    L = {f"none_{kind}": clog(S.logits(Xp[kind], *sv.none(kind, K))) for kind in DD.KINDS}
    for m in "AVT":
        W, b = S.t(sv.p2[f"p2_K{K}_ret_{m}_W"]), S.t(sv.p2[f"p2_K{K}_ret_{m}_b"])
        L[f"ret_{m}"] = clog(S.logits(Xp[m], W, b))
    L["poe_none"] = sum(L[f"none_{m}"] for m in "AVT")
    L["noret_sel"] = sum(float(w) * L[f"none_{m}"] for w, m in zip(sv.wsel[K], "AVT"))
    L["main_q"] = sum(L[f"ret_{m}"] for m in "AVT")
    out = {k: torch.log_softmax(v, -1).cpu().numpy() for k, v in L.items()}
    out.update(src_lp)
    return out


def teacher_key(name: str) -> str:
    if name.startswith("single_"):
        return f"none_{name[-1]}"
    if name.startswith("pair_"):
        return f"none_{name[5:]}"
    if name == "fused_AVT":
        return "none_AVT"
    return name                                        # poe_none，noret_sel，ret_m，main_q，start_AVT，src_*


def per_emotion(y: np.ndarray, lp: np.ndarray) -> tuple:
    """判定の感情ごとの再現率（%），本数，真の感情の確率の平均，4 × 4 の取り違えの数。"""
    p = lp.argmax(1)
    n = np.bincount(y, minlength=4).astype(float)
    if (n == 0).any():
        raise SystemExit("候補に欠ける感情がある")
    conf = np.zeros((4, 4))
    np.add.at(conf, (y, p), 1.0)
    rec = 100.0 * np.diag(conf) / n
    tp = np.array([np.exp(lp[y == e, e]).mean() for e in range(4)])
    return rec, n, tp, conf


def git_state() -> dict:
    cm, dt = dai1.git_commit()
    me = Path(__file__).resolve()
    st_ = subprocess.run(["git", "status", "--porcelain", str(me)], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dict(commit=cm, dirty=dt, script_status=st_ or "clean")


def run_target(T: str, frozen: dict, tt: pd.DataFrame, fam: pd.DataFrame, rec: pd.DataFrame, tm_all: pd.DataFrame):
    fz = frozen[T]
    R = int(fz["n_seeds_eval"])
    lab = DS.Labels(T, DD.ENC1)
    chk = Counter()
    units = [DS.Unit(OUT1 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}", f, r, lab, chk) for f in range(DD.N_FOLDS) for r in range(R)]
    for u in units:
        if not (np.bincount(u.y, minlength=4) > 0).all():
            raise SystemExit(f"{u.base.name}：テストに欠ける感情がある")
    g_eff, _ = ST.g_eff_of(T)
    ctx = DS.Ctx("eval", OUT1, T, T, fz, units, lab, g_eff, defaultdict(list), chk)
    checks = []
    names = {}
    for s in STUDENTS:
        for K in KS:
            nm = [n for n in DS.tt_names(units[0], s, K) if not (s == "AVT" and n == "fused_AVT")] + ["main_q"]
            n_tt = len(tt[(tt["T"] == T) & (tt.student == s) & (tt.K == K)])
            if len(nm) != n_tt - (1 if s == "AVT" else 0) + 1 or len(set(nm)) != len(nm):          # 照合 6
                raise SystemExit(f"{T}/{s}/K{K}：先生の数 {len(nm)} が teacher_types.csv の行の数 {n_tt} と合わない")
            names[(s, K)] = nm
            checks.append(dict(T=T, student=s, K=K, check="n_teachers", value=len(nm), n=n_tt))
    excl = {K: ("same_as_poe_none" if tuple(float(w) for w in fz["w"][str(K)]) == (1.0, 1.0, 1.0) else "") for K in KS}
    U = defaultdict(list)                   # (s, K, name) → 単位ごとの (rec, n, tp)
    Sf = defaultdict(list)                  # (s, K) → 自分の (rec, n, tp)
    CT = defaultdict(lambda: np.zeros((4, 4)))
    urows = []
    diffs = defaultdict(list)
    n_eq_test = Counter()
    for u in units:
        sv = dai3.Saved(OUT1, "eval", T, T, u.f, u.r, fz)
        sid_pool = u.p1["sample_pool"].astype(str)
        if not (sv.sid_pool == sid_pool).all():
            raise SystemExit(f"{u.base.name}：作り直した候補の並びが保存と違う")
        if any(tuple(float(w) for w in sv.wsel[K]) != tuple(float(w) for w in fz["w"][str(K)]) for K in KS):
            raise SystemExit(f"{u.base.name}：単位の指数が固定のファイルと違う")
        ypool = lab.yv[lab.ix(sid_pool)]
        ts = u.r3[u.r3.method == "_teacher_stats"]
        tsd = {}
        for r_ in ts.itertuples():
            k_ = (r_.student if isinstance(r_.student, str) else "", int(r_.K), str(r_.teacher))
            if k_ in tsd:
                raise SystemExit(f"{u.base.name}：_teacher_stats の行が重なる（{k_}）")
            tsd[k_] = float(r_.uar_pool)
        src_lp = source_logprobs(sv)
        for K in KS:
            nk = ~np.isin(sid_pool, u.p1[f"sample_K{K}"].astype(str))
            if nk.sum() != len(sid_pool) - 4 * K:
                raise SystemExit(f"{u.base.name}：K 本を除いた候補の数が合わない（K={K}）")
            if not np.array_equal(np.setdiff1d(np.arange(len(sid_pool)), sv.u.kpos[K]), np.flatnonzero(nk)):
                raise SystemExit(f"{u.base.name}：K 本の印が dai1.Unit の kpos と違う（K={K}）")
            LP = pool_logprobs(sv, K, src_lp)
            y = ypool[nk]
            pe = {k: per_emotion(y, v[nk]) for k, v in LP.items()}
            for s in STUDENTS:
                own = pe[f"none_{s}"]
                Sf[(s, K)].append(own[:3])
                CT[(s, K, "(self)", "self_pool")] += own[3]
                if K in (5, 25):                                                  # 照合 2
                    if s == "AVT":
                        a = np.load(ROOT / "results" / "dai22" / "units" / f"{T}_f{u.f}_r{u.r}.npz", allow_pickle=False)
                        ref = a[f"none__K{K}"]
                    else:
                        a = np.load(ROOT / "results" / "dai14" / "units" / f"{T}_f{u.f}_r{u.r}.npz", allow_pickle=False)
                        ref = a[f"none__K{K}__{s}"]
                    if not (a["sid"].astype(str) == sid_pool).all() or not np.array_equal(a[f"nk__K{K}"].astype(bool), nk):
                        raise SystemExit(f"{u.base.name}：実験 14／22 の候補の並びか K 本の印が違う")
                    nd = int((ref.astype(int) != LP[f"none_{s}"].argmax(1)).sum())
                    if nd:
                        raise SystemExit(f"{u.base.name}：自分の候補の上の判定が実験 14／22 と {nd} 発話違う（{s}，K={K}）")
                pt = S.logits(sv.u.Xt[s], *sv.none(s, K)).argmax(1).cpu().numpy()
                n_eq_test[(s, K)] += int(np.array_equal(pt, u.pred(DS.fam_rows(fz, s, K)["none"])))
                for name in names[(s, K)]:
                    r_, n_, tp_, cf_ = pe[teacher_key(name)]
                    U[(s, K, name)].append((r_, n_, tp_))
                    CT[(s, K, name, "teacher_pool")] += cf_
                    key = ("", K, "one") if name == "main_q" else (s, K, f"tt:{name}")
                    if key not in tsd:
                        raise SystemExit(f"{u.base.name}：_teacher_stats の行が無い（{key}）")
                    diffs[(s, K, name)].append(abs(r_.mean() - tsd[key]))       # 照合 1
                    for e in range(4):
                        urows.append(dict(T=T, f=u.f, r=u.r, student=s, K=K, teacher=name, emotion=e, t_rec=r_[e], s_rec=own[0][e],
                                          n_pool_e=int(n_[e]), t_trueprob=tp_[e], s_trueprob=own[2][e]))
        del sv, src_lp
    print(f"  {T}：候補の上の判定を作り直した", flush=True)
    for (s, K), v in n_eq_test.items():
        checks.append(dict(T=T, student=s, K=K, check="none_test_equal_units", value=v, n=len(units)))
        if v != len(units):
            raise SystemExit(f"{T}/{s}/K{K}：作り直した none のテストの判定が集計の NONE の行と違う単位がある")
    rows = []
    tm = tm_all[tm_all["T"] == T]
    for s in STUDENTS:
        for K in KS:
            Rf = DS.fam_rows(fz, s, K)
            cB = ctx.cell(Rf["none"])
            s_rec = np.array([x[0] for x in Sf[(s, K)]])
            s_tp = np.array([x[2] for x in Sf[(s, K)]])
            for u in units:
                np.add.at(CT[(s, K, "(self)", "none_test")], (u.y, u.pred(Rf["none"])), 1.0)
            mine = set()
            for name in names[(s, K)]:
                d = np.array(diffs[(s, K, name)])
                checks.append(dict(T=T, student=s, K=K, teacher=name, check="teacher_uar_pool_maxdiff", value=float(d.max()), n=int((d > 1e-6).sum())))
                if d.max() > TOL_POOL:
                    raise SystemExit(f"{T}/{s}/K{K}/{name}：先生の候補の上の UAR が uar_pool と {d.max():.3f} 点違う")
                t_rec = np.array([x[0] for x in U[(s, K, name)]])
                t_n = np.array([x[1] for x in U[(s, K, name)]])
                t_tp = np.array([x[2] for x in U[(s, K, name)]])
                specA = Rf["prop"] if name == "main_q" else DS.P3(s, K, method="tt", teacher=name, rho=1.0, a=1)
                cA = ctx.cell(specA)
                for u in units:
                    np.add.at(CT[(s, K, name, "retrained_test")], (u.y, u.pred(specA)), 1.0)
                rng = ctx.rng("teacher_types")
                es = [DS._fmt(ST.effect_stats(DS.emo_of(cA, e), DS.emo_of(cB, e), ctx.g_eff, rng), K, scale=4.0) for e in range(4)]
                ref = (fam[(fam["T"] == T) & (fam.student == s) & (fam.K == K)] if name == "main_q" else
                       tt[(tt["T"] == T) & (tt.student == s) & (tt.K == K) & (tt.teacher == name)])
                if len(ref) != 1:
                    raise SystemExit(f"{T}/{s}/K{K}/{name}：UAR の差の行が {len(ref)} 個")
                d3 = abs(np.mean([x["effect"] for x in es]) - float(ref.effect.iloc[0]))       # 照合 3
                checks.append(dict(T=T, student=s, K=K, teacher=name, check="gain_mean_vs_uar_effect", value=float(d3), n=1))
                if d3 > 1e-9:
                    raise SystemExit(f"{T}/{s}/K{K}/{name}：4 感情の差の平均が UAR の差と {d3:.2e} 違う")
                if name == "main_q":                                                         # 照合 5
                    rr_ = rec[(rec["T"] == T) & (rec.kind == "family1") & (rec.student == s) & (rec.K == K)].sort_values("emotion")
                    if len(rr_) != 4:
                        raise SystemExit(f"{T}/{s}/K{K}：recall.csv の family1 の行が {len(rr_)} 個")
                    d5 = max(abs(es[e]["effect"] - float(rr_.effect.iloc[e])) for e in range(4))
                    d5b = max(max(abs(es[e]["lo"] - float(rr_.lo.iloc[e])), abs(es[e]["hi"] - float(rr_.hi.iloc[e]))) for e in range(4))
                    checks.append(dict(T=T, student=s, K=K, teacher=name, check="family1_recall_effect", value=float(d5), n=1))
                    checks.append(dict(T=T, student=s, K=K, teacher=name, check="family1_recall_lohi", value=float(d5b), n=1))
                    if d5 > 1e-9 or (T != "MOSEI" and d5b > 1e-9):
                        raise SystemExit(f"{T}/{s}/K{K}：感情ごとの差が recall.csv と違う（{d5:.2e}，{d5b:.2e}）")
                mg = (t_rec - s_rec).mean(0)
                ex = excl[K] if name == "noret_sel" else ""
                if not ex:
                    mine.add(name)
                if K in (5, 25):                                                             # 照合 4
                    m4 = tm[(tm.student == s) & (tm.K == K) & (tm.teacher == name)]
                    if len(m4) != 1:
                        raise SystemExit(f"{T}/{s}/K{K}/{name}：teacher_margin.csv の行が {len(m4)} 個")
                    if str(m4.excluded.fillna("").iloc[0]) != ex:
                        raise SystemExit(f"{T}/{s}/K{K}/{name}：除外の印が teacher_margin.csv と違う")
                    d4 = abs(mg.mean() - float(m4.margin.iloc[0]))
                    checks.append(dict(T=T, student=s, K=K, teacher=name, check="margin_mean_vs_teacher_margin", value=float(d4), n=1))
                    if d4 > 1e-6:
                        raise SystemExit(f"{T}/{s}/K{K}/{name}：感情ごとの margin の平均が teacher_margin.csv と {d4:.2e} 違う")
                no_own = (s != "AVT" and name != "main_q" and s not in DS.teacher_mods(name)
                          and not name.startswith(("fused", "poe", "noret", "ret", "start")))      # `dai1_desc.t_teacher_types` と同じ
                for e in range(4):
                    rows.append(dict(T=T, student=s, K=K, teacher=name, excluded=ex, no_own_modality=no_own, emotion=e,
                                     teacher_rec_pool=float(t_rec[:, e].mean()), self_rec_pool=float(s_rec[:, e].mean()),
                                     margin_e=float(mg[e]), margin_uar=float(mg.mean()),
                                     teacher_trueprob_pool=float(t_tp[:, e].mean()), self_trueprob_pool=float(s_tp[:, e].mean()),
                                     n_pool_e_mean=float(t_n[:, e].mean()),
                                     gain_e=float(es[e]["effect"]), gain_e_lo=float(es[e]["lo"]), gain_e_hi=float(es[e]["hi"]),
                                     interval=es[e].get("interval"), n_units=int(es[e]["n_units"]),
                                     gain_uar=float(ref.effect.iloc[0]), gain_uar_lo=float(ref.lo.iloc[0]), gain_uar_hi=float(ref.hi.iloc[0])))
                if name != "main_q":                                                         # no_own_modality の印を teacher_types.csv と照らす
                    if bool(ref.no_own_modality.iloc[0]) != no_own:
                        raise SystemExit(f"{T}/{s}/K{K}/{name}：no_own_modality が teacher_types.csv と違う")
            if K in (5, 25):
                theirs = set(tm[(tm.student == s) & (tm.K == K) & (tm.excluded.fillna("") == "")].teacher)
                if theirs != mine:
                    raise SystemExit(f"{T}/{s}/K{K}：先生の集合が teacher_margin.csv と違う（{sorted(theirs ^ mine)}）")
                checks.append(dict(T=T, student=s, K=K, check="teacher_set_equals_teacher_margin", value=len(mine), n=len(theirs)))
            ctx.clear()
        print(f"  {T} 生徒 {s} 済み", flush=True)
    if any(k[1] in ("missing_row", "mismatch") and v > 0 for k, v in chk.items()):
        raise SystemExit(f"{T}：集計の予測の照合で欠けか食い違いがある {dict(chk)}")
    checks += [dict(T=T, check=f"pred_{a}_{b}", value=v) for (a, b), v in chk.items()]
    crow = [dict(T=T, student=s, K=K, teacher=nm, who=who, true=i, pred=j, count_mean=float(M[i, j] / len(units)))
            for (s, K, nm, who), M in CT.items() for i in range(4) for j in range(4)]
    return rows, checks, urows, crow, sorted({u.commit for u in units})


def summarize(D: pd.DataFrame) -> pd.DataFrame:
    out = []
    D = D[D.excluded == ""].copy()
    key = ["T", "student", "K", "teacher"]
    D["margin_dev"] = D.margin_e - D.groupby(key).margin_e.transform("mean")
    D["gain_dev"] = D.gain_e - D.groupby(key).gain_e.transform("mean")
    groups = ([(("all", K, s), x) for (K, s), x in D.groupby(["K", "student"])] + [(("all", K, "all"), D[D.K == K]) for K in KS]
              + [(("all", "5+25", "all"), D[D.K.isin((5, 25))]), (("all", "all", "all"), D)]
              + [((T, K, s), x) for (T, K, s), x in D.groupby(["T", "K", "student"])] + [((T, K, "all"), x) for (T, K), x in D.groupby(["T", "K"])])
    sp = lambda a, b: float(st.spearmanr(a, b).statistic) if len(a) > 2 else np.nan
    for (T, K, s), x in groups:
        g = x.groupby(key)
        top_same = int(sum(int(gg.margin_e.idxmax() == gg.gain_e.idxmax()) for _, gg in g))
        hurt = x[(x.margin_uar > 0) & (x.gain_uar_hi < 0)]                    # UAR では先生のほうが当たるのに下がった点
        hp = hurt.groupby(key)
        hn, hpo = hurt[hurt.margin_e < 0], hurt[hurt.margin_e >= 0]
        out.append(dict(T=T, K=K, student=s, n_points=g.ngroups, n_rows=len(x),
                        spearman_e=sp(x.margin_e, x.gain_e), spearman_dev=sp(x.margin_dev, x.gain_dev),
                        top_emotion_same=top_same,
                        e_margin_pos=int((x.margin_e > 0).sum()), e_margin_neg=int((x.margin_e < 0).sum()),
                        e_margin_pos_gain_ci_pos=int(((x.margin_e > 0) & (x.gain_e_lo > 0)).sum()),
                        e_margin_pos_gain_ci_neg=int(((x.margin_e > 0) & (x.gain_e_hi < 0)).sum()),
                        e_margin_neg_gain_ci_pos=int(((x.margin_e < 0) & (x.gain_e_lo > 0)).sum()),
                        e_margin_neg_gain_ci_neg=int(((x.margin_e < 0) & (x.gain_e_hi < 0)).sum()),
                        hurt_points=hp.ngroups,
                        hurt_points_with_neg_margin_e=int(sum(int((gg.margin_e < 0).any()) for _, gg in hp)),
                        hurt_points_worst_gain_e_has_neg_margin=int(sum(int(gg.loc[gg.gain_e.idxmin(), "margin_e"] < 0) for _, gg in hp)),
                        hurt_neg_margin_rows=len(hn), hurt_neg_margin_gain_e_mean=float(hn.gain_e.mean()) if len(hn) else np.nan,
                        hurt_pos_margin_rows=len(hpo), hurt_pos_margin_gain_e_mean=float(hpo.gain_e.mean()) if len(hpo) else np.nan))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT1 / "stats"))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    args = ap.parse_args()
    sdir = Path(args.out_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    outs = ("emotion_margin.csv", "emotion_margin_summary.csv", "emotion_margin_units.csv.gz", "emotion_margin_confusion.csv.gz",
            "emotion_margin_check.csv", "emotion_margin_provenance.json")
    for name in outs:
        if (sdir / name).exists():
            raise SystemExit(f"{sdir / name} が既にある（上書きしない）")
    frozen = json.load(open(INPUTS["frozen"]))
    tt = pd.read_csv(INPUTS["teacher_types"], comment="#")
    tt = tt[(tt.B == "none") & tt.K.isin(KS)]
    fam = pd.read_csv(INPUTS["families_flagged"], comment="#")
    famK10 = pd.read_csv(INPUTS["outside_families"], comment="#")
    fam = pd.concat([fam[(fam.family == 1) & fam.K.isin((5, 25))], famK10[(famK10.family == 1) & (famK10.K == 10)]])
    rec = pd.read_csv(INPUTS["recall"], comment="#")
    tm = pd.read_csv(INPUTS["teacher_margin"])
    rows, checks, urows, crows, ucommits = [], [], [], [], {}
    for T in args.targets:
        r_, c_, u_, cf_, uc = run_target(T, frozen, tt, fam, rec, tm)
        rows += r_
        checks += c_
        urows += u_
        crows += cf_
        ucommits[T] = uc
        print(f"{T}：{len(r_)} 行", flush=True)
    D = pd.DataFrame(rows)
    gs = git_state()
    D.assign(commit=gs["commit"], dirty=gs["dirty"]).to_csv(sdir / "emotion_margin.csv", index=False)
    summarize(D).assign(commit=gs["commit"], dirty=gs["dirty"]).to_csv(sdir / "emotion_margin_summary.csv", index=False)
    pd.DataFrame(urows).to_csv(sdir / "emotion_margin_units.csv.gz", index=False)
    pd.DataFrame(crows).to_csv(sdir / "emotion_margin_confusion.csv.gz", index=False)
    pd.DataFrame(checks).to_csv(sdir / "emotion_margin_check.csv", index=False)
    prov = dict(script=str(Path(__file__).resolve().relative_to(ROOT)), script_sha256=sha(Path(__file__)), git=gs, targets=args.targets,
                units_commit=ucommits, inputs={k: dict(path=str(v.relative_to(ROOT)), sha256=sha(v)) for k, v in INPUTS.items()},
                torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, device=str(S.DEV))
    json.dump(prov, open(sdir / "emotion_margin_provenance.json", "w"), indent=1, ensure_ascii=False)
    print(f"書いた：{sdir / 'emotion_margin.csv'}（{len(D)} 行）ほか 5 つ", flush=True)


if __name__ == "__main__":
    main()
