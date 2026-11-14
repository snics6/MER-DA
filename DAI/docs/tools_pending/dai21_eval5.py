"""実験 21（DAI-21）：提案の評価の回の続き．(1) 提案の答えを疑似ラベルにして融合の分類器を学び直す（5 コーパス），(2) CREMA-D を評価のターゲットにする．
設計：DAI/docs/実験21_設計.md．

段：
  selftest   人工の数での検算
  check3     まとめ方の計算（combine_all）を，実験 16 の下調べの保存（3 つの分類器）に当てて保存の判定と照らす（読むだけ）
  check18    CREMA-D の単位の計算を，実験 18 の単位（RAVDESS を除いた回）と同じ設定で流して保存の判定と照らす
  freeze     固定のファイル DAI/frozen/dai21_frozen.json を作る（実験 20 の計算の後．コミットしてから計算する）
  students4  4 コーパス：実験 20 の単位の出力から疑似ラベルを作り，融合の分類器を学び直す
  cremad     CREMA-D：土台（K 本だけの分類器，微調整した声・顔）→ まとめ方 → 学び直し
  summary    UAR と区間，差と区間

実験 20 の計算のあいだは DAI/exp に置かない（DAI/docs/tools_pending/ に置き，計算の後に DAI/exp へ移す）．
使い方：.venv/bin/python <この台本> --stage cremad
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/snics/MER-DA/DAI/exp")
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai15  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai18_probe as P18  # noqa: E402
import dai20_eval as P20  # noqa: E402
import dai3  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT18 = ROOT / "results" / "dai18_probe"
OUT20 = ROOT / "results" / "dai20_eval"
OUT_PROD = ROOT / "results" / "dai21_eval5"
DESIGN = ROOT / "DAI" / "docs" / "実験21_設計.md"
FROZEN = ROOT / "DAI" / "frozen" / "dai21_frozen.json"
FROZEN20 = ROOT / "DAI" / "frozen" / "dai20_frozen.json"
C4 = tuple(DD.CORPORA)
CD, HOLD = "CREMA-D", "RAVDESS"          # CREMA-D の単位は，実験 18 の単位の作り（Unit6）で RAVDESS を除く側に置く．ソースは 4 コーパスになる
KS = (5, 10, 25)
NF = P16.NF
GRID = P16.GRID
NG = len(GRID)
SGRID = P18.SGRID                        # 学び直しの設定の格子：(ρ, a) の 12 点（実験 1 と同じ）
NS = len(SGRID)
SEED = 20261114 + 21
WF_MS = P20.WF_MS
N_WF = P20.N_WF
AGREE_MIN = P18.NONE_AGREE               # 保存の確率（float32）から作り直した判定と保存の判定の一致の下限（0.995）
SEED_SETS = {"all": None, "0-1": (0, 1), "2-9": tuple(range(2, 10))}
TEMP_MODES = P20.TEMP_MODES
B_VERS = P20.B_VERS
A2 = ("prod", "avg", "conf", "M_F2_K", "M_F6_K", "cf_D_l05", "tcal", "temp", "tempb") + tuple(f"wf{m}" for m in WF_MS)     # CREMA-D で比べる相手（別のコーパスで選んだ指数の積は無い）
SCRIPTS = ("dai20_eval.py", "dai18_probe.py", "dai17_probe.py", "dai16_probe.py", "dai15.py", "dai14.py", "dai12_probe.py", "dai11_probe.py", "dai3.py", "dai1.py",
           "dai1_stats.py", "dai_solver.py", "dai_data.py")
RUN = {"commit": "", "dirty": True}


def tables_of(nm: int) -> np.ndarray:
    """重みの表を探す版の表（実験 20 と同じ作り方を，分類器の数 nm に合わせたもの）．3 つなら実験 20 の表そのもの．"""
    if nm == 3:
        return P20.WS
    return np.concatenate([np.ones((1, nm, 4)), nm * np.random.default_rng([SEED, 1]).dirichlet(np.ones(nm), size=(N_WF, 4)).transpose(0, 2, 1)])


def seed_rng(*key) -> np.random.Generator:
    return P16.row_rng("dai21", *key)


# ------------------------------------------------------------------ まとめ方（分類器の数に依らない形）

def wf_search(tables: np.ndarray, LMh: np.ndarray, yk: np.ndarray, fold: np.ndarray, LMpj: list) -> tuple[np.ndarray, int]:
    """実験 20 の wf_search と同じ計算（表を引数で受け取る）：表ごとに，K 本の上の UAR（%）を切片をずらした答えで測る．"""
    dv = P20.dev()
    W = torch.tensor(tables, dtype=torch.float64, device=dv)
    pred = torch.empty((len(tables), len(yk)), dtype=torch.long, device=dv)
    ncap = 0
    for j in range(NF):
        te = np.flatnonzero(fold == j)
        Lj = torch.tensor(LMpj[j], dtype=torch.float64, device=dv)
        tgt = torch.softmax(Lj.sum(0), -1).mean(0)
        step = max(1, int(2.5e7 // (Lj.shape[1] * 4)))
        Lh = torch.tensor(LMh[:, te], dtype=torch.float64, device=dv)
        for a in range(0, len(tables), step):
            Sp = torch.einsum("rky,kny->rny", W[a:a + step], Lj)
            Sp = Sp - torch.logsumexp(Sp, -1, keepdim=True)
            d, nc = P20.match_shift_batch(Sp, tgt)
            ncap += nc
            Sh = torch.einsum("rky,kny->rny", W[a:a + step], Lh)
            Sh = Sh - torch.logsumexp(Sh, -1, keepdim=True)
            pred[a:a + step, te] = (Sh + d[:, None, :]).argmax(-1)
            del Sp
    y = torch.tensor(yk, device=dv)
    hit = (pred == y[None]).double()
    ua = torch.stack([hit[:, y == c].mean(1) for c in range(4) if bool((y == c).any())], 1).mean(1)
    return (100.0 * ua).cpu().numpy(), ncap


def wf_pick(tables: np.ndarray, ua: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    u = ua[:m + 1]
    tied = np.flatnonzero(u >= u.max() - 1e-9)
    return tables[tied].mean(0), tied


def combine_all(arr: dict, rows: list, K: int, Pt: list, Pp: list, Ph: list, P2: list, PJ: list, kpos, fold, yk, yt, yp,
                DK, DO, Dext, Dhome, tables: np.ndarray) -> None:
    """分類器の答え（Pt：テスト，Pp：候補，Ph：K 本の抜いた確率，P2：2 組を抜いた確率，PJ：組 j を抜いた分類器の候補の上の確率）から，
    提案・比べる相手・部品を外した版の判定を作る．式は実験 16 の run_unit と実験 20 の add_extra と同じ（分類器の数だけを一般にした）．"""
    nm = len(Pt)
    nk = np.setdiff1d(np.arange(len(yp)), kpos)
    pairs = list(itertools.combinations(range(NF), 2))
    pidx = {p: n for n, p in enumerate(pairs)}
    LMt, LMp, LMh = (np.stack([P16.centered(p) for p in Ps]) for Ps in (Pt, Pp, Ph))
    LMpj = [np.stack([P16.centered(PJ[k][j]) for k in range(nm)]) for j in range(NF)]
    target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)

    def put(v: str, st: np.ndarray, sp: np.ndarray, **extra) -> None:
        St, Sp = P16.norm_log(st), P16.norm_log(sp)
        if not (np.isfinite(St).all() and np.isfinite(Sp).all()):
            raise SystemExit(f"まとめた点が有限でない（{v} K={K}）")
        dl, rs = P16.match_shift(Sp, target)
        pr, pj, pp = St.argmax(1), (St + dl).argmax(1), (Sp + dl)[nk].argmax(1)
        arr[f"pred_raw_K{K}_{v}"], arr[f"pred_adj_K{K}_{v}"], arr[f"pred_pool_K{K}_{v}"] = pr.astype(np.int8), pj.astype(np.int8), pp.astype(np.int8)
        rows.append(dict(K=K, version=v, uar_raw=100 * S.uar(yt, pr), uar_adj=100 * S.uar(yt, pj), uar_pool=100 * S.uar(yp[nk], pp), adj_resid=rs, **extra))

    def f6(W: np.ndarray, LM: np.ndarray) -> np.ndarray:
        return np.einsum("ky,kny->ny", W, LM)

    put("prod", LMt.sum(0), LMp.sum(0))
    Dm = {"K": DK, "ext": Dext, "orc": DO, **{f"mix{i}": P16.mix_rel(DK, Dext, 4 * K, n0, b_, pr) for i, (n0, b_, pr) in enumerate(GRID)}}
    for r_, D in Dm.items():
        W = P16.to_w(D)
        put(f"M_F6_{r_}", f6(W, LMt), f6(W, LMp))
        if r_ == "orc" or r_.startswith("mix"):
            Wn = P16.noint(W)
            put(f"M_F6n_{r_}", f6(Wn, LMt), f6(Wn, LMp))
    w2 = P16.to_w(DK).mean(1)
    put("M_F2_K", np.tensordot(w2, LMt, axes=1), np.tensordot(w2, LMp, axes=1))
    tabs = [P16.shaped(dai15.table_from(Ph[k], yk), "D", 0.5) for k in range(nm)]
    put("cf_D_l05", sum(P16.lg(P16.reread(Pt[k], tabs[k][0])) for k in range(nm)), sum(P16.lg(P16.reread(Pp[k], tabs[k][0])) for k in range(nm)),
        n_unused=int(sum(t_[1] for t_ in tabs)))
    put("avg", P16.lg(np.mean(Pt, 0)), P16.lg(np.mean(Pp, 0)))
    ct, cp = np.stack([p.max(1) for p in Pt]), np.stack([p.max(1) for p in Pp])
    put("conf", P16.lg((ct[:, :, None] * np.stack(Pt)).sum(0) / ct.sum(0)[:, None]), P16.lg((cp[:, :, None] * np.stack(Pp)).sum(0) / cp.sum(0)[:, None]))
    for v, mode in TEMP_MODES.items():
        w = P20.temp_w(LMh, yk, mode)
        est = np.empty((len(kpos), 4))
        for j in range(NF):
            te, tr = np.flatnonzero(fold == j), np.flatnonzero(fold != j)
            Ltr = np.stack([P16.centered(np.stack([P2[k][pidx[tuple(sorted((int(fold[a]), j)))]][a] for a in tr])) for k in range(nm)])
            wj = P20.temp_w(Ltr, yk[tr], mode)
            dl, _ = P16.match_shift(P16.norm_log(np.tensordot(wj, LMpj[j], axes=1)), np.exp(P16.norm_log(LMpj[j].sum(0))).mean(0))
            est[te] = P16.norm_log(np.tensordot(wj, LMh[:, te], axes=1) + dl)
        put(v, np.tensordot(w, LMt, axes=1), np.tensordot(w, LMp, axes=1), w_bound=float(((w <= 1e-9) | (w >= P16.W_MAX - 1e-9)).mean()),
            est_uar=100 * S.uar(yk, est.argmax(1)), est_ll=P16.bal_ll(est, yk), **{f"w_{k}": float(w[k]) for k in range(nm)})
        arr[f"W_{v}_K{K}"] = w
    ua, ncap = wf_search(tables, LMh, yk, fold, LMpj)
    arr[f"wf_uar_K{K}"] = ua.astype(np.float32)
    for m in WF_MS:
        W, tied = wf_pick(tables, ua, m)
        ex = dict(wf_index=int(tied[0]), wf_n_tied=len(tied), wf_uar_k=float(ua[tied[0]]), wf_uar_k_eq=float(ua[0]), wf_n_capped=ncap)
        arr[f"wf_tied{m}_K{K}"] = tied.astype(np.int16)
        if len(tied) == 1 and tied[0] == 0:                                      # 等しい重みの表だけが最大：積の判定をそのまま使う
            for w_ in ("raw", "adj"):
                arr[f"pred_{w_}_K{K}_wf{m}"] = arr[f"pred_raw_K{K}_prod"]
            arr[f"pred_pool_K{K}_wf{m}"] = arr[f"pred_pool_K{K}_prod"]
            u_ = 100 * S.uar(yt, arr[f"pred_raw_K{K}_prod"].astype(int))
            rows.append(dict(K=K, version=f"wf{m}", uar_raw=u_, uar_adj=u_, uar_pool=100 * S.uar(yp[nk], arr[f"pred_pool_K{K}_prod"].astype(int)), adj_resid=0.0, **ex))
        else:
            put(f"wf{m}", f6(W, LMt), f6(W, LMp), **ex)
        arr[f"W_wf{m}_K{K}"] = W
    for i, (n0, beta, prior) in enumerate(GRID):                              # 相性をソースの中の値（残り方を掛けない）から取る版
        if beta == 1.0:
            lev = P16.mix_rel(DK, Dext, 4 * K, n0, 0.0, prior)
            Wh = P16.to_w(np.maximum(0.0, lev + (Dhome - Dhome.mean(1, keepdims=True))))
            put(f"M_F6h_mix{i}", f6(Wh, LMt), f6(Wh, LMp))


def teacher_probs(LMp: np.ndarray, W: np.ndarray | None) -> np.ndarray:
    """疑似ラベル（候補の上の確率）．W が None なら積，そうでなければ提案（重み W で点を足し，切片をずらす）．実験 18 の teacher_pool と同じ式．"""
    if W is None:
        return np.exp(P16.norm_log(LMp.sum(0)))
    Sp = P16.norm_log(np.einsum("ky,kny->ny", W, LMp))
    dl, _ = P16.match_shift(Sp, np.exp(P16.norm_log(LMp.sum(0))).mean(0))
    return np.exp(P16.norm_log(Sp + dl))


def fit_students(X, Xt, W0, b0, cw, cb, gk, n: int, teachers: dict, label: str):
    """実験 1 の同時に解く版と同じ式（dai1.make_G の joint）：重み ＝ a × K 本 ＋ 疑似ラベル（感情ごとの総量をそろえ，合計 ρ × 候補の本数）．
    疑似ラベル × 設定の格子（SGRID の 12 点）をまとめて解く．戻り：{(疑似ラベルの名前, 格子の番号): (テストの判定, 候補の判定)}，勾配の最大，止まりの数．"""
    names = list(teachers)
    G = torch.stack([float(a_) * gk + S.soft_rows(S.t(teachers[t_]), dai1.mval(r_, n)) for t_ in names for r_, a_ in SGRID])
    res = S.fit(X, G, W0, b0, Cw=cw, Cb=cb)
    if float(res["grad_max"]) > P16.GRAD_TOL or not res["finite_p"].all():
        raise SystemExit(f"学び直しが解けていない（{label}．勾配の最大 {float(res['grad_max']):.1e}）")
    Zt, Zp = S.logits(Xt, res["W"], res["b"]).argmax(-1).cpu().numpy(), S.logits(X, res["W"], res["b"]).argmax(-1).cpu().numpy()
    return {(t_, gi): (Zt[j * NS + gi], Zp[j * NS + gi]) for j, t_ in enumerate(names) for gi in range(NS)}, float(res["grad_max"]), int(res["stalled"])


# ------------------------------------------------------------------ 固定のファイル

def sgrid_index(rho, a) -> int:
    return SGRID.index((float(rho), int(float(a))))


def stage_freeze() -> None:
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} が既にある（作り直すなら消してから）")
    fz1, _ = dai3.load_fz1(OUT1)
    f20 = json.load(open(FROZEN20))
    ssel = pd.read_csv(OUT18 / "summary" / "students_sel.csv")
    fsel = json.load(open(OUT18 / "frozen_sel.json"))["choice"]
    teth18 = json.load(open(OUT18 / "tether.json"))["tether"]

    def sel18(c, K, student, teacher):
        q = ssel[(ssel.c == c) & (ssel.K == K) & (ssel.student == student) & (ssel.teacher == teacher)]
        assert len(q) == 1, (c, K, student, teacher, len(q))
        return sgrid_index(q.iloc[0].rho, q.iloc[0].a)

    four = {}
    for T in C4:
        for K in KS:
            j = fz1[T]["select"][f"AVT|{K}|joint|one"]
            assert fz1[T]["main"][str(K)] == "one"
            four.setdefault(T, {})[str(K)] = dict(prop_grid=int(f20["choice"][T][str(K)]["grid"]), student_main=sgrid_index(j["rho"], j["a"]),
                                                  student_prop_sel=sel18(T, K, "AVT", "M_F6_mixsel"))
    # CREMA-D：実験 18 が，CREMA-D を触らずに残した単位（T = CREMA-D）だけで選んだ値
    ud = OUT18 / "units"
    cd = {}
    for K in KS:
        v = fsel[f"{CD}|{K}|M_F6_mixsel"]
        i = int(v.replace("M_F6_mix", ""))
        rows, seen, n = [], set(), 0
        for p in sorted(ud.glob(f"{CD}_*.csv")):
            d = pd.read_csv(p, low_memory=False)
            d = d[(d.K == K) & d.version.astype(str).str.fullmatch(r"M_F6_mix\d+")]
            if not len(d):
                continue                                                     # 分類器が 1 つの仮のターゲット（RAVDESS）の単位には提案の版が無い
            z = np.load(p.with_suffix(".npz"), allow_pickle=False)
            seen |= {str(x) for k_ in z.files if k_.startswith("srcs_") for x in z[k_]} | set(d.c.astype(str))
            assert (d["T"] == CD).all()
            rows.append(d[["version", "uar_adj"]]); n += 1
        g = pd.concat(rows).groupby("version").uar_adj.mean()
        vals = [float(g[f"M_F6_mix{q}"]) for q in range(NG)]
        v0 = [vals[q] if GRID[q][1] == 0.0 else -np.inf for q in range(NG)]
        if int(np.argmax(vals)) != i or CD in seen:
            raise SystemExit(f"CREMA-D の選び直しが合わない：K={K} 保存 {i} 選び直し {int(np.argmax(vals))} 現れたコーパス {sorted(seen)}")
        cd[str(K)] = dict(prop_grid=i, noprof_sel=int(np.argmax(v0)), n_units=n, corpora_in_units=sorted(seen),
                          student_main=sel18(CD, K, "AV", "prod"), student_prop_sel=sel18(CD, K, "AV", "M_F6_mixsel"),
                          tether={k: teth18[CD][f"{k}|{K}"] for k in ("A", "V", "AV")})
    srcs = [OUT18 / "summary" / "students_sel.csv", OUT18 / "frozen_sel.json", OUT18 / "tether.json", FROZEN20, OUT1 / "select" / "frozen.json"]
    json.dump(dict(four=four, cremad=cd, sgrid=[[r, a] for r, a in SGRID], source_sha256={str(p.relative_to(ROOT)): dai15.sha(p) for p in srcs if p.exists()},
                   wf_tables2_sha256=hashlib.sha256(tables_of(2).tobytes()).hexdigest(),
                   rule="学び直しの主の設定は，積を疑似ラベルにした学び直しのために選ばれた値（4 コーパスは実験 1，CREMA-D は実験 18 が CREMA-D を含まない単位で選んだ値）．"
                        "提案の設定と引き戻しの強さも，評価するコーパスを含まない単位で選んだ値",
                   time=time.strftime("%Y-%m-%d %H:%M:%S"), commit=RUN["commit"]), open(FROZEN, "w"), ensure_ascii=False, indent=1)
    print(f"freeze：{FROZEN}", flush=True)


def load_frozen() -> dict:
    fz = json.load(open(FROZEN))
    for p, h in fz["source_sha256"].items():
        if (ROOT / p).exists() and dai15.sha(ROOT / p) != h:
            raise SystemExit(f"固定のファイルの元（{p}）が変わっている")
    if fz["sgrid"] != [[r, a] for r, a in SGRID] or fz["wf_tables2_sha256"] != hashlib.sha256(tables_of(2).tobytes()).hexdigest():
        raise SystemExit("固定のファイルの格子か乱数の表が台本と違う")
    return fz


# ------------------------------------------------------------------ 4 コーパスの学び直し

def unit_students4(T: str, f: int, r: int, fz: dict, fz1T: dict) -> tuple[list, dict]:
    sv = dai3.Saved(OUT1, "eval", T, T, f, r, fz1T)
    u = sv.u
    z = np.load(OUT20 / "units" / f"{T}_ALL_f{f}_r{r}.npz", allow_pickle=False)
    yt, yp = u.ytest, u.ypool
    if not ((z["y_test"].astype(int) == yt).all() and (z["sid_test"].astype(str) == sv.sid).all()):
        raise SystemExit(f"{T} f{f} r{r}：実験 20 の単位のテストが実験 1 の保存と違う")
    rows, arr = [], dict(y_test=yt.astype(np.int8), cl_test=sv.cl, sid_test=sv.sid)
    for K in KS:
        kpos = u.kpos[K]
        assert (z[f"kpos_K{K}"] == kpos).all()
        nk = np.setdiff1d(np.arange(u.n), kpos)
        i = int(fz["four"][T][str(K)]["prop_grid"])
        LMp = np.stack([P16.centered(z[f"p_pool_ret_{m}_K{K}"].astype(np.float64)) for m in "AVT"])
        W = P16.to_w(P16.mix_rel(z[f"D_M_K_K{K}"], z["SR_M_ext"], 4 * K, *GRID[i]))
        teach = {"prod": teacher_probs(LMp, None), "prop": teacher_probs(LMp, W)}
        ag = {t_: float((teach[t_][nk].argmax(1) == z[f"pred_pool_K{K}_{v}"].astype(int)).mean()) for t_, v in (("prod", "prod"), ("prop", f"M_F6_mix{i}"))}
        if min(ag.values()) < AGREE_MIN:
            raise SystemExit(f"{T} f{f} r{r} K={K}：疑似ラベルの作り直しが実験 20 の保存と違う（{ag}）")
        cw, cb = sv.teth[("AVT", K)]
        W0, b0 = u.start["AVT"]
        out, gmax, stalled = fit_students(u.Xp["AVT"], u.Xt["AVT"], W0, b0, cw, cb, S.balanced_onehot(yp, kpos, u.n), u.n, teach, f"{T} f{f} r{r} K={K}")
        gm = int(fz["four"][T][str(K)]["student_main"])
        joint_saved = z[f"pred_raw_K{K}_joint_AVT"].astype(int)
        for (t_, gi), (pt, pp) in out.items():
            arr[f"pred_K{K}_{t_}_g{gi}"] = pt.astype(np.int8)
            rows.append(dict(K=K, version=f"stud_{t_}", grid=gi, rho=SGRID[gi][0], a=SGRID[gi][1], uar=100 * S.uar(yt, pt), uar_pool=100 * S.uar(yp[nk], pp[nk]),
                             mimic=float((pp[nk] == teach[t_][nk].argmax(1)).mean())))
        rows.append(dict(K=K, version="_check", grad=gmax, stalled=stalled, agree_prod=ag["prod"], agree_prop=ag["prop"],
                         joint_repro=float((out[("prod", gm)][0] == joint_saved).mean()),      # 積を疑似ラベルにして学び直した判定が，実験 1 の保存の「今の提案」と一致する割合
                         teacher_uar_prod=100 * S.uar(yp[nk], teach["prod"][nk].argmax(1)), teacher_uar_prop=100 * S.uar(yp[nk], teach["prop"][nk].argmax(1))))
    return rows, arr


# ------------------------------------------------------------------ CREMA-D

def unit_cremad(u, teth: dict) -> tuple[list, dict]:
    """CREMA-D の 1 単位．teth：{"種類|K": (C′, C_b)}．土台は実験 18 の関数（fit_none，ret_models，ext_rel）をそのまま呼ぶ．"""
    rows, arr = [], {}
    mods, nm = u.mods, len(u.mods)
    yt, yp = u.ytest, u.ypool
    arr["y_test"], arr["cl_test"], arr["sid_test"], arr["y_pool"], arr["cl_pool"] = yt.astype(np.int8), u.cl, u.sid, yp.astype(np.int8), u.cl_pool
    for kind in u.kinds:
        arr[f"srcs_{kind}"] = np.array(u.srcs[kind])
    ext = P18.ext_rel(u.srcs, mods)
    Dext, Dhome = np.stack([ext[m][0] for m in mods]), np.stack([ext[m][2] for m in mods])
    arr["D_M_ext"], arr["D_M_home"], arr["ret_M"] = Dext, Dhome, np.array([ext[m][1] for m in mods])
    pairs = list(itertools.combinations(range(NF), 2))
    tables = tables_of(nm)
    for K in KS:
        kpos = u.kpos[K]
        yk = yp[kpos]
        nk = np.setdiff1d(np.arange(u.n), kpos)
        fold = dai15.kfold_of(yk)
        assert all(np.bincount(fold[yk == i], minlength=NF).min() >= 1 for i in range(4)), "組に感情が欠ける"
        sets = [kpos] + [kpos[fold != j] for j in range(NF)] + [kpos[(fold != i) & (fold != j)] for i, j in pairs]
        Rp, Rt = P18.ret_models(u, K, teth, sets)
        Pp, Pt = [Rp[m][0] for m in mods], [Rt[m][0] for m in mods]
        Ph = [np.stack([Rp[m][1 + fold[a]][kpos[a]] for a in range(len(kpos))]) for m in mods]
        P2 = [np.stack([Rp[m][1 + NF + n][kpos] for n in range(len(pairs))]) for m in mods]
        PJ = [Rp[m][1:1 + NF] for m in mods]
        for k, m in enumerate(mods):
            arr[f"p_pool_ret_{m}_K{K}"], arr[f"p_test_ret_{m}_K{K}"] = Pp[k].astype(np.float32), Pt[k].astype(np.float32)
            arr[f"pcf_{m}_K{K}"], arr[f"pcf2_{m}_K{K}"], arr[f"ppj_{m}_K{K}"] = Ph[k].astype(np.float32), P2[k].astype(np.float32), PJ[k].astype(np.float32)
            arr[f"pred_raw_K{K}_ret_{m}"] = Pt[k].argmax(1).astype(np.int8)
            rows.append(dict(K=K, version=f"ret_{m}", uar_raw=100 * S.uar(yt, Pt[k].argmax(1))))
        for kind in u.kinds:                                                    # K 本だけの分類器
            rn = P18.fit_none(u, kind, K, teth)
            pt = S.logits(u.Xt[kind], rn["W"][0], rn["b"][0]).argmax(1).cpu().numpy()
            arr[f"pred_raw_K{K}_none_{kind}"] = pt.astype(np.int8)
            rows.append(dict(K=K, version=f"none_{kind}", uar_raw=100 * S.uar(yt, pt)))
        DK = np.stack([P16.dprime(Ph[k], yk)[0] for k in range(nm)])
        DO = np.stack([P16.dprime(Pp[k][nk], yp[nk])[0] for k in range(nm)])
        arr[f"D_M_K_K{K}"], arr[f"D_M_orc_K{K}"], arr[f"kpos_K{K}"], arr[f"kfold_K{K}"] = DK, DO, kpos.astype(np.int32), fold.astype(np.int8)
        combine_all(arr, rows, K, Pt, Pp, Ph, P2, PJ, kpos, fold, yk, yt, yp, DK, DO, Dext, Dhome, tables)
        # 学び直し（融合の分類器）：疑似ラベルは積と，提案の格子の全部の点（選ぶ点は集計の段で固定のファイルから引く）
        LMp = np.stack([P16.centered(p) for p in Pp])
        teach = {"prod": teacher_probs(LMp, None)}
        for i in sorted({q for q in range(NG) if GRID[q][1] == 1.0 and GRID[q][0] in (0.0, 20.0)}):   # 提案の点の候補（n0 = 0・20，相性を使う）
            teach[f"mix{i}"] = teacher_probs(LMp, P16.to_w(P16.mix_rel(DK, Dext, 4 * K, *GRID[i])))
        cw, cb = teth[f"{u.fus}|{K}"]
        W0, b0 = u.start[u.fus]
        out, gmax, stalled = fit_students(u.Xp[u.fus], u.Xt[u.fus], W0, b0, cw, cb, u.gK(K), u.n, teach, f"{u.name()} K={K}")
        for (t_, gi), (pt, pp) in out.items():
            arr[f"pred_K{K}_stud_{t_}_g{gi}"] = pt.astype(np.int8)
            rows.append(dict(K=K, version=f"stud_{t_}", grid=gi, rho=SGRID[gi][0], a=SGRID[gi][1], uar=100 * S.uar(yt, pt), uar_pool=100 * S.uar(yp[nk], pp[nk]),
                             mimic=float((pp[nk] == teach[t_][nk].argmax(1)).mean())))
        rows.append(dict(K=K, version="_check", grad=gmax, stalled=stalled, gpu_max_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0)))
    return rows, arr


def cremad_teth(fz: dict) -> dict:
    return {f"{k}|{K}": tuple(v) for K in KS for k, v in fz["cremad"][str(K)]["tether"].items()}


# ------------------------------------------------------------------ 段の回し方

def loop(out: Path, sub: str, todo: list, fn) -> None:
    d = out / sub
    d.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    for name, args in todo:
        if (d / f"{name}.csv").exists():
            if (pd.read_csv(d / f"{name}.csv", low_memory=False).commit == RUN["commit"]).all():
                print(f"{sub} {name}：済み（飛ばす）", flush=True)
                continue
            raise SystemExit(f"{sub} {name} が別のコミットで既にある")
        t0 = time.time()
        rows, arr = fn(*args)
        np.savez_compressed(d / f"{name}.npz", **arr)                       # npz を先に書き，csv を済みの印にする
        df = pd.DataFrame(rows).assign(name=name, commit=RUN["commit"], dirty=RUN["dirty"], sec=time.time() - t0, host=platform.node(), gpu=gpu)
        df.to_csv(d / f"{name}.csv", index=False)
        ck = df[df.version == "_check"]
        print(f"{sub} {name}：{time.time() - t0:.0f} 秒．行 {len(df)}．勾配の最大 {ck.grad.max():.1e}" +
              (f"．今の提案の再現 {ck.joint_repro.min():.4f}．疑似ラベルの一致 {min(ck.agree_prod.min(), ck.agree_prop.min()):.4f}" if "joint_repro" in ck else ""), flush=True)
        del rows, arr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def todo_students4(targets, folds, seeds) -> list:
    fz1, _ = dai3.load_fz1(OUT1)
    return [(f"{T}_ALL_f{f}_r{r}", (T, f, r)) for T in (targets or C4) for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"]))
            if (folds is None or f in folds) and (seeds is None or r in seeds)]


def todo_cremad(folds, seeds) -> list:
    return [(f"{CD}_ALL_f{f}_r{r}", (f, r)) for f in range(DD.N_FOLDS) for r in range(10) if (folds is None or f in folds) and (seeds is None or r in seeds)]


# ------------------------------------------------------------------ 集計

def role_of(v: str) -> str:
    b = v.split("@")[0]
    return ("proposal" if b in ("prop", "stud_prop") else "A" if b in A2 else "ablation" if b in P20.ABL else "oracle" if b in ("M_F6_orc",) else
            "reference" if b in ("M_F6_ext",) else "single" if b.startswith(("ret_", "none_A", "none_V", "none_T")) and b not in ("none_AVT", "none_AV") else
            "selection" if b.endswith("_sel") else "B")


def table_block(cells: dict, con: list, geff: dict, part: str) -> tuple[list, list]:
    """cells：{(コーパス, K, 版): {(分割, 種): 塊ごとの数}}．水準と差を，実験 20 の集計と同じ区間の作り方で出す．"""
    def eff(A, B, c, k_):
        if set(A) != set(B):
            raise SystemExit(f"差の単位がそろわない：{k_}")
        es = ST.effect_stats(A, B, geff[c], seed_rng(*k_))
        return dict(est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], df=es.get("df", np.nan), method=es["method"], n_fr=es["n_units"], n_folds=es["n_folds"],
                    folds_dropped=es.get("folds_dropped", ""), v_cluster=es.get("v_jk", np.nan), v_seed=es.get("v_seed", np.nan), v_seed_all=es.get("v_seed_all", np.nan),
                    mark={"上回る": "↑", "下回る": "↓"}.get(P16.verdict(es["lo"], es["hi"], es["effect"]), ""))

    def zero(A):
        return {k_: (x[0], np.zeros_like(x[1]), x[2]) for k_, x in A.items()}

    def sub(A, seeds):
        return A if seeds is None else {k_: x for k_, x in A.items() if k_[1] in seeds}

    same_as = {}
    for c, K in sorted({(c, K) for c, K, _ in cells}):
        vs = [v for (c2, K2, v) in cells if c2 == c and K2 == K]
        first, size, sig = {}, {}, {}
        for v in vs:
            dd = cells[(c, K, v)]
            sig[v] = hashlib.sha256(b"".join(repr(k_).encode() + np.ascontiguousarray(dd[k_][1]).tobytes() for k_ in sorted(dd))).hexdigest()
            first.setdefault(sig[v], v)
            size[sig[v]] = size.get(sig[v], 0) + 1
        for v in vs:
            same_as[(c, K, v)] = first[sig[v]] if size[sig[v]] > 1 else ""
    lv, ef = [], []
    for (c, K, v), dd in sorted(cells.items()):
        e_ = eff(dd, zero(dd), c, ("L", part, c, K, v))
        lv.append(dict(part=part, c=c, K=K, version=v, role=role_of(v), uar=e_["est"], lo=e_["lo"], hi=e_["hi"], se=e_["se"], df=e_["df"], method=e_["method"],
                       n_fr=e_["n_fr"], same_as=same_as[(c, K, v)]))
    for a, b in con:
        for c, K in sorted({(c, K) for c, K, _ in cells}):
            A, B = cells.get((c, K, a)), cells.get((c, K, b))
            if A is None or B is None:
                raise SystemExit(f"差の相手が無い：{(part, a, b, c, K)}")
            for sn, ss in SEED_SETS.items():
                As, Bs = sub(A, ss), sub(B, ss)
                if not As or (sn != "all" and len(As) == len(A)):
                    continue
                ef.append(dict(part=part, a=a, b=b, c=c, K=K, seeds=sn, role_a=role_of(a), role_b=role_of(b), a_same_as=same_as[(c, K, a)], b_same_as=same_as[(c, K, b)],
                               **eff(As, Bs, c, ("A", part, a, b, c, K, sn))))
    return lv, ef


def stage_summary(out: Path, allow_partial: bool) -> None:
    sdir = out / "summary"
    sdir.mkdir(exist_ok=True)
    fz = load_frozen()
    f20 = P20.load_frozen()
    lv, ef, grid_rows, chk = [], [], [], {}
    # ---- 4 コーパス
    files = sorted((out / "students4").glob("*.csv"))
    if len(files) != len(todo_students4(None, None, None)) and not allow_partial:
        raise SystemExit(f"students4 の出力が {len(files)} 個")
    if files:
        cs = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
        if cs.commit.nunique() != 1 and not allow_partial:
            raise SystemExit("students4 のコミットが 1 つでない")
        cells = {}
        for p in files:
            T, f, r = p.stem.split("_ALL_")[0], int(p.stem.split("_f")[1].split("_r")[0]), int(p.stem.rsplit("_r", 1)[1])
            z, z20 = np.load(p.with_suffix(".npz"), allow_pickle=False), np.load(OUT20 / "units" / f"{p.stem}.npz", allow_pickle=False)
            yt = z["y_test"].astype(int)
            assert (z20["y_test"].astype(int) == yt).all() and (z20["cl_test"] == z["cl_test"]).all()
            for K in KS:
                q = fz["four"][T][str(K)]
                i, gm, gs = int(f20[T][K][0]), int(q["student_main"]), int(q["student_prop_sel"])
                names = {"stud_prop": (z, f"pred_K{K}_prop_g{gm}"), "stud_prod": (z, f"pred_K{K}_prod_g{gm}"), "stud_prop_sel": (z, f"pred_K{K}_prop_g{gs}"),
                         "prop": (z20, f"pred_adj_K{K}_M_F6_mix{i}"), "prod": (z20, f"pred_adj_K{K}_prod"),
                         **{v: (z20, f"pred_raw_K{K}_{v}") for v in ("none_AVT", "joint_AVT") + B_VERS}}
                for v, (src, kk) in names.items():
                    P12._add(cells.setdefault((T, K, v), {}), (f, r), ST.counts(yt, src[kk].astype(int), z["cl_test"]), f"{p.stem} {v}")
        P12._finish(cells)
        con = ([("stud_prop", b) for b in ("joint_AVT", "stud_prod", "none_AVT", "prop", "prod") + B_VERS]
               + [("stud_prop_sel", "joint_AVT"), ("stud_prod", "joint_AVT"), ("joint_AVT", "none_AVT"), ("prop", "joint_AVT")] + [(b, "joint_AVT") for b in B_VERS])
        a, b = table_block(cells, con, {c: ST.g_eff_of(c)[0] for c in C4}, "4corpora")
        lv += a; ef += b
        st = cs[cs.version.astype(str).str.startswith("stud_")].assign(T=lambda d: d.name.str.split("_ALL_").str[0])
        grid_rows.append(st.groupby(["T", "K", "version", "grid", "rho", "a"])[["uar", "mimic"]].mean().reset_index().rename(columns={"T": "c"}).assign(part="4corpora"))
        ck = cs[cs.version == "_check"]
        chk.update(students4=dict(n_units=len(files), grad_max=float(ck.grad.max()), stalled=int(ck.stalled.sum()), joint_repro_min=float(ck.joint_repro.min()),
                                  joint_repro_mean=float(ck.joint_repro.mean()), agree_min=float(min(ck.agree_prod.min(), ck.agree_prop.min())),
                                  commits=sorted(cs.commit.astype(str).unique().tolist())))
    # ---- CREMA-D
    files = sorted((out / "cremad").glob("*.csv"))
    if len(files) != len(todo_cremad(None, None)) and not allow_partial:
        raise SystemExit(f"cremad の出力が {len(files)} 個")
    if files:
        cs = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
        if cs.commit.nunique() != 1 and not allow_partial:
            raise SystemExit("cremad のコミットが 1 つでない")
        cells = {}
        fk = P18.fus(CD)
        for p in files:
            f, r = int(p.stem.split("_f")[1].split("_r")[0]), int(p.stem.rsplit("_r", 1)[1])
            z = np.load(p.with_suffix(".npz"), allow_pickle=False)
            yt = z["y_test"].astype(int)
            for K in KS:
                q = fz["cremad"][str(K)]
                i, gm, gs = int(q["prop_grid"]), int(q["student_main"]), int(q["student_prop_sel"])
                names = {v: f"pred_adj_K{K}_{v}" for v in A2 + ("M_F6_orc", "M_F6_ext")}
                names.update({f"{v}@raw": f"pred_raw_K{K}_{v}" for v in A2})
                names.update(prop=f"pred_adj_K{K}_M_F6_mix{i}", **{"prop@raw": f"pred_raw_K{K}_M_F6_mix{i}"}, prop_noprof=f"pred_adj_K{K}_M_F6_mix{P20.same_but(i, beta=0.0)}",
                             prop_noprof_sel=f"pred_adj_K{K}_M_F6_mix{int(q['noprof_sel'])}", prop_noint=f"pred_adj_K{K}_M_F6n_mix{i}",
                             prop_n0zero=f"pred_adj_K{K}_M_F6_mix{P20.same_but(i, n0=0.0)}", prop_home=f"pred_adj_K{K}_M_F6h_mix{i}",
                             prop_levelfree=f"pred_adj_K{K}_M_F6_mix{GRID.index((float('inf'), 1.0, 'eq'))}",
                             stud_prod=f"pred_K{K}_stud_prod_g{gm}", stud_prop=f"pred_K{K}_stud_mix{i}_g{gm}", stud_prop_sel=f"pred_K{K}_stud_mix{i}_g{gs}",
                             **{f"none_{k}": f"pred_raw_K{K}_none_{k}" for k in ("A", "V", fk)}, ret_A=f"pred_raw_K{K}_ret_A", ret_V=f"pred_raw_K{K}_ret_V")
                for v, kk in names.items():
                    P12._add(cells.setdefault((CD, K, v), {}), (f, r), ST.counts(yt, z[kk].astype(int), z["cl_test"]), f"{p.stem} {v}")
        P12._finish(cells)
        nf = f"none_{fk}"
        con = ([("prop", b) for b in A2 + P20.ABL + (nf, "stud_prod", "M_F6_ext")] + [(a, "prod") for a in A2[1:] + ("prop_noprof", "prop_levelfree", "M_F6_orc", "M_F6_ext")]
               + [("prop_noint", "prop_noprof"), ("prop", "prop@raw")] + [("prop@raw", f"{b}@raw") for b in A2] + [("prop", f"{b}@raw") for b in A2[1:]]
               + [("stud_prop", b) for b in ("stud_prod", nf, "prop", "prod")] + [("stud_prop_sel", "stud_prod"), ("stud_prod", nf), ("stud_prod", "prod")])
        a, b = table_block(cells, con, {CD: P18.g_eff_of(CD)}, "cremad")
        lv += a; ef += b
        st = cs[cs.version.astype(str).str.startswith("stud_")]
        grid_rows.append(st.groupby(["K", "version", "grid", "rho", "a"])[["uar", "mimic"]].mean().reset_index().assign(c=CD, part="cremad"))
        ck = cs[cs.version == "_check"]
        chk.update(cremad=dict(n_units=len(files), grad_max=float(ck.grad.max()), stalled=int(ck.stalled.sum()), adj_resid_max=float(cs.adj_resid.max()),
                               wf_n_capped=int(cs[cs.version == f"wf{N_WF}"].wf_n_capped.fillna(0).sum()), commits=sorted(cs.commit.astype(str).unique().tolist())))
    pd.DataFrame(lv).to_csv(sdir / "levels.csv", index=False)
    pd.DataFrame(ef).to_csv(sdir / "effects.csv", index=False)
    if grid_rows:
        pd.concat(grid_rows, ignore_index=True).to_csv(sdir / "student_grid.csv", index=False)
    json.dump(dict(chk, summary_commit=RUN["commit"]), open(sdir / "convergence.json", "w"), indent=1)
    print(f"summary：水準の行 {len(lv)}，差の行 {len(ef)}．{json.dumps(chk, ensure_ascii=False)[:400]}", flush=True)


# ------------------------------------------------------------------ 検算

def stage_check3() -> None:
    """combine_all を実験 16 の下調べの保存（3 つの分類器）に当て，保存の判定と一致するかを見る（確率は float32 の保存から読むので，同点に近い発話は食い違いうる）．"""
    out = []
    for name in ("AFEW_IEMOCAP_f0_r0", "MELD_AFEW_f1_r1"):
        z = np.load(ROOT / "results" / "dai16_probe" / "units" / f"{name}.npz", allow_pickle=False)
        yt, yp = z["y_test"].astype(int), z["y_pool"].astype(int)
        for K in (5, 25):
            kpos, fold = z[f"kpos_K{K}"], z[f"kfold_K{K}"].astype(int)
            g = lambda pre: [z[f"{pre}_{m}_K{K}"].astype(np.float64) for m in "AVT"]  # noqa: E731
            arr, rows = {}, []
            combine_all(arr, rows, K, g("p_test_ret"), g("p_pool_ret"), g("pcf"), g("pcf2"), g("ppj"), kpos, fold, yp[kpos], yt, yp,
                        z[f"D_M_K_K{K}"], z[f"D_M_orc_K{K}"], z["SR_M_ext"], z["SR_M_home"], tables_of(3))
            vs = ["prod", "M_F6_K", "M_F6_ext", "M_F6_orc", "M_F2_K", "cf_D_l05"] + [f"M_F6_mix{i}" for i in range(NG)] + [f"M_F6n_mix{i}" for i in range(NG)]
            ag = {v: min(float((arr[f"pred_{w}_K{K}_{v}"] == z[f"pred_{w}_K{K}_{v}"]).mean()) for w in ("raw", "adj", "pool")) for v in vs}
            out.append(dict(unit=name, K=K, n_versions=len(vs), agree_min=min(ag.values()), worst=min(ag, key=ag.get), n_exact=sum(v == 1.0 for v in ag.values())))
    d = pd.DataFrame(out)
    print(d.to_string(index=False))
    if d.agree_min.min() < AGREE_MIN:
        raise SystemExit("check3：保存の判定と合わない")
    print("check3：通った")


def stage_check18(out: Path) -> None:
    """CREMA-D の単位の計算を，実験 18 の単位（RAVDESS を除いた回．ソースは 4 コーパス）と同じ引き戻しの強さで流し，保存の判定と照らす．"""
    P18.OUTDIR["out"] = OUT18
    teth = json.load(open(OUT18 / "tether.json"))["tether"][HOLD]
    u = P18.Unit6(HOLD, CD, 0, 0)
    assert all(sorted(u.srcs[k]) == sorted(C4) for k in u.kinds), u.srcs
    t0 = time.time()
    rows, arr = unit_cremad(u, teth)
    z = np.load(OUT18 / "units" / f"{HOLD}_{CD}_f0_r0.npz", allow_pickle=False)
    res = []
    for K in KS:
        vs = ["prod", "M_F6_K", "M_F6_ext", "M_F6_orc"] + [f"M_F6_mix{i}" for i in range(NG)] + [f"M_F6n_mix{i}" for i in range(NG)]
        ag = {v: min(float((arr[f"pred_{w}_K{K}_{v}"] == z[f"pred_{w}_K{K}_{v}"]).mean()) for w in ("raw", "adj", "pool")) for v in vs}
        ag.update({f"none_{k}": float((arr[f"pred_raw_K{K}_none_{k}"] == z[f"pred_raw_K{K}_none_{k}"]).mean()) for k in u.kinds})
        pdev = max(float(np.abs(arr[f"p_test_ret_{m}_K{K}"] - z[f"p_test_ret_{m}_K{K}"]).max()) for m in u.mods)
        res.append(dict(K=K, n_versions=len(ag), agree_min=min(ag.values()), worst=min(ag, key=ag.get), n_exact=sum(v == 1.0 for v in ag.values()), prob_dev_max=pdev))
    d = pd.DataFrame(res)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "check18.csv", index=False)
    print(d.to_string(index=False))
    print(f"check18：{time.time() - t0:.0f} 秒．行 {len(rows)}．配列 {len(arr)}")
    if d.agree_min.min() < AGREE_MIN:
        raise SystemExit("check18：実験 18 の保存と合わない")
    print("check18：通った")


def selftest() -> None:
    assert len(SGRID) == 12 and SGRID[sgrid_index(1.0, 1)] == (1.0, 1) and GRID[14] == (float("inf"), 1.0, "eq")
    assert tables_of(3) is P20.WS
    W2 = tables_of(2)
    assert W2.shape == (N_WF + 1, 2, 4) and np.allclose(W2.sum(1), 2.0) and np.allclose(W2[0], 1.0) and W2.min() >= 0 and np.allclose(W2, tables_of(2))
    rng = np.random.default_rng(0)
    for nm in (2, 3):                                                         # 人工の単位に combine_all を通す
        n, nt, K = 400, 300, 5
        yp, yt = rng.integers(0, 4, n), rng.integers(0, 4, nt)
        kpos = np.sort(np.concatenate([np.flatnonzero(yp == c)[:K] for c in range(4)]))
        yk = yp[kpos]
        fold = dai15.kfold_of(yk)

        def mk(y, m):
            P = rng.dirichlet(np.ones(4), size=(m, len(y)))
            P[:, np.arange(len(y)), y] += 0.5
            return list(P / P.sum(-1, keepdims=True))

        Pt, Pp = mk(yt, nm), mk(yp, nm)
        Ph = [p[kpos] for p in mk(yp, nm)]
        P2 = [np.stack([p[kpos] for p in mk(yp, 10)]) for _ in range(nm)]
        PJ = [np.stack(mk(yp, NF)) for _ in range(nm)]
        DK = np.stack([P16.dprime(Ph[k], yk)[0] for k in range(nm)])
        DO = np.stack([P16.dprime(Pp[k], yp)[0] for k in range(nm)])
        De = rng.uniform(0.3, 1.5, size=(nm, 4))
        arr, rows = {}, []
        combine_all(arr, rows, K, Pt, Pp, Ph, P2, PJ, kpos, fold, yk, yt, yp, DK, DO, De, De * 1.3, tables_of(nm))
        d = pd.DataFrame(rows)
        need = set(A2) | {f"M_F6_mix{i}" for i in range(NG)} | {f"M_F6n_mix{i}" for i in range(NG)} | {"M_F6_orc", "M_F6_ext"} | {f"M_F6h_mix{i}" for i in range(NG) if GRID[i][1] == 1.0}
        assert need <= set(d.version) and not d.duplicated(["K", "version"]).any(), need - set(d.version)
        assert (arr[f"pred_raw_K{K}_prod"] == np.sum([np.log(p) for p in Pt], 0).argmax(1)).all()
        assert (arr[f"pred_adj_K{K}_prod"] == arr[f"pred_raw_K{K}_prod"]).all()            # 積は切片をずらしても同じ判定
        i0 = GRID.index((float("inf"), 0.0, "eq"))                                      # 水準を全部同じにして相性を使わない点は積と同じ
        assert (arr[f"pred_adj_K{K}_M_F6_mix{i0}"] == arr[f"pred_adj_K{K}_prod"]).all()
        assert (arr[f"pred_adj_K{K}_M_F6_mix0"] == arr[f"pred_adj_K{K}_M_F2_K"]).all()     # n0 = 0 で相性を使わない点は水準だけの重みと同じ
        Q0 = teacher_probs(np.stack([P16.centered(p) for p in Pp]), None)
        Q1 = teacher_probs(np.stack([P16.centered(p) for p in Pp]), np.ones((nm, 4)))
        assert np.allclose(Q0.sum(1), 1.0) and np.abs(Q0 - Q1).max() < 1e-9             # 重みが全部 1 の提案の疑似ラベルは積と同じ
    assert role_of("stud_prop") == "proposal" and role_of("wf10@raw") == "A" and role_of("b_kd") == "B" and role_of("none_AV") == "B" and role_of("ret_A") == "single"
    print("selftest：通った")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "check3", "check18", "freeze", "students4", "cremad", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--frozen", default=None, help="検査用：固定のファイルの置き場を替える（本番の置き場では使えない）")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    if args.stage == "selftest":
        selftest()
        return
    dai1.ENC["enc"] = DD.ENC1
    if args.stage == "check3":
        stage_check3()
        return
    out = Path(args.out_dir).resolve()
    prod = out == OUT_PROD.resolve()
    if args.frozen:
        if prod:
            sys.exit("本番の置き場では固定のファイルの置き場を替えない")
        global FROZEN
        FROZEN = Path(args.frozen).resolve()
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    if args.stage == "check18":
        if prod:
            sys.exit("check18 は本番の置き場に書かない")
        stage_check18(out)
        return
    if args.stage == "freeze":
        stage_freeze()
        return
    if dirty and (prod or not args.allow_dirty):
        sys.exit("DAI/exp・measure・features に未コミットの変更がある")
    if prod and not (dai1.committed_clean(DESIGN) and dai1.committed_clean(FROZEN) and dai1.committed_clean(Path(__file__).resolve())):
        sys.exit("設計の文書・固定のファイル・この台本のどれかがコミットされていない")
    if prod and (args.folds is not None or args.seeds is not None or args.targets is not None or args.allow_partial):
        sys.exit("本番の置き場では単位を絞らない")
    if prod and not (ROOT / "logs" / "dai20_eval.done").exists():
        sys.exit("実験 20 の計算が終わっていない")
    P18.OUTDIR["out"] = OUT18
    out.mkdir(parents=True, exist_ok=True)
    fz = load_frozen()
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""), torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, python=platform.python_version(), this_script_sha256=dai15.sha(Path(__file__).resolve()),
                scripts={n: dai15.sha(ROOT / "DAI" / "exp" / n) for n in SCRIPTS}, design_sha256=(dai15.sha(DESIGN) if DESIGN.exists() else ""),
                frozen_sha256=dai15.sha(FROZEN), frozen20_sha256=dai15.sha(FROZEN20), dai20_provenance_sha256=(dai15.sha(OUT20 / "provenance.json") if (OUT20 / "provenance.json").exists() else ""),
                KS=KS, SGRID=[list(x) for x in SGRID], rng_seed=SEED, wf_tables2_sha256=hashlib.sha256(tables_of(2).tobytes()).hexdigest())
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "students4":
        fz1, _ = dai3.load_fz1(OUT1)
        loop(out, "students4", todo_students4(args.targets, args.folds, args.seeds), lambda T, f, r: unit_students4(T, f, r, fz, fz1[T]))
    elif args.stage == "cremad":
        teth = cremad_teth(fz)
        loop(out, "cremad", todo_cremad(args.folds, args.seeds), lambda f, r: unit_cremad(P18.Unit6(HOLD, CD, f, r), teth))
    else:
        stage_summary(out, args.allow_partial)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
