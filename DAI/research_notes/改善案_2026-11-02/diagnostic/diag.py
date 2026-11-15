"""提案手法（oursx）を改善する余地を，仮のターゲットの値だけで測る診断（2026-11-02．CPU だけ）。

読むもの（これ以外の results/ の下のファイルを開こうとしたら止まる．sys.addaudithook で見張る）：
  results/e77_<世界>/<T>_pseudo.csv，results/e78_<世界>/<T>_pseudo.csv，results/e80c_<世界>/<T>_pseudo.csv，results/e80/kept.csv
世界は fuse-audio-video-text（AVT）と fuse-audio-video（AV）．K は 0・5・25．

単位：
  評価のターゲット T（IEMOCAP 台本は仮のターゲットの行が即興の写しなので外し，即興で代表させる．同じことを検算する）
  × 仮のターゲットのコーパス c（IEMOCAP の即興と台本は 1 つのコーパス．値は 2 群の平均）× 生徒 × K．
  候補の値は「c の反復 0〜4（IEMOCAP は 2 群 × 5）の UAR の平均」．

選び方（exp/analyze_e79.py の select の段をまねる）：
  選ぶ点数 ＝ 選択に使う仮のターゲットの全行（群 × 反復）の UAR の単純平均．最大を選び，同点は 'method|pnorm|meff' の文字列の順で先頭．
  none を含む版（incl）は，最良の候補の点数が none の点数より大きいときだけ候補を選ぶ（同点は none）．none を除く版（excl）は必ず候補を選ぶ．
  仮のターゲットの全単位に値の無い候補は外す（E79 と同じ）．
  leave-one-pseudo-out（LOPO）：c を 1 つ抜き，残りの仮のターゲットの行で選び，c で測る．

族：
  oursx    ：dist で教師が kept.csv の（世界, T, 生徒, K）の残した教師，重みの付け方 5 × 総重み 4（E79 の oursx）
  oursx77  ：dist で教師が E77 の教師（単モダリティ pF・pFK・ref，融合 poe0・ref・ref0）．教師をふるわない版（kept.csv は c を含む全部の
             仮のターゲットで決めたので，oursx の LOPO には c の情報が教師の集合を通して入る．oursx77 にはそれが無い）
  kd       ：kd と kdh で教師が kept の教師（E79 の kd）
  kd77     ：kd と kdh で教師が E77 の KD の教師（単モダリティ pF・pFK・ref，融合 poe0・ref）
  groupB   ：entmin，self，im（総重みが 100・500・2000・n），t3a，lp，coral，dann，mme（E79 の groupB．種 0）
  teacher  ：教師そのもの（行の生徒は世界の名前．教師の配列は生徒に依らない）

    CUDA_VISIBLE_DEVICES= ./.venv/bin/python "research_notes/改善案_2026-11-02/diagnostic/diag.py"
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import itertools  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = ROOT / "results"
OUT = HERE / "out"

# ---------------------------------------------------------------- 開いてよいファイルの見張り

ALLOWED = [re.compile(p) for p in (
    r"/results/e7[78]_fuse-audio-video(-text)?/[^/]+_pseudo\.csv$",
    r"/results/e80c_fuse-audio-video(-text)?/[^/]+_pseudo\.csv$",
    r"/results/e80/kept\.csv$",
)]
OPENED: list = []


def _hook(event, args):
    if event != "open" or not args:
        return
    a = args[0]
    if isinstance(a, int):
        return
    try:
        p = os.path.abspath(os.fsdecode(a))
    except TypeError:
        return
    if p.startswith(str(RES) + os.sep):
        if not any(r.search(p) for r in ALLOWED):
            raise PermissionError(f"開いてはいけないファイル：{p}")
        OPENED.append(p)


sys.addaudithook(_hook)


def guard_selftest():
    for bad in (RES / "e77_fuse-audio-video" / "AFEW.csv", RES / "e78_fuse-audio-video-text" / "MELD.csv",
                RES / "e77_fuse-audio-video" / "AFEW_r0_logp.npy", RES / "e80c_fuse-audio-video" / "AFEW.csv"):
        try:
            open(bad, "rb").close()
        except PermissionError:
            continue
        raise SystemExit(f"見張りが働かない：{bad} を開けた")


# ---------------------------------------------------------------- 定数

WORLDS = {"fuse-audio-video-text": ["audio_wavlm", "video", "text_roberta"], "fuse-audio-video": ["audio_wavlm", "video"]}
TARGETS = {"fuse-audio-video-text": ["IEMOCAP-impro", "MOSEI", "MELD"],
           "fuse-audio-video": ["AFEW", "CREMA-D", "IEMOCAP-impro", "MELD", "MOSEI"]}
SCRIPT_CHECK = {"fuse-audio-video-text": "IEMOCAP-script", "fuse-audio-video": "IEMOCAP-script"}
KS = ("0", "5", "25")
MEFFS = ("100", "500", "2000", "n")
WGTS = ("bal", "raw", "hard0.5", "hard0.7", "hard0.9")
SEEDED = {"dann", "mme", "mapctl", "xmuda", "entmin", "im"}
GROUP_B = ("entmin", "self", "im", "t3a", "lp", "coral", "dann", "mme")
E77_DIST = {"uni": ("pF", "pFK", "ref"), "fused": ("poe0", "ref", "ref0")}
E77_KD = {"uni": ("pF", "pFK", "ref"), "fused": ("poe0", "ref")}
K_INDEP_TEACHERS = re.compile(r"^(pF|poe0|agree|ds|mlp|poolC|other_.+)$")   # 配列が K に依らない教師（作り方から）
FAMS = ("oursx", "oursx77", "kd", "kd77", "groupB")


def corp(g):
    return "IEMOCAP" if g.startswith("IEMOCAP") else g


def strip_seed(p):
    return re.sub(r"s\d+$", "", p)


def role_of(world, st):
    return "fused" if st == world else "uni"


# ---------------------------------------------------------------- 読み込み

def read(f):
    return pd.read_csv(f, dtype={"K": str, "param": str, "meff": str, "pseudo": str, "student": str, "method": str})


CHECKS: list = []


def load(world, T):
    d77 = read(RES / f"e77_{world}" / f"{T}_pseudo.csv").assign(wave="e77")
    d78 = read(RES / f"e78_{world}" / f"{T}_pseudo.csv").assign(wave="e78")
    d8c = read(RES / f"e80c_{world}" / f"{T}_pseudo.csv").assign(wave="e80c")
    kk = ["pseudo", "rep", "student", "K"]
    n77 = d77[d77.method == "none"].set_index(kk).uar
    for name, dx in (("e78", d78), ("e80c", d8c)):
        nx = dx[dx.method == "none"].set_index(kk).uar
        j = pd.concat([n77.rename("a"), nx.rename("b")], axis=1, join="inner")
        CHECKS.append({"check": f"none_e77_vs_{name}", "world": world, "T": T, "n": len(j), "n_other": len(nx),
                       "absdiff_max": float((j.a - j.b).abs().max()) if len(j) else np.nan})
        if len(j) < len(nx) or not (j.a - j.b).abs().max() <= 1e-9:
            raise SystemExit(f"{world} {T}：E77 と {name} の none が合わない")
    d78 = d78[d78.method.isin(["coral", "dann", "mme"])]
    d8c = d8c[d8c.method.isin(["dist", "kd", "kdh", "teacher"])]
    P = pd.concat([d77, d78, d8c], ignore_index=True)
    key = ["pseudo", "rep", "student", "method", "param", "meff", "K"]
    dup = P.duplicated(key, keep=False)
    if dup.any():
        raise SystemExit(f"{world} {T}：同じ鍵の行が重なる（{int(dup.sum())} 行）")
    seed = P.param.str.extract(r"s(\d+)$")[0]
    bad = P.method.isin(list(SEEDED)) & seed.notna() & (seed != "0")
    if bad.any():
        raise SystemExit(f"{world} {T}：仮のターゲットに種 1・2 の行がある")
    P["pnorm"] = [strip_seed(p) if m in SEEDED else p for m, p in zip(P.method, P.param)]
    P["key"] = P.method + "|" + P.pnorm + "|" + P.meff
    P["corpus"] = P.pseudo.map(corp)
    return P


def check_script_copy(world):
    """IEMOCAP 台本の仮のターゲットの行が即興の行と同じ値か（同じなら台本を外してよい）。"""
    a = pd.concat([read(RES / f"{w}_{world}" / "IEMOCAP-impro_pseudo.csv").assign(wave=w) for w in ("e77", "e78", "e80c")])
    b = pd.concat([read(RES / f"{w}_{world}" / "IEMOCAP-script_pseudo.csv").assign(wave=w) for w in ("e77", "e78", "e80c")])
    kk = ["wave", "pseudo", "rep", "student", "method", "param", "meff", "K"]
    j = a.set_index(kk).uar.to_frame("a").join(b.set_index(kk).uar.to_frame("b"), how="outer")
    CHECKS.append({"check": "script_equals_impro", "world": world, "n_impro": len(a), "n_script": len(b),
                   "n_unmatched": int(j.isna().any(axis=1).sum()), "absdiff_max": float((j.a - j.b).abs().max())})


def load_kept():
    k = pd.read_csv(RES / "e80" / "kept.csv", dtype={"K": str, "world": str, "target": str, "student": str, "teacher": str})
    k["keep"] = k.keep.map(lambda v: v is True or str(v).strip().lower() in ("true", "1"))
    out = {}
    for (w, T, st, K), g in k.groupby(["world", "target", "student", "K"]):
        out[(w, T, st, K)] = tuple(sorted(g.teacher[g.keep]))
    return out


# ---------------------------------------------------------------- 道具

def fam_mask(keys_df, fam, role, kept):
    m, p, f = keys_df.method, keys_df.pnorm, keys_df.meff
    teacher = p.str.split(":").str[0]
    kdt = p.str.replace(r"T\d+$", "", regex=True)
    if fam == "oursx":
        w = p.str.split(":").str[1]
        return (m == "dist") & teacher.isin(kept) & w.isin(WGTS) & f.isin(MEFFS)
    if fam == "oursx77":
        w = p.str.split(":").str[1]
        return (m == "dist") & teacher.isin(E77_DIST[role]) & w.isin(WGTS) & f.isin(MEFFS)
    if fam == "kd":
        return m.isin(["kd", "kdh"]) & kdt.isin(kept)
    if fam == "kd77":
        return m.isin(["kd", "kdh"]) & kdt.isin(E77_KD[role])
    if fam == "groupB":
        return m.isin(GROUP_B) & ~((m == "im") & ~f.isin(MEFFS))
    raise KeyError(fam)


def teacher_of(key):
    meth, pn, _ = key.split("|")
    if meth == "dist":
        return pn.split(":")[0]
    if meth in ("kd", "kdh"):
        return re.sub(r"T\d+$", "", pn)
    return None


def choose(keys, scores):
    """keys は文字列の順に並べたもの（呼ぶ側で並べる）．最大の点数の先頭。"""
    i = int(np.flatnonzero(scores == np.nanmax(scores))[0])
    return i


def summ(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"mean": np.nan, "sd": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan, "min": float(x.min()), "max": float(x.max()),
            "n": int(len(x))}


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a[ok], b[ok]).statistic)


# ---------------------------------------------------------------- 本体

def main():
    t0 = time.time()
    guard_selftest()
    kept = load_kept()
    lopo, teach_rows, spread_rows, rank_rows, full_rows, drop_rows, tlevel = [], [], [], [], [], [], []
    for world in WORLDS:
        check_script_copy(world)
        for T in TARGETS[world]:
            P = load(world, T)
            # 教師の値：(K, 教師) → 仮のターゲットのコーパスごとの値（反復と群の平均）．K に依らない教師は別の K の行で埋める（一致を検算）
            tch = P[(P.method == "teacher") & (P.student == world)]
            tv = tch.groupby(["K", "param", "pseudo"]).uar.mean().reset_index()
            tv["corpus"] = tv.pseudo.map(corp)
            tvc = tv.groupby(["K", "param", "corpus"]).uar.mean()
            tdict = {}
            for (K, n, c), v in tvc.items():
                tdict[(K, n, c)] = v
            for n in sorted(set(tch.param)):
                if K_INDEP_TEACHERS.match(n):
                    vals = {K: {c: tdict[(K, n, c)] for (K_, n_, c) in tdict if K_ == K and n_ == n} for K in KS}
                    have = [K for K in KS if vals[K]]
                    for K1, K2 in itertools.combinations(have, 2):
                        dmax = max(abs(vals[K1][c] - vals[K2][c]) for c in vals[K1])
                        CHECKS.append({"check": "teacher_K_indep", "world": world, "T": T, "teacher": n, "K1": K1, "K2": K2, "absdiff_max": dmax})
                    for K in KS:
                        if not vals[K] and have:
                            for c, v in vals[have[0]].items():
                                tdict[(K, n, c)] = v
            # agree の教師は pF と argmax が同じ（検算）
            for K in KS:
                for c in sorted(set(P.corpus)):
                    if (K, "agree", c) in tdict and (K, "pF", c) in tdict:
                        CHECKS.append({"check": "agree_equals_pF", "world": world, "T": T, "K": K, "corpus": c,
                                       "absdiff": abs(tdict[(K, "agree", c)] - tdict[(K, "pF", c)])})
            for (K, n, c), v in tdict.items():
                tlevel.append({"world": world, "T": T, "K": K, "teacher": n, "corpus": c, "uar": v})

            for st in WORLDS[world] + [world]:
                role = role_of(world, st)
                for K in KS:
                    S = P[(P.student == st) & (P.K == K) & (P.method != "teacher")]
                    none = S[(S.method == "none") & (S.wave == "e77")]
                    units = sorted(set(zip(none.pseudo, none.rep)))
                    nu = len(units)
                    M = S.pivot_table(index="key", columns=["pseudo", "rep"], values="uar", aggfunc="first")
                    cnt = S.groupby("key").size()
                    M = M.reindex(columns=pd.MultiIndex.from_tuples(units, names=["pseudo", "rep"]))
                    complete = M.notna().all(axis=1) & (cnt.reindex(M.index) == nu)
                    keys_df = pd.DataFrame({"key": M.index})
                    keys_df[["method", "pnorm", "meff"]] = keys_df.key.str.split("|", expand=True)
                    keys_df.index = M.index
                    kept_u = kept.get((world, T, st, K), ())
                    pseudo_of_col = np.array([u[0] for u in units])
                    corp_of_col = np.array([corp(u[0]) for u in units])
                    corpora = sorted(set(corp_of_col))
                    groups = sorted(set(pseudo_of_col))
                    nonekey = "none|-|0"
                    # コーパスの値（群ごとに反復の平均をとり，群の平均）
                    def corpus_vals(Mx):
                        g = {grp: Mx.loc[:, pseudo_of_col == grp].mean(axis=1) for grp in groups}
                        return pd.DataFrame({c: np.mean([g[grp] for grp in groups if corp(grp) == c], axis=0) for c in corpora},
                                            index=Mx.index)
                    Mv = M.to_numpy()
                    for fam in FAMS:
                        fm = fam_mask(keys_df, fam, role, kept_u).to_numpy()
                        drop_rows.append({"world": world, "T": T, "student": st, "K": K, "family": fam, "n_kept_teachers": len(kept_u),
                                          "n_cand": int((fm & complete.to_numpy()).sum()), "n_incomplete": int((fm & ~complete.to_numpy()).sum())})
                    for fam in FAMS:
                        fm = (fam_mask(keys_df, fam, role, kept_u) & complete).to_numpy()
                        if not fm.any():
                            continue
                        fk = np.array(sorted(M.index[fm]))                      # 文字列の順（同点の規則）
                        Mf = M.loc[fk]
                        CV = corpus_vals(Mf)                                     # 候補 × コーパス
                        noneCV = corpus_vals(M.loc[[nonekey]]).iloc[0]
                        A = Mf.to_numpy()
                        an = M.loc[nonekey].to_numpy()
                        # 全部の仮のターゲットで選んだもの（E79 の選択と同じ規則）
                        sc_all = A.mean(axis=1)
                        i_all = choose(fk, sc_all)
                        full_rows.append({"world": world, "T": T, "student": st, "K": K, "family": fam, "chosen_excl": fk[i_all],
                                          "score_excl": sc_all[i_all], "none_score": an.mean(),
                                          "chosen_incl": fk[i_all] if sc_all[i_all] > an.mean() else nonekey, "n_cand": len(fk)})
                        for c in corpora:
                            out_c = corp_of_col != c
                            sc = A[:, out_c].mean(axis=1)
                            ns = an[out_c].mean()
                            i = choose(fk, sc)
                            sel_ex = fk[i]
                            sel_in = sel_ex if sc[i] > ns else nonekey
                            v = CV[c].to_numpy()
                            vnone = float(noneCV[c])
                            o_ex = float(v.max())
                            o_key = fk[int(np.flatnonzero(v == o_ex)[0])]
                            val_ex = float(v[i])
                            val_in = val_ex if sel_in != nonekey else vnone
                            o_in = max(o_ex, vnone)
                            rank_pct = float((v > v[i]).sum()) / max(len(v) - 1, 1)    # 抜いたコーパスでの順位（0 ＝ 最良）
                            # 交差で作ったオラクル：c の反復を 1 つ抜き，残り 4 反復で最良を選び，抜いた反復で測る（5 回の平均）．
                            # 5 反復の平均の最大は雑音の分だけ高く出るので，その上げ底を除いた上限の見積もり
                            reps_col = np.array([u[1] for u in units])
                            cf_ex, cf_in = [], []
                            for r in sorted(set(reps_col[corp_of_col == c])):
                                m_in = (corp_of_col == c) & (reps_col != r)
                                m_out = (corp_of_col == c) & (reps_col == r)
                                s_in = A[:, m_in].mean(axis=1)
                                j = int(np.flatnonzero(s_in == s_in.max())[0])
                                x_out = float(A[j, m_out].mean())
                                cf_ex.append(x_out)
                                cf_in.append(x_out if s_in[j] > an[m_in].mean() else float(an[m_out].mean()))
                            rec = {"world": world, "T": T, "corpus": c, "student": st, "role": role, "K": K, "family": fam,
                                   "n_cand": len(fk), "none": vnone, "oracle_excl": o_ex, "oracle_key": o_key, "oracle_incl": o_in,
                                   "sel_excl": sel_ex, "val_excl": val_ex, "sel_incl": sel_in, "val_incl": val_in,
                                   "same_as_full": sel_ex == fk[i_all], "heldout_rank_pct": rank_pct,
                                   "oracle_cf_excl": float(np.mean(cf_ex)), "oracle_cf_incl": float(np.mean(cf_in)),
                                   "rho_select_vs_heldout": sp(sc, v)}
                            for tag, kk_ in (("oracle", o_key), ("sel_excl", sel_ex)):
                                tn = teacher_of(kk_)
                                rec[f"{tag}_teacher"] = tn
                                rec[f"{tag}_teacher_uar"] = tdict.get((K, tn, c), np.nan) if tn else np.nan
                            if fam in ("oursx", "kd"):
                                tvals = [tdict.get((K, tn, c), np.nan) for tn in kept_u]
                                rec["best_kept_teacher_uar"] = np.nanmax(tvals) if np.isfinite(tvals).any() else np.nan
                                rec["n_kept_teachers"] = len(kept_u)
                                rec["n_kept_teacher_uar_missing"] = int(np.sum(~np.isfinite(tvals)))
                            lopo.append(rec)
                        # 6. 順位の一致：コーパスの平均どうし，1 反復どうし（別のコーパス），同じ群の別の反復
                        rb = [sp(CV[c1], CV[c2]) for c1, c2 in itertools.combinations(corpora, 2)]
                        rb1, rw1 = [], []
                        for g1, g2 in itertools.combinations(groups, 2):
                            if corp(g1) == corp(g2):
                                continue
                            for r in range(5):
                                a1 = A[:, (pseudo_of_col == g1) & (np.array([u[1] for u in units]) == r)]
                                a2 = A[:, (pseudo_of_col == g2) & (np.array([u[1] for u in units]) == r)]
                                if a1.shape[1] and a2.shape[1]:
                                    rb1.append(sp(a1[:, 0], a2[:, 0]))
                        reps_of = np.array([u[1] for u in units])
                        for g1 in groups:
                            for r1, r2 in itertools.combinations(range(5), 2):
                                a1 = A[:, (pseudo_of_col == g1) & (reps_of == r1)]
                                a2 = A[:, (pseudo_of_col == g1) & (reps_of == r2)]
                                if a1.shape[1] and a2.shape[1]:
                                    rw1.append(sp(a1[:, 0], a2[:, 0]))
                        rank_rows.append({"world": world, "T": T, "student": st, "K": K, "family": fam, "n_cand": len(fk),
                                          "rho_between_corpus_mean": np.nanmean(rb) if rb else np.nan, "n_pairs_corpus": len(rb),
                                          "rho_between_single_rep": np.nanmean(rb1) if rb1 else np.nan,
                                          "rho_within_group_single_rep": np.nanmean(rw1) if rw1 else np.nan})
                    # 5. oursx の格子の中の広がり（kept の教師と E77 の教師の 2 通り）
                    for fam in ("oursx", "oursx77"):
                        fm = (fam_mask(keys_df, fam, role, kept_u) & complete).to_numpy()
                        if not fm.any():
                            continue
                        fk = M.index[fm]
                        CV = corpus_vals(M.loc[fk])
                        parts = pd.Series(fk).str.split("|", expand=True)
                        tchr = parts[1].str.split(":").str[0].to_numpy()
                        wgt = parts[1].str.split(":").str[1].to_numpy()
                        mf = parts[2].to_numpy()
                        tlist = sorted(set(tchr))
                        for c in corpora:
                            G = np.full((len(tlist), len(WGTS), len(MEFFS)), np.nan)
                            v = CV[c].to_numpy()
                            for x, a_, b_, c_ in zip(v, tchr, wgt, mf):
                                G[tlist.index(a_), WGTS.index(b_), MEFFS.index(c_)] = x
                            if np.isnan(G).any():
                                CHECKS.append({"check": "grid_incomplete", "world": world, "T": T, "student": st, "K": K, "family": fam, "corpus": c})
                                continue
                            gm = G.mean()
                            ss_tot = float(((G - gm) ** 2).sum())
                            mt, mw, mm = G.mean(axis=(1, 2)), G.mean(axis=(0, 2)), G.mean(axis=(0, 1))
                            ss_t = float(G.shape[1] * G.shape[2] * ((mt - gm) ** 2).sum())
                            ss_w = float(G.shape[0] * G.shape[2] * ((mw - gm) ** 2).sum())
                            ss_m = float(G.shape[0] * G.shape[1] * ((mm - gm) ** 2).sum())
                            it, iw, im_ = np.unravel_index(int(np.argmax(G)), G.shape)
                            o = G[it, iw, im_]
                            # 教師ごとの最良（重みと総重みを教師ごとに最良にしたとき）の幅
                            best_per_t = G.reshape(len(tlist), -1).max(axis=1)
                            best_per_w = G.transpose(1, 0, 2).reshape(len(WGTS), -1).max(axis=1)
                            best_per_m = G.transpose(2, 0, 1).reshape(len(MEFFS), -1).max(axis=1)
                            spread_rows.append({
                                "world": world, "T": T, "corpus": c, "student": st, "K": K, "family": fam, "n_teachers": len(tlist),
                                "grid_range": float(G.max() - G.min()), "grid_sd": float(G.std()),
                                "main_range_teacher": float(np.ptp(mt)), "main_range_wgt": float(np.ptp(mw)), "main_range_meff": float(np.ptp(mm)),
                                "opt_range_teacher": float(o - G[:, iw, im_].min()), "opt_range_wgt": float(o - G[it, :, im_].min()),
                                "opt_range_meff": float(o - G[it, iw, :].min()),
                                "bestof_range_teacher": float(np.ptp(best_per_t)), "bestof_range_wgt": float(np.ptp(best_per_w)),
                                "bestof_range_meff": float(np.ptp(best_per_m)),
                                "share_teacher": ss_t / ss_tot if ss_tot > 0 else np.nan, "share_wgt": ss_w / ss_tot if ss_tot > 0 else np.nan,
                                "share_meff": ss_m / ss_tot if ss_tot > 0 else np.nan,
                                "share_interaction": (ss_tot - ss_t - ss_w - ss_m) / ss_tot if ss_tot > 0 else np.nan,
                                "opt_teacher": tlist[it], "opt_wgt": WGTS[iw], "opt_meff": MEFFS[im_]})
            print(f"{world} {T} 済み（{time.time() - t0:.0f} 秒）", flush=True)
    OUT.mkdir(exist_ok=True)
    L = pd.DataFrame(lopo)
    L.to_csv(OUT / "lopo_cells.csv", index=False)
    pd.DataFrame(spread_rows).to_csv(OUT / "spread_cells.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(OUT / "rank_units.csv", index=False)
    pd.DataFrame(full_rows).to_csv(OUT / "full_select.csv", index=False)
    pd.DataFrame(drop_rows).to_csv(OUT / "candidates.csv", index=False)
    pd.DataFrame(tlevel).to_csv(OUT / "teacher_cells.csv", index=False)
    pd.DataFrame(CHECKS).to_csv(OUT / "checks.csv", index=False)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (OUT / "provenance.json").write_text(json.dumps({
        "script": str(Path(__file__).relative_to(ROOT)), "commit_at_run": commit, "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "opened_results_files": sorted(set(os.path.relpath(p, ROOT) for p in OPENED)), "sec": time.time() - t0}, ensure_ascii=False, indent=1))
    print(f"終わり（{time.time() - t0:.0f} 秒）．開いた results のファイル {len(set(OPENED))} 本", flush=True)


if __name__ == "__main__":
    main()
